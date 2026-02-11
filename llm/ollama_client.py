"""HTTP client to local Ollama (streaming optional).

Jetson Orin Nano 8GB Super — Qwen3:1.7b performance-optimised:
  - num_ctx default 8192 (100% GPU at 2.0 GB; 3.5x faster than old 2048 limit)
  - think=false for plain chat; think=true for tool calls (Qwen3 requires
    reasoning chain to route tool schemas on 1.7b — official Qwen3 behaviour)
  - num_predict 512 (thinking tokens count toward budget; 256 starved content)
  - 30 s timeout for chat; 60 s for tool calls (think chain adds ~2-4 s)
  - On CUDA OOM: unload model, drop kernel caches, retry with smaller context
  - Flash attention + q8_0 KV cache set via systemd env (OLLAMA_FLASH_ATTENTION,
    OLLAMA_KV_CACHE_TYPE)
  - Text-based fallback parser strips JSON / tool-call leakage from small models
"""

import json
import logging
import re
import subprocess
import time

import requests

logger = logging.getLogger(__name__)


def _parse_tool_calls(raw: list) -> list[dict]:
    """Normalize tool_calls from Ollama response: each has 'name' and 'arguments' (dict)."""
    out = []
    for tc in raw or []:
        fn = tc if isinstance(tc, dict) else getattr(tc, "__dict__", {})
        f = fn.get("function") or {}
        name = f.get("name", "") if isinstance(f, dict) else ""
        args = f.get("arguments") if isinstance(f, dict) else None
        if isinstance(args, str):
            try:
                args = json.loads(args) if args else {}
            except json.JSONDecodeError:
                args = {}
        out.append({"name": name, "arguments": args if isinstance(args, dict) else {}})
    return out


def _extract_text_tool_calls(content: str) -> tuple[str, list[dict]]:
    """Parse tool calls leaked into text content by small models.

    Handles patterns like:
      {"name": "tool_name", "parameters": {...}}
      Action: {"tool": "name", "args": {...}}

    Returns (cleaned_content, extracted_tool_calls).
    """
    tool_calls: list[dict] = []

    # Pattern 1: raw JSON object with "name" key
    json_pattern = re.compile(r'\{[^{}]*"name"\s*:\s*"(\w+)"[^{}]*\}', re.DOTALL)
    for m in json_pattern.finditer(content):
        try:
            obj = json.loads(m.group(0))
            name = obj.get("name", "")
            args = obj.get("parameters") or obj.get("arguments") or obj.get("args") or {}
            if name:
                tool_calls.append({"name": name, "arguments": args if isinstance(args, dict) else {}})
        except (json.JSONDecodeError, TypeError):
            pass

    # Pattern 2: Action: { ... }
    action_pattern = re.compile(r'Action:\s*(\{.+?\})', re.DOTALL)
    for m in action_pattern.finditer(content):
        try:
            obj = json.loads(m.group(1))
            name = obj.get("tool") or obj.get("name") or ""
            args = obj.get("args") or obj.get("arguments") or obj.get("parameters") or {}
            if name:
                tool_calls.append({"name": name, "arguments": args if isinstance(args, dict) else {}})
        except (json.JSONDecodeError, TypeError):
            pass

    if not tool_calls:
        return content, []

    # Strip matched JSON / Action blocks from content
    cleaned = json_pattern.sub("", content)
    cleaned = action_pattern.sub("", cleaned)
    cleaned = cleaned.strip().strip("{}").strip()
    return cleaned, tool_calls


def _clean_llm_content(content: str) -> str:
    """Strip thinking tags, JSON fragments, code fences, and meta-commentary.

    Qwen3 emits ``<think>…</think>`` reasoning blocks.  Small models sometimes
    echo context or emit partial JSON.  This ensures the final answer is clean
    natural language suitable for TTS playback.
    """
    if not content:
        return content

    # Strip qwen3-style thinking blocks
    content = re.sub(r'<think>[\s\S]*?</think>', '', content)
    # Remove code fences
    content = re.sub(r'```[\s\S]*?```', '', content)
    # Remove lone JSON objects / arrays
    content = re.sub(r'\{[^{}]*"(?:output|context|objects|reminders|name|type)"[^{}]*\}', '', content, flags=re.DOTALL)
    # Remove parenthetical meta-commentary (e.g. "(no tool call needed)")
    content = re.sub(r'\((?:Exact time|no tool|tool call|Note:)[^)]*\)', '', content, flags=re.IGNORECASE)
    # Remove lines that are just JSON keys/values
    lines = content.split('\n')
    clean_lines = [ln for ln in lines if not re.match(r'^\s*["\'{}\[\]]', ln.strip())]
    content = '\n'.join(clean_lines).strip()

    # If nothing meaningful remains, return empty
    if len(content) < 3:
        return ""
    return content


def _is_oom_error(response) -> bool:
    """True if response body indicates GPU OOM / CUDA allocation failure."""
    try:
        text = (response.text or "").lower()
        return (
            "allocate" in text
            or "buffer" in text
            or "failed to load model" in text
            or "out of memory" in text
            or "nvmapmemalloc" in text
        )
    except Exception:
        return False


def _is_oom_exception(exc: Exception) -> bool:
    """True if a requests exception wraps an OOM error."""
    if hasattr(exc, "response") and exc.response is not None:
        return _is_oom_error(exc.response)
    text = str(exc).lower()
    return "out of memory" in text or "allocate" in text or "failed to load" in text


def _safe_num_ctx(num_ctx: int) -> int:
    """Cap num_ctx to avoid OOM on 8GB Jetson. Use config cap if available."""
    try:
        from config import settings
        cap = getattr(settings, "OLLAMA_NUM_CTX_MAX", 2048)
    except Exception:
        cap = 2048
    return min(max(128, num_ctx), cap)


def _get_perf_options() -> dict:
    """Build Ollama ``options`` dict with performance settings from config."""
    try:
        from config import settings
        return {
            "num_predict": getattr(settings, "OLLAMA_NUM_PREDICT", 256),
            "temperature": getattr(settings, "OLLAMA_TEMPERATURE", 0.6),
        }
    except Exception:
        return {"num_predict": 256, "temperature": 0.6}


def _get_think_flag() -> bool:
    """Return the think flag (False disables Qwen3 reasoning tokens)."""
    try:
        from config import settings
        return getattr(settings, "OLLAMA_THINK", False)
    except Exception:
        return False


def unload_model(base_url: str, model: str) -> bool:
    """Ask Ollama to immediately unload a model from GPU (set keep_alive=0).

    On Jetson unified memory this frees CUDA memory back to the system.
    """
    url = f"{base_url.rstrip('/')}/api/chat"
    try:
        r = requests.post(
            url,
            json={"model": model, "messages": [], "keep_alive": 0},
            timeout=30,
        )
        if r.status_code == 200:
            logger.info("Unloaded model %s from GPU.", model)
            return True
        logger.warning("Unload model %s returned status %s.", model, r.status_code)
        return False
    except Exception as e:
        logger.warning("Unload model %s failed: %s", model, e)
        return False


def _drop_caches() -> None:
    """Drop kernel page/dentry/inode caches (needs sudo).

    On Jetson Orin Nano, buff/cache can hold ~3 GiB that nvmap/cudaMalloc
    cannot reclaim automatically. Dropping caches before model load retries
    makes that memory available to CUDA.
    """
    try:
        subprocess.run(
            ["sudo", "-n", "sh", "-c", "echo 3 > /proc/sys/vm/drop_caches"],
            timeout=5,
            capture_output=True,
        )
        logger.info("Dropped kernel caches to free memory for CUDA.")
    except Exception as e:
        logger.debug("drop_caches skipped (needs passwordless sudo): %s", e)


def _recover_from_oom(base_url: str, model: str) -> None:
    """Best-effort OOM recovery: pause vision, unload model, drop caches, resume.

    Coordinates with vision subsystem to free GPU memory (Tesla-style graceful
    degradation — shed non-critical loads before retrying critical path).
    """
    # Pause vision to free GPU memory for LLM recovery
    try:
        from vision.shared import pause_vision, resume_vision
        pause_vision()
        logger.info("Vision paused for OOM recovery")
    except Exception:
        pass

    unload_model(base_url, model)
    _drop_caches()
    time.sleep(2)  # Allow GPU memory to settle

    # Resume vision after recovery
    try:
        from vision.shared import resume_vision
        resume_vision()
        logger.info("Vision resumed after OOM recovery")
    except Exception:
        pass


# OOM retry sequence: step down context until one fits.
# 8192 → 4096 → 2048 → 1024 covers the realistic range on 8 GB Jetson.
_OOM_RETRY_NUM_CTX = [8192, 4096, 2048, 1024]


def chat(
    base_url: str,
    model: str,
    messages: list[dict],
    stream: bool = False,
    num_ctx: int = 8192,
) -> str:
    """Send chat request to Ollama; return full response content.

    Performance settings applied automatically:
      - num_ctx capped to OLLAMA_NUM_CTX_MAX (default 8192)
      - think=false disables Qwen3 reasoning tokens (fast path)
      - num_predict limits output length
      - 30 s timeout for voice-assistant responsiveness
    On CUDA OOM: unloads model, drops kernel caches, retries with smaller context.
    """
    num_ctx = _safe_num_ctx(num_ctx)
    url = f"{base_url.rstrip('/')}/api/chat"
    perf = _get_perf_options()
    think = _get_think_flag()
    for try_ctx in [num_ctx] + [c for c in _OOM_RETRY_NUM_CTX if c < num_ctx]:
        payload = {
            "model": model,
            "messages": messages,
            "stream": stream,
            "think": think,
            "options": {"num_ctx": try_ctx, **perf},
        }
        try:
            r = requests.post(url, json=payload, timeout=30)
            if r.status_code == 200:
                data = r.json()
                content = (data.get("message") or {}).get("content", "")
                # Strip qwen3-style thinking blocks (safety net if think wasn't honoured)
                content = re.sub(r'<think>[\s\S]*?</think>', '', content).strip()
                return content
            if r.status_code == 500 and _is_oom_error(r):
                logger.warning(
                    "Ollama GPU OOM (num_ctx=%s). Recovering and retrying.",
                    try_ctx,
                )
                _recover_from_oom(base_url, model)
                continue
            r.raise_for_status()
            return ""
        except requests.RequestException as e:
            if _is_oom_exception(e):
                logger.warning("Ollama GPU OOM (num_ctx=%s). Recovering and retrying.", try_ctx)
                _recover_from_oom(base_url, model)
                continue
            logger.warning("Ollama request failed: %s", e)
            return ""
    return ""




def chat_with_tools(
    base_url: str,
    model: str,
    messages: list[dict],
    tools: list[dict],
    stream: bool = False,
    num_ctx: int = 8192,
) -> dict:
    """Send chat request with tools; return dict with 'content' and 'tool_calls'.

    Performance: num_predict, temperature applied automatically.

    ``think`` is set **True** when tools are present.  Qwen3's Hermes-style
    tool template requires the reasoning chain to decide whether to call a
    tool — without it the 1.7b model ignores tool schemas entirely.  This is
    documented Qwen3 behaviour (qwen.readthedocs.io/en/latest/framework/
    function_call.html), not a workaround.  Regular ``chat()`` keeps
    think=False for speed.

    If the model leaks tool calls as text, ``_extract_text_tool_calls`` parses them.
    Final content is cleaned via ``_clean_llm_content`` to strip JSON residue.

    On CUDA OOM: unloads model, drops kernel caches, retries with smaller context.
    """
    num_ctx = _safe_num_ctx(num_ctx)
    url = f"{base_url.rstrip('/')}/api/chat"
    perf = _get_perf_options()
    # Qwen3 needs think=True to reason about tool usage; without it the model
    # ignores tool schemas and generates plain text instead.
    # Empty tools list → no tool reasoning needed → use fast path.
    has_tools = bool(tools)
    think = True if has_tools else _get_think_flag()
    timeout = 60 if has_tools else 30  # think chain adds 2-4 s overhead
    for try_ctx in [num_ctx] + [c for c in _OOM_RETRY_NUM_CTX if c < num_ctx]:
        payload = {
            "model": model,
            "messages": messages,
            "stream": stream,
            "think": think,
            "options": {"num_ctx": try_ctx, **perf},
        }
        # Only include tools key when tools are present; omitting it entirely
        # tells Ollama to skip tool-template injection (saves ~200 prompt tokens).
        if has_tools:
            payload["tools"] = tools
        try:
            r = requests.post(url, json=payload, timeout=timeout)
            if r.status_code == 200:
                data = r.json()
                msg = data.get("message") or {}
                content = (msg.get("content") or "").strip()
                tool_calls = _parse_tool_calls(msg.get("tool_calls") or [])

                # Fallback: small models may leak tool calls as text content
                if not tool_calls and content:
                    cleaned, text_tcs = _extract_text_tool_calls(content)
                    if text_tcs:
                        logger.debug("Extracted %d tool call(s) from text content", len(text_tcs))
                        tool_calls = text_tcs
                        content = cleaned

                # Clean residual JSON / structured data from final content
                if not tool_calls:
                    content = _clean_llm_content(content)

                return {"content": content, "tool_calls": tool_calls}
            if r.status_code == 500 and _is_oom_error(r):
                logger.warning(
                    "Ollama chat_with_tools OOM (num_ctx=%s). Recovering and retrying.",
                    try_ctx,
                )
                _recover_from_oom(base_url, model)
                continue
            return {"content": "", "tool_calls": []}
        except requests.RequestException as e:
            if _is_oom_exception(e):
                logger.warning("Ollama chat_with_tools OOM (num_ctx=%s). Recovering and retrying.", try_ctx)
                _recover_from_oom(base_url, model)
                continue
            logger.warning("Ollama chat_with_tools failed: %s", e)
            return {"content": "", "tool_calls": []}
    return {"content": "", "tool_calls": []}


def is_ollama_available(base_url: str = "http://127.0.0.1:11434") -> bool:
    """Check if Ollama API is reachable."""
    try:
        r = requests.get(f"{base_url.rstrip('/')}/api/tags", timeout=5)
        return r.status_code == 200
    except Exception:
        return False


def is_ollama_model_available(base_url: str, model: str) -> bool:
    """Check if Ollama is reachable and the given model is pulled.

    Matches exactly or by base name (e.g. ``qwen3:1.7b`` matches
    ``qwen3:1.7b`` listed in tags).
    """
    try:
        r = requests.get(f"{base_url.rstrip('/')}/api/tags", timeout=5)
        if r.status_code != 200:
            return False
        data = r.json()
        models = data.get("models") or []
        # Normalise: strip :latest suffix for comparison
        want = model.removesuffix(":latest")
        for m in models:
            name = (m.get("name") or "").removesuffix(":latest")
            if name == want or name == model:
                return True
        return False
    except Exception:
        return False
