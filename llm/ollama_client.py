"""HTTP client to local Ollama (streaming optional).

Jetson Orin Nano 8GB OOM hardening:
  - num_ctx capped to OLLAMA_NUM_CTX_MAX (default 512)
  - On CUDA OOM: unload model, drop kernel caches, retry with smaller context
  - Flash attention and q8_0 KV cache set via systemd env (see scripts/configure-ollama-systemd.sh)
"""

import json
import logging
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
        cap = getattr(settings, "OLLAMA_NUM_CTX_MAX", 512)
    except Exception:
        cap = 512
    return min(max(128, num_ctx), cap)


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
    """Best-effort OOM recovery: unload model, drop caches, brief pause."""
    unload_model(base_url, model)
    _drop_caches()
    time.sleep(1)


# OOM retry sequence: try smaller context until one works
_OOM_RETRY_NUM_CTX = [512, 256, 128]


def chat(
    base_url: str,
    model: str,
    messages: list[dict],
    stream: bool = False,
    num_ctx: int = 512,
) -> str:
    """Send chat request to Ollama; return full response content.

    num_ctx is capped to OLLAMA_NUM_CTX_MAX (default 512) to avoid OOM.
    On CUDA OOM: unloads model, drops kernel caches, retries with smaller context.
    """
    num_ctx = _safe_num_ctx(num_ctx)
    url = f"{base_url.rstrip('/')}/api/chat"
    for try_ctx in [num_ctx] + [c for c in _OOM_RETRY_NUM_CTX if c < num_ctx]:
        payload = {
            "model": model,
            "messages": messages,
            "stream": stream,
            "options": {"num_ctx": try_ctx},
        }
        try:
            r = requests.post(url, json=payload, timeout=120)
            if r.status_code == 200:
                data = r.json()
                return (data.get("message") or {}).get("content", "")
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
    num_ctx: int = 512,
) -> dict:
    """Send chat request with tools; return dict with 'content' and 'tool_calls'.

    On CUDA OOM: unloads model, drops kernel caches, retries with smaller context.
    """
    num_ctx = _safe_num_ctx(num_ctx)
    url = f"{base_url.rstrip('/')}/api/chat"
    for try_ctx in [num_ctx] + [c for c in _OOM_RETRY_NUM_CTX if c < num_ctx]:
        payload = {
            "model": model,
            "messages": messages,
            "stream": stream,
            "tools": tools,
            "options": {"num_ctx": try_ctx},
        }
        try:
            r = requests.post(url, json=payload, timeout=120)
            if r.status_code == 200:
                data = r.json()
                msg = data.get("message") or {}
                return {
                    "content": (msg.get("content") or "").strip(),
                    "tool_calls": _parse_tool_calls(msg.get("tool_calls") or []),
                }
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
    """Check if Ollama is reachable and the given model is pulled (avoids 500 on chat)."""
    try:
        r = requests.get(f"{base_url.rstrip('/')}/api/tags", timeout=5)
        if r.status_code != 200:
            return False
        data = r.json()
        models = data.get("models") or []
        for m in models:
            name = m.get("name") or ""
            if name == model:
                return True
        return False
    except Exception:
        return False
