"""Async orchestrator: wake → STT → LLM (with context + tools) → TTS; proactive idle vision.

Performance budget (Jetson Orin Nano 8 GB Super, MAXN_SUPER):
  Target: user query → spoken reply in <10 s.
  - Vision only for vision-related queries (saves ~1 s)
  - Tool round cap = 2 (rarely need more)
  - Summarisation is fire-and-forget (doesn't block reply)
  - Retry once only (no multi-retry with 30 s timeout)
"""

import asyncio
import logging
import re as _re
import tempfile
import threading
import time
from collections.abc import Callable
from datetime import datetime
from pathlib import Path

from config import settings
from config.prompts import JARVIS_ORCHESTRATOR_SYSTEM_PROMPT
from llm.context import build_messages_with_history
from llm.ollama_client import chat_with_tools, is_ollama_available, is_ollama_model_available
from memory import load_session, maybe_summarize, save_session
from tools import TOOL_SCHEMAS, run_tool
from utils.reminders import format_reminders_for_llm, load_reminders

logger = logging.getLogger(__name__)

MAX_TOOL_ROUNDS = 2
STT_LLM_RETRIES = 1

# Keywords that indicate the user wants vision analysis
_VISION_KEYWORDS = _re.compile(
    r'\b(see|look|camera|watch|show|screen|desk|room|face|person|posture|what.s around'
    r'|what do you see|who.s here|scan|detect|visual)\b',
    _re.IGNORECASE,
)

# ── Intent-based tool routing ────────────────────────────────────────
# Only send tool schemas when the query likely needs a tool.  This prevents
# Qwen3:1.7b from wasting its 512-token think budget reasoning about tools
# on simple greetings/status queries, which caused empty responses.
# This is the standard production pattern (Alexa, Rasa, Dialogflow) for
# resource-constrained NLU: deterministic intent routing before LLM.
_TOOL_INTENT_PATTERNS: dict[str, _re.Pattern] = {
    "tell_joke": _re.compile(
        r'\b(joke|funny|something funny|laugh|humou?r|make me (laugh|smile)|amuse)\b',
        _re.IGNORECASE,
    ),
    "create_reminder": _re.compile(
        r'\b(remind|reminder|don.t forget|set.*(remind|alarm|timer)|remember to)\b',
        _re.IGNORECASE,
    ),
    "toggle_sarcasm": _re.compile(
        r'\b(sarcas[mt]|sarcastic|snark)\b',
        _re.IGNORECASE,
    ),
    # vision_analyze is only for explicit re-scan requests.  Initial vision is
    # pre-fetched by the orchestrator via _VISION_KEYWORDS before the LLM call.
    "vision_analyze": _re.compile(
        r'\b(re-?scan|scan again|look again|check again|another look|refresh.*(camera|view))\b',
        _re.IGNORECASE,
    ),
    "hologram_render": _re.compile(
        r'\b(hologram|3d|three.?d|render|point cloud|depth map|spatial)\b',
        _re.IGNORECASE,
    ),
}


def _select_tools_for_query(query: str) -> list[dict]:
    """Return only the tool schemas whose intent matches the query.

    If no tool intent is detected, returns an empty list so the LLM runs
    without tools (think=false fast path).  This keeps simple queries
    (greetings, status, time, goodnight) at 3-5 s instead of 15+ s.
    """
    matched_names: set[str] = set()
    for tool_name, pattern in _TOOL_INTENT_PATTERNS.items():
        if pattern.search(query):
            matched_names.add(tool_name)

    if not matched_names:
        return []

    return [schema for schema in TOOL_SCHEMAS if schema["function"]["name"] in matched_names]


def _gui_status(status: str) -> None:
    """Default status callback: forward to the Tkinter overlay (if running)."""
    try:
        from gui.overlay import set_status
        set_status(status)
    except Exception:
        pass


def _wake_listener_impl(
    query_queue: asyncio.Queue,
    loop: asyncio.AbstractEventLoop,
    stop_event: threading.Event,
    status_cb: Callable[[str], None] = _gui_status,
) -> None:
    """Run in thread: wake loop; on wake, record + STT and put text on queue."""
    from audio.input import get_default_input_index, record_to_file
    from voice.stt import transcribe
    from voice.wakeword import run_wake_loop

    def on_wake():
        if stop_event.is_set():
            return
        status_cb("Listening (recording)")
        device_index = get_default_input_index()
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            wav_path = f.name
        try:
            if not record_to_file(
                wav_path,
                duration_sec=settings.RECORD_DURATION_SEC,
                device_index=device_index,
            ):
                status_cb("Listening")
                return
            status_cb("Thinking (STT)")
            text = transcribe(wav_path, model_size=settings.STT_MODEL_SIZE)
            text = (text or "").strip()
            loop.call_soon_threadsafe(query_queue.put_nowait, text)
        except Exception as e:
            logger.exception("Wake/record/STT failed: %s", e)
            loop.call_soon_threadsafe(query_queue.put_nowait, "")
        finally:
            Path(wav_path).unlink(missing_ok=True)
            status_cb("Listening")

    stop = run_wake_loop(on_wake, device_index=None)
    try:
        while not stop_event.is_set():
            time.sleep(0.5)
    finally:
        stop.set()


def _run_one_turn_sync(
    query: str,
    memory: dict,
    short_term: list,
    vision_description: str | None,
    vitals_text: str | None = None,
    threat_text: str | None = None,
    bridge_ref: object | None = None,
) -> str:
    """Build messages, ReAct loop with tools, return final answer.

    Runs synchronously (blocking I/O to Ollama).  The orchestrator calls
    this via ``run_in_executor`` so the async event loop stays responsive
    for WebSocket broadcasts and uvicorn in ``--serve`` mode.
    """
    data_dir = Path(memory.get("data_dir", settings.DATA_DIR))
    reminders = load_reminders(data_dir)
    rem_text = format_reminders_for_llm(reminders, max_items=5)
    current_time = datetime.now().strftime("%H:%M %a %b %d")  # compact time format
    try:
        from utils.power import get_system_stats
        sys_stats = get_system_stats()
    except Exception:
        sys_stats = None
    system = JARVIS_ORCHESTRATOR_SYSTEM_PROMPT
    if settings.SARCASM_ENABLED:
        system += " Sarcasm on."

    messages = build_messages_with_history(
        system,
        memory.get("summary", ""),
        short_term,
        query,
        vision_description=vision_description,
        reminders_text=rem_text,
        current_time=current_time,
        system_stats=sys_stats,
        vitals_text=vitals_text,
        threat_text=threat_text,
        max_turns=settings.CONTEXT_MAX_TURNS,
    )

    # Intent-based tool routing: only send tools the query actually needs.
    # Empty list → chat_with_tools uses think=false (fast, no tool reasoning).
    selected_tools = _select_tools_for_query(query)

    max_rounds = min(MAX_TOOL_ROUNDS, settings.MAX_TOOL_CALLS_PER_TURN)
    final_answer = ""
    for _ in range(max_rounds):
        response = chat_with_tools(
            settings.OLLAMA_BASE_URL,
            settings.OLLAMA_MODEL,
            messages,
            selected_tools,
            stream=False,
            num_ctx=settings.OLLAMA_NUM_CTX,
        )
        content = response.get("content", "")
        tool_calls = response.get("tool_calls", [])[: settings.MAX_TOOL_CALLS_PER_TURN]
        messages.append({"role": "assistant", "content": content or "", "tool_calls": tool_calls})
        if not tool_calls:
            final_answer = (content or "").strip()
            break
        for tc in tool_calls:
            name = tc.get("name", "")
            args = tc.get("arguments") or {}
            # Broadcast tool execution step (thread-safe from executor)
            _thinking(bridge_ref, "tool", f"Running {name}...")
            result = run_tool(name, args)
            _thinking(bridge_ref, "tool_done", f"Completed {name}")
            messages.append({"role": "tool", "tool_name": name, "content": result})

    if not final_answer:
        final_answer = "I'm afraid I wasn't able to complete that request, sir."
    return final_answer


async def _run_one_turn(
    query: str,
    memory: dict,
    short_term: list,
    vision_description: str | None,
    vitals_text: str | None = None,
    threat_text: str | None = None,
    bridge_ref: object | None = None,
) -> str:
    """Async wrapper: runs the blocking LLM ReAct loop in an executor thread."""
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(
        None, _run_one_turn_sync, query, memory, short_term,
        vision_description, vitals_text, threat_text, bridge_ref,
    )


def _thinking(bridge: object | None, step: str, detail: str = "") -> None:
    """Broadcast a thinking step to PWA clients (fire-and-forget)."""
    if bridge is not None and hasattr(bridge, "send_thinking_step_threadsafe"):
        bridge.send_thinking_step_threadsafe(step, detail)


async def _thinking_async(bridge: object | None, step: str, detail: str = "") -> None:
    """Broadcast a thinking step from an async context."""
    if bridge is not None and hasattr(bridge, "send_thinking_step"):
        await bridge.send_thinking_step(step, detail)


async def run_orchestrator(
    query_queue: asyncio.Queue | None = None,
    bridge: object | None = None,
    status_callback: Callable[[str], None] | None = None,
) -> None:
    """Main async loop: wait for query (wake+STT) or proactive; run ReAct + TTS; persist session.

    Vision is always enabled.

    Parameters
    ----------
    query_queue : asyncio.Queue, optional
        Shared queue; when provided (by ``--serve``) the orchestrator reads
        from it instead of creating its own.
    bridge : Bridge, optional
        When provided, status/reply updates are broadcast to all connected
        WebSocket clients.
    status_callback : callable, optional
        ``fn(status_str)`` invoked on every state transition.  Defaults to
        the Tkinter GUI overlay.  ``--serve`` passes a callback that also
        broadcasts over WebSocket.
    """
    from audio.output import play_wav
    from voice.tts import synthesize

    status_cb: Callable[[str], None] = status_callback or _gui_status

    data_dir = Path(settings.DATA_DIR)
    data_dir.mkdir(parents=True, exist_ok=True)
    memory = load_session(data_dir)
    short_term: list[dict] = []
    if query_queue is None:
        query_queue = asyncio.Queue()
    loop = asyncio.get_running_loop()
    stop_event = threading.Event()

    if not is_ollama_available(settings.OLLAMA_BASE_URL):
        logger.error("Ollama not available. Start: bash scripts/start-ollama.sh")
        return
    if not is_ollama_model_available(settings.OLLAMA_BASE_URL, settings.OLLAMA_MODEL):
        logger.error("Model %s not pulled. Run: ollama pull %s", settings.OLLAMA_MODEL, settings.OLLAMA_MODEL)
        return

    thread = threading.Thread(
        target=_wake_listener_impl,
        args=(query_queue, loop, stop_event),
        kwargs={"status_cb": status_cb},
        daemon=True,
    )
    thread.start()
    status_cb("Listening")
    idle_since = time.monotonic()
    vision_description: str | None = None
    vitals_text: str | None = None
    threat_text: str | None = None

    try:
        while True:
            # Wait for next query with short timeout to check proactive
            timeout_sec = min(1.0, max(0, settings.PROACTIVE_IDLE_SEC - (time.monotonic() - idle_since)))
            try:
                query = await asyncio.wait_for(query_queue.get(), timeout=timeout_sec)
            except asyncio.TimeoutError:
                # Proactive: every PROACTIVE_IDLE_SEC run enriched vision check
                if (time.monotonic() - idle_since) >= settings.PROACTIVE_IDLE_SEC:
                    try:
                        from tools import vision_analyze_full

                        vision_data = await loop.run_in_executor(
                            None, vision_analyze_full, None,
                        )
                        proactive_desc = vision_data.get("description", "")
                        proactive_vitals = vision_data.get("vitals")
                        proactive_threat = vision_data.get("threat")

                        # Proactive vitals alert
                        say_text = None
                        if proactive_vitals and getattr(proactive_vitals, "fatigue_level", "unknown") in ("moderate", "severe"):
                            say_text = "Sir, you appear quite fatigued. Might I suggest a brief respite. Even Mr. Stark took the occasional break."
                        elif proactive_vitals and getattr(proactive_vitals, "posture_label", "unknown") == "poor":
                            say_text = "Sir, your posture could use some attention. Perhaps a stretch is in order."
                        elif "person" in proactive_desc.lower():
                            say_text = "Sir, you've been at your desk for quite some time. Might I suggest a brief respite?"

                        # Proactive threat alert
                        if proactive_threat and getattr(proactive_threat, "level", 0) >= 5:
                            rec = getattr(proactive_threat, "recommendation", "")
                            say_text = f"Sir, I'm detecting elevated threat conditions. {rec}"

                        if say_text:
                            wav = await loop.run_in_executor(
                                None, synthesize, say_text, settings.TTS_VOICE,
                            )
                            if wav:
                                await loop.run_in_executor(None, play_wav, wav)
                            if bridge is not None:
                                await bridge.send_proactive(say_text)

                        # Broadcast vitals/threat to PWA
                        if bridge is not None and proactive_vitals:
                            await bridge.broadcast({
                                "type": "vitals",
                                "data": {
                                    "fatigue": getattr(proactive_vitals, "fatigue_level", "unknown"),
                                    "posture": getattr(proactive_vitals, "posture_label", "unknown"),
                                    "heart_rate": getattr(proactive_vitals, "heart_rate_bpm", None),
                                    "hr_confidence": getattr(proactive_vitals, "heart_rate_confidence", 0),
                                    "alerts": getattr(proactive_vitals, "alerts", []),
                                },
                            })
                    except Exception as e:
                        logger.debug("Proactive vision check failed: %s", e)

                    idle_since = time.monotonic()
                continue

            if not query or not str(query).strip():
                status_cb("Speaking")
                no_catch = "My apologies, sir. I didn't quite catch that."
                wav = await loop.run_in_executor(
                    None, synthesize, no_catch, settings.TTS_VOICE,
                )
                if wav:
                    await loop.run_in_executor(None, play_wav, wav)
                if bridge is not None:
                    await bridge.send_reply(no_catch)
                status_cb("Listening")
                idle_since = time.monotonic()
                continue

            # Broadcast user transcript to PWA clients
            if bridge is not None:
                await bridge.send_transcript(str(query).strip(), final=True)

            query_text = str(query).strip()

            await _thinking_async(bridge, "heard", "Processing your words...")

            # Only run vision if the query is vision-related (saves ~1 s)
            if _VISION_KEYWORDS.search(query_text):
                await _thinking_async(bridge, "vision", "Scanning the environment...")
                try:
                    from tools import vision_analyze_full

                    vision_data = await loop.run_in_executor(
                        None, vision_analyze_full, None,
                    )
                    vision_description = vision_data.get("description")
                    vitals_text = vision_data.get("vitals_text")
                    threat_text = vision_data.get("threat_text")
                except Exception:
                    vision_description = await loop.run_in_executor(
                        None, run_tool, "vision_analyze", {},
                    )
                    vitals_text = None
                    threat_text = None
                await _thinking_async(bridge, "vision_done", "Environment analyzed")
            # else: reuse previous vision_description (or None)

            await _thinking_async(bridge, "context", "Building context from memory...")
            status_cb("Thinking (LLM)")
            await _thinking_async(bridge, "reasoning", "Analyzing and reasoning...")
            for attempt in range(STT_LLM_RETRIES + 1):
                try:
                    final = await _run_one_turn(
                        query_text,
                        memory,
                        short_term,
                        vision_description,
                        vitals_text=vitals_text,
                        threat_text=threat_text,
                        bridge_ref=bridge,
                    )
                    break
                except Exception as e:
                    logger.exception("Turn failed (attempt %s): %s", attempt + 1, e)
                    if attempt >= STT_LLM_RETRIES:
                        final = "A momentary glitch in my systems, sir. Shall we try that again?"
                    else:
                        await _thinking_async(bridge, "retry", "Retrying reasoning...")
                        final = "Retrying."
            await _thinking_async(bridge, "speaking", "Formulating response...")
            status_cb("Speaking")
            # Broadcast reply to PWA clients
            if bridge is not None:
                await bridge.send_reply(final)
            wav = await loop.run_in_executor(
                None, synthesize, final, settings.TTS_VOICE,
            )
            if wav:
                await loop.run_in_executor(None, play_wav, wav)
            else:
                logger.warning("TTS failed for reply")
            short_term.append({"role": "user", "content": query_text})
            short_term.append({"role": "assistant", "content": final})

            # Fire-and-forget summarisation — don't block the reply pipeline.
            # Uses a background thread so the main loop can immediately return
            # to "Listening" state.
            def _bg_summarize():
                try:
                    maybe_summarize(
                        memory, short_term,
                        settings.OLLAMA_BASE_URL, settings.OLLAMA_MODEL,
                        num_ctx=min(settings.OLLAMA_NUM_CTX, 1024),
                        every_n_turns=settings.SUMMARY_EVERY_N_TURNS,
                    )
                    save_session(memory)
                except Exception as exc:
                    logger.debug("Background summarisation failed: %s", exc)

            threading.Thread(target=_bg_summarize, daemon=True).start()

            await _thinking_async(bridge, "done", "Ready")
            status_cb("Listening")
            idle_since = time.monotonic()
    finally:
        stop_event.set()
        save_session(memory)
