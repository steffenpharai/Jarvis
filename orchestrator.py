"""Async orchestrator: wake → STT → LLM (with context + tools) → TTS; proactive idle vision."""

import asyncio
import logging
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

MAX_TOOL_ROUNDS = 3
STT_LLM_RETRIES = 2


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
) -> str:
    """Build messages, ReAct loop with tools, return final answer.

    Runs synchronously (blocking I/O to Ollama).  The orchestrator calls
    this via ``run_in_executor`` so the async event loop stays responsive
    for WebSocket broadcasts and uvicorn in ``--serve`` mode.
    """
    data_dir = Path(memory.get("data_dir", settings.DATA_DIR))
    reminders = load_reminders(data_dir)
    rem_text = format_reminders_for_llm(reminders, max_items=10)
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        from utils.power import get_system_stats
        sys_stats = get_system_stats()
    except Exception:
        sys_stats = None
    system = JARVIS_ORCHESTRATOR_SYSTEM_PROMPT
    if settings.SARCASM_ENABLED:
        system += " Sarcasm mode is on; you may be dry and slightly sarcastic."

    messages = build_messages_with_history(
        system,
        memory.get("summary", ""),
        short_term,
        query,
        vision_description=vision_description,
        reminders_text=rem_text,
        current_time=current_time,
        system_stats=sys_stats,
        max_turns=settings.CONTEXT_MAX_TURNS,
    )

    max_rounds = min(MAX_TOOL_ROUNDS, settings.MAX_TOOL_CALLS_PER_TURN)
    final_answer = ""
    for _ in range(max_rounds):
        response = chat_with_tools(
            settings.OLLAMA_BASE_URL,
            settings.OLLAMA_MODEL,
            messages,
            TOOL_SCHEMAS,
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
            result = run_tool(name, args)
            messages.append({"role": "tool", "tool_name": name, "content": result})

    if not final_answer:
        final_answer = "I'm unable to complete that, Sir."
    return final_answer


async def _run_one_turn(
    query: str,
    memory: dict,
    short_term: list,
    vision_description: str | None,
) -> str:
    """Async wrapper: runs the blocking LLM ReAct loop in an executor thread."""
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(
        None, _run_one_turn_sync, query, memory, short_term, vision_description,
    )


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

    try:
        while True:
            # Wait for next query with short timeout to check proactive
            timeout_sec = min(1.0, max(0, settings.PROACTIVE_IDLE_SEC - (time.monotonic() - idle_since)))
            try:
                query = await asyncio.wait_for(query_queue.get(), timeout=timeout_sec)
            except asyncio.TimeoutError:
                # Proactive: every PROACTIVE_IDLE_SEC run vision and optionally suggest break
                if (time.monotonic() - idle_since) >= settings.PROACTIVE_IDLE_SEC:
                    obs = await loop.run_in_executor(
                        None, run_tool, "vision_analyze", {"prompt": "person"},
                    )
                    if "person" in obs and "detected" in obs.lower():
                        say_text = "Sir, you appear to be at your desk. A short break is recommended."
                        wav = await loop.run_in_executor(
                            None, synthesize, say_text, settings.TTS_VOICE,
                        )
                        if wav:
                            await loop.run_in_executor(None, play_wav, wav)
                        if bridge is not None:
                            await bridge.send_proactive(say_text)
                    idle_since = time.monotonic()
                continue

            if not query or not str(query).strip():
                status_cb("Speaking")
                no_catch = "I didn't catch that, Sir."
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

            vision_description = await loop.run_in_executor(
                None, run_tool, "vision_analyze", {},
            )
            status_cb("Thinking (LLM)")
            for attempt in range(STT_LLM_RETRIES + 1):
                try:
                    final = await _run_one_turn(
                        str(query).strip(),
                        memory,
                        short_term,
                        vision_description,
                    )
                    break
                except Exception as e:
                    logger.exception("Turn failed (attempt %s): %s", attempt + 1, e)
                    if attempt >= STT_LLM_RETRIES:
                        final = "Brief glitch, Sir — please try again."
                    else:
                        final = "Retrying."
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
            short_term.append({"role": "user", "content": str(query).strip()})
            short_term.append({"role": "assistant", "content": final})
            await loop.run_in_executor(
                None,
                lambda: maybe_summarize(
                    memory, short_term,
                    settings.OLLAMA_BASE_URL, settings.OLLAMA_MODEL,
                    num_ctx=settings.OLLAMA_NUM_CTX,
                    every_n_turns=settings.SUMMARY_EVERY_N_TURNS,
                ),
            )
            await loop.run_in_executor(None, save_session, memory)
            status_cb("Listening")
            idle_since = time.monotonic()
    finally:
        stop_event.set()
        save_session(memory)
