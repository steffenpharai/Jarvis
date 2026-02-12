"""Async orchestrator: wake → STT → LLM (with context + tools) → TTS; proactive idle vision.

Performance budget (Jetson Orin Nano 8 GB Super, MAXN_SUPER):
  Target: user query → spoken reply in <10 s.
  - Background scene context always injected (updated every 5s in bg thread)
  - Vision re-scan only for explicit vision-related queries
  - Tool round cap = 2 (rarely need more)
  - Summarisation is fire-and-forget (doesn't block reply)
  - Retry once only (no multi-retry with 30 s timeout)

Enhancements over v1:
  - VAD-based recording (adaptive end-of-speech, no fixed 5s)
  - BT auto-reconnect daemon (exponential backoff, verbal feedback)
  - Background scene loop (always-on context for every LLM call)
  - Proactive intelligence (person enter/leave, object change, env shift)
  - Preflight checks at startup with verbal status
  - OOM recovery with vision pause coordination

Enhancements v2 (hands-free autonomy):
  - Ambient awareness loop (always-on DIS flow at 160x120, 2Hz)
  - Motion-triggered full perception (no manual trigger required)
  - Proactive verbalization with cooldown (collision, scene change, status)
  - Verbal error recovery ("Brief comms glitch, sir")
  - Battery/thermal adaptive duty cycle
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

# ── Background scene context (always-on, ~5s interval) ───────────────
_bg_scene_description: str | None = None
_bg_scene_lock = threading.Lock()
_bg_scene_stop: threading.Event | None = None
_BG_SCENE_INTERVAL = 5.0  # seconds between background scene updates

# ── Proactive intelligence state ─────────────────────────────────────
_prev_person_count: int = 0
_prev_object_set: set[str] = set()

# Keywords that indicate the user wants vision analysis.
# Broad on purpose: false-positives cost ~1 s of vision latency, but
# false-negatives cause the LLM to answer with NO scene context (or
# worse, stale context from a prior query).
_VISION_KEYWORDS = _re.compile(
    r'\b(see|sees|seeing|saw|look|looking|camera|watch|watching|show|screen|desk|room'
    r'|face|person|people|someone|anyone|posture|what.s around|what.s in front'
    r'|what do you see|who.s here|who.s there|what.s there|what.s going on'
    r'|what.s happening|how does it look|how do I look|how am I looking'
    r'|is.*(there|anyone|someone|safe)|anything.*(around|new|changed|different)'
    r'|check.*(surround|area|room)|environment|surroundings'
    r'|scan|detect|visual|view|observe|spot|notice|identify'
    r'|in front of you|around you|near me|over there)\b',
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


def _start_bg_scene_thread() -> threading.Event:
    """Start background thread that continuously updates scene context.

    Industry pattern (Tesla Autopilot, NVIDIA DRIVE): always-on perception
    loop running independently from the decision/query loop, so context
    is always fresh when needed.
    """
    global _bg_scene_stop
    _bg_scene_stop = threading.Event()

    def _loop():
        global _bg_scene_description
        while not _bg_scene_stop.is_set():
            try:
                from vision.shared import describe_current_scene, is_vision_paused
                if not is_vision_paused():
                    desc = describe_current_scene()
                    with _bg_scene_lock:
                        _bg_scene_description = desc
            except Exception as e:
                logger.debug("Background scene update failed: %s", e)
            _bg_scene_stop.wait(_BG_SCENE_INTERVAL)

    t = threading.Thread(target=_loop, daemon=True, name="bg-scene")
    t.start()
    logger.info("Background scene context thread started (%.0fs interval)", _BG_SCENE_INTERVAL)
    return _bg_scene_stop


def get_bg_scene() -> str | None:
    """Get the latest background scene description (thread-safe)."""
    with _bg_scene_lock:
        return _bg_scene_description


def _check_proactive_changes(vision_data: dict) -> str | None:
    """Detect meaningful environmental changes for proactive alerts.

    Pattern: state-change detection (SpaceX mission control telemetry monitoring).
    Only alert on transitions, not steady state.
    """
    global _prev_person_count, _prev_object_set

    tracked = vision_data.get("tracked", [])

    # Count persons
    current_persons = sum(1 for t in tracked if t.get("class_name", "").lower() == "person")
    current_objects = {t.get("class_name", "unknown") for t in tracked if t.get("class_name")}

    alerts = []

    # Person entered
    if current_persons > _prev_person_count and _prev_person_count == 0:
        alerts.append("Sir, someone has entered the room.")
    elif current_persons > _prev_person_count:
        diff = current_persons - _prev_person_count
        alerts.append(f"Sir, {diff} additional {'person has' if diff == 1 else 'people have'} appeared.")

    # Person left
    if current_persons == 0 and _prev_person_count > 0:
        alerts.append("Sir, the room appears to be clear now.")

    # New significant objects appeared
    new_objects = current_objects - _prev_object_set - {"person"}
    if new_objects and len(new_objects) <= 3:
        obj_str = ", ".join(sorted(new_objects))
        alerts.append(f"Sir, I've noticed new items in view: {obj_str}.")

    _prev_person_count = current_persons
    _prev_object_set = current_objects

    if alerts:
        return " ".join(alerts)
    return None


# ── Ambient awareness thread (hands-free walk-around) ─────────────────

_ambient_stop: threading.Event | None = None
_ambient_thread: threading.Thread | None = None
# Cooldown state for proactive verbalization
_last_proactive_verbal: float = 0.0


def _start_ambient_thread(
    query_queue: asyncio.Queue,
    loop: asyncio.AbstractEventLoop,
    bridge: object | None = None,
) -> threading.Event:
    """Start the ambient awareness background thread.

    Continuously runs ultra-light DIS flow at 160x120 (~2ms per frame)
    to detect motion and scene changes.  When a trigger fires, pushes
    a _ProactiveEvent sentinel onto the query queue so the orchestrator
    can run a full scan and optionally speak.
    """
    global _ambient_stop
    _ambient_stop = threading.Event()

    def _loop():
        try:
            from vision.ambient import AmbientAwareness
            from vision.shared import read_frame
        except ImportError as e:
            logger.warning("Ambient awareness unavailable: %s", e)
            return

        ambient = AmbientAwareness(
            cooldown_sec=getattr(settings, "PROACTIVE_COOLDOWN_SEC", 10.0),
        )
        logger.info("Ambient awareness thread started (%.0f Hz idle)", ambient.current_hz)

        while not _ambient_stop.is_set():
            try:
                frame = read_frame()
                if frame is None:
                    _ambient_stop.wait(1.0)
                    continue

                event = ambient.check_frame(frame)
                if event is not None and event.recommend_full_scan:
                    # Push a sentinel that the orchestrator recognises
                    sentinel = f"__ambient__{event.event_type.value}__{event.detail}"
                    loop.call_soon_threadsafe(query_queue.put_nowait, sentinel)
                    ambient.enter_cooldown()

                _ambient_stop.wait(ambient.interval_sec)
            except Exception as e:
                logger.debug("Ambient loop error: %s", e)
                _ambient_stop.wait(2.0)

    global _ambient_thread
    _ambient_thread = threading.Thread(target=_loop, daemon=True, name="ambient-awareness")
    _ambient_thread.start()
    return _ambient_stop


def _stop_ambient_thread() -> None:
    """Stop the ambient awareness thread."""
    global _ambient_stop, _ambient_thread
    if _ambient_stop is not None:
        _ambient_stop.set()
    _ambient_thread = None


def _is_ambient_event(query: str) -> bool:
    """Check if a query string is an ambient awareness sentinel."""
    return isinstance(query, str) and query.startswith("__ambient__")


def _parse_ambient_event(query: str) -> tuple[str, str]:
    """Parse ambient sentinel into (event_type, detail)."""
    parts = query.split("__", 4)
    # parts: ['', 'ambient', event_type, detail]
    if len(parts) >= 4:
        return parts[2], parts[3]
    return "unknown", ""


async def _handle_ambient_event(
    event_type: str,
    detail: str,
    bridge: object | None,
    loop: asyncio.AbstractEventLoop,
) -> None:
    """Handle an ambient awareness event: run full scan, optionally speak.

    Applies cooldown to prevent verbal spam.
    """
    global _last_proactive_verbal
    from audio.output import play_wav
    from voice.tts import synthesize

    now = time.monotonic()
    cooldown = getattr(settings, "PROACTIVE_COOLDOWN_SEC", 10.0)

    # Run full enriched vision
    say_text = None
    try:
        from tools import vision_analyze_full

        vision_data = await loop.run_in_executor(None, vision_analyze_full, None)

        # Check for collision alerts (bypass cooldown -- safety critical)
        collision_alerts = vision_data.get("collision_alerts", [])
        for ca in collision_alerts:
            if ca.get("severity") in ("critical", "warning"):
                say_text = ca.get("message", "")
                break

        # Check for environmental changes
        if say_text is None:
            change_alert = _check_proactive_changes(vision_data)
            if change_alert and (now - _last_proactive_verbal) > cooldown:
                say_text = change_alert

        # Motion state transitions (with cooldown)
        if say_text is None and (now - _last_proactive_verbal) > cooldown:
            if event_type == "ego_motion_start":
                say_text = "Walking mode activated, sir. Increasing scan rate."
            elif event_type == "ego_motion_stop":
                say_text = "Stationary. Reducing scan rate to conserve power."
            elif event_type == "scene_change":
                say_text = "New area detected, sir. Scanning surroundings."
            elif event_type == "thermal_throttle":
                say_text = "Thermal limit reached, sir. Reducing vision processing."
            elif event_type == "battery_low":
                say_text = "Battery low, sir. Entering conservation mode."

        # Broadcast perception data to PWA
        if bridge is not None:
            perception_text = vision_data.get("perception_text", "")
            if perception_text:
                await bridge.broadcast({
                    "type": "perception",
                    "data": {"summary": perception_text},
                })

    except Exception as e:
        logger.debug("Ambient full scan failed: %s", e)

    # Speak and broadcast
    if say_text:
        _last_proactive_verbal = now
        wav = await loop.run_in_executor(None, synthesize, say_text, settings.TTS_VOICE)
        if wav:
            await loop.run_in_executor(None, play_wav, wav)
        if bridge is not None:
            await bridge.send_proactive(say_text)


async def _verbal_error(
    bridge: object | None,
    message: str,
    loop: asyncio.AbstractEventLoop | None = None,
) -> None:
    """Speak an error message via TTS and broadcast to PWA.

    Used for graceful verbal recovery instead of silent failures.
    """
    try:
        from audio.output import play_wav
        from voice.tts import synthesize

        if loop is None:
            loop = asyncio.get_running_loop()
        wav = await loop.run_in_executor(None, synthesize, message, settings.TTS_VOICE)
        if wav:
            await loop.run_in_executor(None, play_wav, wav)
    except Exception:
        pass
    if bridge is not None:
        try:
            await bridge.send_error(message)
        except Exception:
            pass


def _wake_listener_impl(
    query_queue: asyncio.Queue,
    loop: asyncio.AbstractEventLoop,
    stop_event: threading.Event,
    status_cb: Callable[[str], None] = _gui_status,
) -> None:
    """Run in thread: wake loop; on wake, VAD-record + STT and put text on queue.

    Uses VAD (Voice Activity Detection) instead of fixed 5s recording for
    lower latency and adaptive speech capture.
    """
    from audio.input import get_default_input_index
    from voice.stt import transcribe
    from voice.wakeword import run_wake_loop

    # Try VAD recording first; fall back to fixed-duration
    try:
        import importlib.util
        use_vad = importlib.util.find_spec("webrtcvad") is not None
        if use_vad:
            logger.info("VAD-based recording enabled (webrtcvad)")
        else:
            logger.info("webrtcvad not found; using fixed-duration recording")
    except Exception:
        use_vad = False
        logger.info("VAD not available; using fixed-duration recording")

    # Pre-synthesize "listening" chime at boot for instant playback on wake.
    # This avoids a ~200ms TTS latency each time the wake word fires.
    _listening_chime_wav: str | None = None
    try:
        from voice.tts import synthesize as _synth
        _listening_chime_wav = _synth("Listening, sir.", voice=settings.TTS_VOICE)
        if _listening_chime_wav:
            logger.info("Listening chime pre-synthesized: %s", _listening_chime_wav)
    except Exception as e:
        logger.debug("Failed to pre-synthesize listening chime: %s", e)

    def on_wake():
        if stop_event.is_set():
            return
        status_cb("Listening (recording)")

        # Play the pre-synthesized chime for immediate feedback
        if _listening_chime_wav:
            try:
                from audio.output import play_wav
                play_wav(_listening_chime_wav)
            except Exception:
                pass
        device_index = get_default_input_index()
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            wav_path = f.name
        try:
            if use_vad:
                from audio.vad import record_with_vad
                success = record_with_vad(
                    wav_path,
                    device_index=device_index,
                )
            else:
                from audio.input import record_to_file
                success = record_to_file(
                    wav_path,
                    duration_sec=settings.RECORD_DURATION_SEC,
                    device_index=device_index,
                )

            if not success:
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

    # ── Preflight checks ──────────────────────────────────────────────
    try:
        from utils.autoconfig import run_preflight
        await loop.run_in_executor(
            None, run_preflight, True, True,
        )
    except Exception as e:
        logger.debug("Preflight skipped: %s", e)

    # ── Bluetooth auto-reconnect daemon ───────────────────────────────
    try:
        from audio.bluetooth import start_bt_auto_reconnect

        def _on_bt_reconnect():
            try:
                wav = synthesize("Bluetooth reconnected, sir.", voice=settings.TTS_VOICE)
                if wav:
                    play_wav(wav)
            except Exception:
                pass

        def _on_bt_disconnect():
            try:
                wav = synthesize("Bluetooth audio disconnected. Attempting to reconnect.", voice=settings.TTS_VOICE)
                if wav:
                    play_wav(wav)
            except Exception:
                pass

        start_bt_auto_reconnect(
            on_reconnect=_on_bt_reconnect,
            on_disconnect=_on_bt_disconnect,
        )
    except Exception as e:
        logger.debug("BT auto-reconnect not started: %s", e)

    # ── STT warm-up (pre-load model to avoid cold-start latency) ──────
    try:
        from voice.stt import _get_model
        await loop.run_in_executor(None, _get_model, settings.STT_MODEL_SIZE)
        logger.info("STT model pre-loaded: %s", settings.STT_MODEL_SIZE)
    except Exception as e:
        logger.debug("STT warm-up failed: %s", e)

    # ── Background scene context thread ───────────────────────────────
    bg_scene_stop = _start_bg_scene_thread()

    # ── Ambient awareness thread (hands-free walk-around) ─────────────
    ambient_enabled = getattr(settings, "AMBIENT_AWARENESS_ENABLED", False) or getattr(settings, "PORTABLE_MODE", False)
    ambient_stop: threading.Event | None = None
    if ambient_enabled:
        ambient_stop = _start_ambient_thread(query_queue, loop, bridge)
        logger.info("Ambient awareness enabled (hands-free walk-around mode)")

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

                        # Check for environmental changes (person enter/leave, new objects)
                        change_alert = _check_proactive_changes(vision_data)

                        # Collision alerts (highest priority — immediate safety)
                        collision_alerts = vision_data.get("collision_alerts", [])
                        collision_say = None
                        for ca in collision_alerts:
                            if ca.get("severity") in ("critical", "warning"):
                                collision_say = ca.get("message", "")
                                break

                        # Proactive vitals alert
                        say_text = None
                        if collision_say:
                            say_text = collision_say  # collision overrides everything
                        elif change_alert:
                            say_text = change_alert
                        elif proactive_vitals and getattr(proactive_vitals, "fatigue_level", "unknown") in ("moderate", "severe"):
                            say_text = "Sir, you appear quite fatigued. Might I suggest a brief respite. Even Mr. Stark took the occasional break."
                        elif proactive_vitals and getattr(proactive_vitals, "posture_label", "unknown") == "poor":
                            say_text = "Sir, your posture could use some attention. Perhaps a stretch is in order."
                        elif "person" in proactive_desc.lower():
                            say_text = "Sir, you've been at your desk for quite some time. Might I suggest a brief respite?"

                        # Proactive threat alert (overrides other messages if critical)
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

                        # Broadcast vitals/threat/perception to PWA
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
                        if bridge is not None and collision_alerts:
                            await bridge.broadcast({
                                "type": "collision_alerts",
                                "data": collision_alerts,
                            })
                        # Broadcast perception summary
                        perception_text = vision_data.get("perception_text", "")
                        if bridge is not None and perception_text:
                            await bridge.broadcast({
                                "type": "perception",
                                "data": {"summary": perception_text},
                            })
                    except Exception as e:
                        logger.debug("Proactive vision check failed: %s", e)

                    idle_since = time.monotonic()
                continue

            # ── Handle ambient awareness events (hands-free) ──────
            if _is_ambient_event(str(query)):
                event_type, detail = _parse_ambient_event(str(query))
                logger.debug("Ambient event: %s — %s", event_type, detail)
                try:
                    await _handle_ambient_event(event_type, detail, bridge, loop)
                except Exception as e:
                    logger.debug("Ambient event handling failed: %s", e)
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

            # Always inject background scene context for spatial awareness.
            # This gives the LLM a continuous "sense of the room" without
            # needing explicit vision queries — like how MCU Jarvis always
            # knows what's happening around Tony.
            vision_description = get_bg_scene()
            vitals_text = None
            threat_text = None

            # For explicit vision queries, run full enriched scan (fresh frame)
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

                    # Inject perception context (ego-motion, trajectories, collisions)
                    perception_text = vision_data.get("perception_text", "")
                    if perception_text:
                        vision_description = (
                            (vision_description or "")
                            + f" Motion: {perception_text}"
                        )
                except Exception:
                    vision_description = await loop.run_in_executor(
                        None, run_tool, "vision_analyze", {},
                    )
                    vitals_text = None
                    threat_text = None
                await _thinking_async(bridge, "vision_done", "Environment analyzed")

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
                        await _verbal_error(bridge, "Brief comms glitch, sir. Retrying.", loop)
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
            # If this was a vision turn, tag the response so the LLM knows
            # any scene observations are from a *past* snapshot, not current.
            # This prevents it from parroting old "I see a person" responses
            # when the user moves the camera and asks again.
            if vision_description is not None:
                short_term.append({"role": "assistant", "content": final, "_vision_turn": True})
            else:
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
        bg_scene_stop.set()
        if ambient_stop is not None:
            ambient_stop.set()
        _stop_ambient_thread()
        try:
            from audio.bluetooth import stop_bt_auto_reconnect
            stop_bt_auto_reconnect()
        except Exception:
            pass
        save_session(memory)
