"""Startup auto-configuration and preflight checks.

Called before entering --serve / --orchestrator modes to verify all subsystems
and auto-fix what can be fixed without human intervention.

Design pattern: Tesla vehicle boot sequence — verify all subsystems, log status,
auto-recover where possible, provide verbal feedback on failures.
"""

from __future__ import annotations

import logging
import os
import time

logger = logging.getLogger(__name__)


def _check_ollama(base_url: str, model: str) -> tuple[bool, str]:
    """Check Ollama availability and model presence.  Waits up to 30s."""
    try:
        import requests
    except ImportError:
        return False, "requests library not installed"

    # Wait for Ollama to come up (systemd may still be starting)
    for attempt in range(15):
        try:
            r = requests.get(f"{base_url.rstrip('/')}/api/tags", timeout=3)
            if r.status_code == 200:
                data = r.json()
                models = [
                    (m.get("name", "").removesuffix(":latest"))
                    for m in (data.get("models") or [])
                ]
                want = model.removesuffix(":latest")
                if want in models or model in models:
                    return True, f"Ollama OK, model {model} available"
                return True, f"Ollama OK, but model {model} not found (available: {models})"
            break
        except Exception:
            if attempt < 14:
                time.sleep(2)
    return False, f"Ollama not reachable at {base_url} after 30s"


def _check_camera() -> tuple[bool, str]:
    """Check if a USB camera is available."""
    # Check /dev/video* devices
    video_devs = [f for f in os.listdir("/dev") if f.startswith("video")] if os.path.isdir("/dev") else []
    if not video_devs:
        return False, "No /dev/video* devices found"
    return True, f"Camera devices: {', '.join(sorted(video_devs))}"


def _check_bluetooth() -> tuple[bool, str]:
    """Check Bluetooth status and connected audio devices."""
    try:
        from audio.bluetooth import get_default_sink_name, is_bluetooth_audio_connected
        connected = is_bluetooth_audio_connected()
        sink = get_default_sink_name()
        if connected:
            return True, f"BT audio connected (sink: {sink})"
        return False, f"BT audio not connected (sink: {sink})"
    except Exception as e:
        return False, f"BT check failed: {e}"


def _check_audio_devices() -> tuple[bool, str]:
    """Check that at least one audio input device exists."""
    try:
        from audio.input import list_input_devices
        devices = list_input_devices()
        if devices:
            names = [d.get("name", "?") for d in devices[:3]]
            return True, f"{len(devices)} input device(s): {', '.join(names)}"
        return False, "No audio input devices found"
    except Exception as e:
        return False, f"Audio device check failed: {e}"


def _check_yolo_engine() -> tuple[bool, str]:
    """Check if the YOLOE TensorRT engine exists."""
    from config import settings
    if settings.yolo_engine_exists():
        return True, f"YOLOE engine: {settings.YOLOE_ENGINE_PATH}"
    return False, f"YOLOE engine missing: {settings.YOLOE_ENGINE_PATH}"


def _check_tts_voice() -> tuple[bool, str]:
    """Check if the TTS voice model file exists."""
    from config import settings
    if os.path.isfile(settings.TTS_VOICE):
        return True, f"TTS voice: {settings.TTS_VOICE}"
    # Check if it's a built-in model name (not a path)
    if os.sep not in settings.TTS_VOICE:
        return True, f"TTS voice (built-in): {settings.TTS_VOICE}"
    return False, f"TTS voice missing: {settings.TTS_VOICE}"


def run_preflight(
    verbose: bool = False,
    speak_status: bool = False,
) -> dict[str, tuple[bool, str]]:
    """Run all preflight checks.  Returns dict of check_name -> (ok, message).

    Parameters
    ----------
    verbose : bool
        Log all results at INFO level (not just failures).
    speak_status : bool
        If True, synthesize a TTS summary of any failures.
    """
    from config import settings

    results: dict[str, tuple[bool, str]] = {}

    checks = [
        ("ollama", lambda: _check_ollama(settings.OLLAMA_BASE_URL, settings.OLLAMA_MODEL)),
        ("camera", _check_camera),
        ("bluetooth", _check_bluetooth),
        ("audio_input", _check_audio_devices),
        ("yolo_engine", _check_yolo_engine),
        ("tts_voice", _check_tts_voice),
    ]

    for name, check_fn in checks:
        try:
            ok, msg = check_fn()
        except Exception as e:
            ok, msg = False, f"Check crashed: {e}"
        results[name] = (ok, msg)

        if ok:
            if verbose:
                logger.info("Preflight [%s]: OK — %s", name, msg)
        else:
            logger.warning("Preflight [%s]: FAIL — %s", name, msg)

    # Summary
    failures = [name for name, (ok, _) in results.items() if not ok]
    if failures:
        logger.warning("Preflight: %d/%d checks failed: %s", len(failures), len(results), ", ".join(failures))
        if speak_status:
            _speak_failures(failures)
    else:
        logger.info("Preflight: all %d checks passed", len(results))
        if speak_status:
            _speak_ok()

    return results


def _speak_failures(failures: list[str]) -> None:
    """Best-effort TTS notification of preflight failures."""
    try:
        from audio.output import play_wav
        from config import settings
        from voice.tts import synthesize

        # Map check names to human-friendly descriptions
        friendly = {
            "ollama": "language model",
            "camera": "camera",
            "bluetooth": "Bluetooth audio",
            "audio_input": "microphone",
            "yolo_engine": "vision engine",
            "tts_voice": "voice model",
        }
        names = [friendly.get(f, f) for f in failures[:3]]
        text = f"Sir, {len(failures)} subsystem{'s' if len(failures) > 1 else ''} offline: {', '.join(names)}. I shall continue with reduced capability."
        wav = synthesize(text, voice=settings.TTS_VOICE)
        if wav:
            play_wav(wav)
    except Exception as e:
        logger.debug("Preflight TTS notification failed: %s", e)


def _speak_ok() -> None:
    """Best-effort TTS notification that all systems are nominal."""
    try:
        from audio.output import play_wav
        from config import settings
        from voice.tts import synthesize

        wav = synthesize("All systems nominal, sir. At your service.", voice=settings.TTS_VOICE)
        if wav:
            play_wav(wav)
    except Exception as e:
        logger.debug("Preflight TTS ok notification failed: %s", e)
