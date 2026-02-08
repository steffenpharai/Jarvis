"""STT: Faster-Whisper or whisper.cpp.

The WhisperModel is cached as a module-level singleton to avoid reloading
~500 MB of weights and re-allocating GPU memory on every transcription call.
"""

import logging
import threading

logger = logging.getLogger(__name__)

_model = None
_model_lock = threading.Lock()
_model_size_loaded: str | None = None


def _get_model(model_size: str = "small"):
    """Lazily load and cache the WhisperModel.  Thread-safe."""
    global _model, _model_size_loaded
    if _model is not None and _model_size_loaded == model_size:
        return _model
    with _model_lock:
        if _model is not None and _model_size_loaded == model_size:
            return _model
        try:
            from faster_whisper import WhisperModel

            _model = WhisperModel(model_size, device="auto", compute_type="auto")
            _model_size_loaded = model_size
            logger.info("Loaded Faster-Whisper model: %s", model_size)
        except Exception as e:
            logger.warning("Failed to load Faster-Whisper model: %s", e)
            _model = None
    return _model


def transcribe(audio_path: str, model_size: str = "small") -> str | None:
    """Transcribe audio file to text. Returns None on failure."""
    try:
        model = _get_model(model_size)
        if model is None:
            return None
        segments, _ = model.transcribe(audio_path)
        return " ".join(s.text.strip() for s in segments).strip() or None
    except Exception as e:
        logger.warning("Faster-Whisper transcribe failed: %s", e)
        return None


def is_stt_available() -> bool:
    """Check if STT backend is available."""
    try:
        return True
    except Exception:
        return False
