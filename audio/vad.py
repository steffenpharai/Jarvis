"""Voice Activity Detection (VAD) based recording.

Uses Google WebRTC VAD — the industry standard for end-of-speech detection
(used by Google Assistant, Amazon Alexa, and most modern voice assistants).

Benefits over fixed-duration recording:
  - Stops recording as soon as the user finishes speaking (lower latency)
  - Handles variable-length queries naturally (short "What time is it?" to
    long "Remind me to call the plumber tomorrow at 3pm")
  - Graceful timeout prevents infinite recording if mic is noisy

Pattern: sliding window VAD with trailing silence threshold.
"""

from __future__ import annotations

import logging
import wave
from pathlib import Path

logger = logging.getLogger(__name__)

# ── Defaults ──────────────────────────────────────────────────────────

# VAD aggressiveness: 0 (least aggressive) to 3 (most aggressive filtering)
# 2 is the sweet spot for BT earbuds + ambient noise on Jetson
VAD_AGGRESSIVENESS = 2

# Frame duration in ms (must be 10, 20, or 30 for webrtcvad)
FRAME_DURATION_MS = 30

# How long silence must persist before we stop recording (seconds)
SILENCE_THRESHOLD_SEC = 1.5

# BT audio has higher latency (~200ms) which causes false end-of-speech.
# Use a longer silence threshold when BT audio is connected.
BT_SILENCE_THRESHOLD_SEC = 2.0

# Maximum recording duration (seconds) -- safety cap
MAX_RECORD_SEC = 15.0

# Minimum recording duration before we start checking for silence
MIN_RECORD_SEC = 0.5

# How many consecutive voiced frames needed to confirm speech started
MIN_VOICED_FRAMES = 3

# Pre-speech timeout: if no speech detected within this time, abort
PRE_SPEECH_TIMEOUT_SEC = 5.0


def record_with_vad(
    path: str | Path,
    sample_rate: int = 16000,
    device_index: int | None = None,
    aggressiveness: int = VAD_AGGRESSIVENESS,
    silence_threshold_sec: float | None = None,
    max_duration_sec: float = MAX_RECORD_SEC,
    min_duration_sec: float = MIN_RECORD_SEC,
    pre_speech_timeout_sec: float = PRE_SPEECH_TIMEOUT_SEC,
) -> bool:
    """Record audio using VAD to detect end-of-speech.

    Returns True if audio was captured, False on failure or silence-only.

    If ``silence_threshold_sec`` is None (default), automatically uses
    a longer threshold when Bluetooth audio is connected to compensate
    for BT codec latency that causes false end-of-speech detections.

    Algorithm:
    1. Wait for speech onset (MIN_VOICED_FRAMES consecutive voiced frames)
    2. Record while speech continues
    3. Stop after silence_threshold_sec of continuous silence
    4. Safety: stop after MAX_RECORD_SEC regardless
    """
    # Auto-detect BT audio and use longer silence threshold
    if silence_threshold_sec is None:
        try:
            from audio.bluetooth import is_bluetooth_audio_connected
            if is_bluetooth_audio_connected():
                silence_threshold_sec = BT_SILENCE_THRESHOLD_SEC
                logger.debug("BT audio detected: using %.1fs silence threshold", silence_threshold_sec)
            else:
                silence_threshold_sec = SILENCE_THRESHOLD_SEC
        except Exception:
            silence_threshold_sec = SILENCE_THRESHOLD_SEC
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    try:
        import sounddevice as sd
        import webrtcvad
    except ImportError as e:
        logger.warning("VAD recording requires webrtcvad: %s. Falling back to fixed-duration.", e)
        return _fallback_record(path, sample_rate=sample_rate, device_index=device_index)

    vad = webrtcvad.Vad(aggressiveness)

    # Frame size in samples
    frame_samples = int(sample_rate * FRAME_DURATION_MS / 1000)
    # Bytes per frame (16-bit mono)
    frame_bytes = frame_samples * 2

    # Silence counter
    frames_per_sec = 1000 / FRAME_DURATION_MS
    silence_frames_threshold = int(silence_threshold_sec * frames_per_sec)
    max_frames = int(max_duration_sec * frames_per_sec)
    min_frames = int(min_duration_sec * frames_per_sec)
    pre_speech_frames = int(pre_speech_timeout_sec * frames_per_sec)

    all_audio: list[bytes] = []
    voiced_count = 0
    silence_count = 0
    speech_started = False
    frame_count = 0

    try:
        with sd.RawInputStream(
            samplerate=sample_rate,
            blocksize=frame_samples,
            dtype="int16",
            channels=1,
            device=device_index,
        ) as stream:
            while frame_count < max_frames:
                data, overflowed = stream.read(frame_samples)
                if overflowed:
                    logger.debug("VAD: audio buffer overflow")

                frame_data = bytes(data)
                frame_count += 1

                # Ensure frame is correct length for WebRTC VAD
                if len(frame_data) < frame_bytes:
                    frame_data += b'\x00' * (frame_bytes - len(frame_data))
                elif len(frame_data) > frame_bytes:
                    frame_data = frame_data[:frame_bytes]

                is_speech = vad.is_speech(frame_data, sample_rate)

                if not speech_started:
                    if is_speech:
                        voiced_count += 1
                        all_audio.append(frame_data)
                        if voiced_count >= MIN_VOICED_FRAMES:
                            speech_started = True
                            silence_count = 0
                            logger.debug("VAD: speech detected (frame %d)", frame_count)
                    else:
                        voiced_count = 0
                        # Keep a small pre-buffer (last ~300ms before speech)
                        all_audio.append(frame_data)
                        if len(all_audio) > int(0.3 * frames_per_sec):
                            all_audio.pop(0)

                        # Timeout waiting for speech
                        if frame_count >= pre_speech_frames:
                            logger.debug("VAD: no speech detected within %.1fs", pre_speech_timeout_sec)
                            return False
                else:
                    all_audio.append(frame_data)
                    if is_speech:
                        silence_count = 0
                    else:
                        silence_count += 1

                    # Check for end of speech (enough silence after min duration)
                    if (
                        frame_count >= min_frames
                        and silence_count >= silence_frames_threshold
                    ):
                        logger.debug(
                            "VAD: end of speech (frame %d, %.1fs silence)",
                            frame_count, silence_count / frames_per_sec,
                        )
                        break

        if not speech_started or not all_audio:
            return False

        # Write WAV file
        raw_audio = b"".join(all_audio)
        with wave.open(str(path), "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(sample_rate)
            wav.writeframes(raw_audio)

        duration = len(all_audio) * FRAME_DURATION_MS / 1000
        logger.debug("VAD: recorded %.1fs of audio to %s", duration, path)
        return path.exists()

    except Exception as e:
        logger.warning("VAD recording failed: %s", e)
        return _fallback_record(path, sample_rate=sample_rate, device_index=device_index)


def _fallback_record(
    path: str | Path,
    sample_rate: int = 16000,
    device_index: int | None = None,
    duration_sec: float = 5.0,
) -> bool:
    """Fallback: fixed-duration recording when webrtcvad is not available."""
    from audio.input import record_to_file
    return record_to_file(path, duration_sec=duration_sec, sample_rate=sample_rate, device_index=device_index)
