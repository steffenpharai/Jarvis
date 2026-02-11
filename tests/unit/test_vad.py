"""Unit tests for audio/vad.py (VAD-based recording)."""

from unittest.mock import patch

import pytest


@pytest.mark.unit
class TestVADRecording:
    def test_fallback_when_webrtcvad_missing(self, tmp_path):
        """When webrtcvad is not installed, falls back to fixed-duration recording."""
        from audio.vad import _fallback_record

        wav_path = tmp_path / "test.wav"

        with patch("audio.input.record_to_file", return_value=True) as mock_rec:
            result = _fallback_record(wav_path)
            assert result is True
            mock_rec.assert_called_once()

    def test_record_with_vad_no_webrtcvad(self, tmp_path):
        """record_with_vad falls back gracefully when webrtcvad is missing."""
        import importlib

        wav_path = tmp_path / "test.wav"

        # Simulate webrtcvad not being available
        with patch.dict("sys.modules", {"webrtcvad": None}):
            with patch("audio.input.record_to_file", return_value=True):
                from audio import vad
                importlib.reload(vad)
                result = vad.record_with_vad(wav_path)
                assert result is True


@pytest.mark.unit
class TestVADConstants:
    def test_constants_reasonable(self):
        from audio.vad import (
            FRAME_DURATION_MS,
            MAX_RECORD_SEC,
            MIN_RECORD_SEC,
            MIN_VOICED_FRAMES,
            SILENCE_THRESHOLD_SEC,
            VAD_AGGRESSIVENESS,
        )
        assert 0 <= VAD_AGGRESSIVENESS <= 3
        assert FRAME_DURATION_MS in (10, 20, 30)
        assert 0.5 <= SILENCE_THRESHOLD_SEC <= 5.0
        assert MAX_RECORD_SEC > MIN_RECORD_SEC
        assert MIN_VOICED_FRAMES >= 1
