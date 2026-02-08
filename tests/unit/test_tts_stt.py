"""Unit tests for voice/tts.py and voice/stt.py (mocked subprocess/model)."""

import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from voice.stt import is_stt_available, transcribe
from voice.tts import is_tts_available, synthesize

# ── TTS ───────────────────────────────────────────────────────────────


@pytest.mark.unit
class TestTTS:
    def test_synthesize_success(self, tmp_path):
        """Piper subprocess returns 0 and creates WAV file."""
        out_path = tmp_path / "jarvis_tts.wav"
        out_path.write_bytes(b"RIFF" + b"\x00" * 100)  # fake WAV

        def fake_run(cmd, **kwargs):
            # Create the output file that Piper would produce
            for i, arg in enumerate(cmd):
                if arg == "--output_file" and i + 1 < len(cmd):
                    Path(cmd[i + 1]).write_bytes(b"RIFF" + b"\x00" * 100)
            return subprocess.CompletedProcess(cmd, 0)

        with patch("voice.tts.subprocess.run", side_effect=fake_run):
            result = synthesize("Hello Sir.", out_dir=tmp_path)
            assert result is not None
            assert result.exists()

    def test_synthesize_empty_text(self):
        assert synthesize("") is None
        assert synthesize("   ") is None

    def test_synthesize_piper_not_found(self, tmp_path):
        with patch("voice.tts.subprocess.run", side_effect=FileNotFoundError):
            result = synthesize("Hello", out_dir=tmp_path)
            assert result is None

    def test_synthesize_piper_failure(self, tmp_path):
        proc = subprocess.CompletedProcess([], 1, stderr=b"error")
        with patch("voice.tts.subprocess.run", return_value=proc):
            result = synthesize("Hello", out_dir=tmp_path)
            assert result is None

    def test_synthesize_timeout(self, tmp_path):
        with patch("voice.tts.subprocess.run", side_effect=subprocess.TimeoutExpired([], 30)):
            result = synthesize("Hello", out_dir=tmp_path)
            assert result is None

    def test_is_tts_available_yes(self):
        proc = subprocess.CompletedProcess([], 0)
        with patch("voice.tts.subprocess.run", return_value=proc):
            assert is_tts_available() is True

    def test_is_tts_available_no(self):
        with patch("voice.tts.subprocess.run", side_effect=FileNotFoundError):
            assert is_tts_available() is False


# ── STT ───────────────────────────────────────────────────────────────


@pytest.mark.unit
class TestSTT:
    def test_transcribe_success(self, tmp_path):
        """Faster-Whisper returns segments -> text."""
        audio_file = tmp_path / "test.wav"
        audio_file.write_bytes(b"RIFF" + b"\x00" * 100)

        mock_segment = MagicMock()
        mock_segment.text = " Hello there "
        mock_model = MagicMock()
        mock_model.transcribe.return_value = ([mock_segment], None)

        with patch("voice.stt._get_model", return_value=mock_model):
            result = transcribe(str(audio_file))
            assert result == "Hello there"

    def test_transcribe_empty_segments(self, tmp_path):
        audio_file = tmp_path / "test.wav"
        audio_file.write_bytes(b"RIFF" + b"\x00" * 100)

        mock_model = MagicMock()
        mock_model.transcribe.return_value = ([], None)

        with patch("voice.stt._get_model", return_value=mock_model):
            result = transcribe(str(audio_file))
            assert result is None

    def test_transcribe_model_not_loaded(self, tmp_path):
        audio_file = tmp_path / "test.wav"
        audio_file.write_bytes(b"RIFF" + b"\x00" * 100)

        with patch("voice.stt._get_model", return_value=None):
            result = transcribe(str(audio_file))
            assert result is None

    def test_transcribe_exception(self, tmp_path):
        audio_file = tmp_path / "test.wav"
        audio_file.write_bytes(b"RIFF" + b"\x00" * 100)

        mock_model = MagicMock()
        mock_model.transcribe.side_effect = RuntimeError("CUDA OOM")

        with patch("voice.stt._get_model", return_value=mock_model):
            result = transcribe(str(audio_file))
            assert result is None

    def test_is_stt_available(self):
        assert is_stt_available() is True

    def test_multiple_segments(self, tmp_path):
        audio_file = tmp_path / "test.wav"
        audio_file.write_bytes(b"RIFF")

        seg1 = MagicMock()
        seg1.text = " Hello "
        seg2 = MagicMock()
        seg2.text = " Sir "

        mock_model = MagicMock()
        mock_model.transcribe.return_value = ([seg1, seg2], None)

        with patch("voice.stt._get_model", return_value=mock_model):
            result = transcribe(str(audio_file))
            assert result == "Hello Sir"
