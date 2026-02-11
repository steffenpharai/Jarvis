"""Unit tests for utils/autoconfig.py (preflight checks)."""

from unittest.mock import patch

import pytest


@pytest.mark.unit
class TestPreflight:
    def test_check_camera_with_video_devices(self):
        from utils.autoconfig import _check_camera

        with patch("os.listdir", return_value=["video0", "video1"]):
            with patch("os.path.isdir", return_value=True):
                ok, msg = _check_camera()
                assert ok is True
                assert "video0" in msg

    def test_check_camera_no_devices(self):
        from utils.autoconfig import _check_camera

        with patch("os.listdir", return_value=[]):
            with patch("os.path.isdir", return_value=True):
                ok, msg = _check_camera()
                assert ok is False

    def test_check_bluetooth_connected(self):
        from utils.autoconfig import _check_bluetooth

        with (
            patch("audio.bluetooth.is_bluetooth_audio_connected", return_value=True),
            patch("audio.bluetooth.get_default_sink_name", return_value="bluez_out"),
        ):
            ok, msg = _check_bluetooth()
            assert ok is True
            assert "bluez_out" in msg

    def test_check_bluetooth_disconnected(self):
        from utils.autoconfig import _check_bluetooth

        with (
            patch("audio.bluetooth.is_bluetooth_audio_connected", return_value=False),
            patch("audio.bluetooth.get_default_sink_name", return_value="alsa_out"),
        ):
            ok, msg = _check_bluetooth()
            assert ok is False

    def test_check_tts_voice_file_exists(self):
        from utils.autoconfig import _check_tts_voice

        with patch("os.path.isfile", return_value=True):
            ok, msg = _check_tts_voice()
            assert ok is True

    def test_check_yolo_engine(self):
        from utils.autoconfig import _check_yolo_engine

        with patch("config.settings.yolo_engine_exists", return_value=True):
            ok, msg = _check_yolo_engine()
            assert ok is True

    def test_run_preflight_all_pass(self):
        from utils.autoconfig import run_preflight

        with (
            patch("utils.autoconfig._check_ollama", return_value=(True, "OK")),
            patch("utils.autoconfig._check_camera", return_value=(True, "OK")),
            patch("utils.autoconfig._check_bluetooth", return_value=(True, "OK")),
            patch("utils.autoconfig._check_audio_devices", return_value=(True, "OK")),
            patch("utils.autoconfig._check_yolo_engine", return_value=(True, "OK")),
            patch("utils.autoconfig._check_tts_voice", return_value=(True, "OK")),
        ):
            results = run_preflight(verbose=False, speak_status=False)
            assert all(ok for ok, _ in results.values())
            assert len(results) == 6

    def test_run_preflight_some_fail(self):
        from utils.autoconfig import run_preflight

        with (
            patch("utils.autoconfig._check_ollama", return_value=(True, "OK")),
            patch("utils.autoconfig._check_camera", return_value=(False, "No camera")),
            patch("utils.autoconfig._check_bluetooth", return_value=(False, "No BT")),
            patch("utils.autoconfig._check_audio_devices", return_value=(True, "OK")),
            patch("utils.autoconfig._check_yolo_engine", return_value=(True, "OK")),
            patch("utils.autoconfig._check_tts_voice", return_value=(True, "OK")),
        ):
            results = run_preflight(verbose=False, speak_status=False)
            failures = [name for name, (ok, _) in results.items() if not ok]
            assert len(failures) == 2
            assert "camera" in failures
            assert "bluetooth" in failures
