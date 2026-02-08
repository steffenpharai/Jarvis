"""Unit tests for audio/output.py, audio/bluetooth.py, audio/input.py (mocked)."""

import subprocess
from unittest.mock import MagicMock, patch

import pytest
from audio.bluetooth import (
    get_default_sink_name,
    get_default_source_name,
    is_bluetooth_audio_connected,
    reconnect_bluetooth,
)
from audio.output import play_wav

# ── audio/output.py ───────────────────────────────────────────────────


@pytest.mark.unit
class TestPlayWav:
    def test_play_success(self, tmp_path):
        wav_file = tmp_path / "test.wav"
        wav_file.write_bytes(b"RIFF" + b"\x00" * 100)
        proc = subprocess.CompletedProcess([], 0)
        with patch("audio.output.subprocess.run", return_value=proc):
            assert play_wav(wav_file) is True

    def test_play_file_not_found(self, tmp_path):
        result = play_wav(tmp_path / "nonexistent.wav")
        assert result is False

    def test_play_aplay_fails(self, tmp_path):
        wav_file = tmp_path / "test.wav"
        wav_file.write_bytes(b"RIFF")
        with patch("audio.output.subprocess.run", side_effect=subprocess.CalledProcessError(1, [])):
            assert play_wav(wav_file) is False

    def test_play_aplay_not_installed(self, tmp_path):
        wav_file = tmp_path / "test.wav"
        wav_file.write_bytes(b"RIFF")
        with patch("audio.output.subprocess.run", side_effect=FileNotFoundError):
            assert play_wav(wav_file) is False

    def test_play_timeout(self, tmp_path):
        wav_file = tmp_path / "test.wav"
        wav_file.write_bytes(b"RIFF")
        with patch("audio.output.subprocess.run", side_effect=subprocess.TimeoutExpired([], 30)):
            assert play_wav(wav_file) is False


# ── audio/bluetooth.py ────────────────────────────────────────────────


@pytest.mark.unit
class TestBluetooth:
    def test_default_sink_success(self):
        proc = subprocess.CompletedProcess([], 0, stdout="bluez_output.pixel_buds\n", stderr="")
        with patch("audio.bluetooth.subprocess.run", return_value=proc):
            result = get_default_sink_name()
            assert result == "bluez_output.pixel_buds"

    def test_default_sink_empty(self):
        proc = subprocess.CompletedProcess([], 0, stdout="", stderr="")
        with patch("audio.bluetooth.subprocess.run", return_value=proc):
            result = get_default_sink_name()
            # Empty string => None (due to `or None`)
            assert result is None

    def test_default_sink_pactl_missing(self):
        with patch("audio.bluetooth.subprocess.run", side_effect=FileNotFoundError):
            result = get_default_sink_name()
            assert result is None

    def test_default_source_success(self):
        proc = subprocess.CompletedProcess([], 0, stdout="alsa_input.usb_mic\n", stderr="")
        with patch("audio.bluetooth.subprocess.run", return_value=proc):
            result = get_default_source_name()
            assert result == "alsa_input.usb_mic"

    def test_default_source_failure(self):
        proc = subprocess.CompletedProcess([], 1, stdout="", stderr="error")
        with patch("audio.bluetooth.subprocess.run", return_value=proc):
            result = get_default_source_name()
            assert result is None


@pytest.mark.unit
class TestBluetoothReconnect:
    def test_reconnect_success(self):
        connect_proc = subprocess.CompletedProcess([], 0, stdout="Connection successful", stderr="")
        with patch("audio.bluetooth.subprocess.run", return_value=connect_proc):
            assert reconnect_bluetooth("AA:BB:CC:DD:EE:FF") is True

    def test_reconnect_failure(self):
        fail_proc = subprocess.CompletedProcess([], 1, stdout="Failed to connect", stderr="")
        with patch("audio.bluetooth.subprocess.run", return_value=fail_proc):
            assert reconnect_bluetooth("AA:BB:CC:DD:EE:FF") is False

    def test_reconnect_auto_discover(self):
        paired_proc = subprocess.CompletedProcess(
            [], 0, stdout="Device AA:BB:CC:DD:EE:FF Pixel Buds\n", stderr=""
        )
        connect_proc = subprocess.CompletedProcess([], 0, stdout="Connection successful", stderr="")
        with patch("audio.bluetooth.subprocess.run", side_effect=[paired_proc, connect_proc]):
            assert reconnect_bluetooth() is True

    def test_reconnect_no_paired_devices(self):
        no_paired = subprocess.CompletedProcess([], 0, stdout="", stderr="")
        with patch("audio.bluetooth.subprocess.run", return_value=no_paired):
            assert reconnect_bluetooth() is False


@pytest.mark.unit
class TestIsBluetoothConnected:
    def test_connected_via_sink(self):
        with patch("audio.bluetooth.get_default_sink_name", return_value="bluez_output.pixel_buds"):
            assert is_bluetooth_audio_connected() is True

    def test_connected_via_source(self):
        with (
            patch("audio.bluetooth.get_default_sink_name", return_value="alsa_output.pci"),
            patch("audio.bluetooth.get_default_source_name", return_value="bluez_input.hfp"),
        ):
            assert is_bluetooth_audio_connected() is True

    def test_not_connected(self):
        with (
            patch("audio.bluetooth.get_default_sink_name", return_value="alsa_output.pci"),
            patch("audio.bluetooth.get_default_source_name", return_value="alsa_input.usb"),
        ):
            assert is_bluetooth_audio_connected() is False


# ── audio/input.py ────────────────────────────────────────────────────


@pytest.mark.unit
class TestAudioInput:
    def test_list_input_devices_mocked(self):
        from audio.input import list_input_devices

        mock_devices = [
            {"name": "USB Mic", "max_input_channels": 1, "default_samplerate": 16000},
            {"name": "Monitor", "max_input_channels": 0, "default_samplerate": 44100},
            {"name": "BT HFP", "max_input_channels": 1, "default_samplerate": 16000},
        ]
        mock_sd = MagicMock()
        mock_sd.query_devices.return_value = mock_devices
        with patch.dict("sys.modules", {"sounddevice": mock_sd}):
            devices = list_input_devices()
            # Only input devices (channels > 0)
            assert len(devices) == 2
            assert devices[0]["name"] == "USB Mic"
            assert devices[1]["name"] == "BT HFP"

    def test_get_default_input_index_mocked(self):
        from audio.input import get_default_input_index

        mock_sd = MagicMock()
        mock_sd.default.device = (2, 3)  # (input, output)
        with patch.dict("sys.modules", {"sounddevice": mock_sd}):
            idx = get_default_input_index()
            assert idx == 2
