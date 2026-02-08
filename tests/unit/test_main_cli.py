"""Unit tests for main.py CLI argument parsing and dispatch (mocked hardware)."""

from unittest.mock import patch

import pytest
from main import _set_gui_status, main, parse_args


@pytest.mark.unit
class TestParseArgs:
    def test_no_args(self):
        with patch("sys.argv", ["main.py"]):
            args = parse_args()
            assert args.dry_run is False
            assert args.test_audio is False
            assert args.voice_only is False
            assert args.gui is False
            assert args.e2e is False
            assert args.orchestrator is False
            assert args.serve is False
            assert args.yolo_visualize is False
            assert args.one_shot is None

    def test_dry_run(self):
        with patch("sys.argv", ["main.py", "--dry-run"]):
            args = parse_args()
            assert args.dry_run is True

    def test_one_shot_default(self):
        with patch("sys.argv", ["main.py", "--one-shot"]):
            args = parse_args()
            assert args.one_shot == "What time is it?"

    def test_one_shot_custom(self):
        with patch("sys.argv", ["main.py", "--one-shot", "Hello"]):
            args = parse_args()
            assert args.one_shot == "Hello"

    def test_verbose(self):
        with patch("sys.argv", ["main.py", "-v"]):
            args = parse_args()
            assert args.verbose is True

    def test_serve(self):
        with patch("sys.argv", ["main.py", "--serve"]):
            args = parse_args()
            assert args.serve is True


@pytest.mark.unit
class TestMainDryRun:
    def test_dry_run_returns_0(self):
        with patch("sys.argv", ["main.py", "--dry-run"]):
            result = main()
            assert result == 0


@pytest.mark.unit
class TestMainIdle:
    def test_no_args_returns_0(self):
        with patch("sys.argv", ["main.py"]):
            result = main()
            assert result == 0


@pytest.mark.unit
class TestSetGuiStatus:
    def test_set_gui_status_no_gui(self):
        """Should not raise even when GUI module not available."""
        _set_gui_status("Listening")

    def test_set_gui_status_with_gui(self):
        with patch("gui.overlay.set_status") as mock_set:
            _set_gui_status("Thinking")
            mock_set.assert_called_once_with("Thinking")


@pytest.mark.unit
class TestMainTestAudio:
    def test_test_audio(self):
        with patch("sys.argv", ["main.py", "--test-audio"]):
            with patch("audio.input.list_input_devices", return_value=[]):
                with patch("audio.input.get_default_input_index", return_value=None):
                    with patch("audio.bluetooth.get_default_sink_name", return_value="test_sink"):
                        with patch("audio.bluetooth.get_default_source_name", return_value=None):
                            result = main()
                            assert result == 0
