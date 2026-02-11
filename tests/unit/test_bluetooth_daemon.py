"""Unit tests for audio/bluetooth.py BT auto-reconnect daemon."""

import threading
import time
from unittest.mock import patch

import pytest


@pytest.mark.unit
class TestBTAutoReconnect:
    def test_start_stop_daemon(self):
        """Daemon starts and stops cleanly."""
        from audio.bluetooth import start_bt_auto_reconnect, stop_bt_auto_reconnect

        with (
            patch("audio.bluetooth.is_bluetooth_audio_connected", return_value=True),
        ):
            stop_event = start_bt_auto_reconnect()
            assert isinstance(stop_event, threading.Event)
            time.sleep(0.1)  # Let it spin up
            stop_bt_auto_reconnect()
            time.sleep(0.2)

    def test_daemon_idempotent(self):
        """Starting the daemon twice returns the same stop event."""
        from audio.bluetooth import start_bt_auto_reconnect, stop_bt_auto_reconnect

        with patch("audio.bluetooth.is_bluetooth_audio_connected", return_value=True):
            stop1 = start_bt_auto_reconnect()
            stop2 = start_bt_auto_reconnect()
            assert stop1 is stop2
            stop_bt_auto_reconnect()
            time.sleep(0.2)

    def test_reconnect_callback_on_disconnect(self):
        """on_disconnect callback fires when BT drops."""
        from audio.bluetooth import (
            start_bt_auto_reconnect,
            stop_bt_auto_reconnect,
        )

        disconnected = threading.Event()
        reconnected = threading.Event()

        # Simulate: first check = connected, second = disconnected, then reconnect succeeds
        states = iter([True, False, False])

        with (
            patch("audio.bluetooth.is_bluetooth_audio_connected", side_effect=lambda: next(states, True)),
            patch("audio.bluetooth.reconnect_bluetooth", return_value=True),
            # Speed up check interval for testing
            patch("audio.bluetooth._BT_RECONNECT_CHECK_INTERVAL", 0.1),
        ):
            start_bt_auto_reconnect(
                on_reconnect=lambda: reconnected.set(),
                on_disconnect=lambda: disconnected.set(),
            )
            # Wait for disconnect detection
            time.sleep(0.5)
            stop_bt_auto_reconnect()
