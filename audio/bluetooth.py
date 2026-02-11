"""Device enumeration, default sink/source hints, BT reconnection, and auto-reconnect daemon.

Industry-standard pattern: continuously monitor BT audio connection state and
auto-reconnect with exponential backoff (similar to Tesla vehicle BT module,
Android BluetoothService, and BlueZ ``module-bluetooth-policy``).
"""

from __future__ import annotations

import logging
import subprocess
import threading
from collections.abc import Callable

logger = logging.getLogger(__name__)

# ── Auto-reconnect daemon ────────────────────────────────────────────

_bt_reconnect_stop: threading.Event | None = None
_bt_reconnect_thread: threading.Thread | None = None

# Rate limits for reconnect attempts
_BT_RECONNECT_BASE_DELAY = 2.0      # seconds
_BT_RECONNECT_MAX_DELAY = 60.0      # seconds
_BT_RECONNECT_CHECK_INTERVAL = 15.0  # how often to check connection


def start_bt_auto_reconnect(
    on_reconnect: Callable | None = None,
    on_disconnect: Callable | None = None,
) -> threading.Event:
    """Start a background daemon that monitors BT audio and auto-reconnects.

    Parameters
    ----------
    on_reconnect : callable, optional
        Called (no args) after a successful reconnect.  Typically used to
        synthesize "Bluetooth reconnected, sir" via TTS.
    on_disconnect : callable, optional
        Called (no args) when a disconnect is first detected.

    Returns the stop Event; set it to terminate the daemon.
    """
    global _bt_reconnect_stop, _bt_reconnect_thread

    if _bt_reconnect_thread is not None and _bt_reconnect_thread.is_alive():
        return _bt_reconnect_stop  # already running

    _bt_reconnect_stop = threading.Event()

    def _daemon():
        was_connected = is_bluetooth_audio_connected()
        consecutive_failures = 0
        while not _bt_reconnect_stop.is_set():
            _bt_reconnect_stop.wait(_BT_RECONNECT_CHECK_INTERVAL)
            if _bt_reconnect_stop.is_set():
                break

            connected = is_bluetooth_audio_connected()

            if connected:
                if not was_connected:
                    logger.info("BT audio reconnected")
                    consecutive_failures = 0
                    if on_reconnect:
                        try:
                            on_reconnect()
                        except Exception as e:
                            logger.debug("BT on_reconnect callback error: %s", e)
                was_connected = True
                continue

            # Disconnected
            if was_connected:
                logger.warning("BT audio disconnected — starting auto-reconnect")
                if on_disconnect:
                    try:
                        on_disconnect()
                    except Exception as e:
                        logger.debug("BT on_disconnect callback error: %s", e)
                was_connected = False
                consecutive_failures = 0

            # Attempt reconnect with exponential backoff
            delay = min(
                _BT_RECONNECT_BASE_DELAY * (2 ** consecutive_failures),
                _BT_RECONNECT_MAX_DELAY,
            )
            logger.info(
                "BT reconnect attempt %d (backoff %.1fs)",
                consecutive_failures + 1, delay,
            )
            if reconnect_bluetooth():
                logger.info("BT auto-reconnect succeeded")
                was_connected = True
                consecutive_failures = 0
                if on_reconnect:
                    try:
                        on_reconnect()
                    except Exception:
                        pass
            else:
                consecutive_failures += 1
                # Wait the backoff delay before next attempt
                _bt_reconnect_stop.wait(delay)

    _bt_reconnect_thread = threading.Thread(target=_daemon, daemon=True, name="bt-auto-reconnect")
    _bt_reconnect_thread.start()
    logger.info("BT auto-reconnect daemon started (check every %.0fs)", _BT_RECONNECT_CHECK_INTERVAL)
    return _bt_reconnect_stop


def stop_bt_auto_reconnect() -> None:
    """Stop the auto-reconnect daemon."""
    global _bt_reconnect_stop, _bt_reconnect_thread
    if _bt_reconnect_stop is not None:
        _bt_reconnect_stop.set()
    _bt_reconnect_thread = None


def get_default_sink_name() -> str | None:
    """Query default Pulse sink (e.g. Pixel Buds A2DP)."""
    try:
        out = subprocess.run(
            ["pactl", "get-default-sink"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        return out.stdout.strip() or None if out.returncode == 0 else None
    except Exception as e:
        logger.debug("pactl get-default-sink failed: %s", e)
        return None


def get_default_source_name() -> str | None:
    """Query default Pulse source (e.g. HFP mic or USB mic)."""
    try:
        out = subprocess.run(
            ["pactl", "get-default-source"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        return out.stdout.strip() or None if out.returncode == 0 else None
    except Exception as e:
        logger.debug("pactl get-default-source failed: %s", e)
        return None


def reconnect_bluetooth(mac: str | None = None) -> bool:
    """Attempt to reconnect a Bluetooth device via bluetoothctl.

    If *mac* is None, tries to reconnect the most recently connected device.
    Returns True if the connect command succeeds.
    """
    if not mac:
        # Try to find paired devices and reconnect the first one
        try:
            out = subprocess.run(
                ["bluetoothctl", "devices", "Paired"],
                capture_output=True, text=True, timeout=5,
            )
            if out.returncode != 0 or not out.stdout.strip():
                logger.debug("No paired BT devices found")
                return False
            # First paired device MAC
            line = out.stdout.strip().split("\n")[0]
            parts = line.split()
            if len(parts) >= 2:
                mac = parts[1]
            else:
                return False
        except Exception as e:
            logger.debug("BT device discovery failed: %s", e)
            return False

    try:
        result = subprocess.run(
            ["bluetoothctl", "connect", mac],
            capture_output=True, text=True, timeout=15,
        )
        if result.returncode == 0 and "successful" in result.stdout.lower():
            logger.info("BT reconnected: %s", mac)
            return True
        logger.debug("BT reconnect failed for %s: %s", mac, result.stdout)
        return False
    except Exception as e:
        logger.debug("BT reconnect error: %s", e)
        return False


def is_bluetooth_audio_connected() -> bool:
    """Check if any Bluetooth audio device is currently connected."""
    sink = get_default_sink_name()
    if sink and "bluez" in sink.lower():
        return True
    source = get_default_source_name()
    if source and "bluez" in source.lower():
        return True
    return False
