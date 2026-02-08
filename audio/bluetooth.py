"""Device enumeration, default sink/source hints, and BT reconnection (pactl/wpctl)."""

import logging
import subprocess

logger = logging.getLogger(__name__)


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
