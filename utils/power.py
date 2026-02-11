"""nvpmodel -q, jetson_clocks, thermal check, battery, GPU utilisation for Jetson Orin."""

import logging
import os
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)


def get_power_mode() -> str | None:
    """Return current nvpmodel mode (e.g. MAXN_SUPER)."""
    try:
        out = subprocess.run(
            ["nvpmodel", "-q"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if out.returncode == 0:
            return out.stdout.strip() or None
        return None
    except FileNotFoundError:
        return None
    except Exception as e:
        logger.debug("nvpmodel -q failed: %s", e)
        return None


def get_tegrastats_sample() -> str | None:
    """One-line tegrastats sample (if available).

    Caches the result for 2 seconds to avoid spawning multiple
    tegrastats processes when ``get_system_stats`` and
    ``get_thermal_warning`` are called in sequence.
    """
    import time

    now = time.monotonic()
    prev_time = getattr(get_tegrastats_sample, "_cache_time", 0.0)
    prev_val = getattr(get_tegrastats_sample, "_cache_val", None)
    if prev_val is not None and (now - prev_time) < 2.0:
        return prev_val

    try:
        out = subprocess.run(
            ["tegrastats", "--interval", "1000"],
            capture_output=True,
            text=True,
            timeout=2,
        )
        line = out.stdout.strip().split("\n")[0].strip() if out.stdout else None
        result = line or None
    except Exception:
        result = None

    get_tegrastats_sample._cache_time = now
    get_tegrastats_sample._cache_val = result
    return result


def get_thermal_warning() -> str | None:
    """Return a warning message if thermal throttling is likely.

    Uses the cached tegrastats sample to avoid duplicate subprocess calls.
    Also checks sysfs thermal zones for a more reliable reading.
    """
    # Check sysfs thermal zone first (more reliable than tegrastats parsing)
    temp = get_thermal_temperature()
    if temp is not None and temp > 85.0:
        return f"High temperature: {temp:.0f}°C — consider cooling or reducing load."

    sample = get_tegrastats_sample()
    if not sample:
        return None
    lower = sample.lower()
    if "throttle" in lower or "thermal" in lower:
        return "Thermal/throttling indicated; consider cooling or reducing load."
    return None


def get_system_stats() -> str | None:
    """Return a one-line system stats string for LLM context (power mode + tegrastats).

    Uses the cached tegrastats sample to avoid spawning redundant processes.
    """
    parts = []
    pm = get_power_mode()
    if pm:
        parts.append(f"Power: {pm}")
    sample = get_tegrastats_sample()
    if sample:
        parts.append(f"Stats: {sample[:200]}")
    return "; ".join(parts) if parts else None


# ── Battery / power supply status ────────────────────────────────────


def get_battery_status() -> dict | None:
    """Read battery status from sysfs (for portable Jetson setups with UPS/battery).

    Returns dict with keys: present, capacity_pct, status, voltage_uv, current_ua.
    Returns None if no battery is detected.
    """
    power_supply_dir = Path("/sys/class/power_supply")
    if not power_supply_dir.is_dir():
        return None

    for supply in power_supply_dir.iterdir():
        supply_type_file = supply / "type"
        if not supply_type_file.exists():
            continue
        try:
            stype = supply_type_file.read_text().strip().lower()
        except Exception:
            continue
        if stype != "battery":
            continue

        result = {"present": True, "name": supply.name}
        for key, filename in [
            ("capacity_pct", "capacity"),
            ("status", "status"),
            ("voltage_uv", "voltage_now"),
            ("current_ua", "current_now"),
        ]:
            fpath = supply / filename
            if fpath.exists():
                try:
                    val = fpath.read_text().strip()
                    result[key] = int(val) if val.isdigit() else val
                except Exception:
                    pass
        return result

    return None


# ── GPU utilisation ──────────────────────────────────────────────────


def get_gpu_utilization() -> float | None:
    """Read GPU utilisation percentage from Jetson sysfs.

    Returns 0-100 float, or None if unavailable.
    """
    # Jetson Orin GPU load sysfs path
    gpu_load_paths = [
        "/sys/devices/gpu.0/load",
        "/sys/devices/platform/gpu.0/load",
        "/sys/devices/17000000.ga10b/load",
    ]
    for path in gpu_load_paths:
        if os.path.exists(path):
            try:
                with open(path) as f:
                    val = f.read().strip()
                # Value is typically 0-1000 (permille) on Jetson
                return float(val) / 10.0
            except Exception:
                pass

    # Fallback: try tegrastats parsing
    try:
        out = subprocess.run(
            ["tegrastats", "--interval", "1000"],
            capture_output=True, text=True, timeout=2,
        )
        if out.returncode == 0 and "GR3D" in out.stdout:
            # Parse "GR3D_FREQ 42%" pattern
            for part in out.stdout.split():
                if "%" in part and part.replace("%", "").isdigit():
                    return float(part.replace("%", ""))
    except Exception:
        pass
    return None


# ── Thermal reading ──────────────────────────────────────────────────


def get_thermal_temperature() -> float | None:
    """Read highest thermal zone temperature in Celsius. Returns None if unavailable."""
    thermal_dir = Path("/sys/class/thermal")
    if not thermal_dir.is_dir():
        return None
    max_temp = None
    for zone in thermal_dir.iterdir():
        temp_file = zone / "temp"
        if temp_file.exists():
            try:
                val = int(temp_file.read_text().strip())
                # Value is in millidegrees on most kernels
                temp_c = val / 1000.0 if val > 1000 else float(val)
                if max_temp is None or temp_c > max_temp:
                    max_temp = temp_c
            except Exception:
                pass
    return max_temp


def should_throttle_vision() -> bool:
    """Check if vision should be throttled due to thermal or power constraints.

    Returns True if temperature exceeds THERMAL_PAUSE_THRESHOLD or
    battery is critically low (<10%).
    """
    from config import settings

    threshold = getattr(settings, "THERMAL_PAUSE_THRESHOLD", 80.0)

    # Thermal check
    temp = get_thermal_temperature()
    if temp is not None and temp > threshold:
        logger.warning("Thermal throttle: %.1f°C > %.1f°C threshold", temp, threshold)
        return True

    # Battery check
    battery = get_battery_status()
    if battery and battery.get("capacity_pct") is not None:
        cap = battery["capacity_pct"]
        if isinstance(cap, (int, float)) and cap < 10:
            logger.warning("Battery critical: %s%%", cap)
            return True

    return False


def get_battery_summary() -> str | None:
    """Return a human-readable battery status string for LLM context / TTS.

    Returns None if no battery is detected.
    """
    battery = get_battery_status()
    if not battery or not battery.get("present"):
        return None

    parts = []
    cap = battery.get("capacity_pct")
    status = battery.get("status", "unknown")

    if cap is not None:
        parts.append(f"{cap}%")
    if isinstance(status, str) and status.lower() not in ("unknown", ""):
        parts.append(status.lower())

    voltage = battery.get("voltage_uv")
    if isinstance(voltage, (int, float)) and voltage > 0:
        volts = voltage / 1_000_000
        parts.append(f"{volts:.1f}V")

    if not parts:
        return "Battery detected, status unknown"
    return f"Battery: {', '.join(parts)}"


def get_portable_status() -> dict:
    """Comprehensive status for portable mode display / alerts.

    Returns dict with: temp_c, battery_pct, battery_status, gpu_pct,
    power_mode, warnings (list of human-readable alert strings).
    """
    result: dict = {
        "temp_c": None,
        "battery_pct": None,
        "battery_status": None,
        "gpu_pct": None,
        "power_mode": None,
        "warnings": [],
    }

    result["temp_c"] = get_thermal_temperature()
    result["gpu_pct"] = get_gpu_utilization()
    result["power_mode"] = get_power_mode()

    battery = get_battery_status()
    if battery:
        result["battery_pct"] = battery.get("capacity_pct")
        result["battery_status"] = battery.get("status")

        cap = battery.get("capacity_pct")
        if isinstance(cap, (int, float)):
            if cap < 10:
                result["warnings"].append(f"Battery critical: {cap}%. Connect power immediately.")
            elif cap < 25:
                result["warnings"].append(f"Battery low: {cap}%. Consider connecting power.")

    temp = result["temp_c"]
    if temp is not None and temp > 75:
        result["warnings"].append(f"Temperature elevated: {temp:.0f}C. Reducing processing load.")

    return result
