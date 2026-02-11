"""Proximity alerts for portable/walk-around mode.

Uses depth estimation + object detection to generate audio cues
for obstacles, approaching objects, and environmental awareness.

Pattern: autonomous vehicle proximity warning systems (Tesla Autopilot,
NVIDIA DRIVE) — continuous depth monitoring with distance-based alerting.
"""

from __future__ import annotations

import logging
import time

logger = logging.getLogger(__name__)

# ── Alert thresholds (meters, approximate from monocular depth) ───────

# Critical: immediate collision risk
CRITICAL_DISTANCE_M = 0.5

# Warning: obstacle nearby
WARNING_DISTANCE_M = 1.5

# Notice: something approaching
NOTICE_DISTANCE_M = 3.0

# Minimum time between alerts for the same zone (seconds)
ALERT_COOLDOWN_SEC = 5.0

# ── State ─────────────────────────────────────────────────────────────

_last_alert_time: dict[str, float] = {}


def _can_alert(zone: str) -> bool:
    """Check if enough time has passed since the last alert for this zone."""
    now = time.monotonic()
    last = _last_alert_time.get(zone, 0)
    if now - last < ALERT_COOLDOWN_SEC:
        return False
    _last_alert_time[zone] = now
    return True


def check_proximity(
    tracked_objects: list[dict],
    depth_map=None,
) -> list[dict]:
    """Analyze tracked objects and depth for proximity alerts.

    Parameters
    ----------
    tracked_objects : list of tracked object dicts with 'depth', 'class_name', 'velocity'
    depth_map : numpy array or None, the depth estimation output

    Returns
    -------
    list of alert dicts: {level: 'critical'|'warning'|'notice', message: str, distance: float}
    """
    alerts = []

    for obj in tracked_objects:
        depth = obj.get("depth")
        if depth is None:
            continue

        class_name = obj.get("class_name", "object")
        velocity = obj.get("velocity", [0, 0])

        # Approximate depth in meters (DepthAnything outputs relative depth;
        # we use a rough calibration factor — exact values depend on camera)
        # For monocular depth, lower values = closer
        depth_m = _relative_to_meters(depth)

        if depth_m is None or depth_m > NOTICE_DISTANCE_M:
            continue

        # Check if approaching (velocity towards camera)
        approaching = velocity[1] < -5 if len(velocity) >= 2 else False

        if depth_m < CRITICAL_DISTANCE_M:
            zone = f"critical_{class_name}"
            if _can_alert(zone):
                msg = f"Sir, {class_name} very close ahead, approximately {depth_m:.1f} meters."
                alerts.append({"level": "critical", "message": msg, "distance": depth_m})

        elif depth_m < WARNING_DISTANCE_M:
            zone = f"warning_{class_name}"
            if _can_alert(zone):
                if approaching:
                    msg = f"Sir, {class_name} approaching, about {depth_m:.1f} meters away."
                else:
                    msg = f"Sir, {class_name} nearby at approximately {depth_m:.1f} meters."
                alerts.append({"level": "warning", "message": msg, "distance": depth_m})

        elif depth_m < NOTICE_DISTANCE_M and approaching:
            zone = f"notice_{class_name}"
            if _can_alert(zone):
                msg = f"Sir, {class_name} approaching from {depth_m:.1f} meters."
                alerts.append({"level": "notice", "message": msg, "distance": depth_m})

    # Sort by distance (closest first)
    alerts.sort(key=lambda a: a["distance"])
    return alerts


def _relative_to_meters(relative_depth: float) -> float | None:
    """Convert relative monocular depth to approximate meters.

    DepthAnything V2 outputs inverse relative depth. This is a rough
    linear approximation calibrated for a USB webcam on Jetson.
    Real deployments should calibrate against known distances.
    """
    if relative_depth <= 0:
        return None
    # Rough inverse relationship: closer objects have higher relative depth values
    # This approximation assumes depth values in range ~0-1 (normalized)
    try:
        if relative_depth > 100:
            # Raw disparity values (not normalized)
            meters = 500.0 / max(relative_depth, 1.0)
        else:
            # Normalized 0-1 range
            meters = 5.0 / max(relative_depth, 0.01)
        return min(max(meters, 0.1), 20.0)  # Clamp to 0.1-20m
    except Exception:
        return None


def format_proximity_summary(alerts: list[dict]) -> str:
    """Format proximity alerts as a concise text summary for LLM context."""
    if not alerts:
        return ""
    parts = []
    for a in alerts[:3]:  # Max 3 alerts
        parts.append(f"[{a['level'].upper()}] {a['message']}")
    return " ".join(parts)
