"""Trajectory prediction & collision detection for tracked objects.

Uses constant-velocity / constant-acceleration model to forecast
object positions 1-3 seconds ahead.  Combined with depth estimation,
this enables proactive alerts:

  "Sir, bicycle approaching from left at 8 km/h -- potential collision in 2.4 seconds"

Inspired by Tesla FSD's occupancy flow prediction and SpaceX Dragon's
trajectory forecasting for docking approach.

Optimisations (v2):
  - Stationary objects skipped early (speed < _MIN_SPEED_PX_SEC)
  - Waypoints computed via vectorised numpy batch (all objects at once)
  - Collision risk only computed for approaching objects with depth

Memory: ~0 extra -- pure NumPy computation per tracked object.
"""

import logging
import math
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger(__name__)

# Objects below this speed (px/sec) are tagged "stationary" immediately,
# skipping waypoint and collision computation.  Saves ~0.5-1ms for typical
# scenes with 5-10 mostly-static objects.
_MIN_SPEED_PX_SEC = 5.0


@dataclass
class PredictedTrajectory:
    """Forecasted trajectory for a tracked object."""

    track_id: int = 0
    class_name: str = ""

    # Current state
    position_px: tuple[float, float] = (0.0, 0.0)   # center (cx, cy) in pixels
    velocity_px: tuple[float, float] = (0.0, 0.0)    # pixels/sec
    velocity_mps: tuple[float, float, float] | None = None  # (vx, vy, speed) m/s
    depth_m: float | None = None                      # estimated distance in meters

    # Predicted future positions (list of (cx, cy, t_sec))
    waypoints: list[tuple[float, float, float]] = field(default_factory=list)

    # Collision risk
    collision_risk: float = 0.0        # 0.0-1.0 probability of entering danger zone
    time_to_collision: float | None = None  # seconds until potential collision, or None
    collision_direction: str = ""       # "left", "right", "ahead", "behind"

    # Behaviour classification
    behaviour: str = "stationary"       # stationary, approaching, receding, crossing, orbiting


@dataclass
class CollisionAlert:
    """Proactive collision/proximity alert for the orchestrator."""

    track_id: int = 0
    class_name: str = ""
    speed_mps: float = 0.0
    distance_m: float = 0.0
    time_to_collision: float = 0.0
    direction: str = ""
    severity: str = "notice"  # notice, warning, critical
    message: str = ""


# ── Trajectory predictor ──────────────────────────────────────────────


class TrajectoryPredictor:
    """Predict future positions and detect collision risks.

    Call ``predict_all()`` each frame with tracked objects, flow data,
    and depth info to get trajectories and collision alerts.

    Parameters
    ----------
    prediction_horizon_sec : float
        How far ahead to predict (seconds).
    prediction_steps : int
        Number of waypoints in the prediction horizon.
    collision_zone_m : float
        Radius of "danger zone" around camera (meters).
    approach_angle_deg : float
        Max angle from camera center-line to count as "approaching".
    """

    def __init__(
        self,
        prediction_horizon_sec: float = 3.0,
        prediction_steps: int = 6,
        collision_zone_m: float = 2.0,
        approach_angle_deg: float = 45.0,
    ):
        self.horizon = prediction_horizon_sec
        self.steps = prediction_steps
        self.collision_zone_m = collision_zone_m
        self.approach_angle = approach_angle_deg

        # Per-track acceleration estimator (track_id → prev velocity)
        self._prev_velocities: dict[int, tuple[float, float]] = {}

    def predict_all(
        self,
        tracked_objects: list,
        flow_vectors: list[tuple[float, float] | None] | None = None,
        depth_values: list[float | None] | None = None,
        velocity_mps_list: list[tuple[float, float, float] | None] | None = None,
        frame_size: tuple[int, int] = (320, 240),
        fps: float = 30.0,
    ) -> tuple[list[PredictedTrajectory], list[CollisionAlert]]:
        """Predict trajectories and detect collisions for all tracked objects.

        Parameters
        ----------
        tracked_objects : list of TrackedObject (from ByteTrackLite)
        flow_vectors : per-object (dx, dy) ego-compensated flow, or None
        depth_values : per-object relative depth (0-1), or None
        velocity_mps_list : per-object (vx, vy, speed) in m/s, or None
        frame_size : (W, H) for reference
        fps : camera FPS for velocity conversion

        Returns
        -------
        (trajectories, alerts) -- list of PredictedTrajectory and CollisionAlert
        """
        trajectories = []
        alerts = []
        fw, fh = frame_size
        n_objs = len(tracked_objects)

        if n_objs == 0:
            return trajectories, alerts

        # ── Batch extract attributes ──────────────────────────────
        track_ids = []
        class_names = []
        cxs = np.empty(n_objs, dtype=np.float64)
        cys = np.empty(n_objs, dtype=np.float64)
        vx_arr = np.zeros(n_objs, dtype=np.float64)
        vy_arr = np.zeros(n_objs, dtype=np.float64)
        depth_m_arr = np.full(n_objs, np.nan, dtype=np.float64)
        vel_mps_arr: list[tuple[float, float, float] | None] = [None] * n_objs

        for i, t in enumerate(tracked_objects):
            tid = getattr(t, "track_id", i)
            track_ids.append(tid)
            class_names.append(getattr(t, "class_name", "object"))
            xyxy = getattr(t, "xyxy", [0, 0, 0, 0])
            cxs[i] = (xyxy[0] + xyxy[2]) / 2.0
            cys[i] = (xyxy[1] + xyxy[3]) / 2.0

            # Prefer flow-based velocity
            if flow_vectors and i < len(flow_vectors) and flow_vectors[i] is not None:
                vx_arr[i] = flow_vectors[i][0] * fps
                vy_arr[i] = flow_vectors[i][1] * fps
            else:
                vel = getattr(t, "velocity", [0.0, 0.0])
                vx_arr[i] = vel[0] if len(vel) > 0 else 0.0
                vy_arr[i] = vel[1] if len(vel) > 1 else 0.0

            # Depth
            depth_rel = None
            if depth_values and i < len(depth_values):
                depth_rel = depth_values[i]
            elif hasattr(t, "depth") and t.depth is not None:
                depth_rel = t.depth
            if depth_rel is not None:
                depth_m_arr[i] = depth_rel * 10.0

            # Velocity in m/s
            if velocity_mps_list and i < len(velocity_mps_list):
                vel_mps_arr[i] = velocity_mps_list[i]

        # ── Compute speeds + stationary mask ─────────────────────
        speed_arr = np.sqrt(vx_arr ** 2 + vy_arr ** 2)
        moving_mask = speed_arr >= _MIN_SPEED_PX_SEC

        # ── Acceleration (dampened finite difference) ─────────────
        ax_arr = np.zeros(n_objs, dtype=np.float64)
        ay_arr = np.zeros(n_objs, dtype=np.float64)
        for i, tid in enumerate(track_ids):
            prev = self._prev_velocities.get(tid, (vx_arr[i], vy_arr[i]))
            ax_arr[i] = (vx_arr[i] - prev[0]) * 0.3
            ay_arr[i] = (vy_arr[i] - prev[1]) * 0.3
            self._prev_velocities[tid] = (float(vx_arr[i]), float(vy_arr[i]))

        # ── Vectorised waypoint computation (all moving objects) ──
        dt = self.horizon / self.steps
        # time_steps: shape (steps,)
        time_steps = np.arange(1, self.steps + 1, dtype=np.float64) * dt

        # For moving objects: batch compute waypoints
        # px[i, s] = cx[i] + vx[i]*t[s] + 0.5*ax[i]*t[s]^2
        moving_idx = np.where(moving_mask)[0]
        all_waypoints: list[list[tuple[float, float, float]]] = [[] for _ in range(n_objs)]

        if len(moving_idx) > 0:
            m_cx = cxs[moving_idx][:, np.newaxis]  # (M, 1)
            m_cy = cys[moving_idx][:, np.newaxis]
            m_vx = vx_arr[moving_idx][:, np.newaxis]
            m_vy = vy_arr[moving_idx][:, np.newaxis]
            m_ax = ax_arr[moving_idx][:, np.newaxis]
            m_ay = ay_arr[moving_idx][:, np.newaxis]
            ts = time_steps[np.newaxis, :]  # (1, steps)

            px = m_cx + m_vx * ts + 0.5 * m_ax * ts ** 2  # (M, steps)
            py = m_cy + m_vy * ts + 0.5 * m_ay * ts ** 2

            for j, idx in enumerate(moving_idx):
                wps = []
                for s in range(self.steps):
                    wps.append((
                        round(float(px[j, s]), 1),
                        round(float(py[j, s]), 1),
                        round(float(time_steps[s]), 2),
                    ))
                all_waypoints[idx] = wps

        # ── Build trajectories and alerts ─────────────────────────
        for i in range(n_objs):
            tid = track_ids[i]
            cn = class_names[i]
            cx_i, cy_i = float(cxs[i]), float(cys[i])
            vx_i, vy_i = float(vx_arr[i]), float(vy_arr[i])
            spd = float(speed_arr[i])
            dm = float(depth_m_arr[i]) if not np.isnan(depth_m_arr[i]) else None
            vm = vel_mps_arr[i]

            if not moving_mask[i]:
                # Stationary: skip waypoints/collision, fast path
                traj = PredictedTrajectory(
                    track_id=tid, class_name=cn,
                    position_px=(cx_i, cy_i),
                    velocity_px=(vx_i, vy_i),
                    velocity_mps=vm, depth_m=dm,
                    behaviour="stationary",
                )
                trajectories.append(traj)
                continue

            behaviour = _classify_behaviour(vx_i, vy_i, cx_i, cy_i, fw, fh, spd)

            # Collision risk
            collision_risk = 0.0
            ttc: float | None = None
            direction = ""

            if dm is not None and vm is not None:
                speed_mps = vm[2]
                if behaviour == "approaching" and speed_mps > 0.1:
                    ttc = dm / speed_mps if speed_mps > 0 else None
                    if ttc is not None and ttc < self.horizon:
                        collision_risk = min(1.0, self.collision_zone_m / max(dm, 0.1))

                if cx_i < fw * 0.33:
                    direction = "left"
                elif cx_i > fw * 0.67:
                    direction = "right"
                else:
                    direction = "ahead"

            traj = PredictedTrajectory(
                track_id=tid, class_name=cn,
                position_px=(cx_i, cy_i),
                velocity_px=(vx_i, vy_i),
                velocity_mps=vm, depth_m=dm,
                waypoints=all_waypoints[i],
                collision_risk=collision_risk,
                time_to_collision=ttc,
                collision_direction=direction,
                behaviour=behaviour,
            )
            trajectories.append(traj)

            # Alert
            if ttc is not None and ttc < self.horizon and collision_risk > 0.2:
                speed_display = vm[2] if vm else 0
                alert = _build_alert(tid, cn, speed_display, dm or 0, ttc, direction)
                if alert is not None:
                    alerts.append(alert)

        # Clean up stale velocity history
        active_ids = set(track_ids)
        self._prev_velocities = {
            k: v for k, v in self._prev_velocities.items() if k in active_ids
        }

        return trajectories, alerts

    def reset(self) -> None:
        """Clear prediction state."""
        self._prev_velocities.clear()


# ── Helper functions ──────────────────────────────────────────────────


def _classify_behaviour(
    vx: float, vy: float, cx: float, cy: float,
    fw: int, fh: int, speed: float,
) -> str:
    """Classify object motion behaviour in the image plane."""
    if speed < 10:  # pixels/sec threshold for "stationary"
        return "stationary"

    # Camera center
    cam_cx, cam_cy = fw / 2.0, fh / 2.0

    # Vector from object to camera center
    to_cam_x = cam_cx - cx
    to_cam_y = cam_cy - cy
    to_cam_mag = math.sqrt(to_cam_x ** 2 + to_cam_y ** 2)

    if to_cam_mag < 1:
        return "orbiting"

    # Dot product of velocity with direction-to-camera
    dot = (vx * to_cam_x + vy * to_cam_y) / to_cam_mag
    # Cross product magnitude (lateral component)
    cross = abs(vx * to_cam_y - vy * to_cam_x) / to_cam_mag

    if dot > speed * 0.5:
        return "approaching"
    elif dot < -speed * 0.5:
        return "receding"
    elif cross > speed * 0.5:
        return "crossing"
    else:
        return "moving"


def _build_alert(
    track_id: int,
    class_name: str,
    speed_mps: float,
    distance_m: float,
    ttc: float,
    direction: str,
) -> CollisionAlert | None:
    """Build a collision alert with severity and natural language message."""
    if ttc <= 0:
        return None

    # Convert m/s to km/h for display
    speed_kmh = speed_mps * 3.6

    # Determine severity
    if ttc < 1.0 and distance_m < 2.0:
        severity = "critical"
    elif ttc < 2.0 and distance_m < 4.0:
        severity = "warning"
    elif ttc < 3.0:
        severity = "notice"
    else:
        return None  # not urgent enough

    # Build natural language message (Jarvis-style)
    dir_phrase = f"from the {direction}" if direction else ""
    msg = (
        f"Sir, {class_name} {dir_phrase} at {speed_kmh:.0f} km/h — "
        f"approximately {distance_m:.1f} meters away, "
        f"potential collision in {ttc:.1f} seconds."
    )

    return CollisionAlert(
        track_id=track_id,
        class_name=class_name,
        speed_mps=speed_mps,
        distance_m=distance_m,
        time_to_collision=ttc,
        direction=direction,
        severity=severity,
        message=msg,
    )


def format_trajectory_summary(
    trajectories: list[PredictedTrajectory],
    alerts: list[CollisionAlert],
    ego_motion_type: str = "static",
) -> str:
    """Format trajectory data as concise text for LLM context injection.

    Example: "Ego: walking forward. 2 objects tracked: person approaching
    at 1.2 m/s (3.8m, collision in 2.4s), car crossing left at 5.1 m/s."
    """
    parts = []

    # Ego-motion summary
    if ego_motion_type != "static":
        parts.append(f"Ego: {ego_motion_type}")

    # Object summaries (only moving or close objects)
    moving = [t for t in trajectories if t.behaviour != "stationary"]
    if moving:
        obj_parts = []
        for t in moving[:5]:  # cap at 5 most interesting
            desc = f"{t.class_name} {t.behaviour}"
            if t.velocity_mps:
                desc += f" at {t.velocity_mps[2]:.1f}m/s"
            if t.depth_m is not None:
                desc += f" ({t.depth_m:.1f}m)"
            if t.time_to_collision is not None:
                desc += f" [collision {t.time_to_collision:.1f}s]"
            obj_parts.append(desc)
        parts.append(f"{len(moving)} moving: " + ", ".join(obj_parts))

    # Alerts
    if alerts:
        for a in alerts[:2]:  # cap at 2 most urgent
            parts.append(f"[{a.severity.upper()}] {a.message}")

    return ". ".join(parts) if parts else ""
