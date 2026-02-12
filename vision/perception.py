"""Fused perception pipeline: single-call to run flow + depth + tracking + ego-motion + trajectory.

This is the main entry point for the advanced perception system.  It
coordinates all sub-modules in the correct order and returns a unified
PerceptionResult that the orchestrator and scene description can consume.

Pipeline order (per frame):
  1. Optical flow (between prev and current frame)
  2. YOLOE detection
  3. Depth estimation
  4. Flow-assisted tracking (ByteTrack + flow prediction)
  5. Ego-motion estimation (from sparse flow)
  6. Ego-motion compensation (subtract camera motion from object flow)
  7. 3D velocity estimation (flow + depth → m/s)
  8. Trajectory prediction (Kalman forward projection)
  9. Collision detection (time-to-collision + alerts)

Tesla FSD runs all of these in a single neural backbone; we decompose
them into modular stages for debuggability and testability on the Jetson.

Memory budget: ~0 extra GPU (flow/ego/trajectory are CPU-only).
Only YOLOE + DepthAnything use GPU (already loaded).
"""

import logging
import time
from dataclasses import dataclass, field

import numpy as np

from vision.ego_motion import (
    EgoMotionResult,
    compensate_ego_motion,
    estimate_ego_motion,
    flow_to_velocity_mps,
)
from vision.flow import (
    FlowMethod,
    FlowResult,
    OpticalFlowEstimator,
    compute_motion_energy,
    flow_at_boxes,
)
from vision.trajectory import (
    CollisionAlert,
    PredictedTrajectory,
    TrajectoryPredictor,
    format_trajectory_summary,
)

logger = logging.getLogger(__name__)


@dataclass
class PerceptionResult:
    """Unified result from the full perception pipeline."""

    # Timing
    total_ms: float = 0.0
    flow_ms: float = 0.0
    detection_ms: float = 0.0
    depth_ms: float = 0.0
    tracking_ms: float = 0.0
    ego_motion_ms: float = 0.0
    trajectory_ms: float = 0.0

    # Raw outputs
    flow: FlowResult | None = None
    ego_motion: EgoMotionResult | None = None
    detections: list[dict] = field(default_factory=list)
    tracked: list = field(default_factory=list)
    depth_map: np.ndarray | None = None
    depth_values: list[float | None] = field(default_factory=list)
    trajectories: list[PredictedTrajectory] = field(default_factory=list)
    collision_alerts: list[CollisionAlert] = field(default_factory=list)

    # Derived
    object_velocities_mps: list[tuple[float, float, float] | None] = field(
        default_factory=list
    )
    motion_energy: float = 0.0   # 0-1, fraction of frame with significant motion
    ego_compensated_flows: list[tuple[float, float] | None] = field(
        default_factory=list
    )

    # Scene summary (text for LLM)
    trajectory_summary: str = ""
    ego_summary: str = ""

    # Point cloud (for hologram)
    point_cloud: list[dict] = field(default_factory=list)


class PerceptionPipeline:
    """Stateful perception pipeline — call ``process_frame()`` each frame.

    Manages all sub-module instances and their state across frames.

    Parameters
    ----------
    flow_method : FlowMethod
        Optical flow algorithm (FARNEBACK or DIS).
    flow_resize : tuple[int, int] | None
        Resize frames for flow computation (smaller = faster).
    prediction_horizon : float
        Trajectory prediction look-ahead (seconds).
    collision_zone_m : float
        Collision danger zone radius (meters).
    fps : float
        Camera FPS for velocity conversion.
    """

    def __init__(
        self,
        flow_method: FlowMethod = FlowMethod.FARNEBACK,
        flow_resize: tuple[int, int] | None = (320, 240),
        prediction_horizon: float = 3.0,
        collision_zone_m: float = 2.0,
        fps: float = 30.0,
        portable_mode: bool = False,
    ):
        self.fps = fps
        self.portable_mode = portable_mode
        self.flow_estimator = OpticalFlowEstimator(
            method=flow_method, resize=flow_resize,
        )
        self.trajectory_predictor = TrajectoryPredictor(
            prediction_horizon_sec=prediction_horizon,
            collision_zone_m=collision_zone_m,
        )
        self._flow_resize = flow_resize

    def process_frame(
        self,
        frame: np.ndarray,
        detections: list[dict],
        tracked_objects: list,
        depth_map: np.ndarray | None = None,
        depth_values: list[float | None] | None = None,
    ) -> PerceptionResult:
        """Run the full perception pipeline on one frame.

        Parameters
        ----------
        frame : BGR numpy array
        detections : YOLOE detections (list of dicts with xyxy, conf, cls)
        tracked_objects : list of TrackedObject from ByteTrackLite
        depth_map : HxW float32 depth (0=near, 1=far), or None
        depth_values : per-detection median depth, or None

        Returns
        -------
        PerceptionResult with all fused perception data
        """
        result = PerceptionResult()
        result.detections = detections
        result.tracked = tracked_objects
        result.depth_map = depth_map
        result.depth_values = depth_values or []
        t_start = time.monotonic()

        frame_h, frame_w = frame.shape[:2]
        frame_shape = (frame_h, frame_w)

        # ── 1. Optical Flow ───────────────────────────────────────
        t0 = time.monotonic()
        flow_result = self.flow_estimator.compute(frame)
        result.flow = flow_result
        result.flow_ms = (time.monotonic() - t0) * 1000

        # Motion energy computed lazily -- only when flow is available
        # and only accessed by ambient awareness (not blocking main path)
        if flow_result.flow is not None:
            result.motion_energy = compute_motion_energy(flow_result.flow)

        # ── 2. Per-object flow vectors ────────────────────────────
        raw_flows = flow_at_boxes(
            flow_result.flow, detections, frame_shape=frame_shape
        )

        # ── 3. Ego-motion estimation ─────────────────────────────
        t0 = time.monotonic()
        flow_size = self._flow_resize or (frame_w, frame_h)
        ego = estimate_ego_motion(
            flow_result.prev_points,
            flow_result.curr_points,
            frame_size=flow_size,
            # Skip expensive recoverPose in portable mode -- ego_dx/ego_dy
            # from median inlier flow is sufficient for compensation.
            skip_rotation=self.portable_mode,
        )
        result.ego_motion = ego
        result.ego_motion_ms = (time.monotonic() - t0) * 1000

        # Ego summary for LLM
        if ego.is_moving:
            ego_speed = (ego.ego_dx ** 2 + ego.ego_dy ** 2) ** 0.5
            result.ego_summary = f"Camera {ego.motion_type} (flow={ego_speed:.1f}px/f)"
        else:
            result.ego_summary = "Camera static"

        # ── 4. Ego-motion compensation (vectorised) ───────────────
        compensated = compensate_ego_motion(raw_flows, ego)
        result.ego_compensated_flows = compensated

        # ── 5. 3D velocity estimation ─────────────────────────────
        velocities_mps = []
        for i, comp_flow in enumerate(compensated):
            depth_rel = None
            if depth_values and i < len(depth_values):
                depth_rel = depth_values[i]
            elif i < len(tracked_objects) and hasattr(tracked_objects[i], "depth"):
                depth_rel = tracked_objects[i].depth

            if comp_flow is not None:
                vel = flow_to_velocity_mps(
                    comp_flow[0], comp_flow[1],
                    depth_rel,
                    fps=self.fps,
                    frame_width=flow_size[0],
                )
                velocities_mps.append(vel)
            else:
                velocities_mps.append(None)
        result.object_velocities_mps = velocities_mps

        # ── 6. Trajectory prediction (vectorised batch) ───────────
        t0 = time.monotonic()
        trajectories, alerts = self.trajectory_predictor.predict_all(
            tracked_objects,
            flow_vectors=compensated,
            depth_values=depth_values,
            velocity_mps_list=velocities_mps,
            frame_size=flow_size,
            fps=self.fps,
        )
        result.trajectories = trajectories
        result.collision_alerts = alerts
        result.trajectory_ms = (time.monotonic() - t0) * 1000

        # ── 7. Trajectory summary text ────────────────────────────
        result.trajectory_summary = format_trajectory_summary(
            trajectories, alerts,
            ego_motion_type=ego.motion_type if ego else "static",
        )

        result.total_ms = (time.monotonic() - t_start) * 1000

        # ── Timing guard: warn if perception budget exceeded ──────
        if result.total_ms > 15.0:
            logger.debug(
                "Perception budget exceeded: %.1fms (flow=%.1f ego=%.1f traj=%.1f)",
                result.total_ms, result.flow_ms, result.ego_motion_ms,
                result.trajectory_ms,
            )

        return result

    def reset(self) -> None:
        """Reset all state (e.g. on camera reconnect)."""
        self.flow_estimator.reset()
        self.trajectory_predictor.reset()
