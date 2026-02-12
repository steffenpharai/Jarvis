"""Ambient Awareness: lightweight always-on motion detection for hands-free mode.

Runs DIS optical flow at ultra-low resolution (160x120, ~2ms) to detect
significant scene changes without the cost of full YOLOE/depth inference.
When a change is detected, escalates to the full perception pipeline.

State machine:
  IDLE     -> flow at 2 Hz, minimal CPU
  ACTIVE   -> flow at 5 Hz, full perception triggered
  COOLDOWN -> suppress duplicate triggers for N seconds

Industry patterns:
  - Tesla FSD: continuous low-level perception with escalation to planning
  - Apple Siri proactive: ambient context monitoring triggers suggestions
  - Google Assistant ambient: low-power always-on environmental sensing

Memory: ~1 MB for 160x120 flow buffer.  CPU-only (zero GPU).
"""

import enum
import logging
import time
from dataclasses import dataclass

import numpy as np

from vision.flow import FlowMethod, OpticalFlowEstimator, compute_motion_energy

logger = logging.getLogger(__name__)


# ── Ambient events ────────────────────────────────────────────────────


class AmbientEventType(enum.Enum):
    """Types of ambient awareness events."""
    MOTION_DETECTED = "motion_detected"       # significant object motion in scene
    EGO_MOTION_START = "ego_motion_start"     # camera started moving (user walking)
    EGO_MOTION_STOP = "ego_motion_stop"       # camera stopped (user stationary)
    SCENE_CHANGE = "scene_change"             # large-scale scene content change
    THERMAL_THROTTLE = "thermal_throttle"     # thermal limit reached
    BATTERY_LOW = "battery_low"              # battery below threshold


@dataclass
class AmbientEvent:
    """Event emitted by the ambient awareness system."""
    event_type: AmbientEventType
    timestamp: float = 0.0
    motion_energy: float = 0.0       # 0-1, fraction of moving pixels
    ego_speed: float = 0.0           # px/frame ego-motion magnitude
    detail: str = ""                 # human-readable description
    recommend_full_scan: bool = False  # whether to trigger full YOLOE+perception


# ── Ambient state ─────────────────────────────────────────────────────


class AmbientState(enum.Enum):
    IDLE = "idle"          # low-duty monitoring
    ACTIVE = "active"      # elevated scanning after trigger
    COOLDOWN = "cooldown"  # suppress duplicate triggers


# ── Configuration defaults ────────────────────────────────────────────

_AMBIENT_RESOLUTION = (160, 120)   # ultra-low-res for ~2ms DIS flow
_IDLE_HZ = 2.0                    # check 2x/sec when idle
_ACTIVE_HZ = 5.0                  # check 5x/sec when active
_COOLDOWN_SEC = 10.0              # min time between full scan triggers
_ACTIVE_DURATION_SEC = 30.0       # stay active for 30s after trigger
_EGO_MOTION_THRESHOLD = 3.0       # px/frame, above = user walking
_MOTION_ENERGY_THRESHOLD = 0.08   # above = something significant moving
_SCENE_CHANGE_THRESHOLD = 0.25    # above = major scene change (e.g. new room)
_THERMAL_CHECK_INTERVAL = 30.0    # seconds between thermal/battery checks
_THERMAL_THROTTLE_C = 70.0        # reduce duty above this temp
_THERMAL_PAUSE_C = 80.0           # pause ambient above this temp
_BATTERY_LOW_PCT = 15             # alert below this battery level


class AmbientAwareness:
    """Lightweight always-on motion detection for hands-free autonomy.

    Call ``check_frame(frame)`` at the configured duty cycle (2-5 Hz).
    Returns an ``AmbientEvent`` when a significant change is detected,
    or ``None`` when the scene is stable.

    Parameters
    ----------
    ego_motion_threshold : float
        Ego-motion speed (px/frame) above which user is considered walking.
    motion_energy_threshold : float
        Fraction of moving pixels above which object motion is flagged.
    cooldown_sec : float
        Minimum time between full scan triggers.
    active_duration_sec : float
        How long to stay in ACTIVE state after a trigger.
    """

    def __init__(
        self,
        ego_motion_threshold: float = _EGO_MOTION_THRESHOLD,
        motion_energy_threshold: float = _MOTION_ENERGY_THRESHOLD,
        cooldown_sec: float = _COOLDOWN_SEC,
        active_duration_sec: float = _ACTIVE_DURATION_SEC,
    ):
        self.ego_threshold = ego_motion_threshold
        self.motion_threshold = motion_energy_threshold
        self.cooldown_sec = cooldown_sec
        self.active_duration_sec = active_duration_sec

        # Ultra-low-res flow estimator (~2ms per frame)
        self._flow = OpticalFlowEstimator(
            method=FlowMethod.DIS,
            resize=_AMBIENT_RESOLUTION,
            sparse_max_corners=60,
        )

        # State machine
        self._state = AmbientState.IDLE
        self._state_entered_at = time.monotonic()
        self._last_trigger_time = 0.0
        self._last_thermal_check = 0.0

        # Ego-motion tracking (simple mean flow as proxy)
        self._was_ego_moving = False

        # Scene change detection via mean intensity
        self._prev_mean_intensity: float | None = None

    @property
    def state(self) -> AmbientState:
        return self._state

    @property
    def current_hz(self) -> float:
        """Recommended polling frequency for the current state."""
        if self._state == AmbientState.ACTIVE:
            return _ACTIVE_HZ
        return _IDLE_HZ

    @property
    def interval_sec(self) -> float:
        """Recommended sleep interval between frames."""
        return 1.0 / self.current_hz

    def check_frame(self, frame: np.ndarray) -> AmbientEvent | None:
        """Analyse one frame and return an event if significant change detected.

        Parameters
        ----------
        frame : BGR numpy array from camera

        Returns
        -------
        AmbientEvent if a trigger condition is met, else None.
        """
        now = time.monotonic()

        # ── Auto-transition from ACTIVE/COOLDOWN back to IDLE ─────
        if self._state == AmbientState.ACTIVE:
            if now - self._state_entered_at > self.active_duration_sec:
                self._transition(AmbientState.IDLE)
        elif self._state == AmbientState.COOLDOWN:
            if now - self._state_entered_at > self.cooldown_sec:
                self._transition(AmbientState.IDLE)

        # ── Thermal / battery check (periodic, not every frame) ───
        if now - self._last_thermal_check > _THERMAL_CHECK_INTERVAL:
            self._last_thermal_check = now
            thermal_event = self._check_thermal_battery()
            if thermal_event is not None:
                return thermal_event

        # ── Run ultra-light DIS flow ──────────────────────────────
        flow_result = self._flow.compute(frame)

        if flow_result.flow is None:
            # First frame, no flow yet
            self._prev_mean_intensity = float(np.mean(
                frame[:, :, 0] if len(frame.shape) == 3 else frame
            ))
            return None

        # ── Compute metrics ───────────────────────────────────────
        motion_energy = compute_motion_energy(flow_result.flow, threshold=1.5)
        mean_mag = flow_result.mean_magnitude

        # Scene change: compare mean intensity (cheap proxy for content change)
        gray_mean = float(np.mean(
            frame[:, :, 0] if len(frame.shape) == 3 else frame
        ))
        scene_delta = 0.0
        if self._prev_mean_intensity is not None:
            scene_delta = abs(gray_mean - self._prev_mean_intensity) / max(gray_mean, 1.0)
        self._prev_mean_intensity = gray_mean

        # ── Ego-motion state tracking ─────────────────────────────
        ego_moving = mean_mag > self.ego_threshold
        ego_transition = ego_moving != self._was_ego_moving
        self._was_ego_moving = ego_moving

        # ── Decide if we should emit an event ─────────────────────
        event: AmbientEvent | None = None

        # Priority 1: ego-motion transitions
        if ego_transition:
            if ego_moving:
                event = AmbientEvent(
                    event_type=AmbientEventType.EGO_MOTION_START,
                    timestamp=now,
                    ego_speed=mean_mag,
                    detail=f"Camera moving (flow={mean_mag:.1f}px/f)",
                    recommend_full_scan=True,
                )
            else:
                event = AmbientEvent(
                    event_type=AmbientEventType.EGO_MOTION_STOP,
                    timestamp=now,
                    ego_speed=mean_mag,
                    detail="Camera stopped",
                    recommend_full_scan=True,
                )

        # Priority 2: significant scene change
        elif scene_delta > _SCENE_CHANGE_THRESHOLD:
            event = AmbientEvent(
                event_type=AmbientEventType.SCENE_CHANGE,
                timestamp=now,
                motion_energy=motion_energy,
                detail=f"Scene change detected (delta={scene_delta:.2f})",
                recommend_full_scan=True,
            )

        # Priority 3: object motion in scene (while ego is stable)
        elif not ego_moving and motion_energy > self.motion_threshold:
            event = AmbientEvent(
                event_type=AmbientEventType.MOTION_DETECTED,
                timestamp=now,
                motion_energy=motion_energy,
                ego_speed=mean_mag,
                detail=f"Motion detected (energy={motion_energy:.2f})",
                recommend_full_scan=True,
            )

        # ── Apply cooldown suppression ────────────────────────────
        if event is not None and self._state == AmbientState.COOLDOWN:
            # In cooldown: suppress non-critical triggers
            if event.event_type not in (
                AmbientEventType.THERMAL_THROTTLE,
                AmbientEventType.BATTERY_LOW,
            ):
                return None

        if event is not None:
            self._last_trigger_time = now
            self._transition(AmbientState.ACTIVE)
            logger.debug(
                "Ambient trigger: %s (%s)", event.event_type.value, event.detail,
            )

        return event

    def enter_cooldown(self) -> None:
        """Manually enter cooldown (called after full scan completes)."""
        self._transition(AmbientState.COOLDOWN)

    def reset(self) -> None:
        """Reset all state (e.g. on camera reconnect)."""
        self._flow.reset()
        self._state = AmbientState.IDLE
        self._state_entered_at = time.monotonic()
        self._last_trigger_time = 0.0
        self._was_ego_moving = False
        self._prev_mean_intensity = None

    def _transition(self, new_state: AmbientState) -> None:
        if new_state != self._state:
            logger.debug("Ambient state: %s -> %s", self._state.value, new_state.value)
            self._state = new_state
            self._state_entered_at = time.monotonic()

    def _check_thermal_battery(self) -> AmbientEvent | None:
        """Check thermal and battery status, return event if limits exceeded."""
        try:
            from utils.power import get_battery_status, get_thermal_temperature

            temp = get_thermal_temperature()
            if temp is not None and temp > _THERMAL_PAUSE_C:
                return AmbientEvent(
                    event_type=AmbientEventType.THERMAL_THROTTLE,
                    timestamp=time.monotonic(),
                    detail=f"Thermal limit: {temp:.0f}C > {_THERMAL_PAUSE_C}C",
                )
            if temp is not None and temp > _THERMAL_THROTTLE_C:
                logger.debug("Ambient: thermal throttle at %.0fC", temp)
                # Don't emit event, just log (duty cycle reduction handled externally)

            battery = get_battery_status()
            if battery and battery.get("capacity_pct") is not None:
                cap = battery["capacity_pct"]
                if isinstance(cap, (int, float)) and cap < _BATTERY_LOW_PCT:
                    return AmbientEvent(
                        event_type=AmbientEventType.BATTERY_LOW,
                        timestamp=time.monotonic(),
                        detail=f"Battery low: {cap}%",
                    )
        except Exception as e:
            logger.debug("Ambient thermal/battery check failed: %s", e)
        return None
