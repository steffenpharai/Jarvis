"""Optical flow estimation: dense (Farneback/DIS) and sparse (Lucas-Kanade).

Provides frame-to-frame motion vectors for:
  - Object velocity estimation (combined with depth → m/s)
  - Ego-motion decomposition (camera movement vs object movement)
  - Flow-assisted tracking (predict next bbox position)

All CPU-based (OpenCV) — zero extra GPU memory on the 8 GB Jetson.
Farneback at 320×240 runs ~15 ms; DIS is ~8 ms.  Both well within
the 33 ms frame budget at 30 FPS.

Inspired by Tesla FSD's dense optical flow backbone and SpaceX Dragon's
vision-based motion estimation for docking navigation.
"""

import logging
import time
from dataclasses import dataclass
from enum import Enum

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class FlowMethod(Enum):
    """Supported optical flow algorithms."""
    FARNEBACK = "farneback"
    DIS = "dis"
    LK = "lucas_kanade"  # sparse only


@dataclass
class FlowResult:
    """Result of optical flow computation between two consecutive frames."""

    flow: np.ndarray | None = None           # HxWx2 dense flow (dx, dy) in pixels
    magnitude: np.ndarray | None = None      # HxW flow magnitude
    angle: np.ndarray | None = None          # HxW flow angle (radians)
    mean_magnitude: float = 0.0              # average flow magnitude (motion indicator)
    max_magnitude: float = 0.0               # peak flow magnitude
    compute_time_ms: float = 0.0             # computation time

    # Sparse flow (for ego-motion / feature tracking)
    prev_points: np.ndarray | None = None    # Nx2 previous keypoints
    curr_points: np.ndarray | None = None    # Nx2 current keypoints (matched)
    point_status: np.ndarray | None = None   # N, 1=matched, 0=lost


# ── Flow estimator (stateful, keeps previous frame) ───────────────────


class OpticalFlowEstimator:
    """Stateful optical flow estimator — call ``compute()`` each frame.

    Parameters
    ----------
    method : FlowMethod
        Algorithm to use.  DIS is fastest; Farneback is most accurate for
        the compute budget.
    resize : tuple[int, int] | None
        (W, H) to resize frames before flow computation.  Lower = faster.
        None = use input resolution.  Recommended: (320, 240) for portable.
    sparse_max_corners : int
        Max corners for Lucas-Kanade sparse flow (used for ego-motion).
    """

    def __init__(
        self,
        method: FlowMethod = FlowMethod.FARNEBACK,
        resize: tuple[int, int] | None = (320, 240),
        sparse_max_corners: int = 120,
    ):
        self.method = method
        self.resize = resize
        self.sparse_max_corners = sparse_max_corners

        self._prev_gray: np.ndarray | None = None
        self._prev_points: np.ndarray | None = None
        self._frame_count = 0

        # DIS flow object (reusable, avoids re-allocation)
        self._dis: cv2.DISOpticalFlow | None = None
        # Pre-allocated output buffer for DIS (avoids per-frame allocation, ~1-2ms savings)
        self._flow_buf: np.ndarray | None = None
        if method == FlowMethod.DIS:
            self._dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_FAST)
            # Pre-allocate flow buffer once resize dimensions are known
            if resize is not None:
                w, h = resize
                self._flow_buf = np.zeros((h, w, 2), dtype=np.float32)

        # Shi-Tomasi corner params for sparse flow (120 corners is sufficient
        # for ego-motion estimation and saves ~0.5ms vs 200)
        self._feature_params = dict(
            maxCorners=sparse_max_corners,
            qualityLevel=0.05,
            minDistance=10,
            blockSize=7,
        )

        # Lucas-Kanade params
        self._lk_params = dict(
            winSize=(21, 21),
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.01),
        )

    def compute(self, frame: np.ndarray) -> FlowResult:
        """Compute optical flow between this frame and the previous one.

        Parameters
        ----------
        frame : BGR numpy array from camera

        Returns
        -------
        FlowResult with dense and/or sparse flow data.
        Returns empty FlowResult on first frame (no previous to compare).
        """
        result = FlowResult()
        t0 = time.monotonic()

        # Convert + resize
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if self.resize is not None:
            gray = cv2.resize(gray, self.resize)

        if self._prev_gray is None:
            # First frame — store and return empty
            self._prev_gray = gray
            self._prev_points = cv2.goodFeaturesToTrack(
                gray, mask=None, **self._feature_params
            )
            self._frame_count = 1
            return result

        # ── Dense flow ─────────────────────────────────────────────
        if self.method == FlowMethod.FARNEBACK:
            flow = cv2.calcOpticalFlowFarneback(
                self._prev_gray, gray, None,
                pyr_scale=0.5, levels=3, winsize=15,
                iterations=3, poly_n=5, poly_sigma=1.2,
                flags=0,
            )
        elif self.method == FlowMethod.DIS:
            # Reuse pre-allocated buffer to avoid per-frame allocation (~1-2ms savings)
            if self._flow_buf is not None and self._flow_buf.shape[:2] == gray.shape[:2]:
                flow = self._dis.calc(self._prev_gray, gray, self._flow_buf)
            else:
                flow = self._dis.calc(self._prev_gray, gray, None)
                # Cache the buffer for next frame
                if flow is not None:
                    self._flow_buf = flow
        else:
            flow = None

        if flow is not None:
            result.flow = flow
            mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            result.magnitude = mag
            result.angle = ang
            result.mean_magnitude = float(np.mean(mag))
            result.max_magnitude = float(np.max(mag))

        # ── Sparse flow (for ego-motion estimation) ────────────────
        if self._prev_points is not None and len(self._prev_points) > 0:
            curr_pts, status, _ = cv2.calcOpticalFlowPyrLK(
                self._prev_gray, gray, self._prev_points, None,
                **self._lk_params,
            )
            if curr_pts is not None and status is not None:
                mask = status.ravel() == 1
                result.prev_points = self._prev_points[mask].reshape(-1, 2)
                result.curr_points = curr_pts[mask].reshape(-1, 2)
                result.point_status = status.ravel()

        # Update state for next frame
        self._prev_gray = gray
        # Refresh feature points periodically (every 8 frames) or when too few.
        # 8 frames is sufficient for ego-motion; saves ~0.3ms/frame vs 5.
        self._frame_count += 1
        if (
            self._frame_count % 8 == 0
            or self._prev_points is None
            or len(self._prev_points) < 20
        ):
            self._prev_points = cv2.goodFeaturesToTrack(
                gray, mask=None, **self._feature_params
            )
        elif result.curr_points is not None and len(result.curr_points) > 0:
            self._prev_points = result.curr_points.reshape(-1, 1, 2).astype(
                np.float32
            )

        result.compute_time_ms = (time.monotonic() - t0) * 1000
        return result

    def reset(self) -> None:
        """Clear state (e.g. when camera reconnects or scene changes)."""
        self._prev_gray = None
        self._prev_points = None
        self._frame_count = 0


# ── Utility: flow at bounding boxes ───────────────────────────────────


def flow_at_boxes(
    flow: np.ndarray | None,
    detections: list[dict],
    frame_shape: tuple[int, int] | None = None,
) -> list[tuple[float, float] | None]:
    """Compute median flow vector within each detection bounding box.

    Parameters
    ----------
    flow : HxWx2 dense optical flow, or None
    detections : list of dicts with ``xyxy`` key
    frame_shape : (H, W) of the original frame (for coordinate scaling)

    Returns
    -------
    list of (dx, dy) flow vectors per detection, or None if unavailable
    """
    if flow is None:
        return [None] * len(detections)

    fh, fw = flow.shape[:2]
    results = []

    for det in detections:
        xyxy = det.get("xyxy")
        if xyxy is None or len(xyxy) != 4:
            results.append(None)
            continue

        # Scale bbox to flow resolution if needed
        if frame_shape is not None:
            oh, ow = frame_shape
            sx, sy = fw / ow, fh / oh
        else:
            sx, sy = 1.0, 1.0

        x1 = max(0, int(xyxy[0] * sx))
        y1 = max(0, int(xyxy[1] * sy))
        x2 = min(fw, int(xyxy[2] * sx))
        y2 = min(fh, int(xyxy[3] * sy))

        if x2 <= x1 or y2 <= y1:
            results.append(None)
            continue

        roi = flow[y1:y2, x1:x2]
        dx = float(np.median(roi[..., 0]))
        dy = float(np.median(roi[..., 1]))
        results.append((dx, dy))

    return results


def compute_motion_energy(flow: np.ndarray | None, threshold: float = 1.0) -> float:
    """Compute fraction of pixels with flow magnitude above threshold.

    Returns 0.0 (static scene) to 1.0 (everything moving).
    Useful for adaptive frame rate / wake-on-motion.
    """
    if flow is None:
        return 0.0
    mag = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
    return float(np.mean(mag > threshold))
