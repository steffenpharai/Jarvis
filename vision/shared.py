"""Process-wide singletons for camera, YOLOE engine, MediaPipe, vitals, depth, tracker.

Every consumer in the Jarvis process (MJPEG ``/stream``, orchestrator
``vision_analyze`` tool, ``--e2e`` vision thread, ``--yolo-visualize``) MUST
use these singletons.  Loading duplicate TensorRT engines doubles GPU memory
on the 8 GB Jetson, and opening the camera twice causes V4L2 contention.

Thread safety
-------------
* ``_init_lock`` serialises one-time lazy initialisation of each resource.
* ``_frame_lock`` serialises ``cv2.VideoCapture.read()`` (not re-entrant).
* ``_inference_lock`` serialises TensorRT ``model(frame)`` calls — a single
  CUDA execution context is **not** thread-safe on the Orin's single GPU.
* ``_vitals_lock`` serialises vitals analysis (Face Mesh + Pose).
* ``_depth_lock`` serialises DepthAnything inference.
* ``_vision_paused`` flag allows temporarily pausing vision during OOM.
"""

import logging
import threading
from typing import Any

from config import settings

logger = logging.getLogger(__name__)

# ── Locks ──────────────────────────────────────────────────────────────
_init_lock = threading.Lock()
_frame_lock = threading.Lock()
_inference_lock = threading.Lock()
_vitals_lock = threading.Lock()
_depth_lock = threading.Lock()

# ── Vision pause/resume (for OOM resilience) ──────────────────────────
_vision_paused = False
_vision_pause_lock = threading.Lock()

# ── CUDA check ─────────────────────────────────────────────────────────


def check_cuda() -> tuple[bool, str]:
    """Return ``(ok, message)`` for CUDA availability."""
    try:
        import torch

        if not torch.cuda.is_available():
            return (
                False,
                "PyTorch not compiled with CUDA. "
                "Run: bash scripts/install-pytorch-cuda-nvidia.sh",
            )
        name = torch.cuda.get_device_name(0)
        return True, f"CUDA OK: {name}"
    except ImportError:
        return False, "PyTorch not installed"


# ── Camera singleton ──────────────────────────────────────────────────
_camera: Any | None = None
_camera_initialised = False


def get_camera() -> Any | None:
    """Lazily open and return the shared VideoCapture.  Thread-safe."""
    global _camera, _camera_initialised
    if _camera_initialised:
        return _camera
    with _init_lock:
        if _camera_initialised:
            return _camera
        try:
            from vision.camera import open_camera

            _camera = open_camera(
                settings.CAMERA_INDEX,
                settings.CAMERA_WIDTH,
                settings.CAMERA_HEIGHT,
                settings.CAMERA_FPS,
                device_path=settings.CAMERA_DEVICE,
            )
            if _camera is None:
                logger.warning("Shared camera: device not available")
        except Exception as e:
            logger.warning("Shared camera open failed: %s", e)
            _camera = None
        _camera_initialised = True
    return _camera


_consecutive_frame_failures = 0
_MAX_FRAME_FAILURES = 5
# Number of frames to discard when draining the V4L2 ring buffer.
# USB cameras on Jetson typically buffer 2-4 frames; 4 is safe.
_DRAIN_FRAME_COUNT = 4


def read_frame(drain: bool = False) -> Any | None:
    """Read one frame from the shared camera.  Thread-safe.

    Parameters
    ----------
    drain : bool
        When True, discard buffered V4L2 frames first so the returned frame
        is as close to *now* as possible.  Essential for on-demand vision
        queries after idle periods — the V4L2 ring buffer may hold frames
        captured seconds or minutes ago.  Default False for high-FPS
        streaming (MJPEG, continuous broadcast) where latency is acceptable.

    Auto-reconnects after ``_MAX_FRAME_FAILURES`` consecutive read failures
    (e.g. USB cam disconnected during walk-around).
    """
    global _consecutive_frame_failures

    if is_vision_paused():
        return None

    cap = get_camera()
    if cap is None:
        # Try reconnect
        if reconnect_camera():
            cap = get_camera()
        if cap is None:
            return None

    with _frame_lock:
        from vision.camera import read_frame as _read

        if drain:
            # Drain the V4L2 ring buffer (typically 2-4 frames).
            # This ensures the frame we return is the most recent one,
            # not one buffered while the camera was idle.
            for _ in range(_DRAIN_FRAME_COUNT):
                _read(cap)

        frame = _read(cap)

    if frame is None:
        _consecutive_frame_failures += 1
        if _consecutive_frame_failures >= _MAX_FRAME_FAILURES:
            logger.warning(
                "Camera: %d consecutive read failures — attempting reconnect",
                _consecutive_frame_failures,
            )
            _consecutive_frame_failures = 0
            reconnect_camera()
        return None

    _consecutive_frame_failures = 0
    return frame


def release_camera() -> None:
    """Explicitly release the shared camera (e.g. on shutdown)."""
    global _camera, _camera_initialised
    with _init_lock:
        if _camera is not None:
            try:
                _camera.release()
            except Exception:
                pass
            _camera = None
        _camera_initialised = False


def reconnect_camera() -> bool:
    """Release and re-open the camera.  Useful after USB disconnect/reconnect.

    Returns True if the camera was successfully re-opened.
    """
    release_camera()
    cam = get_camera()
    return cam is not None


# ── YOLO engine singleton ─────────────────────────────────────────────
_yolo_engine: Any | None = None
_yolo_class_names: dict[int, str] | None = None
_yolo_initialised = False


def get_yolo() -> tuple[Any | None, dict[int, str] | None]:
    """Lazily load and return ``(engine, class_names)``.  Thread-safe."""
    global _yolo_engine, _yolo_class_names, _yolo_initialised
    if _yolo_initialised:
        return _yolo_engine, _yolo_class_names
    with _init_lock:
        if _yolo_initialised:
            return _yolo_engine, _yolo_class_names
        ok, msg = check_cuda()
        if not ok:
            logger.warning("YOLOE requires CUDA: %s", msg)
        if settings.yolo_engine_exists():
            try:
                from vision.detector_yolo import get_class_names, load_yolo_engine

                _yolo_engine = load_yolo_engine(settings.YOLOE_ENGINE_PATH)
                _yolo_class_names = (
                    get_class_names(_yolo_engine) if _yolo_engine else None
                )
            except Exception as e:
                logger.warning("Shared YOLOE engine load failed: %s", e)
        else:
            logger.warning(
                "YOLOE engine not found at %s", settings.YOLOE_ENGINE_PATH
            )
        _yolo_initialised = True
    return _yolo_engine, _yolo_class_names


def run_inference_shared(frame: Any) -> list:
    """Run YOLOE inference on *frame* behind the inference lock.

    Returns a list of detection dicts ``{xyxy, conf, cls}`` or ``[]``.
    Serialises calls so concurrent threads don't collide on the single
    TensorRT CUDA execution context.
    """
    engine, _ = get_yolo()
    if engine is None or frame is None:
        return []
    with _inference_lock:
        from vision.detector_yolo import run_inference

        return run_inference(engine, frame)


# ── MediaPipe face detector singleton ─────────────────────────────────
_face_detector: Any | None = None
_face_detector_initialised = False


def get_face_detector() -> Any | None:
    """Lazily create and return the shared MediaPipe face detector."""
    global _face_detector, _face_detector_initialised
    if _face_detector_initialised:
        return _face_detector
    with _init_lock:
        if _face_detector_initialised:
            return _face_detector
        try:
            from vision.detector_mediapipe import create_face_detector

            _face_detector = create_face_detector()
        except Exception as e:
            logger.warning("Shared MediaPipe face detector failed: %s", e)
            _face_detector = None
        _face_detector_initialised = True
    return _face_detector


# ── Face Mesh singleton (for vitals: EAR, rPPG) ──────────────────────
_face_mesh: Any | None = None
_face_mesh_initialised = False


def get_face_mesh() -> Any | None:
    """Lazily create and return the shared MediaPipe Face Mesh."""
    global _face_mesh, _face_mesh_initialised
    if _face_mesh_initialised:
        return _face_mesh
    with _init_lock:
        if _face_mesh_initialised:
            return _face_mesh
        try:
            from vision.vitals import create_face_mesh

            _face_mesh = create_face_mesh()
        except Exception as e:
            logger.warning("Shared Face Mesh init failed: %s", e)
            _face_mesh = None
        _face_mesh_initialised = True
    return _face_mesh


# ── Pose detector singleton (for posture) ─────────────────────────────
_pose_detector: Any | None = None
_pose_detector_initialised = False


def get_pose_detector() -> Any | None:
    """Lazily create and return the shared MediaPipe Pose detector."""
    global _pose_detector, _pose_detector_initialised
    if _pose_detector_initialised:
        return _pose_detector
    with _init_lock:
        if _pose_detector_initialised:
            return _pose_detector
        try:
            from vision.vitals import create_pose_detector

            _pose_detector = create_pose_detector()
        except Exception as e:
            logger.warning("Shared Pose detector init failed: %s", e)
            _pose_detector = None
        _pose_detector_initialised = True
    return _pose_detector


# ── Vitals analyzer singleton ─────────────────────────────────────────
_vitals_analyzer: Any | None = None
_vitals_analyzer_initialised = False


def get_vitals_analyzer():
    """Lazily create and return the shared VitalsAnalyzer."""
    global _vitals_analyzer, _vitals_analyzer_initialised
    if _vitals_analyzer_initialised:
        return _vitals_analyzer
    with _init_lock:
        if _vitals_analyzer_initialised:
            return _vitals_analyzer
        try:
            from vision.vitals import VitalsAnalyzer

            fps = settings.CAMERA_FPS
            if getattr(settings, "PORTABLE_MODE", False):
                fps = getattr(settings, "PORTABLE_FPS", 10)
            _vitals_analyzer = VitalsAnalyzer(fps=fps)
        except Exception as e:
            logger.warning("Shared VitalsAnalyzer init failed: %s", e)
            _vitals_analyzer = None
        _vitals_analyzer_initialised = True
    return _vitals_analyzer


def run_vitals_shared(frame) -> Any | None:
    """Run vitals analysis behind the vitals lock. Returns VitalsResult or None."""
    analyzer = get_vitals_analyzer()
    if analyzer is None or frame is None:
        return None
    with _vitals_lock:
        face_mesh = get_face_mesh()
        pose_det = get_pose_detector()
        return analyzer.analyze(frame, face_mesh=face_mesh, pose_detector=pose_det)


# ── Depth model singleton ─────────────────────────────────────────────
_depth_model: Any | None = None
_depth_model_initialised = False


def get_depth_model() -> Any | None:
    """Lazily load and return the DepthAnything V2 Small TensorRT model."""
    global _depth_model, _depth_model_initialised
    if _depth_model_initialised:
        return _depth_model
    with _init_lock:
        if _depth_model_initialised:
            return _depth_model
        depth_path = getattr(settings, "DEPTH_ENGINE_PATH", None)
        depth_enabled = getattr(settings, "DEPTH_ENABLED", False)
        if not depth_enabled or not depth_path:
            logger.info("Depth estimation disabled or engine path not set")
            _depth_model_initialised = True
            return None
        try:
            from vision.depth import load_depth_model

            _depth_model = load_depth_model(depth_path)
            if _depth_model is None:
                logger.warning("Depth model failed to load from %s", depth_path)
        except Exception as e:
            logger.warning("Shared depth model init failed: %s", e)
            _depth_model = None
        _depth_model_initialised = True
    return _depth_model


def run_depth_shared(frame):
    """Run depth estimation behind the depth lock. Returns depth map or None."""
    model = get_depth_model()
    if model is None or frame is None:
        return None
    with _depth_lock:
        try:
            from vision.depth import estimate_depth

            return estimate_depth(model, frame)
        except Exception as e:
            logger.warning("Depth inference failed: %s", e)
            return None


# ── Tracker singleton ─────────────────────────────────────────────────
_tracker: Any | None = None
_tracker_initialised = False


def get_tracker():
    """Lazily create and return the shared ByteTrackLite tracker."""
    global _tracker, _tracker_initialised
    if _tracker_initialised:
        return _tracker
    with _init_lock:
        if _tracker_initialised:
            return _tracker
        try:
            from vision.tracker import ByteTrackLite

            _tracker = ByteTrackLite()
        except Exception as e:
            logger.warning("Shared tracker init failed: %s", e)
            _tracker = None
        _tracker_initialised = True
    return _tracker


# ── Vision pause/resume for OOM resilience ────────────────────────────


def pause_vision() -> None:
    """Temporarily pause all vision processing (e.g. during heavy LLM inference)."""
    global _vision_paused
    with _vision_pause_lock:
        _vision_paused = True
    logger.info("Vision processing paused")


def resume_vision() -> None:
    """Resume vision processing after a pause."""
    global _vision_paused
    with _vision_pause_lock:
        _vision_paused = False
    logger.info("Vision processing resumed")


def is_vision_paused() -> bool:
    """Check if vision is currently paused."""
    return _vision_paused


# ── High-level convenience ─────────────────────────────────────────────

# Synonym map for prompt-based focus (COCO class names)
_PROMPT_SYNONYMS: dict[str, str] = {
    "coffee mug": "cup",
    "mug": "cup",
    "coffee": "cup",
    "laptop computer": "laptop",
    "mobile": "cell phone",
    "phone": "cell phone",
    "tv": "tv",
    "television": "tv",
    "sofa": "couch",
    "dining table": "dining table",
}


def _parse_prompt_classes(prompt: str) -> list[str]:
    """Parse a prompt string into a list of class names for open-vocab detection.

    Supports comma-separated values: ``"tired person, coffee mug"`` → ``["tired person", "coffee mug"]``.
    Single words pass through as-is.  Synonyms are resolved.
    """
    if not prompt or not prompt.strip():
        return []
    parts = [p.strip() for p in prompt.split(",") if p.strip()]
    resolved = [_PROMPT_SYNONYMS.get(p.lower(), p) for p in parts]
    return resolved


def describe_current_scene(prompt: str | None = None) -> str:
    """Grab a frame, run YOLOE + face detection, return a text description.

    This is the single implementation backing both ``tools.vision_analyze``
    and the orchestrator's proactive-idle vision check.

    If *prompt* contains comma-separated class names (e.g. ``"tired person,
    coffee mug"``), dynamic open-vocabulary prompting is attempted via
    ``set_classes()`` with fallback to post-inference filtering.
    """
    try:
        from vision.detector_mediapipe import detect_faces
        from vision.scene import describe_scene

        # Drain V4L2 buffer so we analyse the *current* view, not a
        # stale buffered frame from before the camera moved.
        frame = read_frame(drain=True)
        if frame is None:
            return "Vision temporarily unavailable (no frame captured)."

        engine, class_names = get_yolo()
        if engine is None and class_names is None:
            return "Vision temporarily unavailable (engine not loaded)."

        # Dynamic prompting: parse comma-separated classes from prompt
        prompt_classes = _parse_prompt_classes(prompt) if prompt else []

        if prompt_classes:
            from vision.detector_yolo import run_inference_with_classes

            with _inference_lock:
                dets = run_inference_with_classes(
                    engine, frame, prompt_classes,
                    class_names=class_names,
                )
        else:
            dets = run_inference_shared(frame)

        face_det = get_face_detector()
        faces = detect_faces(face_det, frame) if face_det else []
        base_desc = describe_scene(dets, face_count=len(faces), class_names=class_names)

        if not prompt or not prompt.strip():
            return f"Objects: {base_desc}. Face count: {len(faces)}."

        q = prompt.strip().lower()
        focus = _PROMPT_SYNONYMS.get(q, q)
        if focus in base_desc.lower():
            return f"Objects: {base_desc}. Face count: {len(faces)}. Note: '{focus}' detected."
        # Check if any of the prompt classes were found
        for pc in prompt_classes:
            if pc.lower() in base_desc.lower():
                return f"Objects: {base_desc}. Face count: {len(faces)}. Note: '{pc}' detected."
        return f"Objects: {base_desc}. Face count: {len(faces)}."
    except Exception as e:
        logger.warning("describe_current_scene failed: %s", e)
        return "Vision temporarily unavailable."


def describe_current_scene_enriched(prompt: str | None = None) -> dict:
    """Full enriched vision pipeline: YOLOE + tracking + depth + vitals + threat.

    Returns a dict with keys:
      description: str (enriched text for LLM context)
      vitals_text: str (compact vitals summary)
      threat_text: str (compact threat summary)
      detections: list[dict] (raw detections)
      tracked: list[dict] (tracked objects as dicts)
      depth_map: ndarray or None
      point_cloud: list[dict] (for hologram)
      vitals: VitalsResult or None
      threat: ThreatAssessment or None
    """
    result = {
        "description": "",
        "vitals_text": "",
        "threat_text": "",
        "detections": [],
        "tracked": [],
        "depth_map": None,
        "point_cloud": [],
        "vitals": None,
        "threat": None,
    }

    try:
        if is_vision_paused():
            result["description"] = "Vision paused (memory conservation mode)."
            return result

        # Thermal throttle check (pause non-essential in portable mode)
        try:
            from utils.power import should_throttle_vision

            if should_throttle_vision():
                logger.info("Thermal/power throttle: skipping enriched vision this cycle")
                # Still do basic YOLOE but skip depth/vitals
        except Exception:
            pass

        from vision.depth import depth_at_boxes, generate_point_cloud
        from vision.detector_mediapipe import detect_faces
        from vision.scene import describe_scene_enriched

        # Drain V4L2 buffer so we analyse the *current* view, not a
        # stale buffered frame from before the camera moved.
        frame = read_frame(drain=True)
        if frame is None:
            result["description"] = "Vision temporarily unavailable (no frame captured)."
            return result

        engine, class_names = get_yolo()

        # 1. YOLOE detection (with optional dynamic prompting)
        prompt_classes = _parse_prompt_classes(prompt) if prompt else []
        if prompt_classes and engine is not None:
            from vision.detector_yolo import run_inference_with_classes

            with _inference_lock:
                dets = run_inference_with_classes(
                    engine, frame, prompt_classes, class_names=class_names,
                )
        elif engine is not None:
            dets = run_inference_shared(frame)
        else:
            dets = []
        result["detections"] = dets

        # 2. Tracking
        tracker = get_tracker()
        tracked = tracker.update(dets) if tracker else []

        # 3. Depth (with portable frame skipping)
        _enriched_call_count_depth = getattr(
            describe_current_scene_enriched, "_depth_counter", 0
        )
        depth_skip = getattr(settings, "PORTABLE_DEPTH_SKIP", 1)
        portable = getattr(settings, "PORTABLE_MODE", False)
        run_depth_this_frame = (not portable) or (_enriched_call_count_depth % depth_skip == 0)
        describe_current_scene_enriched._depth_counter = _enriched_call_count_depth + 1

        depth_map = run_depth_shared(frame) if run_depth_this_frame else None
        result["depth_map"] = depth_map
        depth_values = depth_at_boxes(depth_map, dets) if depth_map is not None else []

        # Attach depth to tracked objects
        if depth_values:
            det_depth_map = {}
            for i, d in enumerate(dets):
                key = tuple(d.get("xyxy", []))
                if i < len(depth_values):
                    det_depth_map[key] = depth_values[i]
            for t in tracked:
                t.depth = det_depth_map.get(tuple(t.xyxy))

        # Only send objects actually detected recently (age <= 1).
        # The tracker keeps "zombie" tracks alive for max_age=30 updates,
        # which at 2-second intervals means 60 seconds of stale overlays.
        # Filtering by age prevents ghost brackets when the camera moves.
        result["tracked"] = [
            {
                "track_id": t.track_id, "xyxy": t.xyxy, "cls": t.cls,
                "class_name": t.class_name, "conf": t.conf,
                "velocity": t.velocity, "depth": t.depth,
                "frames_seen": t.frames_seen,
                "age": t.age,
            }
            for t in tracked
            if t.age <= 1
        ]

        # 4. Point cloud for hologram
        if depth_map is not None:
            result["point_cloud"] = generate_point_cloud(frame, depth_map)

        # 5. Face detection
        face_det = get_face_detector()
        faces = detect_faces(face_det, frame) if face_det else []

        # 6. Vitals (with portable frame skipping)
        _enriched_call_count_vitals = getattr(
            describe_current_scene_enriched, "_vitals_counter", 0
        )
        vitals_skip = getattr(settings, "PORTABLE_VITALS_SKIP", 1)
        run_vitals_this_frame = (not portable) or (_enriched_call_count_vitals % vitals_skip == 0)
        describe_current_scene_enriched._vitals_counter = _enriched_call_count_vitals + 1

        vitals = run_vitals_shared(frame) if run_vitals_this_frame else None
        result["vitals"] = vitals

        # 7. Threat assessment
        # Use a module-level threat scorer (stateful for smoothing)
        threat_scorer = _get_threat_scorer()
        threat = threat_scorer.score_scene(tracked, vitals, depth_map)
        result["threat"] = threat

        # 8. Build enriched description
        result["description"] = describe_scene_enriched(
            dets,
            face_count=len(faces),
            class_names=class_names,
            tracked_objects=tracked,
            depth_values=depth_values,
            vitals=vitals,
            threat=threat,
        )

        # 9. Compact text summaries for LLM context
        if vitals:
            vparts = []
            if vitals.fatigue_level != "unknown":
                vparts.append(vitals.fatigue_level)
            if vitals.posture_label != "unknown":
                vparts.append(f"posture:{vitals.posture_label}")
            if vitals.heart_rate_bpm is not None:
                vparts.append(f"HR:{vitals.heart_rate_bpm:.0f}")
            result["vitals_text"] = ",".join(vparts)

        if threat:
            result["threat_text"] = f"{threat.level}/10 {threat.label}"

        return result
    except Exception as e:
        logger.warning("describe_current_scene_enriched failed: %s", e)
        result["description"] = "Vision temporarily unavailable."
        return result


# ── Threat scorer singleton ───────────────────────────────────────────
_threat_scorer: Any | None = None
_threat_scorer_initialised = False


def _get_threat_scorer():
    """Lazily create and return the shared ThreatScorer."""
    global _threat_scorer, _threat_scorer_initialised
    if _threat_scorer_initialised:
        return _threat_scorer
    with _init_lock:
        if _threat_scorer_initialised:
            return _threat_scorer
        try:
            from vision.threat import ThreatScorer

            _threat_scorer = ThreatScorer()
        except Exception as e:
            logger.warning("Shared ThreatScorer init failed: %s", e)
            _threat_scorer = None
        _threat_scorer_initialised = True
    return _threat_scorer
