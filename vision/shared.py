"""Process-wide singletons for camera, YOLOE engine, and MediaPipe face detector.

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


def read_frame() -> Any | None:
    """Read one frame from the shared camera.  Thread-safe."""
    cap = get_camera()
    if cap is None:
        return None
    with _frame_lock:
        from vision.camera import read_frame as _read

        return _read(cap)


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


def describe_current_scene(prompt: str | None = None) -> str:
    """Grab a frame, run YOLOE + face detection, return a text description.

    This is the single implementation backing both ``tools.vision_analyze``
    and the orchestrator's proactive-idle vision check.
    """
    try:
        from vision.detector_mediapipe import detect_faces
        from vision.scene import COCO_NAMES, describe_scene

        frame = read_frame()
        if frame is None:
            return "Vision temporarily unavailable (no frame captured)."

        _, class_names = get_yolo()
        dets = run_inference_shared(frame)
        if not dets and class_names is None:
            return "Vision temporarily unavailable (engine not loaded)."

        face_det = get_face_detector()
        faces = detect_faces(face_det, frame) if face_det else []
        base_desc = describe_scene(dets, face_count=len(faces), class_names=class_names)

        if not prompt or not prompt.strip():
            return f"Objects: {base_desc}. Face count: {len(faces)}."

        q = prompt.strip().lower()
        focus = _PROMPT_SYNONYMS.get(q, q)
        if focus in base_desc.lower():
            return f"Objects: {base_desc}. Face count: {len(faces)}. Note: '{focus}' detected."
        if focus in [c.lower() for c in COCO_NAMES]:
            return f"Objects: {base_desc}. Face count: {len(faces)}. Note: '{focus}' detected."
        return f"Objects: {base_desc}. Face count: {len(faces)}."
    except Exception as e:
        logger.warning("describe_current_scene failed: %s", e)
        return "Vision temporarily unavailable."
