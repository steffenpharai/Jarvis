"""Draw YOLOE detections on frames and run live camera visualization."""

import logging
from typing import Any

from config import settings

from vision.scene import COCO_NAMES

logger = logging.getLogger(__name__)


def draw_detections_on_frame(
    frame: Any,
    detections: list,
    class_names: dict[int, str] | tuple | None = None,
    box_color: tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2,
    font_scale: float = 0.5,
) -> Any:
    """Draw bounding boxes and labels on frame. Modifies frame in place; returns it."""
    if not detections:
        return frame
    try:
        import cv2
    except ImportError:
        return frame
    if class_names is None:
        class_names = COCO_NAMES
    for d in detections:
        xyxy = d.get("xyxy")
        conf = d.get("conf", 0)
        cls_id = d.get("cls", 0)
        if xyxy is None or len(xyxy) != 4:
            continue
        x1, y1, x2, y2 = [int(round(x)) for x in xyxy]
        if isinstance(class_names, dict):
            label = class_names.get(cls_id, f"class_{cls_id}")
        else:
            label = class_names[cls_id] if cls_id < len(class_names) else f"class_{cls_id}"
        label = f"{label} {conf:.2f}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, thickness)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
        cv2.rectangle(frame, (x1, y1 - th - 4), (x1 + tw, y1), box_color, -1)
        cv2.putText(
            frame,
            label,
            (x1, y1 - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )
    return frame


def run_live_visualization() -> None:
    """Live camera + YOLOE-26N detections in OpenCV window. Press 'q' to quit.

    Uses the process-wide shared camera and engine singletons so no
    duplicate resources are allocated.
    """
    import cv2

    from vision.shared import get_yolo, read_frame, release_camera, run_inference_shared

    engine, class_names = get_yolo()
    if not engine:
        logger.error("YOLOE engine not loaded from %s", settings.YOLOE_ENGINE_PATH)
        return
    logger.info("YOLOE-26N live visualization. Press 'q' to quit.")
    try:
        while True:
            frame = read_frame()
            if frame is None:
                continue
            dets = run_inference_shared(frame)
            draw_detections_on_frame(frame, dets, class_names=class_names)
            cv2.putText(
                frame,
                "YOLOE-26N | q=quit",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )
            cv2.imshow("YOLOE-26N", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        release_camera()
        cv2.destroyAllWindows()
