"""MJPEG streaming: async generator that grabs frames from the shared camera,
runs YOLOE via the shared engine (behind the inference lock), draws detections,
and encodes JPEG.

Uses the process-wide singletons from ``vision.shared`` so the MJPEG stream
and the orchestrator's ``vision_analyze`` tool share a single camera handle
and a single TensorRT engine (critical on the 8 GB Jetson).
"""

import asyncio
import logging
from collections.abc import AsyncGenerator

logger = logging.getLogger(__name__)


def _grab_annotated_jpeg() -> bytes | None:
    """Read one frame, run YOLOE, draw detections, encode as JPEG bytes."""
    try:
        import cv2
        from vision.shared import get_yolo, read_frame, run_inference_shared
        from vision.visualize import draw_detections_on_frame

        frame = read_frame()
        if frame is None:
            return None

        _, class_names = get_yolo()
        dets = run_inference_shared(frame)
        if dets:
            draw_detections_on_frame(frame, dets, class_names=class_names)

        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
        return buf.tobytes() if ok else None
    except Exception as e:
        logger.warning("MJPEG frame grab failed: %s", e)
        return None


# ── Async generator for StreamingResponse ─────────────────────────────


async def mjpeg_generator(fps: int = 10) -> AsyncGenerator[bytes, None]:
    """Yield multipart JPEG frames for use with ``StreamingResponse``."""
    interval = 1.0 / max(fps, 1)
    while True:
        jpeg = await asyncio.get_running_loop().run_in_executor(None, _grab_annotated_jpeg)
        if jpeg is None:
            await asyncio.sleep(interval)
            continue
        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n"
        )
        await asyncio.sleep(interval)
