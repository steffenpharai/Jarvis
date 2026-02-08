"""Unit tests for server/streaming.py – MJPEG frame generation."""

from unittest.mock import patch

import numpy as np
import pytest
from server.streaming import _grab_annotated_jpeg, mjpeg_generator


@pytest.mark.unit
class TestGrabAnnotatedJpeg:
    def test_returns_none_when_no_frame(self):
        with patch("vision.shared.read_frame", return_value=None):
            result = _grab_annotated_jpeg()
            assert result is None

    def test_returns_jpeg_bytes(self):
        fake_frame = np.zeros((480, 640, 3), dtype=np.uint8)

        with (
            patch("vision.shared.read_frame", return_value=fake_frame),
            patch("vision.shared.get_yolo", return_value=(None, None)),
            patch("vision.shared.run_inference_shared", return_value=[]),
        ):
            import cv2

            with patch.object(cv2, "imencode", return_value=(True, np.array([0xFF, 0xD8], dtype=np.uint8))):
                result = _grab_annotated_jpeg()
                assert result is not None
                assert isinstance(result, bytes)

    def test_returns_none_on_encode_failure(self):
        fake_frame = np.zeros((480, 640, 3), dtype=np.uint8)

        with (
            patch("vision.shared.read_frame", return_value=fake_frame),
            patch("vision.shared.get_yolo", return_value=(None, None)),
            patch("vision.shared.run_inference_shared", return_value=[]),
        ):
            import cv2

            with patch.object(cv2, "imencode", return_value=(False, None)):
                result = _grab_annotated_jpeg()
                assert result is None

    def test_handles_exception(self):
        with patch("vision.shared.read_frame", side_effect=RuntimeError("camera error")):
            result = _grab_annotated_jpeg()
            assert result is None


@pytest.mark.asyncio
@pytest.mark.unit
async def test_mjpeg_generator_yields_frames():
    """Generator should yield MJPEG-formatted frames."""
    jpeg_bytes = b"\xff\xd8\xff\xe0fake_jpeg"

    call_count = 0

    def fake_grab():
        nonlocal call_count
        call_count += 1
        if call_count <= 2:
            return jpeg_bytes
        return None  # stop after 2 frames

    with patch("server.streaming._grab_annotated_jpeg", side_effect=fake_grab):
        gen = mjpeg_generator(fps=100)
        frames = []
        async for chunk in gen:
            frames.append(chunk)
            if len(frames) >= 2:
                break

    assert len(frames) == 2
    assert b"--frame" in frames[0]
    assert b"Content-Type: image/jpeg" in frames[0]
