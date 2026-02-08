"""Unit tests for vision/camera.py, vision/detector_yolo.py, vision/visualize.py, vision/detector_mediapipe.py."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from vision.camera import open_camera, read_frame
from vision.detector_mediapipe import create_face_detector, detect_faces
from vision.detector_yolo import get_class_names, load_yolo_engine, run_inference
from vision.visualize import draw_detections_on_frame

# ── vision/camera.py ──────────────────────────────────────────────────


@pytest.mark.unit
class TestCamera:
    def test_open_camera_success(self):
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cv2 = MagicMock()
        mock_cv2.VideoCapture.return_value = mock_cap
        with patch.dict("sys.modules", {"cv2": mock_cv2}):
            cap = open_camera(index=0, width=640, height=480, fps=15)
            assert cap is not None
            assert cap.isOpened()

    def test_open_camera_not_opened(self):
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = False
        mock_cv2 = MagicMock()
        mock_cv2.VideoCapture.return_value = mock_cap
        with patch.dict("sys.modules", {"cv2": mock_cv2}):
            cap = open_camera(index=999)
            assert cap is None

    def test_open_camera_with_device_path(self):
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cv2 = MagicMock()
        mock_cv2.VideoCapture.return_value = mock_cap
        with patch.dict("sys.modules", {"cv2": mock_cv2}):
            open_camera(device_path="/dev/video0")
            mock_cv2.VideoCapture.assert_called_with("/dev/video0")

    def test_read_frame_success(self):
        fake_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        mock_cap = MagicMock()
        mock_cap.read.return_value = (True, fake_frame)
        result = read_frame(mock_cap)
        assert result is not None
        assert result.shape == (480, 640, 3)

    def test_read_frame_failure(self):
        mock_cap = MagicMock()
        mock_cap.read.return_value = (False, None)
        assert read_frame(mock_cap) is None

    def test_read_frame_none_cap(self):
        assert read_frame(None) is None


# ── vision/detector_yolo.py ──────────────────────────────────────────


@pytest.mark.unit
class TestDetectorYolo:
    def test_load_engine_missing(self, tmp_path):
        result = load_yolo_engine(tmp_path / "nonexistent.engine")
        assert result is None

    def test_load_engine_success(self, tmp_path):
        engine_file = tmp_path / "test.engine"
        engine_file.write_bytes(b"\x00" * 100)
        mock_yolo = MagicMock()
        with patch("ultralytics.YOLO", return_value=mock_yolo):
            result = load_yolo_engine(engine_file)
            assert result is mock_yolo

    def test_get_class_names_none(self):
        assert get_class_names(None) is None

    def test_get_class_names_dict(self):
        model = MagicMock()
        model.names = {0: "person", 1: "bicycle"}
        assert get_class_names(model) == {0: "person", 1: "bicycle"}

    def test_get_class_names_list(self):
        model = MagicMock()
        model.names = ["person", "bicycle", "car"]
        result = get_class_names(model)
        assert result == {0: "person", 1: "bicycle", 2: "car"}

    def test_get_class_names_from_inner_model(self):
        model = MagicMock(spec=[])
        inner = MagicMock()
        inner.names = {0: "person"}
        model.model = inner
        # No direct 'names' attribute
        result = get_class_names(model)
        assert result == {0: "person"}

    def test_run_inference_success(self):
        mock_box = MagicMock()
        mock_box.xyxy = [MagicMock(tolist=MagicMock(return_value=[10, 20, 100, 200]))]
        mock_box.conf = [MagicMock(__float__=MagicMock(return_value=0.9))]
        mock_box.cls = [MagicMock(__int__=MagicMock(return_value=0))]

        mock_result = MagicMock()
        mock_result.boxes = [mock_box]

        model = MagicMock()
        model.return_value = [mock_result]

        fake_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        dets = run_inference(model, fake_frame)
        assert len(dets) == 1
        assert dets[0]["cls"] == 0

    def test_run_inference_none_model(self):
        assert run_inference(None, np.zeros((10, 10, 3))) == []

    def test_run_inference_none_frame(self):
        assert run_inference(MagicMock(), None) == []

    def test_run_inference_exception(self):
        model = MagicMock()
        model.side_effect = RuntimeError("CUDA error")
        result = run_inference(model, np.zeros((10, 10, 3)))
        assert result == []


# ── vision/detector_mediapipe.py ──────────────────────────────────────


@pytest.mark.unit
class TestMediaPipe:
    def test_create_face_detector_success(self):
        mock_mp = MagicMock()
        mock_detector = MagicMock()
        mock_mp.solutions.face_detection.FaceDetection.return_value = mock_detector
        with patch.dict("sys.modules", {"mediapipe": mock_mp}):
            det = create_face_detector()
            assert det is mock_detector

    def test_detect_faces_no_detector(self):
        assert detect_faces(None, np.zeros((10, 10, 3))) == []

    def test_detect_faces_no_frame(self):
        assert detect_faces(MagicMock(), None) == []

    def test_detect_faces_no_detections(self):
        mock_cv2 = MagicMock()
        mock_cv2.cvtColor.return_value = np.zeros((10, 10, 3), dtype=np.uint8)
        mock_detector = MagicMock()
        mock_results = MagicMock()
        mock_results.detections = None
        mock_detector.process.return_value = mock_results

        with patch.dict("sys.modules", {"cv2": mock_cv2}):
            result = detect_faces(mock_detector, np.zeros((10, 10, 3)))
            assert result == []


# ── vision/visualize.py ───────────────────────────────────────────────


@pytest.mark.unit
class TestVisualize:
    def test_draw_no_detections(self):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = draw_detections_on_frame(frame, [])
        assert result is frame

    def test_draw_with_detections(self):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        dets = [{"xyxy": [10, 20, 100, 200], "conf": 0.9, "cls": 0}]
        mock_cv2 = MagicMock()
        mock_cv2.getTextSize.return_value = ((50, 10), 0)
        with patch.dict("sys.modules", {"cv2": mock_cv2}):
            result = draw_detections_on_frame(frame, dets, class_names={0: "person"})
            assert result is not None

    def test_draw_with_invalid_xyxy(self):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        dets = [{"xyxy": None, "conf": 0.5, "cls": 0}]
        # Should not crash
        result = draw_detections_on_frame(frame, dets)
        assert result is frame

    def test_draw_with_coco_names(self):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        dets = [{"xyxy": [0, 0, 10, 10], "conf": 0.8, "cls": 56}]  # 56 = chair
        mock_cv2 = MagicMock()
        mock_cv2.getTextSize.return_value = ((30, 10), 0)
        with patch.dict("sys.modules", {"cv2": mock_cv2}):
            result = draw_detections_on_frame(frame, dets)
            assert result is not None
