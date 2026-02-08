"""Unit tests for vision.shared – process-wide singletons (mocked hardware)."""

from unittest.mock import MagicMock, patch

import pytest


@pytest.mark.unit
class TestCheckCuda:
    def test_cuda_available(self):
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.get_device_name.return_value = "Orin (nvgpu)"
        with patch.dict("sys.modules", {"torch": mock_torch}):
            from vision.shared import check_cuda
            ok, msg = check_cuda()
            assert ok is True
            assert "CUDA OK" in msg

    def test_cuda_not_available(self):
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = False
        with patch.dict("sys.modules", {"torch": mock_torch}):
            from vision.shared import check_cuda
            ok, msg = check_cuda()
            assert ok is False
            assert "not compiled" in msg.lower() or "CUDA" in msg


@pytest.mark.unit
class TestSharedCamera:
    def test_get_camera_returns_none_when_no_device(self):
        """When open_camera returns None, get_camera should return None."""
        import vision.shared as vs

        # Reset singleton state
        vs._camera = None
        vs._camera_initialised = False
        with patch("vision.camera.open_camera", return_value=None):
            cam = vs.get_camera()
            assert cam is None
            assert vs._camera_initialised is True
        # Cleanup
        vs._camera_initialised = False

    def test_read_frame_returns_none_without_camera(self):
        import vision.shared as vs

        vs._camera = None
        vs._camera_initialised = True
        frame = vs.read_frame()
        assert frame is None
        vs._camera_initialised = False

    def test_release_camera(self):
        import vision.shared as vs

        mock_cap = MagicMock()
        vs._camera = mock_cap
        vs._camera_initialised = True
        vs.release_camera()
        mock_cap.release.assert_called_once()
        assert vs._camera is None
        assert vs._camera_initialised is False


@pytest.mark.unit
class TestSharedYolo:
    def test_get_yolo_returns_none_when_no_engine(self):
        import vision.shared as vs

        vs._yolo_engine = None
        vs._yolo_class_names = None
        vs._yolo_initialised = False
        with patch.object(vs.settings, "yolo_engine_exists", return_value=False):
            engine, names = vs.get_yolo()
            assert engine is None
            assert names is None
        vs._yolo_initialised = False

    def test_run_inference_shared_returns_empty_without_engine(self):
        import vision.shared as vs

        vs._yolo_engine = None
        vs._yolo_initialised = True
        result = vs.run_inference_shared(MagicMock())
        assert result == []
        vs._yolo_initialised = False


@pytest.mark.unit
class TestSharedFaceDetector:
    def test_get_face_detector_caches(self):
        import vision.shared as vs

        vs._face_detector = None
        vs._face_detector_initialised = False
        mock_det = MagicMock()
        with patch("vision.detector_mediapipe.create_face_detector", return_value=mock_det) as mock_create:
            det1 = vs.get_face_detector()
            det2 = vs.get_face_detector()
            assert det1 is mock_det
            assert det2 is mock_det
            mock_create.assert_called_once()
        vs._face_detector = None
        vs._face_detector_initialised = False


@pytest.mark.unit
class TestDescribeCurrentScene:
    def test_returns_description(self):
        import vision.shared as vs

        fake_frame = MagicMock()
        fake_dets = [{"xyxy": [0, 0, 100, 100], "conf": 0.9, "cls": 0}]
        with (
            patch.object(vs, "read_frame", return_value=fake_frame),
            patch.object(vs, "get_yolo", return_value=(MagicMock(), {0: "person"})),
            patch.object(vs, "run_inference_shared", return_value=fake_dets),
            patch.object(vs, "get_face_detector", return_value=None),
        ):
            result = vs.describe_current_scene()
            assert "Objects:" in result

    def test_returns_unavailable_when_no_frame(self):
        import vision.shared as vs

        with patch.object(vs, "read_frame", return_value=None):
            result = vs.describe_current_scene()
            assert "unavailable" in result.lower() or "no frame" in result.lower()
