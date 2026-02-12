"""Benchmark tests for vision pipeline components.

Measures latency of individual modules with synthetic data.
Target: <100ms per frame at 10 FPS portable, <200ms at 30 FPS desktop.
"""

import time

import numpy as np
import pytest


def _timed(fn, *args, **kwargs):
    """Call fn and return (result, elapsed_ms)."""
    start = time.monotonic()
    result = fn(*args, **kwargs)
    elapsed = (time.monotonic() - start) * 1000
    return result, elapsed


@pytest.mark.e2e
class TestBenchmarkTracker:
    def test_tracker_latency(self):
        """ByteTrackLite update should be <5ms per frame."""
        from vision.tracker import ByteTrackLite

        tracker = ByteTrackLite(min_hits=1)
        dets = [
            {"xyxy": [i * 50, i * 50, i * 50 + 80, i * 50 + 80], "conf": 0.9, "cls": 0}
            for i in range(10)
        ]

        latencies = []
        for _ in range(100):
            _, ms = _timed(tracker.update, dets)
            latencies.append(ms)

        avg = sum(latencies) / len(latencies)
        print(f"\nTracker avg latency: {avg:.2f}ms (100 iterations, 10 detections)")
        assert avg < 50, f"Tracker too slow: {avg:.1f}ms avg (target <5ms)"


@pytest.mark.e2e
class TestBenchmarkThreat:
    def test_threat_scorer_latency(self):
        """ThreatScorer should be <1ms per frame."""
        from vision.threat import ThreatScorer
        from vision.tracker import TrackedObject

        scorer = ThreatScorer()
        tracks = [
            TrackedObject(
                track_id=i, xyxy=[i * 50, i * 50, i * 50 + 80, i * 50 + 80],
                cls=0, class_name="person", velocity=[10.0, 5.0],
                frames_seen=10, age=0, last_seen=time.monotonic(),
            )
            for i in range(5)
        ]

        latencies = []
        for _ in range(100):
            _, ms = _timed(scorer.score_scene, tracks, None, None)
            latencies.append(ms)

        avg = sum(latencies) / len(latencies)
        print(f"\nThreat scorer avg latency: {avg:.3f}ms (100 iterations, 5 tracks)")
        assert avg < 10, f"Threat scorer too slow: {avg:.1f}ms avg (target <1ms)"


@pytest.mark.e2e
class TestBenchmarkDepthUtils:
    def test_depth_at_boxes_latency(self):
        """depth_at_boxes should be <5ms for 20 detections."""
        from vision.depth import depth_at_boxes

        depth_map = np.random.rand(720, 1280).astype(np.float32)
        dets = [
            {"xyxy": [i * 60, i * 30, i * 60 + 80, i * 30 + 80]}
            for i in range(20)
        ]

        latencies = []
        for _ in range(50):
            _, ms = _timed(depth_at_boxes, depth_map, dets)
            latencies.append(ms)

        avg = sum(latencies) / len(latencies)
        print(f"\ndepth_at_boxes avg latency: {avg:.2f}ms (50 iterations, 20 boxes)")
        assert avg < 50, f"depth_at_boxes too slow: {avg:.1f}ms avg (target <5ms)"

    def test_point_cloud_generation_latency(self):
        """Point cloud generation should be <50ms at step=8."""
        from vision.depth import generate_point_cloud

        frame = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)
        depth = np.random.rand(720, 1280).astype(np.float32) * 0.8 + 0.1

        latencies = []
        for _ in range(10):
            _, ms = _timed(generate_point_cloud, frame, depth, 8, 5000)
            latencies.append(ms)

        avg = sum(latencies) / len(latencies)
        print(f"\nPoint cloud avg latency: {avg:.1f}ms (10 iterations, step=8)")
        assert avg < 200, f"Point cloud too slow: {avg:.1f}ms avg (target <50ms)"


@pytest.mark.e2e
class TestBenchmarkScene:
    def test_enriched_scene_description_latency(self):
        """Enriched scene description should be <5ms (no inference, just formatting)."""
        from vision.scene import describe_scene_enriched
        from vision.threat import ThreatAssessment
        from vision.vitals import VitalsResult

        dets = [{"cls": 0, "conf": 0.9}] * 10
        vitals = VitalsResult(fatigue_level="mild", posture_label="good")
        threat = ThreatAssessment(level=2, label="clear")

        latencies = []
        for _ in range(100):
            _, ms = _timed(
                describe_scene_enriched, dets, face_count=1,
                vitals=vitals, threat=threat,
            )
            latencies.append(ms)

        avg = sum(latencies) / len(latencies)
        print(f"\nEnriched scene desc avg latency: {avg:.3f}ms (100 iterations)")
        assert avg < 20, f"Scene description too slow: {avg:.1f}ms avg (target <5ms)"


@pytest.mark.e2e
class TestBenchmarkPerception:
    """Benchmark the fused perception pipeline (flow + ego-motion + trajectory)."""

    def test_perception_pipeline_latency(self):
        """Full perception pipeline should be <15ms at 320x240 with 10 objects."""
        from vision.flow import FlowMethod
        from vision.perception import PerceptionPipeline
        from vision.tracker import TrackedObject

        pipeline = PerceptionPipeline(
            flow_method=FlowMethod.DIS,
            flow_resize=(320, 240),
            fps=10.0,
            portable_mode=True,
        )

        np.random.seed(42)
        base = np.random.randint(0, 200, (240, 320, 3), dtype=np.uint8)
        tracks = [
            TrackedObject(
                track_id=i,
                xyxy=[i * 30, i * 20, i * 30 + 60, i * 20 + 60],
                cls=0, class_name="person",
                velocity=[float(i), float(i * 0.5)],
                frames_seen=10, age=0, last_seen=time.monotonic(),
            )
            for i in range(10)
        ]
        dets = [{"xyxy": t.xyxy, "conf": 0.9, "cls": 0} for t in tracks]

        # Warm up
        pipeline.process_frame(base, dets, tracks)

        latencies = []
        for i in range(100):
            shifted = np.roll(base, i % 10, axis=1)
            result, _ = _timed(pipeline.process_frame, shifted, dets, tracks)
            latencies.append(result.total_ms)

        avg = sum(latencies) / len(latencies)
        p95 = sorted(latencies)[94]
        print(f"\nPerception pipeline: avg={avg:.1f}ms  p95={p95:.1f}ms")
        assert avg < 20, f"Perception avg {avg:.1f}ms too slow (target <15ms)"

    def test_ego_motion_cache_speedup(self):
        """Ego-motion caching should reduce average ego-motion time."""
        from vision.ego_motion import estimate_ego_motion, reset_ego_cache

        np.random.seed(42)
        # Static scene points
        prev = np.random.rand(80, 2).astype(np.float64) * 300 + 10
        curr = prev + 0.3  # tiny motion → static

        reset_ego_cache()

        latencies = []
        for _ in range(50):
            _, ms = _timed(estimate_ego_motion, prev, curr, motion_threshold=1.0)
            latencies.append(ms)

        avg = sum(latencies) / len(latencies)
        print(f"\nEgo-motion (static, with cache): avg={avg:.3f}ms")
        # Should be very fast due to caching
        assert avg < 5, f"Ego-motion with cache too slow: {avg:.1f}ms"
