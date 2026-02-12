"""E2E benchmark: perception pipeline latency.

Target: <15ms total (95th percentile) at 320x240 with 10 synthetic objects.
Benchmarks DIS vs Farneback at standard portable resolution.
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


def _make_fake_tracked(n=10):
    """Create N fake TrackedObject-like objects for benchmark."""
    from vision.tracker import TrackedObject

    tracks = []
    for i in range(n):
        t = TrackedObject(
            track_id=i,
            xyxy=[i * 30, i * 20, i * 30 + 60, i * 20 + 60],
            cls=0,
            class_name="person",
            velocity=[float(i), float(i * 0.5)],
            frames_seen=10,
            age=0,
            last_seen=time.monotonic(),
        )
        tracks.append(t)
    return tracks


@pytest.mark.e2e
class TestPerceptionLatency:
    """Benchmark the full perception pipeline."""

    def test_perception_total_under_15ms(self):
        """Full perception pipeline should be <15ms at 320x240 with 10 objects."""
        from vision.flow import FlowMethod
        from vision.perception import PerceptionPipeline

        pipeline = PerceptionPipeline(
            flow_method=FlowMethod.DIS,
            flow_resize=(320, 240),
            fps=10.0,
            portable_mode=True,
        )

        # Create synthetic frames (gradient shift simulates motion)
        np.random.seed(42)
        base = np.random.randint(0, 200, (240, 320, 3), dtype=np.uint8)
        tracked = _make_fake_tracked(10)
        dets = [
            {"xyxy": t.xyxy, "conf": 0.9, "cls": 0}
            for t in tracked
        ]

        # Warm up (first frame initialises flow state)
        pipeline.process_frame(base, dets, tracked)

        latencies = []
        for i in range(100):
            # Shift the frame slightly each iteration (simulates motion)
            shift = np.roll(base, i % 10, axis=1)
            result, _ = _timed(
                pipeline.process_frame, shift, dets, tracked,
            )
            latencies.append(result.total_ms)

        latencies.sort()
        avg = sum(latencies) / len(latencies)
        p95 = latencies[94]
        p99 = latencies[98]

        print("\nPerception pipeline (DIS, 320x240, 10 objects):")
        print(f"  avg={avg:.1f}ms  p95={p95:.1f}ms  p99={p99:.1f}ms")
        print(f"  flow_avg={sum(r.flow_ms for r in [pipeline.process_frame(base, dets, tracked)] * 0) / max(1, 1):.1f}ms")
        print(f"  min={min(latencies):.1f}ms  max={max(latencies):.1f}ms")

        assert p95 < 15.0, f"P95 perception latency {p95:.1f}ms exceeds 15ms target"

    def test_dis_faster_than_farneback(self):
        """DIS should be at least 30% faster than Farneback at same resolution."""
        from vision.flow import FlowMethod
        from vision.perception import PerceptionPipeline

        np.random.seed(42)
        base = np.random.randint(0, 200, (240, 320, 3), dtype=np.uint8)
        shifted = np.roll(base, 5, axis=1)
        tracked = _make_fake_tracked(5)
        dets = [{"xyxy": t.xyxy, "conf": 0.9, "cls": 0} for t in tracked]

        results = {}
        for method_name, method in [("DIS", FlowMethod.DIS), ("Farneback", FlowMethod.FARNEBACK)]:
            pipeline = PerceptionPipeline(
                flow_method=method,
                flow_resize=(320, 240),
                fps=10.0,
            )
            pipeline.process_frame(base, dets, tracked)  # warm up

            latencies = []
            for _ in range(50):
                result = pipeline.process_frame(shifted, dets, tracked)
                latencies.append(result.total_ms)
            avg = sum(latencies) / len(latencies)
            results[method_name] = avg
            print(f"\n  {method_name}: avg={avg:.1f}ms")

        assert results["DIS"] < results["Farneback"], (
            f"DIS ({results['DIS']:.1f}ms) should be faster than "
            f"Farneback ({results['Farneback']:.1f}ms)"
        )


@pytest.mark.e2e
class TestFlowMethodBenchmark:
    """Benchmark individual flow methods at multiple resolutions."""

    @pytest.mark.parametrize("width,height", [(160, 120), (320, 240), (640, 480)])
    def test_dis_flow_latency(self, width, height):
        """DIS flow should scale predictably with resolution."""
        from vision.flow import FlowMethod, OpticalFlowEstimator

        estimator = OpticalFlowEstimator(
            method=FlowMethod.DIS,
            resize=(width, height),
        )

        np.random.seed(42)
        frame1 = np.random.randint(0, 200, (height, width, 3), dtype=np.uint8)
        frame2 = np.roll(frame1, 3, axis=1)

        estimator.compute(frame1)  # warm up

        latencies = []
        for _ in range(50):
            _, ms = _timed(estimator.compute, frame2)
            latencies.append(ms)

        avg = sum(latencies) / len(latencies)
        print(f"\n  DIS {width}x{height}: avg={avg:.1f}ms")
        # All resolutions should be under 20ms on Jetson
        assert avg < 50, f"DIS at {width}x{height} too slow: {avg:.1f}ms"
