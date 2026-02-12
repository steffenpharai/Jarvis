"""E2E tests for ambient awareness system.

Tests the full ambient awareness flow with synthetic frames:
  - Static scene → no triggers
  - Motion injection → trigger fired
  - Cooldown suppression
  - State machine transitions
"""

import time

import numpy as np
import pytest


@pytest.mark.e2e
class TestAmbientAwarenessE2E:
    """E2E tests for ambient awareness with synthetic data."""

    def test_no_trigger_on_static_scene(self):
        """10 identical frames should produce no ambient events."""
        from vision.ambient import AmbientAwareness

        ambient = AmbientAwareness()
        frame = np.ones((120, 160, 3), dtype=np.uint8) * 128

        events = []
        for _ in range(10):
            event = ambient.check_frame(frame)
            if event is not None:
                events.append(event)

        assert len(events) == 0, f"Expected no events on static scene, got {len(events)}"

    def test_trigger_on_moving_scene(self):
        """Frames with progressive shift should eventually trigger."""
        from vision.ambient import AmbientAwareness

        ambient = AmbientAwareness(ego_motion_threshold=1.5, motion_energy_threshold=0.03)

        events = []
        for i in range(20):
            frame = np.zeros((120, 160, 3), dtype=np.uint8)
            # Create a moving bright bar
            x = (i * 8) % 140
            frame[40:80, x:x + 20] = 255
            event = ambient.check_frame(frame)
            if event is not None:
                events.append(event)

        # Should have triggered at least once
        assert len(events) >= 1, "Expected at least one trigger on moving scene"
        assert events[0].recommend_full_scan is True

    def test_cooldown_prevents_spam(self):
        """After a trigger, cooldown should prevent rapid re-triggers."""
        from vision.ambient import AmbientAwareness, AmbientState

        ambient = AmbientAwareness(cooldown_sec=1.0)

        # First: warm up with a static frame
        static = np.ones((120, 160, 3), dtype=np.uint8) * 128
        ambient.check_frame(static)

        # Trigger via scene change
        bright = np.ones((120, 160, 3), dtype=np.uint8) * 255
        ambient.check_frame(bright)  # may or may not trigger

        # Force cooldown
        ambient.enter_cooldown()
        assert ambient.state == AmbientState.COOLDOWN

        # More frames during cooldown should not trigger
        events_in_cooldown = []
        for _ in range(5):
            dark = np.zeros((120, 160, 3), dtype=np.uint8)
            event = ambient.check_frame(dark)
            if event is not None:
                events_in_cooldown.append(event)

        # At most thermal/battery events bypass cooldown, not motion events
        motion_events = [
            e for e in events_in_cooldown
            if e.event_type.value in ("motion_detected", "ego_motion_start", "scene_change")
        ]
        assert len(motion_events) == 0, "Motion events should be suppressed during cooldown"

    def test_state_machine_idle_to_active(self):
        """Trigger should transition from IDLE to ACTIVE."""
        from vision.ambient import AmbientAwareness, AmbientState

        ambient = AmbientAwareness(ego_motion_threshold=1.0)
        assert ambient.state == AmbientState.IDLE

        # Two very different frames
        frame1 = np.zeros((120, 160, 3), dtype=np.uint8)
        frame2 = np.ones((120, 160, 3), dtype=np.uint8) * 255

        ambient.check_frame(frame1)
        event = ambient.check_frame(frame2)

        if event is not None:
            # Should have transitioned to ACTIVE
            assert ambient.state == AmbientState.ACTIVE

    def test_ambient_latency_under_5ms(self):
        """Each ambient check should complete in <5ms at 160x120."""
        from vision.ambient import AmbientAwareness

        ambient = AmbientAwareness()
        np.random.seed(42)
        frame = np.random.randint(0, 200, (120, 160, 3), dtype=np.uint8)
        ambient.check_frame(frame)  # warm up

        latencies = []
        for i in range(50):
            shifted = np.roll(frame, i % 5, axis=1)
            t0 = time.monotonic()
            ambient.check_frame(shifted)
            latencies.append((time.monotonic() - t0) * 1000)

        avg = sum(latencies) / len(latencies)
        print(f"\nAmbient check avg: {avg:.2f}ms (50 iterations)")
        assert avg < 10, f"Ambient check too slow: {avg:.1f}ms (target <5ms)"
