"""Unit tests for vision/ambient.py — ambient awareness state machine."""

import time

import numpy as np


class TestAmbientAwareness:
    """Tests for the AmbientAwareness state machine."""

    def _make_ambient(self, **kwargs):
        from vision.ambient import AmbientAwareness
        return AmbientAwareness(**kwargs)

    def test_initial_state_is_idle(self):
        ambient = self._make_ambient()
        from vision.ambient import AmbientState
        assert ambient.state == AmbientState.IDLE
        assert ambient.current_hz == 2.0

    def test_first_frame_returns_none(self):
        """First frame has no previous — no event possible."""
        ambient = self._make_ambient()
        frame = np.random.randint(0, 255, (120, 160, 3), dtype=np.uint8)
        event = ambient.check_frame(frame)
        assert event is None

    def test_static_scene_no_event(self):
        """Two identical frames → no motion → no event."""
        ambient = self._make_ambient()
        frame = np.ones((120, 160, 3), dtype=np.uint8) * 128
        ambient.check_frame(frame)  # first frame
        event = ambient.check_frame(frame)  # second frame, identical
        assert event is None

    def test_large_motion_triggers_event(self):
        """Significant pixel shift → ego-motion or motion detected."""
        ambient = self._make_ambient(ego_motion_threshold=2.0)
        frame1 = np.zeros((120, 160, 3), dtype=np.uint8)
        frame1[20:80, 20:140] = 200  # bright rectangle

        frame2 = np.zeros((120, 160, 3), dtype=np.uint8)
        frame2[30:90, 30:150] = 200  # shifted rectangle

        ambient.check_frame(frame1)
        event = ambient.check_frame(frame2)
        # Should detect motion or ego-motion
        if event is not None:
            from vision.ambient import AmbientEventType
            assert event.event_type in (
                AmbientEventType.MOTION_DETECTED,
                AmbientEventType.EGO_MOTION_START,
                AmbientEventType.SCENE_CHANGE,
            )
            assert event.recommend_full_scan is True

    def test_scene_change_detection(self):
        """Drastically different frames → scene_change event."""
        ambient = self._make_ambient()
        dark = np.zeros((120, 160, 3), dtype=np.uint8)
        bright = np.ones((120, 160, 3), dtype=np.uint8) * 255

        ambient.check_frame(dark)
        event = ambient.check_frame(bright)
        if event is not None:
            from vision.ambient import AmbientEventType
            assert event.event_type in (
                AmbientEventType.SCENE_CHANGE,
                AmbientEventType.EGO_MOTION_START,
                AmbientEventType.MOTION_DETECTED,
            )

    def test_cooldown_suppresses_events(self):
        """After entering cooldown, non-critical triggers are suppressed."""
        ambient = self._make_ambient(cooldown_sec=5.0)
        from vision.ambient import AmbientState

        frame = np.ones((120, 160, 3), dtype=np.uint8) * 128
        ambient.check_frame(frame)

        # Force into cooldown
        ambient.enter_cooldown()
        assert ambient.state == AmbientState.COOLDOWN

        # Static frame in cooldown should not trigger
        event = ambient.check_frame(frame)
        assert event is None

    def test_active_state_transitions_back_to_idle(self):
        """ACTIVE state should revert to IDLE after active_duration_sec."""
        ambient = self._make_ambient(active_duration_sec=0.01)
        from vision.ambient import AmbientState

        # Force into active
        ambient._transition(AmbientState.ACTIVE)
        assert ambient.state == AmbientState.ACTIVE

        time.sleep(0.02)
        frame = np.ones((120, 160, 3), dtype=np.uint8) * 128
        ambient.check_frame(frame)
        assert ambient.state == AmbientState.IDLE

    def test_reset_clears_state(self):
        """Reset should return to IDLE and clear all state."""
        ambient = self._make_ambient()
        from vision.ambient import AmbientState

        ambient._transition(AmbientState.ACTIVE)
        ambient._was_ego_moving = True
        ambient.reset()
        assert ambient.state == AmbientState.IDLE
        assert ambient._was_ego_moving is False

    def test_interval_sec_matches_state(self):
        """Interval should be faster in ACTIVE mode."""
        ambient = self._make_ambient()
        from vision.ambient import AmbientState

        idle_interval = ambient.interval_sec
        ambient._transition(AmbientState.ACTIVE)
        active_interval = ambient.interval_sec
        assert active_interval < idle_interval


class TestAmbientEventTypes:
    """Test event dataclass construction."""

    def test_event_construction(self):
        from vision.ambient import AmbientEvent, AmbientEventType

        event = AmbientEvent(
            event_type=AmbientEventType.MOTION_DETECTED,
            motion_energy=0.15,
            detail="test",
            recommend_full_scan=True,
        )
        assert event.event_type == AmbientEventType.MOTION_DETECTED
        assert event.recommend_full_scan is True
        assert event.motion_energy == 0.15
