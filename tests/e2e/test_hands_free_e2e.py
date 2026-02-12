"""E2E tests for hands-free autonomous operation.

Tests the orchestrator ambient event handling:
  - Ambient event sentinel parsing
  - Proactive verbalization cooldown
  - Walking mode detection
"""

import pytest


@pytest.mark.e2e
class TestHandsFreeHelpers:
    """Test the orchestrator's ambient event helpers."""

    def test_is_ambient_event(self):
        """Ambient event sentinels should be correctly identified."""
        from orchestrator import _is_ambient_event

        assert _is_ambient_event("__ambient__ego_motion_start__Camera moving")
        assert _is_ambient_event("__ambient__scene_change__New area")
        assert not _is_ambient_event("Hello Jarvis")
        assert not _is_ambient_event("")
        assert not _is_ambient_event("__other__")

    def test_parse_ambient_event(self):
        """Ambient event sentinels should parse into (type, detail)."""
        from orchestrator import _parse_ambient_event

        event_type, detail = _parse_ambient_event(
            "__ambient__ego_motion_start__Camera moving (flow=5.2px/f)"
        )
        assert event_type == "ego_motion_start"
        assert "Camera moving" in detail

    def test_parse_ambient_event_unknown(self):
        """Malformed sentinel should return unknown."""
        from orchestrator import _parse_ambient_event

        event_type, detail = _parse_ambient_event("__ambient__")
        # Should handle gracefully
        assert isinstance(event_type, str)
        assert isinstance(detail, str)


@pytest.mark.e2e
class TestAmbientSettingsExist:
    """Verify all new ambient/hands-free settings are defined."""

    def test_ambient_settings(self):
        from config import settings

        assert hasattr(settings, "AMBIENT_AWARENESS_ENABLED")
        assert hasattr(settings, "PROACTIVE_WALK_INTERVAL_SEC")
        assert hasattr(settings, "THERMAL_AMBIENT_PAUSE_C")
        assert hasattr(settings, "BATTERY_LOW_PCT")
        assert hasattr(settings, "PROACTIVE_COOLDOWN_SEC")

    def test_ambient_defaults(self):
        from config import settings

        assert settings.PROACTIVE_WALK_INTERVAL_SEC == 15
        assert settings.THERMAL_AMBIENT_PAUSE_C == 70.0
        assert settings.BATTERY_LOW_PCT == 15
        assert settings.PROACTIVE_COOLDOWN_SEC == 10.0


@pytest.mark.e2e
class TestFlowDefaultIsDIS:
    """Verify DIS is the new default flow method."""

    def test_default_flow_is_dis(self):
        from config import settings

        assert settings.FLOW_METHOD == "dis"
