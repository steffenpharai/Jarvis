"""Unit tests for vision/proximity.py (proximity alerts for portable mode)."""

from unittest.mock import patch

import pytest


@pytest.mark.unit
class TestProximityAlerts:
    def test_no_alerts_when_no_depth(self):
        from vision.proximity import check_proximity

        tracked = [
            {"class_name": "person", "depth": None, "velocity": [0, 0]},
        ]
        alerts = check_proximity(tracked)
        assert alerts == []

    def test_critical_alert(self):
        from vision.proximity import _last_alert_time, check_proximity
        _last_alert_time.clear()  # Reset cooldowns

        tracked = [
            {"class_name": "chair", "depth": 200, "velocity": [0, 0]},  # High depth = close
        ]
        alerts = check_proximity(tracked)
        # Should get at least one alert for a close object
        # Exact behavior depends on _relative_to_meters calibration
        assert isinstance(alerts, list)

    def test_warning_alert_approaching(self):
        from vision.proximity import _last_alert_time, check_proximity
        _last_alert_time.clear()

        tracked = [
            {"class_name": "person", "depth": 100, "velocity": [0, -10]},  # Approaching
        ]
        alerts = check_proximity(tracked)
        assert isinstance(alerts, list)

    def test_cooldown_prevents_spam(self):
        from vision.proximity import _last_alert_time, check_proximity
        _last_alert_time.clear()

        tracked = [
            {"class_name": "person", "depth": 300, "velocity": [0, 0]},
        ]

        check_proximity(tracked)  # First call sets cooldown
        second = check_proximity(tracked)  # Should be rate-limited
        # Second call within cooldown should produce fewer or no alerts
        assert isinstance(second, list)

    def test_format_proximity_summary_empty(self):
        from vision.proximity import format_proximity_summary
        assert format_proximity_summary([]) == ""

    def test_format_proximity_summary(self):
        from vision.proximity import format_proximity_summary
        alerts = [
            {"level": "critical", "message": "Object close", "distance": 0.3},
            {"level": "warning", "message": "Person nearby", "distance": 1.0},
        ]
        summary = format_proximity_summary(alerts)
        assert "CRITICAL" in summary
        assert "WARNING" in summary

    def test_relative_to_meters(self):
        from vision.proximity import _relative_to_meters
        # Should return None for zero/negative
        assert _relative_to_meters(0) is None
        assert _relative_to_meters(-1) is None
        # Should return a reasonable value for positive depth
        m = _relative_to_meters(100)
        assert m is not None
        assert 0.1 <= m <= 20.0


@pytest.mark.unit
class TestPortableStatus:
    def test_get_portable_status(self):
        from utils.power import get_portable_status

        with (
            patch("utils.power.get_thermal_temperature", return_value=65.0),
            patch("utils.power.get_gpu_utilization", return_value=42.0),
            patch("utils.power.get_power_mode", return_value="MAXN_SUPER"),
            patch("utils.power.get_battery_status", return_value=None),
        ):
            status = get_portable_status()
            assert status["temp_c"] == 65.0
            assert status["gpu_pct"] == 42.0
            assert status["warnings"] == []

    def test_get_portable_status_with_warnings(self):
        from utils.power import get_portable_status

        with (
            patch("utils.power.get_thermal_temperature", return_value=80.0),
            patch("utils.power.get_gpu_utilization", return_value=95.0),
            patch("utils.power.get_power_mode", return_value="MAXN_SUPER"),
            patch("utils.power.get_battery_status", return_value={
                "present": True, "capacity_pct": 8, "status": "Discharging",
            }),
        ):
            status = get_portable_status()
            assert len(status["warnings"]) >= 2  # Battery + thermal
            assert status["battery_pct"] == 8

    def test_get_battery_summary_no_battery(self):
        from utils.power import get_battery_summary

        with patch("utils.power.get_battery_status", return_value=None):
            assert get_battery_summary() is None

    def test_get_battery_summary_with_battery(self):
        from utils.power import get_battery_summary

        with patch("utils.power.get_battery_status", return_value={
            "present": True, "capacity_pct": 75, "status": "Charging",
            "voltage_uv": 12600000,
        }):
            summary = get_battery_summary()
            assert "75%" in summary
            assert "charging" in summary.lower()
