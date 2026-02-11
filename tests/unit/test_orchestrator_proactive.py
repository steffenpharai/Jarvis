"""Unit tests for orchestrator proactive intelligence."""

import pytest


@pytest.mark.unit
class TestProactiveChangeDetection:
    def test_person_enters_empty_room(self):
        """Alert when a person appears in a previously empty room."""
        import orchestrator
        orchestrator._prev_person_count = 0
        orchestrator._prev_object_set = set()

        vision_data = {
            "description": "person detected",
            "tracked": [{"class_name": "person", "track_id": 1}],
        }
        alert = orchestrator._check_proactive_changes(vision_data)
        assert alert is not None
        assert "entered" in alert.lower() or "someone" in alert.lower()

    def test_person_leaves_room(self):
        """Alert when all persons leave the room."""
        import orchestrator
        orchestrator._prev_person_count = 2
        orchestrator._prev_object_set = {"person", "laptop"}

        vision_data = {
            "description": "empty room",
            "tracked": [{"class_name": "laptop", "track_id": 2}],
        }
        alert = orchestrator._check_proactive_changes(vision_data)
        assert alert is not None
        assert "clear" in alert.lower()

    def test_new_object_appears(self):
        """Alert when a new significant object appears."""
        import orchestrator
        orchestrator._prev_person_count = 1
        orchestrator._prev_object_set = {"person"}

        vision_data = {
            "description": "person with cup",
            "tracked": [
                {"class_name": "person", "track_id": 1},
                {"class_name": "cup", "track_id": 2},
            ],
        }
        alert = orchestrator._check_proactive_changes(vision_data)
        assert alert is not None
        assert "cup" in alert.lower()

    def test_no_change_no_alert(self):
        """No alert when scene hasn't changed."""
        import orchestrator
        orchestrator._prev_person_count = 1
        orchestrator._prev_object_set = {"person", "laptop"}

        vision_data = {
            "description": "person at desk",
            "tracked": [
                {"class_name": "person", "track_id": 1},
                {"class_name": "laptop", "track_id": 2},
            ],
        }
        alert = orchestrator._check_proactive_changes(vision_data)
        assert alert is None

    def test_multiple_people_enter(self):
        """Alert when additional people appear."""
        import orchestrator
        orchestrator._prev_person_count = 1
        orchestrator._prev_object_set = {"person"}

        vision_data = {
            "description": "two persons",
            "tracked": [
                {"class_name": "person", "track_id": 1},
                {"class_name": "person", "track_id": 2},
                {"class_name": "person", "track_id": 3},
            ],
        }
        alert = orchestrator._check_proactive_changes(vision_data)
        assert alert is not None
        assert "additional" in alert.lower() or "people" in alert.lower()


@pytest.mark.unit
class TestBackgroundScene:
    def test_get_bg_scene_initially_none(self):
        """Background scene starts as None before any updates."""
        # Reset state
        import orchestrator
        from orchestrator import get_bg_scene
        orchestrator._bg_scene_description = None
        result = get_bg_scene()
        assert result is None

    def test_get_bg_scene_after_update(self):
        """Background scene returns the latest description."""
        import orchestrator
        orchestrator._bg_scene_description = "A person sitting at a desk with a laptop."
        result = orchestrator.get_bg_scene()
        assert result == "A person sitting at a desk with a laptop."
