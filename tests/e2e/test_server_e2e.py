"""E2E tests for the FastAPI server: health, REST, and WebSocket round-trip.

Starts the server in-process using httpx + websockets test clients.
Mocks the orchestrator queue so no hardware (mic, camera, Ollama) is needed.
"""

import asyncio
import json

import pytest
from fastapi.testclient import TestClient
from server.app import app
from server.bridge import bridge


@pytest.fixture(autouse=True)
def _setup_bridge():
    """Ensure bridge has a queue and loop for every test."""
    loop = asyncio.new_event_loop()
    bridge.set_loop(loop)
    bridge.set_query_queue(asyncio.Queue())
    yield
    loop.close()


@pytest.mark.e2e
class TestHealth:
    def test_health(self):
        with TestClient(app) as client:
            r = client.get("/health")
            assert r.status_code == 200
            data = r.json()
            assert data["status"] == "ok"
            # Enhanced health returns subsystem info
            assert "cuda" in data
            assert "camera" in data
            assert "yolo_engine" in data


@pytest.mark.e2e
class TestRESTEndpoints:
    def test_api_reminders_empty(self, tmp_path, monkeypatch):
        monkeypatch.setattr("config.settings.DATA_DIR", str(tmp_path))
        with TestClient(app) as client:
            r = client.get("/api/reminders")
            assert r.status_code == 200
            assert r.json()["reminders"] == []

    def test_api_create_reminder(self, tmp_path, monkeypatch):
        monkeypatch.setattr("config.settings.DATA_DIR", str(tmp_path))
        with TestClient(app) as client:
            r = client.post("/api/reminders", json={"text": "Buy milk", "time_str": "14:00"})
            assert r.status_code == 200
            assert r.json()["ok"] is True
            # Verify it was persisted
            r2 = client.get("/api/reminders")
            assert len(r2.json()["reminders"]) == 1
            assert r2.json()["reminders"][0]["text"] == "Buy milk"

    def test_api_create_reminder_no_text(self, tmp_path, monkeypatch):
        monkeypatch.setattr("config.settings.DATA_DIR", str(tmp_path))
        with TestClient(app) as client:
            r = client.post("/api/reminders", json={"time_str": "14:00"})
            assert r.status_code == 400

    def test_api_toggle_reminder(self, tmp_path, monkeypatch):
        monkeypatch.setattr("config.settings.DATA_DIR", str(tmp_path))
        with TestClient(app) as client:
            client.post("/api/reminders", json={"text": "Walk dog"})
            # Toggle to done
            r = client.patch("/api/reminders/0")
            assert r.status_code == 200
            assert r.json()["done"] is True
            # Verify persisted
            r2 = client.get("/api/reminders")
            assert r2.json()["reminders"][0]["done"] is True
            # Toggle back to not done
            r3 = client.patch("/api/reminders/0")
            assert r3.json()["done"] is False

    def test_api_toggle_reminder_not_found(self, tmp_path, monkeypatch):
        monkeypatch.setattr("config.settings.DATA_DIR", str(tmp_path))
        with TestClient(app) as client:
            r = client.patch("/api/reminders/99")
            assert r.status_code == 404

    def test_api_delete_reminder(self, tmp_path, monkeypatch):
        monkeypatch.setattr("config.settings.DATA_DIR", str(tmp_path))
        with TestClient(app) as client:
            client.post("/api/reminders", json={"text": "Buy eggs"})
            client.post("/api/reminders", json={"text": "Clean desk"})
            # Delete first
            r = client.delete("/api/reminders/0")
            assert r.status_code == 200
            assert r.json()["removed"]["text"] == "Buy eggs"
            # Only "Clean desk" remains
            r2 = client.get("/api/reminders")
            assert len(r2.json()["reminders"]) == 1
            assert r2.json()["reminders"][0]["text"] == "Clean desk"

    def test_api_delete_reminder_not_found(self, tmp_path, monkeypatch):
        monkeypatch.setattr("config.settings.DATA_DIR", str(tmp_path))
        with TestClient(app) as client:
            r = client.delete("/api/reminders/99")
            assert r.status_code == 404


@pytest.mark.e2e
class TestWebSocket:
    def test_ws_text_roundtrip(self):
        """Send a text message via WS and verify it lands in the bridge queue."""
        with TestClient(app) as client:
            with client.websocket_connect("/ws") as ws:
                ws.send_text(json.dumps({"type": "text", "text": "Hello Jarvis"}))
                # Give bridge a moment to process
                import time
                time.sleep(0.1)
                assert not bridge.query_queue.empty()

    def test_ws_get_status(self):
        """Send get_status command; bridge should broadcast a system_status response."""
        with TestClient(app) as client:
            with client.websocket_connect("/ws") as ws:
                ws.send_text(json.dumps({"type": "get_status"}))
                data = ws.receive_json(mode="text")
                assert data["type"] == "system_status"
