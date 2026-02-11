"""Unit tests for server.bridge – queue injection and broadcast logic."""

import asyncio
import json

import pytest
from server.bridge import Bridge


class FakeWebSocket:
    """Minimal stand-in for fastapi.WebSocket for testing."""

    def __init__(self):
        self.sent: list[dict] = []
        self.closed = False

    async def send_json(self, data: dict):
        if self.closed:
            raise RuntimeError("closed")
        self.sent.append(data)


@pytest.fixture
def bridge_instance():
    b = Bridge()
    loop = asyncio.new_event_loop()
    b.set_loop(loop)
    q: asyncio.Queue = asyncio.Queue()
    b.set_query_queue(q)
    yield b
    loop.close()


@pytest.mark.asyncio
async def test_inject_text(bridge_instance):
    await bridge_instance.inject_text("hello")
    assert not bridge_instance.query_queue.empty()
    val = bridge_instance.query_queue.get_nowait()
    assert val == "hello"


@pytest.mark.asyncio
async def test_broadcast(bridge_instance):
    ws1 = FakeWebSocket()
    ws2 = FakeWebSocket()
    bridge_instance.add_client(ws1)
    bridge_instance.add_client(ws2)

    await bridge_instance.broadcast({"type": "status", "status": "Listening"})
    assert len(ws1.sent) == 1
    assert ws1.sent[0]["type"] == "status"
    assert ws1.sent[0]["status"] == "Listening"
    assert "_seq" in ws1.sent[0]  # Sequence number added by bridge
    assert len(ws2.sent) == 1


@pytest.mark.asyncio
async def test_broadcast_removes_dead_client(bridge_instance):
    ws_alive = FakeWebSocket()
    ws_dead = FakeWebSocket()
    ws_dead.closed = True  # will raise on send

    async def _raise_send(data):
        raise RuntimeError("dead")

    ws_dead.send_json = _raise_send

    bridge_instance.add_client(ws_alive)
    bridge_instance.add_client(ws_dead)
    assert len(bridge_instance._clients) == 2

    await bridge_instance.broadcast({"type": "reply", "text": "hi"})
    assert len(ws_alive.sent) == 1
    assert ws_alive.sent[0]["type"] == "reply"
    assert ws_dead not in bridge_instance._clients


@pytest.mark.asyncio
async def test_handle_text_message(bridge_instance):
    msg = json.dumps({"type": "text", "text": "What time is it?"})
    await bridge_instance.handle_client_message(msg)
    val = bridge_instance.query_queue.get_nowait()
    assert val == "What time is it?"


@pytest.mark.asyncio
async def test_handle_sarcasm_toggle(bridge_instance):
    ws = FakeWebSocket()
    bridge_instance.add_client(ws)
    msg = json.dumps({"type": "sarcasm_toggle", "enabled": True})
    await bridge_instance.handle_client_message(msg)
    assert any("Sarcasm" in m.get("text", "") for m in ws.sent)


@pytest.mark.asyncio
async def test_send_helpers(bridge_instance):
    ws = FakeWebSocket()
    bridge_instance.add_client(ws)
    await bridge_instance.send_status("Thinking")
    await bridge_instance.send_reply("Hello Sir")
    await bridge_instance.send_transcript("test", final=True)
    await bridge_instance.send_detections([{"label": "cup"}], description="a cup")
    await bridge_instance.send_error("oops")
    await bridge_instance.send_wake()
    await bridge_instance.send_proactive("Take a break")
    await bridge_instance.send_hologram({"point_cloud": [], "tracked_objects": []})
    await bridge_instance.send_vitals({"fatigue": "alert", "posture": "good"})
    await bridge_instance.send_threat({"level": "clear", "score": 0.0, "summary": ""})
    await bridge_instance.send_thinking_step("reasoning", "Analyzing and reasoning...")
    assert len(ws.sent) == 11
    types = [m["type"] for m in ws.sent]
    assert types == [
        "status", "reply", "transcript_final",
        "detections", "error", "wake", "proactive",
        "hologram", "vitals", "threat", "thinking_step",
    ]
    # Verify all messages have sequence numbers
    for m in ws.sent:
        assert "_seq" in m


@pytest.mark.asyncio
async def test_thinking_step_messages(bridge_instance):
    """Thinking steps broadcast the correct step/detail fields."""
    import asyncio

    ws = FakeWebSocket()
    bridge_instance.add_client(ws)

    # Clear rate limit state so all messages go through
    bridge_instance._last_broadcast.clear()

    await bridge_instance.send_thinking_step("heard", "Processing your words...")
    # Small delay to avoid rate limiting (100ms between thinking_step messages)
    await asyncio.sleep(0.15)
    await bridge_instance.send_thinking_step("context", "Building context from memory...")
    await asyncio.sleep(0.15)
    await bridge_instance.send_thinking_step("reasoning", "Analyzing and reasoning...")
    await asyncio.sleep(0.15)
    await bridge_instance.send_thinking_step("done")

    assert len(ws.sent) == 4
    assert all(m["type"] == "thinking_step" for m in ws.sent)
    assert ws.sent[0]["step"] == "heard"
    assert ws.sent[0]["detail"] == "Processing your words..."
    assert ws.sent[1]["step"] == "context"
    assert ws.sent[2]["step"] == "reasoning"
    assert ws.sent[3]["step"] == "done"
    assert ws.sent[3]["detail"] == ""


@pytest.mark.asyncio
async def test_handle_hologram_request(bridge_instance):
    """hologram_request WS message triggers hologram broadcast."""
    from unittest.mock import MagicMock, patch

    ws = FakeWebSocket()
    bridge_instance.add_client(ws)

    mock_data = {
        "point_cloud": [{"x": 1, "y": 2, "z": 3, "r": 255, "g": 0, "b": 0}],
        "tracked": [{"track_id": 1, "xyxy": [0, 0, 100, 100], "cls": 0, "class_name": "person"}],
        "description": "test scene",
    }
    with patch("server.bridge.asyncio.get_running_loop") as mock_loop:
        mock_executor = asyncio.Future()
        mock_executor.set_result(mock_data)
        mock_loop.return_value.run_in_executor = MagicMock(return_value=mock_executor)

        msg = json.dumps({"type": "hologram_request"})
        await bridge_instance.handle_client_message(msg)

    # Should have broadcast a hologram message
    holo_msgs = [m for m in ws.sent if m.get("type") == "hologram"]
    assert len(holo_msgs) == 1
    assert holo_msgs[0]["data"]["description"] == "test scene"


@pytest.mark.asyncio
async def test_handle_vitals_request(bridge_instance):
    """vitals_request WS message triggers vitals broadcast."""
    from unittest.mock import MagicMock, patch

    ws = FakeWebSocket()
    bridge_instance.add_client(ws)

    # Simulate no analyzer available
    with patch("server.bridge.asyncio.get_running_loop") as mock_loop:
        mock_executor = asyncio.Future()
        mock_executor.set_result(None)
        mock_loop.return_value.run_in_executor = MagicMock(return_value=mock_executor)

        msg = json.dumps({"type": "vitals_request"})
        await bridge_instance.handle_client_message(msg)

    vitals_msgs = [m for m in ws.sent if m.get("type") == "vitals"]
    assert len(vitals_msgs) == 1
    assert vitals_msgs[0]["data"]["fatigue"] == "unknown"
