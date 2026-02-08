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
    assert ws1.sent[0] == {"type": "status", "status": "Listening"}
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
    assert len(ws.sent) == 7
    types = [m["type"] for m in ws.sent]
    assert types == [
        "status", "reply", "transcript_final",
        "detections", "error", "wake", "proactive",
    ]
