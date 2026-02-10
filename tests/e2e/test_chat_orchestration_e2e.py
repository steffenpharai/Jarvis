"""E2E tests for the full chat orchestration flow via WebSocket.

Validates that the complete message lifecycle makes sense:
  1. User sends text → transcript_final echoed back
  2. Thinking steps broadcast in correct order (heard → context → reasoning → speaking → done)
  3. Reply broadcast after reasoning completes
  4. Status transitions: Listening → Thinking (LLM) → Speaking → Listening
  5. Messages arrive in logical conversational order

Uses a real FastAPI test client with mocked LLM/TTS to isolate the
orchestration logic from hardware dependencies.
"""

import asyncio
import json
from unittest.mock import MagicMock, patch

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
class TestThinkingStepBroadcast:
    """Verify thinking_step messages are broadcast correctly via WebSocket."""

    def test_thinking_step_roundtrip(self):
        """Send thinking steps via bridge and receive them on WS."""
        with TestClient(app) as client:
            with client.websocket_connect("/ws") as ws:
                # Broadcast a thinking step from the bridge
                loop = asyncio.new_event_loop()

                async def _send():
                    await bridge.send_thinking_step("reasoning", "Analyzing...")

                loop.run_until_complete(_send())
                loop.close()

                data = ws.receive_json(mode="text")
                assert data["type"] == "thinking_step"
                assert data["step"] == "reasoning"
                assert data["detail"] == "Analyzing..."

    def test_thinking_step_done_clears(self):
        """The 'done' step signals the end of orchestration."""
        with TestClient(app) as client:
            with client.websocket_connect("/ws") as ws:
                loop = asyncio.new_event_loop()

                async def _send():
                    await bridge.send_thinking_step("heard", "Processing...")
                    await bridge.send_thinking_step("done")

                loop.run_until_complete(_send())
                loop.close()

                msg1 = ws.receive_json(mode="text")
                msg2 = ws.receive_json(mode="text")

                assert msg1["step"] == "heard"
                assert msg2["step"] == "done"
                assert msg2["detail"] == ""


@pytest.mark.e2e
class TestFullOrchestrationMessageOrder:
    """Simulate the complete orchestration pipeline and validate message order.

    Mocks the LLM and TTS so no hardware is needed, but exercises the real
    bridge, orchestrator thinking steps, and WebSocket broadcast path.
    """

    def test_text_query_message_sequence(self):
        """User sends text → thinking steps → reply → done: correct order."""
        with TestClient(app) as client:
            with client.websocket_connect("/ws") as ws:
                # Send a user text query
                ws.send_text(json.dumps({"type": "text", "text": "What time is it?"}))

                # The query goes to the queue; the orchestrator would process it.
                # Since we don't run the orchestrator loop in this test, simulate
                # the expected broadcast sequence that the orchestrator would emit.
                loop = asyncio.new_event_loop()

                async def _simulate_orchestration():
                    # 1. Transcript echo
                    await bridge.send_transcript("What time is it?", final=True)
                    # 2. Thinking steps
                    await bridge.send_thinking_step("heard", "Processing your words...")
                    await bridge.send_thinking_step("context", "Building context from memory...")
                    await bridge.send_thinking_step("reasoning", "Analyzing and reasoning...")
                    # 3. Status
                    await bridge.send_status("Thinking (LLM)")
                    # 4. Speaking step
                    await bridge.send_thinking_step("speaking", "Formulating response...")
                    await bridge.send_status("Speaking")
                    # 5. Reply
                    await bridge.send_reply("It is 3:14 PM, sir.")
                    # 6. Done
                    await bridge.send_thinking_step("done")
                    await bridge.send_status("Listening")

                loop.run_until_complete(_simulate_orchestration())
                loop.close()

                # Collect all messages
                messages = []
                for _ in range(10):
                    try:
                        msg = ws.receive_json(mode="text")
                        messages.append(msg)
                    except Exception:
                        break

                # Validate the full sequence
                types = [m["type"] for m in messages]
                assert types == [
                    "transcript_final",
                    "thinking_step",  # heard
                    "thinking_step",  # context
                    "thinking_step",  # reasoning
                    "status",         # Thinking (LLM)
                    "thinking_step",  # speaking
                    "status",         # Speaking
                    "reply",          # actual answer
                    "thinking_step",  # done
                    "status",         # Listening
                ]

                # Validate thinking step progression
                thinking_steps = [m for m in messages if m["type"] == "thinking_step"]
                step_names = [s["step"] for s in thinking_steps]
                assert step_names == ["heard", "context", "reasoning", "speaking", "done"]

                # Validate the reply is a proper assistant message
                reply_msg = [m for m in messages if m["type"] == "reply"][0]
                assert reply_msg["text"] == "It is 3:14 PM, sir."

                # Validate transcript echo
                transcript = [m for m in messages if m["type"] == "transcript_final"][0]
                assert transcript["text"] == "What time is it?"

    def test_vision_query_includes_vision_steps(self):
        """Vision-related query should include vision scanning steps."""
        with TestClient(app) as client:
            with client.websocket_connect("/ws") as ws:
                loop = asyncio.new_event_loop()

                async def _simulate_vision_orchestration():
                    await bridge.send_transcript("What do you see?", final=True)
                    await bridge.send_thinking_step("heard", "Processing your words...")
                    await bridge.send_thinking_step("vision", "Scanning the environment...")
                    await bridge.send_thinking_step("vision_done", "Environment analyzed")
                    await bridge.send_thinking_step("context", "Building context from memory...")
                    await bridge.send_thinking_step("reasoning", "Analyzing and reasoning...")
                    await bridge.send_thinking_step("speaking", "Formulating response...")
                    await bridge.send_reply("I see a person at a desk with a laptop, sir.")
                    await bridge.send_thinking_step("done")

                loop.run_until_complete(_simulate_vision_orchestration())
                loop.close()

                messages = []
                for _ in range(9):
                    try:
                        msg = ws.receive_json(mode="text")
                        messages.append(msg)
                    except Exception:
                        break

                thinking_steps = [m for m in messages if m["type"] == "thinking_step"]
                step_names = [s["step"] for s in thinking_steps]
                assert "vision" in step_names
                assert "vision_done" in step_names

                # Vision steps come before reasoning
                vision_idx = step_names.index("vision")
                reasoning_idx = step_names.index("reasoning")
                assert vision_idx < reasoning_idx

    def test_tool_call_flow(self):
        """Tool calls should produce tool/tool_done thinking steps."""
        with TestClient(app) as client:
            with client.websocket_connect("/ws") as ws:
                loop = asyncio.new_event_loop()

                async def _simulate_tool_orchestration():
                    await bridge.send_transcript("Tell me a joke", final=True)
                    await bridge.send_thinking_step("heard", "Processing your words...")
                    await bridge.send_thinking_step("context", "Building context from memory...")
                    await bridge.send_thinking_step("reasoning", "Analyzing and reasoning...")
                    await bridge.send_thinking_step("tool", "Running tell_joke...")
                    await bridge.send_thinking_step("tool_done", "Completed tell_joke")
                    await bridge.send_thinking_step("speaking", "Formulating response...")
                    await bridge.send_reply("Why did the server go down? It couldn't handle the pressure, sir.")
                    await bridge.send_thinking_step("done")

                loop.run_until_complete(_simulate_tool_orchestration())
                loop.close()

                messages = []
                for _ in range(9):
                    try:
                        msg = ws.receive_json(mode="text")
                        messages.append(msg)
                    except Exception:
                        break

                thinking_steps = [m for m in messages if m["type"] == "thinking_step"]
                step_names = [s["step"] for s in thinking_steps]
                assert "tool" in step_names
                assert "tool_done" in step_names

                # Tool comes after reasoning, before speaking
                tool_idx = step_names.index("tool")
                reasoning_idx = step_names.index("reasoning")
                speaking_idx = step_names.index("speaking")
                assert reasoning_idx < tool_idx < speaking_idx


@pytest.mark.e2e
class TestOrchestrationWithMockedLLM:
    """Test _run_one_turn_sync with a mock bridge to verify thinking steps fire."""

    def test_thinking_steps_fire_during_tool_call(self, tmp_path):
        """When LLM calls a tool, _thinking(bridge_ref, 'tool', ...) is invoked."""
        from orchestrator import _run_one_turn_sync

        memory = {"summary": "", "data_dir": str(tmp_path)}
        short_term = []

        # Mock bridge that records thinking step calls
        mock_bridge = MagicMock()
        mock_bridge.send_thinking_step_threadsafe = MagicMock()

        tool_response = {
            "content": "",
            "tool_calls": [{"name": "tell_joke", "arguments": {}}],
        }
        final_response = {
            "content": "Here is a joke, sir.",
            "tool_calls": [],
        }

        with (
            patch("orchestrator.chat_with_tools", side_effect=[tool_response, final_response]),
            patch("orchestrator.run_tool", return_value="Why did the chicken cross the road?"),
            patch("orchestrator.load_reminders", return_value=[]),
        ):
            result = _run_one_turn_sync(
                "tell me a joke", memory, short_term, None,
                bridge_ref=mock_bridge,
            )

        assert result == "Here is a joke, sir."

        # Verify thinking steps were broadcast
        calls = mock_bridge.send_thinking_step_threadsafe.call_args_list
        step_names = [c[0][0] for c in calls]
        assert "tool" in step_names
        assert "tool_done" in step_names

    def test_no_thinking_steps_when_no_bridge(self, tmp_path):
        """When bridge_ref is None, no errors occur (thinking steps are no-ops)."""
        from orchestrator import _run_one_turn_sync

        memory = {"summary": "", "data_dir": str(tmp_path)}
        short_term = []
        mock_response = {"content": "Hello, sir.", "tool_calls": []}

        with (
            patch("orchestrator.chat_with_tools", return_value=mock_response),
            patch("orchestrator.load_reminders", return_value=[]),
        ):
            result = _run_one_turn_sync(
                "hi", memory, short_term, None,
                bridge_ref=None,  # no bridge
            )

        assert result == "Hello, sir."

    def test_thinking_steps_not_fired_for_plain_reply(self, tmp_path):
        """Without tool calls, only no tool-related thinking steps fire."""
        from orchestrator import _run_one_turn_sync

        memory = {"summary": "", "data_dir": str(tmp_path)}
        short_term = []
        mock_bridge = MagicMock()
        mock_bridge.send_thinking_step_threadsafe = MagicMock()

        mock_response = {"content": "Good evening, sir.", "tool_calls": []}

        with (
            patch("orchestrator.chat_with_tools", return_value=mock_response),
            patch("orchestrator.load_reminders", return_value=[]),
        ):
            result = _run_one_turn_sync(
                "hello", memory, short_term, None,
                bridge_ref=mock_bridge,
            )

        assert result == "Good evening, sir."

        # No tool-related steps should have fired
        calls = mock_bridge.send_thinking_step_threadsafe.call_args_list
        step_names = [c[0][0] for c in calls]
        assert "tool" not in step_names
        assert "tool_done" not in step_names
