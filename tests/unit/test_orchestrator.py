"""Unit tests for orchestrator.py – mocked LLM, tool loop, vision keywords."""

from unittest.mock import patch

import pytest
from orchestrator import (
    _VISION_KEYWORDS,
    MAX_TOOL_ROUNDS,
    STT_LLM_RETRIES,
    _run_one_turn_sync,
)


@pytest.mark.unit
class TestVisionKeywords:
    def test_matches_see(self):
        assert _VISION_KEYWORDS.search("what do you see")

    def test_matches_camera(self):
        assert _VISION_KEYWORDS.search("check the camera")

    def test_matches_room(self):
        assert _VISION_KEYWORDS.search("scan the room")

    def test_matches_face(self):
        assert _VISION_KEYWORDS.search("is there a face")

    def test_no_match_time(self):
        assert _VISION_KEYWORDS.search("what time is it") is None

    def test_no_match_joke(self):
        assert _VISION_KEYWORDS.search("tell me a joke") is None


@pytest.mark.unit
class TestRunOneTurnSync:
    """Test the synchronous one-turn ReAct loop with mocked Ollama."""

    def test_simple_reply_no_tools(self, tmp_path):
        """LLM returns plain text, no tool calls."""
        memory = {"summary": "", "data_dir": str(tmp_path)}
        short_term = []
        mock_response = {"content": "Hello Sir.", "tool_calls": []}

        with (
            patch("orchestrator.chat_with_tools", return_value=mock_response),
            patch("orchestrator.load_reminders", return_value=[]),
        ):
            result = _run_one_turn_sync("hi", memory, short_term, None)
            assert result == "Hello Sir."

    def test_tool_call_then_answer(self, tmp_path):
        """LLM calls a tool, then returns final answer."""
        memory = {"summary": "", "data_dir": str(tmp_path)}
        short_term = []

        tool_response = {
            "content": "",
            "tool_calls": [{"name": "tell_joke", "arguments": {}}],
        }
        final_response = {
            "content": "Here's a joke for you, Sir.",
            "tool_calls": [],
        }

        with (
            patch("orchestrator.chat_with_tools", side_effect=[tool_response, final_response]),
            patch("orchestrator.run_tool", return_value="Why did the chicken cross the road?"),
            patch("orchestrator.load_reminders", return_value=[]),
        ):
            result = _run_one_turn_sync("tell joke", memory, short_term, None)
            assert "joke" in result.lower() or "Sir" in result

    def test_max_tool_rounds_enforced(self, tmp_path):
        """After MAX_TOOL_ROUNDS, returns fallback."""
        memory = {"summary": "", "data_dir": str(tmp_path)}
        short_term = []

        # Always return tool calls (never a final answer)
        tool_response = {
            "content": "",
            "tool_calls": [{"name": "tell_joke", "arguments": {}}],
        }

        with (
            patch("orchestrator.chat_with_tools", return_value=tool_response),
            patch("orchestrator.run_tool", return_value="joke result"),
            patch("orchestrator.load_reminders", return_value=[]),
        ):
            result = _run_one_turn_sync("keep calling tools", memory, short_term, None)
            assert "unable" in result.lower()

    def test_with_vision_description(self, tmp_path):
        """Vision description is passed through to context."""
        memory = {"summary": "", "data_dir": str(tmp_path)}
        short_term = []
        mock_response = {"content": "I see a person, Sir.", "tool_calls": []}

        with (
            patch("orchestrator.chat_with_tools", return_value=mock_response) as mock_chat,
            patch("orchestrator.load_reminders", return_value=[]),
        ):
            result = _run_one_turn_sync(
                "what do you see?", memory, short_term, "person(1), laptop(1)"
            )
            assert result == "I see a person, Sir."
            # Verify vision was passed into messages via build_messages_with_history
            call_args = mock_chat.call_args
            messages = call_args[0][2]  # third positional arg
            user_msg = [m for m in messages if m.get("role") == "user"][-1]
            assert "person(1)" in user_msg["content"]

    def test_with_reminders_in_context(self, tmp_path):
        """Reminders should be injected into context."""
        memory = {"summary": "", "data_dir": str(tmp_path)}
        short_term = []
        mock_response = {"content": "You have reminders, Sir.", "tool_calls": []}

        with (
            patch("orchestrator.chat_with_tools", return_value=mock_response),
            patch("orchestrator.load_reminders", return_value=[
                {"text": "Call mom", "done": False},
            ]),
        ):
            result = _run_one_turn_sync("what are my reminders?", memory, short_term, None)
            assert "reminders" in result.lower() or "Sir" in result

    def test_empty_reply_fallback(self, tmp_path):
        """Empty LLM reply should produce a fallback message."""
        memory = {"summary": "", "data_dir": str(tmp_path)}
        short_term = []
        mock_response = {"content": "", "tool_calls": []}

        with (
            patch("orchestrator.chat_with_tools", return_value=mock_response),
            patch("orchestrator.load_reminders", return_value=[]),
        ):
            result = _run_one_turn_sync("hi", memory, short_term, None)
            assert "unable" in result.lower()


@pytest.mark.unit
class TestConstants:
    def test_max_tool_rounds(self):
        assert MAX_TOOL_ROUNDS == 2

    def test_stt_llm_retries(self):
        assert STT_LLM_RETRIES == 1
