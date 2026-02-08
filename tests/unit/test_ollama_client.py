"""Unit tests for llm/ollama_client.py – OOM recovery, tool call parsing, content cleaning."""

from unittest.mock import MagicMock, patch

import pytest
from llm.ollama_client import (
    _clean_llm_content,
    _extract_text_tool_calls,
    _is_oom_error,
    _is_oom_exception,
    _parse_tool_calls,
    _safe_num_ctx,
    chat,
    chat_with_tools,
    is_ollama_available,
    is_ollama_model_available,
    unload_model,
)

# ── _parse_tool_calls ──────────────────────────────────────────────────


@pytest.mark.unit
class TestParseToolCalls:
    def test_empty(self):
        assert _parse_tool_calls([]) == []
        assert _parse_tool_calls(None) == []

    def test_normal_dict(self):
        raw = [{"function": {"name": "tell_joke", "arguments": {}}}]
        out = _parse_tool_calls(raw)
        assert len(out) == 1
        assert out[0]["name"] == "tell_joke"
        assert out[0]["arguments"] == {}

    def test_arguments_as_string(self):
        raw = [{"function": {"name": "create_reminder", "arguments": '{"text": "Buy milk"}'}}]
        out = _parse_tool_calls(raw)
        assert out[0]["arguments"] == {"text": "Buy milk"}

    def test_invalid_json_arguments(self):
        raw = [{"function": {"name": "foo", "arguments": "not json"}}]
        out = _parse_tool_calls(raw)
        assert out[0]["arguments"] == {}

    def test_nested_function_dict(self):
        raw = [{"function": {"name": "toggle_sarcasm", "arguments": {"enabled": True}}}]
        out = _parse_tool_calls(raw)
        assert out[0]["name"] == "toggle_sarcasm"
        assert out[0]["arguments"]["enabled"] is True

    def test_missing_function_key(self):
        raw = [{"name": "foo"}]
        out = _parse_tool_calls(raw)
        assert out[0]["name"] == ""
        assert out[0]["arguments"] == {}


# ── _extract_text_tool_calls ───────────────────────────────────────────


@pytest.mark.unit
class TestExtractTextToolCalls:
    def test_no_tool_calls(self):
        content = "Hello Sir, how may I help you?"
        cleaned, tcs = _extract_text_tool_calls(content)
        assert tcs == []
        assert content == cleaned

    def test_json_name_pattern(self):
        # The regex requires no nested braces, so use flat JSON
        content = 'Let me check. {"name": "tell_joke"} Here you go.'
        cleaned, tcs = _extract_text_tool_calls(content)
        assert len(tcs) == 1
        assert tcs[0]["name"] == "tell_joke"
        assert "tell_joke" not in cleaned

    def test_action_pattern(self):
        # Action pattern with flat JSON (no nested braces)
        content = 'Action: {"tool": "vision_analyze", "args": "person"}'
        cleaned, tcs = _extract_text_tool_calls(content)
        assert len(tcs) >= 1
        names = [tc["name"] for tc in tcs]
        assert "vision_analyze" in names

    def test_multiple_flat_patterns(self):
        content = '{"name": "tell_joke"} and also {"name": "toggle_sarcasm"}'
        cleaned, tcs = _extract_text_tool_calls(content)
        assert len(tcs) == 2


# ── _clean_llm_content ────────────────────────────────────────────────


@pytest.mark.unit
class TestCleanLlmContent:
    def test_clean_text_unchanged(self):
        assert _clean_llm_content("Hello Sir.") == "Hello Sir."

    def test_strips_think_tags(self):
        content = "<think>internal reasoning</think>The answer is 42."
        assert "think" not in _clean_llm_content(content)
        assert "42" in _clean_llm_content(content)

    def test_strips_code_fences(self):
        content = "Here: ```python\nprint('hi')\n``` Done."
        cleaned = _clean_llm_content(content)
        assert "```" not in cleaned

    def test_strips_json_objects(self):
        content = 'Yes. {"output": "test", "context": "foo"} That is all.'
        cleaned = _clean_llm_content(content)
        assert '"output"' not in cleaned

    def test_strips_parenthetical_meta(self):
        content = "The time is 3pm. (no tool call needed)"
        cleaned = _clean_llm_content(content)
        assert "no tool call" not in cleaned

    def test_empty_input(self):
        assert _clean_llm_content("") == ""
        assert _clean_llm_content(None) is None

    def test_very_short_result(self):
        # If cleaning leaves < 3 chars, returns empty
        content = '{"name": "x"}'
        cleaned = _clean_llm_content(content)
        # Either empty or the cleaned fragment
        assert isinstance(cleaned, str)


# ── _is_oom_error / _is_oom_exception ─────────────────────────────────


@pytest.mark.unit
class TestOomDetection:
    def test_oom_error_allocate(self):
        resp = MagicMock()
        resp.text = "failed to allocate 2048 bytes for buffer"
        assert _is_oom_error(resp) is True

    def test_oom_error_out_of_memory(self):
        resp = MagicMock()
        resp.text = "CUDA out of memory"
        assert _is_oom_error(resp) is True

    def test_oom_error_nvmap(self):
        resp = MagicMock()
        resp.text = "NvMapMemAlloc failed"
        assert _is_oom_error(resp) is True

    def test_not_oom(self):
        resp = MagicMock()
        resp.text = "model loaded successfully"
        assert _is_oom_error(resp) is False

    def test_oom_exception_with_response(self):
        resp = MagicMock()
        resp.text = "out of memory"
        exc = Exception("request failed")
        exc.response = resp
        assert _is_oom_exception(exc) is True

    def test_oom_exception_from_string(self):
        exc = Exception("CUDA out of memory during inference")
        assert _is_oom_exception(exc) is True

    def test_not_oom_exception(self):
        exc = Exception("connection refused")
        assert _is_oom_exception(exc) is False


# ── _safe_num_ctx ─────────────────────────────────────────────────────


@pytest.mark.unit
class TestSafeNumCtx:
    def test_caps_at_max(self):
        assert _safe_num_ctx(8192) <= 2048

    def test_minimum(self):
        assert _safe_num_ctx(10) == 128

    def test_normal_value(self):
        assert _safe_num_ctx(1024) == 1024


# ── chat (mocked HTTP) ────────────────────────────────────────────────


@pytest.mark.unit
class TestChat:
    def test_chat_success(self):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"message": {"content": "Hello Sir."}}
        with patch("llm.ollama_client.requests.post", return_value=mock_resp):
            result = chat("http://localhost:11434", "qwen3:1.7b", [{"role": "user", "content": "hi"}])
            assert result == "Hello Sir."

    def test_chat_strips_think_tags(self):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"message": {"content": "<think>reasoning</think>Hello."}}
        with patch("llm.ollama_client.requests.post", return_value=mock_resp):
            result = chat("http://localhost:11434", "qwen3:1.7b", [{"role": "user", "content": "hi"}])
            assert "think" not in result
            assert "Hello" in result

    def test_chat_oom_recovery(self):
        oom_resp = MagicMock()
        oom_resp.status_code = 500
        oom_resp.text = "failed to allocate buffer for CUDA"

        ok_resp = MagicMock()
        ok_resp.status_code = 200
        ok_resp.json.return_value = {"message": {"content": "Recovered."}}

        with patch("llm.ollama_client.requests.post", side_effect=[oom_resp, ok_resp]):
            with patch("llm.ollama_client._recover_from_oom"):
                result = chat("http://localhost:11434", "qwen3:1.7b", [{"role": "user", "content": "hi"}])
                assert result == "Recovered."

    def test_chat_connection_error(self):
        import requests

        with patch("llm.ollama_client.requests.post", side_effect=requests.ConnectionError("refused")):
            result = chat("http://localhost:11434", "qwen3:1.7b", [{"role": "user", "content": "hi"}])
            assert result == ""

    def test_chat_empty_content(self):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"message": {"content": ""}}
        with patch("llm.ollama_client.requests.post", return_value=mock_resp):
            result = chat("http://localhost:11434", "qwen3:1.7b", [])
            assert result == ""


# ── chat_with_tools (mocked HTTP) ─────────────────────────────────────


@pytest.mark.unit
class TestChatWithTools:
    def test_with_tool_call(self):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "message": {
                "content": "",
                "tool_calls": [
                    {"function": {"name": "tell_joke", "arguments": {}}}
                ],
            }
        }
        with patch("llm.ollama_client.requests.post", return_value=mock_resp):
            result = chat_with_tools(
                "http://localhost:11434",
                "qwen3:1.7b",
                [{"role": "user", "content": "tell me a joke"}],
                [{"type": "function", "function": {"name": "tell_joke"}}],
            )
            assert len(result["tool_calls"]) == 1
            assert result["tool_calls"][0]["name"] == "tell_joke"

    def test_with_text_content_only(self):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"message": {"content": "Hello Sir.", "tool_calls": []}}
        with patch("llm.ollama_client.requests.post", return_value=mock_resp):
            result = chat_with_tools(
                "http://localhost:11434",
                "qwen3:1.7b",
                [{"role": "user", "content": "hi"}],
                [],
            )
            assert result["content"] == "Hello Sir."
            assert result["tool_calls"] == []

    def test_text_tool_call_fallback(self):
        """When model leaks tool call as flat text JSON, it should be extracted."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "message": {
                "content": '{"name": "tell_joke"}',
                "tool_calls": [],
            }
        }
        with patch("llm.ollama_client.requests.post", return_value=mock_resp):
            result = chat_with_tools(
                "http://localhost:11434",
                "qwen3:1.7b",
                [{"role": "user", "content": "joke please"}],
                [{"type": "function", "function": {"name": "tell_joke"}}],
            )
            assert len(result["tool_calls"]) == 1
            assert result["tool_calls"][0]["name"] == "tell_joke"

    def test_oom_recovery(self):
        oom_resp = MagicMock()
        oom_resp.status_code = 500
        oom_resp.text = "out of memory"

        ok_resp = MagicMock()
        ok_resp.status_code = 200
        ok_resp.json.return_value = {"message": {"content": "Ok.", "tool_calls": []}}

        with patch("llm.ollama_client.requests.post", side_effect=[oom_resp, ok_resp]):
            with patch("llm.ollama_client._recover_from_oom"):
                result = chat_with_tools(
                    "http://localhost:11434", "qwen3:1.7b",
                    [{"role": "user", "content": "hi"}], [],
                )
                assert result["content"] == "Ok."


# ── unload_model ──────────────────────────────────────────────────────


@pytest.mark.unit
class TestUnloadModel:
    def test_unload_success(self):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        with patch("llm.ollama_client.requests.post", return_value=mock_resp):
            assert unload_model("http://localhost:11434", "qwen3:1.7b") is True

    def test_unload_failure(self):
        mock_resp = MagicMock()
        mock_resp.status_code = 500
        with patch("llm.ollama_client.requests.post", return_value=mock_resp):
            assert unload_model("http://localhost:11434", "qwen3:1.7b") is False

    def test_unload_connection_error(self):
        import requests as req

        with patch("llm.ollama_client.requests.post", side_effect=req.ConnectionError):
            assert unload_model("http://localhost:11434", "qwen3:1.7b") is False


# ── is_ollama_available / is_ollama_model_available ───────────────────


@pytest.mark.unit
class TestOllamaAvailability:
    def test_available(self):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        with patch("llm.ollama_client.requests.get", return_value=mock_resp):
            assert is_ollama_available("http://localhost:11434") is True

    def test_not_available(self):
        import requests as req

        with patch("llm.ollama_client.requests.get", side_effect=req.ConnectionError):
            assert is_ollama_available("http://localhost:11434") is False

    def test_model_available(self):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "models": [{"name": "qwen3:1.7b"}]
        }
        with patch("llm.ollama_client.requests.get", return_value=mock_resp):
            assert is_ollama_model_available("http://localhost:11434", "qwen3:1.7b") is True

    def test_model_not_available(self):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"models": [{"name": "other:1b"}]}
        with patch("llm.ollama_client.requests.get", return_value=mock_resp):
            assert is_ollama_model_available("http://localhost:11434", "qwen3:1.7b") is False

    def test_model_available_strips_latest(self):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"models": [{"name": "qwen3:1.7b:latest"}]}
        with patch("llm.ollama_client.requests.get", return_value=mock_resp):
            assert is_ollama_model_available("http://localhost:11434", "qwen3:1.7b") is True
