"""Unit tests for LLM context building."""

import pytest
from llm.context import build_messages, build_messages_with_history


@pytest.mark.unit
def test_build_messages_basic():
    out = build_messages("You are Jarvis.", "What time is it?")
    assert len(out) == 2
    assert out[0]["role"] == "system"
    assert out[0]["content"] == "You are Jarvis."
    assert out[1]["role"] == "user"
    assert "What time is it?" in out[1]["content"]


@pytest.mark.unit
def test_build_messages_with_vision_and_reminders():
    out = build_messages(
        "You are Jarvis.",
        "What do you see?",
        vision_description="person(2), laptop(1)",
        reminders_text="Call mom; Review PR",
    )
    assert out[1]["role"] == "user"
    assert "person(2), laptop(1)" in out[1]["content"]
    assert "Rem:" in out[1]["content"]
    assert "Call mom" in out[1]["content"]


@pytest.mark.unit
def test_build_messages_with_time_and_stats():
    out = build_messages(
        "You are Jarvis.",
        "What time is it?",
        current_time="2026-02-07 12:00:00",
        system_stats="Power mode: MAXN_SUPER",
    )
    assert out[1]["role"] == "user"
    assert "2026-02-07" in out[1]["content"]
    assert "Sys:" in out[1]["content"]
    assert "MAXN_SUPER" in out[1]["content"]


@pytest.mark.unit
def test_build_messages_with_history():
    out = build_messages_with_history(
        "You are Jarvis.",
        "",
        [],
        "What time is it?",
        current_time="2026-02-07 12:00:00",
        max_turns=3,
    )
    assert out[0]["role"] == "system"
    assert out[1]["role"] == "user"
    assert "What time is it?" in out[1]["content"]
    assert "2026-02-07" in out[1]["content"]


@pytest.mark.unit
def test_build_messages_with_history_and_summary():
    out = build_messages_with_history(
        "You are Jarvis.",
        "User asked about the weather.",
        [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello, Sir."}],
        "And the time?",
        max_turns=3,
    )
    assert "Summary:" in out[0]["content"]
    assert "User asked about the weather" in out[0]["content"]
    assert out[1]["role"] == "user"
    assert "Hi" in out[1]["content"]
    assert out[2]["role"] == "assistant"
    assert out[3]["role"] == "user"
    assert "And the time?" in out[3]["content"]
