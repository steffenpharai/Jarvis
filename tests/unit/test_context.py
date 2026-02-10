"""Unit tests for LLM context building (XML-tagged format)."""

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
    # XML-tagged format
    assert "<scene>person(2), laptop(1)</scene>" in out[1]["content"]
    assert "<reminders>" in out[1]["content"]
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
    assert "<time>2026-02-07" in out[1]["content"]
    assert "<sys>" in out[1]["content"]
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
    assert "<time>2026-02-07" in out[1]["content"]


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


@pytest.mark.unit
def test_build_messages_with_vitals_and_threat():
    """Vitals and threat tags should appear in the XML-tagged context."""
    out = build_messages(
        "You are Jarvis.",
        "How am I looking?",
        vitals_text="mild fatigue,posture:fair",
        threat_text="2/10 low",
    )
    assert "<vitals>mild fatigue,posture:fair</vitals>" in out[1]["content"]
    assert "<threat>2/10 low</threat>" in out[1]["content"]


@pytest.mark.unit
def test_vision_turn_history_tagged():
    """Vision-turn assistant responses should be wrapped in <history> tags."""
    short_term = [
        {"role": "user", "content": "What do you see?"},
        {"role": "assistant", "content": "I see a cat, sir.", "_vision_turn": True},
    ]
    out = build_messages_with_history(
        "You are Jarvis.",
        "",
        short_term,
        "What about now?",
        vision_description="dog(1), chair(2)",
        max_turns=4,
    )
    # The old vision response should be wrapped in <history>
    history_msg = [m for m in out if m["role"] == "assistant"][0]
    assert "<history>" in history_msg["content"]
    assert "I see a cat, sir." in history_msg["content"]
    assert "</history>" in history_msg["content"]
    # The new context should have the current scene
    user_msg = [m for m in out if m["role"] == "user"][-1]
    assert "<scene>dog(1), chair(2)</scene>" in user_msg["content"]


@pytest.mark.unit
def test_no_context_no_tags():
    """When no context data is provided, no XML tags should appear."""
    out = build_messages("You are Jarvis.", "Hello")
    assert "<" not in out[1]["content"]
    assert out[1]["content"] == "Hello"
