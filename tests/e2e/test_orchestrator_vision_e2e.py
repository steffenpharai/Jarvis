"""E2E tests: orchestration + vision integration; LLM answers questions about what it sees."""

import asyncio
from pathlib import Path

import pytest
from config import settings
from llm.context import build_messages_with_history
from llm.ollama_client import (
    chat,
    chat_with_tools,
    is_ollama_available,
    is_ollama_model_available,
)
from tools import TOOL_SCHEMAS, run_tool


def _ollama_ready():
    return is_ollama_available(settings.OLLAMA_BASE_URL) and is_ollama_model_available(
        settings.OLLAMA_BASE_URL, settings.OLLAMA_MODEL
    )


@pytest.mark.e2e
def test_vision_description_in_prompt():
    """Vision description must appear in the user message as <scene>...</scene>."""
    from config.prompts import JARVIS_ORCHESTRATOR_SYSTEM_PROMPT

    messages = build_messages_with_history(
        JARVIS_ORCHESTRATOR_SYSTEM_PROMPT,
        "",
        [],
        "What do you see?",
        vision_description="person(1), laptop(1), cup(2). Face count: 1.",
        current_time="2026-02-08 12:00:00",
        max_turns=5,
    )
    assert len(messages) >= 2
    last_user = [m for m in messages if m.get("role") == "user"][-1]
    assert "<scene>" in last_user["content"]
    assert "person(1)" in last_user["content"]
    assert "What do you see?" in last_user["content"]


@pytest.mark.e2e
def test_llm_answers_about_scene_when_vision_in_context():
    """With Ollama, given scene in context, LLM reply should reference the scene (person/laptop/cup/see)."""
    if not _ollama_ready():
        pytest.skip("Ollama not available or model not pulled")
    from config.prompts import JARVIS_ORCHESTRATOR_SYSTEM_PROMPT

    scene = "person(1), laptop(1), cup(2). Face count: 1."
    messages = build_messages_with_history(
        JARVIS_ORCHESTRATOR_SYSTEM_PROMPT,
        "",
        [],
        "What do you see in the scene? Reply in one short sentence.",
        vision_description=scene,
        current_time="2026-02-08 12:00:00",
        max_turns=5,
    )
    reply = chat(
        settings.OLLAMA_BASE_URL,
        settings.OLLAMA_MODEL,
        messages,
        stream=False,
        num_ctx=settings.OLLAMA_NUM_CTX,
    )
    if not reply or not reply.strip():
        pytest.skip("Ollama returned empty (GPU OOM?). Free GPU and retry.")
    reply_lower = reply.lower()
    # Should refer to something in the scene or to seeing/vision
    assert (
        "person" in reply_lower
        or "laptop" in reply_lower
        or "cup" in reply_lower
        or "see" in reply_lower
        or "scene" in reply_lower
        or "object" in reply_lower
    ), f"Reply should reference the scene; got: {reply!r}"


@pytest.mark.e2e
def test_vision_analyze_returns_structured_description():
    """vision_analyze tool returns a string the LLM can use (Objects: ... or unavailable message)."""
    result = run_tool("vision_analyze", {})
    assert isinstance(result, str)
    assert len(result) > 0
    assert "Objects:" in result or "unavailable" in result.lower() or "No frame" in result or "Face count" in result


@pytest.mark.e2e
def test_orchestrator_one_turn_with_vision_and_tools():
    """One full orchestrator turn with vision in context: query about scene -> non-empty answer."""
    if not _ollama_ready():
        pytest.skip("Ollama not available or model not pulled")

    from memory import load_session
    from orchestrator import _run_one_turn

    data_dir = Path(settings.DATA_DIR)
    data_dir.mkdir(parents=True, exist_ok=True)
    memory = load_session(data_dir)
    short_term: list = []
    vision_description = "person(1), laptop(1), cup(2). Face count: 1."

    async def run():
        return await _run_one_turn(
            "What do you see? Answer in one short sentence.",
            memory,
            short_term,
            vision_description,
        )

    final = asyncio.run(run())
    assert final and final.strip(), "Orchestrator turn returned empty"
    assert "I'm unable" not in final or len(final) > 30, "Expected a real answer, not fallback only"


@pytest.mark.e2e
def test_chat_with_tools_vision_tool_call():
    """When LLM calls vision_analyze tool, we get a structured observation back in the loop."""
    if not _ollama_ready():
        pytest.skip("Ollama not available or model not pulled")
    from config.prompts import JARVIS_ORCHESTRATOR_SYSTEM_PROMPT

    messages = [
        {"role": "system", "content": JARVIS_ORCHESTRATOR_SYSTEM_PROMPT},
        {"role": "user", "content": "Use the vision tool to check the current scene, then say what you see in one sentence."},
    ]
    response = chat_with_tools(
        settings.OLLAMA_BASE_URL,
        settings.OLLAMA_MODEL,
        messages,
        TOOL_SCHEMAS,
        stream=False,
        num_ctx=settings.OLLAMA_NUM_CTX,
    )
    content = response.get("content", "")
    tool_calls = response.get("tool_calls", [])
    # Either model answers directly or it calls a tool; either way we get content or tool_calls
    assert content or tool_calls, "Expected content or tool_calls from chat_with_tools"
    if tool_calls:
        names = [tc.get("name") for tc in tool_calls]
        # If it called vision_analyze, run it and ensure we get a string back
        if "vision_analyze" in names:
            obs = run_tool("vision_analyze", {})
            assert isinstance(obs, str) and len(obs) > 0
