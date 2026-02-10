"""Build prompt with XML-tagged context for Jarvis.

Context engineering pattern (industry standard):
  - Live sensor data is injected into the user message using XML tags
    (<scene>, <vitals>, <threat>, <time>, <sys>, <reminders>).
  - The system prompt (config/prompts.py) teaches the model what the tags
    mean and how to ground its response in them.
  - This separation prevents the small model from parroting few-shot
    example data instead of reading live sensor feeds.

Token-budget conscious: every token in the prompt is a token the GPU must
process during prefill.  At num_ctx=8192 with Qwen3:1.7b the model stays
100% GPU (2.0 GB) and prefills ~3.7x faster than the old 2048 limit.
"""


def _build_context_block(
    vision_description: str | None = None,
    reminders_text: str | None = None,
    current_time: str | None = None,
    system_stats: str | None = None,
    vitals_text: str | None = None,
    threat_text: str | None = None,
) -> str:
    """Build XML-tagged context block from live sensor data.

    Each field is wrapped in a short XML tag.  Tags are only emitted when
    the field has data, keeping the token budget tight.  The system prompt
    teaches the model what each tag means.

    Returns empty string if no context data is available.
    """
    parts: list[str] = []
    if current_time:
        parts.append(f"<time>{current_time}</time>")
    if system_stats:
        parts.append(f"<sys>{system_stats}</sys>")
    if vision_description:
        # Cap scene to ~200 chars to save tokens
        parts.append(f"<scene>{vision_description[:200]}</scene>")
    if vitals_text:
        parts.append(f"<vitals>{vitals_text[:100]}</vitals>")
    if threat_text:
        parts.append(f"<threat>{threat_text[:80]}</threat>")
    if reminders_text:
        parts.append(f"<reminders>{reminders_text[:150]}</reminders>")
    return "\n".join(parts)


def build_messages_with_history(
    system_prompt: str,
    long_summary: str,
    short_term_turns: list[dict],
    current_user_message: str,
    vision_description: str | None = None,
    reminders_text: str | None = None,
    current_time: str | None = None,
    system_stats: str | None = None,
    vitals_text: str | None = None,
    threat_text: str | None = None,
    max_turns: int = 4,
) -> list[dict]:
    """Build chat messages: system + last N turns + current user message.

    Context (time, stats, scene, vitals, threat, reminders) is injected as
    XML-tagged data in the current user message so the LLM can ground its
    response in live sensor feeds.

    Long summary capped at 300 chars; history at ``max_turns`` (default 4).
    """
    from config import settings

    max_turns = max_turns or settings.CONTEXT_MAX_TURNS
    system = system_prompt
    if long_summary and long_summary.strip():
        system = system.rstrip() + "\n[Summary: " + long_summary.strip()[:300] + "]"

    messages = [{"role": "system", "content": system}]

    # Last N turns – only user/assistant, trimmed content.
    # Vision-turn assistant responses are tagged so the LLM knows those
    # scene observations are from a *past* snapshot and should not be
    # repeated if the user asks about the current view.
    for msg in short_term_turns[-max_turns:]:
        role = msg.get("role")
        if role not in ("user", "assistant"):
            continue
        content = (msg.get("content") or "").strip()[:300]
        if not content and role == "user":
            continue
        if role == "assistant" and msg.get("_vision_turn"):
            content = "<history>" + content + "</history>"
        messages.append({"role": role, "content": content or "(no text)"})

    # Current user message (with XML-tagged context)
    ctx_block = _build_context_block(
        vision_description=vision_description,
        reminders_text=reminders_text,
        current_time=current_time,
        system_stats=system_stats,
        vitals_text=vitals_text,
        threat_text=threat_text,
    )
    if ctx_block:
        user_content = f"{ctx_block}\n{current_user_message}"
    else:
        user_content = current_user_message
    messages.append({"role": "user", "content": user_content})
    return messages


def build_messages(
    system_prompt: str,
    user_text: str,
    vision_description: str | None = None,
    reminders_text: str | None = None,
    current_time: str | None = None,
    system_stats: str | None = None,
    vitals_text: str | None = None,
    threat_text: str | None = None,
) -> list[dict]:
    """Build chat messages for Ollama: system + context + user (no history)."""
    ctx_block = _build_context_block(
        vision_description=vision_description,
        reminders_text=reminders_text,
        current_time=current_time,
        system_stats=system_stats,
        vitals_text=vitals_text,
        threat_text=threat_text,
    )
    content = f"{ctx_block}\n{user_text}" if ctx_block else user_text
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": content},
    ]
