"""Build prompt with vision + reminders for Jarvis."""


def build_messages_with_history(
    system_prompt: str,
    long_summary: str,
    short_term_turns: list[dict],
    current_user_message: str,
    vision_description: str | None = None,
    reminders_text: str | None = None,
    current_time: str | None = None,
    system_stats: str | None = None,
    max_turns: int = 8,
) -> list[dict]:
    """Build chat messages: system (with optional long summary) + last N turns + current user message.
    short_term_turns is a list of message dicts (role, content) or (role, content, tool_calls).
    Prepends context blocks (time, stats, scene, reminders) to the first user content if provided.
    """
    from config import settings

    max_turns = max_turns or settings.CONTEXT_MAX_TURNS
    system = system_prompt
    if long_summary and long_summary.strip():
        system = system.rstrip() + "\n\n[Previous context summary: " + long_summary.strip()[:800] + "]"

    messages = [{"role": "system", "content": system}]

    # Optional context for this turn – compact single block to save tokens
    ctx_parts = []
    if current_time:
        ctx_parts.append(f"Time: {current_time}")
    if system_stats:
        ctx_parts.append(f"Status: {system_stats}")
    if vision_description:
        ctx_parts.append(f"Scene: {vision_description}")
    if reminders_text:
        ctx_parts.append(f"Reminders: {reminders_text}")
    parts = [f"[Context: {'; '.join(ctx_parts)}]"] if ctx_parts else []

    # Last N turns (assistant may have tool_calls; we only keep role + content for history)
    for msg in short_term_turns[-max_turns:]:
        role = msg.get("role")
        if role not in ("user", "assistant"):
            continue
        content = (msg.get("content") or "").strip()
        if not content and role == "user":
            continue
        messages.append({"role": role, "content": content or "(no text)"})

    # Current user message (with optional context prefix)
    user_content = current_user_message
    if parts:
        user_content = "\n\n".join(parts) + "\n\n" + user_content
    messages.append({"role": "user", "content": user_content})
    return messages


def build_messages(
    system_prompt: str,
    user_text: str,
    vision_description: str | None = None,
    reminders_text: str | None = None,
    current_time: str | None = None,
    system_stats: str | None = None,
) -> list[dict]:
    """Build chat messages for Ollama: system + optional vision/reminders/time/stats + user."""
    ctx_parts = []
    if current_time:
        ctx_parts.append(f"Time: {current_time}")
    if system_stats:
        ctx_parts.append(f"Status: {system_stats}")
    if vision_description:
        ctx_parts.append(f"Scene: {vision_description}")
    if reminders_text:
        ctx_parts.append(f"Reminders: {reminders_text}")
    content = user_text
    if ctx_parts:
        content = f"[Context: {'; '.join(ctx_parts)}]\n\n{content}"
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": content},
    ]
