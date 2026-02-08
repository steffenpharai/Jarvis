"""Build prompt with vision + reminders for Jarvis.

Token-budget conscious: every token in the prompt is a token the GPU must
process during prefill.  On the Orin Nano 8 GB, prefill speed matters as much
as generation speed because the model is partially CPU-offloaded.
"""


def build_messages_with_history(
    system_prompt: str,
    long_summary: str,
    short_term_turns: list[dict],
    current_user_message: str,
    vision_description: str | None = None,
    reminders_text: str | None = None,
    current_time: str | None = None,
    system_stats: str | None = None,
    max_turns: int = 4,
) -> list[dict]:
    """Build chat messages: system + last N turns + current user message.

    Context (time, stats, scene, reminders) is injected as a compact prefix on
    the current user message so the LLM doesn't need to call tools for it.
    Long summary capped at 300 chars; history at ``max_turns`` (default 4).
    """
    from config import settings

    max_turns = max_turns or settings.CONTEXT_MAX_TURNS
    system = system_prompt
    if long_summary and long_summary.strip():
        system = system.rstrip() + "\n[Summary: " + long_summary.strip()[:300] + "]"

    messages = [{"role": "system", "content": system}]

    # Compact context block – semicolon-separated, single line
    ctx_parts = []
    if current_time:
        ctx_parts.append(f"Time:{current_time}")
    if system_stats:
        ctx_parts.append(f"Sys:{system_stats}")
    if vision_description:
        # Truncate scene to ~200 chars to save tokens
        ctx_parts.append(f"Scene:{vision_description[:200]}")
    if reminders_text:
        ctx_parts.append(f"Rem:{reminders_text[:150]}")
    ctx_line = "[" + ";".join(ctx_parts) + "]" if ctx_parts else ""

    # Last N turns – only user/assistant, trimmed content
    for msg in short_term_turns[-max_turns:]:
        role = msg.get("role")
        if role not in ("user", "assistant"):
            continue
        content = (msg.get("content") or "").strip()[:300]
        if not content and role == "user":
            continue
        messages.append({"role": role, "content": content or "(no text)"})

    # Current user message (with context prefix)
    user_content = f"{ctx_line}\n{current_user_message}" if ctx_line else current_user_message
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
    """Build chat messages for Ollama: system + context + user (no history)."""
    ctx_parts = []
    if current_time:
        ctx_parts.append(f"Time:{current_time}")
    if system_stats:
        ctx_parts.append(f"Sys:{system_stats}")
    if vision_description:
        ctx_parts.append(f"Scene:{vision_description[:200]}")
    if reminders_text:
        ctx_parts.append(f"Rem:{reminders_text[:150]}")
    content = user_text
    if ctx_parts:
        content = f"[{';'.join(ctx_parts)}]\n{content}"
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": content},
    ]
