"""Local tools callable by the orchestrator LLM (vision, status, reminders, joke, sarcasm)."""

import logging
import random
from datetime import datetime
from pathlib import Path

from config import settings

logger = logging.getLogger(__name__)


def vision_analyze(prompt: str | None = None) -> str:
    """Run vision on current camera frame; optional prompt to focus (e.g. person, cup).

    Delegates entirely to ``vision.shared.describe_current_scene`` which uses
    the process-wide camera, YOLOE engine, and MediaPipe face-detector singletons.
    """
    from vision.shared import describe_current_scene

    return describe_current_scene(prompt)


def get_jetson_status() -> str:
    """Return GPU/mem/temp/power and mode for the Jetson (jtop or tegrastats)."""
    try:
        from utils.power import get_system_stats, get_thermal_warning

        stats = get_system_stats()
        thermal = get_thermal_warning()
        parts = [stats] if stats else []
        if thermal:
            parts.append(thermal)
        return "; ".join(parts) if parts else "System stats unavailable."
    except Exception as e:
        logger.warning("get_jetson_status failed: %s", e)
        return "System stats unavailable."


def get_current_time() -> str:
    """Return formatted current time and date."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def create_reminder(text: str, time_str: str = "") -> str:
    """Save a reminder to the JSON file. time_str is optional (e.g. '14:00' or 'tomorrow')."""
    try:
        from utils.reminders import add_reminder

        data_dir = Path(settings.DATA_DIR)
        add_reminder(data_dir, text, time_str)
        return f"Reminder added: {text}" + (f" at {time_str}" if time_str else "") + "."
    except Exception as e:
        logger.warning("create_reminder failed: %s", e)
        return "Failed to add reminder."


def list_reminders() -> str:
    """Read and summarize pending reminders."""
    try:
        from utils.reminders import format_reminders_for_llm, load_reminders

        data_dir = Path(settings.DATA_DIR)
        reminders = load_reminders(data_dir)
        out = format_reminders_for_llm(reminders, max_items=10)
        return out if out else "No pending reminders."
    except Exception as e:
        logger.warning("list_reminders failed: %s", e)
        return "Could not list reminders."


_JOKES = [
    "I would avoid the priesthood, Sir. The only thing they're good at is wine and wafer management.",
    "I've calculated the odds of your survival. I'd rather not share them.",
    "Shall I alert the press that the great Tony Stark has misplaced his keys?",
    "Your security protocol appears to be 'hope for the best'. Charming.",
    "I'm afraid the only thing unbreakable in this scenario is my patience.",
]


def tell_joke() -> str:
    """Return a random dry/witty one-liner."""
    return random.choice(_JOKES)


def toggle_sarcasm(enabled: bool) -> str:
    """Turn sarcasm mode on or off. Returns confirmation."""
    settings.SARCASM_ENABLED = bool(enabled)
    return "Sarcasm mode on." if enabled else "Sarcasm mode off."


# Ollama-compatible tool schemas – MINIMAL set to reduce prompt tokens.
# get_current_time, get_jetson_status, and list_reminders are NOT included
# because time, stats, and reminders are already injected into the user context
# by the orchestrator.  The LLM never needs to call tools for those.
TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "vision_analyze",
            "description": "Re-scan camera with optional focus prompt.",
            "parameters": {
                "type": "object",
                "properties": {
                    "prompt": {"type": "string", "description": "Focus: person, cup, etc."},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "create_reminder",
            "description": "Save a reminder with optional time.",
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {"type": "string", "description": "Reminder text."},
                    "time_str": {"type": "string", "description": "Time, e.g. 14:00."},
                },
                "required": ["text"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "tell_joke",
            "description": "Tell a witty one-liner.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "toggle_sarcasm",
            "description": "Toggle sarcasm mode.",
            "parameters": {
                "type": "object",
                "properties": {
                    "enabled": {"type": "boolean", "description": "True=on."},
                },
                "required": ["enabled"],
            },
        },
    },
]

TOOL_REGISTRY = {
    "vision_analyze": vision_analyze,
    "get_jetson_status": get_jetson_status,
    "get_current_time": get_current_time,
    "create_reminder": create_reminder,
    "list_reminders": list_reminders,
    "tell_joke": tell_joke,
    "toggle_sarcasm": toggle_sarcasm,
}


def run_tool(name: str, arguments: dict) -> str:
    """Execute a tool by name with given arguments; return string result."""
    fn = TOOL_REGISTRY.get(name)
    if not fn:
        return f"Unknown tool: {name}"
    try:
        # Ollama may send "time" for create_reminder
        if name == "create_reminder" and "time" in arguments and "time_str" not in arguments:
            arguments = {**arguments, "time_str": arguments.get("time", "")}
        # Only pass known params to each tool
        if name == "vision_analyze":
            args = {"prompt": arguments.get("prompt")}
        elif name == "create_reminder":
            args = {"text": arguments.get("text", ""), "time_str": arguments.get("time_str", arguments.get("time", ""))}
        elif name == "toggle_sarcasm":
            args = {"enabled": arguments.get("enabled", False)}
        else:
            args = {}
        result = fn(**args)
        return str(result)
    except Exception as e:
        logger.warning("run_tool %s failed: %s", name, e)
        return f"Tool error: {e}"
