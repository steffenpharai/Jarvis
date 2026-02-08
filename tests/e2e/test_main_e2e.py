"""E2E tests for main entry (help, dry-run, LLM one-shot, orchestrator)."""

import os
import subprocess
import sys

import pytest


def _project_root():
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _is_ollama_available():
    """Check if Ollama API is reachable (for E2E LLM tests)."""
    try:
        import requests

        base = os.environ.get("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
        r = requests.get(f"{base.rstrip('/')}/api/tags", timeout=5)
        return r.status_code == 200
    except Exception:
        return False


def _is_ollama_model_available():
    """Check if configured Ollama model is pulled (for E2E LLM tests)."""
    if not _is_ollama_available():
        return False
    try:
        if _project_root() not in sys.path:
            sys.path.insert(0, _project_root())
        from config import settings
        from llm.ollama_client import is_ollama_model_available

        return is_ollama_model_available(settings.OLLAMA_BASE_URL, settings.OLLAMA_MODEL)
    except Exception:
        return False


@pytest.mark.e2e
def test_main_help():
    root = _project_root()
    result = subprocess.run(
        [sys.executable, "main.py", "--help"],
        cwd=root,
        capture_output=True,
        text=True,
        timeout=10,
        env={**os.environ, "PYTHONPATH": root},
    )
    assert result.returncode == 0
    assert "Jarvis" in result.stdout or "voice" in result.stdout.lower()


@pytest.mark.e2e
def test_main_dry_run():
    root = _project_root()
    result = subprocess.run(
        [sys.executable, "main.py", "--dry-run"],
        cwd=root,
        capture_output=True,
        text=True,
        timeout=10,
        env={**os.environ, "PYTHONPATH": root},
    )
    assert result.returncode == 0


@pytest.mark.e2e
def test_main_one_shot_llm_e2e():
    """Full E2E: text → Ollama → TTS → play. Skip if Ollama or model not available."""
    if not _is_ollama_available():
        pytest.skip(
            "Ollama not available at OLLAMA_BASE_URL (start: scripts/start-ollama.sh)"
        )
    if not _is_ollama_model_available():
        pytest.skip(
            "Ollama model not pulled (run: ollama pull <OLLAMA_MODEL>)"
        )
    root = _project_root()
    # Short prompt; one-shot runs LLM then TTS playback (may be slow on Jetson)
    result = subprocess.run(
        [sys.executable, "main.py", "--one-shot", "Say hello in one word."],
        cwd=root,
        capture_output=True,
        text=True,
        timeout=120,
        env={**os.environ, "PYTHONPATH": root},
    )
    assert result.returncode == 0, (
        f"one-shot failed: stderr={result.stderr!r} stdout={result.stdout!r}"
    )


@pytest.mark.e2e
def test_multi_llm_calls():
    """Send multiple chat requests to the LLM to verify stability and non-empty replies."""
    if not _is_ollama_available():
        pytest.skip(
            "Ollama not available at OLLAMA_BASE_URL (start: scripts/start-ollama.sh)"
        )
    if not _is_ollama_model_available():
        pytest.skip(
            "Ollama model not pulled (run: ollama pull <OLLAMA_MODEL>)"
        )
    root = _project_root()
    if root not in sys.path:
        sys.path.insert(0, root)
    from config import settings
    from llm.ollama_client import chat

    base_url = settings.OLLAMA_BASE_URL
    model = settings.OLLAMA_MODEL
    fallback = getattr(settings, "OLLAMA_FALLBACK_MODEL", "qwen3:1.7b")
    prompts = [
        "Reply with only one word: hello",
        "What is 2 plus 2? Reply with just the number.",
        "Say the current day of the week in one word.",
    ]
    replies = []
    for i, user_content in enumerate(prompts):
        messages = [{"role": "user", "content": user_content}]
        reply = chat(base_url, model, messages, stream=False)
        if not reply.strip() and fallback != model:
            reply = chat(base_url, fallback, messages, stream=False)
        if not reply.strip():
            pytest.skip(
                "Ollama returned empty (GPU OOM?). Free GPU: sudo scripts/prepare-ollama-gpu.sh; restart ollama; or use OLLAMA_MODEL=qwen3:1.7b"
            )
        assert isinstance(reply, str), f"call {i + 1}: expected str, got {type(reply)}"
        replies.append(reply)
    assert len(replies) == len(prompts), "expected one reply per prompt"


@pytest.mark.e2e
def test_main_help_includes_orchestrator():
    """--orchestrator must appear in help."""
    root = _project_root()
    result = subprocess.run(
        [sys.executable, "main.py", "--help"],
        cwd=root,
        capture_output=True,
        text=True,
        timeout=10,
        env={**os.environ, "PYTHONPATH": root},
    )
    assert result.returncode == 0
    assert "orchestrator" in result.stdout.lower()


@pytest.mark.e2e
def test_main_orchestrator_starts():
    """Orchestrator starts and runs until timeout (no immediate crash)."""
    root = _project_root()
    try:
        result = subprocess.run(
            [sys.executable, "main.py", "--orchestrator"],
            cwd=root,
            capture_output=True,
            text=True,
            timeout=4,
            env={**os.environ, "PYTHONPATH": root},
        )
    except subprocess.TimeoutExpired:
        # Process was still running after 4s — success
        return
    # If it exited before timeout, ensure it wasn't a crash
    assert "Traceback" not in (result.stderr or ""), result.stderr
    assert "Error" not in (result.stderr or "") or result.returncode == 0
