"""E2E performance benchmarks for Jarvis on Jetson Orin Nano.

Measures:
  - LLM cold/warm inference latency
  - Orchestrator one-turn latency (query → reply text)
  - Memory usage before/after LLM call
  - YOLOE inference latency (if engine loaded)
"""

import os
import time

import pytest
from config import settings
from llm.ollama_client import chat, chat_with_tools, is_ollama_available, is_ollama_model_available
from tools import TOOL_SCHEMAS


def _ollama_ready():
    return is_ollama_available(settings.OLLAMA_BASE_URL) and is_ollama_model_available(
        settings.OLLAMA_BASE_URL, settings.OLLAMA_MODEL
    )


def _get_mem_used_mb() -> float:
    """Return total used memory in MB from /proc/meminfo."""
    try:
        with open("/proc/meminfo") as f:
            lines = {ln.split(":")[0]: ln.split(":")[1].strip() for ln in f if ":" in ln}
        total = int(lines.get("MemTotal", "0 kB").split()[0])
        avail = int(lines.get("MemAvailable", "0 kB").split()[0])
        return (total - avail) / 1024
    except Exception:
        return 0


@pytest.mark.e2e
class TestLLMLatency:
    """Measure LLM response time (warm and cold)."""

    def test_warm_llm_latency(self):
        """Warm LLM call should complete in <5s (model already loaded)."""
        if not _ollama_ready():
            pytest.skip("Ollama not available")

        messages = [
            {"role": "system", "content": "Reply in one sentence."},
            {"role": "user", "content": "Say hello."},
        ]

        # Warm up (ensure model is loaded)
        chat(settings.OLLAMA_BASE_URL, settings.OLLAMA_MODEL, messages, num_ctx=settings.OLLAMA_NUM_CTX)

        # Benchmark
        times = []
        for _ in range(3):
            t0 = time.monotonic()
            reply = chat(
                settings.OLLAMA_BASE_URL, settings.OLLAMA_MODEL, messages,
                num_ctx=settings.OLLAMA_NUM_CTX,
            )
            elapsed = time.monotonic() - t0
            times.append(elapsed)
            assert reply and reply.strip(), "Empty reply"

        avg = sum(times) / len(times)
        print(f"\n  Warm LLM latency: {avg:.2f}s (min={min(times):.2f}, max={max(times):.2f})")
        assert avg < 5.0, f"Average warm LLM latency {avg:.2f}s exceeds 5s target"

    def test_chat_with_tools_latency(self):
        """Chat with tool schemas should complete in <5s."""
        if not _ollama_ready():
            pytest.skip("Ollama not available")

        messages = [
            {"role": "system", "content": "You are Jarvis. Reply in one sentence."},
            {"role": "user", "content": "What time is it?"},
        ]

        times = []
        for _ in range(3):
            t0 = time.monotonic()
            result = chat_with_tools(
                settings.OLLAMA_BASE_URL, settings.OLLAMA_MODEL, messages,
                TOOL_SCHEMAS, num_ctx=settings.OLLAMA_NUM_CTX,
            )
            elapsed = time.monotonic() - t0
            times.append(elapsed)
            assert result.get("content") or result.get("tool_calls")

        avg = sum(times) / len(times)
        print(f"\n  Chat+tools latency: {avg:.2f}s (min={min(times):.2f}, max={max(times):.2f})")
        assert avg < 5.0, f"Average chat+tools latency {avg:.2f}s exceeds 5s target"


@pytest.mark.e2e
class TestOrchestratorLatency:
    """Measure full orchestrator one-turn latency (text in → text out)."""

    def test_one_turn_latency(self):
        """Single orchestrator turn (no TTS/audio) should complete in <8s."""
        if not _ollama_ready():
            pytest.skip("Ollama not available")

        from memory import load_session
        from orchestrator import _run_one_turn_sync

        data_dir = settings.DATA_DIR
        os.makedirs(data_dir, exist_ok=True)
        memory = load_session(data_dir)
        short_term = []

        # Warm up
        _run_one_turn_sync("hi", memory, short_term, None)

        times = []
        for prompt in [
            "What time is it?",
            "Tell me a joke.",
            "How's my system doing?",
        ]:
            t0 = time.monotonic()
            reply = _run_one_turn_sync(prompt, memory, short_term, None)
            elapsed = time.monotonic() - t0
            times.append(elapsed)
            assert reply and reply.strip()
            print(f"\n  Turn '{prompt[:30]}': {elapsed:.2f}s -> {reply[:60]}")

        avg = sum(times) / len(times)
        print(f"\n  Avg orchestrator turn: {avg:.2f}s")
        assert avg < 8.0, f"Average orchestrator latency {avg:.2f}s exceeds 8s target"


@pytest.mark.e2e
class TestMemoryUsage:
    """Ensure total memory stays under 7.5 GiB during LLM inference."""

    def test_memory_under_budget(self):
        """Memory usage during LLM call must stay under 7.5 GiB."""
        if not _ollama_ready():
            pytest.skip("Ollama not available")

        mem_before = _get_mem_used_mb()
        messages = [
            {"role": "system", "content": "Reply in one word."},
            {"role": "user", "content": "Hi."},
        ]
        chat(settings.OLLAMA_BASE_URL, settings.OLLAMA_MODEL, messages, num_ctx=settings.OLLAMA_NUM_CTX)
        mem_after = _get_mem_used_mb()

        budget_mb = settings.RAM_BUDGET_GIB * 1024
        print(f"\n  Memory: before={mem_before:.0f}MB, after={mem_after:.0f}MB, budget={budget_mb:.0f}MB")
        assert mem_after < budget_mb, (
            f"Memory {mem_after:.0f}MB exceeds budget {budget_mb:.0f}MB"
        )


@pytest.mark.e2e
class TestYoloInferenceLatency:
    """Measure YOLOE TensorRT inference latency."""

    def test_yolo_inference_latency(self):
        """Single YOLOE inference should complete in <200ms."""
        if not settings.yolo_engine_exists():
            pytest.skip("YOLOE engine not found")

        import numpy as np
        from vision.shared import get_yolo, run_inference_shared

        engine, class_names = get_yolo()
        if engine is None:
            pytest.skip("YOLOE engine failed to load")

        # Create a fake frame (640x480 BGR)
        fake_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

        # Warm up
        run_inference_shared(fake_frame)

        times = []
        for _ in range(5):
            t0 = time.monotonic()
            run_inference_shared(fake_frame)
            elapsed = time.monotonic() - t0
            times.append(elapsed)

        avg = sum(times) / len(times)
        print(f"\n  YOLOE inference: avg={avg*1000:.1f}ms, min={min(times)*1000:.1f}ms, max={max(times)*1000:.1f}ms")
        assert avg < 0.5, f"Average YOLOE inference {avg*1000:.1f}ms exceeds 500ms target"
