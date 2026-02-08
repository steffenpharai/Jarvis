"""Paths, model names, RAM/thermal limits for Jetson Orin Nano 8GB Super (MAXN_SUPER).

Performance tuning for <10 s LLM response:
  - num_ctx 2048 (halves KV cache vs 4096, more GPU layers)
  - num_predict 256 (voice replies are short)
  - think=false (disables Qwen3 hidden reasoning tokens – biggest latency win)
  - Reduced tool set (time/stats/reminders already in context)
"""

import os

# Paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VENV_ROOT = os.path.join(PROJECT_ROOT, "venv")

# RAM budget (GiB) – keep under 7.5 to avoid swap on microSD.
# With Cursor IDE + GNOME the Orin Nano 8 GB typically has ~1–1.5 GiB free.
RAM_BUDGET_GIB = 7.5

# Ollama – local install, GPU.  Default qwen3:1.7b (native tool-calling,
# 8GB-friendly on Jetson Orin Nano Super).  Override with OLLAMA_MODEL env var.
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "qwen3:1.7b")
OLLAMA_FALLBACK_MODEL = os.environ.get("OLLAMA_FALLBACK_MODEL", "qwen3:1.7b")
# Context size (KV cache).  2048 keeps KV cache small so more model layers fit
# on GPU (avoids the 66% CPU / 34% GPU split that kills throughput).
OLLAMA_NUM_CTX = int(os.environ.get("OLLAMA_NUM_CTX", "2048"))
# Hard cap – prevents OOM; 2048 is the safe ceiling with YOLOE + desktop running.
OLLAMA_NUM_CTX_MAX = 2048
# Max output tokens – voice replies should be 1-3 sentences (~60-100 tokens).
# 256 gives ample room without wasting generation time.
OLLAMA_NUM_PREDICT = int(os.environ.get("OLLAMA_NUM_PREDICT", "256"))
# Disable Qwen3 <think> reasoning blocks.  Thinking adds 10–20 s of hidden token
# generation that provides zero value for a concise voice assistant.
OLLAMA_THINK = os.environ.get("OLLAMA_THINK", "0") == "1"
# Temperature – lower = faster convergence, more deterministic for voice.
OLLAMA_TEMPERATURE = float(os.environ.get("OLLAMA_TEMPERATURE", "0.6"))
# Server-side memory settings (must also match systemd env):
#   OLLAMA_FLASH_ATTENTION=1       – flash attention (less KV cache memory)
#   OLLAMA_KV_CACHE_TYPE=q8_0      – quantize KV cache to 8-bit (halves vs f16)
#   OLLAMA_NUM_PARALLEL=1          – single concurrent request (no duplicate KV caches)
#   OLLAMA_MAX_LOADED_MODELS=1     – only one model in GPU at a time
#   OLLAMA_GPU_OVERHEAD=1500000000 – reserve ~1.5 GB for X11/GNOME/Cursor/YOLOE
#   OLLAMA_KEEP_ALIVE=5m           – unload model after 5 min idle
# These are set in systemd, not in-app, but documented here for reference.
OLLAMA_FLASH_ATTENTION = os.environ.get("OLLAMA_FLASH_ATTENTION", "1") == "1"
OLLAMA_KV_CACHE_TYPE = os.environ.get("OLLAMA_KV_CACHE_TYPE", "q8_0")

# Voice
WAKE_WORD_MODEL = "jarvis"  # openWakeWord / Porcupine
STT_MODEL_SIZE = "small"  # faster-whisper: tiny, base, small, medium
# Piper British male: path to .onnx (default from models/voices/)
_DEFAULT_TTS_ONNX = os.path.join(PROJECT_ROOT, "models", "voices", "en_GB-alan-medium.onnx")
TTS_VOICE = os.environ.get("JARVIS_TTS_VOICE", _DEFAULT_TTS_ONNX)
RECORD_DURATION_SEC = 5.0  # seconds to record after wake
SARCASM_ENABLED = False  # toggle for dry/sarcastic replies

# Vision – USB camera (e.g. /dev/video0 on Jetson). Use index 0 by default.
# Set JARVIS_CAMERA_DEVICE=/dev/video0 to force a device path.
CAMERA_DEVICE = os.environ.get("JARVIS_CAMERA_DEVICE")  # None = use index
CAMERA_INDEX = int(os.environ.get("JARVIS_CAMERA_INDEX", "0"))
CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720
CAMERA_FPS = 30
# YOLOE-26N (2026): prompt-free nano for Jetson 8GB; engine built from yoloe-26n-seg-pf.pt
YOLOE_ENGINE_PATH = os.path.join(PROJECT_ROOT, "models", "yoloe26n.engine")
# Legacy alias for backward compatibility
YOLO_ENGINE_PATH = YOLOE_ENGINE_PATH


def yolo_engine_exists() -> bool:
    """True if the TensorRT YOLOE engine file exists (required for vision pipeline)."""
    return os.path.isfile(YOLOE_ENGINE_PATH)


# Orchestrator – context, memory, proactive
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
SUMMARY_PATH = os.path.join(DATA_DIR, "session_summary.json")
CONTEXT_MAX_TURNS = int(os.environ.get("JARVIS_CONTEXT_MAX_TURNS", "4"))
SUMMARY_EVERY_N_TURNS = int(os.environ.get("JARVIS_SUMMARY_EVERY_N", "6"))
PROACTIVE_IDLE_SEC = int(os.environ.get("JARVIS_PROACTIVE_IDLE_SEC", "300"))
MAX_TOOL_CALLS_PER_TURN = int(os.environ.get("JARVIS_MAX_TOOL_CALLS", "3"))

# Optional GUI vision preview (Phase 5)
JARVIS_PREVIEW_PATH = os.path.join(os.environ.get("TMPDIR", "/tmp"), "jarvis_preview.jpg")

# ── Server (FastAPI / WebSocket bridge) ──────────────────────────────────
JARVIS_SERVE_HOST = os.environ.get("JARVIS_SERVE_HOST", "0.0.0.0")
JARVIS_SERVE_PORT = int(os.environ.get("JARVIS_SERVE_PORT", "8000"))
JARVIS_WS_PATH = os.environ.get("JARVIS_WS_PATH", "/ws")
# Optional HTTPS for wss:// (self-signed cert on LAN or Tailscale)
JARVIS_HTTPS_CERT = os.environ.get("JARVIS_HTTPS_CERT")  # path to .pem
JARVIS_HTTPS_KEY = os.environ.get("JARVIS_HTTPS_KEY")    # path to .key
