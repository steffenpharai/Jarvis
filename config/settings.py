"""Paths, model names, RAM/thermal limits for Jetson Orin Nano 8GB Super (MAXN_SUPER).

Performance tuning for <10 s LLM response (Qwen3:1.7b):
  - num_ctx 8192  (100% GPU at 2.0 GB; 3.5x faster than 2048 – eliminates
    KV-cache thrashing that added ~9 s per request with the old Llama3.2 limit)
  - num_predict 512 (Qwen3 think tokens count toward predict budget;
    256 caused empty responses when reasoning chain consumed all tokens)
  - think=false for plain chat; think=true for tool calls (Qwen3 requires
    its reasoning chain to route tool schemas — official Qwen3 behaviour)
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
# Context size (KV cache).  Qwen3:1.7b (1.4 GB weights, Q4_K_M) fits 100% GPU
# up to ~12 288 ctx on 8 GB Jetson.  8192 is the production sweet spot:
#   - 2.0 GB total model footprint (weights + KV), 100% GPU
#   - 3.5 s chat latency vs 12.9 s at 2048 (KV-cache thrashing eliminated)
#   - Ample room for system prompt + 4 tool schemas + 4-turn history
# The old 2048 limit was sized for Llama3.2:3b (~2 GB weights); Qwen3:1.7b
# is 600 MB lighter, freeing that headroom for a larger KV cache.
OLLAMA_NUM_CTX = int(os.environ.get("OLLAMA_NUM_CTX", "8192"))
# Hard cap – 8192 keeps us 100% GPU; 12288 works but increases swap pressure;
# 16384 spills to 30% CPU / 70% GPU (unacceptable latency).
OLLAMA_NUM_CTX_MAX = int(os.environ.get("OLLAMA_NUM_CTX_MAX", "8192"))
# Max output tokens.  Qwen3's thinking tokens (<think>…</think>) count toward
# num_predict.  A typical think chain is ~80-120 tokens; a 1-2 sentence reply
# is ~30-60 tokens.  512 gives comfortable headroom for both.  The old 256
# caused empty responses when think=true consumed the entire budget.
OLLAMA_NUM_PREDICT = int(os.environ.get("OLLAMA_NUM_PREDICT", "512"))
# Default think=false for plain chat (fast).  chat_with_tools() forces
# think=true when tools are present — Qwen3 requires its reasoning chain
# to route tool schemas (confirmed: 1.7b cannot select tools without it).
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


# ── Depth estimation (DepthAnything V2 Small) ────────────────────────
DEPTH_ENGINE_PATH = os.path.join(PROJECT_ROOT, "models", "depth_anything_v2_small.engine")
DEPTH_ENABLED = os.environ.get("JARVIS_DEPTH_ENABLED", "0") == "1"

# ── Advanced perception (optical flow + ego-motion + trajectory) ──────
# Master switch — enables flow, ego-motion, trajectory prediction pipeline.
# CPU-only (OpenCV) — zero extra GPU memory.
PERCEPTION_ENABLED = os.environ.get("JARVIS_PERCEPTION_ENABLED", "1") == "1"
# Optical flow method: "farneback" (accurate, ~15ms) or "dis" (faster, ~8ms)
FLOW_METHOD = os.environ.get("JARVIS_FLOW_METHOD", "dis")
# Flow computation resolution (W, H).  Smaller = faster.
FLOW_WIDTH = int(os.environ.get("JARVIS_FLOW_WIDTH", "320"))
FLOW_HEIGHT = int(os.environ.get("JARVIS_FLOW_HEIGHT", "240"))
# Trajectory prediction horizon (seconds ahead)
TRAJECTORY_HORIZON_SEC = float(os.environ.get("JARVIS_TRAJ_HORIZON", "3.0"))
# Collision danger zone radius (meters) for proactive alerts
COLLISION_ZONE_M = float(os.environ.get("JARVIS_COLLISION_ZONE_M", "2.0"))
# Skip perception every N frames in portable mode (1 = every frame)
PORTABLE_PERCEPTION_SKIP = int(os.environ.get("JARVIS_PORTABLE_PERCEPTION_SKIP", "2"))
# Motion energy threshold to trigger active scanning (0-1)
MOTION_WAKE_THRESHOLD = float(os.environ.get("JARVIS_MOTION_WAKE_THRESHOLD", "0.05"))

# ── Ambient awareness (hands-free walk-around) ───────────────────────
# Master switch for always-on motion detection (auto-enabled in portable mode)
AMBIENT_AWARENESS_ENABLED = os.environ.get("JARVIS_AMBIENT_ENABLED", "0") == "1"
# How often to run full perception in walk mode (seconds)
PROACTIVE_WALK_INTERVAL_SEC = int(os.environ.get("JARVIS_PROACTIVE_WALK_SEC", "15"))
# Thermal threshold for ambient duty cycle reduction
THERMAL_AMBIENT_PAUSE_C = float(os.environ.get("JARVIS_THERMAL_AMBIENT_C", "70"))
# Battery threshold for conservation mode
BATTERY_LOW_PCT = int(os.environ.get("JARVIS_BATTERY_LOW_PCT", "15"))
# Cooldown between non-critical proactive messages (seconds)
PROACTIVE_COOLDOWN_SEC = float(os.environ.get("JARVIS_PROACTIVE_COOLDOWN_SEC", "10"))

# ── Portable / walk-around mode ──────────────────────────────────────
PORTABLE_MODE = os.environ.get("JARVIS_PORTABLE", "0") == "1"
PORTABLE_WIDTH = int(os.environ.get("JARVIS_PORTABLE_WIDTH", "320"))
PORTABLE_HEIGHT = int(os.environ.get("JARVIS_PORTABLE_HEIGHT", "320"))
PORTABLE_FPS = int(os.environ.get("JARVIS_PORTABLE_FPS", "10"))
PORTABLE_DEPTH_SKIP = int(os.environ.get("JARVIS_PORTABLE_DEPTH_SKIP", "3"))   # run depth every Nth frame
PORTABLE_VITALS_SKIP = int(os.environ.get("JARVIS_PORTABLE_VITALS_SKIP", "5"))  # run vitals every Nth frame
# Thermal threshold (Celsius) — pause non-essential vision when exceeded
THERMAL_PAUSE_THRESHOLD = float(os.environ.get("JARVIS_THERMAL_PAUSE_C", "80"))

# Orchestrator – context, memory, proactive
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
SUMMARY_PATH = os.path.join(DATA_DIR, "session_summary.json")
CONTEXT_MAX_TURNS = int(os.environ.get("JARVIS_CONTEXT_MAX_TURNS", "4"))
SUMMARY_EVERY_N_TURNS = int(os.environ.get("JARVIS_SUMMARY_EVERY_N", "6"))
PROACTIVE_IDLE_SEC = int(os.environ.get("JARVIS_PROACTIVE_IDLE_SEC", "300"))
# Continuous vision broadcast interval (seconds) for PWA hologram/vitals/threat
VISION_BROADCAST_INTERVAL = int(os.environ.get("JARVIS_VISION_BROADCAST_SEC", "2"))
# How often (in multiples of VISION_BROADCAST_INTERVAL) to run depth/point cloud
VISION_BROADCAST_DEPTH_EVERY = int(os.environ.get("JARVIS_VISION_DEPTH_EVERY", "3"))
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
