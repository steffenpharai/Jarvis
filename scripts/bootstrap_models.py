#!/usr/bin/env python3
"""Download all models required for Jarvis (openWakeWord, Faster-Whisper, Piper voice, optional YOLOE).
Run from project root with venv active. Use --with-yolo to also build YOLOE-26N TensorRT engine (needs CUDA).
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

# Project root (parent of scripts/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
VOICES_DIR = PROJECT_ROOT / "models" / "voices"
PIPER_BASE = "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_GB/alan/medium"


def _log(msg: str) -> None:
    print(f"[bootstrap] {msg}", flush=True)


def ensure_voices_dir() -> Path:
    VOICES_DIR.mkdir(parents=True, exist_ok=True)
    return VOICES_DIR


def download_piper_voice_if_missing() -> bool:
    """Ensure en_GB-alan-medium.onnx (and .json) exist; download from Hugging Face if missing."""
    onnx_path = VOICES_DIR / "en_GB-alan-medium.onnx"
    json_path = VOICES_DIR / "en_GB-alan-medium.onnx.json"
    if onnx_path.exists() and json_path.exists():
        _log("Piper voice already present: models/voices/en_GB-alan-medium.onnx")
        return True
    ensure_voices_dir()
    try:
        import urllib.request

        for name in ("en_GB-alan-medium.onnx", "en_GB-alan-medium.onnx.json"):
            path = VOICES_DIR / name
            if path.exists():
                continue
            url = f"{PIPER_BASE}/{name}"
            _log(f"Downloading Piper voice: {name}")
            urllib.request.urlretrieve(url, path)
        _log("Piper voice ready: models/voices/en_GB-alan-medium.onnx")
        return True
    except Exception as e:
        _log(f"Piper voice download failed: {e}")
        return False


def download_openwakeword() -> bool:
    """Pre-download openWakeWord hey_jarvis model (TFLite)."""
    try:
        import openwakeword

        WAKE = "hey_jarvis"
        path = openwakeword.MODELS.get(WAKE, {}).get("model_path", "")
        if path and os.path.exists(path):
            _log("openWakeWord model already present: hey_jarvis")
            return True
        _log("Downloading openWakeWord model: hey_jarvis_v0.1")
        openwakeword.utils.download_models(model_names=["hey_jarvis_v0.1"])
        _log("openWakeWord model ready: hey_jarvis")
        return True
    except Exception as e:
        _log(f"openWakeWord download failed: {e}")
        return False


def download_faster_whisper(model_size: str = "small") -> bool:
    """Pre-download Faster-Whisper model (cached by Hugging Face)."""
    try:
        from faster_whisper import WhisperModel

        _log(f"Downloading Faster-Whisper model: {model_size}")
        WhisperModel(model_size, device="auto", compute_type="auto")
        _log(f"Faster-Whisper model ready: {model_size}")
        return True
    except Exception as e:
        _log(f"Faster-Whisper download failed: {e}")
        return False


def run_yolo_export() -> bool:
    """Run scripts/export_yolo_engine.sh (requires CUDA and venv)."""
    script = PROJECT_ROOT / "scripts" / "export_yolo_engine.sh"
    if not script.exists():
        _log("export_yolo_engine.sh not found")
        return False
    _log("Building YOLOE-26N TensorRT engine (this may take several minutes)...")
    try:
        env = os.environ.copy()
        if Path("/etc/profile.d/cuda.sh").exists():
            # Script sources this; ensure we're in project root
            pass
        out = subprocess.run(
            ["bash", str(script)],
            cwd=str(PROJECT_ROOT),
            env=env,
            capture_output=True,
            text=True,
            timeout=600,
        )
        if out.returncode == 0:
            _log("YOLOE-26N TensorRT engine ready: models/yoloe26n.engine")
            return True
        _log(f"YOLOE export failed: {out.stderr or out.stdout}")
        return False
    except subprocess.TimeoutExpired:
        _log("YOLOE export timed out (increase timeout or run manually: bash scripts/export_yolo_engine.sh)")
        return False
    except Exception as e:
        _log(f"YOLOE export failed: {e}")
        return False


def main() -> int:
    parser = argparse.ArgumentParser(description="Download Jarvis models (wake word, STT, TTS, optional YOLO)")
    parser.add_argument(
        "--with-yolo",
        action="store_true",
        help="Also build YOLOE-26N TensorRT engine (requires CUDA, PyTorch, ultralytics)",
    )
    parser.add_argument(
        "--stt-model",
        default="small",
        help="Faster-Whisper model size (default: small)",
    )
    args = parser.parse_args()

    os.chdir(PROJECT_ROOT)

    ok = True
    ok &= download_piper_voice_if_missing()
    ok &= download_openwakeword()
    ok &= download_faster_whisper(model_size=args.stt_model)
    if args.with_yolo:
        ok &= run_yolo_export()
    else:
        engine = PROJECT_ROOT / "models" / "yoloe26n.engine"
        if engine.exists():
            _log("YOLOE-26N engine already present: models/yoloe26n.engine (use --with-yolo to rebuild)")
        else:
            _log("Skipping YOLOE export (use --with-yolo to build models/yoloe26n.engine)")

    if ok:
        _log("Bootstrap complete. Ensure Ollama is running and model is pulled: ollama pull qwen3:1.7b")
    else:
        _log("Some steps failed; check output above.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
