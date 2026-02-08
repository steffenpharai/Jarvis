# Jarvis – Offline Voice Assistant on Jetson Orin Nano Super

A fully offline, Jarvis-like personal AI assistant for the **Jetson Orin Nano Super Developer Kit (8GB)** with JetPack 6.x. Uses Bluetooth (e.g. Google Pixel Buds 2) for mic and TTS output, USB webcam for vision, and runs LLM (Ollama), STT (Faster-Whisper), TTS (Piper), and wake word (openWakeWord) locally.

## Requirements

- **Hardware**: Jetson Orin Nano Super (8GB), 128GB+ storage (microSD or SSD), USB webcam, Bluetooth earbuds (e.g. Pixel Buds 2)
- **OS**: JetPack 6.x (L4T R36.x), Ubuntu 22.04 base
- **RAM**: Keep total usage under ~7.5 GB to avoid swap on microSD

## Power mode (MAXN Super)

For best performance, use MAXN Super and optionally lock max clocks:

```bash
# Check current mode (should show MAXN_SUPER)
nvpmodel -q

# Optional: lock max CPU/GPU clocks
sudo jetson_clocks

# Monitor thermals and power
tegrastats
# or install: sudo pip3 install jetson-stats && jtop
```

## System packages (audio & Bluetooth)

```bash
sudo apt update
sudo apt install -y wireplumber pipewire-audio pipewire-pulse bluez-tools pulseaudio-utils
```

Set default sink/source for Pixel Buds via `pactl` or `wpctl` (after Wireplumber), or use your desktop sound settings.

## Bluetooth pairing (Pixel Buds 2)

1. Put the buds in pairing mode.
2. On the Jetson:

```bash
bluetoothctl
power on
scan on
# Find "Pixel Buds" (or your device), note the MAC address
pair <MAC>
trust <MAC>
connect <MAC>
```

3. **A2DP** is used for high-quality TTS output. For **mic input**, many buds need **HFP/HSP**. If the buds do not appear as an input device, switch the profile to HFP in `bluetoothctl` or via Blueman. If HFP is unreliable on JetPack 6.x, use a **USB microphone** as fallback and keep A2DP for output only.

## Python environment

```bash
cd /home/jarvis/Jarvis
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## CUDA and PyTorch (per NVIDIA)

For GPU-accelerated PyTorch (e.g. YOLO TensorRT export), follow [NVIDIA’s official guide](https://docs.nvidia.com/deeplearning/frameworks/install-pytorch-jetson-platform/index.html):

1. **System packages** (once):
   ```bash
   sudo apt-get -y update
   sudo apt-get install -y python3-pip libopenblas-dev
   ```

2. **cuSPARSELt** (required for PyTorch 24.06+ on JetPack 6.x):
   ```bash
   bash scripts/install-cusparselt.sh   # run with sudo if needed for /usr/local/cuda
   ```

3. **CUDA in PATH** – ensure `/etc/profile.d/cuda.sh` exists (adds `/usr/local/cuda/bin` and `lib64` to PATH/LD_LIBRARY_PATH). Create it if missing:
   ```bash
   sudo bash scripts/install-cuda-path.sh
   ```

4. **PyTorch with CUDA** in the project venv:
   ```bash
   source venv/bin/activate
   . /etc/profile.d/cuda.sh
   bash scripts/install-pytorch-cuda-nvidia.sh
   ```

5. **Verify**:
   ```bash
   . /etc/profile.d/cuda.sh && python -c "import torch; print('CUDA:', torch.cuda.is_available())"
   ```
   You should see `CUDA: True` and device `Orin`. Always source `/etc/profile.d/cuda.sh` (or log in again) so `LD_LIBRARY_PATH` includes CUDA and cuSPARSELt before running Python.

## Ollama (local install – per Jetson AI Lab)

Install and run Ollama **locally** exactly as in [Jetson AI Lab – Ollama on Jetson](https://www.jetson-ai-lab.com/tutorials/ollama/) (native install, no Docker).

**Supported devices (from tutorial):** Jetson AGX Thor, AGX Orin (64GB/32GB), Orin NX (16GB), **Orin Nano (8GB)**. JetPack 5 (r35.x) or JetPack 6 (r36.x). NVMe SSD recommended (space for models >5GB).

### 1. Install Ollama (one command)

```bash
curl -fsSL https://ollama.com/install.sh | sh
```

Or use the project script (same command):

```bash
bash scripts/install-ollama.sh
```

The installer creates a service to run `ollama serve` on startup, so you can use the `ollama` command right away.

### 2. Start Ollama (if not using the system service)

If you want to run the server manually with GPU (e.g. to ensure `OLLAMA_NUM_GPU=1`):

```bash
bash scripts/start-ollama.sh
```

API at `http://127.0.0.1:11434`.

### 3. Run a model

```bash
ollama run llama3.2:1b
```

Or pull then run: `ollama pull llama3.2:1b`. Full model list: [ollama.com/library](https://ollama.com/library).

**Memory:** If your Jetson doesn’t have enough memory for larger models, use smaller models (e.g. `llama3.2:1b`). Jarvis defaults to `OLLAMA_MODEL=llama3.2:1b` in `config/settings.py`. Override with the `OLLAMA_MODEL` environment variable if needed.

## First-time setup: download all models

After `pip install -r requirements.txt`, run the bootstrap script to download openWakeWord (hey_jarvis), Faster-Whisper (small), and the Piper voice (if not already in `models/voices/`). Optionally build the YOLOE-26N TensorRT engine.

```bash
source venv/bin/activate
bash scripts/bootstrap_models.sh
# Optional: build YOLOE-26N TensorRT engine (requires CUDA; source /etc/profile.d/cuda.sh first)
bash scripts/bootstrap_models.sh --with-yolo
```

Ensure Ollama has the default model: `ollama pull llama3.2:1b`. Then run validation (see **Production readiness** below).

## Piper TTS (British male voice)

Piper is installed via **piper-tts** in `requirements.txt` (use the project venv so the `piper` CLI is on your PATH). The **British male voice** (en_GB-alan-medium) is included in the repo under `models/voices/`:

- `models/voices/en_GB-alan-medium.onnx` – voice model  
- `models/voices/en_GB-alan-medium.onnx.json` – config  

Default config uses this path. To use another voice, set `JARVIS_TTS_VOICE` to the full path of a `.onnx` file, or add more voices from [rhasspy/piper-voices](https://huggingface.co/rhasspy/piper-voices) into `models/voices/`.

## Vision (YOLOE-26N TensorRT, optional)

To use the vision pipeline with TensorRT-accelerated **YOLOE-26N** (2026 Ultralytics YOLOE, prompt-free nano), ensure CUDA and PyTorch are set up per the “CUDA and PyTorch (per NVIDIA)” section above, then:

```bash
source venv/bin/activate
. /etc/profile.d/cuda.sh
bash scripts/export_yolo_engine.sh
```

The script exports `yoloe-26n-seg-pf.pt` to TensorRT on device 0 (CUDA). Output: `models/yoloe26n.engine`. Engine build can take several minutes on device. See `docs/jetson-ai-lab-models.md` for details.

**Vision requires the engine**: `--e2e` without `--no-vision` will exit with an error if `models/yoloe26n.engine` is missing. Build it with `bash scripts/export_yolo_engine.sh` before running with vision. **USB camera**: default is index `0` (first device, e.g. `/dev/video0`). Set `JARVIS_CAMERA_DEVICE=/dev/video0` to force a device path, or `JARVIS_CAMERA_INDEX=1` for a second camera.

## Running Jarvis

```bash
source venv/bin/activate

# Validate config
python main.py --dry-run

# List audio devices and default sink/source
python main.py --test-audio

# Phase 1 test: wake word → play "Hello Sir" (no STT/LLM)
python main.py --voice-only

# Full loop: wake → STT → Ollama → TTS (no vision)
python main.py --e2e --no-vision

# Full loop with vision (camera + YOLOE/MediaPipe → LLM context)
python main.py --e2e

# Agentic orchestrator: wake → STT → LLM with tools (vision, time, reminders, jokes) → TTS
python main.py --orchestrator
python main.py --orchestrator --no-vision   # without camera

# With status overlay (Listening / Thinking / Speaking)
python main.py --e2e --gui
```

Stop with `Ctrl+C`.

### Orchestrator (agentic mode)

With `--orchestrator`, Jarvis runs an async loop with **short- and long-term context**, **tool calling** (vision, Jetson status, time, reminders, jokes, sarcasm toggle), and **proactive** idle checks (e.g. vision every ~5 minutes). Session summary and reminders are stored under `data/`. Use `--no-vision` to disable the camera.

### Ollama memory optimization (8GB Jetson)

On Jetson Orin Nano, GPU and CPU share the same 7.6 GiB RAM. With Cursor/Chrome/GNOME running, only ~1.6 GiB may be truly **free** (the rest is buff/cache). CUDA's `cudaMalloc` (via the nvmap kernel driver) needs actual free memory -- it cannot reclaim buff/cache automatically. This causes OOM even when `MemAvailable` looks ample.

The project includes a multi-layer OOM prevention stack:

| Layer | Setting | Effect |
|-------|---------|--------|
| **systemd** | `OLLAMA_FLASH_ATTENTION=1` | Flash attention (dramatically less KV cache memory) |
| **systemd** | `OLLAMA_KV_CACHE_TYPE=q8_0` | 8-bit KV cache (halves memory vs f16) |
| **systemd** | `OLLAMA_NUM_PARALLEL=1` | No duplicate KV caches |
| **systemd** | `OLLAMA_MAX_LOADED_MODELS=1` | Only one model in GPU at a time |
| **systemd** | `OLLAMA_CONTEXT_LENGTH=512` | Default context length for small KV cache |
| **systemd** | `OLLAMA_GPU_OVERHEAD=500000000` | Reserve ~500 MB for X11/GNOME/Cursor |
| **systemd** | `OLLAMA_KEEP_ALIVE=5m` | Unload model after 5 min idle |
| **Python** | `OLLAMA_NUM_CTX=512` / `OLLAMA_NUM_CTX_MAX=512` | Cap context per request |
| **Python** | OOM recovery | On CUDA OOM: unload model + drop kernel caches + retry with smaller context |

**Apply all settings at once:**

```bash
# Configure systemd (writes /etc/systemd/system/ollama.service.d/gpu.conf)
sudo bash scripts/configure-ollama-systemd.sh

# Drop kernel caches to free memory for CUDA
sudo sh -c 'echo 3 > /proc/sys/vm/drop_caches'

# Reload and restart
sudo systemctl daemon-reload && sudo systemctl restart ollama

# Verify model loads (should show 100% GPU, flash_attn enabled)
ollama run llama3.2:1b 'hello' --verbose
```

**Inspect current config and memory:**

```bash
sudo bash scripts/inspect-ollama-config.sh
bash scripts/inspect-jetson-memory.sh
```

## Options

| Option         | Description |
|----------------|-------------|
| `--dry-run`    | Validate config and exit. |
| `--test-audio` | List input devices and default sink/source. |
| `--voice-only` | Wake word only; on trigger, play TTS "Hello Sir". |
| `--e2e`         | Full loop: wake → record → STT → LLM → TTS. |
| `--orchestrator`| Agentic loop: wake → STT → LLM with tools + context → TTS. |
| `--no-vision`   | Disable camera/vision context (use with `--e2e` or `--orchestrator`). |
| `--gui`         | Show status overlay (Listening / Thinking / Speaking). |
| `--verbose`     | Debug logging. |

## Project layout

- `main.py` – Entry point and main loop.
- `orchestrator.py` – Async agentic loop (context, tools, proactive).
- `tools.py` – Local tools (vision, status, time, reminders, joke, sarcasm).
- `memory.py` – Session summary and persistence.
- `config/` – Settings and Jarvis system prompt.
- `audio/` – Mic selection, recording, playback, Bluetooth hints.
- `voice/` – Wake word, STT (Faster-Whisper), TTS (Piper).
- `llm/` – Ollama client and context (vision, reminders, time, stats).
- `vision/` – Camera, YOLOE-26N TensorRT, MediaPipe, scene description.
- `utils/` – Power mode, logging, reminders.
- `gui/` – Optional status overlay with vision preview.

## Troubleshooting

- **Ollama OOM / cudaMalloc failed**: On Jetson 8GB, `cudaMalloc` needs actual **free** RAM (not just `MemAvailable`). Run `sudo sh -c 'echo 3 > /proc/sys/vm/drop_caches'` before model load to reclaim buff/cache. Apply all OOM-prevention settings with `sudo bash scripts/configure-ollama-systemd.sh`. The Python client also auto-recovers: on OOM it unloads the model, drops caches, and retries with smaller context. Inspect memory: `bash scripts/inspect-jetson-memory.sh`.
- **Bluetooth mic not working**: Prefer HFP profile for the buds or use a USB microphone and keep A2DP for output.
- **Piper not found**: Install Piper and ensure the `piper` binary and voice model are on PATH or configured in `TTS_VOICE`.
- **Ollama connection refused**: Start Ollama with `ollama serve` (or equivalent) and check `OLLAMA_BASE_URL`.
- **No camera**: Use `--no-vision` or plug a USB UVC camera. Use `JARVIS_CAMERA_INDEX` (default 0) or `JARVIS_CAMERA_DEVICE=/dev/video0` to select the device.

## Production readiness

Before treating the app as production-ready, ensure:

1. **Models**: Run `bash scripts/bootstrap_models.sh` (and `--with-yolo` if using vision for YOLOE-26N engine). Pull Ollama model: `ollama pull llama3.2:1b`.
2. **Ollama**: Running and reachable at `http://127.0.0.1:11434` (e.g. `bash scripts/start-ollama.sh` or systemd).
3. **Validation**:
   ```bash
   source venv/bin/activate
   ruff check .
   pytest tests/
   pytest tests/ -m e2e
   python main.py --dry-run
   python main.py --one-shot "Say hello."
   python main.py --e2e --no-vision   # full loop (wake → STT → LLM → TTS)
   ```
4. **Ollama systemd** (recommended): Apply memory-optimized settings for 8GB Jetson:
   ```bash
   sudo bash scripts/configure-ollama-systemd.sh
   sudo sh -c 'echo 3 > /proc/sys/vm/drop_caches'
   sudo systemctl daemon-reload && sudo systemctl restart ollama
   ```
5. **Optional**: For vision, build YOLOE engine and run `python main.py --e2e` (with camera).
