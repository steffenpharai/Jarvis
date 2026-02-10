# J.A.R.V.I.S. — Detailed Setup Guide

This guide walks through the complete setup on a fresh Jetson Orin Nano Super with JetPack 6.x.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [System Setup](#system-setup)
3. [Python Environment](#python-environment)
4. [CUDA and PyTorch](#cuda-and-pytorch)
5. [Ollama Setup](#ollama-setup)
6. [Model Downloads](#model-downloads)
7. [TensorRT Engines](#tensorrt-engines)
8. [Bluetooth Audio](#bluetooth-audio)
9. [PWA Frontend](#pwa-frontend)
10. [First Run](#first-run)
11. [Verification](#verification)

---

## Prerequisites

| Requirement | Details |
|:---|:---|
| **Board** | Jetson Orin Nano Super Developer Kit (8 GB) |
| **OS** | JetPack 6.x (L4T R36.x), Ubuntu 22.04 base |
| **Storage** | 128 GB+ (NVMe SSD recommended) |
| **Camera** | USB UVC webcam |
| **Audio** | Bluetooth earbuds or USB mic + speakers |

## System Setup

### Power mode

For best performance, use MAXN_SUPER and lock max clocks:

```bash
sudo nvpmodel -q              # Check current mode
sudo jetson_clocks             # Lock max CPU/GPU/EMC clocks
sudo pip3 install jetson-stats # Install jtop for monitoring
jtop                           # Monitor thermals, power, GPU
```

### System packages

```bash
sudo apt update
sudo apt install -y \
    wireplumber pipewire-audio pipewire-pulse \
    bluez-tools pulseaudio-utils \
    python3-pip python3-venv libopenblas-dev \
    git curl
```

## Python Environment

```bash
cd /path/to/Jarvis
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## CUDA and PyTorch

Follow NVIDIA's official guide for Jetson PyTorch:

```bash
# cuSPARSELt (required for PyTorch 24.06+ on JetPack 6.x)
bash scripts/install-cusparselt.sh

# CUDA in PATH
sudo bash scripts/install-cuda-path.sh

# PyTorch with CUDA
source venv/bin/activate
. /etc/profile.d/cuda.sh
bash scripts/install-pytorch-cuda-nvidia.sh

# Verify
python -c "import torch; print('CUDA:', torch.cuda.is_available())"
# Expected: CUDA: True
```

## Ollama Setup

```bash
# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Pull the model
ollama pull qwen3:1.7b

# Configure for 8GB Jetson (flash attention, q8_0 KV cache, etc.)
sudo bash scripts/configure-ollama-systemd.sh
sudo systemctl daemon-reload && sudo systemctl restart ollama

# Verify: should show 100% GPU, context 8192
ollama ps
```

### What the systemd config does

| Setting | Effect |
|:---|:---|
| `OLLAMA_FLASH_ATTENTION=1` | Dramatically less KV cache memory |
| `OLLAMA_KV_CACHE_TYPE=q8_0` | 8-bit KV cache (halves memory vs f16) |
| `OLLAMA_NUM_PARALLEL=1` | No duplicate KV caches |
| `OLLAMA_MAX_LOADED_MODELS=1` | Only one model in GPU at a time |
| `OLLAMA_CONTEXT_LENGTH=8192` | Default context length |
| `OLLAMA_GPU_OVERHEAD=1500000000` | Reserve ~1.5 GB for OS/desktop/vision |
| `OLLAMA_KEEP_ALIVE=5m` | Unload model after 5 min idle |

## Model Downloads

```bash
source venv/bin/activate
bash scripts/bootstrap_models.sh
```

This downloads:
- **openWakeWord** model
- **Faster-Whisper** (small) for STT
- **Piper** British male voice (en_GB-alan-medium)

## TensorRT Engines

Build on-device (takes several minutes each):

```bash
source venv/bin/activate && . /etc/profile.d/cuda.sh

# YOLOE-26N — required for vision
bash scripts/export_yolo_engine.sh
# Output: models/yoloe26n.engine

# DepthAnything V2 Small — required for 3D holograms
bash scripts/export_depth_engine.sh
# Output: models/depth_anything_v2_small.engine
```

## Bluetooth Audio

### Pairing earbuds (e.g. Pixel Buds)

```bash
bluetoothctl
> power on
> scan on
# Find your device, note the MAC address
> pair <MAC>
> trust <MAC>
> connect <MAC>
> quit
```

**A2DP** = high-quality audio output (TTS playback)
**HFP** = microphone input (voice commands)

If HFP is unreliable on JetPack 6.x, use a USB microphone for input and keep A2DP for output.

## PWA Frontend

```bash
cd pwa
npm install
npm run build
cd ..
```

The built files are served automatically by `--serve` mode.

## First Run

```bash
source venv/bin/activate

# Smoke test
python main.py --dry-run

# Test audio devices
python main.py --test-audio

# Quick text test (no mic needed)
python main.py --one-shot "Hello Jarvis"

# Full-stack launch
python main.py --serve
```

Open `http://<jetson-ip>:8000` from any device on your network.

## Verification

Everything working? You should see:
- PWA loads in browser with camera feed
- Voice wake word triggers recording
- JARVIS responds via TTS through earbuds
- Vision detections appear in camera overlay
- Hologram panel shows 3D point cloud (with depth enabled)
- Vitals panel shows fatigue/posture/heart rate data
- `ollama ps` shows 100% GPU, 8192 context

If anything is off, check the [Troubleshooting section](../README.md#-troubleshooting) in the README.
