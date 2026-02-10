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
11. [Auto-Start on Boot](#auto-start-on-boot)
12. [Verification](#verification)

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

## Auto-Start on Boot

JARVIS can start automatically when the Jetson reboots, so you can access the UI from a phone (e.g. Pixel) on the same LAN without SSH-ing in.

### Boot chain

Three systemd services run in order:

| Order | Service | Purpose |
|:---|:---|:---|
| 1 | `ollama.service` | LLM inference server (installed by `curl \| sh`) |
| 2 | `jetson-clocks.service` | Locks CPU/GPU/EMC at max clocks for MAXN_SUPER |
| 3 | `jarvis.service` | FastAPI + Orchestrator + Vision pipeline on port 8000 |

### Install the services

```bash
# 1. Ollama is already a systemd service after install — just ensure it's enabled:
sudo systemctl enable ollama

# 2. jetson_clocks at boot (locks max performance clocks):
sudo tee /etc/systemd/system/jetson-clocks.service > /dev/null << 'EOF'
[Unit]
Description=Apply jetson_clocks for MAXN_SUPER performance
After=nvpmodel.service
DefaultDependencies=no

[Service]
Type=oneshot
ExecStart=/usr/bin/jetson_clocks
RemainAfterExit=yes

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable jetson-clocks

# 3. JARVIS service:
sudo tee /etc/systemd/system/jarvis.service > /dev/null << 'SVCEOF'
[Unit]
Description=J.A.R.V.I.S. — AI Assistant (FastAPI + Orchestrator + Vision)
After=network-online.target ollama.service jetson-clocks.service
Wants=network-online.target ollama.service jetson-clocks.service

[Service]
Type=simple
User=jarvis
Group=jarvis
WorkingDirectory=/home/jarvis/Jarvis
ExecStartPre=/bin/bash /home/jarvis/Jarvis/scripts/jarvis-boot.sh
ExecStart=/home/jarvis/Jarvis/venv/bin/python main.py --serve
ExecStop=/bin/kill -SIGINT $MAINPID
TimeoutStopSec=15
Restart=on-failure
RestartSec=10
Environment="PATH=/home/jarvis/Jarvis/venv/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
Environment="PYTHONUNBUFFERED=1"
Environment="OLLAMA_BASE_URL=http://127.0.0.1:11434"
Environment="OLLAMA_MODEL=qwen3:1.7b"
Environment="OLLAMA_NUM_CTX=8192"
Environment="JARVIS_SERVE_HOST=0.0.0.0"
Environment="JARVIS_SERVE_PORT=8000"
Environment="JARVIS_DEPTH_ENABLED=0"
LimitNOFILE=65536
LimitMEMLOCK=infinity
StandardOutput=journal
StandardError=journal
SyslogIdentifier=jarvis

[Install]
WantedBy=multi-user.target
SVCEOF

sudo systemctl daemon-reload
sudo systemctl enable jarvis
```

The pre-flight script (`scripts/jarvis-boot.sh`) waits up to 60 s for Ollama to become reachable, then verifies the LLM model is pulled, the camera device exists, and the YOLOE engine is present.

### Managing the service

```bash
# Start / stop / restart
sudo systemctl start jarvis
sudo systemctl stop jarvis
sudo systemctl restart jarvis   # after code changes

# Live logs
sudo journalctl -u jarvis -f

# Status
sudo systemctl status jarvis
```

### Accessing from your phone

After reboot, open `http://<jetson-ip>:8000` in Chrome on your phone (e.g. `http://192.168.86.50:8000`). The PWA can be installed to the home screen for an app-like experience.

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
