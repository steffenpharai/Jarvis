# Jarvis verification

Last verified: 2026-02-08

## Hardware

- **Device**: Jetson Orin Nano Super 8GB, JetPack 6.x (L4T R36.4.7)
- **CUDA**: 12.6, Driver 540.4.0
- **RAM**: 7.4 GiB total (shared CPU/GPU)
- **CPU**: 6 cores (aarch64)
- **Python**: 3.10.12

## Ollama

- **Version**: 0.15.6 (local install, no Docker)
- **Model**: qwen3:1.7b — 100% GPU, context 2048
- **Settings**: flash attention, q8_0 KV cache, GPU overhead 1.5 GB, num_parallel=1, max_loaded=1
- **Install**: `bash scripts/install-ollama.sh`, start: `systemctl start ollama`
- **Systemd**: `sudo bash scripts/configure-ollama-systemd.sh`

## Vision

- **YOLOE engine**: `models/yoloe26n.engine` — present, loads successfully
- **Export**: `bash scripts/export_yolo_engine.sh`
- **Camera**: USB webcam on `/dev/video0`
- **MediaPipe**: Face detection active

## Test results

| Suite | Count | Status |
|-------|-------|--------|
| Unit tests | 176 | ALL PASS |
| E2E tests | 32 | ALL PASS |
| **Total** | **208** | **ALL PASS** |
| Ruff lint | — | CLEAN |

### Test coverage (modules tested)

| Module | Unit tests | E2E tests |
|--------|-----------|-----------|
| `llm/ollama_client.py` | OOM recovery, tool parsing, content cleaning, availability | Multi-LLM calls, one-shot |
| `llm/context.py` | Message building with history, vision, reminders, time | Vision-in-prompt |
| `orchestrator.py` | Vision keywords, one-turn sync (mocked), tool loop, fallback | Orchestrator starts, one-turn with tools |
| `tools.py` | All 7 tools, schemas, registry, vision delegation | Tool calls via LLM |
| `memory.py` | Load/save summary, session persistence | — |
| `voice/tts.py` | Synthesize (success/fail/timeout/empty), availability | One-shot TTS playback |
| `voice/stt.py` | Transcribe (success/empty/fail), multi-segment | — |
| `voice/wakeword.py` | — | Orchestrator wake loop |
| `audio/input.py` | Device listing, default index | Device enumeration |
| `audio/output.py` | Play WAV (success/fail/timeout/missing) | — |
| `audio/bluetooth.py` | Sink/source, reconnect, BT detection | Sink/source queries |
| `vision/camera.py` | Open (success/fail/path), read frame | Invalid index |
| `vision/detector_yolo.py` | Load engine, class names, inference | Engine exists/loads |
| `vision/detector_mediapipe.py` | Create detector, detect faces | — |
| `vision/scene.py` | Describe (empty/detections/faces) | Scene description |
| `vision/shared.py` | CUDA check, camera/YOLO/face singletons, describe scene | Vision analyze |
| `vision/visualize.py` | Draw detections (empty/valid/invalid) | — |
| `server/app.py` | — | Health, REST CRUD, WebSocket |
| `server/bridge.py` | Queue injection, broadcast, dead client removal, all helpers | WS roundtrip, get_status |
| `server/streaming.py` | MJPEG frame grab (success/fail/exception), generator | — |
| `gui/overlay.py` | Status set/get | — |
| `main.py` | Arg parsing, dry-run, idle, test-audio, GUI status | Help, dry-run, one-shot, orchestrator |
| `config/settings.py` | Via dry-run | — |
| `utils/power.py` | Power mode, stats, thermal | — |
| `utils/reminders.py` | CRUD, format for LLM | — |

## Performance benchmarks (Jetson Orin Nano Super, MAXN_SUPER)

| Metric | Result | Target | Status |
|--------|--------|--------|--------|
| Warm LLM latency (qwen3:1.7b) | **0.48s** | <5s | PASS |
| Chat + tools latency | **0.76s** | <5s | PASS |
| Orchestrator turn (text in → text out) | **2.89s** | <8s | PASS |
| YOLOE-26N inference (TensorRT) | **9.9ms** | <500ms | PASS |
| Memory during LLM inference | **5320 MB** | <7680 MB | PASS |
| End-to-end (query → spoken reply) | **~5–8s** | <10s | PASS |

## CLI modes verified

```
python main.py --dry-run           # Config validation
python main.py --test-audio        # Audio device listing
python main.py --one-shot "Hello"  # Text → LLM → TTS → play
python main.py --voice-only        # Wake → play "Hello Sir"
python main.py --e2e               # Full loop: wake → STT → LLM → TTS (with vision)
python main.py --orchestrator      # Agentic: wake → STT → LLM + tools + context → TTS
python main.py --serve             # FastAPI + WebSocket + PWA + orchestrator
python main.py --yolo-visualize    # Live YOLOE camera window
python main.py --gui               # Tkinter status overlay
```

## PWA frontend

- SvelteKit PWA built and served at `pwa/build/`
- Components: ChatPanel, VoiceControls, ListeningOrb, CameraStream, Dashboard, Reminders, SettingsPanel, StatusBar, Toast
- Stores: connection.ts (WebSocket + reconnect), voice.ts (Web Speech API), pwa.ts (install prompt)
- Features: chat deduplication, offline queue, localStorage persistence, fold-aware layout (Pixel Fold), high-contrast mode, PWA install

## Robustness features

- **OOM recovery**: Unload model → drop kernel caches → retry with smaller context (2048 → 1024 → 512)
- **Bluetooth reconnection**: `audio.bluetooth.reconnect_bluetooth()` auto-discovers paired devices
- **Camera reconnection**: `vision.shared.reconnect_camera()` re-opens camera after disconnect
- **Error verbalization**: Orchestrator speaks errors instead of silently failing
- **Proactive checks**: Every 300s, vision scans for person at desk and suggests break
- **Fire-and-forget summarisation**: Background thread, doesn't block reply pipeline
