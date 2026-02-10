# J.A.R.V.I.S. — Social Media Launch Kit

Ready-to-copy posts for maximum visibility. Adjust as needed, but these are optimized for the Jetson/local-LLM/edge-AI communities in 2026.

---

## Priority 1: Record a Demo Video/GIF First

Before posting anywhere, record a **15–20 second screen capture** showing:

1. Wake word trigger ("Hey Jarvis")
2. Voice command → YOLOE detection overlay appears on camera feed
3. Hologram panel renders 3D point cloud
4. JARVIS replies with sarcastic British wit via Bluetooth earbuds
5. Vitals panel showing real-time data

Upload to YouTube (unlisted or public) and create a GIF from it (e.g. via gifcap or ffmpeg). This single asset will 10x your engagement on every platform.

```bash
# GIF from video (ffmpeg)
ffmpeg -i demo.mp4 -vf "fps=15,scale=720:-1" -t 15 docs/assets/demo.gif
```

---

## X / Twitter

### Main launch post (tag people!)

```
I built a fully offline Iron Man J.A.R.V.I.S. on a $249 Jetson Orin Nano (8GB):

- Voice: wake word → STT → LLM → TTS via Bluetooth earbuds
- Vision: TensorRT YOLOE open-vocab detection + 3D holograms
- Vitals: rPPG heart rate, fatigue, posture monitoring
- Brain: Qwen3 1.7b with tool calling, <2s latency
- UI: SvelteKit PWA with Three.js holograms + Iron Man HUD

No cloud. No API keys. 247 tests. MIT licensed.

github.com/steffenpharai/Jarvis

@dustaborin @JetsonHacks @ultraaborlytics #JetsonAI #LocalLLM #EdgeAI
```

### Thread follow-up posts

```
1/ The hardest part: fitting LLM + vision + depth + vitals in 8GB shared RAM.

Qwen3:1.7b (Q4_K_M) @ 8192 tokens: 2.0GB GPU
YOLOE-26N TensorRT: 0.3GB
DepthAnything V2: 0.4GB
Faster-Whisper: 0.5GB
OS + desktop: 3.5GB

Total: 6.8GB. Headroom: 800MB. Every byte counts.
```

```
2/ Intent-based routing makes a 1.7B model feel fast:

"Hey Jarvis" → 0.5s (no tools, no thinking)
"Tell me a joke" → 3.6s (tools + thinking)
"What do you see?" → 0.7s (vision pre-fetched into context)

The same trick Alexa/Rasa use — don't waste small model capacity on routing.
```

```
3/ The vision pipeline runs FIVE models simultaneously:

YOLOE-26N → object detection
ByteTrack → multi-object tracking
DepthAnything V2 → 3D depth
MediaPipe Face → fatigue + heart rate
MediaPipe Pose → posture scoring

All feeding into a real-time Iron Man HUD. On a $249 board.
```

### Short viral format

```
Built J.A.R.V.I.S. IRL.

$249 Jetson. 8GB RAM. Zero cloud.

Voice + vision + 3D holograms + health monitoring.

"At your service, sir."

github.com/steffenpharai/Jarvis
```

---

## Reddit

### r/LocalLLaMA

**Title:** I built a fully offline JARVIS (Iron Man) on Jetson Orin Nano — voice + vision + 3D holograms in 8GB RAM

**Body:**

```markdown
After months of work, I'm sharing J.A.R.V.I.S. — a fully offline, Iron Man-style AI assistant running entirely on a Jetson Orin Nano Super (8GB). No cloud, no API keys, no subscriptions.

**What it does:**
- Wake word → Faster-Whisper STT → Qwen3:1.7b (via Ollama, 100% GPU) → Piper TTS → Bluetooth earbuds
- TensorRT YOLOE-26N open-vocabulary object detection + ByteTrack tracking
- DepthAnything V2 for real-time 3D holograms (Three.js in a SvelteKit PWA)
- MediaPipe vitals: fatigue detection, posture scoring, rPPG heart rate estimation
- Threat detection with proactive LLM alerts
- Full JARVIS MCU persona with sarcasm toggle

**Performance on real hardware:**
- 0.5–0.7s for simple chat (intent routing skips tool overhead)
- 3.6–8.4s for tool calls (jokes, reminders, vision scan)
- 6.8GB total RAM usage (LLM + 5 vision models + OS)
- 247 passing tests

**The trick to making 1.7B feel smart:** Intent-based routing. Simple queries never see tool schemas — they respond in <1s. Tool queries get `think=true` and only the relevant tool. Same pattern Alexa and Rasa use.

MIT licensed: https://github.com/steffenpharai/Jarvis

I'd love feedback, especially from anyone else building on Jetson. What features would you want to see?
```

### r/JetsonNano (also works for r/nvidia, r/selfhosted)

**Title:** Open-source offline AI assistant with voice, vision, and 3D holograms — running on Jetson Orin Nano Super

**Body:**

```markdown
Sharing my project J.A.R.V.I.S. — a fully offline Iron Man-style assistant for the Jetson Orin Nano Super (8GB).

**Stack:**
- Qwen3:1.7b (Ollama) for the brain
- openWakeWord + Faster-Whisper + Piper TTS for voice
- YOLOE-26N (TensorRT) + ByteTrack for object detection/tracking
- DepthAnything V2 for depth + 3D holograms
- MediaPipe for health monitoring (fatigue, posture, heart rate)
- SvelteKit PWA with Three.js, accessible from any device on LAN
- Bluetooth earbuds for wireless I/O

Everything runs locally. No Docker needed (native venv). Setup takes ~30 min on a fresh JetPack 6.x install.

GitHub: https://github.com/steffenpharai/Jarvis

Happy to answer questions about the RAM budgeting, TensorRT optimization, or anything else!
```

---

## NVIDIA Developer Forums

### Jetson Projects category

**Title:** J.A.R.V.I.S.: Open-source offline voice + vision AI assistant on Jetson Orin Nano Super (8GB)

**Body:**

```markdown
Hi everyone,

I'm sharing an open-source project I've been building: **J.A.R.V.I.S.** — a fully offline AI assistant inspired by Iron Man's JARVIS, running entirely on the Jetson Orin Nano Super Developer Kit (8GB).

**Hardware:** Jetson Orin Nano Super 8GB, USB webcam, Bluetooth earbuds
**Software:** JetPack 6.x, Python 3.10, Ollama, SvelteKit

**Key capabilities:**
- **Voice pipeline:** openWakeWord → Faster-Whisper STT → Qwen3:1.7b (Ollama, 100% GPU) → Piper TTS → Bluetooth A2DP/HFP
- **Vision:** TensorRT-accelerated YOLOE-26N with open-vocabulary detection, ByteTrack multi-object tracking, DepthAnything V2 Small for depth estimation and 3D point cloud generation
- **Health monitoring:** MediaPipe Face Mesh for EAR-based fatigue detection and rPPG heart rate estimation, MediaPipe Pose for posture scoring
- **Frontend:** SvelteKit PWA with Three.js 3D holograms, live MJPEG camera feed, Iron Man-style HUD overlay, vitals dashboard
- **Orchestrator:** Async agentic loop with intent-based tool routing, short/long-term memory, proactive vision alerts

**Performance highlights:**
- 0.5–0.7s chat latency (no tools)
- 6.8GB total RAM usage across all models
- Multi-layer CUDA OOM recovery
- 247 unit + E2E tests

Everything runs offline with zero cloud dependencies. MIT licensed.

**GitHub:** https://github.com/steffenpharai/Jarvis

I'd love feedback from the Jetson community, especially on:
1. RAM optimization strategies I might be missing
2. Interest in a pre-built Docker image for JetPack 6.x
3. Ideas for ROS 2 integration

Thanks for building such a great community around Jetson!
```

---

## dusty-nv/jetson-containers Discussion

**Title:** Sharing: Offline JARVIS voice+vision assistant on Orin Nano (Ollama + YOLOE + DepthAnything)

```markdown
Hi @dusty-nv and community,

Wanted to share a project that might interest folks here: **J.A.R.V.I.S.** — a fully offline Iron Man-style AI assistant running on Jetson Orin Nano Super 8GB.

It uses several models simultaneously within the 8GB budget:
- Qwen3:1.7b via Ollama (2.0GB @ 8192 ctx, 100% GPU)
- YOLOE-26N TensorRT (0.3GB)
- DepthAnything V2 Small TensorRT (0.4GB)
- Faster-Whisper small (0.5GB on demand)
- MediaPipe face + pose (CPU, ~0.1GB)

Currently runs natively (no Docker) but I'm exploring containerization. Would love feedback on whether a jetson-containers recipe would be useful for the community.

GitHub: https://github.com/steffenpharai/Jarvis
```

---

## Hacker News (Show HN)

**Title:** Show HN: Offline Iron Man JARVIS on a $249 Jetson – voice, vision, 3D holograms in 8GB

```
I built a fully offline AI assistant inspired by Iron Man's JARVIS, running on a $249 Jetson Orin Nano Super with 8GB shared RAM.

It combines voice (wake word + STT + LLM + TTS through Bluetooth earbuds), computer vision (TensorRT object detection + depth estimation + health monitoring), and a web UI with Three.js 3D holograms — all running simultaneously, all local.

The interesting technical challenge was fitting everything in 8GB shared RAM: the LLM (Qwen3 1.7b), five vision models, and the OS. Intent-based routing keeps simple queries under 1 second while tool calls (scanning, reminders) take 4-8 seconds.

MIT licensed, 247 tests, actively developed.

https://github.com/steffenpharai/Jarvis
```

---

## Timing & Strategy

### Post order (space them 1-2 days apart):

1. **Day 1:** r/LocalLLaMA + X/Twitter main post (biggest audiences for local AI)
2. **Day 2:** r/JetsonNano + NVIDIA Developer Forums (Jetson-specific community)
3. **Day 3:** Hacker News Show HN (broader tech audience)
4. **Day 4:** dusty-nv/jetson-containers discussion (targeted Jetson power users)
5. **Day 5:** X/Twitter thread follow-ups + engage with comments everywhere

### Tips for maximum impact:

- **Post during US morning** (9-11am ET / 6-8am PT) for best Reddit/HN visibility
- **Reply to every comment** within the first 2 hours — engagement boosts ranking
- **Upvote from other accounts is NOT allowed** — just let quality drive it
- **Cross-link** between posts (e.g. "Discussion also on r/LocalLLaMA: [link]")
- **Demo video/GIF is the #1 force multiplier** — record it before posting anything
- **GitHub Discussions** — enable them and link in all posts as the "home base" for questions
