#!/usr/bin/env bash
# jarvis-boot.sh — Pre-flight checks before launching JARVIS server.
#
# Called by jarvis.service ExecStartPre. Ensures:
#   1. Ollama is reachable
#   2. The required LLM model is pulled
#   3. Camera device exists
#   4. YOLOE engine exists
#
# Exits 0 on success; non-zero aborts the service start.
set -e

OLLAMA_URL="${OLLAMA_BASE_URL:-http://127.0.0.1:11434}"
OLLAMA_MODEL="${OLLAMA_MODEL:-qwen3:1.7b}"
PROJECT_ROOT="/home/jarvis/Jarvis"

echo "[jarvis-boot] Waiting for Ollama at ${OLLAMA_URL} ..."
for i in $(seq 1 30); do
    if curl -sf "${OLLAMA_URL}/api/tags" >/dev/null 2>&1; then
        echo "[jarvis-boot] Ollama is up (attempt ${i})."
        break
    fi
    if [ "$i" -eq 30 ]; then
        echo "[jarvis-boot] ERROR: Ollama not reachable after 30 attempts."
        exit 1
    fi
    sleep 2
done

# Check model is pulled
if ! curl -sf "${OLLAMA_URL}/api/tags" | grep -q "${OLLAMA_MODEL}"; then
    echo "[jarvis-boot] WARNING: Model '${OLLAMA_MODEL}' not found. Pulling..."
    ollama pull "${OLLAMA_MODEL}" || echo "[jarvis-boot] WARNING: Pull failed; JARVIS will report model missing at runtime."
fi

# Check camera
if [ -e /dev/video0 ]; then
    echo "[jarvis-boot] Camera /dev/video0 found."
else
    echo "[jarvis-boot] WARNING: /dev/video0 not found; vision will be degraded."
fi

# Check YOLOE engine
if [ -f "${PROJECT_ROOT}/models/yoloe26n.engine" ]; then
    echo "[jarvis-boot] YOLOE engine found."
else
    echo "[jarvis-boot] WARNING: yoloe26n.engine missing; vision will fail."
fi

echo "[jarvis-boot] Pre-flight complete."
