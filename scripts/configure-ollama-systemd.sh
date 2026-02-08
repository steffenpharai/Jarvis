#!/usr/bin/env bash
# Configure systemd ollama service for Jetson Orin Nano 8GB.
#
# Key memory-saving settings (from Jetson AI Labs + Ollama source):
#   OLLAMA_FLASH_ATTENTION=1     – use flash attention (dramatically less KV cache memory)
#   OLLAMA_KV_CACHE_TYPE=q8_0    – quantize KV cache to 8-bit (halves KV memory vs f16)
#   OLLAMA_NUM_PARALLEL=1        – only one concurrent request (avoids duplicate KV caches)
#   OLLAMA_MAX_LOADED_MODELS=1   – only one model in GPU at a time
#   OLLAMA_CONTEXT_LENGTH=512    – default context; matches Jarvis app OLLAMA_NUM_CTX
#   OLLAMA_GPU_OVERHEAD=500000000 – reserve ~500 MB for X11/GNOME/Cursor (bytes)
#   OLLAMA_KEEP_ALIVE=5m         – unload model after 5 min idle to free GPU memory
#   OLLAMA_NUM_GPU=1             – use GPU for inference
#
# Run with:  sudo bash scripts/configure-ollama-systemd.sh
# Override:  OLLAMA_KEEP_ALIVE=-1 sudo bash scripts/configure-ollama-systemd.sh
# Then:      sudo systemctl daemon-reload && sudo systemctl restart ollama
set -e

OVERRIDE_DIR="/etc/systemd/system/ollama.service.d"
OVERRIDE_FILE="${OVERRIDE_DIR}/gpu.conf"

# Defaults optimized for Jetson Orin Nano 8GB with Cursor IDE running
FLASH_ATTENTION="${OLLAMA_FLASH_ATTENTION:-1}"
KV_CACHE_TYPE="${OLLAMA_KV_CACHE_TYPE:-q8_0}"
NUM_PARALLEL="${OLLAMA_NUM_PARALLEL:-1}"
MAX_LOADED="${OLLAMA_MAX_LOADED_MODELS:-1}"
CONTEXT_LENGTH="${OLLAMA_CONTEXT_LENGTH:-512}"
GPU_OVERHEAD="${OLLAMA_GPU_OVERHEAD:-500000000}"
KEEP_ALIVE="${OLLAMA_KEEP_ALIVE:-5m}"

mkdir -p "${OVERRIDE_DIR}"

cat > "${OVERRIDE_FILE}" << EOF
[Service]
Environment="OLLAMA_NUM_GPU=1"
Environment="OLLAMA_FLASH_ATTENTION=${FLASH_ATTENTION}"
Environment="OLLAMA_KV_CACHE_TYPE=${KV_CACHE_TYPE}"
Environment="OLLAMA_NUM_PARALLEL=${NUM_PARALLEL}"
Environment="OLLAMA_MAX_LOADED_MODELS=${MAX_LOADED}"
Environment="OLLAMA_CONTEXT_LENGTH=${CONTEXT_LENGTH}"
Environment="OLLAMA_GPU_OVERHEAD=${GPU_OVERHEAD}"
Environment="OLLAMA_KEEP_ALIVE=${KEEP_ALIVE}"
LimitMEMLOCK=infinity
EOF

echo "=== Ollama systemd override written ==="
echo "File: ${OVERRIDE_FILE}"
echo ""
cat "${OVERRIDE_FILE}"
echo ""
echo "Next steps:"
echo "  1. Drop kernel caches:  sudo sh -c 'echo 3 > /proc/sys/vm/drop_caches'"
echo "  2. Reload + restart:    sudo systemctl daemon-reload && sudo systemctl restart ollama"
echo "  3. Verify:              ollama run llama3.2:1b 'hello' --verbose"
