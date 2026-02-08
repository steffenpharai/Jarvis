#!/usr/bin/env bash
# Inspect effective Ollama configuration (systemd env and overrides).
# Run with sudo for systemd properties: sudo scripts/inspect-ollama-config.sh
# Without sudo, only user-writable paths and hints are shown.
set -e

echo "=== Ollama configuration (Jetson 8GB) ==="
echo ""

# Systemd override files (need sudo to read if restricted)
echo "--- Systemd override files ---"
if [ -d /etc/systemd/system/ollama.service.d ]; then
  for f in /etc/systemd/system/ollama.service.d/*.conf; do
    [ -f "$f" ] || continue
    echo "File: $f"
    cat "$f" 2>/dev/null || sudo cat "$f" 2>/dev/null || echo "(cannot read)"
    echo ""
  done
else
  echo "No /etc/systemd/system/ollama.service.d (Ollama may not be installed as systemd service)"
fi

# Effective environment (requires sudo when run as systemd service)
echo "--- Effective service environment ---"
if systemctl show ollama --property=Environment 2>/dev/null | grep -q Environment; then
  systemctl show ollama --property=Environment 2>/dev/null || sudo systemctl show ollama --property=Environment 2>/dev/null || true
else
  echo "Ollama service not found or not loaded. If using systemd: sudo systemctl show ollama --property=Environment"
fi
echo ""

# User/data paths
echo "--- Paths ---"
echo "Models (default): ${OLLAMA_MODELS:-$HOME/.ollama/models}"
[ -d "${HOME}/.ollama" ] && echo "  $HOME/.ollama exists" || true
[ -d /etc/ollama ] && echo "  /etc/ollama exists" || true
[ -d /var/log/ollama ] && echo "  /var/log/ollama exists" || true
echo ""

# Loaded model
echo "--- Currently loaded model ---"
ollama ps 2>/dev/null || echo "(ollama ps failed)"
echo ""

# Suggestions for 8GB Jetson
echo "--- Suggested for 8GB Jetson ---"
echo "Apply all settings at once:  sudo bash scripts/configure-ollama-systemd.sh"
echo ""
echo "Key settings (set in systemd, not per-request):"
echo '  OLLAMA_FLASH_ATTENTION=1        # flash attention (dramatically less KV cache memory)'
echo '  OLLAMA_KV_CACHE_TYPE=q8_0       # quantize KV cache to 8-bit (halves vs f16)'
echo '  OLLAMA_NUM_PARALLEL=1           # single concurrent request'
echo '  OLLAMA_MAX_LOADED_MODELS=1      # only one model in GPU at a time'
echo '  OLLAMA_CONTEXT_LENGTH=512       # default context; match Jarvis OLLAMA_NUM_CTX'
echo '  OLLAMA_GPU_OVERHEAD=500000000   # reserve ~500 MB for X11/GNOME/Cursor'
echo '  OLLAMA_KEEP_ALIVE=5m            # unload after 5 min idle to free GPU memory'
echo ""
echo "Then: sudo systemctl daemon-reload && sudo systemctl restart ollama"
echo ""

# Current Orin Nano memory
echo "--- Current memory (Orin Nano) ---"
free -h | head -2
if command -v tegrastats &>/dev/null; then
  echo "Tegrastats (one sample):"
  timeout 2 tegrastats --interval 1000 2>/dev/null | head -1 || true
fi
echo ""
echo "Tip: Before loading model, drop caches:  sudo sh -c 'echo 3 > /proc/sys/vm/drop_caches'"
echo "Note: With Cursor IDE, free RAM may be ~1.6 GiB but 'available' is ~4 GiB."
echo "cudaMalloc needs actual free memory; drop caches to reclaim buff/cache for CUDA."
