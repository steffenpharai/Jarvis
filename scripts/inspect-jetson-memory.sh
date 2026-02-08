#!/usr/bin/env bash
# Inspect Orin Nano memory. Run with sudo for full visibility.
# Use this to understand why Ollama might OOM when Cursor/IDE is running.
set -e

echo "=== Jetson Orin Nano memory ==="
echo ""

echo "--- RAM and swap ---"
free -h
echo ""

echo "--- /proc/meminfo (summary) ---"
grep -E "^(MemTotal|MemFree|MemAvailable|Buffers|Cached|SwapTotal|SwapFree|AnonPages|Mapped)" /proc/meminfo 2>/dev/null || sudo grep -E "^(MemTotal|MemFree|MemAvailable|SwapTotal|SwapFree)" /proc/meminfo 2>/dev/null
echo ""

echo "--- Top processes by RSS (memory) ---"
ps -eo pid,rss,comm --sort=-rss 2>/dev/null | head -16 | awk 'NR==1 {print "PID    RSS(MB)  COMMAND"; next} {rss=$2/1024; printf "%6s  %7.0f  %s\n", $1, rss, $3}'
echo ""

if command -v tegrastats &>/dev/null; then
  echo "--- Tegrastats (one sample; RAM/SWAP) ---"
  timeout 2 tegrastats --interval 1000 2>/dev/null | head -1 || true
  echo ""
fi

echo "--- Ollama loaded model ---"
ollama ps 2>/dev/null || echo "(ollama ps failed)"
echo ""

echo "--- Note ---"
echo "Cursor IDE + Chrome + GNOME often use 2-3 GiB. 'free' RAM may show ~1.6 GiB"
echo "while 'available' shows ~4 GiB. The difference is buff/cache."
echo ""
echo "CRITICAL: cudaMalloc (nvmap) needs actual free memory, not 'available'."
echo "Before model load, drop caches:  sudo sh -c 'echo 3 > /proc/sys/vm/drop_caches'"
echo ""
echo "Jarvis OOM prevention stack:"
echo "  1. systemd: OLLAMA_FLASH_ATTENTION=1, OLLAMA_KV_CACHE_TYPE=q8_0"
echo "  2. systemd: OLLAMA_GPU_OVERHEAD=500000000, OLLAMA_NUM_PARALLEL=1"
echo "  3. systemd: OLLAMA_CONTEXT_LENGTH=512, OLLAMA_KEEP_ALIVE=5m"
echo "  4. Python:  OLLAMA_NUM_CTX=512, OLLAMA_NUM_CTX_MAX=512"
echo "  5. Python:  On OOM -> unload model + drop caches + retry smaller ctx"
echo ""
echo "Apply all:  sudo bash scripts/configure-ollama-systemd.sh"
