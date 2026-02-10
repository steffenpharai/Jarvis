# J.A.R.V.I.S. — Dockerfile for Jetson Orin Nano Super (JetPack 6.x)
#
# This is a starter Dockerfile. It builds the Python backend and PWA frontend.
# TensorRT engine files must be built on-device (they're architecture-specific).
#
# Usage:
#   docker build -t jarvis .
#   docker run --runtime nvidia --gpus all --device /dev/video0 \
#     --net host --privileged \
#     -v /run/dbus:/run/dbus \
#     -v $(pwd)/models:/app/models \
#     -v $(pwd)/data:/app/data \
#     jarvis python main.py --serve
#
# Note: For full audio/Bluetooth support, --privileged and dbus mount are needed.
# For production, prefer running natively (see README quickstart).

FROM nvcr.io/nvidia/l4t-jetpack:r36.4.0

# Avoid interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-pip \
    python3-venv \
    python3-dev \
    libopenblas-dev \
    libsndfile1 \
    portaudio19-dev \
    bluez \
    bluez-tools \
    curl \
    git \
    nodejs \
    npm \
    && rm -rf /var/lib/apt/lists/*

# Install Ollama
RUN curl -fsSL https://ollama.com/install.sh | sh

WORKDIR /app

# Python dependencies
COPY requirements.txt .
RUN python3 -m pip install --no-cache-dir --upgrade pip && \
    python3 -m pip install --no-cache-dir -r requirements.txt

# PWA build
COPY pwa/package*.json pwa/
RUN cd pwa && npm ci --production=false

COPY pwa/ pwa/
RUN cd pwa && npm run build

# Copy application code
COPY . .

# Models directory (mount as volume for TensorRT engines)
RUN mkdir -p models data

# Expose server port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default: run in serve mode
CMD ["python3", "main.py", "--serve"]
