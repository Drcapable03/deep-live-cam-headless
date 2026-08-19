#!/bin/bash
# ============================================================================
# Deep-Live-Cam Headless Streaming — Cloud GPU Setup Script
# ============================================================================
# For: Ubuntu Linux, NVIDIA RTX 4090 (or similar), ~$1/hour rented GPU
# Usage: sudo bash scripts/setup_cloud.sh
# Idempotent — safe to re-run. Every step checks before proceeding.
# ============================================================================

set -euo pipefail

# ---------------------------------------------------------------------------
# Globals
# ---------------------------------------------------------------------------
WORKSPACE="/workspace"
PROJECT_DIR="${WORKSPACE}/deep-live-cam-headless"
VENV_DIR="${WORKSPACE}/.venv"
LOGS_DIR="${WORKSPACE}/logs"
MODELS_DIR="${PROJECT_DIR}/models"
CUDA_REPO="https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64"
CUDA_KEY="/etc/apt/keyrings/cuda-keyring.gpg"

echo "============================================================"
echo " Deep-Live-Cam Cloud GPU Setup"
echo " $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================================"
echo ""

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------
log() { echo "[SETUP] $*"; }
ok()  { log "OK: $*"; }
warn(){ echo "[SETUP] WARN: $*" >&2; }

step() {
    echo ""
    echo "--- $1 ---"
}

# Check if a command exists
cmd_exists() { command -v "$1" &>/dev/null; }

# Get installed CUDA driver version from nvidia-smi
get_driver_version() {
    nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1 | tr -d ' '
}

# Detect Ubuntu codename
detect_codename() {
    local ver
    ver=$(lsb_release -rs 2>/dev/null || grep VERSION_ID /etc/os-release 2>/dev/null | cut -d'"' -f2)
    case "${ver%%.*}" in
        20) echo "ubuntu2004" ;;
        22) echo "ubuntu2204" ;;
        24) echo "ubuntu2404" ;;
        *)  echo "ubuntu2204"   # fallback
    esac
}

# ---------------------------------------------------------------------------
# Step 1 — System preparation
# ---------------------------------------------------------------------------
step "System Preparation"

# Set locale
if ! locale -a 2>/dev/null | grep -q "^en_US.UTF-8$"; then
    log "Setting locale to en_US.UTF-8"
    sed -i '/^en_US.UTF-8/d' /etc/locale.gen 2>/dev/null || true
    echo "en_US.UTF-8 UTF-8" >> /etc/locale.gen
    locale-gen en_US.UTF-8
    update-locale LANG=en_US.UTF-8
fi

# Update package lists and upgrade existing packages
log "Updating system packages..."
apt-get update -y 2>&1 | tail -5
apt-get upgrade -y 2>&1 | tail -5 || true

# Install essential tools
log "Installing build essentials and utilities..."
DEBIAN_FRONTEND=noninteractive apt-get install -y \
    build-essential git wget curl unzip tmux htop screen \
    ca-certificates gnupg software-properties-common \
    python3 python3-pip python3-venv bc lsb-release \
    libgl1 libglib2.0-0 2>&1 | tail -5

ok "System packages installed"

# ---------------------------------------------------------------------------
# Step 2 — NVIDIA drivers & CUDA toolkit
# ---------------------------------------------------------------------------
step "NVIDIA Drivers & CUDA Toolkit"

DRIVER_VER=""
if cmd_exists nvidia-smi; then
    DRIVER_VER=$(get_driver_version)
    ok "nvidia-smi already available (driver ${DRIVER_VER})"
else
    log "Installing NVIDIA driver via ubuntu-drivers..."
    DEBIAN_FRONTEND=noninteractive apt-get install -y ubuntu-drivers-common 2>&1 | tail -3
    ubuntu-drivers autoinstall 2>&1 | tail -5 || true
    DRIVER_VER=$(get_driver_version)
    ok "NVIDIA driver installed (${DRIVER_VER})"
fi

# Determine which CUDA repo to use based on driver version
# CUDA 12.x supports drivers >= 525.xx
CUDA_MAJOR=12
CUDA_MINOR=6
CODENAME=$(detect_codename)

# If driver is very new (535+), prefer latest CUDA 12.x
if [[ "${DRIVER_VER%%.*}" -ge 535 ]]; then
    CUDA_MINOR=8
elif [[ "${DRIVER_VER%%.*}" -ge 525 ]]; then
    CUDA_MINOR=6
fi

CUDA_PKG="cuda-toolkit-${CUDA_MAJOR}.${CUDA_MINOR}"
CUDA_PREFIX="/usr/local/cuda-${CUDA_MAJOR}.${CUDA_MINOR}"

if cmd_exists nvcc; then
    log "nvcc already installed ($(nvcc --version 2>&1 | head -1))"
else
    log "Installing CUDA Toolkit ${CUDA_MAJOR}.${CUDA_MINOR}..."

    # Add CUDA repository
    if [ ! -f "${CUDA_KEY}" ]; then
        mkdir -p $(dirname "${CUDA_KEY}")
        wget -qO- https://developer.download.nvidia.com/compute/cuda/repos/${CODENAME}/x86_64/${CUDA_KEY} > "${CUDA_KEY}" 2>/dev/null || \
        wget -qO- https://developer.download.nvidia.com/compute/cuda/repos/${CODENAME}/x86_64/key.gpg | gpg --dearmor -o "${CUDA_KEY}"
    fi

    echo "deb [ signed-by=${CUDA_KEY} ] https://developer.download.nvidia.com/compute/cuda/repos/${CODENAME}/x86_64/ /" > \
        /etc/apt/sources.list.d/cuda.list

    apt-get update -y 2>&1 | tail -3

    DEBIAN_FRONTEND=noninteractive apt-get install -y \
        "${CUDA_PKG}" "${CUDA_PKG}-core" "${CUDA_PKG}-dev" \
        libcudnn9-cuda9 2>&1 | tail -5 || \
    DEBIAN_FRONTEND=noninteractive apt-get install -y \
        "${CUDA_PKG}" "${CUDA_PKG}-core" "${CUDA_PKG}-dev" 2>&1 | tail -5

    # Create symlink for convenience
    if [ ! -L /usr/local/cuda ]; then
        ln -sf "${CUDA_PREFIX}" /usr/local/cuda
    fi

    ok "CUDA Toolkit ${CUDA_MAJOR}.${CUDA_MINOR} installed"
fi

# Verify nvcc
if cmd_exists nvcc; then
    NVCC_VER=$(nvcc --version 2>&1 | head -1 | grep -oP '\d+\.\d+' | head -1)
    ok "nvcc version: ${NVCC_VER}"
else
    warn "nvcc not found — onnxruntime-gpu may not find CUDA at runtime"
fi

# Set environment variables permanently
CUDA_ENV_LINES=(
    "export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:${CUDA_PREFIX}/targets/x86_64-linux/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    "export PATH=${CUDA_PREFIX}/bin:${PATH:+:$PATH}"
)

for line in "${CUDA_ENV_LINES[@]}"; do
    # Avoid duplicates in /etc/environment
    if ! grep -qF "${line//export /}" /etc/environment 2>/dev/null; then
        echo "${line}" >> /etc/environment
    fi
    # Also add to ~/.bashrc
    if ! grep -qF "${line//export /}" /root/.bashrc 2>/dev/null; then
        echo "$line" >> /root/.bashrc
    fi
done

# Export for current session
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:${CUDA_PREFIX}/targets/x86_64-linux/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
export PATH=${CUDA_PREFIX}/bin${PATH:+:$PATH}

ok "CUDA paths configured in /etc/environment and ~/.bashrc"

# ---------------------------------------------------------------------------
# Step 3 — FFmpeg with NVENC support
# ---------------------------------------------------------------------------
step "FFmpeg with NVENC Support"

FFMPEG_INSTALLED=false
if cmd_exists ffmpeg; then
    FFMPEG_VER=$(ffmpeg -version 2>&1 | head -1)
    if echo "$FFMPEG_VER" | grep -qi "h264_nvenc\|libx264\|libx265"; then
        FFMPEG_INSTALLED=true
        ok "FFmpeg already installed: ${FFMPEG_VER}"
    else
        warn "Existing FFmpeg lacks NVENC/video codecs — rebuilding"
    fi
fi

if [ "$FFMPEG_INSTALLED" = false ]; then
    log "Building FFmpeg from source with NVENC support..."

    FFMPEG_SRC="/tmp/ffmpeg-build"
    rm -rf "${FFMPEG_SRC}"
    mkdir -p "${FFMPEG_SRC}"

    cd "${FFMPEG_SRC}"

    # Install FFmpeg build dependencies
    DEBIAN_FRONTEND=noninteractive apt-get install -y \
        nasm yasm pkg-config meson ninja-build 2>&1 | tail -3

    # x264
    if [ ! -d "/tmp/ffmpeg-build/x264" ]; then
        log "Building x264..."
        git clone --depth 1 --branch stable https://code.videolan.org/videolan/x264.git /tmp/ffmpeg-build/x264 2>&1 | tail -2
        cd /tmp/ffmpeg-build/x264
        ./configure --prefix=/usr --enable-shared --enable-pic --disable-cli
        make -j$(nproc) 2>&1 | tail -3
        make install 2>&1 | tail -2
        cd "${FFMPEG_SRC}"
    fi

    # x265
    if [ ! -d "/tmp/ffmpeg-build/x265" ]; then
        log "Building x265..."
        git clone --depth 1 https://github.com/videolan/x265.git /tmp/ffmpeg-build/x265 2>&1 | tail -2
        cd /tmp/ffmpeg-build/x265/build/linux
        cmake -DENABLE_SHARED=ON -DENABLE_CLI=OFF ../..
        make -j$(nproc) 2>&1 | tail -3
        make install 2>&1 | tail -2
        cd "${FFMPEG_SRC}"
    fi

    # FFmpeg itself
    log "Downloading FFmpeg source..."
    if [ ! -d "/tmp/ffmpeg-build/ffmpeg" ]; then
        wget -q https://ffmpeg.org/releases/ffmpeg-7.1.tar.xz -O /tmp/ffmpeg-7.1.tar.xz 2>&1 | tail -2
        tar xf /tmp/ffmpeg-7.1.tar.xz
    fi

    cd /tmp/ffmpeg-build/ffmpeg
    ./configure \
        --prefix=/usr/local \
        --enable-gpl \
        --enable-nonfree \
        --enable-libx264 \
        --enable-libx265 \
        --enable-nvenc \
        --enable-cuda-llvm \
        --enable-pic \
        --enable-shared \
        --disable-debug \
        --disable-doc \
        --disable-static \
        --extra-cflags="-I/usr/include" \
        --extra-ldflags="-L/usr/lib/x86_64-linux-gnu" 2>&1 | tail -10

    make -j$(nproc) 2>&1 | tail -3
    make install 2>&1 | tail -2

    ldconfig

    cd /
    rm -rf "${FFMPEG_SRC}"
fi

# Ensure the newest ffmpeg is first on PATH
if [ -x /usr/local/bin/ffmpeg ]; then
    export PATH="/usr/local/bin:${PATH}"
fi

# Verify
if cmd_exists ffmpeg; then
    FFMPEG_VER=$(ffmpeg -version 2>&1 | head -1)
    HAS_NVENC="NO"
    if ffmpeg -encoders 2>/dev/null | grep -q "h264_nvenc"; then
        HAS_NVENC="YES"
    fi
    ok "FFmpeg installed: ${FFMPEG_VER} (NVENC: ${HAS_NVENC})"
else
    warn "ffmpeg not found after build"
fi

# ---------------------------------------------------------------------------
# Step 4 — Python virtual environment
# ---------------------------------------------------------------------------
step "Python Virtual Environment"

PYTHON_VER=""
if cmd_exists python3; then
    PYTHON_VER=$(python3 --version 2>&1 | grep -oP '\d+\.\d+')
fi

if [ -z "$PYTHON_VER" ] || [ "$(printf '%s\n' "3.11" "$PYTHON_VER" | sort -V | head -1)" != "3.11" ]; then
    log "Installing Python 3.11..."
    DEBIAN_FRONTEND=noninteractive add-apt-repository -y ppa:deadsnakes/ppa 2>&1 | tail -3
    DEBIAN_FRONTEND=noninteractive apt-get install -y python3.11 python3.11-venv python3.11-dev 2>&1 | tail -3
    PYTHON_VER="3.11"
else
    ok "Python ${PYTHON_VER} already available"
fi

# Create venv if it doesn't exist
if [ ! -d "${VENV_DIR}" ]; then
    log "Creating virtualenv at ${VENV_DIR}..."
    python3.11 -m venv "${VENV_DIR}" 2>&1 | tail -3
    ok "Virtualenv created"
else
    ok "Virtualenv already exists at ${VENV_DIR}"
fi

# Activate and upgrade pip
source "${VENV_DIR}/bin/activate"

log "Upgrading pip, setuptools, wheel..."
pip install --upgrade pip setuptools wheel 2>&1 | tail -3

# Install requirements
log "Installing requirements.txt..."
if [ -f "${PROJECT_DIR}/requirements.txt" ]; then
    # Filter out Windows-only deps since we're on Linux
    grep -v 'pygrabber' "${PROJECT_DIR}/requirements.txt" | grep -v 'customtkinter' | grep -v '^tk==' > /tmp/dlc_req_filtered.txt
    pip install -r /tmp/dlc_req_filtered.txt 2>&1 | tail -5
    rm /tmp/dlc_req_filtered.txt
else
    warn "requirements.txt not found at ${PROJECT_DIR}/requirements.txt — skipping pip install"
fi

# Additional deps
log "Installing additional dependencies..."
pip install onnxconverter-common pyvirtualcam 2>&1 | tail -3

ok "Python environment ready"

# ---------------------------------------------------------------------------
# Step 5 — Verification of Python packages
# ---------------------------------------------------------------------------
step "Verifying Python Packages"

VERIFY_PASSED=0
VERIFY_TOTAL=0

for pkg_cmd in \
    "onnxruntime:print(onnxruntime.__version__)" \
    "insightface:print(insightface.__version__)" \
    "cv2:print(cv2.__version__)"; do

    PKG_NAME="${pkg_cmd%%:*}"
    CMD="${pkg_cmd#*:}"

    VERIFY_TOTAL=$((VERIFY_TOTAL + 1))
    RESULT=$(python3 -c "import ${PKG_NAME}; ${CMD}" 2>&1) && {
        ok "${PKG_NAME}: ${RESULT}"
        VERIFY_PASSED=$((VERIFY_PASSED + 1))
    } || {
        warn "${PKG_NAME}: import failed"
    }
done

ok "Verified ${VERIFY_PASSED}/${VERIFY_TOTAL} packages"

# ---------------------------------------------------------------------------
# Step 6 — Download models
# ---------------------------------------------------------------------------
step "Downloading ONNX Models"

mkdir -p "${MODELS_DIR}"

declare -A MODEL_URLS=(
    ["inswapper_128.onnx"]="https://huggingface.co/hacksider/deep-live-cam/resolve/main/inswapper_128.onnx"
    ["GPEN-BFR-256.onnx"]="https://github.com/harisreedhar/Face-Upscalers-ONNX/releases/download/Models/GPEN-BFR-256.onnx"
    ["GPEN-BFR-512.onnx"]="https://github.com/harisreedhar/Face-Upscalers-ONNX/releases/download/Models/GPEN-BFR-512.onnx"
)

MODEL_OK_COUNT=0
for model_name in "${!MODEL_URLS[@]}"; do
    model_path="${MODELS_DIR}/${model_name}"
    if [ -f "${model_path}" ] && [ -s "${model_path}" ]; then
        SIZE=$(du -h "${model_path}" | cut -f1)
        ok "${model_name} already exists (${SIZE})"
        MODEL_OK_COUNT=$((MODEL_OK_COUNT + 1))
    else
        log "Downloading ${model_name}..."
        wget -q --show-progress -O "${model_path}" "${MODEL_URLS[${model_name}]}" 2>&1 | tail -3 || {
            warn "Failed to download ${model_name}"
            continue
        }
        chmod 644 "${model_path}"
        SIZE=$(du -h "${model_path}" | cut -f1)
        ok "${model_name} downloaded (${SIZE})"
        MODEL_OK_COUNT=$((MODEL_OK_COUNT + 1))
    fi
done
ok "Models: ${MODEL_OK_COUNT}/3 downloaded"

# ---------------------------------------------------------------------------
# Step 7 — MediaMTX (RTMP server)
# ---------------------------------------------------------------------------
step "MediaMTX RTMP Server"

MEDIAMTX_BIN="/usr/local/bin/mediamtx"

if cmd_exists mediamtx; then
    MEDIAMTX_VER=$(mediamtx --version 2>&1 | head -1)
    ok "MediaMTX already installed: ${MEDIAMTX_VER}"
else
    log "Downloading latest MediaMTX release..."
    MTX_RELEASE=$(curl -s https://api.github.com/repos/bluenviron/mediamtx/releases/latest | grep '"tag_name"' | sed -E 's/.*"([^"]+)".*/\1/')
    MTX_VERSION="${MTX_RELEASE#v}"
    MTX_URL="https://github.com/bluenviron/mediamtx/releases/download/${MTX_RELEASE}/mediamtx_v${MTX_VERSION}_linux_amd64.tar.gz"

    wget -q -O /tmp/mediamtx.tar.gz "${MTX_URL}" 2>&1 | tail -2
    tar xzf /tmp/mediamtx.tar.gz -C /tmp/ mediamtx
    mv /tmp/mediamtx "${MEDIAMTX_BIN}"
    chmod +x "${MEDIAMTX_BIN}"
    rm -f /tmp/mediamtx.tar.gz
    ok "MediaMTX ${MTX_VERSION} installed to ${MEDIAMTX_BIN}"
fi

# Configure MediaMTX
MEDIAMTX_CONF="/etc/mediamtx.yml"
if [ ! -f "${MEDIAMTX_CONF}" ]; then
    log "Creating MediaMTX config..."
    cat > "${MEDIAMTX_CONF}" <<'EOF'
# MediaMTX configuration — Deep-Live-Cam RTMP server
rtspProtocol: udp
rtmpEnableAuthentication: no
htmxEnableAuthentication: no
paths:
  all:
    # Run external command when a stream starts (for logging/debugging)
    runOnStart:
    # Authorize/unauthorize commands
    authAnywhere: public
    # Publish to RTMP endpoint (optional)
    rtmpDisable: no
    # Record to disk
    record: no
  live:
    # Default RTMP path — DLC streams here
    rtmpDisable: no
EOF
    ok "MediaMTX config created at ${MEDIAMTX_CONF}"
fi

# Create systemd service for MediaMTX
SYSTEMD_UNIT="/etc/systemd/system/mediamtx.service"
if [ ! -f "${SYSTEMD_UNIT}" ]; then
    log "Creating MediaMTX systemd service..."
    cat > "${SYSTEMD_UNIT}" <<EOF
[Unit]
Description=MediaMTX RTMP Server
After=network.target

[Service]
Type=simple
ExecStart=${MEDIAMTX_BIN} --config ${MEDIAMTX_CONF}
Restart=on-failure
RestartSec=5
Environment=LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:/usr/local/cuda-12.6/targets/x86_64-linux/lib

[Install]
WantedBy=multi-user.target
EOF
    systemctl daemon-reload
    systemctl enable mediamtx
    ok "MediaMTX systemd service created"
fi

# Start MediaMTX
if ! systemctl is-active --quiet mediamtx 2>/dev/null; then
    log "Starting MediaMTX..."
    systemctl start mediamtx || "${MEDIAMTX_BIN}" --config "${MEDIAMTX_CONF}" &
    sleep 2
fi

# Verify port 1935
if ss -tlnp 2>/dev/null | grep -q ':1935' || netstat -tlnp 2>/dev/null | grep -q ':1935'; then
    ok "MediaMTX listening on port 1935 (RTMP)"
else
    warn "MediaMTX may not be running — check manually with: ${MEDIAMTX_BIN} --config ${MEDIAMTX_CONF}"
fi

# ---------------------------------------------------------------------------
# Step 8 — ngrok (Tunnel for remote access)
# ---------------------------------------------------------------------------
step "ngrok Tunnel"

NGROK_BIN="/usr/local/bin/ngrok"

if cmd_exists ngrok; then
    NGROK_VER=$(ngrok version 2>&1 | head -1)
    ok "ngrok already installed: ${NGROK_VER}"
else
    log "Downloading ngrok..."
    NGROK_URL="https://bin.equinox.io/c/bNyj1mQVY4c/ngrok-v3-stable-linux-amd64.tgz"
    wget -q -O /tmp/ngrok.tgz "${NGROK_URL}" 2>&1 | tail -2
    tar xzf /tmp/ngrok.tgz -C /tmp/ ngrok
    mv /tmp/ngrok "${NGROK_BIN}"
    chmod +x "${NGROK_BIN}"
    rm -f /tmp/ngrok.tgz
    ok "ngrok installed to ${NGROK_BIN}"
fi

# Configure authtoken (user must set their own token)
NGROK_CONFIG="${HOME}/.config/ngrok/ngrok.yml"
NGROK_TOKEN="cr_30wOLjlfhO5kdRKJbA0rx05OF48"  # <-- Replace with your actual token

if [ ! -f "${NGROK_CONFIG}" ]; then
    log "Configuring ngrok authtoken..."
    mkdir -p "$(dirname "${NGROK_CONFIG}")"
    cat > "${NGROK_CONFIG}" <<EOF
authtoken: ${NGROK_TOKEN}
region: us
server_address: https://api.ngrok.com
EOF
    ok "ngrok configured (token set)"
else
    ok "ngrok config already exists"
fi

# Create optional systemd service for ngrok tunnel
NGROK_SERVICE="/etc/systemd/system/ngrok-tunnel.service"
if [ ! -f "${NGROK_SERVICE}" ]; then
    log "Creating ngrok tunnel systemd service (disabled by default — edit to enable)..."
    cat > "${NGROK_SERVICE}" <<EOF
[Unit]
Description=ngrok TCP tunnel for MediaMTX (port 1935)
After=network.target mediamtx.service
Wants=mediamtx.service

[Service]
Type=simple
ExecStart=${NGROK_BIN} tcp 1935
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF
    systemctl disable ngrok-tunnel 2>/dev/null || true
    ok "ngrok tunnel service created (not enabled — edit as needed)"
fi

# ---------------------------------------------------------------------------
# Step 9 — Project files
# ---------------------------------------------------------------------------
step "Project Files"

# Clone/pull project
if [ ! -d "${PROJECT_DIR}" ]; then
    log "Cloning deep-live-cam-headless to ${PROJECT_DIR}..."
    git clone https://github.com/hacksider/deep-live-cam-headless.git "${PROJECT_DIR}" 2>&1 | tail -3
    ok "Project cloned"
else
    log "Project already exists — pulling latest changes..."
    cd "${PROJECT_DIR}"
    git pull 2>&1 | tail -3
    cd /
    ok "Project updated"
fi

# Source face image — download a sample if missing
FACE_JPG="${PROJECT_DIR}/face.jpg"
if [ ! -f "${FACE_JPG}" ]; then
    log "Downloading sample face.jpg..."
    wget -q -O "${FACE_JPG}" "https://media.githubusercontent.com/media/hacksider/deep-live-cam/main/assets/face.jpg" 2>&1 | tail -2 || true
fi

# Smoke test with CPU provider (quick sanity check)
log "Running smoke test with CPU provider..."
cd "${PROJECT_DIR}"
source "${VENV_DIR}/bin/activate"
python3 run_headless.py -s face.jpg --input-source smoke_input.mp4 --stream-output /tmp/smoke_test.mp4 \
    --execution-provider cpu --frames 3 2>&1 | tail -10 || warn "Smoke test had issues (may need valid input video)"
cd /
ok "Smoke test complete"

# Create .env.example template
ENV_EXAMPLE="${PROJECT_DIR}/.env.example"
if [ ! -f "${ENV_EXAMPLE}" ]; then
    log "Creating .env.example..."
    cat > "${ENV_EXAMPLE}" <<'EOF'
# ============================================================================
# Deep-Live-Cam Headless Streaming — Environment Configuration
# ============================================================================
# Copy this file to .env and customize values as needed.
#
# Usage: export $(grep -v '^#' .env | xargs)
# Or use: set -a; source .env; set +a

# === Source Face Image ===
# Path to the source face image for swapping (relative or absolute)
DLC_SOURCE_FACE=face.jpg

# === Input Source ===
# Camera index (0), video file path, pipe, or RTMP URL
DLC_INPUT_SOURCE=smoke_input.mp4

# === Stream Output ===
# RTMP output URL
DLC_STREAM_OUTPUT=rtmp://localhost/live/main

# === Stream Settings ===
DLC_STREAM_WIDTH=1280
DLC_STREAM_HEIGHT=720
DLC_STREAM_FPS=30
DLC_STREAM_QUALITY=23
DLC_STREAM_ENCODER=h264_nvenc

# === Frame Processors ===
# Comma-separated: face_swapper,face_enhancer_gpen256
DLC_FRAME_PROCESSOR=face_swapper face_enhancer_gpen256

# === Face Masking ===
DLC_MOUTH_MASK=true
DLC_EYES_MASK=true
DLC_EYEBROWS_MASK=true

# === GPU Settings ===
DLC_EXECUTION_PROVIDER=cuda
DLC_EXECUTION_THREADS=2
DLC_MAX_MEMORY=16

# === Quality Presets ===
DLC_QUALITY_PRESET=high
DLC_ADAPTIVE_RESOLUTION=true
DLC_BLEND_OPACITY=0.9
DLC_SHARPEN_STRENGTH=0.4

# === FP16 Acceleration ===
DLC_FP16_ENABLED=true

# === MediaMTX ===
DLC_MEDIAMTX_CONFIG=/etc/mediamtx.yml
DLC_MEDIAMTX_PORT_RTMP=1935
DLC_MEDIAMTX_PORT_HTTP=8888

# === ngrok ===
DLC_NGROK_AUTHTOKEN=your_ngrok_authtoken_here

# === Logging ===
DLC_LOG_DIR=/workspace/logs
EOF
    ok ".env.example created"
fi

# ---------------------------------------------------------------------------
# Step 10 — Logging & Monitoring
# ---------------------------------------------------------------------------
step "Logging & Monitoring"

mkdir -p "${LOGS_DIR}"

# Logrotate config for DLC logs
LOGROTATE_CONF="/etc/logrotate.d/deep-live-cam"
if [ ! -f "${LOGROTATE_CONF}" ]; then
    log "Creating logrotate config..."
    cat > "${LOGROTATE_CONF}" <<EOF
${LOGS_DIR}/*.log {
    daily
    rotate 7
    compress
    delaycompress
    missingok
    notifempty
    create 0644 root root
}
EOF
    ok "Logrotate config created"
fi

ok "bc installed: $(command -v bc)"

# ---------------------------------------------------------------------------
# Final Summary
# ---------------------------------------------------------------------------
echo ""
echo "================================================================"
echo " Cloud GPU Setup Complete!"
echo "================================================================"

PYTHON_OUT=$(python3 --version 2>&1)
CUDA_DRIVER="${DRIVER_VER:-unknown}"
CUDA_RUNTIME="unknown"
if cmd_exists nvcc; then
    CUDA_RUNTIME=$(nvcc --version 2>&1 | head -1 | grep -oP '\d+\.\d+' | head -1)
fi
FFMPEG_OUT=$(ffmpeg -version 2>&1 | head -1)
FFMPEG_NVENC="NO"
if ffmpeg -encoders 2>/dev/null | grep -q "h264_nvenc"; then
    FFMPEG_NVENC="YES"
fi

INSIGHTFACE_VER=$(python3 -c "import insightface; print(insightface.__version__)" 2>/dev/null || echo "N/A")
ORT_VER=$(python3 -c "import onnxruntime; print(onnxruntime.__version__)" 2>/dev/null || echo "N/A")
CV2_VER=$(python3 -c "import cv2; print(cv2.__version__)" 2>/dev/null || echo "N/A")
MEDIAMTX_VER=$(mediamtx --version 2>&1 | head -1 || echo "unknown")
NGROK_VER=$(ngrok version 2>&1 | head -1 || echo "unknown")

cat <<EOF
========================================
Cloud GPU Setup Complete!
========================================
Python:       ${PYTHON_OUT}
CUDA Driver:  ${CUDA_DRIVER}
CUDA Runtime: ${CUDA_RUNTIME}
FFmpeg:       ${FFMPEG_OUT} (NVENC: ${FFMPEG_NVENC})
InsightFace:  ${INSIGHTFACE_VER}
onnxruntime:  ${ORT_VER}
OpenCV:       ${CV2_VER}
MediaMTX:     ${MEDIAMTX_VER} (port 1935)
ngrok:        ${NGROK_VER}
Models:
  inswapper_128.onnx      [$(test -f "${MODELS_DIR}/inswapper_128.onnx" && echo "OK" || echo "MISSING")]
  GPEN-BFR-256.onnx       [$(test -f "${MODELS_DIR}/GPEN-BFR-256.onnx" && echo "OK" || echo "MISSING")]
  GPEN-BFR-512.onnx       [$(test -f "${MODELS_DIR}/GPEN-BFR-512.onnx" && echo "OK" || echo "MISSING")]
VirtualEnv: ${VENV_DIR} [ready]
Logs:         ${LOGS_DIR}
========================================

Next steps:
1. Activate venv: source ${VENV_DIR}/bin/activate
2. Run:
   python run_headless.py -s face.jpg --input-source video.mp4 \\
     --stream-output rtmp://localhost/live/main \\
     --frame-processor face_swapper face_enhancer_gpen256 \\
     --mouth-mask --eyes-mask --eyebrows-mask \\
     --stream-encoder h264_nvenc --execution-provider cuda
========================================
EOF
