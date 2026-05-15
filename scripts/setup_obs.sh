#!/bin/bash
# OBS Virtual Camera Setup Helper for Deep-Live-Cam Headless Streaming
#
# This script helps configure OBS Studio to receive the RTMP stream
# from your cloud GPU and output via the Virtual Camera.
#
# Usage:
#   chmod +x scripts/setup_obs.sh
#   ./scripts/setup_obs.sh

set -e

echo "=========================================="
echo "Deep-Live-Cam OBS Setup Helper"
echo "=========================================="
echo ""

# Detect OS
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    OS="linux"
elif [[ "$OSTYPE" == "darwin"* ]]; then
    OS="macos"
elif [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
    OS="windows"
else
    echo "Unsupported OS: $OSTYPE"
    exit 1
fi

echo "Detected OS: $OS"
echo ""

# Check for OBS
if command -v obs &> /dev/null; then
    OBS_CMD="obs"
elif [ "$OS" == "macos" ] && [ -d "/Applications/OBS.app" ]; then
    OBS_CMD="open -a OBS"
elif [ "$OS" == "windows" ] && [ -f "/c/Program Files/obs-studio/bin/64bit/obs64.exe" ]; then
    OBS_CMD="/c/Program Files/obs-studio/bin/64bit/obs64.exe"
else
    echo "OBS Studio not found. Please install it first:"
    echo "  Linux: sudo apt-get install obs-studio"
    echo "  macOS: brew install --cask obs"
    echo "  Windows: choco install obs-studio"
    exit 1
fi

echo "OBS found: $OBS_CMD"
echo ""

# Function to create OBS scene collection
create_scene_collection() {
    local rtmp_url="$1"
    local collection_name="Deep-Live-Cam-Streaming"

    echo "Creating OBS scene collection: $collection_name"
    echo "RTMP Source: $rtmp_url"
    echo ""

    case "$OS" in
        linux)
            CONFIG_DIR="$HOME/.config/obs-studio"
            ;;
        macos)
            CONFIG_DIR="$HOME/Library/Application Support/obs-studio"
            ;;
        windows)
            CONFIG_DIR="$APPDATA/obs-studio"
            ;;
    esac

    mkdir -p "$CONFIG_DIR/basic/scenes"

    cat > "$CONFIG_DIR/basic/scenes/$collection_name.json" <<EOF
{
    "current_scene": "Deep Live Cam",
    "current_program_scene": "Deep Live Cam",
    "scene_order": ["Deep Live Cam"],
    "sources": [
        {
            "id": "ffmpeg_source",
            "name": "DLC RTMP Stream",
            "settings": {
                "input": "$rtmp_url",
                "input_format": "flv",
                "is_local_file": false,
                "buffering_mb": 1,
                "reconnect_delay_sec": 1,
                "restart_on_activate": true
            },
            "visible": true
        }
    ],
    "transition": "Cut",
    "transitions": []
}
EOF

    echo "Scene collection created at:"
    echo "  $CONFIG_DIR/basic/scenes/$collection_name.json"
    echo ""
    echo "To use it:"
    echo "  1. Open OBS Studio"
    echo "  2. Go to Scene Collections menu"
    echo "  3. Select '$collection_name'"
    echo ""
}

# Main menu
echo "What would you like to do?"
echo ""
echo "1) Full setup (create scene + configure)"
echo "2) Just create OBS scene collection file"
echo "3) Show manual configuration steps"
echo "4) Check system requirements"
echo ""
read -p "Enter choice [1-4]: " choice

case "$choice" in
    1|2)
        echo ""
        read -p "Enter your RTMP stream URL [rtmp://localhost/live/stream]: " rtmp_url
        rtmp_url=${rtmp_url:-"rtmp://localhost/live/stream"}
        create_scene_collection "$rtmp_url"

        if [ "$choice" == "1" ]; then
            echo "=========================================="
            echo "Next steps:"
            echo "=========================================="
            echo ""
            echo "1. Start your RTMP server (e.g., nginx-rtmp, srs, or mediaMTX)"
            echo ""
            echo "2. Start Deep-Live-Cam on cloud GPU:"
            echo "   python run_headless.py -s face.jpg \\"
            echo "     --input-source video.mp4 \\"
            echo "     --stream-output $rtmp_url \\"
            echo "     --execution-provider cuda"
            echo ""
            echo "3. Open OBS Studio and select the '$collection_name' scene"
            echo ""
            echo "4. Click 'Start Virtual Camera' in OBS"
            echo ""
            echo "5. In WhatsApp/Telegram, select 'OBS Virtual Camera' as camera"
            echo ""
        fi
        ;;

    3)
        echo ""
        echo "=========================================="
        echo "Manual Configuration Steps"
        echo "=========================================="
        echo ""
        echo "Step 1: Start RTMP Server"
        echo "  Example using MediaMTX (simplest):"
        echo "    docker run --rm -it -e MTX_PROTOCOLS=tcp \"
        echo "      -p 8554:8554 -p 1935:1935 -p 8888:8888 \"
        echo "      bluenviron/mediamtx:latest"
        echo ""
        echo "Step 2: Configure OBS Source"
        echo "  1. Open OBS Studio"
        echo "  2. In Sources panel, click +"
        echo "  3. Add 'VLC Video Source' or 'Media Source'"
        echo "  4. Enter URL: rtmp://your-server/live/stream"
        echo "  5. Set buffering to low (1-2 MB)"
        echo ""
        echo "Step 3: Enable Virtual Camera"
        echo "  1. In OBS, click 'Start Virtual Camera'"
        echo "  2. Virtual Camera appears as a webcam in other apps"
        echo ""
        echo "Step 4: Use in WhatsApp/Telegram"
        echo "  1. Start a video call"
        echo "  2. Click camera settings"
        echo "  3. Select 'OBS Virtual Camera'"
        echo ""
        echo "Low Latency Tips:"
        echo "  - Use h264_nvenc encoder on cloud GPU"
        echo "  - Reduce buffering in OBS source"
        echo "  - Use local RTMP relay if server is remote"
        echo ""
        ;;

    4)
        echo ""
        echo "=========================================="
        echo "System Requirements Check"
        echo "=========================================="
        echo ""

        echo "--- Python ---"
        python3 --version 2>/dev/null || echo "Python 3 not found"
        echo ""

        echo "--- FFmpeg ---"
        ffmpeg -version 2>/dev/null | head -1 || echo "FFmpeg not found"
        echo ""

        echo "--- OBS Studio ---"
        $OBS_CMD --version 2>/dev/null || echo "OBS not found"
        echo ""

        echo "--- GPU (CUDA) ---"
        if command -v nvidia-smi &> /dev/null; then
            nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
        else
            echo "nvidia-smi not found (no NVIDIA GPU or drivers not installed)"
        fi
        echo ""

        echo "--- Network ---"
        echo "Local IP: $(hostname -I 2>/dev/null | awk '{print $1}')"
        echo ""

        echo "--- Required Python Packages ---"
        pip list 2>/dev/null | grep -E "onnxruntime|opencv|torch|insightface|numpy" || echo "Run: pip install -r requirements.txt"
        echo ""
        ;;

    *)
        echo "Invalid choice"
        exit 1
        ;;
esac

echo "Done!"
