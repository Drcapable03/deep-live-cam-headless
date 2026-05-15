# Deep-Live-Cam Headless Streaming 2.7+

This is an enhanced version of Deep-Live-Cam with headless streaming capabilities, featuring the new 2.7+ face masking system for natural-looking real-time face swaps.

---

## What's New in This Version

### 2.7+ Features (NEW)

| Feature | Description | CLI Flag |
|---------|-------------|----------|
| **Mouth Mask** | Preserves original lip movement while swapping face | `--mouth-mask` |
| **Eyes Mask** | Preserves natural eye blinks and eye movement | `--eyes-mask` |
| **Eyebrows Mask** | Preserves eyebrow expressions (raised, furrowed, etc.) | `--eyebrows-mask` |
| **GPEN 512 Enhancer** | Ultra-sharp face restoration at 512x512 | `--frame-processor face_enhancer_gpen512` |
| **GPEN 256 Enhancer** | High-quality face restoration at 256x256 | `--frame-processor face_enhancer_gpen256` |
| **Color Transfer** | Automatic skin tone matching between source and target | (auto) |
| **Feathered Blending** | Seamless edge blending with configurable feather | `--mask-feather-ratio` |

### Headless Streaming Features

| Feature | Description |
|---------|-------------|
| No GUI required | Runs on cloud servers without display |
| Multiple inputs | Camera, video file, RTMP, named pipe, stdin |
| RTMP streaming | Direct output to RTMP servers via FFmpeg |
| Hardware encoding | NVENC/AMF support for GPU-accelerated streaming |
| Cloud-optimized | Tested on RTX 4090 with CUDA 13 |

---

## Quick Start

### Prerequisites

```bash
# Install dependencies
pip install -r requirements.txt

# Ensure ffmpeg is installed
ffmpeg -version

# Set CUDA library path (for CUDA 13 compatibility)
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:/usr/local/cuda-13.0/targets/x86_64-linux/lib:$LD_LIBRARY_PATH
```

### Start MediaMTX (RTMP Server)

```bash
cd /workspace
wget https://github.com/bluenviron/mediamtx/releases/download/v1.12.0/mediamtx_v1.12.0_linux_amd64.tar.gz
tar xzf mediamtx_v1.12.0_linux_amd64.tar.gz
./mediamtx &
```

### Basic Face Swap (Video File to RTMP)

```bash
python run_headless.py \
  -s face.jpg \
  --input-source test.mp4 \
  --stream-output rtmp://localhost/live/swapped \
  --stream-encoder libx264 \
  --execution-provider cuda
```

### With Face Enhancement (Better Quality)

```bash
python run_headless.py \
  -s face.jpg \
  --input-source test.mp4 \
  --stream-output rtmp://localhost/live/swapped \
  --frame-processor face_swapper face_enhancer_gpen512 \
  --stream-encoder libx264 \
  --execution-provider cuda
```

### With Natural Feature Preservation (Best Results)

```bash
python run_headless.py \
  -s face.jpg \
  --input-source test.mp4 \
  --stream-output rtmp://localhost/live/swapped \
  --frame-processor face_swapper face_enhancer_gpen512 \
  --mouth-mask \
  --eyes-mask \
  --eyebrows-mask \
  --stream-encoder libx264 \
  --execution-provider cuda
```

### Maximum Quality (1080p + All Features)

```bash
python run_headless.py \
  -s face.jpg \
  --input-source test.mp4 \
  --stream-output rtmp://localhost/live/swapped \
  --frame-processor face_swapper face_enhancer_gpen512 \
  --mouth-mask \
  --eyes-mask \
  --eyebrows-mask \
  --stream-width 1920 \
  --stream-height 1080 \
  --stream-fps 30 \
  --stream-encoder h264_nvenc \
  --stream-quality 20 \
  --execution-provider cuda
```

---

## Understanding the Masks

The masks are what make the face swap look natural and "alive":

### Mouth Mask (`--mouth-mask`)
- **What it does**: Preserves the original person's lip movement
- **Why it matters**: Without this, the swapped face has a "frozen" mouth that doesn't move when talking
- **Best for**: Video calls where you want natural speech movement

### Eyes Mask (`--eyes-mask`)
- **What it does**: Preserves natural eye blinks and eye movement
- **Why it matters**: Without this, the eyes look static and unnatural
- **Best for**: Any scenario where eye contact matters

### Eyebrows Mask (`--eyebrows-mask`)
- **What it does**: Preserves eyebrow expressions (surprise, anger, etc.)
- **Why it matters**: Eyebrows convey emotion - without them, the face looks expressionless
- **Best for**: Natural emotional expression

### Using All Three Together
The trending videos you see online use **all three masks simultaneously**. This creates a result where:
- The **face identity** is swapped (who you look like)
- The **expressions** are preserved (how you naturally move)
- The **result** looks like the source person is actually there, mimicking the target

---

## CLI Arguments Reference

### Input/Output

| Argument | Description | Example |
|----------|-------------|---------|
| `-s, --source` | Source face image path | `face.jpg` |
| `--input-source` | Input source (camera/file/pipe/URL) | `0`, `video.mp4`, `rtmp://in` |
| `--stream-output` | Output stream URL | `rtmp://out` |

### Stream Configuration

| Argument | Default | Description |
|----------|---------|-------------|
| `--stream-width` | 1280 | Output width in pixels |
| `--stream-height` | 720 | Output height in pixels |
| `--stream-fps` | 30 | Output frame rate |
| `--stream-quality` | 23 | Quality (CRF for x264/x265, CQ for NVENC) |
| `--stream-encoder` | libx264 | Encoder: `libx264`, `libx265`, `h264_nvenc`, `hevc_nvenc` |

### Face Processing

| Argument | Description |
|----------|-------------|
| `--frame-processor` | Pipeline: `face_swapper`, `face_enhancer_gpen256`, `face_enhancer_gpen512` |
| `--many-faces` | Swap all detected faces |
| `--mouth-mask` | Preserve original lip movement |
| `--eyes-mask` | Preserve original eye movement/blinks |
| `--eyebrows-mask` | Preserve original eyebrow expressions |
| `--mouth-mask-size` | Mouth mask intensity 0-100 (default: 0) |
| `--eyes-mask-size` | Eyes mask intensity 0-100 (default: 0) |
| `--eyebrows-mask-size` | Eyebrows mask intensity 0-100 (default: 0) |
| `--mask-feather-ratio` | Edge feathering 1-100 (default: 12, lower = softer edges) |

### GPU/Performance

| Argument | Description |
|----------|-------------|
| `--execution-provider` | `cuda`, `rocm`, `coreml`, `dml`, `cpu` |
| `--max-memory` | RAM limit in GB |
| `--live-mirror` | Mirror the input horizontally |

---

## ngrok Setup for External Access

```bash
# Install ngrok
wget https://bin.equinox.io/c/bNyj1mQVY4c/ngrok-v3-stable-linux-amd64.tgz
tar xzf ngrok-v3-stable-linux-amd64.tgz
./ngrok config add-authtoken YOUR_TOKEN_HERE

# Start tunnel for MediaMTX
./ngrok tcp 1935 &

# Get the public URL
curl -s http://localhost:4040/api/tunnels | grep -o "tcp://[^\"]*"
# Returns: tcp://2.tcp.ngrok.io:12345

# In OBS, use:
# rtmp://2.tcp.ngrok.io:12345/live/swapped
```

---

## OBS Setup on Windows

1. **Add Media Source**
   - Sources → + → Media Source
   - Uncheck "Local File"
   - Input: `rtmp://YOUR_NGROK_URL/live/swapped`
   - Network Buffering: `200`
   - Reconnect Delay: `1`

2. **Start Virtual Camera**
   - Click "Start Virtual Camera" in OBS

3. **In WhatsApp/Telegram**
   - Start video call
   - Select "OBS Virtual Camera" as camera

---

## File Modifications Summary

| File | Action | Purpose |
|------|--------|---------|
| `modules/core.py` | Modified | New CLI args for masks, headless routing |
| `modules/globals.py` | Modified | New mask settings + streaming config |
| `modules/streaming.py` | Modified | FFmpeg output with deadlock fix |
| `modules/headless_live.py` | Modified | Integrated face masking pipeline |
| `modules/video_capture.py` | Modified | File/pipe/FFmpeg input support |
| `modules/processors/frame/_onnx_enhancer.py` | **NEW** | Base ONNX utilities for GPEN |
| `modules/processors/frame/face_enhancer_gpen256.py` | **NEW** | GPEN 256 enhancer |
| `modules/processors/frame/face_enhancer_gpen512.py` | **NEW** | GPEN 512 enhancer |
| `modules/processors/frame/face_masking.py` | **NEW** | Mouth/eyes/eyebrows masking |
| `run.py` | Modified | Conditional tkinter import |
| `run_headless.py` | **NEW** | Dedicated headless entry point |

---

## Troubleshooting

### FFmpeg "Output pipe broken"
- MediaMTX is not running → Restart MediaMTX
- Wrong RTMP URL → Check URL format: `rtmp://localhost/live/swapped`

### CUDA not found
```bash
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:/usr/local/cuda-13.0/targets/x86_64-linux/lib:$LD_LIBRARY_PATH
```

### ngrok connection issues
- Check authtoken: `cat ~/.config/ngrok/ngrok.yml`
- Free tier disconnects after ~2 hours
- Use `./ngrok tcp 1935 &` to run in background

### Black screen in OBS
- Right-click source → Refresh
- Delete and re-add the source
- Check ngrok URL hasn't changed (changes on restart)

---

## License

Same as the original Deep-Live-Cam project. See LICENSE file.
