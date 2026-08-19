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

### Phase 2 Local Low-Latency Output (NEW)

| Feature | Description | CLI Flag |
|---------|-------------|----------|
| **Virtual camera output** | Push swapped frames directly into the OBS Virtual Camera (`pyvirtualcam`) — WhatsApp/Telegram/Zoom see it as a webcam. No OBS scene, no RTMP, no ngrok. Lowest-latency path when swap + webcam are on the same machine | `--stream-output virtualcam` (or `virtualcam:NAME`) |
| **Audio passthrough** | Mux the input file's audio into stream/file output (AAC 128k) — no more silent recordings/streams | `--stream-audio-source PATH` (auto-detected for local video inputs) |

### Phase 3 FACELESS-Style Acceleration & Quality (NEW)

| Feature | Description | CLI Flag |
|---------|-------------|----------|
| **FP16 acceleration** | Auto-converts inswapper + GPEN models to FP16 (one-time, cached in `models/`) for ~2x Tensor-Core speed on CUDA; gracefully stays FP32 on CPU boxes | (auto on CUDA) / `--no-fp16` |
| **Quality presets** | `high` = 960x540 processing + GPEN-256 enhancer + opacity 0.9 + sharpen 0.4; `normal` = current defaults. Explicit flags always win | `--quality-preset normal\|high` |
| **Sharpening** | Unsharp mask on the swapped face bbox only (background untouched) — fixes "soft/blurry" swapped faces | `--sharpen-strength 0.4` |
| **Opacity blending** | Blend the swapped face over the original (1.0 = full swap; ~0.9 keeps original skin texture = more distinct) | `--blend-opacity 0.9` |
| **Adaptive resolution** | FPS protection: automatically steps processing resolution down (854x480 → 640x360 → ...) to hold the `--stream-fps` budget under load, then restores it — output resolution never changes | `--adaptive-resolution` |

### Phase 1 Performance & Correctness Upgrades

| Feature | Description | CLI Flag |
|---------|-------------|----------|
| **Resolution decoupling** | Detect/swap/enhance at reduced resolution (auto-capped 854x480), upscale to stream size after — ~3x cheaper than processing at 1080p | `--process-width`, `--process-height` (0 = auto) |
| **Per-frame detection** | Face detection runs every frame at processing resolution (~3-6ms on GPU) — swap bbox no longer lags head motion | (auto) |
| **Mask bug fixes** | Mask sizes are now static (were compounding +30/+15/+25 per frame until 100, ballooning the masks) and masks are applied exactly once | (auto) |
| **Landmark-aware detection** | Masks/enhancers now receive faces with 106-point landmarks (previously they silently did nothing) | (auto) |
| **GPEN temporal caching** | GPEN-256/512 inference runs every 2 frames in live mode, reusing the cached enhanced face in between (~50% enhancer cost cut) | (auto) |
| **Single mask source of truth** | All feature-preservation masks moved into `face_masking.py`; the duplicate legacy implementations were removed | (auto) |

---

## Quick Start

### Prerequisites

```bash
# Install dependencies
pip install -r requirements.txt

# Optional but recommended on the GPU box (FP16 auto-conversion needs it):
pip install onnxconverter-common

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

> **Phase 1 tip (live calls):** use `face_enhancer_gpen256` instead of `gpen512` for the live stream (temporal caching + 256px keeps 30fps headroom), and reserve `gpen512` for recorded/batch output. Resolution decoupling (auto 854x480 processing) is on by default — tune with `--process-width 1280 --process-height 720`, or pass your input resolution (e.g. `--process-width 1920 --process-height 1080`) to process at native size.

### Local Virtual Camera (No Streaming Server Needed)

When the swap runs on the **same machine** as your webcam/mic, this is the
lowest-latency path — no OBS scene, no RTMP, no ngrok:

```bash
# Windows: requires OBS Studio (provides the Virtual Camera driver) + pyvirtualcam
pip install pyvirtualcam
python run_headless.py -s face.jpg --input-source webcam \
  --stream-output virtualcam \
  --frame-processor face_swapper face_enhancer_gpen256 \
  --mouth-mask --eyes-mask --eyebrows-mask \
  --stream-width 1280 --stream-height 720 --stream-fps 30 \
  --execution-provider cuda
```

- In WhatsApp/Telegram/Zoom select **"OBS Virtual Camera"** as the camera.
- Audio always travels over the real call (mic) — the virtual cam is video-only.
- Fallback: `--stream-output rtmp://...` remote streaming (ngrok/OBS path above)
  still works when the swap runs on a separate GPU box.

> **Phase 3 recommended for calls:** add `--quality-preset high --adaptive-resolution`
> to the virtual-cam command (or `--sharpen-strength 0.4 --blend-opacity 0.9` with
> `--frame-processor face_swapper face_enhancer_gpen256`) — sharper, more
> distinct face, and adaptive resolution guarantees 30fps even when the box
> struggles. FP16 kicks in automatically on CUDA boxes (first run converts the
> models once, ~15s).

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
| `--process-width` | 0 (auto) | Processing resolution width for resolution decoupling; 0 = auto (capped at 854x480) |
| `--process-height` | 0 (auto) | Processing resolution height for resolution decoupling; 0 = auto (capped at 854x480) |
| `--virtual-cam-name` | None | Virtual camera device name for `--stream-output virtualcam` (optional) |
| `--stream-audio-source` | None | Audio file to mux into stream/file output (auto-detected for local video inputs) |
| `--quality-preset` | normal | Preset bundle: `normal` (defaults) or `high` (960x540 + GPEN-256 + opacity 0.9 + sharpen 0.4). Explicit flags override |
| `--adaptive-resolution` | off | Auto-step processing resolution down (min 0.4x) to hold `--stream-fps`; restores when load drops. Output resolution unchanged |

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
| `--no-fp16` | Disable FP16 model auto-conversion (off by default; auto-enabled only with CUDA) |
| `--blend-opacity` | Swapped-face blend opacity 0-1 (1.0 = full swap; ~0.85-0.9 = keeps original skin texture) |
| `--sharpen-strength` | Unsharp strength on the swapped face bbox (0.0 = off, 0.4 = recommended) |

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
| `modules/processors/frame/face_masking.py` | **NEW** | Mouth/eyes/eyebrows masking (now the single source of truth for all mask + color-transfer logic) |
| `run.py` | Modified | Conditional tkinter import |
| `run_headless.py` | **NEW** | Dedicated headless entry point |

### Phase 1 Refactor Summary

| File | Change |
|------|--------|
| `modules/headless_live.py` | Removed in-loop mask code (was double-applying masks AND compounding global mask sizes every frame); switched to per-frame landmark-aware detection; added resolution decoupling (downscale → process → upscale) |
| `modules/processors/frame/face_swapper.py` | `swap_face` now applies mouth/eyes/eyebrows masks itself via `face_masking.py` (single code path for GUI + headless); removed ~460 lines of duplicate legacy mask/color helpers |
| `modules/processors/frame/face_masking.py` | Added `draw_mouth_mask_visualization` (moved here); is now the sole home of all masking/color code |
| `modules/face_analyser.py` | `_needs_landmark()` now covers eyes/eyebrows masks + size sliders; added `get_one_face_lite` / `get_many_faces_lite` (detection + landmarks, no recognition) |
| `modules/processors/frame/_onnx_enhancer.py` | `enhance_face_onnx(..., temporal=True)` caches the enhanced face + inverse affine, reusing it every other frame (GPEN live-mode cost ~halved) |
| `modules/processors/frame/face_enhancer_gpen256.py` / `gpen512.py` | Wired the temporal cache flag through `process_frame` (auto-enabled when faces are pre-detected in single-face live mode) |
| `modules/ui.py` | GUI detection now uses the same landmark-aware lite detection so masks work in the GUI too |
| `modules/globals.py`, `modules/core.py` | New `--process-width` / `--process-height` CLI flags + globals |

See `PHASE_1_DEVELOPMENT.md` for the full engineering write-up.

### Phase 2 Additions Summary

| File | Change |
|------|--------|
| `modules/streaming.py` | New `VirtualCamOutput` (pyvirtualcam, BGR→RGB, fps pacing, graceful fallback); `FFmpegStreamOutput` gains `audio_source` (2nd FFmpeg input, `-map 1:a:0? -c:a aac`); `create_output` routes `virtualcam`/`pyvirtualcam`/`vcam:` URLs |
| `modules/globals.py`, `modules/core.py` | New `--virtual-cam-name` / `--stream-audio-source` CLI flags + globals |
| `modules/headless_live.py` | Auto-wires audio from local video inputs; forwards `audio_source` to `create_output` |
| `smoke_test.py` | **NEW** — end-to-end smoke test asserting all Phase 1 + Phase 2 invariants (14/14 PASS locally) |

See `PHASE_2_DEVELOPMENT.md` for the full engineering write-up.

### Phase 3 Additions Summary

| File | Change |
|------|--------|
| `modules/onnx_fp16.py` | **NEW** — FP16 auto-conversion helper (cached `<model>_fp16.onnx`, CUDA-only gate, initializer-as-input fix for GPEN models) |
| `modules/processors/frame/face_swapper.py` | Uses `get_fp16_model_path` (replaced the torch-CUDA gate that wrongly blocked CUDA-only boxes) |
| `modules/processors/frame/_onnx_enhancer.py` | `create_onnx_session` prefers cached FP16 GPEN models |
| `modules/globals.py`, `modules/core.py` | `--no-fp16`, `--blend-opacity`, `--sharpen-strength`, `--quality-preset`, `--adaptive-resolution` + preset application (explicit flags win) |
| `modules/headless_live.py` | Adaptive resolution controller (base×scale, EMA frame-time, hysteresis degrade/restore, `stats['adaptive_scale']`) |
| `requirements.txt` | + `onnxconverter-common>=1.14.0` (FP16 conversion) |
| `smoke_test.py` | Phase 3 checks: fp16 gating/fallback, CLI preset precedence (subprocess), e2e sharpen/opacity, adaptive-degrade pass (25/25 and 26/26 PASS locally) |

See `PHASE_3_DEVELOPMENT.md` for the full engineering write-up.

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
