# PHASE 2 — Local Low-Latency Output + Audio Passthrough

Status: **IMPLEMENTED + smoke-tested locally (14/14 PASS, incl. real OBS virtual cam)**

## Goal

Phase 1 removed the pipeline's correctness/speed blockers for remote streaming.
Phase 2 targets the **lowest-latency path for the common case**: the machine doing
the swap is the same machine that holds the webcam/mic and the OBS virtual camera.

Two additions:

1. **`VirtualCamOutput`** — push swapped frames straight into the OBS Virtual
   Camera driver (`pyvirtualcam`). No OBS scene, no RTMP, no ngrok: WhatsApp /
   Telegram / Zoom / Meet simply see the feed as a webcam. This is the lowest
   end-to-end latency path (single machine, in-process, zero network hops).
2. **Audio passthrough** — mux the original media file's audio into stream/file
   output (RTMP, UDP, MP4) via FFmpeg. For local file-to-file/live pipelines the
   audio track is preserved instead of producing a silent output.

## Design

### `modules/streaming.py`

- New class `VirtualCamOutput(StreamOutput)`:
  - `start()` imports `pyvirtualcam`, opens `pyvirtualcam.Camera(width, height,
    fps, fmt=PixelFormat.RGB, device=camera_name)`.
  - Graceful degradation: missing `pyvirtualcam` or missing driver (OBS /
    v4l2loopback) prints an actionable hint and returns `False` — the caller
    keeps running without output instead of crashing.
  - `write()` converts BGR→RGB (`frame[:, :, ::-1]`) for pyvirtualcam and paces
    via `cam.sleep_until_next_frame()` so the consumer sees the requested fps.
  - `close()` releases the camera cleanly.
- `FFmpegStreamOutput` gained `audio_source: Optional[str]`:
  - When set, the FFmpeg command adds a second input (`-i audio_source`) and maps
    `-map 0:v:0 -map 1:a:0?` with `-c:a aac -b:a 128k` (safe for both FLV/RTMP
    and MP4 containers).
  - `-map 1:a:0?` (trailing `?`) keeps the job alive when the audio file has no
    audio stream.
  - **No `-shortest`**: empirically, with video arriving over the raw stdin pipe
    `-shortest` makes the MP4 muxer DROP the audio stream whenever video ends
    first (e.g. early shutdown). Without it, an early stop only leaves a short
    audio tail, and normal operation muxes both streams fully.
- `create_output()` now routes these URL forms:
  - `virtualcam` / `virtualcam:NAME` / `pyvirtualcam` / `vcam` → `VirtualCamOutput`
    (`:NAME` selects a non-default device)
  - `pipe:path` → `PipeOutput`
  - everything else (`rtmp://`, `udp://`, `srt://`, `/path/file.mp4`) →
    `FFmpegStreamOutput` (audio_source forwarded)

### `modules/globals.py` / `modules/core.py`

New globals + CLI flags:

- `--virtual-cam-name NAME` → `globals.virtual_cam_name`
- `--stream-audio-source PATH` → `globals.stream_audio_source`

### `modules/headless_live.py`

`start()` auto-wires audio for local video inputs:

- If no explicit `--stream-audio-source`, and the input is a local media file
  (`file:`-prefixed or a path ending in a known video extension), that file is
  used as the audio source.
- Virtual-cam outputs are skipped (the OS mic provides audio for those).
- `create_output(...)` now receives `audio_source`.

## CLI Examples

```bash
# Local zero-latency virtual camera (same machine as webcam/mic)
python run_headless.py -s face.jpg --input webcam --output virtualcam \
    --processors face_swapper --execution-provider cuda \
    --stream-width 1280 --stream-height 720 --stream-fps 30

# RTMP stream to MediaMTX with the input file's audio preserved
python run_headless.py -s face.jpg --input webcam \
    --output rtmp://127.0.0.1:1935/live/main \
    --stream-audio-source /path/to/music.mp4 \
    --processors face_swapper face_enhancer_gpen256 --execution-provider cuda
```

## Verification (local smoke test)

`smoke_test.py` (new) asserts all Phase 1 + Phase 2 invariants and runs an
end-to-end pipeline:

```bash
# CPU (any machine): 14/14 PASS
python smoke_test.py -s face.jpg -t smoke_input.mp4 --provider cpu --frames 8

# With a real audio track
ffmpeg -y -i test3.mp4 -t 3 -c:v libx264 -preset ultrafast -c:a aac -b:a 128k smoke_input_audio.mp4
python smoke_test.py -s face.jpg -t smoke_input_audio.mp4 --provider cpu --frames 6

# GPU box
python smoke_test.py -s face.jpg -t test.mp4 --provider cuda --frames 30 --enhancer gpen256
```

Verified locally (Windows, Python 3.14 venv, CPU provider):

- All Phase 1 checks: mask sizes stable, landmarks present with masks on,
  recognition skipped in lite mode, resolution decoupling active.
- **Real OBS Virtual Camera started** (`Virtual camera started: OBS Virtual
  Camera 1280x720@30fps`) — pyvirtualcam 0.15.0.
- Audio passthrough: silent input → silent output; audio input → audio output
  (aac, 48 kHz stereo) confirmed by ffmpeg probe.
- Graceful fallback confirmed when `pyvirtualcam` is absent (returns False,
  pipeline continues).

## Latency note

For the WhatsApp/Telegram target: when the swap runs on the same machine as the
webcam, `--output virtualcam` skips the OBS scene + RTMP + ngrok + remote GPU
hop entirely. When the swap must run on a remote GPU box (cloud), the OBS/ngrok
RTMP path from Phase 1 remains the fallback — the virtual cam is a local
option, and audio always travels over the real call (mic) rather than the
stream.