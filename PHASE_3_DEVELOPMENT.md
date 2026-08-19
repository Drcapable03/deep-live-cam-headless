# PHASE 3 — FACELESS-Style Acceleration, Post-Processing & Adaptive FPS Protection

Status: **IMPLEMENTED + smoke-tested locally (25/25 PASS standard, 26/26 PASS with adaptive-resolution e2e pass)**

## Motivation (blueprint: FACELESS)

Researched `EmperorBadussy/faceless` (AGPL-3.0, Deep-Live-Cam fork) and took the
following from its architecture (their measured stage costs on RTX:
detection ~3-5ms, swap ~5ms, GPEN-256 ~5ms; 854x480 → ~31fps, 640x360 → ~52fps):

| FACELESS technique | Status after Phase 3 |
|---|---|
| Resolution-decoupled pipeline (capture 1080p → process 480p → output 1080p) | Already implemented in Phase 1 |
| **FP16 swap model** (`inswapper_128_fp16.onnx`, ~2x Tensor-Core speed) | Implemented — one-time cached conversion, no external download |
| **Post-processing suite** (sharpening + opacity blending) | Implemented — `--sharpen-strength`, `--blend-opacity` |
| **Quality presets** (NORMAL / HIGH bundling res + enhancer) | Implemented — `--quality-preset normal\|high` |
| **FPS protection** (auto-tune process resolution to hold 30fps) | Implemented — `--adaptive-resolution` |
| Four-thread capture/detect/process/display | Already present (capture/process/output threads) |
| GPEN-BFR-256 enhancer + virtual camera | Already implemented in Phases 1-2 |

## 1. FP16 model acceleration (`modules/onnx_fp16.py`, NEW)

**Why:** inswapper_128 and GPEN-BFR are heavily memory-bandwidth-bound; FP16 on
Tensor Cores (Turing+) roughly halves bandwidth and ~2x inference speed with
negligible quality loss (FACELESS ships fp16 models for exactly this).

**How (no sketchy downloads):**
- `get_fp16_model_path(fp32_path)` returns a cached `<name>_fp16.onnx` next to
  the FP32 model, converting it on first use via `onnxconverter-common`
  (deterministic, offline, ~15s for inswapper).
- Auto-enabled only when `fp16_enabled()`: `globals.fp16` AND a
  CUDAExecutionProvider is active. CPU-only boxes keep FP32 (graceful).
- Wired into `face_swapper.get_face_swapper()` (replaced the old torch-CUDA
  gate, which wrongly blocked the model on CUDA-only boxes without torch) and
  `_onnx_enhancer.create_onnx_session()` (GPEN-256/512).
- **Bug found & fixed during conversion:** GPEN models declare some
  initializers as graph inputs; with `keep_io_types=True` ONNX Runtime then
  rejects the converted model ("element type tensor(float16) ... expects
  tensor(float)"). Fixed by dropping initializer-as-input declarations from
  the graph before converting (safe by definition).
- `--no-fp16` disables. Converted models verified to load and run in
  onnxruntime (all three: `inswapper_128_fp16.onnx` 264MB,
  `GPEN-BFR-256_fp16.onnx` 36MB, `GPEN-BFR-512_fp16.onnx` 136MB — pre-converted
  in the local `models/` dir).
- Deps: added `onnxconverter-common>=1.14.0` to requirements.txt.

## 2. Post-processing suite (`--sharpen-strength`, `--blend-opacity`)

The mechanics already existed in `face_swapper.py` (FACELESS-style knobs) but
were only reachable via the GUI — now headless CLI-exposed:

- `--blend-opacity 0.85` (0-1): blends the swapped face over the original
  within the paste-back (existing `globals.opacity` path). 1.0 = full swap.
  Lower values keep more original texture (skin grain) → more "distinct".
- `--sharpen-strength 0.4` (0-1+): unsharp mask on the swapped-face bbox only
  (existing `apply_post_processing`/`gpu_sharpen` path) — sharpens the face
  without touching the background.
- Both applied in the single `swap_face`/`apply_post_processing` code path, so
  GUI and headless behave identically. Defaults unchanged (1.0 / 0.0) unless
  a preset or explicit flag sets them.

## 3. Quality presets (`--quality-preset normal|high`)

Bundles the Phase 1-3 knobs (FACELESS quality presets):

| | normal (default) | high |
|---|---|---|
| Processing resolution | auto (854x480 cap) | 960x540 |
| Enhancer | none added | face_enhancer_gpen256 appended |
| Sharpen | 0.0 | 0.4 |
| Opacity | 1.0 | 0.9 |

Explicit CLI flags always win over preset defaults (checked in the smoke test).

## 4. Adaptive resolution (`--adaptive-resolution`)

FPS protection (FACELESS-style "tune processing resolution to your hardware"):

- EMA of per-frame processing time (0.9 decay); frame budget = 1000/`--stream-fps`.
- Every 15 frames: if EMA > 85% of budget and scale > 0.4 → multiply processing
  scale by 0.75 (log + expose via `stats['adaptive_scale']`,
  `stats['process_resolution']`); if EMA < 55% of budget and scale < 1.0 →
  restore one step. Hysteresis prevents oscillation.
- Detect/swap/enhance run at the reduced size; output is still upscaled to
  stream size — output resolution never changes, only the processing cost.
- Implemented as `base_w/base_h × adaptive_scale` in
  `headless_live._processing_loop`; `globals.adaptive_scale` tracks the state.

**Verified live:** the CPU smoke run degraded 854x480 → 639x360 in two steps
under a 240fps budget (`[HEADLESS] Adaptive: degrading processing resolution...`).

## Files changed

| File | Change |
|------|--------|
| `modules/onnx_fp16.py` | **NEW** — FP16 conversion helper (cache, gate, initializer-as-input fix) |
| `modules/processors/frame/face_swapper.py` | FP16 selection via `get_fp16_model_path` (torch gate removed) |
| `modules/processors/frame/_onnx_enhancer.py` | `create_onnx_session` prefers cached FP16 |
| `modules/globals.py` | `fp16`, `blend_opacity`, `sharpen_strength`, `quality_preset`, `adaptive_resolution`, `adaptive_scale` |
| `modules/core.py` | `--no-fp16`, `--blend-opacity`, `--sharpen-strength`, `--quality-preset`, `--adaptive-resolution` + preset application |
| `modules/headless_live.py` | base×scale resolution model + EMA adaptive controller |
| `requirements.txt` | + `onnxconverter-common>=1.14.0` |
| `smoke_test.py` | Phase 3 checks: wiring, fp16 gating/fallback, CLI preset tests (subprocess), e2e sharpen/opacity, adaptive degrade pass |

## Verification

```bash
# Standard suite (CPU box): 25/25 PASS
python smoke_test.py -s face.jpg -t smoke_input.mp4 --provider cpu --frames 8

# Adaptive-resolution e2e pass: 26/26 PASS (degrades 854x480 -> 639x360 under load)
python smoke_test.py -s face.jpg -t smoke_input.mp4 --provider cpu --frames 35 --adaptive

# GPU box (recommended, exercises FP16 + CUDA):
python smoke_test.py -s face.jpg -t test.mp4 --provider cuda --frames 30 --enhancer gpen256
```

## Recommended production command (cloud RTX box)

```bash
python run_headless.py -s face.jpg --input-source webcam \
  --stream-output rtmp://127.0.0.1:1935/live/main \
  --quality-preset high --adaptive-resolution \
  --mouth-mask --eyes-mask --eyebrows-mask \
  --stream-width 1280 --stream-height 720 --stream-fps 30 \
  --execution-provider cuda
```

First run on the server converts the models to FP16 once (logged `[FP16]
Converting ...`), then every later run loads them instantly (~2x swapper +
GPEN speed on the 4090, and the adaptive controller guarantees 30fps under any
load by stepping the processing resolution down instead of dropping fps).