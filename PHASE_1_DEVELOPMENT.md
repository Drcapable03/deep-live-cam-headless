# Phase 1 — Live Pipeline Refactor: Engineering Write-Up

Date: 2026-08-19
Scope: `deep-live-cam-headless` (fork of hacksider/Deep-Live-Cam 2.1.3 + hand-ported 2.7+ features)
Goal: make the headless streaming pipeline **correct**, then **fast**, so a "total, complete real-time face swap" is achievable over the existing cloud GPU → RTMP → ngrok → OBS → WhatsApp/Telegram chain.

Borrowed blueprints: the resolution-decoupled pipeline (capture native → process ~480p → upscale) from the FACELESS project (EmperorBadussy), and the GPEN temporal-inference cache pattern from this repo's own GFPGAN enhancer (`face_enhancer.py`).

---

## 1. Bugs Found During Review

### 1.1 Mask sizes compounded every frame (show-stopping)
`modules/headless_live.py` (old lines 193-205) ran this every frame:

```python
modules.globals.mouth_mask_size = min(100, modules.globals.mouth_mask_size + 30)
modules.globals.eyes_mask_size  = min(100, modules.globals.eyes_mask_size  + 15)
modules.globals.eyebrows_mask_size = min(100, modules.globals.eyebrows_mask_size + 25)
```

The mask size is a **global** used as an expansion factor in `face_masking.py`
(`expansion_factor = 1 + mask_down_size * mouth_mask_size`). After ~4-7 seconds of
streaming it pinned at 100, expanding the mouth mask ~11x and erasing the swap.

**Fix:** mask sizes are now static CLI values (0-100). Nothing mutates them at runtime.

### 1.2 Masks were silently doing nothing in live mode
The headless loop (and the GUI) detected faces with `detect_one_face_fast` /
`detect_many_faces_fast` — **detection only**, no `landmark_2d_106`. Every mask
function in `face_masking.py` starts with `if face.landmark_2d_106 is None: return empty`.
Result: `--mouth-mask --eyes-mask --eyebrows-mask` had zero effect on live streams.

**Fix:** `get_one_face_lite` / `get_many_faces_lite` in `face_analyser.py` run
detection + landmark model but **skip recognition** (the target's embedding is never
used by `swap_face`, so this saves ~1-2ms/face). The headless loop and GUI both pick
landmark-aware detection automatically whenever masks or enhancers are active
(`_needs_landmark()` — now also checks eyes/eyebrows masks and the `*_mask_size`
sliders, which it previously missed).

### 1.3 Mouth mask applied twice
`swap_face` already applied the mouth mask internally (with its own duplicate
`create_lower_mouth_mask` / `apply_mouth_area`), and `headless_live.py` applied it
again with the `face_masking.py` versions. Two divergent implementations of the same
feature (face_swapper used landmarks 52-71; face_masking used 52-63) produced
unpredictable results.

**Fix:** `swap_face` is now the single entry point that applies **all three masks**
(mouth/eyes/eyebrows) via `face_masking.py` — one code path shared by GUI and
headless. The in-loop mask code in `headless_live.py` was deleted, and ~460 lines of
duplicate legacy helpers at the bottom of `face_swapper.py` were removed.

---

## 2. Performance Upgrades

### 2.1 Resolution decoupling (the big win)
Before: detection, inswapper warp/paste-back and GPEN all ran at the full input
resolution (typically 1080p). The inswapper cost scales with **input** resolution,
not model size (~98ms @1080p vs ~32ms @480p measured in the FACELESS project).

Now (`modules/headless_live.py`):
```
capture (native) → downscale to process size (INTER_AREA)
                 → detection + swap + enhance (landmarks consistent at process scale)
                 → upscale to stream size (INTER_LINEAR) → output
```
- Default process size: **auto, capped at 854x480** (the FACELESS sweet spot).
- Tunable via `--process-width W --process-height H` (0 = auto).
- Pass your input resolution explicitly to process at native size.

### 2.2 Per-frame detection
Before: detection every ~80ms (`det_interval = round(fps * 0.08)`) → the swap bbox
lagged fast head motion (jitter/ghosting).
Now: detection runs **every frame** at the (reduced) processing resolution —
RetinaFace costs ~3-6ms there, well within the 30fps budget.

### 2.3 GPEN temporal caching
Ported the GFPGAN pattern (`face_enhancer.py` `_enh_live_cache`) into
`_onnx_enhancer.py`: in live mode (`detected_faces` provided, single-face mode),
GPEN inference now runs every **2 frames**; in between, the cached enhanced face +
inverse affine is pasted back. Cuts the enhancer cost roughly in half.
Enabled automatically — no new flags.

---

## 3. File-by-File Change Log

| File | Change |
|------|--------|
| `modules/headless_live.py` | Removed mask globals mutation + double masking; landmark-aware per-frame detection; resolution decoupling with auto 854x480 cap; upscale-to-stream-size; docstring updated with Phase 1 notes |
| `modules/processors/frame/face_swapper.py` | Imports mask helpers from `face_masking.py`; new `_mask_enabled()` / `_masks_enabled()` helpers; `swap_face` applies mouth+eyes+eyebrows masks from the pre-swap `original_frame`; legacy duplicate mask/color block (lines 956-1415) deleted |
| `modules/processors/frame/face_masking.py` | Now the **single source of truth** for all mask + color-transfer logic; `draw_mouth_mask_visualization` moved here from `face_swapper.py` |
| `modules/face_analyser.py` | `_needs_landmark()` covers eyes/eyebrows masks + size sliders; added `_analyse_faces_lite`, `get_one_face_lite`, `get_many_faces_lite` (detection + landmarks, no recognition, DML-lock aware) |
| `modules/processors/frame/_onnx_enhancer.py` | `enhance_face_onnx(..., temporal=False)` with shared `_enh_live_cache` + `_ENH_INTERVAL = 2`; refactored so the paste-back path is shared between fresh and cached results |
| `modules/processors/frame/face_enhancer_gpen256.py` / `face_enhancer_gpen512.py` | `enhance_face(..., temporal=False)`; `process_frame` auto-enables temporal caching for pre-detected single-face live mode |
| `modules/ui.py` | GUI live path uses the same landmark-aware lite detection so masks work there too |
| `modules/globals.py` | New `process_width` / `process_height` globals (0 = auto) |
| `modules/core.py` | New `--process-width` / `--process-height` CLI args |

---

## 4. Architecture After Phase 1

```
Thread 1: capture     → source.read() → capture_queue (maxsize 2, drops stale)
Thread 2: process     → downscale → detect (every frame, landmark-aware)
                       → swap_face [swap + masks in one pass]
                       → GPEN (temporal cache, every 2nd frame)
                       → upscale → processed_queue (maxsize 2)
Thread 3: output      → FFmpegStreamOutput.write() (auto-restart on crash)

Mask flow (single path, GUI + headless):
  swap_face() → face_masking.create_face_mask / create_lower_mouth_mask /
                create_eyes_mask / create_eyebrows_mask / apply_mask_area
Detection flow:
  masks or enhancers active → get_*_faces_lite (det + landmarks, no recognition)
  otherwise                 → detect_*_faces_fast (det only)
```

## 5. Verification

- All modified modules pass `python -m py_compile` (local Windows box, no deps
  needed for syntax checks).
- Full runtime validation (fps, VRAM, mask visuals) must happen on the cloud GPU box
  (local machine lacks the pinned Python 3.11 stack + onnxruntime-gpu + insightface).

Suggested smoke test on the server:
```bash
python run_headless.py -s face.jpg --input-source test.mp4 \
  --stream-output /workspace/phase1_smoke.mp4 \
  --frame-processor face_swapper face_enhancer_gpen256 \
  --mouth-mask --eyes-mask --eyebrows-mask \
  --stream-fps 30 --stream-encoder h264_nvenc --execution-provider cuda
```
Check: no mask growth over time, lips/eyes/brows move naturally, fps sustained at
target, output file plays cleanly. Also re-run `benchmark_pipeline.py`.

---

## 6. Next Phases (roadmap)

- **Phase 2 — latency**: pyvirtualcam local output mode (drop RTMP/ngrok/OBS when
  the webcam is on the same machine as processing); optional audio passthrough.
- **Phase 3 — quality ceiling**: 256px swap models (Reswapper/HyperSwap via DLC 2.7
  Ultimate) or DeepFaceLive-trained DFM pairs; batch/recorded GPEN-512 output.
- **Ongoing**: per-frame tracking (optical-flow ROI) to replace per-frame full
  detection when it becomes the bottleneck again on weaker GPUs.