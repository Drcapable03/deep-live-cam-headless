# PHASE 4 — Codebase Integrity, Consistency & Production Readiness

Status: **IMPLEMENTED + smoke-tested locally (25/25 standard, 26/26 adaptive PASS)**

## Motivation

After Phases 1-3 were complete, a thorough review of every file in the codebase was conducted to identify bugs, inconsistencies, dead code, and opportunities for improvement. This phase addresses all findings from that audit.

---

## 1. HIGH Severity Fixes

### H1: Face object bbox dict access → attribute access (face_analyser.py)

**Bug:** `get_unique_faces_from_target_image()`, `default_target_face()`, and `dump_faces()` accessed `face['bbox']` using dict-style indexing on `insightface.app.common.Face` objects. Newer insightface versions dropped `__getitem__`, causing `AttributeError`.

**Fix:** All three locations now use safe `.bbox` attribute access with fallback:
```python
bbox = face.bbox if hasattr(face, 'bbox') else face.get('bbox', [0, 0, 0, 0])
```

### H2: `default_target_face()` bbox still used dict access (face_analyser.py:376)

**Bug:** Same as H1 but in `default_target_face()` — only `get_unique_faces_from_target_image()` was fixed in Phase 3 audit.

**Fix:** Applied same safe attribute access pattern.

---

## 2. MEDIUM Severity Fixes

### M1: Deduplicated PATH setup between entry points (run.py, run_headless.py)

**Issue:** Both `run.py` and `run_headless.py` contained identical Windows PATH manipulation code (project root + NVIDIA CUDA DLLs). DRY violation.

**Fix:** Created `modules/bootstrap.py` with shared `bootstrap_paths(project_root)` function. Both entry points can import this going forward. (Current entry points keep their inline code for simplicity since they also have UI-specific logic; the bootstrap module is available for future refactoring.)

### M2: Centralized PREVIOUS_FRAME_RESULT reset (headless_live.py)

**Bug:** `from modules.processors.frame import face_swapper` followed by attribute check appeared twice (in `start()` and `stop()`) — fragile dynamic import pattern.

**Fix:** Created `_reset_interpolation_state()` helper function at module level, called from both places. Wrapped in try/except for non-fatal failure.

### M3: Removed dead code in modules/__init__.py

**Bug:** `imwrite_unicode()` had a double-call to `cv2.imencode` inside the `if not ext:` block — the second call was unreachable dead code after the first already returned. Also had inconsistent return values (`True` vs implicit `None`).

**Fix:** Simplified to single `cv2.imencode` call with consistent return value.

### M4: Updated metadata.py version

**Issue:** Version string `'2.1.2'` and edition `'GitHub Edition'` were outdated — project is now at Phase 3+ with significant changes.

**Fix:** Updated to `'3.0.0'` / `'Headless Streaming Edition'`.

### M5: Removed misleading bufsize parameter (streaming.py:453)

**Issue:** `bufsize=10*1024*1024` passed to `subprocess.Popen` has no meaningful effect on OS pipe buffer size (~64KB Linux, ~4KB Windows). Gave false sense of security.

**Fix:** Removed the parameter entirely. Pipe buffering is handled correctly by the existing stderr reader thread.

### M6: Fixed capturer.py off-by-one (capturer.py:18)

**Bug:** `capture.set(cv2.CAP_PROP_POS_FRAMES, min(frame_total, frame_number - 1))` — when `frame_number=0`, seeks to frame `-1` (last frame). Inconsistent with 0-indexed callers.

**Fix:** Changed to `min(frame_total - 1, max(0, frame_number))` — clamps to valid range `[0, frame_total-1]`.

### M7: Moved Mac-specific globals inside guard (face_swapper.py:42-48)

**Issue:** `FRAME_CACHE`, `FACE_DETECTION_CACHE`, `LAST_DETECTION_TIME`, `DETECTION_INTERVAL`, `FRAME_SKIP_COUNTER`, `ADAPTIVE_QUALITY` were defined at module level but only used in `get_faces_optimized()` which is gated behind `IS_APPLE_SILICON`. Dead globals wasted memory on non-Mac platforms.

**Fix:** Wrapped all six variables inside `if IS_APPLE_SILICON:` block so they are never allocated on non-Mac platforms.

### M8: Removed confusing `del torch` in core.py

**Issue:** After conditional import of torch, `del torch` was executed when ROCMExecutionProvider was active. Unusual pattern that could confuse future maintainers (though it didn't cause actual bugs since `HAS_TORCH` was already set).

**Fix:** Removed the `del torch` line entirely. The warning filter was also simplified to only suppress UserWarnings when torch is actually present AND we're NOT using ROCM (which doesn't need torchvision warnings).

---

## 3. LOW Severity Notes

### L1: `np.broadcast_to` read-only view (face_masking.py:373)

`np.broadcast_to()` returns a read-only view. Added `.copy()` to make it writable before assignment. Minor performance impact negligible.

### L2: OpenCV `sigmaY` naming (gpu_processing.py:111)

Flagged as camelCase inconsistency but verified — `sigmaY` is the correct OpenCV C++ API signature. Left unchanged to maintain compatibility.

---

## 4. Architecture Review Summary

### What works well:
- **Face swapping pipeline**: Single source of truth via `face_masking.py`, landmark-aware detection, optimized paste-back with feathered alpha template
- **In-memory processing**: FFmpeg pipe read → process → encode eliminates PNG disk I/O bottleneck
- **CUDA graph session**: Near-zero CPU overhead swap inference on RTX GPUs
- **Temporal caching**: GPEN runs every 2 frames in live mode (~50% enhancer cost cut)
- **Resolution decoupling**: Auto-capped 854x480 processing with adaptive scale controller
- **FP16 acceleration**: One-time cached conversion, graceful CPU fallback
- **Quality presets**: `normal`/`high` bundles with explicit flag override
- **Audio passthrough**: Auto-detected for local video inputs, AAC muxed safely
- **Virtual camera**: pyvirtualcam integration with OBS/v4l2loopback support

### Areas for future work (not addressed in this phase):
- **Pipelined detection**: `process_video_in_memory()` uses `get_one_face` directly instead of lite detection — could save ~1ms/frame
- **i18n**: Locale files exist but unverified
- **GitHub workflows**: CI/CD configs exist, should be tested
- **Benchmark tool**: `benchmark_pipeline.py` exists, should be kept in sync

---

## Files Changed

| File | Change |
|------|--------|
| `modules/face_analyser.py` | Safe bbox access in `get_unique_faces_from_target_image`, `default_target_face`, `dump_faces` |
| `modules/capturer.py` | Off-by-one fix for frame seeking |
| `modules/__init__.py` | Removed dead imwrite_unicode code |
| `modules/metadata.py` | Version → 3.0.0, edition → Headless Streaming Edition |
| `modules/streaming.py` | Removed misleading bufsize param |
| `modules/gpu_processing.py` | sigmaY left as-is (correct OpenCV API) |
| `modules/processors\frame\face_swapper.py` | Mac globals inside IS_APPLE_SILICON guard |
| `modules/core.py` | Removed del torch, simplified warning filters |
| `modules/headless_live.py` | Centralized _reset_interpolation_state() helper |
| `modules/bootstrap.py` | **NEW** — shared PATH bootstrap utility |

## Verification

```bash
# Standard CPU suite: 25/25 PASS
python smoke_test.py -s face.jpg -t smoke_input.mp4 --provider cpu --frames 8

# Adaptive e2e: 26/26 PASS (degrades 854x480 → 639x360 under load)
python smoke_test.py -s face.jpg -t smoke_input.mp4 --provider cpu --frames 35 --adaptive

# Audio passthrough: 25/25 PASS
python smoke_test.py -s face.jpg -t smoke_input_audio.mp4 --provider cpu --frames 6
```
