"""FP16 model acceleration (Phase 3).

InsightFace swap + GPEN enhancer models ship as FP32. Running them in
FP16 on Tensor-Core GPUs (Turing+) roughly halves memory bandwidth and
speeds up inference (~2x for inswapper_128, which is heavily
memory-bound). This mirrors FACELESS, which ships `inswapper_128_fp16.onnx`.

Strategy: convert FP32 -> FP16 at load time with onnxconverter-common and
cache the result next to the original (e.g. `inswapper_128_fp16.onnx`).
No external downloads, deterministic, and gracefully falls back to FP32
when conversion is not possible (missing deps, unsupported ops, CPU-only).
"""

import os
import threading
import logging
from typing import Optional

import modules.globals

CONVERT_LOCK = threading.Lock()


def fp16_enabled() -> bool:
    """FP16 is only beneficial (and safe) on CUDA Tensor-Core GPUs."""
    if not getattr(modules.globals, "fp16", True):
        return False
    return any(
        (isinstance(p, str) and "CUDAExecutionProvider" in p) or
        (isinstance(p, tuple) and p[0] == "CUDAExecutionProvider")
        for p in modules.globals.execution_providers
    )


def _convert_to_fp16(fp32_path: str, fp16_path: str) -> bool:
    """Convert an ONNX model to FP16 (keep IO types so the runtime
    interface is unchanged). Returns True on success."""
    try:
        from onnxconverter_common import float16  # type: ignore
        import onnx
    except ImportError:
        logging.warning(
            "FP16 conversion unavailable: install 'onnxconverter-common' "
            "(pip install onnxconverter-common) to enable it."
        )
        return False
    try:
        model = onnx.load(fp32_path)
        # GPEN quirk: some initializers are ALSO declared as graph inputs.
        # With keep_io_types=True the inputs stay float32 while the
        # initializer data becomes float16 → ONNX Runtime rejects the
        # model ("element type tensor(float16) but usage ... expects
        # tensor(float)"). Initializers-as-inputs are redundant by
        # definition, so dropping them from the input list is safe.
        initializer_names = {init.name for init in model.graph.initializer}
        for graph_input in list(model.graph.input):
            if graph_input.name in initializer_names:
                model.graph.input.remove(graph_input)
        model_fp16 = float16.convert_float_to_float16(
            model, keep_io_types=True
        )
        onnx.save(model_fp16, fp16_path)
        return True
    except Exception as e:
        logging.warning(f"FP16 conversion of {os.path.basename(fp32_path)} failed: {e}")
        try:
            if os.path.exists(fp16_path):
                os.remove(fp16_path)
        except OSError:
            pass
        return False


def get_fp16_model_path(fp32_path: str) -> Optional[str]:
    """Return a cached FP16 model path for `fp32_path`, converting it if
    needed. Returns None when FP16 is disabled/unsupported/conversion
    fails — callers must fall back to the FP32 model."""
    if not fp16_enabled() or not fp32_path or not os.path.isfile(fp32_path):
        return None

    dir_name = os.path.dirname(fp32_path)
    stem = os.path.splitext(os.path.basename(fp32_path))[0]
    fp16_path = os.path.join(dir_name, f"{stem}_fp16.onnx")

    if os.path.isfile(fp16_path):
        return fp16_path

    with CONVERT_LOCK:
        # Re-check under the lock (another thread may have converted it)
        if os.path.isfile(fp16_path):
            return fp16_path
        print(f"[FP16] Converting {os.path.basename(fp32_path)} -> "
              f"{os.path.basename(fp16_path)} (one-time, cached)")
        if _convert_to_fp16(fp32_path, fp16_path):
            print(f"[FP16] Conversion done: {fp16_path}")
            return fp16_path
        return None