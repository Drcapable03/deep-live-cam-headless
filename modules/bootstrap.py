"""Shared bootstrapping helpers for entry points (run.py, run_headless.py).

Ensures ffmpeg/ffprobe are on PATH and NVIDIA CUDA DLLs are discoverable
before any ONNX Runtime or InsightFace import happens.
"""

import os
import sys


def bootstrap_paths(project_root: str) -> None:
    """Add project-local bin and NVIDIA CUDA DLLs to the process PATH."""
    # Add the project root to PATH so bundled ffmpeg/ffprobe are found
    os.environ["PATH"] = project_root + os.path.sep + os.environ.get("PATH", "")

    if sys.platform != "win32":
        return

    _site_packages = os.path.join(sys.prefix, "Lib", "site-packages")
    _venv_site_packages = os.path.join(project_root, "venv", "Lib", "site-packages")
    for _sp in (_site_packages, _venv_site_packages):
        _torch_lib = os.path.join(_sp, "torch", "lib")
        if os.path.isdir(_torch_lib):
            os.environ["PATH"] = _torch_lib + os.pathsep + os.environ["PATH"]
        _nvidia_dir = os.path.join(_sp, "nvidia")
        if os.path.isdir(_nvidia_dir):
            for _pkg in os.listdir(_nvidia_dir):
                _bin_dir = os.path.join(_nvidia_dir, _pkg, "bin")
                if os.path.isdir(_bin_dir):
                    os.environ["PATH"] = _bin_dir + os.pathsep + os.environ["PATH"]
