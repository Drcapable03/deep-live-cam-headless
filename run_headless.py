#!/usr/bin/env python3
"""
Headless entry point for Deep-Live-Cam streaming.

Usage examples:

  # Camera input -> RTMP output (cloud GPU streaming)
  python run_headless.py -s face.jpg --input-source 0 --stream-output rtmp://server/live/stream

  # Video file input (loop) -> RTMP output
  python run_headless.py -s face.jpg --input-source /path/to/video.mp4 --stream-output rtmp://server/live/stream

  # FFmpeg input (RTMP/RTSP/HTTP) -> file output
  python run_headless.py -s face.jpg --input-source rtmp://input/stream --stream-output /tmp/output.mp4

  # Named pipe input -> named pipe output
  python run_headless.py -s face.jpg --input-source pipe:/tmp/input_pipe --stream-output pipe:/tmp/output_pipe

  # With enhancer and many faces
  python run_headless.py -s face.jpg --input-source 0 --stream-output rtmp://... \\
    --frame-processor face_swapper face_enhancer --many-faces \\
    --stream-width 1920 --stream-height 1080 --stream-fps 60 \\
    --stream-encoder h264_nvenc --stream-quality 20

Environment variables:
  OMP_NUM_THREADS: Set CPU thread limit (default: 6 with CUDA)
"""

import os
import sys

# Add the project root to PATH so bundled ffmpeg/ffprobe are found
project_root = os.path.dirname(os.path.abspath(__file__))
os.environ["PATH"] = project_root + os.pathsep + os.environ.get("PATH", "")

# On Windows, add NVIDIA CUDA DLL directories to PATH
if sys.platform == "win32":
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

from modules import platform_info
platform_info.print_banner()

from modules import core

if __name__ == '__main__':
    # Force headless mode when using this entry point
    if '--headless' not in sys.argv:
        sys.argv.append('--headless')
    core.run()
