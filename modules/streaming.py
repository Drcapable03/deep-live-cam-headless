"""
Streaming I/O module for headless Deep-Live-Cam.

Handles:
  - Video input from files, named pipes, or camera devices
  - RTMP/UDP streaming output via FFmpeg
  - Raw frame reading/writing for inter-process communication

This module is designed to work without any GUI or display server.
"""

import os
import sys
import cv2
import numpy as np
import subprocess
import threading
import queue
import time
from typing import Optional, Callable, Tuple, Generator
from pathlib import Path

import modules.globals


class FrameSource:
    """Abstract base for frame input sources."""

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        raise NotImplementedError

    def get_properties(self) -> dict:
        raise NotImplementedError

    def release(self) -> None:
        raise NotImplementedError


class CameraSource(FrameSource):
    """Reads frames from a local camera device."""

    def __init__(self, device_index: int = 0):
        self.device_index = device_index
        self.cap: Optional[cv2.VideoCapture] = None
        self.width = 0
        self.height = 0
        self.fps = 30.0

    def start(self, width: int = 1280, height: int = 720, fps: int = 30) -> bool:
        """Initialize the camera."""
        # Try different backends
        backends = [cv2.CAP_V4L2, cv2.CAP_ANY] if sys.platform != "win32" else [cv2.CAP_MSMF, cv2.CAP_DSHOW, cv2.CAP_ANY]

        for backend in backends:
            try:
                self.cap = cv2.VideoCapture(self.device_index, backend)
                if self.cap.isOpened():
                    break
                self.cap.release()
            except Exception:
                continue

        if self.cap is None or not self.cap.isOpened():
            print(f"[STREAMING] Failed to open camera {self.device_index}")
            return False

        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS, fps)

        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or fps

        print(f"[STREAMING] Camera opened: {self.width}x{self.height} @ {self.fps:.1f}fps")
        return True

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        if self.cap is None:
            return False, None
        return self.cap.read()

    def get_properties(self) -> dict:
        return {"width": self.width, "height": self.height, "fps": self.fps}

    def release(self) -> None:
        if self.cap:
            self.cap.release()
            self.cap = None


class VideoFileSource(FrameSource):
    """Reads frames from a video file in a loop (for testing)."""

    def __init__(self, file_path: str):
        self.file_path = file_path
        self.cap: Optional[cv2.VideoCapture] = None
        self.width = 0
        self.height = 0
        self.fps = 30.0
        self.frame_count = 0
        self.loop = True

    def start(self) -> bool:
        if not os.path.isfile(self.file_path):
            print(f"[STREAMING] Video file not found: {self.file_path}")
            return False

        self.cap = cv2.VideoCapture(self.file_path)
        if not self.cap.isOpened():
            print(f"[STREAMING] Failed to open video file: {self.file_path}")
            return False

        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

        print(f"[STREAMING] Video file opened: {self.width}x{self.height} @ {self.fps:.1f}fps, {self.frame_count} frames")
        return True

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        if self.cap is None:
            return False, None
        ret, frame = self.cap.read()
        if not ret and self.loop:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = self.cap.read()
        return ret, frame

    def get_properties(self) -> dict:
        return {"width": self.width, "height": self.height, "fps": self.fps, "frame_count": self.frame_count}

    def release(self) -> None:
        if self.cap:
            self.cap.release()
            self.cap = None


class PipeSource(FrameSource):
    """Reads frames from a named pipe (FIFO) or stdin.

    Expects raw BGR24 frames of the configured width/height.
    """

    def __init__(self, pipe_path: Optional[str] = None, width: int = 1280, height: int = 720):
        self.pipe_path = pipe_path  # None = stdin
        self.width = width
        self.height = height
        self._fifo_fd = None
        self._buffer = b""
        self._frame_size = width * height * 3

    def start(self) -> bool:
        if self.pipe_path is None:
            print("[STREAMING] Reading raw BGR24 frames from stdin")
            return True

        # Create FIFO if it doesn't exist
        if not os.path.exists(self.pipe_path):
            try:
                os.mkfifo(self.pipe_path)
                print(f"[STREAMING] Created FIFO: {self.pipe_path}")
            except Exception as e:
                print(f"[STREAMING] Failed to create FIFO: {e}")
                return False

        print(f"[STREAMING] Waiting for FIFO reader/writer on: {self.pipe_path}")
        try:
            self._fifo_fd = open(self.pipe_path, 'rb')
            print(f"[STREAMING] FIFO opened for reading")
            return True
        except Exception as e:
            print(f"[STREAMING] Failed to open FIFO: {e}")
            return False

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        fd = self._fifo_fd if self._fifo_fd else sys.stdin.buffer
        try:
            while len(self._buffer) < self._frame_size:
                chunk = fd.read(self._frame_size - len(self._buffer))
                if not chunk:
                    return False, None
                self._buffer += chunk

            frame_data = self._buffer[:self._frame_size]
            self._buffer = self._buffer[self._frame_size:]
            frame = np.frombuffer(frame_data, dtype=np.uint8).reshape((self.height, self.width, 3))
            return True, frame.copy()
        except Exception as e:
            print(f"[STREAMING] Pipe read error: {e}")
            return False, None

    def get_properties(self) -> dict:
        return {"width": self.width, "height": self.height, "fps": 30.0}

    def release(self) -> None:
        if self._fifo_fd:
            self._fifo_fd.close()
            self._fifo_fd = None


class FFmpegInputSource(FrameSource):
    """Reads frames via FFmpeg from any URL (RTMP, RTSP, HTTP, file, etc.)."""

    def __init__(self, url: str):
        self.url = url
        self.process: Optional[subprocess.Popen] = None
        self.width = 1280
        self.height = 720
        self.fps = 30.0
        self._frame_size = 0
        self._buffer = b""

    def start(self, width: int = 1280, height: int = 720) -> bool:
        self.width = width
        self.height = height
        self._frame_size = width * height * 3

        cmd = [
            "ffmpeg", "-hide_banner", "-loglevel", "error",
            "-hwaccel", "auto",
            "-i", self.url,
            "-f", "rawvideo", "-pix_fmt", "bgr24",
            "-s", f"{width}x{height}",
            "-r", str(self.fps),
            "-",
        ]

        try:
            self.process = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE
            )
            # Quick test read to verify it's working
            print(f"[STREAMING] FFmpeg input started from: {self.url}")
            return True
        except Exception as e:
            print(f"[STREAMING] Failed to start FFmpeg input: {e}")
            return False

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        if self.process is None or self.process.poll() is not None:
            return False, None

        try:
            while len(self._buffer) < self._frame_size:
                chunk = self.process.stdout.read(self._frame_size - len(self._buffer))
                if not chunk:
                    return False, None
                self._buffer += chunk

            frame_data = self._buffer[:self._frame_size]
            self._buffer = self._buffer[self._frame_size:]
            frame = np.frombuffer(frame_data, dtype=np.uint8).reshape((self.height, self.width, 3))
            return True, frame.copy()
        except Exception as e:
            print(f"[STREAMING] FFmpeg read error: {e}")
            return False, None

    def get_properties(self) -> dict:
        return {"width": self.width, "height": self.height, "fps": self.fps}

    def release(self) -> None:
        if self.process:
            self.process.kill()
            self.process = None


def create_source(source_str: str) -> FrameSource:
    """Factory: create appropriate FrameSource from a string descriptor.

    Args:
        source_str: One of:
            - Integer (e.g., "0", "1") -> camera index
            - "/dev/video0" -> camera device
            - "file:/path/to/video.mp4" -> video file (loops)
            - "pipe" or "pipe:/path/to/fifo" -> named pipe/stdin
            - "rtmp://...", "rtsp://...", "http://..." -> FFmpeg stream
            - Any file path ending in .mp4, .mkv, .avi -> video file

    Returns:
        FrameSource instance
    """
    # Camera index (integer)
    try:
        idx = int(source_str)
        src = CameraSource(idx)
        return src
    except ValueError:
        pass

    # Video file
    video_exts = ('.mp4', '.mkv', '.avi', '.mov', '.flv', '.wmv', '.webm')
    if source_str.lower().startswith("file:"):
        return VideoFileSource(source_str[5:])
    if source_str.lower().endswith(video_exts) and os.path.isfile(source_str):
        return VideoFileSource(source_str)

    # Named pipe
    if source_str.lower() == "pipe" or source_str.lower().startswith("pipe:"):
        pipe_path = None if source_str.lower() == "pipe" else source_str[5:]
        return PipeSource(pipe_path)

    # FFmpeg-compatible URL (RTMP, RTSP, HTTP, etc.)
    if source_str.lower().startswith(("rtmp://", "rtsp://", "http://", "https://", "udp://")):
        return FFmpegInputSource(source_str)

    # Default: try as video file path
    return VideoFileSource(source_str)


# ---------------------------------------------------------------------------
# Output: RTMP / Pipe streaming via FFmpeg
# ---------------------------------------------------------------------------

class StreamOutput:
    """Abstract base for frame output sinks."""

    def write(self, frame: np.ndarray) -> bool:
        raise NotImplementedError

    def close(self) -> None:
        raise NotImplementedError


class FFmpegStreamOutput(StreamOutput):
    """Streams frames to RTMP/UDP/SRT via FFmpeg.

    Example URL formats:
        rtmp://live-server/app/stream_key
        udp://host:port
        srt://host:port
        /path/to/output.mp4  (file recording)
    """

    def __init__(self, url: str, width: int = 1280, height: int = 720,
                 fps: int = 30, encoder: str = "libx264", quality: int = 23):
        self.url = url
        self.width = width
        self.height = height
        self.fps = fps
        self.encoder = encoder
        self.quality = quality
        self.process: Optional[subprocess.Popen] = None
        self._frame_size = width * height * 3
        self._stderr_thread: Optional[threading.Thread] = None

    def _stderr_reader(self):
        """Drain FFmpeg stderr to prevent pipe deadlock."""
        if self.process and self.process.stderr:
            try:
                for line in self.process.stderr:
                    line = line.decode('utf-8', errors='replace').strip()
                    if line:
                        print(f"[FFMPEG] {line}")
            except Exception:
                pass

    def start(self) -> bool:
        # Build encoder options
        encoder_options = []
        use_encoder = self.encoder

        if use_encoder == "h264_nvenc":
            encoder_options = [
                "-preset", "p4", "-tune", "hq", "-rc", "vbr",
                "-cq", str(self.quality), "-b:v", "0",
            ]
        elif use_encoder in ("h264_amf", "hevc_amf"):
            encoder_options = [
                "-quality", "quality", "-rc", "vbr_latency",
                "-qp_i", str(self.quality), "-qp_p", str(self.quality),
            ]
        elif use_encoder == "hevc_nvenc":
            encoder_options = [
                "-preset", "p4", "-tune", "hq", "-rc", "vbr",
                "-cq", str(self.quality), "-b:v", "0",
            ]
        elif use_encoder in ("libx264", "h264"):
            use_encoder = "libx264"
            encoder_options = [
                "-preset", "veryfast",
                "-tune", "zerolatency",
                "-crf", str(self.quality),
            ]
        elif use_encoder == "libx265":
            encoder_options = [
                "-preset", "ultrafast",
                "-crf", str(self.quality),
                "-x265-params", "log-level=error",
            ]
        elif use_encoder == "copy":
            encoder_options = []
        else:
            # Default to libx264
            use_encoder = "libx264"
            encoder_options = [
                "-preset", "veryfast",
                "-tune", "zerolatency",
                "-crf", str(self.quality),
            ]

        cmd = [
            "ffmpeg", "-hide_banner", "-loglevel", "error",
            "-f", "rawvideo",
            "-pix_fmt", "bgr24",
            "-s", f"{self.width}x{self.height}",
            "-r", str(self.fps),
            "-thread_queue_size", "512",
            "-i", "-",  # Read from stdin
            "-c:v", use_encoder,
        ]
        cmd.extend(encoder_options)
        # Determine output container format
        if self.url.startswith("rtmp://"):
            out_format = ["-f", "flv"]
        elif self.url.startswith(("udp://", "srt://")):
            out_format = ["-f", "mpegts"]
        else:
            # Local file: let FFmpeg auto-detect from file extension
            out_format = []

        cmd.extend([
            "-pix_fmt", "yuv420p",
            "-g", str(self.fps * 2),  # Keyframe every 2 seconds
            "-flush_packets", "1",
        ])
        cmd.extend(out_format)
        cmd.extend(["-y", self.url])

        try:
            self.process = subprocess.Popen(
                cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE,
                bufsize=10*1024*1024  # 10MB buffer for large frames
            )
            # Start stderr reader thread to prevent pipe deadlock
            self._stderr_thread = threading.Thread(target=self._stderr_reader, daemon=True)
            self._stderr_thread.start()
            print(f"[STREAMING] FFmpeg output started -> {self.url}")
            print(f"[STREAMING]   Encoder: {use_encoder}, {self.width}x{self.height}@{self.fps}fps")
            return True
        except Exception as e:
            print(f"[STREAMING] Failed to start FFmpeg output: {e}")
            return False

    def write(self, frame: np.ndarray) -> bool:
        if self.process is None or self.process.poll() is not None:
            return False
        try:
            # Ensure frame is correct size
            if frame.shape[0] != self.height or frame.shape[1] != self.width:
                frame = cv2.resize(frame, (self.width, self.height))
            self.process.stdin.write(frame.tobytes())
            return True
        except BrokenPipeError:
            print("[STREAMING] Output pipe broken")
            return False
        except Exception as e:
            print(f"[STREAMING] Write error: {e}")
            return False

    def close(self) -> None:
        if self.process:
            try:
                self.process.stdin.close()
                self.process.wait(timeout=5)
            except Exception:
                self.process.kill()
            self.process = None
        # Wait for stderr reader to finish
        if self._stderr_thread and self._stderr_thread.is_alive():
            self._stderr_thread.join(timeout=1.0)
        print("[STREAMING] FFmpeg output closed")


class PipeOutput(StreamOutput):
    """Writes raw BGR24 frames to a named pipe."""

    def __init__(self, pipe_path: str, width: int = 1280, height: int = 720):
        self.pipe_path = pipe_path
        self.width = width
        self.height = height
        self._fifo_fd = None

    def start(self) -> bool:
        if not os.path.exists(self.pipe_path):
            try:
                os.mkfifo(self.pipe_path)
            except Exception as e:
                print(f"[STREAMING] Failed to create output FIFO: {e}")
                return False

        print(f"[STREAMING] Waiting for reader to open output FIFO: {self.pipe_path}")
        try:
            self._fifo_fd = open(self.pipe_path, 'wb')
            print("[STREAMING] Output FIFO opened")
            return True
        except Exception as e:
            print(f"[STREAMING] Failed to open output FIFO: {e}")
            return False

    def write(self, frame: np.ndarray) -> bool:
        if self._fifo_fd is None:
            return False
        try:
            if frame.shape[1] != self.width or frame.shape[0] != self.height:
                frame = cv2.resize(frame, (self.width, self.height))
            self._fifo_fd.write(frame.tobytes())
            self._fifo_fd.flush()
            return True
        except BrokenPipeError:
            return False
        except Exception as e:
            print(f"[STREAMING] Pipe write error: {e}")
            return False

    def close(self) -> None:
        if self._fifo_fd:
            self._fifo_fd.close()
            self._fifo_fd = None


def create_output(url: str, width: int = 1280, height: int = 720,
                  fps: int = 30, encoder: str = "libx264", quality: int = 23) -> StreamOutput:
    """Factory: create appropriate StreamOutput from a URL string.

    Args:
        url: One of:
            - "rtmp://..." -> FFmpeg RTMP stream
            - "udp://..." -> FFmpeg UDP stream
            - "pipe:/path/to/fifo" -> named pipe output
            - "/path/to/file.mp4" -> file recording
    """
    if url.lower().startswith("pipe:"):
        return PipeOutput(url[5:], width, height)
    return FFmpegStreamOutput(url, width, height, fps, encoder, quality)


# ---------------------------------------------------------------------------
# Frame generator for the processing pipeline
# ---------------------------------------------------------------------------

def frame_generator(source: FrameSource, stop_event: Optional[threading.Event] = None) -> Generator[np.ndarray, None, None]:
    """Yield frames from a source until exhausted or stopped."""
    while stop_event is None or not stop_event.is_set():
        ret, frame = source.read()
        if not ret or frame is None:
            break
        yield frame
