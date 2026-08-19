import cv2
import numpy as np
import sys
import os
import time
from typing import Optional, Tuple, Callable
import platform
import threading

# Only import Windows-specific library if on Windows
if platform.system() == "Windows":
    from pygrabber.dshow_graph import FilterGraph


class VideoCapturer:
    """Unified video capture from camera, file, pipe, or FFmpeg source.

    Supports:
      - Camera devices (by index): 0, 1, 2...
      - Video files: /path/to/video.mp4
      - Named pipes: pipe:/path/to/fifo
      - FFmpeg URLs: rtmp://..., rtsp://..., http://...
    """

    def __init__(self, source: str = "0"):
        """Args:
            source: Camera index ("0", "1"), file path, pipe path, or URL.
        """
        self.source = source
        self.device_index: int = 0
        self.frame_callback = None
        self._current_frame = None
        self._frame_ready = threading.Event()
        self.is_running = False
        self.cap = None
        # Source type detection
        self._source_type = self._detect_source_type(source)
        # Actual values reported by the camera after configuration
        self.actual_width: int = 0
        self.actual_height: int = 0
        self.actual_fps: float = 0.0

        # Initialize Windows-specific components if on Windows
        if platform.system() == "Windows" and self._source_type == "camera":
            self.graph = FilterGraph()
            # Verify device exists
            devices = self.graph.get_input_devices()
            if self.device_index >= len(devices):
                raise ValueError(
                    f"Invalid device index {self.device_index}. Available devices: {len(devices)}"
                )

    def _detect_source_type(self, source: str) -> str:
        """Detect whether source is camera, file, pipe, or ffmpeg URL."""
        # Integer = camera
        try:
            int(source)
            return "camera"
        except ValueError:
            pass

        # URL schemes
        if source.lower().startswith(("rtmp://", "rtsp://", "http://", "https://", "udp://")):
            return "ffmpeg"

        # Pipe prefix
        if source.lower().startswith("pipe:"):
            return "pipe"

        # Video file extensions
        video_exts = ('.mp4', '.mkv', '.avi', '.mov', '.flv', '.wmv', '.webm')
        if source.lower().endswith(video_exts):
            return "file"

        # Default: treat as camera index or file
        return "camera"

    def start(self, width: int = 960, height: int = 540, fps: int = 60) -> bool:
        """Initialize and start video capture from the configured source."""
        try:
            if self._source_type == "camera":
                return self._start_camera(width, height, fps)
            elif self._source_type == "file":
                return self._start_file()
            elif self._source_type == "ffmpeg":
                return self._start_ffmpeg(width, height, fps)
            elif self._source_type == "pipe":
                return self._start_pipe(width, height)
            else:
                return self._start_camera(width, height, fps)
        except Exception as e:
            print(f"[VideoCapturer] Failed to start capture: {str(e)}")
            if self.cap:
                self.cap.release()
            return False

    def _start_camera(self, width: int, height: int, fps: int) -> bool:
        """Initialize camera capture (original behavior)."""
        self.device_index = int(self.source)

        if platform.system() == "Windows":
            capture_methods = [
                (self.device_index, cv2.CAP_MSMF),
                (self.device_index, cv2.CAP_DSHOW),
                (self.device_index, cv2.CAP_ANY),
                (0, cv2.CAP_ANY),
            ]
            for dev_id, backend in capture_methods:
                try:
                    self.cap = cv2.VideoCapture(dev_id, backend)
                    if self.cap.isOpened():
                        break
                    self.cap.release()
                except Exception:
                    continue
        else:
            self.cap = cv2.VideoCapture(self.device_index)

        if not self.cap or not self.cap.isOpened():
            raise RuntimeError("Failed to open camera")

        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS, fps)

        self.actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        reported_fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.actual_fps = self._measure_fps(warmup=10, sample=30, fallback=reported_fps or fps)

        print(f"[VideoCapturer] Camera: {self.actual_width}x{self.actual_height} @ {self.actual_fps:.1f}fps")
        self.is_running = True
        return True

    def _start_file(self) -> bool:
        """Initialize video file capture (loops)."""
        file_path = self.source
        if not os.path.isfile(file_path):
            raise RuntimeError(f"Video file not found: {file_path}")

        self.cap = cv2.VideoCapture(file_path)
        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open video file: {file_path}")

        self.actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.actual_fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self._file_loop = True  # Loop the video

        print(f"[VideoCapturer] File: {self.actual_width}x{self.actual_height} @ {self.actual_fps:.1f}fps ({file_path})")
        self.is_running = True
        return True

    def _start_ffmpeg(self, width: int, height: int, fps: int) -> bool:
        """Initialize FFmpeg-based capture for RTMP/RTSP/HTTP streams."""
        import subprocess

        self._ffmpeg_cmd = [
            "ffmpeg", "-hide_banner", "-loglevel", "error",
            "-hwaccel", "auto",
            "-i", self.source,
            "-f", "rawvideo", "-pix_fmt", "bgr24",
            "-s", f"{width}x{height}",
            "-r", str(fps),
            "-",
        ]
        self._ffmpeg_proc = subprocess.Popen(
            self._ffmpeg_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        self._ffmpeg_frame_size = width * height * 3
        self._ffmpeg_buffer = b""
        self.actual_width = width
        self.actual_height = height
        self.actual_fps = fps

        print(f"[VideoCapturer] FFmpeg: {width}x{height} @ {fps}fps ({self.source})")
        self.is_running = True
        return True

    def _start_pipe(self, width: int, height: int) -> bool:
        """Initialize named pipe capture."""
        pipe_path = self.source[5:] if self.source.lower().startswith("pipe:") else self.source
        self._pipe_fd = None

        if not os.path.exists(pipe_path):
            try:
                os.mkfifo(pipe_path)
            except Exception as e:
                raise RuntimeError(f"Failed to create FIFO: {e}")

        self._pipe_fd = open(pipe_path, 'rb')
        self._pipe_frame_size = width * height * 3
        self._pipe_buffer = b""
        self.actual_width = width
        self.actual_height = height
        self.actual_fps = 30.0

        print(f"[VideoCapturer] Pipe: {width}x{height} ({pipe_path})")
        self.is_running = True
        return True

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Read a frame from the source."""
        if not self.is_running:
            return False, None

        if self._source_type == "camera" or self._source_type == "file":
            return self._read_cv2()
        elif self._source_type == "ffmpeg":
            return self._read_ffmpeg()
        elif self._source_type == "pipe":
            return self._read_pipe()
        return False, None

    def _read_cv2(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Read frame via OpenCV VideoCapture."""
        if not self.cap or not self.cap.isOpened():
            return False, None
        ret, frame = self.cap.read()
        if not ret and hasattr(self, '_file_loop') and self._file_loop:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = self.cap.read()
        if ret:
            self._current_frame = frame
            if self.frame_callback:
                self.frame_callback(frame)
            return True, frame
        return False, None

    def _read_ffmpeg(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Read frame from FFmpeg subprocess stdout."""
        if hasattr(self, '_ffmpeg_proc') and self._ffmpeg_proc.poll() is not None:
            return False, None
        try:
            while len(self._ffmpeg_buffer) < self._ffmpeg_frame_size:
                chunk = self._ffmpeg_proc.stdout.read(self._ffmpeg_frame_size - len(self._ffmpeg_buffer))
                if not chunk:
                    return False, None
                self._ffmpeg_buffer += chunk

            frame_data = self._ffmpeg_buffer[:self._ffmpeg_frame_size]
            self._ffmpeg_buffer = self._ffmpeg_buffer[self._ffmpeg_frame_size:]
            frame = np.frombuffer(frame_data, dtype=np.uint8).reshape(
                (self.actual_height, self.actual_width, 3)).copy()
            return True, frame
        except Exception:
            return False, None

    def _read_pipe(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Read frame from named pipe."""
        if not hasattr(self, '_pipe_fd') or self._pipe_fd is None:
            return False, None
        try:
            while len(self._pipe_buffer) < self._pipe_frame_size:
                chunk = self._pipe_fd.read(self._pipe_frame_size - len(self._pipe_buffer))
                if not chunk:
                    return False, None
                self._pipe_buffer += chunk

            frame_data = self._pipe_buffer[:self._pipe_frame_size]
            self._pipe_buffer = self._pipe_buffer[self._pipe_frame_size:]
            frame = np.frombuffer(frame_data, dtype=np.uint8).reshape(
                (self.actual_height, self.actual_width, 3)).copy()
            return True, frame
        except Exception:
            return False, None

    def release(self) -> None:
        """Stop capture and release resources."""
        self.is_running = False
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        if hasattr(self, '_ffmpeg_proc') and self._ffmpeg_proc:
            self._ffmpeg_proc.kill()
            self._ffmpeg_proc = None
        if hasattr(self, '_pipe_fd') and self._pipe_fd:
            self._pipe_fd.close()
            self._pipe_fd = None

    def _measure_fps(self, warmup: int = 10, sample: int = 30,
                     fallback: float = 30.0) -> float:
        """Read warmup+sample frames and return measured FPS."""
        try:
            for _ in range(warmup):
                self.cap.read()
            t0 = time.perf_counter()
            for _ in range(sample):
                ret, _ = self.cap.read()
                if not ret:
                    return fallback
            elapsed = time.perf_counter() - t0
            if elapsed <= 0:
                return fallback
            return sample / elapsed
        except Exception:
            return fallback

    def set_frame_callback(self, callback: Callable[[np.ndarray], None]) -> None:
        """Set callback for frame processing."""
        self.frame_callback = callback
