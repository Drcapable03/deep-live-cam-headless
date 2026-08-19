# --- START OF FILE globals.py ---

from __future__ import annotations

import os
import threading

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
WORKFLOW_DIR = os.path.join(ROOT_DIR, "workflow")
dml_lock = threading.Lock()

file_types = [
    ("Image", ("*.png", "*.jpg", "*.jpeg", "*.gif", "*.bmp")),
    ("Video", ("*.mp4", "*.mkv")),
]

# Face Mapping Data
source_target_map: List[Dict[str, Any]] = [] # Stores detailed map for image/video processing
simple_map: Dict[str, Any] = {}             # Stores simplified map (embeddings/faces) for live/simple mode

# Paths
source_path: str | None = None
target_path: str | None = None
output_path: str | None = None

# Processing Options
frame_processors: List[str] = []
keep_fps: bool = True
keep_audio: bool = True
keep_frames: bool = False
many_faces: bool = False         # Process all detected faces with default source
map_faces: bool = False          # Use source_target_map or simple_map for specific swaps
poisson_blend: bool = False      # Enable Poisson Blending for smoother face swaps
color_correction: bool = False   # Enable color correction (implementation specific)
nsfw_filter: bool = False

# Video Output Options
video_encoder: str | None = None
video_quality: int | None = None # Typically a CRF value or bitrate

# Live Mode Options
live_mirror: bool = False
live_resizable: bool = True
camera_input_combobox: Any | None = None # Placeholder for UI element if needed
webcam_preview_running: bool = False
show_fps: bool = False

# System Configuration
max_memory: int | None = None        # Memory limit in GB? (Needs clarification)
execution_providers: List[str] = []  # e.g., ['CUDAExecutionProvider', 'CPUExecutionProvider']
execution_threads: int | None = None # Number of threads for CPU execution
headless: bool | None = None         # Run without UI?
log_level: str = "error"             # Logging level (e.g., 'debug', 'info', 'warning', 'error')

# Face Processor UI Toggles (Example)
fp_ui: Dict[str, bool] = {"face_enhancer": False, "face_enhancer_gpen256": False, "face_enhancer_gpen512": False}

# Face Swapper Specific Options
face_swapper_enabled: bool = True # General toggle for the swapper processor
opacity: float = 1.0              # Blend factor for the swapped face (0.0-1.0)
sharpness: float = 0.0            # Sharpness enhancement for swapped face (0.0-1.0+)

# Face Masking Options (2.7+ for natural feature preservation)
eyes_mask: bool = False             # Enable eyes area masking/preservation
mouth_mask: bool = False           # Enable mouth area masking/pasting
show_mouth_mask_box: bool = False  # Visualize the mouth mask area (for debugging)
eyebrows_mask: bool = False        # Enable eyebrows area masking/preservation
mask_feather_ratio: int = 12       # Denominator for feathering calculation (higher = smaller feather)
mask_down_size: float = 0.1        # Expansion factor for lower lip mask (relative)
mask_size: float = 1.0             # Expansion factor for upper lip mask (relative)
mouth_mask_size: float = 0.0       # Mouth mask size (0-100; 0=off, 100=mouth to chin)
eyes_mask_size: float = 0.0        # Eyes mask size (0-100; 0=off, 100=full eyes)
eyebrows_mask_size: float = 0.0    # Eyebrows mask size (0-100; 0=off, 100=full brows)
face_masking_mode: str = "default" # Masking mode: "default", "soft", "aggressive"

# --- START: Added for Frame Interpolation ---
enable_interpolation: bool = True # Toggle temporal smoothing
interpolation_weight: float = 0  # Blend weight for current frame (0.0-1.0). Lower=smoother.
# --- END: Added for Frame Interpolation ---

# --- START: Added for Headless Streaming ---
# Input source for headless live mode: '0' (camera index), '/dev/video0', 'video.mp4', 'pipe', 'rtmp://...'
input_source: str | None = None
# Output stream URL: 'rtmp://server/live/stream', 'udp://...', or 'pipe:///tmp/frame_pipe'
stream_output: str | None = None
# Stream resolution
stream_width: int = 1280
stream_height: int = 720
stream_fps: int = 30
# Stream video quality (CRF for software encoders)
stream_quality: int = 23
# Stream encoder: 'libx264', 'h264_nvenc', 'copy'
stream_encoder: str = 'libx264'
# Resolution decoupling (Phase 1): face detection/swap/enhance run at this
# reduced resolution, then the result is upscaled to stream_size. 0 = auto.
# Default auto rule (headless_live): process at the input resolution, capped
# at 854x480 (the FACELESS 480p sweet spot — inswapper warp + GPEN cost
# scale with input resolution, not model size).
process_width: int = 0
process_height: int = 0
# Phase 2: virtual camera output + audio passthrough
virtual_cam_name: str | None = None
stream_audio_source: str | None = None
# Phase 3: FP16 acceleration, post-processing, quality presets, adaptive res
fp16: bool = True                       # FP16 models on CUDA Tensor-Core GPUs
sharpen_strength: float = 0.0           # 0-1+ unsharp strength on swapped faces
blend_opacity: float = 1.0              # 0-1 blend swapped face over original
quality_preset: str = "normal"          # 'normal' | 'high' (bundled defaults)
adaptive_resolution: bool = False       # auto-degrade process res to hold fps
adaptive_scale: float = 1.0             # current adaptive resolution scale
# Named pipe path for raw frame output (alternative to RTMP)
pipe_output_path: str | None = None
# --- END: Added for Headless Streaming ---

# --- END OF FILE globals.py ---
