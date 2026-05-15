"""
Headless live face-swap pipeline for Deep-Live-Cam.

This module replaces the GUI-based live preview (modules/ui.py) with a
pure headless pipeline:

    Input Source -> Face Detection -> Face Swap -> Enhancement -> Output Stream

No Tkinter, no OpenCV GUI, no display server required. Designed for cloud
GPU environments and streaming workflows.
"""

import os
import sys
import cv2
import numpy as np
import time
import queue
import threading
from typing import Optional, Callable, List

import modules.globals
from modules.face_analyser import (
    get_one_face,
    get_many_faces,
    detect_one_face_fast,
    detect_many_faces_fast,
    get_face_analyser,
)
from modules.processors.frame.core import get_frame_processors_modules
from modules.streaming import (
    FrameSource, StreamOutput,
    create_source, create_output,
    CameraSource, VideoFileSource, PipeSource, FFmpegInputSource,
    FFmpegStreamOutput, PipeOutput,
)
from modules.processors.frame.face_masking import (
    create_lower_mouth_mask,
    create_eyes_mask,
    create_eyebrows_mask,
    create_face_mask,
    apply_mask_area,
)
from modules.gpu_processing import gpu_flip


class HeadlessLivePipeline:
    """Headless real-time face-swap pipeline.

    Usage:
        pipeline = HeadlessLivePipeline()
        pipeline.start(source_face_path="face.jpg")
        # Runs until interrupted or source ends
        pipeline.stop()
    """

    def __init__(self):
        self.source: Optional[FrameSource] = None
        self.output: Optional[StreamOutput] = None
        self.source_face = None
        self.frame_processors = []
        self.is_running = False
        self.stop_event = threading.Event()

        # Threads
        self.capture_thread: Optional[threading.Thread] = None
        self.process_thread: Optional[threading.Thread] = None
        self.output_thread: Optional[threading.Thread] = None

        # Queues for decoupled pipeline stages
        self.capture_queue: queue.Queue = queue.Queue(maxsize=2)
        self.processed_queue: queue.Queue = queue.Queue(maxsize=2)

        # Stats
        self.stats = {
            "frames_captured": 0,
            "frames_processed": 0,
            "frames_written": 0,
            "fps": 0.0,
            "start_time": 0.0,
        }

    # ------------------------------------------------------------------
    # Stage 1: Capture thread
    # ------------------------------------------------------------------
    def _capture_loop(self):
        """Read frames from source, drop stale ones."""
        while not self.stop_event.is_set():
            ret, frame = self.source.read()
            if not ret or frame is None:
                print("[HEADLESS] Source ended or read failed")
                self.stop_event.set()
                break

            try:
                self.capture_queue.put_nowait(frame)
            except queue.Full:
                # Drop oldest, keep newest
                try:
                    self.capture_queue.get_nowait()
                except queue.Empty:
                    pass
                try:
                    self.capture_queue.put_nowait(frame)
                except queue.Full:
                    pass

            self.stats["frames_captured"] += 1

    # ------------------------------------------------------------------
    # Stage 2: Processing thread (the core face-swap logic)
    # ------------------------------------------------------------------
    def _processing_loop(self, camera_fps: float = 30.0):
        """Process frames: detect faces, swap, enhance, post-process.

        This is the extracted frame-processing pipeline from ui.py's
        _processing_thread_func, adapted for headless operation.
        """
        frame_processors = get_frame_processors_modules(modules.globals.frame_processors)
        last_source_path = None
        prev_time = time.time()
        frame_count = 0
        fps = 0.0
        det_count = 0
        cached_target_face = None
        cached_many_faces = None

        # Detect every N frames (~80ms interval)
        det_interval = max(1, round(camera_fps * 0.08))

        while not self.stop_event.is_set():
            try:
                frame = self.capture_queue.get(timeout=0.05)
            except queue.Empty:
                continue

            temp_frame = frame

            if modules.globals.live_mirror:
                temp_frame = gpu_flip(temp_frame, 1)

            # Re-load source face if path changed
            if modules.globals.source_path and modules.globals.source_path != last_source_path:
                last_source_path = modules.globals.source_path
                self.source_face = get_one_face(cv2.imread(modules.globals.source_path))
                if self.source_face is None:
                    print("[HEADLESS] Warning: No face detected in source image")

            # ---- Face detection (throttled) ----
            det_count += 1
            if det_count % det_interval == 0:
                if modules.globals.many_faces:
                    cached_target_face = None
                    cached_many_faces = detect_many_faces_fast(temp_frame)
                else:
                    cached_target_face = detect_one_face_fast(temp_frame)
                    cached_many_faces = None

            # Build face list for enhancers
            _cached_faces = None
            if cached_many_faces:
                _cached_faces = cached_many_faces
            elif cached_target_face is not None:
                _cached_faces = [cached_target_face]

            # ---- Run frame processors ----
            for frame_processor in frame_processors:
                try:
                    if frame_processor.NAME == "DLC.FACE-ENHANCER":
                        if modules.globals.fp_ui.get("face_enhancer", False):
                            temp_frame = frame_processor.process_frame(
                                None, temp_frame, detected_faces=_cached_faces)

                    elif frame_processor.NAME == "DLC.FACE-ENHANCER-GPEN256":
                        if modules.globals.fp_ui.get("face_enhancer_gpen256", False):
                            temp_frame = frame_processor.process_frame(
                                None, temp_frame, detected_faces=_cached_faces)

                    elif frame_processor.NAME == "DLC.FACE-ENHANCER-GPEN512":
                        if modules.globals.fp_ui.get("face_enhancer_gpen512", False):
                            temp_frame = frame_processor.process_frame(
                                None, temp_frame, detected_faces=_cached_faces)

                    elif frame_processor.NAME == "DLC.FACE-SWAPPER":
                        swapped_bboxes = []
                        _all_faces = cached_many_faces if (modules.globals.many_faces and cached_many_faces) else ([cached_target_face] if cached_target_face else [])

                        for t_face in _all_faces:
                            # --- Face Masking (2.7+): Preserve natural features ---
                            original_frame = temp_frame.copy()

                            # Mouth mask: preserve original lip movement
                            if modules.globals.mouth_mask or modules.globals.mouth_mask_size > 0:
                                mouth_mask, mouth_cutout, mouth_box, mouth_polygon = create_lower_mouth_mask(t_face, temp_frame)
                                modules.globals.mouth_mask_size = min(100, modules.globals.mouth_mask_size + 30)

                            # Eyes mask: preserve original eye movement/blinks
                            if modules.globals.eyes_mask or modules.globals.eyes_mask_size > 0:
                                eyes_mask, eyes_cutout, eyes_box, eyes_polygon = create_eyes_mask(t_face, temp_frame)
                                modules.globals.eyes_mask_size = min(100, modules.globals.eyes_mask_size + 15)

                            # Eyebrows mask: preserve original eyebrow expressions
                            if modules.globals.eyebrows_mask or modules.globals.eyebrows_mask_size > 0:
                                eyebrows_mask, eyebrows_cutout, eyebrows_box, eyebrows_polygon = create_eyebrows_mask(t_face, temp_frame)
                                modules.globals.eyebrows_mask_size = min(100, modules.globals.eyebrows_mask_size + 25)

                            # Perform face swap
                            temp_frame = frame_processor.swap_face(self.source_face, t_face, temp_frame)
                            if hasattr(t_face, 'bbox') and t_face.bbox is not None:
                                swapped_bboxes.append(t_face.bbox.astype(int))

                            # --- Apply masks to restore original features ---
                            face_mask = create_face_mask(t_face, temp_frame)

                            if (modules.globals.mouth_mask or modules.globals.mouth_mask_size > 0) and mouth_cutout is not None:
                                temp_frame = apply_mask_area(temp_frame, mouth_cutout, mouth_box, face_mask, mouth_polygon)
                            if (modules.globals.eyes_mask or modules.globals.eyes_mask_size > 0) and eyes_cutout is not None:
                                temp_frame = apply_mask_area(temp_frame, eyes_cutout, eyes_box, face_mask, eyes_polygon)
                            if (modules.globals.eyebrows_mask or modules.globals.eyebrows_mask_size > 0) and eyebrows_cutout is not None:
                                temp_frame = apply_mask_area(temp_frame, eyebrows_cutout, eyebrows_box, face_mask, eyebrows_polygon)

                        # Post-processing (sharpening, interpolation)
                        temp_frame = frame_processor.apply_post_processing(temp_frame, swapped_bboxes)

                    else:
                        temp_frame = frame_processor.process_frame(self.source_face, temp_frame)

                except Exception as e:
                    print(f"[HEADLESS] Processor {frame_processor.NAME} error: {e}")

            # ---- FPS calculation ----
            current_time = time.time()
            frame_count += 1
            if current_time - prev_time >= 0.5:
                fps = frame_count / (current_time - prev_time)
                frame_count = 0
                prev_time = current_time

            if modules.globals.show_fps:
                cv2.putText(temp_frame, f"FPS: {fps:.1f}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            self.stats["fps"] = fps
            self.stats["frames_processed"] += 1

            # ---- Enqueue processed frame ----
            try:
                self.processed_queue.put_nowait(temp_frame)
            except queue.Full:
                try:
                    self.processed_queue.get_nowait()
                except queue.Empty:
                    pass
                try:
                    self.processed_queue.put_nowait(temp_frame)
                except queue.Full:
                    pass

    # ------------------------------------------------------------------
    # Stage 3: Output thread
    # ------------------------------------------------------------------
    def _output_loop(self):
        """Write processed frames to the output stream."""
        while not self.stop_event.is_set():
            try:
                frame = self.processed_queue.get(timeout=0.05)
            except queue.Empty:
                continue

            if self.output:
                success = self.output.write(frame)
                if success:
                    self.stats["frames_written"] += 1

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def start(self, source_face_path: Optional[str] = None,
              input_source: Optional[str] = None,
              output_url: Optional[str] = None,
              width: int = 1280, height: int = 720, fps: int = 30) -> bool:
        """Start the headless pipeline.

        Args:
            source_face_path: Path to source face image (overrides globals.source_path)
            input_source: Input source string (overrides globals.input_source)
            output_url: Output stream URL (overrides globals.stream_output)
            width: Target frame width
            height: Target frame height
            fps: Target frame rate
        """
        # Resolve parameters (CLI args -> function args -> globals)
        if source_face_path:
            modules.globals.source_path = source_face_path
        if input_source:
            modules.globals.input_source = input_source
        if output_url:
            modules.globals.stream_output = output_url

        if not modules.globals.source_path or not os.path.isfile(modules.globals.source_path):
            print("[HEADLESS] ERROR: No valid source face image provided. Use -s/--source")
            return False

        if not modules.globals.input_source:
            print("[HEADLESS] ERROR: No input source provided. Use --input-source")
            return False

        # Pre-load face analyser (required before any face detection)
        print("[HEADLESS] Pre-loading face analyser models...")
        try:
            get_face_analyser()
            print("[HEADLESS] Face analyser ready")
        except Exception as e:
            print(f"[HEADLESS] ERROR: Failed to load face analyser: {e}")
            return False

        # Pre-load face swapper model (avoids first-frame delay)
        if 'face_swapper' in modules.globals.frame_processors:
            print("[HEADLESS] Pre-loading face swapper model...")
            try:
                from modules.processors.frame.face_swapper import get_face_swapper
                if get_face_swapper() is None:
                    print("[HEADLESS] ERROR: Face swapper model failed to load")
                    return False
                print("[HEADLESS] Face swapper ready")
            except Exception as e:
                print(f"[HEADLESS] ERROR: Failed to load face swapper: {e}")
                return False

        # Load source face
        print(f"[HEADLESS] Loading source face: {modules.globals.source_path}")
        source_img = cv2.imread(modules.globals.source_path)
        if source_img is None:
            print("[HEADLESS] ERROR: Could not read source image")
            return False
        self.source_face = get_one_face(source_img)
        if self.source_face is None:
            print("[HEADLESS] ERROR: No face detected in source image")
            return False
        print("[HEADLESS] Source face loaded successfully")

        # Reset interpolation state from previous runs
        from modules.processors.frame import face_swapper
        if hasattr(face_swapper, 'PREVIOUS_FRAME_RESULT'):
            face_swapper.PREVIOUS_FRAME_RESULT = None

        # Initialize frame processors
        self.frame_processors = get_frame_processors_modules(modules.globals.frame_processors)
        for fp in self.frame_processors:
            # NOTE: In live mode, enhancer's pre_start() checks target_path which is None.
            # Enhancers don't need a target file for live processing, so we skip this check.
            if hasattr(fp, 'pre_start'):
                enhancer_names = ("DLC.FACE-ENHANCER", "DLC.FACE-ENHANCER-GPEN256", "DLC.FACE-ENHANCER-GPEN512")
                if fp.NAME in enhancer_names and not modules.globals.target_path:
                    print(f"[HEADLESS] Skipping pre_start for {fp.NAME} (live mode)")
                elif not fp.pre_start():
                    print(f"[HEADLESS] ERROR: Frame processor {fp.NAME} failed pre_start")
                    return False
            print(f"[HEADLESS] Frame processor ready: {fp.NAME}")

        # Create input source
        print(f"[HEADLESS] Initializing input source: {modules.globals.input_source}")
        self.source = create_source(modules.globals.input_source)

        if isinstance(self.source, CameraSource):
            if not self.source.start(width, height, fps):
                return False
            camera_fps = self.source.fps
        elif isinstance(self.source, VideoFileSource):
            if not self.source.start():
                return False
            camera_fps = self.source.fps
        elif isinstance(self.source, PipeSource):
            if not self.source.start():
                return False
            camera_fps = fps
        elif isinstance(self.source, FFmpegInputSource):
            if not self.source.start(width, height):
                return False
            camera_fps = fps
        else:
            camera_fps = fps

        props = self.source.get_properties()
        print(f"[HEADLESS] Input properties: {props}")

        # Create output if configured
        if modules.globals.stream_output:
            print(f"[HEADLESS] Initializing stream output: {modules.globals.stream_output}")
            self.output = create_output(
                modules.globals.stream_output,
                width=modules.globals.stream_width,
                height=modules.globals.stream_height,
                fps=modules.globals.stream_fps,
                encoder=modules.globals.stream_encoder,
                quality=modules.globals.stream_quality,
            )
            if not self.output.start():
                print("[HEADLESS] WARNING: Stream output failed to start, continuing without output")
                self.output = None

        # Start pipeline threads
        self.stop_event.clear()
        self.is_running = True
        self.stats["start_time"] = time.time()

        self.capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.process_thread = threading.Thread(
            target=self._processing_loop, args=(camera_fps,), daemon=True)

        self.capture_thread.start()
        self.process_thread.start()

        if self.output:
            self.output_thread = threading.Thread(target=self._output_loop, daemon=True)
            self.output_thread.start()

        print("[HEADLESS] Pipeline started. Press Ctrl+C to stop.")
        return True

    def run_blocking(self):
        """Block until pipeline stops (Ctrl+C or source ends)."""
        try:
            while self.is_running and not self.stop_event.is_set():
                time.sleep(1.0)
                elapsed = time.time() - self.stats["start_time"]
                if elapsed > 0:
                    print(f"\r[HEADLESS] Cap: {self.stats['frames_captured']} | "
                          f"Proc: {self.stats['frames_processed']} | "
                          f"Out: {self.stats['frames_written']} | "
                          f"FPS: {self.stats['fps']:.1f} | "
                          f"Time: {elapsed:.0f}s", end="", flush=True)
        except KeyboardInterrupt:
            print("\n[HEADLESS] Interrupted by user")
        finally:
            self.stop()

    def stop(self):
        """Stop the pipeline and release all resources."""
        if not self.is_running:
            return

        print("\n[HEADLESS] Stopping pipeline...")
        self.stop_event.set()
        self.is_running = False

        # Wait for threads
        if self.capture_thread:
            self.capture_thread.join(timeout=2.0)
        if self.process_thread:
            self.process_thread.join(timeout=2.0)
        if self.output_thread:
            self.output_thread.join(timeout=2.0)

        # Release resources
        if self.source:
            self.source.release()
        if self.output:
            self.output.close()

        # Reset interpolation state
        from modules.processors.frame import face_swapper
        if hasattr(face_swapper, 'PREVIOUS_FRAME_RESULT'):
            face_swapper.PREVIOUS_FRAME_RESULT = None

        elapsed = time.time() - self.stats["start_time"]
        print(f"\n[HEADLESS] Pipeline stopped. Total time: {elapsed:.1f}s")
        print(f"[HEADLESS] Stats: {self.stats['frames_captured']} captured, "
              f"{self.stats['frames_processed']} processed, "
              f"{self.stats['frames_written']} written")


def run_headless_live():
    """Entry point for headless live streaming mode.

    Called from modules.core when --input-source is provided.
    """
    pipeline = HeadlessLivePipeline()

    success = pipeline.start(
        width=modules.globals.stream_width,
        height=modules.globals.stream_height,
        fps=modules.globals.stream_fps,
    )

    if success:
        pipeline.run_blocking()
    else:
        print("[HEADLESS] Failed to start pipeline")
        sys.exit(1)
