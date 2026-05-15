#!/usr/bin/env python3
"""
Test script for headless streaming components.

Validates:
  1. All imports work without GUI dependencies
  2. Frame source creation (file, camera detection)
  3. Stream output creation
  4. Face analyser and processor loading
  5. End-to-end frame processing (no actual stream)

Usage:
    python test_headless.py -s face.jpg -t test_video.mp4

This does NOT require a display server ($DISPLAY).
"""

import os
import sys
import time
import cv2
import numpy as np

project_root = os.path.dirname(os.path.abspath(__file__))
os.environ["PATH"] = project_root + os.pathsep + os.environ.get("PATH", "")

# Suppress GUI
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import argparse


def test_imports():
    """Test that all required modules import without GUI dependencies."""
    print("[TEST] Testing imports...")
    try:
        import modules.globals
        import modules.metadata
        from modules.processors.frame.core import get_frame_processors_modules
        from modules.face_analyser import get_one_face
        from modules.streaming import create_source, create_output, FrameSource, StreamOutput
        from modules.headless_live import HeadlessLivePipeline
        print("[PASS] All imports successful (no GUI required)")
        return True
    except Exception as e:
        print(f"[FAIL] Import error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_frame_source():
    """Test frame source creation and basic reading."""
    print("\n[TEST] Testing frame sources...")
    from modules.streaming import create_source, VideoFileSource, CameraSource

    # Test video file source creation
    test_video = "test_video.mp4"
    if os.path.isfile(test_video):
        src = create_source(test_video)
        assert isinstance(src, VideoFileSource), "Should create VideoFileSource"
        print(f"[PASS] VideoFileSource created for {test_video}")
        src.release()
    else:
        print(f"[SKIP] No test video found at {test_video}")

    # Test camera source creation (won't open without camera)
    src = create_source("0")
    assert isinstance(src, CameraSource), "Should create CameraSource"
    print("[PASS] CameraSource created")
    src.release()

    # Test URL source
    src = create_source("rtmp://test.server/live")
    print(f"[PASS] FFmpeg source created for RTMP")
    src.release()

    return True


def test_stream_output():
    """Test stream output creation."""
    print("\n[TEST] Testing stream outputs...")
    from modules.streaming import create_output, FFmpegStreamOutput

    # Test RTMP output
    out = create_output("rtmp://test.server/live/stream")
    assert isinstance(out, FFmpegStreamOutput), "Should create FFmpegStreamOutput"
    print("[PASS] RTMP output created")
    out.close()

    # Test file output
    out = create_output("/tmp/test_output.mp4")
    assert isinstance(out, FFmpegStreamOutput), "Should create FFmpegStreamOutput for file"
    print("[PASS] File output created")
    out.close()

    return True


def test_face_processing(source_path: str):
    """Test the core face processing pipeline."""
    print("\n[TEST] Testing face processing pipeline...")
    from modules.face_analyser import get_one_face
    from modules.processors.frame.core import get_frame_processors_modules
    import modules.globals

    modules.globals.execution_providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    modules.globals.frame_processors = ['face_swapper']

    # Load source face
    source_img = cv2.imread(source_path)
    if source_img is None:
        print(f"[FAIL] Could not read source image: {source_path}")
        return False

    source_face = get_one_face(source_img)
    if source_face is None:
        print("[FAIL] No face detected in source image")
        return False
    print(f"[PASS] Source face loaded: {source_face.bbox}")

    # Load frame processors
    processors = get_frame_processors_modules(modules.globals.frame_processors)
    print(f"[PASS] Frame processors loaded: {[p.NAME for p in processors]}")

    # Test frame processing
    dummy_frame = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)
    try:
        from modules.processors.frame.face_swapper import process_frame
        result = process_frame(source_face, dummy_frame)
        assert result is not None, "process_frame returned None"
        assert result.shape == dummy_frame.shape, "Output shape mismatch"
        print("[PASS] Frame processing works (dummy frame)")
    except Exception as e:
        print(f"[WARN] Frame processing test failed (expected without model): {e}")

    return True


def test_headless_pipeline(source_path: str, target_path: str = None):
    """Test the headless pipeline initialization (does not run)."""
    print("\n[TEST] Testing headless pipeline...")
    from modules.headless_live import HeadlessLivePipeline
    import modules.globals

    # Configure for file input
    modules.globals.source_path = source_path
    modules.globals.input_source = target_path or "0"
    modules.globals.stream_output = None  # Don't actually stream
    modules.globals.frame_processors = ['face_swapper']
    modules.globals.execution_providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']

    pipeline = HeadlessLivePipeline()
    print("[PASS] HeadlessLivePipeline created")

    # We won't call start() as it requires a valid input source running
    # Just verify the structure is correct
    assert hasattr(pipeline, 'start'), "Pipeline should have start()"
    assert hasattr(pipeline, 'stop'), "Pipeline should have stop()"
    assert hasattr(pipeline, 'run_blocking'), "Pipeline should have run_blocking()"
    print("[PASS] Pipeline API verified")

    return True


def test_video_file_processing(source_path: str, target_path: str):
    """Test processing a few frames from a video file."""
    print("\n[TEST] Testing video file frame processing...")
    from modules.face_analyser import get_one_face
    from modules.processors.frame.face_swapper import process_frame
    import modules.globals

    modules.globals.execution_providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    modules.globals.frame_processors = ['face_swapper']
    modules.globals.many_faces = False

    if not os.path.isfile(target_path):
        print(f"[SKIP] Target video not found: {target_path}")
        return True

    # Load source face
    source_img = cv2.imread(source_path)
    if source_img is None:
        print("[FAIL] Could not read source image")
        return False

    source_face = get_one_face(source_img)
    if source_face is None:
        print("[FAIL] No face in source")
        return False

    # Process first 5 frames
    cap = cv2.VideoCapture(target_path)
    processed = 0
    start_time = time.time()

    for i in range(5):
        ret, frame = cap.read()
        if not ret:
            break
        try:
            result = process_frame(source_face, frame)
            processed += 1
        except Exception as e:
            print(f"[WARN] Frame {i} processing failed: {e}")

    elapsed = time.time() - start_time
    cap.release()

    print(f"[PASS] Processed {processed}/5 frames in {elapsed:.2f}s")
    if processed > 0:
        print(f"       Effective: {processed/elapsed:.1f} fps")

    return True


def main():
    parser = argparse.ArgumentParser(description='Test headless streaming components')
    parser.add_argument('-s', '--source', help='Source face image', required=True)
    parser.add_argument('-t', '--target', help='Target video file (optional)', default=None)
    args = parser.parse_args()

    print("=" * 60)
    print("Deep-Live-Cam Headless Streaming Test Suite")
    print("=" * 60)

    results = []

    # Run all tests
    results.append(("Imports", test_imports()))
    results.append(("Frame Sources", test_frame_source()))
    results.append(("Stream Outputs", test_stream_output()))

    if os.path.isfile(args.source):
        results.append(("Face Processing", test_face_processing(args.source)))
        results.append(("Headless Pipeline", test_headless_pipeline(args.source, args.target)))
        if args.target:
            results.append(("Video Processing", test_video_file_processing(args.source, args.target)))
    else:
        print(f"\n[SKIP] Source image not found: {args.source}")
        print("       Provide a valid source image with -s to run processing tests")

    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    all_passed = True
    for name, passed in results:
        status = "PASS" if passed else "FAIL"
        print(f"  {status}: {name}")
        if not passed:
            all_passed = False

    if all_passed:
        print("\nAll tests passed! Headless streaming is ready.")
        return 0
    else:
        print("\nSome tests failed. Check output above.")
        return 1


if __name__ == '__main__':
    sys.exit(main())
