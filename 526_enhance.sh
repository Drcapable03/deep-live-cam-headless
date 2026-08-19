#cd /workspace/deep-live-cam-headless

# Fix 1: Fix GPEN 512 download URL
sed -i 's|releases/download/GPEN-BFR/GPEN-BFR-512.onnx|releases/download/Models/GPEN-BFR-512.onnx|' modules/processors/frame/face_enhancer_gpen512.py

# Fix 2: Fix GPEN 256 download URL
sed -i 's|releases/download/GPEN-BFR/GPEN-BFR-256.onnx|releases/download/Models/GPEN-BFR-256.onnx|' modules/processors/frame/face_enhancer_gpen256.py

# Fix 3: Make core.py not crash on failed pre_check
sed -i 's|if not frame_processor.pre_check():$|if not frame_processor.pre_check():|' modules/core.py
sed -i 's|return$|continue|' modules/core.py

# Verify fixes
grep "download/Models" modules/processors/frame/face_enhancer_gpen512.py
grep "continue" modules/core.py
