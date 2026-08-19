# Check current stderr setting
grep -n "stderr=" /workspace/deep-live-cam-headless/modules/streaming.py

# Fix: Change stderr from PIPE to DEVNULL
sed -i 's/stderr=subprocess.PIPE/stderr=subprocess.DEVNULL/' /workspace/deep-live-cam-headless/modules/streaming.py

# Verify
grep -n "stderr=" /workspace/deep-live-cam-headless/modules/streaming.py
