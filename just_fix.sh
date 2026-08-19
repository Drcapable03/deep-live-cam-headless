# Quick fix: Add auto-restart to headless_live.py
python3 << 'EOF'
path = "/workspace/deep-live-cam-headless/modules/headless_live.py"

with open(path) as f:
    c = f.read()

# Replace the _output_loop method
old = '''    def _output_loop(self):
        """Write processed frames to the output stream."""
        while not self.stop_event.is_set():
            try:
                frame = self.processed_queue.get(timeout=0.05)
            except queue.Empty:
                continue

            if self.output:
                success = self.output.write(frame)
                if success:
                    self.stats["frames_written"] += 1'''

new = '''    def _output_loop(self):
        """Write processed frames to the output stream. Auto-restart FFmpeg if it crashes."""
        restart_count = 0
        while not self.stop_event.is_set():
            try:
                frame = self.processed_queue.get(timeout=0.05)
            except queue.Empty:
                # Check if FFmpeg died and restart it
                if self.output and hasattr(self.output, 'process') and self.output.process:
                    if self.output.process.poll() is not None and restart_count < 10:
                        print(f"[HEADLESS] FFmpeg died, restarting... (attempt {restart_count + 1})")
                        try:
                            self.output.close()
                        except Exception:
                            pass
                        if self.output.start():
                            restart_count += 1
                            print("[HEADLESS] FFmpeg restarted successfully")
                        else:
                            print("[HEADLESS] FFmpeg restart failed")
                            self.output = None
                continue

            if self.output:
                success = self.output.write(frame)
                if success:
                    self.stats["frames_written"] += 1
                    restart_count = 0  # Reset counter on successful write'''

c = c.replace(old, new)

with open(path, "w") as f:
    f.write(c)

import ast
ast.parse(c)
print("Auto-restart added to _output_loop")
EOF
