python3 << 'PYEOF'
path = "/workspace/deep-live-cam-headless/modules/core.py"

with open(path) as f:
    lines = f.readlines()

# Fix the corrupted lines 254-281 (0-indexed: 253-280)
# The sed commands inserted "continue" in wrong places throughout

fixes = {
    255: "            continue\n",      # line 256 - pre_start() check (this one is actually OK)
    262: "            return\n",       # line 263 - NSFW check inside image block (was continue)
    263: "        try:\n",             # line 264 - restore indentation
    276: "        return\n",           # line 277 - end of image processing (was continue)
    279: "        return\n",           # line 280 - NSFW check in video section (was continue)
}

for lineno, content in fixes.items():
    if lineno < len(lines):
        lines[lineno] = content

with open(path, "w") as f:
    f.writelines(lines)

print("Fixed. Verifying lines 259-282:")
print("---")
for i in range(258, 282):
    print(f"{i+1}: {lines[i]}", end="")
PYEOF
