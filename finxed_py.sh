python3 << 'EOF'
# Fix face_enhancer_gpen512.py
path = "/workspace/deep-live-cam-headless/modules/processors/frame/face_enhancer_gpen512.py"

with open(path) as f:
    lines = f.readlines()

# Find and replace the corrupted process_frame function
new_lines = []
skip_until_def = False
for line in lines:
    if "def process_frame(source_face: Face | None, temp_frame: Frame" in line:
        # Replace the entire function
        new_lines.append("def process_frame(source_face: Face | None, temp_frame: Frame, detected_faces=None) -> Frame:\n")
        new_lines.append("    if detected_faces:\n")
        new_lines.append("        if isinstance(detected_faces, list) and len(detected_faces) > 0:\n")
        new_lines.append("            target_face = detected_faces[0]\n")
        new_lines.append("        else:\n")
        new_lines.append("            target_face = detected_faces\n")
        new_lines.append("    else:\n")
        new_lines.append("        target_face = get_one_face(temp_frame)\n")
        new_lines.append("    if target_face is None:\n")
        new_lines.append("        return temp_frame\n")
        new_lines.append("    return enhance_face(temp_frame, target_face)\n")
        skip_until_def = True
        continue
    if skip_until_def:
        # Skip old function body until we hit the next function definition or blank line after function
        if line.strip().startswith("def ") or (line.strip() == "" and len(new_lines) > 0 and not new_lines[-1].strip().startswith("def")):
            skip_until_def = False
            new_lines.append(line)
        continue
    new_lines.append(line)

with open(path, "w") as f:
    f.writelines(new_lines)

print("face_enhancer_gpen512.py fixed")

# Fix face_enhancer_gpen256.py the same way
path2 = "/workspace/deep-live-cam-headless/modules/processors/frame/face_enhancer_gpen256.py"

with open(path2) as f:
    lines = f.readlines()

new_lines = []
skip_until_def = False
for line in lines:
    if "def process_frame(source_face: Face | None, temp_frame: Frame" in line:
        new_lines.append("def process_frame(source_face: Face | None, temp_frame: Frame, detected_faces=None) -> Frame:\n")
        new_lines.append("    if detected_faces:\n")
        new_lines.append("        if isinstance(detected_faces, list) and len(detected_faces) > 0:\n")
        new_lines.append("            target_face = detected_faces[0]\n")
        new_lines.append("        else:\n")
        new_lines.append("            target_face = detected_faces\n")
        new_lines.append("    else:\n")
        new_lines.append("        target_face = get_one_face(temp_frame)\n")
        new_lines.append("    if target_face is None:\n")
        new_lines.append("        return temp_frame\n")
        new_lines.append("    return enhance_face(temp_frame, target_face)\n")
        skip_until_def = True
        continue
    if skip_until_def:
        if line.strip().startswith("def ") or (line.strip() == "" and len(new_lines) > 0 and not new_lines[-1].strip().startswith("def")):
            skip_until_def = False
            new_lines.append(line)
        continue
    new_lines.append(line)

with open(path2, "w") as f:
    f.writelines(new_lines)

print("face_enhancer_gpen256.py fixed")
EOF
