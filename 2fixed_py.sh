python3 << 'EOF'
path = "/workspace/deep-live-cam-headless/modules/processors/frame/face_enhancer_gpen512.py"

with open(path) as f:
    content = f.read()

# Replace literal \n with actual newlines
content = content.replace('\\n', '\n')

# Now fix the process_frame function properly
old_func = """def process_frame(source_face: Face | None, temp_frame: Frame, detected_faces=None) -> Frame:
    if detected_faces:
        if isinstance(detected_faces, list) and len(detected_faces) > 0:
            target_face = detected_faces[0]
        else:
            target_face = detected_faces
    else:
        target_face = get_one_face(temp_frame)
    if target_face is None:
        return temp_frame
    return enhance_face(temp_frame, target_face)"""

# Check if it's already correct
if old_func.strip() in content:
    print("Function already correct after \\n fix")
else:
    print("Function still corrupted, need full rewrite")
    # Find and remove the corrupted process_frame function
    lines = content.split('\n')
    new_lines = []
    skip = False
    for line in lines:
        if 'def process_frame(source_face: Face | None, temp_frame: Frame' in line:
            # Insert correct function
            new_lines.append('def process_frame(source_face: Face | None, temp_frame: Frame, detected_faces=None) -> Frame:')
            new_lines.append('    if detected_faces:')
            new_lines.append('        if isinstance(detected_faces, list) and len(detected_faces) > 0:')
            new_lines.append('            target_face = detected_faces[0]')
            new_lines.append('        else:')
            new_lines.append('            target_face = detected_faces')
            new_lines.append('    else:')
            new_lines.append('        target_face = get_one_face(temp_frame)')
            new_lines.append('    if target_face is None:')
            new_lines.append('        return temp_frame')
            new_lines.append('    return enhance_face(temp_frame, target_face)')
            skip = True
            continue
        if skip:
            # Skip until we hit the next function or class
            if line.strip().startswith('def ') or line.strip().startswith('class '):
                skip = False
                new_lines.append(line)
            continue
        new_lines.append(line)
    content = '\n'.join(new_lines)

with open(path, "w") as f:
    f.write(content)

# Verify
import ast
ast.parse(open(path).read())
print("gpen512 FIXED and syntax valid")
EOF
