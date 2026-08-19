"""Face masking module for Deep-Live-Cam 2.7+

Provides mouth mask, eyes mask, eyebrows mask, and color transfer
for natural face swapping with feature preservation.

This is what makes the face swap look "alive" -- it preserves the
original person's lip movement, eye blinks, and eyebrow expressions
while swapping the face identity.
"""

import cv2
import numpy as np
from modules.typing import Face, Frame
import modules.globals
from modules.gpu_processing import gpu_gaussian_blur, gpu_resize, gpu_cvt_color


def apply_color_transfer(source, target):
    """Apply color transfer from target to source image using LAB color space."""
    source_f32 = source.astype(np.float32) / 255.0
    target_f32 = target.astype(np.float32) / 255.0

    source_lab = cv2.cvtColor(source_f32, cv2.COLOR_BGR2LAB)
    target_lab = cv2.cvtColor(target_f32, cv2.COLOR_BGR2LAB)

    source_mean, source_std = cv2.meanStdDev(source_lab)
    target_mean, target_std = cv2.meanStdDev(target_lab)

    source_mean = source_mean.reshape(1, 1, 3).astype(np.float32)
    source_std = np.maximum(source_std.reshape(1, 1, 3), 1e-6).astype(np.float32)
    target_mean = target_mean.reshape(1, 1, 3).astype(np.float32)
    target_std = target_std.reshape(1, 1, 3).astype(np.float32)

    result_lab = (source_lab - source_mean) * (target_std / source_std) + target_mean
    result_bgr = cv2.cvtColor(result_lab, cv2.COLOR_LAB2BGR)
    return np.clip(result_bgr * 255.0, 0, 255).astype(np.uint8)


def create_face_mask(face: Face, frame: Frame) -> np.ndarray:
    """Create a full-face mask with feathered edges for seamless blending."""
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    landmarks = face.landmark_2d_106
    if landmarks is not None:
        landmarks = landmarks.astype(np.int32)
        face_outline = landmarks[0:33]
        hull = cv2.convexHull(face_outline)

        # Calculate padding (5% of face width)
        padding = int(
            np.linalg.norm(face_outline[0] - face_outline[16]) * 0.05
        )

        # Vectorized hull padding -- expand each point outward from center
        center = np.mean(face_outline, axis=0, dtype=np.float32)
        hull_pts = hull.reshape(-1, 2).astype(np.float32)
        directions = hull_pts - center
        norms = np.linalg.norm(directions, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-6)
        directions /= norms
        hull_padded = (hull_pts + directions * padding).astype(np.int32)

        cv2.fillConvexPoly(mask, hull_padded, 255)
        mask = gpu_gaussian_blur(mask, (5, 5), 3)

    return mask


def create_lower_mouth_mask(face: Face, frame: Frame):
    """Create a mask for the lower mouth/lip area to preserve natural lip movement.

    Returns: (mask, mouth_cutout, mouth_box, lower_lip_polygon)
    """
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    mouth_cutout = None
    lower_lip_polygon = None
    mouth_box = (0, 0, 0, 0)

    landmarks = face.landmark_2d_106
    if landmarks is not None:
        # Use outer mouth landmarks (52-63) to capture the lips
        lower_lip_order = list(range(52, 64))

        if max(lower_lip_order) >= landmarks.shape[0]:
            return mask, mouth_cutout, mouth_box, lower_lip_polygon

        lower_lip_landmarks = landmarks[lower_lip_order].astype(np.float32)

        # Calculate center
        center = np.mean(lower_lip_landmarks, axis=0)

        # Expand landmarks using mouth_mask_size
        expansion_factor = (
            1 + modules.globals.mask_down_size * modules.globals.mouth_mask_size
        )
        expanded_landmarks = (lower_lip_landmarks - center) * expansion_factor + center
        expanded_landmarks = expanded_landmarks.astype(np.int32)

        # Calculate bounding box
        min_x, min_y = np.min(expanded_landmarks, axis=0)
        max_x, max_y = np.max(expanded_landmarks, axis=0)

        # Add padding
        padding = int((max_x - min_x) * 0.1)
        min_x = max(0, min_x - padding)
        min_y = max(0, min_y - padding)
        max_x = min(frame.shape[1], max_x + padding)
        max_y = min(frame.shape[0], max_y + padding)

        if max_x <= min_x or max_y <= min_y:
            if (max_x - min_x) <= 1:
                max_x = min_x + 1
            if (max_y - min_y) <= 1:
                max_y = min_y + 1

        # Create mask ROI
        mask_roi = np.zeros((max_y - min_y, max_x - min_x), dtype=np.uint8)
        polygon_relative_to_roi = expanded_landmarks - [min_x, min_y]
        cv2.fillPoly(mask_roi, [polygon_relative_to_roi], 255)
        mask_roi = gpu_gaussian_blur(mask_roi, (15, 15), 5)
        mask[min_y:max_y, min_x:max_x] = mask_roi

        mouth_cutout = frame[min_y:max_y, min_x:max_x].copy()
        lower_lip_polygon = expanded_landmarks
        mouth_box = (min_x, min_y, max_x, max_y)

    return mask, mouth_cutout, mouth_box, lower_lip_polygon


def create_eyes_mask(face: Face, frame: Frame):
    """Create a mask for the eye area to preserve natural eye movement and blinks.

    Returns: (mask, eyes_cutout, eyes_box, eyes_polygon)
    """
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    eyes_cutout = None
    eyes_box = (0, 0, 0, 0)
    eyes_polygon = None

    landmarks = face.landmark_2d_106
    if landmarks is not None:
        left_eye = landmarks[87:96]
        right_eye = landmarks[33:42]

        left_eye_center = np.mean(left_eye, axis=0).astype(np.int32)
        right_eye_center = np.mean(right_eye, axis=0).astype(np.int32)

        def get_eye_dimensions(eye_points):
            x_coords = eye_points[:, 0]
            y_coords = eye_points[:, 1]
            width = int((np.max(x_coords) - np.min(x_coords)) *
                        (1 + modules.globals.mask_down_size * modules.globals.eyes_mask_size))
            height = int((np.max(y_coords) - np.min(y_coords)) *
                         (1 + modules.globals.mask_down_size * modules.globals.eyes_mask_size))
            return width, height

        left_width, left_height = get_eye_dimensions(left_eye)
        right_width, right_height = get_eye_dimensions(right_eye)

        padding = int(max(left_width, right_width) * 0.2)

        min_x = min(left_eye_center[0] - left_width // 2,
                    right_eye_center[0] - right_width // 2) - padding
        max_x = max(left_eye_center[0] + left_width // 2,
                    right_eye_center[0] + right_width // 2) + padding
        min_y = min(left_eye_center[1] - left_height // 2,
                    right_eye_center[1] - right_height // 2) - padding
        max_y = max(left_eye_center[1] + left_height // 2,
                    right_eye_center[1] + right_height // 2) + padding

        min_x = max(0, min_x)
        min_y = max(0, min_y)
        max_x = min(frame.shape[1], max_x)
        max_y = min(frame.shape[0], max_y)

        # Create mask with ellipses for each eye
        mask_roi = np.zeros((max_y - min_y, max_x - min_x), dtype=np.uint8)
        left_center = (left_eye_center[0] - min_x, left_eye_center[1] - min_y)
        right_center = (right_eye_center[0] - min_x, right_eye_center[1] - min_y)
        left_axes = (left_width // 2, left_height // 2)
        right_axes = (right_width // 2, right_height // 2)

        cv2.ellipse(mask_roi, left_center, left_axes, 0, 0, 360, 255, -1)
        cv2.ellipse(mask_roi, right_center, right_axes, 0, 0, 360, 255, -1)
        mask_roi = gpu_gaussian_blur(mask_roi, (15, 15), 5)
        mask[min_y:max_y, min_x:max_x] = mask_roi

        eyes_cutout = frame[min_y:max_y, min_x:max_x].copy()
        eyes_box = (min_x, min_y, max_x, max_y)

        # Create polygon points
        t = np.linspace(0, 2 * np.pi, 32)
        left_points = np.column_stack((
            left_eye_center[0] + left_axes[0] * np.cos(t),
            left_eye_center[1] + left_axes[1] * np.sin(t)
        )).astype(np.int32)
        right_points = np.column_stack((
            right_eye_center[0] + right_axes[0] * np.cos(t),
            right_eye_center[1] + right_axes[1] * np.sin(t)
        )).astype(np.int32)
        eyes_polygon = np.vstack([left_points, right_points])

    return mask, eyes_cutout, eyes_box, eyes_polygon


def create_eyebrows_mask(face: Face, frame: Frame):
    """Create a mask for the eyebrow area to preserve natural eyebrow expressions.

    Returns: (mask, eyebrows_cutout, eyebrows_box, eyebrows_polygon)
    """
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    eyebrows_cutout = None
    eyebrows_box = (0, 0, 0, 0)
    eyebrows_polygon = None

    landmarks = face.landmark_2d_106
    if landmarks is not None:
        left_eyebrow = landmarks[97:105].astype(np.float32)
        right_eyebrow = landmarks[43:51].astype(np.float32)

        all_points = np.vstack([left_eyebrow, right_eyebrow])
        padding_factor = modules.globals.eyebrows_mask_size
        min_x = int(np.min(all_points[:, 0]) - 25 * padding_factor)
        max_x = int(np.max(all_points[:, 0]) + 25 * padding_factor)
        min_y = int(np.min(all_points[:, 1]) - 20 * padding_factor)
        max_y = int(np.max(all_points[:, 1]) + 15 * padding_factor)

        min_x = max(0, min_x)
        min_y = max(0, min_y)
        max_x = min(frame.shape[1], max_x)
        max_y = min(frame.shape[0], max_y)

        mask_roi = np.zeros((max_y - min_y, max_x - min_x), dtype=np.uint8)

        try:
            left_local = left_eyebrow - [min_x, min_y]
            right_local = right_eyebrow - [min_x, min_y]

            # Draw filled polygons for each eyebrow
            cv2.fillPoly(mask_roi, [left_local.astype(np.int32)], 255)
            cv2.fillPoly(mask_roi, [right_local.astype(np.int32)], 255)

            # Multi-stage blurring for natural feathering
            mask_roi = gpu_gaussian_blur(mask_roi, (21, 21), 7)
            mask_roi = cv2.normalize(mask_roi, None, 0, 255, cv2.NORM_MINMAX)

            mask[min_y:max_y, min_x:max_x] = mask_roi
            eyebrows_cutout = frame[min_y:max_y, min_x:max_x].copy()

            eyebrows_polygon = np.vstack([
                left_eyebrow.astype(np.int32),
                right_eyebrow.astype(np.int32)
            ])
            eyebrows_box = (min_x, min_y, max_x, max_y)

        except Exception:
            pass

    return mask, eyebrows_cutout, eyebrows_box, eyebrows_polygon


def draw_mouth_mask_visualization(
    frame: Frame, face: Face, mouth_mask_data: tuple
) -> Frame:
    """Debug overlay that draws the mouth-mask box and polygon on a frame copy."""
    if frame is None or face is None or mouth_mask_data is None or len(mouth_mask_data) != 4:
        return frame

    mask, mouth_cutout, box, lower_lip_polygon = mouth_mask_data
    (min_x, min_y, max_x, max_y) = box

    if lower_lip_polygon is None or not isinstance(lower_lip_polygon, np.ndarray) or len(lower_lip_polygon) < 3:
        return frame

    vis_frame = frame.copy()
    height, width = vis_frame.shape[:2]

    try:
        min_x, min_y = max(0, int(min_x)), max(0, int(min_y))
        max_x, max_y = min(width, int(max_x)), min(height, int(max_y))
    except (ValueError, TypeError):
        return frame

    if max_x <= min_x or max_y <= min_y:
        return frame

    try:
        safe_polygon = lower_lip_polygon.copy()
        safe_polygon[:, 0] = np.clip(safe_polygon[:, 0], 0, width - 1)
        safe_polygon[:, 1] = np.clip(safe_polygon[:, 1], 0, height - 1)
        cv2.polylines(vis_frame, [safe_polygon.astype(np.int32)], isClosed=True,
                      color=(0, 255, 0), thickness=2)
    except Exception:
        pass

    cv2.rectangle(vis_frame, (min_x, min_y), (max_x, max_y), (0, 0, 255), 2)

    label_pos_y = min_y - 10 if min_y > 20 else max_y + 15
    label_pos_x = min_x
    try:
        cv2.putText(vis_frame, "Mouth Mask", (label_pos_x, label_pos_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
    except Exception:
        pass

    return vis_frame


def apply_mask_area(
    frame: np.ndarray,
    cutout: np.ndarray,
    box: tuple,
    face_mask: np.ndarray,
    polygon: np.ndarray,
) -> np.ndarray:
    """Apply a masked cutout area back onto the frame with color correction.

    This is the core blending function that makes the mask seamless.
    """
    min_x, min_y, max_x, max_y = box
    box_width = max_x - min_x
    box_height = max_y - min_y

    if (cutout is None or box_width <= 0 or box_height <= 0
            or face_mask is None or polygon is None):
        return frame

    try:
        resized_cutout = gpu_resize(cutout, (box_width, box_height))
        roi = frame[min_y:max_y, min_x:max_x]

        if roi.shape != resized_cutout.shape:
            resized_cutout = gpu_resize(
                resized_cutout, (roi.shape[1], roi.shape[0])
            )

        color_corrected_area = apply_color_transfer(resized_cutout, roi)

        # Create polygon mask
        polygon_mask = np.zeros(roi.shape[:2], dtype=np.uint8)
        adjusted_polygon = polygon - [min_x, min_y]
        cv2.fillPoly(polygon_mask, [adjusted_polygon], 255)

        polygon_mask = gpu_gaussian_blur(polygon_mask, (21, 21), 7)

        feather_amount = min(
            30,
            box_width // modules.globals.mask_feather_ratio,
            box_height // modules.globals.mask_feather_ratio,
        )
        feathered_mask = cv2.GaussianBlur(
            polygon_mask.astype(np.float32), (0, 0), feather_amount
        )
        max_val = feathered_mask.max()
        if max_val > 1e-6:
            feathered_mask *= np.float32(1.0 / max_val)

        feathered_mask = cv2.GaussianBlur(feathered_mask, (5, 5), 1)

        face_mask_roi = face_mask[min_y:max_y, min_x:max_x]
        combined_mask = feathered_mask * (
            face_mask_roi.astype(np.float32) * np.float32(1.0 / 255.0)
        )

        combined_mask_3ch = combined_mask[:, :, np.newaxis]
        inv_mask = np.float32(1.0) - combined_mask_3ch
        blended = (
            color_corrected_area * combined_mask_3ch + roi * inv_mask
        ).astype(np.uint8)

        # Apply face mask to blended result (np.copy to avoid read-only view from broadcast_to)
        face_mask_f32 = face_mask_roi[:, :, np.newaxis].astype(
            np.float32) * np.float32(1.0 / 255.0)
        face_mask_3channel = np.broadcast_to(face_mask_f32, blended.shape).copy()
        final_blend = blended * face_mask_3channel + roi * (
            np.float32(1.0) - face_mask_3channel)

        frame[min_y:max_y, min_x:max_x] = final_blend.astype(np.uint8)
    except Exception:
        pass

    return frame
