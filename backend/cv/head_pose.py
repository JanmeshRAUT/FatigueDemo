import cv2
import numpy as np
import threading

# --- Global State for Calibration ---
# We assume the user looks at the screen comfortably during the first few seconds.
CALIBRATION_FRAMES_TARGET = 30  # Number of frames to average for calibration
calibration_counter = 0
pitch_accumulator = 0.0
yaw_accumulator = 0.0
roll_accumulator = 0.0

pitch_offset = 0.0
yaw_offset = 0.0
roll_offset = 0.0
is_calibrated = False

# Global Output State
cv_head_angles = {"pitch": 0.0, "yaw": 0.0, "roll": 0.0}
cv_angles_lock = threading.Lock()

def calculate_cv_head_pose(landmarks, img_w, img_h):
    """
    Estimates head pose and applies AUTO-CENTERING calibration.
    
    The first 30 frames of valid face detection are used to define the "Zero" (Center) position.
    This handles any camera angle (tilted laptop, side webcam) automatically.
    """
    global calibration_counter, pitch_accumulator, yaw_accumulator, roll_accumulator
    global pitch_offset, yaw_offset, roll_offset, is_calibrated

    # --- 1. Standard PnP Head Pose Estimation ---
    
    # 2D Image Points
    points_idx = [1, 152, 33, 263, 61, 291]
    image_points = np.array([
        (landmarks[i].x * img_w, landmarks[i].y * img_h) 
        for i in points_idx
    ], dtype="double")

    # 3D Model Points (Y-Down Convention)
    model_points = np.array([
        (0.0, 0.0, 0.0),             # Nose tip
        (0.0, 330.0, -65.0),         # Chin
        (-225.0, -170.0, -135.0),    # Left eye left corner
        (225.0, -170.0, -135.0),     # Right eye right corner
        (-150.0, 150.0, -125.0),     # Left Mouth corner
        (150.0, 150.0, -125.0)       # Right mouth corner
    ], dtype="double")

    focal_length = img_w
    center = (img_w / 2, img_h / 2)
    camera_matrix = np.array([
        [focal_length, 0, center[0]],
        [0, focal_length, center[1]],
        [0, 0, 1]
    ], dtype="double")
    dist_coeffs = np.zeros((4, 1))

    success, rotation_vector, translation_vector = cv2.solvePnP(
        model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE
    )

    if not success:
        return 0, 0, 0, False
    
    rmat, _ = cv2.Rodrigues(rotation_vector)
    angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)
    
    # Raw Angles (Standard OpenCV: +ve Pitch is DOWN)
    raw_pitch = angles[0]
    raw_yaw = angles[1]
    raw_roll = angles[2]

    # --- 2. Calibration Logic ---
    if not is_calibrated:
        if calibration_counter < CALIBRATION_FRAMES_TARGET:
            pitch_accumulator += raw_pitch
            yaw_accumulator += raw_yaw
            roll_accumulator += raw_roll
            calibration_counter += 1
            # While calibrating, assume Center
            return 0.0, 0.0, 0.0, False
        else:
            # Calculate Average (The Baseline)
            pitch_offset = pitch_accumulator / CALIBRATION_FRAMES_TARGET
            yaw_offset = yaw_accumulator / CALIBRATION_FRAMES_TARGET
            roll_offset = roll_accumulator / CALIBRATION_FRAMES_TARGET
            is_calibrated = True
            print(f"[CALIBRATION] ✅ Head Pose Centered! Offsets -> P:{pitch_offset:.2f}, Y:{yaw_offset:.2f}, R:{roll_offset:.2f}")

    # --- 3. Apply Calibration Offset ---
    # Center = Raw - Baseline
    # Example: If Camera tilts back, Raw might be 15 (Looking Down). Offset captures 15.
    # Result = 15 - 15 = 0 (Center). Correct.
    
    # Apply Offset first
    # Inverting because User Hardware produces +ve for Up. We want -ve for Up.
    final_pitch = (raw_pitch - pitch_offset) * -1
    final_yaw = raw_yaw - yaw_offset
    final_roll = raw_roll - roll_offset

    # Clamp
    final_pitch = max(-90, min(90, final_pitch))
    final_yaw = max(-90, min(90, final_yaw))
    final_roll = max(-90, min(90, final_roll))

    return final_pitch, final_yaw, final_roll, is_calibrated
