import cv2
import mediapipe as mp
import numpy as np
from collections import deque

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)

# Landmarks
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
MOUTH = [13, 14, 78, 308]

# Functions
def calculate_ear(eye):
    v1 = np.linalg.norm(eye[1] - eye[5])
    v2 = np.linalg.norm(eye[2] - eye[4])
    h = np.linalg.norm(eye[0] - eye[3])
    return (v1 + v2) / (2.0 * h)

def calculate_mar(mouth):
    v = np.linalg.norm(mouth[0] - mouth[1])
    h = np.linalg.norm(mouth[2] - mouth[3])
    return v / h

# Video
cap = cv2.VideoCapture(0)

# Calibration
calibration_frames = 30
ear_list = []
mar_list = []
calibrated = False

ear_threshold = 0.2
mar_threshold = 0.5

# PERCLOS
window_size = 60
eye_window = deque(maxlen=window_size)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:

            coords = np.array([[int(lm.x * w), int(lm.y * h)] 
                               for lm in face_landmarks.landmark])

            left_eye = coords[LEFT_EYE]
            right_eye = coords[RIGHT_EYE]
            mouth = coords[MOUTH]

            ear = (calculate_ear(left_eye) + calculate_ear(right_eye)) / 2
            mar = calculate_mar(mouth)

            # ---------------- CALIBRATION ----------------
            if not calibrated:
                ear_list.append(ear)
                mar_list.append(mar)

                if len(ear_list) >= calibration_frames:
                    baseline_ear = np.mean(ear_list)
                    baseline_mar = np.mean(mar_list)

                    ear_threshold = baseline_ear * 0.75
                    mar_threshold = baseline_mar * 1.5

                    calibrated = True

                    print("Calibration Done")
                    print(f"EAR Threshold: {ear_threshold:.3f}")
                    print(f"MAR Threshold: {mar_threshold:.3f}")

                cv2.putText(frame, "Calibrating...", (30, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

            else:
                # ---------------- EYE STATE ----------------
                eye_closed = 1 if ear < ear_threshold else 0
                eye_window.append(eye_closed)

                perclos = sum(eye_window) / len(eye_window) if len(eye_window) > 0 else 0

                # ---------------- YAWN ----------------
                yawning = mar > mar_threshold

                # ---------------- STATUS ----------------
                if perclos > 0.55:
                    status = "FATIGUED"
                    color = (0, 0, 255)
                elif perclos > 0.25:
                    status = "DROWSY"
                    color = (0, 165, 255)
                else:
                    status = "ALERT"
                    color = (0, 255, 0)

                if yawning:
                    status += " + YAWN"

                # ---------------- DISPLAY ----------------
                cv2.putText(frame, f"EAR: {ear:.3f}", (30, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)

                cv2.putText(frame, f"MAR: {mar:.3f}", (30, 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)

                cv2.putText(frame, f"PERCLOS: {perclos:.2f}", (30, 110),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)

                cv2.putText(frame, status, (30, 150),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)

    cv2.imshow("Fatigue Detection (Full)", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()