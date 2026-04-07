import cv2
import mediapipe as mp
import numpy as np

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)

LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

def calculate_ear(eye):
    v1 = np.linalg.norm(eye[1] - eye[5])
    v2 = np.linalg.norm(eye[2] - eye[4])
    h = np.linalg.norm(eye[0] - eye[3])
    return (v1 + v2) / (2.0 * h)

cap = cv2.VideoCapture(0)

# Calibration variables
calibration_frames = 30
ear_values = []
calibrated = False
ear_threshold = 0.2  # default fallback

frame_count = 0

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

            coords = []
            for lm in face_landmarks.landmark:
                coords.append([int(lm.x * w), int(lm.y * h)])
            coords = np.array(coords)

            left_eye = coords[LEFT_EYE]
            right_eye = coords[RIGHT_EYE]

            ear = (calculate_ear(left_eye) + calculate_ear(right_eye)) / 2

            # ---------------- CALIBRATION ----------------
            if not calibrated:
                ear_values.append(ear)
                frame_count += 1

                cv2.putText(frame, "Calibrating...", (30, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

                if frame_count >= calibration_frames:
                    baseline = np.mean(ear_values)
                    ear_threshold = baseline * 0.75
                    calibrated = True

                    print(f"Baseline EAR: {baseline:.3f}")
                    print(f"Threshold EAR: {ear_threshold:.3f}")

            else:
                # ---------------- DETECTION ----------------
                if ear < ear_threshold:
                    status = "Eyes Closed"
                    color = (0, 0, 255)
                else:
                    status = "Eyes Open"
                    color = (0, 255, 0)

                cv2.putText(frame, status, (30, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

            # Show EAR
            cv2.putText(frame, f"EAR: {ear:.3f}", (30, 150),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    cv2.imshow("Eye Calibration", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()