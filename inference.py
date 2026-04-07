from typing import Any
import logging
import threading

try:
    import cv2
    CV2_IMPORT_ERROR = None
except Exception as exc:
    cv2 = None
    CV2_IMPORT_ERROR = exc

try:
    import mediapipe as mp
    MEDIAPIPE_IMPORT_ERROR = None
except Exception as exc:
    mp = None
    MEDIAPIPE_IMPORT_ERROR = exc

try:
    import numpy as np
    NUMPY_IMPORT_ERROR = None
except Exception as exc:
    np = None
    NUMPY_IMPORT_ERROR = exc

import math

logger = logging.getLogger(__name__)

TARGET_IMAGE_SIZE = (320, 240)  # Optimized for performance

_face_mesh = None
_face_mesh_lock = threading.Lock()


def _ensure_runtime_dependencies() -> None:
    if CV2_IMPORT_ERROR is not None:
        logger.exception("cv2 import failed", exc_info=CV2_IMPORT_ERROR)
        raise RuntimeError("OpenCV (cv2) is unavailable") from CV2_IMPORT_ERROR
    if NUMPY_IMPORT_ERROR is not None:
        logger.exception("numpy import failed", exc_info=NUMPY_IMPORT_ERROR)
        raise RuntimeError("NumPy is unavailable") from NUMPY_IMPORT_ERROR


def get_face_mesh():
    global _face_mesh
    _ensure_runtime_dependencies()

    if MEDIAPIPE_IMPORT_ERROR is not None:
        logger.exception("mediapipe import failed", exc_info=MEDIAPIPE_IMPORT_ERROR)
        raise RuntimeError("MediaPipe is unavailable") from MEDIAPIPE_IMPORT_ERROR

    if not hasattr(mp, "solutions"):
        raise RuntimeError("MediaPipe package loaded incorrectly (missing solutions module)")

    if _face_mesh is None:
        with _face_mesh_lock:
            if _face_mesh is None:
                try:
                    mp_face_mesh = mp.solutions.face_mesh
                    _face_mesh = mp_face_mesh.FaceMesh(
                        max_num_faces=1,
                        refine_landmarks=True,
                        min_detection_confidence=0.5,
                    )
                    logger.info("MediaPipe FaceMesh initialized")
                except Exception:
                    logger.exception("MediaPipe FaceMesh initialization failed")
                    raise
    return _face_mesh

# Landmark indices
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
MOUTH_INNER = [13, 14, 78, 308]

# Head pose model points and camera matrix (from head_pose.py)
MODEL_POINTS = np.array([
    (0.0, 0.0, 0.0),             # Nose tip
    (0.0, 330.0, -65.0),         # Chin
    (-225.0, -170.0, -135.0),    # Left eye left corner
    (225.0, -170.0, -135.0),     # Right eye right corner
    (-150.0, 150.0, -125.0),     # Left Mouth corner
    (150.0, 150.0, -125.0)       # Right mouth corner
], dtype="double")

POINTS_IDX = [1, 152, 33, 263, 61, 291]


def eye_aspect_ratio(eye):
    if len(eye) != 6:
        return 0
    A = math.dist(eye[1], eye[5])
    B = math.dist(eye[2], eye[4])
    C = math.dist(eye[0], eye[3])
    return (A + B) / (2.0 * C) if C else 0


def mouth_aspect_ratio(mouth):
    if len(mouth) != 4:
        return 0
    v = math.dist(mouth[0], mouth[1])
    h = math.dist(mouth[2], mouth[3])
    return v / h if h else 0


def calculate_head_pose(landmarks, img_w, img_h):
    """Calculate head pose angles."""
    image_points = np.array([
        (landmarks[i].x * img_w, landmarks[i].y * img_h)
        for i in POINTS_IDX
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
        MODEL_POINTS, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE
    )

    if not success:
        return 0.0, 0.0, 0.0

    rmat, _ = cv2.Rodrigues(rotation_vector)
    angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)

    pitch = angles[0] * -1  # Invert for convention
    yaw = angles[1]
    roll = angles[2]

    return pitch, yaw, roll


def extract_features(image: np.ndarray) -> list[float]:
    """Extract 13 features from image for fatigue prediction."""
    try:
        face_mesh = get_face_mesh()
        h, w, _ = image.shape
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)

        if not results.multi_face_landmarks:
            logger.warning("No face detected in image")
            raise ValueError("No face detected in the uploaded image")

        lm = results.multi_face_landmarks[0]

        # Compute EAR
        left_eye = [(lm.landmark[i].x * w, lm.landmark[i].y * h) for i in LEFT_EYE]
        right_eye = [(lm.landmark[i].x * w, lm.landmark[i].y * h) for i in RIGHT_EYE]
        ear = (eye_aspect_ratio(left_eye) + eye_aspect_ratio(right_eye)) / 2

        # Compute MAR
        mouth = [(lm.landmark[i].x * w, lm.landmark[i].y * h) for i in MOUTH_INNER]
        mar = mouth_aspect_ratio(mouth)

        # Compute head pose
        pitch, yaw, roll = calculate_head_pose(lm.landmark, w, h)

        # Sensor features (not available from image, set to 0)
        perclos = 0.0
        blink_rate = 0.0
        heart_rate = 0.0
        spo2 = 0.0
        temperature = 0.0

        # Derived features
        eps = 1e-6
        ear_mar_ratio = ear / (mar + eps)
        perclos_blink_interaction = perclos * blink_rate
        head_motion_sum = abs(pitch) + abs(yaw) + abs(roll)

        features = [
            ear, mar, perclos, blink_rate,
            pitch, yaw, roll,
            heart_rate, spo2, temperature,
            ear_mar_ratio, perclos_blink_interaction, head_motion_sum
        ]

        return features
    except Exception as e:
        logger.error(f"Feature extraction failed: {e}")
        raise ValueError(f"Failed to extract features from image: {e}")


def _normalize_image(image: np.ndarray) -> np.ndarray:
    image = image.astype(np.float32) / 255.0
    return image


def _prepare_tensor_from_image(image: np.ndarray, input_shape: list[Any]) -> np.ndarray:
    resized = cv2.resize(image, TARGET_IMAGE_SIZE, interpolation=cv2.INTER_AREA)
    normalized = _normalize_image(resized)

    if len(input_shape) == 4:
        # NCHW
        if input_shape[1] in (1, 3):
            if input_shape[1] == 1:
                normalized = cv2.cvtColor((normalized * 255.0).astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
                normalized = np.expand_dims(normalized, axis=0)
            else:
                normalized = np.transpose(normalized, (2, 0, 1))
            return np.expand_dims(normalized, axis=0).astype(np.float32)

        # NHWC
        if input_shape[-1] == 1:
            normalized = cv2.cvtColor((normalized * 255.0).astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
            normalized = np.expand_dims(normalized, axis=-1)
        return np.expand_dims(normalized, axis=0).astype(np.float32)

    # Fallback for non-image tensor shapes (e.g. [N, F]).
    flattened = normalized.reshape(1, -1).astype(np.float32)
    if len(input_shape) == 2 and isinstance(input_shape[1], int):
        expected_features = input_shape[1]
        if flattened.shape[1] > expected_features:
            flattened = flattened[:, :expected_features]
        elif flattened.shape[1] < expected_features:
            padding = np.zeros((1, expected_features - flattened.shape[1]), dtype=np.float32)
            flattened = np.concatenate([flattened, padding], axis=1)
    return flattened


def decode_image_from_bytes(image_bytes: bytes) -> np.ndarray:
    _ensure_runtime_dependencies()
    np_buffer = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv2.imdecode(np_buffer, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Invalid image file. Could not decode image bytes.")
    return image


def run_inference_from_image(session: Any, image: np.ndarray) -> tuple[Any, float]:
    """Run inference from an already-decoded OpenCV image."""
    _ensure_runtime_dependencies()

    input_meta = session.get_inputs()[0]
    input_name = input_meta.name
    input_shape = input_meta.shape

    if len(input_shape) == 2 and input_shape[1] == 13:
        features = extract_features(image)
        model_input = np.array([features], dtype=np.float32)
        logger.debug(f"Input shape sent to model: {model_input.shape}")
    else:
        model_input = _prepare_tensor_from_image(image, input_shape)
        logger.debug(f"Input shape sent to model: {model_input.shape}")

    outputs = session.run(None, {input_name: model_input})

    primary = outputs[0]
    if hasattr(primary, "tolist"):
        result = primary.tolist()
    else:
        result = primary

    confidence = 0.0
    if len(outputs) > 1:
        probs = outputs[1]
        if hasattr(probs, "tolist"):
            probs = probs.tolist()
        if isinstance(probs, list) and len(probs) > 0 and isinstance(probs[0], list):
            confidence = max(probs[0])
        elif isinstance(probs, list):
            confidence = max(probs)

    logger.debug(f"Model output: {result}, confidence: {confidence}")
    return result, confidence


def run_inference(session: Any, image_bytes: bytes) -> tuple[Any, float]:
    image = decode_image_from_bytes(image_bytes)
    return run_inference_from_image(session, image)


def analyze_frame(session: Any, image: np.ndarray) -> dict[str, Any]:
    """Analyze a decoded frame and return prediction plus feature metadata."""
    _ensure_runtime_dependencies()

    face_mesh = get_face_mesh()
    h, w, _ = image.shape
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    if not results.multi_face_landmarks:
        logger.warning("No face detected in image")
        raise ValueError("No face detected in the uploaded image")

    lm = results.multi_face_landmarks[0]

    left_eye = [(lm.landmark[i].x * w, lm.landmark[i].y * h) for i in LEFT_EYE]
    right_eye = [(lm.landmark[i].x * w, lm.landmark[i].y * h) for i in RIGHT_EYE]
    ear = (eye_aspect_ratio(left_eye) + eye_aspect_ratio(right_eye)) / 2

    mouth = [(lm.landmark[i].x * w, lm.landmark[i].y * h) for i in MOUTH_INNER]
    mar = mouth_aspect_ratio(mouth)

    pitch, yaw, roll = calculate_head_pose(lm.landmark, w, h)

    perclos = 0.0
    blink_rate = 0.0
    heart_rate = 0.0
    spo2 = 0.0
    temperature = 0.0

    eps = 1e-6
    ear_mar_ratio = ear / (mar + eps)
    perclos_blink_interaction = perclos * blink_rate
    head_motion_sum = abs(pitch) + abs(yaw) + abs(roll)

    input_meta = session.get_inputs()[0]
    input_name = input_meta.name
    input_shape = input_meta.shape
    model_input = np.array([
        ear, mar, perclos, blink_rate,
        pitch, yaw, roll,
        heart_rate, spo2, temperature,
        ear_mar_ratio, perclos_blink_interaction, head_motion_sum,
    ], dtype=np.float32).reshape(1, -1)

    if not (len(input_shape) == 2 and input_shape[1] == 13):
        model_input = _prepare_tensor_from_image(image, input_shape)
        logger.debug(f"Image-based model input shape sent: {model_input.shape}")
    else:
        logger.debug(f"Feature-based model input shape sent: {model_input.shape}")

    outputs = session.run(None, {input_name: model_input})
    primary = outputs[0]
    if hasattr(primary, "tolist"):
        prediction = primary.tolist()
    else:
        prediction = primary

    confidence = 0.0
    if len(outputs) > 1:
        probs = outputs[1]
        if hasattr(probs, "tolist"):
            probs = probs.tolist()
        if isinstance(probs, list) and len(probs) > 0 and isinstance(probs[0], list):
            confidence = max(probs[0])
        elif isinstance(probs, list):
            confidence = max(probs)

    logger.debug(f"Frame analyzed: prediction={prediction}, confidence={confidence}, ear={ear}, mar={mar}, pitch={pitch:.1f}, yaw={yaw:.1f}")
    return {
        "prediction": prediction,
        "confidence": confidence,
        "ear": ear,
        "mar": mar,
        "pitch": pitch,
        "yaw": yaw,
        "roll": roll,
    }
