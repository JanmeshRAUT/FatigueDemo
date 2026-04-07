from typing import Any

import cv2
import numpy as np


TARGET_IMAGE_SIZE = (224, 224)


def _normalize_image(image: np.ndarray) -> np.ndarray:
    image = image.astype(np.float32) / 255.0
    return image


def _prepare_tensor_from_image(image: np.ndarray, input_shape: list[Any]) -> np.ndarray:
    resized = cv2.resize(image, TARGET_IMAGE_SIZE, interpolation=cv2.INTER_AREA)
    normalized = _normalize_image(resized)

    if len(input_shape) == 4:
        # Keep memory usage low by creating only one batch tensor.
        if input_shape[1] in (1, 3):  # NCHW
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
    np_buffer = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv2.imdecode(np_buffer, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Invalid image file. Could not decode image bytes.")
    return image


def run_inference(session: Any, image_bytes: bytes) -> Any:
    image = decode_image_from_bytes(image_bytes)

    input_meta = session.get_inputs()[0]
    input_name = input_meta.name
    input_shape = input_meta.shape

    model_input = _prepare_tensor_from_image(image, input_shape)
    outputs = session.run(None, {input_name: model_input})

    primary = outputs[0]
    if hasattr(primary, "tolist"):
        return primary.tolist()
    return primary
