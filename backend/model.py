import os
from functools import lru_cache

import onnxruntime as ort


BASE_DIR = os.path.dirname(__file__)
DEFAULT_MODEL_PATH = os.path.join(BASE_DIR, "model", "model.onnx")


@lru_cache(maxsize=1)
def get_onnx_session(model_path: str = DEFAULT_MODEL_PATH) -> ort.InferenceSession:
    """Load ONNX model once per process and cache the session."""
    return ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
