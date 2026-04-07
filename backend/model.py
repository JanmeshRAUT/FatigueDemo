import os
from functools import lru_cache

import onnxruntime as ort


BASE_DIR = os.path.dirname(__file__)
DEFAULT_MODEL_PATH = os.path.join(BASE_DIR, "model", "model.onnx")


@lru_cache(maxsize=1)
def get_onnx_session(model_path: str = DEFAULT_MODEL_PATH) -> ort.InferenceSession:
    """Load ONNX model once per process and cache the session."""
    session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])

    # Log model metadata
    print("ONNX Model Loaded:")
    print(f"  Path: {model_path}")
    for input_meta in session.get_inputs():
        print(f"  Input '{input_meta.name}': shape={input_meta.shape}, type={input_meta.type}")
    for output_meta in session.get_outputs():
        print(f"  Output '{output_meta.name}': shape={output_meta.shape}, type={output_meta.type}")

    return session
