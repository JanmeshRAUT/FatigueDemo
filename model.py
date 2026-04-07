import os
import logging
from functools import lru_cache

import onnxruntime as ort

logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODEL_PATH = os.path.join(BASE_DIR, "model", "model.onnx")


@lru_cache(maxsize=1)
def get_onnx_session(model_path: str = DEFAULT_MODEL_PATH) -> ort.InferenceSession:
    """Load ONNX model once per process and cache the session."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"ONNX model not found at {model_path}")

    try:
        session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        logger.info("ONNX Model Loaded:")
        logger.info(f"  Path: {model_path}")
        for input_meta in session.get_inputs():
            logger.info(f"  Input '{input_meta.name}': shape={input_meta.shape}, type={input_meta.type}")
        for output_meta in session.get_outputs():
            logger.info(f"  Output '{output_meta.name}': shape={output_meta.shape}, type={output_meta.type}")
        return session
    except Exception as e:
        logger.error(f"Failed to load ONNX model: {e}")
        raise
