"""Deprecated local webcam script.

Use backend/app.py and call POST /predict with an image file.
"""

from pathlib import Path

import cv2
import numpy as np


def summarize_fatigue_frame(image_path: str) -> dict:
    """Compute lightweight frame statistics for future streaming pipelines."""
    frame = cv2.imread(image_path)
    if frame is None:
        raise ValueError(f"Could not read image: {image_path}")

    resized = cv2.resize(frame, (224, 224), interpolation=cv2.INTER_AREA)
    normalized = resized.astype(np.float32) / 255.0

    return {
        "status": "processed",
        "frame_shape": list(normalized.shape),
        "pixel_mean": float(np.mean(normalized)),
    }


if __name__ == "__main__":
    sample = Path("sample.jpg")
    if sample.exists():
        print(summarize_fatigue_frame(str(sample)))
    else:
        print("Provide an image and use backend/app.py for production inference.")
