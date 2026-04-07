"""Deprecated local webcam script.

Use backend/app.py and call POST /predict with an image file.
"""

from pathlib import Path

import cv2
import numpy as np


def analyze_image_file(image_path: str) -> dict:
    """Return basic eye-state heuristics from a single image file.

    This helper keeps logic reusable for future frame-stream APIs while
    avoiding webcam/GUI dependencies.
    """
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mean_intensity = float(np.mean(gray))

    return {
        "status": "processed",
        "mean_intensity": round(mean_intensity, 3),
    }


if __name__ == "__main__":
    sample = Path("sample.jpg")
    if sample.exists():
        print(analyze_image_file(str(sample)))
    else:
        print("Provide an image and use backend/app.py for production inference.")
