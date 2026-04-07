import argparse
import json
from pathlib import Path

import joblib
import numpy as np
from onnxmltools import convert_xgboost
from onnxmltools.convert.common.data_types import FloatTensorType


# Feature order used by runtime engines.
FEATURE_NAMES = [
    "EAR",
    "MAR",
    "PERCLOS",
    "blink_rate",
    "head_pitch",
    "head_yaw",
    "head_roll",
    "heart_rate",
    "spo2",
    "temperature",
    "ear_mar_ratio",
    "perclos_blink_interaction",
    "head_motion_sum",
]


def main():
    parser = argparse.ArgumentParser(description="Convert XGBoost + scaler artifacts to ONNX deployment assets.")
    parser.add_argument("--model", required=True, help="Path to xgboost .joblib model")
    parser.add_argument("--scaler", required=True, help="Path to sklearn scaler .joblib")
    parser.add_argument("--onnx", required=True, help="Output path for .onnx model")
    parser.add_argument("--scaler-json", required=True, help="Output path for scaler params json")
    args = parser.parse_args()

    model_path = Path(args.model)
    scaler_path = Path(args.scaler)
    onnx_path = Path(args.onnx)
    scaler_json_path = Path(args.scaler_json)

    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)

    if not hasattr(model, "predict_proba"):
        raise ValueError("Loaded model does not support predict_proba. Expected an XGBoost classifier.")

    if not hasattr(scaler, "mean_") or not hasattr(scaler, "scale_"):
        raise ValueError("Scaler must expose mean_ and scale_ (e.g., StandardScaler).")

    initial_type = [("float_input", FloatTensorType([None, len(FEATURE_NAMES)]))]
    onnx_model = convert_xgboost(model, initial_types=initial_type, target_opset=15)

    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    scaler_json_path.parent.mkdir(parents=True, exist_ok=True)

    with open(onnx_path, "wb") as f:
        f.write(onnx_model.SerializeToString())

    scaler_payload = {
        "feature_names": FEATURE_NAMES,
        "mean": np.asarray(scaler.mean_, dtype=np.float32).tolist(),
        "scale": np.asarray(scaler.scale_, dtype=np.float32).tolist(),
    }

    with open(scaler_json_path, "w", encoding="utf-8") as f:
        json.dump(scaler_payload, f, indent=2)

    print(f"Saved ONNX model: {onnx_path}")
    print(f"Saved scaler params: {scaler_json_path}")


if __name__ == "__main__":
    main()
