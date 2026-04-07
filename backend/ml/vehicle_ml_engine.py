import os
import json
import numpy as np
from collections import deque

try:
    import onnxruntime as ort
except Exception:
    ort = None

try:
    import joblib
except Exception:
    joblib = None


class VehicleMLEngine:
    """Vehicle fatigue inference engine with optional XGBoost pipeline."""

    def __init__(self, model_path="vehicle_fatigue_model.pkl", scaler_path=None, label_encoder_path=None):
        self.model = None
        self.model_path = model_path
        self.scaler_path = scaler_path
        self.label_encoder_path = label_encoder_path
        self.scaler = None
        self.label_encoder = None
        self.use_xgb_pipeline = False
        self.use_onnx = False
        self.onnx_session = None
        self.onnx_input_name = None
        self.onnx_scaler = None
        self.labels = {0: "Alert", 1: "Drowsy", 2: "Fatigued"}

        self.window_size = 20
        self.history = deque(maxlen=self.window_size)

        self.ema_probs = None
        self.alpha = 0.15

        self.microsleep_max_frames = 10
        self.microsleep_max_seconds = 1.2

        self.current_state = 0
        self.state_persistence = 0
        self.required_persistence = 5

        self.base_ear = 0.32
        self.calibration_frames = 0
        self.max_calibration = 100

        self.last_vision_values = {"ear": 0.3, "mar": 0}

        self.load_model()

    def load_model(self):
        if self.model_path and self.model_path.lower().endswith(".onnx"):
            if ort is None:
                print("[VehicleML] ❌ onnxruntime is not installed. Cannot load ONNX model.")
            elif os.path.exists(self.model_path):
                try:
                    self.onnx_session = ort.InferenceSession(
                        self.model_path,
                        providers=["CPUExecutionProvider"],
                    )
                    self.onnx_input_name = self.onnx_session.get_inputs()[0].name
                    self.use_onnx = True
                    self._load_onnx_scaler()
                    print(f"[VehicleML] ONNX model loaded: {self.model_path}")
                    return
                except Exception as e:
                    print(f"[VehicleML] Failed to load ONNX model: {e}")

        if not os.path.exists(self.model_path):
            print(f"[VehicleML] Model file not found at {self.model_path}")
            return

        if joblib is None:
            print("[VehicleML] ⚠️ joblib not installed; non-ONNX fallback disabled.")
            return

        try:
            self.model = joblib.load(self.model_path)
            if self.scaler_path and os.path.exists(self.scaler_path):
                self.scaler = joblib.load(self.scaler_path)
            if self.label_encoder_path and os.path.exists(self.label_encoder_path):
                self.label_encoder = joblib.load(self.label_encoder_path)

            self.use_xgb_pipeline = self.scaler is not None
            if self.use_xgb_pipeline:
                print(f"[VehicleML] XGBoost pipeline loaded: {self.model_path}")
            else:
                print(f"[VehicleML] Legacy model loaded: {self.model_path}")
        except Exception as e:
            print(f"[VehicleML] Failed to load model: {e}")

    def _load_onnx_scaler(self):
        if not self.scaler_path:
            return
        if not os.path.exists(self.scaler_path):
            print(f"[VehicleML] ONNX scaler config not found at {self.scaler_path}. Using raw features.")
            return
        try:
            with open(self.scaler_path, "r", encoding="utf-8") as f:
                scaler_data = json.load(f)
            mean = np.array(scaler_data.get("mean", []), dtype=np.float32)
            scale = np.array(scaler_data.get("scale", []), dtype=np.float32)
            if mean.size and scale.size and mean.size == scale.size:
                self.onnx_scaler = {"mean": mean, "scale": np.where(scale == 0, 1.0, scale)}
                print(f"[VehicleML] ONNX scaler loaded: {self.scaler_path}")
        except Exception as e:
            print(f"[VehicleML] Failed to load ONNX scaler config: {e}")

    def _apply_onnx_scaling(self, features):
        if not self.onnx_scaler:
            return features.astype(np.float32)
        mean = self.onnx_scaler["mean"]
        scale = self.onnx_scaler["scale"]
        if features.shape[1] != mean.shape[0]:
            print("[VehicleML] ONNX scaler dimension mismatch. Using raw features.")
            return features.astype(np.float32)
        return ((features - mean) / scale).astype(np.float32)

    def _extract_onnx_probs(self, outputs):
        for out in outputs:
            if isinstance(out, np.ndarray) and out.ndim == 2 and out.shape[0] >= 1:
                return out[0]
            if isinstance(out, list) and out and isinstance(out[0], dict):
                probs = [v for _, v in sorted(out[0].items(), key=lambda kv: int(kv[0]))]
                return np.array(probs, dtype=np.float32)
        raise ValueError("Unable to parse ONNX probability outputs")

    def reset_calibration(self):
        self.base_ear = 0.32
        self.calibration_frames = 0
        self.history.clear()
        self.ema_probs = None
        self.current_state = 0
        print("[VehicleML] Calibration reset")

    def calculate_temporal_features(self, current_data):
        self.history.append(current_data)
        data_matrix = np.array(self.history)

        if len(self.history) < 2:
            return {
                "ear_mean": current_data[0], "ear_std": 0,
                "mar_mean": current_data[1], "mar_std": 0,
                "pitch_mean": current_data[2], "pitch_std": 0,
                "yaw_mean": current_data[3], "yaw_std": 0,
                "roll_mean": current_data[4], "roll_std": 0,
            }

        means = np.mean(data_matrix, axis=0)
        stds = np.std(data_matrix, axis=0)
        return {
            "ear_mean": means[0], "ear_std": stds[0],
            "mar_mean": means[1], "mar_std": stds[1],
            "pitch_mean": means[2], "pitch_std": stds[2],
            "yaw_mean": means[3], "yaw_std": stds[3],
            "roll_mean": means[4], "roll_std": stds[4],
        }

    def _build_xgb_features(self, vision_data, sensor_data):
        ear = float(vision_data.get("ear", 0.3) or 0.3)
        mar = float(vision_data.get("mar", 0.0) or 0.0)
        pitch = float(vision_data.get("pitch", 0.0) or 0.0)
        yaw = float(vision_data.get("yaw", 0.0) or 0.0)
        roll = float(vision_data.get("roll", 0.0) or 0.0)
        perclos = float(vision_data.get("perclos", 0.0) or 0.0)
        perclos_ratio = perclos / 100.0 if perclos > 1.0 else perclos

        if perclos > 1.0:
            perclos = perclos / 100.0

        heart_rate = float(sensor_data.get("hr", 75.0) or 75.0)
        spo2 = float(sensor_data.get("spo2", 98.0) or 98.0)
        temperature = float(sensor_data.get("temperature", 37.0) or 37.0)

        heart_rate = float(np.clip(heart_rate, 40.0, 180.0))
        spo2 = float(np.clip(spo2, 80.0, 100.0))
        temperature = float(np.clip(temperature, 34.0, 41.0))

        eps = 1e-6
        ear_mar_ratio = ear / (mar + eps)
        perclos_blink_interaction = perclos * float(vision_data.get("blink_rate", 0.0) or 0.0)
        head_motion_sum = abs(pitch) + abs(yaw) + abs(roll)

        values = [
            ear, mar, perclos, float(vision_data.get("blink_rate", 0.0) or 0.0), pitch, yaw, roll,
            heart_rate, spo2, temperature, ear_mar_ratio, perclos_blink_interaction, head_motion_sum,
        ]
        return np.array([values], dtype=np.float32)

    def predict(self, vision_data, sensor_data=None):
        if self.model is None and not self.use_onnx:
            return {"status": "Unknown", "confidence": 0, "raw_probs": [0, 0, 0], "microsleep_detected": False}

        if sensor_data is None:
            sensor_data = {}

        vision_status = vision_data.get("status", "Unknown")
        if vision_status in ["No Face", "Unstable"]:
            if self.ema_probs is not None:
                self.ema_probs = self.ema_probs * 0.5 + np.array([1, 0, 0]) * 0.5
            else:
                self.ema_probs = np.array([1, 0, 0])
            return {
                "status": "No Face Detected",
                "confidence": 0.0,
                "raw_probs": self.ema_probs.tolist(),
                "microsleep_detected": False,
            }

        ear = float(vision_data.get("ear", 0.3) or 0.3)
        mar = float(vision_data.get("mar", 0.0) or 0.0)
        pitch = float(vision_data.get("pitch", 0.0) or 0.0)
        yaw = float(vision_data.get("yaw", 0.0) or 0.0)
        roll = float(vision_data.get("roll", 0.0) or 0.0)
        perclos = float(vision_data.get("perclos", 0.0) or 0.0)
        perclos_ratio = perclos / 100.0 if perclos > 1.0 else perclos
        closed_duration_sec = float(vision_data.get("closed_duration_sec", 0.0) or 0.0)

        self.last_vision_values = {"ear": ear, "mar": mar}

        if self.calibration_frames < self.max_calibration and ear > 0.15:
            self.base_ear = (self.base_ear * self.calibration_frames + ear) / (self.calibration_frames + 1)
            self.calibration_frames += 1

        normalized_ear = ear / max(self.base_ear, 0.3)
        current_features = np.array([normalized_ear, mar, pitch, yaw, roll])
        temporal = self.calculate_temporal_features(current_features)

        legacy_x = np.array([
            temporal["ear_mean"], temporal["ear_std"], temporal["mar_mean"], temporal["mar_std"],
            temporal["pitch_mean"], temporal["pitch_std"], temporal["yaw_mean"], temporal["yaw_std"],
            temporal["roll_mean"], temporal["roll_std"], perclos_ratio,
        ])

        try:
            if self.use_onnx and self.onnx_session is not None and self.onnx_input_name:
                xgb_features = self._build_xgb_features(vision_data, sensor_data)
                x_scaled = self._apply_onnx_scaling(xgb_features)
                outputs = self.onnx_session.run(None, {self.onnx_input_name: x_scaled})
                probs = self._extract_onnx_probs(outputs)
            elif self.use_xgb_pipeline:
                xgb_features = self._build_xgb_features(vision_data, sensor_data)
                x_scaled = self.scaler.transform(xgb_features)
                probs = self.model.predict_proba(x_scaled)[0]
            else:
                probs = self.model.predict_proba([legacy_x])[0]
        except Exception as e:
            print(f"[VehicleML] Inference error: {e}")
            return {"status": "Error", "confidence": 0, "raw_probs": [0, 0, 0], "microsleep_detected": False}

        if self.ema_probs is None:
            self.ema_probs = probs
        else:
            self.ema_probs = self.alpha * probs + (1 - self.alpha) * self.ema_probs

        if closed_duration_sec >= self.microsleep_max_seconds:
            self.current_state = 2
            self.ema_probs = np.array([0, 0, 1])
            self.state_persistence = 0

        # Safety override: very high PERCLOS should immediately force Fatigued.
        if perclos_ratio > 0.55:
            self.current_state = 2
            self.ema_probs = np.array([0, 0, 1])
            self.state_persistence = 0
        
        proposed_state = int(np.argmax(self.ema_probs))
        confidence = float(self.ema_probs[proposed_state])

        # Match Standard mode's staged transition feel:
        # Alert -> Drowsy -> Fatigued and Fatigued -> Drowsy -> Alert.
        if perclos_ratio <= 0.55:
            if self.current_state == 0 and proposed_state == 2:
                if confidence < 0.9:
                    proposed_state = 1

            if self.current_state == 2 and proposed_state == 0:
                proposed_state = 1

        if proposed_state != self.current_state:
            self.state_persistence += 1
            if self.state_persistence >= self.required_persistence:
                self.current_state = proposed_state
                self.state_persistence = 0
        else:
            self.state_persistence = 0

        microsleep_detected = bool(perclos_ratio > 0.80 or ear < 0.15 or closed_duration_sec >= self.microsleep_max_seconds)
        confidence = float(self.ema_probs[self.current_state])
        output_label = self.labels[self.current_state]

        if self.label_encoder is not None:
            try:
                decoded = self.label_encoder.inverse_transform([self.current_state])[0]
                if isinstance(decoded, str):
                    normalized = decoded.strip().lower()
                    if normalized in ("0", "1", "2"):
                        output_label = self.labels.get(int(normalized), output_label)
                    elif normalized in ("alert", "drowsy", "fatigued"):
                        output_label = normalized.capitalize()
                    else:
                        output_label = decoded
                else:
                    output_label = self.labels.get(int(decoded), output_label)
            except Exception:
                pass

        return {
            "status": output_label,
            "confidence": round(confidence, 3),
            "raw_probs": [float(p) for p in self.ema_probs],
            "microsleep_detected": microsleep_detected,
        }

    def get_debug_info(self):
        return {
            "model_loaded": self.model is not None,
            "calibration_progress": f"{min(self.calibration_frames, self.max_calibration)}/{self.max_calibration}",
            "base_ear": round(self.base_ear, 4),
            "current_state": self.labels.get(self.current_state, "Unknown"),
            "state_persistence": self.state_persistence,
            "window_size": len(self.history),
        }
