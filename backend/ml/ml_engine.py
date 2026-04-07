import os
import json
import numpy as np
from collections import deque
import time

try:
    import onnxruntime as ort
except Exception:
    ort = None

try:
    import joblib
except Exception:
    joblib = None

class MLEngine:
    def __init__(self, model_path="fatigue_model.pkl", scaler_path=None, label_encoder_path=None, fallback_model_path=None):
        self.model = None
        self.model_path = model_path
        self.scaler_path = scaler_path
        self.label_encoder_path = label_encoder_path
        self.fallback_model_path = fallback_model_path
        self.scaler = None
        self.label_encoder = None
        self.use_xgb_pipeline = False
        self.use_onnx = False
        self.onnx_session = None
        self.onnx_input_name = None
        self.onnx_scaler = None
        self.labels = {0: "Alert", 1: "Drowsy", 2: "Fatigued"}
        
        # INDUSTRIAL UPGRADE: Feature Window
        self.window_size = 20  # WIDE window for "Movie-like" smoothing
        self.history = deque(maxlen=self.window_size)
        
        # ADVANCED SMOOTHING: Exponential Moving Average (EMA)
        self.ema_probs = None 
        self.alpha = 0.15 # VERY LOW = Slow, cinematic transitions
        
        # SAFETY CRITICAL: Microsleep Detection
        # Thresholds now rely on external frame counting for accuracy
        self.microsleep_max_frames = 10   # ~0.33s
        self.microsleep_max_seconds = 1.2
        
        # STATE MACHINE: Hysteresis (Sticky States)
        self.current_state = 0 # Default Alert
        self.state_persistence = 0
        self.required_persistence = 5 # Frames to hold before switching
        
        self.debug_counter = 0

        # ROBUSTNESS 2.0: Adaptive Calibration
        self.base_ear = 0.32   # Dynamic baseline
        self.calibration_frames = 0
        self.max_calibration = 100 # Frames to learn 'Normal' EAR
        
        # Sensor Integrity
        self.last_sensor_values = {"hr": 0, "temp": 0}
        self.sensor_stale_count = 0
        
        self.load_model()
    

    def load_model(self):
        """Loads the trained ML model from disk, with optional fallback support."""
        if self.model_path and self.model_path.lower().endswith(".onnx"):
            if ort is None:
                print("[ML] ❌ onnxruntime is not installed. Cannot load ONNX model.")
            elif os.path.exists(self.model_path):
                try:
                    self.onnx_session = ort.InferenceSession(
                        self.model_path,
                        providers=["CPUExecutionProvider"],
                    )
                    self.onnx_input_name = self.onnx_session.get_inputs()[0].name
                    self.use_onnx = True
                    self._load_onnx_scaler()
                    print(f"[ML] ✅ ONNX model loaded from {self.model_path}")
                    return
                except Exception as e:
                    print(f"[ML] ❌ Failed to load ONNX model at {self.model_path}: {e}")

        if joblib is None:
            print("[ML] ⚠️ joblib not installed; non-ONNX fallback disabled.")
            return

        candidate_paths = [self.model_path]
        if self.fallback_model_path and self.fallback_model_path != self.model_path:
            candidate_paths.append(self.fallback_model_path)

        loaded_path = None
        for path in candidate_paths:
            if not path or not os.path.exists(path):
                continue
            try:
                self.model = joblib.load(path)
                loaded_path = path
                break
            except Exception as e:
                print(f"[ML] ❌ Failed to load model at {path}: {e}")

        if self.model is None:
            print(f"[ML] ⚠️ No model file found/loaded. Tried: {candidate_paths}")
            return

        loaded_is_xgb = self._is_xgb_model(self.model)

        if loaded_is_xgb:
            if self.scaler_path and os.path.exists(self.scaler_path):
                try:
                    self.scaler = joblib.load(self.scaler_path)
                except Exception as e:
                    print(f"[ML] ⚠️ Failed to load scaler at {self.scaler_path}: {e}")

            if self.label_encoder_path and os.path.exists(self.label_encoder_path):
                try:
                    self.label_encoder = joblib.load(self.label_encoder_path)
                except Exception as e:
                    print(f"[ML] ⚠️ Failed to load label encoder at {self.label_encoder_path}: {e}")

            self.use_xgb_pipeline = self.scaler is not None
            if self.use_xgb_pipeline:
                print(f"[ML] ✅ XGBoost pipeline loaded from {loaded_path}")
            else:
                print(f"[ML] ⚠️ XGBoost model loaded but scaler missing. Falling back to legacy feature path.")
        else:
            self.scaler = None
            self.label_encoder = None
            self.use_xgb_pipeline = False
            print(f"[ML] ✅ Legacy model loaded from {loaded_path}")

    def _load_onnx_scaler(self):
        if not self.scaler_path:
            return
        if not os.path.exists(self.scaler_path):
            print(f"[ML] ⚠️ ONNX scaler config not found at {self.scaler_path}. Using raw features.")
            return
        try:
            with open(self.scaler_path, "r", encoding="utf-8") as f:
                scaler_data = json.load(f)
            mean = np.array(scaler_data.get("mean", []), dtype=np.float32)
            scale = np.array(scaler_data.get("scale", []), dtype=np.float32)
            if mean.size and scale.size and mean.size == scale.size:
                self.onnx_scaler = {"mean": mean, "scale": np.where(scale == 0, 1.0, scale)}
                print(f"[ML] ✅ ONNX scaler loaded from {self.scaler_path}")
        except Exception as e:
            print(f"[ML] ⚠️ Failed to load ONNX scaler config: {e}")

    def _apply_onnx_scaling(self, features):
        if not self.onnx_scaler:
            return features.astype(np.float32)
        mean = self.onnx_scaler["mean"]
        scale = self.onnx_scaler["scale"]
        if features.shape[1] != mean.shape[0]:
            print("[ML] ⚠️ ONNX scaler dimension mismatch. Using raw features.")
            return features.astype(np.float32)
        return ((features - mean) / scale).astype(np.float32)

    def _extract_onnx_probs(self, outputs):
        # Typical classifier outputs: [labels, probabilities]
        for out in outputs:
            if isinstance(out, np.ndarray) and out.ndim == 2 and out.shape[0] >= 1:
                return out[0]
            if isinstance(out, list) and out and isinstance(out[0], dict):
                probs = [v for _, v in sorted(out[0].items(), key=lambda kv: int(kv[0]))]
                return np.array(probs, dtype=np.float32)
        raise ValueError("Unable to parse ONNX probability outputs")

    def _is_xgb_model(self, model):
        """Detect whether a loaded model object is an XGBoost estimator."""
        if model is None:
            return False
        module_name = getattr(type(model), "__module__", "")
        class_name = getattr(type(model), "__name__", "")
        return ("xgboost" in module_name.lower()) or class_name.startswith("XGB")

    def _build_xgb_features(self, vision_data, sensor_data):
        ear = float(vision_data.get("ear", 0.3) or 0.3)
        mar = float(vision_data.get("mar", 0.0) or 0.0)
        pitch = float(vision_data.get("head_angle_x", 0.0) or 0.0)
        yaw = float(vision_data.get("head_angle_y", 0.0) or 0.0)
        roll = float(vision_data.get("head_angle_z", 0.0) or 0.0)
        perclos = float(vision_data.get("perclos", 0.0) or 0.0)

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

    def _get_final_label(self, state_index):
        class_value = state_index
        if hasattr(self.model, "classes_") and len(getattr(self.model, "classes_", [])) > state_index:
            class_value = self.model.classes_[state_index]

        if self.label_encoder is not None:
            try:
                decoded = self.label_encoder.inverse_transform([class_value])[0]
                if isinstance(decoded, str):
                    normalized = decoded.strip().lower()
                    if normalized in ("0", "1", "2"):
                        return self.labels.get(int(normalized), self.labels.get(state_index, "Unknown"))
                    if normalized in ("alert", "drowsy", "fatigued"):
                        return normalized.capitalize()
                    return decoded
                return self.labels.get(int(decoded), self.labels.get(state_index, "Unknown"))
            except Exception:
                pass

        try:
            return self.labels.get(int(class_value), self.labels.get(state_index, "Unknown"))
        except Exception:
            return self.labels.get(state_index, "Unknown")

    def _to_python_probs(self, probs):
        """Convert numpy probability arrays/scalars to plain Python floats for JSON serialization."""
        return [round(float(p), 2) for p in probs]

    def reset_calibration(self):
        """Resets the adaptive baseline for a new user/session."""
        self.base_ear = 0.32
        self.calibration_frames = 0
        self.history.clear()
        self.ema_probs = None
        self.current_state = 0
        print("[ML] 🔄 Calibration Reset!")

    def calculate_temporal_features(self, current_data):
        """Computes rolling mean/std from history."""
        self.history.append(current_data)
        data_matrix = np.array(self.history)
        
        if len(self.history) < 2:
            return {
                'ear_mean': current_data[0], 'ear_std': 0,
                'mar_mean': current_data[1], 'mar_std': 0,
                'pitch_mean': current_data[2], 'pitch_std': 0,
                'hr_mean': current_data[4], 'hr_std': 0, 
                'temp_mean': current_data[5]
            }

        means = np.mean(data_matrix, axis=0)
        stds = np.std(data_matrix, axis=0)
        
        return {
            'ear_mean': means[0], 'ear_std': stds[0],
            'mar_mean': means[1], 'mar_std': stds[1],
            'pitch_mean': means[2], 'pitch_std': stds[2],
            'hr_mean': means[4], 'hr_std': stds[4],
            'temp_mean': means[5]
        }

    def predict(self, sensor_data, vision_data):
        if self.model is None and not self.use_onnx:
            return {"status": "Unknown", "confidence": 0, "raw_probs": [0,0,0]}

        # --- 0. VALIDITY CHECK (SAFETY FIRST) ---
        status = vision_data.get("status", "Unknown")
        if status in ["No Face", "Unstable"]:
            # If no face is visible, we should not feed the model garbage data (EAR=0).
            # We decay the probability back towards Alert (0) to prevent stuck alarms.
            if self.ema_probs is not None:
                # Slowly drift towards Alert (0.1% chance shift per frame)
                target_probs = np.array([1.0, 0.0, 0.0])
                self.ema_probs = (0.05 * target_probs) + (0.95 * self.ema_probs)
                
                # Update current state based on drifted probs
                self.current_state = np.argmax(self.ema_probs)
            
            return {
                "status": self.labels.get(self.current_state, "Unknown"),
                "confidence": round(float(np.max(self.ema_probs)), 2) if self.ema_probs is not None else 0,
                "raw_probs": self._to_python_probs(self.ema_probs) if self.ema_probs is not None else [1.0, 0.0, 0.0],
                "flag": f"SKIPPED_{status.upper().replace(' ', '_')}"
            }

        try:
             # --- 1. ROBUST SENSOR IMPUTATION ---
            # Sensors (Arduino) might fail or return 0. We must NOT feed 0 into the model.
            
            # Default "Population Average" values (Fallback of last resort)
            DEFAULT_HR = 75.0
            DEFAULT_TEMP = 37.0
            
            # Check current readings
            curr_hr = sensor_data.get('hr', 0)
            curr_temp = sensor_data.get('temperature', 0)
            
            # If invalid (<=0), try to use history mean, else use default
            if curr_hr <= 0:
                if len(self.history) > 0:
                     # Calculate simple mean of valid HRs in history
                     valid_hrs = [frame[4] for frame in self.history if frame[4] > 0]
                     curr_hr = np.mean(valid_hrs) if valid_hrs else DEFAULT_HR
                else:
                    curr_hr = DEFAULT_HR
                    
            if curr_temp <= 0:
                if len(self.history) > 0:
                     valid_temps = [frame[5] for frame in self.history if frame[5] > 0]
                     curr_temp = np.mean(valid_temps) if valid_temps else DEFAULT_TEMP
                else:
                    curr_temp = DEFAULT_TEMP

            current_ear_raw = vision_data.get('ear', 0.3)
            current_mar = vision_data.get('mar', 0.0)
            closed_frames = vision_data.get('closed_frames', 0)
            closed_duration_sec = float(vision_data.get('closed_duration_sec', 0.0) or 0.0)
            
            head_yaw = abs(vision_data.get('head_angle_y', 0))
            head_pitch = abs(vision_data.get('head_angle_x', 0))

            # --- 2. POSE-BASED EAR CORRECTION ---
            # If head is turned (Yaw), EAR naturally decreases visually.
            # Correction: Slightly boost EAR if looking away to prevent false fatigue.
            correction_factor = 1.0 + (head_yaw / 90.0) * 0.2
            current_ear = current_ear_raw * correction_factor

            # --- 3. SENSOR INTEGRITY GUARD ---
            # Detect if Arduino is sending 'Frozen' data (Stuck sensor)
            if curr_hr == self.last_sensor_values["hr"] and curr_hr > 0:
                self.sensor_stale_count += 1
            else:
                self.sensor_stale_count = 0
                self.last_sensor_values["hr"] = curr_hr
            
            sensor_reliable = self.sensor_stale_count < 50 # Unreliable if stuck for 50 frames
            
            # --- 4. ADAPTIVE BASELINE CALIBRATION ---
            # Learn what 'Alert' looks like for THIS specific user
            if self.current_state == 0 and not closed_frames:
                if self.calibration_frames < self.max_calibration:
                    self.calibration_frames += 1
                    # Running Average for Baseline
                    self.base_ear = (self.base_ear * 0.95) + (current_ear * 0.05)

            # DEBUG LOGGING (Throttle to ~every 5 seconds)
            self.debug_counter += 1
            if self.debug_counter % 20 == 0:
                 print(f"[ML ROBUST] EAR: {current_ear:.3f} (Base: {self.base_ear:.2f}) | Yaw: {head_yaw:.1f} | Reliable: {sensor_reliable}")

            # MICROSLEEP DETECTION (Use reliable frame counter)
            # If eyes closed for > 15 frames (~0.5s), FORCE FATIGUE
            if closed_frames > self.microsleep_max_frames or closed_duration_sec >= self.microsleep_max_seconds:
                print(f"[ML SAFETY] Microsleep Detected! ({closed_frames} frames, {closed_duration_sec:.2f}s)")
                self.current_state = 2 # Force state
                return {
                    "status": "Fatigued",
                    "confidence": 1.0,
                    "raw_probs": [0.0, 0.0, 1.0], # Force probability
                    "flag": "MICROSLEEP"
                }

            # FALLBACK: High PERCLOS (Eyes closing frequently)
            # If PERCLOS > 55%, user is definitely tired regardless of other features.
            current_perclos = vision_data.get('perclos', 0.0)
            if current_perclos > 55.0:
                print(f"[ML SAFETY] High PERCLOS ({current_perclos}%) Detected!")
                self.current_state = 2
                return {
                    "status": "Fatigued",
                    "confidence": 1.0,
                    "raw_probs": [0.0, 0.0, 1.0],
                    "flag": "HIGH_PERCLOS"
                }


            # --- 1.5 SENSOR CONDITION CHECK (Initialize) ---
            sensor_condition_flag = None

            # --- 2. FEATURE ENGINEERING ---
            raw_sample = [
                current_ear,
                current_mar,
                vision_data.get('head_angle_x', 0), # Pitch
                vision_data.get('head_angle_y', 0), # Yaw
                curr_hr,    # IMPUTED
                curr_temp   # IMPUTED
            ]
            
            stats = self.calculate_temporal_features(raw_sample)
            
            feature_vector = [
                vision_data.get('perclos', 0),
                stats['ear_mean'], stats['ear_std'],
                stats['mar_mean'], stats['mar_std'],
                stats['pitch_mean'], stats['pitch_std'],
                stats['hr_mean'], stats['hr_std'],
                stats['temp_mean']
            ]
            
            # --- 3. PROBABILISTIC INFERENCE ---
            # Get raw probabilities [Alert%, Drowsy%, Fatigued%]
            if self.use_onnx and self.onnx_session is not None and self.onnx_input_name:
                xgb_features = self._build_xgb_features({
                    **vision_data,
                    "head_angle_z": vision_data.get("head_angle_z", 0.0),
                }, {
                    **sensor_data,
                    "hr": curr_hr,
                    "temperature": curr_temp,
                })
                x_scaled = self._apply_onnx_scaling(xgb_features)
                outputs = self.onnx_session.run(None, {self.onnx_input_name: x_scaled})
                raw_probs = self._extract_onnx_probs(outputs)
            elif self.use_xgb_pipeline:
                xgb_features = self._build_xgb_features({
                    **vision_data,
                    "head_angle_z": vision_data.get("head_angle_z", 0.0),
                }, {
                    **sensor_data,
                    "hr": curr_hr,
                    "temperature": curr_temp,
                })
                x_scaled = self.scaler.transform(xgb_features)
                raw_probs = self.model.predict_proba(x_scaled)[0]
            else:
                raw_probs = self.model.predict_proba(np.array([feature_vector], dtype=np.float32))[0]

            # --- 3.5 LOGICAL SENSOR OVERRIDES (Soft Integration) ---
            # Instead of a hard return, we bias the probabilities so the 
            # EMA Smoothing and State Machine handle it naturally (No Flicker).
            
            # THERMAL STRESS (> 37.8°C)
            if curr_temp >= 37.8:
                # Strong push towards Drowsy/Fatigued
                # We blend the model's prediction with a strong override vector
                override_vec = np.array([0.05, 0.85, 0.10]) # Mostly Drowsy
                if curr_temp > 39.0:
                    override_vec = np.array([0.0, 0.1, 0.9]) # Mostly Fatigued
                
                # Apply Override (90% weight to sensor, 10% to face)
                raw_probs = (0.1 * raw_probs) + (0.9 * override_vec)
                sensor_condition_flag = "THERMAL_STRESS"

            # BRADYCARDIA (HR < 50)
            elif 0 < curr_hr < 50:
                # Drowsy Bias
                override_vec = np.array([0.1, 0.8, 0.1])
                raw_probs = (0.2 * raw_probs) + (0.8 * override_vec)
                sensor_condition_flag = "CARDIAC_ANOMALY"
            
            # --- 4. ADVANCED SMOOTHING (EMA) ---
            if self.ema_probs is None:
                self.ema_probs = raw_probs
            else:
                # Update EMA: New = alpha * current + (1 - alpha) * old
                self.ema_probs = (self.alpha * raw_probs) + ((1 - self.alpha) * self.ema_probs)
                
            # Normalize to sum to 1
            self.ema_probs /= np.sum(self.ema_probs)
            
            # --- 5. HYSTERESIS / STATE MACHINE ---
            # 0: Alert, 1: Drowsy, 2: Fatigued
            
            proposed_state = np.argmax(self.ema_probs)
            confidence = self.ema_probs[proposed_state]
            
            # Transition Logic
            # "Movie-like" Progression: Alert (0) <-> Drowsy (1) <-> Fatigued (2)
            # You generally shouldn't jump from Alert to Fatigued instantly unless it's a safety override (handled above).
            
            # 1. Enforce Sequential Steps if possible
            if self.current_state == 0 and proposed_state == 2:
                 # IF attempting to jump Alert -> Fatigued, force Drowsy first unless confidence is MASSIVE
                 if confidence < 0.9:
                     proposed_state = 1
                     
            if self.current_state == 2 and proposed_state == 0:
                 # IF attempting to jump Fatigued -> Alert, force Drowsy first
                 proposed_state = 1

            # 2. Persistence Check (Must hold state for N frames)
            if proposed_state != self.current_state:
                self.state_persistence += 1
                if self.state_persistence >= self.required_persistence:
                     self.current_state = proposed_state
                     self.state_persistence = 0
            else:
                self.state_persistence = 0
            
            final_label = self._get_final_label(self.current_state)
            
            # Ensure we have a flag if the model detects fatigue but no sensor override exists
            if sensor_condition_flag is None and self.current_state > 0:
                sensor_condition_flag = "BIO_OCULAR_PATTERN"

            return {
                "status": final_label,
                "confidence": round(float(confidence), 2),
                "raw_probs": self._to_python_probs(self.ema_probs),
                "flag": sensor_condition_flag
            }
            
        except Exception as e:
            print(f"[ML ERROR] {e}")
            return {"status": "Error", "confidence": 0}
