import os


def _as_bool(value, default=False):
    if value is None:
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "on"}

class Config:
    """
    Central Configuration for the Backend Application
    """
    
    # --- Serial / Sensor Configuration ---
    # Defaulting to COM6 as per recent user diagnostics, but serial_reader will also auto-detect.
    ARDUINO_PORT = os.environ.get("ARDUINO_PORT", "COM6") 
    BAUD_RATE = int(os.environ.get("BAUD_RATE", 115200))
    MAX_HISTORY = 100 # Number of sensor readings to keep in memory
    SENSOR_TIMEOUT = 2.0 # Seconds before sensor data is considered "Stale"
    USE_MOCK_DATA = _as_bool(os.environ.get("USE_MOCK_DATA"), False) # Enable random data generation when sensors are disconnected
    
    # --- ML Engine Configuration ---
    MODEL_PATH = os.path.join(os.path.dirname(__file__), "ml", "models", "xgboost_independent.onnx")
    LEGACY_MODEL_PATH = os.path.join(os.path.dirname(__file__), "ml", "models", "fatigue_model.pkl")
    MODEL_SCALER_PATH = os.path.join(os.path.dirname(__file__), "ml", "models", "xgb_scaler_params.json")
    MODEL_LABEL_ENCODER_PATH = os.path.join(os.path.dirname(__file__), "ml", "models", "xgb_label_encoder.joblib")
    LEGACY_VEHICLE_MODEL_PATH = os.path.join(os.path.dirname(__file__), "ml", "models", "vehicle_fatigue_model.pkl")
    VEHICLE_MODEL_PATH = os.path.join(os.path.dirname(__file__), "ml", "models", "xgboost_independent.onnx")
    VEHICLE_SCALER_PATH = os.path.join(os.path.dirname(__file__), "ml", "models", "xgb_scaler_params.json")
    VEHICLE_LABEL_ENCODER_PATH = os.path.join(os.path.dirname(__file__), "ml", "models", "xgb_label_encoder.joblib")
    ML_INTERVAL = 0.5 # Seconds between ML predictions to prevent CPU overload
    
    # --- Logging / Debug ---
    DEBUG = True

def get_config():
    return Config()
