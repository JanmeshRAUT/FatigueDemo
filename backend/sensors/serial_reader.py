"""
Serial communication module for reading sensor data from Arduino.
Handles connection, data parsing, and thread-safe state management.
"""
import logging
import serial
import serial.tools.list_ports
import threading
import time
import math
import traceback
import random
from collections import deque

from config import get_config

logger = logging.getLogger(__name__)
config = get_config()

# --- Global State ---
sensor_lock = threading.Lock()

latest_sensor_data = {
    "temperature": None,
    "ax": None, "ay": None, "az": None,
    "gx": None, "gy": None, "gz": None,
    "hr": None, "spo2": None,
    "timestamp": None
}

sensor_data_history = deque(maxlen=config.MAX_HISTORY)

head_position_data = {
    "position": "Center",
    "angle_x": 0.0,  # up-down
    "angle_y": 0.0,  # left-right
    "timestamp": int(time.time())
}

def generate_mock_sensor_data():
    """Generate realistic mock sensor data for testing (when Arduino unavailable)"""
    return {
        "temperature": round(36.5 + random.uniform(-0.5, 0.5), 1),
        "ax": round(random.uniform(-5, 5), 2),
        "ay": round(random.uniform(-5, 5), 2),
        "az": round(random.uniform(8, 10), 2),
        "gx": round(random.uniform(-2, 2), 2),
        "gy": round(random.uniform(-2, 2), 2),
        "gz": round(random.uniform(-2, 2), 2),
        "hr": random.randint(60, 100),
        "spo2": round(random.uniform(95, 99.5), 1),
        "timestamp": int(time.time())
    }

def parse_raw_sensor_string(raw: str):
    """Parse comma-separated sensor data string into dictionary"""
    parts = [p.strip() for p in raw.split(',') if p.strip()]
    parsed = {}
    for p in parts:
        if ':' not in p:
            continue
        k, v = p.split(':', 1)
        k, v = k.lower(), v.strip()
        try:
            if k in ("t", "temperature"):
                parsed["temperature"] = float(v)
            elif k in ("hr", "bpm", "heart_rate"):
                parsed["hr"] = float(v)
            elif k in ("spo2", "sp02"):
                parsed["spo2"] = float(v)
            elif k in ("ax", "ay", "az", "gx", "gy", "gz"):
                parsed[k] = float(v)
        except ValueError:
            logger.debug(f"Failed to parse sensor value: {k}={v}")
            continue
    return parsed


def find_arduino_port():
    """Auto-detect Arduino port, with fallback to configured port"""
    try:
        ports = [p.device for p in serial.tools.list_ports.comports()]
        logger.debug(f"Available ports: {ports}")
        
        # Try configured port first
        if config.ARDUINO_PORT in ports:
            logger.info(f"Using configured port: {config.ARDUINO_PORT}")
            return config.ARDUINO_PORT
        
        # Auto-detect: prefer CH340 (common Arduino clone) or any available port
        for p in serial.tools.list_ports.comports():
            if 'CH340' in p.description or 'Arduino' in p.description or 'USB' in p.description:
                logger.info(f"Auto-detected Arduino port: {p.device}")
                return p.device
        
        # Fallback: use first available port
        if ports:
            logger.warning(f"No Arduino detected, using first available: {ports[0]}")
            return ports[0]
        
        # No ports available
        logger.warning(f"No serial ports found, defaulting to {config.ARDUINO_PORT}")
        return config.ARDUINO_PORT
        
    except Exception as e:
        logger.warning(f"Error detecting port: {e}, using default {config.ARDUINO_PORT}")
        return config.ARDUINO_PORT

def calculate_head_position(ax, ay, az):
    """Calculate head position angles from accelerometer values"""
    try:
        # PITCH (Up/Down) - Rotation around X-axis
        angle_x = math.degrees(math.atan2(ax, math.sqrt(ay**2 + az**2)))
        
        # YAW (Left/Right) - Rotation around Y-axis
        angle_y = math.degrees(math.atan2(ay, math.sqrt(ax**2 + az**2)))
        
        # ROLL (Tilt Left/Right) - Rotation around Z-axis
        angle_z = math.degrees(math.atan2(ay, az))

        UP_THRESHOLD = 10
        DOWN_THRESHOLD = -10
        LEFT_THRESHOLD = -10
        RIGHT_THRESHOLD = 10

        if angle_x > UP_THRESHOLD:
            vertical = "Up"
        elif angle_x < DOWN_THRESHOLD:
            vertical = "Down"
        else:
            vertical = "Center"

        if angle_y > RIGHT_THRESHOLD:
            horizontal = "Right"
        elif angle_y < LEFT_THRESHOLD:
            horizontal = "Left"
        else:
            horizontal = "Center"
            
        if vertical == "Center" and horizontal == "Center":
            position = "Center"
        elif vertical != "Center" and horizontal == "Center":
            position = vertical
        elif horizontal != "Center" and vertical == "Center":
            position = horizontal
        else:
            position = f"{vertical}-{horizontal}"

        return position, angle_x, angle_y, angle_z

    except Exception as e:
        logger.error(f"Head position calculation error: {e}", exc_info=True)
        return "Unknown", 0.0, 0.0, 0.0

def update_head_position_data():
    """Updates the global head_position_data based on latest sensor values."""
    ax = latest_sensor_data["ax"]
    ay = latest_sensor_data["ay"]
    az = latest_sensor_data["az"]

    if ax is None or ay is None or az is None:
        return head_position_data # Return existing or update to unknown?

    pos, ang_x, ang_y, ang_z = calculate_head_position(ax, ay, az)

    head_position_data.update({
        "position": pos,
        "angle_x": round(ang_x, 2),
        "angle_y": round(ang_y, 2),
        "angle_z": round(ang_z, 2),
        "timestamp": int(time.time())
    })
    return head_position_data

def serial_reader():
    """Main serial reading loop with auto-reconnect and fallback to mock data"""
    connection_attempts = 0
    last_port = None
    using_mock_data = False
    MAX_CONNECTION_ATTEMPTS = 5

    while True:
        # Auto-detect port on each reconnection attempt
        port = find_arduino_port()
        
        # Log port change
        if port != last_port:
            logger.info(f"Attempting connection on {port}...")
            last_port = port
        
        # Switch to mock data after too many failures
        if connection_attempts >= MAX_CONNECTION_ATTEMPTS and not using_mock_data:
            logger.warning(f"[FALLBACK] Switching to mock sensor data after {MAX_CONNECTION_ATTEMPTS} failed attempts")
            using_mock_data = True
        
        # Use mock data mode
        if using_mock_data and config.USE_MOCK_DATA:
            try:
                mock_data = generate_mock_sensor_data()
                with sensor_lock:
                    latest_sensor_data.update(mock_data)
                    sensor_data_history.append(latest_sensor_data.copy())
                time.sleep(1)  # Poll at 1Hz for mock data
                continue
            except Exception as e:
                logger.error(f"Mock data generation error: {e}", exc_info=True)
                time.sleep(5)
                continue
        
        # Try real serial connection
        ser = None
        try:
            # Attempt Connection
            ser = serial.Serial(port=port, baudrate=config.BAUD_RATE, timeout=1)
            logger.info(f"[OK] Connected to {port}")
            using_mock_data = False
            connection_attempts = 0
            time.sleep(2)  # Stabilize
            
            # Reading Loop
            while True:
                if ser.in_waiting > 0:
                    line = ser.readline().decode('utf-8', errors='ignore').strip()
                    if not line:
                        continue
                    
                    logger.debug(f"Raw data from Arduino: {line}")
                    print(f"arduino_data: {line}") # Explicit print for console visibility
                    
                    parsed = parse_raw_sensor_string(line)
                    if not parsed:
                        continue

                    timestamp = int(time.time())
                    with sensor_lock:
                        latest_sensor_data.update({**parsed, "timestamp": timestamp})
                        sensor_data_history.append(latest_sensor_data.copy())
                
                # Prevent CPU hogging
                time.sleep(0.01)

        except (serial.SerialException, PermissionError) as e:
            connection_attempts += 1
            logger.warning(f"[ATTEMPT {connection_attempts}] Serial connection failed: {type(e).__name__}: {e}")
            
        except Exception as e:
            logger.error(f"[ERROR] Unexpected error in serial reader: {e}", exc_info=True)

        finally:
             if ser:
                 try:
                     ser.close()
                 except Exception as e:
                     logger.debug(f"Error closing serial port: {e}")
        
        # Exponential backoff for reconnection (3s, 6s, 9s, 10s max)
        retry_delay = min(10, 3 * (connection_attempts // 3 + 1))
        logger.info(f"Retrying in {retry_delay}s... (will auto-detect available port)")
        time.sleep(retry_delay)


def start_serial_thread():
    """Start serial reader in background thread"""
    t = threading.Thread(target=serial_reader, daemon=True)
    t.start()
    logger.info("Serial reader thread started in background")
