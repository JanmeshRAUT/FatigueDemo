import requests
import serial
import serial.tools.list_ports
import time
import json
import traceback

# --- CONFIGURATION ---
ARDUINO_PORT = "COM6"
BAUD_RATE = 115200
BACKEND_URL = "https://nulliporous-carbolic-lianne.ngrok-free.dev"  
API_ENDPOINT = f"{BACKEND_URL}/sensor_data/ingest"  

def find_arduino():
    ports = list(serial.tools.list_ports.comports())
    for p in ports:
        if "Arduino" in p.description or "Serial" in p.description:
            return p.device
    return ARDUINO_PORT

def bridge():
    print(f"🌉 Starting Hardware Bridge...")
    print(f"🚀 Target Backend: {BACKEND_URL}")
    
    port = find_arduino()
    print(f"🔌 Connecting to Port: {port}")

    try:
        ser = serial.Serial(port, BAUD_RATE, timeout=1)
        time.sleep(2) # Allow reset
        print("✅ Serial Connection Established")
        
        while True:
            if ser.in_waiting > 0:
                line = ser.readline().decode('utf-8', errors='ignore').strip()
                if not line: continue
                
                print(f"📥 [RAW]: {line}")
                
                # Parse
                # Expected format: "T:36.5, HR:72, Ax:0.1..."
                # We can just send the raw string to the backend to parse, 
                # OR parse it here. Sending raw string is easier.
                
                payload = {"raw_sensor_data": line}
                
                try:
                    # POST to Cloud
                    # Note: We need to implement /ingest endpoint in server.py
                    # For now, let's just print what we WOULLD do
                    response = requests.post(API_ENDPOINT, json=payload, timeout=2)
                    # pass
                except Exception as req_e:
                    print(f"⚠️ Cloud Upload Failed: {req_e}")

    except Exception as e:
        print(f"❌ Bridge Error: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    bridge()
