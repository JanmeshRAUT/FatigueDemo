import requests
import csv
import time
import os

# ---------------- CONFIG ----------------
FLASK_URL = "http://localhost:5000/combined_data"  # Flask backend
SAVE_FILE = "data/auto_labeled_fatigue_dataset.csv"
SAMPLE_INTERVAL = 2  

# ---------------- SETUP ----------------
header = [
    "timestamp", "temperature", "hr", "spo2",
    "ax", "ay", "az", "gx", "gy", "gz",
    "perclos", "ear", "mar", "yawn_status",
    "angle_x", "angle_y", "auto_label"
]

if not os.path.exists(SAVE_FILE):
    with open(SAVE_FILE, "w", newline="") as f:
        csv.writer(f).writerow(header)

print("========================================")
print("🤖 AUTO-LABELED FATIGUE DATA COLLECTION")
print("========================================")
print(f"Connected to Flask backend: {FLASK_URL}")
print(f"Saving to: {SAVE_FILE}")
print("----------------------------------------\n")

# ---------------- LABELING LOGIC ----------------
def determine_label(temperature, hr, spo2, perclos, yawn_status):

    try:
        temperature = float(temperature) if temperature else 0
        hr = float(hr) if hr else 0
        spo2 = float(spo2) if spo2 else 0
        perclos = float(perclos) if perclos else 0
        yawn = yawn_status.lower() if isinstance(yawn_status, str) else ""

        if perclos > 50 or yawn == "yawning":
            return "Fatigued"
        elif perclos > 25 or hr > 100 or temperature > 37.5:
            return "Drowsy"
        else:
            return "Alert"
    except Exception:
        return "Unknown"

# ---------------- DATA COLLECTION LOOP ----------------
count = 0
while True:
    try:
        response = requests.get(FLASK_URL, timeout=5)
        if response.status_code != 200:
            print(f"⚠️ Flask returned {response.status_code}")
            time.sleep(SAMPLE_INTERVAL)
            continue

        data = response.json()
        sensor = data.get("sensor", {})
        perclos = data.get("perclos", {})
        head = data.get("head_position", {})

        # Extract features safely
        temperature = sensor.get("temperature")
        hr = sensor.get("hr")
        spo2 = sensor.get("spo2")
        ax, ay, az = sensor.get("ax"), sensor.get("ay"), sensor.get("az")
        gx, gy, gz = sensor.get("gx"), sensor.get("gy"), sensor.get("gz")
        perclos_val = perclos.get("perclos")
        ear = perclos.get("ear")
        mar = perclos.get("mar")
        yawn_status = perclos.get("yawn_status")
        angle_x = head.get("angle_x")
        angle_y = head.get("angle_y")

        # Auto Label
        label = determine_label(temperature, hr, spo2, perclos_val, yawn_status)

        # Save row
        row = [
            int(time.time()),
            temperature, hr, spo2,
            ax, ay, az, gx, gy, gz,
            perclos_val, ear, mar, yawn_status,
            angle_x, angle_y, label
        ]

        with open(SAVE_FILE, "a", newline="") as f:
            csv.writer(f).writerow(row)

        count += 1
        print(f"✅ [{count}] Logged — Label: {label}")
        time.sleep(SAMPLE_INTERVAL)

    except KeyboardInterrupt:
        print("\n🛑 Stopped data collection.")
        break
    except Exception as e:
        print(f"⚠️ Error fetching/saving data: {e}")
        time.sleep(SAMPLE_INTERVAL)
