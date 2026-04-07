# Driver Fatigue Detection System - Project Report

## 1. Problem Statement

Drowsy driving is a significant factor in traffic accidents worldwide. Fatigue diminishes a driver's reaction time, awareness of hazards, and ability to sustain attention. Traditional methods of detecting fatigue (such as simpler lane deviation warnings) are often reactive rather than proactive. There is a critical need for a **non-intrusive, real-time monitoring system** that can accurately detect the early signs of driver fatigue—specifically microsleeps, frequent blinking, and head nodding—before an accident occurs.

## 2. Proposed Solution

We propose a **Multi-Modal Driver Fatigue Detection System** that combines Computer Vision (CV) with Biological Sensors (IoT) for high-accuracy prediction. Unlike systems that rely solely on one metric, our solution fuses visual behavioral data with physiological data to minimize false positives and increase reliability.

### Key Components:

1.  **Computer Vision Module (The "See"):**
    - Utilizes **MediaPipe Face Mesh** for high-precision facial landmark tracking.
    - **PERCLOS (Percentage of Eyelid Closure):** Calculates the Eye Aspect Ratio (EAR) to detect drowsiness and microsleeps.
    - **Head Pose Estimation:** Tracks Pitch, Yaw, and Roll to detect head nodding or distraction (looking away from the road).
    - **Blink Frequency Analysis:** Monitors blink rates to detect rapid blinking associated with early fatigue.

2.  **Biological Sensor Module (The "Feel"):**
    - Integrates an **Arduino-based sensor node** to monitor physiological signals.
    - **Heart Rate Monitoring:** Detects Bradycardia (low heart rate) associated with relaxation and sleep onset.
    - **Temperature Sensing:** Monitors body temperature to detect thermal stress which can induce fatigue.

3.  **Machine Learning Core (The "Brain"):**
    - A **Random Forest Classifier** trained on a hybrid dataset of visual and biological features.
    - **Real-time Inference:** predicts user state (Alert, Drowsy, Fatigued) every ~0.5 seconds.
    - **Features Used:** EAR Mean/Std, MAR (Mouth Aspect Ratio), Head Angles, Heart Rate, Temperature.

4.  **Full-Stack Application:**
    - **Backend:** Python **FastAPI** for high-performance extraction and WebSocket streaming.
    - **Frontend:** **React.js** dashboard for real-time visualization of the driver's face mesh, sensor stats, and alert status.

## 3. Feasibility Analysis

- **Technical Feasibility:** The system is built using open-source, mature technologies (Python, OpenCV, React). It runs efficiently on standard CPUs by optimizing frame processing (analyzing 80% of frames), making it viable for deployment without high-end GPUs.
- **Economic Feasibility:** The hardware requirements are minimal: a standard webcam and low-cost bio-sensors (like Pulse/Temp sensors). This makes the solution affordable for commercial fleets or individual consumers compared to expensive proprietary LIDAR or EEG systems.
- **Operational Feasibility:** The system is non-intrusive. The webcam monitoring requires no contact, and the bio-sensors can be integrated into a steering wheel cover or smartwatch, ensuring driver comfort is not compromised.

## 4. Current Project Status

The project has successfully reached the **Functional Prototype** stage.

- **Core Architecture:** The backend has been fully migrated to **FastAPI**, enabling robust asynchronous WebSocket communication for real-time video streaming.
- **Performance:** The system is currently tuned to process **80% of incoming video frames** (4 out of every 5), achieving a high-precision analysis loop that runs effectively in near real-time.
- **Algorithm:** The **Hybrid Machine Learning Engine** is active. It successfully aggregates:
  - **Visual Data:** PERCLOS (Eye closure), Mouth Aspect Ratio (Yawning), and Head Orientation.
  - **Physiological Data:** Heart rate and temperature inputs from the integrated sensor bridge.
- **User Interface:** The React-based frontend is fully operational, providing a live "Pilot Dashboard" that visualizes the driver's face mesh, real-time sensor statistics, and the system's fatigue prediction confidence.

## 5. Future Scope

1.  **Edge Deployment:** Porting the backend to **NVIDIA Jetson Nano** or **Raspberry Pi 5** for a completely standalone, offline device that sits on the dashboard.
2.  **Mobile Application:** Developing a lightweight Android/iOS version that uses the phone's front camera for detection, democratizing access to safety.
3.  **Connected Fleet Management:** Integrating with 4G/5G modules to send real-time alerts not just to the driver, but to a centralized fleet control center for truck/bus companies.
4.  **Night Vision Support:** Replacing the standard webcam with an **IR (Infrared) Camera** to allow accurate detection in complete darkness or low-light driving conditions.
5.  **GPS Integration:** Correlating fatigue events with location data to identify "fatigue hotspots" on highways (e.g., long monotonous stretches).
