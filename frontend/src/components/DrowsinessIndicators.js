import React, { useEffect, useRef } from "react";
import { CheckCircle, AlertTriangle, XCircle, Volume2, VolumeX } from "lucide-react";
import { useFatigueData } from "../hooks/useFatigueData";
import "./Css/DrowsinessIndicators.css";

const THRESHOLDS = {
  TEMP_HIGH: 38.0,
  PERCLOS_MEDIUM: 33,
  PERCLOS_HIGH: 75,
  SPO2_LOW: 60,
  HR_LOW: 50,
  HR_HIGH: 110,
};

export default function FatigueIndicator() {
  const data = useFatigueData();
  const prevLevelRef = useRef("LOW");
  const prevMicrosleepRef = useRef(false);
  const alertAudioRef = useRef(null);
  const microsleepAudioRef = useRef(null);

  useEffect(() => {
    alertAudioRef.current = new Audio("/sounds/alert.mp3");
    alertAudioRef.current.loop = true;
    alertAudioRef.current.volume = 1.0;
    alertAudioRef.current.preload = "auto";

    microsleepAudioRef.current = new Audio("/sounds/alert.mp3");
    microsleepAudioRef.current.loop = true;
    microsleepAudioRef.current.volume = 1.0;
    microsleepAudioRef.current.playbackRate = 1.2;
    microsleepAudioRef.current.preload = "auto";

    return () => {
      [alertAudioRef.current, microsleepAudioRef.current].forEach((audio) => {
        if (!audio) return;
        audio.pause();
        audio.currentTime = 0;
      });
    };
  }, []);

  const evaluateFatigue = () => {
    // 1. Check Initialization (Direct)
    if (data.system_status === "Initializing") return "LOW";

    // 2. Check Initialization (Implicit via Prediction Status)
    if (data.ml_fatigue_status && data.ml_fatigue_status.startsWith("Initializing")) return "LOW";

    if (data.ml_fatigue_status && data.ml_fatigue_status !== 'Unknown' && data.ml_fatigue_status !== 'Error') {
      const ml = data.ml_fatigue_status;
      if (ml === "Fatigued") return "HIGH";
      if (ml === "Drowsy") return "MEDIUM";
      if (ml === "Alert") return "LOW";
    }

    let level = "LOW";

    const temp = parseFloat(data.temperature) || 0;
    const perclos = parseFloat(data.perclos) || 0;
    const yawn = data.yawn_status || "Closed";
    const hr = parseFloat(data.hr) || 0;
    const spo2 = parseFloat(data.spo2) || 0;
    const eyeStatus = data.status || "Unknown";

    if (temp >= THRESHOLDS.TEMP_HIGH) level = "MEDIUM";
    if (perclos >= THRESHOLDS.PERCLOS_HIGH) level = "HIGH";
    else if (perclos >= THRESHOLDS.PERCLOS_MEDIUM && level !== "HIGH")
      level = "MEDIUM";
    if (yawn === "Yawning" || yawn === "Opening") level = "HIGH";
    if (spo2 && spo2 < THRESHOLDS.SPO2_LOW) level = "HIGH";
    if (hr && (hr < THRESHOLDS.HR_LOW || hr > THRESHOLDS.HR_HIGH)) {
      if (level !== "HIGH") level = "MEDIUM";
    }
    if (eyeStatus === "Closed") level = "HIGH";

    return level;
  };

  const level = evaluateFatigue();
  const isMicrosleep = String(data.ml_flag || "").toUpperCase() === "MICROSLEEP";

  const handleUserInteraction = () => {
    [alertAudioRef.current, microsleepAudioRef.current].forEach((audio) => {
      if (!audio) return;
      audio.play().then(() => {
        audio.pause();
        audio.currentTime = 0;
      }).catch((err) => console.log("Audio unlock failed (harmless if already unlocked):", err));
    });
  };

  const [isMuted, setIsMuted] = React.useState(false);

  useEffect(() => {
    const prevLevel = prevLevelRef.current;
    const prevMicrosleep = prevMicrosleepRef.current;
    
    const alertAudio = alertAudioRef.current;
    const microsleepAudio = microsleepAudioRef.current;
    if (!alertAudio || !microsleepAudio) return;

    if (!isMuted && isMicrosleep) {
      if (!prevMicrosleep || microsleepAudio.paused) {
        microsleepAudio.currentTime = 0;
        microsleepAudio.play().catch((err) => console.error("Microsleep buzzer blocked:", err));
      }
      if (!alertAudio.paused) {
        alertAudio.pause();
        alertAudio.currentTime = 0;
      }
    } else if (!isMuted && level === "HIGH") {
      if (alertAudio.paused || prevLevel !== "HIGH") {
        alertAudio.currentTime = 0;
        alertAudio.play().catch((err) => console.error("Alarm buzzer blocked:", err));
      }
      if (!microsleepAudio.paused) {
        microsleepAudio.pause();
        microsleepAudio.currentTime = 0;
      }
    } else {
      if (!alertAudio.paused) {
        alertAudio.pause();
        alertAudio.currentTime = 0;
      }
      if (!microsleepAudio.paused) {
        microsleepAudio.pause();
        microsleepAudio.currentTime = 0;
      }
    }

    prevLevelRef.current = level;
    prevMicrosleepRef.current = isMicrosleep;
  }, [level, isMuted, isMicrosleep]);

  const getConfig = (lvl) => {
    switch (lvl) {
      case "LOW":
        return { label: "NOMINAL", icon: <CheckCircle color="#16a34a" size={24} /> }; // Industry Term
      case "MEDIUM":
        return { label: "CAUTION", icon: <AlertTriangle color="#ca8a04" size={24} /> };
      case "HIGH":
        return { label: "CRITICAL", icon: <XCircle color="#dc2626" size={24} /> };
      default:
        return { label: "UNKNOWN", icon: <AlertTriangle color="#6b7280" size={24} /> };
    }
  };

  const { label, icon } = getConfig(level);

  return (
    <div className="fatigue-indicator-container">
      <div 
        className={`fatigue-indicator ${level.toLowerCase()}`} 
        onClick={handleUserInteraction}
        title="Status Indicator"
      >
        <div className="indicator-icon">{icon}</div>
        <span className="indicator-label" style={{marginRight: '8px'}}>{label}</span>
        
        <div 
            onClick={(e) => { 
                e.stopPropagation(); 
                setIsMuted(!isMuted); 
                // Don't call handleUserInteraction here to avoid double-trigger logic if needed, 
                // but actually we want to ensure Audio Context is unlocked, so it's fine.
                handleUserInteraction();
            }}
            title={isMuted ? "Unmute Alarm" : "Mute Alarm"}
            style={{
                display: 'flex', alignItems: 'center', justifyContent: 'center',
                padding: '4px',
                borderRadius: '4px',
                background: 'rgba(255,255,255,0.2)',
                cursor: 'pointer',
                transition: 'background 0.2s'
            }}
            className="mute-btn"
        >
            {isMuted ? <VolumeX size={14} color="white" /> : <Volume2 size={14} color="white" />}
        </div>
      </div>
    </div>
  );
}
