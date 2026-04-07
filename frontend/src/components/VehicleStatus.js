import React, { useEffect, useRef, useState } from "react";
import { Cpu, ShieldAlert, RotateCcw, Volume2, VolumeX } from "lucide-react";
import { useVehicleContext } from "../context/VehicleContext";
import { getConnectionMeta, getPerclosRiskBand, getPredictionMeta } from "../utils/vehicleStatus";
import "./Css/VehicleStatus.css";
import "./Css/VehicleShared.css";

function formatLabel(value) {
  if (!value) return "Unknown";
  return String(value)
    .replace(/_/g, " ")
    .replace(/\s+/g, " ")
    .trim()
    .replace(/\b\w/g, (char) => char.toUpperCase());
}

export default function VehicleStatus() {
  const { vehicleData, connectionStatus, resetCalibration } = useVehicleContext();
  const [isMuted, setIsMuted] = useState(false);
  const prevStatusRef = useRef("Unknown");
  const prevMicrosleepRef = useRef(false);
  const alertAudioRef = useRef(null);
  const microsleepAudioRef = useRef(null);

  const prediction = vehicleData?.prediction || {};
  const perclos = vehicleData?.perclos || {};
  const predictedStatus = prediction.status || "Unknown";
  const confidence = typeof prediction.confidence === "number" ? prediction.confidence : 0;
  const microsleep = prediction.microsleep_detected === true;
  const statusMeta = getPredictionMeta(predictedStatus, connectionStatus, Boolean(vehicleData));
  const connectionMeta = getConnectionMeta(connectionStatus);

  const confidencePct = (confidence * 100).toFixed(1);
  const perclosPct = Number(perclos.perclos || 0).toFixed(1);
  const earVal = Number(perclos.ear || 0).toFixed(2);
  const marVal = Number(perclos.mar || 0).toFixed(2);
  const eyeState = perclos.status || "Unknown";
  const yawnState = perclos.yawn_status || "N/A";
  const riskBand = getPerclosRiskBand(Number(perclos.perclos || 0));
  const calibrationProgress = prediction.calibration_progress || null;
  const reasonLabel = formatLabel(prediction.flag || (microsleep ? "MICROSLEEP_EVENT" : riskBand));
  const summaryLine = calibrationProgress
    ? `Calibration ${calibrationProgress}`
    : statusMeta.tone === "danger"
    ? "Escalated fatigue signals detected"
    : statusMeta.tone === "warning"
    ? "Cautionary signal shift detected"
    : "Signal profile remains stable";

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

  useEffect(() => {
    const alertAudio = alertAudioRef.current;
    const microsleepAudio = microsleepAudioRef.current;
    if (!alertAudio || !microsleepAudio) return;

    const prevStatus = prevStatusRef.current;
    const prevMicrosleep = prevMicrosleepRef.current;
    const isHigh = predictedStatus === "Fatigued";

    if (!isMuted && microsleep) {
      if (!prevMicrosleep || microsleepAudio.paused) {
        microsleepAudio.currentTime = 0;
        microsleepAudio.play().catch(() => {});
      }
      if (!alertAudio.paused) {
        alertAudio.pause();
        alertAudio.currentTime = 0;
      }
    } else if (!isMuted && isHigh) {
      if (alertAudio.paused || prevStatus !== "Fatigued") {
        alertAudio.currentTime = 0;
        alertAudio.play().catch(() => {});
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

    prevStatusRef.current = predictedStatus;
    prevMicrosleepRef.current = microsleep;
  }, [predictedStatus, microsleep, isMuted]);

  const unlockAudio = () => {
    [alertAudioRef.current, microsleepAudioRef.current].forEach((audio) => {
      if (!audio) return;
      audio.play().then(() => {
        audio.pause();
        audio.currentTime = 0;
      }).catch(() => {});
    });
  };

  const handleReset = async () => {
    await resetCalibration();
  };

  return (
    <div className="vehicle-status-container">
      <div className="vehicle-status-header">
        <div className="vehicle-status-header-info">
          <div className="vehicle-status-model-icon">
            <Cpu size={16} color="#6366f1" />
            <span className="vehicle-status-model-label">VEHICLE_ML</span>
          </div>
        </div>
        <div className="vehicle-status-header-info">
          <span className={`vehicle-status-chip tone-${connectionMeta.tone}`}>{connectionMeta.label}</span>
          <button className="vehicle-reset-btn" onClick={handleReset} title="Reset vehicle calibration">
            <RotateCcw size={14} />
          </button>
          <button
            className="vehicle-reset-btn"
            onClick={() => {
              setIsMuted((prev) => !prev);
              unlockAudio();
            }}
            title={isMuted ? "Unmute buzzer" : "Mute buzzer"}
          >
            {isMuted ? <VolumeX size={14} /> : <Volume2 size={14} />}
          </button>
        </div>
      </div>

      <div className={`vehicle-prediction-box ${statusMeta.tone}`}>
        <div className="vehicle-prediction-hero">
          <div className="vehicle-prediction-class-header">
            <span className="vehicle-prediction-class-label">LIVE PREDICTION</span>
            <span className="vehicle-prediction-class-value">
              {statusMeta.label}
            </span>
          </div>

          <div className="vehicle-prediction-summary">
            <div className="vehicle-confidence-orb-wrap">
              <div className={`vehicle-confidence-orb orb-${statusMeta.tone}`}>
                <span>{confidencePct}%</span>
              </div>
            </div>
            <div className="vehicle-prediction-copy">
              <div className="vehicle-prediction-copy-title">Inference Reason</div>
              <div className="vehicle-prediction-copy-value">{reasonLabel}</div>
              <div className="vehicle-prediction-copy-subvalue">
                {summaryLine}
              </div>
            </div>
          </div>
        </div>

        {microsleep && (
          <div className="vehicle-microsleep-alert">
            <p>
              <ShieldAlert size={14} /> Microsleep event detected
            </p>
          </div>
        )}

        <div>
          <div className="vehicle-confidence-header">
            <span className="vehicle-confidence-label">MODEL CONFIDENCE</span>
          </div>
          <div className="vehicle-confidence-bar">
            <div
              className={`vehicle-confidence-fill ${statusMeta.tone}`}
              style={{ width: `${confidencePct}%` }}
            ></div>
          </div>
        </div>

        <div className="vehicle-live-tags">
          <span className={`vehicle-live-tag tone-${connectionMeta.tone}`}>Link {connectionMeta.label}</span>
          <span className="vehicle-live-tag tone-neutral">Eyes {eyeState}</span>
          <span className="vehicle-live-tag tone-neutral">Yawn {yawnState}</span>
        </div>
      </div>

      <div className="vehicle-command-snapshot">
        <div className="vehicle-command-snapshot-header">CURRENT SIGNAL SNAPSHOT</div>
        <div className="vehicle-command-metrics">
          <div className="vehicle-command-metric-item">
            <span className="vehicle-command-metric-label">PERCLOS</span>
            <strong className="vehicle-command-metric-value">{perclosPct}%</strong>
          </div>
          <div className="vehicle-command-metric-item">
            <span className="vehicle-command-metric-label">EAR</span>
            <strong className="vehicle-command-metric-value">{earVal}</strong>
          </div>
          <div className="vehicle-command-metric-item">
            <span className="vehicle-command-metric-label">MAR</span>
            <strong className="vehicle-command-metric-value">{marVal}</strong>
          </div>
          <div className="vehicle-command-metric-item">
            <span className="vehicle-command-metric-label">EYE STATE</span>
            <strong className="vehicle-command-metric-value vehicle-command-metric-reason">{formatLabel(eyeState)}</strong>
          </div>
      </div>
      </div>

      <div className="vehicle-status-footnote">
        Live prediction is updated from the current vehicle vision stream.
      </div>
    </div>
  );
}
