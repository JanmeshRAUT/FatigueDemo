import React, { useEffect, useMemo, useState } from "react";
import { Bell, Heart, Thermometer, Activity, Eye, Gauge, ShieldAlert } from "lucide-react";
import { useVehicleContext } from "../context/VehicleContext";
import "./Css/OwnerDashboard.css";

function MetricCard({ icon, label, value, subValue, tone = "neutral" }) {
  return (
    <div className={`owner-metric-card tone-${tone}`}>
      <div className="owner-metric-header">
        <span className="owner-metric-icon">{icon}</span>
        <span className="owner-metric-label">{label}</span>
      </div>
      <div className="owner-metric-value">{value}</div>
      {subValue ? <div className="owner-metric-subvalue">{subValue}</div> : null}
    </div>
  );
}

function formatTimestamp(ts) {
  if (!ts) return "N/A";
  const date = new Date(ts * 1000);
  if (Number.isNaN(date.getTime())) return "N/A";
  return date.toLocaleString();
}

export default function OwnerDashboard() {
  const { vehicleData } = useVehicleContext() || {};
  const [alertHistory, setAlertHistory] = useState([]);
  const [selectedVehicle, setSelectedVehicle] = useState(() => localStorage.getItem("owner_selected_vehicle") || "TRUCK-01");

  const fullData = vehicleData || {};
  const prediction = fullData?.prediction || {};
  const sensor = fullData?.sensor || {};
  const perclos = fullData?.perclos || {};
  const head = fullData?.head_position || {};

  const driverStatus = prediction.status || "Unknown";
  const confidencePct = Math.round((Number(prediction.confidence) || 0) * 100);
  const ownerRiskTone =
    driverStatus === "Fatigued"
      ? "danger"
      : driverStatus === "Drowsy"
      ? "warning"
      : "safe";

  const backendVehicleId = fullData?.vehicle_id || fullData?.truck_id || null;

  useEffect(() => {
    localStorage.setItem("owner_selected_vehicle", selectedVehicle);
  }, [selectedVehicle]);

  useEffect(() => {
    const shouldLogAlert = driverStatus === "Drowsy" || driverStatus === "Fatigued";
    if (!shouldLogAlert) return;

    setAlertHistory((prev) => {
      const last = prev[0];
      if (last && last.status === driverStatus) {
        return prev;
      }

      const entry = {
        id: Date.now(),
        status: driverStatus,
        confidence: confidencePct,
        flag: prediction.flag || "N/A",
        time: new Date().toLocaleTimeString(),
      };
      return [entry, ...prev].slice(0, 12);
    });
  }, [driverStatus, confidencePct, prediction.flag]);

  const motionSummary = useMemo(() => {
    const values = [sensor.ax, sensor.ay, sensor.az].map((v) => Number(v) || 0);
    const magnitude = Math.sqrt(values[0] ** 2 + values[1] ** 2 + values[2] ** 2);
    return magnitude.toFixed(2);
  }, [sensor.ax, sensor.ay, sensor.az]);

  return (
    <div className="owner-dashboard">
      <section className="owner-hero">
        <div>
          <div className="owner-kicker">OWNER MONITOR</div>
          <h2 className="owner-title">Driver Live Health and Fatigue Feed</h2>
          <p className="owner-subtitle">
            Vehicle-mode live stream for owner operations: fatigue status, vision signals, and sensor telemetry.
          </p>
          <div className="owner-vehicle-row">
            <label htmlFor="ownerVehicle" className="owner-vehicle-label">Monitoring Vehicle</label>
            <select
              id="ownerVehicle"
              className="owner-vehicle-select"
              value={selectedVehicle}
              onChange={(e) => setSelectedVehicle(e.target.value)}
            >
              <option value="TRUCK-01">TRUCK-01</option>
              <option value="TRUCK-02">TRUCK-02</option>
              <option value="TRUCK-03">TRUCK-03</option>
            </select>
            <span className="owner-vehicle-badge">Live Source: {backendVehicleId || "Vehicle Mode Feed"}</span>
          </div>
        </div>
        <div className={`owner-status-chip chip-${ownerRiskTone}`}>
          <ShieldAlert size={18} />
          <div>
            <div className="owner-chip-label">Current Driver State</div>
            <div className="owner-chip-value">{driverStatus}</div>
          </div>
        </div>
      </section>

      <section className="owner-grid">
        <MetricCard
          icon={<Bell size={16} />}
          label="Fatigue Confidence"
          value={`${confidencePct}%`}
          subValue={prediction.flag ? `Flag: ${prediction.flag}` : "No active risk flag"}
          tone={ownerRiskTone}
        />
        <MetricCard
          icon={<Heart size={16} />}
          label="Heart Rate"
          value={`${Number(sensor.hr || 0).toFixed(0)} BPM`}
          subValue={`SpO2: ${Number(sensor.spo2 || 0).toFixed(1)}%`}
          tone={Number(sensor.hr || 0) < 45 || Number(sensor.hr || 0) > 120 ? "warning" : "neutral"}
        />
        <MetricCard
          icon={<Thermometer size={16} />}
          label="Body Temperature"
          value={`${Number(sensor.temperature || 0).toFixed(1)} °C`}
          subValue={Number(sensor.temperature || 0) >= 37.8 ? "Elevated thermal risk" : "Within expected range"}
          tone={Number(sensor.temperature || 0) >= 37.8 ? "warning" : "neutral"}
        />
        <MetricCard
          icon={<Gauge size={16} />}
          label="Motion Magnitude"
          value={motionSummary}
          subValue="From accelerometer AX/AY/AZ"
          tone="neutral"
        />
      </section>

      <section className="owner-panels">
        <div className="owner-panel">
          <h3><Eye size={16} /> Vision Signals</h3>
          <div className="owner-kv"><span>PERCLOS</span><strong>{Number(perclos.perclos || 0).toFixed(1)}%</strong></div>
          <div className="owner-kv"><span>EAR</span><strong>{Number(perclos.ear || 0).toFixed(3)}</strong></div>
          <div className="owner-kv"><span>MAR</span><strong>{Number(perclos.mar || 0).toFixed(3)}</strong></div>
          <div className="owner-kv"><span>Eye Status</span><strong>{perclos.status || "N/A"}</strong></div>
          <div className="owner-kv"><span>Yawn</span><strong>{perclos.yawn_status || "N/A"}</strong></div>
        </div>

        <div className="owner-panel">
          <h3><Activity size={16} /> Head Pose</h3>
          <div className="owner-kv"><span>Position</span><strong>{head.position || "Unknown"}</strong></div>
          <div className="owner-kv"><span>Pitch</span><strong>{Number(head.angle_x || 0).toFixed(1)}°</strong></div>
          <div className="owner-kv"><span>Yaw</span><strong>{Number(head.angle_y || 0).toFixed(1)}°</strong></div>
          <div className="owner-kv"><span>Roll</span><strong>{Number(head.angle_z || 0).toFixed(1)}°</strong></div>
          <div className="owner-kv"><span>Source</span><strong>{head.source || "None"}</strong></div>
        </div>

        <div className="owner-panel">
          <h3><Gauge size={16} /> Raw Sensor Telemetry</h3>
          <div className="owner-kv"><span>AX</span><strong>{Number(sensor.ax || 0).toFixed(2)}</strong></div>
          <div className="owner-kv"><span>AY</span><strong>{Number(sensor.ay || 0).toFixed(2)}</strong></div>
          <div className="owner-kv"><span>AZ</span><strong>{Number(sensor.az || 0).toFixed(2)}</strong></div>
          <div className="owner-kv"><span>GX</span><strong>{Number(sensor.gx || 0).toFixed(2)}</strong></div>
          <div className="owner-kv"><span>GY</span><strong>{Number(sensor.gy || 0).toFixed(2)}</strong></div>
          <div className="owner-kv"><span>GZ</span><strong>{Number(sensor.gz || 0).toFixed(2)}</strong></div>
          <div className="owner-kv"><span>Sensor Timestamp</span><strong>{formatTimestamp(sensor.timestamp)}</strong></div>
        </div>
      </section>

      <section className="owner-panel owner-alert-panel">
        <h3><Bell size={16} /> Alert History</h3>
        {alertHistory.length === 0 ? (
          <div className="owner-empty">No drowsy/fatigue alerts captured yet.</div>
        ) : (
          <div className="owner-alert-list">
            {alertHistory.map((alert) => (
              <div key={alert.id} className={`owner-alert-item alert-${alert.status.toLowerCase()}`}>
                <span>{alert.time}</span>
                <strong>{alert.status}</strong>
                <span>{alert.confidence}%</span>
                <span>{alert.flag}</span>
              </div>
            ))}
          </div>
        )}
      </section>
    </div>
  );
}
