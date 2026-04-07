import React, { useRef } from "react";
import { Canvas, useFrame } from "@react-three/fiber";
import { RoundedBox, Environment, ContactShadows } from "@react-three/drei";
import { useHeadPosition } from "../hooks/useHeadPosition";
import { useSmoothHeadPosition } from "../hooks/useSmoothHeadPosition";
import { useFatigueData } from "../hooks/useFatigueData";
import { useVehicleContext } from "../context/VehicleContext";
import "./Css/HeadPositionChart.css";

function HeadModel({ angles, fatigueStatus, source }) {
  const group = useRef();
  
  useFrame(() => {
    if (group.current) {
        // Use x, y, z from smooth animation (work for both modes)
        const angleX = angles?.x ?? 0;
        const angleY = angles?.y ?? 0;
        const angleZ = angles?.z ?? 0;
        
        // Reverse Up/Down (Pitch) for Sensor as requested
        const finalPitch = source === "Sensor" ? -angleX : angleX;

        const radX = (finalPitch * Math.PI) / 180;
        const radY = (-angleY * Math.PI) / 180;
        const radZ = (-angleZ * Math.PI) / 180;
        group.current.rotation.set(radX, radY, radZ, 'YXZ');
    }
  });

  const eyeColor = fatigueStatus === "Fatigued" ? "#ef4444" : (fatigueStatus === "Drowsy" ? "#f59e0b" : "#10b981");
  const eyeScale = fatigueStatus === "Fatigued" ? 0.2 : (fatigueStatus === "Drowsy" ? 0.5 : 1);

  return (
    <group ref={group}>
      <RoundedBox args={[1.4, 1.8, 1.2]} radius={0.3} smoothness={4}>
        <meshStandardMaterial color="#e2e8f0" roughness={0.3} metalness={0.1} />
      </RoundedBox>
      <RoundedBox args={[1.1, 0.8, 0.1]} radius={0.1} smoothness={2} position={[0, 0, 0.61]}>
        <meshStandardMaterial color="#1e293b" roughness={0.2} metalness={0.8} />
      </RoundedBox>

      <group position={[0, 0.1, 0.68]}>
         <mesh position={[-0.25, 0, 0]} scale={[1, eyeScale, 1]}>
            <capsuleGeometry args={[0.08, 0.2, 4, 8]} />
            <meshStandardMaterial color={eyeColor} emissive={eyeColor} emissiveIntensity={2} />
         </mesh>
         
         <mesh position={[0.25, 0, 0]} scale={[1, eyeScale, 1]}>
            <capsuleGeometry args={[0.08, 0.2, 4, 8]} />
            <meshStandardMaterial color={eyeColor} emissive={eyeColor} emissiveIntensity={2} />
         </mesh>
      </group>

      <mesh position={[0, -0.4, 0.68]}>
        <boxGeometry args={[0.3, 0.05, 0.02]} />
        <meshStandardMaterial color="#475569" />
      </mesh>

      <RoundedBox args={[0.2, 0.6, 0.4]} radius={0.05} smoothness={2} position={[-0.75, 0, 0]}>
         <meshStandardMaterial color="#cbd5e1" />
      </RoundedBox>
      <RoundedBox args={[0.2, 0.6, 0.4]} radius={0.05} smoothness={2} position={[0.75, 0, 0]}>
         <meshStandardMaterial color="#cbd5e1" />
      </RoundedBox>
      
      <cylinderGeometry args={[0.4, 0.5, 0.8, 32]} />
    </group>
  );
}

export default function HeadPositionChart({ data = null }) {
  const contextData = useHeadPosition();
  const fatigueData = useFatigueData();
    const vehicleContext = useVehicleContext();

    const getAngle = (obj, key) => {
        if (!obj) return undefined;
        if (typeof obj[key] === "number") return obj[key];
        const altKey = `angle_${key}`;
        if (typeof obj[altKey] === "number") return obj[altKey];
        return undefined;
    };
  
  // Vehicle Mode: use last data point from history array for smooth animation
  const vehicleModeTargetAngles = data && data.length > 0 ? data[data.length - 1] : null;
  
  // Apply smooth animation to Vehicle Mode data
  const vehicleSmoothed = useSmoothHeadPosition(vehicleModeTargetAngles);
  
  // Use Vehicle smoothed data if in Vehicle Mode, otherwise use standard mode
  const angleData = vehicleModeTargetAngles ? vehicleSmoothed : contextData;
    
  const position = angleData?.position || "Center";
    const angle_x = getAngle(angleData, "x") ?? getAngle(contextData, "x") ?? 0;
    const angle_y = getAngle(angleData, "y") ?? getAngle(contextData, "y") ?? 0;
    const angle_z = getAngle(angleData, "z") ?? getAngle(contextData, "z") ?? 0;
  const source = angleData?.source || "None";
  const calibrated = angleData?.calibrated ?? true;
    const isVehicleChart = Boolean(vehicleModeTargetAngles);
  
  const { ml_fatigue_status } = fatigueData;
  
    const isVisionSource = source === "Vision (Fallback)" || source === "Vision";
    const showCalibration = isVisionSource && calibrated === false;
  const isInitializing = source === "None" || source === "Unknown";
    const noFaceFound = ml_fatigue_status === "Unknown" && isVisionSource && position === "Unknown" && calibrated === true;

  // Track if sound has played for this calibration cycle
  const hasPlayedCalibrationSound = React.useRef(false);

  React.useEffect(() => {
    if (isVisionSource && calibrated === true) {
        if (!hasPlayedCalibrationSound.current) {
            try {
                const audio = new Audio("/sounds/correct-356013.mp3");
                audio.volume = 0.5;
                audio.play().catch(e => console.error("Audio playback error:", e));
                hasPlayedCalibrationSound.current = true; // Mark as played
            } catch (e) {
                console.error("Audio initialization failed", e);
            }
        }
    } else if (!calibrated) {
        // Reset whenever calibration is lost or reset
        hasPlayedCalibrationSound.current = false;
    }
    }, [calibrated, isVisionSource]);
  
  const isSafe = Math.abs(angle_x) < 20 && Math.abs(angle_y) < 30 && Math.abs(angle_z) < 20;
  const statusColor = isSafe ? "#10b981" : "#ef4444";
  const statusText = isSafe ? "SAFE" : "DISTRACTED";

  return (
    <div className="head-position-container">
        
        <div className="head-position-header">
            <div className="head-position-title">{position.toUpperCase()}</div>
            <div className="head-position-label">HEAD POSE</div>
        </div>

        <div className="head-position-controls">
            {/* RESET BUTTON for Cloud Calibration */}
            <button 
                className="head-reset-button"
                onClick={async (e) => {
                    e.stopPropagation();
                    if (isVehicleChart && vehicleContext?.resetCalibration) {
                        await vehicleContext.resetCalibration();
                        return;
                    }

                    const { resetCalibration } = await import("../api");
                    await resetCalibration();
                }}
                title="Reset Calibration"
            >
                ↻
            </button> 

            <div className="head-source-badge">
                {isVisionSource ? "VISION" : (source === "Sensor" ? "SENSOR" : "NONE")}
            </div>
        </div>

        {isInitializing && (
            <div className="head-initializing-overlay">
                <div className="head-spinner"></div>
                <div className="head-init-text">
                    <div className="head-init-title">INITIALIZING</div>
                    <div className="head-init-subtitle">WAITING FOR SENSORS...</div>
                </div>
            </div>
        )}

        {showCalibration && !isInitializing && (
            <div className="head-calibration-overlay">
                <div className="head-calibration-spinner"></div>
                <div className="head-calibration-text">CALIBRATING</div>
                <div className="head-calibration-subtitle">Look straight at the camera...</div>
            </div>
        )}

        {noFaceFound && !isInitializing && !showCalibration && (
            <div className="head-no-face-overlay">
                <div className="head-no-face-text">NO FACE DETECTED</div>
                <div className="head-no-face-subtitle">ENSURE FACE IS VISIBLE</div>
            </div>
        )}

        <Canvas camera={{ position: [0, 0, 4], fov: 50 }}>
            <ambientLight intensity={0.5} />
            <spotLight position={[10, 10, 10]} angle={0.15} penumbra={1} intensity={1} castShadow />
            <pointLight position={[-10, -10, -10]} intensity={0.5} />
            
            <HeadModel 
                angles={{ x: angle_x, y: angle_y, z: angle_z }} 
                fatigueStatus={ml_fatigue_status} 
                source={source}
            />
            
            <ContactShadows position={[0, -1.4, 0]} opacity={0.4} scale={10} blur={2.5} far={4} />
            <Environment preset="city" />
        </Canvas>


        <div style={{ position: "absolute", bottom: 12, left: 12, right: 12, display: "flex", justifyContent: "space-between", alignItems: "flex-end", pointerEvents: "none" }}>
             <div style={{ display: "flex", flexDirection: "column", gap: 2, fontSize: "0.65rem", fontFamily: "monospace", color: "#64748b" }}>
                 <div>P: <span style={{color: '#334155', fontWeight: 700}}>{angle_x.toFixed(0)}°</span></div>
                 <div>Y: <span style={{color: '#334155', fontWeight: 700}}>{angle_y.toFixed(0)}°</span></div>
                 <div>R: <span style={{color: '#334155', fontWeight: 700}}>{angle_z.toFixed(0)}°</span></div>
             </div>
             <div style={{ padding: "4px 8px", borderRadius: "6px", background: isSafe ? "#d1fae5" : "#fee2e2", color: statusColor, fontSize: "0.7rem", fontWeight: 800, border: `1px solid ${isSafe ? '#a7f3d0' : '#fecaca'}` }}>
                 {statusText}
             </div>
        </div>
    </div>
  );
}
