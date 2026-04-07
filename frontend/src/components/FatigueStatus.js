import { 
  Activity,
  Cpu,
  RotateCcw
} from "lucide-react";
import { useFatigueData } from "../hooks/useFatigueData";
import { useTheme } from "../context/ThemeContext";
import { resetCalibration } from "../api";
import "./Css/FatigueStatus.css";

export default function FatigueStatus() {
  const data = useFatigueData();
  const { isDarkMode } = useTheme();
  
  const handleReset = async () => {
    const success = await resetCalibration();
    if (success) {
      alert("Calibration reset! Please look at the camera.");
    }
  };

  // Destructure Data
  const { 
    temperature = 0, 
    perclos = 0, 
    yawn_status = "None", 
    hr = 0,
    ml_fatigue_status = "Unknown",
    ml_confidence = 0,
    ml_flag = null,
    system_status = "Active"
  } = data || {};

  const safePerclos = typeof perclos === 'number' ? perclos : 0;
  const safeHR = typeof hr === 'number' ? hr : 0;
  const safeTemp = typeof temperature === 'number' ? temperature : 0;
  const safeYawn = yawn_status || "None";
  
  const normPerclos = Math.min(safePerclos / 100, 1);
  const normYawn = safeYawn === "Yawning" ? 1.0 : (safeYawn === "Opening" ? 0.5 : 0.0);
  const normHR = safeHR > 0 ? (safeHR > 100 ? (safeHR - 100)/50 : (safeHR < 60 ? (60-safeHR)/30 : 0)) : 0;
  
  const features = [
    { id: "f_perclos", label: "PERCLOS", value: normPerclos, raw: `${safePerclos.toFixed(1)}%` },
    { id: "f_yawn", label: "Yawn_Seq", value: normYawn, raw: safeYawn },
    { id: "f_hr", label: "HR_Var", value: Math.min(Math.max(normHR, 0), 1), raw: `${safeHR} bpm` },
    { id: "f_temp", label: "Temp_Delta", value: safeTemp > 37 ? (safeTemp - 37) : 0, raw: `${safeTemp.toFixed(1)}°C` }
  ];
  
  const sortedFeatures = [
      features.find(f => f.id === "f_perclos"),
      features.find(f => f.id === "f_yawn"),
      ...features.filter(f => f.id !== "f_perclos" && f.id !== "f_yawn").sort((a,b) => b.value - a.value)
  ];

  let predictedClass = "NORMAL";
  let confidence = ml_confidence; 
  let color = "low"; 

  const isNoFace = data.status === "No Face";
  const isInitializing = system_status === "Initializing";

  if (isInitializing) {
     predictedClass = "CALIBRATING";
     color = "info"; // New color type
     confidence = 0.0;
  } else if (ml_fatigue_status !== "Unknown" && ml_fatigue_status !== "Error") {
      if (isNoFace) {
          predictedClass = "SEARCHING";
          color = "medium";
          confidence = 0.0;
      } else if (ml_fatigue_status === "Waiting...") {
          predictedClass = "WAITING";
          color = "low";
          confidence = 0.0;
      } else {
           predictedClass = String(ml_fatigue_status || "Unknown").toUpperCase();
          
          if (predictedClass === "DROWSY") color = "medium";
          if (predictedClass === "FATIGUED") color = "high";
      }
  } else {
      predictedClass = "OFFLINE"; 
      confidence = 0.00; 
      color = "low"; 
  }

  const confidencePct = (confidence * 100).toFixed(1);

  const formatDriverReason = (flag) => {
      if (!flag) return "NOMINAL PHYSIOLOGICAL STATE";
      
      const map = {
          "MICROSLEEP": "CRITICAL MICROSLEEP EVENT",
          "HIGH_PERCLOS": "EXCESSIVE OCULAR CLOSURE",
          "SKIPPED_NO_FACE": "SUBJECT NOT VISIBLE", 
          "SKIPPED_UNSTABLE": "SIGNAL INSTABILITY",
          "THERMAL_STRESS": "THERMAL STRESS INDICATED",
          "HYPOXIA_RISK": "HYPOXIA RISK WARNING",
          "CARDIAC_ANOMALY": "CARDIAC RHYTHM ANOMALY",
          "BIO_OCULAR_PATTERN": "BIO-OCULAR FATIGUE PATTERN"
      };
      
      return map[flag] || flag.replace(/_/g, " ");
  };

  return (
    <div className="fatigue-status-container">
      
      <div className="fatigue-header">
          <div className="fatigue-header-info">
             <div className="fatigue-model-icon">
                <Cpu size={16} color={isDarkMode ? '#818cf8' : '#6366f1'} />
                <span className="fatigue-model-label">RF_V3 INFERENCE</span>
             </div>
          </div>
          <div className="fatigue-header-info">
             <button className="fatigue-reset-button" onClick={handleReset} title="Reset Calibration">
                <RotateCcw size={14} color={isDarkMode ? '#cbd5e1' : '#64748b'} />
             </button>
             <div className="fatigue-status-indicator">
                <span 
                  className="fatigue-status-dot" 
                  style={{
                    background: predictedClass === "OFFLINE" ? '#94a3b8' : (predictedClass === "CALIBRATING" ? '#3b82f6' : '#16a34a'),
                    boxShadow: predictedClass === "OFFLINE" ? 'none' : (predictedClass === "CALIBRATING" ? '0 0 4px #3b82f6' : '0 0 4px #16a34a')
                  }}
                ></span>
             </div>
          </div>
      </div>

      <div className={`fatigue-prediction-box ${color}`}>
          <div className="fatigue-class-header">
             <span className="fatigue-class-label">PREDICTED CLASS</span>
             <span className="fatigue-class-value">
                {predictedClass}
             </span>
          </div>
          
          <div className="fatigue-driver-section">
             <span className="fatigue-driver-label">PRIMARY DRIVER</span>
             <span className="fatigue-driver-reason">
                {formatDriverReason(ml_flag)}
             </span>
          </div>
          
          <div>
            <div className="fatigue-confidence-header">
               <span className="fatigue-confidence-label">MODEL CONFIDENCE</span>
               <span className="fatigue-confidence-value">{isInitializing ? "---" : `${confidencePct}%`}</span>
            </div>
            <div className="fatigue-confidence-bar">
               <div 
                 className={`fatigue-confidence-fill ${color} ${isInitializing ? 'pulse' : ''}`}
                 style={{width: isInitializing ? '100%' : `${confidencePct}%`}}
               ></div>
            </div>
          </div>
      </div>

      <div className="fatigue-features-section">
        <div className="fatigue-features-label">
           <Activity size={12} /> LIVE FEATURE WEIGHTS
        </div>
        
        <div className="risk-list-container">
          <ul className="risk-factor-list">
             {sortedFeatures.map((feat) => (
                <li key={feat.id} className="risk-factor-item">
                   <div className="risk-factor-label">
                       <span className="risk-factor-name">{feat.label}</span>
                       <span className="risk-factor-raw">{feat.raw}</span>
                   </div>
                   <div className="risk-factor-bar-container">
                       <div className="risk-factor-bar">
                          <div 
                            className={`risk-factor-bar-fill ${feat.value > 0.5 ? 'high' : 'low'}`}
                            style={{width: `${Math.min(feat.value * 100, 100)}%`}}
                          ></div>
                       </div>
                       <span className="risk-factor-value">
                          {feat.value.toFixed(2)}
                       </span>
                   </div>
                </li>
             ))}
          </ul>
        </div>
      </div>

    </div>
  );
}
