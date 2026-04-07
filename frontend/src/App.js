import React, { useEffect, useState } from "react";
import "./Dashboard.css";
import { 
  Activity, 
  Thermometer, 
  BrainCircuit, 
  User,
  Heart
} from "lucide-react";

// Context
import { FatigueProvider } from "./context/FatigueContext";
import { VehicleProvider, useVehicleContext } from "./context/VehicleContext";
import { ThemeProvider } from "./context/ThemeContext";
import { ModeProvider } from "./context/ModeContext";

// Components
import BodyTemperatureChart from "./components/BodyTemperatureChart";
import HeartRateChart from "./components/HeartRateChart";
import HRVChart from "./components/HRVChart";
import HeadPositionChart from "./components/HeadPositionChart";
import CameraModule from "./components/CameraModule";
import FatigueStatus from "./components/FatigueStatus";
import DrowsinessIndicators from "./components/DrowsinessIndicators";
import ThemeToggle from "./components/ThemeToggle";
import VehicleDashboard from "./components/VehicleDashboard";
import OwnerDashboard from "./components/OwnerDashboard";
import { useFatigueData } from "./hooks/useFatigueData";

// Wrapper Component to access Context for Theme
const StandardModelWithSwitcher = ({ selectedModel, setSelectedModel }) => {
  return (
    <DashboardContent selectedModel={selectedModel} setSelectedModel={setSelectedModel} />
  );
};

// Vehicle Model Wrapper
const VehicleModelWithSwitcher = ({ selectedModel, setSelectedModel }) => {
  return (
    <VehicleDashboardWithHeader selectedModel={selectedModel} setSelectedModel={setSelectedModel} />
  );
};

const OwnerModelWithSwitcher = ({ selectedModel, setSelectedModel }) => {
  return (
    <OwnerDashboardWithHeader selectedModel={selectedModel} setSelectedModel={setSelectedModel} />
  );
};

// Vehicle Dashboard with Header
const VehicleDashboardWithHeader = ({ selectedModel, setSelectedModel }) => {
  const { vehicleData } = useVehicleContext();
  
  // Determine Theme Class based on vehicle prediction
  const predictedStatus = vehicleData?.prediction?.status || "Unknown";
  const themeClass = 
    predictedStatus === "Fatigued" ? "theme-danger" : 
    (predictedStatus === "Drowsy" ? "theme-warning" : "theme-safe");

  return (
    <div className={`dashboard-container ${themeClass}`}>
      <header className="top-header">
        <div className="brand">
          <div className="brand-logo">
            <BrainCircuit size={20} />
          </div>
          <span className="brand-name">FatigueGuard Pro</span>
        </div>
        
        <div className="header-actions">
          <div className="model-switcher">
            <button
              onClick={() => setSelectedModel("standard")}
              className={`model-btn ${selectedModel === "standard" ? "model-btn-active" : ""}`}
            >
              <span className="model-icon">🧠</span>
              <span className="model-text">Standard Model</span>
            </button>
            <button
              onClick={() => setSelectedModel("vehicle")}
              className={`model-btn ${selectedModel === "vehicle" ? "model-btn-active" : ""}`}
            >
              <span className="model-icon">🚗</span>
              <span className="model-text">Vehicle Model</span>
            </button>
            <button
              onClick={() => setSelectedModel("owner")}
              className={`model-btn ${selectedModel === "owner" ? "model-btn-active" : ""}`}
            >
              <span className="model-icon">👤</span>
              <span className="model-text">Owner App</span>
            </button>
          </div>

          <div className="status-badge">
            <span className="live-dot"></span>
            System Active
          </div>
          
          <ThemeToggle />
          
          <div className="user-profile" style={{width: 32, height: 32, borderRadius: '50%', background: '#e2e8f0', display: 'flex', alignItems: 'center', justifyContent: 'center'}}>
            <User size={16} color="#64748b" />
          </div>
        </div>
      </header>

      <main className="dashboard-content vehicle-content">
        <VehicleDashboard />
      </main>
    </div>
  );
};

const OwnerDashboardWithHeader = ({ selectedModel, setSelectedModel }) => {
  const { ml_fatigue_status } = useFatigueData();

  const themeClass =
    ml_fatigue_status === "Fatigued" ? "theme-danger" :
    (ml_fatigue_status === "Drowsy" ? "theme-warning" : "theme-safe");

  return (
    <div className={`dashboard-container ${themeClass}`}>
      <header className="top-header">
        <div className="brand">
          <div className="brand-logo">
            <BrainCircuit size={20} />
          </div>
          <span className="brand-name">FatigueGuard Pro</span>
        </div>

        <div className="header-actions">
          <div className="model-switcher">
            <button
              onClick={() => setSelectedModel("standard")}
              className={`model-btn ${selectedModel === "standard" ? "model-btn-active" : ""}`}
            >
              <span className="model-icon">🧠</span>
              <span className="model-text">Standard Model</span>
            </button>
            <button
              onClick={() => setSelectedModel("vehicle")}
              className={`model-btn ${selectedModel === "vehicle" ? "model-btn-active" : ""}`}
            >
              <span className="model-icon">🚗</span>
              <span className="model-text">Vehicle Model</span>
            </button>
            <button
              onClick={() => setSelectedModel("owner")}
              className={`model-btn ${selectedModel === "owner" ? "model-btn-active" : ""}`}
            >
              <span className="model-icon">👤</span>
              <span className="model-text">Owner App</span>
            </button>
          </div>

          <div className="status-badge">
            <span className="live-dot"></span>
            Owner View Active
          </div>

          <ThemeToggle />

          <div className="user-profile" style={{width: 32, height: 32, borderRadius: '50%', background: '#e2e8f0', display: 'flex', alignItems: 'center', justifyContent: 'center'}}>
            <User size={16} color="#64748b" />
          </div>
        </div>
      </header>

      <main className="dashboard-content owner-content" style={{ padding: 16 }}>
        <OwnerDashboard />
      </main>
    </div>
  );
};

// Wrapper Component to access Context for Theme
const DashboardContent = ({ selectedModel, setSelectedModel }) => {
  const { ml_fatigue_status } = useFatigueData(); 
  
  // Determine Theme Class
  const themeClass = 
    ml_fatigue_status === "Fatigued" ? "theme-danger" : 
    (ml_fatigue_status === "Drowsy" ? "theme-warning" : "theme-safe");

  return (
    <div className={`dashboard-container ${themeClass}`}>
      {/* Top Header (Fixed Height) */}
      <header className="top-header">
        <div className="brand">
          <div className="brand-logo">
            <BrainCircuit size={20} />
          </div>
          <span className="brand-name">FatigueGuard Pro</span>
        </div>
        
        <div className="header-actions">
          <div className="model-switcher">
            <button
              onClick={() => setSelectedModel("standard")}
              className={`model-btn ${selectedModel === "standard" ? "model-btn-active" : ""}`}
            >
              <span className="model-icon">🧠</span>
              <span className="model-text">Standard Model</span>
            </button>
            <button
              onClick={() => setSelectedModel("vehicle")}
              className={`model-btn ${selectedModel === "vehicle" ? "model-btn-active" : ""}`}
            >
              <span className="model-icon">🚗</span>
              <span className="model-text">Vehicle Model</span>
            </button>
            <button
              onClick={() => setSelectedModel("owner")}
              className={`model-btn ${selectedModel === "owner" ? "model-btn-active" : ""}`}
            >
              <span className="model-icon">👤</span>
              <span className="model-text">Owner App</span>
            </button>
          </div>

           {/* Moved Indicator Here as requested */}
           <div style={{marginRight: '12px'}}>
              <DrowsinessIndicators />
           </div>

          <div className="status-badge">
            <span className="live-dot"></span>
            System Active
          </div>
          
          <ThemeToggle />
          
          <div className="user-profile" style={{width: 32, height: 32, borderRadius: '50%', background: '#e2e8f0', display: 'flex', alignItems: 'center', justifyContent: 'center'}}>
            <User size={16} color="#64748b" />
          </div>
        </div>
      </header>

      {/* Main Content (Fills remaining height) */}
      <main className="dashboard-content">
        
        <div className="grid-container">
          
          {/* Left Column: Data Charts */}
          <section className="charts-section">
            
            {/* Row 1: Vitals (Heart Rate & Temp) */}
            <div className="charts-row-1">
              <div className="card">
                <div className="card-header">
                  <span className="card-title"><Heart size={18} className="card-icon"/> Heart Rate</span>
                </div>
                <div className="chart-body-fill">
                   <HeartRateChart />
                </div>
              </div>

              <div className="card">
                <div className="card-header">
                   <span className="card-title"><Thermometer size={18} className="card-icon"/> Temperature</span>
                </div>
                <div className="chart-body-fill">
                   <BodyTemperatureChart />
                </div>
              </div>
            </div>

            {/* Row 2: Secondary Metrics (HRV & Head) */}
            <div className="charts-row-2">
               <div className="card">
                  <div className="card-header">
                    <span className="card-title"><Activity size={18} className="card-icon"/> HRV Analysis</span>
                  </div>
                  <div className="chart-body-fill">
                    <HRVChart />
                  </div>
               </div>
               
               <div className="card">
                  <div className="card-header">
                    <span className="card-title"><User size={18} className="card-icon"/> Head Posture</span>
                  </div>
                  <div className="chart-body-fill">
                    <HeadPositionChart />
                  </div>
               </div>
            </div>

          </section>

          {/* Right Column: Status & Camera */}
          <aside className="side-panel">
            
            {/* Live Camera Feed (Now Top) */}
            <div className="card camera-card">
              <CameraModule />
            </div>

            {/* Fatigue Status Widget (Now Bottom) */}
            <div className="card fatigue-card">
               <div className="card-header">
                  <span className="card-title"><BrainCircuit size={18} className="card-icon"/> Fatigue Status</span>
               </div>
               <FatigueStatus />
            </div>

          </aside>

        </div>
      </main>
    </div>
  );
};

function App() {
  const [mounted, setMounted] = useState(false);
  const [selectedModel, setSelectedModel] = useState("standard"); // "standard" or "vehicle" or "owner"

  useEffect(() => {
    setMounted(true);
  }, []);

  if (!mounted) return null;

  return (
    <ThemeProvider>
      <ModeProvider selectedMode={selectedModel === "owner" ? "standard" : selectedModel}>
        {selectedModel === "vehicle" ? (
          <VehicleProvider autoResetCalibration>
            <VehicleModelWithSwitcher
              selectedModel={selectedModel}
              setSelectedModel={setSelectedModel}
            />
          </VehicleProvider>
        ) : selectedModel === "owner" ? (
          <VehicleProvider>
            <OwnerModelWithSwitcher
              selectedModel={selectedModel}
              setSelectedModel={setSelectedModel}
            />
          </VehicleProvider>
        ) : (
          <FatigueProvider>
            <StandardModelWithSwitcher
              selectedModel={selectedModel}
              setSelectedModel={setSelectedModel}
            />
          </FatigueProvider>
        )}
      </ModeProvider>
    </ThemeProvider>
  );
};

export default App;
