import React, { createContext, useContext, useState, useEffect } from 'react';
import { API_BASE } from '../api';

const VehicleContext = createContext();

export const useVehicleContext = () => {
    const context = useContext(VehicleContext);
    if (!context) {
        console.warn("[VehicleContext] Context not found - using fallback null");
    }
    return context;
};

export const VehicleProvider = ({ children, autoResetCalibration = false }) => {
    // Centralized state for Vehicle Mode
    const [vehicleData, setVehicleData] = useState(null);
    const [headPositionHistory, setHeadPositionHistory] = useState([]);
    const [predictionHistory, setPredictionHistory] = useState([]);
    const [connectionStatus, setConnectionStatus] = useState("connecting");

    const normalizeVehicleData = (json) => {
        const perclos = json?.vision?.perclos || json?.perclos || { perclos: 0, ear: 0, mar: 0, status: "Open" };
        const head = json?.vision?.head_position || json?.head_position || { angle_x: 0, angle_y: 0, angle_z: 0, position: "Center" };
        const prediction = json?.prediction || { status: "Unknown", confidence: 0, microsleep_detected: false };
        const sensor = json?.sensor || { hr: 0, temperature: 0 };

        return {
            ...json,
            status: json?.system_status || "Active",
            model_type: "Vehicle",
            perclos,
            head_position: head,
            prediction,
            sensor,
        };
    };

    const resetCalibration = async () => {
        try {
            const response = await fetch(`${API_BASE}/api/vehicle/reset_calibration`, {
                method: 'POST',
                headers: {
                    "ngrok-skip-browser-warning": "69420",
                    "Content-Type": "application/json"
                }
            });
            return response.ok;
        } catch (error) {
            console.error("[VehicleContext] Reset Error:", error);
            return false;
        }
    };

    useEffect(() => {
        if (!autoResetCalibration) return;

        resetCalibration();
    }, [autoResetCalibration]);

    // Update history when data arrives
    useEffect(() => {
        if (!vehicleData) return;

        // Head Position History from actual head_position data
        const headPos = vehicleData.head_position;
        if (headPos) {
            const newHeadPoint = {
                time: Date.now(),
                angle_x: headPos.angle_x || 0,
                angle_y: headPos.angle_y || 0,
                angle_z: headPos.angle_z || 0,
                position: headPos.position || "Center",
                source: headPos.source || "Vision",
                calibrated: headPos.calibrated || true
            };
            setHeadPositionHistory(prev => {
                const updated = [...prev, newHeadPoint];
                return updated.slice(-30); // Keep last 30 points
            });
        }

        // Prediction History
        if (vehicleData.prediction) {
            const newPredPoint = {
                time: Date.now(),
                status: vehicleData.prediction.status || "Unknown",
                confidence: vehicleData.prediction.confidence || 0,
                microsleep: vehicleData.prediction.microsleep_detected || false
            };
            setPredictionHistory(prev => {
                const updated = [...prev, newPredPoint];
                return updated.slice(-30); // Keep last 30 predictions
            });
        }

    }, [vehicleData]);

    // Polling loop - fetch combined data (Only when Vehicle Mode is active)
    useEffect(() => {
        let isMounted = true;
        let retryCount = 0;
        const maxRetries = 3;

        const fetchData = async () => {
            try {
                const response = await fetch(`${API_BASE}/api/vehicle/combined_data`, {
                    method: 'GET',
                    headers: {
                        "ngrok-skip-browser-warning": "69420",
                        "Content-Type": "application/json"
                    }
                });

                if (!response.ok) {
                    throw new Error(`HTTP ${response.status}: ${response.statusText}`);
                }

                const json = await response.json();

                if (isMounted) {
                    if (json && (json.vision || json.perclos || json.head_position || json.prediction)) {
                        setVehicleData(normalizeVehicleData(json));
                        setConnectionStatus("connected");
                        retryCount = 0;
                    } else {
                        console.warn("[VehicleContext] Invalid data structure:", json);
                        setConnectionStatus("invalid_data");
                    }
                }
            } catch (error) {
                console.error("[VehicleContext] Fetch Error:", error.message);
                if (isMounted) {
                    retryCount++;
                    if (retryCount > maxRetries) {
                        setConnectionStatus("offline");
                        setVehicleData(null);
                    } else {
                        setConnectionStatus("retrying");
                        setVehicleData(prev => prev ? ({...prev, status: "Retrying..."}) : null);
                    }
                }
            }
        };

        // Initial fetch immediately
        fetchData();

        const interval = setInterval(fetchData, 500);

        return () => {
            isMounted = false;
            clearInterval(interval);
        };
    }, []);

    const value = {
        // Data
        vehicleData,
        
        // History
        headPositionHistory,
        predictionHistory,
        
        // Status
        connectionStatus,
        
        // Methods
        resetCalibration
    };

    return (
        <VehicleContext.Provider value={value}>
            {children}
        </VehicleContext.Provider>
    );
};
