import React, { createContext, useContext, useState, useEffect } from 'react';
import { API_BASE } from '../api';

const FatigueContext = createContext();

export const useFatigueContext = () => {
    return useContext(FatigueContext);
};

export const FatigueProvider = ({ children }) => {
    // ---------------- STATE ----------------
    // Centralized state for ALL components
    const [fullData, setFullData] = useState(null);
    
    // --- DERIVED / HISTORY STATES (Centralized Here) ---
    // Moved from useHeartRate
    const [heartRateHistory, setHeartRateHistory] = useState([]);
    
    // Moved from useSensorData
    const [tempHistory, setTempHistory] = useState([]);

    // --- HISTORY UPDATE EFFECT ---
    // Update histories whenever fullData changes (Polling or WebSocket)
    useEffect(() => {
        if (!fullData) return;

        // 1. Heart Rate History
        const hr = fullData.sensor?.hr ?? 0;
        if (hr !== null) {
                setHeartRateHistory(prev => {
                    // Avoid duplicate timestamps if possible, or just push
                    // Since timeTick was local to the loop, we might lose continuity if we switch sources.
                    // But for now, simple array push is fine.
                    // Ideally use fullData.server_time or a counter.
                    const newHrPoint = { time: Date.now(), bpm: hr }; 
                    const updated = [...prev, newHrPoint];
                    return updated.slice(-20);
                });
        }

        // 2. Temperature History (formatted for Chart)
        const temp = fullData.sensor?.temperature ?? 0;
        const newTempPoint = {
            time: new Date().toLocaleTimeString(),
            temperature: temp
        };
        setTempHistory(prev => {
            const updated = [...prev, newTempPoint];
            return updated.slice(-20);
        });

    }, [fullData]);

    // ---------------- POLLING LOOP (Only when Standard Mode is active) ----------------
    useEffect(() => {
        let isMounted = true;
        let interval;

        const fetchData = async () => {
            try {
                // FETCH ONCE for the entire app!
                const response = await fetch(`${API_BASE}/api/combined_data`, {
                    headers: {
                        "ngrok-skip-browser-warning": "69420",
                        "Content-Type": "application/json"
                    }
                });
                if (!response.ok) throw new Error("Network response was not ok");
                
                const json = await response.json();
                
                if (isMounted) {
                    // Inject System Status
                    const richData = { ...json, status: "Active" };
                    setFullData(richData);
                }
            } catch (error) {
                console.error("[FatigueContext] Fetch Error:", error);
                if (isMounted) {
                     setFullData(prev => prev ? ({ ...prev, status: "Offline" }) : null);
                }
            }
        };

        // Start polling immediately
        fetchData();
        interval = setInterval(fetchData, 500);

        return () => {
             isMounted = false;
             if (interval) clearInterval(interval);
        };
    }, []);

    // ---------------- EXPORT ----------------
    const value = {
        fullData,
        setFullData, // Exposed for WebSocket updates
        heartRateHistory,
        tempHistory
    };

    return (
        <FatigueContext.Provider value={value}>
            {children}
        </FatigueContext.Provider>
    );
};
