import { useFatigueContext } from "../context/FatigueContext";

const normalizeMlStatus = (value) => {
    if (value === null || value === undefined) return "Unknown";

    if (typeof value === "number") {
        if (value === 0) return "Alert";
        if (value === 1) return "Drowsy";
        if (value === 2) return "Fatigued";
        return "Unknown";
    }

    const s = String(value).trim();
    const lowered = s.toLowerCase();

    if (lowered === "0") return "Alert";
    if (lowered === "1") return "Drowsy";
    if (lowered === "2") return "Fatigued";
    if (lowered === "alert") return "Alert";
    if (lowered === "drowsy") return "Drowsy";
    if (lowered === "fatigued") return "Fatigued";

    return s || "Unknown";
};

export const useFatigueData = () => {
    const context = useFatigueContext();
    const { fullData } = context || {}; // Handle undefined context safely

    if (!fullData) {
        return {
            temperature: 0,
            hr: 0,
            spo2: 0,
            perclos: 0,
            ear: 0,
            mar: 0,
            status: "Loading...",
            yawn_status: "N/A",
            ml_fatigue_status: "Unknown",
            ml_confidence: 0,
            timestamp: null,
        };
    }

    const sensor = fullData.sensor || {};
    const perclos = fullData.perclos || {};
    const mlData = fullData.prediction || { status: "Unknown", confidence: 0 };

    return {
        temperature: sensor.temperature,
        hr: sensor.hr,
        spo2: sensor.spo2,
        perclos: perclos.perclos,
        ear: perclos.ear,
        mar: perclos.mar,
        status: perclos.status, // "Open" / "Closed"
        yawn_status: perclos.yawn_status,
        // Extract Detailed ML Data
        ml_fatigue_status: normalizeMlStatus(mlData.status),
        ml_confidence: mlData.confidence || 0,
        ml_flag: mlData.flag,
        system_status: fullData.system_status || fullData.status || "Active", // "Active" / "Initializing"
        timestamp: fullData.server_time || Date.now(),
    };
};
