
import axios from 'axios';

export const API_BASE = process.env.REACT_APP_API_URL || "http://localhost:5000";

const api = axios.create({
    baseURL: API_BASE,
    headers: {
        "Content-Type": "application/json",
        "ngrok-skip-browser-warning": "true"
    }
});

// --- API Service Methods ---

export const resetCalibration = async () => {
    try {
        const response = await api.post('/api/reset_calibration');
        return response.data.status === "success";
    } catch (error) {
        console.error("Error resetting calibration:", error);
        return false;
    }
};

export const getCombinedData = async () => {
    try {
        const response = await api.get('/api/combined_data');
        return response.data;
    } catch (error) {
        console.error("Error fetching combined data:", error);
        return null;
    }
};

// --- VEHICLE MODEL API ENDPOINTS ---

export const getVehicleCombinedData = async () => {
    try {
        const response = await api.get('/api/vehicle/combined_data');
        return response.data;
    } catch (error) {
        console.error("Error fetching vehicle data:", error);
        return null;
    }
};

export const resetVehicleCalibration = async () => {
    try {
        const response = await api.post('/api/vehicle/reset_calibration');
        return response.data.status === "OK";
    } catch (error) {
        console.error("Error resetting vehicle calibration:", error);
        return false;
    }
};

export default api;
