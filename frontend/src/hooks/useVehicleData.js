import { useVehicleContext } from '../context/VehicleContext';

/**
 * useVehicleData Hook
 * Provides easy access to vehicle data with convenient selectors
 */
export const useVehicleData = () => {
    const { vehicleData, headPositionHistory, predictionHistory } = useVehicleContext();

    return {
        // Raw data
        vehicleData,
        
        // Vision Data
        perclos: vehicleData?.vision?.perclos,
        headPosition: vehicleData?.vision?.head_position,
        
        // Prediction
        prediction: vehicleData?.prediction,
        status: vehicleData?.prediction?.status,
        confidence: vehicleData?.prediction?.confidence,
        microsleepDetected: vehicleData?.prediction?.microsleep_detected,
        
        // Histories
        headPositionHistory,
        predictionHistory,
        
        // Convenience checks
        isFatigued: vehicleData?.prediction?.status === "Fatigued",
        isDrowsy: vehicleData?.prediction?.status === "Drowsy",
        isAlert: vehicleData?.prediction?.status === "Alert",
        isInitializing: vehicleData?.system_status === "Initializing",
        isOffline: vehicleData?.status === "Offline"
    };
};

export default useVehicleData;
