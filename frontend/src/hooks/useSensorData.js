import { useFatigueContext } from "../context/FatigueContext";

export const useSensorData = () => {
  const { fullData, tempHistory } = useFatigueContext();
  
  const sensorData = fullData?.sensor || {
    temperature: null,
    ax: null, ay: null, az: null,
    status: 'Loading...'
  };

  return { sensorData, dataHistory: tempHistory };
};
