import { useFatigueContext } from "../context/FatigueContext";

export const useHeartRate = () => {
  const { heartRateHistory } = useFatigueContext();
  return heartRateHistory;
};
