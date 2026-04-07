import { useState, useEffect } from 'react';
import { getTemperatureHistory } from '../api';

export const useTemperature = () => {
  const [data, setData] = useState([]);
  const [latestTemp, setLatestTemp] = useState(null);

  useEffect(() => {
    const fetchTemperature = async () => {
      const history = await getTemperatureHistory();
      if (Array.isArray(history) && history.length > 0) {
        const formatted = history.map((item) => ({
          time: new Date(item.timestamp * 1000)
            .toLocaleTimeString()
            .slice(0, 8),
          value: item.temperature,
        }));
        setData(formatted);
        setLatestTemp(formatted[formatted.length - 1]?.value || null);
      }
    };

    fetchTemperature();
    const interval = setInterval(fetchTemperature, 5000);
    return () => clearInterval(interval);
  }, []);

  return { data, latestTemp };
};
