import React, { useState, useEffect } from "react";
import {
  AreaChart,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer
} from "recharts";
import { useTheme } from "../context/ThemeContext";
import "./Css/HRVChart.css";

export default function HRVChart() {
  const [hrvHistory, setHrvHistory] = useState([]);
  const [rawHrBuffer, setRawHrBuffer] = useState([]);
  const { isDarkMode } = useTheme();

  useEffect(() => {
    const fetchHR = async () => {
      try {
        const res = await fetch("http://127.0.0.1:5000/combined_data");
        const data = await res.json();
        
        const currentHr = data.sensor?.hr || 0;

        let newBuffer = [...rawHrBuffer, currentHr];
        if (newBuffer.length > 20) newBuffer.shift(); 
        setRawHrBuffer(newBuffer);

        const validReadings = newBuffer.filter(h => h > 40 && h < 200);
        
        let sdnn = 0;
        if (validReadings.length > 2) {
            const mean = validReadings.reduce((sum, val) => sum + val, 0) / validReadings.length;
            const variance = validReadings.reduce((sum, val) => sum + Math.pow(val - mean, 2), 0) / validReadings.length;
            sdnn = Math.sqrt(variance);
        }

        setHrvHistory(prev => {
           const time = prev.length > 0 ? prev[prev.length-1].time + 1 : 0;
           const newData = [...prev, { time, value: sdnn }];
           if (newData.length > 30) newData.shift();
           return newData;
        });

      } catch (err) {
        console.error("HRV Fetch Error", err);
      }
    };

    const interval = setInterval(fetchHR, 1000); 
    return () => clearInterval(interval);
  }, [rawHrBuffer]);

  return (
    <div className="hrv-container">
      <ResponsiveContainer width="100%" height="100%">
        <AreaChart data={hrvHistory} margin={{ top: 5, right: 0, left: -20, bottom: 0 }}>
          <defs>
            <linearGradient id="colorHrv" x1="0" y1="0" x2="0" y2="1">
              <stop offset="5%" stopColor="#8b5cf6" stopOpacity={0.3}/>
              <stop offset="95%" stopColor="#8b5cf6" stopOpacity={0}/>
            </linearGradient>
          </defs>
          <CartesianGrid strokeDasharray="3 3" vertical={false} stroke={isDarkMode ? '#475569' : '#000'} strokeOpacity={0.05} />
          <XAxis 
             dataKey="time" 
             tick={{fontSize: 10, fill: isDarkMode ? '#94a3b8' : '#cbd5e1'}} 
             tickLine={false}
             axisLine={false}
          />
          <YAxis 
            domain={[0, 20]} 
            hide={true}
          />
          <Tooltip 
             contentStyle={{
                backgroundColor: isDarkMode ? '#1e293b' : '#fff',
                border: `1px solid ${isDarkMode ? '#475569' : '#e2e8f0'}`,
                borderRadius: '6px',
                fontSize: '12px',
                color: isDarkMode ? '#f1f5f9' : '#0f172a'
             }}
             formatter={(val) => [`${val.toFixed(1)}`, 'HRV (SDNN)']}
             labelStyle={{display: 'none'}}
             cursor={{ stroke: '#8b5cf6', strokeWidth: 1, strokeDasharray: '4 4' }}
          />
          <Area
            type="monotone"
            dataKey="value"
            stroke="#8b5cf6"
            strokeWidth={3}
            fillOpacity={1}
            fill="url(#colorHrv)"
            isAnimationActive={false}
          />
        </AreaChart>
      </ResponsiveContainer>

      <div className="hrv-display">
          <div className="hrv-value">
             {hrvHistory.length > 0 ? hrvHistory[hrvHistory.length-1].value.toFixed(1) : "--"}
          </div>
          <div className="hrv-label">ms (SDNN)</div>
      </div>
    </div>
  );
}
