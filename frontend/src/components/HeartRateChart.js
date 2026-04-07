import React from "react";
import {
  AreaChart,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  ReferenceLine
} from "recharts";
import { useHeartRate } from "../hooks/useHeartRate";
import { useTheme } from "../context/ThemeContext";
import "./Css/HeartRateChart.css";

const CustomTooltip = ({ active, payload, label, isDarkMode }) => {
  if (active && payload && payload.length) {
    return (
      <div className="heart-rate-tooltip">
        <p>Time: {label}s</p>
        <p>
          {payload[0].value.toFixed(0)} BPM
        </p>
      </div>
    );
  }
  return null;
};

export default function HeartRateChart() {
  const heartRates = useHeartRate();
  const { isDarkMode } = useTheme();

  return (
    <div className="heart-rate-container">
      <ResponsiveContainer width="100%" height="100%">
        <AreaChart data={heartRates} margin={{ top: 10, right: 0, left: -20, bottom: 0 }}>
          <defs>
            <linearGradient id="colorBpm" x1="0" y1="0" x2="0" y2="1">
              <stop offset="5%" stopColor="#ef4444" stopOpacity={0.3}/>
              <stop offset="95%" stopColor="#ef4444" stopOpacity={0}/>
            </linearGradient>
          </defs>
          <CartesianGrid strokeDasharray="3 3" vertical={false} stroke={isDarkMode ? '#475569' : '#000'} strokeOpacity={0.05} />
          <XAxis 
             dataKey="time" 
             tick={{fontSize: 10, fill: isDarkMode ? '#94a3b8' : '#cbd5e1'}} 
             tickLine={false}
             axisLine={false}
             interval="preserveStartEnd"
          />
          <YAxis 
            domain={[40, 140]} 
            hide={true}
          />
          <Tooltip content={<CustomTooltip isDarkMode={isDarkMode} />} cursor={{ stroke: '#ef4444', strokeWidth: 1, strokeDasharray: '4 4' }}/>
          
          <ReferenceLine y={100} stroke={isDarkMode ? '#7f1d1d' : '#fecaca'} strokeDasharray="3 3" />
          
          <Area
            type="monotone"
            dataKey="bpm"
            stroke="#ef4444"
            strokeWidth={3}
            fillOpacity={1}
            fill="url(#colorBpm)"
            isAnimationActive={false}
          />
        </AreaChart>
      </ResponsiveContainer>

      <div className="heart-rate-display">
          <div className="heart-rate-value">
             {heartRates.length > 0 ? (heartRates[heartRates.length-1].bpm > 0 ? heartRates[heartRates.length-1].bpm : "--") : "--"}
          </div>
          <div className="heart-rate-label">
              BPM {heartRates.length > 0 && heartRates[heartRates.length-1].bpm > 0 ? 
                  ((heartRates[heartRates.length-1].bpm < 50 || heartRates[heartRates.length-1].bpm > 120) ? " (!)" : "") 
                  : "(OFF)"}
          </div>
      </div>
    </div>
  );
}
