import React from "react";
import {
  AreaChart,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
} from "recharts";
import { useSensorData } from "../hooks/useSensorData";

export default function BodyTemperatureChart() {
  const { sensorData, dataHistory } = useSensorData();
  const currentTemp = sensorData?.temperature;

  return (
    <div style={{ width: "100%", height: "100%", position: "relative" }}>
       
       <ResponsiveContainer width="100%" height="100%">
          <AreaChart data={dataHistory} margin={{ top: 5, right: 0, left: -20, bottom: 0 }}>
            <defs>
              <linearGradient id="colorTemp" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor="#f97316" stopOpacity={0.3}/>
                <stop offset="95%" stopColor="#f97316" stopOpacity={0}/>
              </linearGradient>
            </defs>
            <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#000" strokeOpacity={0.05} />
            <XAxis 
               dataKey="time" 
               tickFormatting={(t) => t.split(':').slice(0,2).join(':')}
               tick={{fontSize: 10, fill: '#cbd5e1'}} 
               tickLine={false}
               axisLine={false}
            />
            <YAxis 
              domain={['dataMin - 0.5', 'dataMax + 0.5']} 
              hide={true}
            />
            <Tooltip 
              contentStyle={{
                backgroundColor: '#fff',
                border: '1px solid #e2e82f0',
                borderRadius: '8px',
                boxShadow: '0 4px 6px -1px rgba(0,0,0,0.1)'
              }}
              labelStyle={{ display: 'none' }}
              cursor={{ stroke: '#f97316', strokeWidth: 1, strokeDasharray: '4 4' }}
            />
            <Area
              type="monotone"
              dataKey="temperature"
              stroke="#f97316"
              strokeWidth={3}
              fillOpacity={1}
              fill="url(#colorTemp)"
            />
          </AreaChart>
        </ResponsiveContainer>

        <div style={{
            position: 'absolute', top: 10, right: 20, 
            textAlign: 'right', pointerEvents: 'none'
        }}>
            <div style={{fontSize: '1.8rem', fontWeight: 800, color: '#f97316', lineHeight: 1}}>
               {currentTemp ? currentTemp.toFixed(1) : "--"}
            </div>
            <div style={{fontSize: '0.75rem', fontWeight: 600, color: '#94a3b8'}}>°C ({currentTemp > 37.5 ? "High" : "Normal"})</div>
        </div>
    </div>
  );
}
