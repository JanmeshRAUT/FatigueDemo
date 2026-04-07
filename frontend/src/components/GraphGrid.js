import React from 'react';
import BodyTemperatureChart from './BodyTemperatureChart';
import AccelerometerChart from './AccelerometerChart';
import './Css/GraphGrid.css';

const GraphGrid = () => {
  return (
    <div className="graph-grid">
      <div className="graph-item">
        <BodyTemperatureChart />
      </div>
      <div className="graph-item">
        <AccelerometerChart />
      </div>
    </div>
  );
};

export default GraphGrid;
