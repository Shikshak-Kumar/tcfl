import React, { useState, useEffect, useRef } from 'react';
import axios from 'axios';
import Sidebar from './components/Sidebar';
import LiveCharts from './components/LiveCharts';
import { Car, AlertTriangle, Clock, Route } from 'lucide-react';

const WEBSOCKET_URL = "ws://localhost:8000/api/simulate";
const API_URL = "http://localhost:8000/api/config";

const InfoCard = ({ title, value, icon: Icon, colorClass }) => (
  <div className="glass-panel p-5 flex items-center gap-4">
    <div className={`p-3 rounded-xl bg-dark-900 border ${colorClass}`}>
      <Icon className={`w-6 h-6 ${colorClass.replace('border-', 'text-')}`} />
    </div>
    <div>
      <p className="text-sm font-medium text-slate-400 mb-1">{title}</p>
      <h3 className="text-2xl font-bold text-white tracking-tight">{value}</h3>
    </div>
  </div>
);

function App() {
  const [sysConfig, setSysConfig] = useState({ cities: [], algorithms: [], poi_categories: [] });
  const [simConfig, setSimConfig] = useState({
    city: 'Delhi',
    algorithm: 'Demo (No RL)',
    target_pois: [],
    use_tomtom: true
  });

  const [isRunning, setIsRunning] = useState(false);
  const [simData, setSimData] = useState([]);
  const [latestMetrics, setLatestMetrics] = useState({
    total_queue: 0,
    avg_wait: 0,
    total_vehicles: 0,
    accidents: 0,
    congestion: 1.0
  });

  const wsRef = useRef(null);

  // Fetch initial config from backend
  useEffect(() => {
    axios.get(API_URL)
      .then(res => {
        setSysConfig(res.data);
        if (res.data.cities.length > 0) setSimConfig(p => ({ ...p, city: res.data.cities[0] }));
        if (res.data.algorithms.length > 0) setSimConfig(p => ({ ...p, algorithm: res.data.algorithms[0] }));
      })
      .catch(err => console.error("Failed to load config from backend", err));
  }, []);

  const toggleSimulation = () => {
    if (isRunning) {
      if (wsRef.current) wsRef.current.close();
      setIsRunning(false);
      return;
    }

    // Start Simulation
    setSimData([]);
    setIsRunning(true);

    wsRef.current = new WebSocket(WEBSOCKET_URL);

    wsRef.current.onopen = () => {
      console.log("Connected to simulation server");
      wsRef.current.send(JSON.stringify(simConfig));
    };

    wsRef.current.onmessage = (event) => {
      const data = JSON.parse(event.data);
      if (data.status === "error" || data.status === "complete") {
        setIsRunning(false);
        wsRef.current.close();
        return;
      }

      setLatestMetrics(data);
      setSimData(prev => {
        const newData = [...prev, data];
        // Keep last 50 data points for active chart window
        if (newData.length > 50) return newData.slice(newData.length - 50);
        return newData;
      });
    };

    wsRef.current.onerror = (error) => {
      console.error("WebSocket Error: ", error);
      setIsRunning(false);
    };

    wsRef.current.onclose = () => {
      setIsRunning(false);
    };
  };

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (wsRef.current) wsRef.current.close();
    };
  }, []);

  return (
    <div className="flex h-screen w-full bg-dark-900 overflow-hidden relative">
      {/* Background decoration */}
      <div className="absolute top-0 left-0 w-full h-full overflow-hidden pointer-events-none z-0">
        <div className="absolute top-[-10%] right-[-5%] w-[40%] h-[40%] rounded-full bg-brand-500/5 blur-[120px]"></div>
        <div className="absolute bottom-[-10%] left-[-5%] w-[40%] h-[40%] rounded-full bg-rose-500/5 blur-[120px]"></div>
      </div>

      {/* Sidebar */}
      <div className="z-10 h-full">
        <Sidebar
          config={sysConfig}
          simConfig={simConfig}
          setSimConfig={setSimConfig}
          isRunning={isRunning}
          onToggleSimulation={toggleSimulation}
        />
      </div>

      {/* Main Content */}
      <div className="flex-1 flex flex-col p-6 z-10 gap-6 h-full overflow-hidden">

        {/* Top Stats Row */}
        <div className="grid grid-cols-4 gap-6 shrink-0">
          <InfoCard
            title="Total Vehicles"
            value={latestMetrics.total_vehicles}
            icon={Car}
            colorClass="border-blue-500/30 text-blue-400"
          />
          <InfoCard
            title="Avg Wait Time"
            value={`${latestMetrics.avg_wait}s`}
            icon={Clock}
            colorClass="border-orange-500/30 text-orange-400"
          />
          <InfoCard
            title="Active Queue"
            value={latestMetrics.total_queue}
            icon={Route}
            colorClass="border-rose-500/30 text-rose-400"
          />
          <InfoCard
            title={simConfig.use_tomtom ? "TomTom Congestion" : "Base Congestion"}
            value={`${latestMetrics.congestion || 1.0}x`}
            icon={AlertTriangle}
            colorClass="border-emerald-500/30 text-emerald-400"
          />
        </div>

        {/* Charts Area */}
        <div className="flex-1 min-h-0">
          {simData.length > 0 ? (
            <LiveCharts data={simData} />
          ) : (
            <div className="h-full glass-panel flex flex-col items-center justify-center text-slate-500">
              <Route className="w-16 h-16 mb-4 opacity-50" />
              <p className="text-lg font-medium">No active simulation.</p>
              <p className="text-sm">Configure settings and click Start to stream data.</p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

export default App;
