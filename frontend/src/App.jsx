import { useState, useEffect, useRef } from 'react';
import Sidebar from './components/Sidebar';
import LiveCharts from './components/LiveCharts';
import LocationInput from './components/LocationInput';
import { Activity, Timer, Car, AlertTriangle } from 'lucide-react';
import './App.css';

const API_BASE = 'http://localhost:8000';

function App() {
  const [systemConfig, setSystemConfig] = useState({ cities: {}, algorithms: [], poi_categories: [] });
  const [simConfig, setSimConfig] = useState({
    city: 'Delhi',
    algorithm: 'Demo (No RL)',
    use_tomtom: true,
    intersections: [],
  });
  const [isRunning, setIsRunning] = useState(false);
  const [simData, setSimData] = useState([]);
  const [latestMetrics, setLatestMetrics] = useState(null);
  const [poiResults, setPoiResults] = useState(null);
  const [statusMessage, setStatusMessage] = useState('');
  const wsRef = useRef(null);

  // Fetch config on mount
  useEffect(() => {
    fetch(`${API_BASE}/api/config`)
      .then(r => r.json())
      .then(data => {
        setSystemConfig(data);
        // Set default city
        const cityNames = Object.keys(data.cities || {});
        if (cityNames.length > 0) {
          setSimConfig(prev => ({ ...prev, city: cityNames[0] }));
        }
      })
      .catch(err => console.error('Failed to fetch config:', err));
  }, []);

  // Get city center for map
  const getCityCenter = () => {
    const cityData = systemConfig.cities?.[simConfig.city];
    if (cityData) return [cityData.lat, cityData.lon];
    return [28.6139, 77.2090]; // Default Delhi
  };

  // Handle intersection changes and clear POI results
  const handleIntersectionsChange = (newIntersections) => {
    setSimConfig(prev => ({ ...prev, intersections: newIntersections }));
    setPoiResults(null); // Clear old POI results when pins change
  };

  // POI Detection + Start Simulation
  const toggleSimulation = async () => {
    if (isRunning) {
      // Stop
      if (wsRef.current) wsRef.current.close();
      setIsRunning(false);
      setStatusMessage('Simulation stopped.');
      return;
    }

    if (!simConfig.intersections || simConfig.intersections.length === 0) {
      setStatusMessage('Place at least 1 pin on the map.');
      return;
    }

    setIsRunning(true);
    setSimData([]);
    setLatestMetrics(null);
    setStatusMessage('Detecting POIs...');

    try {
      // Step 1: Detect POIs for each intersection
      const poiResponse = await fetch(`${API_BASE}/api/detect-pois`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          intersections: simConfig.intersections,
          radius_km: 1.0,
        }),
      });
      const poiData = await poiResponse.json();
      setPoiResults(poiData.intersections);
      
      // Step 2: Merge POI tier data into intersections for WebSocket config
      const enrichedIntersections = simConfig.intersections.map((ix, i) => ({
        ...ix,
        ...(poiData.intersections[i] || {}),
      }));

      setStatusMessage('Starting simulation...');

      // Step 3: Open WebSocket
      const ws = new WebSocket(`ws://localhost:8000/api/simulate`);
      wsRef.current = ws;

      ws.onopen = () => {
        const wsConfig = {
          ...simConfig,
          intersections: enrichedIntersections,
        };
        ws.send(JSON.stringify(wsConfig));
        setStatusMessage('Simulation running...');
      };

      ws.onmessage = (event) => {
        const data = JSON.parse(event.data);
        
        if (data.status === 'complete') {
          setIsRunning(false);
          setStatusMessage('Simulation complete.');
          return;
        }
        if (data.status === 'error') {
          setIsRunning(false);
          setStatusMessage(`Error: ${data.message}`);
          return;
        }

        setLatestMetrics(data);
        setSimData(prev => {
          const updated = [...prev, data];
          return updated.slice(-50); // Keep last 50 points
        });
      };

      ws.onclose = () => {
        setIsRunning(false);
      };

      ws.onerror = (err) => {
        console.error('WebSocket error:', err);
        setIsRunning(false);
        setStatusMessage('Connection error.');
      };

    } catch (err) {
      console.error('Error starting simulation:', err);
      setIsRunning(false);
      setStatusMessage('Failed to start simulation.');
    }
  };

  // Compute global metrics from latest data
  const globalMetrics = latestMetrics?.global || {};
  const intersectionData = latestMetrics?.intersections || {};
  const numIntersections = Object.keys(intersectionData).length;

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-950 via-slate-900 to-indigo-950 text-white">
      <div className="flex flex-col lg:flex-row h-screen">
        
        {/* Left Panel: Controls + Map */}
        <div className="lg:w-[380px] flex-shrink-0 flex flex-col gap-4 p-4 overflow-y-auto custom-scrollbar">
          <Sidebar
            simConfig={simConfig}
            onConfigChange={setSimConfig}
            isRunning={isRunning}
            onToggleSimulation={toggleSimulation}
            cities={systemConfig.cities || {}}
            algorithms={systemConfig.algorithms || []}
            poiResults={poiResults}
          />
          
          {/* Location Input */}
          <div className="glass-panel p-4">
            <h3 className="text-sm font-semibold text-slate-300 mb-2">📍 Select Intersections</h3>
            <LocationInput
              intersections={simConfig.intersections}
              onIntersectionsChange={handleIntersectionsChange}
              city={simConfig.city}
              disabled={isRunning}
            />
          </div>
        </div>
        
        {/* Right Panel: Metrics + Charts */}
        <div className="flex-1 p-4 overflow-y-auto custom-scrollbar">
          {/* Status Bar */}
          {statusMessage && (
            <div className="glass-panel px-4 py-2 mb-4 text-sm text-slate-300 flex items-center gap-2">
              <Activity className="w-4 h-4 text-indigo-400 animate-pulse" />
              {statusMessage}
            </div>
          )}

          {/* Global Stats Cards */}
          <div className="grid grid-cols-2 lg:grid-cols-4 gap-3 mb-4">
            <StatCard
              icon={<Car className="w-4 h-4" />}
              label="Total Queue"
              value={globalMetrics.total_queue ?? '—'}
              color="text-rose-400"
            />
            <StatCard
              icon={<Timer className="w-4 h-4" />}
              label="Avg Reward"
              value={globalMetrics.avg_reward ?? '—'}
              color="text-emerald-400"
            />
            <StatCard
              icon={<Activity className="w-4 h-4" />}
              label="Intersections"
              value={numIntersections || simConfig.intersections.length || '—'}
              color="text-indigo-400"
            />
            <StatCard
              icon={<AlertTriangle className="w-4 h-4" />}
              label="Step"
              value={latestMetrics?.step ?? '—'}
              color="text-amber-400"
            />
          </div>

          {/* Per-Intersection Cards */}
          {numIntersections > 0 && (
            <div className="grid grid-cols-1 md:grid-cols-3 gap-3 mb-4">
              {Object.entries(intersectionData).map(([nid, data]) => (
                <div key={nid} className="glass-panel p-3 rounded-xl">
                  <div className="flex items-center gap-2 mb-2">
                    <span className="text-lg">{data.tier === 1 ? '🏥' : data.tier === 2 ? '🏫' : '📍'}</span>
                    <span className="text-sm font-bold">{data.name}</span>
                    <span className={`text-xs px-2 py-0.5 rounded-full ml-auto ${
                      data.tier === 1 ? 'bg-red-500/20 text-red-300' :
                      data.tier === 2 ? 'bg-amber-500/20 text-amber-300' :
                      'bg-blue-500/20 text-blue-300'
                    }`}>
                      {data.tier_label}
                    </span>
                  </div>
                  <div className="grid grid-cols-3 gap-2 text-xs">
                    <div>
                      <p className="text-slate-500">Queue</p>
                      <p className="text-rose-400 font-bold">{data.total_queue}</p>
                    </div>
                    <div>
                      <p className="text-slate-500">Reward</p>
                      <p className="text-emerald-400 font-bold">{data.reward}</p>
                    </div>
                    <div>
                      <p className="text-slate-500">Congestion</p>
                      <p className="text-amber-400 font-bold">{data.congestion}x</p>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          )}

          {/* Charts */}
          <LiveCharts simData={simData} intersections={simConfig.intersections} />
        </div>
      </div>
    </div>
  );
}

function StatCard({ icon, label, value, color }) {
  return (
    <div className="glass-panel p-3 rounded-xl">
      <div className="flex items-center gap-2 mb-1">
        <span className={color}>{icon}</span>
        <span className="text-xs text-slate-400">{label}</span>
      </div>
      <p className={`text-xl font-bold ${color}`}>{value}</p>
    </div>
  );
}

export default App;
