import { useState, useEffect, useRef } from 'react';
import LandingPage from './components/LandingPage';
import Dashboard from './components/Dashboard';
import ComparisonView from './components/ComparisonView';
import AdaptFlowAnalytics from './components/AdaptFlowAnalytics';
import { TrendingUp, BarChart2 } from 'lucide-react';
import './App.css';

const API_BASE = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';
const WS_BASE = import.meta.env.VITE_WS_BASE_URL || 'ws://localhost:8000';

function App() {
  // --- View State (Hash-based routing for back button support) ---
  const [view, setView] = useState(() => {
    const hash = window.location.hash.replace('#', '');
    return hash || localStorage.getItem('appView') || 'landing';
  });

  // --- Simulation Config & Persistence ---
  const [simConfig, setSimConfig] = useState(() => {
    const saved = localStorage.getItem('simConfig');
    return saved ? JSON.parse(saved) : {
      city: 'Delhi',
      algorithm: 'Demo (No RL)',
      use_tomtom: true,
      intersections: [],
    };
  });

  const [systemConfig, setSystemConfig] = useState(() => {
    const saved = localStorage.getItem('systemConfig');
    return saved ? JSON.parse(saved) : { cities: {}, algorithms: [], poi_categories: [] };
  });

  // --- UI State (Not persisted) ---
  const [isRunning, setIsRunning] = useState(false);
  const [simData, setSimData] = useState([]);
  const [latestMetrics, setLatestMetrics] = useState(null);
  const [poiResults, setPoiResults] = useState(null);
  const [statusMessage, setStatusMessage] = useState('');
  const wsRef = useRef(null);

  // Persistence & Routing Effects
  useEffect(() => {
    localStorage.setItem('appView', view);
    if (window.location.hash.replace('#', '') !== view) {
      window.location.hash = view;
    }
  }, [view]);

  useEffect(() => {
    const handleHashChange = () => {
      const hash = window.location.hash.replace('#', '');
      if (['landing', 'dashboard', 'comparison', 'analytics'].includes(hash)) {
        setView(hash);
      }
    };
    window.addEventListener('hashchange', handleHashChange);
    return () => window.removeEventListener('hashchange', handleHashChange);
  }, []);

  useEffect(() => {
    localStorage.setItem('simConfig', JSON.stringify(simConfig));
  }, [simConfig]);

  useEffect(() => {
    localStorage.setItem('systemConfig', JSON.stringify(systemConfig));
  }, [systemConfig]);

  // Fetch config on mount
  useEffect(() => {
    fetch(`${API_BASE}/api/config`)
      .then(r => r.json())
      .then(data => {
        setSystemConfig(data);
        if (data.cities && (!simConfig.city || !data.cities[simConfig.city])) {
          const firstCity = Object.keys(data.cities)[0];
          setSimConfig(prev => ({ ...prev, city: firstCity }));
        }
      })
      .catch(err => console.error('Failed to fetch config:', err));
  }, []);

  const handleIntersectionsChange = (newIntersections) => {
    setSimConfig(prev => ({ ...prev, intersections: newIntersections }));
    setPoiResults(null);
  };

  const toggleSimulation = async () => {
    if (isRunning) {
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

      const enrichedIntersections = simConfig.intersections.map((ix, i) => ({
        ...ix,
        ...(poiData.intersections[i] || {}),
      }));

      setStatusMessage('Starting simulation...');
      const ws = new WebSocket(`${WS_BASE}/api/simulate`);
      wsRef.current = ws;

      ws.onopen = () => {
        const wsConfig = { ...simConfig, intersections: enrichedIntersections };
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
        setSimData(prev => [...prev, data].slice(-50));
      };

      ws.onclose = () => setIsRunning(false);
      ws.onerror = () => {
        setIsRunning(false);
        setStatusMessage('Connection error.');
      };

    } catch (err) {
      setIsRunning(false);
      setStatusMessage('Failed to start simulation.');
    }
  };

  const dashboardProps = {
    systemConfig,
    simConfig,
    setSimConfig,
    isRunning,
    toggleSimulation,
    statusMessage,
    latestMetrics,
    numIntersections: Object.keys(latestMetrics?.intersections || {}).length,
    intersectionData: latestMetrics?.intersections || {},
    simData,
    handleIntersectionsChange,
    poiResults,
    onCompare: () => setView('comparison'),
    onShowAnalytics: () => setView('analytics')
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-950 via-slate-900 to-indigo-950 text-white selection:bg-indigo-500/30">
      {view === 'landing' ? (
        <LandingPage onGetStarted={() => setView('dashboard')} />
      ) : view === 'comparison' ? (
        <ComparisonView onBack={() => setView('dashboard')} />
      ) : view === 'analytics' ? (
        <div className="relative">
           <div className="absolute top-4 right-4 z-50 flex gap-2">
            <button
              onClick={() => setView('dashboard')}
              className="px-3 py-1.5 rounded-lg bg-white/5 border border-white/10 text-xs font-semibold hover:bg-white/10 transition-all flex items-center gap-2"
            >
              ← Back to Dashboard
            </button>
          </div>
          <AdaptFlowAnalytics />
        </div>
      ) : (
        <div className="relative">
          <div className="absolute top-4 right-4 z-50 flex gap-2">
            <button
              onClick={() => setView('landing')}
              className="px-3 py-1.5 rounded-lg bg-white/5 border border-white/10 text-xs font-semibold hover:bg-white/10 transition-all focus:ring-2 focus:ring-white/20 outline-none"
            >
              ← Home
            </button>
          </div>
          <Dashboard {...dashboardProps} />
        </div>
      )}
    </div>
  );
}

export default App;
