import { Activity, Zap, ZapOff } from 'lucide-react';

export default function Sidebar({ simConfig, onConfigChange, isRunning, onToggleSimulation, cities, algorithms, poiResults }) {
  const handleChange = (key, value) => {
    onConfigChange({ ...simConfig, [key]: value });
  };

  return (
    <aside className="glass-panel p-5 flex flex-col gap-5 w-full custom-scrollbar overflow-y-auto" style={{ maxHeight: '100vh' }}>
      <div className="flex items-center gap-2 mb-2">
        <Activity className="w-5 h-5 text-indigo-400" />
        <h2 className="text-lg font-bold tracking-tight">Simulation Control</h2>
      </div>

      {/* Data Source Toggle */}
      <div>
        <label className="block text-xs font-medium text-slate-400 mb-2">Data Source</label>
        <div className="flex rounded-lg overflow-hidden border border-white/10">
          <button
            disabled={isRunning}
            onClick={() => handleChange('use_tomtom', true)}
            className={`flex-1 py-2 text-xs font-semibold transition-all flex items-center justify-center gap-1.5
              ${simConfig.use_tomtom 
                ? 'bg-indigo-500/30 text-indigo-300 border-indigo-400/30' 
                : 'bg-white/5 text-slate-400 hover:bg-white/10'}`}
          >
            <Zap className="w-3.5 h-3.5" /> Real-Time
          </button>
          <button
            disabled={isRunning}
            onClick={() => handleChange('use_tomtom', false)}
            className={`flex-1 py-2 text-xs font-semibold transition-all flex items-center justify-center gap-1.5
              ${!simConfig.use_tomtom 
                ? 'bg-amber-500/30 text-amber-300 border-amber-400/30' 
                : 'bg-white/5 text-slate-400 hover:bg-white/10'}`}
          >
            <ZapOff className="w-3.5 h-3.5" /> Mock
          </button>
        </div>
      </div>

      {/* City Selection */}
      <div>
        <label className="block text-xs font-medium text-slate-400 mb-1.5">City</label>
        <select
          disabled={isRunning}
          value={simConfig.city}
          onChange={(e) => handleChange('city', e.target.value)}
          className="w-full border border-white/10 rounded-lg px-3 py-2 text-sm focus:outline-none focus:ring-1 focus:ring-indigo-500/50"
          style={{ backgroundColor: '#1e293b', color: '#e2e8f0' }}
        >
          {Object.keys(cities).map((c) => (
            <option key={c} value={c} style={{ backgroundColor: '#1e293b', color: '#e2e8f0' }}>{c}</option>
          ))}
        </select>
      </div>

      {/* Algorithm */}
      <div>
        <label className="block text-xs font-medium text-slate-400 mb-1.5">Algorithm</label>
        <select
          disabled={isRunning}
          value={simConfig.algorithm}
          onChange={(e) => handleChange('algorithm', e.target.value)}
          className="w-full border border-white/10 rounded-lg px-3 py-2 text-sm focus:outline-none focus:ring-1 focus:ring-indigo-500/50"
          style={{ backgroundColor: '#1e293b', color: '#e2e8f0' }}
        >
          {algorithms.map((a) => (
            <option key={a} value={a} style={{ backgroundColor: '#1e293b', color: '#e2e8f0' }}>{a}</option>
          ))}
        </select>
      </div>

      {/* POI Detection Results */}
      {poiResults && poiResults.length > 0 && (
        <div>
          <label className="block text-xs font-medium text-slate-400 mb-1.5">POI Detection Results</label>
          <div className="flex flex-col gap-2">
            {poiResults.map((r, i) => (
              <div key={i} className="glass-panel p-3 rounded-lg">
                <div className="flex items-center gap-2 mb-1">
                  <span className="text-lg">{r.tier_emoji}</span>
                  <span className="text-sm font-semibold">{r.name}</span>
                  <span className={`text-xs px-2 py-0.5 rounded-full ${
                    r.tier === 1 ? 'bg-red-500/20 text-red-300' :
                    r.tier === 2 ? 'bg-amber-500/20 text-amber-300' :
                    'bg-blue-500/20 text-blue-300'
                  }`}>
                    Tier {r.tier}
                  </span>
                </div>
                <p className="text-xs text-slate-400">
                  {r.tier_label} — {r.poi_count} POIs within 1km
                </p>
                {r.detected_pois?.length > 0 && (
                  <p className="text-xs text-slate-500 mt-1 italic">
                    {r.detected_pois.slice(0, 3).join(', ')}
                  </p>
                )}
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Start / Stop */}
      <button
        onClick={onToggleSimulation}
        disabled={!isRunning && (!simConfig.intersections || simConfig.intersections.length === 0)}
        className={`w-full py-3 rounded-xl font-bold text-sm tracking-wide transition-all uppercase
          ${isRunning 
            ? 'bg-rose-500/80 hover:bg-rose-500 text-white' 
            : simConfig.intersections?.length > 0
              ? 'bg-indigo-500/80 hover:bg-indigo-500 text-white'
              : 'bg-white/10 text-slate-500 cursor-not-allowed'
          }`}
      >
        {isRunning ? '◼ Stop Simulation' : '▶ Start Simulation'}
      </button>
      
      {!isRunning && (!simConfig.intersections || simConfig.intersections.length === 0) && (
        <p className="text-xs text-center text-slate-500">
          Place at least 1 pin on the map to start
        </p>
      )}
    </aside>
  );
}
