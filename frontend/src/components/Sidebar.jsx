import { Activity, Zap, ZapOff, TrendingUp } from 'lucide-react';

export default function Sidebar({ simConfig, onConfigChange, isRunning, onToggleSimulation, cities, algorithms, poiResults, onCompare }) {
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

      {/* Pareto Legend */}
      {simConfig.algorithm === 'AdaptFlow' && (
        <div className="glass-panel p-3 rounded-lg border border-indigo-500/20">
          <h4 className="text-[10px] uppercase tracking-wider font-bold text-indigo-400 mb-3">Pareto Multi-Objective</h4>
          <div className="grid grid-cols-1 gap-3">
            {[
              {
                label: 'Flow', color: 'text-rose-300',
                def: 'Queue Length Penalty',
                sig: 'Penalises vehicle build-up at the intersection; drives the agent to keep lanes clear'
              },
              {
                label: 'Delay', color: 'text-emerald-300',
                def: 'Wait Time Penalty',
                sig: 'Penalises cumulative vehicle idling; minimising delay reduces emissions & fuel waste'
              },
              {
                label: 'Safe', color: 'text-amber-300',
                def: 'Collision Risk Penalty',
                sig: 'Penalises unsafe phase transitions & near-miss events; hard constraint for real deployment'
              },
            ].map(m => (
              <div key={m.label} className="flex gap-2 text-[10px]">
                <span className={`${m.color} font-bold w-9 flex-shrink-0 mt-0.5`}>{m.label}:</span>
                <div>
                  <p className="text-slate-300 font-semibold leading-tight">{m.def}</p>
                  <p className="text-slate-500 italic mt-0.5 leading-snug">{m.sig}</p>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

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

      {/* Compare Algos Button */}
      <button
        onClick={onCompare}
        className="w-full mt-auto mb-2 flex items-center justify-center gap-2 px-4 py-3 bg-indigo-500/10 hover:bg-indigo-500/20 border border-indigo-500/30 rounded-xl transition-all font-bold text-indigo-300"
      >
        <TrendingUp className="w-4 h-4" />
        Compare Algorithms
      </button>

      {/* Run Simulation Button */}
      <button
        onClick={onToggleSimulation}
        disabled={!isRunning && (!simConfig.intersections || simConfig.intersections.length === 0)}
        className={`w-full flex items-center justify-center gap-2 px-4 py-3 rounded-xl transition-all font-bold
          ${isRunning
            ? 'bg-rose-500 hover:bg-rose-400 text-white shadow-lg shadow-rose-500/20'
            : !simConfig.intersections || simConfig.intersections.length === 0
              ? 'bg-white/10 text-slate-500 cursor-not-allowed'
              : 'bg-emerald-500 hover:bg-emerald-400 text-white shadow-lg shadow-emerald-500/20'
          }`}
      >
        {isRunning ? (
          <>
            <ZapOff className="w-5 h-5 animate-pulse" />
            Stop Simulation
          </>
        ) : (
          <>
            <Zap className="w-5 h-5" />
            Start Simulation
          </>
        )}
      </button>

      {!isRunning && (!simConfig.intersections || simConfig.intersections.length === 0) && (
        <p className="text-xs text-center text-slate-500">
          Place at least 1 pin on the map to start
        </p>
      )}
    </aside>
  );
}
