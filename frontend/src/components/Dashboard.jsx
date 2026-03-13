import { Activity, Timer, Car, AlertTriangle } from 'lucide-react';
import Sidebar from './Sidebar';
import LocationInput from './LocationInput';
import TimeSlotStats from './TimeSlotStats';
import LiveCharts from './LiveCharts';

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

export default function Dashboard({ 
  systemConfig, 
  simConfig, 
  setSimConfig, 
  isRunning, 
  toggleSimulation, 
  statusMessage, 
  latestMetrics, 
  numIntersections, 
  intersectionData, 
  simData, 
  handleIntersectionsChange,
  poiResults,
  onCompare
}) {
  const globalMetrics = latestMetrics?.global || {};

  return (
    <div className="flex flex-col lg:flex-row h-screen overflow-hidden">
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
          onCompare={onCompare}
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
                  <span className={`text-xs px-2 py-0.5 rounded-full ml-auto ${data.tier === 1 ? 'bg-red-500/20 text-red-300' :
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
                    {data.pareto_rewards && (
                      <p className="text-[10px] text-rose-300/60">Flow: {data.pareto_rewards.queue.toFixed(1)}</p>
                    )}
                  </div>
                  <div>
                    <p className="text-slate-500">Reward</p>
                    <p className="text-emerald-400 font-bold">{data.reward}</p>
                    {data.pareto_rewards && (
                      <p className="text-[10px] text-emerald-300/60">Delay: {data.pareto_rewards.wait.toFixed(1)}</p>
                    )}
                  </div>
                  <div>
                    <p className="text-slate-500">Congestion</p>
                    <p className="text-amber-400 font-bold">{data.congestion}x</p>
                    {data.pareto_rewards && (
                       <p className="text-[10px] text-amber-300/60">Safe: {data.pareto_rewards.safety.toFixed(1)}</p>
                    )}
                  </div>
                </div>
              </div>
            ))}
          </div>
        )}

        {/* Historical Time-Slot Data */}
        {!isRunning && <TimeSlotStats city={simConfig.city} />}

        {/* Charts */}
        <LiveCharts simData={simData} intersections={simConfig.intersections} />
      </div>
    </div>
  );
}
