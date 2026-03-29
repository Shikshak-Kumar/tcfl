import { Activity, Timer, Car, AlertTriangle, Box, BarChart3 } from 'lucide-react';
import { useState } from 'react';
import Sidebar from './Sidebar';
import LocationInput from './LocationInput';
import TimeSlotStats from './TimeSlotStats';
import LiveCharts from './LiveCharts';
import Map3DView from './Map3DView';

function StatCard({ icon, label, value, color, description }) {
  return (
    <div className="glass-panel p-3 rounded-xl flex flex-col gap-1 border-b-2 transition-all hover:bg-white/5" style={{ borderBottomColor: color.includes('rose') ? '#ef444433' : color.includes('emerald') ? '#10b98133' : color.includes('indigo') ? '#6366f133' : '#f59e0b33' }}>
      <div className="flex items-center gap-2 mb-1">
        <span className={color}>{icon}</span>
        <span className="text-[10px] uppercase tracking-wider text-slate-500 font-bold">{label}</span>
      </div>
      <p className={`text-xl font-black ${color}`}>{value}</p>
      {description && (
        <p className="text-[9px] text-slate-500 leading-tight mt-1 italic">{description}</p>
      )}
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
  onCompare,
  onShowAnalytics
}) {
  const [activeTab, setActiveTab] = useState('charts'); // 'charts' or '3d'
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
          onShowAnalytics={onShowAnalytics}
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
            description="Sum of all waiting vehicles across the selected network."
          />
          <StatCard
            icon={<Timer className="w-4 h-4" />}
            label="Avg Reward"
            value={globalMetrics.avg_reward ?? '—'}
            color="text-emerald-400"
            description="Model efficiency score. Higher means smarter signal timing."
          />
          <StatCard
            icon={<Activity className="w-4 h-4" />}
            label="Intersections"
            value={numIntersections || simConfig.intersections.length || '—'}
            color="text-indigo-400"
            description="Number of signal-controlled nodes in the simulation."
          />
          <StatCard
            icon={<AlertTriangle className="w-4 h-4" />}
            label="Step"
            value={latestMetrics?.step ?? '—'}
            color="text-amber-400"
            description="Current simulation time-step (1 tick = signal update)."
          />
        </div>

        {/* View Toggles */}
        <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 mb-4 glass-panel p-2">
            <div className="flex gap-2">
                <button 
                    onClick={() => setActiveTab('charts')}
                    className={`flex items-center gap-2 px-4 py-2 rounded-lg text-xs font-bold transition-all ${activeTab === 'charts' ? 'bg-indigo-500 text-white shadow-lg shadow-indigo-500/20' : 'bg-white/5 text-slate-400 hover:bg-white/10'}`}
                >
                    <BarChart3 className="w-4 h-4" />
                    Live Metrics
                </button>
                <button 
                    onClick={() => setActiveTab('3d')}
                    className={`flex items-center gap-2 px-4 py-2 rounded-lg text-xs font-bold transition-all ${activeTab === '3d' ? 'bg-indigo-500 text-white shadow-lg shadow-indigo-500/20' : 'bg-white/5 text-slate-400 hover:bg-white/10'}`}
                >
                    <Box className="w-4 h-4" />
                    3D Density View
                </button>
            </div>
            <p className="text-[10px] text-slate-500 italic pr-2">
                {activeTab === 'charts' 
                    ? "Displaying real-time line charts for rewards, queue lengths, and waiting times." 
                    : "Visualizing traffic density as 3D bars across the geographical map."}
            </p>
        </div>

        {activeTab === '3d' ? (
           <div className="mb-4 h-[600px]">
              <Map3DView 
                intersections={simConfig.intersections} 
                latestMetrics={latestMetrics} 
              />
           </div>
        ) : (
          <>
            {/* Per-Intersection Cards */}
            {numIntersections > 0 && (
              <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-3 mb-4">
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
                        {data.pareto_rewards && simConfig.algorithm === 'AdaptFlow' && (
                          <p className="text-[10px] text-rose-300/60">Flow: {data.pareto_rewards.queue.toFixed(1)}</p>
                        )}
                      </div>
                      <div>
                        <p className="text-slate-500">Reward</p>
                        <p className="text-emerald-400 font-bold">{data.reward}</p>
                        {data.pareto_rewards && simConfig.algorithm === 'AdaptFlow' && (
                          <p className="text-[10px] text-emerald-300/60">Delay: {data.pareto_rewards.wait.toFixed(1)}</p>
                        )}
                      </div>
                      <div>
                        <p className="text-slate-500">Congestion</p>
                        <p className="text-amber-400 font-bold">{data.congestion}x</p>
                        {data.pareto_rewards && simConfig.algorithm === 'AdaptFlow' && (
                           <p className="text-[10px] text-amber-300/60">Safe: {data.pareto_rewards.safety.toFixed(1)}</p>
                        )}
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            )}

            {/* Charts */}
            <LiveCharts simData={simData} intersections={simConfig.intersections} />
          </>
        )}
      </div>
    </div>
  );
}
