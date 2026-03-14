import React, { useState, useEffect } from 'react';
import { Radar, RadarChart, PolarGrid, PolarAngleAxis, PolarRadiusAxis, ResponsiveContainer, Legend, BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Cell } from 'recharts';
import { Info, TrendingUp, Shield, Zap, Activity, Filter, ArrowLeft } from 'lucide-react';

const API_BASE = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';

export default function ComparisonView({ onBack }) {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [selectedAlgos, setSelectedAlgos] = useState(['AdaptFlow', 'FedFlow']);

  useEffect(() => {
    fetch(`${API_BASE}/api/comparison`)
      .then(r => r.json())
      .then(d => {
        setData(d);
        setLoading(false);
      })
      .catch(err => console.error('Failed to fetch comparison:', err));
  }, []);

  if (loading) return (
    <div className="flex flex-col items-center justify-center h-full gap-4 text-slate-400">
      <Activity className="w-8 h-8 animate-spin text-indigo-500" />
      <p>Aggregating simulation results...</p>
    </div>
  );

  const getRadarData = () => {
    // Transform data for Recharts Radar
    // Subjects: Reward, Throughput, Latency, Safety, Stability
    const subjects = ['Reward', 'Throughput', 'Latency', 'Safety', 'Stability'];
    return subjects.map(sub => {
      const row = { subject: sub };
      selectedAlgos.forEach(algo => {
        const algoData = data[algo];
        if (algoData && algoData.radar) {
          const subMetric = algoData.radar.find(r => r.subject === sub);
          row[algo] = subMetric ? subMetric.A : 0;
        }
      });
      return row;
    });
  };

  const colors = {
    AdaptFlow: '#818cf8',
    FedFlow: '#34d399',
    FedCM: '#fbbf24',
    FedAvg: '#f87171'
  };

  return (
    <div className="h-full flex flex-col gap-6 p-6 overflow-y-auto custom-scrollbar">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-3">
          <button onClick={onBack} className="p-2 hover:bg-white/10 rounded-lg transition-colors">
            <ArrowLeft className="w-5 h-5" />
          </button>
          <div>
            <h2 className="text-2xl font-bold tracking-tight">Algorithm Comparison</h2>
            <p className="text-sm text-slate-400 italic">Evaluating federated learning strategies across 5 key dimensions</p>
          </div>
        </div>

        <div className="flex items-center gap-2 p-1 bg-white/5 rounded-xl border border-white/10">
          {Object.keys(data).map(algo => (
            <button
              key={algo}
              onClick={() => setSelectedAlgos(prev =>
                prev.includes(algo) ? prev.filter(a => a !== algo) : [...prev, algo]
              )}
              className={`px-3 py-1.5 rounded-lg text-xs font-bold transition-all ${selectedAlgos.includes(algo)
                ? 'bg-white/10 text-white'
                : 'text-slate-500 hover:text-slate-300'
                }`}
            >
              <div className="flex items-center gap-2">
                <div className="w-2 h-2 rounded-full" style={{ backgroundColor: colors[algo] }} />
                {algo}
              </div>
            </button>
          ))}
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Radar Chart */}
        <div className="glass-panel p-6 min-h-[400px]">
          <div className="flex items-center gap-2 mb-6">
            <Zap className="w-5 h-5 text-indigo-400" />
            <h3 className="font-bold">Performance Spectrum</h3>
            <div className="ml-auto group relative cursor-help">
              <Info className="w-4 h-4 text-slate-500" />
              <div className="absolute right-0 bottom-full mb-2 w-64 p-2 bg-slate-800 border border-slate-700 rounded-lg text-[10px] text-slate-400 hidden group-hover:block z-50">
                Measures how well algorithms balance competing objectives like throughput and delay.
              </div>
            </div>
          </div>

          <div className="h-[320px] w-full">
            <ResponsiveContainer width="100%" height="100%">
              <RadarChart cx="50%" cy="50%" outerRadius="80%" data={getRadarData()}>
                <PolarGrid stroke="#475569" />
                <PolarAngleAxis dataKey="subject" tick={{ fill: '#94a3b8', fontSize: 12 }} />
                <PolarRadiusAxis angle={30} domain={[0, 100]} tick={false} />
                {selectedAlgos.map(algo => (
                  <Radar
                    key={algo}
                    name={algo}
                    dataKey={algo}
                    stroke={colors[algo]}
                    fill={colors[algo]}
                    fillOpacity={0.3}
                  />
                ))}
                <Legend />
              </RadarChart>
            </ResponsiveContainer>
          </div>

          {/* Metric Glossary */}
          <div className="mt-5 pt-5 border-t border-white/5">
            <p className="text-[10px] text-slate-500 uppercase tracking-widest font-bold mb-3">What each axis means</p>
            <div className="grid grid-cols-1 gap-2.5">
              {[
                {
                  label: 'Reward', color: '#818cf8',
                  desc: 'RL agent score — overall intersection management efficiency',
                  sig: 'Primary optimisation target; higher reward = fewer stops & smoother green-wave coordination'
                },
                {
                  label: 'Throughput', color: '#34d399',
                  desc: 'Traffic volume cleared — derived from queue length',
                  sig: 'Critical for urban arteries; poor throughput causes cascading congestion across the network'
                },
                {
                  label: 'Latency', color: '#fbbf24',
                  desc: 'Vehicle delay — derived from cumulative wait time',
                  sig: 'Directly impacts fuel consumption, emissions & driver experience; key SUMO performance metric'
                },
                {
                  label: 'Safety', color: '#f87171',
                  desc: 'Minimisation of sudden stops, tailbacks & collision risk',
                  sig: 'Hard constraint for real-world deployment; accidents negate all efficiency gains'
                },
                {
                  label: 'Stability', color: '#60a5fa',
                  desc: 'Consistency of flow — penalises erratic phase switching',
                  sig: 'Unstable control causes oscillating queues; reflects real-world deployability of the algorithm'
                },
              ].map(m => (
                <div key={m.label} className="flex gap-2.5 text-[11px]">
                  <div className="mt-1 w-2 h-2 rounded-full flex-shrink-0" style={{ backgroundColor: m.color }} />
                  <div>
                    <div className="flex flex-wrap items-baseline gap-x-2">
                      <span className="text-slate-200 font-bold">{m.label}</span>
                      <span className="text-slate-500">{m.desc}</span>
                    </div>
                    <p className="text-[10px] text-slate-600 mt-0.5 leading-snug italic">{m.sig}</p>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>

        {/* Comparison Table / Heatmap Style */}
        <div className="glass-panel p-6 flex flex-col">
          <div className="flex items-center gap-2 mb-6">
            <Shield className="w-5 h-5 text-emerald-400" />
            <h3 className="font-bold">Key Performance Indicators</h3>
          </div>

          <div className="flex-1 space-y-4 overflow-y-auto">
            {Object.keys(data).map(algo => {
              const isSelected = selectedAlgos.includes(algo);
              const m = data[algo];
              return (
                <div key={algo} className={`p-4 rounded-xl border transition-all ${isSelected ? 'bg-indigo-500/5 border-indigo-500/20' : 'bg-white/5 border-white/10 opacity-60'
                  }`}>
                  <div className="flex items-center justify-between mb-3">
                    <span className="font-bold" style={{ color: colors[algo] }}>{algo}</span>
                    <span className="text-[10px] text-slate-500 uppercase tracking-widest font-bold">Latest Eval</span>
                  </div>
                  <div className="grid grid-cols-3 gap-4">
                    <div className="text-center">
                      <p className="text-[10px] text-slate-500 mb-1">Avg Reward</p>
                      <p className="text-sm font-bold text-slate-200">{m.reward.toFixed(1)}</p>
                    </div>
                    <div className="text-center">
                      <p className="text-[10px] text-slate-500 mb-1">Wait Time</p>
                      <p className="text-sm font-bold text-slate-200">{m.waiting_time.toFixed(1)}s</p>
                    </div>
                    <div className="text-center">
                      <p className="text-[10px] text-slate-500 mb-1">Queue Len</p>
                      <p className="text-sm font-bold text-slate-200">{m.queue.toFixed(0)}</p>
                    </div>
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      </div>

      {/* Bar Chart Section */}
      <div className="glass-panel p-6">
        <div className="flex items-center gap-2 mb-6">
          <TrendingUp className="w-5 h-5 text-blue-400" />
          <h3 className="font-bold">Wait Time Benchmarks</h3>
        </div>
        <div className="h-[250px] w-full">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={Object.keys(data).map(a => ({ name: a, val: data[a].waiting_time }))}>
              <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" vertical={false} />
              <XAxis dataKey="name" stroke="#64748b" tick={{ fontSize: 12 }} />
              <YAxis stroke="#64748b" tick={{ fontSize: 12 }} unit="s" />
              <Tooltip
                contentStyle={{ backgroundColor: '#0f172a', border: '1px solid #334155', borderRadius: '8px' }}
                itemStyle={{ color: '#e2e8f0' }}
              />
              <Bar dataKey="val" radius={[4, 4, 0, 0]}>
                {Object.keys(data).map((entry, index) => (
                  <Cell key={`cell-${index}`} fill={colors[entry]} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>
    </div>
  );
}