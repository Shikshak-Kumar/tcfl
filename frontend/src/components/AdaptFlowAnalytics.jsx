import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { 
  Radar, RadarChart, PolarGrid, PolarAngleAxis, PolarRadiusAxis, ResponsiveContainer,
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend,
  BarChart, Bar
} from 'recharts';
import { Network, Fingerprint, Layers, Cpu } from 'lucide-react';

const API_BASE = import.meta.env.VITE_API_URL || 'http://localhost:8000';

export default function AdaptFlowAnalytics() {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [selectedRound, setSelectedRound] = useState(0);
  const [source, setSource] = useState('sim');

  useEffect(() => {
    fetchAnalytics(source);
    const interval = setInterval(() => fetchAnalytics(source), 10000); // Refresh every 10s
    return () => clearInterval(interval);
  }, [source]);

  const fetchAnalytics = async (currentSource) => {
    try {
      const resp = await axios.get(`${API_BASE}/api/adaptflow/analytics?source=${currentSource}`);
      setData(resp.data);
      if (resp.data.num_rounds > 0 && selectedRound >= resp.data.num_rounds) {
        setSelectedRound(resp.data.num_rounds - 1);
      } else if (resp.data.num_rounds > 0 && selectedRound === 0) {
        setSelectedRound(resp.data.num_rounds - 1);
      }
      setLoading(false);
    } catch (err) {
      console.error('Failed to fetch analytics:', err);
      setError('Failed to load AdaptFlow analytics data.');
      setLoading(false);
    }
  };

  if (loading) return <div className="p-8 text-center text-slate-400">Loading AdaptFlow Analytics...</div>;
  if (error) return <div className="p-8 text-center text-rose-400">{error}</div>;
    return (
      <div className="flex flex-col gap-6 p-4">
        <div className="flex items-center justify-center gap-4 glass-panel p-4">
            <button 
                onClick={() => {setSource('sim'); setLoading(true);}}
                className={`px-4 py-2 rounded-lg text-xs font-bold transition-all ${source === 'sim' ? 'bg-indigo-500 text-white shadow-lg shadow-indigo-500/20' : 'text-slate-400 hover:text-slate-200'}`}
            >
                Simulation Analytics
            </button>
            <button 
                onClick={() => {setSource('train'); setLoading(true);}}
                className={`px-4 py-2 rounded-lg text-xs font-bold transition-all ${source === 'train' ? 'bg-indigo-500 text-white shadow-lg shadow-indigo-500/20' : 'text-slate-400 hover:text-slate-200'}`}
            >
                Training Analytics
            </button>
        </div>
        <div className="glass-panel p-8 text-center text-slate-400">
          <Cpu className="w-12 h-12 mx-auto mb-4 opacity-20" />
          <h3 className="text-lg font-semibold mb-2">No {source === 'sim' ? 'Simulation' : 'Training'} Data Found</h3>
          <p className="text-sm">Please run a {source === 'sim' ? 'simulation' : 'training session'} to generate clustering analytics.</p>
        </div>
      </div>
    );

  const currentFingerprints = data.fingerprints[selectedRound] || {};
  const currentSimilarity = data.similarity_matrices[selectedRound] || [];
  const nodeIds = Object.keys(currentFingerprints);

  // Format fingerprint data for Radar Chart
  const FINGERPRINT_LABELS = [
    "Wait Time", "Queue", "Throughput", "Max Queue", "Congested Lanes", "Priority"
  ];

  const getRadarData = (nodeId) => {
    const fp = currentFingerprints[nodeId];
    if (!fp) return [];
    return FINGERPRINT_LABELS.map((label, i) => ({
      subject: label,
      value: fp[i],
      fullMark: 1.0,
    }));
  };

  return (
    <div className="flex flex-col gap-6 p-4">
      {/* Header & Round Selector */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 glass-panel p-4">
        <div className="flex items-center gap-3">
          <div className="p-2 rounded-lg bg-indigo-500/20 text-indigo-400">
            <Layers className="w-5 h-5" />
          </div>
          <div>
            <h2 className="text-lg font-bold text-white">AdaptFlow: Dynamic Clustering Analytics</h2>
            <p className="text-xs text-slate-400">Pairwise similarity & multi-dim fingerprints ({source === 'sim' ? 'Real-time Simulation' : 'Offline Training'})</p>
          </div>
        </div>

        <div className="flex items-center gap-2 p-1 bg-slate-900/50 rounded-lg border border-slate-800">
            <button 
                onClick={() => {setSource('sim'); setLoading(true);}}
                className={`px-3 py-1 rounded-md text-[10px] font-bold transition-all ${source === 'sim' ? 'bg-indigo-500 text-white' : 'text-slate-400 hover:text-slate-300'}`}
            >
                Simulation
            </button>
            <button 
                onClick={() => {setSource('train'); setLoading(true);}}
                className={`px-3 py-1 rounded-md text-[10px] font-bold transition-all ${source === 'train' ? 'bg-indigo-500 text-white' : 'text-slate-400 hover:text-slate-300'}`}
            >
                Training
            </button>
        </div>
        
        <div className="flex items-center gap-2">
          <label className="text-xs text-slate-400">Selected Round:</label>
          <select 
            className="bg-slate-900 border border-slate-700 text-slate-200 text-xs rounded px-2 py-1 outline-none"
            value={selectedRound}
            onChange={(e) => setSelectedRound(parseInt(e.target.value))}
          >
            {Array.from({ length: data.num_rounds }).map((_, i) => (
              <option key={i} value={i}>Round {i + 1}</option>
            ))}
          </select>
        </div>
      </div>

      <div className="grid grid-cols-1 xl:grid-cols-2 gap-6">
        {/* 1. Cosine Similarity Heatmap */}
        <div className="glass-panel p-4 flex flex-col h-[450px]">
          <div className="flex items-center gap-2 mb-4">
            <Network className="w-4 h-4 text-emerald-400" />
            <h3 className="text-sm font-semibold text-slate-200">Similarity Matrix (Cosine)</h3>
          </div>
          
          <div className="flex-1 relative overflow-auto custom-scrollbar">
             <div 
                className="grid gap-1 mb-6" 
                style={{ 
                    gridTemplateColumns: `repeat(${nodeIds.length + 1}, 1fr)`,
                    minWidth: nodeIds.length * 60 + 60
                }}
             >
                <div className="h-8"></div>
                {nodeIds.map(nid => (
                    <div key={nid} className="h-8 text-[10px] text-slate-500 flex items-center justify-center font-mono">
                        {nid.split('_')[1]}
                    </div>
                ))}
                
                {currentSimilarity.map((row, i) => (
                  <React.Fragment key={i}>
                    <div className="w-8 h-8 text-[10px] text-slate-500 flex items-center justify-center font-mono">
                        {nodeIds[i].split('_')[1]}
                    </div>
                    {row.map((val, j) => (
                      <div 
                        key={j} 
                        className="aspect-square rounded-sm flex items-center justify-center text-[10px] font-bold transition-all hover:scale-110"
                        style={{ 
                          backgroundColor: `rgba(16, 185, 129, ${val})`,
                          color: val > 0.6 ? '#064e3b' : '#34d399',
                          border: i === j ? '1px solid rgba(255,255,255,0.2)' : 'none'
                        }}
                        title={`${nodeIds[i]} vs ${nodeIds[j]}: ${(val * 100).toFixed(1)}%`}
                      >
                        {i === j ? '—' : val.toFixed(2)}
                      </div>
                    ))}
                  </React.Fragment>
                ))}
             </div>
          </div>
          <div className="mt-2 flex items-center justify-end gap-2 text-[10px] text-slate-500">
             <span>Low Similarity</span>
             <div className="w-20 h-2 bg-gradient-to-r from-emerald-500/10 to-emerald-500 rounded-full"></div>
             <span>High Similarity</span>
          </div>
        </div>

        {/* 2. Congestion Fingerprints (Radar) */}
        <div className="glass-panel p-4 flex flex-col h-[450px]">
          <div className="flex items-center gap-2 mb-4">
            <Fingerprint className="w-4 h-4 text-indigo-400" />
            <h3 className="text-sm font-semibold text-slate-200">Multivariate Congestion Fingerprint</h3>
          </div>
          
          <div className="flex-1">
            <ResponsiveContainer width="100%" height="100%">
              <RadarChart cx="50%" cy="50%" outerRadius="70%" data={getRadarData(nodeIds[0])}>
                <PolarGrid stroke="#334155" />
                <PolarAngleAxis dataKey="subject" tick={{ fill: '#94a3b8', fontSize: 10 }} />
                <PolarRadiusAxis angle={30} domain={[0, 1]} tick={false} axisLine={false} />
                {nodeIds.map((nid, i) => (
                   <Radar
                    key={nid}
                    name={nid}
                    dataKey="value"
                    data={getRadarData(nid)}
                    stroke={`hsl(${i * (360 / nodeIds.length)}, 70%, 50%)`}
                    fill={`hsl(${i * (360 / nodeIds.length)}, 70%, 50%)`}
                    fillOpacity={0.2}
                  />
                ))}
                <Tooltip 
                  contentStyle={{ backgroundColor: '#0f172a', border: '1px solid #334155', borderRadius: '8px', fontSize: '10px' }}
                />
                <Legend iconType="circle" wrapperStyle={{ fontSize: '10px', paddingTop: '10px' }} />
              </RadarChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* 3. Cluster Transitions Stability */}
        <div className="glass-panel p-4 col-span-full">
            <div className="flex items-center gap-2 mb-4">
                <Layers className="w-4 h-4 text-amber-400" />
                <h3 className="text-sm font-semibold text-slate-200">Cluster Membership & Stability</h3>
            </div>
            <div className="overflow-x-auto custom-scrollbar">
                <table className="w-full text-left text-xs text-slate-300">
                    <thead className="text-slate-500 border-b border-slate-800">
                        <tr>
                            <th className="py-2 px-4">Node ID</th>
                            {data.cluster_history.map((_, i) => (
                                <th key={i} className="py-2 px-4 text-center">Round {i + 1}</th>
                            ))}
                            <th className="py-2 px-4 text-right">Stability</th>
                        </tr>
                    </thead>
                    <tbody className="divide-y divide-slate-800/50">
                        {nodeIds.map(nid => {
                            const memberships = data.cluster_history.map(h => h[nid]);
                            const switches = memberships.filter((m, i) => i > 0 && m !== memberships[i-1]).length;
                            const stability = ((1 - switches / (data.num_rounds - 1 || 1)) * 100).toFixed(0);
                            
                            return (
                                <tr key={nid} className="hover:bg-slate-800/20 transition-colors">
                                    <td className="py-3 px-4 font-mono font-bold text-slate-100">{nid}</td>
                                    {memberships.map((cid, i) => (
                                        <td key={i} className="py-3 px-4 text-center">
                                            <span 
                                                className="px-3 py-1 rounded-full text-[10px] font-bold"
                                                style={{ 
                                                    backgroundColor: `hsla(${cid * 100}, 70%, 40%, 0.2)`,
                                                    color: `hsl(${cid * 100}, 70%, 60%)`,
                                                    border: `1px solid hsla(${cid * 100}, 70%, 40%, 0.4)`
                                                }}
                                            >
                                                Cluster {cid}
                                            </span>
                                        </td>
                                    ))}
                                    <td className="py-3 px-4 text-right">
                                        <div className="flex items-center justify-end gap-2">
                                            <div className="w-16 h-1.5 bg-slate-800 rounded-full overflow-hidden">
                                                <div 
                                                    className={`h-full rounded-full ${parseInt(stability) > 70 ? 'bg-emerald-500' : parseInt(stability) > 30 ? 'bg-amber-500' : 'bg-rose-500'}`}
                                                    style={{ width: `${stability}%` }}
                                                ></div>
                                            </div>
                                            <span className="w-8 font-bold">{stability}%</span>
                                        </div>
                                    </td>
                                </tr>
                            );
                        })}
                    </tbody>
                </table>
            </div>
            <div className="mt-4 p-3 bg-indigo-500/5 rounded-lg border border-indigo-500/10">
                <p className="text-[10px] text-indigo-300 leading-relaxed">
                    <span className="font-bold">Insight:</span> Low stability (high switches) indicates the node is at a congestion boundary, frequently re-adjusting its group to match fluctuating traffic flows. High stability indicates a consistent congestion pattern.
                </p>
            </div>
        </div>
      </div>
    </div>
  );
}
