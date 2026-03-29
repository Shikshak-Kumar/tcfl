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
    const interval = setInterval(() => fetchAnalytics(source), 10000);
    return () => clearInterval(interval);
  }, [source]);

  const fetchAnalytics = async (currentSource) => {
    try {
      const resp = await axios.get(`${API_BASE}/api/adaptflow/analytics?source=${currentSource}`);
      setData(resp.data);
      // Auto-select latest round if we just loaded data for the first time or rounds changed
      if (resp.data.num_rounds > 0) {
          if (selectedRound >= resp.data.num_rounds) {
            setSelectedRound(resp.data.num_rounds - 1);
          }
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

  // 1. Gated Return: If no rounds found, show "No Data" UI but keep the toggle buttons
  if (!data || data.num_rounds === 0) {
    return (
        <div className="flex flex-col gap-6 p-4">
            <div className="flex items-center justify-center gap-4 glass-panel p-4">
                <button 
                    onClick={() => {setSource('sim'); setLoading(true);}}
                    className={`px-4 py-2 rounded-lg text-xs font-bold transition-all ${source === 'sim' ? 'bg-indigo-500 text-white shadow-lg' : 'text-slate-400 hover:text-slate-200'}`}
                >
                    Testing (Sim)
                </button>
                <button 
                    onClick={() => {setSource('train'); setLoading(true);}}
                    className={`px-4 py-2 rounded-lg text-xs font-bold transition-all ${source === 'train' ? 'bg-indigo-500 text-white shadow-lg' : 'text-slate-400 hover:text-slate-200'}`}
                >
                    Training
                </button>
            </div>
            <div className="glass-panel p-20 text-center text-slate-400 flex flex-col items-center justify-center">
                <Cpu className="w-16 h-16 mb-4 opacity-10 animate-pulse" />
                <h3 className="text-xl font-bold text-white mb-2">No {source === 'sim' ? 'Simulation' : 'Training'} Intelligence Found</h3>
                <p className="text-sm text-slate-500 max-w-sm">
                    The requested {source} history file is empty or missing metrics. 
                    Run a session in the dashboard first to generate clustering intelligence.
                </p>
                {source === 'train' && (
                    <div className="mt-6 p-3 bg-indigo-500/10 rounded border border-indigo-500/20 text-[10px] font-mono text-indigo-300">
                        HINT: Run 'python train_adaptflow.py --rounds 3' in backend.
                    </div>
                )}
            </div>
        </div>
    );
  }

  // 2. Data Preparation (Only reached if num_rounds > 0)
  const currentFingerprints = data.fingerprints[selectedRound] || {};
  const currentSimilarity = data.similarity_matrices[selectedRound] || [];
  const nodeIds = Object.keys(currentFingerprints);

  const roundMetricsData = Array.from({ length: data.num_rounds }).map((_, i) => ({
    round: i + 1,
    reward: data.reward_history ? data.reward_history[i] : (data.episode_rewards ? data.episode_rewards[i] : 0),
    queue: data.queue_history ? data.queue_history[i] : 0,
    wait: data.wait_history ? data.wait_history[i] : 0,
    throughput: data.throughput_history ? data.throughput_history[i] : 0,
  }));

  const clusterDistributionData = Array.from({ length: data.num_rounds }).map((_, i) => {
    const roundClusters = data.cluster_history[i] || {};
    const counts = {};
    Object.values(roundClusters).forEach(cid => {
      counts[cid] = (counts[cid] || 0) + 1;
    });
    return { round: source === 'sim' ? `W${i + 1}` : `R${i + 1}`, ...counts };
  });

  const RADAR_LABELS = ["Wait Time", "Queue", "Throughput", "Max Queue", "Congested Lanes", "Priority"];
  const radarData = RADAR_LABELS.map((label, i) => {
    const point = { subject: label, fullMark: 1.0 };
    nodeIds.forEach(nid => {
      point[nid] = currentFingerprints[nid] ? currentFingerprints[nid][i] : 0;
    });
    return point;
  });

  const chartColors = ["#6366f1", "#10b981", "#f59e0b", "#ef4444", "#8b5cf6", "#06b6d4", "#ec4899", "#14b8a6"];

  const snapshotLabel = source === 'sim' ? 'Analysis Window' : 'Training Round';
  const snapshotPrefix = source === 'sim' ? 'Window' : 'Round';

  return (
    <div className="flex flex-col gap-6 p-4">
      {/* Header & Source Toggles */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 glass-panel p-4">
        <div className="flex items-center gap-3">
          <div className="p-2 rounded-lg bg-indigo-500/20 text-indigo-400">
            <Layers className="w-5 h-5" />
          </div>
          <div>
            <h2 className="text-lg font-bold text-white">AdaptFlow: Detailed Algorithm Analysis</h2>
            <p className="text-xs text-slate-400">9-Metric Performance & Clustering Intelligence ({source === 'sim' ? 'Testing snapshots every 50 steps' : 'Cross-round training epochs'})</p>
          </div>
        </div>

        <div className="flex items-center gap-2 p-1 bg-slate-900/50 rounded-lg border border-slate-800">
            <button onClick={() => {setSource('sim'); setLoading(true);}} className={`px-3 py-1 rounded-md text-[10px] font-bold transition-all ${source === 'sim' ? 'bg-indigo-500 text-white shadow-lg' : 'text-slate-400 hover:text-slate-300'}`}>Testing (Sim)</button>
            <button onClick={() => {setSource('train'); setLoading(true);}} className={`px-3 py-1 rounded-md text-[10px] font-bold transition-all ${source === 'train' ? 'bg-indigo-500 text-white shadow-lg' : 'text-slate-400 hover:text-slate-300'}`}>Training</button>
        </div>
        
        <div className="flex items-center gap-4">
          <div className="flex items-center gap-2">
            <label className="text-[10px] uppercase tracking-wider text-slate-500 font-bold">{snapshotPrefix}:</label>
            <select className="bg-slate-900 border border-slate-700 text-slate-200 text-xs rounded px-2 py-1 outline-none font-bold" value={selectedRound} onChange={(e) => setSelectedRound(parseInt(e.target.value))}>
              {Array.from({ length: data.num_rounds }).map((_, i) => (<option key={i} value={i}>{snapshotPrefix} {i + 1}</option>))}
            </select>
          </div>
          <div className="text-[10px] font-mono text-indigo-400 bg-indigo-500/10 px-2 py-1 rounded border border-indigo-500/20">ALPHA v2.1</div>
        </div>
      </div>

      {/* Metric Cards Grid ... */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
          <MetricCard 
            title="Avg Episode Reward" 
            value={data.reward_history ? data.reward_history[selectedRound]?.toFixed(2) : '0'} 
            trend={data.reward_history && selectedRound > 0 ? data.reward_history[selectedRound] - data.reward_history[selectedRound-1] : 0} 
            color="indigo"
            description="Overall performance score. Higher means better traffic flow."
          />
          <MetricCard 
            title="Avg Queue Length" 
            value={data.queue_history ? data.queue_history[selectedRound]?.toFixed(1) : '0'} 
            trend={data.queue_history && selectedRound > 0 ? data.queue_history[selectedRound-1] - data.queue_history[selectedRound] : 0} 
            color="rose" 
            invertTrend 
            description="Number of vehicles stopped at red lights."
          />
          <MetricCard 
            title="Avg Waiting Time" 
            value={data.wait_history ? data.wait_history[selectedRound]?.toFixed(1) + 's' : '0s'} 
            trend={data.wait_history && selectedRound > 0 ? data.wait_history[selectedRound-1] - data.wait_history[selectedRound] : 0} 
            color="emerald" 
            invertTrend 
            description="Total delay experienced per vehicle in seconds."
          />
          <MetricCard 
            title="Throughput Ratio" 
            value={data.throughput_history ? (data.throughput_history[selectedRound] * 100).toFixed(1) + '%' : '0%'} 
            trend={data.throughput_history && selectedRound > 0 ? data.throughput_history[selectedRound] - data.throughput_history[selectedRound-1] : 0} 
            color="amber" 
            description="Percentage of vehicles that successfully crossed."
          />
      </div>

      {/* Main Charts Grid */}
      <div className="grid grid-cols-1 xl:grid-cols-3 gap-6">
        {/* ... Learning Progress and Congestion Trends LineCharts ... */}
        <div className="glass-panel p-4 flex flex-col h-[420px]">
          <div className="mb-4">
            <h3 className="text-[10px] font-bold text-slate-500 uppercase tracking-widest">Learning Progress</h3>
            <p className="text-[9px] text-slate-400 mt-1 leading-relaxed">Tracks AI efficiency over time. Ideally, <span className="text-indigo-400">Reward</span> should increase while <span className="text-amber-400">Throughput</span> stabilizes at a high percentage.</p>
          </div>
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={roundMetricsData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" vertical={false} />
              <XAxis dataKey="round" stroke="#475569" fontSize={10} axisLine={false} tickLine={false} />
              <YAxis yAxisId="left" stroke="#6366f1" fontSize={10} axisLine={false} tickLine={false} />
              <YAxis yAxisId="right" orientation="right" stroke="#f59e0b" fontSize={10} axisLine={false} tickLine={false} />
              <Tooltip contentStyle={{ backgroundColor: '#0f172a', border: '1px solid #334155', borderRadius: '8px', fontSize: '10px' }} />
              <Legend wrapperStyle={{ fontSize: '10px', paddingTop: '10px' }} />
              <Line yAxisId="left" type="monotone" dataKey="reward" name="Reward" stroke="#6366f1" strokeWidth={3} dot={{ r: 4, fill: '#6366f1' }} activeDot={{ r: 6 }} />
              <Line yAxisId="right" type="monotone" dataKey="throughput" name="Throughput" stroke="#f59e0b" strokeWidth={3} dot={{ r: 4, fill: '#f59e0b' }} activeDot={{ r: 6 }} />
            </LineChart>
          </ResponsiveContainer>
        </div>

        <div className="glass-panel p-4 flex flex-col h-[420px]">
          <div className="mb-4">
            <h3 className="text-[10px] font-bold text-slate-500 uppercase tracking-widest">Congestion Trends</h3>
            <p className="text-[9px] text-slate-400 mt-1 leading-relaxed">Lowering both lines means the AI is clearing traffic faster. <span className="text-rose-400">Queue</span> is the car count; <span className="text-emerald-400">Wait Time</span> is the delay.</p>
          </div>
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={roundMetricsData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" vertical={false} />
              <XAxis dataKey="round" stroke="#475569" fontSize={10} axisLine={false} tickLine={false} />
              <YAxis yAxisId="left" stroke="#ef4444" fontSize={10} axisLine={false} tickLine={false} />
              <YAxis yAxisId="right" orientation="right" stroke="#10b981" fontSize={10} axisLine={false} tickLine={false} />
              <Tooltip contentStyle={{ backgroundColor: '#0f172a', border: '1px solid #334155', borderRadius: '8px', fontSize: '10px' }} />
              <Legend wrapperStyle={{ fontSize: '10px', paddingTop: '10px' }} />
              <Line yAxisId="left" type="monotone" dataKey="queue" name="Queue Length" stroke="#ef4444" strokeWidth={3} dot={{ r: 4, fill: '#ef4444' }} />
              <Line yAxisId="right" type="monotone" dataKey="wait" name="Wait Time (s)" stroke="#10b981" strokeWidth={3} dot={{ r: 4, fill: '#10b981' }} />
            </LineChart>
          </ResponsiveContainer>
        </div>

        {/* 3. Node Fingerprints */}
        <div className="glass-panel p-4 flex flex-col h-[420px]">
          <div className="mb-4">
            <h3 className="text-[10px] font-bold text-slate-500 uppercase tracking-widest">Clustering Intelligence: Fingerprints</h3>
            <p className="text-[9px] text-slate-400 mt-1 leading-relaxed">The 'DNA' of each intersection. Intersections with similar web shapes are grouped into the same cluster for coordinated control.</p>
          </div>
          <ResponsiveContainer width="100%" height="100%">
            <RadarChart cx="50%" cy="50%" outerRadius="70%" data={radarData}>
              <PolarGrid stroke="#334155" />
              <PolarAngleAxis dataKey="subject" tick={{ fill: '#94a3b8', fontSize: 9 }} />
              <PolarRadiusAxis angle={30} domain={[0, 1]} tick={false} axisLine={false} />
              {nodeIds.map((nid, i) => (
                <Radar 
                  key={nid} 
                  name={nid} 
                  dataKey={nid} 
                  stroke={chartColors[i % chartColors.length]} 
                  fill={chartColors[i % chartColors.length]} 
                  fillOpacity={0.1 * (nodeIds.length - i) / nodeIds.length} 
                />
              ))}
              <Tooltip contentStyle={{ backgroundColor: '#0f172a', border: '1px solid #334155', borderRadius: '8px', fontSize: '10px' }} />
              <Legend iconType="circle" wrapperStyle={{ fontSize: '10px', paddingTop: '10px' }} />
            </RadarChart>
          </ResponsiveContainer>
        </div>

        {/* 4. Similarity Matrix */}
        <div className="glass-panel p-4 flex flex-col h-[400px]">
          <div className="mb-4">
            <h3 className="text-[10px] font-bold text-slate-500 uppercase tracking-widest">Similarity Matrix (Cosine)</h3>
            <p className="text-[9px] text-slate-400 mt-1 leading-relaxed">Darker green = higher similarity. Intersections with values near 1.00 share almost identical traffic burdens and POI categories.</p>
          </div>
          <div className="flex-1 overflow-auto custom-scrollbar">
             <div className="grid gap-1" style={{ gridTemplateColumns: `repeat(${nodeIds.length + 1}, 1fr)`, minWidth: nodeIds.length * 40 + 40 }}>
                <div className="h-6"></div>
                {nodeIds.map(nid => (<div key={nid} className="h-6 text-[8px] text-slate-500 flex items-center justify-center font-mono">{nid.split('_')[1]}</div>))}
                {currentSimilarity.map((row, i) => (
                  <React.Fragment key={i}>
                    <div className="w-6 h-6 text-[8px] text-slate-500 flex items-center justify-center font-mono">{nodeIds[i].split('_')[1]}</div>
                    {row.map((val, j) => (
                      <div key={j} className="aspect-square rounded-sm flex items-center justify-center text-[8px] font-bold" style={{ backgroundColor: `rgba(16, 185, 129, ${val})`, color: val > 0.6 ? '#064e3b' : '#34d399', opacity: i === j ? 0.3 : 1 }} title={`${nodeIds[i]} vs ${nodeIds[j]}: ${(val * 100).toFixed(1)}%`}>
                        {i === j ? '—' : val.toFixed(2)}
                      </div>
                    ))}
                  </React.Fragment>
                ))}
             </div>
          </div>
        </div>

        {/* 5. Cluster Distribution */}
        <div className="glass-panel p-4 flex flex-col h-[400px]">
          <div className="mb-4">
            <h3 className="text-[10px] font-bold text-slate-500 uppercase tracking-widest">Cluster Size Distribution</h3>
            <p className="text-[9px] text-slate-400 mt-1 leading-relaxed">Shows the balance of the network. Each color block is a cluster; its height indicates how many intersections are in that specific group.</p>
          </div>
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={clusterDistributionData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" vertical={false} />
              <XAxis dataKey="round" stroke="#475569" fontSize={10} axisLine={false} tickLine={false} />
              <YAxis stroke="#475569" fontSize={10} axisLine={false} tickLine={false} />
              <Tooltip contentStyle={{ backgroundColor: '#0f172a', border: '1px solid #334155', borderRadius: '8px', fontSize: '10px' }} cursor={{ fill: '#ffffff05' }} />
              <Legend wrapperStyle={{ fontSize: '10px', paddingTop: '10px' }} />
              {Array.from({ length: 6 }).map((_, i) => (<Bar key={i} dataKey={i} name={`Clust ${i}`} stackId="a" fill={chartColors[i % chartColors.length]} radius={i === 0 ? [0, 0, 4, 4] : [0, 0, 0, 0]} />))}
            </BarChart>
          </ResponsiveContainer>
        </div>

        {/* 6. Stability Tracking */}
        <div className="glass-panel p-4 flex flex-col h-[400px]">
            <div className="mb-4">
                <h3 className="text-[10px] font-bold text-slate-500 uppercase tracking-widest">Stability Tracking</h3>
                <p className="text-[9px] text-slate-400 mt-1 leading-relaxed">How often nodes change groups. High Stability (Stab.) % means the traffic patterns are predictable; low % means high volatility.</p>
            </div>
            <div className="overflow-y-auto custom-scrollbar flex-1">
                <table className="w-full text-left text-[10px] text-slate-300">
                    <thead className="text-slate-500 border-b border-slate-800 sticky top-0 bg-slate-900/95 backdrop-blur z-10">
                        <tr>
                            <th className="py-2 px-2">Node</th>
                            {data.cluster_history.map((_, i) => (<th key={i} className="py-2 px-1 text-center font-mono">{source === 'sim' ? `W${i+1}` : `R${i+1}`}</th>))}
                            <th className="py-2 px-2 text-right">Stab.</th>
                        </tr>
                    </thead>
                    <tbody className="divide-y divide-slate-800/50">
                        {nodeIds.map(nid => {
                            const memberships = data.cluster_history.map(h => h[nid]);
                            const switches = memberships.filter((m, i) => i > 0 && m !== memberships[i-1]).length;
                            const stability = ((1 - switches / (data.num_rounds - 1 || 1)) * 100).toFixed(0);
                            return (
                                <tr key={nid} className="hover:bg-slate-800/20 transition-colors">
                                    <td className="py-2 px-2 font-mono font-bold text-indigo-400">{nid.split('_')[1]}</td>
                                    {memberships.map((cid, i) => (<td key={i} className="py-2 px-1 text-center"><div className="w-2.5 h-2.5 rounded-full mx-auto" style={{ backgroundColor: chartColors[cid % chartColors.length] }} /></td>))}
                                    <td className="py-2 px-2 text-right font-bold text-white">{stability}%</td>
                                </tr>
                            );
                        })}
                    </tbody>
                </table>
            </div>
        </div>
      </div>

      <div className="p-4 bg-indigo-500/5 rounded-xl border border-indigo-500/10 flex items-center gap-4">
          <div className="p-2 bg-indigo-500/20 rounded-lg"><Cpu className="w-4 h-4 text-indigo-400" /></div>
          <p className="text-[10px] text-slate-500 leading-relaxed max-w-4xl font-medium">
              <span className="font-bold text-indigo-300 italic uppercase mr-1">Algorithm Brief:</span> AdaptFlow uses dynamic clustering to group intersections by similar traffic "Fingerprints". This allows 
              the AI to share learning weights between similar nodes, accelerating training. Stability measures group consistency, while Similarity 
              indicates shared congestion characteristics.
          </p>
      </div>
    </div>
  );
}

function MetricCard({ title, value, trend, color, invertTrend = false, description }) {
    const isPositive = trend > 0;
    const DisplayTrend = isPositive ? (invertTrend ? 'rose' : 'emerald') : (invertTrend ? 'emerald' : 'rose');
    return (
        <div className="glass-panel p-4 flex flex-col gap-1 border-l-4 group hover:bg-slate-800/30 transition-all" style={{ borderColor: color === 'indigo' ? '#6366f1' : color === 'rose' ? '#ef4444' : color === 'emerald' ? '#10b981' : '#f59e0b' }}>
            <span className="text-[10px] uppercase tracking-widest text-slate-500 font-bold">{title}</span>
            <div className="flex items-end justify-between">
                <span className="text-xl font-black text-white">{value}</span>
                {trend !== 0 && (<span className={`text-[10px] font-bold px-1.5 py-0.5 rounded ${DisplayTrend === 'emerald' ? 'bg-emerald-500/10 text-emerald-400' : 'bg-rose-500/10 text-rose-400'}`}>{isPositive ? '↑' : '↓'} {Math.abs(trend).toFixed(1)}</span>)}
            </div>
            <p className="text-[9px] text-slate-500 mt-1 italic group-hover:text-slate-400 transition-colors">{description}</p>
        </div>
    );
}
