import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { 
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend,
  AreaChart, Area, ResponsiveContainer
} from 'recharts';
import { Network, Fingerprint, Layers, Cpu, Info, Target, Activity, Clock, BarChart3, HelpCircle, Shuffle } from 'lucide-react';

const API_BASE = import.meta.env.VITE_API_URL || 'http://localhost:8000';

const GLOSSARY = [
  { term: 'AdaptFlow', def: 'An adaptive AI algorithm that groups similar intersections to coordinate traffic lights globally.' },
  { term: 'Clustering', def: 'The process of grouping intersections that experience similar traffic patterns (like "Rush Hour" or "Quiet Suburb").' },
  { term: 'Cosine Similarity', def: 'A mathematical match score (0 to 1). 1.0 means two intersections behave identically.' },
  { term: 'Fingerprint', def: 'A 6-dimension "DNA" of an intersection capturing wait time, queue length, and priority.' },
  { term: 'Parallel Coordinates', def: 'A visualization where each line represents an intersection. When lines follow the same "path", they belong to the same cluster.' },
  { term: 'Reward', def: 'The "score" given to the AI. Higher reward means the AI is doing a better job at clearing traffic.' },
  { term: 'Throughput', def: 'The percentage of vehicles that successfully cross an intersection without being stuck.' },
  { term: 'Q-Len', def: 'Queue Length: the number of vehicles currently waiting at a red light.' },
  { term: 'Wait Time', def: 'The average duration (in seconds) a vehicle spends stopped at an intersection.' }
];

export default function AdaptFlowAnalytics() {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [selectedRound, setSelectedRound] = useState(0);
  const [source, setSource] = useState('sim');
  const [showGlossary, setShowGlossary] = useState(false);
  
  // Selection States
  const [selectedTrainSet, setSelectedTrainSet] = useState('india_urban');
  const [selectedSimSession, setSelectedSimSession] = useState('latest');
  const [sessions, setSessions] = useState([]);

  const trainOptions = [
    { id: 'india_rural_pikhuwa', name: 'India Rural (Pikhuwa)' },
    { id: 'china_rural', name: 'China Rural' },
    { id: 'india_urban', name: 'India Urban (Delhi)' },
    { id: 'china_urban', name: 'China Urban' },
  ];

  useEffect(() => {
    fetchSessions();
  }, []);

  useEffect(() => {
    fetchAnalytics(source, selectedTrainSet, selectedSimSession);
    const interval = setInterval(() => fetchAnalytics(source, selectedTrainSet, selectedSimSession), 10000);
    return () => clearInterval(interval);
  }, [source, selectedTrainSet, selectedSimSession]);

  const fetchSessions = async () => {
      try {
          const resp = await axios.get(`${API_BASE}/api/adaptflow/sessions`);
          setSessions(resp.data);
      } catch (err) {
          console.error("Failed to fetch sessions:", err);
      }
  };

  const fetchAnalytics = async (currentSource, trainSet, simId) => {
    try {
      const resp = await axios.get(`${API_BASE}/api/adaptflow/analytics?source=${currentSource}&dataset=${trainSet}&sim_id=${simId}`);
      setData(resp.data);
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

  if (loading) return (
    <div className="flex flex-col items-center justify-center p-20 glass-panel m-4">
        <Cpu className="w-12 h-12 text-indigo-500 animate-spin mb-4" />
        <div className="text-slate-400 font-bold animate-pulse">Synchronizing Intelligence...</div>
    </div>
  );
  
  if (error) return (
    <div className="p-8 text-center text-rose-400 flex flex-col items-center gap-4">
        <div className="p-4 rounded-full bg-rose-500/10 border border-rose-500/20"><Cpu className="w-8 h-8" /></div>
        <div className="font-bold">{error}</div>
        <button onClick={() => window.location.reload()} className="px-4 py-2 bg-slate-800 rounded-lg text-xs hover:bg-slate-700 transition-all font-bold border border-slate-700">Retry</button>
    </div>
  );

  if (!data || data.num_rounds === 0) {
    return (
        <div className="flex flex-col gap-6 p-4">
            <div className="flex flex-wrap items-center justify-between gap-4 glass-panel p-4">
                <div className="flex items-center gap-2 p-1 bg-slate-900/50 rounded-lg border border-slate-800">
                    <button onClick={() => {setSource('sim'); setLoading(true);}} className={`px-4 py-2 rounded-lg text-xs font-bold transition-all ${source === 'sim' ? 'bg-indigo-500 text-white shadow-lg' : 'text-slate-400 hover:text-slate-200'}`}>Testing (Sim)</button>
                    <button onClick={() => {setSource('train'); setLoading(true);}} className={`px-4 py-2 rounded-lg text-xs font-bold transition-all ${source === 'train' ? 'bg-indigo-500 text-white shadow-lg' : 'text-slate-400 hover:text-slate-200'}`}>Training</button>
                </div>
            </div>
            <div className="glass-panel p-20 text-center text-slate-400 flex flex-col items-center justify-center">
                <Cpu className="w-16 h-16 mb-4 opacity-10 animate-pulse" />
                <h3 className="text-xl font-bold text-white mb-2">No Data Available</h3>
                <p className="text-sm text-slate-500">The requested source has no history records yet.</p>
            </div>
        </div>
    );
  }

  // Data Preparation
  const currentFingerprints = data.fingerprints[selectedRound] || {};
  const currentSimilarity = data.similarity_matrices[selectedRound] || [];
  const nodeIds = Object.keys(currentFingerprints);

  const roundMetricsData = Array.from({ length: data.num_rounds }).map((_, i) => ({
    round: i + 1,
    reward: data.reward_history ? data.reward_history[i] : 0,
    queue: data.queue_history ? data.queue_history[i] : 0,
    wait: data.wait_history ? data.wait_history[i] : 0,
    throughput: data.throughput_history ? data.throughput_history[i] : 0,
  }));

  // NEW: Parallel Coordinates Data
  const dimensions = ["Wait Time", "Queue", "Throughput", "Max Queue", "Congested Lanes", "Priority"];
  const parallelData = dimensions.map((dim, dimIdx) => {
    const point = { name: dim };
    nodeIds.forEach(nid => {
      point[nid] = currentFingerprints[nid] ? currentFingerprints[nid][dimIdx] : 0;
    });
    return point;
  });

  const chartColors = ["#6366f1", "#10b981", "#f59e0b", "#ef4444", "#8b5cf6", "#06b6d4", "#ec4899", "#14b8a6"];
  const snapshotPrefix = source === 'sim' ? 'Window' : 'Round';

  return (
    <div className="flex flex-col gap-6 p-4 max-w-[1600px] mx-auto">
      {/* Header & Definitions Sidebar Toggle */}
      <div className="flex flex-col xl:flex-row xl:items-center justify-between gap-4 glass-panel p-4">
        <div className="flex items-center gap-3">
          <div className="p-2 rounded-lg bg-indigo-500/20 text-indigo-400"><Target className="w-5 h-5" /></div>
          <div>
            <h2 className="text-lg font-bold text-white tracking-tight">AdaptFlow Analytics Dashboard</h2>
            <p className="text-[10px] text-slate-400 font-medium italic">Deep dive into 5 core intelligence metrics</p>
          </div>
        </div>

        <div className="flex flex-wrap items-center gap-3">
            <div className="flex items-center gap-1 p-1 bg-slate-900/50 rounded-lg border border-slate-800">
                <button onClick={() => {setSource('sim'); setLoading(true);}} className={`px-4 py-1.5 rounded-md text-[10px] font-bold transition-all ${source === 'sim' ? 'bg-indigo-500 text-white shadow-lg' : 'text-slate-400 hover:text-slate-300'}`}>Testing</button>
                <button onClick={() => {setSource('train'); setLoading(true);}} className={`px-4 py-1.5 rounded-md text-[10px] font-bold transition-all ${source === 'train' ? 'bg-indigo-500 text-white shadow-lg' : 'text-slate-400 hover:text-slate-300'}`}>Training</button>
            </div>

            <select className="bg-slate-950/80 border border-slate-700 text-slate-200 text-[10px] rounded-md px-3 py-1.5 outline-none font-bold" value={source === 'train' ? selectedTrainSet : selectedSimSession} onChange={(e) => { source === 'train' ? setSelectedTrainSet(e.target.value) : setSelectedSimSession(e.target.value); setLoading(true); }}>
                {source === 'train' ? trainOptions.map(opt => <option key={opt.id} value={opt.id}>{opt.name}</option>) : sessions.map(s => <option key={s.id} value={s.id}>{s.city} ({s.timestamp})</option>)}
            </select>

            <div className="flex items-center gap-2">
                <label className="text-[9px] uppercase tracking-wider text-slate-500 font-black">{snapshotPrefix}:</label>
                <select className="bg-slate-950/80 border border-slate-800 text-indigo-400 text-[10px] rounded-md px-2 py-1.5 outline-none font-black" value={selectedRound} onChange={(e) => setSelectedRound(parseInt(e.target.value))}>
                    {Array.from({ length: data.num_rounds }).map((_, i) => (<option key={i} value={i}>{snapshotPrefix} {i + 1}</option>))}
                </select>
            </div>

            <button onClick={() => setShowGlossary(!showGlossary)} className={`flex items-center gap-2 px-3 py-1.5 rounded-md text-[10px] font-bold border transition-all ${showGlossary ? 'bg-indigo-500/20 border-indigo-500 text-indigo-400 shadow-glow' : 'border-slate-800 text-slate-500 hover:text-slate-300'}`}>
                <HelpCircle className="w-3.5 h-3.5" /> Glossary
            </button>
        </div>
      </div>

      {/* Glossary Pane */}
      {showGlossary && (
        <div className="glass-panel p-4 grid grid-cols-1 md:grid-cols-3 xl:grid-cols-5 gap-4 animate-in slide-in-from-top duration-300 border-indigo-500/30">
            {GLOSSARY.map(g => (
                <div key={g.term} className="p-2 rounded bg-slate-900/40 border border-slate-800">
                    <span className="text-[10px] font-bold text-indigo-400 uppercase tracking-tighter">{g.term}</span>
                    <p className="text-[9px] text-slate-500 mt-1 leading-tight">{g.def}</p>
                </div>
            ))}
        </div>
      )}

      {/* Metric Section 1: Clustering & Similarity */}
      <div className="grid grid-cols-1 xl:grid-cols-2 gap-6">
          {/* 1. Parallel Dimensions Fingerprint */}
          <SectionContainer title="1. Clustering Analytics (Parallel Dimensions)" icon={<Shuffle className="w-4 h-4" />} description="Each line represents an intersection. Intersections that follow the same path across these 6 axes belong to the same cluster. This shows the exact 'behavioral signature' of the network.">
              <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={parallelData} margin={{ top: 20, right: 30, left: 0, bottom: 20 }}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" horizontal={true} vertical={true} />
                      <XAxis dataKey="name" stroke="#64748b" fontSize={10} interval={0} tick={{ fill: '#94a3b8' }} />
                      <YAxis domain={[0, 1]} hide />
                      <Tooltip contentStyle={{ backgroundColor: '#0f172a', border: '1px solid #334155', borderRadius: '8px', fontSize: '10px' }} />
                      <Legend iconType="circle" wrapperStyle={{ fontSize: '9px', paddingTop: '10px' }} />
                      {nodeIds.map((nid, i) => (
                          <Line 
                            key={nid} 
                            type="monotone" 
                            dataKey={nid} 
                            stroke={chartColors[i % chartColors.length]} 
                            strokeWidth={selectedRound % 2 === i % 2 ? 3 : 2} 
                            dot={{ r: 3 }} 
                            activeDot={{ r: 6 }}
                            strokeOpacity={0.8}
                          />
                      ))}
                  </LineChart>
              </ResponsiveContainer>
          </SectionContainer>

          {/* 2. Scaled Similarity Matrix (Single Frame, No Scroll) */}
          <SectionContainer title="Cosine Similarity Matrix" icon={<Network className="w-4 h-4" />} description="A global heatmap of coordination. Values near 1.0 (darker) show intersections that share identical traffic patterns and are perfectly synchronized.">
              <div className="h-full flex flex-col items-center justify-center p-2">
                  <div 
                    className="grid gap-px bg-slate-800 border border-slate-700 rounded overflow-hidden w-full max-w-[400px] aspect-square" 
                    style={{ gridTemplateColumns: `repeat(${nodeIds.length + 1}, 1fr)` }}
                  >
                      {/* Top Header Row */}
                      <div className="bg-slate-900 flex items-center justify-center border-b border-r border-slate-800"></div>
                      {nodeIds.map(nid => (
                        <div key={nid} className="bg-slate-900 flex items-center justify-center text-[7px] md:text-[9px] text-slate-500 font-mono border-b border-r border-slate-800 uppercase tracking-tighter">
                            {nid.split('_')[1]}
                        </div>
                      ))}
                      
                      {/* Grid Rows */}
                      {currentSimilarity.map((row, i) => (
                          <React.Fragment key={i}>
                              <div className="bg-slate-900 flex items-center justify-center text-[7px] md:text-[9px] text-slate-500 font-mono border-r border-b border-slate-800 uppercase tracking-tighter">
                                {nodeIds[i].split('_')[1]}
                              </div>
                              {row.map((val, j) => {
                                  // Determine cell color based on similarity score
                                  const cellBg = i === j ? 'rgba(30, 41, 59, 0.5)' : `rgba(79, 70, 229, ${val})`; 
                                  return (
                                    <div 
                                        key={j} 
                                        className="flex items-center justify-center border-r border-b border-slate-800 transition-all hover:scale-110 z-10" 
                                        style={{ backgroundColor: cellBg }}
                                        title={`${nodeIds[i]} vs ${nodeIds[j]}: ${(val * 100).toFixed(1)}%`}
                                    >
                                        <span className={`text-[7px] md:text-[9px] font-black pointer-events-none ${val > 0.6 ? 'text-white' : 'text-indigo-400'}`}>
                                            {i === j ? '—' : (val > 0 ? val.toFixed(2) : '0')}
                                        </span>
                                    </div>
                                  );
                              })}
                          </React.Fragment>
                      ))}
                  </div>
              </div>
          </SectionContainer>
      </div>

      {/* Metric Section 2: Reward */}
      <div className="grid grid-cols-1 xl:grid-cols-4 gap-6">
          <MetricInfoCard title="2. Intelligence Reward" icon={<Cpu className="text-indigo-400" />} value={data.reward_history ? data.reward_history[selectedRound]?.toFixed(1) : '0'} description="Primary mission success metric. Higher reward indicates the AI is clearing traffic bottlenecks more efficiently across all nodes." />
          <div className="xl:col-span-3 glass-panel p-4 h-[300px]">
              <ResponsiveContainer width="100%" height="100%">
                  <AreaChart data={roundMetricsData}>
                      <defs>
                          <linearGradient id="colorReward" x1="0" y1="0" x2="0" y2="1"><stop offset="5%" stopColor="#6366f1" stopOpacity={0.3}/><stop offset="95%" stopColor="#6366f1" stopOpacity={0}/></linearGradient>
                      </defs>
                      <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" vertical={false} />
                      <XAxis dataKey="round" stroke="#475569" fontSize={10} axisLine={false} tickLine={false} label={{ value: snapshotPrefix, position: 'insideBottom', offset: -5, fontSize: 10, fill: '#475569' }} />
                      <YAxis stroke="#475569" fontSize={10} axisLine={false} tickLine={false} />
                      <Tooltip contentStyle={{ backgroundColor: '#0f172a', border: '1px solid #334155', borderRadius: '8px', fontSize: '10px' }} />
                      <Area type="monotone" dataKey="reward" name="Total Reward" stroke="#6366f1" strokeWidth={3} fillOpacity={1} fill="url(#colorReward)" dot={{ r: 4, fill: '#6366f1' }} />
                  </AreaChart>
              </ResponsiveContainer>
          </div>
      </div>

      {/* Metric Section 3: Throughput */}
      <div className="grid grid-cols-1 xl:grid-cols-4 gap-6">
          <MetricInfoCard title="3. Flow Throughput" icon={<Activity className="text-amber-400" />} value={data.throughput_history ? (data.throughput_history[selectedRound] * 100).toFixed(1) + '%' : '0%'} description="Measures velocity efficiency. A 100% throughput means all traffic is processed through the system without creating new bottlenecks." />
          <div className="xl:col-span-3 glass-panel p-4 h-[300px]">
              <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={roundMetricsData}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" vertical={false} />
                      <XAxis dataKey="round" stroke="#475569" fontSize={10} label={{ value: snapshotPrefix, position: 'insideBottom', offset: -5, fontSize: 10, fill: '#475569' }} />
                      <YAxis stroke="#475569" fontSize={10} tickFormatter={(v) => `${(v * 100).toFixed(0)}%`} />
                      <Tooltip contentStyle={{ backgroundColor: '#0f172a', border: '1px solid #334155', borderRadius: '8px', fontSize: '10px' }} />
                      <Line type="monotone" dataKey="throughput" name="Throughput Ratio" stroke="#f59e0b" strokeWidth={3} dot={{ r: 5, fill: '#f59e0b' }} />
                  </LineChart>
              </ResponsiveContainer>
          </div>
      </div>

      {/* Metric Section 4: Queue Length */}
      <div className="grid grid-cols-1 xl:grid-cols-4 gap-6">
          <MetricInfoCard title="4. Persistent Queue (Q-Len)" icon={<BarChart3 className="text-rose-400" />} value={data.queue_history ? data.queue_history[selectedRound]?.toFixed(1) : '0'} description="The number of vehicles caught at a standstill. Reducing Q-Len directly decreases urban CO2 emissions and time spent idling." />
          <div className="xl:col-span-3 glass-panel p-4 h-[300px]">
              <ResponsiveContainer width="100%" height="100%">
                  <AreaChart data={roundMetricsData}>
                      <defs>
                          <linearGradient id="colorQueue" x1="0" y1="0" x2="0" y2="1"><stop offset="5%" stopColor="#ef4444" stopOpacity={0.2}/><stop offset="95%" stopColor="#ef4444" stopOpacity={0}/></linearGradient>
                      </defs>
                      <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" vertical={false} />
                      <XAxis dataKey="round" stroke="#475569" fontSize={10} label={{ value: snapshotPrefix, position: 'insideBottom', offset: -5, fontSize: 10, fill: '#475569' }} />
                      <YAxis stroke="#475569" fontSize={10} />
                      <Tooltip contentStyle={{ backgroundColor: '#0f172a', border: '1px solid #334155', borderRadius: '8px', fontSize: '10px' }} />
                      <Area type="monotone" dataKey="queue" name="Avg Queue Length" stroke="#ef4444" strokeWidth={3} fill="url(#colorQueue)" dot={{ r: 4, fill: '#ef4444' }} />
                  </AreaChart>
              </ResponsiveContainer>
          </div>
      </div>

      {/* Metric Section 5: Waiting Time */}
      <div className="grid grid-cols-1 xl:grid-cols-4 gap-6 mb-10">
          <MetricInfoCard title="5. Cumulative Delay (Wait)" icon={<Clock className="text-emerald-400" />} value={data.wait_history ? data.wait_history[selectedRound]?.toFixed(1) + 's' : '0s'} description="Average time a driver spends waiting per intersection. Reducing this metric is the key to increasing regional productivity." />
          <div className="xl:col-span-3 glass-panel p-4 h-[300px]">
              <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={roundMetricsData}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" vertical={false} />
                      <XAxis dataKey="round" stroke="#475569" fontSize={10} label={{ value: snapshotPrefix, position: 'insideBottom', offset: -5, fontSize: 10, fill: '#475569' }} />
                      <YAxis stroke="#475569" fontSize={10} />
                      <Tooltip contentStyle={{ backgroundColor: '#0f172a', border: '1px solid #334155', borderRadius: '8px', fontSize: '10px' }} />
                      <Line type="monotone" dataKey="wait" name="Wait Time (s)" stroke="#10b981" strokeWidth={3} dot={{ r: 5, fill: '#10b981' }} />
                  </LineChart>
              </ResponsiveContainer>
          </div>
      </div>
    </div>
  );
}

function SectionContainer({ title, icon, description, children }) {
    return (
        <div className="glass-panel p-5 flex flex-col h-[480px]">
            <div className="mb-4">
                <div className="flex items-center gap-2 mb-1">
                    <div className="p-1.5 rounded bg-slate-800 text-slate-400">{icon}</div>
                    <h3 className="text-[12px] font-black text-white uppercase tracking-wider">{title}</h3>
                </div>
                <p className="text-[10px] text-slate-500 leading-relaxed font-medium">{description}</p>
            </div>
            <div className="flex-1 min-h-0">{children}</div>
        </div>
    );
}

function MetricInfoCard({ title, icon, value, description }) {
    return (
        <div className="glass-panel p-6 flex flex-col justify-center gap-3 border-l-4 border-indigo-500 hover:bg-slate-800/40 transition-all group">
            <div className="flex items-center gap-3">
                <div className="p-2 rounded-lg bg-slate-900 border border-slate-800 group-hover:scale-110 transition-transform">{icon}</div>
                <span className="text-[10px] font-black text-slate-500 uppercase tracking-widest leading-none">{title}</span>
            </div>
            <div className="flex flex-col">
                <span className="text-4xl font-black text-white tracking-tighter">{value}</span>
                <p className="text-[10px] text-slate-500 mt-2 font-medium leading-normal italic line-clamp-3">{description}</p>
            </div>
        </div>
    );
}
