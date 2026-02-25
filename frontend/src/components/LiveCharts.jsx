import React from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend } from 'recharts';

const LiveCharts = ({ data }) => {
    return (
        <div className="grid grid-cols-2 gap-6 h-full p-2">
            {/* Queue & Wait Time Chart */}
            <div className="glass-panel p-5 flex flex-col">
                <h3 className="text-sm font-semibold text-slate-300 mb-6 flex items-center gap-2">
                    Intersection Performance
                    <span className="flex h-2 w-2 relative">
                        <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-brand-400 opacity-75"></span>
                        <span className="relative inline-flex rounded-full h-2 w-2 bg-brand-500"></span>
                    </span>
                </h3>
                <div className="flex-1 w-full min-h-[250px]">
                    <ResponsiveContainer width="100%" height="100%">
                        <LineChart data={data} margin={{ top: 5, right: 30, left: 10, bottom: 5 }}>
                            <CartesianGrid strokeDasharray="3 3" stroke="#334155" vertical={false} />
                            <XAxis dataKey="step" stroke="#94a3b8" fontSize={12} tickLine={false} axisLine={false} />
                            <YAxis yAxisId="left" stroke="#94a3b8" fontSize={12} tickLine={false} axisLine={false} />
                            <YAxis yAxisId="right" orientation="right" stroke="#94a3b8" fontSize={12} tickLine={false} axisLine={false} />
                            <Tooltip
                                contentStyle={{ backgroundColor: '#1e293b', border: '1px solid #334155', borderRadius: '0.5rem' }}
                                itemStyle={{ color: '#e2e8f0' }}
                            />
                            <Legend wrapperStyle={{ paddingTop: '20px' }} />
                            <Line yAxisId="left" type="monotone" dataKey="total_queue" stroke="#f43f5e" name="Queue Length" strokeWidth={2} dot={false} isAnimationActive={false} />
                            <Line yAxisId="right" type="monotone" dataKey="avg_wait" stroke="#f59e0b" name="Avg Wait (s)" strokeWidth={2} dot={false} isAnimationActive={false} />
                        </LineChart>
                    </ResponsiveContainer>
                </div>
            </div>

            {/* Reward Chart */}
            <div className="glass-panel p-5 flex flex-col">
                <h3 className="text-sm font-semibold text-slate-300 mb-6">Algorithm Reward</h3>
                <div className="flex-1 w-full min-h-[250px]">
                    <ResponsiveContainer width="100%" height="100%">
                        <LineChart data={data} margin={{ top: 5, right: 30, left: 10, bottom: 5 }}>
                            <CartesianGrid strokeDasharray="3 3" stroke="#334155" vertical={false} />
                            <XAxis dataKey="step" stroke="#94a3b8" fontSize={12} tickLine={false} axisLine={false} />
                            <YAxis domain={[-1.5, 1.5]} stroke="#94a3b8" fontSize={12} tickLine={false} axisLine={false} />
                            <Tooltip
                                contentStyle={{ backgroundColor: '#1e293b', border: '1px solid #334155', borderRadius: '0.5rem' }}
                            />
                            <Line type="stepAfter" dataKey="reward" stroke="#10b981" name="Reward" strokeWidth={2} dot={false} isAnimationActive={false} />
                        </LineChart>
                    </ResponsiveContainer>
                </div>
            </div>
        </div>
    );
};

export default LiveCharts;
