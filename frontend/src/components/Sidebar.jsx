import React from 'react';
import { Play, Square, Activity, Settings } from 'lucide-react';

const Sidebar = ({ config, simConfig, setSimConfig, isRunning, onToggleSimulation }) => {

    const handleChange = (field, value) => {
        setSimConfig(prev => ({ ...prev, [field]: value }));
    };

    const handlePoiToggle = (poi) => {
        setSimConfig(prev => {
            const currentPois = prev.target_pois || [];
            const newPois = currentPois.includes(poi)
                ? currentPois.filter(p => p !== poi)
                : [...currentPois, poi];
            return { ...prev, target_pois: newPois };
        });
    };

    return (
        <div className="w-80 glass-panel h-full p-6 flex flex-col pt-8">
            <div className="flex items-center gap-3 mb-8">
                <div className="bg-brand-500/20 p-2 rounded-lg border border-brand-500/50">
                    <Activity className="w-6 h-6 text-brand-400" />
                </div>
                <div>
                    <h1 className="text-xl font-bold bg-gradient-to-r from-white to-slate-400 bg-clip-text text-transparent">Smart Traffic</h1>
                    <p className="text-xs text-slate-400 font-medium">Control System</p>
                </div>
            </div>

            <div className="flex-1 space-y-6 overflow-y-auto pr-2 custom-scrollbar">

                {/* Environment Selection */}
                <div className="space-y-3">
                    <label className="text-xs font-semibold text-slate-400 uppercase tracking-wider block flex items-center gap-2">
                        <Settings className="w-3 h-3" /> Data Source
                    </label>
                    <div className="bg-dark-900/50 p-1 rounded-lg flex border border-slate-700/50">
                        <button
                            className={`flex-1 py-2 px-3 text-sm rounded-md font-medium transition-all ${simConfig.use_tomtom ? 'bg-dark-700 text-white shadow-sm' : 'text-slate-400 hover:text-slate-300'}`}
                            onClick={() => handleChange('use_tomtom', true)}
                            disabled={isRunning}
                        >
                            Real-Time
                        </button>
                        <button
                            className={`flex-1 py-2 px-3 text-sm rounded-md font-medium transition-all ${!simConfig.use_tomtom ? 'bg-dark-700 text-white shadow-sm' : 'text-slate-400 hover:text-slate-300'}`}
                            onClick={() => handleChange('use_tomtom', false)}
                            disabled={isRunning}
                        >
                            Mock
                        </button>
                    </div>
                </div>

                {/* Algorithm Selection */}
                <div className="space-y-3">
                    <label className="text-xs font-semibold text-slate-400 uppercase tracking-wider block">Algorithm</label>
                    <select
                        value={simConfig.algorithm}
                        onChange={(e) => handleChange('algorithm', e.target.value)}
                        disabled={isRunning}
                        className="w-full bg-dark-900 border border-slate-700 rounded-lg px-4 py-2.5 text-sm text-slate-200 outline-none focus:border-brand-500/50 transition-colors disabled:opacity-50"
                    >
                        {config.algorithms?.map(algo => (
                            <option key={algo} value={algo}>{algo}</option>
                        ))}
                    </select>
                </div>

                {/* City Selection */}
                <div className="space-y-3">
                    <label className="text-xs font-semibold text-slate-400 uppercase tracking-wider block">Target City</label>
                    <select
                        value={simConfig.city}
                        onChange={(e) => handleChange('city', e.target.value)}
                        disabled={isRunning}
                        className="w-full bg-dark-900 border border-slate-700 rounded-lg px-4 py-2.5 text-sm text-slate-200 outline-none focus:border-brand-500/50 transition-colors disabled:opacity-50"
                    >
                        {config.cities?.map(city => (
                            <option key={city} value={city}>{city}</option>
                        ))}
                    </select>
                </div>

                {/* POI Targeting */}
                <div className="space-y-3">
                    <label className="text-xs font-semibold text-slate-400 uppercase tracking-wider block">Target POIs</label>
                    <div className="flex flex-wrap gap-2">
                        {config.poi_categories?.map(poi => {
                            const isSelected = simConfig.target_pois.includes(poi);
                            return (
                                <button
                                    key={poi}
                                    disabled={isRunning}
                                    onClick={() => handlePoiToggle(poi)}
                                    className={`text-xs px-3 py-1.5 rounded-full border transition-all ${isSelected
                                            ? 'bg-brand-500/20 border-brand-500/50 text-brand-400'
                                            : 'bg-dark-900/50 border-slate-700/50 text-slate-400 hover:border-slate-600 disabled:opacity-50'
                                        }`}
                                >
                                    {poi}
                                </button>
                            );
                        })}
                    </div>
                </div>
            </div>

            <div className="pt-6 mt-4 border-t border-slate-700/50">
                <button
                    onClick={onToggleSimulation}
                    className={`w-full py-3 px-4 rounded-lg flex items-center justify-center gap-2 font-medium transition-all ${isRunning
                            ? 'bg-rose-500/20 text-rose-400 border border-rose-500/50 hover:bg-rose-500/30'
                            : 'bg-brand-500 text-white shadow-lg shadow-brand-500/25 hover:bg-brand-400'
                        }`}
                >
                    {isRunning ? (
                        <><Square className="w-4 h-4 fill-current" /> Stop Simulation</>
                    ) : (
                        <><Play className="w-4 h-4 fill-current" /> Start Simulation</>
                    )}
                </button>
            </div>
        </div>
    );
};

export default Sidebar;
