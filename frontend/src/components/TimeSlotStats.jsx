import React, { useState, useEffect } from 'react';
import { Clock, BarChart3, TrendingUp, Info, Calendar, ChevronLeft, ChevronRight } from 'lucide-react';

const API_BASE = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';

const TimeSlotStats = ({ city = 'Delhi' }) => {
  const [stats, setStats] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [selectedDate, setSelectedDate] = useState(new Date().toISOString().split('T')[0]);

  const fetchStats = async () => {
    setLoading(true);
    try {
      const response = await fetch(`${API_BASE}/api/time-slot-stats?location_id=${city}&date=${selectedDate}`);
      const data = await response.json();
      if (data.status === 'success') {
        setStats(data.data);
      } else {
        setError(data.message);
      }
    } catch (err) {
      setError('Failed to fetch historical stats');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchStats();
    // Refresh every 10 minutes if viewing today's date
    const isToday = selectedDate === new Date().toISOString().split('T')[0];
    let interval;
    if (isToday) {
      interval = setInterval(fetchStats, 600000);
    }
    return () => {
      if (interval) clearInterval(interval);
    };
  }, [city, selectedDate]);

  const handleDateChange = (e) => {
    setSelectedDate(e.target.value);
  };

  const shiftDate = (days) => {
    const d = new Date(selectedDate);
    d.setDate(d.getDate() + days);
    setSelectedDate(d.toISOString().split('T')[0]);
  };

  if (loading && !stats) {
    return (
      <div className="glass-panel p-6 animate-pulse flex flex-col items-center justify-center min-h-[200px]">
        <Clock className="w-8 h-8 text-slate-700 mb-2" />
        <p className="text-slate-500 text-sm">Loading historical traffic slots...</p>
      </div>
    );
  }

  if (error) {
    return (
      <div className="glass-panel p-6 border border-red-500/30 bg-red-500/5">
        <p className="text-red-400 text-sm flex items-center gap-2">
          <Info className="w-4 h-4" /> Error: {error}
        </p>
        <button 
          onClick={() => { setError(null); fetchStats(); }}
          className="mt-2 text-xs text-indigo-400 hover:underline"
        >
          Try Again
        </button>
      </div>
    );
  }

  const getLevelColor = (level) => {
    switch (level) {
      case 'Low': return 'text-emerald-400 bg-emerald-400/10 shadow-[0_0_10px_rgba(52,211,153,0.1)]';
      case 'Moderate': return 'text-amber-400 bg-amber-400/10 shadow-[0_0_10px_rgba(251,191,36,0.1)]';
      case 'High': return 'text-orange-400 bg-orange-400/10 shadow-[0_0_10px_rgba(251,146,60,0.1)]';
      case 'Severe': return 'text-rose-400 bg-rose-400/10 shadow-[0_0_10px_rgba(251,113,133,0.1)]';
      default: return 'text-slate-400 bg-slate-400/10';
    }
  };

  return (
    <div className="mt-6 flex flex-col gap-4">
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 px-2">
        <h2 className="text-lg font-bold flex items-center gap-2">
          <Clock className="w-5 h-5 text-indigo-400" />
          Time-Slot Traffic Analysis
        </h2>
        
        <div className="flex items-center gap-3 bg-slate-900/50 p-1 rounded-lg border border-slate-800">
          <button 
            onClick={() => shiftDate(-1)}
            className="p-1 hover:bg-slate-800 rounded transition-colors"
            title="Previous Day"
          >
            <ChevronLeft className="w-4 h-4 text-slate-400" />
          </button>
          
          <div className="flex items-center gap-2 text-xs font-medium text-slate-300">
            <Calendar className="w-3.5 h-3.5 text-indigo-400" />
            <input 
              type="date" 
              value={selectedDate}
              onChange={handleDateChange}
              max={new Date().toISOString().split('T')[0]}
              className="bg-transparent border-none outline-none cursor-pointer hover:text-white transition-colors"
            />
          </div>

          <button 
            onClick={() => shiftDate(1)}
            disabled={selectedDate === new Date().toISOString().split('T')[0]}
            className="p-1 hover:bg-slate-800 rounded transition-colors disabled:opacity-30 disabled:cursor-not-allowed"
            title="Next Day"
          >
            <ChevronRight className="w-4 h-4 text-slate-400" />
          </button>

          <div className="w-px h-4 bg-slate-800 mx-1" />

          <button 
            onClick={fetchStats}
            className="p-1.5 hover:bg-slate-800 rounded transition-colors"
            title="Refresh Data"
          >
            <TrendingUp className="w-4 h-4 text-indigo-400" />
          </button>
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-5 gap-3">
        {stats?.slots.map((slot, index) => (
          <div key={index} className="glass-panel p-4 flex flex-col gap-3 group hover:border-indigo-500/50 transition-all duration-300">
            <div className="flex flex-col gap-1">
              <span className="text-slate-400 text-[10px] uppercase tracking-wider font-semibold">{slot.time_range}</span>
              <span className="text-sm font-bold group-hover:text-indigo-300 transition-colors uppercase">{slot.slot_name}</span>
            </div>

            {slot.record_count > 0 ? (
              <>
                <div className={`text-[10px] font-bold px-2 py-1 rounded-md text-center inline-block w-fit ${getLevelColor(slot.metrics.congestion_level)}`}>
                  {slot.metrics.congestion_level} Traffic
                </div>

                <div className="space-y-2 mt-1">
                  <div className="flex justify-between items-end">
                    <span className="text-slate-500 text-[10px]">Avg Speed</span>
                    <span className="text-xs font-mono text-indigo-300 font-bold">{slot.metrics.avg_speed_kmh} <span className="text-[8px] font-normal opacity-70">km/h</span></span>
                  </div>
                  <div className="flex justify-between items-end">
                    <span className="text-slate-500 text-[10px]">Density Est.</span>
                    <span className="text-xs font-mono text-emerald-300 font-bold">{slot.metrics.estimated_density_veh_km} <span className="text-[8px] font-normal opacity-70">v/km</span></span>
                  </div>
                </div>
              </>
            ) : (
              <div className="mt-auto py-2 flex flex-col items-center justify-center border border-dashed border-slate-800 rounded-lg">
                <BarChart3 className="w-5 h-5 text-slate-800 mb-1" />
                <span className="text-[10px] text-slate-600">No Data</span>
              </div>
            )}
          </div>
        ))}
      </div>
      
      <p className="text-[10px] text-slate-600 text-center italic mt-1 flex items-center justify-center gap-1">
        <Info className="w-3 h-3" /> Historical metrics collected daily for monitoring.
      </p>
    </div>
  );
};

export default TimeSlotStats;
