import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend } from 'recharts';

// Dynamic color generator for intersections
const getIntersectionColor = (index, total = 8) => {
  // Use HSL to rotate through the hue spectrum
  const count = Math.max(total, 8);
  const hue = (index * (360 / count)) % 360;
  return `hsl(${hue}, 70%, 60%)`;
};

const GLOBAL_COLOR = '#a78bfa'; // Purple for global average

export default function LiveCharts({ simData, intersections }) {
  if (!simData || simData.length === 0) {
    return (
      <div className="glass-panel p-6 rounded-xl text-center text-slate-500 text-sm">
        Start a simulation to see live charts
      </div>
    );
  }

  // Transform simData for charts: extract per-node queue and reward + global
  const chartData = simData.map((d) => {
    const point = { step: d.step };
    
    // Per-intersection data
    if (d.intersections) {
      Object.entries(d.intersections).forEach(([nid, data]) => {
        point[`${nid}_queue`] = data.total_queue;
        point[`${nid}_reward`] = data.reward;
        point[`${nid}_name`] = data.name;
      });
    }
    
    // Global aggregates
    if (d.global) {
      point['global_queue'] = d.global.total_queue;
      point['global_reward'] = d.global.avg_reward;
    }
    
    return point;
  });

  // Get node IDs from the latest data point
  const latestIntersections = simData[simData.length - 1]?.intersections || {};
  const nodeIds = Object.keys(latestIntersections);

  return (
    <div className="flex flex-col gap-4">
      {/* Queue Length Chart */}
      <div className="glass-panel p-4 rounded-xl">
        <h3 className="text-sm font-semibold text-slate-300 mb-3">Queue Length per Intersection</h3>
        <ResponsiveContainer width="100%" height={250}>
          <LineChart data={chartData}>
            <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.06)" />
            <XAxis dataKey="step" stroke="#475569" tick={{ fontSize: 11 }} />
            <YAxis stroke="#475569" tick={{ fontSize: 11 }} />
            <Tooltip
              contentStyle={{ background: '#1e293b', border: '1px solid rgba(255,255,255,0.1)', borderRadius: '8px', fontSize: '12px' }}
              labelStyle={{ color: '#94a3b8' }}
            />
            <Legend wrapperStyle={{ fontSize: '12px' }} />
            {nodeIds.map((nid, index) => (
              <Line
                key={nid}
                type="monotone"
                dataKey={`${nid}_queue`}
                name={latestIntersections[nid]?.name || nid}
                stroke={getIntersectionColor(index, nodeIds.length)}
                strokeWidth={2}
                dot={false}
                isAnimationActive={false}
              />
            ))}
            <Line
              type="monotone"
              dataKey="global_queue"
              name="Global Total"
              stroke={GLOBAL_COLOR}
              strokeWidth={2}
              strokeDasharray="5 5"
              dot={false}
              isAnimationActive={false}
            />
          </LineChart>
        </ResponsiveContainer>
      </div>

      {/* Reward Chart */}
      <div className="glass-panel p-4 rounded-xl">
        <h3 className="text-sm font-semibold text-slate-300 mb-3">Reward per Intersection</h3>
        <ResponsiveContainer width="100%" height={250}>
          <LineChart data={chartData}>
            <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.06)" />
            <XAxis dataKey="step" stroke="#475569" tick={{ fontSize: 11 }} />
            <YAxis stroke="#475569" tick={{ fontSize: 11 }} />
            <Tooltip
              contentStyle={{ background: '#1e293b', border: '1px solid rgba(255,255,255,0.1)', borderRadius: '8px', fontSize: '12px' }}
              labelStyle={{ color: '#94a3b8' }}
            />
            <Legend wrapperStyle={{ fontSize: '12px' }} />
            {nodeIds.map((nid, index) => (
              <Line
                key={nid}
                type="stepAfter"
                dataKey={`${nid}_reward`}
                name={latestIntersections[nid]?.name || nid}
                stroke={getIntersectionColor(index, nodeIds.length)}
                strokeWidth={2}
                dot={false}
                isAnimationActive={false}
              />
            ))}
            <Line
              type="stepAfter"
              dataKey="global_reward"
              name="Avg Reward"
              stroke={GLOBAL_COLOR}
              strokeWidth={2}
              strokeDasharray="5 5"
              dot={false}
              isAnimationActive={false}
            />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
