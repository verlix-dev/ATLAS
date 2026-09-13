import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip,
  Legend, ResponsiveContainer,
} from 'recharts';
import type { Experiment } from '../types';

interface Props {
  deterministic: Experiment[];
  aiGuided: Experiment[];
}

interface TooltipPayload {
  color?: string;
  name?: string;
  value?: number;
}

function CustomTooltip({ active, payload, label }: { active?: boolean; payload?: TooltipPayload[]; label?: string }) {
  if (!active || !payload?.length) return null;
  return (
    <div className="card px-4 py-3" style={{ boxShadow: '0 8px 24px rgba(79,106,247,0.12)' }}>
      <div className="font-mono text-xs font-bold text-gray-500 mb-2">{label}</div>
      {payload.map(p => (
        <div key={p.name} className="flex items-center gap-2 text-sm mb-1">
          <div className="w-2 h-2 rounded-full" style={{ background: p.color }} />
          <span className="text-gray-600 text-xs">{p.name}</span>
          <span className="font-bold text-gray-900 ml-auto font-mono">{p.value?.toFixed(4)}</span>
        </div>
      ))}
    </div>
  );
}

export default function ComparisonChart({ deterministic, aiGuided }: Props) {
  const maxLen = Math.max(deterministic.length, aiGuided.length);
  const data = Array.from({ length: maxLen }, (_, i) => ({
    label: `E${String(i + 1).padStart(2, '0')}`,
    Deterministic: deterministic[i]?.metrics?.f1_macro ?? null,
    'AI Guided': aiGuided[i]?.metrics?.f1_macro ?? null,
  }));

  const allScores = [...deterministic, ...aiGuided].map(e => e.metrics?.f1_macro ?? 0).filter(Boolean);
  const minScore = Math.max(0.4, Math.min(...allScores) - 0.04);
  const maxScore = Math.min(1, Math.max(...allScores) + 0.04);

  return (
    <ResponsiveContainer width="100%" height={280}>
      <LineChart data={data} margin={{ top: 8, right: 24, bottom: 0, left: 0 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="rgba(99,102,241,0.08)" vertical={false} />
        <XAxis
          dataKey="label"
          tick={{ fontSize: 11, fontFamily: 'JetBrains Mono', fill: '#9CA3AF' }}
          axisLine={false}
          tickLine={false}
        />
        <YAxis
          domain={[minScore, maxScore]}
          tick={{ fontSize: 11, fontFamily: 'JetBrains Mono', fill: '#9CA3AF' }}
          axisLine={false}
          tickLine={false}
          width={52}
          tickFormatter={v => v.toFixed(3)}
        />
        <Tooltip content={<CustomTooltip />} cursor={{ stroke: 'rgba(99,102,241,0.15)', strokeDasharray: '4 2' }} />
        <Legend
          wrapperStyle={{ fontSize: '12px', paddingTop: '12px' }}
          formatter={(value) => <span style={{ color: '#6B7280', fontWeight: 600 }}>{value}</span>}
        />
        <Line
          type="monotone"
          dataKey="Deterministic"
          stroke="#4F6AF7"
          strokeWidth={2.5}
          dot={{ r: 4, fill: '#4F6AF7', stroke: 'white', strokeWidth: 2 }}
          connectNulls={false}
        />
        <Line
          type="monotone"
          dataKey="AI Guided"
          stroke="#7C3AED"
          strokeWidth={2.5}
          strokeDasharray="6 3"
          dot={{ r: 4, fill: '#7C3AED', stroke: 'white', strokeWidth: 2 }}
          connectNulls={false}
        />
        <defs>
          <linearGradient id="fillDeterministic" x1="0" y1="0" x2="0" y2="1">
            <stop offset="5%" stopColor="#4F6AF7" stopOpacity={0.15} />
            <stop offset="95%" stopColor="#4F6AF7" stopOpacity={0} />
          </linearGradient>
        </defs>
      </LineChart>
    </ResponsiveContainer>
  );
}
