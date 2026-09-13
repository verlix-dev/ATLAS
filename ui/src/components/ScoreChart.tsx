import { useState } from 'react';
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip,
  ReferenceLine, ResponsiveContainer, Dot,
} from 'recharts';
import type { Experiment } from '../types';

interface Props {
  experiments: Experiment[];
  currentIndex: number;
}

interface TooltipPayload {
  payload?: {
    id: number;
    model: string;
    score: number;
    delta: number;
    isBest: boolean;
    isRegression: boolean;
    params?: Record<string, string | number>;
  };
}

function CustomTooltip({ active, payload }: { active?: boolean; payload?: TooltipPayload[] }) {
  if (!active || !payload?.[0]?.payload) return null;
  const d = payload[0].payload;
  return (
    <div className="card-elevated px-4 py-3 min-w-[180px]" style={{ boxShadow: '0 8px 32px rgba(79,106,247,0.15)' }}>
      <div className="flex items-center justify-between mb-2">
        <span className="font-mono text-xs font-bold text-indigo-500">E{String(d.id).padStart(2, '0')}</span>
        {d.isBest && (
          <span className="text-[10px] bg-violet-100 text-violet-600 font-bold px-2 py-0.5 rounded-full">BEST</span>
        )}
        {d.isRegression && !d.isBest && (
          <span className="text-[10px] bg-orange-100 text-orange-600 font-bold px-2 py-0.5 rounded-full">↓ REGRESSION</span>
        )}
      </div>
      <div className="text-xs text-gray-500 mb-2 font-medium">{d.model}</div>
      <div className="text-2xl font-extrabold text-gray-900 mb-1">{d.score.toFixed(4)}</div>
      <div className="text-xs text-gray-400 mb-2">F1 Macro</div>
      {d.delta !== 0 && (
        <div className={`text-xs font-bold ${d.delta > 0 ? 'text-emerald-600' : 'text-orange-500'}`}>
          Δ Best {d.delta > 0 ? '+' : ''}{d.delta.toFixed(4)}
        </div>
      )}
    </div>
  );
}

interface DotProps {
  cx?: number;
  cy?: number;
  payload?: {
    id: number;
    isBest: boolean;
    isCurrent: boolean;
    isRegression: boolean;
  };
}

function CustomDot(props: DotProps) {
  const { cx = 0, cy = 0, payload } = props;
  if (!payload) return null;
  const { isBest, isCurrent, isRegression } = payload;

  if (isCurrent) {
    return (
      <g>
        <circle cx={cx} cy={cy} r={10} fill="rgba(79,106,247,0.15)" />
        <circle cx={cx} cy={cy} r={6} fill="#4F6AF7" stroke="white" strokeWidth={2} />
      </g>
    );
  }
  if (isBest) {
    return (
      <g className="experiment-point-best">
        <circle cx={cx} cy={cy} r={8} fill="rgba(124,58,237,0.2)" />
        <circle cx={cx} cy={cy} r={5} fill="#7C3AED" stroke="white" strokeWidth={2} />
      </g>
    );
  }
  if (isRegression) {
    return <circle cx={cx} cy={cy} r={4} fill="#F97316" stroke="white" strokeWidth={1.5} />;
  }
  return <circle cx={cx} cy={cy} r={4} fill="#10B981" stroke="white" strokeWidth={1.5} />;
}

export default function ScoreChart({ experiments, currentIndex }: Props) {
  const [hoveredIndex, setHoveredIndex] = useState<number | null>(null);
  const visible = experiments.filter(e => e.metrics);

  let bestScore = 0;
  const data = visible.map((exp, i) => {
    const score = exp.metrics!.f1_macro;
    const isBest = score > bestScore;
    if (isBest) bestScore = score;
    const prevBest = i === 0 ? score : visible.slice(0, i).reduce((b, e) => Math.max(b, e.metrics!.f1_macro), 0);
    return {
      id: exp.id,
      label: `E${String(exp.id).padStart(2, '0')}`,
      score,
      model: exp.model,
      params: exp.params,
      delta: i === 0 ? 0 : score - prevBest,
      isBest: score === Math.max(...visible.map(e => e.metrics!.f1_macro)),
      isCurrent: exp.id === experiments[currentIndex]?.id,
      isRegression: i > 0 && score < visible[i - 1].metrics!.f1_macro,
    };
  });

  const minScore = Math.max(0.4, Math.min(...data.map(d => d.score)) - 0.05);
  const maxScore = Math.min(1, Math.max(...data.map(d => d.score)) + 0.05);
  const bestLine = Math.max(...data.map(d => d.score));

  return (
    <div>
      {/* Legend */}
      <div className="flex items-center gap-5 mb-4 text-xs text-gray-500">
        {[
          { color: '#10B981', label: 'Improvement', shape: 'circle' },
          { color: '#4F6AF7', label: 'Current', shape: 'circle' },
          { color: '#7C3AED', label: 'Best', shape: 'circle' },
          { color: '#F97316', label: 'Regression', shape: 'circle' },
        ].map(({ color, label }) => (
          <div key={label} className="flex items-center gap-1.5">
            <div className="w-2.5 h-2.5 rounded-full" style={{ background: color }} />
            <span>{label}</span>
          </div>
        ))}
      </div>

      <ResponsiveContainer width="100%" height={260}>
        <LineChart data={data} margin={{ top: 8, right: 16, bottom: 0, left: 0 }}
          onMouseMove={(state) => {
            if (state.activeTooltipIndex !== undefined) setHoveredIndex(state.activeTooltipIndex);
          }}
          onMouseLeave={() => setHoveredIndex(null)}
        >
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
          <Tooltip content={<CustomTooltip />} cursor={{ stroke: 'rgba(99,102,241,0.2)', strokeWidth: 1, strokeDasharray: '4 2' }} />
          <ReferenceLine
            y={bestLine}
            stroke="rgba(124,58,237,0.25)"
            strokeDasharray="6 3"
            label={{ value: `Best ${bestLine.toFixed(4)}`, position: 'right', fontSize: 10, fill: '#7C3AED', fontFamily: 'JetBrains Mono' }}
          />
          <Line
            type="monotone"
            dataKey="score"
            stroke="url(#scoreGradient)"
            strokeWidth={2.5}
            dot={<CustomDot />}
            activeDot={false}
            isAnimationActive={true}
            animationDuration={800}
            animationEasing="ease-out"
          />
          <defs>
            <linearGradient id="scoreGradient" x1="0" y1="0" x2="1" y2="0">
              <stop offset="0%" stopColor="#4F6AF7" />
              <stop offset="100%" stopColor="#7C3AED" />
            </linearGradient>
          </defs>
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}
