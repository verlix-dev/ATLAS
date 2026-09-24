import { useMemo } from 'react';
import {
  CartesianGrid, Line, LineChart, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis,
} from 'recharts';
import type { Experiment, Objective } from '../types';
import { formatMetric, formatSigned, isUnavailable } from '../lib/reduce';

interface Props {
  experiments: Experiment[];
  objective: Objective | null;
  currentId: number | null;
}

interface Point {
  id: number;
  label: string;
  score: number;
  model: string;
  params: Record<string, string | number | boolean | null>;
  delta: number;
  isBest: boolean;
  isCurrent: boolean;
  isRegression: boolean;
}

function CustomTooltip({
  active,
  payload,
  metricLabel,
}: {
  active?: boolean;
  payload?: { payload?: Point }[];
  metricLabel: string;
}) {
  if (!active || !payload?.[0]?.payload) return null;
  const d = payload[0].payload;
  return (
    <div className="card-elevated px-4 py-3 min-w-[190px]" style={{ boxShadow: '0 8px 32px rgba(79,106,247,0.15)' }}>
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
      <div className="text-2xl font-extrabold text-gray-900 mb-1 font-mono">{d.score.toFixed(4)}</div>
      <div className="text-xs text-gray-400 mb-2">{metricLabel}</div>
      {d.delta !== 0 && (
        <div className={`text-xs font-bold ${d.delta > 0 ? 'text-emerald-600' : 'text-orange-500'}`}>
          Δ best {formatSigned(d.delta)}
        </div>
      )}
      <div className="mt-2 pt-2 border-t border-indigo-50 space-y-0.5">
        {Object.entries(d.params).slice(0, 4).map(([k, v]) => (
          <div key={k} className="text-[10px] font-mono text-gray-400">
            {k}: {String(v)}
          </div>
        ))}
      </div>
    </div>
  );
}

interface DotProps {
  cx?: number;
  cy?: number;
  payload?: Point;
}

function CustomDot({ cx = 0, cy = 0, payload }: DotProps) {
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
      <g>
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

export default function ScoreChart({ experiments, objective, currentId }: Props) {
  const metric = objective?.primary_metric ?? null;
  const minimize = objective?.direction === 'minimize';
  const metricLabel = metric ? metric.replace(/_/g, ' ').toUpperCase() : 'SCORE';

  /**
   * Only genuinely measured, rankable scores become points.
   *
   * A failed experiment, or one whose primary metric is UNAVAILABLE, is skipped
   * entirely — plotting it as 0 would invent a measurement the engine never
   * made, and would drag the axis to a false floor.
   */
  const data = useMemo<Point[]>(() => {
    if (!metric) return [];
    const points: Point[] = [];
    let best: number | null = null;
    let bestIndex = -1;
    let previous: number | null = null;

    for (const exp of experiments) {
      if (!exp.result?.ok) continue;
      const raw = exp.result.metrics?.[metric];
      if (isUnavailable(raw)) continue;
      const score = Number(raw);
      if (!Number.isFinite(score)) continue;

      const delta = best === null ? score : minimize ? best - score : score - best;
      const beatsBest = best === null || (minimize ? score < best : score > best);
      points.push({
        id: exp.id,
        label: `E${String(exp.id).padStart(2, '0')}`,
        score,
        model: exp.config?.model ?? '—',
        params: exp.config?.params ?? {},
        delta,
        isBest: false,
        isCurrent: exp.id === currentId,
        isRegression:
          previous !== null && (minimize ? score > previous : score < previous),
      });
      if (beatsBest) {
        best = score;
        bestIndex = points.length - 1;
      }
      previous = score;
    }

    // Remember which point won rather than searching back for one whose score
    // equals `best`: on a tie a value search would be ambiguous, and it would
    // rest on float equality. Strict `beatsBest` keeps the EARLIEST of tied
    // scores, which is what the engine's own min()/max() over the history does.
    if (bestIndex >= 0) points[bestIndex].isBest = true;
    return points;
  }, [experiments, metric, minimize, currentId]);

  if (!metric) {
    return (
      <div className="flex items-center justify-center h-[260px] text-sm text-gray-400">
        The objective has not been reported yet.
      </div>
    );
  }

  if (data.length === 0) {
    return (
      <div className="flex items-center justify-center h-[260px] text-sm text-gray-400">
        No experiment has produced a measured {metricLabel} score yet.
      </div>
    );
  }

  const scores = data.map((d) => d.score);
  const lo = Math.min(...scores);
  const hi = Math.max(...scores);
  const pad = (hi - lo) * 0.2 || 0.02;
  const bestLine = minimize ? lo : hi;

  return (
    <div>
      <div className="flex items-center gap-5 mb-4 text-xs text-gray-500">
        {[
          { color: '#10B981', label: 'Improvement' },
          { color: '#4F6AF7', label: 'Current' },
          { color: '#7C3AED', label: 'Best' },
          { color: '#F97316', label: 'Regression' },
        ].map(({ color, label }) => (
          <div key={label} className="flex items-center gap-1.5">
            <div className="w-2.5 h-2.5 rounded-full" style={{ background: color }} />
            <span>{label}</span>
          </div>
        ))}
      </div>

      <ResponsiveContainer width="100%" height={260}>
        <LineChart data={data} margin={{ top: 8, right: 64, bottom: 0, left: 0 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="rgba(99,102,241,0.08)" vertical={false} />
          <XAxis
            dataKey="label"
            tick={{ fontSize: 11, fontFamily: 'JetBrains Mono', fill: '#9CA3AF' }}
            axisLine={false}
            tickLine={false}
          />
          <YAxis
            domain={[lo - pad, hi + pad]}
            tick={{ fontSize: 11, fontFamily: 'JetBrains Mono', fill: '#9CA3AF' }}
            axisLine={false}
            tickLine={false}
            width={56}
            tickFormatter={(v) => Number(v).toFixed(3)}
          />
          <Tooltip
            content={<CustomTooltip metricLabel={metricLabel} />}
            cursor={{ stroke: 'rgba(99,102,241,0.2)', strokeWidth: 1, strokeDasharray: '4 2' }}
          />
          <ReferenceLine
            y={bestLine}
            stroke="rgba(124,58,237,0.25)"
            strokeDasharray="6 3"
            label={{
              value: `best ${formatMetric(bestLine)}`,
              position: 'right',
              fontSize: 10,
              fill: '#7C3AED',
              fontFamily: 'JetBrains Mono',
            }}
          />
          <Line
            type="monotone"
            dataKey="score"
            stroke="url(#scoreGradient)"
            strokeWidth={2.5}
            dot={<CustomDot />}
            activeDot={false}
            isAnimationActive={true}
            animationDuration={600}
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
