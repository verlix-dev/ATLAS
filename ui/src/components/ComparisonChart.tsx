import {
  CartesianGrid, Legend, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis,
} from 'recharts';

/** One scored experiment, keyed by its real experiment id. */
export interface SeriesPoint {
  id: number;
  score: number | null;
}

export interface Series {
  label: string;
  points: SeriesPoint[];
}

interface Props {
  deterministic: Series;
  guided: Series;
  /** Objective metric name, for the tooltip and axis labelling. */
  metricLabel: string;
}

interface Row {
  label: string;
  [key: string]: number | string | null;
}

interface TooltipItem {
  color?: string;
  name?: string;
  value?: number | null;
}

function CustomTooltip({
  active,
  payload,
  label,
  metricLabel,
}: {
  active?: boolean;
  payload?: TooltipItem[];
  label?: string;
  metricLabel: string;
}) {
  if (!active || !payload?.length) return null;
  const shown = payload.filter((p) => p.value !== null && p.value !== undefined);
  return (
    <div className="card px-4 py-3" style={{ boxShadow: '0 8px 24px rgba(79,106,247,0.12)' }}>
      <div className="font-mono text-xs font-bold text-gray-500 mb-2">{label}</div>
      {shown.length === 0 ? (
        <div className="text-xs text-gray-400">no measured {metricLabel} at this step</div>
      ) : (
        shown.map((p) => (
          <div key={p.name} className="flex items-center gap-2 text-sm mb-1">
            <div className="w-2 h-2 rounded-full" style={{ background: p.color }} />
            <span className="text-gray-600 text-xs">{p.name}</span>
            <span className="font-bold text-gray-900 ml-auto font-mono">
              {Number(p.value).toFixed(4)}
            </span>
          </div>
        ))
      )}
    </div>
  );
}

/**
 * Score trajectories for both strategies.
 *
 * The two strategies can run different numbers of experiments — the
 * deterministic ladder may exhaust before its budget while an LLM-guided run
 * keeps proposing. Rows are therefore built by experiment id across the union,
 * and a missing score is `null`, never 0. `connectNulls={false}` keeps the
 * lines honest: a gap stays a gap rather than being drawn through.
 */
export default function ComparisonChart({ deterministic, guided, metricLabel }: Props) {
  const allIds = Array.from(
    new Set([
      ...deterministic.points.map((p) => p.id),
      ...guided.points.map((p) => p.id),
    ]),
  ).sort((a, b) => a - b);

  const scoreFor = (series: Series, id: number): number | null =>
    series.points.find((p) => p.id === id)?.score ?? null;

  const data: Row[] = allIds.map((id) => ({
    label: `E${String(id).padStart(2, '0')}`,
    [deterministic.label]: scoreFor(deterministic, id),
    [guided.label]: scoreFor(guided, id),
  }));

  // Only genuinely measured scores define the axis.
  const measured = [
    ...deterministic.points.map((p) => p.score),
    ...guided.points.map((p) => p.score),
  ].filter((s): s is number => s !== null);

  if (measured.length === 0) {
    return (
      <div className="flex items-center justify-center h-[280px] text-sm text-gray-400">
        Neither strategy produced a measured {metricLabel} score.
      </div>
    );
  }

  const lo = Math.min(...measured);
  const hi = Math.max(...measured);
  const pad = (hi - lo) * 0.2 || 0.02;

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
          domain={[lo - pad, hi + pad]}
          tick={{ fontSize: 11, fontFamily: 'JetBrains Mono', fill: '#9CA3AF' }}
          axisLine={false}
          tickLine={false}
          width={56}
          tickFormatter={(v) => Number(v).toFixed(3)}
        />
        <Tooltip
          content={<CustomTooltip metricLabel={metricLabel} />}
          cursor={{ stroke: 'rgba(99,102,241,0.15)', strokeDasharray: '4 2' }}
        />
        <Legend
          wrapperStyle={{ fontSize: '12px', paddingTop: '12px' }}
          formatter={(value) => (
            <span style={{ color: '#6B7280', fontWeight: 600 }}>{value}</span>
          )}
        />
        <Line
          type="monotone"
          dataKey={deterministic.label}
          stroke="#4F6AF7"
          strokeWidth={2.5}
          dot={{ r: 4, fill: '#4F6AF7', stroke: 'white', strokeWidth: 2 }}
          connectNulls={false}
        />
        <Line
          type="monotone"
          dataKey={guided.label}
          stroke="#7C3AED"
          strokeWidth={2.5}
          strokeDasharray="6 3"
          dot={{ r: 4, fill: '#7C3AED', stroke: 'white', strokeWidth: 2 }}
          connectNulls={false}
        />
      </LineChart>
    </ResponsiveContainer>
  );
}
