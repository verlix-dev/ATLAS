import { useCallback, useEffect, useState } from 'react';
import ComparisonChart, { type Series } from '../components/ComparisonChart';
import { fetchComparison } from '../lib/api';
import { formatMetric } from '../lib/reduce';
import type { ComparisonResponse, RawMissionSummary } from '../types';

interface Props {
  onRestart: () => void;
  /** Dataset and budget carried over from the mission that just ran. */
  csv: string;
  target: string | null;
  budget: number;
}

const DET_COLOR = '#4F6AF7';
const AI_COLOR = '#7C3AED';

/**
 * Turn one side's raw backend summary into a chart series.
 *
 * `score` comes from the backend's own timeline and may be null — an
 * experiment whose primary metric was unavailable, or which failed. Null is
 * carried through as a gap; it is never coerced to 0.
 */
function toSeries(label: string, summary: RawMissionSummary | undefined): Series {
  const timeline = summary?.timeline ?? [];
  return {
    label,
    points: timeline.map((row) => ({ id: row.id, score: row.score ?? null })),
  };
}

function BestModelCard({
  label,
  summary,
  accentColor,
  isWinner,
  metricLabel,
  minimize,
}: {
  label: string;
  summary: RawMissionSummary | undefined;
  accentColor: string;
  isWinner: boolean;
  metricLabel: string;
  minimize: boolean;
}) {
  const timeline = summary?.timeline ?? [];
  const measured = timeline.filter((r) => r.score !== null);
  const scores = measured.map((r) => r.score as number);
  const lo = scores.length ? Math.min(...scores) : 0;
  const hi = scores.length ? Math.max(...scores) : 1;
  const span = hi - lo || 1;

  return (
    <div
      className={`card-elevated flex-1 overflow-hidden ${isWinner ? 'ring-2 ring-offset-2' : ''}`}
      style={isWinner ? ({ '--tw-ring-color': accentColor } as React.CSSProperties) : undefined}
    >
      {isWinner && (
        <div className="px-4 py-2 text-xs font-bold text-white text-center" style={{ background: accentColor }}>
          ★ BETTER FINAL SCORE
        </div>
      )}
      <div className="p-6">
        <div className="flex items-start justify-between mb-4">
          <div>
            <div className="text-xs font-bold uppercase tracking-widest mb-1" style={{ color: accentColor }}>
              {label}
            </div>
            <div className="text-2xl font-extrabold font-mono text-gray-900">
              {formatMetric(summary?.best_score)}
            </div>
            <div className="text-xs text-gray-400 mt-0.5">{metricLabel}</div>
          </div>
          <div className="text-right">
            <div className="text-xs text-gray-400 mb-1">Best Model</div>
            <div className="text-sm font-bold text-gray-700 font-mono">
              {summary?.best_model ?? '—'}
            </div>
          </div>
        </div>

        {/* Score bars — only measured scores get a bar. */}
        <div className="flex flex-col gap-2 mb-4">
          {timeline.map((row) => {
            if (row.score === null) {
              return (
                <div key={row.id} className="flex items-center gap-2">
                  <span className="font-mono text-xs text-gray-400 w-7">E{row.id}</span>
                  <div className="flex-1 h-1.5 bg-gray-100 rounded-full" />
                  <span className="font-mono text-xs text-gray-300 w-14 text-right">N/A</span>
                </div>
              );
            }
            // Bar length tracks quality, not magnitude. Under a minimize
            // objective the lowest score is the best and draws the longest bar.
            const quality = minimize ? (hi - row.score) / span : (row.score - lo) / span;
            const width = quality * 90 + 10;
            const isBest = row.id === summary?.best_id;
            return (
              <div key={row.id} className="flex items-center gap-2">
                <span className="font-mono text-xs text-gray-400 w-7">E{row.id}</span>
                <div className="flex-1 h-1.5 bg-gray-100 rounded-full overflow-hidden">
                  <div
                    className="h-full rounded-full transition-all duration-1000"
                    style={{ width: `${width}%`, background: isBest ? accentColor : `${accentColor}66` }}
                  />
                </div>
                <span className={`font-mono text-xs font-bold w-14 text-right ${isBest ? 'text-gray-900' : 'text-gray-500'}`}>
                  {row.score.toFixed(4)}
                </span>
              </div>
            );
          })}
          {!timeline.length && <div className="text-xs text-gray-400">No experiments recorded.</div>}
        </div>

        <div className="grid grid-cols-2 gap-3 pt-4 border-t border-gray-100 text-xs">
          <div>
            <div className="text-gray-400 mb-0.5">Experiments</div>
            <div className="font-bold text-gray-800">{summary?.experiments ?? 0}</div>
          </div>
          <div>
            <div className="text-gray-400 mb-0.5">Scorable</div>
            <div className="font-bold text-gray-800">
              {measured.length} / {timeline.length}
            </div>
          </div>
          <div className="col-span-2">
            <div className="text-gray-400 mb-0.5">Stop Reason</div>
            <div className="font-bold text-gray-800">{summary?.stop_reason ?? '—'}</div>
          </div>
        </div>
      </div>
    </div>
  );
}

function ExperimentRow({
  id,
  det,
  ai,
  metricLabel,
  minimize,
}: {
  id: number;
  det?: RawMissionSummary['timeline'][number];
  ai?: RawMissionSummary['timeline'][number];
  metricLabel: string;
  minimize: boolean;
}) {
  const detScore = det?.score ?? null;
  const aiScore = ai?.score ?? null;
  const bothMeasured = detScore !== null && aiScore !== null;
  // "Leads" means better under the mission's own objective, not simply larger.
  const detLeads = bothMeasured && (minimize ? detScore! < aiScore! : detScore! > aiScore!);

  const cell = (row: RawMissionSummary['timeline'][number] | undefined, side: 'det' | 'ai') => {
    if (!row) {
      return (
        <div className="card p-3 opacity-50">
          <div className="text-xs text-gray-400 italic">no experiment at this step</div>
        </div>
      );
    }
    const highlighted =
      bothMeasured && ((side === 'det' && detLeads) || (side === 'ai' && !detLeads));
    return (
      <div
        className={`card p-3 transition-all ${
          highlighted ? (side === 'det' ? 'ring-1 ring-indigo-200' : 'ring-1 ring-violet-200') : ''
        }`}
      >
        <div className={`text-xs font-bold mb-1 font-mono ${side === 'det' ? 'text-indigo-500' : 'text-violet-500'}`}>
          {row.model}
        </div>
        <div className={`font-mono font-extrabold ${row.score === null ? 'text-gray-300' : 'text-gray-900'}`}>
          {row.score === null ? 'N/A' : row.score.toFixed(4)}
        </div>
        <div className="text-xs text-gray-400 mt-1 uppercase">{row.decision ?? '—'}</div>
      </div>
    );
  };

  return (
    <div className="grid grid-cols-[1fr_auto_1fr] gap-4 items-center py-4 border-b border-gray-100 last:border-0">
      {cell(det, 'det')}
      <div className="flex flex-col items-center gap-1">
        <span className="font-mono text-xs font-bold text-gray-400">
          E{String(id).padStart(2, '0')}
        </span>
        {bothMeasured ? (
          <>
            <div className="text-xs font-bold" style={{ color: detLeads ? DET_COLOR : AI_COLOR }}>
              {detLeads ? '◀ Det' : 'AI ▶'}
            </div>
            <div className="text-xs font-mono text-gray-400">
              Δ {Math.abs(detScore! - aiScore!).toFixed(4)}
            </div>
          </>
        ) : (
          <div className="text-xs text-gray-300">not comparable</div>
        )}
        <span className="text-[10px] text-gray-300">{metricLabel}</span>
      </div>
      {cell(ai, 'ai')}
    </div>
  );
}

export default function Comparison({ onRestart, csv, target, budget }: Props) {
  const [data, setData] = useState<ComparisonResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  const run = useCallback(() => {
    setLoading(true);
    setError(null);
    fetchComparison({ csv, target, budget })
      .then(setData)
      .catch((cause: unknown) => setError(cause instanceof Error ? cause.message : String(cause)))
      .finally(() => setLoading(false));
  }, [csv, target, budget]);

  useEffect(run, [run]);

  const det = data?.deterministic.summary;
  const ai = data?.guided.summary;
  const objective = det?.objective ?? ai?.objective ?? null;
  const metricLabel = (objective?.primary_metric ?? '').replace(/_/g, ' ').toUpperCase();
  // Both sides share one Objective object server-side, so either reports it.
  const minimize = objective?.direction === 'minimize';

  const detBest = det?.best_score ?? null;
  const aiBest = ai?.best_score ?? null;
  const comparable = detBest !== null && aiBest !== null;
  const delta = comparable ? Math.abs(detBest! - aiBest!) : null;
  const tie = comparable && detBest === aiBest;
  // Better, not bigger: a minimize objective inverts which side won.
  const detWins = comparable && !tie && (minimize ? detBest! < aiBest! : detBest! > aiBest!);
  const winner = !comparable || tie ? null : detWins ? 'Deterministic' : 'AI Guided';

  // Walk the union of experiment ids. Pairing by array position would line up
  // two unrelated experiments as soon as the runs differ in length.
  const ids = Array.from(
    new Set([...(det?.timeline ?? []), ...(ai?.timeline ?? [])].map((r) => r.id)),
  ).sort((a, b) => a - b);
  const rows = ids.map((id) => ({
    id,
    det: det?.timeline.find((r) => r.id === id),
    ai: ai?.timeline.find((r) => r.id === id),
  }));

  return (
    <div className="min-h-screen" style={{ background: 'linear-gradient(160deg, #F4F6FF 0%, #EEF1FF 40%, #F8F6FF 100%)' }}>
      <header className="border-b border-indigo-100" style={{ background: 'rgba(255,255,255,0.8)', backdropFilter: 'blur(12px)', position: 'sticky', top: 0, zIndex: 50 }}>
        <div className="max-w-7xl mx-auto px-6 h-14 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-7 h-7 rounded-lg gradient-primary flex items-center justify-center">
              <svg width="14" height="14" viewBox="0 0 16 16" fill="none">
                <path d="M8 2L14 5.5V10.5L8 14L2 10.5V5.5L8 2Z" stroke="white" strokeWidth="1.5" strokeLinejoin="round" />
                <circle cx="8" cy="8" r="2" fill="white" />
              </svg>
            </div>
            <span className="font-bold text-gray-900 tracking-tight">ATLAS</span>
            <span className="text-xs text-gray-400 font-medium">/ Comparison Results</span>
          </div>
          <button
            onClick={onRestart}
            className="text-sm font-bold text-indigo-600 bg-indigo-50 hover:bg-indigo-100 px-4 py-2 rounded-xl transition-colors"
          >
            ← New Mission
          </button>
        </div>
      </header>

      <div className="max-w-7xl mx-auto px-6 py-10">
        {loading && (
          <div className="text-center py-24">
            <div className="text-sm text-gray-500 mb-2">Running both strategies…</div>
            <p className="text-xs text-gray-400">
              The backend runs a deterministic mission and an LLM-guided mission over the same
              dataset and split, then returns both event logs.
            </p>
          </div>
        )}

        {error && !loading && (
          <div className="card-elevated p-6 border-l-4 border-red-400">
            <div className="text-xs font-bold text-red-600 uppercase tracking-widest mb-2">
              Comparison failed
            </div>
            <p className="text-sm text-gray-700">{error}</p>
            <button onClick={run} className="mt-4 text-xs font-bold text-indigo-600 hover:text-indigo-800">
              Retry
            </button>
          </div>
        )}

        {data && !loading && (
          <>
            {/* Verdict */}
            <div className="text-center mb-12">
              <div className="text-xs font-bold text-gray-400 uppercase tracking-widest mb-3">
                Which Strategy Learned Better?
              </div>
              <h1 className="text-4xl font-extrabold text-gray-900 mb-2">
                {comparable ? (
                  tie ? (
                    <span className="gradient-text">Both strategies tied</span>
                  ) : (
                    <>
                      <span className="gradient-text">{winner}</span> achieved the better score
                    </>
                  )
                ) : (
                  <span className="text-gray-500">Scores not comparable</span>
                )}
              </h1>

              <div className="flex items-center justify-center gap-10 mt-6">
                <div className="text-center">
                  <div className="text-xs font-bold text-indigo-500 uppercase tracking-widest mb-1">Deterministic</div>
                  <div className={`text-5xl font-extrabold font-mono ${detWins ? 'text-gray-900' : 'text-gray-400'}`}>
                    {formatMetric(detBest)}
                  </div>
                </div>
                <div className="flex flex-col items-center gap-1">
                  <div className="text-2xl text-gray-300">vs</div>
                  <div className="text-sm font-bold text-gray-500 font-mono">
                    {delta === null ? 'Δ —' : `Δ ${delta.toFixed(4)}`}
                  </div>
                  <div className="text-[10px] text-gray-400 font-mono">{metricLabel}</div>
                </div>
                <div className="text-center">
                  <div className="text-xs font-bold text-violet-500 uppercase tracking-widest mb-1">AI Guided</div>
                  <div className={`text-5xl font-extrabold font-mono ${comparable && !detWins && !tie ? 'text-gray-900' : 'text-gray-400'}`}>
                    {formatMetric(aiBest)}
                  </div>
                </div>
              </div>

              {comparable && !tie && (
                <div className="mt-6 inline-flex items-start gap-2 bg-amber-50 border border-amber-100 rounded-2xl px-5 py-3 text-left max-w-2xl">
                  <svg width="16" height="16" viewBox="0 0 16 16" fill="none" className="mt-0.5 flex-shrink-0">
                    <path d="M8 2L14.9 14H1.1L8 2Z" stroke="#F59E0B" strokeWidth="1.5" strokeLinejoin="round" />
                    <path d="M8 6v3M8 11v1" stroke="#F59E0B" strokeWidth="1.5" strokeLinecap="round" />
                  </svg>
                  <p className="text-sm text-amber-800 font-medium">
                    ATLAS reports results honestly. {winner} reached a better best {metricLabel}{' '}
                    ({formatMetric(detWins ? detBest : aiBest)} vs {formatMetric(detWins ? aiBest : detBest)}),
                    a difference of {delta!.toFixed(4)}. Neither strategy is universally superior —
                    results may differ on another dataset.
                  </p>
                </div>
              )}
            </div>

            {/* Trajectories */}
            <div className="card-elevated p-6 mb-8">
              <div className="mb-1">
                <div className="text-xs font-bold text-gray-400 uppercase tracking-widest mb-1">
                  Learning Trajectories
                </div>
                <div className="text-sm font-bold text-gray-700">
                  How each strategy navigated the experiment space
                </div>
              </div>
              <ComparisonChart
                deterministic={toSeries('Deterministic', det)}
                guided={toSeries('AI Guided', ai)}
                metricLabel={metricLabel}
              />
            </div>

            {/* Side-by-side */}
            <div className="flex gap-5 mb-8">
              <BestModelCard
                label="Deterministic ATLAS"
                summary={det}
                accentColor={DET_COLOR}
                isWinner={detWins}
                metricLabel={metricLabel}
                minimize={minimize}
              />
              <BestModelCard
                label="AI Guided ATLAS"
                summary={ai}
                accentColor={AI_COLOR}
                isWinner={comparable && !tie && !detWins}
                metricLabel={metricLabel}
                minimize={minimize}
              />
            </div>

            {/* Fairness, asserted by the server from the objects themselves */}
            <div className="card-elevated p-6 mb-8">
              <div className="text-xs font-bold text-gray-400 uppercase tracking-widest mb-4">
                Fair Comparison
              </div>
              <div className="grid grid-cols-4 gap-4 text-xs">
                {[
                  { label: 'Dataset', value: data.csv },
                  { label: 'Target', value: data.target },
                  { label: 'Split', value: `${data.train_rows} train / ${data.test_rows} test` },
                  { label: 'Budget', value: `${data.budget.max_experiments} experiments max` },
                  { label: 'Same Dataset object', value: data.same_dataset_object ? '✓ yes' : '✗ NO' },
                  { label: 'Same Objective object', value: data.same_objective ? '✓ yes' : '✗ NO' },
                  { label: 'Patience', value: String(data.budget.patience) },
                  { label: 'Metric', value: metricLabel || '—' },
                ].map(({ label, value }) => (
                  <div key={label} className="p-3 rounded-xl bg-gray-50/60 border border-gray-100">
                    <div className="text-gray-400 mb-1">{label}</div>
                    <div className="font-mono font-semibold text-gray-800 break-words">{value}</div>
                  </div>
                ))}
              </div>
            </div>

            {/* Experiment by experiment */}
            <div className="card-elevated p-6 mb-8">
              <div className="text-xs font-bold text-gray-400 uppercase tracking-widest mb-5">
                Experiment-by-Experiment
              </div>
              <div className="grid grid-cols-[1fr_auto_1fr] gap-4 mb-2">
                <div className="text-xs font-bold text-indigo-500 uppercase tracking-wider">Deterministic</div>
                <div className="w-16" />
                <div className="text-xs font-bold text-violet-500 uppercase tracking-wider text-right">AI Guided</div>
              </div>
              {rows.map((row) => (
                <ExperimentRow
                  key={row.id}
                  id={row.id}
                  det={row.det}
                  ai={row.ai}
                  metricLabel={metricLabel}
                  minimize={minimize}
                />
              ))}
            </div>

            {/* Novel configurations the proposer reached */}
            <div className="card-elevated p-6 mb-8">
              <div className="text-xs font-bold text-gray-400 uppercase tracking-widest mb-4">
                Configurations Outside the Deterministic Run ({data.novel.length})
              </div>
              {data.novel.length ? (
                <div className="space-y-2">
                  {data.novel.map((c) => (
                    <div key={c.id} className="flex items-center gap-3 text-xs p-3 rounded-xl bg-gray-50/60 border border-gray-100">
                      <span className="font-mono font-bold text-indigo-500 w-8">
                        E{String(c.id).padStart(2, '0')}
                      </span>
                      <span className="font-mono font-semibold text-gray-700">{c.model}</span>
                      <span className="font-mono text-gray-400 truncate">
                        {Object.entries(c.params).map(([k, v]) => `${k}=${v}`).join('  ') || 'defaults'}
                      </span>
                    </div>
                  ))}
                </div>
              ) : (
                <p className="text-xs text-gray-400">
                  The proposer reached nothing the deterministic ladder could not.
                </p>
              )}
            </div>

            {/* Rejections */}
            {data.rejections.length > 0 && (
              <div className="card-elevated p-6">
                <div className="text-xs font-bold text-gray-400 uppercase tracking-widest mb-4">
                  Rejected Proposals ({data.rejections.length})
                </div>
                <div className="p-3 rounded-xl bg-orange-50 border border-orange-200">
                  {data.rejections.map((r, i) => (
                    <div key={i} className="text-xs font-mono text-orange-700">
                      [{r.source}] {r.reason}
                    </div>
                  ))}
                  <div className="text-[10px] text-orange-500 mt-1">
                    These never ran and have no measured result.
                  </div>
                </div>
              </div>
            )}
          </>
        )}
      </div>
    </div>
  );
}
