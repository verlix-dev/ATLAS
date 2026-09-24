import type { Experiment, MissionState, Rejection } from '../types';
import ScoreChart from '../components/ScoreChart';
import {
  baselineDelta, describeConfig, formatMetric, formatSeconds, formatSigned, isUnavailable, progressLabel,
} from '../lib/reduce';

interface Props {
  mission: MissionState;
  error: string | null;
  onComplete: () => void;
  onRestart: () => void;
}

const LOOP = ['Experiment', 'Measure', 'Diagnose', 'Hypothesize', 'Decide'] as const;

/** Maps the last backend stage onto the loop step being narrated. */
const STAGE_TO_INDEX: Record<string, number> = {
  experiment: 0,
  measure: 1,
  diagnose: 2,
  hypothesize: 3,
  decide: 4,
};

function StageIndicator({ stage, complete }: { stage: string | null; complete: boolean }) {
  const idx = stage ? (STAGE_TO_INDEX[stage] ?? -1) : -1;
  return (
    <div className="flex items-center gap-1">
      {LOOP.map((s, i) => (
        <div key={s} className="flex items-center">
          <div
            className={`px-3 py-1 rounded-full text-xs font-semibold transition-all duration-300 ${complete || i < idx
              ? 'bg-emerald-100 text-emerald-600'
              : i === idx
                ? 'bg-indigo-600 text-white shadow-sm'
                : 'bg-gray-100 text-gray-400'}`}
          >
            {s}
          </div>
          {i < LOOP.length - 1 && (
            <div className={`w-4 h-px mx-0.5 ${i < idx ? 'bg-emerald-300' : 'bg-gray-200'}`} />
          )}
        </div>
      ))}
    </div>
  );
}

function MetricCell({ label, value, signed }: { label: string; value: unknown; signed?: boolean }) {
  const unavailable = isUnavailable(value);
  const text = unavailable
    ? 'N/A'
    : signed
      ? formatSigned(Number(value))
      : formatMetric(value);
  return (
    <div className="p-4 rounded-2xl bg-indigo-50/60 border border-indigo-100">
      <div className="text-xs font-semibold text-gray-400 uppercase tracking-wider mb-1">{label}</div>
      <div className={`text-2xl font-extrabold font-mono ${unavailable ? 'text-gray-300' : 'text-gray-900'}`}>
        {text}
      </div>
    </div>
  );
}

function Rejections({ rejections }: { rejections: Rejection[] }) {
  if (!rejections.length) return null;
  return (
    <div className="mx-5 mt-4 p-3 rounded-xl bg-orange-50 border border-orange-200">
      <div className="text-[10px] font-bold text-orange-600 uppercase tracking-widest mb-1">
        {rejections.length} proposal{rejections.length > 1 ? 's' : ''} rejected by the validator
      </div>
      {rejections.map((r, i) => (
        <div key={i} className="text-xs font-mono text-orange-700">
          [{r.source}] {r.reason}
        </div>
      ))}
      <div className="text-[10px] text-orange-500 mt-1">
        Rejected proposals never ran and have no measured result.
      </div>
    </div>
  );
}

function ExperimentCard({ exp, isCurrent, objective }: { exp: Experiment; isCurrent: boolean; objective: MissionState['objective'] }) {
  const metric = objective?.primary_metric ?? null;
  const result = exp.result;
  const failed = result !== null && !result.ok;

  const decisionTone = {
    continue: 'bg-emerald-50 border-emerald-200 text-emerald-700',
    stop: 'bg-violet-50 border-violet-200 text-violet-700',
    revise: 'bg-orange-50 border-orange-200 text-orange-600',
  }[exp.decision?.action ?? ''] ?? 'bg-gray-50 border-gray-200 text-gray-500';

  // Completed experiments collapse to a receipt — the timeline IS the history.
  if (!isCurrent && exp.decision) {
    return (
      <div className="card px-4 py-3 flex items-center gap-4 hover:shadow-md transition-all duration-200">
        <span className="font-mono text-xs font-bold text-indigo-400 w-8">
          E{String(exp.id).padStart(2, '0')}
        </span>
        <div className="flex-1 min-w-0">
          <div className="text-sm font-semibold text-gray-700 truncate font-mono">
            {exp.config?.model ?? '—'}
          </div>
        </div>
        <div className="text-right">
          <div className={`font-mono font-bold text-sm ${failed ? 'text-red-500' : 'text-gray-900'}`}>
            {failed
              ? 'FAILED'
              : metric
                ? formatMetric(result?.metrics?.[metric])
                : '—'}
          </div>
          <div className="text-xs text-gray-400">{metric?.replace(/_/g, ' ') ?? ''}</div>
        </div>
        <div className={`text-[10px] font-bold px-2 py-0.5 rounded-full border uppercase ${decisionTone}`}>
          {exp.decision.action}
        </div>
      </div>
    );
  }

  return (
    <div className={`card-elevated overflow-hidden ${isCurrent ? 'ring-2 ring-indigo-300 ring-offset-2' : ''}`}>
      {/* Header — identity and provenance, from the real events */}
      <div className="p-5 border-b border-indigo-50">
        <div className="flex items-start justify-between mb-2">
          <div>
            <div className="font-mono text-xs font-bold text-indigo-400 mb-1">
              EXPERIMENT {String(exp.id).padStart(2, '0')}
              {isCurrent && (
                <span className="ml-2 text-[10px] bg-indigo-100 text-indigo-600 px-2 py-0.5 rounded-full">
                  CURRENT
                </span>
              )}
            </div>
            <div className="text-lg font-bold text-gray-900 font-mono">
              {exp.config?.model ?? '—'}
            </div>
          </div>
          <div
            className={`text-xs font-bold px-2.5 py-1 rounded-full border ${
              exp.source === 'llm'
                ? 'bg-violet-50 border-violet-200 text-violet-600'
                : 'bg-indigo-50 border-indigo-200 text-indigo-600'
            }`}
          >
            {exp.source === 'llm' ? '✦ AI GUIDED' : exp.source === 'baseline' ? 'BASELINE' : 'ATLAS RULES'}
          </div>
        </div>
        <div className="flex flex-wrap gap-2 mt-2">
          {Object.entries(exp.config?.params ?? {}).map(([k, v]) => (
            <span key={k} className="font-mono text-xs bg-gray-50 border border-gray-100 px-2 py-0.5 rounded-lg text-gray-600">
              {k}: {String(v)}
            </span>
          ))}
        </div>
      </div>

      {/* FACT — what the engine actually measured */}
      {result && (
        <div className="p-5 border-b border-indigo-50">
          <div className="flex items-center gap-2 mb-3">
            <div className="w-1 h-4 rounded-full bg-blue-400" />
            <span className="text-xs font-bold text-blue-500 uppercase tracking-widest">
              Fact — Measured
            </span>
            {result.seconds !== undefined && (
              <span className="ml-auto text-[10px] font-mono text-gray-400">
                {formatSeconds(result.seconds)}
              </span>
            )}
          </div>

          {failed ? (
            // An execution error is not a score of zero, and no diagnosis follows it.
            <div className="p-4 rounded-2xl bg-red-50 border border-red-200">
              <div className="text-xs font-bold text-red-600 uppercase tracking-widest mb-1">
                Execution failed
              </div>
              <div className="text-xs font-mono text-red-700 break-words">
                {result.error ?? 'the experiment raised without a message'}
              </div>
              <div className="text-[10px] text-red-500 mt-2">
                No metrics were produced. This experiment cannot be ranked.
              </div>
            </div>
          ) : (
            <div className="grid grid-cols-2 gap-3">
              {metric && <MetricCell label={metric.replace(/_/g, ' ')} value={result.metrics?.[metric]} />}
              <MetricCell label="Recall" value={result.metrics?.recall} />
              <MetricCell label="Precision" value={result.metrics?.precision} />
              <MetricCell label="Overfit gap" value={result.metrics?.overfit_gap} signed />
            </div>
          )}

          {!failed && result.metrics?.test_rows !== undefined && (
            <div className="mt-3 text-[10px] font-mono text-gray-400">
              {String(result.metrics.test_rows)} held-out rows
              {result.metrics.positive_class ? ` · positive class "${String(result.metrics.positive_class)}"` : ''}
            </div>
          )}
        </div>
      )}

      {/* INTERPRETATION — what ATLAS concluded */}
      {(exp.diagnosis || exp.hypothesis) && (
        <div className="p-5 border-b border-indigo-50 bg-violet-50/30">
          <div className="flex items-center gap-2 mb-3">
            <div className="w-1 h-4 rounded-full bg-violet-400" />
            <span className="text-xs font-bold text-violet-500 uppercase tracking-widest">
              Interpretation — ATLAS Reasoning
            </span>
          </div>

          {exp.diagnosis && (
            <div className="mb-4">
              <div className="text-xs font-bold text-violet-600 uppercase tracking-wider mb-1">
                Diagnosis · {exp.diagnosis.category.replace(/_/g, ' ')}
              </div>
              <p className="text-sm text-gray-700">{exp.diagnosis.summary}</p>
              {exp.diagnosis.findings.length > 0 && (
                <ul className="mt-2 space-y-1">
                  {exp.diagnosis.findings.map((f) => {
                    const cited = exp.hypothesis?.cites?.includes(f) ?? false;
                    return (
                      <li
                        key={f}
                        data-finding={f}
                        className={`text-xs ${cited ? 'text-gray-800 font-medium' : 'text-gray-500'}`}
                      >
                        — {f}
                      </li>
                    );
                  })}
                </ul>
              )}
            </div>
          )}

          {exp.hypothesis && (
            <div>
              <div className="text-xs font-bold text-violet-600 uppercase tracking-wider mb-1">
                Hypothesis · {exp.hypothesis.source === 'llm' ? 'AI guided' : 'ATLAS rules'}
              </div>
              <p className="text-sm text-gray-800 font-medium">{exp.hypothesis.proposedChange}</p>
              <p className="text-xs text-gray-500 mt-1">{exp.hypothesis.reasoning}</p>
              <p className="text-xs text-gray-400 mt-1 italic">
                Expected: {exp.hypothesis.expectedEffect}
              </p>
              {exp.hypothesis.cites.length > 0 && (
                <div className="flex flex-wrap gap-1.5 mt-3">
                  <span className="text-[10px] text-gray-400 uppercase tracking-wider self-center">
                    cites
                  </span>
                  {exp.hypothesis.cites.map((c) => (
                    <span
                      key={c}
                      className="text-[10px] font-mono bg-white border border-violet-200 text-violet-600 px-2 py-0.5 rounded-full"
                    >
                      {c}
                    </span>
                  ))}
                </div>
              )}
            </div>
          )}

          {!exp.hypothesis && (
            <div className="text-xs text-gray-400 italic">No untried hypothesis remains.</div>
          )}
        </div>
      )}

      <Rejections rejections={exp.rejections} />

      {/* ACTION — what ATLAS decided */}
      {exp.decision && (
        <div className="p-5 bg-gradient-to-r from-indigo-50/60 to-violet-50/40">
          <div className="flex items-center gap-3">
            <div className="w-1 h-4 rounded-full bg-indigo-400" />
            <span className="text-xs font-bold text-indigo-500 uppercase tracking-widest">
              Action — Decision
            </span>
            <span className={`ml-auto text-lg font-extrabold uppercase tracking-wide ${
              exp.decision.action === 'continue'
                ? 'text-emerald-600'
                : exp.decision.action === 'stop'
                  ? 'text-violet-600'
                  : 'text-orange-500'
            }`}>
              {exp.decision.action}
            </span>
          </div>
          <p className="text-sm text-gray-600 mt-2">{exp.decision.reason}</p>
          {exp.hypothesis?.nextConfig && (
            <p className="text-xs font-mono text-gray-500 mt-2">
              next → {describeConfig(exp.hypothesis.nextConfig)}
            </p>
          )}
        </div>
      )}
    </div>
  );
}

export default function Dashboard({ mission, error, onComplete, onRestart }: Props) {
  const experiments = mission.experiments;
  const current = experiments.find((e) => e.id === mission.currentId) ?? experiments[experiments.length - 1] ?? null;
  const older = experiments.filter((e) => e.id !== current?.id).reverse();
  const complete = mission.status === 'complete';
  const blocked = mission.status === 'blocked';
  const metric = mission.objective?.primary_metric ?? null;
  const delta = baselineDelta(mission);

  return (
    <div className="min-h-screen" style={{ background: 'linear-gradient(160deg, #F4F6FF 0%, #EEF1FF 40%, #F8F6FF 100%)' }}>
      <header className="border-b border-indigo-100" style={{ background: 'rgba(255,255,255,0.7)', backdropFilter: 'blur(12px)', position: 'sticky', top: 0, zIndex: 50 }}>
        <div className="max-w-7xl mx-auto px-6 h-14 flex items-center justify-between gap-4">
          <div className="flex items-center gap-3">
            <div className="w-8 h-8 rounded-lg gradient-primary flex items-center justify-center">
              <svg width="16" height="16" viewBox="0 0 16 16" fill="none">
                <path d="M8 2L14 5.5V10.5L8 14L2 10.5V5.5L8 2Z" stroke="white" strokeWidth="1.5" strokeLinejoin="round" />
                <circle cx="8" cy="8" r="2" fill="white" />
              </svg>
            </div>
            <span className="font-bold text-gray-900 tracking-tight">ATLAS</span>
          </div>
          <div className="flex items-center gap-3">
            {blocked ? (
              <span className="text-xs font-bold px-4 py-1.5 rounded-full bg-orange-50 text-orange-700 border border-orange-200">
                MISSION BLOCKED
              </span>
            ) : complete ? (
              <span className="text-xs font-bold px-4 py-1.5 rounded-full bg-emerald-50 text-emerald-700 border border-emerald-200">
                MISSION COMPLETE
              </span>
            ) : (
              <StageIndicator stage={mission.stage} complete={false} />
            )}
            <button
              onClick={onRestart}
              className="text-xs font-semibold text-gray-500 hover:text-indigo-600 transition-colors"
            >
              New mission
            </button>
          </div>
        </div>
      </header>

      <div className="max-w-7xl mx-auto px-6 py-8">
        {/* Mission bar */}
        <div className="card p-4 mb-6 flex flex-wrap items-center gap-6">
          <div>
            <div className="text-[10px] font-semibold text-gray-400 uppercase tracking-widest">Target</div>
            <div className="font-mono text-sm text-gray-800">{mission.profile?.target ?? '—'}</div>
          </div>
          <div>
            <div className="text-[10px] font-semibold text-gray-400 uppercase tracking-widest">Objective</div>
            <div className="font-mono text-sm text-gray-800">
              {metric ? `${metric.replace(/_/g, ' ')} ${mission.objective?.direction === 'minimize' ? '↓' : '↑'}` : '—'}
            </div>
          </div>
          <div>
            <div className="text-[10px] font-semibold text-gray-400 uppercase tracking-widest">Experiments</div>
            <div className="font-mono text-sm text-gray-800">{progressLabel(mission)}</div>
          </div>
          {mission.profile && (
            <div>
              <div className="text-[10px] font-semibold text-gray-400 uppercase tracking-widest">Dataset</div>
              <div className="font-mono text-sm text-gray-800">
                {mission.profile.rows} × {mission.profile.columns}
              </div>
            </div>
          )}
          {mission.profile && (
            <div>
              <div className="text-[10px] font-semibold text-gray-400 uppercase tracking-widest">Imbalance</div>
              <div className="font-mono text-sm text-gray-800">
                {mission.profile.imbalance_ratio?.toFixed(2)}:1
              </div>
            </div>
          )}
          {error && (
            <div className="ml-auto text-xs text-red-600 font-medium">{error}</div>
          )}
        </div>

        {/* Data Engineer refusal — nothing ran, and the UI says so. */}
        {blocked && mission.blocked && (
          <div className="card-elevated p-6 mb-6 border-l-4 border-orange-400">
            <div className="text-xs font-bold text-orange-600 uppercase tracking-widest mb-2">
              Data Engineer · target not confirmed
            </div>
            <p className="text-sm text-gray-700">{mission.blocked.reason}</p>
            <p className="text-xs font-mono text-gray-400 mt-2">
              best guess was "{mission.blocked.target}" · available columns:{' '}
              {mission.blocked.columns.join(', ')}
            </p>
            <p className="text-xs text-gray-500 mt-3">
              No experiment ran. ATLAS does not train on an unconfirmed target.
            </p>
          </div>
        )}

        {/* Hero: score progression */}
        {!blocked && (
          <div className="card-elevated p-6 mb-6">
            <div className="flex items-center justify-between mb-4">
              <div>
                <h2 className="text-sm font-bold text-gray-700 uppercase tracking-widest">
                  Score Progression
                </h2>
                <p className="text-xs text-gray-400 mt-0.5">
                  {metric
                    ? `Measured ${metric.replace(/_/g, ' ')} after each experiment`
                    : 'Awaiting the objective'}
                </p>
              </div>
              {mission.best && (
                <div className="text-right">
                  <div className="text-[10px] font-semibold text-gray-400 uppercase tracking-widest">
                    Best {mission.best.provisional ? '(so far)' : '(final)'}
                  </div>
                  <div className="font-mono text-2xl font-extrabold text-gray-900">
                    {formatMetric(mission.best.score)}
                  </div>
                  {delta !== null && (
                    <div className={`text-xs font-bold ${delta >= 0 ? 'text-emerald-600' : 'text-gray-400'}`}>
                      {formatSigned(delta)} vs baseline
                    </div>
                  )}
                </div>
              )}
            </div>
            <ScoreChart
              experiments={experiments}
              objective={mission.objective}
              currentId={mission.currentId}
            />
          </div>
        )}

        {/* Loop narrative */}
        {!blocked && (
          <div className="grid grid-cols-3 gap-6">
            <div className="col-span-2 flex flex-col gap-4">
              {current ? (
                <ExperimentCard exp={current} isCurrent={!complete} objective={mission.objective} />
              ) : (
                <div className="card p-8 text-center">
                  <div className="text-sm font-bold text-gray-700 mb-1">
                    {mission.profile ? 'Dataset profiled' : 'Mission accepted'}
                  </div>
                  <p className="text-xs text-gray-400">
                    {mission.profile
                      ? `${mission.profile.rows} rows across ${mission.profile.columns} features. Training experiment 01…`
                      : 'Waiting for the Data Engineer to profile the dataset…'}
                  </p>
                </div>
              )}

              {older.length > 0 && (
                <>
                  <div className="text-xs font-bold text-gray-400 uppercase tracking-widest mt-2">
                    Earlier experiments
                  </div>
                  {older.map((e) => (
                    <ExperimentCard key={e.id} exp={e} isCurrent={false} objective={mission.objective} />
                  ))}
                </>
              )}
            </div>

            {/* Evidence rail */}
            <div className="flex flex-col gap-4">
              <div className="card p-5">
                <h3 className="text-xs font-bold text-gray-500 uppercase tracking-widest mb-3">
                  Best Result
                </h3>
                {mission.best ? (
                  <>
                    <div className="font-mono text-3xl font-extrabold text-gray-900">
                      {formatMetric(mission.best.score)}
                    </div>
                    <div className="text-xs text-gray-400 mt-1">
                      Experiment {String(mission.best.id).padStart(2, '0')} ·{' '}
                      {metric?.replace(/_/g, ' ')}
                    </div>
                    {mission.best.config && (
                      <div className="text-[10px] font-mono text-gray-500 mt-2 break-words">
                        {describeConfig(mission.best.config)}
                      </div>
                    )}
                    {mission.best.provisional && (
                      <div className="text-[10px] text-indigo-400 mt-2">
                        provisional — awaiting the authoritative summary
                      </div>
                    )}
                  </>
                ) : (
                  <>
                    <div className="font-mono text-3xl font-extrabold text-gray-300">N/A</div>
                    <div className="text-xs text-gray-400 mt-1">
                      {experiments.length ? 'nothing scorable yet' : 'no experiments yet'}
                    </div>
                  </>
                )}
              </div>

              {mission.summary && (
                <div className="card p-5">
                  <h3 className="text-xs font-bold text-gray-500 uppercase tracking-widest mb-3">
                    Mission Summary
                  </h3>
                  <div className="space-y-2 text-xs">
                    <div className="flex justify-between gap-2">
                      <span className="text-gray-400">Experiments</span>
                      <span className="font-mono text-gray-800">{mission.summary.experiments}</span>
                    </div>
                    <div className="flex justify-between gap-2">
                      <span className="text-gray-400">Best experiment</span>
                      <span className="font-mono text-gray-800">
                        {mission.summary.bestId !== null
                          ? `E${String(mission.summary.bestId).padStart(2, '0')}`
                          : 'N/A'}
                      </span>
                    </div>
                    <div className="flex justify-between gap-2">
                      <span className="text-gray-400">Best score</span>
                      <span className="font-mono text-gray-800">
                        {formatMetric(mission.summary.bestScore)}
                      </span>
                    </div>
                    <div className="pt-2 border-t border-gray-100">
                      <div className="text-gray-400 mb-1">Stop reason</div>
                      <div className="text-gray-700">{mission.summary.reason}</div>
                    </div>
                  </div>
                </div>
              )}

              <button
                onClick={onComplete}
                disabled={!complete}
                className={`w-full rounded-xl py-3 font-bold text-sm transition-all ${complete
                  ? 'gradient-primary text-white hover:shadow-lg hover:shadow-indigo-200'
                  : 'bg-gray-100 text-gray-400 cursor-not-allowed'}`}
              >
                Compare strategies →
              </button>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
