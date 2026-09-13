import { useState, useEffect, useRef } from 'react';
import type { MissionState, Experiment, Stage } from '../types';
import ScoreChart from '../components/ScoreChart';
import { deterministicExperiments } from '../data/demoData';

interface Props {
  mission: MissionState;
  onComplete: () => void;
}

const STAGES: Stage[] = ['experiment', 'measure', 'diagnose', 'hypothesize', 'decide'];
const STAGE_LABELS: Record<Stage, string> = {
  experiment: 'Experiment',
  measure: 'Measure',
  diagnose: 'Diagnose',
  hypothesize: 'Hypothesize',
  decide: 'Decide',
};

function StageIndicator({ current }: { current: Stage }) {
  const idx = STAGES.indexOf(current);
  return (
    <div className="flex items-center gap-1">
      {STAGES.map((s, i) => (
        <div key={s} className="flex items-center">
          <div
            className={`px-3 py-1 rounded-full text-xs font-semibold transition-all duration-300 ${i === idx
              ? 'bg-indigo-600 text-white shadow-sm'
              : i < idx
              ? 'bg-emerald-100 text-emerald-600'
              : 'bg-gray-100 text-gray-400'}`}
          >
            {STAGE_LABELS[s]}
          </div>
          {i < STAGES.length - 1 && (
            <div className={`w-4 h-px mx-0.5 ${i < idx ? 'bg-emerald-300' : 'bg-gray-200'}`} />
          )}
        </div>
      ))}
    </div>
  );
}

function MetricBadge({ label, value, sub }: { label: string; value: string | number; sub?: string }) {
  return (
    <div className="p-4 rounded-2xl bg-indigo-50/60 border border-indigo-100">
      <div className="text-xs font-semibold text-gray-400 uppercase tracking-wider mb-1">{label}</div>
      <div className="text-2xl font-extrabold text-gray-900 font-mono">{value}</div>
      {sub && <div className="text-xs text-gray-400 mt-0.5">{sub}</div>}
    </div>
  );
}

function ExperimentCard({ exp, isCurrent }: { exp: Experiment; isCurrent: boolean }) {
  const stage = exp.revealStage;
  const showMeasured = ['measure', 'diagnose', 'hypothesize', 'decide', 'complete'].includes(stage);
  const showDiagnosis = ['diagnose', 'hypothesize', 'decide', 'complete'].includes(stage);
  const showHypothesis = ['hypothesize', 'decide', 'complete'].includes(stage);
  const showDecision = ['decide', 'complete'].includes(stage);

  const decisionColor = {
    CONTINUE: 'bg-emerald-50 border-emerald-200 text-emerald-700',
    STOP: 'bg-violet-50 border-violet-200 text-violet-700',
    REVISE: 'bg-orange-50 border-orange-200 text-orange-600',
  }[exp.decision ?? 'CONTINUE'];

  if (!isCurrent && stage === 'complete') {
    return (
      <div className="card px-4 py-3 flex items-center gap-4 group hover:shadow-md transition-all duration-200">
        <span className="font-mono text-xs font-bold text-indigo-400 w-8">E{String(exp.id).padStart(2, '0')}</span>
        <div className="flex-1 min-w-0">
          <div className="text-sm font-semibold text-gray-700 truncate">{exp.model}</div>
        </div>
        <div className="text-right">
          <div className="font-mono font-bold text-gray-900 text-sm">{exp.metrics?.f1_macro.toFixed(4)}</div>
          <div className="text-xs text-gray-400">F1 Macro</div>
        </div>
        <div className={`text-[10px] font-bold px-2 py-0.5 rounded-full border ${decisionColor}`}>
          {exp.decision}
        </div>
      </div>
    );
  }

  return (
    <div className={`card-elevated overflow-hidden animate-in ${isCurrent ? 'ring-2 ring-indigo-300 ring-offset-2' : ''}`}>
      {/* Header */}
      <div className="p-5 border-b border-indigo-50">
        <div className="flex items-start justify-between mb-2">
          <div>
            <div className="font-mono text-xs font-bold text-indigo-400 mb-1">
              EXPERIMENT {String(exp.id).padStart(2, '0')}
              {isCurrent && <span className="ml-2 text-[10px] bg-indigo-100 text-indigo-600 px-2 py-0.5 rounded-full non-mono">CURRENT</span>}
            </div>
            <div className="text-lg font-bold text-gray-900">{exp.model}</div>
          </div>
          <div className={`text-xs font-bold px-2.5 py-1 rounded-full border ${exp.provenance === 'AI_GUIDED' ? 'bg-violet-50 border-violet-200 text-violet-600' : 'bg-indigo-50 border-indigo-200 text-indigo-600'}`}>
            {exp.provenance === 'AI_GUIDED' ? '✦ AI GUIDED' : 'ATLAS RULES'}
          </div>
        </div>
        <div className="flex flex-wrap gap-2 mt-2">
          {Object.entries(exp.params).map(([k, v]) => (
            <span key={k} className="font-mono text-xs bg-gray-50 border border-gray-100 px-2 py-0.5 rounded-lg text-gray-600">
              {k}: {v}
            </span>
          ))}
        </div>
      </div>

      {/* FACT */}
      {showMeasured && exp.metrics && (
        <div className="p-5 border-b border-indigo-50 animate-in">
          <div className="flex items-center gap-2 mb-3">
            <div className="w-1 h-4 rounded-full bg-blue-400" />
            <span className="text-xs font-bold text-blue-500 uppercase tracking-widest">FACT — Measured</span>
          </div>
          <div className="grid grid-cols-4 gap-3">
            <MetricBadge label="F1 Macro" value={exp.metrics.f1_macro.toFixed(4)} />
            <MetricBadge label="Recall" value={exp.metrics.recall.toFixed(4)} />
            <MetricBadge label="Precision" value={exp.metrics.precision.toFixed(4)} />
            <MetricBadge
              label="Overfit Gap"
              value={`+${exp.metrics.overfit_gap.toFixed(4)}`}
              sub={exp.metrics.overfit_gap > 0.12 ? '⚠ High' : 'Acceptable'}
            />
          </div>
        </div>
      )}

      {/* INTERPRETATION — Diagnosis */}
      {showDiagnosis && exp.diagnosis && (
        <div className="p-5 border-b border-indigo-50 animate-in">
          <div className="flex items-center gap-2 mb-3">
            <div className="w-1 h-4 rounded-full bg-violet-400" />
            <span className="text-xs font-bold text-violet-500 uppercase tracking-widest">INTERPRETATION — Diagnosis</span>
          </div>
          <p className="text-sm text-gray-700 leading-relaxed font-medium">{exp.diagnosis}</p>
        </div>
      )}

      {/* INTERPRETATION — Hypothesis */}
      {showHypothesis && exp.hypothesis && (
        <div className="p-5 border-b border-indigo-50 animate-in">
          <div className="flex items-center gap-2 mb-3">
            <div className="w-1 h-4 rounded-full bg-cyan-400" />
            <span className="text-xs font-bold text-cyan-500 uppercase tracking-widest">INTERPRETATION — Hypothesis</span>
          </div>
          <p className="text-sm text-gray-700 leading-relaxed font-medium">{exp.hypothesis}</p>
          {exp.evidence && exp.evidence.length > 0 && (
            <div className="flex flex-wrap gap-2 mt-3">
              <span className="text-xs text-gray-400 mr-1 self-center">Evidence:</span>
              {exp.evidence.map(ev => (
                <span key={ev} className="text-xs bg-cyan-50 border border-cyan-100 text-cyan-700 font-semibold px-2.5 py-1 rounded-full">
                  {ev}
                </span>
              ))}
            </div>
          )}
        </div>
      )}

      {/* ACTION — Decision */}
      {showDecision && exp.decision && (
        <div className="p-5 animate-in">
          <div className="flex items-center gap-2 mb-3">
            <div className="w-1 h-4 rounded-full bg-emerald-400" />
            <span className="text-xs font-bold text-emerald-600 uppercase tracking-widest">ACTION — Decision</span>
          </div>
          <div className="flex items-start gap-4">
            <div className={`text-sm font-extrabold px-4 py-2 rounded-xl border-2 ${decisionColor}`}>
              {exp.decision}
            </div>
            <p className="text-sm text-gray-600 font-medium self-center">{exp.decisionReason}</p>
          </div>
        </div>
      )}

      {/* Running shimmer */}
      {isCurrent && stage === 'experiment' && (
        <div className="p-5">
          <div className="shimmer h-4 rounded-lg mb-2" />
          <div className="shimmer h-4 rounded-lg w-3/4" />
        </div>
      )}
    </div>
  );
}

export default function Dashboard({ mission, onComplete }: Props) {
  const [experiments, setExperiments] = useState<Experiment[]>([]);
  const [currentIndex, setCurrentIndex] = useState(0);
  const [stage, setStage] = useState<Stage>('experiment');
  const [missionComplete, setMissionComplete] = useState(false);
  const [stopReason, setStopReason] = useState('');
  const timerRef = useRef<ReturnType<typeof setTimeout>>();

  const source = deterministicExperiments.slice(0, mission.config.maxExperiments);

  useEffect(() => {
    let expIdx = 0;
    let stageIdx = 0;
    const DELAYS: Record<Stage, number> = {
      experiment: 900,
      measure: 1100,
      diagnose: 1300,
      hypothesize: 1100,
      decide: 900,
    };

    function advance() {
      const exp = source[expIdx];
      if (!exp) return;
      const currentStage = STAGES[stageIdx] as Stage;

      setStage(currentStage);
      setExperiments(prev => {
        const next = [...prev];
        const existingIdx = next.findIndex(e => e.id === exp.id);
        const updated: Experiment = { ...exp, status: 'running', revealStage: currentStage };
        if (existingIdx >= 0) {
          next[existingIdx] = updated;
        } else {
          next.push(updated);
        }
        return next;
      });
      setCurrentIndex(expIdx);

      stageIdx++;
      if (stageIdx >= STAGES.length) {
        // finish this experiment
        setExperiments(prev => {
          const next = [...prev];
          const idx = next.findIndex(e => e.id === exp.id);
          if (idx >= 0) next[idx] = { ...exp, status: 'complete', revealStage: 'complete' };
          return next;
        });

        expIdx++;
        stageIdx = 0;

        if (expIdx >= source.length || exp.decision === 'STOP') {
          timerRef.current = setTimeout(() => {
            setMissionComplete(true);
            setStopReason(exp.decision === 'STOP' ? 'Objective reached' : `${source.length} experiments evaluated`);
            setCurrentIndex(expIdx - 1);
          }, 1200);
          return;
        }
        timerRef.current = setTimeout(advance, 1400);
      } else {
        timerRef.current = setTimeout(advance, DELAYS[currentStage]);
      }
    }

    timerRef.current = setTimeout(advance, 600);
    return () => clearTimeout(timerRef.current);
  }, []);

  const bestExp = experiments
    .filter(e => e.metrics)
    .reduce<Experiment | null>((best, e) => {
      if (!best || e.metrics!.f1_macro > best.metrics!.f1_macro) return e;
      return best;
    }, null);

  const currentExp = experiments[currentIndex];

  return (
    <div className="min-h-screen" style={{ background: 'linear-gradient(160deg, #F4F6FF 0%, #EEF1FF 40%, #F8F6FF 100%)' }}>
      {/* Header */}
      <header className="border-b border-indigo-100" style={{ background: 'rgba(255,255,255,0.8)', backdropFilter: 'blur(12px)', position: 'sticky', top: 0, zIndex: 50 }}>
        <div className="max-w-7xl mx-auto px-6 h-14 flex items-center justify-between gap-4">
          <div className="flex items-center gap-3">
            <div className="w-7 h-7 rounded-lg gradient-primary flex items-center justify-center">
              <svg width="14" height="14" viewBox="0 0 16 16" fill="none">
                <path d="M8 2L14 5.5V10.5L8 14L2 10.5V5.5L8 2Z" stroke="white" strokeWidth="1.5" strokeLinejoin="round" />
                <circle cx="8" cy="8" r="2" fill="white" />
              </svg>
            </div>
            <span className="font-bold text-gray-900 tracking-tight">ATLAS</span>
          </div>

          {/* Status */}
          <div className={`flex items-center gap-2 px-4 py-1.5 rounded-full text-sm font-bold ${missionComplete ? 'bg-emerald-50 text-emerald-700 border border-emerald-200' : 'bg-indigo-50 text-indigo-700 border border-indigo-200'}`}>
            {missionComplete ? (
              <>
                <svg width="14" height="14" viewBox="0 0 14 14" fill="none">
                  <circle cx="7" cy="7" r="6" fill="#10B981" />
                  <path d="M4 7l2 2 4-4" stroke="white" strokeWidth="1.5" strokeLinecap="round" />
                </svg>
                MISSION COMPLETE — {stopReason}
              </>
            ) : (
              <>
                <div className="w-2 h-2 rounded-full bg-indigo-500 pulse-dot" />
                ATLAS IS EXPERIMENTING — Running Experiment {String((currentIndex + 1)).padStart(2, '0')}
              </>
            )}
          </div>

          {/* Stage indicator */}
          {!missionComplete && <StageIndicator current={stage} />}

          {missionComplete && (
            <button
              onClick={onComplete}
              className="gradient-primary text-white text-sm font-bold px-5 py-2 rounded-xl hover:shadow-md hover:shadow-indigo-200 transition-all"
            >
              View Comparison →
            </button>
          )}
        </div>
      </header>

      <div className="max-w-7xl mx-auto px-6 py-6 grid grid-cols-12 gap-5">
        {/* LEFT — Mission Context */}
        <div className="col-span-2 flex flex-col gap-4">
          <div className="card p-4">
            <div className="text-xs font-bold text-gray-400 uppercase tracking-widest mb-4">Mission</div>
            <div className="text-sm font-bold text-gray-900 mb-4 leading-snug">{mission.config.task.slice(0, 80)}{mission.config.task.length > 80 ? '…' : ''}</div>
            {[
              { label: 'Dataset', value: <span className="font-mono text-xs">{mission.config.dataset}</span> },
              { label: 'Rows', value: mission.config.rows.toLocaleString() },
              { label: 'Target', value: <span className="font-mono text-xs text-indigo-600">{mission.config.target}</span> },
              { label: 'Task', value: mission.config.problemType },
              { label: 'Metric', value: mission.config.primaryMetric },
              { label: 'Priority', value: mission.config.priority },
            ].map(({ label, value }) => (
              <div key={label} className="mb-3">
                <div className="text-[10px] text-gray-400 uppercase tracking-wider font-semibold mb-0.5">{label}</div>
                <div className="text-xs font-semibold text-gray-700">{value}</div>
              </div>
            ))}

            {/* Budget */}
            <div className="mt-4 pt-4 border-t border-gray-100">
              <div className="text-[10px] text-gray-400 uppercase tracking-wider font-semibold mb-2">Budget</div>
              <div className="text-sm font-bold text-gray-900 mb-2">
                {missionComplete ? experiments.filter(e => e.status === 'complete').length : currentIndex + 1} of ≤{mission.config.maxExperiments}
              </div>
              <div className="flex gap-1">
                {Array.from({ length: mission.config.maxExperiments }, (_, i) => (
                  <div
                    key={i}
                    className="flex-1 h-1.5 rounded-full transition-all duration-500"
                    style={{
                      background: i < experiments.filter(e => e.status === 'complete').length
                        ? 'linear-gradient(90deg, #4F6AF7, #7C3AED)'
                        : i === currentIndex && !missionComplete ? '#E0E7FF' : '#F3F4F6'
                    }}
                  />
                ))}
              </div>
            </div>
          </div>
        </div>

        {/* CENTER — Score Graph + Experiment Cards */}
        <div className="col-span-7 flex flex-col gap-5">
          {/* Score Chart */}
          <div className="card-elevated p-5">
            <div className="flex items-center justify-between mb-1">
              <div>
                <div className="text-xs font-bold text-gray-400 uppercase tracking-widest mb-1">Score Progression</div>
                <div className="text-sm font-bold text-gray-700">Primary Metric — F1 Macro</div>
              </div>
              {bestExp && (
                <div className="text-right">
                  <div className="text-xs text-gray-400 mb-0.5">Current Best</div>
                  <div className="font-mono font-extrabold text-indigo-600 text-lg">{bestExp.metrics!.f1_macro.toFixed(4)}</div>
                </div>
              )}
            </div>
            {experiments.filter(e => e.metrics).length > 0 ? (
              <ScoreChart experiments={experiments} currentIndex={currentIndex} />
            ) : (
              <div className="h-[260px] flex items-center justify-center">
                <div className="text-center">
                  <div className="w-8 h-8 border-2 border-indigo-300 border-t-indigo-600 rounded-full animate-spin mx-auto mb-3" />
                  <div className="text-sm text-gray-400 font-medium">Running first experiment…</div>
                </div>
              </div>
            )}
          </div>

          {/* Experiment cards — current first, then history */}
          <div className="flex flex-col gap-3">
            {currentExp && (
              <ExperimentCard key={`current-${currentExp.id}`} exp={currentExp} isCurrent={!missionComplete} />
            )}
            {experiments
              .filter(e => e.id !== currentExp?.id || missionComplete)
              .reverse()
              .map(exp => (
                <ExperimentCard key={exp.id} exp={exp} isCurrent={false} />
              ))}
          </div>
        </div>

        {/* RIGHT — Best + Stats */}
        <div className="col-span-3 flex flex-col gap-4">
          {/* Best Result */}
          {bestExp ? (
            <div className="card-elevated overflow-hidden">
              <div className="px-4 pt-4 pb-3" style={{ background: 'linear-gradient(135deg, rgba(124,58,237,0.06) 0%, rgba(79,106,247,0.06) 100%)' }}>
                <div className="flex items-center gap-2 mb-3">
                  <svg width="14" height="14" viewBox="0 0 14 14" fill="none">
                    <path d="M7 1l1.8 3.6 4 .58-2.9 2.83.68 3.99L7 10l-3.58 1.99.68-3.99L1.2 5.18l4-.58L7 1z" fill="#7C3AED" />
                  </svg>
                  <span className="text-xs font-bold text-violet-600 uppercase tracking-widest">Best Result</span>
                </div>
                <div className="font-mono text-xs text-violet-400 mb-1">Experiment {String(bestExp.id).padStart(2, '0')}</div>
                <div className="text-base font-bold text-gray-900 mb-3">{bestExp.model}</div>
                <div className="text-4xl font-extrabold text-gray-900 font-mono">{bestExp.metrics!.f1_macro.toFixed(4)}</div>
                <div className="text-xs text-gray-400 mt-1 mb-3">F1 Macro</div>
                {experiments.filter(e => e.metrics && e.id !== bestExp.id).length > 0 && (() => {
                  const prev = experiments
                    .filter(e => e.metrics && e.id !== bestExp.id)
                    .sort((a, b) => b.metrics!.f1_macro - a.metrics!.f1_macro)[0];
                  const delta = bestExp.metrics!.f1_macro - prev.metrics!.f1_macro;
                  return (
                    <div className="flex items-center gap-2 text-xs">
                      <span className="text-gray-400">vs previous</span>
                      <span className={`font-bold ${delta > 0 ? 'text-emerald-600' : 'text-orange-500'}`}>
                        {delta > 0 ? '+' : ''}{delta.toFixed(4)}
                      </span>
                    </div>
                  );
                })()}
              </div>
              <div className="px-4 py-3 border-t border-indigo-50">
                <div className="text-xs text-gray-400 mb-2 font-semibold">Configuration</div>
                {Object.entries(bestExp.params).slice(0, 3).map(([k, v]) => (
                  <div key={k} className="flex items-center justify-between text-xs mb-1">
                    <span className="font-mono text-gray-500">{k}</span>
                    <span className="font-mono font-bold text-gray-800">{v}</span>
                  </div>
                ))}
              </div>
            </div>
          ) : (
            <div className="card p-4 text-center text-sm text-gray-400">
              <div className="shimmer h-4 rounded mb-2" />
              <div className="shimmer h-4 rounded w-3/4 mx-auto" />
            </div>
          )}

          {/* Live stats */}
          {experiments.filter(e => e.metrics).length > 0 && (
            <div className="card p-4">
              <div className="text-xs font-bold text-gray-400 uppercase tracking-widest mb-3">Live Stats</div>
              {(() => {
                const withMetrics = experiments.filter(e => e.metrics);
                const scores = withMetrics.map(e => e.metrics!.f1_macro);
                const improvements = withMetrics.filter((e, i) => i > 0 && e.metrics!.f1_macro > withMetrics[i-1].metrics!.f1_macro).length;
                const regressions = withMetrics.filter((e, i) => i > 0 && e.metrics!.f1_macro < withMetrics[i-1].metrics!.f1_macro).length;
                return (
                  <div className="flex flex-col gap-2">
                    {[
                      { label: 'Experiments', value: withMetrics.length },
                      { label: 'Improvements', value: improvements, color: 'text-emerald-600' },
                      { label: 'Regressions', value: regressions, color: 'text-orange-500' },
                      { label: 'Total Gain', value: `+${(Math.max(...scores) - scores[0]).toFixed(4)}`, color: 'text-indigo-600' },
                    ].map(({ label, value, color }) => (
                      <div key={label} className="flex items-center justify-between text-sm">
                        <span className="text-gray-500">{label}</span>
                        <span className={`font-bold font-mono ${color ?? 'text-gray-800'}`}>{value}</span>
                      </div>
                    ))}
                  </div>
                );
              })()}
            </div>
          )}

          {/* Strategy */}
          <div className="card p-4">
            <div className="text-xs font-bold text-gray-400 uppercase tracking-widest mb-3">Strategy</div>
            <div className={`text-sm font-bold px-3 py-2 rounded-xl ${mission.config.aiGuided ? 'bg-violet-50 text-violet-700 border border-violet-200' : 'bg-indigo-50 text-indigo-700 border border-indigo-200'}`}>
              {mission.config.aiGuided ? '✦ AI Guided' : '⬡ ATLAS Rules'}
            </div>
            <p className="text-xs text-gray-400 mt-2">
              {mission.config.aiGuided
                ? 'An LLM evaluates each result and proposes the next experiment.'
                : 'Deterministic rules guide hypothesis generation and decisions.'}
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}
