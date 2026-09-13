import { deterministicExperiments, aiGuidedExperiments } from '../data/demoData';
import ComparisonChart from '../components/ComparisonChart';

interface Props {
  onRestart: () => void;
}

const det = deterministicExperiments;
const ai = aiGuidedExperiments;
const detBest = Math.max(...det.map(e => e.metrics!.f1_macro));
const aiBest = Math.max(...ai.map(e => e.metrics!.f1_macro));
const detWins = detBest > aiBest;

function BestModelCard({ label, experiments, accentColor, borderColor, isDet }: {
  label: string;
  experiments: typeof det;
  accentColor: string;
  borderColor: string;
  isDet: boolean;
}) {
  const best = experiments.reduce((b, e) => e.metrics!.f1_macro > b.metrics!.f1_macro ? e : b);
  const scores = experiments.map(e => e.metrics!.f1_macro);
  const winner = isDet ? detWins : !detWins;

  return (
    <div className={`card-elevated flex-1 overflow-hidden ${winner ? 'ring-2 ring-offset-2' : ''}`}
      style={{ ringColor: accentColor }}>
      {winner && (
        <div className="px-4 py-2 text-xs font-bold text-white text-center" style={{ background: accentColor }}>
          ★ WINNER — Higher Final Score
        </div>
      )}
      <div className="p-6">
        <div className="flex items-start justify-between mb-4">
          <div>
            <div className="text-xs font-bold uppercase tracking-widest mb-1" style={{ color: accentColor }}>{label}</div>
            <div className="text-2xl font-extrabold font-mono text-gray-900">{(isDet ? detBest : aiBest).toFixed(4)}</div>
            <div className="text-xs text-gray-400 mt-0.5">F1 Macro</div>
          </div>
          <div className="text-right">
            <div className="text-xs text-gray-400 mb-1">Best Model</div>
            <div className="text-sm font-bold text-gray-700">{best.model}</div>
          </div>
        </div>

        {/* Score bars */}
        <div className="flex flex-col gap-2 mb-4">
          {experiments.map(e => {
            const w = ((e.metrics!.f1_macro - 0.45) / (0.75 - 0.45)) * 100;
            const isBest = e.metrics!.f1_macro === (isDet ? detBest : aiBest);
            return (
              <div key={e.id} className="flex items-center gap-2">
                <span className="font-mono text-xs text-gray-400 w-7">E{e.id}</span>
                <div className="flex-1 h-1.5 bg-gray-100 rounded-full overflow-hidden">
                  <div
                    className="h-full rounded-full transition-all duration-1000"
                    style={{ width: `${w}%`, background: isBest ? accentColor : borderColor }}
                  />
                </div>
                <span className={`font-mono text-xs font-bold ${isBest ? 'text-gray-900' : 'text-gray-500'}`}>
                  {e.metrics!.f1_macro.toFixed(4)}
                </span>
              </div>
            );
          })}
        </div>

        <div className="grid grid-cols-2 gap-3 pt-4 border-t border-gray-100 text-xs">
          {[
            { label: 'Experiments', value: experiments.length },
            { label: 'Improvement', value: `+${(Math.max(...scores) - scores[0]).toFixed(4)}` },
            { label: 'Regressions', value: experiments.filter((e, i) => i > 0 && e.metrics!.f1_macro < experiments[i-1].metrics!.f1_macro).length },
            { label: 'Stop Reason', value: experiments[experiments.length - 1].decision },
          ].map(({ label, value }) => (
            <div key={label}>
              <div className="text-gray-400 mb-0.5">{label}</div>
              <div className="font-bold text-gray-800">{value}</div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

function ExperimentRow({ det, ai }: { det: typeof deterministicExperiments[0]; ai: typeof aiGuidedExperiments[0] }) {
  const detScore = det.metrics!.f1_macro;
  const aiScore = ai.metrics!.f1_macro;
  const detLeads = detScore > aiScore;

  return (
    <div className="grid grid-cols-[1fr_auto_1fr] gap-4 items-center py-4 border-b border-gray-100 last:border-0">
      {/* Det */}
      <div className={`card p-3 transition-all ${detLeads ? 'ring-1 ring-indigo-200' : ''}`}>
        <div className="text-xs font-bold text-indigo-500 mb-1">{det.model}</div>
        <div className="font-mono font-extrabold text-gray-900">{detScore.toFixed(4)}</div>
        <div className="text-xs text-gray-400 mt-1">{det.decision}</div>
      </div>

      {/* Spacer / label */}
      <div className="flex flex-col items-center gap-1">
        <span className="font-mono text-xs font-bold text-gray-400">E{det.id}</span>
        <div className="text-xs font-bold" style={{ color: detLeads ? '#4F6AF7' : '#7C3AED' }}>
          {detLeads ? '◀ Det' : 'AI ▶'}
        </div>
        <div className="text-xs font-mono text-gray-400">
          Δ {Math.abs(detScore - aiScore).toFixed(4)}
        </div>
      </div>

      {/* AI */}
      <div className={`card p-3 transition-all ${!detLeads ? 'ring-1 ring-violet-200' : ''}`}>
        <div className="text-xs font-bold text-violet-500 mb-1">{ai.model}</div>
        <div className="font-mono font-extrabold text-gray-900">{aiScore.toFixed(4)}</div>
        <div className="text-xs text-gray-400 mt-1">{ai.decision}</div>
      </div>
    </div>
  );
}

export default function Comparison({ onRestart }: Props) {
  const delta = Math.abs(detBest - aiBest);
  const winner = detWins ? 'Deterministic' : 'AI Guided';
  const winnerScore = detWins ? detBest : aiBest;
  const loserScore = detWins ? aiBest : detBest;

  return (
    <div className="min-h-screen" style={{ background: 'linear-gradient(160deg, #F4F6FF 0%, #EEF1FF 40%, #F8F6FF 100%)' }}>
      {/* Header */}
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
        {/* Hero verdict */}
        <div className="text-center mb-12">
          <div className="text-xs font-bold text-gray-400 uppercase tracking-widest mb-3">Which Strategy Learned Better?</div>
          <h1 className="text-4xl font-extrabold text-gray-900 mb-2">
            <span className="gradient-text">{winner}</span> Strategy Won
          </h1>
          <div className="flex items-center justify-center gap-10 mt-6">
            <div className="text-center">
              <div className="text-xs font-bold text-indigo-500 uppercase tracking-widest mb-1">Deterministic</div>
              <div className={`text-5xl font-extrabold font-mono ${detWins ? 'text-gray-900' : 'text-gray-400'}`}>{detBest.toFixed(4)}</div>
            </div>
            <div className="flex flex-col items-center gap-1">
              <div className="text-2xl text-gray-300">vs</div>
              <div className="text-sm font-bold text-gray-500 font-mono">
                Δ {detWins ? '-' : '+'}{delta.toFixed(4)}
              </div>
            </div>
            <div className="text-center">
              <div className="text-xs font-bold text-violet-500 uppercase tracking-widest mb-1">AI Guided</div>
              <div className={`text-5xl font-extrabold font-mono ${!detWins ? 'text-gray-900' : 'text-gray-400'}`}>{aiBest.toFixed(4)}</div>
            </div>
          </div>

          {/* Honest result note */}
          <div className="mt-6 inline-flex items-start gap-2 bg-amber-50 border border-amber-100 rounded-2xl px-5 py-3 text-left max-w-xl">
            <svg width="16" height="16" viewBox="0 0 16 16" fill="none" className="mt-0.5 flex-shrink-0">
              <path d="M8 2L14.9 14H1.1L8 2Z" stroke="#F59E0B" strokeWidth="1.5" strokeLinejoin="round" />
              <path d="M8 6v3M8 11v1" stroke="#F59E0B" strokeWidth="1.5" strokeLinecap="round" />
            </svg>
            <p className="text-sm text-amber-800 font-medium">
              ATLAS reports results honestly. {winner} achieved a higher final F1 ({winnerScore.toFixed(4)} vs {loserScore.toFixed(4)}), a difference of {delta.toFixed(4)}. Neither strategy is universally superior — strategies may perform differently on other datasets.
            </p>
          </div>
        </div>

        {/* Trajectory chart */}
        <div className="card-elevated p-6 mb-8">
          <div className="mb-1">
            <div className="text-xs font-bold text-gray-400 uppercase tracking-widest mb-1">Learning Trajectories</div>
            <div className="text-sm font-bold text-gray-700">How each strategy navigated the experiment space</div>
          </div>
          <ComparisonChart deterministic={det} aiGuided={ai} />
        </div>

        {/* Side-by-side model cards */}
        <div className="flex gap-5 mb-8">
          <BestModelCard
            label="Deterministic ATLAS"
            experiments={det}
            accentColor="#4F6AF7"
            borderColor="rgba(79,106,247,0.4)"
            isDet={true}
          />
          <BestModelCard
            label="AI Guided ATLAS"
            experiments={ai}
            accentColor="#7C3AED"
            borderColor="rgba(124,58,237,0.4)"
            isDet={false}
          />
        </div>

        {/* Experiment-by-experiment comparison */}
        <div className="card-elevated p-6 mb-8">
          <div className="text-xs font-bold text-gray-400 uppercase tracking-widest mb-5">Experiment-by-Experiment</div>
          <div className="grid grid-cols-[1fr_auto_1fr] gap-4 mb-2">
            <div className="text-xs font-bold text-indigo-500 uppercase tracking-wider">Deterministic</div>
            <div className="w-16" />
            <div className="text-xs font-bold text-violet-500 uppercase tracking-wider text-right">AI Guided</div>
          </div>
          {det.map((d, i) => (
            <ExperimentRow key={d.id} det={d} ai={ai[i]} />
          ))}
        </div>

        {/* Summary table */}
        <div className="card-elevated p-6">
          <div className="text-xs font-bold text-gray-400 uppercase tracking-widest mb-5">Summary</div>
          <div className="grid grid-cols-4 gap-4">
            {[
              {
                metric: 'Best Score',
                det: detBest.toFixed(4),
                ai: aiBest.toFixed(4),
                winner: detWins ? 'det' : 'ai',
              },
              {
                metric: 'Experiments Run',
                det: det.length,
                ai: ai.length,
                winner: det.length <= ai.length ? 'det' : 'ai',
                note: 'fewer is efficient',
              },
              {
                metric: 'Total Improvement',
                det: `+${(detBest - det[0].metrics!.f1_macro).toFixed(4)}`,
                ai: `+${(aiBest - ai[0].metrics!.f1_macro).toFixed(4)}`,
                winner: (detBest - det[0].metrics!.f1_macro) > (aiBest - ai[0].metrics!.f1_macro) ? 'det' : 'ai',
              },
              {
                metric: 'Regressions',
                det: det.filter((e, i) => i > 0 && e.metrics!.f1_macro < det[i-1].metrics!.f1_macro).length,
                ai: ai.filter((e, i) => i > 0 && e.metrics!.f1_macro < ai[i-1].metrics!.f1_macro).length,
                winner: det.filter((e, i) => i > 0 && e.metrics!.f1_macro < det[i-1].metrics!.f1_macro).length <=
                        ai.filter((e, i) => i > 0 && e.metrics!.f1_macro < ai[i-1].metrics!.f1_macro).length ? 'det' : 'ai',
                note: 'fewer is better',
              },
              {
                metric: 'Novel Configs Tried',
                det: new Set(det.map(e => e.model)).size,
                ai: new Set(ai.map(e => e.model)).size,
                winner: new Set(ai.map(e => e.model)).size >= new Set(det.map(e => e.model)).size ? 'ai' : 'det',
              },
              {
                metric: 'Hypothesis Source',
                det: 'ATLAS Rules',
                ai: 'LLM Reasoning',
                winner: null,
              },
              {
                metric: 'Final Decision',
                det: det[det.length - 1].decision,
                ai: ai[ai.length - 1].decision,
                winner: null,
              },
              {
                metric: 'Stop Reason',
                det: det[det.length - 1].decisionReason?.slice(0, 28) ?? 'N/A',
                ai: ai[ai.length - 1].decisionReason?.slice(0, 28) ?? 'N/A',
                winner: null,
              },
            ].map(({ metric, det: dv, ai: av, winner: w, note }) => (
              <div key={metric} className="p-4 rounded-xl bg-gray-50/60 border border-gray-100">
                <div className="text-xs text-gray-400 font-semibold mb-2">{metric}</div>
                {note && <div className="text-[10px] text-gray-300 mb-2 italic">{note}</div>}
                <div className="flex gap-2">
                  <div className={`flex-1 text-xs font-bold text-center py-1.5 rounded-lg ${w === 'det' ? 'bg-indigo-100 text-indigo-700' : 'text-gray-600'}`}>
                    {dv}
                  </div>
                  <div className={`flex-1 text-xs font-bold text-center py-1.5 rounded-lg ${w === 'ai' ? 'bg-violet-100 text-violet-700' : 'text-gray-600'}`}>
                    {av}
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}
