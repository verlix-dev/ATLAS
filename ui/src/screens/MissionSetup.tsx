import { useCallback, useEffect, useRef, useState } from 'react';
import { fetchDatasets, fetchPlan, uploadDataset, type StartMissionBody } from '../lib/api';
import type { DatasetListing, MissionPlan } from '../types';

interface Props {
  /** Resolves to the real mission id, or null when the backend refused it. */
  onStart: (body: StartMissionBody) => Promise<string | null>;
  starting: boolean;
  startError: string | null;
}

const EXAMPLE_TASKS = [
  'Predict whether a customer will churn. Missing a churner is more costly than a false alarm.',
  'Detect fraudulent transactions. Missing real fraud is far more costly than a false alarm.',
];

/** Shortest task the Planner is asked to read. One constant, so the gate that
 *  starts planning and the gate that enables Start can never disagree. */
const MIN_TASK_CHARS = 10;

/**
 * "ATLAS Understands" — rendered from the real Planner, never from heuristics.
 * While the task is empty or being planned the panel says so, rather than
 * inventing a metric.
 */
function AtlasUnderstandsPanel({
  plan,
  status,
  maxExp,
  aiGuided,
  target,
}: {
  plan: MissionPlan | null;
  status: 'idle' | 'planning' | 'ready' | 'error';
  maxExp: number;
  aiGuided: boolean;
  target: string;
}) {
  if (status === 'idle') {
    return (
      <div className="card-elevated p-6">
        <div className="text-xs font-semibold tracking-widest uppercase text-gray-400 mb-3">
          ATLAS Understands
        </div>
        <p className="text-sm text-gray-400">
          Describe a mission objective to see how the Planner interprets it.
        </p>
      </div>
    );
  }

  if (status === 'planning') {
    return (
      <div className="card-elevated p-6">
        <div className="text-xs font-semibold tracking-widest uppercase text-gray-400 mb-3">
          ATLAS Understands
        </div>
        <p className="text-sm text-gray-400">Reading the objective…</p>
      </div>
    );
  }

  if (status === 'error' || !plan) {
    return (
      <div className="card-elevated p-6">
        <div className="text-xs font-semibold tracking-widest uppercase text-gray-400 mb-3">
          ATLAS Understands
        </div>
        <p className="text-sm text-orange-500">
          The Planner could not interpret this objective. Try describing the problem more concretely.
        </p>
      </div>
    );
  }

  const facts = [
    { label: 'Problem type', value: plan.problem_type ?? 'not stated' },
    { label: 'Target', value: plan.target_candidate ?? `auto (${target || 'inferred'})` },
    { label: 'Primary metric', value: plan.primary_metric.replace(/_/g, ' ') },
    { label: 'Priority', value: plan.priority ?? 'none stated' },
    { label: 'Budget', value: `≤ ${plan.experiment_budget ?? maxExp} experiments` },
    { label: 'Strategy', value: aiGuided ? 'AI Guided' : 'Deterministic Rules' },
  ];

  return (
    <div className="card-elevated p-6 animate-in">
      <div className="flex items-center gap-2 mb-5">
        <div className="w-2 h-2 rounded-full bg-emerald-500" />
        <span className="text-xs font-semibold tracking-widest uppercase text-emerald-600">
          ATLAS Understands
        </span>
        <span className="ml-auto text-[10px] font-mono text-gray-400 uppercase">
          {plan.metric_confidence}
        </span>
      </div>

      <div className="grid grid-cols-2 gap-4">
        {facts.map(({ label, value }) => (
          <div key={label}>
            <div className="text-xs font-medium text-gray-400 uppercase tracking-wider mb-1">
              {label}
            </div>
            <div className="text-sm font-semibold text-gray-800 font-mono">{value}</div>
          </div>
        ))}
      </div>

      {/* The Planner's own justification, verbatim. */}
      <div className="mt-5 pt-4 border-t border-indigo-50">
        <div className="text-xs font-medium text-gray-400 uppercase tracking-wider mb-1">
          Why this metric
        </div>
        <p className="text-xs text-gray-600 leading-relaxed">{plan.metric_reason}</p>
      </div>

      {plan.constraints.length > 0 && (
        <div className="mt-4">
          <div className="text-xs font-medium text-gray-400 uppercase tracking-wider mb-1">
            Constraints
          </div>
          <ul className="text-xs text-gray-600 space-y-1">
            {plan.constraints.map((c) => (
              <li key={c}>— {c}</li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}

export default function MissionSetup({ onStart, starting, startError }: Props) {
  const [datasets, setDatasets] = useState<DatasetListing[]>([]);
  const [datasetError, setDatasetError] = useState<string | null>(null);
  const [selected, setSelected] = useState<string>('');
  const [task, setTask] = useState('');
  const [maxExperiments, setMaxExperiments] = useState(6);
  const [aiGuided, setAiGuided] = useState(false);
  const [target, setTarget] = useState('');
  const [plan, setPlan] = useState<MissionPlan | null>(null);
  const [planStatus, setPlanStatus] = useState<'idle' | 'planning' | 'ready' | 'error'>('idle');
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [uploadError, setUploadError] = useState<string | null>(null);
  const [uploaded, setUploaded] = useState<DatasetListing | null>(null);
  const fileInput = useRef<HTMLInputElement>(null);
  const planTimer = useRef<ReturnType<typeof setTimeout> | null>(null);

  // The dataset list is whatever the backend actually has on disk.
  useEffect(() => {
    let cancelled = false;
    fetchDatasets()
      .then((rows) => {
        if (cancelled) return;
        setDatasets(rows);
        if (rows.length) setSelected(rows[0].csv);
      })
      .catch((cause: unknown) => {
        if (!cancelled) {
          setDatasetError(cause instanceof Error ? cause.message : String(cause));
        }
      });
    return () => {
      cancelled = true;
    };
  }, []);

  // Debounced planner call. Every displayed value comes from the response.
  useEffect(() => {
    if (planTimer.current) clearTimeout(planTimer.current);
    const trimmed = task.trim();
    if (trimmed.length < MIN_TASK_CHARS) {
      setPlan(null);
      setPlanStatus('idle');
      return;
    }
    setPlanStatus('planning');
    planTimer.current = setTimeout(() => {
      fetchPlan(trimmed)
        .then((result) => {
          setPlan(result);
          setPlanStatus(result ? 'ready' : 'error');
          if (result?.experiment_budget) setMaxExperiments(result.experiment_budget);
        })
        .catch(() => {
          setPlan(null);
          setPlanStatus('error');
        });
    }, 350);
    return () => {
      if (planTimer.current) clearTimeout(planTimer.current);
    };
  }, [task]);

  const activeDataset = datasets.find((d) => d.csv === selected) ?? null;
  const canStart =
    Boolean(activeDataset) && task.trim().length >= MIN_TASK_CHARS && planStatus !== 'planning';

  /**
   * Upload, then select. The new row comes straight from the backend response,
   * so the list shows what the server actually stored — nothing is optimistically
   * invented client-side.
   */
  const handleFile = useCallback(async (file: File | undefined) => {
    if (!file) return;
    setUploadError(null);
    setUploaded(null);
    setUploading(true);
    try {
      const dataset = await uploadDataset(file);
      setDatasets((rows) => [...rows.filter((r) => r.csv !== dataset.csv), dataset]);
      setSelected(dataset.csv);
      setUploaded(dataset);
    } catch (cause) {
      setUploadError(cause instanceof Error ? cause.message : String(cause));
    } finally {
      setUploading(false);
      // Let the same file be picked again after a failure.
      if (fileInput.current) fileInput.current.value = '';
    }
  }, []);

  const handleStart = useCallback(() => {
    if (!activeDataset || !canStart) return;
    void onStart({
      csv: activeDataset.csv,
      target: target.trim() || null,
      task: task.trim(),
      budget: maxExperiments,
      // The backend field is `llm`; "offline" selects the LLM-guided proposer.
      llm: aiGuided ? 'offline' : null,
    });
  }, [activeDataset, canStart, onStart, target, task, maxExperiments, aiGuided]);

  return (
    <div className="min-h-screen" style={{ background: 'linear-gradient(160deg, #F4F6FF 0%, #EEF1FF 40%, #F8F6FF 100%)' }}>
      <header className="border-b border-indigo-100" style={{ background: 'rgba(255,255,255,0.7)', backdropFilter: 'blur(12px)', position: 'sticky', top: 0, zIndex: 50 }}>
        <div className="max-w-6xl mx-auto px-6 h-14 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-8 h-8 rounded-lg gradient-primary flex items-center justify-center">
              <svg width="16" height="16" viewBox="0 0 16 16" fill="none">
                <path d="M8 2L14 5.5V10.5L8 14L2 10.5V5.5L8 2Z" stroke="white" strokeWidth="1.5" strokeLinejoin="round" />
                <circle cx="8" cy="8" r="2" fill="white" />
              </svg>
            </div>
            <span className="font-bold text-gray-900 tracking-tight">ATLAS</span>
            <span className="text-xs text-gray-400 font-medium">Autonomous Training, Learning &amp; Analytics System</span>
          </div>
        </div>
      </header>

      <div className="max-w-6xl mx-auto px-6 py-12">
        <div className="mb-12 text-center">
          <div className="inline-flex items-center gap-2 bg-indigo-50 border border-indigo-100 rounded-full px-4 py-1.5 mb-6">
            <div className="w-1.5 h-1.5 rounded-full bg-indigo-500" />
            <span className="text-xs font-semibold text-indigo-600 tracking-wide">NEW MISSION</span>
          </div>
          <h1 className="text-5xl font-extrabold text-gray-900 mb-4 leading-tight tracking-tight">
            Configure your<br />
            <span className="gradient-text">ML Mission</span>
          </h1>
          <p className="text-gray-500 text-lg max-w-xl mx-auto font-medium">
            ATLAS will autonomously run experiments, learn from results, and decide what to try next.
          </p>
        </div>

        <div className="grid grid-cols-5 gap-8">
          <div className="col-span-3 flex flex-col gap-6">
            {/* 01 — Dataset */}
            <div className="card p-6">
              <h2 className="text-sm font-bold text-gray-700 uppercase tracking-widest mb-5">01 — Dataset</h2>

              {datasetError && (
                <div className="mb-4 p-4 rounded-xl bg-orange-50 border border-orange-200 text-xs text-orange-700">
                  Could not reach the backend: {datasetError}
                </div>
              )}

              {/* Real datasets from GET /api/datasets — the repo's demo CSVs
                  plus anything uploaded, both discovered server-side. */}
              <div className="grid gap-2">
                {datasets.map((ds) => (
                  <button
                    key={ds.csv}
                    onClick={() => setSelected(ds.csv)}
                    className={`p-3 rounded-xl border text-left transition-all duration-150 text-xs flex items-center gap-3 ${selected === ds.csv
                      ? 'border-indigo-400 bg-indigo-50 text-indigo-700'
                      : 'border-gray-100 hover:border-indigo-200 hover:bg-indigo-50/30 text-gray-600'}`}
                  >
                    <span className="font-mono font-semibold">{ds.name}</span>
                    {ds.source === 'upload' && (
                      <span className="text-[10px] font-bold uppercase tracking-wider bg-violet-100 text-violet-600 px-1.5 py-0.5 rounded">
                        yours
                      </span>
                    )}
                    <span className="ml-auto text-gray-400">
                      {(ds.bytes / 1024).toFixed(1)} KB
                    </span>
                  </button>
                ))}
                {!datasets.length && !datasetError && (
                  <div className="text-xs text-gray-400">Loading datasets…</div>
                )}
              </div>

              {/* Upload — the same list, one more row once it lands. */}
              <div className="mt-4 pt-4 border-t border-gray-100">
                <div className="text-xs font-semibold text-gray-500 mb-2">Upload your own CSV</div>
                <input
                  ref={fileInput}
                  type="file"
                  accept=".csv,text/csv"
                  className="hidden"
                  onChange={(e) => void handleFile(e.target.files?.[0])}
                />
                <button
                  onClick={() => fileInput.current?.click()}
                  disabled={uploading}
                  className={`text-xs font-semibold px-3 py-2 rounded-xl border transition-colors ${uploading
                    ? 'border-gray-100 text-gray-400 cursor-not-allowed'
                    : 'border-indigo-200 text-indigo-600 hover:bg-indigo-50'}`}
                >
                  {uploading ? 'Uploading…' : 'Browse…'}
                </button>

                {uploaded && !uploadError && (
                  <div className="mt-2 text-xs text-emerald-600 font-medium">
                    ✓ {uploaded.name} uploaded
                    {uploaded.rows !== undefined && uploaded.columns !== undefined && (
                      <span className="text-gray-400 font-normal">
                        {' '}
                        — {uploaded.rows} rows × {uploaded.columns} columns
                      </span>
                    )}
                  </div>
                )}
                {uploadError && (
                  <div className="mt-2 p-3 rounded-xl bg-orange-50 border border-orange-200 text-xs text-orange-700">
                    {uploadError}
                  </div>
                )}
              </div>

              {/* The Data Engineer profiles the dataset at mission start; the
                  real profile arrives in the mission_start event. */}
              <p className="mt-4 text-xs text-gray-400 leading-relaxed">
                ATLAS profiles the dataset with its Data Engineer when the mission starts —
                rows, dtypes, missingness and class balance come from the file itself.
              </p>
            </div>

            {/* 02 — Objective */}
            <div className="card p-6">
              <h2 className="text-sm font-bold text-gray-700 uppercase tracking-widest mb-2">02 — Mission Objective</h2>
              <p className="text-xs text-gray-400 mb-4 font-medium">
                Describe what you want ATLAS to accomplish in natural language
              </p>
              <textarea
                value={task}
                onChange={(e) => setTask(e.target.value)}
                placeholder="Predict whether a customer will churn. Missing a churner is more costly than a false alarm."
                rows={4}
                className="w-full rounded-xl border border-indigo-100 bg-indigo-50/30 px-4 py-3 text-sm text-gray-800 placeholder:text-gray-300 resize-none focus:outline-none focus:ring-2 focus:ring-indigo-300 focus:border-indigo-300 transition-all"
              />
              <div className="flex gap-2 mt-2 flex-wrap">
                {EXAMPLE_TASKS.map((t) => (
                  <button
                    key={t}
                    onClick={() => setTask(t)}
                    className="text-xs text-indigo-500 bg-indigo-50 hover:bg-indigo-100 px-2 py-1 rounded-lg transition-colors truncate max-w-[220px]"
                  >
                    {t.slice(0, 42)}…
                  </button>
                ))}
              </div>
            </div>

            {/* 03 — Configuration */}
            <div className="card p-6">
              <h2 className="text-sm font-bold text-gray-700 uppercase tracking-widest mb-5">03 — Configuration</h2>
              <div className="grid grid-cols-2 gap-5">
                <div>
                  <label className="text-xs font-semibold text-gray-500 mb-2 block">Target Column</label>
                  <input
                    value={target}
                    onChange={(e) => setTarget(e.target.value)}
                    placeholder={plan?.target_candidate ?? 'inferred by Data Engineer'}
                    className="w-full rounded-xl border border-gray-100 bg-gray-50/50 px-3 py-2.5 text-sm text-gray-700 font-mono focus:outline-none focus:ring-2 focus:ring-indigo-300"
                  />
                  <p className="text-xs text-gray-400 mt-1">
                    Leave blank to use the Planner's candidate, or type the exact column name.
                  </p>
                </div>
                <div>
                  <label className="text-xs font-semibold text-gray-500 mb-2 block">
                    Max Experiments <span className="text-gray-400 font-normal">(budget)</span>
                  </label>
                  <div className="flex items-center gap-3">
                    <input
                      type="range" min={1} max={20} value={maxExperiments}
                      onChange={(e) => setMaxExperiments(Number(e.target.value))}
                      className="flex-1 accent-indigo-600"
                    />
                    <span className="font-mono font-bold text-indigo-700 w-6 text-center">{maxExperiments}</span>
                  </div>
                  <p className="text-xs text-gray-400 mt-1">A ceiling — ATLAS may stop earlier</p>
                </div>
              </div>

              <div className="mt-5 flex items-center justify-between p-4 rounded-xl bg-gray-50/50 border border-gray-100">
                <div>
                  <div className="text-sm font-semibold text-gray-700">LLM Guidance</div>
                  <div className="text-xs text-gray-400 mt-0.5">
                    An offline evidence-led proposer runs behind the same validator as the deterministic ladder
                  </div>
                </div>
                <button
                  onClick={() => setAiGuided(!aiGuided)}
                  aria-pressed={aiGuided}
                  aria-label="Toggle LLM guidance"
                  className={`relative w-12 h-6 rounded-full transition-all duration-200 ${aiGuided ? 'bg-indigo-500' : 'bg-gray-200'}`}
                >
                  <div className={`absolute w-5 h-5 rounded-full bg-white shadow top-0.5 transition-all duration-200 ${aiGuided ? 'left-6' : 'left-0.5'}`} />
                </button>
              </div>

              <button
                onClick={() => setShowAdvanced(!showAdvanced)}
                className="text-xs text-gray-400 hover:text-indigo-500 mt-4 flex items-center gap-1 transition-colors"
              >
                <svg width="12" height="12" viewBox="0 0 12 12" fill="none" style={{ transform: showAdvanced ? 'rotate(180deg)' : 'none', transition: 'transform 0.2s' }}>
                  <path d="M2 4l4 4 4-4" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" />
                </svg>
                Advanced configuration
              </button>
              {showAdvanced && (
                <div className="mt-3 p-4 rounded-xl bg-gray-50/50 border border-gray-100 text-xs text-gray-500 animate-in">
                  ATLAS currently supports three model families — logistic regression, random forest
                  and histogram gradient boosting — with median/mode imputation, optional scaling and
                  one-hot encoding. Model choice and hyperparameters are decided by the hypothesis engine.
                </div>
              )}
            </div>
          </div>

          {/* Right — planner + CTA */}
          <div className="col-span-2 flex flex-col gap-6">
            <AtlasUnderstandsPanel
              plan={plan}
              status={planStatus}
              maxExp={maxExperiments}
              aiGuided={aiGuided}
              target={target}
            />

            <div className="card p-6">
              <h3 className="text-xs font-bold text-gray-500 uppercase tracking-widest mb-4">How ATLAS Works</h3>
              {['Experiment', 'Measure', 'Diagnose', 'Hypothesize', 'Decide'].map((step, i) => (
                <div key={step} className="flex items-start gap-3 mb-3 last:mb-0">
                  <div className="w-5 h-5 rounded-full border-2 border-indigo-200 bg-indigo-50 flex items-center justify-center flex-shrink-0 mt-0.5">
                    <span className="text-[10px] font-bold text-indigo-500">{i + 1}</span>
                  </div>
                  <div>
                    <div className="text-sm font-semibold text-gray-700">{step}</div>
                    <div className="text-xs text-gray-400 mt-0.5">
                      {[
                        'Train a model on the dataset',
                        'Evaluate metrics on the held-out split',
                        'Classify what actually happened',
                        'Propose the next configuration',
                        'Continue, Revise, or Stop',
                      ][i]}
                    </div>
                  </div>
                </div>
              ))}
            </div>

            <button
              onClick={handleStart}
              disabled={!canStart || starting}
              className={`w-full rounded-2xl py-5 font-bold text-base tracking-wide text-white transition-all duration-300 relative overflow-hidden group ${canStart && !starting
                ? 'gradient-primary hover:shadow-lg hover:shadow-indigo-200 hover:-translate-y-0.5 active:translate-y-0'
                : 'bg-gray-200 text-gray-400 cursor-not-allowed'}`}
            >
              {starting ? (
                <span className="flex items-center justify-center gap-2">
                  <svg className="animate-spin w-4 h-4" viewBox="0 0 16 16" fill="none">
                    <circle cx="8" cy="8" r="6" stroke="white" strokeWidth="2" strokeOpacity="0.3" />
                    <path d="M8 2a6 6 0 016 6" stroke="white" strokeWidth="2" strokeLinecap="round" />
                  </svg>
                  ATLAS is taking control…
                </span>
              ) : (
                <span className="flex items-center justify-center gap-2">
                  START MISSION
                  <svg width="16" height="16" viewBox="0 0 16 16" fill="none">
                    <path d="M3 8h10M9 4l4 4-4 4" stroke="white" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
                  </svg>
                </span>
              )}
            </button>

            {startError && (
              <div className="p-4 rounded-xl bg-red-50 border border-red-200 text-xs text-red-700">
                {startError}
              </div>
            )}
            {!canStart && !startError && (
              <p className="text-xs text-gray-400 text-center">
                {!activeDataset
                  ? 'Select a dataset to continue'
                  : planStatus === 'planning'
                    ? 'Reading your objective…'
                    : 'Describe your mission objective above'}
              </p>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
