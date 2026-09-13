import { useState, useRef, useCallback } from 'react';
import type { MissionConfig } from '../types';

interface Props {
  onStart: (config: MissionConfig) => void;
}

const EXAMPLE_DATASETS = [
  { name: 'churn.csv', rows: 400, columns: 21, target: 'churned', missing: '2.1%', classes: { 'Not Churned': '72%', 'Churned': '28%' } },
  { name: 'fraud_transactions.csv', rows: 1240, columns: 31, target: 'is_fraud', missing: '0.4%', classes: { 'Legitimate': '89%', 'Fraud': '11%' } },
  { name: 'employee_attrition.csv', rows: 1470, columns: 35, target: 'attrition', missing: '0.0%', classes: { 'No': '84%', 'Yes': '16%' } },
];

const EXAMPLE_TASKS = [
  "Predict whether a customer will churn. Prioritize catching potential churners while maintaining reasonable precision.",
  "Detect fraudulent transactions. Missing real fraud is far more costly than false alarms.",
  "Identify employees at risk of leaving. Focus on recall to ensure at-risk employees aren't missed.",
];

function AtlasUnderstandsPanel({ task, maxExp, aiGuided, dataset }: { task: string; maxExp: number; aiGuided: boolean; dataset: { name: string; target: string } | null }) {
  if (!task.trim() && !dataset) return null;

  const problemType = task.toLowerCase().includes('fraud') ? 'Binary Classification' :
    task.toLowerCase().includes('churn') ? 'Binary Classification' :
    task.toLowerCase().includes('predict') ? 'Binary Classification' : 'Binary Classification';
  const target = dataset?.target || 'auto-detected';
  const metric = task.toLowerCase().includes('recall') || task.toLowerCase().includes('miss') ? 'F1 Macro' : 'F1 Macro';
  const priority = task.toLowerCase().includes('recall') || task.toLowerCase().includes('miss') || task.toLowerCase().includes('catching') ?
    'Minimize false negatives' : 'Balance precision/recall';

  return (
    <div className="card-elevated p-6 animate-in" style={{ animationDelay: '0.1s' }}>
      <div className="flex items-center gap-2 mb-5">
        <div className="w-2 h-2 rounded-full bg-emerald-500" style={{ animation: 'pulse-ring 2s infinite' }} />
        <span className="text-xs font-semibold tracking-widest uppercase text-emerald-600">ATLAS Understands</span>
      </div>
      <div className="grid grid-cols-2 gap-4">
        {[
          { label: 'Problem', value: problemType },
          { label: 'Target', value: <span className="font-mono text-sm text-indigo-600">{target}</span> },
          { label: 'Primary Metric', value: metric },
          { label: 'Priority', value: priority },
          { label: 'Experiment Budget', value: `≤ ${maxExp} experiments` },
          { label: 'Strategy', value: aiGuided ? 'AI Guided' : 'Deterministic Rules' },
        ].map(({ label, value }) => (
          <div key={label}>
            <div className="text-xs font-medium text-gray-400 uppercase tracking-wider mb-1">{label}</div>
            <div className="text-sm font-semibold text-gray-800">{value}</div>
          </div>
        ))}
      </div>
    </div>
  );
}

export default function MissionSetup({ onStart }: Props) {
  const [dragActive, setDragActive] = useState(false);
  const [uploadedFile, setUploadedFile] = useState<typeof EXAMPLE_DATASETS[0] | null>(null);
  const [selectedDataset, setSelectedDataset] = useState<string>('');
  const [task, setTask] = useState('');
  const [maxExperiments, setMaxExperiments] = useState(6);
  const [aiGuided, setAiGuided] = useState(true);
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [target, setTarget] = useState('auto');
  const [launching, setLaunching] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const activeDataset = uploadedFile || EXAMPLE_DATASETS.find(d => d.name === selectedDataset) || null;

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setDragActive(false);
    const file = e.dataTransfer.files[0];
    if (file && file.name.endsWith('.csv')) {
      setUploadedFile({ name: file.name, rows: 400, columns: 21, target: 'churned', missing: '2.1%', classes: { 'Not Churned': '72%', 'Churned': '28%' } });
      setSelectedDataset('');
    }
  }, []);

  const handleStart = () => {
    if (!activeDataset || !task.trim()) return;
    setLaunching(true);
    setTimeout(() => {
      onStart({
        task,
        dataset: activeDataset.name,
        rows: activeDataset.rows,
        columns: activeDataset.columns,
        target: target === 'auto' ? activeDataset.target : target,
        maxExperiments,
        aiGuided,
        problemType: 'Binary Classification',
        primaryMetric: 'F1 Macro',
        priority: task.toLowerCase().includes('catching') || task.toLowerCase().includes('miss') ?
          'Minimize false negatives' : 'Balance precision/recall',
      });
    }, 600);
  };

  const canStart = !!activeDataset && task.trim().length > 10;

  return (
    <div className="min-h-screen" style={{ background: 'linear-gradient(160deg, #F4F6FF 0%, #EEF1FF 40%, #F8F6FF 100%)' }}>
      {/* Header */}
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
            <span className="text-xs text-gray-400 font-medium">Autonomous Training, Learning & Analytics System</span>
          </div>
          <div className="text-xs font-mono text-indigo-400 bg-indigo-50 px-3 py-1 rounded-full">v2.0</div>
        </div>
      </header>

      <div className="max-w-6xl mx-auto px-6 py-12">
        {/* Hero */}
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
          {/* Left — main config */}
          <div className="col-span-3 flex flex-col gap-6">

            {/* Dataset */}
            <div className="card p-6">
              <h2 className="text-sm font-bold text-gray-700 uppercase tracking-widest mb-5">01 — Dataset</h2>

              {/* Upload zone */}
              <div
                className={`border-2 border-dashed rounded-2xl p-8 text-center cursor-pointer transition-all duration-200 mb-4 ${dragActive ? 'drag-active border-indigo-500 bg-indigo-50/40' : 'border-indigo-100 hover:border-indigo-300 hover:bg-indigo-50/20'}`}
                onDragOver={(e) => { e.preventDefault(); setDragActive(true); }}
                onDragLeave={() => setDragActive(false)}
                onDrop={handleDrop}
                onClick={() => fileInputRef.current?.click()}
              >
                <input ref={fileInputRef} type="file" accept=".csv" className="hidden" onChange={(e) => {
                  const file = e.target.files?.[0];
                  if (file) {
                    setUploadedFile({ name: file.name, rows: 400, columns: 21, target: 'churned', missing: '2.1%', classes: { 'Not Churned': '72%', 'Churned': '28%' } });
                    setSelectedDataset('');
                  }
                }} />
                <div className="w-10 h-10 rounded-xl bg-indigo-50 border border-indigo-100 flex items-center justify-center mx-auto mb-3">
                  <svg width="20" height="20" viewBox="0 0 20 20" fill="none">
                    <path d="M10 2v10M6 6l4-4 4 4" stroke="#6366F1" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
                    <path d="M3 14v2a1 1 0 001 1h12a1 1 0 001-1v-2" stroke="#6366F1" strokeWidth="1.5" strokeLinecap="round" />
                  </svg>
                </div>
                <p className="text-sm font-semibold text-gray-600 mb-1">Drop your dataset here</p>
                <p className="text-xs text-gray-400">CSV files supported · or <span className="text-indigo-500 font-medium">Browse files</span></p>
              </div>

              {/* Or choose existing */}
              <div className="relative flex items-center gap-3 mb-4">
                <div className="flex-1 h-px bg-gray-100" />
                <span className="text-xs text-gray-400 font-medium">or choose existing</span>
                <div className="flex-1 h-px bg-gray-100" />
              </div>
              <div className="grid grid-cols-3 gap-2">
                {EXAMPLE_DATASETS.map(ds => (
                  <button
                    key={ds.name}
                    onClick={() => { setSelectedDataset(ds.name); setUploadedFile(null); }}
                    className={`p-3 rounded-xl border text-left transition-all duration-150 text-xs ${selectedDataset === ds.name
                      ? 'border-indigo-400 bg-indigo-50 text-indigo-700'
                      : 'border-gray-100 hover:border-indigo-200 hover:bg-indigo-50/30 text-gray-600'}`}
                  >
                    <div className="font-mono font-semibold truncate mb-1">{ds.name}</div>
                    <div className="text-gray-400">{ds.rows} rows · {ds.columns} cols</div>
                  </button>
                ))}
              </div>

              {/* Dataset preview */}
              {activeDataset && (
                <div className="mt-4 p-4 rounded-xl bg-indigo-50/50 border border-indigo-100 animate-in">
                  <div className="flex items-center gap-2 mb-3">
                    <svg width="14" height="14" viewBox="0 0 14 14" fill="none">
                      <path d="M7 1L13 4V10L7 13L1 10V4L7 1Z" stroke="#4F46E5" strokeWidth="1.2" />
                      <circle cx="7" cy="7" r="1.5" fill="#4F46E5" />
                    </svg>
                    <span className="text-xs font-bold text-indigo-700 uppercase tracking-wider">Dataset Analysis</span>
                  </div>
                  <div className="grid grid-cols-3 gap-3 text-xs">
                    {[
                      { label: 'Filename', value: <span className="font-mono">{activeDataset.name}</span> },
                      { label: 'Rows', value: activeDataset.rows.toLocaleString() },
                      { label: 'Columns', value: activeDataset.columns },
                      { label: 'Target', value: <span className="font-mono text-indigo-600">{activeDataset.target}</span> },
                      { label: 'Missing', value: activeDataset.missing },
                      { label: 'Status', value: <span className="text-emerald-600 font-semibold">Ready</span> },
                    ].map(({ label, value }) => (
                      <div key={label}>
                        <div className="text-gray-400 mb-0.5">{label}</div>
                        <div className="font-semibold text-gray-700">{value}</div>
                      </div>
                    ))}
                  </div>
                  <div className="mt-3 pt-3 border-t border-indigo-100">
                    <div className="text-gray-400 text-xs mb-2">Class Distribution</div>
                    <div className="flex gap-3">
                      {Object.entries(activeDataset.classes).map(([cls, pct]) => (
                        <div key={cls} className="flex items-center gap-1.5">
                          <div className="w-2 h-2 rounded-full" style={{ background: cls.includes('Not') || cls.includes('Legitimate') || cls.includes('No') ? '#10B981' : '#4F6AF7' }} />
                          <span className="text-xs text-gray-600">{cls} <span className="font-semibold text-gray-800">{pct}</span></span>
                        </div>
                      ))}
                    </div>
                  </div>
                </div>
              )}
            </div>

            {/* Task */}
            <div className="card p-6">
              <h2 className="text-sm font-bold text-gray-700 uppercase tracking-widest mb-2">02 — Mission Objective</h2>
              <p className="text-xs text-gray-400 mb-4 font-medium">Describe what you want ATLAS to accomplish in natural language</p>
              <div className="relative">
                <textarea
                  value={task}
                  onChange={e => setTask(e.target.value)}
                  placeholder={"Predict whether a customer will churn. Prioritize catching potential churners while maintaining reasonable precision."}
                  rows={4}
                  className="w-full rounded-xl border border-indigo-100 bg-indigo-50/30 px-4 py-3 text-sm text-gray-800 placeholder:text-gray-300 resize-none focus:outline-none focus:ring-2 focus:ring-indigo-300 focus:border-indigo-300 transition-all"
                />
              </div>
              <div className="flex gap-2 mt-2 flex-wrap">
                {EXAMPLE_TASKS.map(t => (
                  <button key={t} onClick={() => setTask(t)} className="text-xs text-indigo-500 bg-indigo-50 hover:bg-indigo-100 px-2 py-1 rounded-lg transition-colors truncate max-w-[200px]">
                    {t.slice(0, 38)}…
                  </button>
                ))}
              </div>
            </div>

            {/* Config */}
            <div className="card p-6">
              <h2 className="text-sm font-bold text-gray-700 uppercase tracking-widest mb-5">03 — Configuration</h2>
              <div className="grid grid-cols-2 gap-5">
                <div>
                  <label className="text-xs font-semibold text-gray-500 mb-2 block">Target Column</label>
                  <select
                    value={target}
                    onChange={e => setTarget(e.target.value)}
                    className="w-full rounded-xl border border-gray-100 bg-gray-50/50 px-3 py-2.5 text-sm text-gray-700 focus:outline-none focus:ring-2 focus:ring-indigo-300 focus:border-indigo-300"
                  >
                    <option value="auto">Auto-detect</option>
                    {activeDataset && <option value={activeDataset.target}>{activeDataset.target}</option>}
                  </select>
                </div>
                <div>
                  <label className="text-xs font-semibold text-gray-500 mb-2 block">
                    Max Experiments <span className="text-gray-400 font-normal">(budget)</span>
                  </label>
                  <div className="flex items-center gap-3">
                    <input
                      type="range" min={2} max={12} value={maxExperiments}
                      onChange={e => setMaxExperiments(Number(e.target.value))}
                      className="flex-1 accent-indigo-600"
                    />
                    <span className="font-mono font-bold text-indigo-700 w-6 text-center">{maxExperiments}</span>
                  </div>
                  <p className="text-xs text-gray-400 mt-1">ATLAS may stop earlier if the objective is reached</p>
                </div>
              </div>

              <div className="mt-5 flex items-center justify-between p-4 rounded-xl bg-gray-50/50 border border-gray-100">
                <div>
                  <div className="text-sm font-semibold text-gray-700">AI-Guided Experimentation</div>
                  <div className="text-xs text-gray-400 mt-0.5">Uses an LLM to generate and evaluate hypotheses</div>
                </div>
                <button
                  onClick={() => setAiGuided(!aiGuided)}
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
                  Advanced options such as custom metric functions, hyperparameter search spaces, and hardware configuration are available via the ATLAS API.
                </div>
              )}
            </div>
          </div>

          {/* Right — live preview */}
          <div className="col-span-2 flex flex-col gap-6">
            <AtlasUnderstandsPanel task={task} maxExp={maxExperiments} aiGuided={aiGuided} dataset={activeDataset} />

            {/* How ATLAS works */}
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
                      {['Train a model on the dataset', 'Evaluate metrics on holdout set', 'Analyze failures and patterns', 'Generate a testable theory', 'Choose: Continue, Revise, or Stop'][i]}
                    </div>
                  </div>
                </div>
              ))}
            </div>

            {/* CTA */}
            <button
              onClick={handleStart}
              disabled={!canStart || launching}
              className={`w-full rounded-2xl py-5 font-bold text-base tracking-wide text-white transition-all duration-300 relative overflow-hidden group ${canStart && !launching
                ? 'gradient-primary hover:shadow-lg hover:shadow-indigo-200 hover:-translate-y-0.5 active:translate-y-0'
                : 'bg-gray-200 text-gray-400 cursor-not-allowed'}`}
            >
              {launching ? (
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
            {!canStart && (
              <p className="text-xs text-gray-400 text-center -mt-3">
                {!activeDataset ? 'Select or upload a dataset to continue' : 'Describe your mission objective above'}
              </p>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
