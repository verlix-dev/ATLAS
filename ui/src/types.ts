export type Screen = 'setup' | 'dashboard' | 'comparison';
export type Stage = 'experiment' | 'measure' | 'diagnose' | 'hypothesize' | 'decide';
export type Decision = 'CONTINUE' | 'STOP' | 'REVISE';
export type Provenance = 'ATLAS_RULES' | 'AI_GUIDED';

export interface Metrics {
  f1_macro: number;
  recall: number;
  precision: number;
  overfit_gap: number;
}

export interface Experiment {
  id: number;
  model: string;
  params: Record<string, string | number>;
  metrics?: Metrics;
  diagnosis?: string;
  hypothesis?: string;
  evidence?: string[];
  decision?: Decision;
  decisionReason?: string;
  provenance: Provenance;
  status: 'running' | 'complete' | 'failed';
  revealStage: Stage | 'complete';
}

export interface MissionConfig {
  task: string;
  dataset: string;
  rows: number;
  columns: number;
  target: string;
  maxExperiments: number;
  aiGuided: boolean;
  problemType: string;
  primaryMetric: string;
  priority: string;
}

export interface MissionState {
  config: MissionConfig;
  experiments: Experiment[];
  status: 'running' | 'complete';
  currentStage: Stage;
  currentExperimentIndex: number;
  stopReason?: string;
}
