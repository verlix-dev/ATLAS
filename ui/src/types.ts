/**
 * ATLAS — types for the real backend contract.
 *
 * These mirror what server.py actually returns and what atlas/loop.py actually
 * emits as SSE events. Two honesty constraints shape them:
 *
 *   1. A metric can be the string "unavailable" (atlas/objective.py UNAVAILABLE).
 *      It is never a number, and it never silently becomes 0.
 *   2. A score can be null — an experiment that failed, or whose primary metric
 *      could not be computed, is excluded from ranking rather than scored 0.
 */

/* ------------------------------------------------------------------ events */

export type EventStage =
  | 'mission_start'
  | 'experiment_start'
  | 'experiment_result'
  | 'diagnosis'
  | 'hypothesis'
  | 'decision'
  | 'mission_end'
  | 'data_engineer';

/** The engine's marker for "this could not be measured". Mirrors objective.py. */
export const UNAVAILABLE = 'unavailable';

export type MetricValue = number | typeof UNAVAILABLE;
export type MaybeMetric = MetricValue | undefined | null;

export interface Metrics {
  f1_macro?: MetricValue;
  accuracy?: MetricValue;
  precision?: MetricValue;
  recall?: MetricValue;
  roc_auc?: MetricValue;
  pr_auc?: MetricValue;
  positive_class?: string | null;
  train_f1_macro?: number;
  overfit_gap?: MetricValue;
  per_class_recall?: Record<string, number>;
  confusion_matrix?: number[][];
  labels?: string[];
  test_rows?: number;
  predicted_classes?: number;
  [key: string]: unknown;
}

export interface Objective {
  primary_metric: string;
  direction: 'maximize' | 'minimize' | string;
  secondary_metrics: string[];
}

export interface Budget {
  max_experiments: number;
  target_score: number;
  patience: number;
  max_seconds: number;
  max_consecutive_errors: number;
}

export interface Profile {
  rows: number;
  columns: number;
  target: string;
  problem_type: string;
  target_confidence: string;
  target_reason: string;
  numeric: string[];
  categorical: string[];
  dtypes: Record<string, string>;
  missing_per_column: Record<string, number>;
  missing_fraction: number;
  duplicate_rows: number;
  constant_columns: string[];
  class_counts: Record<string, number>;
  minority_class: string;
  imbalance_ratio: number;
}

export interface MissionPlan {
  objective: string;
  problem_type: string | null;
  target_candidate: string | null;
  primary_metric: string;
  metric_confidence: string;
  metric_reason: string;
  priority: string | null;
  constraints: string[];
  experiment_budget: number | null;
}

export interface ExperimentConfig {
  model: string;
  params: Record<string, string | number | boolean | null>;
  preprocess: Record<string, string | number | boolean | null>;
}

export interface Rejection {
  source: string;
  reason: string;
}

/** One raw frame from the SSE stream. Forwarded verbatim; never reshaped. */
export interface AtlasEvent {
  seq: number;
  t: number;
  stage: EventStage;
  [key: string]: unknown;
}

/* ------------------------------------------------------------ mission state */

export interface ExperimentResult {
  ok: boolean;
  metrics: Metrics;
  error: string | null;
  seconds?: number;
}

export interface Diagnosis {
  category: string;
  summary: string;
  findings: string[];
}

export interface Hypothesis {
  problem: string;
  proposedChange: string;
  reasoning: string;
  expectedEffect: string;
  nextConfig: ExperimentConfig | null;
  /** Who proposed it. Drives provenance UI; never inferred client-side. */
  source: 'llm' | 'deterministic' | string | null;
  cites: string[];
}

export interface Decision {
  action: 'continue' | 'revise' | 'stop' | string;
  reason: string;
}

export interface Experiment {
  id: number;
  config: ExperimentConfig | null;
  /** "baseline", or e.g. "exp1: [llm] <proposed change>". */
  origin: string | null;
  source: 'baseline' | 'llm' | 'deterministic' | string;
  result: ExperimentResult | null;
  diagnosis: Diagnosis | null;
  hypothesis: Hypothesis | null;
  decision: Decision | null;
  /** Proposals the validator refused. These never ran. */
  rejections: Rejection[];
}

export interface MissionSummary {
  experiments: number;
  bestId: number | null;
  bestScore: number | null;
  bestConfig: ExperimentConfig | null;
  objective: Objective | null;
  reason: string;
}

/**
 * The shape the BACKEND returns from `Mission.summary()`.
 *
 * Deliberately separate from MissionSummary above: that one is the reducer's
 * normalised view (camelCase), while this is the raw wire format. Note the two
 * are not just renamed — the SSE `mission_end` event carries `reason`, whereas
 * `summary()` carries `stop_reason`.
 */
export interface RawMissionSummary {
  experiments: number;
  best_id: number | null;
  best_score: number | null;
  best_model: string | null;
  objective: Objective;
  /** null when the mission produced neither a decision nor a block reason. */
  stop_reason: string | null;
  timeline: {
    id: number;
    model: string;
    params: Record<string, string | number | boolean | null>;
    /** null when the experiment was unrankable — never coerce this to 0. */
    score: number | null;
    primary_metric: string;
    category: string;
    decision: string | null;
    origin: string;
  }[];
}

export interface BlockedInfo {
  status: string;
  target: string;
  reason: string;
  columns: string[];
}

export type MissionStatus = 'idle' | 'running' | 'complete' | 'blocked';

export interface MissionState {
  status: MissionStatus;
  profile: Profile | null;
  objective: Objective | null;
  budget: Budget | null;
  plan: MissionPlan | null;
  experiments: Experiment[];
  currentId: number | null;
  /** Which loop stage the backend last reported. */
  stage: EventStage | null;
  best: {
    id: number;
    score: number;
    config: ExperimentConfig | null;
    /** True while derived locally; false once mission_end confirms it. */
    provisional: boolean;
  } | null;
  summary: MissionSummary | null;
  blocked: BlockedInfo | null;
  lastSeq: number;
  eventCount: number;
}

/* ---------------------------------------------------------------- REST DTOs */

export interface DatasetListing {
  name: string;
  /** The token the backend resolves back to a file; send it back verbatim. */
  csv: string;
  bytes: number;
  /** Where it came from. Absent on older responses, so treated as builtin. */
  source?: 'builtin' | 'upload';
  /** Counted by the backend on upload only. Never inferred client-side. */
  rows?: number;
  columns?: number;
}

export interface MissionRunResponse {
  mission_id: string;
}

export interface ComparisonSide {
  /** Verbatim SSE frames from that strategy's run. */
  events: AtlasEvent[];
  /** Raw backend summary — snake_case, authoritative for best_id/best_score. */
  summary: RawMissionSummary;
}

export interface NovelConfig {
  id: number;
  model: string;
  params: Record<string, string | number | boolean | null>;
  preprocess: Record<string, string | number | boolean | null>;
}

export interface ComparisonResponse {
  csv: string;
  target: string;
  budget: Budget;
  same_dataset_object: boolean;
  same_objective: boolean;
  train_rows: number;
  test_rows: number;
  deterministic: ComparisonSide;
  guided: ComparisonSide;
  novel: NovelConfig[];
  rejections: Rejection[];
}

/* ------------------------------------------------------------------- screens */

export type Screen = 'setup' | 'dashboard' | 'comparison';
