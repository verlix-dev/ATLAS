/**
 * Pure event reducer: SSE event -> MissionState.
 *
 * The backend's event stream is an append-only log, so UI state is a fold over
 * it. This module has no DOM, no network and no timers, which makes it testable
 * in isolation and makes the duplicate-protection below verifiable.
 *
 * Three rules it exists to enforce:
 *
 *   1. An unavailable metric stays unavailable. It never becomes 0.
 *   2. A null score is never a winner and is never plotted.
 *   3. The backend owns final ranking. A live "best" is derived locally only
 *      until mission_end arrives, and is then replaced verbatim.
 */

import {
  UNAVAILABLE,
  type AtlasEvent,
  type Diagnosis,
  type Experiment,
  type ExperimentConfig,
  type ExperimentResult,
  type Hypothesis,
  type MissionState,
  type Objective,
  type Rejection,
} from '../types';

export const LOOP_STAGES = [
  'experiment',
  'measure',
  'diagnose',
  'hypothesize',
  'decide',
] as const;

/** Maps each backend stage onto the loop step the UI narrates. */
const STAGE_OF: Record<string, string> = {
  experiment_start: 'experiment',
  experiment_result: 'measure',
  diagnosis: 'diagnose',
  hypothesis: 'hypothesize',
  decision: 'decide',
};

export function isUnavailable(value: unknown): boolean {
  return value === undefined || value === null || value === UNAVAILABLE;
}

/** Format a metric for display. Unavailable reads as "N/A" -- never "0". */
export function formatMetric(value: unknown, digits = 4): string {
  if (isUnavailable(value)) return 'N/A';
  const n = Number(value);
  if (!Number.isFinite(n)) return 'N/A';
  return n.toFixed(digits);
}

export function formatSigned(value: number, digits = 4): string {
  return `${value >= 0 ? '+' : ''}${value.toFixed(digits)}`;
}

export function formatSeconds(seconds: unknown): string {
  const n = Number(seconds);
  if (!Number.isFinite(n)) return '—';
  return `${n}s`;
}

/** "hist_gb · learning_rate=0.04 · max_iter=300" */
export function describeConfig(config: ExperimentConfig | null): string {
  if (!config) return '—';
  const params = Object.entries(config.params ?? {}).map(([k, v]) => `${k}=${v}`);
  const prep = Object.entries(config.preprocess ?? {}).map(([k, v]) => `${k}=${v}`);
  return [config.model, ...params, ...prep].join(' · ');
}

export function initialState(): MissionState {
  return {
    status: 'idle',
    profile: null,
    objective: null,
    budget: null,
    plan: null,
    experiments: [],
    currentId: null,
    stage: null,
    best: null,
    summary: null,
    blocked: null,
    lastSeq: -1,
    eventCount: 0,
  };
}

function blankExperiment(id: number): Experiment {
  return {
    id,
    config: null,
    origin: null,
    source: 'baseline',
    result: null,
    diagnosis: null,
    hypothesis: null,
    decision: null,
    rejections: [],
  };
}

/**
 * The score ATLAS ranks an experiment by, or null when it cannot be ranked.
 * Mirrors ExperimentResult.score(): a failed run and an unavailable primary
 * metric are both unrankable, and neither is worth zero.
 */
export function scoreOf(
  experiment: Experiment | undefined,
  objective: Objective | null,
): number | null {
  if (!experiment?.result?.ok || !objective) return null;
  const raw = experiment.result.metrics?.[objective.primary_metric];
  if (isUnavailable(raw)) return null;
  const n = Number(raw);
  return Number.isFinite(n) ? n : null;
}

/** Provisional best while the mission runs. Direction-aware; skips unrankable rows. */
function provisionalBest(
  experiments: Experiment[],
  objective: Objective | null,
): MissionState['best'] {
  if (!objective) return null;
  const scored = experiments
    .map((e) => ({ experiment: e, score: scoreOf(e, objective) }))
    .filter((row): row is { experiment: Experiment; score: number } => row.score !== null);
  if (!scored.length) return null;
  const minimize = objective.direction === 'minimize';
  const winner = scored.reduce((a, b) =>
    minimize ? (b.score < a.score ? b : a) : b.score > a.score ? b : a,
  );
  return {
    id: winner.experiment.id,
    score: winner.score,
    config: winner.experiment.config,
    provisional: true,
  };
}

/** Replace one experiment immutably, creating it if the id is new. */
function withExperiment(
  experiments: Experiment[],
  id: number,
  patch: Partial<Experiment>,
): Experiment[] {
  const next = experiments.slice();
  const index = next.findIndex((e) => e.id === id);
  if (index === -1) next.push({ ...blankExperiment(id), ...patch });
  else next[index] = { ...next[index], ...patch };
  return next;
}

/**
 * Fold one event into state. Pure: returns a new object and mutates nothing.
 *
 * Duplicate protection: the SSE endpoint replays from seq 0 on connect, and a
 * reconnect replays again. Every event carries a monotonic `seq`, so an event
 * already applied is dropped. Without this, a reconnect would double-count.
 */
export function reduce(state: MissionState, event: AtlasEvent): MissionState {
  if (!event || typeof event.stage !== 'string') return state;
  if (typeof event.seq === 'number' && event.seq <= state.lastSeq) return state;

  const next: MissionState = {
    ...state,
    lastSeq: typeof event.seq === 'number' ? event.seq : state.lastSeq,
    eventCount: state.eventCount + 1,
  };
  if (STAGE_OF[event.stage]) next.stage = event.stage;

  switch (event.stage) {
    case 'mission_start': {
      next.status = 'running';
      next.profile = (event.profile as MissionState['profile']) ?? null;
      next.objective = (event.objective as Objective) ?? null;
      next.budget = (event.budget as MissionState['budget']) ?? null;
      next.plan = (event.plan as MissionState['plan']) ?? null;
      return next;
    }

    // The Data Engineer refused the target, so no experiment will run.
    case 'data_engineer': {
      next.status = 'blocked';
      next.blocked = {
        status: String(event.status ?? 'needs_clarification'),
        target: String(event.target ?? ''),
        reason: String(event.reason ?? ''),
        columns: (event.columns as string[]) ?? [],
      };
      next.profile = (event.profile as MissionState['profile']) ?? next.profile;
      next.plan = (event.plan as MissionState['plan']) ?? next.plan;
      return next;
    }

    case 'experiment_start': {
      const id = Number(event.id);
      next.experiments = withExperiment(next.experiments, id, {
        config: (event.config as ExperimentConfig) ?? null,
        origin: (event.origin as string) ?? null,
        source: originSource(event.origin as string),
      });
      next.currentId = id;
      return next;
    }

    case 'experiment_result': {
      const id = Number(event.id);
      next.experiments = withExperiment(next.experiments, id, {
        result: {
          ok: Boolean(event.ok),
          metrics: (event.metrics as ExperimentResult['metrics']) ?? {},
          error: (event.error as string | null) ?? null,
          seconds: event.seconds as number | undefined,
        },
      });
      if (!next.summary) next.best = provisionalBest(next.experiments, next.objective);
      return next;
    }

    case 'diagnosis': {
      const id = Number(event.id);
      next.experiments = withExperiment(next.experiments, id, {
        diagnosis: {
          category: String(event.category ?? ''),
          summary: String(event.summary ?? ''),
          findings: (event.findings as string[]) ?? [],
        } satisfies Diagnosis,
      });
      return next;
    }

    case 'hypothesis': {
      const id = Number(event.id);
      // proposed_change === null means the engine ran out of untried ideas.
      const hypothesis: Hypothesis | null = event.proposed_change
        ? {
            problem: String(event.problem ?? ''),
            proposedChange: String(event.proposed_change),
            reasoning: String(event.reasoning ?? ''),
            expectedEffect: String(event.expected_effect ?? ''),
            nextConfig: (event.next_config as ExperimentConfig) ?? null,
            source: (event.source as string | null) ?? null,
            cites: (event.cites as string[]) ?? [],
          }
        : null;
      next.experiments = withExperiment(next.experiments, id, {
        hypothesis,
        rejections: (event.rejected as Rejection[]) ?? [],
      });
      return next;
    }

    case 'decision': {
      const id = Number(event.id);
      next.experiments = withExperiment(next.experiments, id, {
        decision: {
          action: String(event.action ?? ''),
          reason: String(event.reason ?? ''),
        },
      });
      return next;
    }

    case 'mission_end': {
      next.status = next.status === 'blocked' ? 'blocked' : 'complete';
      next.summary = {
        experiments: Number(event.experiments ?? next.experiments.length),
        bestId: (event.best_id as number | null) ?? null,
        bestScore: (event.best_score as number | null) ?? null,
        bestConfig: (event.best_config as ExperimentConfig | null) ?? null,
        objective: (event.objective as Objective) ?? next.objective,
        reason: String(event.reason ?? ''),
      };
      // The backend owns ranking: replace the provisional best verbatim.
      if (next.summary.bestId === null || next.summary.bestScore === null) {
        next.best = null;
      } else {
        next.best = {
          id: next.summary.bestId,
          score: next.summary.bestScore,
          config: next.summary.bestConfig,
          provisional: false,
        };
      }
      return next;
    }

    default:
      // An unknown stage is retained in the count but changes nothing, so a
      // backend addition cannot break the UI.
      return next;
  }
}

/** Provenance of the experiment an origin string describes. */
export function originSource(origin: string | null | undefined): string {
  if (!origin || origin === 'baseline') return 'baseline';
  if (origin.includes('[llm]')) return 'llm';
  if (origin.includes('[deterministic]')) return 'deterministic';
  return 'deterministic';
}

/** Fold a whole log. */
export function reduceAll(state: MissionState, events: AtlasEvent[]): MissionState {
  return events.reduce(reduce, state);
}

/** Points for the score chart: only genuinely measured, rankable scores. */
export function scoreSeries(
  state: MissionState,
): { id: number; score: number; experiment: Experiment }[] {
  return state.experiments
    .map((experiment) => ({ experiment, score: scoreOf(experiment, state.objective) }))
    .filter(
      (row): row is { experiment: Experiment; score: number } => row.score !== null,
    )
    .map((row) => ({ id: row.experiment.id, score: row.score, experiment: row.experiment }));
}

/** "3 of ≤6" — the budget is a ceiling, and a mission may stop before it. */
export function progressLabel(state: MissionState): string {
  const run = state.experiments.length;
  const cap = state.budget?.max_experiments;
  return cap ? `${run} of ≤${cap}` : String(run);
}

/** Improvement over the first experiment ATLAS ran. Arithmetic on measured scores. */
export function baselineDelta(state: MissionState): number | null {
  if (!state.best || state.experiments.length < 2 || !state.objective) return null;
  const first = scoreOf(state.experiments[0], state.objective);
  if (first === null || state.best.id === state.experiments[0].id) return null;
  return state.objective.direction === 'minimize'
    ? first - state.best.score
    : state.best.score - first;
}
