/**
 * The one place the frontend talks to the backend.
 *
 * Every request and response shape here was read out of server.py, not guessed.
 * Nothing in this file invents data: if a call fails, it throws an Error whose
 * message comes from the backend, and the caller renders that.
 */

import type {
  AtlasEvent,
  ComparisonResponse,
  DatasetListing,
  MissionPlan,
  MissionRunResponse,
} from '../types';

/**
 * Empty in local development so Vite's /api proxy remains the transport.
 * Set VITE_API_BASE_URL to the separately deployed backend (for example on
 * Render) for a production frontend. Strip a trailing slash so paths below
 * always join correctly. VITE_ATLAS_API remains a harmless compatibility
 * fallback for existing local configuration.
 */
export const API_BASE = (import.meta.env.VITE_API_BASE_URL ?? import.meta.env.VITE_ATLAS_API ?? '').replace(/\/$/, '');

class ApiError extends Error {
  constructor(
    message: string,
    readonly status: number,
  ) {
    super(message);
    this.name = 'ApiError';
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let response: Response;
  try {
    response = await fetch(`${API_BASE}${path}`, {
      // FormData must set its own Content-Type: the browser appends the
      // multipart boundary, and overriding it here would corrupt the body.
      headers:
        init?.body && !(init.body instanceof FormData)
          ? { 'Content-Type': 'application/json' }
          : undefined,
      ...init,
    });
  } catch (cause) {
    // Network-level failure: the backend is not reachable at all.
    throw new ApiError(
      cause instanceof Error ? cause.message : 'backend unreachable',
      0,
    );
  }

  let payload: unknown = null;
  const text = await response.text();
  if (text) {
    try {
      payload = JSON.parse(text);
    } catch {
      payload = null;
    }
  }

  if (!response.ok) {
    // server.py returns {"error": "..."} on every failure path.
    const message =
      payload && typeof payload === 'object' && 'error' in payload
        ? String((payload as { error: unknown }).error)
        : `HTTP ${response.status}`;
    throw new ApiError(message, response.status);
  }
  return payload as T;
}

/** GET /api/datasets — the CSVs actually present in the repo's data/ directory. */
export async function fetchDatasets(): Promise<DatasetListing[]> {
  const payload = await request<{ datasets: DatasetListing[] }>('/api/datasets');
  return payload.datasets;
}

/**
 * POST /api/datasets/upload — one CSV, multipart.
 *
 * Returns the same row shape /api/datasets uses, so the caller can drop it
 * straight into the dataset list. Throws with the backend's own reason when the
 * file is rejected (wrong type, empty, unparseable).
 */
export async function uploadDataset(file: File): Promise<DatasetListing> {
  const form = new FormData();
  form.append('file', file);
  const payload = await request<{ dataset: DatasetListing }>('/api/datasets/upload', {
    method: 'POST',
    body: form,
  });
  return payload.dataset;
}

/**
 * POST /api/plan — real Planner output.
 * Returns null when the task is blank; throws when the Planner rejects it.
 */
export async function fetchPlan(task: string): Promise<MissionPlan | null> {
  const payload = await request<{ plan: MissionPlan | null }>('/api/plan', {
    method: 'POST',
    body: JSON.stringify({ task }),
  });
  return payload.plan;
}

export interface StartMissionBody {
  csv: string;
  target?: string | null;
  task?: string;
  budget?: number;
  /** NOTE: the field is `llm`, not `use_llm`. "offline" enables the proposer. */
  llm?: string | null;
}

/** POST /api/missions — returns the real mission id, or throws with the reason. */
export async function startMission(body: StartMissionBody): Promise<string> {
  const payload = await request<MissionRunResponse>('/api/missions', {
    method: 'POST',
    body: JSON.stringify(body),
  });
  return payload.mission_id;
}

/** POST /api/compare — runs both strategies server-side and returns both logs. */
export async function fetchComparison(body: {
  csv: string;
  target?: string | null;
  budget?: number;
}): Promise<ComparisonResponse> {
  return request<ComparisonResponse>('/api/compare', {
    method: 'POST',
    body: JSON.stringify(body),
  });
}

/** Uses the configured API origin for cross-origin SSE, or Vite's local proxy. */
export function eventsUrl(missionId: string): string {
  return `${API_BASE}/api/missions/${missionId}/events`;
}

/** Parse an SSE `data:` payload. Returns null on malformed input rather than throwing. */
export function parseEvent(data: string): AtlasEvent | null {
  try {
    const parsed = JSON.parse(data) as AtlasEvent;
    return typeof parsed?.stage === 'string' ? parsed : null;
  } catch {
    return null;
  }
}

export { ApiError };
