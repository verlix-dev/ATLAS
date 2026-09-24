/**
 * The live-mission hook: mission creation, SSE consumption, and cleanup.
 *
 * It owns exactly one responsibility — turning the backend's event stream into
 * MissionState — and delegates both the transport (api.ts) and the fold
 * (reduce.ts) elsewhere.
 *
 * On pacing: the engine finishes a mission in a couple of seconds and the
 * server emits each event the instant it is produced. This hook does NOT delay,
 * sleep, or fake anything — every state it exposes is the result of an event
 * that actually arrived. Events are applied as they land.
 */

import { useCallback, useEffect, useRef, useState } from 'react';

import { eventsUrl, parseEvent, startMission } from './api';
import type { StartMissionBody } from './api';
import { initialState, reduce } from './reduce';
import type { AtlasEvent, MissionState } from '../types';

export interface UseMission {
  state: MissionState;
  missionId: string | null;
  /** Set when the stream or a REST call fails; the reason comes from the backend. */
  error: string | null;
  starting: boolean;
  start: (body: StartMissionBody) => Promise<string | null>;
  /** Reattach to a mission already running server-side. */
  attach: (missionId: string) => void;
  reset: () => void;
}

export function useMission(): UseMission {
  const [state, setState] = useState<MissionState>(initialState);
  const [missionId, setMissionId] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [starting, setStarting] = useState(false);

  const sourceRef = useRef<EventSource | null>(null);
  const doneRef = useRef(false);

  const closeStream = useCallback(() => {
    if (sourceRef.current) {
      sourceRef.current.close();
      sourceRef.current = null;
    }
  }, []);

  /** Fold one frame. Malformed payloads are ignored rather than crashing the UI. */
  const apply = useCallback((event: AtlasEvent) => {
    setState((previous) => reduce(previous, event));
  }, []);

  const attach = useCallback(
    (id: string) => {
      closeStream();
      doneRef.current = false;
      setMissionId(id);
      setError(null);

      // The endpoint replays from seq 0 and then follows live, so a late
      // subscriber sees the whole run with no gap. reduce() drops any seq it
      // has already applied, which makes a reconnect safe.
      const source = new EventSource(eventsUrl(id));
      sourceRef.current = source;

      const stages = [
        'mission_start',
        'experiment_start',
        'experiment_result',
        'diagnosis',
        'hypothesis',
        'decision',
        'mission_end',
        'data_engineer',
      ];

      for (const stage of stages) {
        source.addEventListener(stage, (message) => {
          const event = parseEvent((message as MessageEvent).data);
          if (event) apply(event);
        });
      }

      // Terminal frame the server sends once the worker is finished. The
      // mission_end event already carried the authoritative ranking, so this
      // frame only reports whether the worker itself crashed.
      source.addEventListener('done', (message) => {
        doneRef.current = true;
        try {
          const payload = JSON.parse((message as MessageEvent).data) as {
            done: boolean;
            error: string | null;
          };
          if (payload.error) setError(payload.error);
        } catch {
          /* the terminal frame is advisory */
        }
        closeStream();
      });

      source.onerror = () => {
        // The server closes the socket when the mission ends; that is normal.
        if (doneRef.current) return;
        // Otherwise the connection genuinely dropped.
        setError('event stream disconnected');
        closeStream();
      };
    },
    [apply, closeStream],
  );

  const start = useCallback(
    async (body: StartMissionBody): Promise<string | null> => {
      setStarting(true);
      setError(null);
      try {
        const id = await startMission(body);
        setState(initialState());
        attach(id);
        return id;
      } catch (cause) {
        setError(cause instanceof Error ? cause.message : String(cause));
        return null;
      } finally {
        setStarting(false);
      }
    },
    [attach],
  );

  const reset = useCallback(() => {
    closeStream();
    doneRef.current = false;
    setState(initialState());
    setMissionId(null);
    setError(null);
  }, [closeStream]);

  // Never leave a stream open behind an unmounted component.
  useEffect(() => closeStream, [closeStream]);

  return { state, missionId, error, starting, start, attach, reset };
}
