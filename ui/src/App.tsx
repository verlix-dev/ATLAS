import { useCallback, useState } from 'react';

import Dashboard from './screens/Dashboard';
import Comparison from './screens/Comparison';
import MissionSetup from './screens/MissionSetup';
import { useMission } from './lib/useMission';
import type { Screen } from './types';

/**
 * Screen routing and nothing else.
 *
 * All mission logic lives in useMission (transport + fold) and lib/reduce
 * (state derivation). This component holds only which screen is showing and
 * the dataset/budget the comparison needs.
 */
export default function App() {
  const [screen, setScreen] = useState<Screen>('setup');
  const { state, error, starting, start, reset } = useMission();

  // Carried from the mission that just ran, so the comparison uses the same
  // dataset and budget rather than guessing.
  const [runConfig, setRunConfig] = useState({
    csv: 'data/churn.csv',
    target: null as string | null,
    budget: 6,
  });

  const handleStart = useCallback(
    async (body: Parameters<typeof start>[0]) => {
      setRunConfig({
        csv: body.csv,
        target: body.target ?? null,
        budget: body.budget ?? 6,
      });
      const id = await start(body);
      // Only navigate once the backend has actually accepted the mission.
      if (id) setScreen('dashboard');
      return id;
    },
    [start],
  );

  const handleRestart = useCallback(() => {
    reset();
    setScreen('setup');
  }, [reset]);

  if (screen === 'setup') {
    return <MissionSetup onStart={handleStart} starting={starting} startError={error} />;
  }

  if (screen === 'dashboard') {
    return (
      <Dashboard
        mission={state}
        error={error}
        onComplete={() => setScreen('comparison')}
        onRestart={handleRestart}
      />
    );
  }

  return (
    <Comparison
      onRestart={handleRestart}
      csv={runConfig.csv}
      target={runConfig.target}
      budget={runConfig.budget}
    />
  );
}
