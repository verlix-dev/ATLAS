import { useState } from 'react';
import type { Screen, MissionState, MissionConfig } from './types';
import MissionSetup from './screens/MissionSetup';
import Dashboard from './screens/Dashboard';
import Comparison from './screens/Comparison';
const initialMissionState = (config: MissionConfig): MissionState => ({
  config,
  experiments: [],
  status: 'running',
  currentStage: 'experiment',
  currentExperimentIndex: 0,
});

export default function App() {
  const [screen, setScreen] = useState<Screen>('setup');
  const [mission, setMission] = useState<MissionState | null>(null);

  const handleStart = (config: MissionConfig) => {
    setMission(initialMissionState(config));
    setScreen('dashboard');
  };

  const handleComplete = () => {
    setScreen('comparison');
  };

  const handleRestart = () => {
    setMission(null);
    setScreen('setup');
  };

  if (screen === 'setup') {
    return <MissionSetup onStart={handleStart} />;
  }

  if (screen === 'dashboard' && mission) {
    return <Dashboard mission={mission} onComplete={handleComplete} />;
  }

  if (screen === 'comparison') {
    return <Comparison onRestart={handleRestart} />;
  }

  return null;
}
