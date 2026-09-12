"""Decision: deterministic guards around the loop.

ATLAS never stops because a model said it felt done. It stops when a budget,
a target, a plateau, or an absence of hypotheses says so.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

from atlas.objective import reached
from atlas.state import Action, Category, Decision, Diagnosis, History, Hypothesis


@dataclass(frozen=True)
class Budget:
    max_experiments: int = 6
    target_score: float = 0.95
    patience: int = 3  # experiments without improvement before stopping
    max_seconds: float = 300.0
    max_consecutive_errors: int = 2


def decide(
    diagnosis: Diagnosis,
    hypothesis: Hypothesis | None,
    history: History,
    budget: Budget,
    started: float,
) -> Decision:
    if len(history) >= budget.max_experiments:
        return Decision(Action.STOP, f"experiment budget reached ({budget.max_experiments})")

    elapsed = time.perf_counter() - started
    if elapsed >= budget.max_seconds:
        return Decision(Action.STOP, f"time budget reached ({elapsed:.1f}s)")

    recent = history.experiments[-budget.max_consecutive_errors :]
    if len(recent) == budget.max_consecutive_errors and all(not e.result.ok for e in recent):
        return Decision(Action.STOP, f"{budget.max_consecutive_errors} consecutive failed experiments")

    best = history.best_score()
    if best is not None and reached(best, budget.target_score, history.objective):
        return Decision(
            Action.STOP,
            f"objective reached (best {best:.3f} meets target {budget.target_score} "
            f"on {history.objective.primary_metric})",
        )

    if hypothesis is None:
        return Decision(Action.STOP, "no untried hypothesis remains")

    stalled = history.rounds_since_improvement()
    if stalled >= budget.patience:
        return Decision(Action.STOP, f"no improvement in {stalled} experiments (patience {budget.patience})")

    if hypothesis.touches_data or diagnosis.category is Category.DATA_ISSUE:
        return Decision(Action.REVISE, f"data preparation must change: {hypothesis.proposed_change}")

    return Decision(Action.CONTINUE, f"testing next hypothesis: {hypothesis.proposed_change}")
