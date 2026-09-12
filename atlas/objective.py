"""The objective layer: what ATLAS is optimising, and how to score against it.

One Objective per Mission, one score() function, one direction table. The
Planner says which metric the user wants; this module knows what is
computable and which way is better.

A leaf module by design — it imports no other ATLAS component, so state.py
can depend on it without inverting the dependency graph.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # annotation only: keeps this module a leaf at runtime
    from atlas.planner import MissionPlan

# Marker written into a metrics dict when a metric genuinely cannot be
# computed. Never a number, so it can never be mistaken for a measurement.
UNAVAILABLE = "unavailable"

DEFAULT_METRIC = "f1_macro"  # what ATLAS optimises when no plan says otherwise

# Direction is a property of the metric, not of the user's phrasing.
DIRECTIONS = {
    "accuracy": "maximize",
    "precision": "maximize",
    "recall": "maximize",
    "f1_macro": "maximize",
    "roc_auc": "maximize",
    "pr_auc": "maximize",
    "r2": "maximize",
    "mae": "minimize",
    "rmse": "minimize",
}

# The fixed classification reporting set. Not user-configurable this milestone.
REPORTED = ("accuracy", "precision", "recall", "f1_macro", "roc_auc", "pr_auc")

# The Planner can name a metric more loosely than the engine computes it.
ALIASES = {"f1": "f1_macro", "average_precision": "pr_auc", "auc": "roc_auc"}


@dataclass(frozen=True)
class Objective:
    """What to optimise. One field of state; direction and reporting follow from it."""

    primary_metric: str = DEFAULT_METRIC

    def __post_init__(self) -> None:
        # Trust boundary: primary_metric originates in free-text task input.
        if self.primary_metric not in DIRECTIONS:
            raise ValueError(
                f"unknown metric {self.primary_metric!r}; known: {sorted(DIRECTIONS)}"
            )

    @property
    def direction(self) -> str:
        return DIRECTIONS[self.primary_metric]

    @property
    def secondary_metrics(self) -> tuple[str, ...]:
        return tuple(m for m in REPORTED if m != self.primary_metric)

    def as_dict(self) -> dict[str, Any]:
        """Event-payload shape. asdict() would drop the derived properties."""
        return {
            "primary_metric": self.primary_metric,
            "direction": self.direction,
            "secondary_metrics": list(self.secondary_metrics),
        }


DEFAULT_OBJECTIVE = Objective()


def from_plan(plan: MissionPlan | None) -> Objective:
    """Build the Mission's Objective. No plan means the historical default."""
    if plan is None:
        return DEFAULT_OBJECTIVE
    name = ALIASES.get(plan.primary_metric, plan.primary_metric)
    return Objective(primary_metric=name)


def metric_value(metrics: dict[str, Any], name: str) -> float | None:
    """Read one metric. Missing or UNAVAILABLE reads as None, never as zero."""
    value = metrics.get(name)
    if value is None or value == UNAVAILABLE:
        return None
    return float(value)


def score(result: Any, objective: Objective) -> float | None:
    """The objective's metric for one result, or None when it cannot be scored.

    Duck-typed on `result` (needs .ok and .metrics) so this module stays a leaf.
    None means "excluded from best-selection" — never substitute another metric.
    """
    if not result.ok:
        return None
    return metric_value(result.metrics, objective.primary_metric)


def reached(value: float, target: float, objective: Objective) -> bool:
    """Has `value` met `target`, read in the objective's own direction?

    Lives here because DIRECTIONS lives here. Decision asks; it does not
    re-derive which way "better" runs.
    """
    return value <= target if objective.direction == "minimize" else value >= target


def improvement(value: float, baseline: float, objective: Objective) -> float:
    """How much `value` improves on `baseline`, in the objective's direction.

    Positive always means better, whichever way the metric runs, so callers can
    compare against a single threshold without knowing the direction.
    """
    return baseline - value if objective.direction == "minimize" else value - baseline
