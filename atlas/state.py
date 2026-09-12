"""Explicit experiment state. Structured data passed between ATLAS components."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, replace
from enum import StrEnum
from typing import Any

from atlas.objective import DEFAULT_METRIC, DEFAULT_OBJECTIVE, Objective, metric_value, score


class Category(StrEnum):
    """Diagnosis categories. Not every bad result is a runtime error."""

    EXECUTION_ERROR = "execution_error"
    DATA_ISSUE = "data_issue"
    MODEL_ISSUE = "model_issue"
    INSUFFICIENT = "insufficient"
    IMPROVEMENT = "improvement"


class Action(StrEnum):
    CONTINUE = "continue"
    REVISE = "revise"
    STOP = "stop"


@dataclass(frozen=True)
class ExperimentConfig:
    model: str
    params: dict[str, Any] = field(default_factory=dict)
    preprocess: dict[str, Any] = field(default_factory=lambda: {"impute": "median", "scale": True})

    def fingerprint(self) -> str:
        """Identity of an experiment, for repeated-experiment detection."""
        return json.dumps(asdict(self), sort_keys=True, default=str)

    def evolve(self, **changes: Any) -> ExperimentConfig:
        """Copy with shallow-merged params/preprocess so callers can't mutate history."""
        params = {**self.params, **changes.pop("params", {})}
        preprocess = {**self.preprocess, **changes.pop("preprocess", {})}
        return replace(self, params=params, preprocess=preprocess, **changes)


@dataclass
class ExperimentResult:
    ok: bool
    metrics: dict[str, Any] = field(default_factory=dict)
    error: str | None = None
    traceback: str | None = None
    stdout: str = ""
    stderr: str = ""
    warnings: list[str] = field(default_factory=list)
    seconds: float = 0.0
    # Which metric was primary when this experiment ran, so a selection can be
    # explained after the fact. The engine sets it from the Mission's objective.
    primary_metric: str = DEFAULT_METRIC

    def score(self) -> float | None:
        """This result's own primary metric. None when absent or unavailable."""
        return metric_value(self.metrics, self.primary_metric) if self.ok else None


@dataclass
class Diagnosis:
    category: Category
    summary: str
    findings: list[str] = field(default_factory=list)
    evidence: dict[str, Any] = field(default_factory=dict)


@dataclass
class Hypothesis:
    problem: str
    proposed_change: str
    reasoning: str
    expected_effect: str
    config: ExperimentConfig
    touches_data: bool = False  # True -> routes back to data preparation (REVISE)
    # Provenance. Set by the validator for an LLM proposal, never self-reported:
    # an LLM cannot claim to be the deterministic engine, or vice versa.
    source: str = "deterministic"
    cites: list[str] = field(default_factory=list)  # diagnosis evidence leaned on


@dataclass
class Decision:
    action: Action
    reason: str


@dataclass
class Experiment:
    id: int
    config: ExperimentConfig
    result: ExperimentResult
    diagnosis: Diagnosis
    hypothesis: Hypothesis | None = None  # what ATLAS wants to try NEXT
    decision: Decision | None = None
    origin: str = "baseline"  # hypothesis that produced this experiment


class History:
    """In-process experiment memory. ATLAS must know what it already tried.

    Ranking is objective-aware: one Objective per Mission, since REVISE
    changes preprocessing and model config but never the objective.
    """

    def __init__(self, objective: Objective | None = None) -> None:
        self.experiments: list[Experiment] = []
        self.objective = objective or DEFAULT_OBJECTIVE

    def __len__(self) -> int:
        return len(self.experiments)

    def add(self, experiment: Experiment) -> None:
        self.experiments.append(experiment)

    def tried(self, config: ExperimentConfig) -> bool:
        fp = config.fingerprint()
        return any(e.config.fingerprint() == fp for e in self.experiments)

    def successful(self) -> list[Experiment]:
        return [e for e in self.experiments if e.result.ok]

    def scorable(self) -> list[tuple[Experiment, float]]:
        """Successful experiments whose objective metric was actually computed.

        An experiment whose primary metric is UNAVAILABLE stays in History with
        that fact recorded, but is excluded from best-selection rather than
        being scored as zero or silently rescored on a different metric.
        """
        scored = ((e, score(e.result, self.objective)) for e in self.successful())
        return [(e, s) for e, s in scored if s is not None]

    def best(self) -> Experiment | None:
        candidates = self.scorable()
        if not candidates:
            return None
        pick = min if self.objective.direction == "minimize" else max
        return pick(candidates, key=lambda pair: pair[1])[0]

    def best_score(self) -> float | None:
        best = self.best()
        return score(best.result, self.objective) if best else None

    def rounds_since_improvement(self) -> int:
        """How many experiments since the current best was found."""
        best = self.best()
        return len(self.experiments) - (self.experiments.index(best) + 1) if best else len(self.experiments)
