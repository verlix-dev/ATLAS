"""Hypothesis Engine: evidence + history + diagnosis -> the next experiment.

Two proposers, one output. The deterministic ladder is authoritative and always
present; an LLM, when supplied, gets first refusal but must pass llm.validate()
before its proposal is allowed to become an experiment. Returning None is
meaningful — it tells Decision no useful hypothesis remains.
"""

from __future__ import annotations

from typing import Any

from atlas import llm as llm_seam
from atlas.ml_engineer import LADDER
from atlas.objective import Objective
from atlas.state import Category, Diagnosis, ExperimentConfig, History, Hypothesis

# Regularisation strengths to walk when a linear model overfits.
WEAKER_C = [0.1, 0.01]


def _next_model(history: History, current: str) -> str | None:
    """Next unused model on the escalation ladder."""
    used = {e.config.model for e in history.experiments}
    start = LADDER.index(current) + 1 if current in LADDER else 0
    return next((m for m in LADDER[start:] if m not in used), None)


def _candidates(
    diagnosis: Diagnosis, current: ExperimentConfig, profile: dict[str, Any], history: History
) -> list[Hypothesis]:
    """Ordered candidate hypotheses, most-targeted first. Filtered for repeats later."""
    out: list[Hypothesis] = []
    evidence = diagnosis.evidence
    weak = evidence.get("weak_classes") or {}
    # Absent or UNAVAILABLE means not measured: never state a ratio we do not have.
    ratio = evidence.get("imbalance_ratio")
    if not isinstance(ratio, (int, float)):
        ratio = profile.get("imbalance_ratio")
    imbalance_note = (
        f"imbalance is {ratio:.2f}:1, so the loss is dominated by the majority class"
        if isinstance(ratio, (int, float))
        else "the imbalance ratio was not measured, but minority recall is weak"
    )

    if diagnosis.category is Category.DATA_ISSUE:
        out.append(
            Hypothesis(
                problem=diagnosis.summary,
                proposed_change="switch numeric imputation to most_frequent and rebuild encoding",
                reasoning="the failure signature points at input values, not at model code",
                expected_effect="the pipeline completes and produces metrics",
                config=current.evolve(preprocess={"impute": "most_frequent"}),
                touches_data=True,
            )
        )

    if diagnosis.category is Category.EXECUTION_ERROR:
        fallback = _next_model(history, current.model) or "random_forest"
        out.append(
            Hypothesis(
                problem=diagnosis.summary,
                proposed_change=f"drop the failing configuration and run {fallback} with defaults",
                reasoning="an execution error invalidates this configuration; a known-good one restores signal",
                expected_effect="a completed run that can be measured",
                config=ExperimentConfig(model=fallback),
            )
        )
        return out

    # Weak minority-class recall is the most actionable performance finding.
    if weak and "class_weight" not in current.params:
        out.append(
            Hypothesis(
                problem=f"minority class recall is {min(weak.values()):.3f}",
                proposed_change="set class_weight='balanced'",
                reasoning=imbalance_note,
                expected_effect="higher minority-class recall lifts macro f1",
                config=current.evolve(params={"class_weight": "balanced"}),
            )
        )

    # UNAVAILABLE (a failed experiment measured no gap) must not be compared
    # numerically, and absence must not read as a measured 0.0.
    gap = evidence.get("overfit_gap")
    if isinstance(gap, (int, float)) and gap > 0.15:
        if current.model == "logreg":
            for c in WEAKER_C:
                if current.params.get("C") != c:
                    out.append(
                        Hypothesis(
                            problem=f"train/test gap is {gap:.3f}",
                            proposed_change=f"tighten regularisation to C={c}",
                            reasoning="a smaller C penalises coefficient magnitude and reduces variance",
                            expected_effect="a smaller gap and steadier test f1",
                            config=current.evolve(params={"C": c}),
                        )
                    )
        else:
            out.append(
                Hypothesis(
                    problem=f"train/test gap is {gap:.3f}",
                    proposed_change="constrain tree depth to 6 and require 5 samples per leaf",
                    reasoning="unconstrained trees memorise the training split",
                    expected_effect="a smaller gap at similar test f1",
                    config=current.evolve(params={"max_depth": 6} | ({"min_samples_leaf": 5} if current.model == "random_forest" else {})),
                )
            )

    # Escalate capacity when the current family has been explored.
    upgrade = _next_model(history, current.model)
    if upgrade:
        carried = {"class_weight": "balanced"} if weak or "class_weight" in current.params else {}
        out.append(
            Hypothesis(
                problem=diagnosis.summary,
                proposed_change=f"escalate from {current.model} to {upgrade}",
                reasoning="the current family has been explored; a higher-capacity model may capture non-linear structure",
                expected_effect="a macro f1 above the current best",
                config=ExperimentConfig(
                    model=upgrade,
                    params=carried,
                    preprocess={"impute": current.preprocess.get("impute", "median"), "scale": False},
                ),
            )
        )
    return out


def propose(
    diagnosis: Diagnosis,
    history: History,
    profile: dict[str, Any],
    *,
    llm: llm_seam.Llm | None = None,
    objective: Objective | None = None,
    budget: Any = None,
    rejections: list[dict[str, str]] | None = None,
) -> Hypothesis | None:
    """First usable candidate: the LLM's if it passes validation, else the ladder.

    `llm=None` is the Milestone 1-4 path, unchanged. Every LLM failure or
    rejection is appended to `rejections` and falls through to the deterministic
    ladder, so the loop always has the provable engine underneath it.
    """
    current = history.experiments[-1].config

    if llm is not None:
        if objective is None or budget is None:
            # Fail loudly rather than silently degrading to the deterministic
            # path: a caller that wanted an LLM should know it got none.
            raise ValueError("an llm proposer needs `objective` and `budget` to build context")
        payload = llm_seam.context(diagnosis, history, profile, objective, budget, current)
        raw, reason = llm_seam._ask(llm, payload)
        if raw is not None:
            candidate, reason = llm_seam.validate(raw, diagnosis, history, current)
            if candidate is not None:
                return candidate
        if rejections is not None:
            rejections.append({"source": "llm", "reason": reason})

    for candidate in _candidates(diagnosis, current, profile, history):
        if not history.tried(candidate.config):
            return candidate
    return None
