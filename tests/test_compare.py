"""M6 verification: the comparison itself is fair and reports what actually happened.

Deliberately small. M5 already proves the LLM cannot fabricate metrics, bypass
validation, or escape the control flow; none of that is re-tested here. These
four tests cover only the claims the *comparison* adds.
"""

from __future__ import annotations

import pytest

from atlas.data_engineer import load
from atlas.decision import Budget
from compare_strategies import EvidenceLedProposer, run_comparison

BUDGET = Budget(max_experiments=4)


@pytest.fixture(scope="module")
def report() -> dict:
    return run_comparison(load("data/churn.csv", "churned"), BUDGET)


def _trajectory(run) -> list[tuple]:
    return [
        (e.config.model, tuple(sorted(e.config.params.items())),
         tuple(sorted(e.config.preprocess.items())), round(e.result.score(), 8))
        for e in run.history.experiments
    ]


def test_both_strategies_run_under_identical_conditions(report) -> None:
    """One Dataset, one objective, one budget — the whole fairness claim."""
    deterministic, guided = report["deterministic"], report["guided"]

    # The same split object, not two equal-looking loads.
    assert deterministic.data is guided.data
    assert deterministic.data.X_train is guided.data.X_train
    assert deterministic.data.y_test is guided.data.y_test

    assert deterministic.objective is guided.objective
    assert deterministic.objective.primary_metric == "f1_macro"
    assert deterministic.budget == guided.budget == BUDGET

    # Neither side got extra runway.
    assert len(deterministic.history) == len(guided.history) == BUDGET.max_experiments
    assert deterministic.history is not guided.history


def test_reported_best_is_the_actual_best(report) -> None:
    """The headline number is the argmax over what was really measured."""
    for run in (report["deterministic"], report["guided"]):
        summary = run.summary()
        scorable = run.history.scorable()
        assert summary["best_score"] == max(s for _, s in scorable)

        best = run.history.best()
        assert summary["best_id"] == best.id
        # ...and it is that experiment's own engine-measured metric, not a
        # number the comparison computed for itself.
        assert summary["best_score"] == best.result.metrics["f1_macro"]


def test_llm_reaches_a_configuration_the_deterministic_run_does_not(report) -> None:
    """Without this the comparison would be demonstrating nothing."""
    ladder = {e.config.fingerprint() for e in report["deterministic"].history.experiments}
    assert report["novel"], "the proposer produced no configuration outside the ladder"
    for e in report["novel"]:
        assert e.config.fingerprint() not in ladder
        assert "[llm]" in e.origin
        assert e.result.ok  # it really ran

    # At least one differs in model parameters, not merely in preprocessing.
    ladder_params = {
        (e.config.model, tuple(sorted(e.config.params.items())))
        for e in report["deterministic"].history.experiments
    }
    assert any(
        (e.config.model, tuple(sorted(e.config.params.items()))) not in ladder_params
        for e in report["novel"]
    ), "every 'novel' config differs only in preprocessing"


def test_the_comparison_is_reproducible() -> None:
    """Same inputs, same trajectories — the proposer is a pure function of context."""
    budget = Budget(max_experiments=3)
    first = run_comparison(load("data/churn.csv", "churned"), budget)
    second = run_comparison(load("data/churn.csv", "churned"), budget)

    for side in ("deterministic", "guided"):
        assert _trajectory(first[side]) == _trajectory(second[side])
        assert first[side].summary()["best_score"] == second[side].summary()["best_score"]
    assert first["proposer"].decisions == second["proposer"].decisions
    assert isinstance(first["proposer"], EvidenceLedProposer)
