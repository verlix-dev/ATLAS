"""Planner verification: task text in, a structured MissionPlan out."""

from __future__ import annotations

import pandas as pd
import pytest

from atlas.data_engineer import TargetConfidence, load
from atlas.decision import Budget
from atlas.loop import Mission, mission
from atlas.planner import DEFAULT_METRIC, MetricConfidence, MissionPlan, plan
from atlas.state import Action

FRAUD_TASK = "Build a fraud detection model. Missing fraud is more costly than false alarms."


# --- problem type --------------------------------------------------------------------


def test_classification_task_is_detected() -> None:
    p = plan("Build a model to classify whether a customer will churn.")
    assert p.problem_type == "classification"


def test_regression_task_is_detected() -> None:
    p = plan("Forecast how much revenue each store will generate next quarter.")
    assert p.problem_type == "regression"


def test_silent_task_states_no_problem_type() -> None:
    """No cue either way: say nothing rather than default to classification."""
    p = plan("Do something useful with this data.")
    assert p.problem_type is None


# --- target extraction ---------------------------------------------------------------


def test_explicit_target_phrasing_is_extracted() -> None:
    assert plan("Train a model. The target is churned.").target_candidate == "churned"
    assert plan("Predict the 'fraud' column.").target_candidate == "fraud"
    assert plan("Build a model where the label is spam.").target_candidate == "spam"


def test_target_extracted_from_the_domain_phrasing() -> None:
    assert plan(FRAUD_TASK).target_candidate == "fraud"
    assert plan("Predict customer churn from usage data.").target_candidate == "churn"


def test_no_target_mentioned_stays_none() -> None:
    p = plan("Build the best classifier you can from this dataset.")
    assert p.target_candidate is None


def test_target_candidate_preserves_source_casing() -> None:
    """The candidate is passed to load() verbatim, so casing must survive."""
    assert plan("The target is Churned.").target_candidate == "Churned"


# --- metric policy -------------------------------------------------------------------


def test_balanced_classification_prefers_accuracy() -> None:
    p = plan("Classify these balanced product categories.")
    assert p.primary_metric == "accuracy"
    assert p.metric_confidence is MetricConfidence.INFERRED
    assert p.priority is None


def test_cost_of_missed_positives_prioritises_recall() -> None:
    p = plan(FRAUD_TASK)
    assert p.primary_metric == "pr_auc"
    assert p.priority == "recall"
    assert p.metric_confidence is MetricConfidence.INFERRED
    assert "missed positives cost more" in p.metric_reason


def test_disease_phrasing_also_prioritises_recall() -> None:
    p = plan("Screen patients for disease. We cannot afford to miss a positive case.")
    assert p.priority == "recall"
    assert p.primary_metric == "pr_auc"


def test_cost_of_false_positives_prioritises_precision() -> None:
    p = plan("Build a spam filter. False positives are worse than letting spam through.")
    assert p.primary_metric == "precision"
    assert p.priority == "precision"
    assert p.metric_confidence is MetricConfidence.INFERRED


def test_explicitly_named_metric_wins() -> None:
    p = plan("Build a churn classifier and optimise roc_auc.")
    assert p.primary_metric == "roc_auc"
    assert p.metric_confidence is MetricConfidence.EXPLICIT


def test_ambiguous_objective_is_reported_not_guessed() -> None:
    p = plan("Do something useful with this data.")
    assert p.metric_confidence is MetricConfidence.AMBIGUOUS
    assert p.primary_metric == DEFAULT_METRIC  # matches the loop, claims nothing
    assert "neither a problem type nor an error preference" in p.metric_reason


def test_both_errors_costly_is_ambiguous_not_arbitrarily_resolved() -> None:
    p = plan("Detect fraud. Missing fraud is costly and false alarms are costly too.")
    assert p.metric_confidence is MetricConfidence.AMBIGUOUS
    assert p.priority is None
    assert "without ranking them" in p.metric_reason


def test_bare_cost_phrasing_still_matches_recall_side() -> None:
    """Regression: only comparative phrasing used to match, so a task naming both
    costs without ranking them was silently resolved to precision."""
    p = plan("Detect fraud. Missing a fraudulent transaction is costly.")
    assert p.priority == "recall"
    assert p.metric_confidence is MetricConfidence.INFERRED


def test_regression_metric_selection() -> None:
    assert plan("Forecast revenue for each store.").primary_metric == "rmse"
    assert plan("Forecast revenue; be robust to outliers.").primary_metric == "mae"
    assert plan("Forecast revenue and report variance explained.").primary_metric == "r2"


# --- constraints and budget ----------------------------------------------------------


def test_constraints_are_extracted() -> None:
    p = plan("Predict churn. The model must be interpretable for regulators. Run on CPU only.")
    assert len(p.constraints) == 2
    assert any("interpretable" in c for c in p.constraints)
    assert any("CPU only" in c for c in p.constraints)


def test_no_constraints_gives_an_empty_list() -> None:
    assert plan("Predict churn from this data.").constraints == []


def test_budget_is_extracted() -> None:
    assert plan("Detect fraud. Try at most 5 experiments.").experiment_budget == 5
    assert plan("Detect fraud. Keep the experiment budget to 3.").experiment_budget == 3
    assert plan("Detect fraud.").experiment_budget is None


def test_implausible_budget_is_ignored() -> None:
    """A number about something else must not become the budget."""
    assert plan("Detect fraud across 50000 experiments.").experiment_budget is None


# --- invalid input -------------------------------------------------------------------


def test_empty_task_is_rejected() -> None:
    for bad in ("", "   ", "\n\t"):
        with pytest.raises(ValueError, match="task is empty"):
            plan(bad)


def test_plan_is_a_structured_object_not_a_dict() -> None:
    assert isinstance(plan(FRAUD_TASK), MissionPlan)


# --- integration: the candidate reaches the Data Engineer ----------------------------


def test_planner_candidate_takes_the_existing_explicit_path() -> None:
    """A valid candidate is accepted exactly as a human-supplied --target is."""
    p = plan(FRAUD_TASK)
    data = load("data/fraud.csv", p.target_candidate)  # the real call site
    assert p.target_candidate == "fraud"
    assert data.profile.target == "fraud"
    assert data.profile.target_confidence is TargetConfidence.EXPLICIT
    assert data.profile.target_reason == "target supplied by the user"


def test_no_candidate_leaves_data_engineer_inference_untouched() -> None:
    p = plan("Build the best classifier you can from this dataset.")
    assert p.target_candidate is None
    data = load("data/fraud.csv", p.target_candidate)
    assert data.profile.target == "fraud"  # positional inference, unchanged
    assert data.profile.target_confidence is TargetConfidence.INFERRED


def test_planner_does_not_intercept_the_blocked_path() -> None:
    """A structurally invalid candidate must reach the existing blocked path intact."""
    p = plan("Predict the price of each house. The target is price.")
    assert p.target_candidate == "price"  # the Planner extracted it and did NOT judge it

    seen: list[dict] = []
    run_ = mission("data/houses.csv", p.target_candidate, sink=seen.append, plan=p)
    experiments = list(run_.run())

    assert experiments == []
    assert [e["stage"] for e in seen] == ["data_engineer", "mission_end"]
    assert seen[0]["status"] == "needs_clarification"
    assert "classification only" in seen[0]["reason"]
    assert [e["seq"] for e in seen] == [0, 1]


def test_unknown_candidate_column_still_raises() -> None:
    """Milestone 2 behaviour: a column absent from the frame raises, not blocks."""
    p = plan("Predict customer churn.")
    assert p.target_candidate == "churn"  # the real column is 'churned'
    with pytest.raises(ValueError, match=r"target 'churn' not in columns"):
        load("data/churn.csv", p.target_candidate)


# --- integration: the full pipeline --------------------------------------------------


def test_full_pipeline_keeps_every_milestone_guarantee() -> None:
    p = plan(FRAUD_TASK + " Try at most 3 experiments.")
    run_ = mission("data/fraud.csv", p.target_candidate, budget=Budget(max_experiments=p.experiment_budget), plan=p)
    experiments = list(run_.run())

    assert p.experiment_budget == 3
    assert len(experiments) == 3
    for previous, current in zip(experiments, experiments[1:]):
        assert current.config == previous.hypothesis.config  # autonomy intact
    assert experiments[-1].decision.action is Action.STOP  # termination intact
    fingerprints = [e.config.fingerprint() for e in experiments]
    assert len(set(fingerprints)) == len(fingerprints)  # no repeats
    assert run_.summary()["best_score"] is not None


def test_event_stream_shape_is_unchanged_by_the_planner() -> None:
    seen: list[str] = []
    p = plan(FRAUD_TASK)
    run_ = mission("data/fraud.csv", p.target_candidate, budget=Budget(max_experiments=2), sink=lambda e: seen.append(e["stage"]), plan=p)
    list(run_.run())

    assert seen[0] == "mission_start" and seen[-1] == "mission_end"
    assert seen[1:6] == ["experiment_start", "experiment_result", "diagnosis", "hypothesis", "decision"]
    assert "planner" not in seen and "planner_start" not in seen  # no new stages
    assert [e["seq"] for e in run_.events] == list(range(len(run_.events)))


def test_plan_rides_along_in_mission_start() -> None:
    p = plan(FRAUD_TASK)
    run_ = mission("data/fraud.csv", p.target_candidate, budget=Budget(max_experiments=1), plan=p)
    list(run_.run())
    payload = run_.events[0]["plan"]
    assert isinstance(payload, dict)
    assert payload["primary_metric"] == "pr_auc"
    assert payload["target_candidate"] == "fraud"


def test_mission_without_a_planner_still_works() -> None:
    """The Planner is optional: the Milestone 2 entry point is unchanged."""
    run_ = mission("data/fraud.csv", "fraud", budget=Budget(max_experiments=1))
    list(run_.run())
    assert run_.plan is None
    assert run_.events[0]["plan"] is None
    assert run_.summary()["best_score"] is not None


def test_planner_metric_now_drives_ranking() -> None:
    """Milestone 4 inverts Milestone 3's assertion: the mismatch is gone.

    This test previously asserted 'pr_auc' not in metrics and that score() was
    f1_macro. Both are now false by design — the engine computes the plan's
    metric and History ranks on it.
    """
    p = plan(FRAUD_TASK)
    run_ = mission("data/fraud.csv", p.target_candidate, budget=Budget(max_experiments=1), plan=p)
    list(run_.run())

    assert p.primary_metric == "pr_auc"
    assert run_.objective.primary_metric == "pr_auc"
    best = run_.history.best()
    assert "pr_auc" in best.result.metrics  # the engine now computes it
    assert best.result.score() == best.result.metrics["pr_auc"]
    assert best.result.score() != best.result.metrics["f1_macro"]
