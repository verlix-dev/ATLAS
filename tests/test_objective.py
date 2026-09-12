"""Objective-aware scoring and ranking.

The central claim under test: the same two experiment results rank differently
depending only on the Mission's objective. Every test routes through the real
Objective / score() / History / decide() path — none reimplements comparison.
"""

from __future__ import annotations

import io
from contextlib import redirect_stdout

import pandas as pd
import pytest

from atlas.data_engineer import load
from atlas.decision import Budget, decide
from atlas.diagnose import diagnose
from atlas.engine import run
from atlas.loop import Mission, mission
from atlas.objective import (
    DEFAULT_OBJECTIVE,
    DIRECTIONS,
    REPORTED,
    UNAVAILABLE,
    Objective,
    from_plan,
    metric_value,
    score,
)
from atlas.planner import MissionPlan, plan
from atlas.state import (
    Action,
    Category,
    Decision,
    Diagnosis,
    Experiment,
    ExperimentConfig,
    ExperimentResult,
    History,
    Hypothesis,
)

# --- the adversarial pair ------------------------------------------------------------
# A wins on f1_macro, B wins on pr_auc. Built once, ranked under both objectives.
A_METRICS = {"f1_macro": 0.80, "pr_auc": 0.70, "accuracy": 0.88, "recall": 0.55, "precision": 0.91}
B_METRICS = {"f1_macro": 0.75, "pr_auc": 0.85, "accuracy": 0.82, "recall": 0.80, "precision": 0.66}


def _experiment(id: int, metrics: dict, model: str) -> Experiment:
    return Experiment(
        id=id,
        config=ExperimentConfig(model=model),
        result=ExperimentResult(ok=True, metrics=dict(metrics)),
        diagnosis=Diagnosis(Category.IMPROVEMENT, "x"),
        decision=Decision(Action.CONTINUE, "x"),
    )


@pytest.fixture
def pair() -> tuple[Experiment, Experiment]:
    """One A and one B. The SAME objects are ranked under both objectives."""
    return _experiment(1, A_METRICS, "logreg"), _experiment(2, B_METRICS, "random_forest")


def _history(objective: Objective, experiments) -> History:
    history = History(objective)
    for e in experiments:
        history.add(e)
    return history


# --- THE critical test: ranking flips with the objective -----------------------------


def test_f1_objective_selects_a(pair) -> None:
    a, b = pair
    history = _history(Objective("f1_macro"), [a, b])
    assert history.best() is a
    assert history.best_score() == 0.80


def test_pr_auc_objective_selects_b(pair) -> None:
    a, b = pair
    history = _history(Objective("pr_auc"), [a, b])
    assert history.best() is b
    assert history.best_score() == 0.85


def test_same_results_rank_differently_under_each_objective() -> None:
    """Both objectives see byte-identical inputs; only the objective differs."""
    a, b = _experiment(1, A_METRICS, "logreg"), _experiment(2, B_METRICS, "random_forest")

    by_f1 = _history(Objective("f1_macro"), [a, b]).best()
    by_pr = _history(Objective("pr_auc"), [a, b]).best()

    assert by_f1 is a and by_pr is b
    assert by_f1 is not by_pr
    # The results themselves were never mutated between rankings.
    assert a.result.metrics == A_METRICS and b.result.metrics == B_METRICS


def test_decide_follows_the_objective_not_a_hardcoded_metric(pair) -> None:
    """decide() reads score() — the same pair produces opposite stop decisions."""
    import time

    a, b = pair
    now = time.perf_counter()
    budget = Budget(max_experiments=9, target_score=0.82, patience=5)
    proposal = None  # forces STOP either way; the reason is what differs

    by_f1 = decide(Diagnosis(Category.IMPROVEMENT, "x"), proposal, _history(Objective("f1_macro"), [a, b]), budget, now)
    by_pr = decide(Diagnosis(Category.IMPROVEMENT, "x"), proposal, _history(Objective("pr_auc"), [a, b]), budget, now)

    # best f1 is 0.80, under the 0.82 target -> not "objective reached"
    assert "objective reached" not in by_f1.reason
    # best pr_auc is 0.85, over the same target -> "objective reached"
    assert "objective reached" in by_pr.reason
    assert by_pr.action is Action.STOP


def test_rounds_since_improvement_follows_the_objective(pair) -> None:
    a, b = pair
    assert _history(Objective("f1_macro"), [a, b]).rounds_since_improvement() == 1  # best is A, one ago
    assert _history(Objective("pr_auc"), [a, b]).rounds_since_improvement() == 0  # best is B, the last one


# --- direction -----------------------------------------------------------------------


def test_maximize_direction_picks_the_highest() -> None:
    low, high = _experiment(1, {"recall": 0.40}, "logreg"), _experiment(2, {"recall": 0.90}, "hist_gb")
    objective = Objective("recall")
    assert objective.direction == "maximize"
    assert _history(objective, [low, high]).best() is high


def test_minimize_direction_picks_the_lowest() -> None:
    """MAE is typed as minimize. Regression is not wired up, so score()'s
    direction logic is exercised in isolation, per the milestone's scope note."""
    worse, better = _experiment(1, {"mae": 12.5}, "logreg"), _experiment(2, {"mae": 3.2}, "hist_gb")
    objective = Objective("mae")
    assert objective.direction == "minimize"
    assert _history(objective, [worse, better]).best() is better
    assert _history(objective, [worse, better]).best_score() == 3.2


def test_direction_table_covers_every_planner_metric() -> None:
    from atlas.planner import NAMED_METRICS
    from atlas.objective import ALIASES

    for name in NAMED_METRICS:
        resolved = ALIASES.get(name, name)
        assert resolved in DIRECTIONS, f"planner can emit {name!r} with no direction"


def test_decide_honours_the_objective_direction_at_the_target() -> None:
    """A minimize objective is reached by going BELOW the target, not above it."""
    import time

    now = time.perf_counter()
    proposal = Hypothesis("p", "c", "r", "e", ExperimentConfig(model="hist_gb"))
    ok = Diagnosis(Category.IMPROVEMENT, "x")

    def reason(objective: Objective, metrics: dict, target: float) -> str:
        history = _history(objective, [_experiment(1, metrics, "logreg")])
        budget = Budget(max_experiments=9, target_score=target, patience=5)
        return decide(ok, proposal, history, budget, now).reason

    # minimize: 3.2 beats a 5.0 target; 12.5 does not. The old `>=` had these
    # exactly backwards and stopped on the worse value.
    assert "objective reached" in reason(Objective("mae"), {"mae": 3.2}, 5.0)
    assert "objective reached" not in reason(Objective("mae"), {"mae": 12.5}, 5.0)

    # maximize is unchanged.
    assert "objective reached" in reason(Objective("f1_macro"), {"f1_macro": 0.96}, 0.95)
    assert "objective reached" not in reason(Objective("f1_macro"), {"f1_macro": 0.80}, 0.95)


def test_diagnose_gain_means_improvement_in_either_direction() -> None:
    """gain is 'amount of improvement', so one MIN_GAIN threshold works both ways."""
    mae_history = _history(Objective("mae"), [_experiment(1, {"mae": 12.5}, "logreg")])

    # minimize, the right way: 12.5 -> 3.2 is a 9.3 improvement, not a -9.3 loss.
    better = diagnose(ExperimentResult(ok=True, metrics={"mae": 3.2}, primary_metric="mae"), mae_history, {})
    assert better.evidence["gain"] == pytest.approx(9.3)
    assert better.category is Category.IMPROVEMENT

    # minimize, the wrong way: a rising mae must never read as an improvement.
    worse = diagnose(ExperimentResult(ok=True, metrics={"mae": 20.0}, primary_metric="mae"), mae_history, {})
    assert worse.evidence["gain"] == pytest.approx(-7.5)
    assert worse.category is not Category.IMPROVEMENT

    # maximize is unchanged.
    f1_history = _history(Objective("f1_macro"), [_experiment(1, {"f1_macro": 0.60}, "logreg")])
    up = diagnose(ExperimentResult(ok=True, metrics={"f1_macro": 0.75}), f1_history, {})
    assert up.evidence["gain"] == pytest.approx(0.15)
    assert up.category is Category.IMPROVEMENT


def test_unknown_metric_is_rejected_at_construction() -> None:
    with pytest.raises(ValueError, match="unknown metric 'nonsense'"):
        Objective("nonsense")


# --- unavailable metrics -------------------------------------------------------------


def test_unavailable_metric_scores_as_none_not_zero() -> None:
    result = ExperimentResult(ok=True, metrics={"f1_macro": 0.7, "pr_auc": UNAVAILABLE})
    assert score(result, Objective("pr_auc")) is None
    assert score(result, Objective("f1_macro")) == 0.7
    assert metric_value(result.metrics, "pr_auc") is None
    assert metric_value(result.metrics, "missing_entirely") is None


def test_unscorable_experiment_is_excluded_but_retained() -> None:
    """The documented fallback: recorded in History, excluded from best-selection."""
    scorable = _experiment(1, {"f1_macro": 0.60, "pr_auc": 0.55}, "logreg")
    unscorable = _experiment(2, {"f1_macro": 0.99, "pr_auc": UNAVAILABLE}, "hist_gb")
    history = _history(Objective("pr_auc"), [scorable, unscorable])

    assert len(history) == 2  # both retained
    assert len(history.successful()) == 2
    assert len(history.scorable()) == 1  # only one rankable
    assert history.best() is scorable  # NOT the 0.99 f1 one: no silent substitution
    assert history.best_score() == 0.55


def test_unavailable_throughout_gives_no_best() -> None:
    a = _experiment(1, {"f1_macro": 0.8, "pr_auc": UNAVAILABLE}, "logreg")
    b = _experiment(2, {"f1_macro": 0.9, "pr_auc": UNAVAILABLE}, "hist_gb")
    history = _history(Objective("pr_auc"), [a, b])
    assert history.best() is None
    assert history.best_score() is None
    assert len(history) == 2  # still remembered


def test_failed_experiment_never_scores() -> None:
    failed = ExperimentResult(ok=False, error="boom", metrics={"pr_auc": 0.99})
    assert score(failed, Objective("pr_auc")) is None


def test_diagnosis_reports_unavailable_rather_than_a_zero_score() -> None:
    """An unmeasured metric must read as unavailable, never stand in as 0.0."""
    result = ExperimentResult(
        ok=True,
        metrics={
            "f1_macro": 0.71,
            "pr_auc": UNAVAILABLE,
            "overfit_gap": 0.01,
            "per_class_recall": {"0": 0.9, "1": 0.8},
            "labels": ["0", "1"],
            "predicted_classes": 2,
        },
        primary_metric="pr_auc",
    )
    history = History(Objective("pr_auc"))

    first = diagnose(result, history, {})
    assert first.evidence["score"] == UNAVAILABLE  # not 0.0
    assert first.evidence["gain"] is None
    assert "pr_auc" in first.summary and UNAVAILABLE in first.summary
    assert "0.000" not in first.summary

    # With a scorable baseline in history, the gain is still not invented:
    # the old `score or 0.0` would have claimed a -0.65 regression here.
    history.add(_experiment(1, {"pr_auc": 0.65}, "logreg"))
    later = diagnose(result, history, {})
    assert later.evidence["score"] == UNAVAILABLE
    assert later.evidence["gain"] is None
    assert later.evidence["previous_best"] == 0.65
    assert "0.000" not in later.summary
    assert history.best_score() == 0.65  # ranking behaviour unchanged


def test_unavailable_metric_never_reaches_the_event_stream_as_a_number(tmp_path) -> None:
    """End-to-end: a whole mission on an unrankable objective prints no fake score."""
    frame = pd.DataFrame({"x": list(range(60)), "grade": ["a", "b", "c"] * 20})
    csv = tmp_path / "multi.csv"
    frame.to_csv(csv, index=False)
    fake_plan = MissionPlan(
        objective="x", problem_type="classification", target_candidate="grade",
        primary_metric="pr_auc", metric_confidence="inferred", metric_reason="test",
    )
    events: list[dict] = []
    run_ = mission(csv, "grade", budget=Budget(max_experiments=3, patience=2),
                   sink=events.append, plan=fake_plan)
    list(run_.run())

    summaries = [e["summary"] for e in events if e["stage"] == "diagnosis"]
    assert summaries
    for summary in summaries:
        assert UNAVAILABLE in summary
        assert "0.000" not in summary
    assert run_.history.best_score() is None  # still excluded from ranking
    assert run_.summary()["best_id"] is None


def test_multiclass_target_marks_auc_unavailable(tmp_path) -> None:
    """>2 classes: reported as unavailable, never fabricated by one-vs-rest."""
    frame = pd.DataFrame(
        {
            "x": list(range(60)),
            "y": [i % 5 for i in range(60)],
            "grade": ["a", "b", "c"] * 20,
        }
    )
    csv = tmp_path / "multi.csv"
    frame.to_csv(csv, index=False)
    data = load(csv, "grade")

    result = run(data, ExperimentConfig(model="random_forest"), Objective("pr_auc"))
    assert result.ok
    assert result.metrics["roc_auc"] == UNAVAILABLE
    assert result.metrics["pr_auc"] == UNAVAILABLE
    assert isinstance(result.metrics["f1_macro"], float)  # still measured
    assert any("3 classes" in w for w in result.warnings)
    assert any("excluded from best-selection" in w for w in result.warnings)
    assert result.score() is None


def test_multiclass_mission_stops_instead_of_looping(tmp_path) -> None:
    """An unrankable mission must still terminate, not spin forever."""
    frame = pd.DataFrame({"x": list(range(60)), "grade": ["a", "b", "c"] * 20})
    csv = tmp_path / "multi.csv"
    frame.to_csv(csv, index=False)

    fake_plan = MissionPlan(
        objective="x", problem_type="classification", target_candidate="grade",
        primary_metric="pr_auc", metric_confidence="inferred", metric_reason="test",
    )
    run_ = mission(csv, "grade", budget=Budget(max_experiments=6, patience=2), plan=fake_plan)
    experiments = list(run_.run())

    assert len(experiments) >= 1
    assert experiments[-1].decision.action is Action.STOP
    assert run_.summary()["best_id"] is None
    assert run_.summary()["experiments"] == len(experiments)  # all retained


# --- the engine computes the reported set --------------------------------------------


def test_engine_computes_every_reported_metric() -> None:
    data = load("data/fraud.csv", "fraud")
    result = run(data, ExperimentConfig(model="random_forest", params={"class_weight": "balanced"}))
    for name in REPORTED:
        assert name in result.metrics, f"{name} not computed"
        assert isinstance(result.metrics[name], float), f"{name} is {result.metrics[name]!r}"
    assert 0.0 <= result.metrics["pr_auc"] <= 1.0
    assert 0.0 <= result.metrics["roc_auc"] <= 1.0


def test_precision_recall_measure_the_minority_class() -> None:
    """Binary recall means recall of the class being detected, not the macro average."""
    data = load("data/fraud.csv", "fraud")
    result = run(data, ExperimentConfig(model="logreg", params={"class_weight": "balanced"}))
    assert result.metrics["positive_class"] == "1"  # the minority class
    assert result.metrics["recall"] == pytest.approx(result.metrics["per_class_recall"]["1"])


def test_engine_stamps_the_primary_metric_on_the_result() -> None:
    data = load("data/fraud.csv", "fraud")
    assert run(data, ExperimentConfig(model="logreg"), Objective("recall")).primary_metric == "recall"
    assert run(data, ExperimentConfig(model="logreg")).primary_metric == "f1_macro"  # default


# --- objective is preserved in history -----------------------------------------------


def test_objective_is_retrievable_after_the_mission_ends() -> None:
    p = plan("Build a fraud detection model. Missing fraud is more costly than false alarms.")
    run_ = mission("data/fraud.csv", "fraud", budget=Budget(max_experiments=2), plan=p)
    list(run_.run())

    assert run_.objective.primary_metric == "pr_auc"
    assert run_.history.objective is run_.objective  # one objective, not a copy
    assert run_.summary()["objective"] == {
        "primary_metric": "pr_auc",
        "direction": "maximize",
        "secondary_metrics": ["accuracy", "precision", "recall", "f1_macro", "roc_auc"],
    }


def test_each_experiment_records_which_metric_was_primary() -> None:
    p = plan("Detect fraud. We cannot afford to miss a positive case.")
    run_ = mission("data/fraud.csv", "fraud", budget=Budget(max_experiments=2), plan=p)
    list(run_.run())

    for entry in run_.summary()["timeline"]:
        assert entry["primary_metric"] == run_.objective.primary_metric
        assert entry["score"] is not None
    for e in run_.history.experiments:
        assert e.result.primary_metric == "pr_auc"
        assert e.result.metrics["pr_auc"] == e.result.score()


def test_objective_rides_on_mission_start_and_end() -> None:
    p = plan("Build a fraud detection model. Missing fraud is more costly than false alarms.")
    seen: list[dict] = []
    run_ = mission("data/fraud.csv", "fraud", budget=Budget(max_experiments=1), sink=seen.append, plan=p)
    list(run_.run())

    start = next(e for e in seen if e["stage"] == "mission_start")
    end = next(e for e in seen if e["stage"] == "mission_end")
    assert start["objective"]["primary_metric"] == "pr_auc"
    assert start["objective"]["direction"] == "maximize"
    assert end["objective"] == start["objective"]
    assert [e["seq"] for e in seen] == list(range(len(seen)))  # seq discipline intact


# --- end to end ----------------------------------------------------------------------


def test_planner_to_objective_to_ranking_end_to_end() -> None:
    """The success criterion: the fraud task's plan drives the final selection."""
    p = plan("Build a fraud detection model. Missing fraud is more costly than false alarms.")
    run_ = mission("data/fraud.csv", p.target_candidate, budget=Budget(max_experiments=4), plan=p)
    list(run_.run())

    assert p.primary_metric == "pr_auc"
    assert run_.objective.primary_metric == "pr_auc"

    best = run_.history.best()
    scorable = run_.history.scorable()
    assert len(scorable) >= 2
    # The winner maximises pr_auc among everything that ran.
    assert best.result.metrics["pr_auc"] == max(s for _, s in scorable)
    assert run_.summary()["best_score"] == best.result.metrics["pr_auc"]


def test_final_selection_can_differ_from_the_f1_winner() -> None:
    """Proof on real data that the objective, not f1, decides the final model."""
    p = plan("Build a fraud detection model. Missing fraud is more costly than false alarms.")
    run_ = mission("data/fraud.csv", "fraud", budget=Budget(max_experiments=4), plan=p)
    list(run_.run())

    by_objective = run_.history.best()
    by_f1 = max(run_.history.successful(), key=lambda e: e.result.metrics["f1_macro"])
    assert by_objective.result.metrics["pr_auc"] >= by_f1.result.metrics["pr_auc"]
    # If they disagree, the objective wins. If they agree, the assertion above still holds.


def test_from_plan_defaults_when_there_is_no_plan() -> None:
    assert from_plan(None) is DEFAULT_OBJECTIVE
    assert from_plan(None).primary_metric == "f1_macro"
    assert from_plan(None).direction == "maximize"


def test_planner_f1_alias_resolves_to_the_computed_metric() -> None:
    """The Planner can say 'f1'; the engine computes 'f1_macro'."""
    p = plan("Build a churn classifier and optimise f1.")
    assert p.primary_metric == "f1"
    assert from_plan(p).primary_metric == "f1_macro"


# --- backward compatibility ----------------------------------------------------------


def test_no_plan_path_is_unchanged() -> None:
    """The Milestone 1-3 programmatic entry point keeps its exact behaviour.

    Asserts the sequence ATLAS *chose* and the metric it ranked by, not the
    float scikit-learn happened to produce. The six configs below are the
    Milestone 1 escalation, verified identical on scikit-learn 1.6 and 1.9;
    the scores themselves move with scikit-learn's tree internals (1.9.0
    #31529 changed the random-forest bootstrap under sample weights, #29641
    changed HistGB bin edges), which is not ATLAS behaviour to pin.
    """
    run_ = mission("data/churn.csv", "churned", budget=Budget(max_experiments=6))
    experiments = list(run_.run())

    assert run_.objective is DEFAULT_OBJECTIVE
    assert run_.history.objective.primary_metric == "f1_macro"

    assert [(e.config.model, e.config.params) for e in experiments] == [
        ("logreg", {}),
        ("logreg", {"class_weight": "balanced"}),
        ("random_forest", {"class_weight": "balanced"}),
        ("random_forest", {"class_weight": "balanced", "max_depth": 6, "min_samples_leaf": 5}),
        ("hist_gb", {"class_weight": "balanced"}),
        ("hist_gb", {"class_weight": "balanced", "max_depth": 6}),
    ]

    # Every score is the experiment's own f1_macro -- the legacy metric, not pr_auc.
    scores = [e.result.score() for e in experiments]
    assert scores == [e.result.metrics["f1_macro"] for e in experiments]

    # Ranking is the f1_macro argmax, and the run improved on its baseline.
    summary = run_.summary()
    assert summary["best_score"] == max(scores) > scores[0]
    assert summary["best_id"] == experiments[scores.index(max(scores))].id
    assert summary["stop_reason"] == "experiment budget reached (6)"


def test_history_defaults_to_the_f1_objective() -> None:
    assert History().objective is DEFAULT_OBJECTIVE
    result = ExperimentResult(ok=True, metrics={"f1_macro": 0.5})
    assert result.primary_metric == "f1_macro"
    assert result.score() == 0.5


# --- the CLI reports, it does not decide ---------------------------------------------


def test_cli_reports_the_backends_choice_and_never_ranks() -> None:
    """The CLI must print the backend's best_id, not compute its own."""
    from run_atlas import printer

    p = plan("Build a fraud detection model. Missing fraud is more costly than false alarms.")
    events: list[dict] = []
    run_ = mission("data/fraud.csv", "fraud", budget=Budget(max_experiments=3), sink=events.append, plan=p)
    list(run_.run())

    out = io.StringIO()
    show = printer()
    with redirect_stdout(out):
        for event in events:
            show(event)
    printed = out.getvalue()

    summary = run_.summary()
    assert f"best=experiment {summary['best_id']}" in printed
    assert f"pr_auc={summary['best_score']:.4f}" in printed
    # The f1 winner's id is only absent from the 'best=' line; the objective decided.
    assert "METRIC MISMATCH" not in printed


def test_cli_module_contains_no_ranking_logic() -> None:
    """Structural guard: no comparison over experiments anywhere in the CLI."""
    import pathlib

    source = pathlib.Path("run_atlas.py").read_text(encoding="utf-8")
    for banned in ("max(", "min(", "sorted(", ".best(", "score("):
        assert banned not in source, f"run_atlas.py must not rank: found {banned!r}"
