"""Milestone 1 verification: the loop is autonomous, evidence-driven, and terminates."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from atlas import hypothesis as hypothesis_engine
from atlas import ml_engineer
from atlas.decision import Budget, decide
from atlas.diagnose import diagnose
from atlas.data_engineer import Dataset, load
from atlas.engine import run
from atlas.loop import Mission, mission
from atlas.objective import UNAVAILABLE
from atlas.state import (
    Action,
    Category,
    Decision,
    Diagnosis,
    Experiment,
    ExperimentConfig,
    ExperimentResult,
    History,
)

CSV = "data/churn.csv"


@pytest.fixture(scope="module")
def data() -> Dataset:
    return load(CSV, "churned")


def _experiment(id: int, score: float | None, config: ExperimentConfig, ok: bool = True) -> Experiment:
    result = ExperimentResult(ok=ok, metrics={"f1_macro": score} if ok else {}, error=None if ok else "boom")
    return Experiment(
        id=id,
        config=config,
        result=result,
        diagnosis=Diagnosis(Category.IMPROVEMENT, "x"),
        decision=Decision(Action.CONTINUE, "x"),
    )


# --- data loading / trust boundaries -------------------------------------------------


def test_profile_measures_real_dataset_facts(data: Dataset) -> None:
    p = data.profile
    assert p.rows == 1200
    assert p.imbalance_ratio > 1.5
    assert p.missing_per_column["monthly_charge"] == 60  # injected by make_data.py
    assert set(data.categorical) == {"plan", "contract"}


def test_load_rejects_bad_input(tmp_path) -> None:
    with pytest.raises(FileNotFoundError):
        load(tmp_path / "nope.csv", "y")
    csv = tmp_path / "d.csv"
    csv.write_text("a,b\n1,2\n")
    with pytest.raises(ValueError, match="target 'y' not in columns"):
        load(csv, "y")


# --- item 7: execution errors are captured, not crashes ------------------------------


def test_bad_params_return_structured_failure(data: Dataset) -> None:
    result = run(data, ExperimentConfig(model="logreg", params={"C": -1}))
    assert result.ok is False
    assert result.error and result.traceback
    assert result.score() is None


def test_unknown_model_is_a_failed_result_not_an_exception(data: Dataset) -> None:
    result = run(data, ExperimentConfig(model="not_a_model"))
    assert result.ok is False
    assert "unknown model" in result.error


def test_mission_survives_a_failing_experiment(data: Dataset) -> None:
    """A crash mid-loop must be diagnosed and recovered from, not kill the mission."""
    run_ = Mission(data, budget=Budget(max_experiments=3))
    broken = ExperimentConfig(model="logreg", params={"C": -1})
    original = ml_engineer.baseline
    ml_engineer.baseline = lambda: broken  # first experiment fails on purpose
    try:
        experiments = list(run_.run())
    finally:
        ml_engineer.baseline = original

    assert experiments[0].result.ok is False
    assert experiments[0].diagnosis.category is Category.EXECUTION_ERROR
    assert len(experiments) > 1, "ATLAS must recover and keep going"
    assert experiments[1].result.ok is True


# --- diagnosis distinguishes the five categories -------------------------------------


def test_diagnosis_categories() -> None:
    empty = History()
    profile = {"imbalance_ratio": 4.0}

    data_err = ExperimentResult(ok=False, error="ValueError: could not convert string to float: 'x'")
    assert diagnose(data_err, empty, profile).category is Category.DATA_ISSUE

    code_err = ExperimentResult(ok=False, error="TypeError: unexpected keyword")
    assert diagnose(code_err, empty, profile).category is Category.EXECUTION_ERROR

    weak = ExperimentResult(ok=True, metrics={"f1_macro": 0.6, "overfit_gap": 0.02, "per_class_recall": {"0": 0.9, "1": 0.2}, "labels": ["0", "1"], "predicted_classes": 2})
    first = diagnose(weak, empty, profile)
    assert first.category is Category.MODEL_ISSUE
    assert any("weak recall" in f for f in first.findings)

    history = History()
    history.add(_experiment(1, 0.60, ExperimentConfig(model="logreg")))
    clean = {"f1_macro": 0.0, "overfit_gap": 0.01, "per_class_recall": {"0": 0.8, "1": 0.8}, "labels": ["0", "1"], "predicted_classes": 2}

    better = ExperimentResult(ok=True, metrics={**clean, "f1_macro": 0.75})
    assert diagnose(better, history, profile).category is Category.IMPROVEMENT

    flat = ExperimentResult(ok=True, metrics={**clean, "f1_macro": 0.602})
    assert diagnose(flat, history, profile).category is Category.INSUFFICIENT


def test_overfitting_is_detected_from_the_gap() -> None:
    result = ExperimentResult(ok=True, metrics={"f1_macro": 0.7, "overfit_gap": 0.4, "per_class_recall": {"0": 0.9, "1": 0.8}, "labels": ["0", "1"], "predicted_classes": 2})
    assert any("overfitting" in f for f in diagnose(result, History(), {}).findings)


# --- evidence integrity: absence is reported, never defaulted ------------------------


WEAK = {
    "f1_macro": 0.6,
    "overfit_gap": 0.01,
    "per_class_recall": {"0": 0.9, "1": 0.2},
    "labels": ["0", "1"],
    "predicted_classes": 2,
}


def test_absent_profile_field_is_not_treated_as_a_measurement() -> None:
    """A missing imbalance_ratio must not read as a measured 1.00:1 (i.e. balanced)."""
    result = ExperimentResult(ok=True, metrics=dict(WEAK))

    measured = diagnose(result, History(), {"imbalance_ratio": 4.0})
    assert measured.evidence["imbalance_ratio"] == 4.0
    assert any("class imbalance 4.00:1" in f for f in measured.findings)

    absent = diagnose(result, History(), {})  # the profile carries no ratio at all
    assert absent.evidence["imbalance_ratio"] == UNAVAILABLE  # not 1.0
    assert not any("imbalance" in f for f in absent.findings)

    # ...and the hypothesis built from it claims no ratio it does not have.
    history = History()
    history.add(_experiment(1, 0.6, ExperimentConfig(model="logreg")))
    proposal = hypothesis_engine.propose(absent, history, {})
    assert proposal.config.params["class_weight"] == "balanced"  # still acts on weak recall
    assert "1.00:1" not in proposal.reasoning
    assert "not measured" in proposal.reasoning


def test_failed_experiment_reports_no_measured_evidence() -> None:
    """A crash measured nothing, so its evidence must say so rather than report zeros."""
    failed = ExperimentResult(ok=False, error="TypeError: unexpected keyword")
    evidence = diagnose(failed, History(), {}).evidence

    assert evidence["score"] == UNAVAILABLE
    assert evidence["overfit_gap"] == UNAVAILABLE  # not 0.0
    assert "weak_classes" not in evidence  # absent, not an empty measurement

    # propose() must not compare UNAVAILABLE numerically (this raised TypeError
    # before the guard) nor read it as a measured 0.0.
    history = History()
    history.add(_experiment(1, 0.6, ExperimentConfig(model="logreg")))
    stale = Diagnosis(Category.MODEL_ISSUE, "x", [], {"overfit_gap": UNAVAILABLE, "weak_classes": {}})
    proposal = hypothesis_engine.propose(stale, history, {})
    assert proposal is not None
    assert "gap" not in proposal.problem  # no overfit hypothesis from a non-measurement


def test_data_signals_need_a_word_boundary() -> None:
    """'maintenance' contains 'nan' but is not a data issue."""

    def category(error: str) -> Category:
        return diagnose(ExperimentResult(ok=False, error=error), History(), {}).category

    assert category("RuntimeError: maintenance window exceeded") is Category.EXECUTION_ERROR
    assert category("ValueError: finance column missing") is Category.EXECUTION_ERROR

    # Real data signals still classify, including suffixed forms.
    assert category("ValueError: could not convert string to float: 'x'") is Category.DATA_ISSUE
    assert category("ValueError: Input contains NaN") is Category.DATA_ISSUE
    assert category("ValueError: nans detected in column") is Category.DATA_ISSUE
    assert category("TypeError: dtypes are incompatible") is Category.DATA_ISSUE


# --- item 5: hypotheses are derived from evidence and never repeat -------------------


def test_hypothesis_targets_the_diagnosed_problem() -> None:
    history = History()
    history.add(_experiment(1, 0.58, ExperimentConfig(model="logreg")))
    diagnosis = Diagnosis(
        Category.MODEL_ISSUE,
        "weak minority",
        ["weak recall on class '1'"],
        {"weak_classes": {"1": 0.2}, "overfit_gap": 0.01, "score": 0.58},
    )
    proposal = hypothesis_engine.propose(diagnosis, history, {"imbalance_ratio": 4.0})
    assert proposal.config.params["class_weight"] == "balanced"
    assert proposal.reasoning and proposal.expected_effect


def test_hypothesis_never_repeats_a_tried_config() -> None:
    balanced = ExperimentConfig(model="logreg", params={"class_weight": "balanced"})
    history = History()
    history.add(_experiment(1, 0.58, ExperimentConfig(model="logreg")))
    history.add(_experiment(2, 0.60, balanced))
    diagnosis = Diagnosis(Category.MODEL_ISSUE, "weak", [], {"weak_classes": {"1": 0.2}, "overfit_gap": 0.01})

    proposal = hypothesis_engine.propose(diagnosis, history, {"imbalance_ratio": 4.0})
    assert not history.tried(proposal.config)


def test_hypothesis_returns_none_when_out_of_ideas() -> None:
    history = History()
    for i, model in enumerate(ml_engineer.LADDER, start=1):
        history.add(_experiment(i, 0.6, ExperimentConfig(model=model)))
    diagnosis = Diagnosis(Category.INSUFFICIENT, "flat", [], {"overfit_gap": 0.0, "weak_classes": {}})
    assert hypothesis_engine.propose(diagnosis, history, {}) is None


# --- item 6: decision logic can stop -------------------------------------------------


def test_decision_stops_on_every_termination_condition() -> None:
    import time

    now = time.perf_counter()
    budget = Budget(max_experiments=3, target_score=0.9, patience=2)
    proposal = hypothesis_engine.Hypothesis("p", "c", "r", "e", ExperimentConfig(model="hist_gb"))
    ok = Diagnosis(Category.IMPROVEMENT, "fine", [], {})

    full = History()
    for i in range(3):
        full.add(_experiment(i + 1, 0.5, ExperimentConfig(model="logreg", params={"i": i})))
    assert decide(ok, proposal, full, budget, now).action is Action.STOP

    hit = History()
    hit.add(_experiment(1, 0.95, ExperimentConfig(model="logreg")))
    assert "objective reached" in decide(ok, proposal, hit, budget, now).reason

    one = History()
    one.add(_experiment(1, 0.5, ExperimentConfig(model="logreg")))
    assert decide(ok, None, one, budget, now).action is Action.STOP  # no hypothesis

    assert decide(ok, proposal, one, budget, now - 999).action is Action.STOP  # timeout

    errors = History()
    for i in range(2):
        errors.add(_experiment(i + 1, None, ExperimentConfig(model="logreg", params={"i": i}), ok=False))
    assert "consecutive failed" in decide(ok, proposal, errors, budget, now).reason


def test_decision_stops_on_plateau() -> None:
    import time

    history = History()
    history.add(_experiment(1, 0.70, ExperimentConfig(model="logreg")))  # best, never beaten
    history.add(_experiment(2, 0.69, ExperimentConfig(model="random_forest")))
    history.add(_experiment(3, 0.68, ExperimentConfig(model="hist_gb")))
    proposal = hypothesis_engine.Hypothesis("p", "c", "r", "e", ExperimentConfig(model="logreg", params={"C": 0.1}))
    decision = decide(Diagnosis(Category.INSUFFICIENT, "flat", [], {}), proposal, history, Budget(max_experiments=9, patience=2), time.perf_counter())
    assert decision.action is Action.STOP
    assert "no improvement" in decision.reason


def test_data_issue_routes_to_revise() -> None:
    import time

    history = History()
    history.add(_experiment(1, None, ExperimentConfig(model="logreg"), ok=False))
    diagnosis = Diagnosis(Category.DATA_ISSUE, "bad values", [], {})
    proposal = hypothesis_engine.propose(diagnosis, history, {})
    assert proposal.touches_data
    assert decide(diagnosis, proposal, history, Budget(), time.perf_counter()).action is Action.REVISE


# --- items 3, 4, 5: the closed loop --------------------------------------------------


def test_loop_runs_multiple_experiments_and_each_follows_from_the_last(data: Dataset) -> None:
    run_ = Mission(data, budget=Budget(max_experiments=4))
    experiments = list(run_.run())

    assert len(experiments) >= 2, "the loop must produce more than a baseline"
    assert experiments[0].origin == "baseline"

    for previous, current in zip(experiments, experiments[1:]):
        # Item 4: the next experiment IS the previous hypothesis, not a fresh guess.
        assert previous.hypothesis is not None
        assert current.config == previous.hypothesis.config
        assert current.origin.startswith(f"exp{previous.id}:")

    fingerprints = [e.config.fingerprint() for e in experiments]
    assert len(set(fingerprints)) == len(fingerprints), "no experiment may be repeated"


def test_history_is_retained_and_best_is_selected(data: Dataset) -> None:
    run_ = Mission(data, budget=Budget(max_experiments=4))
    list(run_.run())
    summary = run_.summary()

    assert len(run_.history) == summary["experiments"] >= 2
    assert summary["timeline"][0]["origin"] == "baseline"
    scores = [e.result.score() for e in run_.history.successful()]
    assert summary["best_score"] == max(scores)
    assert run_.history.best().id == summary["best_id"]


def test_loop_always_terminates_and_reports_why(data: Dataset) -> None:
    run_ = Mission(data, budget=Budget(max_experiments=2))
    experiments = list(run_.run())
    assert len(experiments) == 2
    assert experiments[-1].decision.action is Action.STOP
    assert run_.summary()["stop_reason"]


def test_metrics_come_only_from_real_training(data: Dataset) -> None:
    """Every score in history must match a re-run of that exact config."""
    run_ = Mission(data, budget=Budget(max_experiments=2))
    list(run_.run())
    for experiment in run_.history.successful():
        assert run(data, experiment.config).metrics["f1_macro"] == pytest.approx(
            experiment.result.score()
        )


def test_events_expose_every_stage_for_a_ui(data: Dataset) -> None:
    seen: list[str] = []
    run_ = Mission(data, budget=Budget(max_experiments=2), sink=lambda e: seen.append(e["stage"]))
    list(run_.run())
    assert seen[0] == "mission_start" and seen[-1] == "mission_end"
    for stage in ("experiment_start", "experiment_result", "diagnosis", "hypothesis", "decision"):
        assert stage in seen
    assert [e["seq"] for e in run_.events] == list(range(len(run_.events)))


def test_atlas_improves_over_the_baseline(data: Dataset) -> None:
    run_ = mission(CSV, "churned", budget=Budget(max_experiments=5))
    experiments = list(run_.run())
    assert run_.history.best_score() > experiments[0].result.score()


def test_engine_handles_a_single_class_dataset(tmp_path) -> None:
    """Degenerate input must produce a structured result, not an unhandled crash."""
    csv = tmp_path / "one.csv"
    pd.DataFrame({"a": np.arange(20), "y": [1] * 20}).to_csv(csv, index=False)
    result = run(load(csv, "y"), ExperimentConfig(model="logreg"))
    assert result.ok or result.error  # either way, structured
