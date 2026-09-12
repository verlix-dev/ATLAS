"""M5 verification: the LLM proposes, the deterministic system decides.

Every test here uses a fake proposer — a plain Callable[[dict], str]. No test
reaches a network. The central claim: a hostile LLM cannot change what ATLAS
measures, ranks, or decides, and a *useful* LLM can reach a configuration the
deterministic ladder cannot express.
"""

from __future__ import annotations

import json
import pathlib

import pandas as pd
import pytest

from atlas import llm as llm_seam
from atlas import hypothesis as hypothesis_engine
from atlas.decision import Budget
from atlas.diagnose import diagnose
from atlas.llm import validate
from atlas.loop import Mission, mission
from atlas.objective import UNAVAILABLE, Objective
from atlas.planner import plan
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
TARGET = "churned"


# --- fakes ---------------------------------------------------------------------------


def fake(response: str | dict):
    """A proposer that always answers the same way. Records what it was asked."""
    payloads: list[dict] = []

    def proposer(payload: dict) -> str:
        payloads.append(payload)
        return response if isinstance(response, str) else json.dumps(response)

    proposer.payloads = payloads  # type: ignore[attr-defined]
    return proposer


def boom(exc: Exception):
    """A proposer that fails the way a real client fails."""

    def proposer(payload: dict) -> str:
        raise exc

    return proposer


def citing(**overrides):
    """A proposer that cites a real finding from the context it was handed.

    This is what a real model does — read `diagnosis.findings` and quote one —
    and it keeps these tests independent of the exact recall value, which moves
    with scikit-learn's tree internals.
    """
    payloads: list[dict] = []

    def proposer(payload: dict) -> str:
        payloads.append(payload)
        findings = payload["diagnosis"]["findings"] or list(payload["diagnosis"]["evidence"])
        return json.dumps(proposal(cites=[findings[0]], **overrides))

    proposer.payloads = payloads  # type: ignore[attr-defined]
    return proposer


def proposal(**overrides) -> dict:
    """A well-formed proposal; override one field to make it malformed."""
    body = {
        "problem": "logreg underfits the minority class",
        "proposed_change": "escalate to random_forest with balanced weights",
        "reasoning": "reweighting the linear loss moved pr_auc very little",
        "expected_effect": "a higher primary metric",
        "cites": ["weak recall on class '1': 0.201"],
        "model": "random_forest",
        "params": {"class_weight": "balanced"},
        "preprocess": {"impute": "median", "scale": False},
    }
    body.update(overrides)
    return body


def _history_with_one(model: str = "logreg") -> tuple[History, Diagnosis]:
    """One experiment run, diagnosed. The state validate() is called against."""
    metrics = {
        "f1_macro": 0.60,
        "overfit_gap": 0.01,
        "per_class_recall": {"0": 0.9, "1": 0.201},
        "labels": ["0", "1"],
        "predicted_classes": 2,
    }
    result = ExperimentResult(ok=True, metrics=metrics)
    history = History()
    diagnosis = diagnose(result, History(), {"imbalance_ratio": 3.0})
    history.add(
        Experiment(
            id=1,
            config=ExperimentConfig(model=model),
            result=result,
            diagnosis=diagnosis,
            decision=Decision(Action.CONTINUE, "x"),
        )
    )
    return history, diagnosis


def _validate(body, history=None, diagnosis=None, current=None):
    """Run the real validator against a realistic diagnosis/history."""
    if history is None or diagnosis is None:
        history, diagnosis = _history_with_one()
    raw = body if isinstance(body, str) else json.dumps(body)
    return validate(raw, diagnosis, history, current or ExperimentConfig(model="logreg"))


# --- the deterministic baseline every hostile run is compared against ----------------


@pytest.fixture(scope="module")
def baseline() -> dict:
    run_ = mission(CSV, TARGET, budget=Budget(max_experiments=4))
    experiments = list(run_.run())
    return {
        "metrics": [e.result.metrics for e in experiments],
        "configs": [(e.config.model, e.config.params) for e in experiments],
        "summary": run_.summary(),
    }


def _assert_matches_baseline(run_, experiments, baseline: dict) -> None:
    """A rejected LLM must leave the mission indistinguishable from llm=None."""
    assert [e.result.metrics for e in experiments] == baseline["metrics"]
    assert [(e.config.model, e.config.params) for e in experiments] == baseline["configs"]
    assert run_.summary()["best_id"] == baseline["summary"]["best_id"]
    assert run_.summary()["best_score"] == baseline["summary"]["best_score"]
    assert run_.summary()["stop_reason"] == baseline["summary"]["stop_reason"]
    assert all(e.hypothesis.source == "deterministic" for e in experiments if e.hypothesis)


# --- 1. a valid proposal is accepted -------------------------------------------------


def test_valid_proposal_is_accepted_and_stamped() -> None:
    hypothesis, reason = _validate(proposal())
    assert reason == "accepted"
    assert hypothesis.source == "llm"  # stamped by the validator
    assert hypothesis.cites == ["weak recall on class '1': 0.201"]
    assert hypothesis.config.model == "random_forest"
    assert hypothesis.config.params == {"class_weight": "balanced"}
    # preprocess is canonicalised so it fingerprints like a deterministic config
    assert hypothesis.config.preprocess == {"impute": "median", "scale": False}
    assert hypothesis.touches_data is True  # derived: scale changed


def test_partial_preprocess_is_canonicalised_for_duplicate_detection() -> None:
    """{"scale": false} and the full dict behave the same, so they must fingerprint the same."""
    partial, _ = _validate(proposal(preprocess={"scale": False}))
    full, _ = _validate(proposal(preprocess={"impute": "median", "scale": False}))
    assert partial.config.fingerprint() == full.config.fingerprint()


def test_source_cannot_be_self_reported() -> None:
    for field in ("source", "id", "touches_data", "objective", "primary_metric", "target"):
        _, reason = _validate(proposal(**{field: "anything"}))
        assert reason.startswith("schema: unknown field"), f"{field} was not rejected"


# --- 2. gate 1: JSON -----------------------------------------------------------------


def test_invalid_json_is_rejected() -> None:
    for raw in ("not json at all", "", "{", "[1,2,3]", "null", '"a string"'):
        hypothesis, reason = _validate(raw)
        assert hypothesis is None
        assert reason.startswith("invalid_json"), f"{raw!r} -> {reason}"


def test_markdown_wrapped_json_is_rejected() -> None:
    """Tolerating fences would hide a prompt that needs fixing."""
    body = json.dumps(proposal())
    for raw in (f"```json\n{body}\n```", f"Sure! Here you go:\n{body}", f"{body}\nHope that helps!"):
        hypothesis, reason = _validate(raw)
        assert hypothesis is None
        assert reason.startswith("invalid_json")


def test_non_string_response_is_rejected() -> None:
    hypothesis, reason = validate(proposal(), *_history_with_one()[::-1], ExperimentConfig(model="logreg"))
    assert hypothesis is None
    assert reason.startswith("invalid_json: expected a string")


# --- 3. gate 2: schema ---------------------------------------------------------------


def test_fake_metrics_are_rejected() -> None:
    for smuggled in ({"metrics": {"pr_auc": 0.99}}, {"score": 0.99}, {"ok": True}):
        _, reason = _validate(proposal(**smuggled))
        assert reason.startswith("schema: unknown field"), f"{smuggled} -> {reason}"


def test_fake_stop_or_decision_is_rejected() -> None:
    for smuggled in ({"action": "stop"}, {"stop": True}, {"decision": "stop"}):
        _, reason = _validate(proposal(**smuggled))
        assert reason.startswith("schema: unknown field"), f"{smuggled} -> {reason}"


def test_missing_and_malformed_fields_are_rejected() -> None:
    body = proposal()
    del body["reasoning"]
    _, reason = _validate(body)
    assert reason == "schema: missing field(s) ['reasoning']"

    _, reason = _validate(proposal(problem=""))
    assert reason == "schema: problem must be a non-empty string"
    _, reason = _validate(proposal(problem=123))
    assert reason == "schema: problem must be a non-empty string"
    _, reason = _validate(proposal(reasoning="x" * 501))
    assert "exceeds 500 characters" in reason
    _, reason = _validate(proposal(params=["not", "a", "dict"]))
    assert reason == "schema: params must be an object"
    _, reason = _validate(proposal(preprocess="median"))
    assert reason == "schema: preprocess must be an object"


# --- 4. gates 3-6: model, params, preprocessing --------------------------------------


def test_unknown_model_is_rejected() -> None:
    for name in ("xgboost", "lightgbm", "", None, 42, "LogReg"):
        _, reason = _validate(proposal(model=name))
        assert reason.startswith("unknown_model"), f"{name!r} -> {reason}"


def test_unknown_parameter_is_rejected() -> None:
    _, reason = _validate(proposal(params={"booster": "gbtree"}))
    assert reason.startswith("unknown_params")
    assert "random_forest" in reason


def test_parameter_names_are_checked_against_the_named_model() -> None:
    """`C` is a logreg parameter; random_forest must not accept it."""
    _, reason = _validate(proposal(model="random_forest", params={"C": 0.1}))
    assert reason.startswith("unknown_params")
    hypothesis, accepted = _validate(
        proposal(model="logreg", params={"C": 0.1}, preprocess={"impute": "median", "scale": True})
    )
    assert accepted == "accepted" and hypothesis.config.params == {"C": 0.1}


def test_non_scalar_parameter_values_are_rejected() -> None:
    for value in ([1, 2], {"a": 1}):
        _, reason = _validate(proposal(params={"max_depth": value}))
        assert reason.startswith("param_type"), f"{value!r} -> {reason}"


def test_invalid_preprocessing_is_rejected() -> None:
    _, reason = _validate(proposal(preprocess={"drop_outliers": True}))
    assert reason.startswith("preprocess: unknown key(s)")
    _, reason = _validate(proposal(preprocess={"impute": "interpolate"}))
    assert reason.startswith("preprocess: impute")
    _, reason = _validate(proposal(preprocess={"scale": "yes"}))
    assert reason.startswith("preprocess: scale must be a boolean")


# --- 5. gate 7: duplicate prevention -------------------------------------------------


def test_duplicate_proposal_is_rejected() -> None:
    history, diagnosis = _history_with_one()
    already = history.experiments[0].config
    body = proposal(
        model=already.model, params=dict(already.params), preprocess=dict(already.preprocess)
    )
    hypothesis, reason = _validate(body, history, diagnosis)
    assert hypothesis is None
    assert reason == "duplicate: already run as experiment 1"


def test_duplicate_prevention_uses_the_same_path_as_the_ladder() -> None:
    """Not a second implementation: the validator calls history.tried()."""
    history, diagnosis = _history_with_one()
    accepted, _ = _validate(proposal(), history, diagnosis)
    assert not history.tried(accepted.config)
    history.add(
        Experiment(
            id=2,
            config=accepted.config,
            result=ExperimentResult(ok=True, metrics={"f1_macro": 0.7}),
            diagnosis=diagnosis,
        )
    )
    _, reason = _validate(proposal(), history, diagnosis)
    assert reason == "duplicate: already run as experiment 2"


# --- 6. gate 8: evidence citation ----------------------------------------------------


def test_missing_or_invented_citations_are_rejected() -> None:
    _, reason = _validate(proposal(cites=[]))
    assert reason == "schema: cites must be a non-empty list of strings"
    _, reason = _validate(proposal(cites="a string"))
    assert reason == "schema: cites must be a non-empty list of strings"
    _, reason = _validate(proposal(cites=[1, 2]))
    assert reason == "schema: cites must be a non-empty list of strings"

    _, reason = _validate(proposal(cites=["the data is noisy"]))
    assert reason.startswith("uncited: 'the data is noisy'")


def test_citing_only_an_evidence_key_is_rejected_when_findings_exist() -> None:
    """Evidence keys are real, but a diagnosis with findings must be engaged with."""
    _, reason = _validate(proposal(cites=["overfit_gap"]))
    assert reason == "uncited: the diagnosis reported findings, but none are cited"


def test_evidence_keys_are_citable_when_there_are_no_findings() -> None:
    clean = Diagnosis(Category.INSUFFICIENT, "flat", [], {"score": 0.6, "gain": 0.0})
    history, _ = _history_with_one()
    hypothesis, reason = _validate(proposal(cites=["gain"]), history, clean)
    assert reason == "accepted" and hypothesis.cites == ["gain"]


# --- 7. transport failure: everything degrades to deterministic ----------------------


def test_llm_exception_falls_back(baseline) -> None:
    for exc in (
        TimeoutError("deadline exceeded"),
        ConnectionError("dns failure"),
        RuntimeError("429 rate_limit_error"),
        PermissionError("401 authentication_error"),
        ValueError("unexpected"),
    ):
        events: list[dict] = []
        run_ = mission(
            CSV, TARGET, budget=Budget(max_experiments=4), sink=events.append, llm=boom(exc)
        )
        experiments = list(run_.run())
        _assert_matches_baseline(run_, experiments, baseline)
        rejected = [r for e in events if e["stage"] == "hypothesis" for r in e["rejected"]]
        assert rejected, f"{exc!r} produced no recorded rejection"
        assert all(r["reason"].startswith("llm_unavailable") for r in rejected)
        assert type(exc).__name__ in rejected[0]["reason"]


def test_llm_returning_none_falls_back(baseline) -> None:
    events: list[dict] = []
    run_ = mission(
        CSV, TARGET, budget=Budget(max_experiments=4), sink=events.append, llm=lambda p: None
    )
    experiments = list(run_.run())
    _assert_matches_baseline(run_, experiments, baseline)


# --- 8. THE hostile LLM --------------------------------------------------------------


HOSTILE = [
    ("prose", "I think you should try a neural network!", "invalid_json"),
    ("fenced", f"```json\n{json.dumps(proposal())}\n```", "invalid_json"),
    ("empty", "", "invalid_json"),
    ("fake_metrics", json.dumps(proposal(metrics={"pr_auc": 0.99})), "schema"),
    ("fake_score", json.dumps(proposal(score=0.99, ok=True)), "schema"),
    ("forced_stop", json.dumps(proposal(action="stop")), "schema"),
    ("self_stamped", json.dumps(proposal(source="deterministic")), "schema"),
    ("override_objective", json.dumps(proposal(primary_metric="accuracy")), "schema"),
    ("override_target", json.dumps(proposal(target="monthly_charge")), "schema"),
    ("route_revise", json.dumps(proposal(touches_data=True)), "schema"),
    ("no_cites", json.dumps(proposal(cites=[])), "schema"),
    ("unknown_model", json.dumps(proposal(model="xgboost")), "unknown_model"),
    ("unknown_param", json.dumps(proposal(params={"booster": "gbtree"})), "unknown_params"),
    ("bad_preprocess", json.dumps(proposal(preprocess={"drop_rows": True})), "preprocess"),
    ("invented_cites", json.dumps(proposal(cites=["the model is bad"])), "uncited"),
]


GATES = (
    "invalid_json", "schema", "unknown_model", "unknown_params",
    "param_type", "preprocess", "duplicate", "uncited", "llm_unavailable",
)


@pytest.mark.parametrize("name,response,expected", HOSTILE, ids=[h[0] for h in HOSTILE])
def test_hostile_llm_cannot_change_the_mission(name, response, expected, baseline) -> None:
    """Every forbidden move is rejected at its own gate, and the mission is
    indistinguishable from llm=None."""
    events: list[dict] = []
    run_ = mission(
        CSV, TARGET, budget=Budget(max_experiments=4), sink=events.append, llm=fake(response)
    )
    experiments = list(run_.run())

    _assert_matches_baseline(run_, experiments, baseline)
    rejected = [r for e in events if e["stage"] == "hypothesis" for r in e["rejected"]]
    assert rejected, f"{name}: nothing was rejected"
    assert all(r["source"] == "llm" for r in rejected)

    # The first rejection is at the intended gate -- rejected for the right
    # reason, not merely rejected.
    assert rejected[0]["reason"].startswith(expected), (
        f"{name}: expected {expected!r}, got {rejected[0]['reason']!r}"
    )
    # Later rejections may fire at an earlier gate as history accumulates (a
    # config the ladder has since run is caught by `duplicate` before `uncited`),
    # but every one must be a named gate, never an unhandled error.
    assert all(r["reason"].startswith(GATES) for r in rejected), [r["reason"] for r in rejected]


def test_hostile_llm_cannot_alter_the_objective() -> None:
    p = plan("Build a fraud detection model. Missing fraud is more costly than false alarms.")
    run_ = mission(
        "data/fraud.csv",
        "fraud",
        budget=Budget(max_experiments=3),
        plan=p,
        llm=fake(proposal(objective="accuracy", primary_metric="accuracy")),
    )
    list(run_.run())
    assert run_.objective.primary_metric == "pr_auc"
    assert run_.history.objective is run_.objective
    assert run_.summary()["objective"]["primary_metric"] == "pr_auc"
    best = run_.history.best()
    assert run_.summary()["best_score"] == best.result.metrics["pr_auc"]


def test_hostile_llm_cannot_alter_target_validity(tmp_path) -> None:
    """The Data Engineer blocks first; no proposer is ever consulted."""
    csv = tmp_path / "ambiguous.csv"
    pd.DataFrame({"a": range(30), "b": range(30, 60), "ident": range(60, 90)}).to_csv(
        csv, index=False
    )
    proposer = fake(proposal(target="ident"))
    events: list[dict] = []
    run_ = mission(csv, sink=events.append, llm=proposer)

    assert list(run_.run()) == []
    assert [e["stage"] for e in events] == ["data_engineer", "mission_end"]
    assert proposer.payloads == []  # never even asked
    assert run_.summary()["best_id"] is None


def test_hostile_llm_cannot_bypass_budget_or_patience() -> None:
    """An endlessly inventive LLM still stops when the deterministic budget says so."""
    counter = {"n": 0}

    def endless(payload: dict) -> str:
        counter["n"] += 1
        # Always novel, always valid, always well-cited.
        finding = payload["diagnosis"]["findings"] or list(payload["diagnosis"]["evidence"])
        return json.dumps(
            proposal(
                model="hist_gb",
                params={"max_iter": 50 + counter["n"]},
                preprocess={"impute": "median", "scale": True},
                cites=[finding[0]],
            )
        )

    run_ = mission(CSV, TARGET, budget=Budget(max_experiments=3, patience=99), llm=endless)
    experiments = list(run_.run())
    assert len(experiments) == 3
    assert experiments[-1].decision.action is Action.STOP
    assert run_.summary()["stop_reason"] == "experiment budget reached (3)"

    run_ = mission(CSV, TARGET, budget=Budget(max_experiments=99, patience=2), llm=endless)
    experiments = list(run_.run())
    assert len(experiments) < 99
    assert "no improvement" in run_.summary()["stop_reason"]


def test_llm_cannot_force_a_stop() -> None:
    """Withholding a proposal is the most an LLM can do, and the ladder covers it."""
    run_ = mission(CSV, TARGET, budget=Budget(max_experiments=3), llm=fake('{"action":"stop"}'))
    experiments = list(run_.run())
    assert len(experiments) == 3  # the ladder kept going
    assert run_.summary()["stop_reason"] == "experiment budget reached (3)"


def test_llm_prose_numbers_never_become_metrics() -> None:
    """An accepted proposal's claims are recorded as claims, never as measurements."""
    run_ = mission(
        CSV,
        TARGET,
        budget=Budget(max_experiments=2),
        llm=citing(
            expected_effect="pr_auc will be 0.99",
            reasoning="trust me, this reaches 0.99 accuracy",
        ),
    )
    experiments = list(run_.run())

    llm_hypotheses = [e.hypothesis for e in experiments if e.hypothesis and e.hypothesis.source == "llm"]
    assert llm_hypotheses, "the proposal should have been accepted"
    assert "0.99" in llm_hypotheses[0].expected_effect  # the claim is preserved verbatim

    for e in experiments:
        assert 0.99 not in e.result.metrics.values()
        assert e.result.score() == e.result.metrics[run_.objective.primary_metric]
    scorable = run_.history.scorable()
    assert run_.summary()["best_score"] == max(s for _, s in scorable)


# --- 9. the context is data, and only measured data ---------------------------------


def _leaves(obj, path="") -> list[str]:
    if isinstance(obj, dict):
        return [p for k, v in obj.items() for p in _leaves(v, f"{path}.{k}")]
    if isinstance(obj, list):
        return [p for v in obj for p in _leaves(v, path)]
    return [path]


def test_llm_receives_only_serialisable_non_executable_context() -> None:
    proposer = fake("not json")
    run_ = mission(CSV, TARGET, budget=Budget(max_experiments=3), llm=proposer)
    list(run_.run())

    assert proposer.payloads, "the proposer was never called"
    for payload in proposer.payloads:
        assert isinstance(payload, dict)
        assert json.loads(json.dumps(payload)) == payload  # pure data, round-trips
        assert set(payload) == {
            "objective", "dataset", "diagnosis", "current", "history", "allowed", "budget",
        }


def test_raw_dataset_and_paths_never_reach_the_llm() -> None:
    proposer = fake("not json")
    run_ = mission(CSV, TARGET, budget=Budget(max_experiments=2), llm=proposer)
    list(run_.run())

    blob = json.dumps(proposer.payloads)
    for leaked in ("X_train", "X_test", "y_train", "y_test", "churn.csv", ".csv", "data/",
                   "target_reason", "target_confidence", "DataFrame", "sklearn"):
        assert leaked not in blob, f"{leaked!r} leaked into the context"

    for payload in proposer.payloads:
        # Column names are present (needed to reason); values are not.
        assert "tenure_months" in payload["dataset"]["numeric"]
        assert "target" not in payload["dataset"]
        # No handle of any kind survives serialisation, so check the shape too.
        assert all(isinstance(row["id"], int) for row in payload["history"])


def test_context_keeps_unavailable_metrics_unavailable(tmp_path) -> None:
    """The prompt must not be the one place a missing measurement becomes a number."""
    csv = tmp_path / "multi.csv"
    pd.DataFrame({"x": list(range(60)), "grade": ["a", "b", "c"] * 20}).to_csv(csv, index=False)
    from atlas.planner import MissionPlan

    p = MissionPlan(
        objective="x", problem_type="classification", target_candidate="grade",
        primary_metric="pr_auc", metric_confidence="inferred", metric_reason="test",
    )
    proposer = fake("not json")
    run_ = mission(
        csv, "grade", budget=Budget(max_experiments=3, patience=2), plan=p, llm=proposer
    )
    list(run_.run())

    scores = [row["score"] for payload in proposer.payloads for row in payload["history"]]
    assert scores, "no history reached the context"
    assert all(s == UNAVAILABLE for s in scores)
    assert 0.0 not in scores


def test_context_budget_counters_are_read_only_facts() -> None:
    proposer = fake("not json")
    run_ = mission(CSV, TARGET, budget=Budget(max_experiments=4, patience=3), llm=proposer)
    list(run_.run())
    for i, payload in enumerate(proposer.payloads, start=1):
        assert payload["budget"] == {
            "experiments_run": i,
            "max_experiments": 4,
            "rounds_since_improvement": payload["budget"]["rounds_since_improvement"],
            "patience": 3,
        }


# --- 10. structural: the module cannot execute anything ------------------------------


def test_llm_module_cannot_reach_execution() -> None:
    source = pathlib.Path("atlas/llm.py").read_text(encoding="utf-8")
    for banned in (
        "import sklearn", "from sklearn", "import pandas", "from pandas",
        "atlas.engine", "atlas.decision", "atlas.data_engineer", "atlas.loop",
        ".fit(", ".predict(", "read_csv", "train_test_split", "build_estimator",
        "open(", "subprocess", "eval(", "exec(", "__import__", "pickle",
    ):
        assert banned not in source, f"atlas/llm.py must not reach execution: {banned!r}"


def test_llm_module_never_constructs_an_estimator() -> None:
    """allowed_params reads a signature; it must not instantiate anything."""
    from atlas.ml_engineer import MODELS, allowed_params

    for name in MODELS:
        names = allowed_params(name)
        assert isinstance(names, set) and names
    assert "get_params" not in pathlib.Path("atlas/ml_engineer.py").read_text(encoding="utf-8")
    assert "class_weight" in allowed_params("random_forest")
    assert "C" in allowed_params("logreg")
    assert "C" not in allowed_params("hist_gb")


def test_validator_is_pure() -> None:
    """Same inputs, same answer, and history is not mutated by a rejection."""
    history, diagnosis = _history_with_one()
    before = len(history)
    first = _validate(proposal(model="xgboost"), history, diagnosis)
    second = _validate(proposal(model="xgboost"), history, diagnosis)
    assert first[1] == second[1]
    assert len(history) == before


# --- 11. provenance in history and the event stream ---------------------------------


def test_accepted_llm_hypothesis_is_recorded_in_history() -> None:
    events: list[dict] = []
    run_ = mission(
        CSV, TARGET, budget=Budget(max_experiments=2), sink=events.append, llm=citing()
    )
    experiments = list(run_.run())

    first = experiments[0]
    cited = first.diagnosis.findings[0]  # what the proposer actually quoted
    assert first.hypothesis.source == "llm"
    assert first.hypothesis.cites == [cited]
    assert cited in first.diagnosis.findings  # the citation is real, not invented
    # Provenance is visible on the experiment it produced, and in the timeline.
    assert experiments[1].origin.startswith("exp1: [llm]")
    assert run_.summary()["timeline"][1]["origin"].startswith("exp1: [llm]")

    hypothesis_event = next(e for e in events if e["stage"] == "hypothesis")
    assert hypothesis_event["source"] == "llm"
    assert hypothesis_event["cites"] == [cited]
    assert hypothesis_event["rejected"] == []


def test_rejection_is_recorded_and_fallback_is_labelled() -> None:
    events: list[dict] = []
    run_ = mission(
        CSV, TARGET, budget=Budget(max_experiments=2), sink=events.append, llm=fake("nonsense")
    )
    experiments = list(run_.run())

    hypothesis_event = next(e for e in events if e["stage"] == "hypothesis")
    assert hypothesis_event["source"] == "deterministic"
    assert len(hypothesis_event["rejected"]) == 1
    assert hypothesis_event["rejected"][0]["source"] == "llm"
    assert hypothesis_event["rejected"][0]["reason"].startswith("invalid_json")
    assert experiments[1].origin.startswith("exp1: [deterministic]")


def test_deterministic_hypotheses_carry_no_citations() -> None:
    run_ = mission(CSV, TARGET, budget=Budget(max_experiments=3))
    experiments = list(run_.run())
    for e in experiments:
        if e.hypothesis:
            assert e.hypothesis.source == "deterministic"
            assert e.hypothesis.cites == []


# --- 12. THE POINT: a novel configuration the ladder cannot reach -------------------


def test_llm_reaches_a_configuration_the_ladder_cannot_express() -> None:
    """M5's actual value: ATLAS really runs a config the deterministic engine
    would never propose, and the measured metrics come from that config."""
    novel = {
        "model": "hist_gb",
        "params": {"learning_rate": 0.03, "max_iter": 300, "min_samples_leaf": 40},
        "preprocess": {"impute": "most_frequent", "scale": True},
    }

    def deterministic_configs() -> set[str]:
        run_ = mission(CSV, TARGET, budget=Budget(max_experiments=6))
        return {e.config.fingerprint() for e in run_.run()}

    target_config = ExperimentConfig(
        model=novel["model"], params=novel["params"], preprocess=novel["preprocess"]
    )
    assert target_config.fingerprint() not in deterministic_configs(), (
        "the ladder can already reach this config; the test proves nothing"
    )

    once = {"done": False}

    def proposer(payload: dict) -> str:
        if once["done"]:
            return "no further ideas"
        once["done"] = True
        finding = payload["diagnosis"]["findings"] or list(payload["diagnosis"]["evidence"])
        return json.dumps(proposal(cites=[finding[0]], **novel))

    run_ = mission(CSV, TARGET, budget=Budget(max_experiments=3), llm=proposer)
    experiments = list(run_.run())

    ran = experiments[1]
    assert ran.config.model == "hist_gb"
    assert ran.config.params == novel["params"]
    assert ran.config.preprocess == novel["preprocess"]
    assert ran.origin.startswith("exp1: [llm]")
    # And it was really trained: the metrics are measured, not asserted.
    assert ran.result.ok
    assert isinstance(ran.result.metrics["f1_macro"], float)
    assert ran.result.metrics["test_rows"] == 300
    assert ran.result.score() == ran.result.metrics["f1_macro"]


def test_llm_novel_config_still_competes_on_the_objective_only() -> None:
    """A novel LLM config wins only if it actually measures better."""
    def proposer(payload: dict) -> str:
        finding = payload["diagnosis"]["findings"] or list(payload["diagnosis"]["evidence"])
        return json.dumps(
            proposal(
                model="logreg",
                params={"C": 0.001},  # deliberately weak
                preprocess={"impute": "median", "scale": True},
                cites=[finding[0]],
                expected_effect="this will be the best model by far",
            )
        )

    run_ = mission(CSV, TARGET, budget=Budget(max_experiments=3), llm=proposer)
    list(run_.run())

    best = run_.history.best()
    scorable = run_.history.scorable()
    assert run_.summary()["best_score"] == max(s for _, s in scorable)
    assert best.result.score() == max(s for _, s in scorable)


# --- 13. backward compatibility ------------------------------------------------------


def test_llm_none_is_the_m4_5_path() -> None:
    run_ = mission("data/churn.csv", "churned", budget=Budget(max_experiments=6))
    experiments = list(run_.run())
    assert [(e.config.model, e.config.params) for e in experiments] == [
        ("logreg", {}),
        ("logreg", {"class_weight": "balanced"}),
        ("random_forest", {"class_weight": "balanced"}),
        ("random_forest", {"class_weight": "balanced", "max_depth": 6, "min_samples_leaf": 5}),
        ("hist_gb", {"class_weight": "balanced"}),
        ("hist_gb", {"class_weight": "balanced", "max_depth": 6}),
    ]
    assert all(e.hypothesis.source == "deterministic" for e in experiments if e.hypothesis)


def test_propose_without_objective_or_budget_fails_loudly() -> None:
    """Silently degrading to deterministic would hide a wiring bug."""
    history, diagnosis = _history_with_one()
    with pytest.raises(ValueError, match="needs `objective` and `budget`"):
        hypothesis_engine.propose(diagnosis, history, {}, llm=fake(proposal()))


def test_mission_accepts_a_proposer_without_a_plan() -> None:
    run_ = Mission(mission(CSV, TARGET).data, budget=Budget(max_experiments=2), llm=citing())
    experiments = list(run_.run())
    assert run_.objective.primary_metric == "f1_macro"  # default objective intact
    assert experiments[0].hypothesis.source == "llm"
