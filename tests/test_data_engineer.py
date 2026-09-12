"""Data Engineer verification: CSV in, correct profile and split Dataset out."""

from __future__ import annotations

import pandas as pd
import pytest

from atlas.data_engineer import (
    TEST_SIZE,
    DataProfile,
    Dataset,
    TargetConfidence,
    detect_target,
    load,
)
from atlas.decision import Budget
from atlas.engine import run
from atlas.loop import Mission, mission
from atlas.state import Action, ExperimentConfig

# A hand-built fixture with every property the milestone must handle:
# numeric + categorical features, missing values, class imbalance, and one
# exact duplicate (row 10 repeats row 0) so duplicate detection has something
# real to find. 12 rows: 9 x "no", 3 x "yes" -> exactly 3.0:1.
FIXTURE = pd.DataFrame(
    {
        "age": [25, 31, 47, 52, 33, None, 41, 29, 38, 45, 25, 60],
        "score": [0.5, 0.9, 0.2, 0.7, 0.4, 0.6, None, 0.8, 0.3, 0.55, 0.5, 0.15],
        "region": ["north", "south", "north", None, "south", "north", "south", "north", "north", "south", "north", "south"],
        "tier": ["a", "b", "a", "b", "a", "a", "b", "a", "b", "a", "a", "b"],
        "churned": ["no", "yes", "no", "no", "no", "yes", "no", "no", "no", "no", "no", "yes"],
    }
)


@pytest.fixture
def fixture_csv(tmp_path):
    path = tmp_path / "fixture.csv"
    FIXTURE.to_csv(path, index=False)
    return path


# --- ingestion and trust boundaries --------------------------------------------------


def test_load_rejects_unreadable_and_malformed_input(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="dataset not found"):
        load(tmp_path / "missing.csv", "y")

    headers_only = tmp_path / "empty.csv"
    headers_only.write_text("a,b,y\n")
    with pytest.raises(ValueError, match="dataset is empty"):
        load(headers_only, "y")

    no_target = tmp_path / "no_target.csv"
    no_target.write_text("a,b\n1,2\n3,4\n")
    with pytest.raises(ValueError, match=r"target 'y' not in columns"):
        load(no_target, "y")

    blank_target = tmp_path / "blank.csv"
    blank_target.write_text("a,y\n1,\n2,\n")
    with pytest.raises(ValueError, match="no rows remain"):
        load(blank_target, "y")


def test_ragged_csv_is_a_pandas_error_not_a_silent_truncation(tmp_path) -> None:
    ragged = tmp_path / "ragged.csv"
    ragged.write_text("a,b,y\n1,2,no\n3,4,5,6,yes\n")
    with pytest.raises(pd.errors.ParserError):
        load(ragged, "y")


# --- profile correctness against the known fixture -----------------------------------


def test_profile_matches_the_fixture_exactly(fixture_csv) -> None:
    p = load(fixture_csv, "churned").profile

    assert isinstance(p, DataProfile)
    assert p.rows == 12
    assert p.columns == 4  # 5 columns minus the target
    assert p.target == "churned"
    assert p.problem_type == "classification"
    assert p.target_confidence is TargetConfidence.EXPLICIT


def test_numeric_and_categorical_detection(fixture_csv) -> None:
    p = load(fixture_csv, "churned").profile
    assert p.numeric == ["age", "score"]
    assert p.categorical == ["region", "tier"]
    assert p.dtypes["age"] == "float64"  # None forces float
    # The profile reports the dtype pandas actually gave us: "object" before
    # pandas 3, "str" once infer_string became the default. Either proves the
    # column was recorded as text rather than silently coerced to a number.
    assert p.dtypes["tier"] in ("object", "str")


def test_missing_values_and_duplicates(fixture_csv) -> None:
    p = load(fixture_csv, "churned").profile
    assert p.missing_per_column == {"age": 1, "score": 1, "region": 1}
    assert p.missing_fraction == pytest.approx(3 / (12 * 4))
    assert p.duplicate_rows == 1  # row 10 repeats row 0
    assert p.constant_columns == []


def test_duplicate_rows_are_counted(tmp_path) -> None:
    doubled = pd.concat([FIXTURE, FIXTURE.iloc[[0, 1]]], ignore_index=True)
    path = tmp_path / "dupes.csv"
    doubled.to_csv(path, index=False)
    p = load(path, "churned").profile
    assert p.duplicate_rows == 3  # the fixture's own duplicate, plus the two added
    assert p.rows == 14  # counted and reported, not silently dropped


def test_constant_column_is_flagged(tmp_path) -> None:
    frame = FIXTURE.assign(country="uk")
    path = tmp_path / "constant.csv"
    frame.to_csv(path, index=False)
    assert load(path, "churned").profile.constant_columns == ["country"]


def test_class_distribution_is_exact(fixture_csv) -> None:
    p = load(fixture_csv, "churned").profile
    assert p.class_counts == {"no": 9, "yes": 3}
    assert p.minority_class == "yes"
    assert p.imbalance_ratio == pytest.approx(3.0)


def test_profile_matches_the_real_churn_csv() -> None:
    p = load("data/churn.csv", "churned").profile
    assert p.rows == 1200
    assert p.columns == 5
    assert p.numeric == ["tenure_months", "monthly_charge", "support_calls"]
    assert p.categorical == ["plan", "contract"]
    assert p.missing_per_column == {"monthly_charge": 60}
    assert sum(p.class_counts.values()) == 1200
    assert p.imbalance_ratio == pytest.approx(
        p.class_counts["0"] / p.class_counts["1"]
    )


# --- target identification -----------------------------------------------------------


def test_explicit_target_wins(fixture_csv) -> None:
    p = load(fixture_csv, "tier").profile
    assert p.target == "tier"
    assert p.target_confidence is TargetConfidence.EXPLICIT
    assert "churned" in p.categorical  # the unused column becomes a feature


def test_conventional_name_is_inferred() -> None:
    frame = pd.DataFrame({"a": [1, 2, 3, 4], "label": ["x", "y", "x", "y"], "b": [5, 6, 7, 8]})
    target, confidence, reason = detect_target(frame)
    assert target == "label"
    assert confidence is TargetConfidence.INFERRED
    assert "conventional target name" in reason


def test_last_low_cardinality_column_is_inferred(fixture_csv) -> None:
    p = load(fixture_csv).profile  # no target supplied
    assert p.target == "churned"
    assert p.target_confidence is TargetConfidence.INFERRED
    assert "2 distinct values" in p.target_reason


def test_ambiguous_target_needs_clarification() -> None:
    """A column with one row per value is an identifier, not a target."""
    frame = pd.DataFrame({"a": [1, 2, 3, 4], "note": ["p", "q", "r", "s"]})
    target, confidence, reason = detect_target(frame)
    assert confidence is TargetConfidence.NEEDS_CLARIFICATION
    assert "pass the target explicitly" in reason


def test_high_cardinality_last_column_needs_clarification() -> None:
    frame = pd.DataFrame({"a": range(100), "code": [f"c{i}" for i in range(100)]})
    _, confidence, _ = detect_target(frame)
    assert confidence is TargetConfidence.NEEDS_CLARIFICATION


def test_classes_must_repeat_to_be_inferred() -> None:
    """Same 2 distinct values, but enough rows for them to repeat -> inferred."""
    frame = pd.DataFrame({"a": range(10), "flag": ["x", "y"] * 5})
    target, confidence, _ = detect_target(frame)
    assert target == "flag"
    assert confidence is TargetConfidence.INFERRED


def test_competing_conventional_names_need_clarification() -> None:
    frame = pd.DataFrame({"target": [0, 1, 0], "label": ["a", "b", "a"], "x": [1, 2, 3]})
    _, confidence, reason = detect_target(frame)
    assert confidence is TargetConfidence.NEEDS_CLARIFICATION
    assert "will not choose between them" in reason


def test_continuous_target_is_blocked_as_out_of_scope(tmp_path) -> None:
    frame = pd.DataFrame({"x": range(60), "price": [i * 1.5 for i in range(60)]})
    path = tmp_path / "reg.csv"
    frame.to_csv(path, index=False)
    p = load(path, "price").profile
    assert p.problem_type == "regression"
    assert p.target_confidence is TargetConfidence.NEEDS_CLARIFICATION
    assert "classification only" in p.target_reason
    assert not p.usable()


def test_single_class_target_is_blocked(tmp_path) -> None:
    frame = pd.DataFrame({"x": range(10), "y": [1] * 10})
    path = tmp_path / "single.csv"
    frame.to_csv(path, index=False)
    p = load(path, "y").profile
    assert p.target_confidence is TargetConfidence.NEEDS_CLARIFICATION
    assert "nothing to classify" in p.target_reason


# --- preprocessing output shape ------------------------------------------------------


def test_split_shape_and_stratification(fixture_csv) -> None:
    data = load(fixture_csv, "churned")
    assert isinstance(data, Dataset)
    assert len(data.X_test) == 3  # 12 * 0.25
    assert len(data.X_train) == 9
    assert len(data.y_train) == 9 and len(data.y_test) == 3
    assert list(data.X_train.columns) == ["age", "score", "region", "tier"]
    assert "churned" not in data.X_train.columns
    assert data.classes == ["no", "yes"]
    # Stratified: the minority class appears on both sides.
    assert "yes" in set(data.y_train) and "yes" in set(data.y_test)


def test_split_is_deterministic_and_disjoint(fixture_csv) -> None:
    a, b = load(fixture_csv, "churned"), load(fixture_csv, "churned")
    assert list(a.X_test.index) == list(b.X_test.index)
    assert not set(a.X_train.index) & set(a.X_test.index)


def test_split_fraction_matches_the_constant() -> None:
    data = load("data/churn.csv", "churned")
    assert len(data.X_test) == pytest.approx(1200 * TEST_SIZE, abs=1)


def test_prepared_dataset_trains_all_three_models(fixture_csv) -> None:
    """The Data Engineer's output must satisfy every currently supported model."""
    data = load(fixture_csv, "churned")
    for model in ("logreg", "random_forest", "hist_gb"):
        result = run(data, ExperimentConfig(model=model))
        assert result.ok, f"{model} failed on the prepared dataset: {result.error}"
        assert 0.0 <= result.metrics["f1_macro"] <= 1.0


def test_missing_values_and_categoricals_survive_preprocessing(fixture_csv) -> None:
    """NaNs and strings reach the engine raw; the pipeline imputes and encodes them."""
    data = load(fixture_csv, "churned")
    assert data.X_train.isna().to_numpy().any() or data.X_test.isna().to_numpy().any()
    assert pd.api.types.is_string_dtype(data.X_train["region"])  # raw text, not encoded
    assert run(data, ExperimentConfig(model="logreg")).ok


# --- integration: CSV in, Milestone 1 guarantees intact ------------------------------


def test_csv_reaches_the_loop_with_all_milestone_1_guarantees(fixture_csv) -> None:
    run_ = mission(fixture_csv, "churned", budget=Budget(max_experiments=3))
    experiments = list(run_.run())

    assert len(experiments) >= 2
    for previous, current in zip(experiments, experiments[1:]):
        assert current.config == previous.hypothesis.config  # autonomy preserved
    assert experiments[-1].decision.action is Action.STOP  # termination preserved
    assert len(run_.history) == len(experiments)  # history retained
    fingerprints = [e.config.fingerprint() for e in experiments]
    assert len(set(fingerprints)) == len(fingerprints)  # no repeats


def test_inferred_target_reaches_the_loop(fixture_csv) -> None:
    run_ = mission(fixture_csv, budget=Budget(max_experiments=2))  # no target given
    experiments = list(run_.run())
    assert run_.data.profile.target == "churned"
    assert len(experiments) == 2
    assert run_.summary()["best_score"] is not None


def test_event_stream_shape_is_unchanged_on_the_happy_path(fixture_csv) -> None:
    """Milestone 1's contract: same stages, same order, monotonic seq."""
    seen: list[str] = []
    run_ = Mission(load(fixture_csv, "churned"), budget=Budget(max_experiments=2), sink=lambda e: seen.append(e["stage"]))
    list(run_.run())

    assert seen[0] == "mission_start" and seen[-1] == "mission_end"
    assert "data_engineer" not in seen  # only ever emitted on the blocked path
    assert seen[1:6] == ["experiment_start", "experiment_result", "diagnosis", "hypothesis", "decision"]
    assert [e["seq"] for e in run_.events] == list(range(len(run_.events)))


def test_mission_start_payload_carries_the_profile_as_a_dict(fixture_csv) -> None:
    run_ = Mission(load(fixture_csv, "churned"), budget=Budget(max_experiments=1))
    list(run_.run())
    profile = run_.events[0]["profile"]
    assert isinstance(profile, dict)
    assert profile["rows"] == 12
    assert profile["target"] == "churned"
    assert profile["imbalance_ratio"] == pytest.approx(3.0)
    # Downstream components still receive a plain dict, unchanged from Milestone 1.
    assert run_.profile == profile


def test_needs_clarification_is_observable_and_never_trains(tmp_path) -> None:
    """A bad target must surface on the event stream, not crash or reach engine.run()."""
    frame = pd.DataFrame({"x": range(60), "price": [i * 1.5 for i in range(60)]})
    path = tmp_path / "reg.csv"
    frame.to_csv(path, index=False)

    seen: list[dict] = []
    run_ = mission(path, sink=seen.append)
    experiments = list(run_.run())

    assert experiments == []  # no experiment ever ran
    assert len(run_.history) == 0
    stages = [e["stage"] for e in seen]
    assert stages == ["data_engineer", "mission_end"]
    assert "mission_start" not in stages  # mutually exclusive, no seq collision
    assert [e["seq"] for e in seen] == [0, 1]
    assert seen[0]["status"] == "needs_clarification"
    assert "classification only" in seen[0]["reason"]
    assert run_.summary()["stop_reason"]
    assert run_.summary()["best_score"] is None


def test_unconfirmed_target_cannot_reach_training_even_without_a_mission(tmp_path) -> None:
    """Target validity is the Data Engineer's to decide, and the Engine enforces it.

    Mission.run() already blocks this, but it is not the only way in: load()
    straight into run() must refuse too, or the authority is only advisory.
    """
    ambiguous = tmp_path / "ambiguous.csv"
    pd.DataFrame({"a": range(30), "b": range(30, 60), "ident": range(60, 90)}).to_csv(
        ambiguous, index=False
    )

    data = load(ambiguous)  # no target passed, and none is inferable
    assert data.profile.usable() is False
    assert data.profile.target_confidence is TargetConfidence.NEEDS_CLARIFICATION
    # The guessed target is still baked into the split, which is exactly why the
    # Engine -- not the caller -- has to be the one to refuse.
    assert data.y_train.name == "ident"

    result = run(data, ExperimentConfig(model="logreg"))  # Mission bypassed entirely
    assert result.ok is False
    assert "has not confirmed target" in result.error
    assert result.metrics == {}  # nothing measured
    assert result.score() is None  # and nothing scorable

    # A confirmed target on the same low-level path still trains, unchanged.
    clear = tmp_path / "clear.csv"
    pd.DataFrame(
        {"a": range(30), "plan": ["x", "y", "z"] * 10, "label": [0, 1] * 15}
    ).to_csv(clear, index=False)
    good = load(clear)
    assert good.profile.usable() is True
    assert good.profile.target == "label"
    assert run(good, ExperimentConfig(model="logreg")).ok is True
