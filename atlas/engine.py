"""Experiment Engine: the only component that actually trains models.

Everything numeric ATLAS reasons about is measured here. No component
upstream is allowed to invent a metric.
"""

from __future__ import annotations

import io
import time
import traceback
import warnings as warnings_mod
from contextlib import redirect_stderr, redirect_stdout
from typing import Any

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from atlas.data_engineer import Dataset
from atlas.ml_engineer import build_estimator
from atlas.objective import DEFAULT_METRIC, UNAVAILABLE, Objective, score
from atlas.state import ExperimentConfig, ExperimentResult

MAX_CAPTURE = 4000  # keep captured streams inspectable, not unbounded


def _preprocessor(data: Dataset, preprocess: dict[str, Any]) -> ColumnTransformer:
    impute = preprocess.get("impute", "median")
    numeric_steps: list[tuple[str, Any]] = [("impute", SimpleImputer(strategy=impute))]
    if preprocess.get("scale", True):
        numeric_steps.append(("scale", StandardScaler()))
    return ColumnTransformer(
        [
            ("num", Pipeline(numeric_steps), data.numeric),
            (
                "cat",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="most_frequent")),
                        ("encode", OneHotEncoder(handle_unknown="ignore", min_frequency=2)),
                    ]
                ),
                data.categorical,
            ),
        ],
        remainder="drop",
    )


def _positive_class(data: Dataset):
    """The class precision/recall are measured against in a binary problem.

    The minority class: in an imbalanced problem it is the one being detected
    (fraud, disease), and it is the one a recall-priority objective means.
    Measured from the full dataset, so it is stable across experiments.
    """
    counts = data.profile.class_counts
    return min(data.classes, key=lambda c: counts.get(str(c), 0))


def _threshold_metrics(pipeline, X_test, y_test, data: Dataset) -> tuple[dict, list[str]]:
    """roc_auc and pr_auc, or an explicit UNAVAILABLE with the reason recorded."""
    notes: list[str] = []
    if len(data.classes) != 2:
        notes.append(
            f"roc_auc/pr_auc unavailable: {len(data.classes)} classes; "
            "binary only this milestone (no one-vs-rest averaging)"
        )
        return {"roc_auc": UNAVAILABLE, "pr_auc": UNAVAILABLE}, notes
    if len(np.unique(y_test)) < 2:
        notes.append("roc_auc/pr_auc unavailable: the validation split holds a single class")
        return {"roc_auc": UNAVAILABLE, "pr_auc": UNAVAILABLE}, notes
    if not hasattr(pipeline, "predict_proba"):
        notes.append(f"roc_auc/pr_auc unavailable: {type(pipeline[-1]).__name__} has no predict_proba")
        return {"roc_auc": UNAVAILABLE, "pr_auc": UNAVAILABLE}, notes

    positive = _positive_class(data)
    column = list(pipeline.classes_).index(positive)
    probabilities = pipeline.predict_proba(X_test)[:, column]
    actual = (y_test == positive).astype(int)  # binarise, so string labels are fine
    return {
        "roc_auc": float(roc_auc_score(actual, probabilities)),
        "pr_auc": float(average_precision_score(actual, probabilities)),
    }, notes


def run(data: Dataset, config: ExperimentConfig, objective: Objective | None = None) -> ExperimentResult:
    """Train and evaluate one configuration. Never raises: failures come back structured.

    The split is fixed by the Data Engineer, so every experiment is comparable.
    """
    # Target validity belongs to the Data Engineer. This is the only component
    # that trains, so refusing here closes every path into training -- not just
    # the one Mission.run() happens to guard.
    if not data.profile.usable():
        return ExperimentResult(
            ok=False,
            error=(
                f"data engineer has not confirmed target {data.profile.target!r}: "
                f"{data.profile.target_reason}"
            ),
            primary_metric=objective.primary_metric if objective else DEFAULT_METRIC,
        )

    out, err = io.StringIO(), io.StringIO()
    started = time.perf_counter()
    captured: list[str] = []
    try:
        with redirect_stdout(out), redirect_stderr(err), warnings_mod.catch_warnings(record=True) as caught:
            warnings_mod.simplefilter("always")
            estimator = build_estimator(config)
            X_train, X_test = data.X_train, data.X_test
            y_train, y_test = data.y_train, data.y_test
            pipeline = Pipeline(
                [("prep", _preprocessor(data, config.preprocess)), ("model", estimator)]
            )
            pipeline.fit(X_train, y_train)
            predictions = pipeline.predict(X_test)
            train_predictions = pipeline.predict(X_train)

            labels = data.classes
            binary = len(labels) == 2
            positive = _positive_class(data) if binary else None
            # Binary: measure the class being detected. Multiclass: macro, as f1 is.
            shared = (
                {"pos_label": positive, "average": "binary"}
                if binary
                else {"average": "macro"}
            )
            recalls = recall_score(y_test, predictions, labels=labels, average=None, zero_division=0)
            test_f1 = float(f1_score(y_test, predictions, average="macro", zero_division=0))
            train_f1 = float(f1_score(y_train, train_predictions, average="macro", zero_division=0))
            threshold, threshold_notes = _threshold_metrics(pipeline, X_test, y_test, data)
            captured.extend(threshold_notes)
            metrics = {
                "f1_macro": test_f1,
                "accuracy": float(accuracy_score(y_test, predictions)),
                "precision": float(precision_score(y_test, predictions, zero_division=0, **shared)),
                "recall": float(recall_score(y_test, predictions, zero_division=0, **shared)),
                **threshold,
                "positive_class": str(positive) if binary else None,
                "train_f1_macro": train_f1,
                "overfit_gap": train_f1 - test_f1,
                "per_class_recall": {str(l): float(r) for l, r in zip(labels, recalls)},
                "confusion_matrix": confusion_matrix(y_test, predictions, labels=labels).tolist(),
                "labels": [str(l) for l in labels],
                "test_rows": int(len(y_test)),
                "predicted_classes": int(len(np.unique(predictions))),
            }
            captured.extend(f"{w.category.__name__}: {w.message}" for w in caught)
    except Exception as exc:  # noqa: BLE001 - deliberate: errors are evidence, not crashes
        return ExperimentResult(
            ok=False,
            error=f"{type(exc).__name__}: {exc}",
            traceback=traceback.format_exc()[-MAX_CAPTURE:],
            stdout=out.getvalue()[-MAX_CAPTURE:],
            stderr=err.getvalue()[-MAX_CAPTURE:],
            warnings=captured,
            seconds=time.perf_counter() - started,
            primary_metric=objective.primary_metric if objective else DEFAULT_METRIC,
        )
    result = ExperimentResult(
        ok=True,
        metrics=metrics,
        stdout=out.getvalue()[-MAX_CAPTURE:],
        stderr=err.getvalue()[-MAX_CAPTURE:],
        warnings=captured,
        seconds=time.perf_counter() - started,
        primary_metric=objective.primary_metric if objective else DEFAULT_METRIC,
    )
    if objective and score(result, objective) is None:
        # Framed for diagnose(): the run succeeded but cannot be ranked.
        result.warnings.append(
            f"objective metric {objective.primary_metric!r} is unavailable for this experiment; "
            "it stays in History but is excluded from best-selection"
        )
    return result
