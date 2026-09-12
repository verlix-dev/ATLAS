"""Data Engineer: a CSV path in, a profiled and split Dataset out.

Runs once, before the experiment loop. Every fact here is measured from the
file; no component downstream is allowed to guess at dataset structure.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

# Column names that conventionally denote a supervised target.
CONVENTIONAL_NAMES = frozenset({"target", "label", "class", "y", "outcome"})

MAX_CLASSES = 50  # above this, a column is not a classification target
CONTINUOUS_UNIQUE = 20  # a numeric column with more distinct values looks continuous
TEST_SIZE = 0.25


class TargetConfidence(StrEnum):
    EXPLICIT = "explicit"  # the user named the column
    INFERRED = "inferred"  # a conservative heuristic fired
    NEEDS_CLARIFICATION = "needs_clarification"  # ATLAS must not guess


@dataclass
class DataProfile:
    """Structured dataset facts. Field names match the Milestone 1 event payload."""

    rows: int
    columns: int
    target: str
    problem_type: str
    target_confidence: TargetConfidence
    target_reason: str
    numeric: list[str]
    categorical: list[str]
    dtypes: dict[str, str]
    missing_per_column: dict[str, int]
    missing_fraction: float
    duplicate_rows: int
    constant_columns: list[str]
    class_counts: dict[str, int]
    minority_class: str
    imbalance_ratio: float

    def usable(self) -> bool:
        return self.target_confidence is not TargetConfidence.NEEDS_CLARIFICATION


@dataclass
class Dataset:
    """A profiled, split dataset. The contract the Experiment Engine consumes."""

    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    numeric: list[str]
    categorical: list[str]
    classes: list  # every label present, so a class absent from train is still scored
    profile: DataProfile


def detect_target(df: pd.DataFrame) -> tuple[str, TargetConfidence, str]:
    """Infer the target column conservatively, reporting which heuristic fired."""
    named = [c for c in df.columns if str(c).strip().lower() in CONVENTIONAL_NAMES]
    if len(named) == 1:
        return named[0], TargetConfidence.INFERRED, f"column {named[0]!r} matches a conventional target name"
    if len(named) > 1:
        return (
            named[0],
            TargetConfidence.NEEDS_CLARIFICATION,
            f"{len(named)} columns match conventional target names ({named}); ATLAS will not choose between them",
        )

    last = df.columns[-1]
    distinct = int(df[last].nunique(dropna=True))
    # A target's classes must repeat. A column with one row per value is an
    # identifier, however few distinct values a tiny frame happens to have.
    repeats = distinct <= max(2, len(df) // 2)
    if 2 <= distinct <= 10 and repeats:
        return (
            last,
            TargetConfidence.INFERRED,
            f"last column {last!r} has {distinct} distinct values across {len(df)} rows",
        )
    return (
        last,
        TargetConfidence.NEEDS_CLARIFICATION,
        f"no conventional target name, and last column {last!r} has {distinct} distinct values "
        f"across {len(df)} rows — not a confident classification target; pass the target explicitly",
    )


def _classification_check(y: pd.Series) -> tuple[str, str | None]:
    """Return (problem_type, blocking_reason). ATLAS supports classification only."""
    distinct = int(y.nunique(dropna=True))
    if distinct < 2:
        return "degenerate", f"target has {distinct} distinct value(s); nothing to classify"
    numeric = pd.api.types.is_numeric_dtype(y)
    if numeric and (pd.api.types.is_float_dtype(y) or distinct > CONTINUOUS_UNIQUE):
        if distinct > CONTINUOUS_UNIQUE:
            return "regression", (
                f"target looks continuous ({distinct} distinct numeric values); "
                "ATLAS supports classification only"
            )
    if distinct > MAX_CLASSES:
        return "classification", f"target has {distinct} classes, above the {MAX_CLASSES} supported"
    return "classification", None


def profile_frame(df: pd.DataFrame, target: str, confidence: TargetConfidence, reason: str) -> DataProfile:
    """Measure every structural fact ATLAS reasons about."""
    duplicates = int(df.duplicated().sum())
    y = df[target]
    X = df.drop(columns=[target])

    numeric = X.select_dtypes(include="number").columns.tolist()
    categorical = [c for c in X.columns if c not in numeric]

    problem_type, blocker = _classification_check(y)
    if blocker:
        confidence = TargetConfidence.NEEDS_CLARIFICATION
        reason = f"{reason}; {blocker}"

    counts = y.value_counts()
    small = len(counts) <= MAX_CLASSES  # keep the event payload bounded on a bad target
    return DataProfile(
        rows=int(len(df)),
        columns=int(X.shape[1]),
        target=target,
        problem_type=problem_type,
        target_confidence=confidence,
        target_reason=reason,
        numeric=numeric,
        categorical=categorical,
        dtypes={str(c): str(t) for c, t in X.dtypes.items()},
        missing_per_column={str(k): int(v) for k, v in X.isna().sum().items() if v},
        missing_fraction=float(X.isna().to_numpy().mean()) if X.shape[1] else 0.0,
        duplicate_rows=duplicates,
        constant_columns=[c for c in X.columns if X[c].nunique(dropna=False) <= 1],
        class_counts={str(k): int(v) for k, v in counts.items()} if small else {},
        minority_class=str(counts.idxmin()) if small else "",
        imbalance_ratio=float(counts.max() / counts.min()) if small else 1.0,
    )


def load(csv: str | Path, target: str | None = None, *, seed: int = 0) -> Dataset:
    """Ingest a CSV, profile it, and produce the split Dataset the engine consumes.

    Raises only on trust-boundary violations (unreadable file, named target
    missing, no usable rows). An ambiguous target is reported through the
    profile as needs_clarification, not raised.
    """
    path = Path(csv)
    if not path.is_file():
        raise FileNotFoundError(f"dataset not found: {path}")
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"dataset is empty: {path}")

    if target is None:
        target, confidence, reason = detect_target(df)
    else:
        if target not in df.columns:
            raise ValueError(f"target {target!r} not in columns: {list(df.columns)}")
        confidence, reason = TargetConfidence.EXPLICIT, "target supplied by the user"

    df = df.dropna(subset=[target])
    if df.empty:
        raise ValueError("no rows remain after dropping missing target values")

    profile = profile_frame(df, target, confidence, reason)

    y = df[target]
    X = df.drop(columns=[target])
    counts = y.value_counts()
    # Stratify only when every class can appear on both sides of the split.
    stratify = y if counts.min() >= 2 and len(counts) <= MAX_CLASSES else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=seed, stratify=stratify
    )
    return Dataset(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        numeric=profile.numeric,
        categorical=profile.categorical,
        classes=sorted(y.unique(), key=str),
        profile=profile,
    )
