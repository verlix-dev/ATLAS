"""ML Engineer: picks the baseline, owns the model set, turns a hypothesis into a config."""

from __future__ import annotations

import inspect

from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from atlas.state import ExperimentConfig, Hypothesis

# Small, reliable tabular set. Iteration matters more than algorithm count.
# All three accept class_weight, so no capability table is needed.
MODELS = {
    "logreg": (LogisticRegression, {"max_iter": 1000}),
    "random_forest": (RandomForestClassifier, {"n_estimators": 200, "random_state": 0, "n_jobs": -1}),
    "hist_gb": (HistGradientBoostingClassifier, {"random_state": 0}),
}

# Escalation order used when a result is merely insufficient.
LADDER = ["logreg", "random_forest", "hist_gb"]


def build_estimator(config: ExperimentConfig) -> object:
    """Instantiate the estimator. Raises on an unknown model or invalid params."""
    if config.model not in MODELS:
        raise ValueError(f"unknown model {config.model!r}; known: {sorted(MODELS)}")
    cls, defaults = MODELS[config.model]
    return cls(**{**defaults, **config.params})


def allowed_params(model: str) -> set[str]:
    """Parameter names this estimator accepts, read from its own signature.

    No second table to drift out of step with scikit-learn, and nothing is
    constructed to find out. Value *ranges* are not our business: scikit-learn
    remains the authority on those and reports them through engine.run().
    """
    if model not in MODELS:
        raise ValueError(f"unknown model {model!r}; known: {sorted(MODELS)}")
    return set(inspect.signature(MODELS[model][0]).parameters)


def baseline() -> ExperimentConfig:
    """Simplest defensible starting point: a scaled linear model."""
    return ExperimentConfig(model="logreg", params={}, preprocess={"impute": "median", "scale": True})


def apply(hypothesis: Hypothesis) -> ExperimentConfig:
    """The next experiment is whatever the hypothesis specified."""
    return hypothesis.config
