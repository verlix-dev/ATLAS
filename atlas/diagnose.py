"""Error Analyst: classifies what actually happened, from measured evidence only.

Deliberately deterministic. Classifying a result is measurement, not opinion,
so an LLM has no job here.
"""

from __future__ import annotations

import re
from typing import Any

from atlas.objective import UNAVAILABLE, improvement
from atlas.state import Category, Diagnosis, ExperimentResult, History

MIN_GAIN = 0.01  # below this, an experiment "worked" but didn't earn its place
OVERFIT_GAP = 0.15
WEAK_RECALL = 0.5

# Substrings in an exception that point at the data rather than the code.
DATA_SIGNALS = ("could not convert", "nan", "infinity", "dtype", "n_splits", "least populated")
# Matched with a LEADING word boundary so a bare token cannot fire on an
# unrelated word ("nan" must not match "maintenance"). No trailing boundary,
# so plural and suffixed forms ("nans", "dtypes") still count.
DATA_SIGNAL_RE = re.compile("|".join(rf"\b{re.escape(s)}" for s in DATA_SIGNALS))


def diagnose(result: ExperimentResult, history: History, profile: dict[str, Any]) -> Diagnosis:
    baseline_score = history.best_score()

    if not result.ok:
        error = (result.error or "").lower()
        is_data = bool(DATA_SIGNAL_RE.search(error))
        return Diagnosis(
            category=Category.DATA_ISSUE if is_data else Category.EXECUTION_ERROR,
            summary=result.error or "execution failed",
            findings=[result.traceback.strip().splitlines()[-1]] if result.traceback else [],
            # Nothing was measured, so nothing numeric is reported. weak_classes
            # is omitted entirely: absent means "not measured", and propose()
            # already reads it as "nothing actionable".
            evidence={
                "error": result.error,
                "data_related": is_data,
                "score": UNAVAILABLE,
                "overfit_gap": UNAVAILABLE,
            },
        )

    score = result.score()  # None when the objective's metric was not computable
    findings: list[str] = []
    metrics = result.metrics

    gap = metrics.get("overfit_gap", 0.0)
    if gap > OVERFIT_GAP:
        findings.append(f"overfitting: train f1 exceeds test f1 by {gap:.3f}")

    weak = {c: r for c, r in metrics.get("per_class_recall", {}).items() if r < WEAK_RECALL}
    if weak:
        worst = min(weak, key=weak.get)
        findings.append(f"weak recall on class {worst!r}: {weak[worst]:.3f}")

    if metrics.get("predicted_classes", 2) < len(metrics.get("labels", [])):
        findings.append("model never predicts some classes")

    # Absent means "not measured". Never assume 1.0, which would read as a
    # measured claim that the classes are perfectly balanced.
    imbalance = profile.get("imbalance_ratio")
    if imbalance is not None and imbalance > 1.5 and weak:
        findings.append(f"class imbalance {imbalance:.2f}:1 with weak minority recall")

    gain = (
        improvement(score, baseline_score, history.objective)
        if score is not None and baseline_score is not None
        else None
    )
    evidence = {
        "score": score if score is not None else UNAVAILABLE,
        "previous_best": baseline_score,
        "gain": gain,
        "overfit_gap": gap,
        "imbalance_ratio": imbalance if imbalance is not None else UNAVAILABLE,
        "weak_classes": weak,
    }

    if score is None:
        # The objective's metric was not computed, so there is no number to
        # report and no gain to claim. The category is chosen exactly as the
        # scorable paths below would choose it, leaving loop behaviour intact.
        return Diagnosis(
            category=Category.MODEL_ISSUE
            if findings
            else Category.IMPROVEMENT
            if baseline_score is None
            else Category.INSUFFICIENT,
            summary=f"{result.primary_metric} is {UNAVAILABLE} for this experiment; "
            "excluded from ranking"
            + (f"; {len(findings)} issue(s) found" if findings else ""),
            findings=findings,
            evidence=evidence,
        )

    if baseline_score is None:
        category = Category.MODEL_ISSUE if findings else Category.IMPROVEMENT
        return Diagnosis(
            category=category,
            summary=f"baseline established at {score:.3f}"
            + (f"; {len(findings)} issue(s) found" if findings else ""),
            findings=findings,
            evidence=evidence,
        )

    if gain is not None and gain >= MIN_GAIN:
        return Diagnosis(
            category=Category.IMPROVEMENT,
            summary=f"improved {baseline_score:.3f} -> {score:.3f} (+{gain:.3f})",
            findings=findings,
            evidence=evidence,
        )

    if findings:
        return Diagnosis(
            category=Category.MODEL_ISSUE,
            summary=f"no gain ({score:.3f} vs best {baseline_score:.3f}); model issues identified",
            findings=findings,
            evidence=evidence,
        )

    return Diagnosis(
        category=Category.INSUFFICIENT,
        summary=f"trained cleanly but gain {gain:+.3f} is below the {MIN_GAIN} threshold",
        findings=[],
        evidence=evidence,
    )
