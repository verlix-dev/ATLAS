"""Planner: a natural-language task in, a MissionPlan out.

Runs exactly once, before the mission starts. It reads the task text and
nothing else — never the CSV, never a model. Everything about *what to try
next* stays with diagnose() -> propose() -> decide().

Deterministic keyword extraction, mirroring propose()'s shape so an LLM can
replace the body later: plan(task) -> MissionPlan.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import StrEnum

DEFAULT_METRIC = "f1_macro"  # what engine.py actually optimises today


class MetricConfidence(StrEnum):
    EXPLICIT = "explicit"  # the task named a metric
    INFERRED = "inferred"  # a stated preference implied one
    AMBIGUOUS = "ambiguous"  # nothing in the task justifies a choice


@dataclass
class MissionPlan:
    objective: str
    problem_type: str | None
    target_candidate: str | None
    primary_metric: str
    metric_confidence: MetricConfidence
    metric_reason: str
    priority: str | None = None
    constraints: list[str] = field(default_factory=list)
    experiment_budget: int | None = None


# --- problem type --------------------------------------------------------------------

CLASSIFICATION_CUES = (
    "classif", "categor", "detect", "fraud", "churn", "spam", "diagnos", "disease",
    "binary", "which class", "whether", "will they", "yes or no", "label",
)
REGRESSION_CUES = (
    "regress", "forecast", "how much", "how many", "estimate the", "price", "revenue",
    "sales", "temperature", "continuous", "numeric value", "amount of",
)

# --- target extraction ---------------------------------------------------------------

# Words that are never the target itself, only scaffolding around it.
# Prepositions and connectives matter: they terminate a captured phrase, so
# "predict customer churn from usage data" yields 'churn', not 'from'.
#
# Interrogative and modal scaffolding matters for the same reason, one step
# earlier: "predict whether a customer will churn" captures 'whether a customer',
# and without these words _pick() would return 'whether' -- a function word that
# is not a column in any dataset, yet would be handed to load() as an EXPLICIT
# target and abort the mission. Stripping them empties the phrase instead, so
# _target_candidate() returns None and the Data Engineer infers the target,
# which is the conservative outcome for prose that never names a column.
STOPWORDS = frozenset({
    "the", "a", "an", "of", "for", "each", "this", "that", "our", "their", "its",
    "customer", "user", "client", "patient", "house", "home", "product", "model",
    "is", "column", "value", "values", "data", "dataset",
    "from", "in", "on", "with", "using", "by", "at", "to", "and", "or", "based",
    "given", "across", "per", "next", "last", "new", "these", "those",
    "whether", "if", "will", "not",
})

TARGET_PATTERNS = (
    r"target\s+(?:column\s+)?(?:is|=|:)\s*['\"`]?([A-Za-z_][\w ]*?)['\"`]?(?:[.,;]|$)",
    r"predict(?:ing)?\s+(?:the\s+)?['\"`]([A-Za-z_]\w*)['\"`]",
    r"label\s+(?:column\s+)?(?:is|=|:)\s*['\"`]?([A-Za-z_]\w*)",
    r"\b([A-Za-z_]\w*)\s+detection\b",
    r"\bdetect\s+((?:[A-Za-z_]\w*\s+){0,2}[A-Za-z_]\w*)\b",
    r"\bpredict(?:ing)?\s+((?:[A-Za-z_]\w*\s+){0,2}[A-Za-z_]\w*)\b",
)


def _pick(phrase: str) -> str | None:
    """Last meaningful word of a matched phrase ('customer churn' -> 'churn')."""
    words = [w for w in re.split(r"\s+", phrase.strip()) if w and w.lower() not in STOPWORDS]
    return words[-1] if words else None


def _target_candidate(task: str) -> str | None:
    for pattern in TARGET_PATTERNS:
        match = re.search(pattern, task, re.IGNORECASE)
        if match and (word := _pick(match.group(1))):
            return word
    return None


# --- metric policy -------------------------------------------------------------------

# "missing fraud costs more than a false alarm" -> recall matters most.
# Both comparative ("is more costly than") and bare ("is costly") phrasing must
# match: if only the comparative form did, a task naming both costs without
# ranking them would match precision alone and be reported as certain.
RECALL_PATTERNS = (
    r"miss(?:ing|ed)?\s+(?:\w+\s+){0,3}(?:is|are)\s+(?:more|worse|costly|expensive|bad|serious)",
    r"(?:false negative|missed case|missed detection)s?\s+(?:are|is)\s+(?:more|worse|costly|expensive)",
    r"(?:cannot|can't|must not|don't)\s+(?:afford to\s+)?miss",
    r"catch\s+(?:as many|every|all)",
    r"worse\s+to\s+miss",
    r"\brecall\b",
)
# "a legitimate email in spam is worse than spam in the inbox" -> precision.
PRECISION_PATTERNS = (
    r"false\s+(?:positive|alarm)s?\s+(?:are|is)\s+(?:more|worse|expensive|costly)",
    r"(?:avoid|minimi[sz]e)\s+false\s+(?:positive|alarm)",
    r"(?:wrongly|incorrectly|falsely)\s+(?:flag|block|reject|classif)",
    r"\bprecision\b",
)
NAMED_METRICS = ("f1_macro", "pr_auc", "roc_auc", "f1", "accuracy", "recall", "precision", "rmse", "mae", "r2")


def _stated_metric(task: str) -> str | None:
    for name in NAMED_METRICS:  # longest-first so "f1_macro" wins over "f1"
        if re.search(rf"\b{re.escape(name)}\b", task, re.IGNORECASE):
            return name
    return None


def _metric(task: str, problem_type: str | None) -> tuple[str, MetricConfidence, str, str | None]:
    """Return (metric, confidence, reason, priority). Never guesses silently."""
    recall_first = any(re.search(p, task, re.IGNORECASE) for p in RECALL_PATTERNS)
    precision_first = any(re.search(p, task, re.IGNORECASE) for p in PRECISION_PATTERNS)

    if named := _stated_metric(task):
        priority = named if named in ("recall", "precision") else None
        return named, MetricConfidence.EXPLICIT, f"the task names {named!r}", priority

    if recall_first and not precision_first:
        return "pr_auc", MetricConfidence.INFERRED, "the task states that missed positives cost more than false alarms", "recall"
    if precision_first and not recall_first:
        return "precision", MetricConfidence.INFERRED, "the task states that false positives are the costly error", "precision"
    if recall_first and precision_first:
        return (
            DEFAULT_METRIC,
            MetricConfidence.AMBIGUOUS,
            "the task describes both missed positives and false alarms as costly, without ranking them",
            None,
        )

    if problem_type == "regression":
        if re.search(r"outlier|robust", task, re.IGNORECASE):
            return "mae", MetricConfidence.INFERRED, "regression, and the task asks for robustness to outliers", None
        if re.search(r"variance explained|goodness of fit", task, re.IGNORECASE):
            return "r2", MetricConfidence.INFERRED, "regression, and the task asks about explained variance", None
        return "rmse", MetricConfidence.INFERRED, "regression with no stated error preference", None

    if problem_type == "classification":
        if re.search(r"\bbalanc(?:ed|e)\b", task, re.IGNORECASE) and not re.search(r"imbalanc", task, re.IGNORECASE):
            return "accuracy", MetricConfidence.INFERRED, "the task describes balanced classes", None
        if re.search(r"imbalanc|rare|skewed|minority", task, re.IGNORECASE):
            return DEFAULT_METRIC, MetricConfidence.INFERRED, "the task describes class imbalance with no stated error preference", None
        return DEFAULT_METRIC, MetricConfidence.INFERRED, "classification with no stated error preference", None

    return (
        DEFAULT_METRIC,
        MetricConfidence.AMBIGUOUS,
        "the task states neither a problem type nor an error preference",
        None,
    )


# --- constraints and budget ----------------------------------------------------------

CONSTRAINT_CUES = (
    "interpretab", "explainab", "no deep learning", "no neural", "cpu only", "cpu-only",
    "no gpu", "latency", "real time", "real-time", "regulat", "gdpr", "audit",
    "memory limit", "must run", "offline",
)
BUDGET_PATTERNS = (
    r"(\d+)\s+(?:experiment|run|trial|iteration)s?\b",
    r"(?:budget|at most|up to|maximum of|max(?:imum)?)\D{0,20}?(\d+)",
)


def _constraints(task: str) -> list[str]:
    sentences = [s.strip() for s in re.split(r"[.;\n]", task) if s.strip()]
    return [s for s in sentences if any(cue in s.lower() for cue in CONSTRAINT_CUES)]


def _budget(task: str) -> int | None:
    for pattern in BUDGET_PATTERNS:
        if match := re.search(pattern, task, re.IGNORECASE):
            value = int(match.group(1))
            if 1 <= value <= 1000:  # anything else is a number about something else
                return value
    return None


def _problem_type(task: str) -> str | None:
    lowered = task.lower()
    classification = sum(cue in lowered for cue in CLASSIFICATION_CUES)
    regression = sum(cue in lowered for cue in REGRESSION_CUES)
    if classification > regression:
        return "classification"
    if regression > classification:
        return "regression"
    return None  # tied or silent: say nothing rather than invent a type


def plan(task: str) -> MissionPlan:
    """Read a task description into a MissionPlan. Raises only on an empty task.

    `target_candidate` is a hint for the Data Engineer, never a decision: it
    goes straight into load(target=...), which owns all target validation.
    """
    if not task or not task.strip():
        raise ValueError("task is empty; the Planner needs a description of the objective")

    task = task.strip()
    problem_type = _problem_type(task)
    metric, confidence, reason, priority = _metric(task, problem_type)
    first_sentence = re.split(r"(?<=[.!?])\s+", task)[0].strip()

    return MissionPlan(
        objective=first_sentence or task,
        problem_type=problem_type,
        target_candidate=_target_candidate(task),
        primary_metric=metric,
        metric_confidence=confidence,
        metric_reason=reason,
        priority=priority,
        constraints=_constraints(task),
        experiment_budget=_budget(task),
    )
