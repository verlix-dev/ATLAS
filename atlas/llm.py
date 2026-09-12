"""The LLM seam: build a context, validate a proposal. Nothing here executes.

The only crossing to a model is `Callable[[dict], str]` — a JSON-serialisable
dict in, a string out. No dataset, split, path, estimator or callable ever
crosses, so a response cannot cause execution: it can only be rejected, or
turned into the same Hypothesis the deterministic engine already produces.

This module reads ml_engineer's model table to validate against it. It never
constructs or fits an estimator — `allowed_params` works off the signature.
"""

from __future__ import annotations

import json
from typing import Any, Callable

from atlas.ml_engineer import MODELS, allowed_params
from atlas.objective import UNAVAILABLE, Objective
from atlas.state import Diagnosis, ExperimentConfig, ExperimentResult, History, Hypothesis

# The whole model interface. Deliberately not a class: there is nothing to hold.
Llm = Callable[[dict], str]

# Exactly the fields a proposal may carry. Anything else is a rejection, which
# is what stops a model smuggling a metric, a decision, or its own provenance.
ALLOWED_FIELDS = (
    "problem",
    "proposed_change",
    "reasoning",
    "expected_effect",
    "cites",
    "model",
    "params",
    "preprocess",
)
PROSE_FIELDS = ("problem", "proposed_change", "reasoning", "expected_effect")
MAX_PROSE = 500

# engine._preprocessor reads exactly these two keys; nothing else is wired up,
# so anything else would be silently ignored rather than applied.
PREPROCESS_KEYS = ("impute", "scale")
IMPUTE_STRATEGIES = ("mean", "median", "most_frequent", "constant")

# Scalars only. A nested structure is never a scikit-learn scalar param and is
# the obvious place to hide a payload.
PARAM_TYPES = (str, bool, int, float, type(None))

# Measured dataset facts the model may see. A whitelist, so a new DataProfile
# field is not exposed by accident. Note the absence of target, target_reason
# and target_confidence: target validity is settled before the loop starts and
# is not the model's business.
PROFILE_FIELDS = (
    "rows",
    "columns",
    "problem_type",
    "numeric",
    "categorical",
    "dtypes",
    "missing_per_column",
    "duplicate_rows",
    "constant_columns",
    "class_counts",
    "minority_class",
    "imbalance_ratio",
)

SYSTEM_PROMPT = """You are the hypothesis component of ATLAS, an automated ML \
experimentation system. You are given measured facts about one dataset, the \
diagnosis of the most recent experiment, and every experiment already run.

Propose the single next experiment to run. Reason from the measured results in \
`history` — what has already been tried, and what it actually scored on the \
objective's primary metric.

Reply with one JSON object and nothing else. No markdown fences, no commentary.

{
  "problem":         "what the last result shows is wrong, in one sentence",
  "proposed_change": "the change you are making, in one sentence",
  "reasoning":       "why this change addresses that problem, citing the numbers",
  "expected_effect": "what you expect to happen to the primary metric",
  "cites":           ["exact string from diagnosis.findings, or an evidence key"],
  "model":           "one of allowed.models",
  "params":          {"param_name": value},
  "preprocess":      {"impute": "median", "scale": true}
}

Rules:
- `model` must be a key of `allowed.models`.
- every key of `params` must appear in that model's list in `allowed.models`.
- `preprocess` may only use the keys and values in `allowed.preprocess`.
- every entry of `cites` must be an exact string from `diagnosis.findings` or an
  exact key of `diagnosis.evidence`. If `findings` is non-empty, cite at least
  one of them.
- do not repeat a configuration that already appears in `history`.
- never report a metric, a score, or a decision about whether to stop. You do not
  measure anything and you do not decide anything; you propose one experiment.
"""


def _score(result: ExperimentResult) -> Any:
    """A measured score, or the canonical marker. Never a stand-in number."""
    value = result.score()
    return UNAVAILABLE if value is None else value


def context(
    diagnosis: Diagnosis,
    history: History,
    profile: dict[str, Any],
    objective: Objective,
    budget: Any,
    current: ExperimentConfig,
) -> dict[str, Any]:
    """Everything the model may see: measured facts, plus the action space.

    Pure data — no path, no rows, no split, no handle. An unavailable metric
    stays UNAVAILABLE here too: the prompt must not be the one place a missing
    measurement quietly becomes a number.
    """
    return {
        "objective": objective.as_dict(),
        "dataset": {k: profile[k] for k in PROFILE_FIELDS if k in profile},
        "diagnosis": {
            "category": str(diagnosis.category),
            "summary": diagnosis.summary,
            "findings": list(diagnosis.findings),
            "evidence": diagnosis.evidence,
        },
        "current": {
            "model": current.model,
            "params": dict(current.params),
            "preprocess": dict(current.preprocess),
        },
        "history": [
            {
                "id": e.id,
                "model": e.config.model,
                "params": dict(e.config.params),
                "preprocess": dict(e.config.preprocess),
                "ok": e.result.ok,
                "score": _score(e.result),
            }
            for e in history.experiments
        ],
        "allowed": {
            "models": {name: sorted(allowed_params(name)) for name in MODELS},
            "preprocess": {"impute": list(IMPUTE_STRATEGIES), "scale": [True, False]},
        },
        # Read-only counters, so the model can reason about how much runway is
        # left. It has no lever on any of them.
        "budget": {
            "experiments_run": len(history),
            "max_experiments": budget.max_experiments,
            "rounds_since_improvement": history.rounds_since_improvement(),
            "patience": budget.patience,
        },
    }


def _ask(llm: Llm, payload: dict[str, Any]) -> tuple[str | None, str | None]:
    """Call the model. Returns (raw, None) or (None, reason). Never raises.

    Timeouts belong to the client: ATLAS calls a plain Callable[[dict], str],
    so a timeout, a network drop, a rate limit and an auth failure all arrive
    here as the same thing — an exception — and all become one recorded
    rejection plus a deterministic fallback.

    ponytail: one attempt, no retry. Add a single reprompt only if observed
    parse-failure rates justify it.
    """
    try:
        raw = llm(payload)
    except Exception as exc:  # noqa: BLE001 - deliberate: an outage is not a crash
        return None, f"llm_unavailable: {type(exc).__name__}: {exc}"
    return raw, None


def validate(
    raw: Any, diagnosis: Diagnosis, history: History, current: ExperimentConfig
) -> tuple[Hypothesis | None, str]:
    """Eight ordered gates. Returns (hypothesis, "accepted") or (None, reason).

    A gate, not an oracle. It guarantees shape and identity — the model exists,
    the params are real names for it, the config is novel, the claim points at
    measured evidence. Parameter *ranges* stay with scikit-learn, which reports
    them through engine.run() as a structured failed experiment.
    """
    # 1. JSON
    if not isinstance(raw, str):
        return None, f"invalid_json: expected a string, got {type(raw).__name__}"
    try:
        proposal = json.loads(raw)
    except ValueError as exc:  # JSONDecodeError is a subclass
        return None, f"invalid_json: {exc}"
    if not isinstance(proposal, dict):
        return None, f"invalid_json: expected an object, got {type(proposal).__name__}"

    # 2. schema
    if unknown := sorted(set(proposal) - set(ALLOWED_FIELDS)):
        return None, f"schema: unknown field(s) {unknown}"
    if missing := sorted(set(ALLOWED_FIELDS) - set(proposal)):
        return None, f"schema: missing field(s) {missing}"
    for field_name in PROSE_FIELDS:
        value = proposal[field_name]
        if not isinstance(value, str) or not value.strip():
            return None, f"schema: {field_name} must be a non-empty string"
        if len(value) > MAX_PROSE:
            return None, f"schema: {field_name} exceeds {MAX_PROSE} characters"
    if not isinstance(proposal["params"], dict):
        return None, "schema: params must be an object"
    if not isinstance(proposal["preprocess"], dict):
        return None, "schema: preprocess must be an object"
    cites = proposal["cites"]
    if not isinstance(cites, list) or not cites or not all(isinstance(c, str) for c in cites):
        return None, "schema: cites must be a non-empty list of strings"

    # 3. model
    model = proposal["model"]
    if not isinstance(model, str) or model not in MODELS:
        return None, f"unknown_model: {model!r} not in {sorted(MODELS)}"

    # 4. parameter names
    params = proposal["params"]
    if unknown := sorted(set(params) - allowed_params(model)):
        return None, f"unknown_params: {unknown} not accepted by {model}"

    # 5. parameter value types
    for name, value in params.items():
        if not isinstance(value, PARAM_TYPES):
            return None, f"param_type: {name}={value!r} is a {type(value).__name__}"

    # 6. preprocessing
    preprocess = proposal["preprocess"]
    if unknown := sorted(set(preprocess) - set(PREPROCESS_KEYS)):
        return None, f"preprocess: unknown key(s) {unknown}"
    if "impute" in preprocess and preprocess["impute"] not in IMPUTE_STRATEGIES:
        return None, (
            f"preprocess: impute {preprocess['impute']!r} not in {list(IMPUTE_STRATEGIES)}"
        )
    if "scale" in preprocess and not isinstance(preprocess["scale"], bool):
        return None, f"preprocess: scale must be a boolean, got {preprocess['scale']!r}"

    # 7. duplicate. Merge onto the dataclass default so a partial preprocess
    # fingerprints identically to the equivalent deterministic config -- without
    # this, {"scale": false} and {"impute": "median", "scale": false} behave the
    # same but look different to history.tried().
    canonical = {**ExperimentConfig(model=model).preprocess, **preprocess}
    config = ExperimentConfig(model=model, params=dict(params), preprocess=canonical)
    if history.tried(config):
        prior = next(
            e.id for e in history.experiments if e.config.fingerprint() == config.fingerprint()
        )
        return None, f"duplicate: already run as experiment {prior}"

    # 8. evidence citation
    findings = set(diagnosis.findings)
    supported = findings | set(diagnosis.evidence)
    if invented := [c for c in cites if c not in supported]:
        return None, f"uncited: {invented[0]!r} is not in the diagnosis"
    if findings and not any(c in findings for c in cites):
        return None, "uncited: the diagnosis reported findings, but none are cited"

    return (
        Hypothesis(
            problem=proposal["problem"],
            proposed_change=proposal["proposed_change"],
            reasoning=proposal["reasoning"],
            expected_effect=proposal["expected_effect"],
            config=config,
            # Decision-adjacent (it routes REVISE vs CONTINUE), so derived here
            # and never taken from the model.
            touches_data=canonical != current.preprocess,
            source="llm",
            cites=list(cites),
        ),
        "accepted",
    )
