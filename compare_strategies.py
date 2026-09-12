"""M6: one reproducible comparison — deterministic ATLAS vs LLM-assisted ATLAS.

Same CSV, same target, same split, same objective, same budget. The split is
shared by construction: load() runs once and the single returned Dataset object
is handed to both Missions, so there is no second split to drift.

Runs offline. The "LLM" is a deterministic stand-in that sees only the context
dict atlas.llm.context() builds — no CSV, no rows, no estimator, no future
result. It chooses *configurations*; every *score* printed is measured by the
Experiment Engine.

    python compare_strategies.py
    python compare_strategies.py --csv data/fraud.csv --target fraud
"""

from __future__ import annotations

import argparse
import json
from typing import Any

from atlas.data_engineer import Dataset, load
from atlas.decision import Budget
from atlas.loop import Mission

BAR = "=" * 78
RULE = "-" * 78
TIE = 1e-4  # below this the two strategies are reporting the same number


class EvidenceLedProposer:
    """A deterministic stand-in for an LLM, implementing Callable[[dict], str].

    Three rules, in priority order, each keyed off the *diagnosis* rather than
    off any particular dataset — so the same proposer is meaningful on any CSV
    and nothing here is tuned to a known result:

      1. the run is overfitting      -> cut capacity for the current family
      2. a class is under-recalled   -> reweight the loss, and impute by mode
                                        when the profile reports missing values
      3. nothing is blocking         -> escalate to an untried family with a
                                        deliberately regularised parameterisation

    Rules 1 and 3 reach parameter combinations the deterministic ladder cannot
    express. Rule 2 usually agrees with the ladder, which is a legitimate
    outcome and is reported as such.

    It cannot fabricate a result: the only numbers it emits are hyperparameters.
    """

    def __init__(self) -> None:
        self.decisions: list[str] = []

    # -- helpers ---------------------------------------------------------------

    @staticmethod
    def _cite(findings: list[str], evidence: dict, prefer: str) -> list[str]:
        """Quote a real finding. Gate 8 requires one when findings exist."""
        for finding in findings:
            if prefer in finding:
                return [finding]
        if findings:
            return [findings[0]]
        return [next(iter(evidence))] if evidence else []

    @staticmethod
    def _untried(history: list[dict], allowed: dict) -> str | None:
        """First model family with no experiment against it yet."""
        used = {row["model"] for row in history}
        return next((m for m in ("hist_gb", "random_forest", "logreg") if m not in used), None)

    # -- the proposer ----------------------------------------------------------

    def __call__(self, payload: dict) -> str:
        diagnosis = payload["diagnosis"]
        findings, evidence = diagnosis["findings"], diagnosis["evidence"]
        current, history = payload["current"], payload["history"]
        model = current["model"]
        last = history[-1]
        missing = bool(payload["dataset"].get("missing_per_column"))

        gap = evidence.get("overfit_gap")
        gap = gap if isinstance(gap, (int, float)) else None
        weak = evidence.get("weak_classes") or {}
        seen = str(last["score"])

        # RULE 1 -- measured train/test gap says the model memorised the split.
        if gap is not None and gap > 0.15:
            if model == "logreg":
                params = {**current["params"], "C": 0.05}
                change = "tighten regularisation to C=0.05"
            elif model == "random_forest":
                params = {**current["params"], "max_depth": 4,
                          "min_samples_leaf": 20, "n_estimators": 400}
                change = "shallow forest: depth 4, leaf floor 20, 400 trees"
            else:
                params = {**current["params"], "max_depth": 3, "learning_rate": 0.05,
                          "max_iter": 250, "min_samples_leaf": 20}
                change = "slow boosting: depth 3, lr 0.05, 250 iters, leaf floor 20"
            why = (
                f"train f1 exceeds test f1 by {gap:.3f} at a measured "
                f"{payload['objective']['primary_metric']} of {seen}, so the ceiling "
                "is variance, not capacity"
            )
            return self._emit(change, why, "a smaller gap at similar or better score",
                              model, params, current["preprocess"],
                              self._cite(findings, evidence, "overfitting"))

        # RULE 2 -- a class is being missed and the loss is not yet reweighted.
        if weak and "class_weight" not in current["params"]:
            params = {**current["params"], "class_weight": "balanced"}
            prep = dict(current["preprocess"])
            if missing:
                prep["impute"] = "most_frequent"
            worst = min(weak.values())
            why = (
                f"recall on the weakest class is {worst:.3f} while the run scores {seen}; "
                "reweighting the loss is the direct lever"
                + (", and the profile reports missing values that mode-imputation "
                   "handles without inventing a median" if missing else "")
            )
            return self._emit("reweight the loss with class_weight='balanced'"
                              + (" and impute categorically by mode" if missing else ""),
                              why, "higher minority-class recall", model, params, prep,
                              self._cite(findings, evidence, "recall"))

        # RULE 3 -- nothing is blocking; spend the next slot on new capacity.
        upgrade = self._untried(history, payload["allowed"]) or model
        params: dict[str, Any] = {"class_weight": "balanced"} if weak or "class_weight" in current["params"] else {}
        if upgrade == "hist_gb":
            params |= {"learning_rate": 0.04, "max_iter": 300, "min_samples_leaf": 15}
            change = "escalate to hist_gb, slow and heavily regularised (lr 0.04, 300 iters)"
        elif upgrade == "random_forest":
            params |= {"n_estimators": 500, "min_samples_leaf": 3, "max_features": "log2"}
            change = "escalate to random_forest, 500 trees with log2 feature sampling"
        else:
            params |= {"C": 0.5, "max_iter": 2000}
            change = "retune logreg at C=0.5 with a longer solve"
        prep = {"impute": "most_frequent" if missing else "median", "scale": upgrade == "logreg"}
        why = (
            f"experiment {last['id']} measured {seen} with no blocking defect; "
            f"{upgrade} explores non-linear structure the current family cannot reach"
        )
        return self._emit(change, why, "a higher primary metric than the current best",
                          upgrade, params, prep, self._cite(findings, evidence, "imbalance"))

    def _emit(self, change, why, effect, model, params, preprocess, cites) -> str:
        self.decisions.append(change)
        return json.dumps(
            {
                "problem": "the last experiment leaves the objective short of its ceiling",
                "proposed_change": change,
                "reasoning": why,
                "expected_effect": effect,
                "cites": cites,
                "model": model,
                "params": params,
                "preprocess": preprocess,
            }
        )


def run_comparison(data: Dataset, budget: Budget) -> dict[str, Any]:
    """Run both strategies against the same Dataset object. Returns the facts."""
    deterministic = Mission(data, budget=budget)
    list(deterministic.run())

    proposer = EvidenceLedProposer()
    events: list[dict] = []
    guided = Mission(data, budget=budget, sink=events.append, llm=proposer)
    list(guided.run())

    ladder = {e.config.fingerprint() for e in deterministic.history.experiments}
    novel = [
        e for e in guided.history.experiments
        if e.origin.startswith("exp") and "[llm]" in e.origin
        and e.config.fingerprint() not in ladder
    ]
    rejections = [r for e in events if e["stage"] == "hypothesis" for r in e["rejected"]]

    return {
        "data": data,
        "budget": budget,
        "deterministic": deterministic,
        "guided": guided,
        "novel": novel,
        "rejections": rejections,
        "proposer": proposer,
    }


def _params(config) -> str:
    return ", ".join(f"{k}={v}" for k, v in config.params.items()) or "defaults"


def _prep(config) -> str:
    return ", ".join(f"{k}={v}" for k, v in sorted(config.preprocess.items()))


def _trajectory(run, show_source: bool) -> None:
    for e in run.history.experiments:
        source = "baseline" if e.origin == "baseline" else ("llm" if "[llm]" in e.origin else "det")
        score = "  n/a " if e.result.score() is None else f"{e.result.score():.4f}"
        head = f"  {e.id}  " + (f"{source:<9}" if show_source else "")
        print(f"{head}{score}  {e.config.model:<15}{_params(e.config)}")
        print(f"       diagnosis: [{e.diagnosis.category}] {e.diagnosis.summary}")
        if e.hypothesis:
            print(f"       next:      [{e.hypothesis.source}] {e.hypothesis.proposed_change}")
        else:
            print("       next:      none - out of untried ideas")


def render(report: dict[str, Any], csv: str, target: str) -> None:
    deterministic, guided = report["deterministic"], report["guided"]
    d_sum, g_sum = deterministic.summary(), guided.summary()
    objective = d_sum["objective"]

    print(f"\n{BAR}\nATLAS STRATEGY COMPARISON\n{BAR}")
    print(f"Dataset:   {csv}")
    print(f"Target:    {target}")
    print(f"Objective: {objective['primary_metric']} ({objective['direction']})")
    print(f"Budget:    {report['budget'].max_experiments} experiments")
    print(f"Split:     one load(); the same Dataset object drives both runs "
          f"({report['data'].X_train.shape[0]} train / {report['data'].X_test.shape[0]} test rows)")

    print(f"\n{RULE}\nDETERMINISTIC  (llm=None)\n{RULE}")
    print(f"  #  {'Score':<6}  {'Model':<15}Params")
    _trajectory(deterministic, show_source=False)
    print(f"\n  experiments: {d_sum['experiments']}   best: #{d_sum['best_id']} "
          f"{d_sum['best_score']:.4f}   stopped: {d_sum['stop_reason']}")

    print(f"\n{RULE}\nLLM-GUIDED  (M5 seam, deterministic offline proposer)\n{RULE}")
    print(f"  #  {'Source':<9}{'Score':<6}  {'Model':<15}Params")
    _trajectory(guided, show_source=True)
    print(f"\n  experiments: {g_sum['experiments']}   best: #{g_sum['best_id']} "
          f"{g_sum['best_score']:.4f}   stopped: {g_sum['stop_reason']}")

    d_best, g_best = d_sum["best_score"], g_sum["best_score"]
    delta = g_best - d_best
    print(f"\n{RULE}\nCOMPARISON\n{RULE}")
    print(f"Best {objective['primary_metric']}:")
    print(f"  deterministic: {d_best:.4f}  (#{d_sum['best_id']}, {d_sum['best_model']})")
    print(f"  LLM-guided:    {g_best:.4f}  (#{g_sum['best_id']}, {g_sum['best_model']})")
    print(f"Difference:      {delta:+.4f}")
    print(f"Experiments used: deterministic {d_sum['experiments']}, LLM-guided {g_sum['experiments']}")
    print(f"LLM novel configurations (fingerprint absent from the deterministic run): "
          f"{len(report['novel'])}")
    for e in report["novel"]:
        # Show preprocessing too: some of these differ from a ladder config only
        # in imputation, and the count would otherwise look padded.
        print(f"  #{e.id} {e.config.model} [{_params(e.config)}] prep[{_prep(e.config)}]")
    print(f"LLM rejections: {len(report['rejections'])}")
    for r in report["rejections"]:
        print(f"  {r['reason']}")

    if delta > TIE:
        verdict = (f"LLM-guided found the better model (+{delta:.4f} "
                   f"{objective['primary_metric']}) on this dataset and budget.")
    elif delta < -TIE:
        verdict = (f"Deterministic found the better model ({-delta:.4f} "
                   f"{objective['primary_metric']} ahead) on this dataset and budget.")
    else:
        verdict = "Tie: both strategies reached the same best score."
    print(f"\nResult:\n  {verdict}")
    print("  One dataset, one budget, one seed — indicative, not a benchmark.")
    print(BAR)


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare deterministic and LLM-guided ATLAS.")
    parser.add_argument("--csv", default="data/churn.csv")
    parser.add_argument("--target", default="churned")
    parser.add_argument("--budget", type=int, default=6)
    args = parser.parse_args()

    # One load(), one split, both strategies. This is the whole fairness mechanism.
    data = load(args.csv, args.target)
    report = run_comparison(data, Budget(max_experiments=args.budget))
    render(report, args.csv, args.target)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
