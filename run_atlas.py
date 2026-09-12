"""CLI entry point. Streams the mission to the terminal as it happens."""

from __future__ import annotations

import argparse
import json
import sys
from typing import Callable

from atlas import llm, planner
from atlas.decision import Budget
from atlas.loop import mission
from atlas.objective import UNAVAILABLE
from atlas.planner import MissionPlan

BAR = "-" * 72
DEFAULT_LLM_MODEL = "claude-opus-5"


def anthropic_proposer(model: str, timeout: float = 120.0) -> Callable[[dict], str]:
    """One concrete client for --llm. Lives here, outside the atlas package.

    The core stays provider-independent: atlas.hypothesis knows only
    Callable[[dict], str], so swapping providers means replacing this function
    and nothing else.

    The raw text is handed back unparsed on purpose — atlas.llm.validate() owns
    the proposal schema, and declaring it a second time as a structured-output
    JSON schema here would be two sources of truth for the same shape.
    """
    try:
        import anthropic
    except ModuleNotFoundError as exc:  # pragma: no cover - environment-dependent
        raise SystemExit("--llm requires the anthropic package: pip install anthropic") from exc

    client = anthropic.Anthropic(timeout=timeout)

    def propose(payload: dict) -> str:
        message = client.messages.create(
            model=model,
            max_tokens=16000,
            thinking={"type": "adaptive"},
            system=llm.SYSTEM_PROMPT,
            messages=[{"role": "user", "content": json.dumps(payload, indent=2)}],
        )
        return "".join(block.text for block in message.content if block.type == "text")

    return propose


def show_plan(plan: MissionPlan) -> None:
    print(f"{BAR}\nATLAS PLANNER\n{BAR}")
    print(f"  objective:   {plan.objective}")
    print(f"  problem:     {plan.problem_type or 'not stated'}")
    print(f"  target hint: {plan.target_candidate or 'none - Data Engineer will infer'}")
    print(f"  metric:      {plan.primary_metric} ({plan.metric_confidence}) - {plan.metric_reason}")
    if plan.priority:
        print(f"  priority:    {plan.priority}")
    if plan.constraints:
        print(f"  constraints: {plan.constraints}")
    if plan.experiment_budget:
        print(f"  budget:      {plan.experiment_budget} experiments")


def _metric_text(metrics: dict, objective: dict) -> str:
    """Primary metric first, then the reported secondaries, unavailable marked."""
    def fmt(name: str) -> str:
        value = metrics.get(name)
        if value is None or value == UNAVAILABLE:
            return f"{name}=n/a"
        return f"{name}={value:.4f}"

    primary = objective["primary_metric"]
    return f"*{fmt(primary)} " + " ".join(fmt(m) for m in objective["secondary_metrics"])


def printer() -> Callable[[dict], None]:
    """Build the CLI sink for one mission.

    It remembers the objective announced at mission_start only so each result
    line can mark which metric is primary. It never ranks anything: best_id
    and best_score are read straight off the backend's mission_end event.
    """
    objective: dict = {}

    def show(event: dict) -> None:
        stage = event["stage"]
        if stage == "mission_start":
            objective.update(event["objective"])
            p = event["profile"]
            print(f"{BAR}\nATLAS MISSION\n{BAR}")
            print(f"DATA ENGINEER  target={p['target']!r} ({p['target_confidence']}) - {p['target_reason']}")
            print(f"  {p['problem_type']}: rows={p['rows']} features={p['columns']} classes={p['class_counts']} imbalance={p['imbalance_ratio']:.2f}:1")
            print(f"  numeric={p['numeric']}")
            print(f"  categorical={p['categorical']}")
            print(f"  missing={p['missing_per_column'] or 'none'} duplicates={p['duplicate_rows']} constant={p['constant_columns'] or 'none'}")
            o = event["objective"]
            print(f"OBJECTIVE      {o['primary_metric']} ({o['direction']}); also reporting {', '.join(o['secondary_metrics'])}")
            print(f"budget={event['budget']}")
        elif stage == "data_engineer":
            print(f"{BAR}\nATLAS MISSION BLOCKED\n{BAR}")
            print(f"DATA ENGINEER  {event['status']}: {event['reason']}")
            print(f"  best guess was {event['target']!r}; columns available: {event['columns']}")
            print("  re-run with --target <column> to proceed.")
        elif stage == "experiment_start":
            print(f"\n{BAR}\nEXPERIMENT {event['id']}  [{event['origin']}]")
            print(f"  config: {event['config']['model']} params={event['config']['params']} prep={event['config']['preprocess']}")
        elif stage == "experiment_result":
            if event["ok"]:
                m = event["metrics"]
                print(f"  RESULT   {_metric_text(m, objective)}  ({event['seconds']}s)")
                print(f"           gap={m['overfit_gap']:+.4f} recall_per_class={ {k: round(v, 3) for k, v in m['per_class_recall'].items()} }")
            else:
                print(f"  FAILED   {event['error']}")
        elif stage == "diagnosis":
            print(f"  DIAGNOSIS[{event['category']}] {event['summary']}")
            for finding in event["findings"]:
                print(f"           - {finding}")
        elif stage == "hypothesis":
            for rejection in event.get("rejected") or []:
                print(f"  REJECTED [{rejection['source']}] {rejection['reason']}")
            if event["proposed_change"]:
                print(f"  HYPOTHESIS [{event.get('source', 'deterministic')}] {event['proposed_change']}")
                print(f"           why: {event['reasoning']}")
                print(f"           expect: {event['expected_effect']}")
                if event.get("cites"):
                    print(f"           cites: {event['cites']}")
            else:
                print("  HYPOTHESIS none - out of untried ideas")
        elif stage == "decision":
            print(f"  DECISION [{event['action'].upper()}] {event['reason']}")
        elif stage == "mission_end":
            metric = event["objective"]["primary_metric"]
            print(f"\n{BAR}\nFINAL\n{BAR}")
            if event["best_id"] is None:
                # No rankable experiment: blocked mission, every run failed, or
                # the objective metric was unavailable throughout.
                print(f"experiments={event['experiments']} best=none (nothing scorable on {metric})")
            else:
                print(f"experiments={event['experiments']} best=experiment {event['best_id']} {metric}={event['best_score']:.4f}")
                print(f"model={event['best_config']}")
            print(f"stopped because: {event['reason']}")

    return show


def main() -> int:
    parser = argparse.ArgumentParser(description="Run an ATLAS mission on a tabular CSV.")
    parser.add_argument("--csv", default="data/churn.csv")
    parser.add_argument("--task", default=None, help="natural-language objective; the Planner reads it")
    parser.add_argument("--target", default=None, help="target column; overrides the Planner's candidate")
    parser.add_argument("--max-experiments", type=int, default=None, help="overrides the Planner's budget")
    parser.add_argument("--target-score", type=float, default=0.95)
    parser.add_argument(
        "--llm",
        metavar="MODEL",
        nargs="?",
        const=DEFAULT_LLM_MODEL,
        default=None,
        help=f"propose hypotheses with this Claude model (default {DEFAULT_LLM_MODEL}); "
        "proposals are still validated and can be rejected",
    )
    args = parser.parse_args()

    # Planner runs once, here, before anything else. Its output is a hint:
    # explicit CLI flags always win over what it derived from the task text.
    mission_plan = None
    if args.task:
        try:
            mission_plan = planner.plan(args.task)
        except ValueError as exc:
            print(f"cannot plan mission: {exc}", file=sys.stderr)
            return 2
        show_plan(mission_plan)

    target = args.target or (mission_plan.target_candidate if mission_plan else None)
    max_experiments = args.max_experiments or (mission_plan.experiment_budget if mission_plan else None) or 6

    budget = Budget(max_experiments=max_experiments, target_score=args.target_score)
    proposer = anthropic_proposer(args.llm) if args.llm else None
    try:
        run = mission(args.csv, target, budget=budget, sink=printer(), plan=mission_plan, llm=proposer)
    except (FileNotFoundError, ValueError) as exc:
        print(f"cannot start mission: {exc}", file=sys.stderr)
        return 2

    for _ in run.run():
        pass

    print(f"\ntimeline: {run.summary()['timeline']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
