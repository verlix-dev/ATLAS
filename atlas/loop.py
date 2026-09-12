"""The autonomous loop: ML Engineer -> Engine -> Error Analyst -> Hypothesis -> Decision -> repeat.

Emits an event per stage. That stream is the whole UI contract for now.
"""

from __future__ import annotations

import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable, Iterator

from atlas import diagnose as analyst
from atlas import hypothesis as hypothesis_engine
from atlas import llm as llm_seam
from atlas import ml_engineer
from atlas import objective as objective_layer
from atlas.data_engineer import Dataset, load
from atlas.decision import Budget, decide
from atlas.engine import run
from atlas.planner import MissionPlan
from atlas.state import Action, Experiment, History

Event = dict[str, Any]
Sink = Callable[[Event], None]


class Mission:
    """One ATLAS run over one dataset."""

    def __init__(
        self,
        data: Dataset,
        *,
        budget: Budget | None = None,
        sink: Sink | None = None,
        plan: MissionPlan | None = None,
        llm: llm_seam.Llm | None = None,
    ) -> None:
        self.data = data
        self.budget = budget or Budget()
        self.plan = plan
        # A proposal/reasoning component only. It never reaches the engine, the
        # objective, the data or the decision -- see atlas/llm.py.
        self._llm = llm
        # One Objective per Mission: REVISE changes preprocessing and model
        # config, never what the run is optimising.
        self.objective = objective_layer.from_plan(plan)
        self.history = History(self.objective)
        self.events: list[Event] = []
        self._sink = sink
        # Downstream components (diagnose, propose) take the profile as a plain
        # dict, and so does the event payload. Convert once, here.
        self.profile = asdict(data.profile)
        # The plan is observability only: nothing in the loop reads it back.
        self._plan_payload = asdict(plan) if plan else None

    def _emit(self, stage: str, **payload: Any) -> None:
        event = {"seq": len(self.events), "t": time.time(), "stage": stage, **payload}
        self.events.append(event)
        if self._sink:
            self._sink(event)

    def run(self) -> Iterator[Experiment]:
        """Drive the loop to termination, yielding each completed experiment."""
        started = time.perf_counter()

        # The Data Engineer can block the mission before any experiment runs.
        # This is the only path that emits `data_engineer`, and it never emits
        # `mission_start`, so the two are mutually exclusive and `seq` is unaffected.
        if not self.data.profile.usable():
            self._emit(
                "data_engineer",
                status="needs_clarification",
                target=self.data.profile.target,
                reason=self.data.profile.target_reason,
                columns=self.data.profile.numeric + self.data.profile.categorical,
                profile=self.profile,
                plan=self._plan_payload,
            )
            self._emit(
                "mission_end",
                experiments=0,
                best_id=None,
                best_score=None,
                best_config=None,
                objective=self.objective.as_dict(),
                reason=f"data engineer could not confirm the target: {self.data.profile.target_reason}",
            )
            return

        self._emit(
            "mission_start",
            profile=self.profile,
            budget=asdict(self.budget),
            plan=self._plan_payload,
            objective=self.objective.as_dict(),
        )

        config = ml_engineer.baseline()
        origin = "baseline"

        while True:
            experiment_id = len(self.history) + 1
            self._emit("experiment_start", id=experiment_id, config=asdict(config), origin=origin)

            result = run(self.data, config, self.objective)
            self._emit(
                "experiment_result",
                id=experiment_id,
                ok=result.ok,
                metrics=result.metrics,
                error=result.error,
                seconds=round(result.seconds, 3),
            )

            diagnosis = analyst.diagnose(result, self.history, self.profile)
            self._emit("diagnosis", id=experiment_id, category=str(diagnosis.category), summary=diagnosis.summary, findings=diagnosis.findings)

            experiment = Experiment(
                id=experiment_id, config=config, result=result, diagnosis=diagnosis, origin=origin
            )
            self.history.add(experiment)

            rejections: list[dict[str, str]] = []
            proposal = hypothesis_engine.propose(
                diagnosis,
                self.history,
                self.profile,
                llm=self._llm,
                objective=self.objective,
                budget=self.budget,
                rejections=rejections,
            )
            if proposal:
                self._emit(
                    "hypothesis",
                    id=experiment_id,
                    problem=proposal.problem,
                    proposed_change=proposal.proposed_change,
                    reasoning=proposal.reasoning,
                    expected_effect=proposal.expected_effect,
                    next_config=asdict(proposal.config),
                    source=proposal.source,
                    cites=proposal.cites,
                    rejected=rejections,
                )
            else:
                self._emit(
                    "hypothesis",
                    id=experiment_id,
                    problem=diagnosis.summary,
                    proposed_change=None,
                    source=None,
                    cites=[],
                    rejected=rejections,
                )

            decision = decide(diagnosis, proposal, self.history, self.budget, started)
            experiment.hypothesis = proposal
            experiment.decision = decision
            self._emit("decision", id=experiment_id, action=str(decision.action), reason=decision.reason)

            yield experiment

            if decision.action is Action.STOP:
                break

            assert proposal is not None  # decide() STOPs when there is no proposal
            config = ml_engineer.apply(proposal)
            origin = f"exp{experiment_id}: [{proposal.source}] {proposal.proposed_change}"

        best = self.history.best()
        self._emit(
            "mission_end",
            experiments=len(self.history),
            best_id=best.id if best else None,
            best_score=self.history.best_score(),
            best_config=asdict(best.config) if best else None,
            objective=self.objective.as_dict(),
            reason=self.history.experiments[-1].decision.reason,
        )
        return

    def summary(self) -> dict[str, Any]:
        best = self.history.best()
        last = self.history.experiments[-1] if len(self.history) else None
        blocked = [e for e in self.events if e["stage"] == "data_engineer"]
        return {
            "experiments": len(self.history),
            "best_id": best.id if best else None,
            "best_score": self.history.best_score(),
            "best_model": best.config.model if best else None,
            "objective": self.objective.as_dict(),
            "stop_reason": (last.decision.reason if last and last.decision else None)
            or (blocked[0]["reason"] if blocked else None),
            "timeline": [
                {
                    "id": e.id,
                    "model": e.config.model,
                    "params": e.config.params,
                    "score": e.result.score(),
                    "primary_metric": e.result.primary_metric,
                    "category": str(e.diagnosis.category),
                    "decision": str(e.decision.action) if e.decision else None,
                    "origin": e.origin,
                }
                for e in self.history.experiments
            ],
        }


def mission(
    csv: str | Path,
    target: str | None = None,
    *,
    budget: Budget | None = None,
    sink: Sink | None = None,
    plan: MissionPlan | None = None,
    llm: llm_seam.Llm | None = None,
) -> Mission:
    """Run the Data Engineer over a CSV and return a ready-to-run Mission.

    `plan` is optional and observability-only. Callers using a Planner pass
    `target=plan.target_candidate` themselves — the target still reaches the
    Data Engineer through the same parameter a human uses.
    """
    return Mission(load(csv, target), budget=budget, sink=sink, plan=plan, llm=llm)
