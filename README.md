# ATLAS

**Autonomous Training, Learning & Analytics System**

ATLAS is an autonomous machine learning experimentation system. Given a tabular
dataset and an objective described in plain English, it profiles the data, runs
real experiments, measures actual results, diagnoses what happened, forms a
hypothesis about what to try next, and decides whether to continue, revise, or
stop — without a human selecting each subsequent experiment.

This is not a chatbot wrapper around a machine learning library. The system that
trains models, computes metrics, and terminates a mission is deterministic Python
code. Reasoning about what to try next may optionally use a large language model,
but that reasoning is validated before it is allowed to influence anything that
gets measured, ranked, or reported.

---

## Table of Contents

- [Why ATLAS](#why-atlas)
- [How It Works](#how-it-works)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [The LLM-Assisted Hypothesis Engine](#the-llm-assisted-hypothesis-engine)
- [Deterministic vs. LLM-Guided Comparison](#deterministic-vs-llm-guided-comparison)
- [Testing](#testing)
- [Current Scope and Limitations](#current-scope-and-limitations)
- [Results](#results)
- [Roadmap](#roadmap)

---

## Why ATLAS

Most of the effort in applied machine learning is not spent writing the code that
trains a single model. It is spent deciding what to do after seeing a result:
whether the model underfit or overfit, whether the data has a problem, whether a
different algorithm or configuration is worth trying, and when to stop. This
decision loop is usually manual, repeated by hand across many iterations.

ATLAS automates the loop itself, not just the training step. Two design
principles distinguish it from a general-purpose coding agent or a conventional
AutoML search:

- **Execution and reasoning are separated.** Python and scikit-learn are the
  sole authority on measured results. A reasoning component — deterministic
  rules, or optionally a language model — may propose what to try next, but it
  cannot fabricate a metric, force a stopping decision, or bypass validation.
- **Every step is evidence-linked.** A hypothesis must cite the specific
  diagnostic finding that motivated it. The system can explain, for any given
  experiment, exactly which prior result caused it to run.

---

## How It Works

```
USER TASK
    |
    v
PLANNER            interprets the objective into a structured mission
    |
    v
DATA ENGINEER      profiles the CSV; refuses to guess an unclear target
    |
    v
ML ENGINEER        selects the next model/configuration to try
    |
    v
EXPERIMENT ENGINE  actually trains the model and measures real metrics
    |
    v
DIAGNOSIS          classifies the result: execution error, data issue,
    |               model issue, insufficient improvement, or improvement
    v
HYPOTHESIS         proposes a change, grounded in the diagnosis
    |               (deterministic ladder, or optionally an LLM proposal
    |                behind a multi-stage validator)
    v
DECISION           continue, revise, or stop
    |
    v
NEXT EXPERIMENT    (loop back to ML ENGINEER, or terminate)
```

Termination is governed by deterministic constraints, not by a language model
deciding it is finished: a maximum experiment budget, a patience limit on
consecutive experiments without improvement, a maximum runtime, a maximum
number of consecutive execution failures, and an optional target score.

---

## Architecture

| Component | Responsibility |
|---|---|
| Planner | Converts a natural-language task into a structured mission: problem type, candidate target, primary metric, metric confidence, priority, and experiment budget. |
| Data Engineer | Profiles a raw CSV — shape, dtypes, missing values, duplicates, class balance — identifies the target, and declines to proceed when the target cannot be confidently established. |
| ML Engineer | Converts a hypothesis into a concrete, executable experiment configuration. |
| Experiment Engine | The sole trainer. Executes the model, computes metrics, and returns a structured result. Never raises past this boundary — failures are captured and reported. |
| Diagnostic Engine | Classifies a result into one of five categories: execution error, data issue, model issue, insufficient improvement, or improvement. |
| Hypothesis Engine | Proposes the next experiment from the diagnosis, objective, and history. Uses a deterministic rule ladder by default; can optionally consult a validated LLM proposal first. |
| Objective | Owns the selected metric and its optimization direction (maximize or minimize). Controls ranking, scoring, and final model selection — not just what is displayed. |
| Decision | Chooses exactly one of CONTINUE, REVISE, or STOP under deterministic budget and improvement constraints. |
| History | Records every experiment, its measured result, diagnosis, hypothesis, and decision. Prevents repeating a configuration that has already been tried. |

No component outside the Experiment Engine can produce a value that is treated
as a measured metric. An unavailable metric (for example, ROC-AUC on a
single-class validation split) is represented explicitly as unavailable and is
never defaulted to a numeric placeholder.

---

## Project Structure

```
atlas/
    planner.py          natural-language task -> structured mission plan
    data_engineer.py     CSV profiling, target detection, preprocessing
    ml_engineer.py        model construction and parameter application
    engine.py             experiment execution; the sole trainer
    diagnose.py            result classification
    hypothesis.py          next-experiment proposal (deterministic + LLM seam)
    llm.py                  LLM context construction and proposal validation
    objective.py            metric direction, scoring, and ranking
    decision.py             continue / revise / stop logic
    state.py                 core data structures and the experiment history

data/
    churn.csv, fraud.csv, houses.csv    demo datasets

tests/                       176+ tests, including adversarial and
                              mutation-tested coverage of the LLM validator

make_data.py                  regenerates the demo datasets deterministically
run_atlas.py                    command-line entry point
compare_strategies.py             deterministic vs. LLM-guided comparison
```

---

## Installation

Requires Python 3.13.

```bash
git clone <repository-url>
cd atlas
python -m venv .venv
source .venv/bin/activate        # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
python make_data.py               # generates the demo datasets
```

To enable the optional LLM-assisted hypothesis proposer:

```bash
pip install anthropic
export ANTHROPIC_API_KEY=your-key-here
```

ATLAS runs fully offline without this step. The deterministic hypothesis engine
requires no external service.

---

## Usage

Run a mission on a demo dataset:

```bash
python run_atlas.py --csv data/churn.csv --target churned
```

Describe the objective in natural language instead of specifying a metric
directly:

```bash
python run_atlas.py --csv data/fraud.csv --task \
  "Build a fraud detection model. Missing fraud is more costly than false alarms. Try at most 6 experiments."
```

Enable the LLM-assisted hypothesis proposer:

```bash
python run_atlas.py --csv data/churn.csv --target churned --llm
```

Run the deterministic vs. LLM-guided comparison:

```bash
python compare_strategies.py
```

This runs both strategies against the identical loaded dataset, split,
objective, and experiment budget, and prints a side-by-side trajectory and an
honest verdict — including cases where the deterministic strategy outperforms
the LLM-guided one.

---

## The LLM-Assisted Hypothesis Engine

When enabled, a language model may propose the next experiment. The proposal
crosses a single, narrow boundary: the model receives a JSON-serializable
context (objective, dataset profile, diagnosis, experiment history, and the
allowed model/parameter space) and returns a string. It receives no dataset,
no file path, no estimator, and no executable handle of any kind.

Every proposal passes through eight ordered validation gates before it can
become an experiment:

1. Valid JSON
2. Correct schema — required fields present, no unknown fields, correct types
3. A supported model
4. Known parameter names for that model
5. Valid parameter value types
6. Valid preprocessing configuration
7. Not a duplicate of an already-tried configuration
8. Cites real evidence from the diagnosis — an unsupported claim is rejected

A rejected proposal falls back to the deterministic hypothesis ladder
automatically. If the LLM is unavailable, times out, or fails for any reason,
the mission continues without it. The LLM is assistive; it is never required
for the system to function.

This is verified, not merely designed: an adversarial test suite submits
proposals that attempt to fabricate metrics, force a stop, invent provenance,
repeat a duplicate experiment, or cite evidence that does not exist, and
confirms each is rejected for the correct reason. Mutation testing —
deliberately bypassing the validator and confirming that the relevant tests
fail — was used to verify that this safety net actually works rather than
passing by coincidence.

---

## Deterministic vs. LLM-Guided Comparison

`compare_strategies.py` runs one controlled comparison: the same dataset, the
same train/test split, the same objective, and the same experiment budget,
once with the deterministic hypothesis ladder and once with an LLM-guided
proposer. The result is reported as measured, whichever strategy performs
better.

In the reference run on `churn.csv` (budget of six experiments, macro F1), the
deterministic strategy reached a best score of 0.6924 and the LLM-guided
strategy reached 0.6817 — the deterministic strategy ahead by 0.0108. The
LLM-guided run reached a configuration the deterministic ladder cannot express
(an early escalation to a gradient-boosted model at a specific learning rate
and iteration count) and was briefly ahead of the deterministic trajectory
before falling behind. Both outcomes are reported as they occurred; the
comparison exists to measure the effect of evidence-driven LLM guidance, not
to demonstrate that it wins.

---

## Testing

```bash
python -m pytest
```

The test suite includes unit and integration tests for every component,
adversarial tests against the LLM validator, and mutation testing at several
points in the project — deliberately reverting a fix or bypassing a
safeguard and confirming that the corresponding tests fail, then restoring
and confirming they pass again. This establishes that the tests depend on the
implementation being correct, rather than passing incidentally.

---

## Current Scope and Limitations

**Implemented and tested:**
Tabular supervised classification; CSV ingestion and profiling; natural-language
objective interpretation; autonomous experiment selection with duplicate
prevention; objective-aware, direction-aware scoring and ranking; deterministic
termination; an optional, validated LLM hypothesis proposer; a controlled
deterministic-vs-LLM comparison.

**Not implemented:**
Regression as a validated end-to-end path (the scoring architecture supports
optimization direction generically, but this has not been exercised against a
real regression model). Production deployment, monitoring, and continuous
learning. A trained or fine-tuned policy model — the LLM integration uses an
existing model as a proposal generator, not a model trained specifically for
this system. Persistent learning across missions — adaptation currently occurs
only within a single mission's experiment history.

**Supported models:** Logistic Regression, Random Forest, and
HistGradientBoosting, all via scikit-learn.

---

## Results

Across six self-generated experiments on the reference dataset, with no human
selecting any experiment after the first, macro F1 improved from 0.585 to
0.704. Each experiment was derived from the measured result of the one before
it: an experiment with a large train/test gap was followed by a more
regularized configuration; a plateaued score was followed by escalation to a
stronger model family.

---

## Roadmap

- Validated regression support
- Feature-engineering hypotheses under the same evidence and validation
  discipline already applied to model hypotheses
- Evaluation across multiple datasets, splits, and random seeds
- Accumulating experiment trajectories across missions to build a learned
  prior for hypothesis generation, rather than a fixed rule ladder
- A live dashboard exposing the experiment loop as it runs, with measured and
  reasoned information visually distinguished
