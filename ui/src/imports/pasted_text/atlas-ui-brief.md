Design a premium, modern, visually striking web application UI for an autonomous machine-learning experimentation platform called ATLAS — Autonomous Training, Learning & Analytics System.

ATLAS is an autonomous ML experimentation system that behaves like an AI machine-learning engineer.

Its core loop is:

Experiment → Measure → Diagnose → Hypothesize → Decide → Improve → Experiment again

IMPORTANT:
The current UI is considered a failed first prototype. Do NOT simply restyle it. Redesign the visual experience from scratch while preserving the actual ATLAS product concept and backend behavior.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. VISUAL DIRECTION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Create a LIGHT-FIRST modern interface.

DO NOT use:
- black backgrounds
- near-black backgrounds
- dark-mode aesthetics
- military/command-center aesthetics
- CRT/terminal styling
- hacker aesthetics
- brutalist styling
- industrial control-panel styling
- excessive monospace typography

The UI should feel like a sophisticated, premium AI product.

Desired feeling:

Intelligent
Autonomous
Analytical
Trustworthy
Modern
Experimental
Premium
Visually exciting

Think:
- high-end AI product
- modern research platform
- sophisticated data visualization
- premium scientific software
- polished experimentation workspace

Use:
- white / off-white backgrounds
- very light gray surfaces
- subtle blue/lavender background tints
- gradients
- soft shadows
- rounded cards
- subtle glass effects
- blur
- gradient borders
- elegant hover states
- modern iconography
- smooth micro-interactions
- tasteful animations
- strong visual hierarchy

Gradients are explicitly allowed and encouraged when they improve the visual design.

Possible accent palette:
- electric blue
- indigo
- violet
- cyan
- emerald
- coral for warnings

Use gradients such as blue → indigo → violet tastefully.

Do not make everything glassmorphic or colorful. Maintain sophistication and readability.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
2. ATLAS PRODUCT IDENTITY
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

ATLAS is NOT just a dashboard showing ML metrics.

ATLAS autonomously:

1. Runs an experiment
2. Measures the result
3. Diagnoses what happened
4. Generates a hypothesis
5. Decides what to do next
6. Runs another experiment
7. Learns from the new evidence

The interface must visually communicate this autonomous reasoning loop.

The central storytelling principle is:

FACT → INTERPRETATION → ACTION

FACT:
What the ML system actually measured.

INTERPRETATION:
What ATLAS concluded from the measurements.

ACTION:
What ATLAS decided to do next.

The UI must make these distinctions visually obvious.

Example:

FACT

F1 Macro
0.6759

Recall
0.4000

Overfit Gap
+0.1690


INTERPRETATION

The model improved the primary metric, but the remaining train/test gap suggests excessive model capacity.


ACTION

CONTINUE

Constrain model depth and evaluate again.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
3. APPLICATION STRUCTURE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Create exactly THREE primary screens:

1. Mission Setup
2. Live Experiment Dashboard
3. Comparison / Results

They should feel like one continuous product workflow.

The transition should feel like:

Setup → ATLAS takes control → Autonomous experimentation → Results

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
4. SCREEN 1 — MISSION SETUP
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Create a visually impressive setup experience.

The user should be able to provide:

DATASET

Include:

Upload CSV

and:

Choose existing dataset

The user must NOT be limited to predefined datasets.

Create a polished drag-and-drop upload area.

Example:

Drop your dataset here

CSV files supported
or Browse files

After upload, display:

- filename
- number of rows
- number of columns
- detected target
- missing values
- class distribution
- dataset status

Make this feel like a real AI workspace, not a file picker.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
5. NATURAL-LANGUAGE TASK
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Create a large, prominent natural-language input.

Label:

What do you want ATLAS to accomplish?

Example:

"Predict whether a customer will churn. Prioritize catching potential churners while maintaining reasonable precision."

This is important because ATLAS uses the task description to create its mission plan.

The user should be able to describe the ML objective naturally rather than configuring everything manually.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
6. MISSION CONFIGURATION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Allow the user to configure:

Target:
- Auto-detect
- Select column

Maximum experiments:
- numeric input / selector

AI-guided experimentation:
- ON / OFF

Optional advanced configuration may be collapsed.

Do NOT imply that every mission must execute exactly six experiments.

The experiment budget represents a MAXIMUM.

ATLAS can stop earlier based on its decision logic.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
7. LIVE PLANNER PREVIEW
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

While the user configures the mission, show a beautiful preview:

ATLAS UNDERSTANDS

Problem
Binary Classification

Target
churned

Primary Metric
F1 Macro

Priority
Minimize false negatives

Experiment Budget
6

Make this look like ATLAS intelligently interpreted the user's request.

Do NOT display raw JSON.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
8. START MISSION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Create a prominent primary CTA:

START MISSION →

Use a beautiful modern gradient.

The transition from setup to dashboard should feel like ATLAS taking control.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
9. SCREEN 2 — LIVE EXPERIMENT DASHBOARD
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

This is the most important screen.

Make it visually exceptional.

Use a three-column layout:

LEFT:
Mission Context

CENTER:
Current Experiment + Experiment Timeline

RIGHT:
Score Progression + Best Result

The center and right sections should receive the most visual attention.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
10. HERO FEATURE — EXPERIMENT SCORE GRAPH
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

This is one of the MOST IMPORTANT elements in the entire application.

Create a large, beautiful score progression graph.

The graph must show EVERY experiment ATLAS has actually executed.

Example:

Primary Metric — F1 Macro

E1   0.5845
E2   0.6044
E3   0.6531
E4   0.6717
E5   0.6587
E6   0.6924

Visualize this as a polished line chart.

Each experiment is a point.

The graph must communicate:

- improvement
- regression
- plateau
- current experiment
- best experiment
- overall trajectory

Do NOT only show the final score.

The entire experimentation journey is important.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
11. SCORE GRAPH INTERACTION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

When hovering an experiment point, show a beautiful tooltip containing:

Experiment 04

HistGradientBoosting

F1 Macro
0.6817

Δ Best
+0.0058

Model parameters

learning_rate: 0.04
iterations: 300

Clearly identify whether this experiment became the best result.

Visually distinguish:

Improvement
Current experiment
Best experiment
Regression

The graph should feel like a major product feature, not a small dashboard chart.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
12. CURRENT EXPERIMENT — HERO CARD
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Create a large current experiment card.

Example:

EXPERIMENT 04

HistGradientBoosting

learning_rate 0.04
300 iterations
leaf 15


MEASURED

F1 Macro

0.6759

Recall
0.4000

Precision
0.XXXX

Overfit Gap
+0.1690


DIAGNOSIS

The model improved substantially over the previous best, but the train/test gap remains significant.


HYPOTHESIS

Reducing model complexity may improve generalization.

Evidence:

[ overfit gap ]
[ weak recall ]


DECISION

CONTINUE

Test a lower model depth.

The card should feel like the receipt of what ATLAS actually did.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
13. EXPERIMENT TIMELINE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Create a beautiful vertical experimentation timeline.

Example:

EXPERIMENT 01
Logistic Regression
0.5845

↓

EXPERIMENT 02
Balanced Logistic Regression
0.6044

↓

EXPERIMENT 03
Random Forest
0.6531

↓

EXPERIMENT 04
Random Forest + constraints
0.6717

↓

EXPERIMENT 05
HistGradientBoosting
0.6587

↓

EXPERIMENT 06
HistGradientBoosting + depth constraint
0.6924

Older experiments may collapse into compact cards.

The current experiment should receive substantially more visual space.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
14. LEFT SIDEBAR — MISSION CONTEXT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Keep the left rail visually quieter.

Display:

MISSION

Customer Churn Prediction

Dataset
churn.csv

Rows
400

Target
churned

Task
Binary Classification

Primary Metric
F1 Macro

Priority
Minimize false negatives

Budget

3 of ≤6 experiments

IMPORTANT:

The experiment count must be dynamic.

Never hardcode "6 experiments".

The budget is a maximum.

ATLAS may stop before the maximum if:
- the objective is reached
- improvement becomes insufficient
- execution fails
- another deterministic stopping condition is reached

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
15. AI PROVENANCE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Clearly communicate whether each hypothesis came from:

ATLAS RULES

or:

AI GUIDED

Do not hide the provenance.

For AI-generated hypotheses, display:

Why this experiment?

Then show actual evidence citations.

Example:

Cited evidence:

[ Overfit gap ]
[ Weak minority recall ]

Citations must correspond to actual diagnostic findings.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
16. RIGHT SIDEBAR — BEST SO FAR
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Create a visually strong BEST RESULT section.

Example:

BEST RESULT

Experiment 06

HistGradientBoosting

0.6924

F1 Macro

Previous Best
0.6817

Improvement
+0.0107

Show the winning configuration in a compact form.

The best result should visually stand out.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
17. MISSION STATUS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

At the top of the dashboard create a polished status indicator.

Possible states:

ATLAS IS EXPERIMENTING
● Running Experiment 04


MISSION COMPLETE
✓ Objective reached


MISSION COMPLETE
6 experiments evaluated

Use subtle motion to make the active state feel alive.

Do not over-animate.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
18. AUTONOMOUS LOOP INDICATOR
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Do NOT create a giant circular AI-agent diagram.

Instead use a compact process indicator:

Experiment
→
Measure
→
Diagnose
→
Hypothesize
→
Decide

Highlight the current stage.

This indicator complements the experiment cards.

It should NOT replace them.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
19. SCREEN 3 — COMPARISON
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Create a polished comparison screen.

Compare:

Deterministic ATLAS

vs.

AI-Guided ATLAS

The comparison should be immediately understandable.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
20. COMPARISON HERO
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Create a strong verdict-first section.

Example:

WHICH STRATEGY LEARNED BETTER?

DETERMINISTIC

0.6924

AI GUIDED

0.6817

Δ −0.0107

Do not hide an unfavorable result.

ATLAS must communicate results honestly.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
21. COMPARISON TRAJECTORY GRAPH
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Create a large line chart.

X-axis:

Experiment 1
Experiment 2
Experiment 3
Experiment 4
Experiment 5
Experiment 6

Y-axis:

Primary Metric

Plot both:

Deterministic

AI Guided

The graph should clearly show how the two strategies navigated the experimentation space differently.

The purpose is not simply benchmarking.

It demonstrates how different reasoning strategies behave.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
22. COMPARISON SUMMARY
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Show useful information such as:

Best Score
Experiment Count
Improvement Trajectory
Novel Configurations
Rejected Proposals
Final Stopping Reason

Avoid meaningless metrics.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
23. TYPOGRAPHY
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Use a modern sans-serif as the primary typeface.

Good choices:

Inter
Geist
Manrope
Plus Jakarta Sans

Use monospace ONLY for:

- experiment IDs
- model parameters
- dataset column names
- technical configuration
- technical metadata

Do NOT make the entire interface monospace.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
24. CARD DESIGN
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Cards should use:

- 16–24px border radius
- generous whitespace
- subtle borders
- soft shadows
- layered depth
- strong typography hierarchy

Not every piece of information needs to be inside a card.

Use whitespace to create hierarchy.

The interface should breathe.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
25. DATA VISUALIZATION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Data visualization is a major part of ATLAS.

Prioritize:

1. Experiment Score Progression
2. Deterministic vs AI-Guided Trajectory
3. Improvement / Regression
4. Best-So-Far Progression

Charts should be:
- clean
- beautiful
- interactive
- presentation-ready
- easy to understand

Avoid decorative charts that do not communicate meaningful information.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
26. ANIMATION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Use subtle modern motion.

Examples:

- experiment card entering timeline
- new graph point appearing
- score line extending
- current experiment status pulse
- score counter animation
- graph drawing itself
- setup → dashboard transition
- card expansion
- hover elevation

Motion should communicate that ATLAS is actively working.

Avoid:
- excessive bouncing
- excessive spinning
- distracting effects
- gimmicky animations

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
27. BACKEND CONNECTIVITY — HARD CONSTRAINT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

This is a REAL backend-connected product, not a static visual concept.

The UI must preserve compatibility with these APIs:

GET  /api/datasets

POST /api/plan

POST /api/missions

GET  /api/missions/{id}/events

GET  /api/missions/{id}/summary

POST /api/compare

Do not invent new backend behavior unless explicitly required.

Do not hardcode experiment results into the production interface.

Do not fabricate metrics.

Do not fabricate AI reasoning.

The design must be implementable with the existing backend.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
28. REAL-TIME SSE EVENTS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

The dashboard receives real-time events through Server-Sent Events.

Relevant event types:

mission_start

experiment_start

experiment_result

diagnosis

hypothesis

decision

mission_end

The interface should progressively update as these events arrive.

Visual sequence:

Experiment starts
↓
Configuration appears
↓
Measured metrics appear
↓
Diagnosis appears
↓
Hypothesis appears
↓
Decision appears
↓
Next experiment begins
↓
Score graph receives a new point

This progressive experience is a CORE part of ATLAS.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
29. DATA HONESTY
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

ATLAS is an evidence-driven ML system.

The UI must NEVER imply information that the backend did not actually establish.

If a metric is unavailable:

N/A

If there is no score:

No score

If an experiment failed:

Execution failed

If a proposal was rejected:

Rejected

Never convert missing information into fake zeros.

Never fabricate AI reasoning.

Never fabricate experiment results.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
30. IMPORTANT PRODUCT STORY
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

The UI should communicate this sentence:

"ATLAS doesn't just train a model. It runs experiments, learns from their results, and decides what to try next."

A user should immediately be able to answer:

What did ATLAS try?

↓

What happened?

↓

What did ATLAS learn?

↓

What did ATLAS decide?

↓

Did the next experiment improve?

The SCORE PROGRESSION GRAPH is the visual backbone connecting these questions.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
31. TECHNICAL PRODUCT FLOW
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

The actual product flow is:

USER TASK
↓
ATLAS PLANNER
↓
DATA ENGINEER
↓
ML ENGINEER
↓
EXPERIMENT ENGINE
↓
ERROR / DIAGNOSTIC ENGINE
↓
HYPOTHESIS ENGINE
↓
DECISION
↓
CONTINUE / REVISE / STOP
↓
NEW EXPERIMENT

The UI should visually represent this intelligence without turning it into a complicated architecture diagram.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
32. IMPLEMENTATION STYLE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

The eventual frontend is Vanilla HTML/CSS/JavaScript.

No React requirement.

No build-step dependency should be assumed.

The Figma design should therefore be practical to reproduce with:

HTML
CSS
JavaScript

Design components that can realistically be implemented.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
33. RESPONSIVE DESIGN
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Design for desktop first because the main demo will be desktop.

Also provide sensible tablet/mobile behavior.

On smaller screens:

Mission Context
↓
Score Graph
↓
Current Experiment
↓
Timeline
↓
Best Result

Do not simply shrink the desktop layout.

Reflow the information hierarchy intelligently.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
34. FINAL DESIGN QUALITY BAR
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

The final design should look good enough for:

- Hackathon final demo
- AI/ML portfolio
- investor/product presentation
- university technical presentation
- GitHub showcase

It must NOT look like:

- Streamlit dashboard
- generic SaaS admin panel
- generic AI landing page
- dark developer tool
- terminal
- military command center
- collection of random cards
- basic CRUD application

It should look like a REAL premium AI experimentation product.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
35. MOST IMPORTANT PRIORITIES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

PRIORITY 1
Beautiful modern LIGHT interface.

PRIORITY 2
Large, visually compelling experiment SCORE PROGRESSION GRAPH showing every experiment.

PRIORITY 3
Beautiful current-experiment presentation.

PRIORITY 4
Clear FACT → INTERPRETATION → ACTION hierarchy.

PRIORITY 5
Experiment timeline.

PRIORITY 6
Best-so-far visualization.

PRIORITY 7
AI provenance and evidence citations.

PRIORITY 8
Excellent natural-language mission setup.

PRIORITY 9
Beautiful Deterministic vs AI-Guided comparison.

PRIORITY 10
Real backend compatibility and data honesty.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
FINAL INSTRUCTION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Do NOT preserve the existing visual design.

Treat the existing UI as a failed first prototype.

Redesign the visual language from scratch.

Be creatively ambitious with:
- gradients
- colors
- typography
- glass effects
- shadows
- charts
- animations
- depth
- layout
- visual hierarchy

But preserve the actual ATLAS product concept and backend behavior.

The goal is NOT to make the current UI slightly prettier.

The goal is to create an interface where someone sees ATLAS for the first time and immediately thinks:

"This is a serious, polished autonomous AI/ML system."

The score progression graph should be one of the defining visual elements of the entire product.