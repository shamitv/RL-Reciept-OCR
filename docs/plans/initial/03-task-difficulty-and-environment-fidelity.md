# Plan: Task Difficulty And Environment Fidelity

## Status

Current state: complete for the current environment design

- Task-specific visibility rules are enforced.
- Candidate ranking noise is deterministic and task-aware.
- Task behavior differences are covered by tests.

## Goal

Turn the current easy, medium, and hard task presets into genuinely different environment conditions that reward better policies and make the task progression defensible to judges.

## Problem Statement

The repository already defines task configuration values such as `visible_windows`, `corruption_level`, and `ranking_noise`, but most of those values do not currently alter the agent experience at runtime. As a result, the difficulty ladder is weaker than the design intends.

## Desired End State

After this plan is complete:

- each difficulty changes the observable or decision-relevant environment behavior
- the differences are deterministic under a seed
- the task descriptions in code and docs match what the environment actually does
- graders remain stable and interpretable across tasks

## Design Constraints

- preserve reproducibility
- do not leak hidden gold labels into observation state
- do not introduce randomness that cannot be replayed from seed and sample identity
- keep the action space stable unless a strong reason emerges to change it

## Candidate Difficulty Levers

### Lever 1: Visibility restrictions

Examples:

- easy reveals more useful regions with broad actions
- medium requires more targeted windowing or neighbor inspection
- hard starts with fewer visible clues and weaker reveal actions

### Lever 2: Candidate ranking quality

Examples:

- easy returns high-quality candidate ordering
- medium inserts plausible distractors near the top
- hard weakens ranking confidence and requires validation or manual correction

### Lever 3: OCR corruption handling

Examples:

- easy uses relatively clean normalized OCR
- medium preserves more noisy formatting and ambiguous date strings
- hard includes stronger corruption or split-field ambiguity

### Lever 4: Budget pressure

Examples:

- easy allows redundant steps
- hard punishes inefficient exploration through tight action budgets

Recommendation:

Use at least three levers, not only budget. Budget alone is not enough to defend the progression.

## Implementation Plan

### Step 1: Define explicit task semantics

Tasks:

- write short, code-adjacent definitions of what makes each task harder
- map each task to concrete runtime behaviors
- remove any config field that does not have a planned runtime effect

Acceptance criteria:

- every difficulty-related config field has a defined behavioral consequence

### Step 2: Enforce visibility rules in the environment

Tasks:

- tie `visible_windows` and related rules to what `list_text_regions` can reveal
- decide whether `view_receipt` changes per task
- ensure hidden regions remain hidden until allowed by a valid action

Acceptance criteria:

- the same action sequence yields meaningfully different observations across tasks when intended

### Step 3: Apply deterministic candidate noise or ambiguity

Tasks:

- modify candidate retrieval to use task-specific ranking behavior
- introduce seeded distractors or reranking noise where justified
- preserve the ability to inspect evidence behind each candidate

Acceptance criteria:

- candidate lists differ by difficulty in a controlled, testable way

### Step 4: Clarify normalization and corruption policy

Tasks:

- decide whether corruption is applied at dataset load time or observation time
- ensure corruption does not alter hidden gold labels
- document which transformations are visible to the agent and which are only for grading

Acceptance criteria:

- corruption logic is deterministic and traceable

### Step 5: Align rewards with harder tasks

Tasks:

- verify that the shaped reward still gives informative partial progress under noisier tasks
- adjust penalties or bonuses only if task changes break reward usefulness
- avoid overfitting reward logic to one heuristic policy

Acceptance criteria:

- the reward remains bounded, interpretable, and useful across all tasks

### Step 6: Add difficulty-focused tests

Tasks:

- add tests that compare task observations under the same seed
- add tests that confirm visibility limits are enforced
- add tests that confirm ranking noise is deterministic
- add tests for budget exhaustion behavior under each task

Acceptance criteria:

- task differences are verified by tests, not only by documentation

## Suggested Runtime Differences By Task

### Easy

- broad reveals succeed
- top candidate quality is high
- normalization is forgiving
- budget is generous

### Medium

- some useful fields require window targeting or neighbor inspection
- candidate lists include plausible distractors
- date and address ambiguity increases
- budget is moderate

### Hard

- default reveals expose less complete evidence
- top-ranked candidates are less reliable
- more fields require validation or manual composition
- budget is tight

## File-Level Work Estimate

### Likely files to change

- `env/tasks.py`
- `env/environment.py`
- `env/candidate_retrieval.py`
- possibly `env/rewards.py`
- tests covering tasks and environment behavior

## Risks And Mitigations

### Risk: difficulty becomes arbitrary rather than realistic

Mitigation:

- ground each change in plausible receipt extraction failure modes
- prefer ambiguity based on OCR structure over synthetic chaos

### Risk: deterministic tests become brittle

Mitigation:

- centralize seeded perturbation logic
- avoid hidden dependence on dictionary ordering or filesystem order

### Risk: heuristic baseline performance collapses completely

Mitigation:

- accept some drop in score, but keep the environment solvable
- use baseline results to calibrate task severity after implementation

## Exit Criteria

This plan is complete when easy, medium, and hard can be defended as meaningfully different tasks based on observable runtime behavior, not just configuration labels.