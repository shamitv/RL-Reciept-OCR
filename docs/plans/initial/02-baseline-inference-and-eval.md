# Plan: Baseline Inference And Evaluation

## Status

Current state: complete for Path A

- The repository now uses a heuristic-only baseline path.
- The CLI evaluates all tasks by default and supports seed, episode, format, and verbose options.
- README baseline numbers were regenerated from the current codebase.

## Goal

Create a truthful, reproducible baseline evaluation flow that can be run in one command and reported directly in the submission README.

## Problem Statement

The current `inference.py` exposes an `openai` mode but does not implement one. It also evaluates only a single task per run, which makes it awkward to produce the required all-task baseline summary.

## Desired End State

After this plan is complete:

- the default evaluation path runs all three tasks in a deterministic way
- output includes per-task and aggregate metrics
- agent mode names match what is actually implemented
- optional API-backed agent behavior is either real or explicitly removed
- baseline scores can be copied directly into the README

## Scope

### In scope

- CLI redesign for evaluation behavior
- per-task and aggregate reporting
- reproducibility controls such as seed and episode count
- structured output formats such as JSON or Markdown-ready summary
- optional OpenAI-backed action selection if justified

### Out of scope

- training PPO or BC policies
- leaderboard optimization
- rich experiment tracking infrastructure

## Decision Point

Before implementation, choose one of these two paths:

### Path A: Honest heuristic baseline only

Use the deterministic heuristic as the official baseline. Remove or rename any `openai` mode that is not genuinely implemented.

When to choose it:

- the goal is fast submission hardening
- the heuristic is acceptable as the required baseline
- there is not enough time to engineer a reliable LLM prompting loop

### Path B: Real optional OpenAI-backed agent mode

Keep the heuristic baseline, but also implement a real API-driven policy mode using the OpenAI client.

When to choose it:

- the hackathon team wants a stronger demonstration path
- prompt design and response parsing can be made reliable quickly

Recommendation:

Start with Path A to remove ambiguity. Add Path B only if it can be delivered without weakening determinism or clarity.

## Implementation Plan

### Step 1: Redesign the CLI around evaluation use cases

Tasks:

- define a default mode that runs all tasks
- allow single-task runs for debugging
- add flags for seed, episode count, and output format
- make the command naming reflect whether it is a one-off run or an evaluation sweep

Acceptance criteria:

- one command produces a complete baseline summary across `easy`, `medium`, and `hard`

### Step 2: Standardize result reporting

Tasks:

- report per-task final score, success rate, step count, and reward totals
- include an aggregate summary at the end
- optionally emit JSON for scripting and README generation
- print the exact seed and agent mode used

Acceptance criteria:

- the same run can be compared across commits without manual parsing

### Step 3: Clean up agent mode semantics

Tasks for Path A:

- remove `openai` mode or rename it to something truthful
- keep `heuristic` as the canonical baseline

Tasks for Path B:

- implement prompt construction from observation state
- constrain the action schema tightly
- validate model output before calling `env.step()`
- define clear fallback behavior for invalid responses

Acceptance criteria:

- users cannot accidentally invoke a mode that overclaims its behavior

### Step 4: Add evaluation helpers

Tasks:

- create a helper that runs one episode and returns structured metrics
- create a helper that runs a full task suite and aggregates results
- separate terminal logging from core evaluation logic so tests remain simple

Acceptance criteria:

- evaluation logic is testable without relying on CLI stdout parsing

### Step 5: Add tests for baseline behavior

Tasks:

- verify that all-task evaluation covers exactly three tasks by default
- verify deterministic results under a fixed seed for the heuristic path
- verify the CLI rejects invalid agent modes or missing API credentials cleanly
- if Path B is implemented, test response parsing and invalid output handling with mocks

Acceptance criteria:

- baseline reporting behavior is covered by unit or integration tests

### Step 6: Capture publishable baseline numbers

Tasks:

- run the baseline on the finalized environment
- capture per-task and aggregate results
- store these values in README and possibly a checked-in report artifact

Acceptance criteria:

- baseline scores in docs are generated from a reproducible command, not handwritten estimates

## Suggested Output Format

### Minimum console summary

- agent mode
- seed
- task name
- final score
- success flag
- steps used
- cumulative reward

### Recommended machine-readable artifact

- `artifacts/baseline_results.json` or similar

Fields:

- timestamp
- git commit if available
- agent mode
- seed
- per-task metrics
- aggregate mean score
- aggregate success rate

## File-Level Work Estimate

### Likely files to change

- `inference.py`
- possibly a new evaluation helper module under `training/` or `env/`
- README for baseline documentation
- tests for evaluation behavior

## Risks And Mitigations

### Risk: OpenAI mode becomes fragile

Mitigation:

- make it optional
- keep heuristic as the guaranteed reproducible path
- use strict schema validation for actions

### Risk: baseline numbers shift while environment changes

Mitigation:

- treat final score capture as a late-stage activity after environment semantics stabilize

### Risk: noisy console output hurts reproducibility

Mitigation:

- separate structured metrics from human-readable logs

## Exit Criteria

This plan is complete when one documented command produces a reproducible baseline summary across all three tasks and every advertised agent mode is implemented honestly.