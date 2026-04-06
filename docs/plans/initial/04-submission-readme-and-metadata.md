# Plan: Submission README And Metadata Hardening

## Status

Current state: complete

- The top-level README was rewritten as a submission-facing document.
- `openenv.yaml` and package metadata were tightened to match current behavior.
- Scaffold-era language was removed from submission-facing metadata.

## Goal

Rewrite the repository documentation and metadata so the project reads like a finished hackathon submission rather than a scaffold.

## Problem Statement

The current README is useful for developers, but it openly describes the project as scaffolding. That candor is accurate, yet it weakens the submission and leaves several rubric-required topics under-documented.

## Desired End State

After this plan is complete:

- the README matches the judging rubric directly
- the environment description reflects implemented behavior, not intended future work
- baseline numbers are documented and reproducible
- `openenv.yaml` is tightened to match the final runtime contract
- no stale scaffold language remains in user-facing docs

## Scope

### In scope

- top-level README rewrite
- `openenv.yaml` review and hardening
- package and metadata wording cleanup where relevant
- documenting validation and deployment commands

### Out of scope

- internal architecture deep dives beyond what supports evaluation
- long-form design essays better suited for `docs/hackathon/`

## README Target Structure

### 1. Project overview

Explain:

- what the environment simulates
- why receipt extraction is a real-world sequential decision task
- what the agent is expected to do

### 2. Environment definition

Explain:

- observation structure
- action space
- terminal conditions
- reward design summary

### 3. Task definitions

Explain:

- easy, medium, hard objectives
- what changes across tasks
- why the progression is meaningful

### 4. Dataset and assumptions

Explain:

- data source
- preparation assumptions
- any limitations or preprocessing choices

### 5. Setup and local usage

Explain:

- environment setup
- running tests
- running the server
- running baseline evaluation

### 6. Baseline results

Explain:

- evaluation command
- seed and evaluation conditions
- per-task scores and aggregate summary

### 7. Deployment and validation

Explain:

- Docker build and run commands
- `openenv validate` usage
- Hugging Face Space notes if applicable

## Metadata Hardening Plan

### Step 1: Audit current claims

Tasks:

- remove language that still calls the project a scaffold unless that remains intentionally true
- ensure README, `openenv.yaml`, and package description agree on project status and scope

Acceptance criteria:

- there are no contradictory statements across top-level docs and metadata

### Step 2: Rewrite README from rubric requirements backward

Tasks:

- map each hackathon documentation requirement to a specific README section
- use concrete examples from the implemented action and observation models
- keep the README concise but complete enough for judges to evaluate quickly

Acceptance criteria:

- every required README topic from the brief has a visible section

### Step 3: Document baseline results from a real run

Tasks:

- run the finalized baseline command
- capture per-task and aggregate numbers
- include enough context that another evaluator can reproduce them

Acceptance criteria:

- baseline scores in docs are traceable to a command and seed

### Step 4: Tighten `openenv.yaml`

Tasks:

- verify metadata fields are current and sufficient
- ensure task listings match implemented tasks
- ensure entrypoint and runtime details match the deployed server behavior

Acceptance criteria:

- metadata does not describe deprecated behavior or scaffolding

### Step 5: Add a short limitations section only if necessary

Tasks:

- document genuine remaining limitations without undermining the submission
- avoid TODO language in the main narrative

Acceptance criteria:

- any limitation notes are factual, bounded, and do not advertise missing core requirements

## File-Level Work Estimate

### Likely files to change

- `README.md`
- `openenv.yaml`
- `pyproject.toml` description if needed
- possibly `docs/hackathon/` references if they conflict with the final README

## Risks And Mitigations

### Risk: docs overclaim behavior

Mitigation:

- update docs only after the corresponding code path is verified
- prefer exact descriptions over marketing language

### Risk: README becomes too long and hard to scan

Mitigation:

- optimize for judges reading quickly
- keep implementation detail in supporting docs, not in the main README

### Risk: metadata drifts again after last-minute code changes

Mitigation:

- schedule a final docs pass immediately before validation

## Exit Criteria

This plan is complete when the README and metadata describe the repository as it actually behaves today, satisfy the brief directly, and can be read as a finished submission artifact.