# Initial Plan Index

This directory contains the initial execution plans for closing the gaps identified in the hackathon solution review.

## Purpose

The repository already has a usable environment scaffold. These plans focus on the work required to turn that scaffold into a submission-ready OpenEnv project.

## Tracking

Use [CHECKLIST.md](CHECKLIST.md) as the master progress tracker across all plan documents.

## Current Status

- `00-roadmap`: complete for the current implementation cycle
- `01-real-dataset-integration`: complete
- `02-baseline-inference-and-eval`: complete for the heuristic baseline path
- `03-task-difficulty-and-environment-fidelity`: complete for the current task model
- `04-submission-readme-and-metadata`: complete
- `05-validation-and-deployment`: partially complete

Remaining gap:

- Docker build and container startup have not been verified on this machine because the `docker` CLI is unavailable.

## Planning Principles

- Close pass-fail submission risks before optimizing polish.
- Prefer changes that improve both hackathon scoring and long-term maintainability.
- Keep the environment deterministic where the brief expects reproducible behavior.
- Make each completed phase leave behind tests, documentation, and measurable acceptance criteria.

## Recommended Execution Order

1. [00-roadmap.md](00-roadmap.md)
2. [01-real-dataset-integration.md](01-real-dataset-integration.md)
3. [03-task-difficulty-and-environment-fidelity.md](03-task-difficulty-and-environment-fidelity.md)
4. [02-baseline-inference-and-eval.md](02-baseline-inference-and-eval.md)
5. [04-submission-readme-and-metadata.md](04-submission-readme-and-metadata.md)
6. [05-validation-and-deployment.md](05-validation-and-deployment.md)

## Plan Set Overview

### [00-roadmap.md](00-roadmap.md)

Top-level execution roadmap. Defines delivery sequence, milestones, dependencies, and exit criteria across the full gap-closure effort.

### [01-real-dataset-integration.md](01-real-dataset-integration.md)

Detailed plan for replacing the embedded mock samples with a prepared receipt dataset flow, including schema normalization, split handling, deterministic sampling, and tests.

### [02-baseline-inference-and-eval.md](02-baseline-inference-and-eval.md)

Detailed plan for making the baseline path honest and reproducible. Covers heuristic baseline evaluation, all-task reporting, optional OpenAI-backed policy mode, and metrics output.

### [03-task-difficulty-and-environment-fidelity.md](03-task-difficulty-and-environment-fidelity.md)

Detailed plan for making easy, medium, and hard differ in actual environment behavior through visibility rules, candidate noise, OCR corruption handling, and stronger task definitions.

### [04-submission-readme-and-metadata.md](04-submission-readme-and-metadata.md)

Detailed plan for rewriting the repository documentation as a submission artifact rather than a scaffold note, including README structure, baseline results, and metadata tightening.

### [05-validation-and-deployment.md](05-validation-and-deployment.md)

Detailed plan for final verification: `openenv validate`, Docker build and run checks, API smoke tests, and deployment readiness for Hugging Face Spaces.

## Completion Criteria For The Full Plan Set

The plan set can be considered executed when all of the following are true:

- The environment runs on prepared receipt data rather than embedded mocks.
- Task difficulty settings materially affect behavior and scoring conditions.
- The baseline runner reports reproducible results across all three tasks.
- The README reads as a finished submission and matches the implemented behavior.
- `openenv validate` is run and recorded.
- The container and deployment path are verified end to end.

Current state:

- All listed conditions are satisfied except the final container and deployment verification step.

## Notes

- These are implementation plans, not design notes. Each document is intended to be actionable.
- The sequence above is deliberate. Later phases assume the earlier ones are complete.