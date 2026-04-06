# Plan: Validation And Deployment Readiness

## Goal

Verify that the final repository satisfies the technical submission gates beyond local unit tests: validator compatibility, container startup, API behavior, and deployment readiness.

## Problem Statement

The project includes a Dockerfile and API server, but there is no recorded evidence yet that the final integrated submission passes `openenv validate`, starts cleanly in a container, or is ready for Hugging Face Spaces deployment.

## Desired End State

After this plan is complete:

- the repository passes its unit and integration checks
- the container builds and runs successfully
- API endpoints behave correctly in a local smoke test
- `openenv validate` passes and is documented
- deployment-specific assumptions are explicit and verified

## Scope

### In scope

- validator execution
- Docker build and run verification
- server endpoint smoke testing
- deployment checklist for Hugging Face Spaces
- documenting the verification steps and outcomes

### Out of scope

- post-submission monitoring
- scaling or performance optimization beyond what affects startup and validation

## Validation Matrix

### Layer 1: Python and unit validation

Checks:

- install dependencies in a clean environment
- run `pytest`
- run baseline evaluation command

Purpose:

- catch local regressions before container or validator work

### Layer 2: API validation

Checks:

- start the FastAPI server locally
- call `/reset`, `/step`, and `/state`
- confirm response models serialize correctly on real dataset-backed episodes

Purpose:

- ensure the service contract still works after integration changes

### Layer 3: Container validation

Checks:

- run `docker build`
- run the container locally
- confirm the server starts on the expected port
- repeat the API smoke test against the containerized service

Purpose:

- detect packaging, path, and dependency issues that local runs may hide

### Layer 4: OpenEnv validation

Checks:

- install or otherwise access the validator toolchain
- run `openenv validate`
- capture the exact output and remediate failures

Purpose:

- satisfy the explicit compliance gate from the brief

### Layer 5: Hugging Face Space readiness

Checks:

- confirm startup command and port expectations
- confirm required environment variables are documented
- confirm repository layout is compatible with chosen deployment mode

Purpose:

- avoid last-mile deployment surprises

## Implementation Plan

### Step 1: Create a repeatable verification checklist

Tasks:

- define the exact commands to run for tests, baseline evaluation, Docker, and validator
- keep this checklist in docs so it can be reused before submission

Acceptance criteria:

- a teammate can run the same sequence without guessing missing steps

### Step 2: Add or refine smoke-test scripts if needed

Tasks:

- extend existing smoke test helpers if they do not cover API endpoints sufficiently
- ensure scripts fail fast on startup or contract issues

Acceptance criteria:

- local smoke testing is automated enough to be rerun after changes

### Step 3: Validate Docker end to end

Tasks:

- ensure the Docker image installs all runtime dependencies
- ensure dataset paths expected at runtime exist inside the image
- confirm the server binds correctly and survives a simple request cycle

Acceptance criteria:

- container run matches local server behavior for the basic API flow

### Step 4: Run and fix `openenv validate`

Tasks:

- execute the validator against the finalized repo
- classify failures into metadata, API, or packaging problems
- fix issues with the smallest compatible change set

Acceptance criteria:

- validator passes without relying on undocumented manual intervention

### Step 5: Capture deployment notes

Tasks:

- document required environment variables
- document the expected startup command
- document any repository assumptions for Hugging Face Spaces

Acceptance criteria:

- deployment setup is explicit and reproducible

## Suggested Deliverables

- updated README deployment section
- validator notes or transcript summary
- refined smoke test script if needed
- optional `docs/plans/final-validation-checklist.md` if the team wants a dedicated runbook

## File-Level Work Estimate

### Likely files to change

- `Dockerfile`
- `README.md`
- `scripts/smoke_test.py`
- possibly `openenv.yaml`
- optional validation helper script

## Risks And Mitigations

### Risk: validator requirements differ from current assumptions

Mitigation:

- treat validator output as authoritative
- avoid speculative metadata changes before seeing the real failures

### Risk: dataset files are missing inside the container

Mitigation:

- validate file copy paths explicitly during Docker testing
- fail fast if dataset roots are missing at startup

### Risk: API passes locally but not in container

Mitigation:

- test the same endpoint sequence in both environments
- keep port and host binding explicit

## Exit Criteria

This plan is complete when the repository has documented evidence that tests pass, the container works, the API contract is healthy, and `openenv validate` succeeds on the final codebase.