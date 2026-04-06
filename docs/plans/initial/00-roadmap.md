# Roadmap To Submission Readiness

## Status

Current state: in progress

- Phases 1 through 4 are complete.
- Phase 5 is complete for local Python validation and OpenEnv validation.
- Remaining blocker: Docker and container startup have not been verified on this machine.

## Objective

Convert the current repository from a strong prototype into a submission-ready OpenEnv environment that can survive pass-fail validation and score competitively against the hackathon rubric.

## Current State Summary

The repository already includes:

- a typed environment model
- deterministic graders and rewards
- a functional `reset()` / `step()` / `state()` API
- a baseline heuristic flow
- tests for core behavior
- a Dockerfile and minimal metadata

The largest remaining gaps are:

- the environment still runs on mock in-code samples
- baseline inference overclaims its `openai` mode and does not report all tasks by default
- difficulty settings are only partially enforced in runtime behavior
- submission documentation is incomplete
- validation and deployment evidence is missing

## Delivery Strategy

### Phase 1: Replace scaffolding with real data flow

Goal:

Make the environment operate on prepared receipt dataset records rather than in-memory demo objects.

Why this goes first:

- It closes the most obvious submission blocker.
- It prevents later plans from tuning logic against toy inputs.
- It provides the real data distribution needed for baseline and difficulty work.

Primary output:

- a deterministic dataset loader over the prepared receipt corpus

### Phase 2: Make difficulty progression real

Goal:

Ensure `easy`, `medium`, and `hard` differ in observable and decision-relevant ways, not only by step budget.

Why this goes second:

- Baseline evaluation should reflect the real task split.
- README claims should be grounded in implemented behavior.

Primary output:

- task-specific visibility, noise, corruption, or candidate quality behavior that changes the agent experience

### Phase 3: Make the baseline honest, reproducible, and reportable

Goal:

Provide a default evaluation path that produces reproducible scores over all tasks and accurately describes which agents are actually implemented.

Why this goes third:

- Baseline quality depends on the real dataset and real task split.
- Reported scores should come after the environment semantics settle.

Primary output:

- a baseline script that evaluates all three tasks and emits structured results

### Phase 4: Rewrite the repo as a submission artifact

Goal:

Replace scaffold-style README language with submission-quality documentation tied to real behavior and measured results.

Why this goes fourth:

- Documentation should reflect the system that now exists, not the system originally intended.

Primary output:

- a README that maps directly to the judging rubric

### Phase 5: Validate and harden deployment

Goal:

Collect end-to-end evidence that the repo satisfies validator, container, and deployment expectations.

Why this goes last:

- Validation should be run against the final integrated system.
- Any issues discovered here can still feed back into a small hardening pass.

Primary output:

- reproducible submission verification notes and fixes

## Work Breakdown

### Workstream A: Data layer

- define canonical receipt sample schema
- ingest prepared annotations and OCR into that schema
- handle dataset split selection
- keep deterministic seeded sampling
- surface actionable loader errors when data is missing or malformed

### Workstream B: Environment realism

- attach task config to runtime logic
- enforce visibility restrictions by task
- inject controlled ambiguity or ranking noise deterministically
- ensure graders remain deterministic even with task perturbations

### Workstream C: Agent and evaluation layer

- separate heuristic and API-driven modes clearly
- make all-task evaluation first-class
- export summary metrics in machine-readable format
- record reproducibility controls such as seed and dataset split

### Workstream D: Documentation and packaging

- rewrite README to match the judging rubric
- tighten `openenv.yaml`
- ensure package metadata reflects actual behavior, not scaffold status

### Workstream E: Validation and release readiness

- run test suite after each phase
- run API smoke tests against the served environment
- verify Docker build and startup
- run `openenv validate`

## Dependencies

### Hard dependencies

- prepared receipt dataset format must be finalized before the data plan can complete
- exact OpenEnv validator expectations should be confirmed before final metadata hardening

### Soft dependencies

- if OpenAI mode remains optional, the heuristic baseline can still satisfy reproducibility needs as long as the claims are accurate
- HF deployment can proceed after local Docker validation, but not before

## Milestones

### Milestone 1: Real dataset cutover

Done when:

- `env/dataset.py` no longer relies on embedded samples as the primary runtime path
- tests cover dataset loading and deterministic sample selection
- environment reset succeeds against the prepared dataset

### Milestone 2: Behavioral difficulty split

Done when:

- task configs influence at least visibility, candidate quality, or corruption behavior
- easy, medium, and hard are measurably different in runtime traces
- the task descriptions in docs match the code

### Milestone 3: Baseline reporting

Done when:

- one command runs baseline evaluation on all tasks
- outputs include per-task scores and aggregate summary
- agent mode names accurately describe behavior

### Milestone 4: Submission documentation

Done when:

- README includes motivation, environment model, actions, observations, tasks, setup, baseline scores, and deployment notes
- no scaffold disclaimers remain unless they describe an intentional limitation

### Milestone 5: Submission verification

Done when:

- tests pass
- Docker build and run pass
- API endpoints pass a smoke test
- `openenv validate` passes and is recorded in documentation

## Acceptance Criteria

The roadmap is complete only when the repository can credibly support the following claim:

"This is a complete, validated, containerized OpenEnv submission for receipt extraction with reproducible baselines across three meaningful task difficulties."

## Risks

### Data preparation mismatch

Risk:

The on-disk receipt corpus may not align with the assumptions baked into current models.

Mitigation:

- create explicit parsing and normalization helpers
- validate a subset before full cutover
- keep a small fixture set for tests

### Overcomplicating difficulty logic

Risk:

Adding too much stochastic or heuristic complexity may make the environment hard to reason about and hard to test.

Mitigation:

- keep perturbations seeded and deterministic
- make each task rule explicit and testable

### Documentation drift

Risk:

The README may fall behind implementation changes.

Mitigation:

- rewrite docs near the end but before final validation
- derive baseline numbers from a repeatable command

## Recommended Implementation Order Inside The Codebase

1. Update dataset parsing and tests.
2. Update environment runtime logic to consume task config.
3. Update candidate retrieval or observation construction as needed.
4. Update inference and evaluation scripts.
5. Update README and metadata.
6. Run validator, container, and deployment checks.