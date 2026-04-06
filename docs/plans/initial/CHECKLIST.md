# Master Execution Checklist

This checklist tracks execution progress across the initial gap-closure plans.

## Status Snapshot

- Completed through plan 04.
- Plan 05 is substantially complete in the local Python environment.
- Remaining external gap: Docker build and container startup have not been verified on this machine because the `docker` CLI is unavailable.

## Usage

- Mark items complete only when the acceptance criteria in the linked plan are actually satisfied.
- Prefer checking off milestones in batches after verification, not while work is still partially in progress.
- Update this file alongside implementation changes so the planning state stays useful.

## Global Milestones

- [x] Real dataset path replaces embedded mock runtime data.
- [x] Task difficulty settings change runtime behavior in meaningful, deterministic ways.
- [x] Baseline evaluation reports reproducible results across all three tasks.
- [x] README and metadata are submission-ready and no longer scaffold-oriented.
- [ ] Docker, API smoke tests, and `openenv validate` are all verified.

## 00 Roadmap

Reference: [00-roadmap.md](00-roadmap.md)

- [x] Confirm the intended prepared receipt dataset source of truth.
- [x] Sequence the implementation according to roadmap dependencies.
- [x] Track milestone completion against explicit acceptance criteria.
- [x] Re-run tests and validation gates after each major phase.

## 01 Real Dataset Integration

Reference: [01-real-dataset-integration.md](01-real-dataset-integration.md)

- [x] Inspect representative dataset annotation files and image mappings.
- [x] Define the canonical on-disk record contract.
- [x] Implement parsing from prepared dataset files into `ReceiptSample`.
- [x] Separate file reading from sample-model construction.
- [x] Preserve deterministic seeded sampling.
- [x] Replace sample ID substring matching with explicit difficulty mapping.
- [x] Add tests for valid parsing, malformed input, and deterministic sample selection.
- [x] Remove or isolate embedded mock samples from the default runtime path.

## 02 Baseline Inference And Evaluation

Reference: [02-baseline-inference-and-eval.md](02-baseline-inference-and-eval.md)

- [x] Redesign the CLI to support all-task evaluation by default.
- [x] Add flags for seed, episode count, and output format.
- [x] Standardize per-task and aggregate metrics output.
- [x] Decide whether to remove, rename, or fully implement the `openai` agent mode.
- [x] Extract evaluation helpers so they are testable outside CLI logging.
- [x] Add tests for deterministic heuristic evaluation behavior.
- [x] Record publishable baseline numbers from a finalized environment run.

## 03 Task Difficulty And Environment Fidelity

Reference: [03-task-difficulty-and-environment-fidelity.md](03-task-difficulty-and-environment-fidelity.md)

- [x] Define explicit runtime semantics for `easy`, `medium`, and `hard`.
- [x] Enforce task-specific visibility rules in the environment.
- [x] Apply deterministic candidate ranking noise or ambiguity where appropriate.
- [x] Clarify whether corruption is applied at load time or observation time.
- [x] Verify reward shaping remains useful across all task variants.
- [x] Add tests that prove difficulty differences are real and deterministic.

## 04 Submission README And Metadata

Reference: [04-submission-readme-and-metadata.md](04-submission-readme-and-metadata.md)

- [x] Audit current docs and metadata for scaffold-era language.
- [x] Rewrite README around the hackathon rubric requirements.
- [x] Document observation space, action space, task definitions, and reward summary.
- [x] Add reproducible baseline results and evaluation command details.
- [x] Tighten `openenv.yaml` to match the final runtime contract.
- [x] Remove stale or contradictory top-level project claims.

## 05 Validation And Deployment Readiness

Reference: [05-validation-and-deployment.md](05-validation-and-deployment.md)

- [x] Create or refine a repeatable verification checklist for tests, baseline, Docker, and validator.
- [x] Extend smoke-test coverage if the current script is insufficient.
- [ ] Verify Docker build and container startup against the finalized repo.
- [ ] Run API smoke tests against the local server and containerized server.
- [x] Run `openenv validate` and resolve all reported failures.
- [x] Document deployment assumptions and startup requirements for Hugging Face Spaces.

## Final Release Gate

- [x] Tests pass in the intended submission environment.
- [x] Baseline results are regenerated on the final codebase.
- [x] README, metadata, and validation results agree with the implementation.
- [x] The repository can be presented as a finished submission without scaffold disclaimers.