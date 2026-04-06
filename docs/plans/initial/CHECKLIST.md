# Master Execution Checklist

This checklist tracks execution progress across the initial gap-closure plans.

## Usage

- Mark items complete only when the acceptance criteria in the linked plan are actually satisfied.
- Prefer checking off milestones in batches after verification, not while work is still partially in progress.
- Update this file alongside implementation changes so the planning state stays useful.

## Global Milestones

- [ ] Real dataset path replaces embedded mock runtime data.
- [ ] Task difficulty settings change runtime behavior in meaningful, deterministic ways.
- [ ] Baseline evaluation reports reproducible results across all three tasks.
- [ ] README and metadata are submission-ready and no longer scaffold-oriented.
- [ ] Docker, API smoke tests, and `openenv validate` are all verified.

## 00 Roadmap

Reference: [00-roadmap.md](00-roadmap.md)

- [ ] Confirm the intended prepared receipt dataset source of truth.
- [ ] Sequence the implementation according to roadmap dependencies.
- [ ] Track milestone completion against explicit acceptance criteria.
- [ ] Re-run tests and validation gates after each major phase.

## 01 Real Dataset Integration

Reference: [01-real-dataset-integration.md](01-real-dataset-integration.md)

- [ ] Inspect representative dataset annotation files and image mappings.
- [ ] Define the canonical on-disk record contract.
- [ ] Implement parsing from prepared dataset files into `ReceiptSample`.
- [ ] Separate file reading from sample-model construction.
- [ ] Preserve deterministic seeded sampling.
- [ ] Replace sample ID substring matching with explicit difficulty mapping.
- [ ] Add tests for valid parsing, malformed input, and deterministic sample selection.
- [ ] Remove or isolate embedded mock samples from the default runtime path.

## 02 Baseline Inference And Evaluation

Reference: [02-baseline-inference-and-eval.md](02-baseline-inference-and-eval.md)

- [ ] Redesign the CLI to support all-task evaluation by default.
- [ ] Add flags for seed, episode count, and output format.
- [ ] Standardize per-task and aggregate metrics output.
- [ ] Decide whether to remove, rename, or fully implement the `openai` agent mode.
- [ ] Extract evaluation helpers so they are testable outside CLI logging.
- [ ] Add tests for deterministic heuristic evaluation behavior.
- [ ] Record publishable baseline numbers from a finalized environment run.

## 03 Task Difficulty And Environment Fidelity

Reference: [03-task-difficulty-and-environment-fidelity.md](03-task-difficulty-and-environment-fidelity.md)

- [ ] Define explicit runtime semantics for `easy`, `medium`, and `hard`.
- [ ] Enforce task-specific visibility rules in the environment.
- [ ] Apply deterministic candidate ranking noise or ambiguity where appropriate.
- [ ] Clarify whether corruption is applied at load time or observation time.
- [ ] Verify reward shaping remains useful across all task variants.
- [ ] Add tests that prove difficulty differences are real and deterministic.

## 04 Submission README And Metadata

Reference: [04-submission-readme-and-metadata.md](04-submission-readme-and-metadata.md)

- [ ] Audit current docs and metadata for scaffold-era language.
- [ ] Rewrite README around the hackathon rubric requirements.
- [ ] Document observation space, action space, task definitions, and reward summary.
- [ ] Add reproducible baseline results and evaluation command details.
- [ ] Tighten `openenv.yaml` to match the final runtime contract.
- [ ] Remove stale or contradictory top-level project claims.

## 05 Validation And Deployment Readiness

Reference: [05-validation-and-deployment.md](05-validation-and-deployment.md)

- [ ] Create or refine a repeatable verification checklist for tests, baseline, Docker, and validator.
- [ ] Extend smoke-test coverage if the current script is insufficient.
- [ ] Verify Docker build and container startup against the finalized repo.
- [ ] Run API smoke tests against the local server and containerized server.
- [ ] Run `openenv validate` and resolve all reported failures.
- [ ] Document deployment assumptions and startup requirements for Hugging Face Spaces.

## Final Release Gate

- [ ] Tests pass in the intended submission environment.
- [ ] Baseline results are regenerated on the final codebase.
- [ ] README, metadata, and validation results agree with the implementation.
- [ ] The repository can be presented as a finished submission without scaffold disclaimers.