# RL Receipt OCR Hackathon Review

Date: April 6, 2026

## Verdict

This repository is a credible scaffold for the OpenEnv hackathon problem, but it is not yet a submission-ready solution.

The strongest parts are the typed environment models, deterministic grading, reward shaping, API surface, tests, and basic containerization. The main blockers are that the environment still runs on embedded mock samples instead of the actual receipt dataset, the baseline inference path does not truly use the OpenAI client path it advertises, and the documented task difficulty progression is only partially enforced in code.

## Findings

### 1. High: The environment still uses mock embedded samples instead of the real receipt dataset

The solution is framed as a real-world receipt extraction environment, but the active dataset implementation is still hard-coded in `env/dataset.py`.

Why this matters:

- The hackathon statement requires a real-world task simulation.
- The current environment only exercises three in-memory examples with `mock://receipt/...` image references.
- The README explicitly says the dataset loader is still mock and should be replaced for the actual submission.

Evidence:

- `env/dataset.py` defines `SAMPLES` directly in code and returns those samples at runtime.
- `README.md` states that the dataset loader currently ships with mock seeded samples.

Impact:

This is a submission-level blocker. As written, the repo demonstrates the environment shape, but not a complete real-world environment over the actual receipt corpus.

### 2. High: The baseline inference script does not satisfy the stated baseline requirement

The repository includes `inference.py`, but the `openai` agent mode is not actually implemented.

Why this matters:

- The hackathon brief requires a baseline inference script that uses the OpenAI API client.
- In the current code, `llm_action()` ignores the client and always falls back to the heuristic policy.
- The script also runs only one task at a time, rather than producing a reproducible score across all three tasks by default.

Evidence:

- `llm_action()` returns `heuristic_action(env)` unconditionally.
- The CLI accepts a single `--task` choice and calls `run_episode()` once.

Impact:

This is another submission-level blocker. The repository has a useful deterministic heuristic baseline, but not the baseline contract described in the brief.

### 3. Medium: Difficulty progression is declared in configuration, but only weakly enforced in environment behavior

The repository defines `easy`, `medium`, and `hard` task configs with different budgets and noise-related fields, but most of those difficulty knobs are not used by the environment.

Why this matters:

- Judges are asked to assess whether tasks have meaningful difficulty progression.
- `max_steps` is enforced.
- `visible_windows`, `corruption_level`, and `ranking_noise` are present in the task model, but they are not materially consumed in the current environment flow.

Evidence:

- `env/tasks.py` sets different values for `visible_windows`, `corruption_level`, and `ranking_noise`.
- `env/environment.py` uses task difficulty for sampling and `max_steps` for budget, but the other task controls are not applied.

Impact:

This weakens the quality of the easy-to-hard progression. Right now, the task split is partly real and partly aspirational.

### 4. Medium: The README is not yet written as submission-ready documentation

The README is candid about the repo being a scaffold, but that also means it does not fully satisfy the documentation requirements of the brief.

Why this matters:

- The brief asks for environment motivation, action and observation space definitions, task descriptions, setup and usage instructions, and baseline scores.
- The current README provides quick-start instructions and caveats, but it does not yet read like a polished hackathon submission.

Evidence:

- The README contains a short layout and quick-start section.
- The notes explicitly describe the Dockerfile and `openenv.yaml` as scaffolding.
- Baseline scores are not documented.

Impact:

This is not the biggest technical blocker, but it would cost points and makes the project look unfinished.

## What Is Working Well

- The environment surface is clear and deterministic: `reset()`, `step()`, and `state()` are implemented.
- The Pydantic models for observation, action, state, and grading are clean and appropriate for the task.
- The grading logic is deterministic and bounded, with field-level scoring and a final weighted score.
- Reward shaping includes partial progress signals and penalties for repetition and errors.
- The FastAPI wrapper is minimal but functional.
- Tests are present and currently passing in the local workspace.
- A Dockerfile exists and is plausible for local container execution.

## Runtime Notes

A local heuristic smoke check across the three tasks produced deterministic but mixed results:

- easy: final score `0.9375`
- medium: final score `0.6000`
- hard: final score `0.6143`

The heuristic exhausted the step budget in all three cases. That is not inherently invalid, but it does show the current baseline is still fairly weak on medium and hard.

## Overall Assessment Against The Hackathon Statement

### Likely passes

- Typed models are present.
- `reset()` / `step()` / `state()` are implemented.
- Three tasks exist.
- Deterministic graders and shaped rewards exist.
- Basic Docker packaging exists.
- The project is testable locally.

### Likely does not yet pass cleanly

- Real-world data integration is incomplete.
- Baseline inference does not match the stated OpenAI-driven contract.
- Difficulty progression is not fully realized in behavior.
- Documentation is still scaffold-level.
- OpenEnv validation status was not demonstrated in this repository.
- Hugging Face Space deployment readiness was not verified.

## Recommended Next Steps

1. Replace the mock dataset path with actual prepared receipt OCR and labels from the intended corpus.
2. Make `inference.py` run all three tasks by default and either implement the real OpenAI action path or rename the mode so it does not overclaim.
3. Apply task-specific corruption, ranking noise, and visibility constraints so `easy`, `medium`, and `hard` differ in actual environment behavior.
4. Rewrite the README as a submission document with action space, observation space, task descriptions, baseline scores, setup, and deployment notes.
5. Run and record `openenv validate`, then verify the container and Hugging Face Space startup path end to end.

## Bottom Line

This repo is a strong foundation and a plausible prototype for the hackathon brief, but it is still short of a competitive submission in its current state. The remaining work is not cosmetic; it is concentrated in the parts judges are likely to treat as pass-fail or high-weight scoring criteria.