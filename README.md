# RL Receipt OCR OpenEnv

Starter workspace for an OpenEnv hackathon submission based on Option A: a deterministic receipt extraction environment with a separate trainable policy.

## Status

This repository is scaffolded and git-ready. It includes:

- typed environment models
- a deterministic `reset()` / `step()` / `state()` API
- easy / medium / hard task presets
- heuristic candidate generation and grading
- reward shaping boilerplate
- FastAPI server starter
- reproducible heuristic inference entrypoint
- tests for normalizers, graders, rewards, and environment behavior

## Layout

- `env/` core environment package
- `training/` training placeholders for BC and PPO
- `tests/` deterministic unit tests
- `scripts/` local prep and smoke test helpers

## Quick Start

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
pytest
python scripts/smoke_test.py
python inference.py --agent heuristic --task easy
```

## Notes

- The dataset loader currently ships with mock seeded samples so the workspace runs before SROIE ingestion is added.
- Replace the mock dataset in `env/dataset.py` with prepared SROIE OCR and labels for the actual submission.
- The Dockerfile and `openenv.yaml` are included as submission scaffolding and should be tightened once the final server contract is fixed.
