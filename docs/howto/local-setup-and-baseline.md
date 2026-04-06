# Local Setup And Baseline Usage

This guide covers the local developer setup for OpenEnv Receipt Understanding and the main commands for running the baseline environment flows.

If you want the eval dashboard and API, also see [run-eval-api-ui.md](run-eval-api-ui.md). If you want bulk image evaluation, also see [evaluate_dataset_images.md](eval/evaluate_dataset_images.md).

## 1. Create A Local Environment

From the repo root:

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

`requirements.txt` includes the PPO runtime dependency (`torch`) for local checkpoint-backed inference.

If you prefer a package install with an explicit PPO extra:

```powershell
pip install -e ".[ppo]"
```

## 2. Optional `.env` Configuration

Create a `.env` file in the repo root if you want local environment variables loaded automatically.

```dotenv
API_BASE_URL=https://router.huggingface.co/v1
MODEL_NAME=gpt-4o-mini
EVAL_API_BASE_URL=https://router.huggingface.co/v1
EVAL_MODEL=gpt-4.1
OPENAI_API_KEY=your-key-here
HF_TOKEN=your-hf-token-here
RECEIPT_DATASET_ROOT=dataset/Receipt dataset/ds0
RECEIPT_EVAL_OUTPUT_DIR=artifacts/eval/dataset-image-eval
```

Notes:

- `RECEIPT_DATASET_ROOT` is optional. If unset, the loader defaults to the bundled prepared dataset.
- `RECEIPT_EVAL_OUTPUT_DIR` is optional. If unset, the eval pipeline writes to `artifacts/eval/dataset-image-eval`.
- `MODEL_NAME` / `API_BASE_URL` and `EVAL_MODEL` / `EVAL_API_BASE_URL` are only required for the dataset image evaluation pipeline, not for the local heuristic baseline.

## 3. Run The Baseline

The main baseline entrypoint is:

```powershell
.\.venv\Scripts\python inference.py --format text
```

Useful variants:

```powershell
.\.venv\Scripts\python inference.py --task easy
.\.venv\Scripts\python inference.py --task medium
.\.venv\Scripts\python inference.py --task hard
.\.venv\Scripts\python inference.py --format json
.\.venv\Scripts\python inference.py --episodes 3 --seed 7
```

The checked-in reference scores are documented in [baseline-scores.md](../hackathon/baseline-scores.md).

## 4. Run PPO Inference From A Checkpoint

Use this path if you already have a compatible PPO checkpoint:

```powershell
.\.venv\Scripts\python inference.py --agent ppo --checkpoint checkpoints\policy.pt
```

Useful variants:

```powershell
.\.venv\Scripts\python inference.py --agent ppo --checkpoint checkpoints\policy.pt --task hard
.\.venv\Scripts\python inference.py --agent ppo --checkpoint checkpoints\policy.pt --episodes 3
.\.venv\Scripts\python inference.py --agent ppo --checkpoint checkpoints\policy.pt --device cpu
```

`--agent ppo` requires both a checkpoint and `torch`.

## 5. Run Local Checks

Tests:

```powershell
.\.venv\Scripts\python -m pytest
```

Validation bundle:

```powershell
.\.venv\Scripts\python scripts\validate_local.py
```

Smoke test:

```powershell
.\.venv\Scripts\python scripts\smoke_test.py
```

OpenEnv validator:

```powershell
.\.venv\Scripts\openenv.exe validate
```
