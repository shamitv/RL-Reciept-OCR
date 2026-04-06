# RL Receipt OCR OpenEnv

RL Receipt OCR OpenEnv is a sequential receipt extraction environment for the OpenEnv hackathon. An agent must inspect receipt text regions, request candidate values, edit a structured draft, validate uncertain fields, and decide when to submit the final extraction.

The target schema is fixed to four fields:

- `company`
- `date`
- `address`
- `total`

The environment is designed as a real-world document extraction task rather than a single-step classifier. Agents operate through repeated `reset()` / `step()` / `state()` interactions and are scored by deterministic graders against annotated receipt data.

## Learning Agent Design

The intended learning agent in this project is an external policy trained against the environment, not the frozen LLM itself.

The planned setup is:

- Environment: the OpenEnv receipt extraction world implemented in this repo
- Agent: a policy that reads the observation or encoded state and chooses the next action
- Learning: PPO updates that policy from reward signals
- Frozen helper: an LLM may assist with interpretation or reranking, but it does not learn

In other words, the learning loop is:

```text
observation -> policy network -> action
                    ^
                    |
             updated by PPO from reward
```

Under that design, the agent learns:

- what action to take next
- which field to focus on
- whether to inspect more OCR evidence
- which candidate to choose
- when to validate
- when to submit

That is a real learning agent because its future behavior is meant to improve based on past reward.

### Trainable vs. Frozen Components

Trainable component:

- an external policy module, intended to be trained with PPO

Frozen components:

- the base LLM, if one is used as a helper
- the OCR evidence already present in the dataset
- the deterministic graders and reward logic

What this design does not do:

- fine-tune the base LLM
- directly learn OCR
- directly generate raw text end to end via RL

## Environment Definition

### Observation space

Each observation includes:

- task id and difficulty
- instruction string
- receipt image reference
- currently visible OCR regions
- candidate lists requested so far
- current extraction draft
- validation feedback
- last action result
- remaining step budget
- step index
- terminal permission flag

### Action space

The environment accepts typed structured actions, including:

- `view_receipt`
- `list_text_regions`
- `inspect_bbox`
- `inspect_neighbors`
- `query_candidates`
- `set_field_from_candidate`
- `set_field_manual`
- `merge_spans`
- `normalize_field`
- `check_total_consistency`
- `check_date_format`
- `clear_field`
- `submit`

### Reward and termination

- Step rewards provide partial progress signals when the draft improves.
- Repeated actions and invalid actions are penalized.
- Terminal reward is based on the deterministic final grade.
- Episodes end on `submit` or when the task budget is exhausted.

## Tasks

The environment defines three task presets with deterministic behavior differences.

### Easy

- broader starter reveal after `view_receipt`
- all windows available for `list_text_regions`
- lowest candidate reranking noise
- largest step budget

### Medium

- narrower starter reveal
- `all` window is unavailable
- moderate deterministic candidate reranking noise
- medium step budget

### Hard

- restricted starter reveal
- only `top` and `bottom` windows available
- highest deterministic candidate reranking noise
- smallest step budget

## Dataset

The default runtime dataset is loaded from the prepared receipt annotations in `dataset/Receipt dataset/ds0`.

At load time, the environment:

- reads rectangle annotations from `ann/`
- reconstructs OCR regions from annotation boxes and transcriptions
- derives gold `company`, `address`, `date`, and `total` fields from labeled categories
- filters out incomplete records that do not contain all four target fields
- buckets valid records into deterministic easy, medium, and hard sample pools

You can override the dataset location through `RECEIPT_DATASET_ROOT`.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
pytest
```

`requirements.txt` now includes the local PPO inference runtime dependency (`torch`) so a standard repo setup can run both heuristic and checkpoint-backed PPO inference.

### Optional PPO package extra

If you are installing the project as a package instead of using `requirements.txt`, you can still opt into the PPO runtime with the package extra.

```bash
pip install -e ".[ppo]"
```

### Optional `.env` configuration

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

`RECEIPT_DATASET_ROOT` is optional. If unset, the loader defaults to the bundled prepared dataset.
`RECEIPT_EVAL_OUTPUT_DIR` is optional. If unset, the eval pipeline writes to `artifacts/eval/dataset-image-eval`.

## Local Usage

### Run tests

```bash
pytest
```

### Run the local validation bundle

```bash
python scripts/validate_local.py
```

### Smoke test the environment package

```bash
python scripts/smoke_test.py
```

### Run heuristic baseline evaluation

```bash
python inference.py
```

Useful options:

- `python inference.py --task easy`
- `python inference.py --task medium`
- `python inference.py --task hard`
- `python inference.py --format json`
- `python inference.py --episodes 3 --seed 7`

### Run PPO inference from a checkpoint

```bash
python inference.py --agent ppo --checkpoint checkpoints/policy.pt
```

Useful options:

- `python inference.py --agent ppo --checkpoint checkpoints/policy.pt --task hard`
- `python inference.py --agent ppo --checkpoint checkpoints/policy.pt --episodes 3`
- `python inference.py --agent ppo --checkpoint checkpoints/policy.pt --device cpu`

`--agent ppo` requires both a checkpoint and `torch`. A standard local setup via `pip install -r requirements.txt` now includes it; package installs can use `pip install -e ".[ppo]"`.

### Run dataset-wide image evaluation

```bash
python scripts/evaluate_dataset_images.py
```

Useful options:

- `python scripts/evaluate_dataset_images.py --limit 10`
- `python scripts/evaluate_dataset_images.py --resume`
- `python scripts/evaluate_dataset_images.py --output-dir artifacts/eval/dataset-image-eval`

Artifacts written by the evaluator:

- `artifacts/eval/dataset-image-eval/results.jsonl`
- `artifacts/eval/dataset-image-eval/summary.json`
- `artifacts/eval/dataset-image-eval/report.md`

The evaluator walks every annotation file in the dataset, records skipped items explicitly, grades runnable samples deterministically against gold fields, and uses a larger eval model for validation notes.

### Run the API server

```bash
uvicorn env.server:app --host 0.0.0.0 --port 7860
```

Endpoints:

- `POST /reset`
- `POST /step`
- `GET /state`
- `GET /api/eval/summary`
- `GET /api/eval/receipts`
- `GET /api/eval/receipts/{sample_id}`
- `GET /api/eval/receipts/{sample_id}/image`
- `GET /api/eval/report`
- `GET /eval`
- `GET /eval/receipts/{sample_id}`

## Baseline Results

Current reproducible heuristic baseline:

- command: `python inference.py --format text`
- base seed: `7`
- episodes per task: `1`

Current scores:

- `easy`: mean score `0.400`
- `medium`: mean score `0.400`
- `hard`: mean score `0.400`
- aggregate mean score: `0.400`

These numbers reflect the current deterministic heuristic and current task constraints. They should be regenerated if task logic, grading, or dataset filtering changes.

## Project Layout

- `agents/` heuristic and checkpoint-backed PPO inference runtimes
- `env/` environment package, graders, rewards, dataset loader, API server
- `server/templates/` built-in eval UI templates
- `server/static/` built-in eval UI styles
- `training/` PPO and BC entrypoints; training remains placeholder-only
- `tests/` deterministic unit tests
- `scripts/` local preparation, smoke-test helpers, and dataset eval runner
- `docs/` hackathon notes and execution plans

## Deployment And Validation

### Docker

```bash
docker build -t rl-receipt-ocr .
docker run -p 7860:7860 rl-receipt-ocr
```

### OpenEnv validation

The repository now passes `openenv validate` in the local development environment after adding:

- `openenv-core>=0.2.3` as a declared dependency
- `server/app.py` as the deployment entrypoint wrapper
- a generated `uv.lock`

The validation command used was:

```bash
openenv validate
```

### Hugging Face Spaces

The repository includes a container entrypoint suitable for a Space-style deployment, but final deployment verification is still pending.

### Docker status

Docker validation has not been executed on this machine because the `docker` CLI is not installed in the current environment. The intended commands are:

```bash
docker build -t rl-receipt-ocr .
docker run -p 7860:7860 rl-receipt-ocr
```

## Current Status

- real annotated receipt data is the default runtime source
- the environment implements typed models and `reset()` / `step()` / `state()`
- task difficulty now affects runtime behavior, not only step budget
- the heuristic baseline evaluates all three tasks deterministically by default
- checkpoint-backed PPO inference is implemented behind `python inference.py --agent ppo --checkpoint ...`
- the intended learning architecture is an external PPO-trained policy over environment observations, while any helper LLM remains frozen
- local pytest currently passes
- `openenv validate` currently passes in the local Python environment

## Limitations

- PPO and behavior-cloning training are still placeholders, so learned-policy optimization is not implemented yet
- PPO inference requires a compatible checkpoint and the optional `torch` extra
- dataset-wide image eval depends on configured OpenAI-compatible model endpoints for extraction and validation
- final `openenv validate` and deployment verification are still pending
