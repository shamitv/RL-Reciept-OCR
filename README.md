---
title: Rl Hackathon 26
emoji: "🐠"
colorFrom: red
colorTo: pink
sdk: docker
pinned: false
license: apache-2.0
---

# OpenEnv Receipt Understanding

This README includes the core hackathon submission details:

- [Environment description and motivation](#environment-description-and-motivation)
- [Action and observation space definitions](#action-and-observation-space-definitions)
- [Task descriptions with expected difficulty](#task-descriptions-with-expected-difficulty)
- [Baseline results](#baseline-results)
- [Quickstart and detailed docs](#quickstart-and-detailed-docs)

## Environment Description And Motivation

OpenEnv Receipt Understanding is a sequential receipt extraction environment for the OpenEnv hackathon. The motivation is to help small on-device models turn messy receipt images into reliable structured data even when they cannot solve the whole document in one shot.

That matters for expense capture, bookkeeping, reimbursement, and merchant workflows where privacy, latency, and cost make small local models attractive, but messy real receipts still break one-shot extraction.

The environment now uses a task-aware receipt schema:

- `easy`: `company`, `date`, `address`, `total`
- `medium`: `company`, `date`, `subtotal`, `tax`, `total`
- `hard`: `company`, `date`, `subtotal`, `tax`, `total`, `line_items`

The environment is designed as a real-world document extraction task rather than a single-step classifier. Agents operate through repeated `reset()` / `step()` / `state()` interactions and are scored by deterministic graders against annotated receipt data.

The real-world pain is that receipts are rarely clean. Mobile captures are often skewed or rotated, shadows hide totals, thermal paper fades, contrast is low, long receipts get cropped badly, and OCR misses parts of the layout. A lightweight model running on-device may be cheap and private, but it will often need help recovering from those errors.

That is why this environment treats receipt understanding as a decision-making problem instead of a single prediction. A good agent should learn when to inspect more evidence, when to query candidates, when to validate arithmetic consistency, when to gather more line-item evidence, and when the draft is good enough to submit. As the environment evolves, the same framework can support recovery actions such as changing contrast, sharpening, rotating, or cropping before retrying extraction. Those image-transformation actions are part of the motivating direction for the project, not part of the currently implemented action space.

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

## Action And Observation Space Definitions

### Observation space

Each observation includes:

- task id and difficulty
- instruction string
- receipt image reference
- currently visible OCR regions
- candidate lists requested so far, including `subtotal` and `tax`
- line-item candidates for hard receipts
- current extraction draft
- validation feedback
- reconciliation feedback and delta
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
- `query_line_item_candidates`
- `add_line_item_from_candidate`
- `add_line_item_manual`
- `remove_line_item`
- `check_receipt_consistency`
- `clear_field`
- `submit`

### Reward and termination

- Step rewards provide partial progress signals when the draft improves.
- Repeated actions and invalid actions are penalized.
- Terminal reward is based on the deterministic final grade.
- Episodes end on `submit` or when the task budget is exhausted.

## Task Descriptions With Expected Difficulty

The environment keeps the public task IDs `easy`, `medium`, and `hard`, but they now correspond to different receipt objectives as well as different visibility/noise/budget settings.

### Easy

- expected difficulty: lowest
- header extraction for `company`, `date`, `address`, and `total`
- broader starter reveal after `view_receipt`
- all windows available for `list_text_regions`
- lowest candidate reranking noise
- largest step budget

### Medium

- expected difficulty: medium
- monetary-summary extraction for `company`, `date`, `subtotal`, `tax`, and `total`
- deterministic reconciliation between `subtotal + tax` and `total`
- narrower starter reveal
- `all` window is unavailable
- moderate deterministic candidate reranking noise
- medium step budget

### Hard

- expected difficulty: highest
- summary extraction plus line-item reconstruction
- deterministic reconciliation between line-item totals, `subtotal`, `tax`, and `total`
- restricted starter reveal
- only `top` and `bottom` windows available
- highest deterministic candidate reranking noise
- smallest step budget

## Dataset

The default runtime dataset is loaded from the prepared receipt annotations in `dataset/Receipt dataset/ds0`.

At load time, the environment:

- reads rectangle annotations from `ann/`
- reconstructs OCR regions from annotation boxes and transcriptions
- derives gold `company`, `address`, `date`, `subtotal`, `tax`, and `total` fields from labeled categories
- derives gold line-item rows directly from `Item information` labels
- builds deterministic task pools from label coverage:
  - `easy` requires header labels
  - `medium` requires summary labels
  - `hard` requires summary labels plus line-item labels

You can override the dataset location through `RECEIPT_DATASET_ROOT`.

## Baseline Results

Submitted LLM baseline:

- command: `python inference.py`
- required environment variables: `API_BASE_URL`, `MODEL_NAME`, `HF_TOKEN`
- default sample set: the selected 50-image manifest at `artifacts/datasets/receipt-selection-50/selected_manifest.json`, when present
- LLM client: the root inference script builds an OpenAI-compatible client from those variables and uses the existing receipt-image extraction pipeline before submitting the extracted draft through the environment.
- fallback all-task smoke run: `python inference.py --no-manifest`

Offline reproducible heuristic baseline:

- command: `python inference.py --agent heuristic --format text`
- base seed: `7`
- episodes per task: `1`

Current scores:

- `easy`: mean score `0.400`
- `medium`: mean score `0.200`
- `hard`: mean score `0.100`
- aggregate mean score: `0.233`

These numbers reflect the current deterministic heuristic and current task constraints. They should be regenerated if task logic, grading, or dataset filtering changes.

The full checked-in baseline report is in [docs/hackathon/baseline-scores.md](D:/work/RL-Reciept-OCR/docs/hackathon/baseline-scores.md).

## Quickstart And Detailed Docs

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
$env:API_BASE_URL="https://router.huggingface.co/v1"
$env:MODEL_NAME="<your-model-id>"
$env:HF_TOKEN="<your-token>"
python inference.py
```

This quickstart is enough to install the project and run the submitted LLM baseline. `requirements.txt` includes the local PPO inference runtime dependency (`torch`) so a standard repo setup can also run heuristic and checkpoint-backed PPO inference.

Detailed runbooks:

- [Local setup and baseline usage](D:/work/RL-Reciept-OCR/docs/howto/local-setup-and-baseline.md)
- [Run the eval API and UI](D:/work/RL-Reciept-OCR/docs/howto/run-eval-api-ui.md)
- [Evaluate dataset images](D:/work/RL-Reciept-OCR/docs/howto/eval/evaluate_dataset_images.md)
- [Docs index](D:/work/RL-Reciept-OCR/docs/README.md)

Evaluator-facing shortcuts:

- submitted LLM baseline entrypoint: `python inference.py`
- selected 50-image manifest: [artifacts/datasets/receipt-selection-50/selected_manifest.json](D:/work/RL-Reciept-OCR/artifacts/datasets/receipt-selection-50/selected_manifest.json)
- offline heuristic baseline entrypoint: `python inference.py --agent heuristic --format text`
- environment API entrypoint: `uvicorn env.server:app --host 0.0.0.0 --port 7860`
- eval dashboard: `GET /eval`
- full baseline report: [docs/hackathon/baseline-scores.md](D:/work/RL-Reciept-OCR/docs/hackathon/baseline-scores.md)

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

### Hackathon pre-submission validator

The repository also includes the hackathon-style pre-submission validator:

```bash
bash scripts/validate-submission.sh https://your-space.hf.space .
```

It checks that the submitted Space responds to `POST /reset`, the Docker image builds, and `openenv validate` passes.

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
- the submitted inference path uses the configured LLM through an OpenAI-compatible client and evaluates all three tasks by default
- the heuristic baseline remains available with `python inference.py --agent heuristic --format text`
- checkpoint-backed PPO inference is implemented behind `python inference.py --agent ppo --checkpoint ...`
- the intended learning architecture is an external PPO-trained policy over environment observations, while any helper LLM remains frozen
- local pytest currently passes
- `openenv validate` currently passes in the local Python environment

## Limitations

- PPO and behavior-cloning training are still placeholders, so learned-policy optimization is not implemented yet
- PPO inference requires a compatible checkpoint and the optional `torch` extra
- dataset-wide image eval depends on configured OpenAI-compatible model endpoints for extraction and validation
- final `openenv validate` and deployment verification are still pending
