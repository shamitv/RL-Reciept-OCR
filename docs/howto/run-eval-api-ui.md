# How To Run The Eval API And UI

This guide shows how to:

- generate the evaluation artifacts the UI reads
- start the FastAPI server
- open the API and browser UI locally

## 1. Prerequisites

From the repo root:

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

If you want to use a `.env` file, create one in the repo root.

Minimum useful eval config:

```dotenv
API_BASE_URL=https://router.huggingface.co/v1
MODEL_NAME=gpt-4o-mini
EVAL_API_BASE_URL=https://router.huggingface.co/v1
EVAL_MODEL=gpt-4.1
OPENAI_API_KEY=your-key-here
HF_TOKEN=your-key-here
RECEIPT_DATASET_ROOT=dataset/Receipt dataset/ds0
RECEIPT_EVAL_OUTPUT_DIR=artifacts/eval/dataset-image-eval
```

Notes:

- `MODEL_NAME` and `API_BASE_URL` are used for the primary extraction pass.
- `EVAL_MODEL` and `EVAL_API_BASE_URL` are used for the larger validation model.
- `RECEIPT_EVAL_OUTPUT_DIR` is optional. If omitted, artifacts go to `artifacts/eval/dataset-image-eval`.

## 2. Generate Eval Artifacts

The API and UI are read-only viewers over saved eval results. Run the evaluator first:

```powershell
.\.venv\Scripts\python scripts\evaluate_dataset_images.py
```

Useful variants:

```powershell
.\.venv\Scripts\python scripts\evaluate_dataset_images.py --limit 10
.\.venv\Scripts\python scripts\evaluate_dataset_images.py --resume
.\.venv\Scripts\python scripts\evaluate_dataset_images.py --output-dir artifacts/eval/dataset-image-eval
```

Expected outputs:

- `artifacts/eval/dataset-image-eval/results.jsonl`
- `artifacts/eval/dataset-image-eval/summary.json`
- `artifacts/eval/dataset-image-eval/report.md`

If you skip this step, the UI still starts, but `/eval` will show an empty-state message instead of receipt results.

## 3. Start The API Server

Run either command from the repo root:

```powershell
.\.venv\Scripts\python -m uvicorn env.server:app --host 0.0.0.0 --port 7860
```

or:

```powershell
.\.venv\Scripts\python server\app.py
```

Default local URL:

```text
http://127.0.0.1:7860
```

## 4. Open The UI

In a browser, open:

```text
http://127.0.0.1:7860/eval
```

Main UI pages:

- `/eval` - summary dashboard and receipt list
- `/eval/receipts/{sample_id}` - per-receipt detail view

The detail page shows:

- the receipt image
- gold fields
- predicted fields
- field-by-field correct, partial, missing, or incorrect status
- judge summary and failure reasons

## 5. Call The API Directly

Core eval endpoints:

- `GET /api/eval/summary`
- `GET /api/eval/receipts`
- `GET /api/eval/receipts/{sample_id}`
- `GET /api/eval/receipts/{sample_id}/image`
- `GET /api/eval/report`

Examples:

```powershell
curl http://127.0.0.1:7860/api/eval/summary
curl "http://127.0.0.1:7860/api/eval/receipts?status=failed"
curl http://127.0.0.1:7860/api/eval/receipts/sample-1
```

The original environment endpoints are still available:

- `POST /reset`
- `POST /step`
- `GET /state`

## 6. Troubleshooting

If `scripts/evaluate_dataset_images.py` fails immediately:

- check that `MODEL_NAME` and `API_BASE_URL` are set
- check that `EVAL_MODEL` and `EVAL_API_BASE_URL` are set
- check that `OPENAI_API_KEY`, `HF_TOKEN`, or `API_KEY` is available

If `/eval` opens but shows no results:

- confirm `results.jsonl` exists in `RECEIPT_EVAL_OUTPUT_DIR`
- confirm the server is reading the same `RECEIPT_EVAL_OUTPUT_DIR` you used when generating artifacts

If an image does not render in the detail page:

- confirm the dataset image file still exists at the path recorded in `results.jsonl`
- try `GET /api/eval/receipts/{sample_id}/image` directly
