# Plan: Dataset-Wide Image Eval With LLM Validation

## Status

- Current state: implemented
- Implemented outputs:
  - `scripts/evaluate_dataset_images.py`
  - `env/evaluation.py`
  - `artifacts/eval/dataset-image-eval/{results.jsonl,summary.json,report.md}` output contract
- Implemented behavior:
  - dataset-wide audit over every annotation file
  - resumable JSONL artifact generation
  - deterministic grading plus larger-model validation notes
  - tests covering audit coverage, skip reasons, and artifact generation

## Summary

- Add a new plan doc at `docs/plans/eval/01-dataset-image-eval.md`.
- Implement a separate evaluator script at `scripts/evaluate_dataset_images.py`; do not repurpose `inference.py`, which should stay as the OpenEnv baseline entrypoint.
- Make the evaluator walk every annotation/image pair in the dataset root, not just `ReceiptDataset.samples`, so it can report both evaluated and skipped records.
- Load `.env` at startup with `env.config.load_environment()`.
- Use deterministic grading as the source of truth and a larger model as a hybrid validator/explainer.
- Emit a machine-readable result set plus a Markdown report that clearly shows what worked, what partially worked, what failed, and what was skipped.

## Implementation Changes

- Add a raw dataset audit/enumeration helper that returns one record per annotation file with status:
  - `runnable`
  - `skipped_missing_labels`
  - `skipped_unparseable_gold`
  - `skipped_missing_image`
- Reuse the current normalizers and `grade_receipt()` for gold-field comparison so the evaluator matches repo grading rules.
- For each `runnable` record:
  - run the primary extraction model against the image using the existing inference config
  - parse a strict four-field JSON result: `company`, `date`, `address`, `total`
  - compute deterministic field scores and overall score
  - call a separate OpenAI-compatible judge model with the image, prediction, gold fields, and deterministic scores to explain success/failure and tag likely failure reasons
- Write incremental output so long runs can resume:
  - `artifacts/eval/dataset-image-eval/results.jsonl`
  - `artifacts/eval/dataset-image-eval/summary.json`
  - `artifacts/eval/dataset-image-eval/report.md`
- Report classifications:
  - `worked`: normalized exact match on all four fields, score `1.0`
  - `partial`: score `> 0` and `< 1.0`
  - `failed`: parse failure, empty extraction, or score `0`
  - `skipped`: record could not be evaluated, with explicit reason
- Current repo data should be treated as a coverage requirement: there are 192 annotation/image pairs, but only 92 are loader-valid today, so the new evaluator must account for the full 192 instead of silently shrinking to the loader subset.

## Interfaces

- New CLI:
  - `python scripts/evaluate_dataset_images.py --dataset-root ... --output-dir artifacts/eval/dataset-image-eval`
  - support `--limit` and `--resume` for debugging and interrupted runs
- Env vars:
  - primary extractor keeps using existing repo config: `MODEL_NAME` and `API_BASE_URL`
  - eval judge requires `EVAL_MODEL`
  - eval judge requires `EVAL_API_BASE_URL`
  - API key continues to come from existing key env vars already used by the repo
- Update `.env.example` and README to document the new eval-only config and usage.

## Test Plan

- Add unit tests proving raw dataset enumeration accounts for every annotation file and produces explicit skip reasons.
- Cover records with required labels missing, records with labels present but unparseable dates/totals, and odd filenames like `1158-receipt.jpg.png`.
- Mock the OpenAI client for extractor and judge calls to verify env loading, request payloads, JSON parsing, and failure handling.
- Add report-generation tests to verify stable counts and the presence of worked/failed/skipped sections.
- Keep live model calls out of CI; document one manual full-run command for local validation.

## Assumptions

- `MODEL_NAME` and `API_BASE_URL` remain the primary extraction-model config; only validation gets separate `EVAL_*` variables.
- The large model is a hybrid validator and explainer, not the pass/fail authority; deterministic grading remains canonical.
- The implementation should fail fast with a clear message if `EVAL_MODEL` or `EVAL_API_BASE_URL` is missing.
