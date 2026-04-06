# Plan: Eval API And UI

## Status

- Current state: implemented
- Implemented outputs:
  - `env/eval_api.py`
  - new `/api/eval/*` endpoints in the existing FastAPI app
  - server-rendered UI under `/eval`
  - templates and styles in `server/templates/` and `server/static/`
- Implemented behavior:
  - artifact-backed summary, list, detail, image, and report endpoints
  - per-receipt UI showing image, gold fields, predicted fields, and correct/incorrect field states
  - tests covering API responses, image serving, and UI rendering

## Summary

- Add a new plan doc at `docs/plans/eval/02-eval-api-ui.md`.
- Extend the existing FastAPI app to expose evaluation result APIs backed by the dataset-wide eval artifacts from plan 01.
- Add a lightweight UI served by FastAPI to browse each receipt image, view model output, and see which fields are correct or incorrect.
- Keep the UI stack simple and repo-native: server-rendered HTML plus static assets, not a separate frontend framework.
- Treat the API and UI as read-only inspection tooling for eval results, not as a second execution pipeline.

## Implementation Changes

- Introduce an eval result loader/service that reads:
  - `artifacts/eval/dataset-image-eval/results.jsonl`
  - `artifacts/eval/dataset-image-eval/summary.json`
  - `artifacts/eval/dataset-image-eval/report.md`
- Normalize each record into a stable response shape with:
  - sample id
  - image path
  - eval status: `worked`, `partial`, `failed`, `skipped`
  - gold fields
  - predicted fields
  - per-field deterministic scores
  - overall score
  - judge explanation and failure reasons
- Extend the FastAPI app with a dedicated eval router instead of mixing eval responses into the OpenEnv `/reset`, `/step`, and `/state` endpoints.
- Add a small UI layer served from FastAPI:
  - an index page with summary counts and filters
  - a per-receipt detail page showing the receipt image, gold values, predicted values, and field-by-field correct/incorrect state
  - simple navigation for previous/next receipt and filter-preserving links
- Mount receipt image files as read-only static content so the UI can render the actual dataset image beside the eval output.
- Surface skipped records explicitly in both API and UI so missing labels or unparseable gold data do not disappear from coverage metrics.

## Interfaces

- New API endpoints under `/api/eval`:
  - `GET /api/eval/summary` for aggregate counts and score breakdowns
  - `GET /api/eval/receipts` for paginated receipt records with filter params like `status`, `sample_id`, and `has_errors`
  - `GET /api/eval/receipts/{sample_id}` for one receipt’s full eval payload
  - `GET /api/eval/report` for the generated Markdown report or rendered HTML summary
- New UI routes:
  - `GET /eval` for the summary and searchable receipt list
  - `GET /eval/receipts/{sample_id}` for the detail view
- UI behavior defaults:
  - green for exact/correct fields
  - amber for partial-match fields
  - red for incorrect or missing fields
  - gray for skipped records
- API responses should be derived from the saved eval artifacts, not recomputed live on request.

## Test Plan

- Add API tests for summary, list, and detail endpoints using a small fixture artifact set.
- Verify pagination, filtering, missing-record handling, and skipped-record visibility.
- Add UI response tests to confirm the summary page and detail page render expected receipt data and correctness states.
- Add tests for image/static serving to ensure receipt images referenced by eval artifacts render correctly when present and fail cleanly when absent.
- Keep the test fixture small and deterministic; do not depend on live model calls or a full dataset run.

## Assumptions

- Plan 01 is implemented first and produces the canonical eval artifacts consumed here.
- The API/UI will live in the existing FastAPI app rather than a separate service.
- A simple server-rendered UI is the default path because the repo does not currently contain a frontend build system.
- This phase is read-only: triggering new eval runs from the browser is out of scope unless requested later.
