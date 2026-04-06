# Plan: Receipt Reconciliation And Line-Item Metrics

## Status

- Current state: planned
- Current implementation gap:
  - `easy`, `medium`, and `hard` mainly differ by visibility, noise, and budget, not by receipt-specific objectives
  - the environment only grades `company`, `date`, `address`, and `total`
  - there is no metric for subtotal/tax/discount consistency or line-item reconstruction

## Summary

- Keep the public task IDs `easy`, `medium`, and `hard`, but make each one correspond to a more receipt-native objective.
- Expand the environment from header-only extraction into receipt-summary extraction and full line-item reconstruction.
- Add deterministic grading metrics for summary fields, arithmetic reconciliation, and line-item matching.
- Preserve the existing OpenEnv loop and typed models while extending them for richer receipt structure.

## Task Design

- `easy` remains header-focused:
  - extract `company`, `date`, `address`, and `total`
  - keep the current broad reveal, full window access, lowest noise, and largest step budget
- `medium` becomes receipt-summary extraction and reconciliation:
  - extract `subtotal`, `tax`, `discount`, `service_charge`, and `total`
  - continue extracting `company` and `date`; keep `address` optional for medium scoring
  - require the agent to validate whether the extracted monetary fields reconcile to the final total
  - keep the current narrower reveal, no `all` window, moderate noise, and medium step budget
- `hard` becomes full receipt understanding:
  - extract all summary amounts plus line items
  - reconstruct a normalized line-item table with per-row amounts
  - require line-item totals and summary totals to reconcile
  - keep the current restricted reveal, `top`/`bottom`-only windows, highest noise, and smallest step budget

## Data And Model Changes

- Extend `ReceiptDraft` with:
  - `subtotal`
  - `tax`
  - `discount`
  - `service_charge`
  - `line_items`
- Add a new typed model:
  - `ReceiptLineItem(description, quantity, unit_price, line_total)`
- Add task-specific gold metadata on top of the current OCR-region dataset:
  - keep `env/dataset.py` responsible for OCR regions and header fields
  - add a curated overlay file for summary fields and line items keyed by `sample_id`
  - receipts without the required overlay data remain eligible for easier tasks but are excluded from harder task pools
- Define task-specific sample eligibility:
  - `easy`: requires existing header labels only
  - `medium`: requires summary overlay for at least `subtotal` and `total`, plus any present `tax`, `discount`, and `service_charge`
  - `hard`: requires full line-item overlay and summary overlay

## Environment And Action Changes

- Keep the existing `reset()` / `step()` / `state()` contract unchanged.
- Expand `ReceiptObservation` and `ReceiptState` to expose:
  - summary-field candidate lists for the new monetary fields
  - current line-item draft list
  - reconciliation feedback
  - current reconciliation delta or status
- Reuse `query_candidates` / `set_field_from_candidate` for the new summary fields by extending the field-name enum.
- Add typed line-item actions:
  - `query_line_item_candidates`
  - `add_line_item_from_candidate`
  - `update_line_item_field`
  - `remove_line_item`
  - `check_receipt_consistency`
- Keep `submit` as the terminal action for all tasks.

## Grading And Metrics

- Keep grading deterministic and task-aware.
- `easy` score remains header-centric:
  - `company`: 0.20
  - `date`: 0.20
  - `address`: 0.25
  - `total`: 0.35
- `medium` final score uses:
  - `header_score`: 0.15
  - `summary_score`: 0.55
  - `reconciliation_score`: 0.30
- `hard` final score uses:
  - `header_score`: 0.10
  - `summary_score`: 0.25
  - `line_items_score`: 0.45
  - `reconciliation_score`: 0.20

Metric definitions:

- `header_score`:
  - weighted aggregate over `company`, `date`, and optionally `address` depending on task profile
- `summary_score`:
  - weighted aggregate over `subtotal`, `tax`, `discount`, `service_charge`, and `total`
  - exact monetary match after normalization
  - if a gold field is missing, the correct prediction is `None`
- `reconciliation_score`:
  - compute `expected_total = subtotal + tax + service_charge - discount`
  - compare `expected_total` against predicted `total`
  - score `1.0` if absolute difference is `<= 0.01`
  - score `0.5` if absolute difference is `<= 0.05`
  - else score `0.0`
  - for `hard`, also require the summed predicted line-item totals to match predicted `subtotal` within the same tolerance for a full `1.0`
- `line_items_score`:
  - use deterministic bipartite matching between gold and predicted rows
  - row similarity:
    - description token F1: 0.45
    - quantity exact match after normalization: 0.15
    - unit price exact match after normalization: 0.15
    - line total exact match after normalization: 0.25
  - final line-item score = matched similarity sum divided by `max(len(gold_items), len(predicted_items))`

## Reward Changes

- Preserve the current repeated-action and invalid-action penalties.
- Continue giving partial reward when any tracked field improves.
- Add receipt-native shaping:
  - positive reward when a summary field improves over the best prior value
  - positive reward when a new line item is added that increases `line_items_score`
  - positive reward when reconciliation error decreases
  - penalty when an edit makes reconciliation error worse
- Keep all step and terminal rewards clamped to `[-1.0, 1.0]`.

## Documentation And Interfaces

- Update `README.md` to describe the new task objectives, the expanded draft schema, and the new receipt-native metrics.
- Update architecture docs to show:
  - task-specific semantic progression
  - summary-field grading
  - line-item grading
  - reconciliation checks
- Keep `openenv.yaml` task IDs unchanged unless a later migration explicitly chooses new public task names.

## Test Plan

- Add dataset tests for task-specific sample eligibility:
  - header-only receipts appear in `easy`
  - summary-labeled receipts appear in `medium`
  - line-item-labeled receipts appear in `hard`
- Add grader tests for:
  - monetary normalization and exact summary-field scoring
  - reconciliation tolerance thresholds
  - deterministic line-item matching
  - penalties for extra or missing line items
- Add environment tests for:
  - new line-item actions
  - reconciliation feedback generation
  - task-specific observation content
  - harder tasks failing when required metadata is unavailable
- Update inference and PPO tests so observation encoding and action handling reflect the expanded draft and action space.

## Assumptions

- The current task IDs stay `easy`, `medium`, and `hard` to avoid API churn.
- Summary and line-item labels will come from a curated overlay rather than trying to infer everything directly from the base annotation JSON.
- `address` remains part of `easy`, but medium and hard scoring prioritize monetary reasoning and line-item reconstruction over address quality.
- Arithmetic checks use normalized decimal strings with a tolerance of `0.01` for exact consistency and `0.05` for partial consistency.
