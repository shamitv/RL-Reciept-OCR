# Plan: Receipt Reconciliation And Line-Item Metrics

## Status

- Current state: implemented
- Validation status:
  - `pytest` passes with the receipt-native task rewrite in place
  - `openenv validate` passes after the environment, eval API, and UI updates

## Implemented Summary

- Kept the public task IDs `easy`, `medium`, and `hard`.
- Reworked them into semantically distinct receipt tasks instead of three variants of the same four-field header extraction.
- Extended the environment, grading, rewards, PPO runtime, eval artifacts, API payloads, and dashboard UI for receipt-summary and line-item metrics.

## Implemented Task Design

- `easy`:
  - extracts `company`, `date`, `address`, and `total`
  - keeps the easiest reveal policy, widest window access, lowest reranking noise, and largest budget
- `medium`:
  - extracts `company`, `date`, `subtotal`, `tax`, and `total`
  - grades deterministic reconciliation for `subtotal + tax ~= total`
  - keeps the medium reveal policy, no `all` window, moderate reranking noise, and medium budget
- `hard`:
  - extracts `company`, `date`, `subtotal`, `tax`, `total`, and line items
  - grades deterministic reconciliation for both `subtotal + tax ~= total` and `sum(line_items) ~= subtotal`
  - keeps the most restricted reveal policy, `top`/`bottom` windows only, highest reranking noise, and smallest budget

## Implemented Data And Model Changes

- Extended `ReceiptDraft` with:
  - `subtotal`
  - `tax`
  - `line_items`
- Added typed models for:
  - `ReceiptLineItem`
  - `ReceiptLineItemCandidate`
- Extended `ReceiptSample` with `gold_line_items`.
- Used the existing dataset categories directly in v1:
  - `Subtotal`
  - `Tax`
  - `Item information`
- Did not add an overlay file in v1.
- Built task-specific sample pools directly from parsed annotations:
  - `easy`: header labels present and parseable
  - `medium`: summary labels present and parseable
  - `hard`: summary labels present and parseable plus at least one parseable line item

## Implemented Environment And Action Changes

- Kept the `reset()` / `step()` / `state()` contract unchanged.
- Expanded observations and state with:
  - candidate lists for `subtotal` and `tax`
  - line-item candidates
  - current draft line items
  - reconciliation feedback
  - reconciliation delta and status
- Reused scalar-field actions for `subtotal` and `tax`.
- Added typed line-item and reconciliation actions:
  - `query_line_item_candidates`
  - `add_line_item_from_candidate`
  - `remove_line_item`
  - `check_receipt_consistency`
- Updated the heuristic agent and PPO runtime so they can operate on the expanded field and action space.

## Implemented Grading And Metrics

- `easy` scoring:
  - `company`: 0.20
  - `date`: 0.20
  - `address`: 0.25
  - `total`: 0.35
- `medium` scoring:
  - `header_score`: 0.20 from `company` and `date`
  - `summary_score`: 0.55 from `subtotal`, `tax`, and `total`
  - `reconciliation_score`: 0.25
- `hard` scoring:
  - `header_score`: 0.10 from `company` and `date`
  - `summary_score`: 0.20 from `subtotal`, `tax`, and `total`
  - `line_items_score`: 0.45
  - `reconciliation_score`: 0.25

Metric details:

- `summary_score` uses exact normalized amount matching.
- `reconciliation_score` uses:
  - `1.0` when delta `<= 0.01`
  - `0.5` when delta `<= 0.05`
  - `0.0` otherwise
- `line_items_score` uses deterministic greedy row matching over:
  - description token overlap
  - exact normalized `line_total` when parseable

## Implemented Reward Changes

- Preserved repeated-action and invalid-action penalties.
- Added reward signal for:
  - improved `subtotal` and `tax`
  - improved line-item score
  - improved reconciliation score
- Kept rewards clamped to `[-1.0, 1.0]`.

## Implemented API And Dashboard Updates

- Extended eval artifacts and API payloads with:
  - `header_score`
  - `summary_score`
  - `line_items_score`
  - `reconciliation_score`
  - `reconciliation_delta`
  - `reconciliation_status`
  - predicted vs. gold line-item rows
- Updated the dashboard/detail UI to show:
  - overall score plus component scores
  - summary-field correctness for `subtotal`, `tax`, and `total`
  - reconciliation status and numeric delta
  - line-item comparison tables
- Kept the UI backward-compatible with older artifact sets by relying on optional/defaulted fields.

## Follow-On Work

- `discount` and `service_charge` remain deferred because the current dataset does not label them reliably enough for v1.
- PPO training is still a separate follow-on implementation; this work updated environment semantics and learned-policy inference compatibility, not PPO optimization.
