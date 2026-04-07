# Receipt Score Examples Walkthrough

This walkthrough documents representative examples from the selected 50-image inference set. The goal is to show how final score changes across receipts where totals, line-item counts, and reconciliation checks behave differently.

Source data:

- Manifest: `artifacts/datasets/receipt-selection-50/selected_manifest.json`
- Sample list: `artifacts/datasets/receipt-selection-50/selected_samples.csv`
- Image JSON assets: `artifacts/datasets/receipt-selection-50/dataset/img_json/`

Scoring terms:

- `score`: final environment score for the receipt.
- `total_score`: whether the extracted total field matched the annotation.
- `line_items_score`: line-item extraction component of the score. In the hard task examples below, this component is capped at `0.45`.
- `summary reconciliation`: whether predicted `subtotal + tax` matches predicted `total`.
- `line-item reconciliation`: whether predicted line-item totals reconcile with predicted subtotal.
- `reconciliation_status`: combined reconciliation status used for the hard task.

## Spectrum Summary

| Range | Image | Task | Score | Total score | Line items | Summary recon | Line-item recon | What it shows |
| --- | --- | --- | ---: | ---: | --- | --- | --- | --- |
| High | [1159-receipt.jpg.json](../../artifacts/datasets/receipt-selection-50/dataset/img_json/1159-receipt.jpg.json) | hard | 1.000 | 1.0 | 1 -> 1 | pass | pass | Simple receipt where fields, one line item, and totals all match. |
| High | [1177-receipt.jpg.json](../../artifacts/datasets/receipt-selection-50/dataset/img_json/1177-receipt.jpg.json) | hard | 1.000 | 1.0 | 4 -> 4 | pass | pass | Multi-item receipt where item count and item totals reconcile cleanly. |
| Medium | [1124-receipt.jpg.json](../../artifacts/datasets/receipt-selection-50/dataset/img_json/1124-receipt.jpg.json) | hard | 0.750 | 1.0 | 11 -> 11 | pass | fail | Count and summary totals match, but some line-item amounts are missing. |
| Medium | [1183-receipt.jpg.json](../../artifacts/datasets/receipt-selection-50/dataset/img_json/1183-receipt.jpg.json) | hard | 0.750 | 1.0 | 2 -> 2 | fail | pass | Extracted fields match, but subtotal plus tax does not equal the printed total. |
| Lower | [1019-receipt.jpg.json](../../artifacts/datasets/receipt-selection-50/dataset/img_json/1019-receipt.jpg.json) | hard | 0.500 | 1.0 | 9 -> 9 | fail | fail | Count matches, but amount alignment and summary fields are wrong. |
| Low | [1058-receipt.jpg.json](../../artifacts/datasets/receipt-selection-50/dataset/img_json/1058-receipt.jpg.json) | hard | 0.317 | 0.0 | 5 -> 4 | pass | fail | Predicted values reconcile internally, but not against the receipt labels. |
| Worst | [1022-receipt.jpg.json](../../artifacts/datasets/receipt-selection-50/dataset/img_json/1022-receipt.jpg.json) | easy | 0.000 | 0.0 | 1 -> 0 | n/a | n/a | Empty extraction for an easy-task receipt. |

## High Score Examples

### 1159-receipt.jpg

This Taco Time receipt is a clean single-item success case. The model extracted the company, date, address, subtotal, tax, and total correctly after normalization.

Key checks:

- Gold line items: `1`; predicted line items: `1`; delta: `0`.
- Gold item: `SFT TACO BF`, `5.99`; predicted item: `SFT TACO BF`, `5.99`.
- Predicted summary fields reconcile: `5.99 + 0.60 = 6.59`.
- Line-item total reconciles with subtotal: `5.99 = 5.99`.
- Final score: `1.000`.

### 1177-receipt.jpg

This El Charro Cafe Ventana receipt is a stronger high-score example because it has four line items, not just one.

Key checks:

- Gold line items: `4`; predicted line items: `4`; delta: `0`.
- Items match the annotation: `Coors Light 5.00`, `Ice Tea 3.25`, `Shrimp Fajita 18.95`, and `Stuffed Chicken 17.95`.
- Predicted line items sum to subtotal: `5.00 + 3.25 + 18.95 + 17.95 = 45.15`.
- Predicted summary fields reconcile: `45.15 + 2.75 = 47.90`.
- Final score: `1.000`.

## Medium Score Examples

### 1124-receipt.jpg

This Firefly American Bistro receipt shows that matching the line-item count is not enough by itself. The model found all 11 line-item rows and matched the summary fields, but several modifier rows had missing line totals.

Key checks:

- Gold line items: `11`; predicted line items: `11`; delta: `0`.
- `total_score`: `1.0`.
- Summary reconciliation passes: predicted subtotal, tax, and total are internally consistent.
- Line-item reconciliation fails because extracted modifiers such as `Add One Meatball`, `Add Two Meatball`, and `Add Steak` have null line totals.
- Final score: `0.750`.

### 1183-receipt.jpg

This JALISCO receipt is useful because the line items are correct, but the summary arithmetic does not reconcile with the printed total.

Key checks:

- Gold line items: `2`; predicted line items: `2`; delta: `0`.
- Items match: `Water 0.00` and `Caldo de Camaron 8.99`.
- Line-item reconciliation passes: `0.00 + 8.99 = 8.99`.
- Summary reconciliation fails: `8.99 + 0.74 = 9.73`, while the receipt total is `10.00`.
- Final score: `0.750`.

## Lower Score Examples

### 1019-receipt.jpg

This Grotto Pizzeria & Tavern receipt shows a common failure mode where the line-item count matches but amounts are shifted across rows.

Key checks:

- Gold line items: `9`; predicted line items: `9`; delta: `0`.
- `total_score`: `1.0`, because the final total `45.58` was extracted correctly.
- The date was extracted as `5/12/2017`, but the annotation date is `2017-12-05`.
- Predicted subtotal and tax are wrong: `25.00` and `43.00`, while the annotation has subtotal `43.00` and tax `12.58`.
- Summary reconciliation fails with delta `22.42`.
- Line-item reconciliation fails with delta `23.00`.
- Final score: `0.500`.

### 1058-receipt.jpg

This Caldera Brewery receipt is useful because the predicted fields reconcile internally, but they do not match the receipt labels.

Key checks:

- Gold line items: `5`; predicted line items: `4`; delta: `1`.
- `total_score`: `0.0`, because predicted total `43.50` does not match annotated total `37.60`.
- Predicted summary reconciliation passes internally: `36.50 + 7.00 = 43.50`.
- Line-item reconciliation fails with delta `15.00`.
- The model merged or shifted items, such as combining beer rows and assigning incorrect amounts to food rows.
- Final score: `0.317`.

### 1022-receipt.jpg

This Moonstar Restaurant receipt is the low-end easy-task example. It is useful as a smoke-test case for empty or failed extraction output.

Key checks:

- Gold company: `Moonstar Restaurant`; predicted company: `null`.
- Gold total: `37.37`; predicted total: `null`.
- Gold line items: `1`; predicted line items: `0`; delta: `1`.
- Reconciliation is not evaluated in this easy-task example.
- Final score: `0.000`.

## Debugging Takeaways

- A perfect or high score needs both field correctness and reconciliation, especially on hard tasks.
- Matching line-item count is useful but insufficient; amount alignment matters.
- Passing summary reconciliation only proves the predicted fields are internally consistent. It does not prove they match the receipt.
- A receipt can have correct extracted fields but still lose reconciliation credit if the printed subtotal, tax, and total do not add up exactly.
- Low-score examples are useful regression tests for null output, merged item rows, shifted amounts, and date normalization errors.
