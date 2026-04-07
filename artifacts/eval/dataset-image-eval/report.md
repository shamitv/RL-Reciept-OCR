# Dataset Image Evaluation Report

- Generated at: `2026-04-07T19:00:56.582476Z`
- Dataset root: `D:\work\RL-Reciept-OCR\dataset\Receipt dataset\ds0`
- Output dir: `D:\work\RL-Reciept-OCR\artifacts\eval\dataset-image-eval`
- Expected records: `192`
- Completed records: `192`
- Mean score (non-skipped): `0.728`

## Counts

- Worked: `10`
- Partial: `134`
- Failed: `2`
- Skipped: `46`

## Component Means

- `header_score`: `0.206`
- `summary_score`: `0.146`
- `line_items_score`: `0.269`
- `reconciliation_score`: `0.106`
- `line_item_count_score`: `0.835`

## Line Item Gold Availability

- `with_gold_line_items`: `143`
- `without_gold_line_items`: `49`

## Dataset Audit Status

- `runnable`: `146`
- `skipped_missing_labels`: `38`
- `skipped_unparseable_gold`: `8`

## Top Failure Reasons

- `missing_required_labels`: `38`
- `date_format_mismatch`: `34`
- `missing_line_items`: `16`
- `line_item_merging`: `15`
- `missing_line_item`: `11`
- `unparseable_gold_fields`: `8`
- `incorrect_subtotal`: `7`
- `reconciliation_failure`: `6`
- `company_name_mismatch`: `5`
- `date_mismatch`: `5`

## Worked Records

- `1032-receipt.jpg` score=`1.000` (task=easy; judge=tax_as_line_item, incorrect_subtotal, extra_fields_extracted)
- `1057-receipt.jpg` score=`1.000` (task=hard; reconciliation=pass)
- `1137-receipt.jpg` score=`1.000` (task=hard; reconciliation=pass)
- `1145-receipt.jpg` score=`1.000` (task=hard; reconciliation=pass)
- `1158-receipt.jpg.png` score=`1.000` (task=easy; judge=tax_mismatch, line_item_description_mismatch)
- `1159-receipt.jpg` score=`1.000` (task=hard; reconciliation=pass)
- `1167-receipt.jpg` score=`1.000` (task=hard; reconciliation=pass)
- `1177-receipt.jpg` score=`1.000` (task=hard; reconciliation=pass)
- `1178-receipt.jpg` score=`1.000` (task=easy; judge=description_mismatch, missing_raw_text)
- `1191-receipt.jpg` score=`1.000` (task=hard; reconciliation=pass)

## Partial Records

- `1007-receipt.jpg` score=`0.838` (task=hard; reconciliation=pass; judge=missing_line_items, incorrect_line_item_description, incorrect_line_item_total, date_format_mismatch)
- `1009-receipt.jpg` score=`0.920` (task=hard; reconciliation=pass; judge=date_format_mismatch, address_mismatch_due_to_gold_typos, missing_line_item_metadata)
- `1010-receipt.jpg` score=`0.665` (task=hard; reconciliation=fail; judge=line_item_mismatch, hallucination_relative_to_gold, missing_item, formatting_mismatch)
- `1011-receipt.jpg` score=`0.388` (task=hard; reconciliation=fail; judge=line_item_misalignment, hallucinated_line_items, incorrect_line_item_totals, company_name_mismatch)
- `1014-receipt.jpg` score=`0.885` (task=hard; reconciliation=pass; judge=line_item_merging, line_item_count_mismatch)
- `1015-receipt.jpg` score=`0.781` (task=easy; judge=date_mismatch, missing_line_items, header_field_mismatch)
- `1016-receipt.jpg` score=`0.963` (task=hard; reconciliation=pass)
- `1017-receipt.jpg` score=`0.775` (task=hard; reconciliation=pass; judge=line_item_count_mismatch, line_item_incorrect_grouping)
- `1019-receipt.jpg` score=`0.500` (task=hard; reconciliation=fail; judge=incorrect_subtotal, incorrect_tax, date_format_mismatch, line_item_price_shift)
- `1020-receipt.jpg` score=`0.675` (task=hard; reconciliation=fail; judge=missing_line_item)
- `1021-receipt.jpg` score=`0.832` (task=hard; reconciliation=pass; judge=formatting_discrepancy, incomplete_extraction, description_mismatch)
- `1023-receipt.jpg` score=`0.960` (task=easy; judge=line_item_merging, missing_line_items, description_hallucination)
- `1025-receipt.jpg` score=`0.781` (task=hard; reconciliation=pass; judge=Incorrect date extraction (predicted 2013 instead of 2019), Line item merging (multiple items combined into one description), Missing line item (Special Request was not captured as a separate entity), Spelling errors in line item descriptions (e.g., 'Lettuc' instead of 'Lettuce'))
- `1026-receipt.jpg` score=`0.958` (task=easy; judge=tax_misidentification, line_item_omission, address_truncation)
- `1027-receipt.jpg` score=`0.664` (task=easy; judge=line_item_merging, missing_line_items, incorrect_tax_value, date_format_mismatch)
- `1029-receipt.jpg` score=`0.894` (task=hard; reconciliation=pass; judge=date_format_mismatch)
- `1030-receipt.jpg` score=`0.935` (task=hard; reconciliation=pass; judge=date_format_mismatch, address_typo)
- `1031-receipt.jpg` score=`0.950` (task=hard; reconciliation=pass; judge=date_format_mismatch)
- `1033-receipt.jpg` score=`0.815` (task=hard; reconciliation=pass; judge=company_mismatch, line_item_merging, incorrect_item_count)
- `1036-receipt.jpg` score=`0.930` (task=hard; reconciliation=pass; judge=gold_label_error)
- `1037-receipt.jpg` score=`0.636` (task=hard; reconciliation=fail; judge=Incorrect date extraction (predicted 2013 instead of 2016), Missing line item (the largest item '1 PARTY PK 18-22' was omitted), Incomplete line item extraction (missing raw_text and evidence_ids))
- `1038-receipt.jpg` score=`0.733` (task=easy; judge=field_noise, format_mismatch)
- `1039-receipt.jpg` score=`0.350` (task=hard; reconciliation=fail; judge=Incorrect financial field extraction (subtotal, tax, and total are wrong), Line item value misattribution (line totals are being populated with the receipt total or incorrect values), Failure to correctly parse line item descriptions and prices)
- `1040-receipt.jpg` score=`0.581` (task=hard; reconciliation=fail; judge=Incomplete line item extraction: Two items (the 'LUNCH' descriptors/modifiers) were merged into the main item descriptions, and two items were missing entirely., Address inaccuracy: The predicted address contains incorrect street numbers and zip codes compared to the gold label., Formatting issues: Currency symbols ($) were included in numeric fields (subtotal, tax, total) which deviates from the gold format., Line item count mismatch: The model extracted 5 line items instead of the 7 present in the gold standard.)
- `1042-receipt.jpg` score=`0.358` (task=hard; reconciliation=fail; judge=line_item_splitting, missing_item, over_extraction)

## Failed Records

- `1012-receipt.jpg` score=`0.000` (task=hard; reconciliation=fail; judge=total_extraction_failure, missing_all_header_fields, missing_all_line_items)
- `1022-receipt.jpg` score=`0.000` (task=hard; reconciliation=fail; judge=all_fields_null, no_line_items_extracted)

## Skipped Records

- `1008-receipt.jpg` score=`0.000` (skip=missing_required_labels; judge=missing_required_labels)
- `1013-receipt.jpg` score=`0.000` (skip=missing_required_labels; judge=missing_required_labels)
- `1018-receipt.jpg` score=`0.000` (skip=missing_required_labels; judge=missing_required_labels)
- `1024-receipt.jpg` score=`0.000` (task=easy; skip=unparseable_gold_fields; judge=unparseable_gold_fields)
- `1028-receipt.jpg` score=`0.000` (skip=missing_required_labels; judge=missing_required_labels)
- `1034-receipt.jpg` score=`0.000` (skip=missing_required_labels; judge=missing_required_labels)
- `1035-receipt.jpg` score=`0.000` (task=easy; skip=unparseable_gold_fields; judge=unparseable_gold_fields)
- `1041-receipt.jpg` score=`0.000` (skip=missing_required_labels; judge=missing_required_labels)
- `1044-receipt.jpg` score=`0.000` (skip=missing_required_labels; judge=missing_required_labels)
- `1045-receipt.jpg` score=`0.000` (skip=missing_required_labels; judge=missing_required_labels)
- `1055-receipt.jpg` score=`0.000` (skip=missing_required_labels; judge=missing_required_labels)
- `1056-receipt.jpg` score=`0.000` (task=hard; skip=unparseable_gold_fields; judge=unparseable_gold_fields)
- `1060-receipt.jpg` score=`0.000` (skip=missing_required_labels; judge=missing_required_labels)
- `1064-receipt.jpg` score=`0.000` (skip=missing_required_labels; judge=missing_required_labels)
- `1066-receipt.jpg` score=`0.000` (skip=missing_required_labels; judge=missing_required_labels)
- `1067-receipt.jpg` score=`0.000` (skip=missing_required_labels; judge=missing_required_labels)
- `1072-receipt.jpg` score=`0.000` (skip=missing_required_labels; judge=missing_required_labels)
- `1073-receipt.jpg` score=`0.000` (skip=missing_required_labels; judge=missing_required_labels)
- `1082-receipt.jpg` score=`0.000` (skip=missing_required_labels; judge=missing_required_labels)
- `1087-receipt.jpg` score=`0.000` (skip=missing_required_labels; judge=missing_required_labels)
- `1093-receipt.jpg` score=`0.000` (skip=missing_required_labels; judge=missing_required_labels)
- `1094-receipt.jpg` score=`0.000` (task=hard; skip=unparseable_gold_fields; judge=unparseable_gold_fields)
- `1095-receipt.jpg` score=`0.000` (skip=missing_required_labels; judge=missing_required_labels)
- `1096-receipt.jpg` score=`0.000` (skip=missing_required_labels; judge=missing_required_labels)
- `1097-receipt.jpg` score=`0.000` (skip=missing_required_labels; judge=missing_required_labels)
