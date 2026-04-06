from __future__ import annotations

import json
from pathlib import Path

from env.dataset import ReceiptDataset
from env.environment import ReceiptExtractionEnv
from env.graders import grade_receipt
from env.models import OCRRegion, ReceiptAction, ReceiptDraft, ReceiptLineItem, ReceiptSample


def _annotation_object(object_id: int, transcription: str, category: str, bbox: tuple[int, int, int, int]) -> dict:
    left, top, right, bottom = bbox
    return {
        "id": object_id,
        "tags": [
            {"name": "Transcription", "value": transcription},
            {"name": "Category", "value": category},
        ],
        "points": {"exterior": [[float(left), float(top)], [float(right), float(bottom)]], "interior": []},
    }


def test_dataset_extracts_summary_fields_and_line_items_for_hard_pool(tmp_path: Path) -> None:
    ann_dir = tmp_path / "ann"
    img_dir = tmp_path / "img"
    ann_dir.mkdir()
    img_dir.mkdir()
    payload = {
        "objects": [
            _annotation_object(1, "Demo Cafe", "Business name", (10, 10, 120, 30)),
            _annotation_object(2, "25/03/2019", "Time and date", (10, 40, 120, 60)),
            _annotation_object(3, "LATTE 4.50", "Item information", (10, 110, 140, 130)),
            _annotation_object(4, "MUFFIN 3.50", "Item information", (10, 135, 140, 155)),
            _annotation_object(5, "SUBTOTAL 8.00", "Subtotal", (10, 190, 140, 210)),
            _annotation_object(6, "TAX 0.48", "Tax", (10, 210, 140, 230)),
            _annotation_object(7, "TOTAL 8.48", "Total", (10, 235, 140, 255)),
        ]
    }
    annotation_path = ann_dir / "hard-receipt.jpg.json"
    annotation_path.write_text(json.dumps(payload), encoding="utf-8")
    (img_dir / "hard-receipt.jpg").write_bytes(b"image")

    dataset = ReceiptDataset(dataset_root=tmp_path)

    assert dataset.eligible_task_counts()["hard"] == 1
    sample = dataset.samples_by_task["hard"][0]
    assert sample.gold_fields.subtotal == "8.00"
    assert sample.gold_fields.tax == "0.48"
    assert [item.description for item in sample.gold_line_items] == ["LATTE", "MUFFIN"]


def test_grade_receipt_reconciliation_and_line_item_scores() -> None:
    gold = ReceiptDraft(company="Demo Cafe", date="2019-03-25", subtotal="8.00", tax="0.48", total="8.48")
    gold_items = [
        ReceiptLineItem(description="LATTE", line_total="4.50"),
        ReceiptLineItem(description="MUFFIN", line_total="3.50"),
    ]

    exact = grade_receipt(
        ReceiptDraft(
            company="Demo Cafe",
            date="2019-03-25",
            subtotal="8.00",
            tax="0.48",
            total="8.48",
            line_items=gold_items,
        ),
        gold,
        task_id="hard",
        gold_line_items=gold_items,
    )
    partial = grade_receipt(
        ReceiptDraft(company="Demo Cafe", date="2019-03-25", subtotal="8.00", tax="0.50", total="8.48"),
        gold,
        task_id="medium",
        gold_line_items=[],
    )

    assert exact.reconciliation_status == "pass"
    assert exact.line_items_score == 0.45
    assert exact.line_item_gold_available is True
    assert exact.gold_line_item_count == 2
    assert exact.predicted_line_item_count == 2
    assert exact.line_item_count_delta == 0
    assert exact.line_item_count_score == 1.0
    assert partial.reconciliation_status == "partial"
    assert partial.reconciliation_score == 0.125
    assert partial.line_item_reconciliation_status == "not_evaluated"
    assert partial.line_item_count_score is None


def test_grade_receipt_marks_line_item_metrics_not_evaluated_without_gold_rows() -> None:
    grade = grade_receipt(
        ReceiptDraft(
            company="Demo Cafe",
            date="2019-03-25",
            subtotal="8.00",
            tax="0.48",
            total="8.48",
            line_items=[ReceiptLineItem(description="LATTE", line_total="4.50")],
        ),
        ReceiptDraft(company="Demo Cafe", date="2019-03-25", subtotal="8.00", tax="0.48", total="8.48"),
        task_id="hard",
        gold_line_items=[],
    )

    assert grade.line_item_gold_available is False
    assert grade.line_item_count_score is None
    assert grade.line_item_reconciliation_status == "not_evaluated"
    assert grade.reconciliation_status == "pass"


def test_environment_hard_line_item_actions_and_reconciliation_feedback() -> None:
    env = ReceiptExtractionEnv()
    sample = ReceiptSample(
        sample_id="synthetic-hard",
        image_ref="synthetic://receipt",
        regions=[
            OCRRegion(region_id="r1", text="Demo Cafe", bbox=(0, 10, 120, 30)),
            OCRRegion(region_id="r2", text="25/03/2019", bbox=(0, 40, 120, 60)),
            OCRRegion(region_id="r3", text="LATTE 4.50", bbox=(0, 155, 120, 175)),
            OCRRegion(region_id="r4", text="MUFFIN 3.50", bbox=(0, 180, 120, 200)),
            OCRRegion(region_id="r5", text="SUBTOTAL 8.00", bbox=(0, 190, 120, 210)),
            OCRRegion(region_id="r6", text="TAX 0.48", bbox=(0, 205, 120, 225)),
            OCRRegion(region_id="r7", text="TOTAL 8.48", bbox=(0, 225, 120, 245)),
        ],
        gold_fields=ReceiptDraft(company="Demo Cafe", date="2019-03-25", subtotal="8.00", tax="0.48", total="8.48"),
        gold_line_items=[
            ReceiptLineItem(description="LATTE", line_total="4.50"),
            ReceiptLineItem(description="MUFFIN", line_total="3.50"),
        ],
    )
    env.dataset.sample = lambda task_name, rng: sample.model_copy(deep=True)
    env.reset(task_name="hard", seed=3)
    env.task.max_steps = 20
    env.hidden_state.remaining_budget = 20

    env.step(ReceiptAction(action_type="view_receipt"))
    env.step(ReceiptAction(action_type="list_text_regions", window="bottom"))
    env.step(ReceiptAction(action_type="query_candidates", field="subtotal"))
    env.step(
        ReceiptAction(
            action_type="set_field_from_candidate",
            field="subtotal",
            candidate_id=env.last_observation.candidate_lists["subtotal"][0].candidate_id,
        )
    )
    env.step(ReceiptAction(action_type="query_candidates", field="tax"))
    env.step(
        ReceiptAction(
            action_type="set_field_from_candidate",
            field="tax",
            candidate_id=env.last_observation.candidate_lists["tax"][0].candidate_id,
        )
    )
    env.step(ReceiptAction(action_type="query_candidates", field="total"))
    env.step(
        ReceiptAction(
            action_type="set_field_from_candidate",
            field="total",
            candidate_id=env.last_observation.candidate_lists["total"][0].candidate_id,
        )
    )
    env.step(ReceiptAction(action_type="query_line_item_candidates"))
    add_result = env.step(
        ReceiptAction(
            action_type="add_line_item_from_candidate",
            candidate_id=env.last_observation.line_item_candidates[0].candidate_id,
        )
    )
    check_result = env.step(ReceiptAction(action_type="check_receipt_consistency"))

    assert add_result.observation.current_draft.line_items
    assert check_result.observation.current_reconciliation_status in {"pass", "partial", "fail"}
    assert check_result.observation.reconciliation_feedback
