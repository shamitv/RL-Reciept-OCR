from __future__ import annotations

import base64
import json
from pathlib import Path

from fastapi.testclient import TestClient

from env.evaluation import DatasetAuditRecord, EvalSummary, FieldEvaluation, JudgeEvaluation, ReceiptEvalRecord, ReceiptDraft, render_markdown_report, write_eval_artifacts
from env.server import app


def _write_image_json(root: Path, image_id: str, payload: bytes = b"image") -> Path:
    image_json_dir = root / "img_json"
    image_json_dir.mkdir(parents=True, exist_ok=True)
    path = image_json_dir / f"{image_id}.json"
    path.write_text(
        json.dumps(
            {
                "image_id": image_id,
                "mime_type": "image/jpeg",
                "image_data": base64.b64encode(payload).decode("ascii"),
            }
        ),
        encoding="utf-8",
    )
    return path


def _build_artifacts(output_dir: Path) -> None:
    image_json_path = _write_image_json(output_dir, "sample-1")

    records = [
        ReceiptEvalRecord(
            sample_id="sample-1",
            annotation_path=str(output_dir / "sample-1.json"),
            image_id="sample-1",
            image_json_path=str(image_json_path),
            dataset_status="runnable",
            status="worked",
            gold_fields=ReceiptDraft(company="Store", date="2019-03-25", address="12 Road", total="31.00"),
            predicted_fields=ReceiptDraft(company="Store", date="2019-03-25", address="12 Road", total="31.00"),
            field_results={
                "company": FieldEvaluation(predicted="Store", gold="Store", score=1.0, status="correct"),
                "date": FieldEvaluation(predicted="2019-03-25", gold="2019-03-25", score=1.0, status="correct"),
                "address": FieldEvaluation(predicted="12 Road", gold="12 Road", score=1.0, status="correct"),
                "total": FieldEvaluation(predicted="31.00", gold="31.00", score=1.0, status="correct"),
            },
            overall_score=1.0,
            deterministic_success=True,
            judge=JudgeEvaluation(summary="Perfect extraction"),
            created_at="2026-04-06T00:00:00Z",
        ),
        ReceiptEvalRecord(
            sample_id="sample-2",
            annotation_path=str(output_dir / "sample-2.json"),
            image_id="sample-2",
            image_json_path=None,
            dataset_status="runnable",
            status="failed",
            gold_fields=ReceiptDraft(company="Cafe", date="2019-03-26", address="9 Street", total="10.00"),
            predicted_fields=ReceiptDraft(company=None, date=None, address=None, total=None),
            field_results={
                "company": FieldEvaluation(predicted=None, gold="Cafe", score=0.0, status="missing"),
                "date": FieldEvaluation(predicted=None, gold="2019-03-26", score=0.0, status="missing"),
                "address": FieldEvaluation(predicted=None, gold="9 Street", score=0.0, status="missing"),
                "total": FieldEvaluation(predicted=None, gold="10.00", score=0.0, status="missing"),
            },
            overall_score=0.0,
            deterministic_success=False,
            error="model timeout",
            judge=JudgeEvaluation(summary="Model failed to extract fields", failure_reasons=["runtime_error"]),
            created_at="2026-04-06T00:00:00Z",
        ),
    ]

    summary = EvalSummary(
        generated_at="2026-04-06T00:00:00Z",
        dataset_root="D:/tmp/dataset",
        output_dir=str(output_dir),
        expected_total_records=2,
        completed_records=2,
        counts={"worked": 1, "failed": 1},
        dataset_status_counts={"runnable": 2},
        mean_score=0.5,
        records_with_errors=1,
        field_mean_scores={"company": 0.5, "date": 0.5, "address": 0.5, "total": 0.5},
        top_failure_reasons={"runtime_error": 1},
    )
    write_eval_artifacts(output_dir, records, summary, render_markdown_report(summary, records))


def test_eval_api_and_ui_endpoints(monkeypatch, tmp_path: Path) -> None:
    output_dir = tmp_path / "eval-output"
    _build_artifacts(output_dir)
    monkeypatch.setenv("RECEIPT_EVAL_OUTPUT_DIR", str(output_dir))

    client = TestClient(app)

    summary_response = client.get("/api/eval/summary")
    assert summary_response.status_code == 200
    assert summary_response.json()["expected_total_records"] == 2

    listing_response = client.get("/api/eval/receipts", params={"has_errors": "true"})
    assert listing_response.status_code == 200
    assert listing_response.json()["total"] == 1
    assert listing_response.json()["items"][0]["sample_id"] == "sample-2"

    detail_response = client.get("/api/eval/receipts/sample-1")
    assert detail_response.status_code == 200
    assert detail_response.json()["status"] == "worked"
    assert "line_item_gold_available" in detail_response.json()
    assert "score_explanation" in detail_response.json()
    assert "formula" in detail_response.json()["score_explanation"]

    image_response = client.get("/api/eval/receipts/sample-1/image")
    assert image_response.status_code == 200
    assert image_response.content == b"image"
    assert image_response.headers["cache-control"] == "no-store, max-age=0"

    report_response = client.get("/api/eval/report")
    assert report_response.status_code == 200
    assert "Dataset Image Evaluation Report" in report_response.text

    dashboard_response = client.get("/eval")
    assert dashboard_response.status_code == 200
    assert "Receipt Eval Dashboard" in dashboard_response.text
    assert "sample-1" in dashboard_response.text
    assert "score 1.000" in dashboard_response.text
    assert 'href="/static/eval.css?v=' in dashboard_response.text
    assert "http://testserver/static/eval.css" not in dashboard_response.text

    image_detail_page_response = client.get("/eval/receipts/sample-1")
    assert image_detail_page_response.status_code == 200
    assert "Click to enlarge" in image_detail_page_response.text
    assert "Score Formula" in image_detail_page_response.text

    detail_page_response = client.get("/eval/receipts/sample-2")
    assert detail_page_response.status_code == 200
    assert "Model failed to extract fields" in detail_page_response.text
    assert "Availability And Count" in detail_page_response.text


def test_eval_ui_includes_unprocessed_receipts_and_single_run_action(monkeypatch, tmp_path: Path) -> None:
    output_dir = tmp_path / "eval-output"
    _build_artifacts(output_dir)
    monkeypatch.setenv("RECEIPT_EVAL_OUTPUT_DIR", str(output_dir))
    monkeypatch.setenv("MODEL_NAME", "extractor-test")
    monkeypatch.setenv("API_BASE_URL", "https://extractor.test/v1")
    monkeypatch.setenv("EVAL_MODEL", "judge-test")
    monkeypatch.setenv("EVAL_API_BASE_URL", "https://judge.test/v1")

    sample_1_image_json = output_dir / "img_json" / "sample-1.json"
    extra_image_json = _write_image_json(output_dir, "sample-3", b"image-3")

    audits = [
        DatasetAuditRecord(sample_id="sample-1", annotation_path=str(output_dir / "sample-1.json"), image_id="sample-1", image_json_path=str(sample_1_image_json), dataset_status="runnable", gold_fields=ReceiptDraft(company="Store", date="2019-03-25", address="12 Road", total="31.00")),
        DatasetAuditRecord(sample_id="sample-2", annotation_path=str(output_dir / "sample-2.json"), image_id="sample-2", image_json_path=None, dataset_status="runnable", gold_fields=ReceiptDraft(company="Cafe", date="2019-03-26", address="9 Street", total="10.00")),
        DatasetAuditRecord(sample_id="sample-3", annotation_path=str(output_dir / "sample-3.json"), image_id="sample-3", image_json_path=str(extra_image_json), dataset_status="runnable", gold_fields=ReceiptDraft(company="Bakery", date="2019-03-27", address="5 Lane", total="4.20")),
    ]

    monkeypatch.setattr("env.evaluation.audit_dataset", lambda dataset_root=None: audits)

    def fake_run(sample_id: str):
        return ReceiptEvalRecord(
            sample_id=sample_id,
            annotation_path=str(output_dir / f"{sample_id}.json"),
            image_id=sample_id,
            image_json_path=str(extra_image_json),
            dataset_status="runnable",
            status="worked",
            gold_fields=ReceiptDraft(company="Bakery", date="2019-03-27", address="5 Lane", total="4.20"),
            predicted_fields=ReceiptDraft(company="Bakery", date="2019-03-27", address="5 Lane", total="4.20"),
            field_results={
                "company": FieldEvaluation(predicted="Bakery", gold="Bakery", score=1.0, status="correct"),
                "date": FieldEvaluation(predicted="2019-03-27", gold="2019-03-27", score=1.0, status="correct"),
                "address": FieldEvaluation(predicted="5 Lane", gold="5 Lane", score=1.0, status="correct"),
                "total": FieldEvaluation(predicted="4.20", gold="4.20", score=1.0, status="correct"),
            },
            overall_score=1.0,
            deterministic_success=True,
            judge=JudgeEvaluation(summary="Perfect extraction"),
            created_at="2026-04-06T00:00:00Z",
        )

    monkeypatch.setattr("env.eval_api.evaluate_single_receipt", fake_run)

    client = TestClient(app)

    dashboard_response = client.get("/eval")
    assert dashboard_response.status_code == 200
    assert "sample-3" in dashboard_response.text
    assert "extractor-test" in dashboard_response.text
    assert "judge-test" in dashboard_response.text

    detail_response = client.get("/eval/receipts/sample-3")
    assert detail_response.status_code == 200
    assert "has not been processed by the extraction and judge LLMs yet" in detail_response.text
    assert "Run extraction + judge" in detail_response.text

    api_run_response = client.post("/api/eval/receipts/sample-3/run")
    assert api_run_response.status_code == 200
    assert api_run_response.json()["sample_id"] == "sample-3"

    ui_run_response = client.post("/eval/receipts/sample-3/run", follow_redirects=False)
    assert ui_run_response.status_code == 303
    assert ui_run_response.headers["location"] == "/eval/receipts/sample-3"


def test_eval_ui_falls_back_to_processed_records_when_dataset_missing(monkeypatch, tmp_path: Path) -> None:
    output_dir = tmp_path / "eval-output"
    _build_artifacts(output_dir)
    monkeypatch.setenv("RECEIPT_EVAL_OUTPUT_DIR", str(output_dir))
    monkeypatch.setattr(
        "env.evaluation.audit_dataset",
        lambda dataset_root=None: (_ for _ in ()).throw(FileNotFoundError("dataset missing")),
    )

    client = TestClient(app)

    dashboard_response = client.get("/eval")
    assert dashboard_response.status_code == 200
    assert "sample-1" in dashboard_response.text

    detail_response = client.get("/eval/receipts/sample-1")
    assert detail_response.status_code == 200
    assert "Perfect extraction" in detail_response.text


def test_eval_image_endpoint_404s_for_missing_image_json(monkeypatch, tmp_path: Path) -> None:
    output_dir = tmp_path / "eval-output"
    _build_artifacts(output_dir)
    monkeypatch.setenv("RECEIPT_EVAL_OUTPUT_DIR", str(output_dir))

    records = [
        ReceiptEvalRecord(
            sample_id="missing-image",
            annotation_path=str(output_dir / "missing-image.json"),
            image_id="missing-image",
            image_json_path=str(output_dir / "img_json" / "missing-image.json"),
            dataset_status="skipped_missing_image",
            status="skipped",
            skip_reason="missing_image_json",
            created_at="2026-04-06T00:00:00Z",
        )
    ]
    summary = EvalSummary(
        generated_at="2026-04-06T00:00:00Z",
        dataset_root="D:/tmp/dataset",
        output_dir=str(output_dir),
        expected_total_records=1,
        completed_records=1,
        counts={"skipped": 1},
        dataset_status_counts={"skipped_missing_image": 1},
    )
    write_eval_artifacts(output_dir, records, summary, render_markdown_report(summary, records))

    client = TestClient(app)

    image_response = client.get("/api/eval/receipts/missing-image/image")

    assert image_response.status_code == 404
    assert "image JSON" in image_response.json()["detail"]
