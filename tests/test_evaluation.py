from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from env.evaluation import audit_dataset, evaluate_dataset_images, load_results_jsonl


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


def _write_annotation(root: Path, name: str, objects: list[dict], image_name: str | None = None) -> None:
    ann_dir = root / "ann"
    ann_dir.mkdir(parents=True, exist_ok=True)
    path = ann_dir / name
    path.write_text(json.dumps({"objects": objects}), encoding="utf-8")
    if image_name is not None:
        img_dir = root / "img"
        img_dir.mkdir(parents=True, exist_ok=True)
        (img_dir / image_name).write_bytes(b"image")


class _FakeCompletions:
    def __init__(self, responses: list[str]) -> None:
        self.responses = responses
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        index = len(self.calls) - 1
        content = self.responses[index]
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])


class _FakeClient:
    def __init__(self, responses: list[str]) -> None:
        self.completions = _FakeCompletions(responses)
        self.chat = SimpleNamespace(completions=self.completions)


def test_audit_dataset_accounts_for_all_annotation_files_and_skip_reasons(tmp_path: Path) -> None:
    _write_annotation(
        tmp_path,
        "good-receipt.jpg.json",
        [
            _annotation_object(1, "Store Name", "Business name", (10, 10, 120, 30)),
            _annotation_object(2, "123 Main St", "Business address", (10, 40, 140, 60)),
            _annotation_object(3, "Springfield", "Business address", (10, 65, 120, 85)),
            _annotation_object(4, "25/03/2019", "Time and date", (10, 90, 110, 110)),
            _annotation_object(5, "TOTAL 31.00", "Total", (10, 200, 120, 220)),
        ],
        image_name="good-receipt.jpg",
    )
    _write_annotation(
        tmp_path,
        "missing-label-receipt.jpg.json",
        [
            _annotation_object(1, "Store Name", "Business name", (10, 10, 120, 30)),
            _annotation_object(2, "123 Main St", "Business address", (10, 40, 140, 60)),
            _annotation_object(3, "25/03/2019", "Time and date", (10, 90, 110, 110)),
        ],
        image_name="missing-label-receipt.jpg",
    )
    _write_annotation(
        tmp_path,
        "odd-receipt.jpg.png.json",
        [
            _annotation_object(1, "Cafe", "Business name", (10, 10, 120, 30)),
            _annotation_object(2, "1 Example Road", "Business address", (10, 40, 160, 60)),
            _annotation_object(3, "Apr 01, 2019", "Time and date", (10, 90, 150, 110)),
            _annotation_object(4, "TOTAL 8.50", "Total", (10, 200, 120, 220)),
        ],
        image_name="odd-receipt.jpg.png",
    )
    _write_annotation(
        tmp_path,
        "missing-image-receipt.jpg.json",
        [
            _annotation_object(1, "Shop", "Business name", (10, 10, 120, 30)),
            _annotation_object(2, "10 Street", "Business address", (10, 40, 140, 60)),
            _annotation_object(3, "25/03/2019", "Time and date", (10, 90, 110, 110)),
            _annotation_object(4, "TOTAL 31.00", "Total", (10, 200, 120, 220)),
        ],
        image_name=None,
    )

    records = audit_dataset(tmp_path)

    assert len(records) == 4
    statuses = {record.sample_id: record.dataset_status for record in records}
    assert statuses["good-receipt.jpg"] == "runnable"
    assert statuses["missing-label-receipt.jpg"] == "skipped_missing_labels"
    assert statuses["odd-receipt.jpg.png"] == "skipped_unparseable_gold"
    assert statuses["missing-image-receipt.jpg"] == "skipped_missing_image"


def test_evaluate_dataset_images_writes_artifacts_and_respects_resume(monkeypatch, tmp_path: Path) -> None:
    _write_annotation(
        tmp_path,
        "good-receipt.jpg.json",
        [
            _annotation_object(1, "Store Name", "Business name", (10, 10, 120, 30)),
            _annotation_object(2, "123 Main St", "Business address", (10, 40, 140, 60)),
            _annotation_object(3, "Springfield", "Business address", (10, 65, 120, 85)),
            _annotation_object(4, "25/03/2019", "Time and date", (10, 90, 110, 110)),
            _annotation_object(5, "TOTAL 31.00", "Total", (10, 200, 120, 220)),
        ],
        image_name="good-receipt.jpg",
    )
    _write_annotation(
        tmp_path,
        "missing-label-receipt.jpg.json",
        [
            _annotation_object(1, "Store Name", "Business name", (10, 10, 120, 30)),
            _annotation_object(2, "123 Main St", "Business address", (10, 40, 140, 60)),
            _annotation_object(3, "25/03/2019", "Time and date", (10, 90, 110, 110)),
        ],
        image_name="missing-label-receipt.jpg",
    )
    output_dir = tmp_path / "artifacts"

    monkeypatch.setenv("MODEL_NAME", "extractor-model")
    monkeypatch.setenv("API_BASE_URL", "https://extractor.example/v1")
    monkeypatch.setenv("EVAL_MODEL", "judge-model")
    monkeypatch.setenv("EVAL_API_BASE_URL", "https://judge.example/v1")

    extractor_client = _FakeClient(['{"company":"Store Name","date":"2019-03-25","address":"123 Main St Springfield","total":"31.00"}'])
    judge_client = _FakeClient(['{"summary":"Perfect extraction","failure_reasons":[],"field_notes":{"total":"Matched exactly."}}'])

    result = evaluate_dataset_images(
        dataset_root=tmp_path,
        output_dir=output_dir,
        extractor_client=extractor_client,
        judge_client=judge_client,
    )

    records = load_results_jsonl(output_dir)
    assert result.processed_records == 2
    assert result.expected_total_records == 2
    assert len(records) == 2
    assert records[0].status == "worked"
    assert records[1].status == "skipped"
    assert (output_dir / "summary.json").exists()
    assert (output_dir / "report.md").exists()
    assert extractor_client.completions.calls
    assert judge_client.completions.calls

    resume_result = evaluate_dataset_images(
        dataset_root=tmp_path,
        output_dir=output_dir,
        resume=True,
        extractor_client=extractor_client,
        judge_client=judge_client,
    )

    assert resume_result.processed_records == 0
