from __future__ import annotations

import json
from pathlib import Path
from random import Random

from env.dataset import ReceiptDataset


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


def test_dataset_loads_from_prepared_directory(tmp_path: Path) -> None:
    ann_dir = tmp_path / "ann"
    img_dir = tmp_path / "img"
    ann_dir.mkdir()
    img_dir.mkdir()
    payload = {
        "objects": [
            _annotation_object(1, "Store Name", "Business name", (10, 10, 120, 30)),
            _annotation_object(2, "123 Main St", "Business address", (10, 40, 140, 60)),
            _annotation_object(3, "Springfield", "Business address", (10, 65, 120, 85)),
            _annotation_object(4, "25/03/2019", "Time and date", (10, 90, 110, 110)),
            _annotation_object(5, "TOTAL 31.00", "Total", (10, 200, 120, 220)),
        ]
    }
    annotation_path = ann_dir / "demo-receipt.jpg.json"
    annotation_path.write_text(json.dumps(payload), encoding="utf-8")
    (img_dir / "demo-receipt.jpg").write_bytes(b"image")

    dataset = ReceiptDataset(dataset_root=tmp_path)
    sample = dataset.sample("easy", Random(0))

    assert sample.sample_id == "demo-receipt.jpg"
    assert sample.image_ref == str(img_dir / "demo-receipt.jpg")
    assert sample.gold_fields.company == "Store Name"
    assert sample.gold_fields.address == "123 Main St Springfield"
    assert sample.gold_fields.date == "2019-03-25"
    assert sample.gold_fields.total == "31.00"


def test_dataset_skips_incomplete_records(tmp_path: Path) -> None:
    ann_dir = tmp_path / "ann"
    img_dir = tmp_path / "img"
    ann_dir.mkdir()
    img_dir.mkdir()
    payload = {
        "objects": [
            _annotation_object(1, "Only Name", "Business name", (10, 10, 120, 30)),
            _annotation_object(2, "No Total", "Business address", (10, 40, 140, 60)),
            _annotation_object(3, "25/03/2019", "Time and date", (10, 90, 110, 110)),
        ]
    }
    annotation_path = ann_dir / "incomplete-receipt.jpg.json"
    annotation_path.write_text(json.dumps(payload), encoding="utf-8")
    (img_dir / "incomplete-receipt.jpg").write_bytes(b"image")

    dataset = ReceiptDataset(dataset_root=tmp_path)

    assert all(str(sample.image_ref).startswith("mock://") for sample in dataset.samples)


def test_default_dataset_prefers_real_receipt_samples() -> None:
    dataset = ReceiptDataset()

    assert dataset.samples
    assert any(not str(sample.image_ref).startswith("mock://") for sample in dataset.samples)