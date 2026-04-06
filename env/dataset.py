from __future__ import annotations

import json
import os
from random import Random
from pathlib import Path

from env.models import OCRRegion, ReceiptDraft, ReceiptSample
from env.normalizers import normalize_amount, normalize_date

MOCK_SAMPLES = [
    ReceiptSample(
        sample_id="sample_easy_1",
        image_ref="mock://receipt/1",
        regions=[
            OCRRegion(region_id="r1", text="MART CORNER SDN BHD", bbox=(0, 0, 200, 20)),
            OCRRegion(region_id="r2", text="12 JALAN BINTANG", bbox=(0, 30, 220, 50)),
            OCRRegion(region_id="r3", text="50000 KUALA LUMPUR", bbox=(0, 55, 220, 75)),
            OCRRegion(region_id="r4", text="25/03/2019", bbox=(0, 90, 120, 110)),
            OCRRegion(region_id="r5", text="TOTAL 31.00", bbox=(0, 220, 140, 240)),
        ],
        gold_fields=ReceiptDraft(company="MART CORNER SDN BHD", address="12 JALAN BINTANG 50000 KUALA LUMPUR", date="2019-03-25", total="31.00"),
    ),
    ReceiptSample(
        sample_id="sample_medium_1",
        image_ref="mock://receipt/2",
        regions=[
            OCRRegion(region_id="r1", text="FRESH HUB MARKET", bbox=(0, 0, 200, 20)),
            OCRRegion(region_id="r2", text="88 JLN MERDEKA", bbox=(0, 28, 200, 48)),
            OCRRegion(region_id="r3", text="SHAH ALAM SELANGOR", bbox=(0, 50, 220, 70)),
            OCRRegion(region_id="r4", text="26-03-19", bbox=(0, 92, 100, 112)),
            OCRRegion(region_id="r5", text="SUBTOTAL 19.90", bbox=(0, 180, 140, 200)),
            OCRRegion(region_id="r6", text="TOTAL DUE 21.10", bbox=(0, 220, 160, 240)),
        ],
        gold_fields=ReceiptDraft(company="FRESH HUB MARKET", address="88 JLN MERDEKA SHAH ALAM SELANGOR", date="2019-03-26", total="21.10"),
    ),
    ReceiptSample(
        sample_id="sample_hard_1",
        image_ref="mock://receipt/3",
        regions=[
            OCRRegion(region_id="r1", text="CITY CAFE", bbox=(0, 0, 120, 20)),
            OCRRegion(region_id="r2", text="LOT 9 KOMPLEKS NIAGA", bbox=(0, 30, 220, 50)),
            OCRRegion(region_id="r3", text="PETALING JAYA", bbox=(0, 55, 180, 75)),
            OCRRegion(region_id="r4", text="2019-03-27", bbox=(0, 88, 100, 108)),
            OCRRegion(region_id="r5", text="CASH 12.00", bbox=(0, 180, 120, 200)),
            OCRRegion(region_id="r6", text="TOTAL 12.00", bbox=(0, 215, 120, 235)),
        ],
        gold_fields=ReceiptDraft(company="CITY CAFE", address="LOT 9 KOMPLEKS NIAGA PETALING JAYA", date="2019-03-27", total="12.00"),
    ),
]


class ReceiptDataset:
    def __init__(self, dataset_root: str | Path | None = None) -> None:
        self.dataset_root = self._resolve_dataset_root(dataset_root)
        self.samples = self._load_samples()
        self.samples_by_difficulty = self._bucket_by_difficulty(self.samples)

    def sample(self, difficulty: str, rng: Random) -> ReceiptSample:
        candidates = self.samples_by_difficulty.get(difficulty, [])
        if not candidates:
            candidates = self.samples
        index = rng.randrange(len(candidates))
        return candidates[index].model_copy(deep=True)

    def _resolve_dataset_root(self, dataset_root: str | Path | None) -> Path:
        if dataset_root is not None:
            return Path(dataset_root)
        configured_root = os.getenv("RECEIPT_DATASET_ROOT")
        if configured_root:
            return Path(configured_root)
        return Path(__file__).resolve().parents[1] / "dataset" / "Receipt dataset" / "ds0"

    def _load_samples(self) -> list[ReceiptSample]:
        ann_dir = self.dataset_root / "ann"
        img_dir = self.dataset_root / "img"
        if not ann_dir.exists() or not img_dir.exists():
            return MOCK_SAMPLES

        samples: list[ReceiptSample] = []
        for annotation_path in sorted(ann_dir.glob("*.json")):
            sample = self._parse_annotation(annotation_path, img_dir)
            if sample is not None:
                samples.append(sample)
        return samples or MOCK_SAMPLES

    def _parse_annotation(self, annotation_path: Path, image_dir: Path) -> ReceiptSample | None:
        payload = json.loads(annotation_path.read_text(encoding="utf-8"))
        objects = payload.get("objects", [])
        grouped_regions: dict[str, list[OCRRegion]] = {}
        visible_regions: list[OCRRegion] = []

        for obj in objects:
            region = self._build_region(obj)
            if region is None:
                continue
            visible_regions.append(region)
            category = self._tag_value(obj, "Category")
            if category:
                grouped_regions.setdefault(category, []).append(region)

        company = self._join_text(grouped_regions.get("Business name", []))
        address = self._join_text(grouped_regions.get("Business address", []))
        date = self._pick_date(grouped_regions.get("Time and date", []))
        total = self._pick_total(grouped_regions.get("Total", []))
        if not company or not address or not date or not total:
            return None

        image_name = annotation_path.name[:-5]
        image_path = image_dir / image_name
        if not image_path.exists():
            return None

        return ReceiptSample(
            sample_id=annotation_path.stem,
            image_ref=str(image_path),
            regions=sorted(visible_regions, key=lambda region: (region.bbox[1], region.bbox[0], region.region_id)),
            gold_fields=ReceiptDraft(company=company, address=address, date=date, total=total),
        )

    def _build_region(self, obj: dict) -> OCRRegion | None:
        transcription = self._tag_value(obj, "Transcription")
        points = obj.get("points", {}).get("exterior", [])
        if not transcription or len(points) != 2:
            return None
        (x1, y1), (x2, y2) = points
        left, right = sorted((int(round(x1)), int(round(x2))))
        top, bottom = sorted((int(round(y1)), int(round(y2))))
        return OCRRegion(
            region_id=str(obj.get("id", f"region-{left}-{top}")),
            text=transcription,
            bbox=(left, top, right, bottom),
        )

    def _tag_value(self, obj: dict, tag_name: str) -> str:
        for tag in obj.get("tags", []):
            if tag.get("name") == tag_name and tag.get("value"):
                return str(tag["value"])
        return ""

    def _join_text(self, regions: list[OCRRegion]) -> str:
        if not regions:
            return ""
        ordered = sorted(regions, key=lambda region: (region.bbox[1], region.bbox[0], region.region_id))
        return " ".join(region.text.strip() for region in ordered if region.text.strip())

    def _pick_date(self, regions: list[OCRRegion]) -> str:
        for region in sorted(regions, key=lambda item: (item.bbox[1], item.bbox[0], item.region_id)):
            normalized = normalize_date(region.text)
            if normalized:
                return normalized
        return ""

    def _pick_total(self, regions: list[OCRRegion]) -> str:
        ordered = sorted(regions, key=lambda item: (item.bbox[1], item.bbox[0], item.region_id))
        for region in reversed(ordered):
            normalized = normalize_amount(region.text)
            if normalized:
                return normalized
        return ""

    def _bucket_by_difficulty(self, samples: list[ReceiptSample]) -> dict[str, list[ReceiptSample]]:
        if not samples:
            return {"easy": [], "medium": [], "hard": []}

        ranked = sorted(samples, key=lambda sample: (self._complexity_score(sample), sample.sample_id))
        total = len(ranked)
        first_cut = max(1, total // 3)
        second_cut = max(first_cut + 1, (2 * total) // 3) if total > 1 else total
        return {
            "easy": ranked[:first_cut],
            "medium": ranked[first_cut:second_cut] or ranked[:first_cut],
            "hard": ranked[second_cut:] or ranked[-1:],
        }

    def _complexity_score(self, sample: ReceiptSample) -> tuple[int, int, int, int, int]:
        texts = [region.text for region in sample.regions]
        address_lines = sum(1 for text in texts if any(char.isdigit() for char in text) and any(char.isalpha() for char in text))
        numeric_lines = sum(1 for text in texts if any(char.isdigit() for char in text))
        long_lines = sum(1 for text in texts if len(text.strip()) >= 20)
        return (len(sample.regions), address_lines, numeric_lines, long_lines, len(sample.gold_fields.address or ""))
