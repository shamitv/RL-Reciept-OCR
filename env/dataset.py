from __future__ import annotations

import json
import os
import re
from random import Random
from pathlib import Path

from env.models import OCRRegion, ReceiptDraft, ReceiptLineItem, ReceiptSample
from env.normalizers import normalize_amount, normalize_date, normalize_text

LINE_ITEM_AMOUNT_PATTERN = re.compile(r"(\d+[.]\d{1,2})(?!.*\d+[.]\d{1,2})")
TASK_REQUIREMENTS: dict[str, tuple[str, ...]] = {
    "easy": ("company", "address", "date", "total"),
    "medium": ("company", "date", "subtotal", "tax", "total"),
    "hard": ("company", "date", "subtotal", "tax", "total"),
}


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
        gold_fields=ReceiptDraft(
            company="MART CORNER SDN BHD",
            address="12 JALAN BINTANG 50000 KUALA LUMPUR",
            date="2019-03-25",
            total="31.00",
        ),
    ),
    ReceiptSample(
        sample_id="sample_medium_1",
        image_ref="mock://receipt/2",
        regions=[
            OCRRegion(region_id="r1", text="FRESH HUB MARKET", bbox=(0, 0, 200, 20)),
            OCRRegion(region_id="r2", text="26-03-19", bbox=(0, 92, 100, 112)),
            OCRRegion(region_id="r3", text="SUBTOTAL 19.90", bbox=(0, 180, 140, 200)),
            OCRRegion(region_id="r4", text="TAX 1.20", bbox=(0, 200, 140, 220)),
            OCRRegion(region_id="r5", text="TOTAL DUE 21.10", bbox=(0, 220, 160, 240)),
        ],
        gold_fields=ReceiptDraft(
            company="FRESH HUB MARKET",
            date="2019-03-26",
            subtotal="19.90",
            tax="1.20",
            total="21.10",
        ),
    ),
    ReceiptSample(
        sample_id="sample_hard_1",
        image_ref="mock://receipt/3",
        regions=[
            OCRRegion(region_id="r1", text="CITY CAFE", bbox=(0, 0, 120, 20)),
            OCRRegion(region_id="r2", text="2019-03-27", bbox=(0, 88, 100, 108)),
            OCRRegion(region_id="r3", text="LATTE 4.50", bbox=(0, 140, 120, 160)),
            OCRRegion(region_id="r4", text="MUFFIN 3.50", bbox=(0, 165, 120, 185)),
            OCRRegion(region_id="r5", text="SUBTOTAL 8.00", bbox=(0, 190, 120, 210)),
            OCRRegion(region_id="r6", text="TAX 0.48", bbox=(0, 205, 120, 225)),
            OCRRegion(region_id="r7", text="TOTAL 8.48", bbox=(0, 225, 120, 245)),
        ],
        gold_fields=ReceiptDraft(company="CITY CAFE", date="2019-03-27", subtotal="8.00", tax="0.48", total="8.48"),
        gold_line_items=[
            ReceiptLineItem(item_id="li-1", description="LATTE", line_total="4.50", raw_text="LATTE 4.50"),
            ReceiptLineItem(item_id="li-2", description="MUFFIN", line_total="3.50", raw_text="MUFFIN 3.50"),
        ],
    ),
]


class ReceiptDataset:
    def __init__(self, dataset_root: str | Path | None = None) -> None:
        self.dataset_root = self._resolve_dataset_root(dataset_root)
        self.samples = self._load_samples()
        self.samples_by_task = self._bucket_by_task(self.samples)

    def sample(self, task_name: str, rng: Random) -> ReceiptSample:
        candidates = self.samples_by_task.get(task_name, [])
        if not candidates:
            candidates = self.samples
        index = rng.randrange(len(candidates))
        return candidates[index].model_copy(deep=True)

    def eligible_task_counts(self) -> dict[str, int]:
        return {task_name: len(samples) for task_name, samples in self.samples_by_task.items()}

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

        image_name = annotation_path.name[:-5]
        image_path = image_dir / image_name
        if not image_path.exists():
            return None

        company = self._join_text(grouped_regions.get("Business name", []))
        address = self._join_text(grouped_regions.get("Business address", []))
        date = self._pick_date(grouped_regions.get("Time and date", []))
        subtotal = self._pick_amount(grouped_regions.get("Subtotal", []))
        tax = self._pick_amount(grouped_regions.get("Tax", []))
        total = self._pick_amount(grouped_regions.get("Total", []))
        gold_line_items = self._extract_line_items(grouped_regions.get("Item information", []))

        if not company or not date or not total:
            return None

        return ReceiptSample(
            sample_id=annotation_path.stem,
            image_ref=str(image_path),
            regions=sorted(visible_regions, key=lambda region: (region.bbox[1], region.bbox[0], region.region_id)),
            gold_fields=ReceiptDraft(
                company=company or None,
                address=address or None,
                date=date or None,
                subtotal=subtotal or None,
                tax=tax or None,
                total=total or None,
            ),
            gold_line_items=gold_line_items,
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

    def _pick_amount(self, regions: list[OCRRegion]) -> str:
        ordered = sorted(regions, key=lambda item: (item.bbox[1], item.bbox[0], item.region_id))
        for region in reversed(ordered):
            normalized = normalize_amount(region.text)
            if normalized:
                return normalized
        return ""

    def _extract_line_items(self, regions: list[OCRRegion]) -> list[ReceiptLineItem]:
        items: list[ReceiptLineItem] = []
        for region in sorted(regions, key=lambda item: (item.bbox[1], item.bbox[0], item.region_id)):
            item = self._line_item_from_region(region)
            if item is not None:
                items.append(item)
        return items

    def _line_item_from_region(self, region: OCRRegion) -> ReceiptLineItem | None:
        raw_text = region.text.strip()
        if not raw_text:
            return None
        amount_match = LINE_ITEM_AMOUNT_PATTERN.search(raw_text)
        line_total = normalize_amount(amount_match.group(1) if amount_match else raw_text) or None
        description_text = raw_text
        if amount_match:
            description_text = raw_text[: amount_match.start()].strip(" -:$")
        description = normalize_text(description_text) or None
        if not description and not line_total:
            return None
        return ReceiptLineItem(
            item_id=f"gold:{region.region_id}",
            description=description,
            line_total=line_total,
            raw_text=raw_text,
            evidence_ids=[region.region_id],
        )

    def _bucket_by_task(self, samples: list[ReceiptSample]) -> dict[str, list[ReceiptSample]]:
        buckets = {"easy": [], "medium": [], "hard": []}
        for sample in samples:
            if self._is_task_eligible(sample, "easy"):
                buckets["easy"].append(sample)
            if self._is_task_eligible(sample, "medium"):
                buckets["medium"].append(sample)
            if self._is_task_eligible(sample, "hard"):
                buckets["hard"].append(sample)
        return buckets

    def _is_task_eligible(self, sample: ReceiptSample, task_name: str) -> bool:
        required_fields = TASK_REQUIREMENTS[task_name]
        if not all(getattr(sample.gold_fields, field) for field in required_fields):
            return False
        if task_name == "hard":
            return bool(sample.gold_line_items)
        return True
