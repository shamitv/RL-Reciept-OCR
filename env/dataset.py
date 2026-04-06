from __future__ import annotations

from random import Random

from env.models import OCRRegion, ReceiptDraft, ReceiptSample

SAMPLES = [
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
    def __init__(self) -> None:
        self.samples = SAMPLES

    def sample(self, difficulty: str, rng: Random) -> ReceiptSample:
        candidates = [sample for sample in self.samples if difficulty in sample.sample_id]
        if not candidates:
            candidates = self.samples
        index = rng.randrange(len(candidates))
        return candidates[index].model_copy(deep=True)
