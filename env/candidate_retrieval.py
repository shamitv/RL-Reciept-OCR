from __future__ import annotations

import hashlib
import re

from env.models import FieldCandidate, OCRRegion
from env.normalizers import normalize_address, normalize_amount, normalize_date, normalize_text

FIELDS = ("company", "date", "address", "total")


def _window_score(region: OCRRegion) -> float:
    return max(0.0, 1.0 - (region.bbox[1] / 300.0))


def company_candidates(regions: list[OCRRegion]) -> list[FieldCandidate]:
    candidates: list[FieldCandidate] = []
    for region in regions:
        text = normalize_text(region.text)
        if any(char.isdigit() for char in text):
            continue
        score = 0.7 + 0.3 * _window_score(region)
        candidates.append(FieldCandidate(candidate_id=f"company:{region.region_id}", field="company", value=text, evidence_ids=[region.region_id], heuristic_score=round(score, 4)))
    return sorted(candidates, key=lambda item: item.heuristic_score, reverse=True)


def date_candidates(regions: list[OCRRegion]) -> list[FieldCandidate]:
    candidates: list[FieldCandidate] = []
    for region in regions:
        normalized = normalize_date(region.text)
        if not normalized:
            continue
        score = 0.8 + 0.2 * _window_score(region)
        candidates.append(FieldCandidate(candidate_id=f"date:{region.region_id}", field="date", value=normalized, evidence_ids=[region.region_id], heuristic_score=round(score, 4)))
    return sorted(candidates, key=lambda item: item.heuristic_score, reverse=True)


def total_candidates(regions: list[OCRRegion]) -> list[FieldCandidate]:
    candidates: list[FieldCandidate] = []
    for region in regions:
        if "TOTAL" not in normalize_text(region.text) and not re.search(r"\d+[.]\d{1,2}", region.text):
            continue
        amount = normalize_amount(region.text)
        if not amount:
            continue
        bottom_bias = region.bbox[1] / 300.0
        score = 0.75 + min(0.25, bottom_bias)
        candidates.append(FieldCandidate(candidate_id=f"total:{region.region_id}", field="total", value=amount, evidence_ids=[region.region_id], heuristic_score=round(score, 4)))
    return sorted(candidates, key=lambda item: item.heuristic_score, reverse=True)


def address_candidates(regions: list[OCRRegion]) -> list[FieldCandidate]:
    text_regions = [region for region in regions if any(char.isalpha() for char in region.text)]
    ordered = sorted(text_regions, key=lambda region: region.bbox[1])
    candidates: list[FieldCandidate] = []
    for index in range(len(ordered)):
        for width in (1, 2, 3):
            window = ordered[index : index + width]
            if len(window) != width:
                continue
            value = normalize_address(" ".join(region.text for region in window))
            if len(value) < 8:
                continue
            evidence_ids = [region.region_id for region in window]
            has_digit = any(any(char.isdigit() for char in region.text) for region in window)
            top_penalty = 0.12 if index == 0 and not has_digit else 0.0
            score = 0.62 + 0.08 * width + (0.08 if has_digit else 0.0) - top_penalty
            candidates.append(FieldCandidate(candidate_id=f"address:{'-'.join(evidence_ids)}", field="address", value=value, evidence_ids=evidence_ids, heuristic_score=round(score, 4)))
    return sorted(candidates, key=lambda item: item.heuristic_score, reverse=True)


def _stable_noise(key: str) -> float:
    digest = hashlib.sha256(key.encode("utf-8")).digest()
    value = int.from_bytes(digest[:8], "big") / float(2**64)
    return (value * 2.0) - 1.0


def _rerank_candidates(candidates: list[FieldCandidate], ranking_noise: float, noise_key: str) -> list[FieldCandidate]:
    if ranking_noise <= 0.0:
        return candidates

    reranked: list[FieldCandidate] = []
    for candidate in candidates:
        adjusted_score = candidate.heuristic_score + (_stable_noise(f"{noise_key}:{candidate.candidate_id}") * ranking_noise)
        reranked.append(candidate.model_copy(update={"heuristic_score": round(adjusted_score, 4)}))
    return sorted(reranked, key=lambda item: (item.heuristic_score, item.candidate_id), reverse=True)


def query_candidates(field: str, visible_regions: list[OCRRegion], ranking_noise: float = 0.0, noise_key: str = "") -> list[FieldCandidate]:
    if field == "company":
        candidates = company_candidates(visible_regions)
        return _rerank_candidates(candidates, ranking_noise, noise_key)
    if field == "date":
        candidates = date_candidates(visible_regions)
        return _rerank_candidates(candidates, ranking_noise, noise_key)
    if field == "address":
        candidates = address_candidates(visible_regions)
        return _rerank_candidates(candidates, ranking_noise, noise_key)
    if field == "total":
        candidates = total_candidates(visible_regions)
        return _rerank_candidates(candidates, ranking_noise, noise_key)
    return []
