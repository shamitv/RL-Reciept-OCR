from __future__ import annotations

from collections import Counter

from env.models import GradeResult, ReceiptDraft
from env.normalizers import normalize_address, normalize_amount, normalize_date, normalize_text, tokenize
from env.utils import clamp


def token_f1(predicted: str | None, gold: str | None) -> float:
    predicted_tokens = tokenize(predicted)
    gold_tokens = tokenize(gold)
    if not predicted_tokens and not gold_tokens:
        return 1.0
    if not predicted_tokens or not gold_tokens:
        return 0.0
    predicted_counter = Counter(predicted_tokens)
    gold_counter = Counter(gold_tokens)
    overlap = sum((predicted_counter & gold_counter).values())
    precision = overlap / len(predicted_tokens)
    recall = overlap / len(gold_tokens)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def company_score(pred: str | None, gold: str | None) -> float:
    return token_f1(normalize_text(pred), normalize_text(gold))


def date_score(pred: str | None, gold: str | None) -> float:
    return 1.0 if normalize_date(pred) and normalize_date(pred) == normalize_date(gold) else 0.0


def address_score(pred: str | None, gold: str | None) -> float:
    return token_f1(normalize_address(pred), normalize_address(gold))


def total_score(pred: str | None, gold: str | None) -> float:
    return 1.0 if normalize_amount(pred) and normalize_amount(pred) == normalize_amount(gold) else 0.0


def grade_receipt(prediction: ReceiptDraft, gold: ReceiptDraft) -> GradeResult:
    field_scores = {
        "company": company_score(prediction.company, gold.company),
        "date": date_score(prediction.date, gold.date),
        "address": address_score(prediction.address, gold.address),
        "total": total_score(prediction.total, gold.total),
    }
    final_score = clamp(
        0.20 * field_scores["company"]
        + 0.20 * field_scores["date"]
        + 0.25 * field_scores["address"]
        + 0.35 * field_scores["total"]
    )
    return GradeResult(score=final_score, success=final_score >= 0.85, field_scores=field_scores)
