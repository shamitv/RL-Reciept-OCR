from __future__ import annotations

from collections import Counter
from decimal import Decimal, InvalidOperation
from math import inf

from env.models import GradeResult, ReceiptDraft, ReceiptLineItem
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


def amount_score(pred: str | None, gold: str | None) -> float:
    return 1.0 if normalize_amount(pred) and normalize_amount(pred) == normalize_amount(gold) else 0.0


def subtotal_score(pred: str | None, gold: str | None) -> float:
    return amount_score(pred, gold)


def tax_score(pred: str | None, gold: str | None) -> float:
    return amount_score(pred, gold)


def total_score(pred: str | None, gold: str | None) -> float:
    return amount_score(pred, gold)


def _decimal_amount(value: str | None) -> Decimal | None:
    normalized = normalize_amount(value)
    if not normalized:
        return None
    try:
        return Decimal(normalized)
    except InvalidOperation:
        return None


def _line_item_similarity(predicted: ReceiptLineItem, gold: ReceiptLineItem) -> float:
    description_component = token_f1(predicted.description, gold.description)
    predicted_total = normalize_amount(predicted.line_total)
    gold_total = normalize_amount(gold.line_total)
    if predicted_total and gold_total:
        amount_component = 1.0 if predicted_total == gold_total else 0.0
        return (0.6 * description_component) + (0.4 * amount_component)
    return description_component


def score_line_items(
    predicted_items: list[ReceiptLineItem],
    gold_items: list[ReceiptLineItem],
) -> float:
    if not predicted_items and not gold_items:
        return 1.0
    if not predicted_items or not gold_items:
        return 0.0

    remaining_indices = set(range(len(gold_items)))
    matched_similarity = 0.0
    for predicted in predicted_items:
        best_index = None
        best_score = 0.0
        for index in remaining_indices:
            similarity = _line_item_similarity(predicted, gold_items[index])
            if similarity > best_score:
                best_score = similarity
                best_index = index
        if best_index is not None:
            matched_similarity += best_score
            remaining_indices.remove(best_index)

    denominator = max(len(predicted_items), len(gold_items))
    if denominator == 0:
        return 1.0
    return clamp(matched_similarity / float(denominator))


def reconciliation_component(
    prediction: ReceiptDraft,
    task_id: str,
) -> tuple[float, str, float | None]:
    subtotal = _decimal_amount(prediction.subtotal)
    tax = _decimal_amount(prediction.tax)
    total = _decimal_amount(prediction.total)
    primary_delta: Decimal | None = None

    if subtotal is not None and tax is not None and total is not None:
        primary_delta = abs((subtotal + tax) - total)

    line_items_delta: Decimal | None = None
    if task_id == "hard":
        if subtotal is None or not prediction.line_items:
            return 0.0, "fail", float(primary_delta) if primary_delta is not None else None
        line_item_amounts = [_decimal_amount(item.line_total) for item in prediction.line_items]
        if any(amount is None for amount in line_item_amounts):
            return 0.0, "fail", float(primary_delta) if primary_delta is not None else None
        line_items_total = sum((amount for amount in line_item_amounts if amount is not None), Decimal("0.00"))
        line_items_delta = abs(line_items_total - subtotal)

    if primary_delta is None:
        return 0.0, "fail", None

    combined_delta = max(primary_delta, line_items_delta or Decimal("0.00"))
    if combined_delta <= Decimal("0.01"):
        return 1.0, "pass", float(combined_delta)
    if combined_delta <= Decimal("0.05"):
        return 0.5, "partial", float(combined_delta)
    return 0.0, "fail", float(combined_delta)


def grade_receipt(
    prediction: ReceiptDraft,
    gold: ReceiptDraft,
    task_id: str = "easy",
    gold_line_items: list[ReceiptLineItem] | None = None,
) -> GradeResult:
    gold_line_items = gold_line_items or []
    field_scores = {
        "company": company_score(prediction.company, gold.company),
        "date": date_score(prediction.date, gold.date),
        "address": address_score(prediction.address, gold.address),
        "subtotal": subtotal_score(prediction.subtotal, gold.subtotal),
        "tax": tax_score(prediction.tax, gold.tax),
        "total": total_score(prediction.total, gold.total),
    }

    header_score = 0.0
    summary_score = 0.0
    line_items_score = 0.0
    reconciliation_score = 0.0
    reconciliation_delta: float | None = None
    reconciliation_status: str | None = None

    if task_id == "easy":
        header_score = clamp(
            0.20 * field_scores["company"]
            + 0.20 * field_scores["date"]
            + 0.25 * field_scores["address"]
            + 0.35 * field_scores["total"]
        )
        final_score = header_score
    elif task_id == "medium":
        header_score = clamp((0.10 * field_scores["company"]) + (0.10 * field_scores["date"]))
        summary_score = clamp(
            (0.15 * field_scores["subtotal"]) + (0.15 * field_scores["tax"]) + (0.25 * field_scores["total"])
        )
        reconciliation_component_score, reconciliation_status, reconciliation_delta = reconciliation_component(prediction, task_id)
        reconciliation_score = clamp(0.25 * reconciliation_component_score)
        final_score = clamp(header_score + summary_score + reconciliation_score)
    else:
        header_score = clamp((0.05 * field_scores["company"]) + (0.05 * field_scores["date"]))
        summary_score = clamp(
            (0.05 * field_scores["subtotal"]) + (0.05 * field_scores["tax"]) + (0.10 * field_scores["total"])
        )
        line_items_score = clamp(0.45 * score_line_items(prediction.line_items, gold_line_items))
        reconciliation_component_score, reconciliation_status, reconciliation_delta = reconciliation_component(prediction, task_id)
        reconciliation_score = clamp(0.25 * reconciliation_component_score)
        final_score = clamp(header_score + summary_score + line_items_score + reconciliation_score)

    return GradeResult(
        score=final_score,
        success=final_score >= 0.85,
        field_scores=field_scores,
        header_score=header_score,
        summary_score=summary_score,
        line_items_score=line_items_score,
        reconciliation_score=reconciliation_score,
        reconciliation_delta=reconciliation_delta,
        reconciliation_status=reconciliation_status,
    )
