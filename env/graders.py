from __future__ import annotations

from collections import Counter
from decimal import Decimal, InvalidOperation
from typing import Any

from env.models import GradeResult, ReceiptDraft, ReceiptLineItem
from env.normalizers import normalize_address, normalize_amount, normalize_date, normalize_text, tokenize
from env.utils import clamp

COMMON_SCORE_FORMULA_NOTES = (
    "company and address use token F1 over normalized text.",
    "date, subtotal, tax, and total use exact normalized date/amount matches.",
    "Final scores are clamped to the 0.000-1.000 range.",
)

SCORE_FORMULAS: dict[str, dict[str, Any]] = {
    "easy": {
        "title": "Easy task",
        "formula": "score = clamp(0.20*company + 0.20*date + 0.25*address + 0.35*total)",
        "terms": (
            {"label": "Company", "source_key": "company", "weight": 0.20, "component": "header_score", "source": "company field score"},
            {"label": "Date", "source_key": "date", "weight": 0.20, "component": "header_score", "source": "date field score"},
            {"label": "Address", "source_key": "address", "weight": 0.25, "component": "header_score", "source": "address field score"},
            {"label": "Total", "source_key": "total", "weight": 0.35, "component": "header_score", "source": "total field score"},
        ),
        "notes": (),
    },
    "medium": {
        "title": "Medium task",
        "formula": (
            "score = clamp(0.10*company + 0.10*date + 0.15*subtotal + "
            "0.15*tax + 0.25*total + 0.25*summary_reconciliation)"
        ),
        "terms": (
            {"label": "Company", "source_key": "company", "weight": 0.10, "component": "header_score", "source": "company field score"},
            {"label": "Date", "source_key": "date", "weight": 0.10, "component": "header_score", "source": "date field score"},
            {"label": "Subtotal", "source_key": "subtotal", "weight": 0.15, "component": "summary_score", "source": "subtotal field score"},
            {"label": "Tax", "source_key": "tax", "weight": 0.15, "component": "summary_score", "source": "tax field score"},
            {"label": "Total", "source_key": "total", "weight": 0.25, "component": "summary_score", "source": "total field score"},
            {
                "label": "Summary reconciliation",
                "source_key": "summary_reconciliation",
                "weight": 0.25,
                "component": "reconciliation_score",
                "source": "subtotal + tax vs total",
            },
        ),
        "notes": (
            "summary reconciliation gives 1.000 when subtotal + tax equals total within 0.01, 0.500 within 0.05, otherwise 0.000.",
        ),
    },
    "hard": {
        "title": "Hard task",
        "formula": (
            "score = clamp(0.05*company + 0.05*date + 0.05*subtotal + "
            "0.05*tax + 0.10*total + 0.45*line_items + 0.25*reconciliation)"
        ),
        "terms": (
            {"label": "Company", "source_key": "company", "weight": 0.05, "component": "header_score", "source": "company field score"},
            {"label": "Date", "source_key": "date", "weight": 0.05, "component": "header_score", "source": "date field score"},
            {"label": "Subtotal", "source_key": "subtotal", "weight": 0.05, "component": "summary_score", "source": "subtotal field score"},
            {"label": "Tax", "source_key": "tax", "weight": 0.05, "component": "summary_score", "source": "tax field score"},
            {"label": "Total", "source_key": "total", "weight": 0.10, "component": "summary_score", "source": "total field score"},
            {
                "label": "Line items",
                "source_key": "line_items",
                "weight": 0.45,
                "component": "line_items_score",
                "source": "best-match line item similarity",
            },
            {
                "label": "Reconciliation",
                "source_key": "reconciliation",
                "weight": 0.25,
                "component": "reconciliation_score",
                "source": "summary and line-item reconciliation",
            },
        ),
        "notes": (
            "line items use best-match row similarity: 0.60*description token F1 + 0.40*amount exact match when both amounts exist.",
            "line item count is diagnostic only and is not included in the final deterministic score.",
        ),
    },
}


def scoring_task_id(task_id: str | None) -> str:
    normalized = str(task_id if task_id is not None else "easy").lower()
    if normalized == "easy":
        return "easy"
    if normalized == "medium":
        return "medium"
    return "hard"


def score_formula_definition(task_id: str | None) -> dict[str, Any]:
    resolved_task_id = scoring_task_id(task_id)
    definition = SCORE_FORMULAS[resolved_task_id]
    return {
        "task_id": resolved_task_id,
        "title": definition["title"],
        "formula": definition["formula"],
        "terms": [dict(term) for term in definition["terms"]],
        "notes": list(COMMON_SCORE_FORMULA_NOTES + definition["notes"]),
    }


def _safe_score_value(value: Any) -> float:
    try:
        return clamp(float(value or 0.0))
    except (TypeError, ValueError):
        return 0.0


def score_formula_term_contributions(task_id: str | None, source_scores: dict[str, Any]) -> list[dict[str, Any]]:
    contributions: list[dict[str, Any]] = []
    for term in score_formula_definition(task_id)["terms"]:
        source_score = _safe_score_value(source_scores.get(term["source_key"]))
        contributions.append(
            {
                "label": term["label"],
                "weight": term["weight"],
                "source_score": source_score,
                "contribution": round(term["weight"] * source_score, 6),
                "source": term["source"],
            }
        )
    return contributions


def score_formula_numeric(terms: list[dict[str, Any]], overall_score: float) -> str:
    if not terms:
        return f"score = {overall_score:.3f}"
    contributions = " + ".join(f"{term['contribution']:.3f}" for term in terms)
    return f"score = clamp({contributions}) = {overall_score:.3f}"


def _score_formula_component(task_id: str, source_scores: dict[str, Any], component: str) -> float:
    return clamp(
        sum(
            float(term["weight"]) * _safe_score_value(source_scores.get(term["source_key"]))
            for term in SCORE_FORMULAS[task_id]["terms"]
            if term["component"] == component
        )
    )


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


def score_line_item_count(
    predicted_items: list[ReceiptLineItem],
    gold_items: list[ReceiptLineItem],
) -> tuple[float | None, int | None]:
    if not gold_items:
        return None, None
    predicted_count = len(predicted_items)
    gold_count = len(gold_items)
    delta = abs(predicted_count - gold_count)
    denominator = max(predicted_count, gold_count, 1)
    return clamp(1.0 - (delta / float(denominator))), delta


def _delta_to_status(delta: Decimal | None) -> tuple[float, str, float | None]:
    if delta is None:
        return 0.0, "not_evaluated", None
    if delta <= Decimal("0.01"):
        return 1.0, "pass", float(delta)
    if delta <= Decimal("0.05"):
        return 0.5, "partial", float(delta)
    return 0.0, "fail", float(delta)


def summary_reconciliation_component(prediction: ReceiptDraft) -> tuple[float, str, float | None]:
    subtotal = _decimal_amount(prediction.subtotal)
    tax = _decimal_amount(prediction.tax)
    total = _decimal_amount(prediction.total)
    if subtotal is None or tax is None or total is None:
        return 0.0, "not_evaluated", None
    return _delta_to_status(abs((subtotal + tax) - total))


def line_item_reconciliation_component(
    prediction: ReceiptDraft,
    gold_line_items: list[ReceiptLineItem],
) -> tuple[float, str, float | None]:
    if not gold_line_items:
        return 0.0, "not_evaluated", None

    subtotal = _decimal_amount(prediction.subtotal)
    if subtotal is None or not prediction.line_items:
        return 0.0, "fail", None

    line_item_amounts = [_decimal_amount(item.line_total) for item in prediction.line_items]
    if any(amount is None for amount in line_item_amounts):
        return 0.0, "fail", None
    line_items_total = sum((amount for amount in line_item_amounts if amount is not None), Decimal("0.00"))
    return _delta_to_status(abs(line_items_total - subtotal))


def combine_reconciliation(
    summary_result: tuple[float, str, float | None],
    line_item_result: tuple[float, str, float | None],
) -> tuple[float, str, float | None]:
    summary_score_value, summary_status, summary_delta = summary_result
    line_item_score_value, line_item_status, line_item_delta = line_item_result

    if line_item_status == "not_evaluated":
        return summary_score_value, summary_status, summary_delta
    if summary_status == "not_evaluated":
        return line_item_score_value, line_item_status, line_item_delta

    combined_delta_values = [value for value in (summary_delta, line_item_delta) if value is not None]
    combined_delta = max(combined_delta_values) if combined_delta_values else None

    if "fail" in {summary_status, line_item_status}:
        return 0.0, "fail", combined_delta
    if "partial" in {summary_status, line_item_status}:
        return 0.5, "partial", combined_delta
    return 1.0, "pass", combined_delta


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
    summary_reconciliation_delta: float | None = None
    summary_reconciliation_status: str | None = None
    line_item_reconciliation_delta: float | None = None
    line_item_reconciliation_status: str | None = None
    line_item_gold_available = bool(gold_line_items)
    gold_line_item_count = len(gold_line_items)
    predicted_line_item_count = len(prediction.line_items)
    line_item_count_score, line_item_count_delta = score_line_item_count(prediction.line_items, gold_line_items)
    resolved_task_id = scoring_task_id(task_id)
    formula_source_scores: dict[str, Any] = dict(field_scores)

    if resolved_task_id == "medium":
        reconciliation_component_score, reconciliation_status, reconciliation_delta = summary_reconciliation_component(prediction)
        summary_reconciliation_status = reconciliation_status
        summary_reconciliation_delta = reconciliation_delta
        line_item_reconciliation_status = "not_evaluated"
        formula_source_scores["summary_reconciliation"] = reconciliation_component_score
    elif resolved_task_id == "hard":
        formula_source_scores["line_items"] = score_line_items(prediction.line_items, gold_line_items)
        summary_result = summary_reconciliation_component(prediction)
        line_item_result = line_item_reconciliation_component(prediction, gold_line_items)
        summary_reconciliation_status = summary_result[1]
        summary_reconciliation_delta = summary_result[2]
        line_item_reconciliation_status = line_item_result[1]
        line_item_reconciliation_delta = line_item_result[2]
        reconciliation_component_score, reconciliation_status, reconciliation_delta = combine_reconciliation(summary_result, line_item_result)
        formula_source_scores["reconciliation"] = reconciliation_component_score

    header_score = _score_formula_component(resolved_task_id, formula_source_scores, "header_score")
    summary_score = _score_formula_component(resolved_task_id, formula_source_scores, "summary_score")
    line_items_score = _score_formula_component(resolved_task_id, formula_source_scores, "line_items_score")
    reconciliation_score = _score_formula_component(resolved_task_id, formula_source_scores, "reconciliation_score")
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
        summary_reconciliation_delta=summary_reconciliation_delta,
        summary_reconciliation_status=summary_reconciliation_status,
        line_item_reconciliation_delta=line_item_reconciliation_delta,
        line_item_reconciliation_status=line_item_reconciliation_status,
        line_item_gold_available=line_item_gold_available,
        gold_line_item_count=gold_line_item_count,
        predicted_line_item_count=predicted_line_item_count,
        line_item_count_delta=line_item_count_delta,
        line_item_count_score=line_item_count_score,
    )
