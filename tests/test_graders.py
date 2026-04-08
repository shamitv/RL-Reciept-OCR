from env.graders import address_score, grade_receipt, score_formula_definition, score_formula_term_contributions
from env.models import ReceiptDraft, ReceiptLineItem


def test_address_score_partial_match() -> None:
    assert 0.0 < address_score("12 JALAN BINTANG", "12 JALAN BINTANG KL") < 1.0


def test_grade_receipt_exact_match() -> None:
    gold = ReceiptDraft(company="SHOP", date="2019-03-25", address="12 ROAD KL", total="31.00")
    result = grade_receipt(gold, gold)
    assert result.score == 1.0
    assert result.success is True


def test_hard_grade_matches_shared_formula_terms() -> None:
    gold = ReceiptDraft(
        company="SHOP",
        date="2019-03-25",
        address="12 ROAD KL",
        subtotal="30.00",
        tax="1.00",
        total="31.00",
        line_items=[ReceiptLineItem(description="Sandwich", line_total="30.00")],
    )
    result = grade_receipt(gold, gold, task_id="hard", gold_line_items=gold.line_items)
    definition = score_formula_definition("hard")
    weights = {term["source_key"]: term["weight"] for term in definition["terms"]}
    source_scores = {
        **result.field_scores,
        "line_items": result.line_items_score / weights["line_items"],
        "reconciliation": result.reconciliation_score / weights["reconciliation"],
    }

    terms = score_formula_term_contributions("hard", source_scores)

    assert definition["title"] == "Hard task"
    assert result.score == sum(term["contribution"] for term in terms)
    assert any(
        "line_item_count_score is diagnostic only and is not included directly in the final deterministic score"
        in note
        for note in definition["notes"]
    )
