from env.graders import address_score, grade_receipt
from env.models import ReceiptDraft


def test_address_score_partial_match() -> None:
    assert 0.0 < address_score("12 JALAN BINTANG", "12 JALAN BINTANG KL") < 1.0


def test_grade_receipt_exact_match() -> None:
    gold = ReceiptDraft(company="SHOP", date="2019-03-25", address="12 ROAD KL", total="31.00")
    result = grade_receipt(gold, gold)
    assert result.score == 1.0
    assert result.success is True
