from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation

from dateutil import parser


DATE_PATTERN = re.compile(r"\b(?:\d{1,4}[/-]\d{1,2}[/-]\d{1,4}|\d{8})\b")


def normalize_text(value: str | None) -> str:
    if not value:
        return ""
    value = value.upper().strip()
    value = re.sub(r"\s+", " ", value)
    return value


def normalize_address(value: str | None) -> str:
    value = normalize_text(value)
    value = value.replace("\n", " ")
    value = re.sub(r"\s+", " ", value)
    value = re.sub(r"\s*[,;]+\s*", " ", value)
    return value.strip()


def normalize_amount(value: str | None) -> str:
    if not value:
        return ""
    cleaned = re.sub(r"[^0-9.]", "", value)
    if not cleaned:
        return ""
    try:
        amount = Decimal(cleaned)
    except InvalidOperation:
        return ""
    return f"{amount:.2f}"


def normalize_date(value: str | None) -> str:
    if not value:
        return ""
    candidate = value.strip()
    if not DATE_PATTERN.search(candidate):
        return ""
    try:
        parsed = parser.parse(candidate, dayfirst=True, fuzzy=False)
    except (ValueError, TypeError, OverflowError):
        try:
            parsed = parser.parse(candidate, dayfirst=False, fuzzy=False)
        except (ValueError, TypeError, OverflowError):
            return ""
    return parsed.strftime("%Y-%m-%d")


def tokenize(value: str | None) -> list[str]:
    normalized = normalize_text(value)
    if not normalized:
        return []
    return [token for token in re.split(r"[^A-Z0-9.]+", normalized) if token]
