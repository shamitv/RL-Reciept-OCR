from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation

from dateutil import parser


MONTH_PATTERN = r"jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?"
NUMERIC_DATE_PATTERN = re.compile(r"(?<!\d)(?:\d{1,4}[./-]\d{1,2}[./-]\d{1,4}|\d{8})(?!\d)", re.IGNORECASE)
MONTH_FIRST_DATE_PATTERN = re.compile(
    rf"\b(?:{MONTH_PATTERN})\.?\s*\d{{1,2}}(?:st|nd|rd|th)?\s*(?:[,./'-]|\s)\s*'?\d{{2,4}}\b",
    re.IGNORECASE,
)
DAY_FIRST_MONTH_DATE_PATTERN = re.compile(
    rf"(?<![#\w])\d{{1,2}}\s*[-/\s]?\s*(?:{MONTH_PATTERN})\.?\s*[-,.'/\s]\s*'?\d{{2,4}}\b",
    re.IGNORECASE,
)
PERCENT_PATTERN = re.compile(r"(?<![\d.])\d+(?:\s*\.\s*\d+)?\s*%")
AMOUNT_PATTERN = re.compile(r"(?<![\d.])(?:[$\u20ac\u00a3*]\s*)?(\d{1,6}(?:\s*\.\s*\d{1,2})|\.\s*\d{1,2}|\d{1,6})(?![\d.])")
AMOUNT_CONTEXT_PATTERN = re.compile(r"[$\u20ac\u00a3*]|TOTAL|SUBTOTAL|TAX|VAT|GST|SST|AMOUNT|BALANCE|DUE|CASH|CHANGE|PRICE|PAID|TIP", re.IGNORECASE)
DATE_CONTEXT_PATTERN = re.compile(rf"(?:{MONTH_PATTERN})|\d{{1,4}}[./-]\d{{1,2}}[./-]\d{{1,4}}", re.IGNORECASE)


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
    candidate_text = re.sub(r"(\d)\s*\.\s*(\d)", r"\1.\2", value.strip())
    has_amount_context = bool(AMOUNT_CONTEXT_PATTERN.search(candidate_text))
    has_decimal = bool(re.search(r"\d\s*\.\s*\d", candidate_text))
    is_bare_number = bool(re.fullmatch(r"\s*[$\u20ac\u00a3*]?\s*\d+(?:\s*\.\s*\d+)?\s*", candidate_text))
    if DATE_CONTEXT_PATTERN.search(candidate_text) and not has_amount_context and not is_bare_number:
        return ""

    candidate_text = PERCENT_PATTERN.sub(" ", candidate_text)
    matches = list(AMOUNT_PATTERN.finditer(candidate_text))
    if not matches:
        return ""

    if not has_amount_context and not has_decimal and not is_bare_number and len(matches) > 1:
        return ""

    raw_amount = matches[-1].group(1).replace(" ", "")
    if raw_amount.startswith("."):
        raw_amount = f"0{raw_amount}"
    try:
        amount = Decimal(raw_amount)
    except InvalidOperation:
        return ""
    return f"{amount:.2f}"


def normalize_date(value: str | None) -> str:
    if not value:
        return ""
    candidate_text = value.strip()
    date_matches = [
        *NUMERIC_DATE_PATTERN.finditer(candidate_text),
        *MONTH_FIRST_DATE_PATTERN.finditer(candidate_text),
        *DAY_FIRST_MONTH_DATE_PATTERN.finditer(candidate_text),
    ]
    candidates = [match.group(0) for match in sorted(date_matches, key=lambda match: match.start())]
    for candidate in candidates:
        dayfirst_options = (True, False) if NUMERIC_DATE_PATTERN.fullmatch(candidate) else (False, True)
        for dayfirst in dayfirst_options:
            try:
                parsed = parser.parse(candidate, dayfirst=dayfirst, fuzzy=True)
            except (ValueError, TypeError, OverflowError):
                continue
            if 1990 <= parsed.year <= 2035:
                return parsed.strftime("%Y-%m-%d")
    return ""


def tokenize(value: str | None) -> list[str]:
    normalized = normalize_text(value)
    if not normalized:
        return []
    return [token for token in re.split(r"[^A-Z0-9.]+", normalized) if token]
