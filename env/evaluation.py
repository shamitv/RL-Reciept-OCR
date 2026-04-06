from __future__ import annotations

import base64
import json
import logging
import mimetypes
import os
from collections import Counter
from datetime import datetime, timezone
from json import JSONDecodeError, JSONDecoder
from pathlib import Path
from statistics import mean
from typing import Any, Literal

from openai import OpenAI
from pydantic import BaseModel, Field

from env.dataset import ReceiptDataset
from env.graders import grade_receipt
from env.llm_cache import cached_chat_completion
from env.models import ReceiptDraft, ReceiptLineItem

logger = logging.getLogger(__name__)

FIELD_NAMES = ("company", "date", "address", "subtotal", "tax", "total")
DEFAULT_EVAL_OUTPUT_DIR = Path(__file__).resolve().parents[1] / "artifacts" / "eval" / "dataset-image-eval"

DatasetStatus = Literal["runnable", "skipped_missing_labels", "skipped_unparseable_gold", "skipped_missing_image"]
EvalStatus = Literal["worked", "partial", "failed", "skipped"]
FieldStatus = Literal["correct", "partial", "incorrect", "missing", "not_evaluated"]

EXTRACTION_SYSTEM_PROMPT = (
    "You extract receipt fields from a single receipt image. "
    "Return strict JSON only with keys company, date, address, subtotal, tax, total, line_items. "
    "line_items must be an array of objects with keys description and line_total. "
    "Use null for unknown scalar values and [] for unknown line items."
)

JUDGE_SYSTEM_PROMPT = (
    "You are validating receipt extraction output against gold labels and deterministic scores. "
    "Return strict JSON only with keys summary, failure_reasons, field_notes."
)


class DatasetAuditRecord(BaseModel):
    sample_id: str
    task_id: str | None = None
    annotation_path: str
    image_path: str | None = None
    dataset_status: DatasetStatus
    skip_reason: str | None = None
    missing_categories: list[str] = Field(default_factory=list)
    unparseable_fields: list[str] = Field(default_factory=list)
    gold_fields: ReceiptDraft | None = None
    gold_line_items: list[ReceiptLineItem] = Field(default_factory=list)


class FieldEvaluation(BaseModel):
    predicted: str | None = None
    gold: str | None = None
    score: float = 0.0
    status: FieldStatus = "not_evaluated"


class JudgeEvaluation(BaseModel):
    summary: str
    failure_reasons: list[str] = Field(default_factory=list)
    field_notes: dict[str, str] = Field(default_factory=dict)


class ReceiptEvalRecord(BaseModel):
    sample_id: str
    task_id: str | None = None
    annotation_path: str
    image_path: str | None = None
    dataset_status: DatasetStatus
    status: EvalStatus
    skip_reason: str | None = None
    gold_fields: ReceiptDraft | None = None
    gold_line_items: list[ReceiptLineItem] = Field(default_factory=list)
    predicted_fields: ReceiptDraft | None = None
    predicted_line_items: list[ReceiptLineItem] = Field(default_factory=list)
    field_results: dict[str, FieldEvaluation] = Field(default_factory=dict)
    overall_score: float = 0.0
    header_score: float = 0.0
    summary_score: float = 0.0
    line_items_score: float = 0.0
    line_item_gold_available: bool = False
    gold_line_item_count: int = 0
    predicted_line_item_count: int = 0
    line_item_count_delta: int | None = None
    line_item_count_score: float | None = None
    reconciliation_score: float = 0.0
    reconciliation_delta: float | None = None
    reconciliation_status: str | None = None
    summary_reconciliation_delta: float | None = None
    summary_reconciliation_status: str | None = None
    line_item_reconciliation_delta: float | None = None
    line_item_reconciliation_status: str | None = None
    deterministic_success: bool = False
    error: str | None = None
    judge: JudgeEvaluation | None = None
    created_at: str


class EvalSummary(BaseModel):
    generated_at: str
    dataset_root: str
    output_dir: str
    expected_total_records: int
    completed_records: int
    counts: dict[str, int] = Field(default_factory=dict)
    dataset_status_counts: dict[str, int] = Field(default_factory=dict)
    mean_score: float = 0.0
    records_with_errors: int = 0
    field_mean_scores: dict[str, float] = Field(default_factory=dict)
    component_mean_scores: dict[str, float] = Field(default_factory=dict)
    line_item_availability_counts: dict[str, int] = Field(default_factory=dict)
    top_failure_reasons: dict[str, int] = Field(default_factory=dict)


class EvalRunResult(BaseModel):
    output_dir: str
    processed_records: int
    expected_total_records: int
    results_path: str
    summary_path: str
    report_path: str


def now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def resolve_eval_output_dir(output_dir: str | Path | None = None) -> Path:
    if output_dir is not None:
        return Path(output_dir)
    configured = os.getenv("RECEIPT_EVAL_OUTPUT_DIR")
    if configured:
        return Path(configured)
    return DEFAULT_EVAL_OUTPUT_DIR


def load_results_jsonl(output_dir: str | Path | None = None) -> list[ReceiptEvalRecord]:
    resolved_output_dir = resolve_eval_output_dir(output_dir)
    results_path = resolved_output_dir / "results.jsonl"
    if not results_path.exists():
        return []

    records_by_sample_id: dict[str, ReceiptEvalRecord] = {}
    for line in results_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        record = ReceiptEvalRecord.model_validate_json(stripped)
        records_by_sample_id[record.sample_id] = record
    return sorted(records_by_sample_id.values(), key=lambda item: item.sample_id)


def load_eval_summary(output_dir: str | Path | None = None) -> EvalSummary | None:
    resolved_output_dir = resolve_eval_output_dir(output_dir)
    summary_path = resolved_output_dir / "summary.json"
    if not summary_path.exists():
        return None
    return EvalSummary.model_validate_json(summary_path.read_text(encoding="utf-8"))


def load_eval_report_markdown(output_dir: str | Path | None = None) -> str:
    resolved_output_dir = resolve_eval_output_dir(output_dir)
    report_path = resolved_output_dir / "report.md"
    if not report_path.exists():
        return ""
    return report_path.read_text(encoding="utf-8")


def extract_json_object(text: str) -> dict[str, Any]:
    decoder = JSONDecoder()
    for index, char in enumerate(text):
        if char != "{":
            continue
        try:
            payload, _ = decoder.raw_decode(text[index:])
        except JSONDecodeError:
            continue
        if isinstance(payload, dict):
            return payload
    raise ValueError("No JSON object found in model response")


def message_text_from_completion(completion: Any) -> str:
    try:
        content = completion.choices[0].message.content
    except (AttributeError, IndexError, KeyError, TypeError) as exc:
        raise ValueError("Model response did not contain a message content payload") from exc

    if isinstance(content, str):
        return content
    if isinstance(content, list):
        text_parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                text_parts.append(item)
                continue
            if isinstance(item, dict) and item.get("type") == "text":
                text_parts.append(str(item.get("text", "")))
                continue
            maybe_text = getattr(item, "text", None)
            if maybe_text:
                text_parts.append(str(maybe_text))
        return "\n".join(part for part in text_parts if part).strip()
    return str(content).strip()


def _line_item_from_payload(payload: Any, index: int) -> ReceiptLineItem | None:
    if not isinstance(payload, dict):
        return None
    description = payload.get("description")
    line_total = payload.get("line_total")
    if description is not None:
        description = str(description).strip() or None
    if line_total is not None:
        line_total = str(line_total).strip() or None
    if not description and not line_total:
        return None
    return ReceiptLineItem(item_id=f"predicted:{index}", description=description, line_total=line_total)


def draft_from_payload(payload: dict[str, Any]) -> ReceiptDraft:
    values: dict[str, Any] = {}
    for field in FIELD_NAMES:
        value = payload.get(field)
        if value is None:
            values[field] = None
        elif isinstance(value, str):
            normalized = value.strip()
            values[field] = normalized or None
        else:
            values[field] = str(value).strip() or None

    raw_line_items = payload.get("line_items", [])
    line_items: list[ReceiptLineItem] = []
    if isinstance(raw_line_items, list):
        for index, item in enumerate(raw_line_items):
            parsed = _line_item_from_payload(item, index)
            if parsed is not None:
                line_items.append(parsed)
    values["line_items"] = line_items
    return ReceiptDraft(**values)


def field_status_from_score(score: float, predicted: str | None, gold: str | None) -> FieldStatus:
    if gold is None:
        return "not_evaluated"
    if score >= 1.0:
        return "correct"
    if score > 0.0:
        return "partial"
    if predicted:
        return "incorrect"
    return "missing"


def build_field_results(
    predicted_fields: ReceiptDraft | None,
    gold_fields: ReceiptDraft | None,
    field_scores: dict[str, float] | None = None,
) -> dict[str, FieldEvaluation]:
    results: dict[str, FieldEvaluation] = {}
    field_scores = field_scores or {}
    for field in FIELD_NAMES:
        predicted = getattr(predicted_fields, field) if predicted_fields is not None else None
        gold = getattr(gold_fields, field) if gold_fields is not None else None
        score = round(float(field_scores.get(field, 0.0)), 6)
        results[field] = FieldEvaluation(
            predicted=predicted,
            gold=gold,
            score=score,
            status=field_status_from_score(score, predicted, gold),
        )
    return results


def classify_eval_status(dataset_status: DatasetStatus, score: float, predicted_fields: ReceiptDraft | None, error: str | None) -> EvalStatus:
    if dataset_status != "runnable":
        return "skipped"
    if error:
        return "failed"
    if predicted_fields is None:
        return "failed"
    if score >= 1.0:
        return "worked"
    if score > 0.0:
        return "partial"
    return "failed"


def image_to_data_url(image_path: str | Path) -> str:
    path = Path(image_path)
    mime_type, _ = mimetypes.guess_type(path.name)
    mime_type = mime_type or "application/octet-stream"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def resolve_api_key() -> str:
    for key_name in ("OPENAI_API_KEY", "HF_TOKEN", "API_KEY"):
        value = os.getenv(key_name, "").strip()
        if value:
            return value
    return "not-needed"


def require_env(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise ValueError(f"Missing required environment variable: {name}")
    return value


def build_model_client(base_url: str) -> OpenAI:
    return OpenAI(base_url=base_url, api_key=resolve_api_key())


def build_extraction_messages(record: DatasetAuditRecord) -> list[dict[str, Any]]:
    return [
        {"role": "system", "content": EXTRACTION_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": (
                        "Extract the receipt into JSON with keys company, date, address, subtotal, tax, total, line_items. "
                        "Each line_items entry should include description and line_total. Use null or [] for unknown values."
                    ),
                },
                {
                    "type": "image_url",
                    "image_url": {"url": image_to_data_url(record.image_path or "")},
                },
            ],
        },
    ]


def build_judge_messages(record: DatasetAuditRecord, prediction: ReceiptDraft | None, grade: dict[str, Any], error: str | None) -> list[dict[str, Any]]:
    gold_fields = record.gold_fields.model_dump() if record.gold_fields is not None else {}
    predicted_fields = prediction.model_dump() if prediction is not None else {field: None for field in FIELD_NAMES}
    payload = {
        "sample_id": record.sample_id,
        "task_id": record.task_id,
        "gold_fields": gold_fields,
        "gold_line_items": [item.model_dump() for item in record.gold_line_items],
        "predicted_fields": predicted_fields,
        "predicted_line_items": [item.model_dump() for item in (prediction.line_items if prediction is not None else [])],
        "grade": grade,
        "error": error,
        "instructions": (
            "Explain whether the extraction worked, what is wrong if it did not, "
            "and identify likely failure reasons. Return JSON only."
        ),
    }
    return [
        {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": json.dumps(payload, indent=2)},
                {
                    "type": "image_url",
                    "image_url": {"url": image_to_data_url(record.image_path or "")},
                },
            ],
        },
    ]


def run_extraction_model(record: DatasetAuditRecord, client: Any, model_name: str) -> ReceiptDraft:
    completion = cached_chat_completion(
        client,
        model=model_name,
        messages=build_extraction_messages(record),
        temperature=0,
        response_format={"type": "json_object"},
    )
    payload = extract_json_object(message_text_from_completion(completion))
    return draft_from_payload(payload)


def run_judge_model(
    record: DatasetAuditRecord,
    prediction: ReceiptDraft | None,
    grade_payload: dict[str, Any],
    error: str | None,
    client: Any,
    model_name: str,
) -> JudgeEvaluation:
    completion = cached_chat_completion(
        client,
        model=model_name,
        messages=build_judge_messages(record, prediction, grade_payload, error),
        temperature=0,
        response_format={"type": "json_object"},
    )
    payload = extract_json_object(message_text_from_completion(completion))
    summary = str(payload.get("summary", "")).strip() or "No judge summary returned."
    failure_reasons = payload.get("failure_reasons", [])
    if not isinstance(failure_reasons, list):
        failure_reasons = [str(failure_reasons)]
    field_notes = payload.get("field_notes", {})
    if not isinstance(field_notes, dict):
        field_notes = {}
    return JudgeEvaluation(
        summary=summary,
        failure_reasons=[str(reason).strip() for reason in failure_reasons if str(reason).strip()],
        field_notes={str(key): str(value) for key, value in field_notes.items() if str(value).strip()},
    )


def fallback_judge_evaluation(record: DatasetAuditRecord, field_results: dict[str, FieldEvaluation], error: str | None) -> JudgeEvaluation:
    if record.dataset_status != "runnable":
        return JudgeEvaluation(
            summary=f"Skipped during dataset audit: {record.skip_reason or record.dataset_status}",
            failure_reasons=[record.skip_reason or record.dataset_status],
        )

    incorrect_fields = [field for field, result in field_results.items() if result.status in {"incorrect", "missing", "partial"}]
    if error:
        return JudgeEvaluation(summary=f"Evaluation failed before scoring completed: {error}", failure_reasons=["runtime_error"])
    if not incorrect_fields:
        return JudgeEvaluation(summary="All evaluated receipt fields matched the gold labels under deterministic grading.")
    return JudgeEvaluation(
        summary=f"Deterministic grading found issues in: {', '.join(incorrect_fields)}.",
        failure_reasons=[f"{field}_mismatch" for field in incorrect_fields],
    )


def evaluate_audit_record(
    record: DatasetAuditRecord,
    extractor_client: Any,
    extractor_model: str,
    judge_client: Any,
    judge_model: str,
) -> ReceiptEvalRecord:
    created_at = now_utc_iso()
    if record.dataset_status != "runnable":
        return ReceiptEvalRecord(
            sample_id=record.sample_id,
            task_id=record.task_id,
            annotation_path=record.annotation_path,
            image_path=record.image_path,
            dataset_status=record.dataset_status,
            status="skipped",
            skip_reason=record.skip_reason,
            gold_fields=record.gold_fields,
            gold_line_items=record.gold_line_items,
            field_results=build_field_results(None, record.gold_fields),
            judge=fallback_judge_evaluation(record, {}, None),
            created_at=created_at,
        )

    prediction: ReceiptDraft | None = None
    grade_payload: dict[str, Any] = {}
    overall_score = 0.0
    header_score = 0.0
    summary_score = 0.0
    line_items_score = 0.0
    line_item_gold_available = bool(record.gold_line_items)
    gold_line_item_count = len(record.gold_line_items)
    predicted_line_item_count = 0
    line_item_count_delta: int | None = None
    line_item_count_score: float | None = None
    reconciliation_score = 0.0
    reconciliation_delta: float | None = None
    reconciliation_status: str | None = None
    summary_reconciliation_delta: float | None = None
    summary_reconciliation_status: str | None = None
    line_item_reconciliation_delta: float | None = None
    line_item_reconciliation_status: str | None = None
    deterministic_success = False
    error: str | None = None

    try:
        prediction = run_extraction_model(record, extractor_client, extractor_model)
        grade = grade_receipt(
            prediction,
            record.gold_fields or ReceiptDraft(),
            task_id=record.task_id or "easy",
            gold_line_items=record.gold_line_items,
        )
        grade_payload = grade.model_dump(mode="json")
        overall_score = float(grade.score)
        header_score = float(grade.header_score)
        summary_score = float(grade.summary_score)
        line_items_score = float(grade.line_items_score)
        line_item_gold_available = bool(grade.line_item_gold_available)
        gold_line_item_count = int(grade.gold_line_item_count)
        predicted_line_item_count = int(grade.predicted_line_item_count)
        line_item_count_delta = grade.line_item_count_delta
        line_item_count_score = grade.line_item_count_score
        reconciliation_score = float(grade.reconciliation_score)
        reconciliation_delta = grade.reconciliation_delta
        reconciliation_status = grade.reconciliation_status
        summary_reconciliation_delta = grade.summary_reconciliation_delta
        summary_reconciliation_status = grade.summary_reconciliation_status
        line_item_reconciliation_delta = grade.line_item_reconciliation_delta
        line_item_reconciliation_status = grade.line_item_reconciliation_status
        deterministic_success = bool(grade.success)
    except Exception as exc:  # pragma: no cover - exact client failures vary
        error = str(exc)

    field_results = build_field_results(prediction, record.gold_fields, grade_payload.get("field_scores") if grade_payload else {})
    status = classify_eval_status(record.dataset_status, overall_score, prediction, error)

    judge: JudgeEvaluation | None = None
    try:
        judge = run_judge_model(record, prediction, grade_payload, error, judge_client, judge_model)
    except Exception:  # pragma: no cover - exact client failures vary
        judge = fallback_judge_evaluation(record, field_results, error)

    return ReceiptEvalRecord(
        sample_id=record.sample_id,
        task_id=record.task_id,
        annotation_path=record.annotation_path,
        image_path=record.image_path,
        dataset_status=record.dataset_status,
        status=status,
        skip_reason=record.skip_reason,
        gold_fields=record.gold_fields,
        gold_line_items=record.gold_line_items,
        predicted_fields=prediction,
        predicted_line_items=list(prediction.line_items) if prediction is not None else [],
        field_results=field_results,
        overall_score=round(overall_score, 6),
        header_score=round(header_score, 6),
        summary_score=round(summary_score, 6),
        line_items_score=round(line_items_score, 6),
        line_item_gold_available=line_item_gold_available,
        gold_line_item_count=gold_line_item_count,
        predicted_line_item_count=predicted_line_item_count,
        line_item_count_delta=line_item_count_delta,
        line_item_count_score=None if line_item_count_score is None else round(float(line_item_count_score), 6),
        reconciliation_score=round(reconciliation_score, 6),
        reconciliation_delta=reconciliation_delta,
        reconciliation_status=reconciliation_status,
        summary_reconciliation_delta=summary_reconciliation_delta,
        summary_reconciliation_status=summary_reconciliation_status,
        line_item_reconciliation_delta=line_item_reconciliation_delta,
        line_item_reconciliation_status=line_item_reconciliation_status,
        deterministic_success=deterministic_success,
        error=error,
        judge=judge,
        created_at=created_at,
    )


def _missing(values: dict[str, str | None], required_fields: tuple[str, ...]) -> list[str]:
    return [field for field in required_fields if not values.get(field)]


def _task_from_labels(label_presence: dict[str, bool]) -> tuple[str | None, list[str]]:
    hard_required = {
        "Business name": "company",
        "Time and date": "date",
        "Subtotal": "subtotal",
        "Tax": "tax",
        "Total": "total",
        "Item information": "line_items",
    }
    medium_required = {
        "Business name": "company",
        "Time and date": "date",
        "Subtotal": "subtotal",
        "Tax": "tax",
        "Total": "total",
    }
    easy_required = {
        "Business name": "company",
        "Business address": "address",
        "Time and date": "date",
        "Total": "total",
    }
    if all(label_presence.get(label, False) for label in hard_required):
        return "hard", []
    if all(label_presence.get(label, False) for label in medium_required):
        return "medium", []
    missing_easy = [label for label in easy_required if not label_presence.get(label, False)]
    if not missing_easy:
        return "easy", []
    return None, missing_easy


def audit_dataset(dataset_root: str | Path | None = None) -> list[DatasetAuditRecord]:
    dataset = ReceiptDataset(dataset_root=dataset_root)
    ann_dir = dataset.dataset_root / "ann"
    img_dir = dataset.dataset_root / "img"
    if not ann_dir.exists() or not img_dir.exists():
        raise FileNotFoundError(f"Dataset root does not contain ann/ and img/: {dataset.dataset_root}")

    records: list[DatasetAuditRecord] = []
    task_requirements = {
        "easy": ("company", "address", "date", "total"),
        "medium": ("company", "date", "subtotal", "tax", "total"),
        "hard": ("company", "date", "subtotal", "tax", "total"),
    }
    for annotation_path in sorted(ann_dir.glob("*.json")):
        payload = json.loads(annotation_path.read_text(encoding="utf-8"))
        objects = payload.get("objects", [])
        grouped_regions: dict[str, list[Any]] = {}
        label_presence: dict[str, bool] = {}

        for obj in objects:
            region = dataset._build_region(obj)
            if region is None:
                continue
            category = dataset._tag_value(obj, "Category")
            if category:
                grouped_regions.setdefault(category, []).append(region)
                label_presence[category] = True

        sample_id = annotation_path.stem
        image_name = annotation_path.name[:-5]
        image_path = img_dir / image_name
        if not image_path.exists():
            records.append(
                DatasetAuditRecord(
                    sample_id=sample_id,
                    annotation_path=str(annotation_path),
                    image_path=str(image_path),
                    dataset_status="skipped_missing_image",
                    skip_reason="missing_image",
                )
            )
            continue

        labeled_task_id, missing_categories = _task_from_labels(label_presence)
        if labeled_task_id is None:
            records.append(
                DatasetAuditRecord(
                    sample_id=sample_id,
                    annotation_path=str(annotation_path),
                    image_path=str(image_path),
                    dataset_status="skipped_missing_labels",
                    skip_reason="missing_required_labels",
                    missing_categories=missing_categories,
                )
            )
            continue

        values = {
            "company": dataset._join_text(grouped_regions.get("Business name", [])) or None,
            "address": dataset._join_text(grouped_regions.get("Business address", [])) or None,
            "date": dataset._pick_date(grouped_regions.get("Time and date", [])) or None,
            "subtotal": dataset._pick_amount(grouped_regions.get("Subtotal", [])) or None,
            "tax": dataset._pick_amount(grouped_regions.get("Tax", [])) or None,
            "total": dataset._pick_amount(grouped_regions.get("Total", [])) or None,
        }
        gold_line_items = dataset._extract_line_items(grouped_regions.get("Item information", []))
        gold_fields = ReceiptDraft(**values)

        if labeled_task_id == "hard" and gold_line_items and not _missing(values, task_requirements["hard"]):
            resolved_task_id = "hard"
            unparseable_fields: list[str] = []
        elif labeled_task_id in {"hard", "medium"} and not _missing(values, task_requirements["medium"]):
            resolved_task_id = "medium"
            unparseable_fields = []
        elif not _missing(values, task_requirements["easy"]):
            resolved_task_id = "easy"
            unparseable_fields = []
        else:
            resolved_task_id = labeled_task_id
            required_fields = task_requirements["easy"] if labeled_task_id == "easy" else task_requirements["medium"]
            unparseable_fields = _missing(values, required_fields)
            if labeled_task_id == "hard" and not gold_line_items:
                unparseable_fields.append("line_items")

        if unparseable_fields:
            records.append(
                DatasetAuditRecord(
                    sample_id=sample_id,
                    task_id=resolved_task_id,
                    annotation_path=str(annotation_path),
                    image_path=str(image_path),
                    dataset_status="skipped_unparseable_gold",
                    skip_reason="unparseable_gold_fields",
                    unparseable_fields=sorted(set(unparseable_fields)),
                    gold_fields=gold_fields,
                    gold_line_items=gold_line_items,
                )
            )
            continue

        records.append(
            DatasetAuditRecord(
                sample_id=sample_id,
                task_id=resolved_task_id,
                annotation_path=str(annotation_path),
                image_path=str(image_path),
                dataset_status="runnable",
                gold_fields=gold_fields,
                gold_line_items=gold_line_items,
            )
        )

    return records


def build_eval_summary(
    records: list[ReceiptEvalRecord],
    dataset_root: str | Path,
    output_dir: str | Path,
    expected_total_records: int,
) -> EvalSummary:
    counts = Counter(record.status for record in records)
    dataset_status_counts = Counter(record.dataset_status for record in records)
    scored_records = [record for record in records if record.status != "skipped"]
    field_mean_scores: dict[str, float] = {}
    for field in FIELD_NAMES:
        scores = [record.field_results[field].score for record in scored_records if field in record.field_results]
        field_mean_scores[field] = round(mean(scores), 6) if scores else 0.0

    failure_reasons = Counter()
    for record in records:
        if record.judge is None:
            continue
        for reason in record.judge.failure_reasons:
            failure_reasons[reason] += 1

    line_item_availability_counts = Counter(
        "with_gold_line_items" if record.line_item_gold_available else "without_gold_line_items"
        for record in records
    )
    component_mean_scores = {
        "header_score": round(mean(record.header_score for record in scored_records), 6) if scored_records else 0.0,
        "summary_score": round(mean(record.summary_score for record in scored_records), 6) if scored_records else 0.0,
        "line_items_score": round(mean(record.line_items_score for record in scored_records), 6) if scored_records else 0.0,
        "reconciliation_score": round(mean(record.reconciliation_score for record in scored_records), 6) if scored_records else 0.0,
        "line_item_count_score": round(
            mean(record.line_item_count_score for record in scored_records if record.line_item_count_score is not None),
            6,
        )
        if any(record.line_item_count_score is not None for record in scored_records)
        else 0.0,
    }

    return EvalSummary(
        generated_at=now_utc_iso(),
        dataset_root=str(dataset_root),
        output_dir=str(output_dir),
        expected_total_records=expected_total_records,
        completed_records=len(records),
        counts=dict(counts),
        dataset_status_counts=dict(dataset_status_counts),
        mean_score=round(mean(record.overall_score for record in scored_records), 6) if scored_records else 0.0,
        records_with_errors=sum(1 for record in records if record.error),
        field_mean_scores=field_mean_scores,
        component_mean_scores=component_mean_scores,
        line_item_availability_counts=dict(line_item_availability_counts),
        top_failure_reasons=dict(failure_reasons.most_common(10)),
    )


def render_markdown_report(summary: EvalSummary, records: list[ReceiptEvalRecord]) -> str:
    grouped: dict[str, list[ReceiptEvalRecord]] = {status: [] for status in ("worked", "partial", "failed", "skipped")}
    for record in records:
        grouped.setdefault(record.status, []).append(record)

    lines = [
        "# Dataset Image Evaluation Report",
        "",
        f"- Generated at: `{summary.generated_at}`",
        f"- Dataset root: `{summary.dataset_root}`",
        f"- Output dir: `{summary.output_dir}`",
        f"- Expected records: `{summary.expected_total_records}`",
        f"- Completed records: `{summary.completed_records}`",
        f"- Mean score (non-skipped): `{summary.mean_score:.3f}`",
        "",
        "## Counts",
        "",
        f"- Worked: `{summary.counts.get('worked', 0)}`",
        f"- Partial: `{summary.counts.get('partial', 0)}`",
        f"- Failed: `{summary.counts.get('failed', 0)}`",
        f"- Skipped: `{summary.counts.get('skipped', 0)}`",
        "",
        "## Component Means",
        "",
    ]

    for name, score in summary.component_mean_scores.items():
        lines.append(f"- `{name}`: `{score:.3f}`")

    if summary.line_item_availability_counts:
        lines.extend(["", "## Line Item Gold Availability", ""])
        for name, count in sorted(summary.line_item_availability_counts.items()):
            lines.append(f"- `{name}`: `{count}`")

    lines.extend(["", "## Dataset Audit Status", ""])
    for status_name, count in sorted(summary.dataset_status_counts.items()):
        lines.append(f"- `{status_name}`: `{count}`")

    if summary.top_failure_reasons:
        lines.extend(["", "## Top Failure Reasons", ""])
        for reason, count in summary.top_failure_reasons.items():
            lines.append(f"- `{reason}`: `{count}`")

    for section_name in ("worked", "partial", "failed", "skipped"):
        lines.extend(["", f"## {section_name.title()} Records", ""])
        section_records = grouped.get(section_name, [])
        if not section_records:
            lines.append("- None")
            continue

        for record in section_records[:25]:
            issue_bits: list[str] = []
            if record.task_id:
                issue_bits.append(f"task={record.task_id}")
            if record.skip_reason:
                issue_bits.append(f"skip={record.skip_reason}")
            if record.error:
                issue_bits.append(f"error={record.error}")
            if record.reconciliation_status:
                issue_bits.append(f"reconciliation={record.reconciliation_status}")
            if record.judge and record.judge.failure_reasons:
                issue_bits.append(f"judge={', '.join(record.judge.failure_reasons)}")
            issue_suffix = f" ({'; '.join(issue_bits)})" if issue_bits else ""
            lines.append(f"- `{record.sample_id}` score=`{record.overall_score:.3f}`{issue_suffix}")

    return "\n".join(lines) + "\n"


def write_eval_artifacts(
    output_dir: str | Path,
    records: list[ReceiptEvalRecord],
    summary: EvalSummary,
    report_markdown: str,
) -> None:
    resolved_output_dir = resolve_eval_output_dir(output_dir)
    resolved_output_dir.mkdir(parents=True, exist_ok=True)

    results_path = resolved_output_dir / "results.jsonl"
    results_body = "\n".join(record.model_dump_json() for record in records)
    if results_body:
        results_body += "\n"
    results_path.write_text(results_body, encoding="utf-8")

    (resolved_output_dir / "summary.json").write_text(summary.model_dump_json(indent=2), encoding="utf-8")
    (resolved_output_dir / "report.md").write_text(report_markdown, encoding="utf-8")


def append_eval_record(output_dir: str | Path, record: ReceiptEvalRecord) -> None:
    resolved_output_dir = resolve_eval_output_dir(output_dir)
    resolved_output_dir.mkdir(parents=True, exist_ok=True)
    with (resolved_output_dir / "results.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(record.model_dump_json())
        handle.write("\n")


def existing_result_sample_ids(output_dir: str | Path | None = None) -> set[str]:
    return {record.sample_id for record in load_results_jsonl(output_dir)}


def get_audit_record(sample_id: str, dataset_root: str | Path | None = None) -> DatasetAuditRecord | None:
    for record in audit_dataset(dataset_root):
        if record.sample_id == sample_id:
            return record
    return None


def evaluate_dataset_images(
    dataset_root: str | Path | None = None,
    output_dir: str | Path | None = None,
    limit: int | None = None,
    resume: bool = False,
    extractor_client: Any | None = None,
    judge_client: Any | None = None,
) -> EvalRunResult:
    resolved_output_dir = resolve_eval_output_dir(output_dir)
    audits = audit_dataset(dataset_root)
    dataset = ReceiptDataset(dataset_root=dataset_root)
    completed_ids = existing_result_sample_ids(resolved_output_dir) if resume else set()

    extractor_model = require_env("MODEL_NAME")
    extractor_base_url = require_env("API_BASE_URL")
    judge_model = require_env("EVAL_MODEL")
    judge_base_url = require_env("EVAL_API_BASE_URL")

    extractor_client = extractor_client or build_model_client(extractor_base_url)
    judge_client = judge_client or build_model_client(judge_base_url)

    processed = 0
    for audit in audits:
        if audit.sample_id in completed_ids:
            continue
        if limit is not None and processed >= limit:
            break

        logger.info("Evaluating receipt: %s", audit.sample_id)
        record = evaluate_audit_record(audit, extractor_client, extractor_model, judge_client, judge_model)
        append_eval_record(resolved_output_dir, record)
        processed += 1

    records = load_results_jsonl(resolved_output_dir)
    summary = build_eval_summary(records, dataset.dataset_root, resolved_output_dir, len(audits))
    report_markdown = render_markdown_report(summary, records)
    write_eval_artifacts(resolved_output_dir, records, summary, report_markdown)

    return EvalRunResult(
        output_dir=str(resolved_output_dir),
        processed_records=processed,
        expected_total_records=len(audits),
        results_path=str(resolved_output_dir / "results.jsonl"),
        summary_path=str(resolved_output_dir / "summary.json"),
        report_path=str(resolved_output_dir / "report.md"),
    )


def evaluate_single_receipt(
    sample_id: str,
    dataset_root: str | Path | None = None,
    output_dir: str | Path | None = None,
    extractor_client: Any | None = None,
    judge_client: Any | None = None,
) -> ReceiptEvalRecord:
    resolved_output_dir = resolve_eval_output_dir(output_dir)
    audits = audit_dataset(dataset_root)
    audit = next((record for record in audits if record.sample_id == sample_id), None)
    if audit is None:
        raise KeyError(f"Unknown sample_id={sample_id}")

    logger.info("Evaluating single receipt: %s", sample_id)

    dataset = ReceiptDataset(dataset_root=dataset_root)
    extractor_model = require_env("MODEL_NAME")
    extractor_base_url = require_env("API_BASE_URL")
    judge_model = require_env("EVAL_MODEL")
    judge_base_url = require_env("EVAL_API_BASE_URL")

    extractor_client = extractor_client or build_model_client(extractor_base_url)
    judge_client = judge_client or build_model_client(judge_base_url)

    updated_record = evaluate_audit_record(audit, extractor_client, extractor_model, judge_client, judge_model)
    records_by_sample_id = {record.sample_id: record for record in load_results_jsonl(resolved_output_dir)}
    records_by_sample_id[updated_record.sample_id] = updated_record
    merged_records = sorted(records_by_sample_id.values(), key=lambda item: item.sample_id)

    summary = build_eval_summary(merged_records, dataset.dataset_root, resolved_output_dir, len(audits))
    report_markdown = render_markdown_report(summary, merged_records)
    write_eval_artifacts(resolved_output_dir, merged_records, summary, report_markdown)
    return updated_record


class EvalArtifactStore:
    def __init__(self, output_dir: str | Path | None = None) -> None:
        self.output_dir = resolve_eval_output_dir(output_dir)

    @property
    def results_path(self) -> Path:
        return self.output_dir / "results.jsonl"

    @property
    def summary_path(self) -> Path:
        return self.output_dir / "summary.json"

    @property
    def report_path(self) -> Path:
        return self.output_dir / "report.md"

    def exists(self) -> bool:
        return self.results_path.exists()

    def records(self) -> list[ReceiptEvalRecord]:
        return load_results_jsonl(self.output_dir)

    def summary(self) -> EvalSummary | None:
        return load_eval_summary(self.output_dir)

    def report_markdown(self) -> str:
        return load_eval_report_markdown(self.output_dir)

    def get_record(self, sample_id: str) -> ReceiptEvalRecord | None:
        for record in self.records():
            if record.sample_id == sample_id:
                return record
        return None

    def audit_records(self) -> list[DatasetAuditRecord]:
        return audit_dataset()

    def receipt_menu(self) -> list[dict[str, Any]]:
        records_by_sample_id = {record.sample_id: record for record in self.records()}
        items: list[dict[str, Any]] = []
        seen_sample_ids: set[str] = set()
        for audit in self.audit_records():
            record = records_by_sample_id.get(audit.sample_id)
            seen_sample_ids.add(audit.sample_id)
            items.append(
                {
                    "sample_id": audit.sample_id,
                    "task_id": audit.task_id,
                    "image_path": audit.image_path,
                    "dataset_status": audit.dataset_status,
                    "status": record.status if record is not None else "not_run",
                    "overall_score": record.overall_score if record is not None else None,
                    "processed": record is not None,
                    "processable": audit.dataset_status == "runnable",
                }
            )
        for record in records_by_sample_id.values():
            if record.sample_id in seen_sample_ids:
                continue
            items.append(
                {
                    "sample_id": record.sample_id,
                    "task_id": record.task_id,
                    "image_path": record.image_path,
                    "dataset_status": record.dataset_status,
                    "status": record.status,
                    "overall_score": record.overall_score,
                    "processed": True,
                    "processable": record.dataset_status == "runnable",
                }
            )
        items.sort(key=lambda item: item["sample_id"])
        return items

    def list_records(
        self,
        status: str | None = None,
        sample_id: str | None = None,
        has_errors: bool | None = None,
        page: int = 1,
        page_size: int = 25,
    ) -> dict[str, Any]:
        records = self.records()
        if status:
            records = [record for record in records if record.status == status]
        if sample_id:
            needle = sample_id.lower()
            records = [record for record in records if needle in record.sample_id.lower()]
        if has_errors is True:
            records = [record for record in records if record.status in {"partial", "failed"} or bool(record.error)]
        elif has_errors is False:
            records = [record for record in records if record.status == "worked" and not record.error]

        page = max(page, 1)
        page_size = max(min(page_size, 100), 1)
        start = (page - 1) * page_size
        items = records[start : start + page_size]
        return {
            "items": [record.model_dump(mode="json") for record in items],
            "page": page,
            "page_size": page_size,
            "total": len(records),
            "pages": max((len(records) + page_size - 1) // page_size, 1),
        }
