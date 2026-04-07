from __future__ import annotations

import os
from pathlib import Path
from typing import Any
from urllib.parse import quote, urlencode

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import HTMLResponse, PlainTextResponse, RedirectResponse, Response
from fastapi.templating import Jinja2Templates

from env.evaluation import EvalArtifactStore, build_field_results, evaluate_single_receipt, get_audit_record
from env.graders import score_formula_definition, score_formula_numeric, score_formula_term_contributions
from env.image_store import ImageStoreError, decode_image_json_bytes

BASE_DIR = Path(__file__).resolve().parents[1]
TEMPLATES = Jinja2Templates(directory=str(BASE_DIR / "server" / "templates"))
TEMPLATES.env.globals["static_asset_version"] = str(int((BASE_DIR / "server" / "static" / "eval.css").stat().st_mtime))

api_router = APIRouter(prefix="/api/eval", tags=["eval"])
ui_router = APIRouter(tags=["eval-ui"])

TASK_IDS = ("easy", "medium", "hard")
TASK_LABELS = {"easy": "Easy", "medium": "Medium", "hard": "Hard"}


def get_store() -> EvalArtifactStore:
    return EvalArtifactStore()


def eval_model_config() -> dict[str, str]:
    return {
        "extractor_model": os.getenv("MODEL_NAME", "").strip() or "not-configured",
        "extractor_base_url": os.getenv("API_BASE_URL", "").strip() or "not-configured",
        "judge_model": os.getenv("EVAL_MODEL", "").strip() or "not-configured",
        "judge_base_url": os.getenv("EVAL_API_BASE_URL", "").strip() or "not-configured",
    }


def line_item_rows(record_payload: dict[str, Any]) -> list[dict[str, Any]]:
    predicted_items = record_payload.get("predicted_line_items") or []
    gold_items = record_payload.get("gold_line_items") or []
    row_count = max(len(predicted_items), len(gold_items))
    rows: list[dict[str, Any]] = []
    for index in range(row_count):
        predicted = predicted_items[index] if index < len(predicted_items) else {}
        gold = gold_items[index] if index < len(gold_items) else {}
        rows.append(
            {
                "index": index,
                "predicted_description": predicted.get("description"),
                "predicted_total": predicted.get("line_total"),
                "gold_description": gold.get("description"),
                "gold_total": gold.get("line_total"),
            }
        )
    return rows


def _safe_float(value: Any) -> float:
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _field_score(record_payload: dict[str, Any], field_name: str) -> float:
    field_result = (record_payload.get("field_results") or {}).get(field_name) or {}
    if hasattr(field_result, "score"):
        return _safe_float(field_result.score)
    if isinstance(field_result, dict):
        return _safe_float(field_result.get("score"))
    return 0.0


def _formula_source_scores(record_payload: dict[str, Any], formula_terms: list[dict[str, Any]]) -> dict[str, float]:
    source_scores: dict[str, float] = {}
    field_results = record_payload.get("field_results") or {}
    for term in formula_terms:
        source_key = term["source_key"]
        if source_key in field_results:
            source_scores[source_key] = _field_score(record_payload, source_key)
            continue

        component_score = _safe_float(record_payload.get(term["component"]))
        weight = _safe_float(term["weight"])
        source_scores[source_key] = component_score / weight if weight > 0.0 else 0.0
    return source_scores


def score_explanation_payload(record_payload: dict[str, Any]) -> dict[str, Any]:
    raw_task_id = record_payload.get("task_id")
    task_id = str(raw_task_id or "").lower()
    overall_score = _safe_float(record_payload.get("overall_score"))
    notes: list[str] = []

    if not record_payload.get("processed", True):
        notes.append("This receipt has not been run yet, so displayed component scores are placeholders.")

    dataset_status = record_payload.get("dataset_status")
    if dataset_status and dataset_status != "runnable":
        reason = record_payload.get("skip_reason") or dataset_status
        return {
            "title": "Scoring skipped",
            "formula": "score = 0.000 because the dataset audit marked this receipt as non-runnable.",
            "numeric_formula": f"dataset_status={dataset_status}; reason={reason}",
            "terms": [],
            "notes": [
                "The deterministic grader only applies weighted scoring to runnable receipts.",
            ],
        }

    formula_task_id = raw_task_id or "easy"
    if not raw_task_id and dataset_status == "runnable":
        notes.append("No task_id was recorded, so this explanation uses the grader default easy formula.")

    if raw_task_id and task_id not in {"easy", "medium", "hard"}:
        notes.append(f"Unrecognized task_id={task_id}; this explanation uses the grader's hard-task fallback.")

    formula_definition = score_formula_definition(formula_task_id)
    source_scores = _formula_source_scores(record_payload, formula_definition["terms"])
    terms = score_formula_term_contributions(formula_task_id, source_scores)
    notes.extend(formula_definition["notes"])

    reconciliation_status = record_payload.get("reconciliation_status")
    if reconciliation_status:
        reconciliation_delta = record_payload.get("reconciliation_delta")
        if reconciliation_delta is None:
            notes.append(f"reconciliation status: {reconciliation_status}.")
        else:
            notes.append(f"reconciliation status: {reconciliation_status}; delta={_safe_float(reconciliation_delta):.2f}.")

    return {
        "title": formula_definition["title"],
        "formula": formula_definition["formula"],
        "numeric_formula": score_formula_numeric(terms, overall_score),
        "terms": terms,
        "notes": notes,
    }


def enrich_detail_payload(payload: dict[str, Any]) -> dict[str, Any]:
    payload["line_item_rows"] = line_item_rows(payload)
    payload["score_explanation"] = score_explanation_payload(payload)
    return payload


def merge_audit_metadata(payload: dict[str, Any], audit: Any) -> dict[str, Any]:
    for key in ("task_id", "image_id", "image_json_path"):
        if not payload.get(key):
            payload[key] = getattr(audit, key, None)
    if not payload.get("annotation_path"):
        payload["annotation_path"] = audit.annotation_path
    return payload


def audit_detail_payload(audit: Any) -> dict[str, Any]:
    return {
        "sample_id": audit.sample_id,
        "task_id": audit.task_id,
        "annotation_path": audit.annotation_path,
        "image_id": audit.image_id,
        "image_json_path": audit.image_json_path,
        "has_image": bool(audit.image_json_path and audit.dataset_status != "skipped_missing_image"),
        "dataset_status": audit.dataset_status,
        "status": "not_run" if audit.dataset_status == "runnable" else "skipped",
        "skip_reason": audit.skip_reason,
        "gold_fields": audit.gold_fields.model_dump(mode="json") if audit.gold_fields else None,
        "gold_line_items": [item.model_dump(mode="json") for item in audit.gold_line_items],
        "predicted_fields": None,
        "predicted_line_items": [],
        "field_results": {
            name: result.model_dump(mode="json")
            for name, result in build_field_results(None, audit.gold_fields).items()
        },
        "overall_score": 0.0,
        "header_score": 0.0,
        "summary_score": 0.0,
        "line_items_score": 0.0,
        "line_item_gold_available": bool(audit.gold_line_items),
        "gold_line_item_count": len(audit.gold_line_items),
        "predicted_line_item_count": 0,
        "line_item_count_delta": None,
        "line_item_count_score": None,
        "reconciliation_score": 0.0,
        "reconciliation_delta": None,
        "reconciliation_status": None,
        "summary_reconciliation_delta": None,
        "summary_reconciliation_status": None,
        "line_item_reconciliation_delta": None,
        "line_item_reconciliation_status": "not_evaluated" if not audit.gold_line_items else None,
        "deterministic_success": False,
        "error": None,
        "judge": None,
        "created_at": None,
        "processed": False,
        "processable": audit.dataset_status == "runnable",
    }


def detail_record_payload(store: EvalArtifactStore, sample_id: str) -> dict[str, Any] | None:
    try:
        audit = get_audit_record(sample_id)
    except FileNotFoundError:
        audit = None
    record = store.get_record(sample_id)

    if audit is None and record is None:
        return None

    if audit is None and record is not None:
        payload = record.model_dump(mode="json")
        payload["processed"] = True
        payload["processable"] = record.dataset_status == "runnable"
        payload["has_image"] = bool(record.image_json_path and record.dataset_status != "skipped_missing_image")
        return enrich_detail_payload(payload)

    if record is not None:
        if record.dataset_status != audit.dataset_status:
            return enrich_detail_payload(audit_detail_payload(audit))
        payload = record.model_dump(mode="json")
        payload = merge_audit_metadata(payload, audit)
        payload["processed"] = True
        payload["processable"] = audit.dataset_status == "runnable"
        payload["has_image"] = bool(payload.get("image_json_path") and audit.dataset_status != "skipped_missing_image")
        return enrich_detail_payload(payload)

    return enrich_detail_payload(audit_detail_payload(audit))


def pagination_window(page: int, pages: int) -> list[int]:
    start = max(1, page - 2)
    end = min(pages, page + 2)
    return list(range(start, end + 1))


def optional_bool_query(value: str | None, parameter_name: str) -> bool | None:
    if value is None:
        return None
    normalized = value.strip().lower()
    if not normalized:
        return None
    if normalized in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise HTTPException(
        status_code=422,
        detail=[
            {
                "type": "bool_parsing",
                "loc": ["query", parameter_name],
                "msg": "Input should be a valid boolean, unable to interpret input",
                "input": value,
            }
        ],
    )


def optional_task_query(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = value.strip().lower()
    if not normalized:
        return None
    if normalized in TASK_IDS:
        return normalized
    raise HTTPException(
        status_code=422,
        detail=[
            {
                "type": "literal_error",
                "loc": ["query", "task_id"],
                "msg": "Input should be 'easy', 'medium', or 'hard'",
                "input": value,
            }
        ],
    )


def eval_href(path: str, **params: str | None) -> str:
    clean_params = {key: value for key, value in params.items() if value}
    query_string = urlencode(clean_params)
    return f"{path}?{query_string}" if query_string else path


def receipt_href(sample_id: str, task_id: str | None = None) -> str:
    return eval_href(f"/eval/receipts/{quote(sample_id, safe='')}", task_id=task_id)


def _audit_listing_payload(audit: Any) -> dict[str, Any]:
    return {
        "sample_id": audit.sample_id,
        "task_id": audit.task_id,
        "annotation_path": audit.annotation_path,
        "image_id": audit.image_id,
        "image_json_path": audit.image_json_path,
        "has_image": bool(audit.image_json_path and audit.dataset_status != "skipped_missing_image"),
        "dataset_status": audit.dataset_status,
        "status": "not_run",
        "skip_reason": audit.skip_reason,
        "overall_score": None,
        "header_score": None,
        "summary_score": None,
        "line_items_score": None,
        "line_item_gold_available": bool(audit.gold_line_items),
        "gold_line_item_count": len(audit.gold_line_items),
        "line_item_count_score": None,
        "reconciliation_score": None,
        "reconciliation_status": None,
        "deterministic_success": False,
        "error": None,
        "judge": None,
        "created_at": None,
        "processed": False,
        "processable": audit.dataset_status == "runnable",
    }


def _record_listing_payload(record: Any, audit: Any | None = None) -> dict[str, Any]:
    if audit is not None and record.dataset_status != audit.dataset_status:
        return _audit_listing_payload(audit)
    payload = record.model_dump(mode="json")
    if audit is not None:
        payload = merge_audit_metadata(payload, audit)
        if not payload.get("gold_line_item_count") and audit.gold_line_items:
            payload["gold_line_item_count"] = len(audit.gold_line_items)
            payload["line_item_gold_available"] = True
    payload["processed"] = True
    payload["processable"] = payload.get("dataset_status") == "runnable"
    payload["has_image"] = bool(payload.get("image_json_path") and payload.get("dataset_status") != "skipped_missing_image")
    return payload


def runnable_receipt_scope(store: EvalArtifactStore) -> list[dict[str, Any]]:
    records_by_sample_id = {record.sample_id: record for record in store.records()}
    items: list[dict[str, Any]] = []
    seen_sample_ids: set[str] = set()

    for audit in store.audit_records():
        if audit.dataset_status != "runnable":
            continue
        seen_sample_ids.add(audit.sample_id)
        record = records_by_sample_id.get(audit.sample_id)
        if record is None:
            items.append(_audit_listing_payload(audit))
        else:
            items.append(_record_listing_payload(record, audit))

    for record in records_by_sample_id.values():
        if record.sample_id in seen_sample_ids or record.dataset_status != "runnable":
            continue
        items.append(_record_listing_payload(record))

    return sorted(items, key=lambda item: item["sample_id"])


def filter_receipt_items(
    items: list[dict[str, Any]],
    task_id: str | None = None,
    status: str | None = None,
    sample_id: str | None = None,
    has_errors: bool | None = None,
) -> list[dict[str, Any]]:
    filtered = list(items)
    if task_id:
        filtered = [item for item in filtered if str(item.get("task_id") or "").lower() == task_id]
    if status:
        filtered = [item for item in filtered if item.get("status") == status]
    if sample_id:
        needle = sample_id.lower()
        filtered = [item for item in filtered if needle in item["sample_id"].lower()]
    if has_errors is True:
        filtered = [item for item in filtered if item.get("status") in {"partial", "failed"} or bool(item.get("error"))]
    elif has_errors is False:
        filtered = [item for item in filtered if item.get("status") == "worked" and not item.get("error")]
    return filtered


def paginate_receipt_items(items: list[dict[str, Any]], page: int, page_size: int) -> dict[str, Any]:
    page = max(page, 1)
    page_size = max(min(page_size, 100), 1)
    start = (page - 1) * page_size
    return {
        "items": items[start : start + page_size],
        "page": page,
        "page_size": page_size,
        "total": len(items),
        "pages": max((len(items) + page_size - 1) // page_size, 1),
    }


def _mean_score(items: list[dict[str, Any]], key: str) -> float:
    values = [_safe_float(item.get(key)) for item in items if item.get("processed") and item.get(key) is not None]
    return sum(values) / len(values) if values else 0.0


def runnable_scope_summary(items: list[dict[str, Any]]) -> dict[str, Any]:
    counts: dict[str, int] = {}
    for item in items:
        status = item.get("status") or "not_run"
        counts[status] = counts.get(status, 0) + 1

    with_gold_line_items = sum(1 for item in items if item.get("line_item_gold_available") or (item.get("gold_line_item_count") or 0) > 0)
    return {
        "total": len(items),
        "completed": sum(1 for item in items if item.get("processed")),
        "counts": counts,
        "mean_score": _mean_score(items, "overall_score"),
        "component_mean_scores": {
            "header_score": _mean_score(items, "header_score"),
            "summary_score": _mean_score(items, "summary_score"),
            "line_items_score": _mean_score(items, "line_items_score"),
            "reconciliation_score": _mean_score(items, "reconciliation_score"),
            "line_item_count_score": _mean_score(items, "line_item_count_score"),
        },
        "line_item_availability_counts": {
            "with_gold_line_items": with_gold_line_items,
            "without_gold_line_items": len(items) - with_gold_line_items,
        },
    }


def task_nav_payload(items: list[dict[str, Any]], active_task_id: str | None, link_to_receipts: bool = False) -> dict[str, Any]:
    nav_items: list[dict[str, Any]] = []
    for task_id in TASK_IDS:
        task_items = [item for item in items if str(item.get("task_id") or "").lower() == task_id]
        first_sample_id = task_items[0]["sample_id"] if task_items else None
        href = receipt_href(first_sample_id, task_id) if link_to_receipts and first_sample_id else eval_href("/eval", task_id=task_id)
        nav_items.append(
            {
                "task_id": task_id,
                "label": TASK_LABELS[task_id],
                "count": len(task_items),
                "href": href,
                "active": active_task_id == task_id,
                "disabled": first_sample_id is None,
            }
        )

    return {
        "items": nav_items,
        "total": len(items),
        "all_href": "/eval",
        "active_task_id": active_task_id,
    }


@api_router.get("/summary")
def eval_summary() -> dict[str, Any]:
    store = get_store()
    summary = store.summary()
    if summary is None:
        raise HTTPException(status_code=404, detail="Evaluation summary not found. Run scripts/evaluate_dataset_images.py first.")
    return summary.model_dump(mode="json")


@api_router.get("/receipts")
def eval_receipts(
    status: str | None = None,
    sample_id: str | None = None,
    has_errors: str | None = Query(default=None),
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=25, ge=1, le=100),
) -> dict[str, Any]:
    parsed_has_errors = optional_bool_query(has_errors, "has_errors")
    store = get_store()
    if not store.exists():
        raise HTTPException(status_code=404, detail="Evaluation results not found. Run scripts/evaluate_dataset_images.py first.")
    return store.list_records(status=status, sample_id=sample_id, has_errors=parsed_has_errors, page=page, page_size=page_size)


@api_router.get("/receipts/{sample_id}")
def eval_receipt_detail(sample_id: str) -> dict[str, Any]:
    store = get_store()
    record = detail_record_payload(store, sample_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"Evaluation record not found for sample_id={sample_id}")
    return record


@api_router.post("/receipts/{sample_id}/run")
def eval_receipt_run(sample_id: str) -> dict[str, Any]:
    try:
        record = evaluate_single_receipt(sample_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return record.model_dump(mode="json")


@api_router.get("/receipts/{sample_id}/image")
def eval_receipt_image(sample_id: str) -> Response:
    store = get_store()
    record = store.get_record(sample_id)
    image_json_path = record.image_json_path if record is not None else None
    if not image_json_path:
        audit = get_audit_record(sample_id)
        image_json_path = audit.image_json_path if audit is not None else None
    if not image_json_path:
        raise HTTPException(status_code=404, detail=f"Receipt image JSON not found for sample_id={sample_id}")

    try:
        image_bytes, media_type = decode_image_json_bytes(image_json_path)
    except ImageStoreError as exc:
        raise HTTPException(status_code=404, detail=f"Receipt image JSON unavailable for sample_id={sample_id}: {exc}") from exc
    return Response(
        content=image_bytes,
        media_type=media_type,
        headers={
            "Cache-Control": "no-store, max-age=0",
            "Pragma": "no-cache",
        },
    )


@api_router.get("/report", response_model=None)
def eval_report(format: str = Query(default="markdown", pattern="^(markdown|html)$")):
    store = get_store()
    report_markdown = store.report_markdown()
    if not report_markdown:
        raise HTTPException(status_code=404, detail="Evaluation report not found. Run scripts/evaluate_dataset_images.py first.")

    if format == "html":
        return HTMLResponse(f"<html><body><pre>{report_markdown}</pre></body></html>")
    return PlainTextResponse(report_markdown, media_type="text/markdown")


@ui_router.get("/eval", response_class=HTMLResponse)
def eval_dashboard(
    request: Request,
    task_id: str | None = None,
    status: str | None = None,
    sample_id: str | None = None,
    has_errors: str | None = Query(default=None),
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=25, ge=1, le=100),
) -> HTMLResponse:
    parsed_has_errors = optional_bool_query(has_errors, "has_errors")
    active_task_id = optional_task_query(task_id)
    store = get_store()
    summary = store.summary()
    runnable_items = runnable_receipt_scope(store)
    if summary is None or not store.exists():
        return TEMPLATES.TemplateResponse(
            request=request,
            name="eval_empty.html",
            context={
                "title": "Receipt Eval Dashboard",
                "output_dir": str(store.output_dir),
                "task_nav": task_nav_payload(runnable_items, active_task_id),
                "active_task_id": active_task_id,
                "eval_models": eval_model_config(),
            },
        )

    task_scope_items = filter_receipt_items(runnable_items, task_id=active_task_id)
    listing_items = filter_receipt_items(task_scope_items, status=status, sample_id=sample_id, has_errors=parsed_has_errors)
    listing = paginate_receipt_items(listing_items, page=page, page_size=page_size)
    return TEMPLATES.TemplateResponse(
        request=request,
        name="eval_index.html",
        context={
            "title": "Receipt Eval Dashboard",
            "summary": summary.model_dump(mode="json"),
            "scope_summary": runnable_scope_summary(task_scope_items),
            "listing": listing,
            "task_nav": task_nav_payload(runnable_items, active_task_id),
            "eval_models": eval_model_config(),
            "active_task_id": active_task_id,
            "status_filter": status or "",
            "sample_filter": sample_id or "",
            "has_errors_filter": parsed_has_errors,
            "page_numbers": pagination_window(listing["page"], listing["pages"]),
        },
    )


@ui_router.get("/eval/receipts/{sample_id}", response_class=HTMLResponse)
def eval_detail(request: Request, sample_id: str, task_id: str | None = None) -> HTMLResponse:
    active_task_id = optional_task_query(task_id)
    store = get_store()
    record = detail_record_payload(store, sample_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"Evaluation record not found for sample_id={sample_id}")
    if not record.get("processable"):
        raise HTTPException(status_code=404, detail=f"Receipt {sample_id} is not runnable and is outside the eval UI scope.")

    record_task_id = str(record.get("task_id") or "").lower()
    if active_task_id and record_task_id != active_task_id:
        active_task_id = None
    runnable_items = runnable_receipt_scope(store)
    scope_items = filter_receipt_items(runnable_items, task_id=active_task_id)
    sample_ids = [item["sample_id"] for item in scope_items]
    index = sample_ids.index(sample_id) if sample_id in sample_ids else -1
    previous_id = sample_ids[index - 1] if index > 0 else None
    next_id = sample_ids[index + 1] if index >= 0 and index + 1 < len(sample_ids) else None

    return TEMPLATES.TemplateResponse(
        request=request,
        name="eval_detail.html",
        context={
            "title": f"Receipt Eval: {sample_id}",
            "record": record,
            "task_nav": task_nav_payload(runnable_items, active_task_id, link_to_receipts=True),
            "eval_models": eval_model_config(),
            "active_task_id": active_task_id,
            "previous_id": previous_id,
            "next_id": next_id,
        },
    )


@ui_router.post("/eval/receipts/{sample_id}/run")
def eval_detail_run(sample_id: str, task_id: str | None = None) -> RedirectResponse:
    active_task_id = optional_task_query(task_id)
    evaluate_single_receipt(sample_id)
    return RedirectResponse(url=receipt_href(sample_id, active_task_id), status_code=303)
