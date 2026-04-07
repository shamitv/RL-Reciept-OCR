from __future__ import annotations

import os
from pathlib import Path
from typing import Any

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
        payload = record.model_dump(mode="json")
        payload = merge_audit_metadata(payload, audit)
        payload["processed"] = True
        payload["processable"] = audit.dataset_status == "runnable"
        payload["has_image"] = bool(payload.get("image_json_path") and audit.dataset_status != "skipped_missing_image")
        return enrich_detail_payload(payload)

    payload = {
        "sample_id": audit.sample_id,
        "task_id": audit.task_id,
        "annotation_path": audit.annotation_path,
        "image_id": audit.image_id,
        "image_json_path": audit.image_json_path,
        "has_image": bool(audit.image_json_path and audit.dataset_status != "skipped_missing_image"),
        "dataset_status": audit.dataset_status,
        "status": "not_run",
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
    return enrich_detail_payload(payload)


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
    status: str | None = None,
    sample_id: str | None = None,
    has_errors: str | None = Query(default=None),
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=25, ge=1, le=100),
) -> HTMLResponse:
    parsed_has_errors = optional_bool_query(has_errors, "has_errors")
    store = get_store()
    summary = store.summary()
    if summary is None or not store.exists():
        return TEMPLATES.TemplateResponse(
            request=request,
            name="eval_empty.html",
            context={
                "title": "Receipt Eval Dashboard",
                "output_dir": str(store.output_dir),
                "receipt_menu": store.receipt_menu(),
                "eval_models": eval_model_config(),
            },
        )

    listing = store.list_records(status=status, sample_id=sample_id, has_errors=parsed_has_errors, page=page, page_size=page_size)
    return TEMPLATES.TemplateResponse(
        request=request,
        name="eval_index.html",
        context={
            "title": "Receipt Eval Dashboard",
            "summary": summary.model_dump(mode="json"),
            "listing": listing,
            "receipt_menu": store.receipt_menu(),
            "eval_models": eval_model_config(),
            "status_filter": status or "",
            "sample_filter": sample_id or "",
            "has_errors_filter": parsed_has_errors,
            "page_numbers": pagination_window(listing["page"], listing["pages"]),
        },
    )


@ui_router.get("/eval/receipts/{sample_id}", response_class=HTMLResponse)
def eval_detail(request: Request, sample_id: str) -> HTMLResponse:
    store = get_store()
    record = detail_record_payload(store, sample_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"Evaluation record not found for sample_id={sample_id}")

    menu_items = store.receipt_menu()
    sample_ids = [item["sample_id"] for item in menu_items]
    index = sample_ids.index(sample_id)
    previous_id = sample_ids[index - 1] if index > 0 else None
    next_id = sample_ids[index + 1] if index + 1 < len(sample_ids) else None

    return TEMPLATES.TemplateResponse(
        request=request,
        name="eval_detail.html",
        context={
            "title": f"Receipt Eval: {sample_id}",
            "record": record,
            "receipt_menu": menu_items,
            "eval_models": eval_model_config(),
            "previous_id": previous_id,
            "next_id": next_id,
        },
    )


@ui_router.post("/eval/receipts/{sample_id}/run")
def eval_detail_run(sample_id: str) -> RedirectResponse:
    evaluate_single_receipt(sample_id)
    return RedirectResponse(url=f"/eval/receipts/{sample_id}", status_code=303)
