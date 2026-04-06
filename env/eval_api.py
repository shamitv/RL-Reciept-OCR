from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import FileResponse, HTMLResponse, PlainTextResponse, RedirectResponse
from fastapi.templating import Jinja2Templates

from env.evaluation import EvalArtifactStore, build_field_results, evaluate_single_receipt, get_audit_record

BASE_DIR = Path(__file__).resolve().parents[1]
TEMPLATES = Jinja2Templates(directory=str(BASE_DIR / "server" / "templates"))

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


def detail_record_payload(store: EvalArtifactStore, sample_id: str) -> dict[str, Any] | None:
    audit = get_audit_record(sample_id)
    record = store.get_record(sample_id)

    if audit is None and record is None:
        return None

    if audit is None and record is not None:
        payload = record.model_dump(mode="json")
        payload["processed"] = True
        payload["processable"] = record.dataset_status == "runnable"
        payload["line_item_rows"] = line_item_rows(payload)
        return payload

    if record is not None:
        payload = record.model_dump(mode="json")
        payload["processed"] = True
        payload["processable"] = audit.dataset_status == "runnable"
        payload["line_item_rows"] = line_item_rows(payload)
        return payload

    payload = {
        "sample_id": audit.sample_id,
        "task_id": audit.task_id,
        "annotation_path": audit.annotation_path,
        "image_path": audit.image_path,
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
        "reconciliation_score": 0.0,
        "reconciliation_delta": None,
        "reconciliation_status": None,
        "deterministic_success": False,
        "error": None,
        "judge": None,
        "created_at": None,
        "processed": False,
        "processable": audit.dataset_status == "runnable",
    }
    payload["line_item_rows"] = line_item_rows(payload)
    return payload


def pagination_window(page: int, pages: int) -> list[int]:
    start = max(1, page - 2)
    end = min(pages, page + 2)
    return list(range(start, end + 1))


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
    has_errors: bool | None = None,
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=25, ge=1, le=100),
) -> dict[str, Any]:
    store = get_store()
    if not store.exists():
        raise HTTPException(status_code=404, detail="Evaluation results not found. Run scripts/evaluate_dataset_images.py first.")
    return store.list_records(status=status, sample_id=sample_id, has_errors=has_errors, page=page, page_size=page_size)


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
def eval_receipt_image(sample_id: str) -> FileResponse:
    store = get_store()
    record = store.get_record(sample_id)
    image_path_str = record.image_path if record is not None else None
    if not image_path_str:
        audit = get_audit_record(sample_id)
        image_path_str = audit.image_path if audit is not None else None
    if not image_path_str:
        raise HTTPException(status_code=404, detail=f"Receipt image not found for sample_id={sample_id}")

    image_path = Path(image_path_str)
    if not image_path.exists():
        raise HTTPException(status_code=404, detail=f"Receipt image file missing for sample_id={sample_id}")
    return FileResponse(image_path)


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
    has_errors: bool | None = None,
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=25, ge=1, le=100),
) -> HTMLResponse:
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

    listing = store.list_records(status=status, sample_id=sample_id, has_errors=has_errors, page=page, page_size=page_size)
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
            "has_errors_filter": has_errors,
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
