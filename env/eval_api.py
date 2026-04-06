from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import FileResponse, HTMLResponse, PlainTextResponse
from fastapi.templating import Jinja2Templates

from env.evaluation import EvalArtifactStore

BASE_DIR = Path(__file__).resolve().parents[1]
TEMPLATES = Jinja2Templates(directory=str(BASE_DIR / "server" / "templates"))

api_router = APIRouter(prefix="/api/eval", tags=["eval"])
ui_router = APIRouter(tags=["eval-ui"])


def get_store() -> EvalArtifactStore:
    return EvalArtifactStore()


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
    record = store.get_record(sample_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"Evaluation record not found for sample_id={sample_id}")
    return record.model_dump(mode="json")


@api_router.get("/receipts/{sample_id}/image")
def eval_receipt_image(sample_id: str) -> FileResponse:
    store = get_store()
    record = store.get_record(sample_id)
    if record is None or not record.image_path:
        raise HTTPException(status_code=404, detail=f"Receipt image not found for sample_id={sample_id}")

    image_path = Path(record.image_path)
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
            "status_filter": status or "",
            "sample_filter": sample_id or "",
            "has_errors_filter": has_errors,
            "page_numbers": pagination_window(listing["page"], listing["pages"]),
        },
    )


@ui_router.get("/eval/receipts/{sample_id}", response_class=HTMLResponse)
def eval_detail(request: Request, sample_id: str) -> HTMLResponse:
    store = get_store()
    record = store.get_record(sample_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"Evaluation record not found for sample_id={sample_id}")

    records = store.records()
    sample_ids = [item.sample_id for item in records]
    index = sample_ids.index(sample_id)
    previous_id = sample_ids[index - 1] if index > 0 else None
    next_id = sample_ids[index + 1] if index + 1 < len(sample_ids) else None

    return TEMPLATES.TemplateResponse(
        request=request,
        name="eval_detail.html",
        context={
            "title": f"Receipt Eval: {sample_id}",
            "record": record.model_dump(mode="json"),
            "previous_id": previous_id,
            "next_id": next_id,
        },
    )
