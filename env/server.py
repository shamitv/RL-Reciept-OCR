from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from env.config import load_environment
from env.environment import ReceiptExtractionEnv
from env.eval_api import api_router as eval_api_router
from env.eval_api import ui_router as eval_ui_router
from env.models import ReceiptAction, StepResult

load_environment()

app = FastAPI(title="RL Receipt OCR OpenEnv")
ENV = ReceiptExtractionEnv()
STATIC_DIR = Path(__file__).resolve().parents[1] / "server" / "static"

app.mount("/static", StaticFiles(directory=str(STATIC_DIR), check_dir=False), name="static")
app.include_router(eval_api_router)
app.include_router(eval_ui_router)


@app.get("/")
def root() -> dict[str, object]:
    return {
        "name": "rl-receipt-ocr",
        "status": "ok",
        "openenv": {
            "reset": "/reset",
            "step": "/step",
            "state": "/state",
        },
        "health": {
            "live": "/healthz",
            "ready": "/readyz",
        },
        "ui": "/eval",
    }


@app.get("/healthz")
def healthz() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/readyz")
def readyz() -> dict[str, object]:
    task_counts = ENV.dataset.eligible_task_counts()
    task_ids = sorted(task_name for task_name, count in task_counts.items() if count > 0)
    return {"status": "ready", "tasks": task_ids, "counts": task_counts}


@app.post("/reset", response_model=StepResult)
def reset(task_name: str | None = None, seed: int | None = None) -> StepResult:
    return ENV.reset(task_name=task_name, seed=seed)


@app.post("/step", response_model=StepResult)
def step(action: ReceiptAction) -> StepResult:
    return ENV.step(action)


@app.get("/state")
def state() -> dict:
    return ENV.state().model_dump()
