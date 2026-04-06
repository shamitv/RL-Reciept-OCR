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


@app.post("/reset", response_model=StepResult)
def reset(task_name: str | None = None, seed: int | None = None) -> StepResult:
    return ENV.reset(task_name=task_name, seed=seed)


@app.post("/step", response_model=StepResult)
def step(action: ReceiptAction) -> StepResult:
    return ENV.step(action)


@app.get("/state")
def state() -> dict:
    return ENV.state().model_dump()
