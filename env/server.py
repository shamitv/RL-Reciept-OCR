from __future__ import annotations

from fastapi import FastAPI

from env.environment import ReceiptExtractionEnv
from env.models import ReceiptAction, StepResult

app = FastAPI(title="RL Receipt OCR OpenEnv")
ENV = ReceiptExtractionEnv()


@app.post("/reset", response_model=StepResult)
def reset(task_name: str | None = None, seed: int | None = None) -> StepResult:
    return ENV.reset(task_name=task_name, seed=seed)


@app.post("/step", response_model=StepResult)
def step(action: ReceiptAction) -> StepResult:
    return ENV.step(action)


@app.get("/state")
def state() -> dict:
    return ENV.state().model_dump()
