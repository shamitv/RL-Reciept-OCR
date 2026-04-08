from __future__ import annotations

from pathlib import Path
from typing import Any
from uuid import uuid4

import gradio as gr
from fastapi import Body, FastAPI, HTTPException, WebSocket, WebSocketDisconnect, status
from fastapi.responses import RedirectResponse
from openenv.core.env_server.gradio_theme import OPENENV_GRADIO_CSS, OPENENV_GRADIO_THEME
from openenv.core.env_server.gradio_ui import build_gradio_app
from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import Action, EnvironmentMetadata, Observation, State
from openenv.core.env_server.web_interface import (
    WebInterfaceManager,
    _extract_action_fields,
    get_quick_start_markdown,
)
from pydantic import Field

from env.environment import ReceiptExtractionEnv
from env.models import (
    ActionType,
    Difficulty,
    FieldCandidate,
    OCRRegion,
    ReceiptAction,
    ReceiptDraft,
    ReceiptLineItem,
    ReceiptLineItemCandidate,
    ReconciliationStatus,
)


def _read_repo_readme() -> str | None:
    readme_path = Path(__file__).resolve().parents[1] / "README.md"
    if not readme_path.exists():
        return None
    return readme_path.read_text(encoding="utf-8")


class OpenEnvReceiptAction(Action):
    action_type: ActionType
    field: str | None = None
    window: str | None = None
    bbox_id: str | None = None
    radius_bucket: int | None = None
    candidate_id: str | None = None
    line_item_index: int | None = None
    span_ids: list[str] | None = None
    join_mode: str | None = None
    mode: str | None = None
    value: str | None = None
    line_total: str | None = None
    quantity: str | None = None
    evidence_ids: list[str] | None = None


class OpenEnvReceiptObservation(Observation):
    task_id: str
    difficulty: Difficulty
    instruction: str
    image_ref: str | None
    image_id: str | None = None
    image_json_path: str | None = None
    visible_regions: list[OCRRegion] = Field(default_factory=list)
    candidate_lists: dict[str, list[FieldCandidate]] = Field(default_factory=dict)
    line_item_candidates: list[ReceiptLineItemCandidate] = Field(default_factory=list)
    current_draft: ReceiptDraft = Field(default_factory=ReceiptDraft)
    validation_feedback: list[str] = Field(default_factory=list)
    reconciliation_feedback: list[str] = Field(default_factory=list)
    current_reconciliation_delta: float | None = None
    current_reconciliation_status: ReconciliationStatus | None = None
    last_action_result: str = ""
    remaining_budget: int
    step_index: int
    terminal_allowed: bool


class OpenEnvReceiptState(State):
    sample_id: str = ""
    difficulty: str = "easy"
    task_id: str = "easy"
    current_draft: ReceiptDraft = Field(default_factory=ReceiptDraft)
    revealed_region_ids: list[str] = Field(default_factory=list)
    history: list[dict[str, Any]] = Field(default_factory=list)
    step_index: int = 0
    remaining_budget: int = 0
    cumulative_reward: float = 0.0
    done: bool = False
    current_reconciliation_delta: float | None = None
    current_reconciliation_status: ReconciliationStatus | None = None
    reconciliation_feedback: list[str] = Field(default_factory=list)
    last_error: str | None = None


class OpenEnvReceiptWebEnvironment(Environment[OpenEnvReceiptAction, OpenEnvReceiptObservation, OpenEnvReceiptState]):
    def __init__(self) -> None:
        super().__init__()
        self._env = ReceiptExtractionEnv()
        self._episode_id: str | None = None

    def get_metadata(self) -> EnvironmentMetadata:
        return EnvironmentMetadata(
            name="rl-receipt-ocr",
            description=(
                "Sequential receipt extraction environment for receipt headers, "
                "monetary summaries, reconciliation, and line-item extraction."
            ),
            version="0.1.0",
            readme_content=_read_repo_readme(),
        )

    def reset(
        self,
        seed: int | None = None,
        episode_id: str | None = None,
        task_name: str | None = None,
        **kwargs: Any,
    ) -> OpenEnvReceiptObservation:
        self._episode_id = episode_id or str(uuid4())
        result = self._env.reset(task_name=task_name, seed=seed)
        return self._serialize_observation(result.observation, reward=result.reward, done=result.done)

    def step(
        self,
        action: OpenEnvReceiptAction,
        timeout_s: float | None = None,
        **kwargs: Any,
    ) -> OpenEnvReceiptObservation:
        receipt_action = ReceiptAction.model_validate(action.model_dump(exclude={"metadata"}, exclude_none=True))
        result = self._env.step(receipt_action)
        return self._serialize_observation(result.observation, reward=result.reward, done=result.done)

    @property
    def state(self) -> OpenEnvReceiptState:
        state = self._env.state().model_dump(mode="json")
        state["episode_id"] = self._episode_id
        state["step_count"] = int(state.get("step_index", 0))
        return OpenEnvReceiptState.model_validate(state)

    def close(self) -> None:
        return None

    @staticmethod
    def _serialize_observation(
        observation: Any,
        *,
        reward: float | None,
        done: bool,
    ) -> OpenEnvReceiptObservation:
        payload = observation.model_dump(mode="json")
        payload["reward"] = reward
        payload["done"] = done
        return OpenEnvReceiptObservation.model_validate(payload)


def mount_openenv_web_interface(app: FastAPI) -> FastAPI:
    if getattr(app.state, "openenv_web_interface_mounted", False):
        return app

    web_env = OpenEnvReceiptWebEnvironment()
    metadata = web_env.get_metadata()
    web_manager = WebInterfaceManager(
        web_env,
        OpenEnvReceiptAction,
        OpenEnvReceiptObservation,
        metadata,
    )
    action_fields = _extract_action_fields(OpenEnvReceiptAction)
    quick_start_md = get_quick_start_markdown(
        metadata,
        OpenEnvReceiptAction,
        OpenEnvReceiptObservation,
    )
    gradio_blocks = build_gradio_app(
        web_manager,
        action_fields,
        metadata,
        is_chat_env=False,
        title=metadata.name,
        quick_start_md=quick_start_md,
    )

    @app.get("/web", include_in_schema=False)
    async def openenv_web_root() -> RedirectResponse:
        return RedirectResponse(url="/web/")

    @app.get("/web/metadata")
    async def openenv_web_metadata() -> dict[str, Any]:
        return web_manager.metadata.model_dump()

    @app.websocket("/ws/ui")
    async def openenv_websocket_ui_endpoint(websocket: WebSocket) -> None:
        await web_manager.connect_websocket(websocket)
        try:
            while True:
                await websocket.receive_text()
        except WebSocketDisconnect:
            await web_manager.disconnect_websocket(websocket)

    @app.post("/web/reset")
    async def openenv_web_reset(request: dict[str, Any] | None = Body(default=None)) -> dict[str, Any]:
        return await web_manager.reset_environment(request)

    @app.post("/web/step")
    async def openenv_web_step(request: dict[str, Any]) -> dict[str, Any]:
        if "message" in request:
            return await web_manager.step_environment({"message": request["message"]})
        if "action" not in request:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="Request body must include an action object.",
            )
        return await web_manager.step_environment(request.get("action") or {})

    @app.get("/web/state")
    async def openenv_web_state() -> dict[str, Any]:
        try:
            return web_manager.get_state()
        except RuntimeError as exc:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=str(exc),
            ) from exc

    mounted_app = gr.mount_gradio_app(
        app,
        gradio_blocks,
        path="/web",
        theme=OPENENV_GRADIO_THEME,
        css=OPENENV_GRADIO_CSS,
    )
    mounted_app.state.openenv_web_interface_mounted = True
    return mounted_app
