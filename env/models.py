from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

FieldName = Literal["company", "date", "address", "total"]
Difficulty = Literal["easy", "medium", "hard"]
ActionType = Literal[
    "view_receipt",
    "list_text_regions",
    "inspect_bbox",
    "inspect_neighbors",
    "query_candidates",
    "set_field_from_candidate",
    "set_field_manual",
    "merge_spans",
    "normalize_field",
    "check_total_consistency",
    "check_date_format",
    "clear_field",
    "submit",
]


class OCRRegion(BaseModel):
    region_id: str
    text: str
    bbox: tuple[int, int, int, int]
    confidence: float | None = None
    revealed: bool = True


class FieldCandidate(BaseModel):
    candidate_id: str
    field: FieldName
    value: str
    evidence_ids: list[str] = Field(default_factory=list)
    heuristic_score: float


class ReceiptDraft(BaseModel):
    company: str | None = None
    date: str | None = None
    address: str | None = None
    total: str | None = None


class ReceiptAction(BaseModel):
    action_type: ActionType
    field: str | None = None
    window: str | None = None
    bbox_id: str | None = None
    radius_bucket: int | None = None
    candidate_id: str | None = None
    span_ids: list[str] | None = None
    join_mode: str | None = None
    mode: str | None = None
    value: str | None = None
    evidence_ids: list[str] | None = None


class ReceiptObservation(BaseModel):
    task_id: str
    difficulty: Difficulty
    instruction: str
    image_ref: str | None
    visible_regions: list[OCRRegion] = Field(default_factory=list)
    candidate_lists: dict[str, list[FieldCandidate]] = Field(default_factory=dict)
    current_draft: ReceiptDraft = Field(default_factory=ReceiptDraft)
    validation_feedback: list[str] = Field(default_factory=list)
    last_action_result: str = ""
    remaining_budget: int
    step_index: int
    terminal_allowed: bool


class ReceiptState(BaseModel):
    sample_id: str
    difficulty: str
    current_draft: ReceiptDraft
    revealed_region_ids: list[str] = Field(default_factory=list)
    history: list[dict] = Field(default_factory=list)
    step_index: int
    remaining_budget: int
    cumulative_reward: float
    done: bool
    last_error: str | None = None


class StepResult(BaseModel):
    observation: ReceiptObservation
    reward: float
    done: bool
    info: dict = Field(default_factory=dict)


class GradeResult(BaseModel):
    score: float
    success: bool
    field_scores: dict[str, float]


class TaskConfig(BaseModel):
    task_id: str
    difficulty: Difficulty
    max_steps: int
    visible_windows: list[str]
    corruption_level: float = 0.0
    ranking_noise: float = 0.0


class ReceiptSample(BaseModel):
    sample_id: str
    image_ref: str | None = None
    regions: list[OCRRegion]
    gold_fields: ReceiptDraft
