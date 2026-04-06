from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

FieldName = Literal["company", "date", "address", "subtotal", "tax", "total"]
Difficulty = Literal["easy", "medium", "hard"]
ReconciliationStatus = Literal["pass", "partial", "fail", "not_evaluated"]
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
    "query_line_item_candidates",
    "add_line_item_from_candidate",
    "remove_line_item",
    "check_receipt_consistency",
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


class ReceiptLineItem(BaseModel):
    item_id: str | None = None
    description: str | None = None
    line_total: str | None = None
    quantity: str | None = None
    raw_text: str | None = None
    evidence_ids: list[str] = Field(default_factory=list)


class ReceiptLineItemCandidate(BaseModel):
    candidate_id: str
    description: str | None = None
    line_total: str | None = None
    raw_text: str | None = None
    evidence_ids: list[str] = Field(default_factory=list)
    heuristic_score: float


class ReceiptDraft(BaseModel):
    company: str | None = None
    date: str | None = None
    address: str | None = None
    subtotal: str | None = None
    tax: str | None = None
    total: str | None = None
    line_items: list[ReceiptLineItem] = Field(default_factory=list)


class ReceiptAction(BaseModel):
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
    evidence_ids: list[str] | None = None


class ReceiptObservation(BaseModel):
    task_id: str
    difficulty: Difficulty
    instruction: str
    image_ref: str | None
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


class ReceiptState(BaseModel):
    sample_id: str
    difficulty: str
    task_id: str = "easy"
    current_draft: ReceiptDraft
    revealed_region_ids: list[str] = Field(default_factory=list)
    history: list[dict] = Field(default_factory=list)
    step_index: int
    remaining_budget: int
    cumulative_reward: float
    done: bool
    current_reconciliation_delta: float | None = None
    current_reconciliation_status: ReconciliationStatus | None = None
    reconciliation_feedback: list[str] = Field(default_factory=list)
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
    header_score: float = 0.0
    summary_score: float = 0.0
    line_items_score: float = 0.0
    reconciliation_score: float = 0.0
    reconciliation_delta: float | None = None
    reconciliation_status: ReconciliationStatus | None = None
    summary_reconciliation_delta: float | None = None
    summary_reconciliation_status: ReconciliationStatus | None = None
    line_item_reconciliation_delta: float | None = None
    line_item_reconciliation_status: ReconciliationStatus | None = None
    line_item_gold_available: bool = False
    gold_line_item_count: int = 0
    predicted_line_item_count: int = 0
    line_item_count_delta: int | None = None
    line_item_count_score: float | None = None


class TaskConfig(BaseModel):
    task_id: str
    difficulty: Difficulty
    max_steps: int
    visible_windows: list[str]
    corruption_level: float = 0.0
    ranking_noise: float = 0.0
    instruction: str
    target_fields: list[FieldName] = Field(default_factory=list)
    requires_line_items: bool = False


class ReceiptSample(BaseModel):
    sample_id: str
    image_ref: str | None = None
    regions: list[OCRRegion]
    gold_fields: ReceiptDraft
    gold_line_items: list[ReceiptLineItem] = Field(default_factory=list)
