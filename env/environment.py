from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field

from env.candidate_retrieval import query_candidates, query_line_item_candidates
from env.dataset import ReceiptDataset
from env.graders import grade_receipt
from env.models import (
    OCRRegion,
    ReceiptAction,
    ReceiptDraft,
    ReceiptLineItem,
    ReceiptLineItemCandidate,
    ReceiptObservation,
    ReceiptSample,
    ReceiptState,
    StepResult,
    TaskConfig,
)
from env.normalizers import normalize_address, normalize_amount, normalize_date, normalize_text
from env.rewards import RewardTracker
from env.tasks import get_task
from env.utils import make_rng


@dataclass
class HiddenState:
    sample_id: str = ""
    difficulty: str = "easy"
    task_id: str = "easy"
    image_ref: str | None = None
    image_id: str | None = None
    image_json_path: str | None = None
    gold_fields: ReceiptDraft = field(default_factory=ReceiptDraft)
    gold_line_items: list[ReceiptLineItem] = field(default_factory=list)
    all_regions: list[OCRRegion] = field(default_factory=list)
    revealed_region_ids: list[str] = field(default_factory=list)
    candidate_lists: dict[str, list] = field(default_factory=dict)
    line_item_candidates: list[ReceiptLineItemCandidate] = field(default_factory=list)
    current_draft: ReceiptDraft = field(default_factory=ReceiptDraft)
    validation_feedback: list[str] = field(default_factory=list)
    reconciliation_feedback: list[str] = field(default_factory=list)
    current_reconciliation_delta: float | None = None
    current_reconciliation_status: str | None = None
    history: list[dict] = field(default_factory=list)
    step_index: int = 0
    remaining_budget: int = 0
    done: bool = False
    last_error: str | None = None
    last_action_result: str = "episode reset"
    cumulative_reward: float = 0.0


class ReceiptExtractionEnv:
    def __init__(self) -> None:
        self.dataset = ReceiptDataset()
        self.rng = make_rng(0)
        self.hidden_state = HiddenState()
        self.task: TaskConfig = get_task("easy")
        self.last_observation = ReceiptObservation(
            task_id=self.task.task_id,
            difficulty=self.task.difficulty,
            instruction=self.task.instruction,
            image_ref=None,
            remaining_budget=0,
            step_index=0,
            terminal_allowed=False,
        )
        self.reward_tracker = RewardTracker()

    def reset(self, task_name: str | None = None, seed: int | None = None) -> StepResult:
        self.rng = make_rng(seed)
        self.task = get_task(task_name)
        sample = self.dataset.sample(self.task.task_id, self.rng)
        return self._reset_from_sample(sample)

    def reset_with_sample(self, sample: ReceiptSample, task_name: str | None = None, seed: int | None = None) -> StepResult:
        self.rng = make_rng(seed)
        self.task = get_task(task_name)
        return self._reset_from_sample(sample)

    def _reset_from_sample(self, sample: ReceiptSample) -> StepResult:
        self.reward_tracker = RewardTracker()
        self.hidden_state = HiddenState(
            sample_id=sample.sample_id,
            difficulty=self.task.difficulty,
            task_id=self.task.task_id,
            image_ref=sample.image_ref,
            image_id=sample.image_id,
            image_json_path=sample.image_json_path,
            gold_fields=sample.gold_fields,
            gold_line_items=sample.gold_line_items,
            all_regions=sample.regions,
            current_draft=ReceiptDraft(),
            step_index=0,
            remaining_budget=self.task.max_steps,
        )
        self.last_observation = self._build_observation("episode reset")
        return StepResult(observation=self.last_observation, reward=0.0, done=False, info={})

    def state(self) -> ReceiptState:
        return ReceiptState(
            sample_id=self.hidden_state.sample_id,
            difficulty=self.hidden_state.difficulty,
            task_id=self.hidden_state.task_id,
            current_draft=self.hidden_state.current_draft.model_copy(deep=True),
            revealed_region_ids=list(self.hidden_state.revealed_region_ids),
            history=list(self.hidden_state.history),
            step_index=self.hidden_state.step_index,
            remaining_budget=self.hidden_state.remaining_budget,
            cumulative_reward=self.hidden_state.cumulative_reward,
            done=self.hidden_state.done,
            current_reconciliation_delta=self.hidden_state.current_reconciliation_delta,
            current_reconciliation_status=self.hidden_state.current_reconciliation_status,
            reconciliation_feedback=list(self.hidden_state.reconciliation_feedback),
            last_error=self.hidden_state.last_error,
        )

    def close(self) -> None:
        return None

    def step(self, action: ReceiptAction) -> StepResult:
        if self.hidden_state.done:
            return StepResult(observation=self.last_observation, reward=0.0, done=True, info={"error": "episode already done"})

        prev_draft = deepcopy(self.hidden_state.current_draft)
        self.hidden_state.last_error = None
        message = self._execute_action(action)
        self.hidden_state.step_index += 1
        self.hidden_state.remaining_budget -= 1

        info: dict = {}
        if action.action_type == "submit":
            final = grade_receipt(
                self.hidden_state.current_draft,
                self.hidden_state.gold_fields,
                task_id=self.task.task_id,
                gold_line_items=self.hidden_state.gold_line_items,
            )
            blank_fields = self._blank_field_count()
            reward = self.reward_tracker.compute_terminal_reward(final, blank_fields)
            self.hidden_state.done = True
            info = {
                "success": final.success,
                "final_score": final.score,
                "field_scores": final.field_scores,
                "header_score": final.header_score,
                "summary_score": final.summary_score,
                "line_items_score": final.line_items_score,
                "reconciliation_score": final.reconciliation_score,
                "reconciliation_delta": final.reconciliation_delta,
                "reconciliation_status": final.reconciliation_status,
            }
        else:
            reward = self.reward_tracker.compute_step_reward(
                prev_draft=prev_draft,
                current_state=self.state(),
                action=action,
                gold=self.hidden_state.gold_fields,
                action_result=message,
                gold_line_items=self.hidden_state.gold_line_items,
                task=self.task,
            )

        if self.hidden_state.remaining_budget <= 0 and not self.hidden_state.done:
            final = grade_receipt(
                self.hidden_state.current_draft,
                self.hidden_state.gold_fields,
                task_id=self.task.task_id,
                gold_line_items=self.hidden_state.gold_line_items,
            )
            reward += self.reward_tracker.compute_terminal_reward(final, self._blank_field_count())
            self.hidden_state.done = True
            info.update(
                {
                    "success": final.success,
                    "final_score": final.score,
                    "field_scores": final.field_scores,
                    "header_score": final.header_score,
                    "summary_score": final.summary_score,
                    "line_items_score": final.line_items_score,
                    "reconciliation_score": final.reconciliation_score,
                    "reconciliation_delta": final.reconciliation_delta,
                    "reconciliation_status": final.reconciliation_status,
                    "budget_exhausted": True,
                }
            )

        self.hidden_state.cumulative_reward += reward
        self.hidden_state.history.append({"action": action.model_dump(exclude_none=True), "message": message, "reward": reward})
        self.hidden_state.last_action_result = message
        self.last_observation = self._build_observation(message)
        return StepResult(observation=self.last_observation, reward=reward, done=self.hidden_state.done, info=info)

    def _blank_field_count(self) -> int:
        blank_fields = sum(1 for field in self.task.target_fields if getattr(self.hidden_state.current_draft, field) is None)
        if self.task.requires_line_items and not self.hidden_state.current_draft.line_items:
            blank_fields += 1
        return blank_fields

    def _visible_regions(self) -> list[OCRRegion]:
        ids = set(self.hidden_state.revealed_region_ids)
        return [region for region in self.hidden_state.all_regions if region.region_id in ids]

    def _reveal_window(self, window: str) -> int:
        if window not in self.task.visible_windows:
            self.hidden_state.last_error = f"window {window} unavailable for task"
            return 0
        regions = self.hidden_state.all_regions
        if window == "top":
            selected = [region for region in regions if region.bbox[1] <= 100]
        elif window == "middle":
            selected = [region for region in regions if 50 <= region.bbox[1] <= 180]
        elif window == "bottom":
            selected = [region for region in regions if region.bbox[1] >= 150]
        else:
            selected = list(regions)
        before = len(self.hidden_state.revealed_region_ids)
        for region in selected:
            if region.region_id not in self.hidden_state.revealed_region_ids:
                self.hidden_state.revealed_region_ids.append(region.region_id)
        return len(self.hidden_state.revealed_region_ids) - before

    def _default_reveal_windows(self) -> list[str]:
        if self.task.difficulty == "easy":
            return [window for window in ("top", "bottom") if window in self.task.visible_windows]
        return [window for window in ("top",) if window in self.task.visible_windows]

    def _candidate_noise(self) -> float:
        return self.task.ranking_noise + (self.task.corruption_level / 2.0)

    def _execute_action(self, action: ReceiptAction) -> str:
        if action.action_type == "view_receipt":
            total_revealed = 0
            for window in self._default_reveal_windows():
                total_revealed += self._reveal_window(window)
            return f"Viewing receipt {self.hidden_state.sample_id}; revealed {total_revealed} starter regions"

        if action.action_type == "list_text_regions":
            window = action.window or "all"
            revealed = self._reveal_window(window)
            if self.hidden_state.last_error:
                return f"Window {window} is unavailable for {self.task.difficulty}"
            return f"Revealed {revealed} regions from {window}"

        if action.action_type == "inspect_bbox":
            if not action.bbox_id or action.bbox_id not in {region.region_id for region in self.hidden_state.all_regions}:
                self.hidden_state.last_error = "invalid bbox_id"
                return "Invalid bbox inspection request"
            if action.bbox_id not in self.hidden_state.revealed_region_ids:
                self.hidden_state.revealed_region_ids.append(action.bbox_id)
            return f"Revealed bbox {action.bbox_id}"

        if action.action_type == "inspect_neighbors":
            if not action.bbox_id:
                self.hidden_state.last_error = "missing bbox_id"
                return "Missing bbox_id"
            source = next((region for region in self.hidden_state.all_regions if region.region_id == action.bbox_id), None)
            if source is None:
                self.hidden_state.last_error = "invalid bbox_id"
                return "Invalid bbox_id"
            radius = 40 * (action.radius_bucket or 1)
            added = 0
            for region in self.hidden_state.all_regions:
                if abs(region.bbox[1] - source.bbox[1]) <= radius and region.region_id not in self.hidden_state.revealed_region_ids:
                    self.hidden_state.revealed_region_ids.append(region.region_id)
                    added += 1
            return f"Revealed {added} neighboring regions"

        if action.action_type == "query_candidates":
            if not action.field:
                self.hidden_state.last_error = "missing field"
                return "Missing field"
            candidates = query_candidates(
                action.field,
                self._visible_regions(),
                ranking_noise=self._candidate_noise(),
                noise_key=f"{self.hidden_state.sample_id}:{self.task.task_id}:{action.field}",
            )
            self.hidden_state.candidate_lists[action.field] = candidates
            return f"Generated {len(candidates)} candidates for {action.field}"

        if action.action_type == "query_line_item_candidates":
            if not self.task.requires_line_items:
                self.hidden_state.last_error = "line items unsupported"
                return "Line-item candidates are unavailable for this task"
            candidates = query_line_item_candidates(
                self._visible_regions(),
                ranking_noise=self._candidate_noise(),
                noise_key=f"{self.hidden_state.sample_id}:{self.task.task_id}:line_items",
            )
            self.hidden_state.line_item_candidates = candidates
            return f"Generated {len(candidates)} line-item candidates"

        if action.action_type == "set_field_from_candidate":
            if not action.field or not action.candidate_id:
                self.hidden_state.last_error = "missing field or candidate_id"
                return "Missing field or candidate_id"
            candidates = self.hidden_state.candidate_lists.get(action.field, [])
            selected = next((candidate for candidate in candidates if candidate.candidate_id == action.candidate_id), None)
            if selected is None:
                self.hidden_state.last_error = "candidate not found"
                return "Candidate not found"
            setattr(self.hidden_state.current_draft, action.field, selected.value)
            return f"Set {action.field} from candidate"

        if action.action_type == "add_line_item_from_candidate":
            if not action.candidate_id:
                self.hidden_state.last_error = "missing candidate_id"
                return "Missing candidate_id"
            selected = next((candidate for candidate in self.hidden_state.line_item_candidates if candidate.candidate_id == action.candidate_id), None)
            if selected is None:
                self.hidden_state.last_error = "candidate not found"
                return "Candidate not found"
            if any(item.item_id == selected.candidate_id for item in self.hidden_state.current_draft.line_items):
                self.hidden_state.last_error = "line item already added"
                return "Line item already added"
            self.hidden_state.current_draft.line_items.append(
                ReceiptLineItem(
                    item_id=selected.candidate_id,
                    description=selected.description,
                    line_total=selected.line_total,
                    raw_text=selected.raw_text,
                    evidence_ids=list(selected.evidence_ids),
                )
            )
            return "Added line item from candidate"

        if action.action_type == "add_line_item_manual":
            if not self.task.requires_line_items:
                self.hidden_state.last_error = "line items unsupported"
                return "Line items are unavailable for this task"
            description = normalize_text(action.value) or action.value
            line_total = normalize_amount(action.line_total) or action.line_total
            if not description and not line_total:
                self.hidden_state.last_error = "missing line item payload"
                return "Missing line item payload"
            item_index = len(self.hidden_state.current_draft.line_items)
            self.hidden_state.current_draft.line_items.append(
                ReceiptLineItem(
                    item_id=f"manual:{item_index}",
                    description=description,
                    line_total=line_total,
                    quantity=action.quantity,
                    raw_text=" ".join(part for part in (description, line_total) if part),
                    evidence_ids=list(action.evidence_ids or []),
                )
            )
            return "Added manual line item"

        if action.action_type == "remove_line_item":
            if action.line_item_index is None:
                self.hidden_state.last_error = "missing line_item_index"
                return "Missing line_item_index"
            if action.line_item_index < 0 or action.line_item_index >= len(self.hidden_state.current_draft.line_items):
                self.hidden_state.last_error = "invalid line_item_index"
                return "Invalid line_item_index"
            removed = self.hidden_state.current_draft.line_items.pop(action.line_item_index)
            return f"Removed line item {removed.description or removed.item_id or action.line_item_index}"

        if action.action_type == "set_field_manual":
            if not action.field or not action.value:
                self.hidden_state.last_error = "missing manual payload"
                return "Missing manual payload"
            setattr(self.hidden_state.current_draft, action.field, action.value)
            return f"Set {action.field} manually"

        if action.action_type == "merge_spans":
            if not action.field or not action.span_ids:
                self.hidden_state.last_error = "missing merge payload"
                return "Missing merge payload"
            regions = [region.text for region in self._visible_regions() if region.region_id in action.span_ids]
            if not regions:
                self.hidden_state.last_error = "no visible spans"
                return "No visible spans"
            setattr(self.hidden_state.current_draft, action.field, " ".join(regions))
            return f"Merged {len(regions)} spans into {action.field}"

        if action.action_type == "normalize_field":
            if not action.field:
                self.hidden_state.last_error = "missing field"
                return "Missing field"
            current = getattr(self.hidden_state.current_draft, action.field)
            if action.field == "date":
                setattr(self.hidden_state.current_draft, action.field, normalize_date(current) or None)
            elif action.field in {"subtotal", "tax", "total"}:
                setattr(self.hidden_state.current_draft, action.field, normalize_amount(current) or None)
            elif action.field == "address":
                setattr(self.hidden_state.current_draft, action.field, normalize_address(current) or None)
            elif action.field == "company":
                setattr(self.hidden_state.current_draft, action.field, normalize_text(current) or None)
            return f"Normalized {action.field}"

        if action.action_type == "check_total_consistency":
            total = self.hidden_state.current_draft.total
            subtotal = self.hidden_state.current_draft.subtotal
            tax = self.hidden_state.current_draft.tax
            totals = [candidate.value for candidate in self.hidden_state.candidate_lists.get("total", [])]
            if total and subtotal and tax:
                normalized_subtotal = normalize_amount(subtotal)
                normalized_tax = normalize_amount(tax)
                normalized_total = normalize_amount(total)
                if normalized_subtotal and normalized_tax and normalized_total:
                    if round(float(normalized_subtotal) + float(normalized_tax), 2) == round(float(normalized_total), 2):
                        message = "Total reconciles against subtotal and tax"
                    else:
                        message = "Total does not reconcile against subtotal and tax"
                else:
                    message = "Total is weakly supported"
            else:
                valid = bool(total and total in totals)
                message = "Total matches visible evidence" if valid else "Total is weakly supported"
            self.hidden_state.validation_feedback.append(message)
            return message

        if action.action_type == "check_receipt_consistency":
            grade = grade_receipt(
                self.hidden_state.current_draft,
                self.hidden_state.gold_fields,
                task_id=self.task.task_id,
                gold_line_items=self.hidden_state.gold_line_items,
            )
            self.hidden_state.current_reconciliation_delta = grade.reconciliation_delta
            self.hidden_state.current_reconciliation_status = grade.reconciliation_status
            if grade.reconciliation_status == "pass":
                message = "Receipt reconciliation passed"
            elif grade.reconciliation_status == "partial":
                message = "Receipt reconciliation is close"
            else:
                message = "Receipt reconciliation failed"
            if grade.reconciliation_delta is not None:
                message = f"{message} (delta={grade.reconciliation_delta:.2f})"
            self.hidden_state.reconciliation_feedback.append(message)
            return message

        if action.action_type == "check_date_format":
            date_value = self.hidden_state.current_draft.date
            valid = bool(date_value and normalize_date(date_value))
            message = "Date format looks valid" if valid else "Date format is invalid"
            self.hidden_state.validation_feedback.append(message)
            return message

        if action.action_type == "clear_field":
            if not action.field:
                self.hidden_state.last_error = "missing field"
                return "Missing field"
            setattr(self.hidden_state.current_draft, action.field, None)
            return f"Cleared {action.field}"

        if action.action_type == "submit":
            return "Submitted current draft"

        self.hidden_state.last_error = "unsupported action"
        return "Unsupported action"

    def _build_observation(self, message: str) -> ReceiptObservation:
        return ReceiptObservation(
            task_id=self.task.task_id,
            difficulty=self.task.difficulty,
            instruction=self.task.instruction,
            image_ref=self.hidden_state.image_ref,
            image_id=self.hidden_state.image_id,
            image_json_path=self.hidden_state.image_json_path,
            visible_regions=self._visible_regions(),
            candidate_lists=self.hidden_state.candidate_lists,
            line_item_candidates=list(self.hidden_state.line_item_candidates),
            current_draft=self.hidden_state.current_draft.model_copy(deep=True),
            validation_feedback=list(self.hidden_state.validation_feedback),
            reconciliation_feedback=list(self.hidden_state.reconciliation_feedback),
            current_reconciliation_delta=self.hidden_state.current_reconciliation_delta,
            current_reconciliation_status=self.hidden_state.current_reconciliation_status,
            last_action_result=message,
            remaining_budget=self.hidden_state.remaining_budget,
            step_index=self.hidden_state.step_index,
            terminal_allowed=True,
        )
