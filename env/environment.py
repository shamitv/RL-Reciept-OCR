from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field

from env.candidate_retrieval import query_candidates
from env.dataset import ReceiptDataset
from env.graders import grade_receipt
from env.models import OCRRegion, ReceiptAction, ReceiptDraft, ReceiptObservation, ReceiptState, StepResult, TaskConfig
from env.normalizers import normalize_amount, normalize_date
from env.rewards import RewardTracker
from env.tasks import get_task
from env.utils import make_rng


@dataclass
class HiddenState:
    sample_id: str = ""
    difficulty: str = "easy"
    image_ref: str | None = None
    gold_fields: ReceiptDraft = field(default_factory=ReceiptDraft)
    all_regions: list[OCRRegion] = field(default_factory=list)
    revealed_region_ids: list[str] = field(default_factory=list)
    candidate_lists: dict[str, list] = field(default_factory=dict)
    current_draft: ReceiptDraft = field(default_factory=ReceiptDraft)
    validation_feedback: list[str] = field(default_factory=list)
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
        self.hidden_state = HiddenState()
        self.last_observation = ReceiptObservation(
            task_id="easy",
            difficulty="easy",
            instruction="Extract company, date, address, and total.",
            image_ref=None,
            remaining_budget=0,
            step_index=0,
            terminal_allowed=False,
        )
        self.task: TaskConfig = get_task("easy")
        self.reward_tracker = RewardTracker()

    def reset(self, task_name: str | None = None, seed: int | None = None) -> StepResult:
        rng = make_rng(seed)
        self.task = get_task(task_name)
        sample = self.dataset.sample(self.task.difficulty, rng)
        self.reward_tracker = RewardTracker()
        self.hidden_state = HiddenState(
            sample_id=sample.sample_id,
            difficulty=self.task.difficulty,
            image_ref=sample.image_ref,
            gold_fields=sample.gold_fields,
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
            current_draft=self.hidden_state.current_draft.model_copy(deep=True),
            revealed_region_ids=list(self.hidden_state.revealed_region_ids),
            history=list(self.hidden_state.history),
            step_index=self.hidden_state.step_index,
            remaining_budget=self.hidden_state.remaining_budget,
            cumulative_reward=self.hidden_state.cumulative_reward,
            done=self.hidden_state.done,
            last_error=self.hidden_state.last_error,
        )

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
            final = grade_receipt(self.hidden_state.current_draft, self.hidden_state.gold_fields)
            blank_fields = sum(1 for field in ("company", "date", "address", "total") if getattr(self.hidden_state.current_draft, field) is None)
            reward = self.reward_tracker.compute_terminal_reward(final, blank_fields)
            self.hidden_state.done = True
            info = {"success": final.success, "final_score": final.score, "field_scores": final.field_scores}
        else:
            reward = self.reward_tracker.compute_step_reward(prev_draft, self.state(), action, self.hidden_state.gold_fields, message)

        if self.hidden_state.remaining_budget <= 0 and not self.hidden_state.done:
            final = grade_receipt(self.hidden_state.current_draft, self.hidden_state.gold_fields)
            reward += self.reward_tracker.compute_terminal_reward(final, 0)
            self.hidden_state.done = True
            info.update({"success": final.success, "final_score": final.score, "budget_exhausted": True})

        self.hidden_state.cumulative_reward += reward
        self.hidden_state.history.append({"action": action.model_dump(exclude_none=True), "message": message, "reward": reward})
        self.hidden_state.last_action_result = message
        self.last_observation = self._build_observation(message)
        return StepResult(observation=self.last_observation, reward=reward, done=self.hidden_state.done, info=info)

    def _visible_regions(self) -> list[OCRRegion]:
        ids = set(self.hidden_state.revealed_region_ids)
        return [region for region in self.hidden_state.all_regions if region.region_id in ids]

    def _reveal_window(self, window: str) -> int:
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

    def _execute_action(self, action: ReceiptAction) -> str:
        if action.action_type == "view_receipt":
            return f"Viewing receipt {self.hidden_state.sample_id}"

        if action.action_type == "list_text_regions":
            window = action.window or "all"
            revealed = self._reveal_window(window)
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
            candidates = query_candidates(action.field, self._visible_regions())
            self.hidden_state.candidate_lists[action.field] = candidates
            return f"Generated {len(candidates)} candidates for {action.field}"

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
                setattr(self.hidden_state.current_draft, action.field, normalize_date(current))
            elif action.field == "total":
                setattr(self.hidden_state.current_draft, action.field, normalize_amount(current))
            return f"Normalized {action.field}"

        if action.action_type == "check_total_consistency":
            total = self.hidden_state.current_draft.total
            totals = [candidate.value for candidate in self.hidden_state.candidate_lists.get("total", [])]
            valid = bool(total and total in totals)
            message = "Total matches visible evidence" if valid else "Total is weakly supported"
            self.hidden_state.validation_feedback.append(message)
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
            instruction="Extract company, date, address, and total from the receipt.",
            image_ref=self.hidden_state.image_ref,
            visible_regions=self._visible_regions(),
            candidate_lists=self.hidden_state.candidate_lists,
            current_draft=self.hidden_state.current_draft.model_copy(deep=True),
            validation_feedback=list(self.hidden_state.validation_feedback),
            last_action_result=message,
            remaining_budget=self.hidden_state.remaining_budget,
            step_index=self.hidden_state.step_index,
            terminal_allowed=True,
        )
