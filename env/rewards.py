from __future__ import annotations

from env.graders import address_score, company_score, date_score, grade_receipt, subtotal_score, tax_score, total_score
from env.models import GradeResult, ReceiptAction, ReceiptDraft, ReceiptLineItem, ReceiptState, TaskConfig
from env.utils import clamp

FIELD_SCORERS = {
    "company": company_score,
    "date": date_score,
    "address": address_score,
    "subtotal": subtotal_score,
    "tax": tax_score,
    "total": total_score,
}


class RewardTracker:
    def __init__(self) -> None:
        self.best_scores = {field: 0.0 for field in FIELD_SCORERS}
        self.best_line_items_score = 0.0
        self.best_reconciliation_score = 0.0
        self.seen_actions: set[str] = set()

    def snapshot(self) -> dict[str, float]:
        return dict(self.best_scores)

    def score_draft(self, draft: ReceiptDraft, gold: ReceiptDraft) -> dict[str, float]:
        return {
            field: scorer(getattr(draft, field), getattr(gold, field))
            for field, scorer in FIELD_SCORERS.items()
        }

    def compute_step_reward(
        self,
        prev_draft: ReceiptDraft,
        current_state: ReceiptState,
        action: ReceiptAction,
        gold: ReceiptDraft,
        action_result: str,
        gold_line_items: list[ReceiptLineItem] | None = None,
        task: TaskConfig | None = None,
    ) -> float:
        reward = 0.0
        action_key = action.model_dump_json(exclude_none=True)
        if action_key in self.seen_actions:
            reward -= 0.01
        self.seen_actions.add(action_key)

        prev_scores = self.score_draft(prev_draft, gold)
        curr_scores = self.score_draft(current_state.current_draft, gold)
        for field, score in curr_scores.items():
            if score > self.best_scores[field]:
                reward += 0.02 if field not in {"subtotal", "tax", "total"} else 0.03
                self.best_scores[field] = score
            elif score < prev_scores[field]:
                reward -= 0.03

        if task is not None:
            gold_line_items = gold_line_items or []
            prev_grade = grade_receipt(prev_draft, gold, task_id=task.task_id, gold_line_items=gold_line_items)
            curr_grade = grade_receipt(current_state.current_draft, gold, task_id=task.task_id, gold_line_items=gold_line_items)

            if curr_grade.line_items_score > self.best_line_items_score:
                reward += 0.04
                self.best_line_items_score = curr_grade.line_items_score
            elif curr_grade.line_items_score < prev_grade.line_items_score:
                reward -= 0.03

            if curr_grade.reconciliation_score > self.best_reconciliation_score:
                reward += 0.03
                self.best_reconciliation_score = curr_grade.reconciliation_score
            elif curr_grade.reconciliation_score < prev_grade.reconciliation_score:
                reward -= 0.02

        if "revealed" in action_result.lower():
            reward += 0.02
        if current_state.last_error:
            reward -= 0.02
        if current_state.remaining_budget < 3:
            reward -= 0.01
        return clamp(reward, -1.0, 1.0)

    def compute_terminal_reward(self, final_grade: GradeResult, blank_fields: int) -> float:
        reward = final_grade.score
        if blank_fields >= 2:
            reward -= 0.05
        return clamp(reward, -1.0, 1.0)
