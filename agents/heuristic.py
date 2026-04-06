from __future__ import annotations

from dataclasses import dataclass

from env.environment import ReceiptExtractionEnv
from env.models import ReceiptAction

FIELD_ORDER = ("company", "date", "address", "total")


def tried_windows(env: ReceiptExtractionEnv) -> set[str]:
    return {
        entry.get("action", {}).get("window")
        for entry in env.state().history
        if entry.get("action", {}).get("action_type") == "list_text_regions" and entry.get("action", {}).get("window")
    }


def next_window_to_reveal(env: ReceiptExtractionEnv) -> str | None:
    preferred_order = [window for window in ("all", "top", "middle", "bottom") if window in env.task.visible_windows]
    attempted = tried_windows(env)
    for window in preferred_order:
        if window not in attempted:
            return window
    return None


@dataclass(frozen=True)
class HeuristicAgent:
    name: str = "heuristic"

    def select_action(self, env: ReceiptExtractionEnv) -> ReceiptAction:
        state = env.state()
        obs = env.last_observation

        if state.step_index == 0:
            return ReceiptAction(action_type="view_receipt")

        if not obs.visible_regions:
            window = next_window_to_reveal(env) or env.task.visible_windows[0]
            return ReceiptAction(action_type="list_text_regions", window=window)

        for field in FIELD_ORDER:
            if not obs.candidate_lists.get(field):
                if field in obs.candidate_lists and not obs.candidate_lists[field]:
                    next_window = next_window_to_reveal(env)
                    if next_window is not None:
                        return ReceiptAction(action_type="list_text_regions", window=next_window)
                return ReceiptAction(action_type="query_candidates", field=field)
            if getattr(obs.current_draft, field) is None and obs.candidate_lists[field]:
                return ReceiptAction(
                    action_type="set_field_from_candidate",
                    field=field,
                    candidate_id=obs.candidate_lists[field][0].candidate_id,
                )

        if obs.current_draft.date and not any("date" in item.lower() for item in obs.validation_feedback):
            return ReceiptAction(action_type="check_date_format")

        if obs.current_draft.total and not any("total" in item.lower() for item in obs.validation_feedback):
            return ReceiptAction(action_type="check_total_consistency")

        return ReceiptAction(action_type="submit")
