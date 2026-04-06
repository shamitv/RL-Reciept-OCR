from env.models import ReceiptAction, ReceiptDraft, ReceiptState
from env.rewards import RewardTracker


def make_state() -> ReceiptState:
    return ReceiptState(
        sample_id="sample",
        difficulty="easy",
        current_draft=ReceiptDraft(company="SHOP"),
        step_index=1,
        remaining_budget=5,
        cumulative_reward=0.0,
        done=False,
    )


def test_reward_improvement() -> None:
    tracker = RewardTracker()
    reward = tracker.compute_step_reward(
        prev_draft=ReceiptDraft(),
        current_state=make_state(),
        action=ReceiptAction(action_type="set_field_manual", field="company", value="SHOP"),
        gold=ReceiptDraft(company="SHOP"),
        action_result="Set company manually",
    )
    assert reward > 0.0


def test_terminal_penalty_for_blanks() -> None:
    tracker = RewardTracker()
    reward = tracker.compute_terminal_reward(type("Grade", (), {"score": 0.8})(), blank_fields=2)
    assert reward == 0.75
