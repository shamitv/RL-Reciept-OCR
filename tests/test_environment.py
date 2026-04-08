from types import SimpleNamespace

from env.environment import ReceiptExtractionEnv
from env.models import ReceiptAction


def test_reset_returns_valid_observation() -> None:
    env = ReceiptExtractionEnv()
    result = env.reset(task_name="easy", seed=1)
    assert result.done is False
    assert result.observation.task_id == "easy"


def test_submit_terminates_episode() -> None:
    env = ReceiptExtractionEnv()
    env.reset(task_name="easy", seed=1)
    result = env.step(ReceiptAction(action_type="submit"))
    assert result.done is True
    assert "final_score" in result.info


def test_invalid_action_is_deterministic() -> None:
    env = ReceiptExtractionEnv()
    env.reset(task_name="easy", seed=1)
    result = env.step(ReceiptAction(action_type="inspect_bbox", bbox_id="missing"))
    assert result.observation.last_action_result == "Invalid bbox inspection request"


def test_budget_exhaustion_returns_field_scores() -> None:
    env = ReceiptExtractionEnv()
    env.reset(task_name="hard", seed=1)

    result = None
    for _ in range(env.task.max_steps):
        result = env.step(ReceiptAction(action_type="view_receipt"))

    assert result is not None
    assert result.done is True
    assert result.info["budget_exhausted"] is True
    assert "field_scores" in result.info


def test_submit_publishes_final_score_inside_open_interval(monkeypatch) -> None:
    env = ReceiptExtractionEnv()
    env.reset(task_name="easy", seed=1)

    monkeypatch.setattr(
        "env.environment.grade_receipt",
        lambda *args, **kwargs: SimpleNamespace(
            score=1.0,
            success=True,
            field_scores={},
            header_score=1.0,
            summary_score=0.0,
            line_items_score=0.0,
            reconciliation_score=0.0,
            reconciliation_delta=None,
            reconciliation_status="pass",
        ),
    )

    result = env.step(ReceiptAction(action_type="submit"))

    assert result.info["final_score"] == 0.999
