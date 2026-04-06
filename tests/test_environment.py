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
