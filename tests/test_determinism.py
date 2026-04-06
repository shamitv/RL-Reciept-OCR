from env.environment import ReceiptExtractionEnv
from env.models import ReceiptAction


def rollout() -> tuple[list[str], list[float]]:
    env = ReceiptExtractionEnv()
    env.reset(task_name="easy", seed=5)
    actions = [
        ReceiptAction(action_type="view_receipt"),
        ReceiptAction(action_type="list_text_regions", window="all"),
        ReceiptAction(action_type="query_candidates", field="company"),
    ]
    messages = []
    rewards = []
    for action in actions:
        result = env.step(action)
        messages.append(result.observation.last_action_result)
        rewards.append(result.reward)
    return messages, rewards


def test_same_seed_same_rollout() -> None:
    first = rollout()
    second = rollout()
    assert first == second
