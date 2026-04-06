from __future__ import annotations

from agents.encoding import OBSERVATION_DIM, encode_observation_values
from env.environment import ReceiptExtractionEnv
from env.models import ReceiptDraft


def make_env(task_name: str = "easy", seed: int = 7) -> ReceiptExtractionEnv:
    env = ReceiptExtractionEnv()
    env.reset(task_name=task_name, seed=seed)
    return env


def test_encode_observation_values_has_expected_dimension() -> None:
    env = make_env()

    encoded = encode_observation_values(env.last_observation, env.task)

    assert len(encoded) == OBSERVATION_DIM


def test_encode_observation_values_is_deterministic() -> None:
    first_env = make_env()
    second_env = make_env()

    first = encode_observation_values(first_env.last_observation, first_env.task)
    second = encode_observation_values(second_env.last_observation, second_env.task)

    assert first == second


def test_encode_observation_values_handles_empty_regions_and_candidates() -> None:
    env = make_env()

    encoded = encode_observation_values(env.last_observation, env.task)

    assert encoded[3] == 0.0
    assert encoded[8:14] == [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]


def test_encode_observation_values_marks_filled_fields() -> None:
    env = make_env()
    env.hidden_state.current_draft = ReceiptDraft(company="Shop", date="2019-03-25", address=None, total="12.00")
    env.last_observation = env._build_observation("filled draft")

    encoded = encode_observation_values(env.last_observation, env.task)

    assert encoded[22:28] == [1.0, 1.0, 0.0, 0.0, 0.0, 1.0]
