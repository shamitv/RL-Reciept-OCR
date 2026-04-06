from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from agents.ppo import (
    CheckpointIncompatibleError,
    PARAMETER_HEAD_REQUIREMENTS,
    SUPPORTED_PPO_ACTIONS,
    action_type_mask,
    parameter_choices,
    validate_checkpoint_payload,
)
from env.environment import ReceiptExtractionEnv
from env.models import ReceiptAction


def make_env(task_name: str = "easy", seed: int = 7) -> ReceiptExtractionEnv:
    env = ReceiptExtractionEnv()
    env.reset(task_name=task_name, seed=seed)
    return env


def make_checkpoint_payload() -> dict:
    param_heads = {
        "list_text_regions": {"window": 4},
        "inspect_bbox": {"bbox_index": 8},
        "inspect_neighbors": {"bbox_index": 8, "radius_bucket": 3},
        "query_candidates": {"field": 4},
        "set_field_from_candidate": {"field": 4, "candidate_index": 8},
        "normalize_field": {"field": 4},
        "clear_field": {"field": 4},
    }
    assert set(PARAMETER_HEAD_REQUIREMENTS).issubset(param_heads)
    return {
        "model_state_dict": {},
        "obs_dim": 26,
        "action_types": list(SUPPORTED_PPO_ACTIONS),
        "param_heads": param_heads,
        "architecture": {"hidden_sizes": [16, 16], "activation": "relu"},
        "encoder_version": "receipt-obs-v1",
    }


def test_action_type_mask_blocks_submit_until_a_field_is_filled() -> None:
    env = make_env()

    initial_mask = action_type_mask(env)
    assert initial_mask["submit"] is False

    env.step(ReceiptAction(action_type="view_receipt"))
    env.step(ReceiptAction(action_type="query_candidates", field="company"))
    env.step(ReceiptAction(action_type="set_field_from_candidate", field="company", candidate_id=env.last_observation.candidate_lists["company"][0].candidate_id))

    updated_mask = action_type_mask(env)
    assert updated_mask["submit"] is True


def test_parameter_choices_reflect_task_constraints() -> None:
    env = make_env(task_name="hard")

    assert parameter_choices(env, "list_text_regions")["window"] == ["top", "bottom"]


def test_parameter_choices_reflect_visible_bbox_and_candidate_state() -> None:
    env = make_env()
    env.step(ReceiptAction(action_type="view_receipt"))
    env.step(ReceiptAction(action_type="query_candidates", field="company"))

    bbox_choices = parameter_choices(env, "inspect_bbox")["bbox_index"]
    field_choices = parameter_choices(env, "set_field_from_candidate")["field"]

    assert bbox_choices
    assert "company" in field_choices


def test_validate_checkpoint_payload_rejects_incompatible_action_set() -> None:
    payload = make_checkpoint_payload()
    payload["action_types"] = ["view_receipt"]

    with pytest.raises(CheckpointIncompatibleError, match="action_types"):
        validate_checkpoint_payload(payload)


def test_validate_checkpoint_payload_rejects_missing_fields() -> None:
    payload = make_checkpoint_payload()
    del payload["architecture"]

    with pytest.raises(CheckpointIncompatibleError, match="architecture"):
        validate_checkpoint_payload(payload)


TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None


def create_fake_checkpoint(path: Path) -> None:
    if not TORCH_AVAILABLE:
        pytest.skip("torch is not installed")

    import torch
    from agents.ppo import build_policy_network

    payload = make_checkpoint_payload()
    model = build_policy_network(
        torch,
        obs_dim=payload["obs_dim"],
        action_types=tuple(payload["action_types"]),
        architecture=payload["architecture"],
        param_heads=payload["param_heads"],
    )
    payload["model_state_dict"] = model.state_dict()
    torch.save(payload, path)


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch is not installed")
def test_ppo_agent_loads_fake_checkpoint(tmp_path: Path) -> None:
    from agents.ppo import PPOAgent

    checkpoint_path = tmp_path / "policy.pt"
    create_fake_checkpoint(checkpoint_path)

    agent = PPOAgent(checkpoint_path=checkpoint_path)

    assert agent.name == "ppo"


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch is not installed")
def test_ppo_agent_runs_episode_with_fake_checkpoint(tmp_path: Path) -> None:
    from agents.ppo import PPOAgent

    checkpoint_path = tmp_path / "policy.pt"
    create_fake_checkpoint(checkpoint_path)

    env = make_env()
    agent = PPOAgent(checkpoint_path=checkpoint_path)
    steps = 0
    done = False

    while not done and steps < env.task.max_steps:
        steps += 1
        action = agent.select_action(env)
        assert action.action_type in SUPPORTED_PPO_ACTIONS
        result = env.step(action)
        done = result.done

    assert done is True
