from __future__ import annotations

from dataclasses import dataclass

import pytest

from agents.heuristic import HeuristicAgent
from env.environment import ReceiptExtractionEnv
from env.models import ReceiptAction
from inference import TASK_ORDER, build_agent, episode_seed, evaluate_tasks, resolve_tasks, run_episode


@dataclass(frozen=True)
class WrappedHeuristicAgent:
    name: str = "ppo-mock"

    def select_action(self, env: ReceiptExtractionEnv) -> ReceiptAction:
        return HeuristicAgent().select_action(env)


def test_resolve_tasks_defaults_to_all_tasks() -> None:
    assert resolve_tasks("all") == list(TASK_ORDER)


def test_episode_seed_is_deterministic_and_task_specific() -> None:
    assert episode_seed(7, "easy", 0) == 7
    assert episode_seed(7, "medium", 0) == 1007
    assert episode_seed(7, "hard", 2) == 2009


def test_run_episode_is_deterministic_for_fixed_seed() -> None:
    first = run_episode(task="easy", seed=7)
    second = run_episode(task="easy", seed=7)

    assert first == second


def test_evaluate_tasks_returns_all_task_summary() -> None:
    summary = evaluate_tasks(tasks=list(TASK_ORDER), seed=7, episodes=1)

    assert summary["agent"] == "heuristic"
    assert summary["episodes_per_task"] == 1
    assert [task_summary["task"] for task_summary in summary["tasks"]] == list(TASK_ORDER)
    assert summary["aggregate"]["task_count"] == 3


def test_build_agent_defaults_to_heuristic() -> None:
    agent = build_agent()

    assert isinstance(agent, HeuristicAgent)


def test_build_agent_requires_checkpoint_for_ppo() -> None:
    with pytest.raises(ValueError, match="--checkpoint is required when --agent ppo"):
        build_agent(agent_name="ppo")


def test_custom_agent_name_flows_into_summary() -> None:
    summary = evaluate_tasks(tasks=["easy"], seed=7, episodes=1, agent=WrappedHeuristicAgent())

    assert summary["agent"] == "ppo-mock"


def test_custom_agent_keeps_summary_schema() -> None:
    heuristic_summary = evaluate_tasks(tasks=["easy"], seed=7, episodes=1)
    wrapped_summary = evaluate_tasks(tasks=["easy"], seed=7, episodes=1, agent=WrappedHeuristicAgent())

    assert set(heuristic_summary) == set(wrapped_summary)
    assert set(heuristic_summary["aggregate"]) == set(wrapped_summary["aggregate"])
