from inference import TASK_ORDER, episode_seed, evaluate_tasks, resolve_tasks, run_episode


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