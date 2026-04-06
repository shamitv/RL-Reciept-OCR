from __future__ import annotations

import argparse
import json
from statistics import mean
from typing import Any

from env.config import load_environment
from env.environment import ReceiptExtractionEnv
from env.models import ReceiptAction
from env.tasks import TASKS

load_environment()

TASK_ORDER = tuple(TASKS.keys())


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


def log_start(task: str, env_name: str, agent: str, seed: int) -> None:
    print(f"[START] task={task} env={env_name} agent={agent} seed={seed}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: str | None) -> None:
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={str(done).lower()} error={error or 'null'}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: list[float]) -> None:
    rewards_str = ",".join(f"{reward:.2f}" for reward in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}", flush=True)


def heuristic_action(env: ReceiptExtractionEnv) -> ReceiptAction:
    state = env.state()
    obs = env.last_observation

    if state.step_index == 0:
        return ReceiptAction(action_type="view_receipt")

    if not obs.visible_regions:
        window = next_window_to_reveal(env) or env.task.visible_windows[0]
        return ReceiptAction(action_type="list_text_regions", window=window)

    for field in ("company", "date", "address", "total"):
        if not obs.candidate_lists.get(field):
            if field in obs.candidate_lists and not obs.candidate_lists[field]:
                next_window = next_window_to_reveal(env)
                if next_window is not None:
                    return ReceiptAction(action_type="list_text_regions", window=next_window)
            return ReceiptAction(action_type="query_candidates", field=field)
        if getattr(obs.current_draft, field) is None and obs.candidate_lists[field]:
            return ReceiptAction(action_type="set_field_from_candidate", field=field, candidate_id=obs.candidate_lists[field][0].candidate_id)

    if obs.current_draft.date and not any("date" in item.lower() for item in obs.validation_feedback):
        return ReceiptAction(action_type="check_date_format")

    if obs.current_draft.total and not any("total" in item.lower() for item in obs.validation_feedback):
        return ReceiptAction(action_type="check_total_consistency")

    return ReceiptAction(action_type="submit")


def episode_seed(base_seed: int, task: str, episode_index: int) -> int:
    return base_seed + (TASK_ORDER.index(task) * 1000) + episode_index


def run_episode(task: str, seed: int, verbose: bool = False) -> dict[str, Any]:
    env = ReceiptExtractionEnv()
    result = env.reset(task_name=task, seed=seed)
    rewards: list[float] = []
    steps = 0
    score = 0.0
    success = False

    if verbose:
        log_start(task=task, env_name="rl-receipt-ocr", agent="heuristic", seed=seed)

    while not result.done and steps < env.task.max_steps:
        steps += 1
        action = heuristic_action(env)
        result = env.step(action)
        rewards.append(result.reward)
        if verbose:
            log_step(steps, action.model_dump_json(), result.reward, result.done, env.state().last_error)
        if result.done:
            score = float(result.info.get("final_score", 0.0))
            success = bool(result.info.get("success", False))
            break

    if verbose:
        log_end(success=success, steps=steps, score=score, rewards=rewards)

    return {
        "task": task,
        "seed": seed,
        "sample_id": env.state().sample_id,
        "success": success,
        "score": score,
        "steps": steps,
        "max_steps": env.task.max_steps,
        "cumulative_reward": round(sum(rewards), 6),
        "reward_trace": [round(reward, 6) for reward in rewards],
        "budget_exhausted": bool(result.info.get("budget_exhausted", False)),
        "field_scores": result.info.get("field_scores", {}),
    }


def summarize_task_runs(task: str, runs: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "task": task,
        "episodes": len(runs),
        "mean_score": round(mean(run["score"] for run in runs), 6),
        "success_rate": round(mean(1.0 if run["success"] else 0.0 for run in runs), 6),
        "mean_steps": round(mean(run["steps"] for run in runs), 6),
        "mean_cumulative_reward": round(mean(run["cumulative_reward"] for run in runs), 6),
        "runs": runs,
    }


def evaluate_tasks(tasks: list[str], seed: int = 7, episodes: int = 1, verbose: bool = False) -> dict[str, Any]:
    task_summaries: list[dict[str, Any]] = []
    for task in tasks:
        runs = [run_episode(task=task, seed=episode_seed(seed, task, episode_index), verbose=verbose) for episode_index in range(episodes)]
        task_summaries.append(summarize_task_runs(task, runs))

    return {
        "agent": "heuristic",
        "base_seed": seed,
        "episodes_per_task": episodes,
        "tasks": task_summaries,
        "aggregate": {
            "task_count": len(task_summaries),
            "mean_score": round(mean(task_summary["mean_score"] for task_summary in task_summaries), 6),
            "mean_success_rate": round(mean(task_summary["success_rate"] for task_summary in task_summaries), 6),
            "mean_steps": round(mean(task_summary["mean_steps"] for task_summary in task_summaries), 6),
            "mean_cumulative_reward": round(mean(task_summary["mean_cumulative_reward"] for task_summary in task_summaries), 6),
        },
    }


def resolve_tasks(task_arg: str) -> list[str]:
    if task_arg == "all":
        return list(TASK_ORDER)
    return [task_arg]


def print_text_summary(summary: dict[str, Any]) -> None:
    print(
        f"[SUMMARY] agent={summary['agent']} base_seed={summary['base_seed']} episodes_per_task={summary['episodes_per_task']}",
        flush=True,
    )
    for task_summary in summary["tasks"]:
        print(
            "[TASK] "
            f"task={task_summary['task']} episodes={task_summary['episodes']} mean_score={task_summary['mean_score']:.3f} "
            f"success_rate={task_summary['success_rate']:.3f} mean_steps={task_summary['mean_steps']:.2f} "
            f"mean_reward={task_summary['mean_cumulative_reward']:.3f}",
            flush=True,
        )
    aggregate = summary["aggregate"]
    print(
        "[AGGREGATE] "
        f"task_count={aggregate['task_count']} mean_score={aggregate['mean_score']:.3f} "
        f"mean_success_rate={aggregate['mean_success_rate']:.3f} mean_steps={aggregate['mean_steps']:.2f} "
        f"mean_reward={aggregate['mean_cumulative_reward']:.3f}",
        flush=True,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent", choices=["heuristic"], default="heuristic")
    parser.add_argument("--task", choices=[*TASK_ORDER, "all"], default="all")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--format", choices=["text", "json"], default="text")
    parser.add_argument("--verbose", action="store_true")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    tasks = resolve_tasks(args.task)
    summary = evaluate_tasks(tasks=tasks, seed=args.seed, episodes=args.episodes, verbose=args.verbose)
    if args.format == "json":
        print(json.dumps(summary, indent=2), flush=True)
    else:
        print_text_summary(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
