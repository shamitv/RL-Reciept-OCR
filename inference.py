from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from statistics import mean
from typing import Any

from agents.base import Agent
from agents.heuristic import HeuristicAgent
from env.config import load_environment
from env.environment import ReceiptExtractionEnv
from env.evaluation import DatasetAuditRecord, build_model_client, require_env, run_extraction_model
from env.image_store import IMAGE_JSON_DIR_NAME
from env.models import ReceiptAction, ReceiptDraft, ReceiptLineItem, ReceiptSample
from env.tasks import TASKS

load_environment()

# Submission-facing environment variables are declared here to mirror the
# sample inference contract. LOCAL_IMAGE_NAME stays optional because this
# environment does not use from_docker_image().
API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
MODEL_NAME = os.getenv("MODEL_NAME") or "Qwen/Qwen2.5-72B-Instruct"
HF_TOKEN = os.getenv("HF_TOKEN", "").strip()
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")

TASK_ORDER = tuple(TASKS.keys())
ENV_NAME = "rl-receipt-ocr"
DEFAULT_SELECTION_MANIFEST = Path(__file__).resolve().parent / "artifacts" / "datasets" / "receipt-selection-50" / "selected_manifest.json"


def one_line(value: object) -> str:
    text = str(value)
    return " ".join(text.splitlines()).strip()


def log_start(task: str, env_name: str, model: str) -> None:
    print(f"[START] task={one_line(task)} env={one_line(env_name)} model={one_line(model)}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: str | None) -> None:
    error_value = one_line(error) if error else "null"
    print(
        f"[STEP] step={step} action={one_line(action)} reward={reward:.2f} done={str(done).lower()} error={error_value}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: list[float]) -> None:
    rewards_str = ",".join(f"{reward:.2f}" for reward in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}", flush=True)


def close_env(env: object) -> None:
    close = getattr(env, "close", None)
    if callable(close):
        try:
            close()
        except Exception:
            pass


def default_agent() -> Agent:
    return HeuristicAgent()


def build_agent(agent_name: str = "heuristic", checkpoint: str | None = None, device: str = "cpu") -> Agent:
    if agent_name == "heuristic":
        return HeuristicAgent()
    if agent_name == "ppo":
        if not checkpoint:
            raise ValueError("--checkpoint is required when --agent ppo")
        from agents.ppo import PPOAgent

        return PPOAgent(checkpoint_path=checkpoint, device=device)
    raise ValueError(f"unsupported agent: {agent_name}")


def episode_seed(base_seed: int, task: str, episode_index: int) -> int:
    return base_seed + (TASK_ORDER.index(task) * 1000) + episode_index


def run_episode(task: str, seed: int, agent: Agent | None = None, verbose: bool = False) -> dict[str, Any]:
    selected_agent = agent or default_agent()
    env = ReceiptExtractionEnv()
    result = None
    rewards: list[float] = []
    steps = 0
    score = 0.0
    success = False

    if verbose:
        log_start(task=task, env_name=ENV_NAME, model=selected_agent.name)

    try:
        result = env.reset(task_name=task, seed=seed)
        while not result.done and steps < env.task.max_steps:
            steps += 1
            action = selected_agent.select_action(env)
            result = env.step(action)
            rewards.append(result.reward)
            if verbose:
                log_step(steps, action.model_dump_json(), result.reward, result.done, env.state().last_error)
            if result.done:
                score = float(result.info.get("final_score", 0.0))
                success = bool(result.info.get("success", False))
                break
    finally:
        close_env(env)
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
        "budget_exhausted": bool(result.info.get("budget_exhausted", False)) if result is not None else False,
        "field_scores": result.info.get("field_scores", {}) if result is not None else {},
    }


def build_llm_client_from_env() -> tuple[Any, str]:
    if not HF_TOKEN:
        require_env("HF_TOKEN")
    return build_model_client(API_BASE_URL), MODEL_NAME


def audit_record_from_env(env: ReceiptExtractionEnv) -> DatasetAuditRecord:
    hidden_state = env.hidden_state
    image_json_path = hidden_state.image_json_path or hidden_state.image_ref
    if not image_json_path:
        raise ValueError("LLM inference requires the selected receipt to have an image JSON asset")
    return DatasetAuditRecord(
        sample_id=hidden_state.sample_id,
        task_id=hidden_state.task_id,
        annotation_path=f"{hidden_state.sample_id}.json",
        image_id=hidden_state.image_id or hidden_state.sample_id,
        image_json_path=image_json_path,
        dataset_status="runnable",
        gold_fields=hidden_state.gold_fields,
        gold_line_items=hidden_state.gold_line_items,
    )


def _maybe_existing_path(value: object) -> Path | None:
    if value is None:
        return None
    path = Path(str(value))
    return path if path.exists() else None


def _dataset_root_candidates_from_manifest(manifest_path: Path) -> list[Path]:
    candidates: list[Path] = []
    seen: set[Path] = set()

    for ancestor in manifest_path.parents:
        candidate = ancestor / "dataset" / "Receipt dataset" / "ds0"
        if candidate in seen:
            continue
        seen.add(candidate)
        candidates.append(candidate)

    return candidates


def _resolve_selected_dataset_path(
    *,
    manifest_path: Path,
    sample_id: str,
    subdir: str,
    manifest_value: object,
) -> Path:
    candidates: list[Path] = [manifest_path.parent / "dataset" / subdir / f"{sample_id}.json"]

    manifest_candidate = _maybe_existing_path(manifest_value)
    if manifest_candidate is not None:
        candidates.append(manifest_candidate)

    for dataset_root in _dataset_root_candidates_from_manifest(manifest_path):
        candidates.append(dataset_root / subdir / f"{sample_id}.json")

    seen: set[Path] = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        if candidate.exists():
            return candidate

    return Path(str(manifest_value if manifest_value is not None else candidates[0]))


def audit_record_from_selected_record(record: dict[str, Any], manifest_path: Path) -> DatasetAuditRecord:
    sample_id = str(record["sample_id"])
    image_json_path = _resolve_selected_dataset_path(
        manifest_path=manifest_path,
        sample_id=sample_id,
        subdir=IMAGE_JSON_DIR_NAME,
        manifest_value=record.get("image_json_path"),
    )
    annotation_path = _resolve_selected_dataset_path(
        manifest_path=manifest_path,
        sample_id=sample_id,
        subdir="ann",
        manifest_value=record.get("annotation_path"),
    )

    gold_fields_payload = record.get("gold_fields") or {}
    gold_line_items_payload = record.get("gold_line_items") or []
    return DatasetAuditRecord(
        sample_id=sample_id,
        task_id=record.get("task_id"),
        annotation_path=str(annotation_path),
        image_id=str(record.get("image_id") or sample_id),
        image_json_path=str(image_json_path),
        dataset_status=record.get("dataset_status", "runnable"),
        skip_reason=record.get("skip_reason"),
        missing_categories=list(record.get("missing_categories") or []),
        unparseable_fields=list(record.get("unparseable_fields") or []),
        gold_fields=ReceiptDraft.model_validate(gold_fields_payload) if gold_fields_payload else None,
        gold_line_items=[ReceiptLineItem.model_validate(item) for item in gold_line_items_payload],
    )


def load_selected_audit_records(manifest_path: str | Path, task_filter: str = "all") -> list[DatasetAuditRecord]:
    resolved_manifest_path = Path(manifest_path)
    payload = json.loads(resolved_manifest_path.read_text(encoding="utf-8"))
    records = payload.get("records", [])
    if not isinstance(records, list):
        raise ValueError(f"Selected manifest does not contain a records list: {resolved_manifest_path}")

    audit_records = [
        audit_record_from_selected_record(record, resolved_manifest_path)
        for record in records
        if isinstance(record, dict)
    ]
    if task_filter != "all":
        audit_records = [record for record in audit_records if record.task_id == task_filter]
    if not audit_records:
        raise ValueError(f"No selected records found for task={task_filter} in {resolved_manifest_path}")
    return audit_records


def format_action(action: ReceiptAction) -> str:
    return action.model_dump_json(exclude_none=True)


def actions_from_llm_prediction(
    prediction: ReceiptDraft,
    field_order: list[str],
    requires_line_items: bool,
    max_actions: int,
) -> list[ReceiptAction]:
    actions: list[ReceiptAction] = []
    for field in field_order:
        value = getattr(prediction, field, None)
        if value is not None and str(value).strip():
            actions.append(ReceiptAction(action_type="set_field_manual", field=field, value=str(value).strip()))

    if requires_line_items:
        for item in prediction.line_items:
            description = item.description or item.raw_text
            if description or item.line_total:
                actions.append(
                    ReceiptAction(
                        action_type="add_line_item_manual",
                        value=description,
                        line_total=item.line_total,
                        quantity=item.quantity,
                        evidence_ids=item.evidence_ids or None,
                    )
                )

    return actions[:max_actions]


def run_llm_episode(task: str, seed: int, client: Any, model_name: str, emit_logs: bool = False) -> dict[str, Any]:
    env = ReceiptExtractionEnv()
    result = None
    rewards: list[float] = []
    steps = 0
    score = 0.0
    success = False
    sample_id: str | None = None
    field_scores: dict[str, float] = {}
    error: str | None = None

    if emit_logs:
        log_start(task=task, env_name=ENV_NAME, model=model_name)

    try:
        result = env.reset(task_name=task, seed=seed)
        sample_id = env.state().sample_id
        prediction = run_extraction_model(audit_record_from_env(env), client, model_name)
        actions = actions_from_llm_prediction(
            prediction=prediction,
            field_order=env.task.target_fields,
            requires_line_items=env.task.requires_line_items,
            max_actions=max(env.task.max_steps - 1, 0),
        )

        for action in actions:
            if result.done:
                break
            steps += 1
            result = env.step(action)
            rewards.append(result.reward)
            if emit_logs:
                log_step(steps, format_action(action), result.reward, result.done, env.state().last_error)

        if not result.done:
            action = ReceiptAction(action_type="submit")
            steps += 1
            result = env.step(action)
            rewards.append(result.reward)
            if emit_logs:
                log_step(steps, format_action(action), result.reward, result.done, env.state().last_error)
    except Exception as exc:  # pragma: no cover - exact model/client failures vary
        error = str(exc)
        if result is not None and not result.done and steps < env.task.max_steps:
            action = ReceiptAction(action_type="submit")
            steps += 1
            result = env.step(action)
            rewards.append(result.reward)
            if emit_logs:
                log_step(steps, format_action(action), result.reward, result.done, error)
    finally:
        close_env(env)
        if result is not None and result.done:
            score = float(result.info.get("final_score", 0.0))
            success = bool(result.info.get("success", False))
            field_scores = result.info.get("field_scores", {})
        if emit_logs:
            log_end(success=success, steps=steps, score=score, rewards=rewards)

    return {
        "task": task,
        "seed": seed,
        "sample_id": sample_id,
        "success": success,
        "score": score,
        "steps": steps,
        "max_steps": env.task.max_steps,
        "cumulative_reward": round(sum(rewards), 6),
        "reward_trace": [round(reward, 6) for reward in rewards],
        "budget_exhausted": bool(result.info.get("budget_exhausted", False)) if result is not None else False,
        "field_scores": field_scores,
        "error": error,
    }


def sample_from_audit_record(record: DatasetAuditRecord) -> ReceiptSample:
    return ReceiptSample(
        sample_id=record.sample_id,
        image_ref=record.image_json_path,
        image_id=record.image_id,
        image_json_path=record.image_json_path,
        regions=[],
        gold_fields=record.gold_fields or ReceiptDraft(),
        gold_line_items=record.gold_line_items,
    )


def run_llm_audit_record(record: DatasetAuditRecord, client: Any, model_name: str, emit_logs: bool = False) -> dict[str, Any]:
    task = record.task_id or "easy"
    env = ReceiptExtractionEnv()
    result = None
    rewards: list[float] = []
    steps = 0
    score = 0.0
    success = False
    field_scores: dict[str, float] = {}
    error: str | None = None

    if record.dataset_status != "runnable":
        raise ValueError(f"Selected inference records must be runnable: {record.sample_id}")

    if emit_logs:
        log_start(task=task, env_name=ENV_NAME, model=model_name)

    try:
        result = env.reset_with_sample(sample_from_audit_record(record), task_name=task)
        prediction = run_extraction_model(record, client, model_name)
        actions = actions_from_llm_prediction(
            prediction=prediction,
            field_order=env.task.target_fields,
            requires_line_items=env.task.requires_line_items,
            max_actions=max(env.task.max_steps - 1, 0),
        )

        for action in actions:
            if result.done:
                break
            steps += 1
            result = env.step(action)
            rewards.append(result.reward)
            if emit_logs:
                log_step(steps, format_action(action), result.reward, result.done, env.state().last_error)

        if not result.done:
            action = ReceiptAction(action_type="submit")
            steps += 1
            result = env.step(action)
            rewards.append(result.reward)
            if emit_logs:
                log_step(steps, format_action(action), result.reward, result.done, env.state().last_error)
    except Exception as exc:  # pragma: no cover - exact model/client failures vary
        error = str(exc)
        if result is not None and not result.done and steps < env.task.max_steps:
            action = ReceiptAction(action_type="submit")
            steps += 1
            result = env.step(action)
            rewards.append(result.reward)
            if emit_logs:
                log_step(steps, format_action(action), result.reward, result.done, error)
    finally:
        close_env(env)
        if result is not None and result.done:
            score = float(result.info.get("final_score", 0.0))
            success = bool(result.info.get("success", False))
            field_scores = result.info.get("field_scores", {})
        if emit_logs:
            log_end(success=success, steps=steps, score=score, rewards=rewards)

    return {
        "task": task,
        "seed": None,
        "sample_id": record.sample_id,
        "success": success,
        "score": score,
        "steps": steps,
        "max_steps": env.task.max_steps,
        "cumulative_reward": round(sum(rewards), 6),
        "reward_trace": [round(reward, 6) for reward in rewards],
        "budget_exhausted": bool(result.info.get("budget_exhausted", False)) if result is not None else False,
        "field_scores": field_scores,
        "error": error,
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


def aggregate_task_summaries(
    task_summaries: list[dict[str, Any]],
    agent_name: str,
    seed: int | None,
    episodes: int,
    model_name: str | None = None,
) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "agent": agent_name,
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
    if model_name is not None:
        summary["model"] = model_name
    return summary


def evaluate_tasks(tasks: list[str], seed: int = 7, episodes: int = 1, agent: Agent | None = None, verbose: bool = False) -> dict[str, Any]:
    selected_agent = agent or default_agent()
    task_summaries: list[dict[str, Any]] = []
    for task in tasks:
        runs = [
            run_episode(task=task, seed=episode_seed(seed, task, episode_index), agent=selected_agent, verbose=verbose)
            for episode_index in range(episodes)
        ]
        task_summaries.append(summarize_task_runs(task, runs))

    return aggregate_task_summaries(task_summaries, selected_agent.name, seed, episodes)


def evaluate_tasks_with_llm(
    tasks: list[str],
    seed: int,
    episodes: int,
    client: Any,
    model_name: str,
    emit_logs: bool = False,
) -> dict[str, Any]:
    task_summaries: list[dict[str, Any]] = []
    for task in tasks:
        runs = [
            run_llm_episode(
                task=task,
                seed=episode_seed(seed, task, episode_index),
                client=client,
                model_name=model_name,
                emit_logs=emit_logs,
            )
            for episode_index in range(episodes)
        ]
        task_summaries.append(summarize_task_runs(task, runs))

    return aggregate_task_summaries(task_summaries, "llm", seed, episodes, model_name)


def evaluate_selected_records_with_llm(
    records: list[DatasetAuditRecord],
    client: Any,
    model_name: str,
    emit_logs: bool = False,
) -> dict[str, Any]:
    runs = [
        run_llm_audit_record(record=record, client=client, model_name=model_name, emit_logs=emit_logs)
        for record in records
    ]
    task_summaries = [
        summarize_task_runs(task, [run for run in runs if run["task"] == task])
        for task in TASK_ORDER
        if any(run["task"] == task for run in runs)
    ]
    summary = aggregate_task_summaries(task_summaries, "llm", None, 1, model_name)
    summary["sample_count"] = len(runs)
    return summary


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
    parser.add_argument("--agent", choices=["llm", "heuristic", "ppo"], default="llm")
    parser.add_argument("--checkpoint")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--task", choices=[*TASK_ORDER, "all"], default="all")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--format", choices=["text", "json"], default="text")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--manifest", default=str(DEFAULT_SELECTION_MANIFEST))
    parser.add_argument("--no-manifest", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        tasks = resolve_tasks(args.task)
        if args.agent == "llm":
            client, model_name = build_llm_client_from_env()
            manifest_path = Path(args.manifest)
            if not args.no_manifest and manifest_path.exists():
                selected_records = load_selected_audit_records(manifest_path, task_filter=args.task)
                summary = evaluate_selected_records_with_llm(
                    records=selected_records,
                    client=client,
                    model_name=model_name,
                    emit_logs=args.format == "text",
                )
            else:
                summary = evaluate_tasks_with_llm(
                    tasks=tasks,
                    seed=args.seed,
                    episodes=args.episodes,
                    client=client,
                    model_name=model_name,
                    emit_logs=args.format == "text",
                )
        else:
            agent = build_agent(agent_name=args.agent, checkpoint=args.checkpoint, device=args.device)
            summary = evaluate_tasks(tasks=tasks, seed=args.seed, episodes=args.episodes, agent=agent, verbose=args.verbose)
    except (FileNotFoundError, ImportError, ValueError, RuntimeError) as exc:
        parser.error(str(exc))

    if args.agent == "llm" and args.format == "text":
        return 0
    if args.format == "json":
        print(json.dumps(summary, indent=2), flush=True)
    else:
        print_text_summary(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
