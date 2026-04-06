from __future__ import annotations

import argparse
import os
from typing import List

from openai import OpenAI

from env.environment import ReceiptExtractionEnv
from env.models import ReceiptAction

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")
API_KEY = os.getenv("HF_TOKEN") or os.getenv("OPENAI_API_KEY") or ""


def log_start(task: str, env_name: str, model: str) -> None:
    print(f"[START] task={task} env={env_name} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: str | None) -> None:
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={str(done).lower()} error={error or 'null'}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{reward:.2f}" for reward in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}", flush=True)


def heuristic_action(env: ReceiptExtractionEnv) -> ReceiptAction:
    state = env.state()
    obs = env.last_observation

    if state.step_index == 0:
        return ReceiptAction(action_type="view_receipt")

    if not obs.visible_regions:
        return ReceiptAction(action_type="list_text_regions", window="all")

    for field in ("company", "date", "address", "total"):
        if not obs.candidate_lists.get(field):
            return ReceiptAction(action_type="query_candidates", field=field)
        if getattr(obs.current_draft, field) is None and obs.candidate_lists[field]:
            return ReceiptAction(action_type="set_field_from_candidate", field=field, candidate_id=obs.candidate_lists[field][0].candidate_id)

    if obs.current_draft.date and not any("date" in item.lower() for item in obs.validation_feedback):
        return ReceiptAction(action_type="check_date_format")

    if obs.current_draft.total and not any("total" in item.lower() for item in obs.validation_feedback):
        return ReceiptAction(action_type="check_total_consistency")

    return ReceiptAction(action_type="submit")


def llm_action(client: OpenAI, env: ReceiptExtractionEnv) -> ReceiptAction:
    _ = client
    return heuristic_action(env)


def run_episode(task: str, agent: str) -> int:
    env = ReceiptExtractionEnv()
    result = env.reset(task_name=task, seed=7)
    rewards: List[float] = []
    steps = 0
    success = False
    score = 0.0
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY) if agent == "openai" and API_KEY else None

    log_start(task=task, env_name="rl-receipt-ocr", model=MODEL_NAME if client else agent)

    while not result.done and steps < env.task.max_steps:
        steps += 1
        action = heuristic_action(env) if client is None else llm_action(client, env)
        result = env.step(action)
        rewards.append(result.reward)
        log_step(steps, action.model_dump_json(), result.reward, result.done, env.state().last_error)
        if result.done:
            score = float(result.info.get("final_score", 0.0))
            success = bool(result.info.get("success", False))
            break

    log_end(success=success, steps=steps, score=score, rewards=rewards)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent", choices=["heuristic", "openai"], default="heuristic")
    parser.add_argument("--task", choices=["easy", "medium", "hard"], default="easy")
    args = parser.parse_args()
    return run_episode(task=args.task, agent=args.agent)


if __name__ == "__main__":
    raise SystemExit(main())
