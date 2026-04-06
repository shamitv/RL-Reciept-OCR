from __future__ import annotations

from env.models import TaskConfig

TASKS: dict[str, TaskConfig] = {
    "easy": TaskConfig(task_id="easy", difficulty="easy", max_steps=12, visible_windows=["all", "top", "middle", "bottom"], corruption_level=0.02, ranking_noise=0.02),
    "medium": TaskConfig(task_id="medium", difficulty="medium", max_steps=10, visible_windows=["top", "middle", "bottom"], corruption_level=0.12, ranking_noise=0.12),
    "hard": TaskConfig(task_id="hard", difficulty="hard", max_steps=8, visible_windows=["top", "bottom"], corruption_level=0.25, ranking_noise=0.25),
}


def get_task(task_name: str | None) -> TaskConfig:
    if task_name is None:
        return TASKS["easy"]
    return TASKS[task_name]
