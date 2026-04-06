from __future__ import annotations

from env.models import TaskConfig

TASKS: dict[str, TaskConfig] = {
    "easy": TaskConfig(
        task_id="easy",
        difficulty="easy",
        max_steps=12,
        visible_windows=["all", "top", "middle", "bottom"],
        corruption_level=0.02,
        ranking_noise=0.02,
        instruction="Extract company, date, address, and total from the receipt.",
        target_fields=["company", "date", "address", "total"],
        requires_line_items=False,
    ),
    "medium": TaskConfig(
        task_id="medium",
        difficulty="medium",
        max_steps=10,
        visible_windows=["top", "middle", "bottom"],
        corruption_level=0.12,
        ranking_noise=0.12,
        instruction="Extract company, date, subtotal, tax, and total. Check whether subtotal plus tax matches total.",
        target_fields=["company", "date", "subtotal", "tax", "total"],
        requires_line_items=False,
    ),
    "hard": TaskConfig(
        task_id="hard",
        difficulty="hard",
        max_steps=8,
        visible_windows=["top", "bottom"],
        corruption_level=0.25,
        ranking_noise=0.25,
        instruction=(
            "Extract company, date, subtotal, tax, total, and line items. "
            "Check whether line items reconcile to subtotal and subtotal plus tax reconciles to total."
        ),
        target_fields=["company", "date", "subtotal", "tax", "total"],
        requires_line_items=True,
    ),
}


def get_task(task_name: str | None) -> TaskConfig:
    if task_name is None:
        return TASKS["easy"]
    return TASKS[task_name]
