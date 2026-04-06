from __future__ import annotations

from statistics import mean

from env.models import OCRRegion, ReceiptObservation, TaskConfig

FIELD_ORDER = ("company", "date", "address", "total")
DIFFICULTY_ORDER = ("easy", "medium", "hard")
OBSERVATION_DIM = 26
ENCODER_VERSION = "receipt-obs-v1"


def _require_torch():
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - exercised by runtime error paths
        raise ImportError(
            "torch is required for --agent ppo. Install with: pip install -e '.[ppo]'"
        ) from exc
    return torch


def _region_bucket(region: OCRRegion) -> str:
    center_y = (region.bbox[1] + region.bbox[3]) / 2.0
    if center_y < 100:
        return "top"
    if center_y < 200:
        return "middle"
    return "bottom"


def _bucket_fraction(regions: list[OCRRegion], bucket: str) -> float:
    if not regions:
        return 0.0
    count = sum(1 for region in regions if _region_bucket(region) == bucket)
    return count / float(len(regions))


def _feedback_flag(messages: list[str], needles: tuple[str, ...]) -> float:
    lowered = [message.lower() for message in messages]
    return 1.0 if any(any(needle in message for needle in needles) for message in lowered) else 0.0


def encode_observation_values(obs: ReceiptObservation, task: TaskConfig) -> list[float]:
    visible_regions = list(obs.visible_regions)
    max_steps = max(1, task.max_steps)
    confidence_values = [region.confidence if region.confidence is not None else 1.0 for region in visible_regions]

    values: list[float] = []
    values.extend(1.0 if obs.difficulty == difficulty else 0.0 for difficulty in DIFFICULTY_ORDER)
    values.append(obs.step_index / float(max_steps))
    values.append(obs.remaining_budget / float(max_steps))
    values.append(float(len(visible_regions)))
    values.append(mean(confidence_values) if confidence_values else 0.0)
    values.append(_bucket_fraction(visible_regions, "top"))
    values.append(_bucket_fraction(visible_regions, "middle"))
    values.append(_bucket_fraction(visible_regions, "bottom"))
    for field in FIELD_ORDER:
        values.append(float(len(obs.candidate_lists.get(field, []))))
    for field in FIELD_ORDER:
        candidates = obs.candidate_lists.get(field, [])
        values.append(max((candidate.heuristic_score for candidate in candidates), default=0.0))
    for field in FIELD_ORDER:
        values.append(1.0 if getattr(obs.current_draft, field) else 0.0)
    values.append(float(len(obs.validation_feedback)))
    values.append(_feedback_flag(obs.validation_feedback, ("valid", "matches")))
    values.append(_feedback_flag(obs.validation_feedback, ("invalid", "weakly")))
    values.append(float(obs.terminal_allowed))
    return values


def encode_observation(obs: ReceiptObservation, task: TaskConfig):
    torch = _require_torch()
    values = encode_observation_values(obs, task)
    return torch.tensor(values, dtype=torch.float32)
