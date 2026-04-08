from __future__ import annotations

from random import Random


def make_rng(seed: int | None) -> Random:
    return Random(seed if seed is not None else 0)


def clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def strict_unit_interval(value: float, epsilon: float = 0.001) -> float:
    return max(epsilon, min(1.0 - epsilon, value))
