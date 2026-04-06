from __future__ import annotations

from typing import Protocol

from env.environment import ReceiptExtractionEnv
from env.models import ReceiptAction


class Agent(Protocol):
    name: str

    def select_action(self, env: ReceiptExtractionEnv) -> ReceiptAction: ...
