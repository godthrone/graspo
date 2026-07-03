from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class Experience:
    sequences: Any
    old_log_probs: Any
    advantages: Any
    attention_mask: Any
    action_mask: Any
    rewards: Any
    metadata: dict[str, Any] | None = None
    decision: str | None = None  # "trainable_max_correct" | "trainable_not_correct" | …


class ReplayBuffer:
    def __init__(self, limit: int = 0) -> None:
        self.limit = limit
        self.items: list[Experience] = []

    def append_many(self, items: list[Experience]) -> None:
        self.items.extend(items)
        if self.limit > 0 and len(self.items) > self.limit:
            self.items = self.items[-self.limit :]

    def clear(self) -> None:
        self.items.clear()

    def take(self, count: int) -> list[Experience]:
        return self.items[:count]

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> Experience:
        return self.items[index]
