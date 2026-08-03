"""经验回放缓冲：Experience 数据容器 + ReplayBuffer（算法层数据结构）。"""

from typing import Any

from pydantic import BaseModel, ConfigDict


class Experience(BaseModel):
    """单条 rollout 经验：序列、旧 log-prob、advantage、reward 等。

    Tensor 字段使用 ``arbitrary_types_allowed=True``，因为 pydantic 不原生校验
    torch.Tensor。编码到 CPU 后传入，确保跨进程序列化安全。
    """

    model_config = ConfigDict(extra="forbid", frozen=True, arbitrary_types_allowed=True)

    sequences: Any
    old_log_probs: Any
    advantages: Any
    attention_mask: Any
    action_mask: Any
    rewards: Any
    metadata: dict[str, Any] | None = None


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
