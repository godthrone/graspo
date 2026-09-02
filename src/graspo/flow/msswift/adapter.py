"""Graspo JSONL 数据集格式 → ms-swift dataset 格式的适配器。

Graspo 使用 JSONL 格式存储训练数据（每行一个 Sample），
ms-swift 使用 HuggingFace datasets 或自定义 dataset 格式。
本模块提供双向适配，确保数据格式兼容。

当前状态：骨架实现，标记 TODO 待后续完善。
"""

from __future__ import annotations

from typing import Any


class GraspoToMsSwiftAdapter:
    """将 Graspo JSONL 样本转换为 ms-swift 兼容的 dataset 格式。

    TODO: 实现完整的字段映射和格式转换：
    - messages → ms-swift conversation format
    - targets → ms-swift reward/verification format
    - tools → ms-swift tool_call format
    - media → ms-swift multimodal format
    """

    def __init__(self) -> None:
        pass

    def convert_sample(self, sample: dict[str, Any]) -> dict[str, Any]:
        """将单条 Graspo 样本转换为 ms-swift 格式。

        TODO: 实现字段映射逻辑。
        """
        raise NotImplementedError("GraspoToMsSwiftAdapter.convert_sample is not yet implemented")

    def convert_dataset(self, samples: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """批量转换 Graspo 样本。

        TODO: 实现批量转换逻辑（可考虑流式处理）。
        """
        return [self.convert_sample(s) for s in samples]


class MsSwiftToGraspoAdapter:
    """将 ms-swift dataset 格式转换回 Graspo JSONL 格式。

    TODO: 实现逆向转换。
    """

    def __init__(self) -> None:
        pass

    def convert_sample(self, sample: dict[str, Any]) -> dict[str, Any]:
        """将单条 ms-swift 样本转换为 Graspo 格式。

        TODO: 实现字段映射逻辑。
        """
        raise NotImplementedError("MsSwiftToGraspoAdapter.convert_sample is not yet implemented")