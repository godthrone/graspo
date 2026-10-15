"""F2：标准 PEFT/LoRA adapter 目录 → ``merged-hf``（**P3 实现；本包先给声明式处置**）。

**为什么现在就有本文件**：``CheckpointForm.PEFT_ADAPTER`` 自 P1 起就是形态词汇表的一员，
而 T12 要求"每个枚举成员都必须有已登记的处置"。真实导出器（搬自
``eval/merged_export.py`` 的 ``merge_peft_checkpoint`` / 多模态守卫）属 **P3**；
在此之前用 :class:`~graspo.export.schema.UnsupportedForm` **显式声明"已识别但暂不支持"**，
⛔ 不退化成"未知形态"（§17.7：状态必须显式）。

P3 落地时：把本文件的 ``PEFT_ADAPTER_UNSUPPORTED`` 替换为真实的 ``PeftAdapterExporter``，
其余登记（pyproject + registry 回退表）**不需要改动**（这正是"新增实现只需一行注册"）。
"""

from __future__ import annotations

from pathlib import Path

from graspo.export.schema import CheckpointForm, UnsupportedForm

_ADAPTER_CONFIG = "adapter_config.json"


def match_peft_adapter(root: Path) -> tuple[bool, tuple[str, ...]]:
    """检测器：目录根有 ``adapter_config.json``（标准 HF PEFT/LoRA adapter 的标志）。"""
    config = root / _ADAPTER_CONFIG
    if not config.is_file():
        return False, ()
    return True, (f"found {_ADAPTER_CONFIG}",)


PEFT_ADAPTER_UNSUPPORTED = UnsupportedForm(
    form=CheckpointForm.PEFT_ADAPTER,
    reason=(
        "PEFT adapter → merged-hf 的合并实现（含多模态 loader 类别判定与命名空间守卫）"
        "按设计属 P3 交付；在此之前 fail-closed，⛔ 不产出未经合并校验的产物"
    ),
    detector=match_peft_adapter,
)


__all__ = ["PEFT_ADAPTER_UNSUPPORTED", "match_peft_adapter"]
