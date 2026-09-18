"""msswift 后端：把 graspo ripple 算法注入 ms-swift 的训练循环。

分层（各文件一事一责，宪法 §1.1）：

- ``trainer.py``      RL(GRPO) 训练器与 ``graspo.backends`` 工厂（算法注入点）
- ``sft_trainer.py``  SFT 入口与 ``graspo.sft_backends`` 工厂（决策 D2）
- ``cpt_trainer.py``  CPT（继续预训练）入口与 ``graspo.cpt_backends`` 工厂
- ``opd_trainer.py``  OPD（on-policy 蒸馏 / GKD）入口与 ``graspo.opd_backends`` 工厂
- ``dataset.py``      graspo/ARD JSONL → ms-swift 数据集（SFT/GRPO/CPT/OPD 四形态）
- ``_config_mapping.py``  graspo 配置 → ms-swift 参数（纯计算）
- ``reward.py``       graspo 奖励在 ms-swift 奖励通道上的适配器
- ``adapter.py`` / ``_ard_contract.py``  ARD ↔ graspo 数据契约（决策 D4）

公开 API 用 PEP 562 惰性导出（§8.6）：只 import ``graspo.flow.msswift._config_mapping``
这类子模块时，**不会**连带导入 torch / ms-swift，使纯映射与契约测试保持轻量。
"""

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - 仅供类型检查
    from graspo.flow.msswift.cpt_trainer import (
        MsSwiftCptTrainer,
        create_msswift_cpt_trainer,
    )
    from graspo.flow.msswift.opd_trainer import (
        MsSwiftOpdTrainer,
        create_msswift_opd_trainer,
    )
    from graspo.flow.msswift.sft_trainer import MsSwiftSftTrainer, create_msswift_sft_trainer
    from graspo.flow.msswift.trainer import MsSwiftRlTrainer, create_msswift_trainer

__all__ = [
    "MsSwiftCptTrainer",
    "MsSwiftOpdTrainer",
    "MsSwiftRlTrainer",
    "MsSwiftSftTrainer",
    "create_msswift_cpt_trainer",
    "create_msswift_opd_trainer",
    "create_msswift_sft_trainer",
    "create_msswift_trainer",
]

_EXPORTS = {
    "MsSwiftRlTrainer": "graspo.flow.msswift.trainer",
    "create_msswift_trainer": "graspo.flow.msswift.trainer",
    "MsSwiftSftTrainer": "graspo.flow.msswift.sft_trainer",
    "create_msswift_sft_trainer": "graspo.flow.msswift.sft_trainer",
    "MsSwiftCptTrainer": "graspo.flow.msswift.cpt_trainer",
    "create_msswift_cpt_trainer": "graspo.flow.msswift.cpt_trainer",
    "MsSwiftOpdTrainer": "graspo.flow.msswift.opd_trainer",
    "create_msswift_opd_trainer": "graspo.flow.msswift.opd_trainer",
}


def __getattr__(name: str) -> Any:
    """惰性导出公开 API（PEP 562）：只有真正取用时才导入对应的重模块。"""
    module_name = _EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    return getattr(importlib.import_module(module_name), name)
