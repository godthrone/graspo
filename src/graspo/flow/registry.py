"""flow 侧的开发模式回退注册表（★ X4：core 不硬编码 flow 路径）。

- 生产真相源 = `pyproject.toml` 的 entry_points；
- 本表只是"包未 pip install"时的开发回退；
- **入口**：`graspo.flow.__init__` 导入本模块即完成注册（幂等）。

⚠ 目录搬迁（P2：`flow/backend/native/trainer` → `flow/backend/native/trainer` 等）时**必须同步本表**
——这是重构期最容易漏的接线（§1.4 单一真相源）。
"""

from __future__ import annotations

from graspo.core.discovery import register_dev_fallbacks

register_dev_fallbacks(
    "graspo.adapters",
    {
        "qwen3": "graspo.flow.backend.native.models.qwen3.model_adapter:Qwen3ModelAdapter",
        "qwen35_36": "graspo.flow.backend.native.models.qwen35_36.model_adapter:Qwen35ModelAdapter",
    },
)
register_dev_fallbacks(
    "graspo.backends",
    {
        "native": "graspo.flow.backend.native:create_native_backend",
        "msswift": "graspo.flow.backend.msswift:create_msswift_backend",
    },
)
# SFT 后端注册表：与 RL（graspo.backends）分开登记，因为两者的训练器形状不同
# （SFT 消费「已 tokenize 的样本」，RL 消费 rollout group）。新增后端 SFT 支持 =
# 本表加一行 + pyproject entry_points 加一行，现有选择逻辑 0 行改动（§1.2）。
register_dev_fallbacks(
    "graspo.sft_backends",
    {
        "native": "graspo.flow.backend.native.trainer.sft_trainer:create_native_sft_trainer",
        "msswift": "graspo.flow.backend.msswift.sft_trainer:create_msswift_sft_trainer",
    },
)
# CPT（继续预训练）：走 ms-swift 的预训练管线；**只有 ms-swift 实现**。
register_dev_fallbacks(
    "graspo.cpt_backends",
    {
        "msswift": "graspo.flow.backend.msswift.cpt_trainer:create_msswift_cpt_trainer",
    },
)
# OPD（on-policy 蒸馏）：走 ms-swift 的 GKD 路径；**只有 ms-swift 实现**。
register_dev_fallbacks(
    "graspo.opd_backends",
    {
        "msswift": "graspo.flow.backend.msswift.opd_trainer:create_msswift_opd_trainer",
    },
)
# ★ native 后端的**默认值**（X4：`core/schema.py` 不得硬编码 flow 路径）
register_dev_fallbacks(
    "graspo.native_defaults",
    {
        "native": "graspo.flow.backend.native.config:native_defaults",
    },
)

__all__: list[str] = []
