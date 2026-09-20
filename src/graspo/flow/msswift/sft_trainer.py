"""msswift 后端的 SFT 训练器入口与工厂（决策 D2：SFT 双后端）。

**边界（一事一责，宪法 §1.1）**

- **本模块**：``(train_method="sft", backend="msswift")`` 这条路径的**入口与执行**——
  把 graspo 的 SFT 数据与超参交给 ms-swift 自己的 SFT 训练循环。
- **不负责**：GRPO/RL 与算法注入（``trainer.py``）、参数映射（``_config_mapping.py``）、
  数据集转换（``dataset.py``）。

**执行路线（决策 D6：Python API，进程内）**

``swift.pipelines.sft_main`` 是 ms-swift 的**库入口**——它自己的 ``swift sft`` CLI
就是 ``from swift.pipelines import sft_main; sft_main()``（见 ms-swift 4.5.3
``swift/cli/sft.py``）。本模块直接调用同一个函数并显式传入参数向量：
**没有子进程、没有 shell、不把 graspo YAML 喂给 ms-swift**。

**为什么 SFT 不需要重写训练器**：graspo 的算法注入（字符级标注 / token 级 advantage /
PPO-clip loss）是 **RL 专有**的；SFT 的语义是"在给定目标文本上做监督学习"，
graspo 与 ms-swift 在这一层没有分歧——因此 msswift 的 SFT 直接复用 ms-swift 训练循环，
只有**数据形态**（``dataset.py``，与 native 共用 ``build_sft_target_text``）与
**参数映射**（``_config_mapping.py``）由 graspo 提供。

**API 实测事实（T1 复验，ms-swift 4.5.3，2026-09-16）**

``swift.llm`` 子模块**不存在**（4.3.2 与 4.5.3 均实测缺失）→ 决策 D6 写作时的
``from swift.llm import ...`` 已失效；实际入口是 ``swift.pipelines``（``sft_main`` /
``rlhf_main``）与 ``swift.trainers.Trainer`` / ``TrainerFactory``。

**前置条件不是"伪造通过"**：ms-swift 未安装时本模块不静默降级、不返回假训练器
——它抛 ``RuntimeError`` 并指明缺什么。训练路径要么真跑，要么明确失败。
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

#: ms-swift 未安装时，统一的可操作提示（单一真相源，禁止在别处再写一份）。
MS_SWIFT_SFT_PREREQUISITE = (
    "backend='msswift' SFT requires the ms-swift package "
    "(pip install graspo[msswift]). Native SFT is fully available: "
    "set backend: native in the YAML."
)


def ms_sft_available() -> bool:
    """ms-swift 的 SFT 训练循环是否可用（判据 = ms-swift 是否可导入）。

    判据集中在这一个函数，测试与 CLI 用同一真相源判断，不各自探测。
    注意：它是**诊断/编排用**的查询，不是工厂的闸门——工厂照常返回可调用对象，
    真正的失败发生在执行时（失败信息精确可操作），而不是在注册表解析阶段
    抛异常导致调用方看不到分派结果。
    """
    try:
        import swift  # noqa: F401
    except ImportError:
        return False
    return True


def _require_ms_swift() -> type:
    """解析 ms-swift 的 SFT 训练器类（``swift.trainers.Trainer``）。

    Raises:
        RuntimeError: ms-swift 未安装 —— 附可操作指引（不是静默降级）。
    """
    try:
        from swift.trainers import Trainer as SwiftTrainer
    except ImportError as exc:
        raise RuntimeError(MS_SWIFT_SFT_PREREQUISITE) from exc
    return SwiftTrainer


class MsSwiftSftTrainer:
    """msswift 后端的 SFT 训练器入口（延迟解析 ms-swift，避免分派期重量级导入）。

    **接口契约**（与 native 侧 ``SFTTrainer`` 一致，``cli/train_worker.py`` 依赖）：

        trainer = factory(config, selection)   # 本类实例，构造不导入 torch/ms-swift
        trainer.train(smoke=bool)              # 执行；未接入时给出精确可操作错误

    构造时**不**导入 ms-swift / torch（宪法 §1.3：计算与设施分离），因此
    "解析注册表 → 构造 trainer" 在任何环境都可完成，失败只发生在 ``train()``。
    """

    def __init__(self, config: Any, selection: Any = None) -> None:
        self.config = config
        self.selection = selection

    def __repr__(self) -> str:  # pragma: no cover - 调试辅助
        return (
            f"<MsSwiftSftTrainer config.train_method={getattr(self.config, 'train_method', '?')!r}>"
        )

    def train(self, *, smoke: bool = False) -> None:
        """运行 ms-swift SFT 训练（Python API，进程内）。

        Args:
            smoke: 冒烟边界（跑 1 个 optimizer step 即停），与 native 侧语义一致。

        Raises:
            RuntimeError: ms-swift 未安装 —— 消息给出安装方式与 native 替代路径。
        """
        _require_ms_swift()  # 未安装 → RuntimeError(MS_SWIFT_SFT_PREREQUISITE)
        from swift.pipelines import sft_main

        from graspo.flow.msswift._config_mapping import (
            graspo_to_ms_swift_argv,
            validate_combinations,
        )
        from graspo.flow.msswift._rope_compat import rope_parameters_compatible
        from graspo.flow.msswift.dataset import prepare_ms_swift_dataset

        # 前置校验（E2b 缺陷 3/4 的防御性报错改进，**非能力修复**）：把上游的
        # 组合约束变成启动前可读的报错，而不是训练跑到一半的 ChildFailedError。
        validate_combinations(self.config)
        _warn_native_only_fields(self.config)
        output_dir = Path(str(self.config.training.output_dir))
        work_dir = output_dir / "msswift"
        dataset_path = prepare_ms_swift_dataset(self.config, stage="sft", work_dir=work_dir)

        extra_argv: list[str] = []
        if smoke:
            # 冒烟要拿到 optimizer step 与 checkpoint 两样产物证据，因此显式规定
            # "只跑 1 步、每步保存一次"——这是运行边界参数，不改训练语义（§10.1）。
            extra_argv = [
                "--max_steps",
                "1",
                "--save_strategy",
                "steps",
                "--save_steps",
                "1",
                "--save_total_limit",
                "1",
                "--logging_steps",
                "1",
            ]
        argv = graspo_to_ms_swift_argv(
            self.config,
            stage="sft",
            dataset_path=dataset_path,
            output_dir=str(output_dir),
            extra_argv=extra_argv,
        )
        logger.info("graspo msswift SFT argv: %s", " ".join(argv))
        # RoPE 键名适配（E2b 实测缺陷 1）：ms-swift 把旧格式 ``rope_scaling`` 写进模型
        # config，而 transformers 5.x 读新键 ``rope_parameters["rope_type"]``。适配在
        # ms-swift 的模型加载边界上**作用域内**完成，不改 ms-swift 源码（§1.2）。
        # 未配置 ``rope_scaling`` 时该上下文不装任何补丁。
        with rope_parameters_compatible(self.config.msswift.rope_scaling):
            sft_main(argv)


def _warn_native_only_fields(config: Any) -> None:
    """透明退路（§3.2）：graspo 有、msswift 后端不映射的字段逐条 WARNING。"""
    from graspo.flow.msswift._config_mapping import native_only_field_notes, native_only_fields

    notes = native_only_field_notes()
    for field in native_only_fields(config):
        logger.warning("graspo: %s is not mapped by the msswift backend — %s", field, notes[field])


def create_msswift_sft_trainer(config: Any, selection: Any = None) -> MsSwiftSftTrainer:
    """``graspo.sft_backends`` 注册表中 ``msswift`` 项的工厂。

    返回 ``MsSwiftSftTrainer`` 实例，满足与 native 侧相同的接口契约
    ``factory(config, selection) -> 含 train(smoke) 的训练器``。

    Args:
        config: GraspoConfig 实例。
        selection: BackendSelection 实例（可选，供 E2 做后端内部分派）。

    Returns:
        ``MsSwiftSftTrainer``；构造不触发 torch / ms-swift 导入。
    """
    return MsSwiftSftTrainer(config, selection)
