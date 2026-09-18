"""msswift 后端的 **CPT（继续预训练）** 训练器入口与工厂。

**边界（一事一责，宪法 §1.1）**

- **本模块**：``(train_method="cpt", backend="msswift")`` 这条路径的**入口与执行**——
  把 graspo 的纯文本续训数据与超参交给 ms-swift 自己的**预训练**训练循环。
- **不负责**：SFT/GRPO 与算法注入（``sft_trainer.py`` / ``trainer.py``）、
  参数映射（``_config_mapping.py``）、数据集转换（``dataset.py``）、
  **路由**（路由归属 ``core.discovery`` 的路由表，§1.4）。

**执行路线（决策 D6：Python API，进程内）**

``swift.pipelines.pretrain_main`` 是 ms-swift 的**库入口**——它自己的 ``swift pt``
CLI 就是 ``from swift.pipelines import pretrain_main; pretrain_main()``（见 ms-swift
4.5.3 ``swift/cli/pt.py``）。本模块直接调用同一个函数并显式传入参数向量：
**没有子进程、没有 shell、不把 graspo YAML 喂给 ms-swift**。

**为什么 CPT 不需要重写训练器**：CPT 的语义 = "在纯文本上做 next-token 续训、
每个 token 都计损、不套对话模板"，ms-swift 的 ``swift pt`` 就是这个语义的既有
实现（``swift pt`` ≡ ``swift sft --use_chat_template false --loss_scale all``，
ms-swift 4.5.3 ``docs/source_en/Instruction/Pre-training-and-Fine-tuning.md``）。
graspo 提供的是**数据形态**（``dataset.py::build_cpt_rows``）与**参数映射**
（``_config_mapping.py`` 的 ``stage="cpt"`` 分支），算法本体不重复实现（§1.3）。

**已知边界（如实声明，不假装支持）**：native 后端**没有** CPT 实现
（``docs/capability-matrix.md`` §4「CPT · native」= ``⛔ 不支持``），因此这条通道
只有 ms-swift 一个实现；配置层对 ``train_method: cpt`` + ``backend: native``
**fail-closed**（``core/schema.py::validate_train_method_combination``）。

**前置条件不是"伪造通过"**：ms-swift 未安装时本模块不静默降级、不返回假训练器
——它抛 ``RuntimeError`` 并指明缺什么。训练路径要么真跑，要么明确失败。
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

#: ms-swift 未安装时，统一的可操作提示（单一真相源，禁止在别处再写一份）。
MS_SWIFT_CPT_PREREQUISITE = (
    "backend='msswift' CPT (continued pre-training) requires the ms-swift package "
    "(pip install graspo[msswift]). CPT has no native implementation "
    "(docs/capability-matrix.md §4: CPT · native is unsupported)."
)


def ms_cpt_available() -> bool:
    """ms-swift 的预训练入口是否可用（判据 = ms-swift 是否可导入）。

    与 ``sft_trainer.ms_sft_available`` 同口径：这是**诊断用**查询，不是工厂的闸门
    ——工厂照常返回可调用对象，真正的失败发生在执行时（失败信息精确可操作），
    而不是在注册表解析阶段抛异常导致调用方看不到分派结果。
    """
    try:
        import swift  # noqa: F401
    except ImportError:
        return False
    return True


def _require_ms_swift() -> Any:
    """解析 ms-swift 的预训练入口（``swift.pipelines.pretrain_main``）。

    Raises:
        RuntimeError: ms-swift 未安装 —— 附可操作指引（不是静默降级）。
    """
    try:
        from swift.pipelines import pretrain_main
    except ImportError as exc:
        raise RuntimeError(MS_SWIFT_CPT_PREREQUISITE) from exc
    return pretrain_main


class MsSwiftCptTrainer:
    """msswift 后端的 CPT 训练器入口（延迟解析 ms-swift，避免分派期重量级导入）。

    **接口契约**（与 native SFT / msswift SFT+RL+OPD 一致，``cli/train_worker.py`` 依赖）::

        trainer = factory(config, selection)   # 本类实例，构造不导入 torch/ms-swift
        trainer.train(smoke=bool)              # 执行；未接入时给出精确可操作错误
    """

    def __init__(self, config: Any, selection: Any = None) -> None:
        self.config = config
        self.selection = selection

    def __repr__(self) -> str:  # pragma: no cover - 调试辅助
        return (
            f"<MsSwiftCptTrainer config.train_method={getattr(self.config, 'train_method', '?')!r}>"
        )

    def train(self, *, smoke: bool = False) -> None:
        """运行 ms-swift 预训练（Python API，进程内）。

        Args:
            smoke: 冒烟边界（跑 1 个 optimizer step 即停），与其它通道语义一致。

        Raises:
            RuntimeError: ms-swift 未安装 —— 消息给出安装方式与"CPT 无 native 替代"。
        """
        pretrain_main = _require_ms_swift()

        from graspo.flow.msswift._config_mapping import (
            graspo_to_ms_swift_argv,
            native_only_field_notes,
            native_only_fields,
            validate_combinations,
        )
        from graspo.flow.msswift._rope_compat import rope_parameters_compatible
        from graspo.flow.msswift.dataset import prepare_ms_swift_dataset

        validate_combinations(self.config)
        notes = native_only_field_notes()
        for field in native_only_fields(self.config):
            logger.warning(
                "graspo: %s is not mapped by the msswift backend — %s", field, notes[field]
            )

        output_dir = Path(str(self.config.training.output_dir))
        work_dir = output_dir / "msswift"
        dataset_path = prepare_ms_swift_dataset(self.config, stage="cpt", work_dir=work_dir)

        extra_argv: list[str] = []
        if smoke:
            # 冒烟要拿到 optimizer step 与 checkpoint 两样产物证据，因此显式规定
            # "只跑 1 步、每步保存一次"——这是运行边界参数，不改训练语义（§10.1）。
            extra_argv = [
                "--max_steps", "1",
                "--save_strategy", "steps",
                "--save_steps", "1",
                "--save_total_limit", "1",
                "--logging_steps", "1",
            ]
        argv = graspo_to_ms_swift_argv(
            self.config,
            stage="cpt",
            dataset_path=dataset_path,
            output_dir=str(output_dir),
            extra_argv=extra_argv,
        )
        logger.info("graspo msswift CPT argv: %s", " ".join(argv))
        # RoPE 键名适配（E2b 实测缺陷 1）与 SFT 通道同一处理：在 ms-swift 的模型加载
        # 边界上作用域内完成，不改 ms-swift 源码（§1.2）。未配置 rope_scaling 时不装补丁。
        with rope_parameters_compatible(self.config.msswift.rope_scaling):
            pretrain_main(argv)


def create_msswift_cpt_trainer(config: Any, selection: Any = None) -> MsSwiftCptTrainer:
    """``graspo.cpt_backends`` 注册表中 ``msswift`` 项的工厂。

    Returns:
        ``MsSwiftCptTrainer``；构造不触发 ms-swift / torch 导入。
    """
    return MsSwiftCptTrainer(config, selection)
