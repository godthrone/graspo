"""msswift 后端的 **OPD（on-policy 蒸馏）** 训练器入口与工厂。

**边界（一事一责，宪法 §1.1）**

- **本模块**：``(train_method="opd", backend="msswift")`` 这条路径的**入口与执行**——
  把 graspo 的纯提示词数据与教师/学生配置交给 ms-swift 的 **GKD** 训练循环。
- **不负责**：CPT（``cpt_trainer.py``）、SFT（``sft_trainer.py``）、
  graspo RL 的算法注入（``trainer.py``）、参数映射（``_config_mapping.py``）、
  数据集转换（``dataset.py``）、**路由**（路由归属 ``core.discovery``，§1.4）。

**执行路线（决策 D6：Python API，进程内）**

``swift.pipelines.rlhf_main`` 是 ms-swift 的库入口；``--rlhf_type gkd`` 由它的
``TrainerFactory`` 解析成 ``swift.rlhf_trainers.GKDTrainer``（ms-swift 4.5.3
``swift/trainers/trainer_factory.py:27/44``）。**没有子进程、没有 shell。**

**为什么走 GKD 而不是 OPD-RL**（判据：ms-swift 4.5.3 ``docs/source_en/Instruction/Distillation.md`` §3）

ms-swift 提供两条在线蒸馏路径：

- **Path A ``--rlhf_type gkd``**：教师散度**直接作为 loss**（全词表或 top-k），
  教师是同一训练进程里的**独立冻结模型**（``--teacher_model``）；
- **Path B ``--rlhf_type grpo`` + teacher**：教师 log-ratio 注入 GRPO 的 per-token
  advantage，需要 ``--teacher_kl_coef`` 一类系数，且优势估计的方差更高。

graspo 选 **Path A**：它是"学生现场采样 → 教师现场打分 → 逐 token 稠密信号"这条
on-policy 蒸馏语义的**最短通路**，且**不需要任何教师 logprob 数据契约**——教师
logits 在进程内前向得到。这意味着本通道**不消费** ``_ard_contract.py`` 的
ARD→graspo 转换（那份契约是数据集侧的），也不产生"教师 logprob 数据列"。
**缺口登记**：若将来改走"外部教师服务"（``--teacher_model_server``）或 Path B，
才需要一份教师 logprob 的边界契约；本期不发明它（§6.1 简单优先 + §3.3 不预支）。

**教师/学生对**：用户 2026-09-18 拍板 教师 = ``Qwen3.8-27B``、学生 =
``Qwen3.5-9B``；教师路径来自**后端中立**的 ``distill.teacher_model_path``
（§1.4 单一真相源），学生就是既有的 ``model.model_path``——不发明第二个字段。

**已知边界**：native 后端**没有** OPD 实现（``docs/capability-matrix.md`` §4
「OPD · native」= ``⛔ 不支持``），配置层对 ``train_method: opd`` +
``backend: native`` **fail-closed**。

**前置条件不是"伪造通过"**：ms-swift 未安装时抛 ``RuntimeError`` 并指明缺什么。
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

#: ms-swift 未安装时，统一的可操作提示（单一真相源，禁止在别处再写一份）。
MS_SWIFT_OPD_PREREQUISITE = (
    "backend='msswift' OPD (on-policy distillation, GKD) requires the ms-swift package "
    "(pip install graspo[msswift]). OPD has no native implementation "
    "(docs/capability-matrix.md §4: OPD · native is unsupported)."
)


def ms_opd_available() -> bool:
    """ms-swift 的 RLHF/GKD 入口是否可用（判据 = ms-swift 是否可导入）。

    与 ``sft_trainer.ms_sft_available`` 同口径：**诊断用**查询，不是工厂的闸门。
    """
    try:
        import swift  # noqa: F401
    except ImportError:
        return False
    return True


def _require_ms_swift() -> Any:
    """解析 ms-swift 的 RLHF 入口（``swift.pipelines.rlhf_main``）。

    Raises:
        RuntimeError: ms-swift 未安装 —— 附可操作指引（不是静默降级）。
    """
    try:
        from swift.pipelines import rlhf_main
    except ImportError as exc:
        raise RuntimeError(MS_SWIFT_OPD_PREREQUISITE) from exc
    return rlhf_main


class MsSwiftOpdTrainer:
    """msswift 后端的 OPD(GKD) 训练器入口（延迟解析 ms-swift，避免分派期重量级导入）。

    **接口契约**（与其它四条通道一致，``cli/train_worker.py`` 依赖）::

        trainer = factory(config, selection)   # 本类实例，构造不导入 torch/ms-swift
        trainer.train(smoke=bool)              # 执行；未接入时给出精确可操作错误
    """

    def __init__(self, config: Any, selection: Any = None) -> None:
        self.config = config
        self.selection = selection

    def __repr__(self) -> str:  # pragma: no cover - 调试辅助
        return (
            f"<MsSwiftOpdTrainer config.train_method={getattr(self.config, 'train_method', '?')!r}>"
        )

    def train(self, *, smoke: bool = False) -> None:
        """运行 ms-swift GKD 训练（Python API，进程内）。

        Args:
            smoke: 冒烟边界（跑 1 个 optimizer step 即停），与其它通道语义一致。

        Raises:
            RuntimeError: ms-swift 未安装 —— 消息给出安装方式与"OPD 无 native 替代"。
        """
        rlhf_main = _require_ms_swift()

        from graspo.flow.msswift._config_mapping import (
            graspo_to_ms_swift_argv,
            native_only_field_notes,
            native_only_fields,
            validate_combinations,
        )
        from graspo.flow.msswift._rollout_seed import (
            assert_rollout_seed_applied,
            rollout_seed_deterministic,
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
        dataset_path = prepare_ms_swift_dataset(self.config, stage="opd", work_dir=work_dir)

        extra_argv: list[str] = []
        if smoke:
            # 与 RL 通道同一约定：冒烟 = 1 个优化轮次 + 每步保存，拿到两样产物证据。
            extra_argv = [
                "--max_steps", "1",
                "--save_strategy", "steps",
                "--save_steps", "1",
                "--save_total_limit", "1",
                "--logging_steps", "1",
            ]
        argv = graspo_to_ms_swift_argv(
            self.config,
            stage="opd",
            dataset_path=dataset_path,
            output_dir=str(output_dir),
            extra_argv=extra_argv,
        )
        logger.info("graspo msswift OPD(GKD) argv: %s", " ".join(argv))
        # **不注册 graspo 奖励、不传 `--reward_funcs`**：GKD 的监督信号来自教师
        # 现场 logits（Path A），奖励函数在这条路径上没有消费者。传一个没人消费的
        # 奖励等于制造"看起来生效、实际被忽略"的假配置（§1.4 / §7.2）。
        #
        # **rollout 播种**（可复现性措施，不改变算法语义）：本通道 on-policy 的学生
        # 现场采样在 ``use_vllm=false`` 下走 ``TransformersEngine``，它从**全局
        # torch RNG** 取随机数且不接 ``RequestConfig.seed``；ms-swift 只在 trainer
        # ``__init__`` 播一次种（``grpo_trainer.py:143``），rollout 前不重播 ⇒ 同 config
        # 同 seed 的两跑首步监督信号可以不同（T046 的 A4 就是被这个打掉的）。
        # 这里在**每次 rollout 生成之前**把全局 RNG 钉到 ``training.seed``——只钉
        # 随机性起点，不动温度/top_p/top_k/lmbda/采样分布，也不强制 greedy。
        with (
            rope_parameters_compatible(self.config.msswift.rope_scaling),
            rollout_seed_deterministic(self.config) as seed_ledger,
        ):
            rlhf_main(argv)
        # fail-closed：装了补丁却一次都没触发 ⇒ 这次运行的可复现性并未被保证，
        # 不能当成功交付（§2.3）。
        assert_rollout_seed_applied(seed_ledger)


def create_msswift_opd_trainer(config: Any, selection: Any = None) -> MsSwiftOpdTrainer:
    """``graspo.opd_backends`` 注册表中 ``msswift`` 项的工厂。

    Returns:
        ``MsSwiftOpdTrainer``；构造不触发 ms-swift / torch 导入。
    """
    return MsSwiftOpdTrainer(config, selection)
