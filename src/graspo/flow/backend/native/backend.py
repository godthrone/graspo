"""flow.backend.native.backend — ``NativeBackend``：native 后端的 :class:`RLBackend` 实现。

**组合结构**（backend-interface.md §2.3）：``NativeBackend`` 继承 :class:`RLBackend`，
**组合** native 运行时 ``GraspoFlowRuntime`` 与既有的成熟训练循环 ``GraspoFlowTrainer``。

**行为等价声明（P3）**：本后端沿用既有循环作为执行路径（``_run`` 显式委托），
因为该循环承载 resume / retry / smoke / 确定性 / 检查点等大量既有语义，在**没有对拍
证据**的前提下重写成钩子骨架会改变行为。9 项能力的钩子按 ABC 契约**逐项实现/登记**
（见 :meth:`NativeBackend.capability_matrix` 与 ``phase-3-report.md`` 的能力矩阵）；
其中分片（能力 3）与优化步进入（能力 7）将在 **P4（β1b 均衡分配）** 接线。

生命周期由既有循环内部完成，故 ``_setup`` / ``_close`` 为**显式空实现**（不是遗漏）。
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from graspo.flow.backend.base import BackendCapability, RLBackend


class NativeBackend(RLBackend):
    """native 后端（TP/DP/PP/SP 自研运行时 + 自研训练循环）。"""

    name = "native"
    capabilities = frozenset(BackendCapability)  # 9 项全声明（执行路径见能力矩阵）

    def __init__(self, config: Any, selection: Any | None = None, *, trainer: Any | None = None) -> None:
        super().__init__(config, selection)
        # 组合：既有成熟循环（构造廉价，不触发 GPU / torch.distributed 初始化）
        self._trainer = trainer if trainer is not None else self._build_trainer()

    def _build_trainer(self) -> Any:
        from graspo.flow.backend.native.trainer import GraspoFlowTrainer

        return GraspoFlowTrainer(self.config, selection=self.selection)

    # ── 组合对象的读取口（供 P4/P5 与外部诊断使用；不是新的真相源）──────────
    @property
    def trainer(self) -> Any:
        return self._trainer

    @property
    def runtime(self) -> Any:
        return self._trainer.runtime

    # ── 能力 1：生命周期 ─────────────────────────────────────────────────────
    def validate(self) -> None:
        """无副作用前置检查（**不代替**既有循环内部的 ``runtime.validate()``）。"""
        if self.config is None:
            raise ValueError("NativeBackend 需要 config")

    def _setup(self) -> None:
        """显式空实现：设备/模型/进程组的建立在 ``_run``（既有循环）内部完成。"""

    def _close(self) -> None:
        """显式空实现：进程组销毁在 ``_run``（既有循环）的 finally 分支完成。"""

    # ── 训练主体（★ 行为等价的唯一执行路径）─────────────────────────────────
    def _run(self, *, smoke: bool = False) -> None:
        """委托既有 ``GraspoFlowTrainer.train()`` —— 与 P2 及以前逐字同路径。"""
        self._trainer.train(smoke=smoke)

    # ── 能力 3：分片（★ 整组不拆；组内 rollout 可切 G//m）─────────────────────
    def _step_batches(self, epoch: Any) -> Iterator[Any]:
        """能力 3：按 β1b **整组均衡分配**产出**本卡份额**。

        :param epoch: 可迭代的**全局组键**序列（分片前的口径）。

        ★ **两段式语义（2026-10-16 主席裁定 rollout 分卡后的真语义）**：

        1. **条目维（β1b，本方法实现）**：分配单位是**组**，整组不拆、全局组集合不变、
           各分片组数差 ≤ 1（:func:`...assignment.assert_balanced_assignment` 机核）。
           默认模式与既有 ``[dp_rank::dp_size]`` **逐位等价** ⇒ 本钩子是"把既有分片
           收编到契约面"，而不是改变分布。
        2. **组内 rollout 维（U-C，训练循环实现）**：切片模式下**不做条目分片**
           （各卡持**全部条目**），只把组内 rollout 槽位切成 `G//m`
           （:func:`...assignment.rollout_slice`）。⇒ **交给算法层之前必须装配**：
           组的 G 条要已从**同一 dp 组**的各成员处拼齐
           （:func:`graspo.flow.adapter.contract.assert_assembled_group_complete`），
           拼不齐 = **显式失败**（⛔ 不允许"算法在部分组上做决策"）。

        ⛔ 两段**不同维度**：本方法管"组整体落在哪个分片"，切片管"组内 rollout 由哪张卡
        生成/训练"。**I1（G 条齐）一个字都没放宽** —— 变的只是"它在哪一层被断言"。

        ★ **本钩子只实现第 1 段**（条目维 β1b；这是 P4 收编的契约面）。
        ★ 第 2 段（组内 rollout 切片）由**训练循环**实现（``trainer.py`` 的 epoch 循环
        ＋ ``rollout.py`` 的决策前会合）—— 本钩子**不改**它的行为：
        ``_step_batches`` 不是训练取数路径（``NativeBackend._run`` 直接委托既有
        ``GraspoFlowTrainer.train()``，见 ``_run`` 的"行为等价"注释），
        且把"条目维是否分片"绑到本钩子上会让**能力矩阵的既有断言**（默认模式与
        ``[dp_rank::dp_size]`` 逐位等价）失真。⇒ ⛔ 保持逐位等价。
        """
        from graspo.flow.backend.native.assignment import (
            assign_whole_groups,
            assert_balanced_assignment,
        )

        keys = list(epoch)
        degree = getattr(self.runtime, "parallel_degree", None)
        n_shards = int(getattr(degree, "dp_size", 1) or 1)
        rank = int(getattr(degree, "dp_rank", 0) or 0)
        if n_shards <= 1:
            yield from keys
            return
        shard_of = assign_whole_groups(keys, n_shards)
        assert_balanced_assignment(keys, shard_of, n_shards=n_shards)
        yield from (key for key in keys if shard_of[key] == rank)

    # ── 能力 1（epoch）/ 4（生成）/ 5（评分）/ 7（优化步进入）─────────────────
    def _epochs(self) -> Iterator[Any]:
        raise NotImplementedError("native 的 epoch 推进由既有循环内部完成；ABC 钩子待接线")

    def _generate(self, batch: Any) -> list[Any]:
        """能力 4：生成走既有 rollout 循环（``GraspoFlowRuntime.generate_*``）。"""
        runtime = self.runtime
        if isinstance(batch, dict):
            return runtime.generate_groups(**batch)
        raise NotImplementedError(
            "native 生成走既有 rollout 循环；向量化入参请用 _run 路径（ABC 钩子待接线）"
        )

    def _score_and_align(self, groups: list[Any]) -> list[Any]:
        raise NotImplementedError("native 评分/对齐由既有 rollout 循环内部完成；ABC 钩子待接线")

    def _optimize_step(self, groups: list[Any]) -> dict[str, Any]:
        """待 P4 接线：优化步进入（★各 rank micro-batch 次数相等）的产出点。"""
        raise NotImplementedError(
            "native 优化步进入由既有循环 + dp 集合完成；ABC 钩子待 P4 接线"
        )

    # ── 能力 8：检查点 ───────────────────────────────────────────────────────
    def save_checkpoint(self, path: str) -> None:
        self.runtime.save_checkpoint(path)

    def load_checkpoint(self, path: str) -> dict[str, Any] | None:
        return self.runtime.load_checkpoint(path)

    # ── 能力 9：指标 ─────────────────────────────────────────────────────────
    def metrics(self) -> dict[str, Any]:
        """逐步读数：既有循环若提供 ``metrics()`` 则转发，否则返回已归集的快照。"""
        getter = getattr(self._trainer, "metrics", None)
        if callable(getter):
            return dict(getter())
        return dict(self._metrics_snapshot)


__all__ = ["NativeBackend"]
