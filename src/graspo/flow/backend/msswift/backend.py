"""flow.backend.msswift.backend — ``MsSwiftBackend``：ms-swift 后端的 :class:`RLBackend` 实现。

**组合结构**（backend-interface.md §2.3）：``MsSwiftBackend`` 继承 :class:`RLBackend`，
**组合**上游 ms-swift 训练器（经 ``GraspoMsSwiftGRPOTrainer`` 覆写钩子）与既有的
``MsSwiftRlTrainer``。

**行为等价声明（P3）**：本后端沿用既有 ``MsSwiftRlTrainer.train()`` 作为执行路径
（``_run`` 显式委托）。ms-swift 侧的 G1（整组分片）/ G2（按键对齐）/ retry 接线是
**P5** 的工作，本轮**不动**；9 项能力按 ABC 契约逐项实现/登记（见能力矩阵）。

**构造廉价**：``MsSwiftRlTrainer.__init__`` 不触发 ms-swift / torch 导入；重量级导入
发生在 ``train()`` 内部（因此本后端的 :meth:`validate` 不探测 ms-swift，避免把
"缺 ms-swift" 这条既有 fail-closed 路径提前到构造期而改变错误行为）。
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from graspo.flow.backend.base import BackendCapability, RLBackend


class MsSwiftBackend(RLBackend):
    """ms-swift 后端（Megatron / DeepSpeed / vLLM 由上游承载）。"""

    name = "msswift"
    capabilities = frozenset(BackendCapability)  # 9 项全声明（分片/评分待 P5 修正）

    def __init__(self, config: Any, selection: Any | None = None, *, trainer: Any | None = None) -> None:
        super().__init__(config, selection)
        self._trainer = trainer if trainer is not None else self._build_trainer()

    def _build_trainer(self) -> Any:
        from graspo.flow.backend.msswift.trainer import MsSwiftRlTrainer

        return MsSwiftRlTrainer(self.config, self.selection)

    @property
    def trainer(self) -> Any:
        return self._trainer

    # ── 能力 1：生命周期 ─────────────────────────────────────────────────────
    def validate(self) -> None:
        """无副作用前置检查（不探测 ms-swift —— 那会改变既有 fail-closed 的时机）。"""
        if self.config is None:
            raise ValueError("MsSwiftBackend 需要 config")

    def _setup(self) -> None:
        """显式空实现：上游训练器的建立在 ``_run`` 内部完成。"""

    def _close(self) -> None:
        """显式空实现：释放由 ``_run``（上游训练器）内部完成。"""

    # ── 训练主体（★ 行为等价的唯一执行路径）─────────────────────────────────
    def _run(self, *, smoke: bool = False) -> None:
        """委托既有 ``MsSwiftRlTrainer.train()`` —— 与 P2 及以前逐字同路径。"""
        self._trainer.train(smoke=smoke)

    # ── 9 项能力的钩子（契约面；执行路径沿用既有循环，G1/G2 待 P5）───────────
    def _epochs(self) -> Iterator[Any]:
        raise NotImplementedError("msswift 的 epoch 推进由上游训练循环完成；ABC 钩子待接线")

    def _step_batches(self, epoch: Any) -> Iterator[Any]:
        """能力 3（★整组不拆）：待 **P5/G1** 用 ``group_sharding`` 接线。"""
        raise NotImplementedError("msswift 分片由上游 sampler 完成；整组分片待 P5/G1 接线")

    def _generate(self, batch: Any) -> list[Any]:
        raise NotImplementedError("msswift 生成由上游 rollout 管道完成；ABC 钩子待接线")

    def _score_and_align(self, groups: list[Any]) -> list[Any]:
        """能力 5（★按键对齐）：待 **P5/G2** 删掉 positional fallback 后接线。"""
        raise NotImplementedError("msswift 评分/对齐在 _postprocess_batch；按键对齐待 P5/G2 接线")

    def _optimize_step(self, groups: list[Any]) -> dict[str, Any]:
        raise NotImplementedError("msswift 优化步进入由上游训练循环完成；ABC 钩子待接线")

    # ── 能力 8：检查点 ───────────────────────────────────────────────────────
    def save_checkpoint(self, path: str) -> None:
        saver = getattr(self._trainer, "save_checkpoint", None)
        if not callable(saver):
            raise NotImplementedError("MsSwiftRlTrainer 未提供 save_checkpoint（能力 8）")
        saver(path)

    def load_checkpoint(self, path: str) -> dict[str, Any] | None:
        loader = getattr(self._trainer, "load_checkpoint", None)
        if not callable(loader):
            raise NotImplementedError("MsSwiftRlTrainer 未提供 load_checkpoint（能力 8）")
        return loader(path)

    # ── 能力 9：指标 ─────────────────────────────────────────────────────────
    def metrics(self) -> dict[str, Any]:
        getter = getattr(self._trainer, "metrics", None)
        if callable(getter):
            return dict(getter())
        return dict(self._metrics_snapshot)


__all__ = ["MsSwiftBackend", "create_msswift_backend"]


# ★ P3：ABC 工厂（`graspo.backends` 注册表项）—— 返回 `RLBackend` 实现。
def create_msswift_backend(config: Any, selection: Any | None = None) -> RLBackend:
    """ms-swift 后端的 ABC 工厂（入口：`graspo.backends` 注册表）。

    返回 :class:`MsSwiftBackend`；**构造廉价**（不触发 ms-swift / torch 导入）。
    """
    return MsSwiftBackend(config, selection)
