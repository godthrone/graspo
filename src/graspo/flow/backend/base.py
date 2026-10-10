"""flow.backend.base — ★ ``RLBackend``：可切换训练后端的**接口边界**（ABC）。

依据：``task-arch-design/backend-interface.md`` §2（9 项能力 + §9.2 模板方法 + §9.3 注册表）。

## 一个 backend 必须提供什么（9 项能力）

| # | 能力 | 语义 | 不变量 |
|:--:|---|---|---|
| 1 | **生命周期** | `validate()` / `_setup()` / `_close()` | 启动期校验失败必须 fail-closed |
| 2 | **设备管理** | device / rank / world_size / 卡位守卫 / 并行组 | 卡守卫规则**唯一来源**在 `core/gpu_guard` + `flow/guard` |
| 3 | **分片** | 把全局 prompt/组清单切成"本卡要做的部分" | ★**必须整组不拆**（β1） |
| 4 | **生成** | 产出 `RolloutGroup`（G 条 completion + 原始序列/张量） | ★**必须 G 条齐**（I1） |
| 5 | **评分** | 对每条 completion 打分，产出"奖励 ↔ 样本"的按键映射 | ★**按键对齐**（I2），禁行号回退 |
| 6 | **同步** | dp/tp/pp 集合通信、会合点预算与看门狗 | 会合点可枚举、可预算（NCCL 600 s 对照面） |
| 7 | **优化步进入** | 决定本 step 全局训哪些组、各 rank 训多少 | ★**各 rank micro-batch 次数相等** |
| 8 | **检查点** | `save_checkpoint(path)` / `load_checkpoint(path)` | 产物落盘由 config 决定 |
| 9 | **指标** | 逐步结构化读数 | 口径跨 backend 一致，否则验证台无法对拍 |

## 形状（§9.2 模板方法）

公开骨架 :meth:`RLBackend.train` 定顺序；子类实现 ``_`` 前缀钩子。子类**不得**覆盖
:meth:`train`；要改执行路径请覆盖 :meth:`_run`（默认实现 = 钩子骨架）。

## 边界

- 本模块**不 import torch / ms-swift**（可在无 GPU 开发机单测）；
- 算法（``ripple/``）与契约（``flow/adapter/``）**都不认识** :class:`RLBackend` 的子类；
- 数据方向单向：backend → adapter → ripple → adapter → backend（不回调，§1.4）。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator
from enum import StrEnum
from typing import Any, ClassVar


class BackendCapability(StrEnum):
    """9 项能力的**枚举名**（供能力矩阵与 `supports()` 查询；语义见模块 docstring）。"""

    LIFECYCLE = "lifecycle"
    DEVICE = "device"
    SHARDING = "sharding"
    GENERATE = "generate"
    SCORE = "score"
    SYNC = "sync"
    OPTIMIZE_STEP = "optimize_step"
    CHECKPOINT = "checkpoint"
    METRICS = "metrics"


#: 9 项能力的**契约面**（名称 → 语义 → 不变量）—— 能力矩阵的单一真相源（§1.4）。
CAPABILITY_SPECS: tuple[tuple[BackendCapability, str, str], ...] = (
    (BackendCapability.LIFECYCLE, "validate/_setup/_close", "启动期校验失败 fail-closed"),
    (BackendCapability.DEVICE, "device/rank/world_size/并行组", "卡守卫规则唯一来源"),
    (BackendCapability.SHARDING, "全局清单 → 本卡份额", "★整组不拆（β1）"),
    (BackendCapability.GENERATE, "产出 RolloutGroup", "★G 条齐（I1）"),
    (BackendCapability.SCORE, "奖励 ↔ 样本按键映射", "★按键对齐（I2），禁行号回退"),
    (BackendCapability.SYNC, "dp/tp/pp 集合通信与看门狗", "会合点可枚举、可预算"),
    (BackendCapability.OPTIMIZE_STEP, "本 step 训哪些组、各 rank 训多少", "★各 rank micro-batch 次数相等"),
    (BackendCapability.CHECKPOINT, "save/load checkpoint", "产物落盘由 config 决定"),
    (BackendCapability.METRICS, "逐步结构化读数", "读数口径跨 backend 一致"),
)

CAPABILITY_NAMES: frozenset[str] = frozenset(c.value for c, _, _ in CAPABILITY_SPECS)


class RLBackend(ABC):
    """可切换训练后端。算法（ripple）与契约（flow/adapter）都不认识本类的子类。"""

    #: 注册表键（``graspo.backends`` 的 entry point 名）。
    name: ClassVar[str] = ""
    #: 本后端**声明**提供的能力（子类必须显式声明；见 :func:`assert_backend_contract`）。
    capabilities: ClassVar[frozenset[BackendCapability]] = frozenset()

    def __init__(self, config: Any, selection: Any | None = None) -> None:
        self.config = config
        self.selection = selection
        self._metrics_snapshot: dict[str, Any] = {}

    # ── 公开骨架（★ 子类不得覆盖；要改执行路径请覆盖 `_run`）────────────────
    def train(self, *, smoke: bool = False) -> None:
        """**唯一对外入口**：validate → setup → 训练主体 → close（§9.2 模板方法）。"""
        self.validate()
        self._setup()
        try:
            self._run(smoke=smoke)
        finally:
            self._close()

    def _run(self, *, smoke: bool = False) -> None:
        """默认训练主体：epoch → step_batch → 生成 → 评分 → 优化步 → 上报。

        子类可覆盖以沿用既有成熟循环 —— 但必须**保持同一语义顺序**
        （分片 → 生成 → 评分/对齐 → 优化步进入 → 上报），且不得绕过 ``validate/_close``。
        """
        for epoch in self._epochs():
            for step_batch in self._step_batches(epoch):
                groups = self._generate(step_batch)
                groups = self._score_and_align(groups)
                metrics = self._optimize_step(groups)
                self._report(step_batch, metrics)

    # ── 能力查询 ─────────────────────────────────────────────────────────────
    def supports(self, capability: BackendCapability) -> bool:
        return capability in self.capabilities

    def capability_matrix(self) -> dict[str, bool]:
        return {name: self.supports(BackendCapability(name)) for name in sorted(CAPABILITY_NAMES)}

    # ── 子类钩子（能力 1–7、9）──────────────────────────────────────────────
    @abstractmethod
    def validate(self) -> None:
        """启动期校验（能力 1）：不满足前置条件时**显式失败**（fail-closed）。"""

    @abstractmethod
    def _setup(self) -> None:
        """建立设备 / 并行组 / 模型（能力 1–2）。"""

    @abstractmethod
    def _epochs(self) -> Iterator[Any]:
        """产出 epoch（能力 1）。"""

    @abstractmethod
    def _step_batches(self, epoch: Any) -> Iterator[Any]:
        """分片：把全局清单切成"本卡要做的部分"（能力 3，★整组不拆）。"""

    @abstractmethod
    def _generate(self, batch: Any) -> list[Any]:
        """生成 ``RolloutGroup``（能力 4，★G 条齐 = I1）。"""

    @abstractmethod
    def _score_and_align(self, groups: list[Any]) -> list[Any]:
        """评分并按键对齐（能力 5，★禁行号回退 = I2）。"""

    @abstractmethod
    def _optimize_step(self, groups: list[Any]) -> dict[str, Any]:
        """优化步进入：决定训哪些组、各 rank 训多少（能力 7，★次数相等）。"""

    @abstractmethod
    def metrics(self) -> dict[str, Any]:
        """逐步结构化读数（能力 9）。"""

    @abstractmethod
    def _close(self) -> None:
        """释放设备 / 进程组（能力 1）。"""

    # ── 可选能力（能力 8；默认未实现，子类按需覆盖）─────────────────────────
    def save_checkpoint(self, path: str) -> None:
        """保存检查点（能力 8）。默认未实现 ⇒ 显式失败（不静默跳过）。"""
        raise NotImplementedError(f"{type(self).__name__} 未提供 save_checkpoint（能力 8）")

    def load_checkpoint(self, path: str) -> dict[str, Any] | None:
        """加载检查点（能力 8）。默认未实现 ⇒ 显式失败（不静默跳过）。"""
        raise NotImplementedError(f"{type(self).__name__} 未提供 load_checkpoint（能力 8）")

    # ── 默认上报（能力 9）────────────────────────────────────────────────────
    def _report(self, step_batch: Any, metrics: dict[str, Any]) -> None:
        """默认上报：把本步读数并入快照（子类可覆盖为日志/落盘）。"""
        self._metrics_snapshot.update(metrics)


def assert_backend_contract(backend: RLBackend) -> None:
    """契约一致性校验（供"新增 backend 0 行改动"演练与两 backend 一致性测试）。

    Raises:
        TypeError: ``name`` 为空 / 能力未声明 / 未实现任一抽象钩子 / 覆盖了公开骨架。
    """
    if not backend.name:
        raise TypeError(f"{type(backend).__name__} 未声明注册表名 `name`")

    missing = sorted(c.value for c in BackendCapability if c not in backend.capabilities)
    if missing:
        raise TypeError(f"{type(backend).__name__} 未声明能力 {missing}（9 项须逐项声明）")

    for hook in ("validate", "_setup", "_epochs", "_step_batches", "_generate",
                 "_score_and_align", "_optimize_step", "metrics", "_close"):
        fn = getattr(type(backend), hook, None)
        if fn is None or getattr(fn, "__isabstractmethod__", False):
            raise TypeError(f"{type(backend).__name__} 未实现钩子 {hook}")
    if "train" in type(backend).__dict__:
        raise TypeError(f"{type(backend).__name__} 覆盖了公开骨架 train()（§9.2 禁止）")
    if "train" not in RLBackend.__dict__:
        raise TypeError("RLBackend 必须提供公开骨架 train()")


__all__ = [
    "CAPABILITY_NAMES",
    "CAPABILITY_SPECS",
    "BackendCapability",
    "RLBackend",
    "assert_backend_contract",
]
