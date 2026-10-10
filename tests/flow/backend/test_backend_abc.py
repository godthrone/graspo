"""``flow.backend.base.RLBackend`` 的接口契约测试（P3）。

覆盖（每条都带**阳性/阴性对照**，⛔ 不接受恒 PASS）：
1. **骨架**：``train()`` 按 §9.2 顺序跑完 5 个阶段 + ``finally`` 必 ``_close``（异常也关）；
2. **9 项能力**：``CAPABILITY_SPECS`` 恰好 9 项；两个真 backend 逐项声明；矩阵可机核；
3. **禁止覆盖公开骨架**：子类覆盖 ``train()`` ⇒ :func:`assert_backend_contract` 判红；
4. **两个真 backend 的契约一致性**：``NativeBackend`` / ``MsSwiftBackend`` 通过契约校验；
5. **行为等价**：两 backend 的 ``_run`` 逐字委托既有循环（smoke 透传）；
6. **0 行改动演练**：注册第三个 backend ⇒ ``create_backend`` 立刻可用，既有代码 0 行改动。
"""

from __future__ import annotations

from typing import Any

import pytest

from graspo.flow.backend.base import (
    CAPABILITY_NAMES,
    CAPABILITY_SPECS,
    BackendCapability,
    RLBackend,
    assert_backend_contract,
)


# ── 测试替身 ─────────────────────────────────────────────────────────────────
class _FakeTrainer:
    """记录 ``train(smoke=...)`` 调用的替身（不发 torch / GPU）。"""

    def __init__(self, *, fail: bool = False) -> None:
        self.calls: list[bool] = []
        self._fail = fail

    def train(self, *, smoke: bool = False) -> None:
        self.calls.append(smoke)
        if self._fail:
            raise RuntimeError("boom")


class _FakeBackend(RLBackend):
    """合规替身：用同一个 **fake RolloutGroup 流**跑通骨架。"""

    name = "fake"
    capabilities = frozenset(BackendCapability)

    def __init__(self, *, fail: bool = False) -> None:
        super().__init__(config=object())
        self.events: list[Any] = []
        self._fail = fail

    def validate(self) -> None:
        self.events.append("validate")

    def _setup(self) -> None:
        self.events.append("setup")

    def _epochs(self):
        yield "e0"

    def _step_batches(self, epoch):
        yield f"{epoch}:b0"

    def _generate(self, batch):
        self.events.append(("generate", batch))
        return [{"group_key": "p1", "completions": tuple(range(8)), "group_size": 8}]

    def _score_and_align(self, groups):
        self.events.append(("score", len(groups)))
        return groups

    def _optimize_step(self, groups):
        self.events.append(("optimize", len(groups)))
        if self._fail:
            raise RuntimeError("boom")
        return {"loss": 1.0}

    def metrics(self) -> dict[str, Any]:
        return dict(self._metrics_snapshot)

    def _close(self) -> None:
        self.events.append("close")


class _IllegalOverrideBackend(_FakeBackend):
    """违规替身：覆盖公开骨架 ``train()``（§9.2 禁止）。"""

    def train(self, *, smoke: bool = False) -> None:  # type: ignore[override]
        self.validate()


def _native_backend(trainer: _FakeTrainer):
    from graspo.flow.backend.native.backend import NativeBackend

    return NativeBackend(config=object(), trainer=trainer)


def _msswift_backend(trainer: _FakeTrainer):
    from graspo.flow.backend.msswift.backend import MsSwiftBackend

    return MsSwiftBackend(config=object(), trainer=trainer)


# ── 1. 骨架（模板方法）────────────────────────────────────────────────────────
def test_skeleton_runs_phases_in_order_then_closes():
    backend = _FakeBackend()
    backend.train(smoke=True)
    assert backend.events == [
        "validate",
        "setup",
        ("generate", "e0:b0"),
        ("score", 1),
        ("optimize", 1),
        "close",
    ]


def test_skeleton_closes_even_when_optimize_raises():
    """★ 阳性对照：主体抛错 ⇒ ``finally`` 仍必须 ``_close``。"""
    backend = _FakeBackend(fail=True)
    with pytest.raises(RuntimeError, match="boom"):
        backend.train()
    assert backend.events[-1] == "close"


# ── 2. 9 项能力 ──────────────────────────────────────────────────────────────
def test_capability_specs_are_exactly_nine():
    assert len(CAPABILITY_SPECS) == 9
    assert CAPABILITY_NAMES == {
        "lifecycle", "device", "sharding", "generate", "score",
        "sync", "optimize_step", "checkpoint", "metrics",
    }


def test_capability_matrix_is_machine_checkable():
    backend = _FakeBackend()
    matrix = backend.capability_matrix()
    assert set(matrix) == CAPABILITY_NAMES
    assert all(matrix.values())
    assert backend.supports(BackendCapability.SHARDING)


# ── 3. 契约校验（含阳性/阴性对照）────────────────────────────────────────────
def test_assert_backend_contract_accepts_compliant_backend():
    assert_backend_contract(_FakeBackend())


def test_assert_backend_contract_rejects_train_override():
    """★ 阳性对照：覆盖公开骨架必须判红。"""
    with pytest.raises(TypeError, match="覆盖了公开骨架"):
        assert_backend_contract(_IllegalOverrideBackend())


def test_assert_backend_contract_rejects_missing_capability():
    class _MissingCap(_FakeBackend):
        capabilities = frozenset(BackendCapability) - {BackendCapability.SYNC}

    with pytest.raises(TypeError, match="未声明能力"):
        assert_backend_contract(_MissingCap())


def test_assert_backend_contract_rejects_empty_name():
    class _NoName(_FakeBackend):
        name = ""

    with pytest.raises(TypeError, match="未声明注册表名"):
        assert_backend_contract(_NoName())


# ── 4. 两个真 backend 的契约一致性 ──────────────────────────────────────────
@pytest.mark.parametrize("factory", [_native_backend, _msswift_backend])
def test_real_backends_satisfy_the_abc_contract(factory):
    backend = factory(_FakeTrainer())
    assert_backend_contract(backend)
    assert backend.name in {"native", "msswift"}
    assert set(backend.capability_matrix()) == CAPABILITY_NAMES


def test_real_backends_declare_all_nine_capabilities():
    native = _native_backend(_FakeTrainer())
    msswift = _msswift_backend(_FakeTrainer())
    for backend in (native, msswift):
        assert backend.capabilities == frozenset(BackendCapability), backend.name


# ── 5. 行为等价：`_run` 逐字委托既有循环 ────────────────────────────────────
@pytest.mark.parametrize("factory", [_native_backend, _msswift_backend])
@pytest.mark.parametrize("smoke", [False, True])
def test_run_delegates_to_the_legacy_loop_with_smoke_passthrough(factory, smoke):
    """★ P3 的行为等价声明：backend.train() 的**唯一执行路径** = 既有循环。"""
    trainer = _FakeTrainer()
    factory(trainer).train(smoke=smoke)
    assert trainer.calls == [smoke]


@pytest.mark.parametrize("factory", [_native_backend, _msswift_backend])
def test_real_backends_have_no_train_override(factory):
    """两 backend 都**没有**覆盖 `train()` ⇒ 骨架的 validate/_close 语义仍受基类管辖。"""
    assert "train" not in factory(_FakeTrainer()).__class__.__dict__


# ── 6. 新增 backend 的"0 行改动"演练 ─────────────────────────────────────────
def test_third_backend_needs_zero_code_changes_in_factories(tmp_path, monkeypatch):
    """★ 注册第三个 backend ⇒ `create_backend` 立刻可用，既有代码 0 行改动（§1.2）。"""
    import sys

    import graspo.core.discovery as discovery
    import graspo.flow.registry  # noqa: F401
    from graspo.flow.backend.selection import BackendSelection, create_backend

    # 把"第三方 backend"写成一个临时可导入模块（模拟 `flow/backend/<name>/` 新增一个包）
    (tmp_path / "third_backend_stub.py").write_text(
        "from graspo.flow.backend.base import RLBackend\n"
        "from graspo.flow.backend.base import BackendCapability\n"
        "class ThirdBackend(RLBackend):\n"
        "    name = 'thirdparty'\n"
        "    capabilities = frozenset(BackendCapability)\n"
        "    def __init__(self, config, selection=None):\n"
        "        super().__init__(config, selection)\n"
        "    def validate(self): ...\n"
        "    def _setup(self): ...\n"
        "    def _epochs(self): return iter(())\n"
        "    def _step_batches(self, epoch): return iter(())\n"
        "    def _generate(self, batch): return []\n"
        "    def _score_and_align(self, groups): return groups\n"
        "    def _optimize_step(self, groups): return {}\n"
        "    def metrics(self): return {}\n"
        "    def _close(self): ...\n"
        "def create_third_backend(config, selection=None):\n"
        "    return ThirdBackend(config, selection)\n",
        encoding="utf-8",
    )
    monkeypatch.syspath_prepend(str(tmp_path))

    fallbacks = dict(discovery._REGISTERED_FALLBACKS)
    fallbacks["graspo.backends"] = dict(fallbacks["graspo.backends"])
    fallbacks["graspo.backends"]["thirdparty"] = "third_backend_stub:create_third_backend"
    monkeypatch.setattr(discovery, "_REGISTERED_FALLBACKS", fallbacks)

    # 本机已安装本包 ⇒ entry_points 非空会**优先**接管；演练的是"开发期回退表"路径，
    # 因此把 entry_points 探测临时置空（生产侧等价动作 = pyproject 加一行 entry_point）。
    class _NoEntryPoints:
        def select(self, **_kwargs):
            return []

    monkeypatch.setattr(discovery.importlib.metadata, "entry_points", lambda: _NoEntryPoints())

    try:
        selection = BackendSelection(name="thirdparty", reason="", requested="thirdparty")
        backend = create_backend(config=object(), selection=selection)
        assert backend.name == "thirdparty"
        assert_backend_contract(backend)
    finally:
        sys.modules.pop("third_backend_stub", None)
