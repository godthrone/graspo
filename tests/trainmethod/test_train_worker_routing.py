"""``cli/train_worker`` 的**分派契约**测试（不依赖 torch / PyYAML，本机可跑）。

本文件守护的是指挥官 2026-09-18 的裁定 2：

- **路由归属路由表**（``core.discovery._REGISTRY_BY_TRAIN_METHOD``），
  ``cli/train_worker`` **只调用一次** ``resolve_backend_builder``，
  自己不再按算法分支——旧版 ``if config.train_method == "sft": ... else: <RL>``
  会把 CPT / OPD 静默送进 RL 注册表。
- **不存在第二套路由**：``flow/msswift/trainer.py`` 的 RL 工厂不得按
  ``train_method`` 分派（那是"为守边界而把设计做坏"的绕行方案，已撤销）。

用**行为**断言而不是读实现细节：跑四次 ``main()``，看它把哪个
``(backend, train_method)`` 交给了哪个工厂，以及工厂返回的训练器是否被 ``train(smoke)``
调用过。
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from graspo.core.schema import GraspoConfig

_REPO_ROOT = Path(__file__).resolve().parents[2]

STUDENT = "/models/Qwen3.5-9B"
TEACHER = "/models/Qwen3.8-27B"


def _config(train_method: str) -> GraspoConfig:
    data: dict[str, object] = {"train_method": train_method, "backend": "msswift"}
    if train_method == "opd":
        data["distill"] = {"teacher_model_path": TEACHER}
    return GraspoConfig.model_validate(data)


class _RecordingTrainer:
    def __init__(self) -> None:
        self.calls: list[bool] = []

    def train(self, *, smoke: bool = False) -> None:
        self.calls.append(smoke)


@pytest.mark.parametrize("train_method", ["graspo", "sft", "cpt", "opd"])
def test_train_worker_dispatches_through_the_single_router(monkeypatch, train_method: str):
    """★ 四条路由都走**同一条**路径：`resolve_backend_builder(backend, train_method)`。"""
    import graspo.cli.train_worker as worker

    config = _config(train_method)
    seen: list[tuple[str, str]] = []
    trainer = _RecordingTrainer()

    class _ConfigLoader:
        @staticmethod
        def from_yaml(_path: str) -> GraspoConfig:
            return config

    def _fake_resolve(backend: str, *, train_method: str):  # noqa: ANN202
        seen.append((backend, train_method))

        def _factory(_config: object, _selection: object) -> _RecordingTrainer:
            return trainer

        return _factory

    monkeypatch.setattr(worker, "GraspoConfig", _ConfigLoader)
    monkeypatch.setattr(worker, "require_gpu_lock_or_exit", lambda: [0])
    monkeypatch.setattr(worker, "resolve_backend_builder", _fake_resolve)
    # 按**模块对象**打补丁（不按 "pkg.sub.attr" 字符串）——字符串形式会走
    # `getattr(graspo, "flow")`，在"命名空间垫片只填了 sys.modules、没填父包属性"的
    # 环境下会以与真因无关的 AttributeError 失败。
    import graspo.flow.backend_selection as backend_selection

    monkeypatch.setattr(
        backend_selection,
        "select_backend",
        lambda _config, requested=None: SimpleNamespace(name="msswift", requested="msswift"),
    )
    monkeypatch.setattr(sys, "argv", ["train_worker", "--config", "unit.yaml", "--smoke"])

    worker.main()

    # ① 恰好调用一次路由，且把**算法**与**后端**都交给了路由表
    assert seen == [("msswift", train_method)]
    # ② 路由返回的工厂被调用，且调用方真的执行了 train(smoke=True)
    assert trainer.calls == [True]


def test_train_worker_has_no_algorithm_branch():
    """★ 源码级守护：worker 里不得再出现按 ``train_method`` 的算法分支。"""
    source = (_REPO_ROOT / "src" / "graspo" / "cli" / "train_worker.py").read_text(
        encoding="utf-8"
    )
    assert 'resolve_backend_builder(selection.name, train_method=config.train_method)' in source
    # 旧分支形态（"非 sft 即 RL"）：出现即说明路由被复制回了调用方
    assert 'config.train_method == "sft"' not in source
    assert "create_trainer(" not in source


def test_msswift_rl_factory_does_not_route():
    """★ 源码级守护：ms-swift 的 RL 工厂**不得**承担分派职责（绕行方案已撤销）。

    路由只有一处（``core.discovery``）；后端实现按"哪个后端怎么跑"负责，
    不按"该走哪张注册表"负责（宪法 §1.1 模块边界 / §1.4 单一真相源）。
    """
    source = (
        _REPO_ROOT / "src" / "graspo" / "flow" / "msswift" / "trainer.py"
    ).read_text(encoding="utf-8")
    assert "resolve_backend_builder" not in source
    assert "_REGISTRY_BY_TRAIN_METHOD" not in source
