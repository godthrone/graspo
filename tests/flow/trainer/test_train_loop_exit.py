"""训练循环**退出路径**的契约测试（纯逻辑，不触 GPU、不建真实进程组）。

**背景（2026-09-22 T035 实测，9B GRASPO 全参 · native · 2 卡 pp=2，rc=1 / 890.7s）**：
``train()`` 的 ``finally`` 无条件执行 ``dist.barrier()``（WORLD）。当一侧 stage 抛错时，
另一 stage 通常正阻塞在 PP P2P / ``pp_group`` 集合上 ⇒ 这道 barrier 永远等不到对端，
只能等满 600s ``nccl_collective_timeout``，把**首个异常**掩盖成
``c10::DistBackendError`` + SIGABRT（容器里 rank1 的 traceback 只到
``trainer.py`` 的 barrier，真正的首错取证链断裂）。

容器的决定性证据（228 ``runs/verify-p1/r1/T035/stdout.log``）：

- rank1：``last enqueued work: 3``（= barrier）、``last completed work: 2``；
  rank0：``last enqueued work: 2``、``last completed work: 2``
  ⇒ rank0 **从未 enqueue 那道 barrier**（不是"同刻提交不同算子"，而是"一侧已退出
  训练循环、另一侧还在循环里"）。
- ``pp_debug.log`` 只有 prefill 的 3 行（mtime 06:00:27），此后无任何 PP forward；
  ``events.jsonl`` 只有 ``backend_selected`` / ``run_start``。
- ``entry.sh`` 不传 ``--smoke``（且 ``--smoke`` 是 ``store_true``）⇒
  ``_smoke_boundary=False`` ⇒ ``trainer.py`` 的 ``stop_requested`` 恒 False ⇒
  不存在"smoke 早退"这条正常出口 ⇒ rank1 只能经**异常**退出 try 块。

本文件把修后的契约钉死：

1. **正常退出**（含 ``return``，如 smoke 边界）：行为与修复前**逐字不变**——
   先 WORLD barrier 再 ``runtime.close()``（对 ``pp_size==1`` 的既有档位零影响）；
2. **异常退出**：**不得**做任何 WORLD 对齐（否则重新引入 600s 掩盖），
   只做非阻塞 abort；清理失败也只记 WARNING，绝不替换正在传播的首错。
"""

from __future__ import annotations

import inspect
import logging

import pytest

import graspo.flow.trainer.trainer as trainer_module
from graspo.flow.trainer.trainer import GraspoFlowTrainer


class _FakeDist:
    """最小 ``torch.distributed`` 鸭子类型：只记录被调用的集合通信入口。"""

    def __init__(self, *, initialized: bool = True, destroy_raises: bool = False) -> None:
        self.calls: list[str] = []
        self._initialized = initialized
        self._destroy_raises = destroy_raises

    def is_available(self) -> bool:
        return True

    def is_initialized(self) -> bool:
        return self._initialized

    def barrier(self, *args: object, **kwargs: object) -> None:
        self.calls.append("barrier")

    def destroy_process_group(self, *args: object, **kwargs: object) -> None:
        self.calls.append("destroy_process_group")
        if self._destroy_raises:
            raise RuntimeError("destroy_process_group 失败（模拟）")


class _FakeRuntime:
    def __init__(self) -> None:
        self.close_calls = 0

    def close(self) -> None:
        self.close_calls += 1


def _make_trainer() -> tuple[GraspoFlowTrainer, _FakeRuntime]:
    """只造出退出路径需要的两个协作者（绕过 ``__init__``：不建 runtime、不碰 GPU）。"""
    trainer = GraspoFlowTrainer.__new__(GraspoFlowTrainer)
    runtime = _FakeRuntime()
    trainer.runtime = runtime  # type: ignore[assignment]
    return trainer, runtime


@pytest.fixture
def fake_dist(monkeypatch: pytest.MonkeyPatch):
    def _install(*, initialized: bool = True, destroy_raises: bool = False) -> _FakeDist:
        fake = _FakeDist(initialized=initialized, destroy_raises=destroy_raises)
        monkeypatch.setattr(trainer_module, "dist", fake)
        return fake

    return _install


# ── 正常退出路径：与修复前逐字不变（pp=1 的 46 档不受影响）──────────────────


class TestNormalExitUnchanged:
    def test_barrier_then_close(self, fake_dist) -> None:
        fake = fake_dist()
        trainer, runtime = _make_trainer()
        trainer._teardown_after_train_loop()
        assert fake.calls == ["barrier"], "正常退出必须先 WORLD 对齐（原语义）"
        assert runtime.close_calls == 1

    def test_without_distributed_only_closes(self, fake_dist) -> None:
        """未初始化进程组（单进程 / CPU 单测）⇒ 不调 barrier，直接 close（原语义）。"""
        fake = fake_dist(initialized=False)
        trainer, runtime = _make_trainer()
        trainer._teardown_after_train_loop()
        assert fake.calls == []
        assert runtime.close_calls == 1

    def test_barrier_uses_world_group(self, fake_dist) -> None:
        """契约：对齐范围仍是 WORLD（不传 group ⇒ default_pg）——不改既有语义。"""
        fake = fake_dist()
        trainer, _ = _make_trainer()
        trainer._teardown_after_train_loop()
        assert fake.calls == ["barrier"]


# ── 异常退出路径：不得 rendezvous（修 T035 的 600s 掩盖）────────────────────


class TestExceptionalExitNeverRendezvous:
    def test_no_barrier_on_exception_path(self, fake_dist) -> None:
        """★ 核心回归：异常路径**绝不**碰 barrier（碰了就是重新引入 600s 掩盖）。"""
        fake = fake_dist()
        trainer, runtime = _make_trainer()
        trainer._abort_distributed()
        assert "barrier" not in fake.calls
        assert fake.calls == ["destroy_process_group"]
        # 也**不得**走 runtime.close()：其内部 destroy_parallel_state 仍带 WORLD barrier。
        assert runtime.close_calls == 0

    def test_abort_is_non_blocking_when_not_initialized(self, fake_dist) -> None:
        fake = fake_dist(initialized=False)
        trainer, _ = _make_trainer()
        trainer._abort_distributed()
        assert fake.calls == []

    def test_abort_never_masks_the_original_error(self, fake_dist, caplog) -> None:
        """清理失败只记 WARNING：异常路径的二次失败绝不能替换正在传播的首错。"""
        fake = fake_dist(destroy_raises=True)
        trainer, _ = _make_trainer()
        with caplog.at_level(logging.WARNING, logger="graspo.trainer"):
            trainer._abort_distributed()  # 不得抛出
        assert fake.calls == ["destroy_process_group"]
        assert any("original error is preserved" in record.message for record in caplog.records), (
            caplog.text
        )


# ── 接线契约：train() 必须把两条出口分派到上述两个方法 ──────────────────────


class TestTrainExitWiring:
    def test_train_dispatches_exception_exit_to_abort(self) -> None:
        src = inspect.getsource(GraspoFlowTrainer.train)
        assert "except BaseException" in src, "异常退出必须被显式捕获以走非阻塞 abort"
        assert "self._abort_distributed()" in src
        assert "raise" in src, "捕获后必须原样抛出首个异常（不得吞掉）"

    def test_train_keeps_rendezvous_on_normal_exit(self) -> None:
        src = inspect.getsource(GraspoFlowTrainer.train)
        assert "if not aborted:" in src, "正常退出分支必须保留（含 smoke 的 return 路径）"
        assert "self._teardown_after_train_loop()" in src
        # WORLD barrier 只允许存在于正常退出路径的方法里；train() 自身不得直接调。
        assert "dist.barrier(" not in src

    def test_barrier_lives_only_in_the_normal_exit_method(self) -> None:
        normal = inspect.getsource(GraspoFlowTrainer._teardown_after_train_loop)
        abnormal = inspect.getsource(GraspoFlowTrainer._abort_distributed)
        assert "dist.barrier(" in normal
        # 断言的是**调用**（不是 docstring 里的散文）：异常路径不得调用 barrier。
        assert "dist.barrier(" not in abnormal
        assert "destroy_process_group" in abnormal
