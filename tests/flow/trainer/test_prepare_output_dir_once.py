"""RC-3 回归用例：多 rank 不得并发 ``rmtree`` 同一个输出目录（纯 CPU，无 GPU）。

缺陷（228 实测 ``T030`` run2）：每个 rank 各自 ``prepare_output_dir(overwrite=True)``
⇒ 多个 ``shutil.rmtree`` 撞车 ⇒ ``FileNotFoundError: PosixPath('/out/T030')``。

本文件验证 ``helpers.prepare_output_dir_once`` 的四条契约：
1. **只有 primary rank 删**（其余 rank 对文件系统零改动）；
2. **其余 rank 在广播上等待**（广播返回即"目录已就绪"，没人在 rmtree 窗口内建目录）；
3. ``overwrite`` 语义不变（非空 + 未授权 ⇒ ``FileExistsError``；授权 ⇒ 清干净）；
4. primary 的失败被广播 ⇒ 其余 rank 一起 fail-closed，**不会**干等看门狗超时。
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
import torch.distributed as dist

from graspo.flow.trainer.helpers import prepare_output_dir_once


class _FakeDist:
    """把 ``torch.distributed`` 的四个接口替换成单进程可观测的替身。

    ``on_first_broadcast`` 让用例模拟"另一个 rank（rank0）已完成清目录"这一
    事实——真实运行里它是 ``dist.broadcast`` 的阻塞语义带来的同步点。
    """

    def __init__(self, *, world_size: int = 4, on_first_broadcast=None) -> None:
        self.world_size = world_size
        self.on_first_broadcast = on_first_broadcast
        self.calls: list[int] = []

    def install(self, monkeypatch: pytest.MonkeyPatch) -> "_FakeDist":
        monkeypatch.setattr(dist, "is_available", lambda: True)
        monkeypatch.setattr(dist, "is_initialized", lambda: True)
        monkeypatch.setattr(dist, "get_world_size", lambda: self.world_size)
        monkeypatch.setattr(dist, "broadcast", self.broadcast, raising=False)
        return self

    def broadcast(self, tensor: torch.Tensor, *, src: int = 0, group=None) -> None:
        assert src == 0, "同步点必须锚在 rank0"
        self.calls.append(src)
        if len(self.calls) == 1 and self.on_first_broadcast is not None:
            self.on_first_broadcast()


def _populate(path: Path) -> Path:
    (path / "logs").mkdir(parents=True)
    (path / "logs" / "old.jsonl").write_text("stale\n", encoding="utf-8")
    (path / "final").mkdir()
    (path / "final" / "rank_00000.pt").write_bytes(b"old-weights")
    return path


def test_non_primary_never_touches_the_filesystem(tmp_path, monkeypatch):
    """其余 rank **一次都不删**：调用前后目录内容逐字不变。"""
    out = _populate(tmp_path / "T030")
    before = sorted(p.relative_to(out).as_posix() for p in out.rglob("*"))

    calls: list[int] = []

    def probe(output_dir, *, overwrite):
        calls.append(1)
        raise AssertionError("非 primary rank 不得执行破坏性的 prepare_output_dir")

    monkeypatch.setattr("graspo.flow.lora.lora_io.prepare_output_dir", probe)
    fake = _FakeDist(on_first_broadcast=None).install(monkeypatch)

    returned = prepare_output_dir_once(out, overwrite=True, is_primary=False)

    assert calls == [], "非 primary rank 不得调用 prepare_output_dir"
    assert returned == Path(out)
    assert sorted(p.relative_to(out).as_posix() for p in out.rglob("*")) == before
    assert fake.calls == [0, 0], "非 primary rank 必须经过广播（状态+报错载荷）才继续"


def test_non_primary_returns_only_after_rank0_finished_cleaning(tmp_path, monkeypatch):
    """等待语义：广播返回时目录必须已经"清干净 + 重建"（没人在窗口内建目录）。"""
    out = _populate(tmp_path / "T030")
    observed: list[bool] = []

    def rank0_cleanup() -> None:
        # 模拟 rank0 在另一个进程里完成 prepare_output_dir(overwrite=True)
        import shutil

        shutil.rmtree(out)
        out.mkdir(parents=True)

    fake = _FakeDist(on_first_broadcast=rank0_cleanup).install(monkeypatch)
    monkeypatch.setattr(
        "graspo.flow.lora.lora_io.prepare_output_dir",
        lambda output_dir, *, overwrite: (observed.append(True), Path(output_dir))[1],
    )

    returned = prepare_output_dir_once(out, overwrite=True, is_primary=False)

    assert fake.calls == [0, 0]
    assert observed == [], "等待期间不得由非 primary rank 自己动手"
    assert returned == Path(out)
    assert returned.is_dir() and list(returned.iterdir()) == [], "返回时目录已就绪（空）"


def test_primary_clears_and_recreates_populated_dir(tmp_path, monkeypatch):
    out = _populate(tmp_path / "T030")
    fake = _FakeDist().install(monkeypatch)

    returned = prepare_output_dir_once(out, overwrite=True, is_primary=True)

    assert returned == Path(out)
    assert returned.is_dir()
    assert list(returned.iterdir()) == [], "overwrite=True 仍然清掉上次全部产物"
    assert fake.calls == [0, 0]


def test_primary_raises_when_overwrite_not_authorized_and_all_ranks_fail(tmp_path, monkeypatch):
    """rank0 的 FileExistsError 被广播 ⇒ 每个 rank 都收到同一判决（fail-closed，不干等）。"""
    out = _populate(tmp_path / "T030")
    fake = _FakeDist().install(monkeypatch)

    with pytest.raises(FileExistsError) as excinfo:
        prepare_output_dir_once(out, overwrite=False, is_primary=True)
    assert "overwrite_output_dir" in str(excinfo.value)

    # 非 primary 侧：rank0 判定失败（status=1）由广播带回 ⇒ 同样抛 FileExistsError
    seen: list[int] = []

    def broadcast_with_primary_status(tensor, *, src=0, group=None):
        seen.append(1)
        if len(seen) == 1:
            tensor.fill_(1)  # rank0 的判决：目录非空且未授权覆盖
        return None

    monkeypatch.setattr(dist, "broadcast", broadcast_with_primary_status, raising=False)
    monkeypatch.setattr(
        "graspo.flow.lora.lora_io.prepare_output_dir",
        lambda output_dir, *, overwrite: Path(output_dir),
    )
    with pytest.raises(FileExistsError):
        prepare_output_dir_once(out, overwrite=False, is_primary=False)


def test_primary_hard_failure_is_broadcast_as_runtime_error(tmp_path, monkeypatch):
    """rank0 遇到非预期异常（如权限）⇒ 其余 rank 收到 RuntimeError，而不是在集合点干等。"""
    out = tmp_path / "T030"
    out.mkdir()
    _FakeDist().install(monkeypatch)

    def boom(output_dir, *, overwrite):
        raise PermissionError("EPERM: /out/T030")

    monkeypatch.setattr("graspo.flow.lora.lora_io.prepare_output_dir", boom)
    with pytest.raises(RuntimeError, match="rank0 准备输出目录失败"):
        prepare_output_dir_once(out, overwrite=True, is_primary=True)

    text = "PermissionError: EPERM: /out/T030"

    seen: list[int] = []

    def broadcast_with_failure(tensor, *, src=0, group=None):
        seen.append(1)
        if len(seen) == 1:
            tensor.fill_(2)
        else:
            payload = torch.zeros_like(tensor)
            payload[: len(text)] = torch.tensor(list(text.encode("utf-8")), dtype=tensor.dtype)
            tensor.copy_(payload)
        return None

    monkeypatch.setattr(dist, "broadcast", broadcast_with_failure, raising=False)
    with pytest.raises(RuntimeError) as excinfo:
        prepare_output_dir_once(out, overwrite=True, is_primary=False)
    assert "EPERM" in str(excinfo.value)


def test_single_card_without_process_group_is_a_passthrough(tmp_path, monkeypatch):
    """单卡（进程组未初始化）⇒ 行为与旧实现逐字相同。"""
    out = _populate(tmp_path / "T028")
    monkeypatch.setattr(dist, "is_available", lambda: True)
    monkeypatch.setattr(dist, "is_initialized", lambda: False)

    returned = prepare_output_dir_once(out, overwrite=True, is_primary=True)
    assert returned.is_dir() and list(returned.iterdir()) == []

    out2 = _populate(tmp_path / "T028b")
    monkeypatch.setattr(dist, "is_initialized", lambda: False)
    with pytest.raises(FileExistsError):
        prepare_output_dir_once(out2, overwrite=False, is_primary=True)
