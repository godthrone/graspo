"""``graspo.eval.guard`` 的单测：评测链路的锁卡防呆必须 fail-closed。

这些测试是**安全回归**：任何一条挂掉都意味着"评测链路可能跑到允许集合之外的卡上"
或"显存采样可能混入非目标卡读数"。

★ 卡位契约（**2026-10-17 主席令**，逐字）：「**228 先到先得，随便用。**」
＋「**单个作业不限制卡数。**」⇒ **允许卡 = 本机全卡 0–7（先到先得）**；≥8 不存在。
⛔ 不预留、⛔ 不禁用、⛔ 无"生产锁卡"、⛔ 无"邻居卡"、⛔ **无单作业卡数上限**。
故本文件的负例**只在 ≥8**（唯一"集合外"），正例覆盖 0–7 全部；
⛔ 不再有"0–3 是第三方所以拒绝""最多 4 卡"的用例。
"""

from __future__ import annotations

import subprocess

import pytest

from graspo.core.gpu_guard import (
    ALLOWED_INDICES,
    GpuGuardError,
    GpuPlan,
    resolve_gpu_plan,
)
from graspo.eval import guard as eval_guard
from graspo.eval.guard import (
    ALLOWED_GPU_INDICES,
    RESERVED_GPU_INDICES,
    repo_relative,
    sample_device_memory_mib,
)


def test_allowed_set_and_limit_come_from_core_guard():
    """边界常量必须与 core.gpu_guard 一致（单一真相源，不得各自定义）。

    ★ B10：允许集合 = ``ALLOWED_INDICES`` 本身（默认 = **本机全卡 0–7**），
    ⛔ 不是硬编码的小集合、⛔ 不是 ``range(max+1)`` 之类的第二定义；
    「保留卡」已退役 ⇒ 别名必须为空元组。
    """
    from graspo.core.gpu_guard import ALLOWED_MAX_INDEX, MAX_CARDS, RESERVED_INDICES

    assert ALLOWED_GPU_INDICES == frozenset(ALLOWED_INDICES) == frozenset(range(8))
    assert ALLOWED_MAX_INDEX == max(ALLOWED_INDICES) == 7
    # ★ 2026-10-17：评测侧**不设卡数常量**——旧透传别名已删（它本身不构成拦截，
    #   却是"这里有个卡数上限"的误解诱因）。本机卡数这类派生量直接引用真相源。
    assert not hasattr(eval_guard, "MAX_GPU_COUNT")
    from graspo.core import gpu_guard as core_guard

    assert MAX_CARDS == core_guard.MAX_CARDS == len(core_guard.ALLOWED_INDICES)
    # ★ 「生产卡 / reserved」语义退役：两处都必须为空
    assert RESERVED_GPU_INDICES == ()
    assert RESERVED_INDICES == ()


@pytest.mark.parametrize("raw", [None, "", "   "])
def test_unset_device_list_is_rejected(raw):
    """未显式给卡 → 拒绝启动（fail-closed）。没有默认值是有意的。"""
    with pytest.raises(GpuGuardError):
        resolve_gpu_plan(raw)


@pytest.mark.parametrize("raw", ["all", "void", "none"])
def test_sentinel_device_lists_are_rejected(raw):
    with pytest.raises(GpuGuardError):
        resolve_gpu_plan(raw)


@pytest.mark.parametrize(
    "raw",
    [
        "8",  # ≥8：不存在（8 卡机只有 0–7）
        "9",  # 更远的越界
        "8,4",  # 一张越界一张合法 ⇒ 整组拒绝
        "0,8",  # 同上（越界在另一侧）
        "7,8,9",  # 多张越界
    ],
)
def test_cards_outside_the_allowlist_are_always_rejected(raw):
    """★B10：允许集合 = 全卡 0–7 ⇒ **唯一在集合外的是 ≥8**；任何组合里出现都必须拒绝。"""
    with pytest.raises(GpuGuardError):
        resolve_gpu_plan(raw)


@pytest.mark.parametrize("raw", ["0", "1", "2", "3", "4", "5", "6", "7"])
def test_every_allowlist_card_is_accepted(raw):
    """★ 正向对照：**0–7 每一张**都必须被接受（B10 起 0–3 不再被拒）。

    这条与上面的负例成对存在，用来证明负例**不是同义反复**：
    同一判据在 0–7 内接受、在 ≥8 拒绝。
    """
    assert resolve_gpu_plan(raw).devices == (int(raw),)


def test_card_count_is_not_capped():
    """★★ 2026-10-17 主席令「单个作业不限制卡数」⇒ **5 张卡必须放行**。

    （本用例由旧的 `test_more_than_four_cards_is_rejected` **反转**而来：
    旧口径下 ``"0,1,2,3,4"`` 走到 ``MAX_CARDS`` 分支被拒；该平台级上限已删除。）
    ★ 没放松边界：越界卡（≥8）仍拒（见下一条用例）。
    """
    assert resolve_gpu_plan("0,1,2,3,4").devices == (0, 1, 2, 3, 4)
    assert resolve_gpu_plan("0,1,2,3,4,5,6,7").count == 8


def test_four_cards_is_accepted():
    plan = resolve_gpu_plan("4,5,6,7")
    assert plan.devices == (4, 5, 6, 7)
    assert plan.count == 4


def test_accepted_plan_exposes_docker_flag_and_csv():
    plan = resolve_gpu_plan("5,6")
    assert plan.csv == "5,6"
    assert plan.docker_gpus_flag() == '"device=5,6"'


def test_plan_is_frozen_and_typed():
    """GpuPlan 是不可变契约——防止拿错卡后又被就地改掉。"""
    plan = GpuPlan(devices=(0,))
    with pytest.raises(Exception):
        plan.devices = (1,)  # type: ignore[misc]


def test_non_integer_and_duplicate_indices_rejected():
    with pytest.raises(GpuGuardError):
        resolve_gpu_plan("0,x")
    with pytest.raises(GpuGuardError):
        resolve_gpu_plan("0,0")


def test_sampling_uses_nvidia_smi_with_explicit_id_and_pinned_visibility(monkeypatch):
    """采样必须带 --id 且把 CUDA_VISIBLE_DEVICES 钉在可见卡上。

    这是已确证事故的回归测试：不带 -i 的采样会把**非目标卡**（历史上是生产卡
    GPU6/7；现行契约下是任何未点名的卡）读数混进来。
    """
    captured: dict[str, object] = {}

    def fake_run(command, **kwargs):  # noqa: ANN001
        captured["command"] = command
        captured["env"] = dict(kwargs.get("env") or {})
        return subprocess.CompletedProcess(
            args=command,
            returncode=0,
            stdout="4, NVIDIA A800-SXM4-80GB, 11, 81920\n5, NVIDIA A800-SXM4-80GB, 22, 81920\n",
            stderr="",
        )

    monkeypatch.setattr(subprocess, "run", fake_run)
    readings = sample_device_memory_mib(resolve_gpu_plan("4,5"))

    command = captured["command"]
    assert isinstance(command, list)
    # 查询必须**精确限定**在计划卡上（不是全部 8 张、也不是 6/7）
    assert command[1] == "--id=4,5"
    env = captured["env"]
    assert env["CUDA_VISIBLE_DEVICES"] == "4,5"
    assert [reading.index for reading in readings] == [4, 5]
    assert readings[0].used_mib == 11
    assert readings[1].total_mib == 81920


def test_sampling_rejects_empty_plan():
    """没有可见卡就不能采样——不存在"那就采全部"的退路。"""
    with pytest.raises(GpuGuardError):
        sample_device_memory_mib(GpuPlan(devices=()))


def test_sampling_reports_missing_nvidia_smi_as_actionable_error(monkeypatch):
    def fake_run(command, **kwargs):  # noqa: ANN001
        raise FileNotFoundError("nvidia-smi")

    monkeypatch.setattr(subprocess, "run", fake_run)
    with pytest.raises(GpuGuardError, match="nvidia-smi not found"):
        sample_device_memory_mib(resolve_gpu_plan("4"))


def test_sampling_reports_command_failure(monkeypatch):
    def fake_run(command, **kwargs):  # noqa: ANN001
        raise subprocess.CalledProcessError(returncode=9, cmd=command, stderr="boom")

    monkeypatch.setattr(subprocess, "run", fake_run)
    with pytest.raises(GpuGuardError, match="exit 9"):
        sample_device_memory_mib(resolve_gpu_plan("4"))


def test_sampling_rejects_unparsable_output(monkeypatch):
    def fake_run(command, **kwargs):  # noqa: ANN001
        return subprocess.CompletedProcess(
            args=command, returncode=0, stdout="garbage\n", stderr=""
        )

    monkeypatch.setattr(subprocess, "run", fake_run)
    with pytest.raises(GpuGuardError, match="no parsable device lines"):
        sample_device_memory_mib(resolve_gpu_plan("4"))


def test_repo_relative_strips_cwd_prefix(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    target = tmp_path / "a" / "b.json"
    assert repo_relative(target) == "a/b.json"
    assert repo_relative("/somewhere/else/b.json") == "/somewhere/else/b.json"
