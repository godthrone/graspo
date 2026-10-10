"""``graspo.eval.guard`` 的单测：评测链路的锁卡防呆必须 fail-closed。

这些测试是**安全回归**：任何一条挂掉都意味着"评测链路可能跑到允许集合之外的卡上"
或"显存采样可能混入非目标卡读数"。

★ 卡位契约（B7 收口后正式口径）：**允许卡 = ``{4,5,6,7}``；其余一律拒绝**。
0–3 属第三方工作负载（先到先得，**不是**"保留卡"）；≥8 不存在。
「生产卡 / reserved」语义**已退役**。故本文件的负例一律取**允许集合之外**的取值
（0–3 / ≥8），正例一律取集合之内——⛔ 不再有"6/7 是生产卡所以拒绝"的用例。
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
from graspo.eval.guard import (
    ALLOWED_GPU_INDICES,
    MAX_GPU_COUNT,
    RESERVED_GPU_INDICES,
    repo_relative,
    sample_device_memory_mib,
)


def test_allowed_set_and_limit_come_from_core_guard():
    """边界常量必须与 core.gpu_guard 一致（单一真相源，不得各自定义）。

    ★ B7：允许集合是**正向白名单** ``{4,5,6,7}`` 本身，⛔ 不是 ``range(max+1)``
    （后者会把 0–3 也算成允许卡）；「保留卡」已退役 ⇒ 别名必须为空元组。
    """
    from graspo.core.gpu_guard import ALLOWED_MAX_INDEX, MAX_CARDS, RESERVED_INDICES

    assert ALLOWED_GPU_INDICES == frozenset(ALLOWED_INDICES) == frozenset({4, 5, 6, 7})
    assert ALLOWED_MAX_INDEX == max(ALLOWED_INDICES)
    assert MAX_GPU_COUNT == MAX_CARDS
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
        "0",  # 低于允许集合（第三方先到先得，不是"保留卡"）
        "8",  # 高于允许集合（8 卡机只有 0–7）
        "0,4",  # 一张越界一张合法 ⇒ 整组拒绝
        "3,7",  # 同上（另一侧）
        "1,2",  # 整组越界
    ],
)
def test_cards_outside_the_allowlist_are_always_rejected(raw):
    """允许集合 ``{4,5,6,7}`` 之外的卡——任何组合里出现都必须拒绝（fail-closed）。"""
    with pytest.raises(GpuGuardError):
        resolve_gpu_plan(raw)


@pytest.mark.parametrize("raw", ["4", "5", "6", "7"])
def test_every_allowlist_card_is_accepted(raw):
    """★ 正向对照：4/5/6/7 每一张都**必须被接受**。

    这条与上面的负例成对存在，用来证明负例**不是同义反复**：
    同一判据在集合内接受、在集合外拒绝。特别是 6/7——它们**不再是"生产卡"**。
    """
    assert resolve_gpu_plan(raw).devices == (int(raw),)


def test_more_than_four_cards_is_rejected():
    """5 张卡（必然是"允许集合外 × 卡数超限"的混合输入）→ 拒绝。

    ★ 登记：现行白名单只有 4 个元素 ⇒ **纯"超 4 卡"输入在显式卡号路径上不可构造**
    （5 张互不相同的卡必然含集合外卡号，先被白名单分支拒绝）。``MAX_CARDS`` 分支
    在此路径上**已不可达**（B7 只登记、⛔ 未删）。
    """
    with pytest.raises(GpuGuardError):
        resolve_gpu_plan("4,5,6,7,0")


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
