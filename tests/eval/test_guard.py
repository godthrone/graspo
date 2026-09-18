"""``graspo.eval.guard`` 的单测：评测链路的锁卡防呆必须 fail-closed。

这些测试是**安全回归**：任何一条挂掉都意味着"评测链路可能踩上生产卡 GPU6/7"
或"显存采样可能混入生产读数"。
"""

from __future__ import annotations

import subprocess

import pytest

from graspo.eval.guard import (
    ALLOWED_GPU_INDICES,
    MAX_GPU_COUNT,
    RESERVED_GPU_INDICES,
    GpuGuardError,
    GpuPlan,
    repo_relative,
    resolve_gpu_plan,
    sample_device_memory_mib,
)


def test_allowed_set_and_limit_come_from_core_guard():
    """边界常量必须与 core.gpu_guard 一致（单一真相源，不得各自定义）。"""
    from graspo.core.gpu_guard import ALLOWED_MAX_INDEX, MAX_CARDS

    assert ALLOWED_GPU_INDICES == frozenset(range(ALLOWED_MAX_INDEX + 1))
    assert MAX_GPU_COUNT == MAX_CARDS
    assert 6 in RESERVED_GPU_INDICES and 7 in RESERVED_GPU_INDICES


@pytest.mark.parametrize("raw", [None, "", "   "])
def test_unset_device_list_is_rejected(raw):
    """未显式给卡 → 拒绝启动（fail-closed）。没有默认值是有意的。"""
    with pytest.raises(GpuGuardError):
        resolve_gpu_plan(raw)


@pytest.mark.parametrize("raw", ["all", "void", "none"])
def test_sentinel_device_lists_are_rejected(raw):
    with pytest.raises(GpuGuardError):
        resolve_gpu_plan(raw)


@pytest.mark.parametrize("raw", ["6", "7", "0,6", "5,7", "6,7"])
def test_production_cards_are_always_rejected(raw):
    """GPU6/7 是生产卡——任何组合里出现都必须拒绝。"""
    with pytest.raises(GpuGuardError):
        resolve_gpu_plan(raw)


def test_more_than_four_cards_is_rejected():
    with pytest.raises(GpuGuardError):
        resolve_gpu_plan("0,1,2,3,4")


def test_four_cards_is_accepted():
    plan = resolve_gpu_plan("0,1,2,3")
    assert plan.devices == (0, 1, 2, 3)
    assert plan.count == 4


def test_accepted_plan_exposes_docker_flag_and_csv():
    plan = resolve_gpu_plan("2,3")
    assert plan.csv == "2,3"
    assert plan.docker_gpus_flag() == '"device=2,3"'


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

    这是已确证事故的回归测试：不带 -i 的采样会把生产卡读数混进来。
    """
    captured: dict[str, object] = {}

    def fake_run(command, **kwargs):  # noqa: ANN001
        captured["command"] = command
        captured["env"] = dict(kwargs.get("env") or {})
        return subprocess.CompletedProcess(
            args=command,
            returncode=0,
            stdout="0, NVIDIA A800-SXM4-80GB, 11, 81920\n1, NVIDIA A800-SXM4-80GB, 22, 81920\n",
            stderr="",
        )

    monkeypatch.setattr(subprocess, "run", fake_run)
    readings = sample_device_memory_mib(resolve_gpu_plan("0,1"))

    command = captured["command"]
    assert isinstance(command, list)
    assert "--id=0,1" in command
    assert "6" not in ",".join(command)
    env = captured["env"]
    assert env["CUDA_VISIBLE_DEVICES"] == "0,1"
    assert [reading.index for reading in readings] == [0, 1]
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
        sample_device_memory_mib(resolve_gpu_plan("0"))


def test_sampling_reports_command_failure(monkeypatch):
    def fake_run(command, **kwargs):  # noqa: ANN001
        raise subprocess.CalledProcessError(returncode=9, cmd=command, stderr="boom")

    monkeypatch.setattr(subprocess, "run", fake_run)
    with pytest.raises(GpuGuardError, match="exit 9"):
        sample_device_memory_mib(resolve_gpu_plan("0"))


def test_sampling_rejects_unparsable_output(monkeypatch):
    def fake_run(command, **kwargs):  # noqa: ANN001
        return subprocess.CompletedProcess(
            args=command, returncode=0, stdout="garbage\n", stderr=""
        )

    monkeypatch.setattr(subprocess, "run", fake_run)
    with pytest.raises(GpuGuardError, match="no parsable device lines"):
        sample_device_memory_mib(resolve_gpu_plan("0"))


def test_repo_relative_strips_cwd_prefix(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    target = tmp_path / "a" / "b.json"
    assert repo_relative(target) == "a/b.json"
    assert repo_relative("/somewhere/else/b.json") == "/somewhere/else/b.json"
