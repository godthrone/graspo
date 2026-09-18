"""graspo.core.gpu_guard 的单元测试——锁卡守卫的负向覆盖（安全关键）。

必须覆盖五种情形：未设置 / 含 6 / 含 7 / 5 卡 / 合法 4 卡。
纯逻辑测试，不触 GPU、不读真实环境（env 通过 mapping 注入）。
"""

import pytest

from graspo.core.gpu_guard import (
    GpuLockError,
    assert_gpu_lock,
    assert_gpu_lock_from_env,
    parse_device_list,
    select_sample_targets,
)


def _reason(exc: pytest.ExceptionInfo[GpuLockError]) -> str:
    return str(exc.value)


# ── 必须覆盖的五种情形 ──────────────────────────────────────────────────────


def test_rejects_when_visible_devices_unset():
    """未设置 NVIDIA_VISIBLE_DEVICES → 拒绝（fail-closed）。"""
    with pytest.raises(GpuLockError) as exc:
        assert_gpu_lock_from_env({})
    assert "未设置" in _reason(exc)
    assert "正确用法" in _reason(exc)


def test_rejects_when_devices_contain_gpu6():
    """含生产卡 GPU6 → 拒绝。"""
    with pytest.raises(GpuLockError) as exc:
        assert_gpu_lock("0,1,6")
    assert "6" in _reason(exc)
    assert "生产卡" in _reason(exc)


def test_rejects_when_devices_contain_gpu7():
    """含生产卡 GPU7 → 拒绝。"""
    with pytest.raises(GpuLockError) as exc:
        assert_gpu_lock("7")
    assert "7" in _reason(exc)


def test_rejects_when_five_cards_requested():
    """5 卡超过上限 → 拒绝。"""
    with pytest.raises(GpuLockError) as exc:
        assert_gpu_lock("0,1,2,3,4")
    assert "超过上限" in _reason(exc)


def test_accepts_legal_four_cards():
    """合法 4 卡（首选 {0,1,2,3}）→ 通过，返回设备元组。"""
    assert assert_gpu_lock("0,1,2,3") == (0, 1, 2, 3)
    assert assert_gpu_lock("0, 4, 5, 2") == (0, 4, 5, 2)
    assert assert_gpu_lock_from_env({"NVIDIA_VISIBLE_DEVICES": "1,2"}) == (1, 2)


# ── 其它非法取值 ────────────────────────────────────────────────────────────


@pytest.mark.parametrize("raw", ["all", "", "  ", "void", "GPU-abc", "0,,1", "0,0"])
def test_parse_rejects_non_explicit_or_malformed(raw):
    """all / 空 / void / UUID / 空元素 / 重复卡号 → 一律拒绝。"""
    with pytest.raises(GpuLockError):
        parse_device_list(raw)


def test_rejects_six_cards_with_reserved_first():
    """含 6/7 的错误优先于卡数错误——先报生产卡。"""
    with pytest.raises(GpuLockError) as exc:
        assert_gpu_lock("0,1,2,3,4,7")
    assert "7" in _reason(exc)


# ── 采样目标选择（可信显存采样的边界）──────────────────────────────────────


def test_select_sample_targets_defaults_to_visible():
    """未显式指定时，采样目标 = 可见卡。"""
    assert select_sample_targets(None, "0,3") == (0, 3)
    assert select_sample_targets("", "0,3") == (0, 3)
    assert select_sample_targets("all", "0,3") == (0, 3)


def test_select_sample_targets_rejects_outside_visible():
    """显式采样卡越出可见集 → 拒绝（防混入生产卡）。"""
    with pytest.raises(GpuLockError) as exc:
        select_sample_targets("0,6", "0,3")
    assert "越出可见卡" in _reason(exc)


def test_select_sample_targets_requires_visible():
    """可见集本身未设置 → 拒绝。"""
    with pytest.raises(GpuLockError):
        select_sample_targets(None, None)
