"""graspo.core.gpu_guard 的单元测试——锁卡守卫的负向覆盖（安全关键）。

必须覆盖五种情形：未设置 / 含 6 / 含 7 / 5 卡 / 合法 4 卡。
纯逻辑测试，不触 GPU、不读真实环境（env 通过 mapping 注入）。

**F-1 追加（守卫与容器 runtime 语义不一致）**：nvidia-container-runtime 按设备
收窄可见集时会把容器内 ``NVIDIA_VISIBLE_DEVICES`` 覆写成哨兵 ``void``。旧守卫把它
判为非法 ⇒ 目标 GPU 服务器上所有训练入口必然拒绝启动（false reject）。修后：
``void`` + 实测可见卡（≤4）→ 通过；``void`` + 实测 8 卡 → 仍拒绝；
未设置/``all`` → 仍拒绝；显式 ``0,1,6`` → 仍拒绝（6/7 由宿主侧按真实卡号判）。
"""

import pytest

from graspo.core.gpu_guard import (
    GpuInventory,
    GpuLockError,
    assert_gpu_idle,
    assert_gpu_lock,
    assert_gpu_lock_from_env,
    assert_gpu_lock_inventory,
    is_runtime_managed,
    parse_device_list,
    require_gpu_lock_or_exit,
    resolve_device_source,
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


@pytest.mark.parametrize("raw", ["all", "", "  ", "GPU-abc", "0,,1", "0,0"])
def test_parse_rejects_non_explicit_or_malformed(raw):
    """all / 空 / UUID / 空元素 / 重复卡号 → 一律拒绝（显式卡号通道）。"""
    with pytest.raises(GpuLockError):
        parse_device_list(raw)


def test_parse_device_list_rejects_runtime_marker_because_it_wants_host_indices():
    """``parse_device_list`` 是"宿主卡号"通道：``void`` 不是卡号，必须拒绝。

    这不是 F-1 的回退点——该走实测可见卡的调用方用
    :func:`assert_gpu_lock_inventory` / :func:`select_sample_targets_for_inventory`。
    """
    with pytest.raises(GpuLockError) as exc:
        parse_device_list("void")
    assert "托管哨兵" in _reason(exc)


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


# ── F-1：runtime 哨兵（void）按实测可见卡判定 ────────────────────────────────


def test_void_is_recognised_as_runtime_managed_not_as_unset():
    """``void``/``none`` 是 runtime 哨兵（不是"没锁卡"）；其它取值不是。"""
    assert is_runtime_managed("void")
    assert is_runtime_managed(" VOID ")
    assert is_runtime_managed("none")
    assert not is_runtime_managed("all")
    assert not is_runtime_managed(None)
    assert not is_runtime_managed("")


def test_void_with_single_visible_card_passes_the_guard():
    """★F-1 核心回归：``void`` + 实测 1 卡 → **不再误拒**（目标 GPU 服务器上训练入口能起来了）。

    实测事实：runtime 已把可见卡收窄为 1 张（`nvidia-smi -L` 一行、
    `torch.cuda.device_count()==1`），旧守卫却因 env 取值 `void` 拒绝启动。
    """
    inventory = GpuInventory(source="nvidia-smi -L", count=1, indices=(0,))

    devices = assert_gpu_lock_inventory("void", inventory)

    assert devices == (0,)


def test_void_with_four_visible_cards_passes_and_reports_local_indices():
    """``void`` + 实测 4 卡（上限）→ 通过，返回容器内本地序号。"""
    inventory = GpuInventory(source="nvidia-smi -L", count=4, indices=(0, 1, 2, 3))

    assert assert_gpu_lock_inventory("void", inventory) == (0, 1, 2, 3)


def test_void_with_eight_visible_cards_is_still_rejected():
    """★不放松"不得超 4 卡"：runtime 没收住（可见 8 张卡，含生产 6/7）时必须拒绝。

    本地序号与宿主卡号无对应关系，所以这条断言就是"可见集失守"的兜底信号。
    """
    inventory = GpuInventory(source="nvidia-smi -L", count=8, indices=tuple(range(8)))

    with pytest.raises(GpuLockError) as exc:
        assert_gpu_lock_inventory("void", inventory)

    assert "超过上限" in _reason(exc)
    assert "8" in _reason(exc)


@pytest.mark.parametrize("raw", [None, "", "  ", "all"])
def test_void_fallback_does_not_relax_unset_or_all(raw):
    """★不放松"必须显式锁卡"：未设置 / 空 / ``all`` 即便给出实测可见卡也仍拒绝。"""
    inventory = GpuInventory(source="nvidia-smi -L", count=1, indices=(0,))

    with pytest.raises(GpuLockError):
        assert_gpu_lock_inventory(raw, inventory)


def test_void_without_inventory_is_fail_closed():
    """拿不到实测可见卡 → 拒绝（不把"测不到"当作"没锁卡"放行）。"""
    with pytest.raises(GpuLockError) as exc:
        assert_gpu_lock_inventory("void", None)

    assert "未提供实测可见卡" in _reason(exc)


def test_zero_visible_cards_is_fail_closed():
    """实测 0 卡（runtime 没生效 / 探针失败）→ 拒绝。"""
    inventory = GpuInventory(source="nvidia-smi -L", count=0, indices=())

    with pytest.raises(GpuLockError) as exc:
        assert_gpu_lock_inventory("void", inventory)

    assert "可见卡数为 0" in _reason(exc)


def test_explicit_devices_still_reject_reserved_cards_with_inventory_present():
    """★不放松"不得含 6/7"：即使同时给了实测卡，显式宿主卡号仍按真实卡号判。"""
    inventory = GpuInventory(source="nvidia-smi -L", count=2, indices=(0, 1))

    with pytest.raises(GpuLockError) as exc:
        assert_gpu_lock_inventory("0,1,6", inventory)

    assert "生产卡" in _reason(exc)


def test_explicit_declared_count_must_match_observed_count():
    """可见卡数 ≠ 声明卡数 → 拒绝（runtime 收窄不改变卡数）。"""
    inventory = GpuInventory(source="nvidia-smi -L", count=4, indices=(0, 1, 2, 3))

    with pytest.raises(GpuLockError) as exc:
        assert_gpu_lock_inventory("0,1", inventory)

    assert "!=" in _reason(exc)


def test_resolve_device_source_picks_channel_explicitly():
    """来源判定显式化：显式卡号 → explicit；``void`` → inventory（二者恰一个非 None）。"""
    explicit = resolve_device_source("0,1")
    assert explicit.explicit == (0, 1)
    assert explicit.inventory is None

    inventory = GpuInventory(source="nvidia-smi -L", count=2, indices=(0, 1))
    managed = resolve_device_source("void", inventory)
    assert managed.explicit is None
    assert managed.inventory == inventory


def test_require_gpu_lock_or_exit_uses_injected_probe_only_for_void():
    """入口包装：``void`` 走注入的探针；显式卡号**不调用**探针（宿主侧零设施调用）。"""
    calls: list[str] = []

    def probe() -> GpuInventory:
        calls.append("probe")
        return GpuInventory(source="nvidia-smi -L", count=1, indices=(0,))

    assert require_gpu_lock_or_exit({"NVIDIA_VISIBLE_DEVICES": "void"}, inventory_probe=probe) == (
        0,
    )
    assert calls == ["probe"]

    assert require_gpu_lock_or_exit({"NVIDIA_VISIBLE_DEVICES": "2,3"}, inventory_probe=probe) == (
        2,
        3,
    )
    assert calls == ["probe"], "显式卡号路径不得触发实测探测"


def test_require_gpu_lock_or_exit_exits_on_void_with_eight_visible_cards():
    """入口包装在可见集失守时 **SystemExit(1)**——不把事故放到模型加载之后。"""
    probe = lambda: GpuInventory(  # noqa: E731
        source="nvidia-smi -L", count=8, indices=tuple(range(8))
    )

    with pytest.raises(SystemExit) as exc:
        require_gpu_lock_or_exit({"NVIDIA_VISIBLE_DEVICES": "void"}, inventory_probe=probe)

    assert "超过上限" in str(exc.value.code)


def test_require_gpu_lock_or_exit_exits_when_probe_fails():
    """探针失败 → **SystemExit**（fail-closed），不是原样抛 RuntimeError traceback。"""

    def broken_probe() -> GpuInventory:
        raise RuntimeError("nvidia-smi 不可用")

    with pytest.raises(SystemExit) as exc:
        require_gpu_lock_or_exit({"NVIDIA_VISIBLE_DEVICES": "void"}, inventory_probe=broken_probe)

    assert "探测失败" in str(exc.value.code)


# ── F-10：目标卡实测空闲断言 ────────────────────────────────────────────────


def test_idle_accepts_driver_resident_footprint_below_tolerance():
    """空载卡有 driver 常驻占用（几 MiB）——64 MiB 容差内必须通过。"""
    assert_gpu_idle(0, 4.0, 0.0)
    assert_gpu_idle(0, 64.0, 5.0)  # 边界值（严格大于才拒）


def test_idle_rejects_busy_memory():
    """★F-10：目标卡被他人占用 299 MiB（实测场景：跑 4 卡作业时撞上 GPU3）→ 拒绝。"""
    with pytest.raises(GpuLockError) as exc:
        assert_gpu_idle(3, 299.0, 0.0)

    assert "GPU3" in _reason(exc)
    assert "宁等不抢" in _reason(exc)


def test_idle_rejects_high_utilization_even_with_no_memory_claim():
    """util 高但显存没涨（别人在做推理）同样拒绝——两个门槛都要看。"""
    with pytest.raises(GpuLockError) as exc:
        assert_gpu_idle(2, 0.0, 90.0)

    assert "利用率" in _reason(exc)
