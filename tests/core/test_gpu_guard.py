"""graspo.core.gpu_guard 的单元测试——锁卡守卫的负向覆盖（安全关键）。

★ 卡位契约（**B10 再推导**，主席 2026-10-12 批准）：
**允许卡 = 本机全卡 0–7（先到先得）；唯一要求 = 不破坏别人的服务**。
旧的「只允许 ``{4,5,6,7}`` / GPU0–3 留给他人」白名单**已退役**；≥8 不存在；
「生产卡 / reserved」语义更早已退役。

★ 空闲判据（B10 再推导）：**可用 ⟺ 剩余显存 ≥ 本档头寸 ∧ util == 0**；
旧绝对阈值（``memory.used ≤ 64 MiB``）**已作废**。起跑前须**连续采样**（≥2 帧）全满足；
运行期另用「非我方 PID / 显存跌破余量」判退让（我们自己会让 util>0）。

纯逻辑测试，不触 GPU、不读真实环境（env 通过 mapping 注入）。

**F-1 追加（守卫与容器 runtime 语义不一致）**：nvidia-container-runtime 按设备
收窄可见集时会把容器内 ``NVIDIA_VISIBLE_DEVICES`` 覆写成哨兵 ``void``。旧守卫把它
判为非法 ⇒ 目标 GPU 服务器上所有训练入口必然拒绝启动（false reject）。修后：
``void`` + 实测可见卡（≤4）→ 通过；``void`` + 实测 8 卡 → 仍拒绝；
未设置/``all`` → 仍拒绝；显式 ``4,5,0`` → 通过（0 现在也在允许集合内）。
"""

import pytest

from graspo.core.gpu_guard import (
    ALLOWED_INDICES,
    DEFAULT_GPU_POSITION_MIB,
    GpuInventory,
    GpuLockError,
    assert_card_available,
    assert_gpu_lock,
    assert_samples_all_idle,
    evaluate_runtime_yield,
    parse_device_list,
    resolve_device_source,
    resolve_gpu_position_mib,
    resolve_gpu_yield_margin_mib,
)
from graspo.flow.guard import (
    assert_gpu_idle,
    assert_gpu_idle_sampled,
    assert_gpu_lock_from_env,
    assert_gpu_lock_inventory,
    is_runtime_managed,
    require_gpu_lock_or_exit,
    select_sample_targets,
)


def _reason(exc: pytest.ExceptionInfo[GpuLockError]) -> str:
    return str(exc.value)


# ── B10：默认全卡 + env 覆盖 ────────────────────────────────────────────────


def test_default_allowlist_is_the_full_card_set():
    """★B10：默认允许集合 = 全卡 0–7（不再限制卡号）。"""
    assert tuple(ALLOWED_INDICES) == tuple(range(8))


def test_rejects_when_visible_devices_unset():
    """未设置 NVIDIA_VISIBLE_DEVICES → 拒绝（fail-closed，不放松）。"""
    with pytest.raises(GpuLockError) as exc:
        assert_gpu_lock_from_env({})
    assert "未设置" in _reason(exc)
    assert "正确用法" in _reason(exc)


def test_cards_below_the_old_allowlist_are_now_accepted():
    """★B10 再推导：0–3 **不再**被拒（旧用例 ``"4,5,0"`` 断言拒绝，语义已失效）。"""
    assert assert_gpu_lock("0,1,2,3") == (0, 1, 2, 3)
    assert assert_gpu_lock("4,5,0") == (4, 5, 0)
    assert assert_gpu_lock_from_env({"NVIDIA_VISIBLE_DEVICES": "3,7"}) == (3, 7)


def test_rejects_when_devices_contain_a_card_above_the_allowlist():
    """含允许集合**之上**的卡（GPU8 起不存在）→ 拒绝。"""
    with pytest.raises(GpuLockError) as exc:
        assert_gpu_lock("8")
    assert "8" in _reason(exc)
    assert "白名单外" in _reason(exc)


def test_env_override_narrows_the_allowlist():
    """★B10：``GRASPO_ALLOWED_GPU_INDICES`` 覆盖生效（唯一解析入口）。"""
    env = {"NVIDIA_VISIBLE_DEVICES": "0", "GRASPO_ALLOWED_GPU_INDICES": "4,5,6,7"}
    with pytest.raises(GpuLockError) as exc:
        assert_gpu_lock_from_env(env)
    assert "白名单外" in _reason(exc)

    ok = {"NVIDIA_VISIBLE_DEVICES": "4", "GRASPO_ALLOWED_GPU_INDICES": "4,5,6,7"}
    assert assert_gpu_lock_from_env(ok) == (4,)


def test_env_override_rejects_malformed_value():
    """⛔ fail-closed：env 取值非法（非整数）一律拒绝，不静默回落。"""
    env = {"NVIDIA_VISIBLE_DEVICES": "0", "GRASPO_ALLOWED_GPU_INDICES": "0,GPU-x"}
    with pytest.raises(GpuLockError) as exc:
        assert_gpu_lock_from_env(env)
    assert "取值非法" in _reason(exc)


def test_rejects_when_five_cards_requested():
    """★B10 再推导：0–7 全在集合内 ⇒ 5 张卡首次真正走到 **卡数上限** 分支。"""
    with pytest.raises(GpuLockError) as exc:
        assert_gpu_lock("0,1,2,3,4")
    assert "超过上限" in _reason(exc)


def test_accepts_legal_four_cards():
    """合法 4 卡（0–7 内任取）→ 通过，返回设备元组（保序）。"""
    assert assert_gpu_lock("4,5,6,7") == (4, 5, 6, 7)
    assert assert_gpu_lock("5, 4, 7, 6") == (5, 4, 7, 6)
    assert assert_gpu_lock("0,1,2,3") == (0, 1, 2, 3)
    assert assert_gpu_lock_from_env({"NVIDIA_VISIBLE_DEVICES": "4,5"}) == (4, 5)


# ── ★ 防线分层（2026-10-16 补）：运行期守卫**不做**禁用卡过滤 ────────────────
# 背景：`scripts/gpu_lock_guard.py` 的 docstring 曾声称"含 6/7 或 >4 卡即拒绝"，
# 而实现（本文件被测的 `assert_gpu_lock`）**只有**四条判据（显式设置 / 允许集合
# ⊆0..7 / ≤4 卡 / 空元素·重复卡）。实测 `--visible 2` / `2,3` / `6` / `6,7` /
# `4,5,6,7` **全 rc=0**。
# ★ 处置口径（2026-10-16 指挥裁定；★ **不推翻 B10**）：运行期守卫语义保持不动
#   （B10 = 2026-10-12 主席批准「允许卡 = 本机全卡 0–7、先到先得；生产卡/预留卡
#   语义已退役」）⇒ 只把 docstring 改回**已批准口径** ＋ 把"邻卡/生产锁卡的拦截
#   面"写清在三层：① 清单生成器 `NEIGHBOR_CARDS=(2,)` / `LOCKED_CARDS=(2,6,7)`；
#   ② 投放冒烟 (b2)（全 54 档声明不得含 GPU0/GPU2）；③ `gpu_assignment.check_gpus()`。
# 下面两条**只新增**、⛔ 不改任何既有期望值：它们把"本层不拦"这一契约钉住，使
# "把禁用卡判据塞进运行期守卫"这类改法必须显式过测试（⇒ 必须显式过裁定），而不是
# 静默改变运行期语义（那等于推翻 B10，须主席批准）。


def _load_module_from_path(name: str, relpath: str):
    """按文件路径加载仓内模块（与 `scripts/gpu_lock_guard.py` 同款做法）。"""
    import importlib.util
    from pathlib import Path as _Path

    root = _Path(__file__).resolve().parents[2]
    source = root / relpath
    spec = importlib.util.spec_from_file_location(name, source)
    assert spec is not None and spec.loader is not None, f"加载失败：{source}"
    module = importlib.util.module_from_spec(spec)
    import sys as _sys

    _sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_docstring_names_the_four_judgements_and_disclaims_forbidden_cards():
    """★ 契约钉住：docstring 与实现一致（四条判据），且**明写本层不拦 2/6/7**。"""
    guard_cli = _load_module_from_path("_gpu_lock_guard_cli_doc", "scripts/gpu_lock_guard.py")
    doc = guard_cli.__doc__ or ""
    for needle in (
        "必须显式设置",
        "允许集合",
        "卡数 ≤ 4",
        "重复卡",
        '不做任何"禁用卡号"过滤',
        "NEIGHBOR_CARDS",
        "LOCKED_CARDS",
        "GPU_FORBIDDEN_PRODUCTION",
    ):
        assert needle in doc, f"docstring 缺少防线分层要素：{needle!r}"
    assert "含 6/7 或 >4 卡即拒绝" not in doc, "旧的过度承诺句仍在 ⇒ 文档未回到已批准口径"


def test_neighbour_and_production_cards_map_exactly_to_generator_side_constants():
    """★ 三方常量一致性：运行期允许集合 = 生成器侧「`GPU_ALLOWED` ∪ 禁用集合」。

    生成器侧：`gpu_assignment.GPU_ALLOWED = (0..5)` /
    `GPU_FORBIDDEN_PRODUCTION = (6, 7)`；`generate_matrix_manifest_source` 另有
    `NEIGHBOR_CARDS = (2,)` / `LOCKED_CARDS = (2, 6, 7)`。
    ⇒ 「运行期允许 0–7」与「生成/投放侧禁 2/6/7」**不是矛盾，而是分层**：本用例把
    这个等式钉住，任一侧改动都会在这里现形（而不是靠两条线各自的口头约定）。
    """
    import sys

    assignment = _load_module_from_path("gpu_assignment", "tests/e2e/gpu_assignment.py")
    sys.modules.setdefault("gpu_assignment", assignment)
    genv2 = _load_module_from_path(
        "generate_matrix_manifest_source", "tests/e2e/generate_matrix_manifest_source.py"
    )
    locked = set(assignment.GPU_FORBIDDEN_PRODUCTION) | set(genv2.NEIGHBOR_CARDS)
    assert locked == {2, 6, 7}, f"禁用集合 ≠ {{2,6,7}}：{sorted(locked)}"
    assert tuple(genv2.LOCKED_CARDS) == (2, 6, 7)
    assert set(ALLOWED_INDICES) == set(assignment.GPU_ALLOWED) | locked, (
        "运行期允许集合 ≠ 生成器侧「允许卡 ∪ 禁用卡」⇒ 两侧口径分叉，需显式裁定"
    )


# ── 其它非法取值 ────────────────────────────────────────────────────────────


@pytest.mark.parametrize("raw", ["all", "", "  ", "GPU-abc", "0,,1", "0,0"])
def test_parse_rejects_non_explicit_or_malformed(raw):
    """all / 空 / UUID / 空元素 / 重复卡号 → 一律拒绝（显式卡号通道）。"""
    with pytest.raises(GpuLockError):
        parse_device_list(raw)


def test_parse_device_list_rejects_runtime_marker_because_it_wants_host_indices():
    """``parse_device_list`` 是"宿主卡号"通道：``void`` 不是卡号，必须拒绝。"""
    with pytest.raises(GpuLockError) as exc:
        parse_device_list("void")
    assert "托管哨兵" in _reason(exc)


def test_card_count_limit_is_reachable_and_checked_after_the_allowlist():
    """★B10 再推导：集合外卡仍先于卡数被拒；而 0–7 内的 6 张卡走卡数分支。

    （旧用例名是 "out_of_allowlist_error_precedes_the_card_count_check"，
    在白名单只有 4 个元素时"6 张合法卡"不可构造；B10 后可构造，两个分支都能钉住。）
    """
    with pytest.raises(GpuLockError) as exc:
        assert_gpu_lock("0,1,2,3,4,8")  # 集合外 + 超卡数
    assert "白名单外" in _reason(exc)
    assert "超过上限" not in _reason(exc)

    with pytest.raises(GpuLockError) as exc2:
        assert_gpu_lock("0,1,2,3,4,5")  # 全在集合内 + 超卡数
    assert "超过上限" in _reason(exc2)


# ── 采样目标选择（可信显存采样的边界）──────────────────────────────────────


def test_select_sample_targets_defaults_to_visible():
    """未显式指定时，采样目标 = 可见卡。"""
    assert select_sample_targets(None, "4,6") == (4, 6)
    assert select_sample_targets("", "4,6") == (4, 6)
    assert select_sample_targets("all", "4,6") == (4, 6)
    assert select_sample_targets(None, "0,1") == (0, 1)


def test_select_sample_targets_rejects_outside_visible():
    """显式采样卡越出可见集 → 拒绝（防混入非目标卡读数）。"""
    with pytest.raises(GpuLockError) as exc:
        select_sample_targets("4,7", "4,6")
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
    """★F-1 核心回归：``void`` + 实测 1 卡 → **不再误拒**。"""
    inventory = GpuInventory(source="nvidia-smi -L", count=1, indices=(0,))
    assert assert_gpu_lock_inventory("void", inventory) == (0,)


def test_void_with_four_visible_cards_passes_and_reports_local_indices():
    """``void`` + 实测 4 卡（上限）→ 通过，返回容器内本地序号。"""
    inventory = GpuInventory(source="nvidia-smi -L", count=4, indices=(0, 1, 2, 3))
    assert assert_gpu_lock_inventory("void", inventory) == (0, 1, 2, 3)


def test_void_with_eight_visible_cards_is_still_rejected():
    """★不放松"不得超 4 卡"：runtime 没收住（可见 8 张卡）时必须拒绝。"""
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


def test_explicit_devices_still_reject_out_of_allowlist_cards_with_inventory_present():
    """★不放松"必须落在允许集合内"：即使同时给了实测卡，显式宿主卡号仍按真实卡号判。"""
    inventory = GpuInventory(source="nvidia-smi -L", count=2, indices=(4, 5))
    with pytest.raises(GpuLockError) as exc:
        assert_gpu_lock_inventory("4,5,8", inventory)
    assert "白名单外" in _reason(exc)


def test_explicit_declared_count_must_match_observed_count():
    """可见卡数 ≠ 声明卡数 → 拒绝（runtime 收窄不改变卡数）。"""
    inventory = GpuInventory(source="nvidia-smi -L", count=4, indices=(4, 5, 6, 7))
    with pytest.raises(GpuLockError) as exc:
        assert_gpu_lock_inventory("4,5", inventory)
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

    assert require_gpu_lock_or_exit({"NVIDIA_VISIBLE_DEVICES": "4,5"}, inventory_probe=probe) == (
        4,
        5,
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


# ── B10：起跑前可用断言（头寸 + util 双判据）────────────────────────────────


def test_position_defaults_to_48_gib_and_is_env_overridable():
    """★B10：头寸默认 48 GiB；``GRASPO_GPU_POSITION_MIB`` 覆盖生效；非法值 fail-closed。"""
    assert DEFAULT_GPU_POSITION_MIB == 48 * 1024
    assert resolve_gpu_position_mib({}) == 48 * 1024
    assert resolve_gpu_position_mib({"GRASPO_GPU_POSITION_MIB": "65536"}) == 65536.0
    with pytest.raises(GpuLockError):
        resolve_gpu_position_mib({"GRASPO_GPU_POSITION_MIB": "0"})
    with pytest.raises(GpuLockError):
        resolve_gpu_position_mib({"GRASPO_GPU_POSITION_MIB": "abc"})


def test_idle_accepts_when_free_memory_meets_the_position():
    """空载卡：剩余 ≈80 GiB ≥ 头寸 48 GiB 且 util=0 → 通过。"""
    assert_gpu_idle(0, free_mib=80000.0, utilization_pct=0.0)
    assert_gpu_idle(0, free_mib=49152.0, utilization_pct=0.0)  # 边界：恰好等于头寸


def test_idle_accepts_neighbour_resident_but_not_computing():
    """★B10 核心再推导：邻居**常驻但不在算**（剩余 60 GiB、util=0）⇒ **通过**。

    旧判据（``memory.used > 64 MiB`` 即拒）在此误拒；新判据只看"剩余 ≥ 头寸 ∧ util==0"。
    """
    assert_gpu_idle(3, free_mib=60 * 1024, utilization_pct=0.0)


def test_idle_rejects_when_free_memory_below_the_position():
    """★B10：剩余显存 < 头寸 → 拒绝（消息点明头寸）。"""
    with pytest.raises(GpuLockError) as exc:
        assert_gpu_idle(3, free_mib=40 * 1024, utilization_pct=0.0)
    assert "GPU3" in _reason(exc)
    assert "头寸" in _reason(exc)


def test_idle_rejects_nonzero_utilization():
    """util != 0（邻居在算）→ 拒绝，即便剩余显存远超头寸。"""
    with pytest.raises(GpuLockError) as exc:
        assert_gpu_idle(2, free_mib=80 * 1024, utilization_pct=90.0)
    assert "利用率" in _reason(exc)


def test_idle_position_can_be_passed_explicitly_or_via_env():
    """按档声明头寸：显式入参优先；env 亦可（两者都不满足时拒绝）。"""
    assert_gpu_idle(1, free_mib=40 * 1024, utilization_pct=0.0, position_mib=32 * 1024)
    with pytest.raises(GpuLockError):
        assert_gpu_idle(
            1,
            free_mib=40 * 1024,
            utilization_pct=0.0,
            env={"GRASPO_GPU_POSITION_MIB": str(64 * 1024)},
        )


def test_idle_requires_at_least_two_consecutive_frames():
    """★B10：连续采样**至少 2 帧**——单帧会被"邻居正好不在算"骗过。"""
    with pytest.raises(GpuLockError) as exc:
        assert_samples_all_idle(0, [(80000.0, 0.0)], position_mib=48 * 1024)
    assert "帧数不足" in _reason(exc)


def test_idle_sampled_accepts_when_every_frame_is_ok():
    """连续采样全部满足 → 通过（注入 sleeper，免真等待）。"""
    frames = iter([(80000.0, 0.0), (79000.0, 0.0), (78500.0, 0.0)])
    slept: list[float] = []
    assert_gpu_idle_sampled(
        0,
        lambda _idx: next(frames),
        samples=3,
        position_mib=48 * 1024,
        sleeper=slept.append,
    )
    assert slept == [12.0, 12.0]


def test_idle_sampled_rejects_when_any_frame_is_busy():
    """★B10：**只要有一帧**不满足就拒绝（防"两次 kernel 之间"）。"""
    frames = iter([(80000.0, 0.0), (80000.0, 45.0), (80000.0, 0.0)])
    with pytest.raises(GpuLockError) as exc:
        assert_gpu_idle_sampled(
            1, lambda _idx: next(frames), samples=3, position_mib=48 * 1024, sleeper=lambda _s: None
        )
    assert "第 2/3 帧" in _reason(exc)


def test_assert_card_available_reports_both_reasons():
    """双判据同时不满足时，两条原因都要报出来。"""
    with pytest.raises(GpuLockError) as exc:
        assert_card_available(5, free_mib=1000.0, utilization_pct=99.0, position_mib=48 * 1024)
    text = _reason(exc)
    assert "头寸" in text and "利用率" in text


# ── B10：运行期退让（util==0 只是起跑前判据）────────────────────────────────


def test_runtime_yield_when_a_foreign_pid_appears():
    """★B10①：本卡出现**非我方 PID** 的计算进程 ⇒ 退让。"""
    decision = evaluate_runtime_yield(gpu_index=4, foreign_compute_pids=[12345])
    assert decision.should_yield is True
    assert "非我方计算进程" in decision.reason
    assert "12345" in decision.reason


def test_runtime_yield_when_free_memory_drops_below_the_margin():
    """★B10②：剩余显存跌破安全余量 ⇒ 退让/停跑上报。"""
    decision = evaluate_runtime_yield(gpu_index=4, free_mib=1024.0, margin_mib=8 * 1024)
    assert decision.should_yield is True
    assert "安全余量" in decision.reason


def test_runtime_yield_is_quiet_when_healthy():
    """★B10：无外来 PID 且剩余充足 ⇒ **不退让**（我们自己在算 util>0 是正常的）。"""
    decision = evaluate_runtime_yield(gpu_index=4, foreign_compute_pids=(), free_mib=40 * 1024)
    assert decision.should_yield is False
    assert decision.reason == ""


def test_runtime_yield_margin_defaults_and_env_override():
    """安全余量默认 8 GiB，``GRASPO_GPU_YIELD_MARGIN_MIB`` 可覆盖；负数 fail-closed。"""
    assert resolve_gpu_yield_margin_mib({}) == 8 * 1024
    assert resolve_gpu_yield_margin_mib({"GRASPO_GPU_YIELD_MARGIN_MIB": "2048"}) == 2048.0
    with pytest.raises(GpuLockError):
        resolve_gpu_yield_margin_mib({"GRASPO_GPU_YIELD_MARGIN_MIB": "-1"})
