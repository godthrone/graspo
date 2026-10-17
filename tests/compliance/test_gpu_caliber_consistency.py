#!/usr/bin/env python3
"""③3 卡池口径**跨件一致性**机核 —— 把"文档/清单说的"与"代码做的"钉在一起。

## 现行口径（唯一标尺）

★★ 主席 **2026-10-17** 裁定，逐字：「**228 先到先得，随便用。**」
＋「**单个作业不限制卡数。**」
⇒ 本机全卡 0–7 一律可用（⛔ 无预留/禁用/生产锁卡/邻居卡、⛔ 无卡数上限）；
唯一保留的硬要求 =「**不抢占他人**」（起跑前逐卡实测：**剩余 ≥ 本档头寸 ∧ util == 0**）。

## 为什么需要这个文件（与姊妹机核的分工，§1.4 不重复）

- `test_gpu_contract_retired_sweep.py`：管**词表与符号**（旧口径不许留下痕迹）；
- `test_gpu_allowlist_single_source.py`：管**允许集合只有一处定义**（stdlib-only 派生例外已登记）；
- **本文件**：管**跨件相等 + 判据本身**（"允许集/卡数/空闲判据/自述"四处必须互相咬合）。
  ⛔ 不重抄姊妹机核的规则，只做**它们都不做**的那几件事。

1. **允许集合两个家必须逐值相等**：`core.gpu_guard.ALLOWED_INDICES`（真相源）
   == `tests/e2e/gpu_assignment.GPU_ALLOWED`（stdlib-only 派生，生成器/运行期用）
   == `(0..7)`。
2. **卡数上限是派生量、不是阈值**：`MAX_CARDS == len(ALLOWED_INDICES)`；
   且 `assert_gpu_lock` 的**默认**路径不判卡数（5 张、8 张都放行）。
   ★ 阳性对照：显式传 `max_cards` 时仍收窄（保留给"确实要自限"的调用方）。
3. **空闲判据 = B10**：`scripts/gpu_idle_assert.py` / `flow.guard.idle` 生效判据是
   「剩余 ≥ 本档头寸 ∧ util == 0」；旧绝对阈值（`used > 64 MiB`）**不得**再是判据。
   ★ 阳性对照：伪造 `nvidia-smi` 让目标卡被占（剩余 < 头寸 或 util != 0）⇒ **必须拒**；
     让邻居**常驻但不在算**（剩余充足 ∧ util == 0）⇒ **必须放行**（"先到先得"的直接体现）。
4. **「不抢占他人」没有退化成"杀别人"**：空闲/守卫面**不得含** kill/pkill/killall/
   `docker stop|kill|rm`/`--gpu-reset` 这类"对他人动手"的能力（本进程自有信号处理器除外）。
5. **清单自述与代码一致**：`matrix54_manifest_source.json::gpu_card_caliber` 的
   `gpu_allowed` / `max_cards_per_job` / `no_preemption` 与上面 1–3 逐值对齐。
"""
from __future__ import annotations

import importlib.util
import json
import re
import sys
import types
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]

#: 需要做"不得杀他人"扫描的**在役守卫面**（具名，⛔ 不用通配符）。
_GUARD_SURFACE = (
    "src/graspo/core/gpu_guard.py",
    "src/graspo/flow/guard/lock.py",
    "src/graspo/flow/guard/sampling.py",
    "src/graspo/flow/guard/idle.py",
    "src/graspo/cli/gpu_monitor.py",
    "scripts/gpu_lock_guard.py",
    "scripts/gpu_idle_assert.py",
    "scripts/run_guards.py",
)

#: "对**他人**动手"的能力（命中即 FAIL）。`signal.signal(...)` 是本进程自有处理器，另计。
_KILL_OTHERS_RX = re.compile(
    r"\b(pkill|killall|docker\s+(stop|kill|rm)|nvidia-smi\s+--gpu-reset|\bkill\s+-\d|\bos\.kill\b|SIGKILL)",
)
_OWN_SIGNAL_RX = re.compile(r"signal\.signal\(")


def _load_by_path(name: str, relpath: str):
    spec = importlib.util.spec_from_file_location(name, REPO / relpath)
    assert spec is not None and spec.loader is not None, relpath
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _gpu_assignment():
    """加载 `tests/e2e/gpu_assignment.py`（stdlib-only，裸解释器可加载）。"""
    return _load_by_path("_caliber_gpu_assignment", "tests/e2e/gpu_assignment.py")


# ── 1. 允许集合：两个家逐值相等 ─────────────────────────────────────────────


def test_allowed_set_is_the_same_in_the_truth_source_and_the_stdlib_derivation():
    from graspo.core.gpu_guard import ALLOWED_INDICES

    assignment = _gpu_assignment()
    assert tuple(ALLOWED_INDICES) == tuple(range(8)), "真相源允许集合 != 本机全卡 0–7"
    assert tuple(assignment.GPU_ALLOWED) == tuple(ALLOWED_INDICES), (
        "stdlib-only 派生（生成器/运行期）与真相源允许集合**分叉** ⇒ §1.4 破，必须显式裁定"
    )
    assert assignment.GPU_MAX_INDEX == max(ALLOWED_INDICES)


def test_all_machine_cards_are_accepted_and_only_out_of_range_is_rejected():
    """★ 逐卡正向覆盖 0–7 ＋ 越界（≥8）负向 → 证明"允许集合"确实只剩**存在性**判据。"""
    from graspo.core.gpu_guard import assert_gpu_lock

    for card in range(8):
        assert assert_gpu_lock(str(card)) == (card,), card
    assert assert_gpu_lock("0,1,2,3,4,5,6,7") == tuple(range(8))
    for bad in ("8", "99", "0,8"):
        with pytest.raises(Exception):
            assert_gpu_lock(bad)


# ── 2. 卡数是派生量、默认不设上限 ───────────────────────────────────────────


def test_max_cards_is_a_derived_machine_width_not_a_per_job_threshold():
    from graspo.core import gpu_guard

    assert gpu_guard.MAX_CARDS == len(gpu_guard.ALLOWED_INDICES), (
        "MAX_CARDS 必须是**派生量**（本机卡数），⛔ 不得是硬编码的作业卡数上限"
    )


def test_default_path_has_no_per_job_card_count_cap():
    """★ 主席令「单个作业不限制卡数」：默认路径 5 张 / 8 张都放行。"""
    from graspo.core.gpu_guard import assert_gpu_lock

    assert assert_gpu_lock("0,1,2,3,4") == (0, 1, 2, 3, 4)
    assert assert_gpu_lock("0,1,2,3,4,5,6,7") == tuple(range(8))
    # 阳性对照：调用方**显式**自限仍生效（默认与显式必须可区分）
    with pytest.raises(Exception) as exc:
        assert_gpu_lock("0,1,2,3,4", max_cards=4)
    assert "显式设置的自限" in str(exc.value)


def test_manifest_layer_has_no_per_job_card_count_cap():
    assignment = _gpu_assignment()
    assert not hasattr(assignment, "MAX_CARDS_PER_JOB"), "清单侧卡数上限常量复活了"
    assert assignment.check_gpus("TX", list(range(8)), 8) == tuple(range(8))
    # 阳性对照：卡数 != 该档 cards 仍拒绝（唯一的卡数判据）
    with pytest.raises(assignment.GpuAssignmentError):
        assignment.check_gpus("TX", [0, 1, 2, 3, 4], 4)


# ── 3. 空闲判据 = B10（剩余 ≥ 头寸 ∧ util == 0）─────────────────────────────


def _load_gpu_monitor():
    """按 `scripts/gpu_lock_guard.py` 同款命名空间垫片加载 `cli/gpu_monitor.py`（避开 torch）。"""
    for name, rel in (("graspo", "graspo"), ("graspo.core", "graspo/core"), ("graspo.flow", "graspo/flow")):
        if name not in sys.modules:
            m = types.ModuleType(name)
            m.__path__ = [str(REPO / "src" / rel)]
            m.__package__ = name
            sys.modules[name] = m
    guard = _load_by_path("graspo.flow.guard", "src/graspo/flow/guard/__init__.py")
    sys.modules["graspo.flow.guard"] = guard
    sys.modules["graspo.flow.guard.inventory"] = _load_by_path(
        "_caliber_ginv", "src/graspo/flow/guard/inventory.py"
    )
    return _load_by_path("graspo.cli._caliber_monitor", "src/graspo/cli/gpu_monitor.py")


def _nvidia_smi_stub(used_mib: float, util_pct: float):
    """伪 `nvidia-smi`：IDLE_FIELDS = (index, memory.used, memory.total, utilization.gpu)。"""

    def runner(argv):
        idx = "0"
        for i, a in enumerate(argv):
            if a == "-i":
                idx = argv[i + 1]
        return f"{idx}, {used_mib}, 81920, {util_pct}\n"

    return runner


def test_idle_criterion_is_free_memory_vs_position_and_util_zero():
    """★ 生效判据 = B10：剩余 ≥ 本档头寸 ∧ util == 0。"""
    gm = _load_gpu_monitor()
    # 邻居**常驻但不在算**（used 64 GiB、util=0，剩余 16 GiB < 48 GiB 头寸）⇒ **必须拒**
    with pytest.raises(Exception) as exc:
        gm.assert_target_gpus_idle(
            [3], runner=_nvidia_smi_stub(65536, 0), samples=1, interval_s=0, sleeper=lambda _s: None
        )
    assert "头寸" in str(exc.value), str(exc.value)


def test_neighbour_resident_but_not_computing_is_accepted():
    """★「先到先得」的直接体现：剩余 ≥ 头寸 ∧ util == 0 ⇒ 占用多少都放行。

    （这与旧口径「`memory.used > 64 MiB` 即拒」**直接冲突** —— 旧口径已作废。）
    """
    gm = _load_gpu_monitor()
    # used = 20 GiB、util = 0 ⇒ 剩余 ≈ 60 GiB ≥ 48 GiB 头寸 ⇒ 放行（旧口径下会被误拒）
    gm.assert_target_gpus_idle(
        [3], runner=_nvidia_smi_stub(20 * 1024, 0), samples=1, interval_s=0, sleeper=lambda _s: None
    )


def test_occupied_card_is_rejected_by_the_idle_assertion():
    """★ 阳性对照：目标卡被他人**在算**（util != 0）⇒ 必须拒（这是"不抢占他人"的判据面）。"""
    gm = _load_gpu_monitor()
    with pytest.raises(Exception) as exc:
        gm.assert_target_gpus_idle(
            [3], runner=_nvidia_smi_stub(64, 95), samples=1, interval_s=0, sleeper=lambda _s: None
        )
    msg = str(exc.value)
    assert "util" in msg and "绝不 kill" in msg, msg


# ── 4. 「不抢占他人」不得退化成"杀别人" ─────────────────────────────────────


def test_guard_surface_has_no_capability_to_kill_or_stop_others():
    """★ 守卫面（含空闲断言与择卡）**不得含**对他人动手的能力。

    允许的唯一信号用法 = 本进程自己的 `signal.signal(SIGINT/SIGTERM, ...)` 优雅退出
    （那是"别人停我们时我们收拾干净"，⛔ 不是"我们去停别人"）。
    """
    offenders, own = [], []
    for rel in _GUARD_SURFACE:
        path = REPO / rel
        if not path.is_file():
            continue
        for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if _KILL_OTHERS_RX.search(line):
                offenders.append(f"{rel}:{i}: {line.strip()}")
            elif _OWN_SIGNAL_RX.search(line):
                own.append(f"{rel}:{i}: {line.strip()}")
    assert offenders == [], f"守卫面出现『对他人动手』的能力：{offenders}"
    # 自有信号处理器允许存在，但必须真的是**本进程**的 handler（不是 kill 调用）
    for line in own:
        assert "kill" not in line.lower(), line


def test_idle_assertion_message_points_at_yield_or_wait_never_kill():
    """★ 守卫文案本身必须指向"换卡或等"，并**明写**"绝不 kill 他人"（可读性即防线）。"""
    gm = _load_gpu_monitor()
    with pytest.raises(Exception) as exc:
        gm.assert_target_gpus_idle(
            [3], runner=_nvidia_smi_stub(70000, 95), samples=1, interval_s=0, sleeper=lambda _s: None
        )
    msg = str(exc.value)
    assert "先到先得" in msg and "绝不 kill" in msg, msg


# ── 5. 清单自述 ↔ 代码（跨件）──────────────────────────────────────────────


def test_manifest_self_description_matches_the_code():
    """★ `matrix54_manifest_source.json` 的机读自述必须与**代码**逐值对齐。"""
    from graspo.core.gpu_guard import ALLOWED_INDICES

    p = REPO / "tests" / "e2e" / "matrix54_manifest_source.json"
    if not p.is_file():
        pytest.skip("v2 清单不在盘")
    cal = json.loads(p.read_text(encoding="utf-8"))["gpu_card_caliber"]
    assert cal["gpu_allowed"] == list(ALLOWED_INDICES) == list(range(8))
    assert cal["max_cards_per_job"] is None
    assert cal["ruling"] == "228 先到先得，随便用。／单个作业不限制卡数。"
    for key in ("reserve", "disable", "production_lock", "third_party_occupancy"):
        assert cal["static_bans"][key] == [], (key, cal["static_bans"][key])
    assert cal["no_preemption"]["required"] is True
    assert "48 GiB" in cal["no_preemption"]["criterion"]
    assert "util == 0" in cal["no_preemption"]["criterion"]
    assert "换卡或等" in cal["no_preemption"]["on_busy"]
    assert "绝不 kill" in cal["no_preemption"]["on_busy"]
    assert "建议首选卡位" in cal["gpus_field_semantics"]
    # ★ 实现清单必须指向**真实存在**的文件/模块（防"自述指向不存在的东西"）
    assert (REPO / "scripts" / "gpu_idle_assert.py").is_file()
    assert any("gpu_idle_assert" in s for s in cal["no_preemption"]["implementations"])


# ══════════════════════════════════════════════════════════════════════════════
# ⑥ 宿主入口 `run.sh` 与评测链 `scripts/eval_*.sh` —— 不得再写旧口径
#
# 病史（2026-10-17 机算发现，硬闸门⑤）：口径换代后，**在役强制点**上仍留着旧口径：
#   · `run.sh:85-86`  `MAX_CARDS=4` + `MAX_INDEX=5`  ⇒ 自动择卡被硬限在 GPU0–5 且 ≤4 张；
#     这是"用户不带 --gpus 起跑"的**最常用路径** ⇒ 不修则主席令在该路径上不生效。
#   · `scripts/eval_serve_vllm.sh` / `eval_export_checkpoint.sh` / `eval_cleanup.sh`
#     `ALLOWED_GPUS="0 1 2 3 4 5"` + `MAX_GPU_COUNT=4` ⇒ **硬拒绝** GPU6/7 与 >4 卡。
# 本段把"宿主入口/评测链也不得回到旧口径"固化成**能失败**的断言（判据是纯文本匹配，
# 但由 `_host_caliber_problems()` 单一实现承担，且下面用变异体证明它**能红**）。
# ⛔ 本段只断言**文本口径 + 必须保留的"不抢占他人"要素**，不改这些脚本的行为。
# ══════════════════════════════════════════════════════════════════════════════

#: 宿主入口/评测链的**旧口径痕迹**（命中即 FAIL）。★ 只列"静态禁用/卡数上限"这一类。
_HOST_OLD_CALIBER_PATTERNS: tuple[tuple[str, str], ...] = (
    ("run.sh", "MAX_CARDS="),
    ("scripts/eval_serve_vllm.sh", "MAX_GPU_COUNT="),
    ("scripts/eval_export_checkpoint.sh", "MAX_GPU_COUNT="),
)
#: 旧口径的**用户可见文案**（"生产卡 / 禁用 / 只允许 0–5"这类已作废的静态名单表述）。
_HOST_OLD_WORDING: tuple[tuple[str, str], ...] = (
    ("scripts/eval_serve_vllm.sh", "production"),
    ("scripts/eval_export_checkpoint.sh", "production"),
    ("scripts/eval_cleanup.sh", "production"),
    ("scripts/eval_cleanup.sh", "allowed eval cards are only"),
    ("run.sh", "生产卡"),
    ("run.sh", "GPU0-5"),
)
#: 现行口径的**必须出现**要素（缺一即 FAIL —— 防"把旧口径删掉却没写清新口径"）。
_HOST_REQUIRED: tuple[tuple[str, str], ...] = (
    ("run.sh", "MAX_INDEX=7"),
    ("run.sh", "不限制卡数"),
    ("run.sh", "绝不 kill"),
    ("run.sh", "gpu_idle_assert.py"),          # 指向权威空闲判据实现
    ("scripts/eval_serve_vllm.sh", 'ALLOWED_GPUS="0 1 2 3 4 5 6 7"'),
    ("scripts/eval_serve_vllm.sh", "不抢占他人"),
    ("scripts/eval_export_checkpoint.sh", 'ALLOWED_GPUS="0 1 2 3 4 5 6 7"'),
    ("scripts/eval_export_checkpoint.sh", "不抢占他人"),
    ("scripts/eval_cleanup.sh", 'ALLOWED_GPUS="0 1 2 3 4 5 6 7"'),
)
#: 旧卡数上限在**说明文字**里的残留（`卡数 ≤ 4` 之类）。
_HOST_OLD_DOC_LINES: tuple[tuple[str, str, str], ...] = (
    ("src/graspo/core/schema.py", "卡数 ≤ 4", "EvalConfig docstring 仍称卡数 ≤ 4（与 resolve_gpu_plan 实现分叉）"),
    ("tools/matrix/extract_spec.py", "待 B10 更新", "口径已定案，不再是「待更新」"),
)


def _host_caliber_problems(texts: dict[str, str]) -> list[str]:
    """纯函数：按 `_HOST_*` 三张表判定，返回问题清单（空 = 全过）。供变异体阳性对照复用。"""
    problems: list[str] = []
    for path, needle in _HOST_OLD_CALIBER_PATTERNS:
        if needle in texts.get(path, ""):
            problems.append(f"{path}: 旧口径痕迹 {needle!r}（静态禁用/卡数上限）")
    for path, needle in _HOST_OLD_WORDING:
        if needle in texts.get(path, ""):
            problems.append(f"{path}: 旧口径文案 {needle!r}（已作废的静态名单表述）")
    for path, needle in _HOST_REQUIRED:
        if needle not in texts.get(path, ""):
            problems.append(f"{path}: 缺现行口径要素 {needle!r}")
    for path, needle, why in _HOST_OLD_DOC_LINES:
        if needle in texts.get(path, ""):
            problems.append(f"{path}: {why}（命中 {needle!r}）")
    return problems


def _read_host_sources() -> dict[str, str]:
    paths = {p for p, _ in _HOST_OLD_CALIBER_PATTERNS} | {p for p, _ in _HOST_OLD_WORDING}
    paths |= {p for p, _ in _HOST_REQUIRED} | {p for p, _, _ in _HOST_OLD_DOC_LINES}
    texts: dict[str, str] = {}
    for rel in sorted(paths):
        f = REPO / rel
        texts[rel] = f.read_text(encoding="utf-8") if f.is_file() else ""
    return texts


def test_host_entries_and_eval_chain_do_not_carry_the_old_caliber() -> None:
    """★ 真树：`run.sh` 与 `scripts/eval_*.sh` 必须在新口径上，且保留"不抢占他人"要素。"""
    problems = _host_caliber_problems(_read_host_sources())
    assert problems == [], "宿主入口/评测链仍带旧口径：\n  - " + "\n  - ".join(problems)


def test_host_caliber_checker_actually_bites() -> None:
    """★ **能失败对照**（本段机核自己的阳性对照）：逐项把旧值/旧文案塞回 ⇒ 必须报出来。

    这一条证明上面那条真树断言**不是恒绿**：它用的就是同一个 `_host_caliber_problems()`。
    """
    good = _read_host_sources()
    assert _host_caliber_problems(good) == [], "前提：真树本应全过"
    mutants = (
        ("run.sh", "MAX_INDEX=7", "MAX_INDEX=5"),
        ("run.sh", "# ★ 不抢占他人", "# 生产卡 GPU6/7 永不自动选中"),
        ("scripts/eval_serve_vllm.sh", 'ALLOWED_GPUS="0 1 2 3 4 5 6 7"', 'ALLOWED_GPUS="0 1 2 3 4 5"'),
        ("scripts/eval_serve_vllm.sh", "ALLOWED_GPUS=\"0 1 2 3 4 5 6 7\"", "ALLOWED_GPUS=\"0 1 2 3 4 5\"\nMAX_GPU_COUNT=4"),
        ("scripts/eval_export_checkpoint.sh", 'ALLOWED_GPUS="0 1 2 3 4 5 6 7"', 'ALLOWED_GPUS="0 1 2 3 4 5"'),
        ("scripts/eval_cleanup.sh", "the cards that exist here", "allowed eval cards are only"),
        ("src/graspo/core/schema.py", "卡数不设上限", "卡数 ≤ 4"),
        ("tools/matrix/extract_spec.py", "2026-10-17 主席令已定案", "⇒ 待 B10 更新"),
    )
    for path, old, new in mutants:
        assert old in good[path], (path, old)
        mutated = dict(good)
        mutated[path] = good[path].replace(old, new, 1)
        assert _host_caliber_problems(mutated), f"变异体未被抓到（本段断言失效）：{path} {old!r}→{new!r}"
