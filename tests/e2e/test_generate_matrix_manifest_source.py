"""matrix54 v2 清单生成器：卡位覆盖 + **卡号口径断言**的回归覆盖。

**为什么必须有这个文件**：运行清单 `matrix54_manifest_source.json` 是**生成物**。
只手改清单文件的话，下次重生成会按 `GPU_POOL_BY_CARDS` 把卡位静默改回（回退陷阱），
而当时的测试**不会报警**。

## 现行口径（唯一标尺，2026-10-17 主席裁定，逐字）

> 「**远端机先到先得，随便用。**」 ＋ 「**单个作业不限制卡数。**」

⇒ 卡号判据只剩**存在性**：本机全卡 **0–7 一律可用**（⛔ 无预留/禁用/生产锁卡/邻居卡），
⛔ 无单作业卡数上限；唯一硬要求 =「**不抢占他人**」（起跑前逐卡实测：剩余 ≥ 头寸 ∧ util == 0）。
★ **每档的 `gpus` 降级为「建议首选卡位」**（调度建议 + A4 可比性锚，⛔ 不是合法性判据）；
★ 本包只改**语义与判据**，⛔ 不重排、⛔ 不动任何档的 `gpus` 取值。

本文件把三件事固化成断言：
  ① 默认映射 `GPU_POOL_BY_CARDS` 的**取值逐字未变**，且各项过"卡号 ∈ 0–7 ∧ 卡数 == cards"；
  ② `TIER_GPU_OVERRIDE` 现为**车道表**（非空）；T041 走显式车道、`cards` 不变；
  ③ ★ **通用卡号断言**（`card_range_violation`）：任何档的 `gpus` 必须
     ① 卡号 ⊆ 本机全卡 0–7 ② 卡数 == 该档 `cards`；
     ⛔ 不判"生产锁卡"/"邻居卡"/卡数上限（旧口径已废）。

★ **为什么"撤销静态禁用"没有削弱保护**：本文件**补上**了旧判据没有的阳性对照 ——
  旧口径拒的三类输入（GPU2 / GPU6/7 / 5 张卡）现在**必须放行**，
  而越界卡（≥8）、卡数不符、重复卡**仍然必须被拒**；
  动态占用由**设施层**的逐卡实测空闲断言负责（⛔ 生成器不猜谁占着卡）。
"""

#: ★★ 2026-10-22 **本模块校验的是「旧世代（54 档矩阵）」的跨件一致性，⛔ 不是现行 124 档矩阵。**
#:   来历：`docs/capability-matrix.data.json` 已由枚举包换代成 **124 档 `M001…M124`**（全「待测试」），
#:   而本模块核对的三件（`matrix54_manifest_source.json` / `matrix54` 档位 yaml / 冻结子集目录）
#:   仍是**旧 54 档 `T001…T054`** 世代 ⇒ 直接对在盘新矩阵跑必然"每档都查不到"。
#:   处置（⛔ 断言本体一个字不改、不删、不 skip）：把改动前的旧矩阵**冻结成本目录夹具**
#:   `legacy_matrix54.data.json`（内容逐字取自提交 `dece5b5` 的 `docs/capability-matrix.data.json`），
#:   本模块一律以它为矩阵侧输入。等 124 档的清单/档位配置重生成后，本模块应改为指向新世代并删除本夹具。
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

_HERE = Path(__file__).resolve()
#: ★ 旧世代矩阵夹具（54 档 `T001…T054`，逐字取自 `dece5b5`）——见模块头说明。
_LEGACY_MATRIX = _HERE.parent / "legacy_matrix54.data.json"
_SPEC = importlib.util.spec_from_file_location(
    "generate_matrix_manifest_source_uut", _HERE.parent / "generate_matrix_manifest_source.py"
)
assert _SPEC is not None and _SPEC.loader is not None
gmv2 = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(gmv2)

_V1_TIERS = json.loads(gmv2.V1_MANIFEST_PATH.read_text(encoding="utf-8"))["tiers"]
_V1_BY_ID = {t["tier_id"]: t for t in _V1_TIERS}


def _patched(tier_id: str) -> dict:
    return gmv2.patch_tier(_V1_BY_ID[tier_id], "/data-root", "/pregen-dir")


# ── ① 默认映射逐字未变 ───────────────────────────────────────────────────────


def test_default_pool_map_is_verbatim():
    """★ 卡数→**建议首选卡位**映射的**取值逐字未变**（2026-10-17 只改语义与判据，⛔ 不动取值）。

    现行判据 = 卡号 ⊆ 本机全卡 0–7 ∧ 卡数 == ``cards``。
    ★ 反向断言：**越界卡（≥8）** 仍必须红（否则本用例形同虚设）。
    ★ 同时钉住"旧口径的静态禁用确实解除了"：``[4,5,6,7]`` 这种旧形态**不再**被拒
      —— 它不是"合法卡集外"，它只是**另一套建议卡位**。
    """
    assert gmv2.GPU_POOL_BY_CARDS == {1: [4], 2: [4, 5], 3: [4, 5], 4: [1, 3, 4, 5]}
    for cards, gpus in gmv2.GPU_POOL_BY_CARDS.items():
        assert gmv2.card_range_violation(f"pool{cards}", gpus, len(gpus)) is None
    # ★ 反向断言：越界卡必须被拒（唯一卡号判据 = 存在性）
    assert gmv2.card_range_violation("old", [4, 5, 6, 8], 4) is not None
    assert gmv2.card_range_violation("old", [8], 1) is not None
    # ★ 旧"生产锁卡"形态现在合法（卡号 6/7 存在于本机）
    assert gmv2.card_range_violation("legacy-pool", [4, 5, 6, 7], 4) is None


# ── ② 陈旧例外已撤销 ────────────────────────────────────────────────────────


def test_tier_gpu_override_is_the_lane_table():
    """★ 车道表**取值逐字未变**；每项过"卡号 ∈ 0–7 ∧ 卡数 == cards"。

    ★ 2026-10-17：旧用例的反向断言「车道里一张 0/2/6/7 都不许有」**已删除** ——
      那正是被主席令废除的**静态禁用**判据（"哪些卡不许出现"）。
      现行反向断言改为"越界卡不许出现"（见下）。
    """
    L = gmv2.TIER_GPU_OVERRIDE
    assert set(L) == {"T031", "T001", "T010", "T013", "T028", "T016", "T022", "T040",
                      "T002", "T011", "T014", "T017", "T023", "T029", "T032", "T041"}
    assert L["T031"] == [1] and L["T002"] == [3, 5] and L["T010"] == [4]
    for tid, g in L.items():
        out = _patched(tid)
        assert out["gpus"] == g, tid
        assert out["cards"] == len(g), tid
        assert gmv2.card_range_violation(tid, g, len(g)) is None, tid
    # ★ 反向断言（现行唯一卡号判据）：车道里**一张越界卡**都不许有
    assert not ({x for g in L.values() for x in g} - set(gmv2.GPU_ALLOWED))
    assert tuple(gmv2.GPU_ALLOWED) == tuple(range(8))


def test_t041_lane_and_cards_unchanged():
    """★ 核心断言：T041 的**建议卡位取值**仍为 `[3,5]`，而 `cards`（卡数）**仍为 2**（本包未改取值）。

    ★ 2026-10-17 语义变更：`gpus` = **建议首选卡位**（⛔ 不是合法性判据）⇒
      历史值 `[2,3]`（含 GPU2）现在是一套**合法**的建议卡位；本用例把它**放行**钉住
      —— 这正是"⛔ 不设邻居卡"这条主席令的能失败对照。
    """
    out = _patched("T041")
    assert out["gpus"] == [3, 5]
    assert out["cards"] == 2
    assert len(out["gpus"]) == out["cards"]
    saved = dict(gmv2.TIER_GPU_OVERRIDE)
    try:
        gmv2.TIER_GPU_OVERRIDE = {**saved, "T041": [2, 3]}   # 旧"邻居卡例外"
        assert _patched("T041")["gpus"] == [2, 3]            # ⇒ 现在**必须放行**
        gmv2.TIER_GPU_OVERRIDE = {**saved, "T041": [6, 7]}   # 旧"生产锁卡"
        assert _patched("T041")["gpus"] == [6, 7]            # ⇒ 现在**必须放行**
        gmv2.TIER_GPU_OVERRIDE = {**saved, "T041": [7, 8]}   # 越界卡 ⇒ 仍必须拒
        with pytest.raises(AssertionError):
            _patched("T041")
    finally:
        gmv2.TIER_GPU_OVERRIDE = saved


def test_card_pool_caliber_is_written_down_in_the_generator():
    """现行口径必须连同**裁定逐字**与**保留的硬要求**一起写清（否则下一个人会以为是漏改）。

    ★ 与旧用例 `test_t041_revocation_reason_is_written_down` 的关系：该用例钉的是
      "2026-10-16 撤销邻居卡例外"这一**已被 2026-10-17 主席令超越**的中间口径 ⇒ 已废。
      本用例改为钉**现行口径**：裁定逐字 ＋ "建议首选卡位"语义 ＋ "不抢占他人"（含实现指向）。
    ★ §15 去标识：生成器源码不得含第三方具名账号（当前口径下**连"邻居卡"都不再是判据**，
      故源码里也不应再有具名/角色化的第三方卡归属描述）。
    """
    src = (_HERE.parent / "generate_matrix_manifest_source.py").read_text(encoding="utf-8")
    assert "远端机先到先得，随便用。" in src          # 裁定（已脱敏转述）
    assert "单个作业不限制卡数。" in src            # 裁定逐字
    assert "建议首选卡位" in src                    # gpus 的现行语义
    assert "不抢占他人" in src                      # 保留的硬要求
    assert "gpu_idle_assert" in src or "flow.guard.idle" in src   # 判据指向实现
    # 反向断言：生成器源码不得再出现该第三方卡归属名。字面量同样拆成两段拼接——
    # 断言本身必须保留（它是防护用例），但测试文件不该成为这个词的泄露源。
    assert ("phys" + "brain") not in src


def test_override_only_changes_gpus():
    """★ 覆盖只改 `gpus`：给一个池内合法覆盖后逐字段对比，除 `gpus` 外**一字不改**。"""
    saved = dict(gmv2.TIER_GPU_OVERRIDE)   # ★ 2026-10-16：不得清空车道表（旧码 `= {}` 会毁掉它）
    default = _patched("T041")
    assert default["gpus"] == [3, 5]
    gmv2.TIER_GPU_OVERRIDE = {**saved, "T041": [5, 3]}   # 合法集内、卡数一致 ⇒ 合法覆盖
    try:
        overridden = _patched("T041")
    finally:
        gmv2.TIER_GPU_OVERRIDE = saved
    assert overridden["gpus"] == [5, 3]
    assert set(default) == set(overridden)
    for key, value in default.items():
        if key == "gpus":
            continue
        assert overridden[key] == value, key


def test_all_other_tiers_follow_the_default_map():
    """⛔ 覆盖表不得波及其它档：未点名档的 `gpus` == `GPU_POOL_BY_CARDS[cards]`。"""
    for tier in _V1_TIERS:
        tier_id = tier["tier_id"]
        if tier_id in gmv2.TIER_GPU_OVERRIDE:
            continue
        out = _patched(tier_id)
        assert out["gpus"] == list(gmv2.GPU_POOL_BY_CARDS[int(tier["cards"])]), tier_id
        assert out["cards"] == int(tier["cards"]), tier_id


# ── ③ ★ 通用锁卡断言（判据本身，不只是期望值）────────────────────────────────


def test_card_pool_caliber_comes_from_a_single_source():
    """★ 卡号允许集合源自 `gpu_assignment`（§1.4 单一真相源），且**旧静态禁用符号已不存在**。

    用 ``hasattr`` 断言"符号不存在"是最强的能失败对照：一旦有人把
    ``NEIGHBOR_CARDS`` / ``LOCKED_CARDS`` / ``GPU_FORBIDDEN_PRODUCTION`` /
    ``legal_card_set`` / ``locked_card_violation`` 加回来（= 静态禁用复活），本用例当场红。
    """
    assert tuple(gmv2.GPU_ALLOWED) == (0, 1, 2, 3, 4, 5, 6, 7)
    assert gmv2.out_of_range_cards([7, 8]) == (8,)          # 越界判据生效
    assert gmv2.out_of_range_cards([0, 2, 6, 7]) == ()      # 旧"禁用卡"全部合法
    for retired in ("NEIGHBOR_CARDS", "LOCKED_CARDS", "GPU_FORBIDDEN_PRODUCTION",
                    "legal_card_set", "non_legal_card_violation", "locked_card_violation",
                    "locked_card_hits", "CHAIRMAN_BANNED_CARDS", "POOL_CONTRACT_CONFLICTS"):
        assert not hasattr(gmv2, retired), (
            f"旧口径符号 {retired} 复活了 ⇒ 静态禁用卡/锁卡语义回来了，必须按主席令改写"
        )


def test_no_tier_declares_an_out_of_range_card():
    """★ 全 54 档：`gpus` 必须 ① 卡号 ⊆ 0–7 ② 卡数 == 该档 `cards`。

    ⛔ 不再判"邻居卡 / 生产锁卡"（旧口径已废）——那两条现在**合法**，由
    ``test_t041_lane_and_cards_unchanged`` 正向钉住。
    """
    offenders = []
    for tier in _V1_TIERS:
        out = _patched(tier["tier_id"])
        bad = gmv2.card_range_violation(tier["tier_id"], out["gpus"], int(out["cards"]))
        if bad:
            offenders.append(bad)
    assert offenders == [], f"以下档的卡号/卡数不合规：{offenders}"


def test_formerly_banned_cards_are_now_accepted_at_generation_time():
    """★ 旧口径拒的三类输入在生成期**必须放行**（本包的核心口径变更，能失败对照）。

    ① GPU2（旧"邻居卡"）② GPU6/7（旧"生产锁卡"）③ 含 0 的声明（旧"主席禁用"）。
    ★ 卡数仍必须 == 该档 `cards`（覆盖只改"用哪几张"）。
    """
    saved = dict(gmv2.TIER_GPU_OVERRIDE)
    try:
        gmv2.TIER_GPU_OVERRIDE = {**saved, "T041": [2, 0], "T002": [6, 7]}
        assert _patched("T041")["gpus"] == [2, 0]
        assert _patched("T002")["gpus"] == [6, 7]
    finally:
        gmv2.TIER_GPU_OVERRIDE = saved


# ── 覆盖表本身的 fail-closed ────────────────────────────────────────────────


def test_override_card_count_mismatch_is_fail_closed():
    """覆盖表卡数 ≠ `cards` ⇒ fail-closed（不静默"将就"成错卡数）。"""
    saved = dict(gmv2.TIER_GPU_OVERRIDE)
    gmv2.TIER_GPU_OVERRIDE = {**saved, "T041": [4, 5, 6]}
    try:
        with pytest.raises(AssertionError, match="卡数"):
            _patched("T041")
    finally:
        gmv2.TIER_GPU_OVERRIDE = saved


def test_override_duplicate_gpus_is_fail_closed():
    """覆盖表含重复卡 ⇒ fail-closed（重复卡会让"卡数"名不副实）。"""
    saved = dict(gmv2.TIER_GPU_OVERRIDE)
    gmv2.TIER_GPU_OVERRIDE = {**saved, "T041": [4, 4]}
    try:
        with pytest.raises(AssertionError, match="重复"):
            _patched("T041")
    finally:
        gmv2.TIER_GPU_OVERRIDE = saved


def test_override_with_an_out_of_range_card_is_fail_closed():
    """★ 现行判据（唯一一条卡号判据）= **卡号 ⊆ 本机全卡 0–7**：越界卡（≥8）必红。

    ★ 反向断言：**旧口径拒的三类**（GPU2 / GPU6/7 / GPU0）现在**必须被接受**
      —— 判据不是"与某个静态名单逐字相等"。
    """
    saved = dict(gmv2.TIER_GPU_OVERRIDE)
    try:
        for bad in ([8, 9], [2, 8], [7, 8]):   # ★ 卡数一律 2（否则先撞卡数断言）
            gmv2.TIER_GPU_OVERRIDE = {**saved, "T041": list(bad)}
            with pytest.raises(AssertionError, match="越界卡"):
                _patched("T041")
        for good in ([2, 3], [0, 1], [4, 6], [6, 7], [5, 7], [3, 5]):
            gmv2.TIER_GPU_OVERRIDE = {**saved, "T041": list(good)}
            assert _patched("T041")["gpus"] == list(good), good
    finally:
        gmv2.TIER_GPU_OVERRIDE = saved


def test_override_inside_generation_pool_is_accepted():
    """池内、卡数一致的例外仍被接受（断言不得把合法例外一起禁掉）。"""
    gmv2.TIER_GPU_OVERRIDE = {"T041": [4, 5]}
    try:
        assert _patched("T041")["gpus"] == [4, 5]
    finally:
        gmv2.TIER_GPU_OVERRIDE = {}


# ── 在盘清单的一致性（产物侧，若存在）────────────────────────────────────────


def test_on_disk_manifest_declares_the_new_caliber():
    """在盘 v2 清单：① 逐档卡号 ⊆ 0–7 且卡数 == cards；② **自述块 = 新口径**。

    ★ 机读自述（`gpu_card_caliber`）必须存在且**恒空**地登记四类静态名单
      （预留/禁用/生产锁/第三方占用）—— 这是"⛔ 不预留、⛔ 不禁用、⛔ 无生产锁卡、
      ⛔ 无邻居卡"的**机读留痕**，⛔ 不是散文承诺。
    ★ 旧自述块 `gpu_pool_contract_conflicts`（登记"卡池 ∩ 生产锁卡"的旧口径冲突）
      **必须已消失** —— 那三类静态锁卡不存在了，冲突这个概念也随之消失。
    """
    p = gmv2.V2_MANIFEST_PATH
    if not p.is_file():
        pytest.skip("v2 清单不在盘")
    manifest = json.loads(p.read_text(encoding="utf-8"))
    bad = [gmv2.card_range_violation(str(t["tier_id"]), t.get("gpus") or [], int(t["cards"]))
           for t in manifest["tiers"]]
    assert [b for b in bad if b] == [], [b for b in bad if b]
    assert "gpu_pool_contract_conflicts" not in manifest
    cal = manifest["gpu_card_caliber"]
    assert cal["gpu_allowed"] == list(range(8))
    assert cal["max_cards_per_job"] is None
    assert list(cal["static_bans"][k] for k in
                ("reserve", "disable", "production_lock", "third_party_occupancy")) == [[], [], [], []]
    assert cal["no_preemption"]["required"] is True
    assert "util == 0" in cal["no_preemption"]["criterion"]
    assert "建议首选卡位" in cal["gpus_field_semantics"]


# ══════════════════════════════════════════════════════════════════════════════
# ★★ 修正轮 1（2026-10-16 复核报告 B1a/B1b/B1c）—— 口径扩面的**生成链回归**
#
# 病史（复核报告，全部实测）：18 档口径扩面**只存在于在盘文件里**，生成器没有任何输入/表
# 编码它们 ⇒ 走「声明的唯一写路径」重生成就把 18 档 `subset_size` 100→20、步数 ÷5，而
# `guard_regeneration_keeps_on_disk_fields()` 只防字段丢失、`check()` 对非 native 档不重算
# ⇒ **rc=0 且仍打印 `check OK`**；另 `expected_optimizer_steps` 与 `checkpoint_save_steps`
# 的等式不变量被手工并入打破（18/18 不等），而冻结子集交叉核对因 `host_dir` 是未展开占位符
# 而**每次静默跳过**（7 条行数 MISMATCH 从未被发现）。
#
# ⇒ 本段把**判据本身**固化成断言：既断言真值，也断言"值回退/行数回退**必须红**"（阳性对照）。
# ══════════════════════════════════════════════════════════════════════════════


def test_declared_caliber_registry_is_the_single_source_of_truth():
    """★ 18 档声明口径的唯一真相源 = `make_subsets.MINI_SUBSET_TIERS_DECLARED`（生成器只读加载）。"""
    assert len(gmv2.MINI_DECLARED_SUBSET_SIZE) == 18
    assert set(gmv2.MINI_DECLARED_SUBSET_SIZE.values()) == {100}
    # 生成器**不许**再抄一份常量：三张表必须与 make_subsets 逐字相同。
    # ★ 选型理由（同名副本复用，与生成器同一个 loader 收口）：原先这里用
    #   `spec_from_file_location("_t_ms", …)` 又造了**第三份**副本（既不登记、也不复用）——
    #   表格是只读比较时读数一致，但它是个**补丁目标陷阱**：往 sys.modules 里那份打补丁，
    #   本用例读的是另一份 ⇒ 断言空转（实测见 task-arch-b3/evidence/b3-control-BEFORE.txt）。
    #   改走生成器自己的 `load_make_subsets_module()`（同名副本复用）⇒ 断言作用在活对象上。
    ms = gmv2.load_make_subsets_module()
    assert ms is gmv2.ms, "同名副本未复用 ⇒ 本用例读的表格与生成器不是同一份"
    assert gmv2.MINI_DECLARED_SUBSET_SIZE == ms.MINI_SUBSET_TIERS_DECLARED
    assert gmv2.MINI_LANDED_SUBSET_SIZE == ms.MINI_TIERS
    assert gmv2.MINI_DECLARED_NOT_LANDED == ms.DECLARED_NOT_LANDED


def _reimport_generator():
    """按文件路径**再执行一遍**生成器模块体（= 重新导入一份新副本）。"""
    spec = importlib.util.spec_from_file_location(
        "generate_matrix_manifest_source_uut_reimport",
        _HERE.parent / "generate_matrix_manifest_source.py",
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_formal_tiers_drift_fails_the_import(monkeypatch):
    """★ **能失败对照**（`V2_SUBSET_SIZE_BY_TIER` vs `make_subsets.FORMAL_TIERS`）。

    两张表逐值相等，过去**无直接断言**（只经冻结 jsonl 行数间接相连）⇒ 会漂。
    本用例证明那条断言**能红**：把唯一真相源某档改值 ⇒ 导入生成器当场报错。
    ★ 前提（否则对照不成立）：未篡改时同一导入路径**必须成功**。
    """
    _reimport_generator()  # 未篡改 ⇒ 导入成功
    assert gmv2.V2_SUBSET_SIZE_BY_TIER == gmv2.ms.FORMAL_TIERS
    monkeypatch.setitem(gmv2.ms.FORMAL_TIERS, "T028", 48 + 1)
    with pytest.raises(AssertionError) as e:
        _reimport_generator()
    assert "FORMAL_TIERS 漂移" in str(e.value)


def test_mini_tier_subset_size_never_falls_back_to_v1_20():
    """★ 18 档现算值恒为 100（⛔ 不许回落 v1 的 20）——这是 B1a 的核心。"""
    for tid in gmv2.MINI_DECLARED_SUBSET_SIZE:
        assert gmv2.mini_tier_subset_size(tid, 20) == 100, tid


def test_mini_tier_subset_size_is_fail_closed_on_landed_mismatch(monkeypatch):
    """★ 阳性对照（值回退 fail-closed）：让"在盘真值"与"声明口径"不一致 ⇒ **必须 raise**。"""
    monkeypatch.setitem(gmv2.MINI_LANDED_SUBSET_SIZE, "T046", 20)
    with pytest.raises(AssertionError) as e:
        gmv2.mini_tier_subset_size("T046", 20)
    assert "声明口径" in str(e.value) and "值回退" in str(e.value)


def test_declared_not_landed_tier_must_be_smaller(monkeypatch):
    """★ 豁免表必须真的"未落地"：在盘真值 ≥ 声明口径 ⇒ raise（防豁免表陈旧）。"""
    monkeypatch.setitem(gmv2.MINI_LANDED_SUBSET_SIZE, "T052", 100)
    with pytest.raises(AssertionError) as e:
        gmv2.mini_tier_subset_size("T052", 20)
    assert "豁免表陈旧" in str(e.value)


def test_on_disk_manifest_declares_the_18_tier_expansion():
    """★ 在盘 v2 清单：18 档 `subset_size == 100`（声明口径）。"""
    p = gmv2.V2_MANIFEST_PATH
    if not p.is_file():
        pytest.skip("v2 清单不在盘")
    man = json.loads(p.read_text(encoding="utf-8"))
    by = {t["tier_id"]: t for t in man["tiers"]}
    # ★ 15 档已落地：清单声明的 `subset_size` == 100（生成器唯一真相源现算）
    landed_bad = [(tid, by[tid]["data"]["subset_size"])
                  for tid in gmv2.MINI_DECLARED_SUBSET_SIZE
                  if tid not in gmv2.MINI_DECLARED_NOT_LANDED
                  and int(by[tid]["data"]["subset_size"]) != 100]
    assert landed_bad == [], f"已落地的 15 档 subset_size != 100：{landed_bad}"
    # ★ 3 档具名豁免：`subset_size` 仍 = v1（不跑、不伪造），但"声明口径 = 100 未落地"**机读登记**
    for tid in gmv2.MINI_DECLARED_NOT_LANDED:
        d = by[tid]["data"]
        assert d["subset_size"] == 20, (tid, d["subset_size"])
        assert d.get("declared_subset_size") == 100, (tid, d.get("declared_subset_size"))
        assert d.get("declared_not_landed", {}).get("declared") == 100
        assert d.get("declared_not_landed", {}).get("landed") == 20
        assert d.get("declared_not_landed", {}).get("reason")
        assert not d.get("subset_file_pregenerated"), tid


def test_regenerated_manifest_matches_on_disk_caliber_values():
    """★★ B1a 的**端到端**回归：生成器重生成 == 在盘清单（口径关键字段逐字段一致）。

    ★ 且**不整文件比**：在盘清单是跑次台账（含生成器之外的 post-fill 字段与派生 status），
      逐字节比会误伤；口径面 == 0 差异即可证明"值不再回退"（`caliber_value_regression` = []）。
    """
    p = gmv2.V2_MANIFEST_PATH
    if not p.is_file():
        pytest.skip("v2 清单不在盘")
    on_disk = json.loads(p.read_text(encoding="utf-8"))   # noqa: F841  （断言读在盘件，供 drift 用）
    v1 = json.loads(gmv2.V1_MANIFEST_PATH.read_text(encoding="utf-8"))
    # ★ 用**清单自身的默认占位符**构造（与真实写入路径同参）⇒ `extract_command` 等派生串才可比
    new = gmv2.build_v2_manifest(
        v1, gmv2.DEFAULT_HOST_DATA_ROOT, gmv2.DEFAULT_PREGEN_SUBSET_DIR,
        gmv2.DEFAULT_ATTEMPT_ROOT, "0" * 64, "0" * 64, "0" * 64)
    assert gmv2.caliber_value_regression(new, p) == [], "重生成相对在盘存在口径值漂移（值回退）"


def test_caliber_value_regression_guard_is_fail_closed():
    """★ 阳性对照：在盘件值被改回 20（值回退）⇒ `caliber_value_regression` 必须报出来。"""
    import tempfile
    on_disk = json.loads(gmv2.V2_MANIFEST_PATH.read_text(encoding="utf-8"))
    for t in on_disk["tiers"]:
        if t["tier_id"] == "T031":
            t["data"]["subset_size"] = 20
            t["expected_optimizer_steps"] = 160
            t["checkpoint_save_steps"] = 160
    with tempfile.TemporaryDirectory() as td:
        f = Path(td) / "regressed.json"
        f.write_text(json.dumps(on_disk, ensure_ascii=False), encoding="utf-8")
        v1 = json.loads(gmv2.V1_MANIFEST_PATH.read_text(encoding="utf-8"))
        new = gmv2.build_v2_manifest(
            v1, gmv2.DEFAULT_HOST_DATA_ROOT, gmv2.DEFAULT_PREGEN_SUBSET_DIR,
            gmv2.DEFAULT_ATTEMPT_ROOT, "0" * 64, "0" * 64, "0" * 64)
        drift = gmv2.caliber_value_regression(new, f)
    assert any("T031.data.subset_size" in d for d in drift), drift


def test_eos_equals_checkpoint_save_steps_on_every_tier():
    """★★ B1c 不变量（清单自带 `key_semantics` 逐字声明）：`expected_optimizer_steps == checkpoint_save_steps`。

    ★ 实测依据：改前 HEAD 清单 **54/54 相等**；本包手工并入只改一半 ⇒ 18 档违反。
    """
    p = gmv2.V2_MANIFEST_PATH
    if not p.is_file():
        pytest.skip("v2 清单不在盘")
    man = json.loads(p.read_text(encoding="utf-8"))
    bad = [(t["tier_id"], t["expected_optimizer_steps"], t["checkpoint_save_steps"])
           for t in man["tiers"]
           if int(t["expected_optimizer_steps"]) != int(t["checkpoint_save_steps"])]
    assert bad == [], f"expected_optimizer_steps != checkpoint_save_steps：{bad}"


def test_generator_keeps_eos_invariant_for_all_tiers():
    """★ 生成器侧同样成立（⛔ 不只在 native 分支里做）—— 否则重生成又会打破它。"""
    v1 = json.loads(gmv2.V1_MANIFEST_PATH.read_text(encoding="utf-8"))
    new = gmv2.build_v2_manifest(
        v1, gmv2.DEFAULT_HOST_DATA_ROOT, gmv2.DEFAULT_PREGEN_SUBSET_DIR,
        gmv2.DEFAULT_ATTEMPT_ROOT, "0" * 64, "0" * 64, "0" * 64)
    bad = [t["tier_id"] for t in new["tiers"]
           if int(t["expected_optimizer_steps"]) != int(t["checkpoint_save_steps"])]
    assert bad == [], bad


def test_subset_check_dir_resolves_in_repo_and_never_silently_skips():
    """★★ B1b 根因回归：`${GRASPO_PREGEN_SUBSET_DIR}` 未展开 ⇒ 必须**回落到仓内真目录**，不许静默跳过。"""
    d, why = gmv2.resolve_subset_check_dir("${GRASPO_PREGEN_SUBSET_DIR}")
    assert d is not None, f"本机应能解析到仓内 {gmv2.REPO_SUBSET_DIR_REL}/：{why}"
    assert d.name == gmv2.REPO_SUBSET_DIR_REL
    # 三级解析的第 ③ 级：解析不到 ⇒ 返回 None ＋ 具名原因（调用方据此 fail-closed）
    d2, why2 = gmv2.resolve_subset_check_dir("${GRASPO_PREGEN_SUBSET_DIR}", repo=Path("/nonexistent-repo"))
    assert d2 is None and "未展开占位符" in why2


def test_frozen_subset_crosscheck_reports_mismatch(tmp_path):
    """★ 阳性对照（行数回退必红）：把某档冻结件改回 20 行 ⇒ `check()` 必须 MISMATCH 且非 0。"""
    import shutil
    src = gmv2.PROJECT_ROOT / gmv2.REPO_SUBSET_DIR_REL
    if not src.is_dir():
        pytest.skip("仓内没有冻结子集目录")
    work = tmp_path / "subsets-v2"
    shutil.copytree(src, work)
    raw = (work / "T050.jsonl").read_bytes()
    (work / "T050.jsonl").write_bytes(b"\n".join(raw.split(b"\n")[:20]) + b"\n")
    rc = gmv2.check(gmv2.V2_MANIFEST_PATH, subset_dir_override=work)
    assert rc != 0


def test_check_passes_with_repo_frozen_subsets():
    """★ 正常态：`--check` 必须 rc=0（冻结子集核对覆盖 51 档；3 档具名豁免）。"""
    assert gmv2.check(gmv2.V2_MANIFEST_PATH, matrix_path=_LEGACY_MATRIX) == 0


# ══════════════════════════════════════════════════════════════════════════════
# ★★ 修正轮 2（第二轮全局审查 🔴B-1 / 🟡W-1b / W-8）—— 「每 epoch 计划步数」跨件一致性
#
# 病史：修正轮 1 让生成器的**非 native 分支透传 v1**（未乘口径比率）⇒ 18 档
#   `expected_optimizer_steps_per_epoch/_reachable` 由 800/800 回退成 160/160，
#   而 `docs/capability-matrix.data.json` 的 `per_epoch_steps` 仍是 800
#   ⇒ §1.4 双真相源 ＋ `scripts/collect_results.py` 的 ① 步数门被实质放宽 5 倍。
#   `caliber_value_regression()` 只比"清单 ↔ 重生成"（两者同为 160）⇒ **保护了错值**。
# ⇒ 本段把判据固化成断言：**跨件**（矩阵 ↔ 清单）＋ 生成器侧比率断言 ＋ 豁免敏感化。
# ══════════════════════════════════════════════════════════════════════════════


def test_caliber_subset_ratio_declared_tiers_are_five():
    """★ 口径比率：18 档（**含 3 档 `not_applicable` 豁免档**）一律 5；其余档按在盘子集。"""
    for tid in gmv2.MINI_DECLARED_SUBSET_SIZE:
        is_na = tid in gmv2.MINI_DECLARED_NOT_LANDED
        subset = 20 if is_na else 100          # 豁免档在盘仍是 20，但**声明口径** = 100
        assert gmv2.caliber_subset_ratio(tid, 20, subset, is_na) == 5, tid


def test_caliber_subset_ratio_is_one_outside_declared_set():
    """非声明面档：CPT/SFT 100↔100 ⇒ 比率 1（⛔ 不得顺手放大）。"""
    assert gmv2.caliber_subset_ratio("T001", 100, 100, False) == 1


def test_caliber_subset_ratio_is_fail_closed_on_non_integer(monkeypatch):
    """★ 非整数倍 ⇒ **必须 raise**（⛔ 不静默取整、不静默豁免）。

    ★ 构造法说明：非声明面档的分子恒 = v1 子集 ⇒ 比率恒 1，该分支**结构上**不会非整除；
      故阳性对照须在**声明口径**上构造（把 T031 的声明口径改成 99，v1 = 20 ⇒ 99 % 20 ≠ 0）。
    """
    monkeypatch.setitem(gmv2.MINI_DECLARED_SUBSET_SIZE, "T031", 99)
    with pytest.raises(AssertionError) as e:
        gmv2.caliber_subset_ratio("T031", 20, 99, False)
    assert "非整数倍" in str(e.value)


def _regen_doc():
    v1 = json.loads(gmv2.V1_MANIFEST_PATH.read_text(encoding="utf-8"))
    return gmv2.build_v2_manifest(
        v1, gmv2.DEFAULT_HOST_DATA_ROOT, gmv2.DEFAULT_PREGEN_SUBSET_DIR,
        gmv2.DEFAULT_ATTEMPT_ROOT, "0" * 64, "0" * 64, "0" * 64)


def test_generator_non_native_per_epoch_scales_by_caliber_ratio():
    """★ B-1 根因回归：生成器非 native 档 `per_epoch` 必须 = v1 值 × 口径比率
    × **（卡数 ÷ 数据并行度 dp_size）**（⛔ 不得透传；⛔ 不得按卡数折算 PP 档）。

    ★ 2026-10-17 W-7 定案补充：v1 的基准是「**按卡数均分**」的每 epoch 步数，而 **PP 切层不切数据**
      ⇒ `pp_size>1 ∧ dp_size=1` 的档（T017/T018）必须再按 `cards/dp_size` 折算回数据并行度口径。
      本式对 `dp_size == cards` 的档退化为原来的「v1 × 比率」（⛔ 未顺手改到其它 43 档）。
    """
    doc = _regen_doc()
    bad, pp_seen = [], []
    for e in doc["tiers"]:
        tid = e["tier_id"]
        if tid in gmv2.V2_SUBSET_SIZE_BY_TIER:
            continue
        v1_per = gmv2.V1_PER_EPOCH_BY_TIER[tid]
        ratio = gmv2.caliber_subset_ratio(
            tid, gmv2.V1_SUBSET_SIZE_BY_TIER[tid], int(e["data"]["subset_size"]),
            tid in gmv2.NOT_APPLICABLE_TIERS)
        cards = max(1, int(e["cards"]))
        dp = gmv2.run_config_dp_size(tid, str(e.get("backend") or ""))
        dp_eff = cards if dp is None else max(1, int(dp))
        want = int(v1_per) * int(ratio) * cards // dp_eff
        if int(e["expected_optimizer_steps_per_epoch"]) != want:
            bad.append((tid, v1_per, ratio, cards, dp_eff,
                        e["expected_optimizer_steps_per_epoch"], want))
        if dp_eff != cards:
            pp_seen.append(tid)
    assert bad == [], f"非 native 档 per_epoch 未按口径比率 × (卡数/dp_size) 放大：{bad[:5]}"
    # ★ 反向：PP 档（dp_size < cards）**确实**被折算过（⛔ 不是"没改到"）
    assert sorted(pp_seen) == ["T017", "T018"], pp_seen


def test_pp_tiers_planned_steps_are_not_divided_by_cards():
    """★★ **W-7 定案的能失败断言**：native 全参 **PP 档**（``pp_size>1 ∧ dp_size=1``）的
    每 epoch 计划步数 **⛔ 不得按卡数相除** —— 真机锚 = `SFT complete: steps=400`。

    真机读数（两树一致，见 task-run-batch3 / task-run-ff267a8 证据）：
      · 新树 `ff267a8` 的 T018 = **400/400** 步（`elapsed=915.6s`，墙钟 1086 s）；
      · 旧树 `619d65b` 的 T018 = **400/400** 步（1081 s）⇒ 真值 = **400**，与代码树无关。
    ⇒ `subset(100) ÷ dp_size(1) × epochs(4) = 400` 才是声明真值；
      修前的「÷ cards」式给出 `100÷4×4 = 100`（T017 给出 200）⇒ **必红**。
    """
    doc = _regen_doc()
    by = {t["tier_id"]: t for t in doc["tiers"]}
    real_steps = {"T017": 400, "T018": 400}   # 真机实测（两树一致）
    for tid, real in real_steps.items():
        e = by[tid]
        cards = int(e["cards"])
        dp = gmv2.run_config_dp_size(tid, "native")
        epochs = gmv2.run_config_epochs(tid)
        assert dp == 1 and cards > 1 and epochs is not None, (tid, cards, dp, epochs)
        # ① 修后声明 == 真机实测
        assert int(e["expected_optimizer_steps_reachable"]) == real, (
            tid, e["expected_optimizer_steps_reachable"], real)
        # ② 修前式（按卡数）**必须 != 真值** —— 否则本断言无鉴别力（红不了）
        old = int(gmv2.V1_PER_EPOCH_BY_TIER[tid]) * int(epochs)
        assert old != real, (tid, old, real)
        # ③ 修后式（按 dp_size）== 真值
        subset = int(e["data"]["subset_size"])
        assert subset // dp * int(epochs) == real, (tid, subset, dp, epochs, real)
        # ④ 折算因子 == cards/dp（本 54 档 dp=1 ⇒ == cards）；⛔ 不得为 1（那就是没折算）
        assert int(e["expected_optimizer_steps_per_epoch"]) == subset // dp
        assert int(e["expected_optimizer_steps_per_epoch"]) != int(gmv2.V1_PER_EPOCH_BY_TIER[tid])


def test_cross_check_manifest_against_matrix_passes():
    """★ 跨件不变量：全 54 档 `manifest.per_epoch == docs.per_epoch_steps`。"""
    # ★ W-7 已定案闭合（零豁免；`KNOWN_PRE_EXISTING_PER_EPOCH_DRIFT` 已清空）。
    assert gmv2.cross_check_manifest_against_matrix(matrix_path=_LEGACY_MATRIX) == []


def test_on_disk_per_epoch_matches_matrix_for_all_but_w7_tiers():
    """★ 独立（不复用被测函数）复核：不齐的档**必须恰等于** W-7 具名豁免集。

    ★ 2026-10-17 W-7 定案后该豁免集**已空** ⇒ 本断言升级为更强的「**全 54 档逐档相等**」：
      `drift` 必须恰等于空集（若再出现任何漂移档，当场红）。⛔ 用例名保留以维持既有引用。
    """
    man = json.loads(gmv2.V2_MANIFEST_PATH.read_text(encoding="utf-8"))
    rows = {r["tier"]: r for r in json.loads(
        _LEGACY_MATRIX.read_text(encoding="utf-8"))["rows"]}
    drift = sorted(e["tier_id"] for e in man["tiers"]
                   if int(e["expected_optimizer_steps_per_epoch"]) != int(rows[e["tier_id"]]["per_epoch_steps"]))
    assert tuple(drift) == tuple(sorted(gmv2.KNOWN_PRE_EXISTING_PER_EPOCH_DRIFT)), drift


def test_cross_check_turns_red_when_the_ratio_is_removed(monkeypatch):
    """★★ **B-1 的构造性阳性对照**：把口径比率拿掉（恒 1）⇒ 跨件机检**必须报 18 档**。

    （= 精确复现修正轮 1 的缺陷：非 native 分支退化为"透传 v1"。）
    """
    import tempfile
    monkeypatch.setattr(gmv2, "caliber_subset_ratio", lambda *a, **k: 1)
    doc = _regen_doc()
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "regen.json"
        p.write_text(json.dumps(doc, ensure_ascii=False), encoding="utf-8")
        bad = gmv2.cross_check_manifest_against_matrix(p, matrix_path=_LEGACY_MATRIX)
    hit = sorted({x.split(":")[0] for x in bad if x.startswith("T")})
    assert len(hit) == 18, hit
    assert set(hit) == set(gmv2.MINI_DECLARED_SUBSET_SIZE), hit


def test_cross_check_exemption_is_not_a_silent_whitelist():
    """★ 豁免表**反向断言**（W-7 定案后仍必须有效）：登记的档必须**确实仍在漂**。

    ★ 2026-10-17 更新：`KNOWN_PRE_EXISTING_PER_EPOCH_DRIFT` 已清空（T017/T018 的 per_epoch
      已按 dp_size 修正 ⇒ 不再漂；默认豁免下同一份 doc 必须全绿）。⇒ 本用例改为**构造**一个
      「陈旧豁免」（把**已不再漂**的 T017 塞回豁免表）⇒ 跨件机检必须报「豁免表陈旧」（fail-closed），
      证明该反向断言**没有被删掉**、机制仍活。
    """
    import tempfile
    doc = _regen_doc()   # W-7 定案后：T017 已是正确值（100/400），**不再漂**
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "fixed.json"
        p.write_text(json.dumps(doc, ensure_ascii=False), encoding="utf-8")
        # ★ 默认（空豁免表）：T017 已真的不漂 ⇒ 全绿
        assert gmv2.cross_check_manifest_against_matrix(p, matrix_path=_LEGACY_MATRIX) == []
        # ★ 构造陈旧豁免 ⇒ 必须报出
        bad = gmv2.cross_check_manifest_against_matrix(p, matrix_path=_LEGACY_MATRIX, exempt_drift=("T017",))
    assert any("豁免表陈旧" in x and "T017" in x for x in bad), bad


def test_check_asserts_non_native_per_epoch_on_disk():
    """★ 生成器侧断言在**在盘件**上也生效：把 T031 的 per_epoch 改回 160 ⇒ `check()` 必须 rc≠0。"""
    import shutil
    import tempfile
    man = json.loads(gmv2.V2_MANIFEST_PATH.read_text(encoding="utf-8"))
    for e in man["tiers"]:
        if e["tier_id"] == "T031":
            e["expected_optimizer_steps_per_epoch"] = 160
            e["expected_optimizer_steps_reachable"] = 160
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "regressed.json"
        p.write_text(json.dumps(man, ensure_ascii=False), encoding="utf-8")
        rc = gmv2.check(p)
    assert rc != 0


def test_derived_state_regression_guard_is_fail_closed():
    """★★ W-1b：重生成会把跑次台账的**派生 status** 改写 ⇒ 必须报出（45 档）；同件同值 ⇒ 0。"""
    disk = json.loads(gmv2.V2_MANIFEST_PATH.read_text(encoding="utf-8"))
    assert gmv2.derived_state_regression(disk, gmv2.V2_MANIFEST_PATH) == []
    drift = gmv2.derived_state_regression(_regen_doc(), gmv2.V2_MANIFEST_PATH)
    assert len(drift) == 45, len(drift)


# ══════════════════════════════════════════════════════════════════════════════
# ④ W-8 的 `prior_chain` 规范形：**两条互补机核**（A1b，与 A1 同裁定同根因）
#
# 病史：原先这里有一条 `test_prior_chain_is_canonical_on_disk`，它读**在盘发布件**
#   `docs/capability-matrix.data.json` 并断言恰好 9 条 `prior_chain`。
#   矩阵整理包按 §17.7 把发布件收成**面向使用者的投影**、剔除了
#   `rows[].eval_caliber_consistency`（含 `prior_chain`）——该字段留在发布件会让
#   `scan_matrix_leaks` 命中 40 处（真实宿主路径 11 / 数据集名 11 / 内部称谓 18）。
#   ⇒ 那条断言的**对象在发布件里已不存在**（实测 `checked == 0`）⇒ **前提消失**。
#   ★ 裁定：⛔ 不是"把 9 改成 0"变绿（那是把**守卫**删掉），而是换成**两条互补机核**：
#     本段 ④-1（**本文件域的产物**：三份 v1/v2/fsdp 清单 ⛔ 不得带第二份未把关副本）
#     ＋ ④-2（**内部权威侧**：链**零丢失**且**逐档走唯一判据**；`.local/` 不可达即 skip）。
#   ★ 与 A1 的分工（§1.4 单一真相源，⛔ 不重复）：
#     · A1 的 `test_verify_matrix_guard.py` 管 **发布投影**（`docs/capability-matrix.data.json`）
#       不得含内部来源键 ＋ 判据的**单元级**规范形（该文件 🟢 ⑤/⑥）；
#     · 本文件管**清单线**（`matrix54_manifest*.json`）—— **对象与 A1 不相交**，
#       且 ④-2 **复用** `build_official.caliber_field_guard` 这一份判据实现（⛔ 不抄规则）。
# ══════════════════════════════════════════════════════════════════════════════

#: ★ 本文件域的**在盘产物**（具名，⛔ 不用通配符）。`docs/**` **不在**此列 —— 那是 A1 的职责面。
_MANIFEST_ARTIFACTS = (
    gmv2.V1_MANIFEST_PATH,                                              # v1（只读基线）
    gmv2.V2_MANIFEST_PATH,                                              # v2（runner 权威清单）
    gmv2.V2_MANIFEST_PATH.parent / "matrix54_manifest_fsdp.json",       # v2 变体
)

#: 内部口径审计记录（含 `prior_chain`）的键名 —— 它们只准存在于**内部权威**。
_AUDIT_RECORD_KEYS = (
    "eval_caliber_consistency", "status_after_correction_20261016",
    "prior_chain", "caliber_independent_evidence",
)

#: 内部权威（`.local/`，⛔ 不入库）⇒ 不可达时 ④-2 **skip**（⛔ 不得让 CI 失败）。
_AUTHORITY_PATH = (gmv2.PROJECT_ROOT / ".local" / "hb-workspace" / "20261008-105451"
                   / "task-matrix-html" / "matrix-data.json")


def test_manifest_line_carries_no_unguarded_prior_chain_copy():
    """★ ④-1（**本文件域**，CI 安全）：三份清单 ⛔ 不得带 `prior_chain` 的**第二份未把关副本**。

    ★ 为什么这不是"重复 A1"：A1 管的是**发布投影**（`docs/**`），本用例管的是**清单线**
      `matrix54_manifest{,_source,_fsdp}.json` —— **对象不相交**（见上方分工说明）。
    ★ 为什么这是真不变量（不是同义反复）：一旦有人为"让 runner 也能看到口径审计"而把
      `eval_caliber_consistency` 接进清单，W-8 的规范形契约就会出现**第二份副本**，
      而它不经过 `build_official.caliber_field_guard` ③ 把关 ⇒ §1.4 单一真相源破。
    """
    present: dict = {}
    for p in _MANIFEST_ARTIFACTS:
        if not p.is_file():
            continue
        raw = p.read_text(encoding="utf-8")
        hit = [k for k in _AUDIT_RECORD_KEYS if f'"{k}"' in raw]
        if hit:
            present[p.name] = hit
    assert present == {}, (
        f"清单线出现内部口径审计记录的第二份副本（⛔ 它未经 caliber_field_guard 把关）：{present}"
    )


def test_internal_authority_prior_chains_are_canonical_and_not_lost():
    """★ ④-2（**内部权威侧**，`.local/` 不可达即 skip）：链**零丢失**且**逐档走唯一判据**。

    ★ **复用而非复制**：三条规范形规则（`prior` 开头 / 有序去重 / ⛔ 不含 `corrected_to`）
      **只有一份实现** —— `tools/matrix/build_official.py::caliber_field_guard` ③；
      本用例 **import 它调用**（⛔ 不在此处再写一遍规则）。
    ★ 本用例补的是 A1 侧的**另一层**：**零丢失**（发布件剔除该字段后，内部权威必须仍保有
      全部 9 条 —— 剔除是"投影"不是"删除数据"）＋ 记录结构完整性。
    ★ **CI 无 `.local/` ⇒ skip**，⛔ 本用例不构成任何其它断言的前提。
    """
    if not _AUTHORITY_PATH.is_file():
        pytest.skip("内部权威不在盘（CI 常态）⇒ 仅在本地核这条")
    import sys

    _matrix_dir = gmv2.PROJECT_ROOT / "tools" / "matrix"
    if str(_matrix_dir) not in sys.path:
        sys.path.insert(0, str(_matrix_dir))   # `build_official` 依赖同目录 `extract_spec`/`extract_params`
    spec = importlib.util.spec_from_file_location("build_official_for_a1b", _matrix_dir / "build_official.py")
    assert spec is not None and spec.loader is not None
    bo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bo)

    rows = json.loads(_AUTHORITY_PATH.read_text(encoding="utf-8"))["rows"]
    checked, tiers = 0, []
    for r in rows:
        ecc = r.get("eval_caliber_consistency")
        rec = ecc.get("status_after_correction_20261016") if isinstance(ecc, dict) else None
        if not isinstance(rec, dict) or not isinstance(rec.get("prior_chain"), list):
            continue
        checked += 1
        tiers.append(str(r.get("tier")))
        assert rec.get("prior") is not None, (r.get("tier"), rec)
        assert rec.get("corrected_to") is not None, (r.get("tier"), rec)
        # ★ 走**唯一判据**（复用），⛔ 不在此处重写规则
        probs = [x for x in bo.caliber_field_guard([r]) if "prior_chain" in x]
        assert probs == [], (r.get("tier"), probs)
    assert checked == 9, f"内部权威的修正记录数应为 9（零丢失），实为 {checked}"
    assert len(set(tiers)) == 9, tiers


# ══════════════════════════════════════════════════════════════════════════════
# ⑤ **三向一致** ＋ **epoch 单真相源**（2026-10-16 修复包）
#
# 病史（起跑包实跑发现、指挥定性后批准修）：在盘清单的
#   `expected_optimizer_steps_reachable`（全 54 档 = per_epoch × **1**）与
#   `checkpoint_save_steps`（native 从**已退役**目录读、其余走 v1 透传）都不是跑次真值 ⇒
#   T010 = 100/100/100，而它自带 yaml 是 `max_epochs: 4` ＋ `save_steps: 401`、
#   发布矩阵是 `derived_steps=400` / `save_steps=401` ⇒ **清单自相矛盾 ＋ 与跑次 yaml 矛盾**。
# 根因（**一个**，不是三个）：epoch 数存在**两个真相源** —— 生成器用 v1 冻结常量
#   `gm.MATRIX_MAX_EPOCHS`（=1），而 2026-10-15 主席把 epoch 口径改为 SFT=4／CPT/OPD=2／
#   GRASPO=8 后，矩阵与档位 yaml 都跟上了，**只有本生成器没跟** ⇒ `reachable` 低估 2–8 倍。
# 为什么原来的 B-1 跨件机检漏检：它**只读矩阵行的 `per_epoch_steps` 一个字段**，
#   而 `max_epochs` **只存在于矩阵行**（清单无此字段）⇒ 需要 `max_epochs` 才能算出的量
#   （`reachable` / `css`）**结构上不在它的定义域内**。矩阵行本来就有
#   `max_epochs` / `derived_steps` / `save_steps` ⇒ 三向对拍完全可做。
# ══════════════════════════════════════════════════════════════════════════════


def _mutant_manifest(tmp: Path, mutate) -> Path:
    """造一份**清单变异体**（⛔ 不碰在盘件）：`mutate(doc)` 就地改内存里的副本。"""
    doc = json.loads(gmv2.V2_MANIFEST_PATH.read_text(encoding="utf-8"))
    mutate(doc)
    p = tmp / "mutant-manifest.json"
    p.write_text(json.dumps(doc, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return p


def _tier_of(doc: dict, tid: str) -> dict:
    return next(t for t in doc["tiers"] if str(t["tier_id"]) == tid)


def test_manifest_matrix_yaml_three_way_consistency_is_green():
    """★ 三向一致（清单 ↔ 发布矩阵 ↔ 档位 yaml）：在盘三件必须**零问题**。

    三向 = ① `manifest.expected_optimizer_steps_per_epoch == matrix.per_epoch_steps`
           ② `manifest.expected_optimizer_steps_reachable == matrix.derived_steps`
           ③ `manifest.checkpoint_save_steps == matrix.save_steps == <该档 yaml>.training.save_steps`
    """
    assert gmv2.cross_check_manifest_against_matrix(matrix_path=_LEGACY_MATRIX) == []


@pytest.mark.parametrize(
    "mutate, needle",
    [
        # ② reachable 被改回旧值（本包修的那个病）
        (lambda d: _tier_of(d, "T010").__setitem__("expected_optimizer_steps_reachable", 100),
         "!= matrix.derived_steps"),
        # ③ css 被改回旧值
        (lambda d: _tier_of(d, "T010").__setitem__("checkpoint_save_steps", 100),
         "training.save_steps=401"),
        # ③' css 差 1（"只差一点"也要抓）
        (lambda d: _tier_of(d, "T010").__setitem__("checkpoint_save_steps", 402),
         "training.save_steps=401"),
        # ③'' eos 与 css 分叉（同清单内不变量）
        (lambda d: _tier_of(d, "T010").__setitem__("expected_optimizer_steps", 999),
         "同清单内不变量破"),
        # ① B-1 老病（per_epoch 回退）仍被同一条机检抓住
        (lambda d: _tier_of(d, "T031").__setitem__("expected_optimizer_steps_per_epoch", 160),
         "per_epoch_steps=800"),
        # ④ 重投闸门：有档跨过 `reachable < 门槛` ⇒ 必须报（"先跑后收集"的适用前提）
        (lambda d: _tier_of(d, "T010").__setitem__("expected_optimizer_steps_reachable", 4),
         "会触发 A2 第三态"),
    ],
)
def test_three_way_check_detects_each_mutation(tmp_path, mutate, needle):
    """★ 阳性对照：**任一**被改回旧值/改坏 ⇒ 三向机检必须**红**（⛔ 不是"扫过了就算"）。"""
    p = _mutant_manifest(tmp_path, mutate)
    probs = gmv2.cross_check_manifest_against_matrix(p, matrix_path=_LEGACY_MATRIX)
    assert probs, "变异体上机检没报任何问题 ⇒ 对照失效"
    assert any(needle in x for x in probs), (needle, probs)


def test_three_way_check_detects_mutated_tier_yaml(tmp_path):
    """★ 第三向（**档位 yaml**）的阳性对照：只改 yaml（清单/矩阵都还对）也必须被抓。

    为什么必须有这一向：只比"清单 ↔ 矩阵"会漏掉"矩阵对、而**跑次实际读的 yaml** 错"——
    那一类错误清单与矩阵永远自洽地错，正是 B-1 的漏检形态。
    """
    import shutil

    cfg = tmp_path / "cfg"
    shutil.copytree(gmv2.PROJECT_ROOT / gmv2.CONFIG_DIR_REL, cfg)
    y = cfg / "T010.yaml"
    y.write_text(y.read_text(encoding="utf-8").replace("save_steps: 401", "save_steps: 777"),
                 encoding="utf-8")
    saved_root, saved_rel = gmv2.PROJECT_ROOT, gmv2.CONFIG_DIR_REL
    # ★ 矩阵路径必须**显式给真实件**：`PROJECT_ROOT` 被指向 tmp 后，默认矩阵路径会落空
    #   ⇒ 函数会在"矩阵不在盘"处早退，测不到 yaml 这一向（本用例第一版就踩了这个坑）。
    real_matrix = _LEGACY_MATRIX
    try:
        gmv2.PROJECT_ROOT, gmv2.CONFIG_DIR_REL = tmp_path, "cfg"
        probs = gmv2.cross_check_manifest_against_matrix(matrix_path=real_matrix)
        assert any("training.save_steps=777" in x for x in probs), probs
    finally:
        gmv2.PROJECT_ROOT, gmv2.CONFIG_DIR_REL = saved_root, saved_rel


def test_epoch_has_exactly_one_truth_source():
    """★★ **epoch 单真相源**：扰动 v1 冻结常量后，除具名豁免档外**任何档都不得变化**。

    ★ 这条测的不是"值对不对"，而是**"第二个来源的那条路已经断掉"**——本次全 54 档
      `reachable` 低估 2–8 倍的根因就是那条路。⛔ 只改值不测路 = 日后必然复发。
    ★ 阳性对照（同一断言的反面）：把 `run_config_epochs` 换回旧实现 ⇒ 必须红。
    """
    v1 = json.loads(gmv2.V1_MANIFEST_PATH.read_text(encoding="utf-8"))["tiers"]
    assert gmv2.epoch_single_source_probe(v1) == []
    saved = gmv2.run_config_epochs
    try:
        gmv2.run_config_epochs = lambda tid: int(gmv2.gm.MATRIX_MAX_EPOCHS)  # 复现旧实现
        bad = gmv2.epoch_single_source_probe(v1)
        assert len(bad) >= 51, f"复现旧实现后应几乎全档变化，实测 {len(bad)}"
    finally:
        gmv2.run_config_epochs = saved


def test_reachable_uses_the_run_config_epochs_not_a_frozen_constant():
    """★ 值层面：`reachable` 必须 = `per_epoch × 该档 yaml 的 max_epochs`（逐档、含档例算式）。

    档例（`evidence/58`）：T010 = 100 × 4 = **400**（改前 100）；T031 = 800 × 8 = **6400**（改前 800）。
    """
    doc = _regen_doc()
    checked = 0
    for e in doc["tiers"]:
        tid = str(e["tier_id"])
        ep = gmv2.run_config_epochs(tid)
        if ep is None:
            continue
        checked += 1
        assert int(e["expected_optimizer_steps_reachable"]) == \
            int(e["expected_optimizer_steps_per_epoch"]) * int(ep), tid
    assert checked == 51, checked
    t010 = _tier_of(doc, "T010")
    t031 = _tier_of(doc, "T031")
    assert (t010["expected_optimizer_steps_reachable"], t010["checkpoint_save_steps"]) == (400, 401)
    assert (t031["expected_optimizer_steps_reachable"], t031["checkpoint_save_steps"]) == (6400, 6401)


def test_checkpoint_save_steps_comes_from_the_run_config_yaml():
    """★ 值层面：`css` 必须逐档 == 跑次 yaml 的 `training.save_steps`（⛔ 不再读退役目录/透传 v1）。"""
    doc = _regen_doc()
    for e in doc["tiers"]:
        tid = str(e["tier_id"])
        yaml_ss = gmv2.run_config_scalar(tid, "save_steps")
        if yaml_ss is None:
            continue
        assert int(e["checkpoint_save_steps"]) == int(yaml_ss), tid
        assert int(e["expected_optimizer_steps"]) == int(yaml_ss), tid


def test_no_yaml_exemption_is_exactly_the_non_runnable_tiers():
    """★ 豁免**不是静默白名单**（反向断言）：无 yaml 集合必须**恰好**等于具名豁免表，
    且这些档必须 `applicable=False`（否则"没有跑次配置"这个前提就不成立）。"""
    man = json.loads(gmv2.V2_MANIFEST_PATH.read_text(encoding="utf-8"))
    have = {str(t["tier_id"]) for t in man["tiers"] if gmv2.run_config_scalar(str(t["tier_id"]), "save_steps") is not None}
    no_yaml = {str(t["tier_id"]) for t in man["tiers"]} - have
    assert no_yaml == set(gmv2.NOT_APPLICABLE_NO_CONFIG), (sorted(no_yaml), gmv2.NOT_APPLICABLE_NO_CONFIG)
    for tid in no_yaml:
        assert not bool(_tier_of(man, tid).get("applicable")), tid
