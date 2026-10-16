"""matrix54 v2 清单生成器：卡位覆盖 + **通用锁卡断言**的回归覆盖。

**为什么必须有这个文件**：运行清单 `matrix54_manifest_source.json` 是**生成物**。
只手改清单文件的话，下次重生成会按 `GPU_POOL_BY_CARDS` 把卡位静默改回（回退陷阱），
而当时的测试**不会报警**（既有 `test_generate_matrix.py` 断言的是 **v1 生成器**的产物，
不校 v2 清单文件本身）。

本文件把三件事固化成断言：
  ① 2026-10-05 主席令的默认映射 `GPU_POOL_BY_CARDS` **逐字未变**；
  ② **2026-10-16 撤销的陈旧例外**：`TIER_GPU_OVERRIDE` 现为**空表**；T041 走默认映射
     ⇒ `gpus == [4,5]`、`cards` **仍为 2**（卡数属基础列，⛔ 不随之改）；
  ③ ★ **通用锁卡断言**（`locked_card_violation` + 「例外必须落在本世代卡池内」）：
     **任何档**的 `gpus` 都不得含邻居常驻卡（GPU2）/ 生产锁卡（GPU6/7）/ 越界卡。

★ 病史（为什么 ③ 不能只写成"把 `[2,3]` 改成 `[4,5]`"）：只改期望值 = 把**能抓到这条缺陷的
  能力**一起删掉。2026-10-09 的例外 `[2,3]` 把第三方账号占用中的常驻邻居卡 GPU2 写进了我方声明
  （`docs/card-pool-contract.md` 规则 4「邻居资源只读不占…只观测、不排队、不干预」），
  而当时的断言只锁「覆盖表 == {T041: [2,3]}」⇒ **只固化了错误，抓不到错误**。
  ⇒ 现在：既断言新权威值，也断言**判据本身**（阳性对照 =
  `test_locked_card_assertion_rejects_the_stale_override` /
  `test_override_outside_generation_pool_is_fail_closed`）。

★ 另一条已知冲突（**只登记、⛔ 不擅自改池**，由指挥统一处置）：本世代 4 卡池
  `[4,5,6,7]` 与 `gpu_assignment.GPU_FORBIDDEN_PRODUCTION = (6,7)` 相交 ⇒
  `locked_card_violation` 对"恰等于卡池"的值**不 raise**，但会登记进
  `POOL_CONTRACT_CONFLICTS` 并写入清单 `gpu_pool_contract_conflicts`
  （见 `test_pool_contract_conflict_is_registered_not_silently_allowed`）。
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

_HERE = Path(__file__).resolve()
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
    """2026-10-05 主席令的卡数→卡位映射**逐字未变**（覆盖表不得顶替它）。"""
    assert gmv2.GPU_POOL_BY_CARDS == {1: [4], 2: [4, 5], 3: [4, 5, 6], 4: [4, 5, 6, 7]}


# ── ② 陈旧例外已撤销 ────────────────────────────────────────────────────────


def test_tier_gpu_override_is_now_empty():
    """★ 2026-10-16：陈旧例外（T041 `[2,3]`，2026-10-09 的一次性调度值）已撤销 ⇒ 覆盖表为空。"""
    assert gmv2.TIER_GPU_OVERRIDE == {}


def test_t041_falls_back_to_pool_and_cards_unchanged():
    """★ 核心断言：T041 走默认映射 ⇒ `gpus == [4,5]`，而 `cards`（卡数）**仍为 2**。"""
    out = _patched("T041")
    assert out["gpus"] == [4, 5]
    assert out["cards"] == 2
    assert len(out["gpus"]) == out["cards"]


def test_t041_revocation_reason_is_written_down():
    """⛔ 撤销必须连同**理由**一起写清（否则下一个人会以为是漏改）。"""
    src = (_HERE.parent / "generate_matrix_manifest_source.py").read_text(encoding="utf-8")
    assert "2026-10-16 撤销" in src
    assert "邻居" in src
    # ★ §15 去标识（修正轮 1 / W4）：正式件里第三方账号**一律角色化** ⇒ 生成器源码亦不得含账号字面量。
    assert "第三方账号占用" in src
    assert "remote" not in src
    assert "docs/card-pool-contract.md" in src


def test_override_only_changes_gpus():
    """★ 覆盖只改 `gpus`：给一个池内合法覆盖后逐字段对比，除 `gpus` 外**一字不改**。"""
    default = _patched("T041")
    assert default["gpus"] == [4, 5]
    gmv2.TIER_GPU_OVERRIDE = {"T041": [5, 4]}          # 池内、卡数一致 ⇒ 合法覆盖
    try:
        overridden = _patched("T041")
    finally:
        gmv2.TIER_GPU_OVERRIDE = {}
    assert overridden["gpus"] == [5, 4]
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


def test_locked_cards_come_from_a_single_source():
    """具名锁卡 = 生产锁卡 ∪ 邻居卡，且**取自** `gpu_assignment`（§1.4 单一真相源）；
    越界卡不枚举、由 `GPU_ALLOWED` 成员判定（见 `test_locked_card_assertion_rejects_each_locked_class`）。"""
    assert gmv2.NEIGHBOR_CARDS == (2,)                 # GPU2 = 第三方账号占用中的常驻邻居卡
    assert tuple(gmv2.GPU_FORBIDDEN_PRODUCTION) == (6, 7)
    assert tuple(gmv2.GPU_ALLOWED) == (0, 1, 2, 3, 4, 5)
    assert gmv2.LOCKED_CARDS == (2, 6, 7)
    assert gmv2.locked_card_hits([8]) == (8,)           # 越界 leg 生效
    assert gmv2.locked_card_hits([0]) == ()             # 合法卡不误伤


def test_no_tier_declares_a_locked_card():
    """★ 全 54 档：`gpus` 不得含邻居卡 / 生产锁卡 / 越界卡（本世代卡池自身的冲突除外）。"""
    offenders = []
    for tier in _V1_TIERS:
        out = _patched(tier["tier_id"])
        hits = gmv2.locked_card_hits(out["gpus"])
        if not hits:
            continue
        pool = tuple(gmv2.GPU_POOL_BY_CARDS[int(out["cards"])])
        if tuple(out["gpus"]) == pool:
            continue                                    # 卡池自身冲突 ⇒ 已登记分支
        offenders.append((tier["tier_id"], out["gpus"], hits))
    assert offenders == [], f"以下档声明了被锁的卡：{offenders}"


def test_no_tier_declares_the_neighbour_card():
    """★ 邻居卡 GPU2 在我方**全部 54 档**声明里必须是 **0 次**（本包的核心修复目标）。"""
    with_gpu2 = [t["tier_id"] for t in _V1_TIERS if 2 in _patched(t["tier_id"])["gpus"]]
    assert with_gpu2 == [], f"仍声明了邻居 GPU2 的档：{with_gpu2}"


def test_locked_card_assertion_rejects_the_stale_override():
    """★ **阳性对照固化**：把改前的陈旧例外 `[2,3]` 放回去，断言**必须**拒（否则断言是假的）。"""
    reason = gmv2.locked_card_violation("T041", [2, 3], 2)
    assert reason is not None
    assert "GPU2" in reason and "邻居" in reason


def test_locked_card_assertion_rejects_each_locked_class():
    """三类锁卡各自必须被拒（逐类阳性对照）。"""
    assert gmv2.locked_card_violation("X", [4, 2], 2) is not None      # 邻居卡
    assert gmv2.locked_card_violation("X", [6, 7], 2) is not None      # 生产锁卡
    assert gmv2.locked_card_violation("X", [4, 8], 2) is not None      # 越界卡（∉ GPU_ALLOWED）


def test_locked_card_assertion_passes_for_a_clean_set():
    assert gmv2.locked_card_violation("X", [4, 5], 2) is None
    assert gmv2.locked_card_hits([4, 5]) == ()


def test_pool_contract_conflict_is_registered_not_silently_allowed():
    """★ 本世代 4 卡池 `[4,5,6,7]` 与 `GPU_FORBIDDEN_PRODUCTION=(6,7)` 相交 ⇒
    **不 raise**（⛔ 本生成器不擅自改池）但**必须登记**（机读、可审计）。"""
    gmv2.POOL_CONTRACT_CONFLICTS.clear()
    gmv2._uncaptured_pool_conflicts.clear()
    assert gmv2.locked_card_violation("T003", [4, 5, 6, 7], 4) is None
    assert gmv2.POOL_CONTRACT_CONFLICTS == {(4, 5, 6, 7): (6, 7)}
    assert [t for t, _p, _h in gmv2._uncaptured_pool_conflicts] == ["T003"]
    gmv2.POOL_CONTRACT_CONFLICTS.clear()
    gmv2._uncaptured_pool_conflicts.clear()


# ── 覆盖表本身的 fail-closed ────────────────────────────────────────────────


def test_override_card_count_mismatch_is_fail_closed():
    """覆盖表卡数 ≠ `cards` ⇒ fail-closed（不静默"将就"成错卡数）。"""
    gmv2.TIER_GPU_OVERRIDE = {"T041": [4, 5, 6]}
    try:
        with pytest.raises(AssertionError, match="卡数"):
            _patched("T041")
    finally:
        gmv2.TIER_GPU_OVERRIDE = {}


def test_override_duplicate_gpus_is_fail_closed():
    """覆盖表含重复卡 ⇒ fail-closed（重复卡会让"卡数"名不副实）。"""
    gmv2.TIER_GPU_OVERRIDE = {"T041": [4, 4]}
    try:
        with pytest.raises(AssertionError, match="重复"):
            _patched("T041")
    finally:
        gmv2.TIER_GPU_OVERRIDE = {}


def test_override_outside_generation_pool_is_fail_closed():
    """★ 显式例外**必须落在本世代卡池内** —— 这就是「陈旧调度值」缺陷的根因判据。"""
    for bad in ([2, 3], [0, 1], [4, 6]):
        gmv2.TIER_GPU_OVERRIDE = {"T041": list(bad)}
        try:
            with pytest.raises(AssertionError, match="卡池外"):
                _patched("T041")
        finally:
            gmv2.TIER_GPU_OVERRIDE = {}


def test_override_inside_generation_pool_is_accepted():
    """池内、卡数一致的例外仍被接受（断言不得把合法例外一起禁掉）。"""
    gmv2.TIER_GPU_OVERRIDE = {"T041": [4, 5]}
    try:
        assert _patched("T041")["gpus"] == [4, 5]
    finally:
        gmv2.TIER_GPU_OVERRIDE = {}


# ── 在盘清单的一致性（产物侧，若存在）────────────────────────────────────────


def test_on_disk_manifest_has_no_locked_card_declaration():
    """在盘 v2 清单：⛔ 不得声明邻居卡；生产锁卡只允许出现在"等于本世代卡池"的形态里。"""
    p = gmv2.V2_MANIFEST_PATH
    if not p.is_file():
        pytest.skip("v2 清单不在盘")
    manifest = json.loads(p.read_text(encoding="utf-8"))
    assert [t["tier_id"] for t in manifest["tiers"] if 2 in (t.get("gpus") or [])] == []
    non_pool = []
    for t in manifest["tiers"]:
        hits = gmv2.locked_card_hits(t.get("gpus") or [])
        if not hits:
            continue
        if tuple(t["gpus"]) != tuple(gmv2.GPU_POOL_BY_CARDS[int(t["cards"])]):
            non_pool.append((t["tier_id"], t["gpus"], hits))
    assert non_pool == [], f"在盘清单含非卡池形态的锁卡声明：{non_pool}"


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
    # 生成器**不许**再抄一份常量：三张表必须与 make_subsets 逐字相同
    import importlib.util as _iu
    spec = _iu.spec_from_file_location("_t_ms", gmv2.MAKE_SUBSETS_PATH)
    ms = _iu.module_from_spec(spec)
    spec.loader.exec_module(ms)
    assert gmv2.MINI_DECLARED_SUBSET_SIZE == ms.MINI_SUBSET_TIERS_DECLARED
    assert gmv2.MINI_LANDED_SUBSET_SIZE == ms.MINI_TIERS
    assert gmv2.MINI_DECLARED_NOT_LANDED == ms.DECLARED_NOT_LANDED


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
    assert gmv2.check(gmv2.V2_MANIFEST_PATH) == 0


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
    """★ B-1 根因回归：生成器非 native 档 `per_epoch` 必须 = v1 值 × 口径比率（⛔ 不得透传）。"""
    doc = _regen_doc()
    bad = []
    for e in doc["tiers"]:
        tid = e["tier_id"]
        if tid in gmv2.V2_SUBSET_SIZE_BY_TIER:
            continue
        v1_per = gmv2.V1_PER_EPOCH_BY_TIER[tid]
        ratio = gmv2.caliber_subset_ratio(
            tid, gmv2.V1_SUBSET_SIZE_BY_TIER[tid], int(e["data"]["subset_size"]),
            tid in gmv2.NOT_APPLICABLE_TIERS)
        if int(e["expected_optimizer_steps_per_epoch"]) != int(v1_per) * int(ratio):
            bad.append((tid, v1_per, ratio, e["expected_optimizer_steps_per_epoch"]))
    assert bad == [], f"非 native 档 per_epoch 未按口径比率放大：{bad[:5]}"


def test_cross_check_manifest_against_matrix_passes():
    """★ 跨件不变量：全 54 档 `manifest.per_epoch == docs.per_epoch_steps`（豁免 T017/T018 = W-7）。"""
    assert gmv2.cross_check_manifest_against_matrix() == []


def test_on_disk_per_epoch_matches_matrix_for_all_but_w7_tiers():
    """★ 独立（不复用被测函数）复核：不齐的档**必须恰等于** W-7 具名豁免集。"""
    man = json.loads(gmv2.V2_MANIFEST_PATH.read_text(encoding="utf-8"))
    rows = {r["tier"]: r for r in json.loads(
        (gmv2.PROJECT_ROOT / "docs" / "capability-matrix.data.json").read_text(encoding="utf-8"))["rows"]}
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
        bad = gmv2.cross_check_manifest_against_matrix(p)
    hit = sorted({x.split(":")[0] for x in bad if x.startswith("T")})
    assert len(hit) == 18, hit
    assert set(hit) == set(gmv2.MINI_DECLARED_SUBSET_SIZE), hit


def test_cross_check_exemption_is_not_a_silent_whitelist():
    """★ 豁免表**反向断言**：把 T017 人为修好 ⇒ 必须报「豁免表陈旧」（逼下一轮删出）。"""
    import tempfile
    doc = _regen_doc()
    for e in doc["tiers"]:
        if e["tier_id"] == "T017":
            e["expected_optimizer_steps_per_epoch"] = 100
            e["expected_optimizer_steps_reachable"] = 100
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "fixed.json"
        p.write_text(json.dumps(doc, ensure_ascii=False), encoding="utf-8")
        bad = gmv2.cross_check_manifest_against_matrix(p)
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
