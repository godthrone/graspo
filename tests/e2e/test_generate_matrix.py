"""54 档矩阵生成器的结构自检（纯 stdlib，不依赖 torch/graspo 包）。

生成器自身就是"台账逐档对应"的防线：这里把它的断言固化成测试，防止
后续改动悄悄把档位数量、行序或口径改掉。
"""

import importlib.util
import json
import re
import sys
from pathlib import Path

import pytest

#: ``runtime`` 段里禁止出现的宿主标识键（§15.1：manifest 不得承载宿主真实值）。
_FORBIDDEN_HOST_KEYS: tuple[str, ...] = ("host", "hostname", "host_name", "node", "server")

#: 键名含 ``host`` 时允许的三种形式——``*_env_var``（环境变量名）、
#: ``*_default``（``<GRASPO_...>`` 占位符）、``*_note``（说明性文字）。
#: 其余含 ``host`` 的键（如 ``host`` / ``host_root`` / ``host_dir``）一律拒绝：
#: 它们是**真实值的载体**，正是本用例要挡住的东西。
_ALLOWED_HOST_KEY_SUFFIXES: tuple[str, ...] = ("_env_var", "_default", "_note")

#: 宿主信息的两种合法表示：环境变量名 / ``<GRASPO_...>`` 占位符。
_HOST_ENV_VAR_RE = re.compile(r"\AGRASPO_[A-Z0-9_]+\Z")
_HOST_PLACEHOLDER_PREFIX = "<GRASPO_"

#: 形如**宿主**绝对路径的字符串（容器内路径如 ``/models``、``/work/elam-v5`` 不在此列）。
#: Windows 盘符必须锚定在串首，否则会误伤 ``-v "$MODELS_ROOT:/models:ro"`` 这类挂载串。
_HOST_ABS_PATH_RE = re.compile(
    r"(\A~|(?<![\w.])/(?:home|mnt|Users|root|media|srv)(?:/|\Z)|\A[A-Za-z]:[\\/])"
)


def _walk_manifest(node: object, prefix: tuple[str, ...] = ()):
    """深度遍历 manifest，产出 ``(键路径, 键名, 值)``；列表元素用下标字符串作键名。"""
    if isinstance(node, dict):
        for key, value in node.items():
            yield from _walk_manifest(value, (*prefix, str(key)))
    elif isinstance(node, list):
        for index, value in enumerate(node):
            yield from _walk_manifest(value, (*prefix, str(index)))
    else:
        yield prefix, (prefix[-1] if prefix else ""), node


_GENERATOR_PATH = Path(__file__).resolve().parent / "generate_matrix.py"
_spec = importlib.util.spec_from_file_location("generate_matrix", _GENERATOR_PATH)
assert _spec is not None and _spec.loader is not None
generate_matrix = importlib.util.module_from_spec(_spec)
sys.modules["generate_matrix"] = generate_matrix
_spec.loader.exec_module(generate_matrix)

#: 卡集合覆盖点的共享实现（生成期与运行期同一份；这里独立加载以便直接断言其函数）。
_gpu_spec = importlib.util.spec_from_file_location(
    "gpu_assignment", Path(__file__).resolve().parent / "gpu_assignment.py"
)
assert _gpu_spec is not None and _gpu_spec.loader is not None
gpu_assignment = importlib.util.module_from_spec(_gpu_spec)
sys.modules["gpu_assignment"] = gpu_assignment
_gpu_spec.loader.exec_module(gpu_assignment)

#: 生成链防丢失的共享判定实现（清单生成器与测试同一份，§1.4 单一真相源）。
_guard_spec = importlib.util.spec_from_file_location(
    "regen_field_guard", Path(__file__).resolve().parent / "regen_field_guard.py"
)
assert _guard_spec is not None and _guard_spec.loader is not None
regen_guard = importlib.util.module_from_spec(_guard_spec)
sys.modules["regen_field_guard"] = regen_guard
_guard_spec.loader.exec_module(regen_guard)


def test_ledger_has_54_tiers_with_expected_counts():
    tiers = generate_matrix.build_ledger()

    generate_matrix.assert_ledger(tiers, 54)
    assert len(tiers) == 54


def test_ledger_counts_match_capability_matrix_section_6():
    tiers = generate_matrix.build_ledger()

    assert generate_matrix._count_by(tiers, "algorithm") == {
        "CPT": 9,
        "SFT": 18,
        "GRASPO": 18,
        "OPD": 9,
    }
    assert generate_matrix._count_by_cards(tiers) == {1: 18, 2: 18, 4: 18}
    assert generate_matrix._count_by(tiers, "backend") == {"ms-swift": 36, "native": 18}
    assert generate_matrix._count_by(tiers, "mode") == {"LoRA": 36, "全量": 18}


def test_ledger_fingerprints_lock_row_order():
    """抽样锁死行序——防止"数量对、顺序错"。"""
    tiers = {tier["tier_id"]: tier for tier in generate_matrix.build_ledger()}

    def fingerprint(tier_id):
        tier = tiers[tier_id]
        return (
            tier["model"],
            tier["algorithm"],
            tier["mode"],
            tier["backend"],
            tier["cards"],
        )

    assert fingerprint("T001") == ("9B", "CPT", "LoRA", "ms-swift", 1)
    assert fingerprint("T010") == ("9B", "SFT", "LoRA", "native", 1)
    assert fingerprint("T021") == ("9B", "SFT", "全量", "ms-swift", 4)
    assert fingerprint("T028") == ("9B", "GRASPO", "LoRA", "native", 1)
    assert fingerprint("T046") == ("9B", "OPD", "LoRA", "ms-swift", 1)
    assert fingerprint("T054") == ("27B", "OPD", "LoRA", "ms-swift", 4)


def test_wrong_expected_count_fails_assertion():
    tiers = generate_matrix.build_ledger()

    with pytest.raises(AssertionError):
        generate_matrix.assert_ledger(tiers, 53)


def test_gpu_sets_are_within_allowed_range():
    for tier in generate_matrix.build_ledger():
        gpus = tier["gpus"]
        assert len(gpus) == tier["cards"]
        assert max(gpus) <= 3
        assert len(gpus) <= 4


def test_native_full_uses_pipeline_parallel_only():
    """native 全参只支持 PP 分片——生成器必须产出 pp_size=卡数、dp=tp=1。"""
    tiers = generate_matrix.build_ledger()
    for tier in tiers:
        if tier["mode"] != "全量" or tier["backend"] != "native":
            continue
        config = generate_matrix.build_config(tier)
        layout = config["native"]
        if tier["cards"] > 1:
            assert layout["pp_size"] == tier["cards"], tier["tier_id"]
        assert layout["dp_size"] == 1, tier["tier_id"]
        assert layout["tp_size"] == 1, tier["tier_id"]


def test_expressibility_counts():
    """CPT/OPD **可表达**（WP-X3 已落地通道）；native 全参 1 卡给不出可行配方；
    其余全参可表达但未验证。

    ★ 本条的口径变更（2026-09-19）：旧断言 ``{"ready": 24, "unverified": 10, "blocked": 20}``
    编码的是 "CPT/OPD 零实现" 这一**已解决**的事实。现在 18 档从 ``blocked`` 转
    ``ready``（LoRA，常规路径）/ ``unverified``（全量，⚠️ 开发目标），与 SFT/GRASPO 同口径。
    **注意 `ready`/`unverified` 只表示"可表达"，不表示"跑通"或"显存可行"**。
    """
    tiers = generate_matrix.build_ledger()

    counts = generate_matrix.assert_expressibility(tiers)

    # 24 + 9(CPT/OPD 9B LoRA) = 33；10 + 6(CPT/OPD 全量) = 16；
    # ★ 2026-09-19：T052–T054（台账的 27B OPD 档）由 ready 改判 not_applicable（确实不适用）。
    # ★ 2026-09-23：T016/T034（native 全参 1 卡）**解除 blocked** —— native 侧实现了全参
    #   优化器态 CPU offload（`native.offload_optimizer_state`），1 卡 pp=1 配方已真机实测
    #   通过（`rig/pp1-offload-T017PA.yaml`：exit=0、50/50 步、峰值 38.18 GiB；无 offload
    #   对照 79.24 GiB OOM）。⇒ `blocked` 2→0，`unverified` 16→18。
    #   注：该取证档为历史件，已随 rig/ 退役；原件见 .local/repo-moved-out/rig/。
    #   为什么落 `unverified` 而非 `ready`：本档（T016 SFT / T034 GRASPO）**尚未上机**，
    #   只有取证档 T017PA 跑通 ⇒ 与其它全量档同口径（"配方给得出、能力未验证"）。
    assert counts == {"ready": 33, "unverified": 18, "blocked": 0, "not_applicable": 3}
    blocked = {
        tier["tier_id"]
        for tier in tiers
        if generate_matrix.classify_expressibility(tier)[0] == "blocked"
    }
    # ★ 2026-09-23：**没有任何档**再落 blocked（native 全参 1 卡两档已由 offload 配方解除）
    assert blocked == set()


def test_cpt_and_opd_are_expressed_by_the_train_method_enum():
    """★ CPT/OPD 的 `train_method` 是真枚举值，且配置里带各自的中立段。"""
    tiers = {tier["tier_id"]: tier for tier in generate_matrix.build_ledger()}

    cpt = generate_matrix.build_config(tiers["T001"])  # 9B CPT LoRA 1 卡
    assert cpt["train_method"] == "cpt"
    assert cpt["pretrain"] == {"loss_scale": "all", "use_chat_template": False}

    opd = generate_matrix.build_config(tiers["T046"])  # 9B OPD LoRA 1 卡
    assert opd["train_method"] == "opd"
    assert opd["distill"]["teacher_model_path"] == generate_matrix.MODELS["27B"]["path"]
    assert opd["distill"]["offload_teacher_model"] is True
    # 学生 = 台账的 model 列；教师必须**不同**（27B → 9B）
    assert opd["model"]["model_path"] != opd["distill"]["teacher_model_path"]

    # 18 档**通道**都可表达：没有 .blocked 产物（T052–T054 走 not_applicable，见下方专项用例）
    for tier_id, tier in tiers.items():
        if tier["algorithm"] in {"CPT", "OPD"}:
            status, _ = generate_matrix.classify_expressibility(tier)
            assert status != "blocked", tier_id


# ── ★ 裁定 A（2026-09-19）：T052–T054 由"自蒸馏"改判"确实不适用" ───────────────


def test_27b_opd_tiers_are_not_applicable_with_reason():
    """★ `T052`–`T054`（台账的 27B OPD 档）必须是 `not_applicable`，且理由可读。

    **为什么要单独锁死**：这三档原口径是"27B→27B 自蒸馏"，现在按用户拍板
    （OPD 教师 = 27B 常量、学生 = 9B，省显存）改判为**确实不适用**——
    若哪天有人把 `opd_not_applicable()` 的判据改回去（或让它们重新渲染成自蒸馏配置），
    这条断言必须失败。三件事同时锁：
    ① 可表达性 = `not_applicable`；② 可行性 verdict = `not_applicable`（不是 unmeasured）；
    ③ 理由含"省显存""50.10""100 GiB"三个可核数字，且**不产出可跑配置**。
    """
    tiers = {tier["tier_id"]: tier for tier in generate_matrix.build_ledger()}

    for tier_id in ("T052", "T053", "T054"):
        tier = tiers[tier_id]
        assert tier["model"] == "27B" and tier["algorithm"] == "OPD", tier_id
        status, reason = generate_matrix.classify_expressibility(tier)
        assert status == "not_applicable", (tier_id, status)
        verdict, detail = generate_matrix.feasibility_verdict(tier)
        assert verdict == generate_matrix.VERDICT_NOT_APPLICABLE, (tier_id, verdict)
        # 不是"待上机测量"——不适用档没有该测量的配置
        assert "待上机测量" not in detail, tier_id
        for token in ("省显存", "50.10", "100 GiB"):
            assert token in detail, (tier_id, token)
        assert reason and "确实不适用" in reason, tier_id
        # 9 档 OPD 里只有这三档不适用；其余 6 档（9B）保持可测/待测量
        assert generate_matrix.opd_not_applicable(tier)[0] is True, tier_id

    # 不可产出可跑配置（查渲染结果，不查磁盘——磁盘状态依赖"生成器是否跑过"，
    # 那样用例顺序会左右结论；渲染结果才是生成器的确定输出）
    rendered = dict(generate_matrix.render_all_tiers(list(tiers.values())))
    for tier_id in ("T052", "T053", "T054"):
        assert f"{tier_id}.yaml" not in rendered, tier_id
        assert f"{tier_id}.not_applicable.md" in rendered, tier_id


def test_9b_opd_teacher_relation_is_independent_not_self_distillation():
    """`T046`–`T051`（9B OPD）必须仍是 **independent（独立教师）**，且不得被写成自蒸馏。"""
    tiers = {tier["tier_id"]: tier for tier in generate_matrix.build_ledger()}

    for tier_id in ("T046", "T047", "T048", "T049", "T050", "T051"):
        tier = tiers[tier_id]
        assert tier["model"] == "9B" and tier["algorithm"] == "OPD", tier_id
        assert generate_matrix.opd_teacher_relation(tier) == "independent", tier_id
        header = generate_matrix.opd_relation_header(tier)
        assert "independent" in header, tier_id
        assert "self-distillation" not in header, tier_id
        assert "not-applicable" not in header, tier_id

    # 不适用档的 header（防御性渲染）也不得写成自蒸馏
    for tier_id in ("T052", "T053", "T054"):
        header = generate_matrix.opd_relation_header(tiers[tier_id])
        assert "not-applicable" in header, tier_id
        assert "self-distillation" not in header, tier_id


def test_not_applicable_tiers_render_a_stub_not_a_runnable_config():
    """不适用档的产物是 `T###.not_applicable.md`，且**明确区分** blocked 与"不适用"。

    ★ 2026-09-23：本用例原先同时断言"`T016.blocked.md` 仍在产物里（blocked 档保持原样）"。
    T016/T034 解除 blocked 后该断言反过来——**`T016.blocked.md` 必须消失、改产 `T016.yaml`**
    （这是"解除 blocked"的可观测后果，必须由测试钉住，否则会留下一个过期的 blocked 存根）。
    ⇒ 本用例拆成两半：① T052 的"不适用"存根语义不变；② T016 的"已解除 blocked"由
    :func:`test_native_full_one_card_offload_recipe_replaces_the_blocked_stub` 守。
    """
    tiers = [t for t in generate_matrix.build_ledger() if t["tier_id"] in {"T052", "T016"}]
    rendered = dict(generate_matrix.render_all_tiers(tiers))

    assert "T052.not_applicable.md" in rendered
    assert "T052.yaml" not in rendered
    stub = rendered["T052.not_applicable.md"]
    assert "确实不适用" in stub
    assert "合法终态" in stub
    assert "无需排期" in stub
    # ★ 2026-09-23：blocked 存根已随判定解除而消失，改产可跑配置。
    assert "T016.blocked.md" not in rendered
    assert "T016.yaml" in rendered


def test_native_full_one_card_offload_recipe_replaces_the_blocked_stub():
    """★ 2026-09-23：native 全参 1 卡（T016/T034）由 `blocked` 转 `unverified`，
    产物从 `.blocked.md` 改为**可跑 yaml**，且配方含 `offload_optimizer_state: true`。

    为什么必须单独钉这条（§18 不留债）：解除 blocked 是个**判定反转**，它的可观测后果有三处
    ——①`classify_expressibility` 不再返回 blocked；②产物形态从存根变 yaml；③**配方字段**
    真的带上了 offload（若只解除判定却漏配 offload，档位会以"无分片 1 卡"的形态被放行 ⇒
    必然 OOM，比 blocked 更糟）。三处必须同时成立。
    """
    tiers = {t["tier_id"]: t for t in generate_matrix.build_ledger()}
    ledger = [tiers["T016"], tiers["T034"]]
    rendered = dict(generate_matrix.render_all_tiers(ledger))

    for tier_id in ("T016", "T034"):
        tier = tiers[tier_id]
        status, _ = generate_matrix.classify_expressibility(tier)
        assert status == generate_matrix.NATIVE_FULL_1CARD_EXPRESSIBILITY == "unverified", tier_id

        config = generate_matrix.build_config(tier)
        native = config["native"]
        assert native["pp_size"] == 1, tier_id
        assert native["tp_size"] == 1 and native["dp_size"] == 1, tier_id
        # ★ 唯一语义开关：offload（这是"1 卡装得下"的全部依据）
        assert native["offload_optimizer_state"] is True, tier_id

        # 产物：blocked 存根消失、yaml 出现
        assert f"{tier_id}.blocked.md" not in rendered, tier_id
        assert f"{tier_id}.yaml" in rendered, tier_id

        # 显存算式吃的是 offload 口径，且不超预算
        gib, basis = generate_matrix.estimate_config_per_card_gib(tier, config)
        assert "offload" in basis, tier_id
        assert gib <= generate_matrix.CARD_BUDGET_GIB, (tier_id, gib)


def test_manifest_marks_t052_t054_as_not_applicable():
    """manifest 必须逐档带 `applicable: false` + 理由，且 `relation_by_tier` 不再写自蒸馏。"""
    manifest = generate_matrix.build_manifest(generate_matrix.build_ledger())
    entries = {e["tier_id"]: e for e in manifest["tiers"]}

    for tier_id in ("T052", "T053", "T054"):
        entry = entries[tier_id]
        assert entry["applicable"] is False, tier_id
        assert entry["status"] == "not_applicable", tier_id
        assert entry["opd_teacher_relation"] == "not-applicable", tier_id
        assert entry["config"] is None, tier_id
        assert "省显存" in entry["applicability_reason"], tier_id
        assert entry["feasibility"]["verdict"] == "not_applicable", tier_id
        assert entry["feasibility"]["feasible"] is None, tier_id

    relations = manifest["feasibility_model"]["opd_teacher"]["relation_by_tier"]
    assert relations["T046"] == "independent"
    assert relations["T051"] == "independent"
    for tier_id in ("T052", "T053", "T054"):
        assert relations[tier_id] == "not-applicable", tier_id
    # 全 manifest 不得再出现"自蒸馏"三档的旧结论
    assert "T052" in manifest["feasibility_model"]["opd_teacher"]["not_applicable_tiers"]


def test_opd_tiers_are_unmeasured_never_claimed_feasible():
    """★ OPD 的**可执行**档（`T046`–`T051`，9B）**不得**被写成"可行"（P-25）。

    教师（27B）在训练进程内额外占一份显存，本生成器**没有**教师侧实测数字 ⇒
    判定必须是 ``unmeasured``（"待上机测量"），既不是 ``feasible``（那是没有依据的
    结论），也不是 ``infeasible``（那是把没量过的东西说成跑不了）。

    ★ 2026-09-19 口径变更：`T052`–`T054`（27B OPD）已判 **`not_applicable`**（确实不适用），
    它们**既不**是 `unmeasured` **也不**是 `feasible`——不适用档没有"待测"这回事。
    """
    tiers = {tier["tier_id"]: tier for tier in generate_matrix.build_ledger()}

    opd_ids = [tid for tid, t in tiers.items() if t["algorithm"] == "OPD"]
    assert len(opd_ids) == 9
    applicable_ids = [tid for tid in opd_ids if tid not in {"T052", "T053", "T054"}]
    assert applicable_ids == ["T046", "T047", "T048", "T049", "T050", "T051"]
    for tier_id in applicable_ids:
        verdict, detail = generate_matrix.feasibility_verdict(tiers[tier_id])
        assert verdict == generate_matrix.VERDICT_UNMEASURED, (tier_id, verdict)
        assert "待上机测量" in detail, tier_id
        feasible, _ = generate_matrix.feasibility_for(tiers[tier_id])
        assert feasible is None, tier_id  # 不是 True（声称可行），也不是 False（判定不可行）

    # 阈值：1 卡档必须被点名（T046/T049 分别对应 9B LoRA / 9B 全量的 1 卡）
    for tier_id in ("T046", "T049"):
        assert tiers[tier_id]["cards"] == 1
        assert (
            generate_matrix.feasibility_verdict(tiers[tier_id])[0]
            == generate_matrix.VERDICT_UNMEASURED
        )

    # T052（27B OPD 1 卡）走"不适用"这条完全不同的路
    for tier_id in ("T052", "T053", "T054"):
        assert tiers[tier_id]["cards"] in (1, 2, 4)
        verdict, detail = generate_matrix.feasibility_verdict(tiers[tier_id])
        assert verdict == generate_matrix.VERDICT_NOT_APPLICABLE, (tier_id, verdict)
        assert "待上机测量" not in detail, tier_id


def test_cpt_tiers_are_deterministically_feasible():
    """CPT 没有额外显存消费者 ⇒ 走确定性算式（这里顺带覆盖 LoRA 与全量两种配方）。"""
    tiers = {tier["tier_id"]: tier for tier in generate_matrix.build_ledger()}

    for tier_id in ("T001", "T004", "T007"):  # 9B LoRA / 9B 全量 / 27B LoRA
        verdict, _ = generate_matrix.feasibility_verdict(tiers[tier_id])
        assert verdict == generate_matrix.VERDICT_FEASIBLE, tier_id


# ── P1：全参可行配方 + 生成期显存可行性断言 ────────────────────────────────


def test_ms_swift_full_tiers_have_viable_deepspeed_recipe():
    """6 个 ms-swift 全量档必须带可行配方（1/2 卡 `zero2_offload`；4 卡 **纯 `zero2`/DP4**）。

    ★ 2026-09-23 改口径（路线 a）：4 卡那两档**去掉 `deepspeed_autotp_size: 4`**。
    为什么（详见 `ms_swift_full_recipe` 的 docstring）：AutoTP 的预设路径对 Qwen3.5
    视觉塔错切（上游 DeepSpeed 缺陷，pin 0.18.9）、RL 档另有 ms-swift 缺 `DataLoaderMixin`
    的独立缺陷、且 `tp=4 ⇒ dp=1` 时 ZeRO-2 无 DP 维可分片（显存与纯 DP4 相同）
    ⇒ 去 AutoTP **不花显存**、只去掉两条错误路径。
    """
    tiers = {tier["tier_id"]: tier for tier in generate_matrix.build_ledger()}

    expected = {
        "T019": {"deepspeed": "zero2_offload"},
        "T020": {"deepspeed": "zero2_offload"},
        "T021": {"deepspeed": "zero2"},
        "T037": {"deepspeed": "zero2_offload"},
        "T038": {"deepspeed": "zero2_offload"},
        "T039": {"deepspeed": "zero2"},
    }
    for tier_id, recipe in expected.items():
        msswift = generate_matrix.build_config(tiers[tier_id])["msswift"]
        for key, value in recipe.items():
            assert msswift.get(key) == value, (tier_id, key)
        # ★ 4 卡档**必须不含** AutoTP（这是本次改动的核心不变式；只断言 "0 卡 recipe"
        #   不够——`deepspeed_autotp_size` 若以别的值残留，仍然会开 TP 路径）
        if int(tiers[tier_id]["cards"]) == 4:
            assert "deepspeed_autotp_size" not in msswift, tier_id
            assert not (msswift.get("deepspeed_autotp_size") or 0), tier_id


def test_feasibility_gate_rejects_opd_tier_with_infeasible_recipe(monkeypatch):
    """★ 负向：OPD 的 ``unmeasured`` **不得**成为"任何 OPD 配置都放行"的后门。

    两种超预算形态都必须被拒绝生成：
    ① 确定性算式本身超预算（抽掉 deepspeed）；
    ② 教师**未换出**（27B 权重每卡常驻）—— 这是权重常驻的算术事实。
    """
    original_build_config = generate_matrix.build_config

    # ① 抽掉深速配方 ⇒ OPD 全量档的学生侧确定性算式超预算
    def stripped_deepspeed(tier):
        config = original_build_config(tier)
        if tier["algorithm"] == "OPD" and tier["mode"] == "全量":
            config["msswift"].pop("deepspeed", None)
            config["msswift"].pop("deepspeed_autotp_size", None)
        return config

    monkeypatch.setattr(generate_matrix, "build_config", stripped_deepspeed)
    tier = next(t for t in generate_matrix.build_ledger() if t["tier_id"] == "T049")
    verdict, _ = generate_matrix.feasibility_verdict(tier)
    assert verdict == generate_matrix.VERDICT_INFEASIBLE
    with pytest.raises(AssertionError, match="拒绝生成"):
        generate_matrix.assert_memory_feasible(generate_matrix.build_ledger())

    # ② 教师不换出 ⇒ 教师权重常驻，确定性超预算
    def resident_teacher(tier):
        config = original_build_config(tier)
        if tier["algorithm"] == "OPD":
            config["distill"]["offload_teacher_model"] = False
        return config

    monkeypatch.setattr(generate_matrix, "build_config", resident_teacher)
    verdict, detail = generate_matrix.feasibility_verdict(tier)
    assert verdict == generate_matrix.VERDICT_INFEASIBLE
    assert "未换出" in detail
    with pytest.raises(AssertionError, match="拒绝生成"):
        generate_matrix.assert_memory_feasible(generate_matrix.build_ledger())


def test_all_executable_tiers_are_memory_feasible():
    """可执行档的显存判定必须是三态之一：``feasible`` 或（仅有未测算消费者时）``unmeasured``。

    ★ 四态口径（2026-09-19）：`infeasible` 会被 ``assert_memory_feasible`` 直接拒绝生成，
    因此执行到这里不可能出现；`unmeasured` 只允许由**未测算的额外消费者**引起
    （当前唯一来源 = OPD 教师），理由必须显式写清"待上机测量"。
    ``not_applicable`` 档**不参与**可行性断言（确实不适用：不是"放不下"、也不是"待测量"）。
    """
    tiers = generate_matrix.build_ledger()

    verdicts = generate_matrix.assert_memory_feasible(tiers)

    for tier in tiers:
        status, _ = generate_matrix.classify_expressibility(tier)
        if status in ("blocked", "not_applicable"):
            # 两类都不进 `verdicts`（断言只覆盖"可执行"档）
            assert tier["tier_id"] not in verdicts, tier["tier_id"]
            continue
        verdict, detail = verdicts[tier["tier_id"]]
        assert verdict in (
            generate_matrix.VERDICT_FEASIBLE,
            generate_matrix.VERDICT_UNMEASURED,
        ), (tier["tier_id"], verdict, detail)
        if verdict == generate_matrix.VERDICT_UNMEASURED:
            assert tier["algorithm"] == "OPD", tier["tier_id"]
            assert "待上机测量" in detail, tier["tier_id"]
        else:
            assert "可行" in detail, (tier["tier_id"], detail)


def test_feasibility_gate_rejects_ms_swift_full_without_deepspeed(monkeypatch):
    """负向：把深速配方抽掉后，生成期断言必须**拒绝生成**（这正是 P1 的坑）。"""
    original_build_config = generate_matrix.build_config

    def stripped(tier):
        config = original_build_config(tier)
        if tier["backend"] == "ms-swift" and tier["mode"] == "全量":
            config["msswift"].pop("deepspeed", None)
            config["msswift"].pop("deepspeed_autotp_size", None)
        return config

    monkeypatch.setattr(generate_matrix, "build_config", stripped)

    # 先确认单档估算确实判定为"必然 OOM"
    tier = next(t for t in generate_matrix.build_ledger() if t["tier_id"] == "T019")
    gib, detail = generate_matrix.estimate_config_per_card_gib(tier, stripped(tier))
    assert gib > generate_matrix.CARD_BUDGET_GIB
    assert "未配 deepspeed" in detail

    import pytest

    with pytest.raises(AssertionError) as exc:
        generate_matrix.assert_memory_feasible(generate_matrix.build_ledger())
    assert "拒绝生成" in str(exc.value)


def test_native_full_one_card_offload_note_keeps_the_honest_history():
    """★ 2026-09-23（替换原 `test_native_full_one_card_is_blocked_with_honest_reason`）：
    native 全参 1 卡**已由 offload 配方解除 blocked**；旧 blocked 文案保留为**留痕**，
    且留痕必须继续遵守原口径铁律——**不得**把未实测的假设说成"必然 OOM"。

    为什么保留这条（§18 不留债）：解除 blocked 之后，"当初为什么判 blocked、后来为什么
    改判"必须还能从代码里读到；而那段旧文案里**两条诚实性约束**（不写"必然 OOM"、
    明标假设）不得因为"它不再被消费"就退化。
    """
    tiers = {tier["tier_id"]: tier for tier in generate_matrix.build_ledger()}

    for tier_id in ("T016", "T034"):
        status, _ = generate_matrix.classify_expressibility(tiers[tier_id])
        # ① 判定已反转：不再 blocked
        assert status == "unverified", tier_id

    # ② 留痕仍在，且两条诚实性约束不被破坏
    note = generate_matrix.NATIVE_FULL_1CARD_OFFLOAD_NOTE
    assert "必然 OOM" not in note
    assert "已上机实测证实" in note  # 现在是实测支撑的结论（不再是"判定不可靠"）
    legacy = generate_matrix._legacy_blocked_reasons()
    assert legacy["native_full_1card"] == note
    # ③ 表变空 = 已无 blocked 档可表达（防止"表还在 = 还有 blocked"的误读）
    assert set(legacy) == {"native_full_1card"}
    all_blocked = {
        tier["tier_id"]
        for tier in generate_matrix.build_ledger()
        if generate_matrix.classify_expressibility(tier)[0] == "blocked"
    }
    assert all_blocked == set()


def test_native_full_accounting_uses_k8_and_labels_it_as_assumption():
    """🟡-1：native 全参常驻口径统一为 k=8（bf16 Adam 假设），且不冒充实测。"""
    tiers = {tier["tier_id"]: tier for tier in generate_matrix.build_ledger()}

    # k = 2(bf16 权重) + 2(bf16 梯度) + 4(bf16 m/v) = 8
    assert generate_matrix.NATIVE_OPTIMIZER_BYTES_PER_PARAM == 8.0
    # ms-swift 侧保守沿用 fp32 优化器态（k=12），口径差异必须在 manifest 里写明。
    assert generate_matrix.MSSWIFT_OPTIMIZER_BYTES_PER_PARAM == 12.0
    assert "未经上机实测" in generate_matrix.NATIVE_ACCOUNTING_ASSUMPTION

    tier = tiers["T017"]  # 9B 全参 native 2 卡（PP=2）
    gib, detail = generate_matrix.estimate_config_per_card_gib(
        tier, generate_matrix.build_config(tier)
    )
    assert "k=8 B/param" in detail
    assert gib == pytest.approx(9.0e9 * 8.0 / 2 / (1024.0**3) + 12.0, rel=1e-6)


def test_manifest_has_no_host_identity_and_declares_model_mount():
    """🔴-2 + 🔴-1：manifest 不得含宿主标识；必须声明模型挂载来源与前置断言。

    **防呆目的（本用例存在的意义）**：manifest 一旦被写入真实宿主信息就必须失败。
    因此这里断言的是「含不含宿主标识」这一**结构性不变量**，而不是某个具体的
    主机编号字面量——把内网标识写进断言本身即构成泄露（§15.1），
    等于把防线变成泄露点。

    三条不变量：
    1. ``runtime`` 段不得出现任何宿主标识键（host/hostname/host_name/node/server）；
    2. 全 manifest 中凡键名含 ``host`` 的，只允许 ``*_env_var`` / ``*_default`` /
       ``*_note`` 三种形式；且 ``*_env_var`` 的值必须是 ``GRASPO_*`` 环境变量名、
       ``*_default`` 的值必须以 ``<GRASPO_`` 占位符开头——宿主信息**只能**以这两种
       形式出现，不得是真实值；
    3. 任何字符串值都不得是宿主绝对路径（``/home``、``/mnt``、``/root``、``~/``、
       Windows 盘符等）；容器内路径（``/models``、``/work/elam-v5``）不受限。
    """
    manifest = generate_matrix.build_manifest(generate_matrix.build_ledger())

    # ── 不变量 1：runtime 段不得出现宿主标识键 ──────────────────────────────
    for key in _FORBIDDEN_HOST_KEYS:
        assert key not in manifest["runtime"], (
            f"manifest runtime must not declare host identity key {key!r}"
        )

    # ── 不变量 2 + 3：逐条遍历全部叶子 ──────────────────────────────────────
    for path, key, value in _walk_manifest(manifest):
        location = ".".join(path) or "<manifest root>"
        if "host" in key.lower():
            assert key.endswith(_ALLOWED_HOST_KEY_SUFFIXES), (
                f"{location}: host-related key {key!r} carries a real host value; "
                "host information may only appear as a *_env_var name or a "
                "<GRASPO_...> placeholder"
            )
            if key.endswith("_env_var"):
                assert _HOST_ENV_VAR_RE.match(str(value)), (
                    f"{location}: {key!r} must be a GRASPO_* env-var name, got {value!r}"
                )
            if key.endswith("_default"):
                assert str(value).startswith(_HOST_PLACEHOLDER_PREFIX), (
                    f"{location}: {key!r} must start with {_HOST_PLACEHOLDER_PREFIX!r}, "
                    f"got {value!r}"
                )
        if isinstance(value, str):
            assert not _HOST_ABS_PATH_RE.search(value), (
                f"{location}: manifest must never contain a real host absolute path, got {value!r}"
            )

    models = manifest["models"]
    assert models["container_root"] == "/models"
    assert models["host_root_env_var"] == "GRASPO_MODELS_HOST_ROOT"
    assert models["overrides"]["9B"]["host_dir_env_var"] == "GRASPO_9B_HOST_DIR"
    assert models["overrides"]["9B"]["host_dir_default"].startswith(_HOST_PLACEHOLDER_PREFIX)


def test_rendered_output_has_no_legacy_terms():
    tiers = generate_matrix.build_ledger()
    # 与正式生成**同一条渲染路径**（含 blocked / not_applicable / yaml 三类产物）
    rendered = generate_matrix.render_all_tiers(tiers)

    generate_matrix.assert_no_legacy_terms(rendered)


def test_dry_run_does_not_write_files():
    manifest = generate_matrix.MANIFEST_PATH
    before = manifest.read_text(encoding="utf-8") if manifest.exists() else None

    assert generate_matrix.main(["--dry-run", "--assert-count", "54"]) == 0

    after = manifest.read_text(encoding="utf-8") if manifest.exists() else None
    assert after == before


# ══════════════════════════════════════════════════════════════════════════════
# LoRA target_preset 的**条件选择**（T028 阻断的配置侧修复）
# ══════════════════════════════════════════════════════════════════════════════
#
# 阻断（228 实测 T028，2026-09-19）：native GRASPO 档 `lora.target_preset: language_safe`
# ⇒ 视觉塔零可训参数 ⇒ `RuntimeError: multimodal preflight failed: no trainable visual
# parameters found`。真因是**生成器对所有 LoRA 档硬编码语言预设**，与卡数/显存无关。
# 修法：`lora_target_preset(tier)` 按"模型有视觉塔 且 该档数据含图"条件选择。
# 下面既锁"条件真的分流"，也锁"只改这一格、其它档语义不变"。


def _tier(**overrides: object) -> dict[str, object]:
    tier: dict[str, object] = {
        "model": "9B",
        "algorithm": "GRASPO",
        "backend": "native",
        "mode": "LoRA",
        "cards": 1,
        "tier_id": "T999",
    }
    tier.update(overrides)
    return tier


def test_lora_preset_is_vision_common_when_model_has_vision_and_data_has_media():
    """判据主路：视觉基座 + 含图数据 ⇒ 视觉预设（T028 的失败路径由此消失）。"""
    assert generate_matrix.lora_target_preset(_tier()) == "vision_common"


def test_lora_preset_falls_back_to_language_safe_without_vision():
    """★ mutation：模型无视觉塔 ⇒ 必须回落语言预设（"只训语言塔"仍是合法组合）。"""
    original = generate_matrix.MODELS["9B"].get("vision")
    generate_matrix.MODELS["9B"]["vision"] = False
    try:
        assert generate_matrix.lora_target_preset(_tier()) == "language_safe"
    finally:
        generate_matrix.MODELS["9B"]["vision"] = original


def test_lora_preset_falls_back_to_language_safe_without_media():
    """★ mutation：数据是纯文本 ⇒ 同样回落语言预设（不无条件改成视觉预设）。"""
    original = generate_matrix.DATA_HAS_MEDIA_BY_ALGORITHM["GRASPO"]
    generate_matrix.DATA_HAS_MEDIA_BY_ALGORITHM["GRASPO"] = False
    try:
        assert generate_matrix.lora_target_preset(_tier()) == "language_safe"
    finally:
        generate_matrix.DATA_HAS_MEDIA_BY_ALGORITHM["GRASPO"] = original


def test_lora_preset_is_never_hardcoded_in_build_config():
    """★ 反向：`build_config` 不得写死语言预设（旧缺陷的复发路径）。"""
    for tier in generate_matrix.build_ledger():
        if tier["mode"] != "LoRA":
            continue
        config = generate_matrix.build_config(tier)
        if config.get("lora") is None:
            continue
        expected = generate_matrix.lora_target_preset(tier)
        assert config["lora"]["target_preset"] == expected, tier["tier_id"]


def test_every_lora_tier_gets_the_preset_its_criterion_implies():
    """★ 判据一致性：每个 LoRA 档的产物 == 判据算出的值（无一档被写死）。

    本矩阵两个基座都有视觉塔、四个算法的子集都含图 ⇒ **33 个可跑 LoRA 档**（T052–T054
    为 ``not_applicable``，不产出配置）全部满足"视觉基座 + 含图数据"，因此全部是
    ``vision_common``。等将来某档换成纯文本基座/纯文本子集，这条断言会自动要求它回落
    ``language_safe``——判据与产物始终同源（§1.4）。
    """
    tiers = generate_matrix.build_ledger()
    lora_tiers = [t for t in tiers if t["mode"] == "LoRA"]
    assert len(lora_tiers) == 36, [t["tier_id"] for t in lora_tiers]
    for tier in lora_tiers:
        preset = generate_matrix.lora_target_preset(tier)
        want = (
            "vision_common"
            if generate_matrix.has_vision(str(tier["model"]))
            and generate_matrix.data_has_media(str(tier["algorithm"]))
            else "language_safe"
        )
        assert preset == want, tier["tier_id"]
        config = generate_matrix.build_config(tier)
        assert config["lora"]["target_preset"] == preset, tier["tier_id"]
    # 36 个 LoRA 档里 T052-T054 是 not_applicable（渲染成 .not_applicable.md，不产出
    # 可跑 YAML）⇒ 真正落盘的 LoRA 配置是 33 份。
    rendered_names = {name for name, _ in generate_matrix.render_all_tiers(tiers)}
    lora_yaml = [
        f"{t['tier_id']}.yaml" for t in lora_tiers if f"{t['tier_id']}.yaml" in rendered_names
    ]
    assert len(lora_yaml) == 33, lora_yaml


def test_t028_family_is_exactly_the_previously_blocked_native_graspo_tiers():
    """★ 修复面定位：native GRASPO LoRA 一族就是 T028 实测失败的那一族。

    该族（T028-T030 / T040-T042）现在拿到视觉预设；其余档得到什么由各自判据决定。
    """
    tiers = {t["tier_id"]: t for t in generate_matrix.build_ledger()}
    family = {"T028", "T029", "T030", "T040", "T041", "T042"}
    for tier_id in family:
        tier = tiers[tier_id]
        assert tier["algorithm"] == "GRASPO" and tier["backend"] == "native", tier
        assert tier["mode"] == "LoRA", tier
        assert generate_matrix.lora_target_preset(tier) == "vision_common", tier_id


def test_generated_tier_configs_load_and_carry_a_known_preset():
    """产物可被配置模型加载，且预设是 `core/lora.py` 认识的具名预设（不是随手字符串）。"""
    import yaml  # noqa: PLC0415 仅测试用

    from graspo.core.lora import LORA_TARGET_PRESETS  # noqa: PLC0415
    from graspo.core.schema import GraspoConfig  # noqa: PLC0415

    config_dir = generate_matrix.CONFIG_DIR
    checked = 0
    for tier in generate_matrix.build_ledger():
        if tier["mode"] != "LoRA":
            continue
        path = config_dir / f"{tier['tier_id']}.yaml"
        if not path.is_file():
            continue  # not_applicable / blocked 不产出配置
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict) or "lora" not in raw:
            continue
        preset = raw["lora"]["target_preset"]
        assert preset in LORA_TARGET_PRESETS, (tier["tier_id"], preset)
        GraspoConfig.model_validate(raw)  # 加载即校验（§2.1 契约即防呆）
        checked += 1
    assert checked == 33, checked


def test_regenerated_tier_configs_differ_only_in_target_preset():
    """★ 重建不变量：重生成后每档配置**只有 target_preset 一行**变化。

    用"重新渲染"与磁盘产物逐字节比对：除该字段外一字不改 ⇒ 证明修复面被限制在
    这一格，没有顺手改别的语义（其余内容由 `render_all_tiers` 决定，是唯一真相源）。
    """
    from pathlib import Path as _Path

    config_dir = _Path(generate_matrix.CONFIG_DIR)
    tiers = generate_matrix.build_ledger()
    rendered = dict(generate_matrix.render_all_tiers(tiers))
    for name, text in rendered.items():
        on_disk = (config_dir / name).read_text(encoding="utf-8")
        if on_disk != text:
            # 只允许 target_preset 这一行不同
            differing = [(a, b) for a, b in zip(on_disk.splitlines(), text.splitlines()) if a != b]
            assert all("target_preset" in a or "target_preset" in b for a, b in differing), (
                name,
                differing[:4],
            )
            assert len(differing) <= 1, (name, differing[:4])


# ── RLHF 步数口径（2026-09-20 修复：ms-swift GRASPO 的 `rollout_group_size` 乘数）──────
#
# 缺陷：`expected_optimizer_steps()` 原先是 `子集 ÷ 卡数`，**漏了 RLHF 的数据集重复**。
# ms-swift 的 `RepeatSampler(mini_repeat_count=G)` 会把每条 prompt 重复
# `G = training.rollout_group_size`（默认 8）次再进训练循环 ⇒ 总样本 = 子集 × G。
# 真机实测：T031/T043（1 卡）`global_step/max_steps` 跑到 **160/160**（= 20 × 8），
# 且保留块日志显示真落了 8 份 ckpt（`checkpoint-{20,40,…,160}`）⇒ 声明值低 8 倍，
# `save_steps` 也因此低 8 倍、每个 run 落 8 份（与"每档只落一份"的设计相悖）。
#
# ★ **不做一刀切**：只有走 ms-swift `stage == "rlhf"` 的档才重复数据集。
#   native 的 GRASPO 档不重复（`T028` 的 `events.jsonl`：`samples_seen` 8→16、
#   `samples_total` 20 ⇒ 一个 epoch 约 3 步），SFT/CPT/OPD 也不重复。

_SCHEMA_PATH = Path(__file__).resolve().parents[2] / "src" / "graspo" / "core" / "schema.py"


def test_rollout_group_size_constant_matches_the_schema_default():
    """★单一真相源 parity：生成器里的 G 必须等于 `schema.py` 的 `rollout_group_size` 默认值。

    为什么用**源码文本级** parity 而不是 `import graspo.core.schema`：本模块开头就声明
    "纯 stdlib，不依赖 torch/graspo 包"——那条 import 会拉起 pydantic + ripple→torch，
    在没装 torch 的机器上连本模块都进不来。schema 是 tracked 文件，文本级 parity
    同样能让"默认值被改动"立刻判红（§1.4 单一真相源：不许有两份会漂移的真相）。
    """
    text = _SCHEMA_PATH.read_text(encoding="utf-8")
    match = re.search(r"^    rollout_group_size: int = (\d+)$", text, re.M)
    assert match is not None, "schema 里找不到 `rollout_group_size` 的默认值（字段被改名/改形了？）"
    assert int(match.group(1)) == generate_matrix.ROLLOUT_GROUP_SIZE, (
        "生成器的 ROLLOUT_GROUP_SIZE 与 src/graspo/core/schema.py 的默认值不一致"
        f"（{generate_matrix.ROLLOUT_GROUP_SIZE} vs {match.group(1)}）"
    )


def test_expected_optimizer_steps_scales_only_ms_swift_graspo_tiers():
    """★修复面锁定：只有 **ms-swift 的 GRASPO** 档乘 G，其余档**逐档不变**。

    真机证据（工位 grep 全站的 `[ckpt-retention] KEEP` + `global_step/max_steps`）：
      · ms-swift GRASPO 1 卡：T031/T043 = **160**（= 20 × 8）
      · SFT 1/2/4 卡：T025/T026/T027 = **100/50/25**
      · OPD(GKD) 2/4 卡：T047/T048 = **10/5**
      · native GRASPO 1 卡：T028 全程约 **3** 步（8 samples/step × 20 samples）⇒ 不乘
    """
    tiers = {tier["tier_id"]: tier for tier in generate_matrix.build_ledger()}
    g = generate_matrix.ROLLOUT_GROUP_SIZE
    assert g == 8, "本用例锁的是默认口径；默认值变了要连带更新证据"

    # ms-swift GRASPO：乘 G
    assert generate_matrix.expected_optimizer_steps(tiers["T031"]) == 20 * g // 1 == 160
    assert generate_matrix.expected_optimizer_steps(tiers["T032"]) == 20 * g // 2 == 80
    assert generate_matrix.expected_optimizer_steps(tiers["T033"]) == 20 * g // 4 == 40
    assert generate_matrix.expected_optimizer_steps(tiers["T043"]) == 160
    assert generate_matrix.expected_optimizer_steps(tiers["T045"]) == 40

    # SFT（ms-swift 与 native）：不乘，真机 100/50/25
    for tier_id, expected in (("T013", 100), ("T014", 50), ("T015", 25)):
        assert generate_matrix.expected_optimizer_steps(tiers[tier_id]) == expected, tier_id
    # CPT：不乘
    assert generate_matrix.expected_optimizer_steps(tiers["T001"]) == 100
    assert generate_matrix.expected_optimizer_steps(tiers["T003"]) == 25
    # OPD(GKD)：不乘，真机 10/5
    assert generate_matrix.expected_optimizer_steps(tiers["T047"]) == 10
    assert generate_matrix.expected_optimizer_steps(tiers["T048"]) == 5
    # ★ native 的 GRASPO：**不乘** G（乘了会高 8 倍）——它的步数由**阈值 + force flush**
    #   决定，不是"样本 ÷ 卡数"。但 `expected_optimizer_steps` 是**计划值**（朴素均分，
    #   供 ckpt `save_steps`），对 native **刻意保留偏大值**（2026-09-21 指挥官裁定：
    #   回退 `5c6bde0` 把计划值压到可达上限的那一改）。真实可达上限是另一个量
    #   `expected_optimizer_steps_reachable`（3/2/1），见下面的 `test_native_...`。
    assert generate_matrix.expected_optimizer_steps(tiers["T028"]) == 20
    assert generate_matrix.expected_optimizer_steps(tiers["T029"]) == 10
    assert generate_matrix.expected_optimizer_steps(tiers["T042"]) == 5
    # 且它**绝不**等于"乘 G"的值（防回归：说明 native 档确实没被顺手乘 8）
    assert generate_matrix.expected_optimizer_steps(tiers["T028"]) != 20 * g

    # 全量断言：非「ms-swift GRASPO」的所有档（**含 native GRASPO**）一律等于
    # `子集 ÷ 卡数`（计划口径；没有别的档被顺手改到）。唯一走专用公式的是
    # `expected_optimizer_steps_per_epoch` / `..._reachable` 的 native 分支。
    for tier in generate_matrix.build_ledger():
        if generate_matrix._repeats_dataset_by_rollout_group(tier):
            continue
        plain = generate_matrix.subset_size(str(tier["algorithm"])) // max(1, int(tier["cards"]))
        assert generate_matrix.expected_optimizer_steps(tier) == plain, tier["tier_id"]

    # 判据本身：正好 9 档走 ms-swift RLHF
    rlhf = [
        tier["tier_id"]
        for tier in generate_matrix.build_ledger()
        if generate_matrix._repeats_dataset_by_rollout_group(tier)
    ]
    assert rlhf == ["T031", "T032", "T033", "T037", "T038", "T039", "T043", "T044", "T045"]


def test_save_steps_equals_the_declared_total_for_every_tier():
    """★自洽性：`save_steps` == **计划**步数（`_planned_save_steps`）⇒ 每档只落一份 ckpt。

    这是 ckpt 保留策略的**配置侧**那一半（另一半是容器内保留块）。修前 GRASPO 档的
    `save_steps` 只有真总步数的 1/8 ⇒ 每个 run 落 8 份、靠容器内保留块删掉 7 份。

    ★ **`save_steps` 与计划口径解耦**（2026-09-21 指挥官裁定，AH1 §⑤-5.2）：本用例断言
    `save_steps == _planned_save_steps`（计划），**不**断言它等于任何"可达上限"——
    native GRASPO 那 9 档的可达上限（3/2/1）比计划值（20/10/5）低，且**这正是要的**：
    native 中间段叫 `step_<N>`、保留块只清 `checkpoint-*`，计划值偏大 ⇒ 不触发中间段
    ⇒ 只落终态 `final/`（保留块显式承认它）⇒ 恰好一份。
    """
    for tier in generate_matrix.build_ledger():
        assert generate_matrix.checkpoint_save_steps(tier) == generate_matrix._planned_save_steps(
            tier
        ), tier["tier_id"]
    # 计划口径 = `expected_optimizer_steps` 的字段值（两者同源，见生成器 docstring）
    for tier in generate_matrix.build_ledger():
        assert generate_matrix.expected_optimizer_steps(
            tier
        ) == generate_matrix._planned_save_steps(tier), tier["tier_id"]
    tiers = {tier["tier_id"]: tier for tier in generate_matrix.build_ledger()}
    assert generate_matrix.checkpoint_save_steps(tiers["T043"]) == 160
    # ★ **更正**（2026-09-21 指挥官裁定，回退 `5c6bde0`）：native GRASPO 的 `save_steps`
    # 回到**计划值** 20/10/5，**不是**可达上限 3/2/1。旧断言（3 / 1）建立在"旧值会让
    # ckpt 策略失效"这个**已被证伪**的理由上：保留块**显式承认**非空 `final/`
    # （`generate_matrix.py` 的 `retain_single_checkpoint`；真机 `task-e1-smoke-t010`
    # 的 `KEEP /out/T010/final` + `ckpt_retention.state=ok`）。反过来把计划值压到可达上限
    # 会让 native 每档真落 `step_<N>` + `final/` = **2 份重 ckpt**，与 `keep_per_tier=1` 冲突。
    assert generate_matrix.checkpoint_save_steps(tiers["T028"]) == 20
    assert generate_matrix.checkpoint_save_steps(tiers["T030"]) == 5
    # 且计划值**严格大于**可达上限（这是 native 只落 `final/` 的结构性依据）
    for tier_id in ("T028", "T029", "T030", "T035", "T036", "T040", "T041", "T042"):
        assert generate_matrix.checkpoint_save_steps(
            tiers[tier_id]
        ) > generate_matrix.expected_optimizer_steps_reachable(tiers[tier_id]), tier_id


# ── 卡集合覆盖点（唯一覆盖点：--gpu-override；判据在 tests/e2e/gpu_assignment.py）─────


def test_default_gpu_sets_unchanged_and_still_pass_the_guard():
    """默认行为不变：不设覆盖时逐档卡集合 == `GPU_SETS[cards]`，且仍过范围守卫。"""
    tiers = generate_matrix.build_ledger()
    for tier in tiers:
        assert tier["gpus"] == list(generate_matrix.GPU_SETS[tier["cards"]]), tier["tier_id"]
    by_cards = {tier["cards"]: tier["gpus"] for tier in tiers}
    assert by_cards == {1: [0], 2: [0, 1], 4: [0, 1, 2, 3]}
    assert generate_matrix.GPU_SETS == {1: (0,), 2: (0, 1), 4: (0, 1, 2, 3)}


def test_gpu_override_moves_only_the_named_tiers():
    """覆盖以「档 → 卡集合」映射给出：只动被点名的档，其余沿用默认。"""
    override = generate_matrix.parse_override("T001=4;T002=4,5;T046=5")
    tiers = {tier["tier_id"]: tier for tier in generate_matrix.build_ledger(override)}

    assert tiers["T001"]["gpus"] == [4]
    assert tiers["T002"]["gpus"] == [4, 5]
    assert tiers["T046"]["gpus"] == [5]  # 1 卡 OPD 档也能被挪走
    assert tiers["T003"]["gpus"] == [0, 1, 2, 3]  # 未点名 ⇒ 默认
    assert tiers["T010"]["gpus"] == [0]
    assert [tiers[tid]["cards"] for tid in ("T001", "T002", "T046")] == [1, 2, 1]


def test_gpu_override_keeps_cards_and_every_other_field():
    """★覆盖只改 `gpus`：cards 与其它字段逐字段相同（防"覆盖把卡数也改了"）。"""
    default = {t["tier_id"]: t for t in generate_matrix.build_ledger()}
    overridden = {
        t["tier_id"]: t
        for t in generate_matrix.build_ledger(generate_matrix.parse_override("T001=4;T002=4,5"))
    }
    assert set(default) == set(overridden)
    for tier_id, tier in default.items():
        for key, value in tier.items():
            if key == "gpus":
                continue
            assert overridden[tier_id][key] == value, (tier_id, key)


@pytest.mark.parametrize(
    ("spec", "needle"),
    [
        ("T001=4,5", "卡数不符"),  # 1 卡档给 2 张 ⇒ fail-closed（不"将就"）
        ("T002=4", "卡数不符"),  # 2 卡档给 1 张
        ("T001=6", "生产卡 GPU[6]"),  # GPU6 是生产 vLLM，永不触碰
        ("T001=7", "生产卡 GPU[7]"),
        ("T002=6,7", "生产卡 GPU[6, 7]"),
        ("T001=9", "越界卡"),  # 只允许 GPU0–5
        ("T001=", "卡号列表为空"),
        ("T001=4,4", "含重复卡"),  # 去重
        ("T999=4", "未知档号"),  # 写错档号不得被静默忽略
        ("T001=4;T001=5", "出现多次"),
        ("T001", "缺少 '='"),
        ("*1=4", "不合规"),
    ],
)
def test_gpu_override_is_fail_closed(spec: str, needle: str):
    """负向用例：越界 / 含 6,7 / 卡数不符 / 空 / 重复 / 未知档 / 语法错 ⇒ 一律拒绝。"""
    with pytest.raises(generate_matrix.GpuAssignmentError) as excinfo:
        override = generate_matrix.parse_override(spec)
        generate_matrix.build_ledger(override)
    assert needle in str(excinfo.value), str(excinfo.value)


def test_gpu_override_accepts_partial_maps_and_allowed_edges():
    """合法边界：空覆盖值 = 不覆盖；GPU4/5、跨 NUMA 的 2 卡组合都允许（跨 NUMA 只提示）。"""
    assert generate_matrix.parse_override("") == {}
    assert generate_matrix.parse_override(" T001 = 4 ; ") == {"T001": (4,)}
    tiers = {
        t["tier_id"]: t
        for t in generate_matrix.build_ledger(
            generate_matrix.parse_override("T002=4,5;T003=3,4,5,2")
        )
    }
    assert tiers["T002"]["gpus"] == [4, 5]
    assert tiers["T003"]["gpus"] == [3, 4, 5, 2]  # 4 卡允许 0–5 的任意组合
    assert gpu_assignment.crosses_numa([4, 5]) is False
    assert gpu_assignment.crosses_numa([3, 4]) is True
    assert gpu_assignment.numa_nodes([4, 5]) == (1,)


def test_default_manifest_has_no_gpu_assignment_section():
    """默认清单**不得**新增任何字段（向后兼容：逐字节不变的充分条件之一）。"""
    manifest = generate_matrix.build_manifest(generate_matrix.build_ledger())
    assert "gpu_assignment" not in manifest
    assert list(manifest.keys())[-1] == "tiers"


def test_override_manifest_records_spec_and_fingerprint():
    """有覆盖时清单带 `gpu_assignment` 记录：覆盖值、边界常量、映射指纹。"""
    override = generate_matrix.parse_override("T001=4;T002=4,5")
    tiers = generate_matrix.build_ledger(override)
    manifest = generate_matrix.build_manifest(
        tiers, gpu_assignment=generate_matrix.gpu_assignment_record("T001=4;T002=4,5", tiers)
    )
    record = manifest["gpu_assignment"]
    assert record["source"] == "override"
    assert record["override_spec"] == "T001=4;T002=4,5"
    assert record["allowed_gpus"] == [0, 1, 2, 3, 4, 5]
    assert record["forbidden_production_gpus"] == [6, 7]
    assert record["max_cards_per_job"] == 4
    assert record["default_by_cards"] == {"1": [0], "2": [0, 1], "4": [0, 1, 2, 3]}

    entries = {e["tier_id"]: e for e in manifest["tiers"]}
    assert entries["T001"]["gpus"] == [4]
    assert entries["T002"]["gpus"] == [4, 5]
    expected = generate_matrix.gpu_assignment_fingerprint(
        (e["tier_id"], e["gpus"]) for e in manifest["tiers"]
    )
    assert record["fingerprint_sha256"] == expected


def test_override_manifest_differs_only_in_gpus_and_override_record():
    """★覆盖清单与默认清单逐字段对比：差异只允许出现在 `gpus` 与 `gpu_assignment`。"""
    default = generate_matrix.build_manifest(generate_matrix.build_ledger())
    override = generate_matrix.parse_override("T001=4")
    tiers = generate_matrix.build_ledger(override)
    overridden = generate_matrix.build_manifest(
        tiers, gpu_assignment=generate_matrix.gpu_assignment_record("T001=4", tiers)
    )
    assert set(overridden) - set(default) == {"gpu_assignment"}
    default_entries = {e["tier_id"]: e for e in default["tiers"]}
    for entry in overridden["tiers"]:
        tier_id = entry["tier_id"]
        for key, value in default_entries[tier_id].items():
            if key == "gpus":
                continue
            assert entry[key] == value, (tier_id, key)


def test_override_does_not_touch_configs_or_train_path():
    """覆盖不得改档位 YAML，也不得改任何 `data.train_path`（只改 gpus）。"""
    override = generate_matrix.parse_override("T001=4")
    default_tiers = generate_matrix.build_ledger()
    tiers = generate_matrix.build_ledger(override)
    assert dict(generate_matrix.render_all_tiers(default_tiers)) == dict(
        generate_matrix.render_all_tiers(tiers)
    )
    default_manifest = generate_matrix.build_manifest(default_tiers)
    overridden_manifest = generate_matrix.build_manifest(
        tiers, gpu_assignment=generate_matrix.gpu_assignment_record("T001=4", tiers)
    )
    default_data = {e["tier_id"]: e["data"]["train_path"] for e in default_manifest["tiers"]}
    overridden_data = {e["tier_id"]: e["data"]["train_path"] for e in overridden_manifest["tiers"]}
    assert default_data == overridden_data


def test_default_regeneration_matches_generator_field_universe_and_runner_bytes(
    tmp_path, monkeypatch
):
    """★向后兼容硬证据（2026-10-16 收紧口径，改名以名副其实）。

    无覆盖重生成 ⇒ **runner 逐字节相同**；**manifest 在「生成器字段全集」上逐值相同**，
    且在盘**多出的**字段必须**全部登记**在 `generate_matrix.POST_FILL_*` 注册表里。

    ★ 为什么不再断言 manifest 整份逐字节相同（⛔ 不是"把断言删掉/放宽成只要 config 相同"）：
    在盘 `matrix54_manifest.json` 含生成器**不产出**的后填字段（跑次结论 status/matrix_conclusion/
    availability、实测显存 feasibility_model.graspo_*_cards、OPD 教师测量 …，实测 106 条路径）。
    整体重生成会把它们抹掉 ⇒ 生成器侧已加 **fail-closed** 防线（`generate()` 在写盘前拒绝，
    见 `ManifestFieldLossError`）。本用例负责另一半：**把这 106 条路径逐条钉进注册表**——
    任何未登记的新增字段、或生成器字段值漂移，都会让本用例失败。
    """
    if not generate_matrix.LOCAL_MINI_JSONL.is_file():
        pytest.skip("开发机缺 mini 数据集本地镜像，无法重生成 manifest")
    on_disk_manifest = generate_matrix.MANIFEST_PATH.read_bytes()
    on_disk_runner = generate_matrix.RUNNER_PATH.read_bytes()

    monkeypatch.setattr(generate_matrix, "CONFIG_DIR", tmp_path / "configs")
    monkeypatch.setattr(generate_matrix, "MANIFEST_PATH", tmp_path / "manifest.json")
    monkeypatch.setattr(generate_matrix, "RUNNER_PATH", tmp_path / "runner.sh")
    monkeypatch.setattr(generate_matrix, "write_feasibility_table", lambda tiers: None)

    generate_matrix.generate(54)

    # ① runner：**仍然**逐字节相同（这一条没有放宽）。
    assert (tmp_path / "runner.sh").read_bytes() == on_disk_runner

    regen = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    on_disk = json.loads(on_disk_manifest.decode("utf-8"))

    # ② 在盘**多出**的字段路径 ⇒ 必须全部登记为 post-fill（陌生漂移即失败）。
    regen_paths = regen_guard.field_paths(regen)
    on_disk_paths = regen_guard.field_paths(on_disk)
    extra = sorted(on_disk_paths - regen_paths)
    unregistered = [p for p in extra if not generate_matrix.is_post_fill_extra(p)]
    assert not unregistered, (
        f"在盘清单多出 {len(unregistered)} 条**未登记**的字段路径（既非生成器产出、也不在 "
        f"POST_FILL_EXTRA_PATTERNS 注册表里）——请判定归属后再登记：{unregistered[:20]}"
    )
    # 反向：生成器产出的字段在盘必须都有（在盘不得落后于生成器）。
    assert not sorted(regen_paths - on_disk_paths), (
        f"生成器产出但清单缺失的字段路径：{sorted(regen_paths - on_disk_paths)[:20]}"
    )

    # ③ 生成器字段全集上，在盘与新生成**逐值相同**（仅登记的 post-fill 覆盖字段除外）。
    projected = regen_guard.prune_like(on_disk, regen)
    projected = regen_guard.apply_like_at(
        projected, regen, "", generate_matrix.POST_FILL_OVERRIDE_PATTERNS
    )
    assert projected == regen, "在盘清单在「生成器字段全集」上与重生成结果不一致（值漂移）"
    assert extra, "在盘清单不含任何 post-fill 字段——注册表形同虚设，请复核本用例的前提"


def test_gpu_assignment_cli_fingerprint_is_stable_and_detects_drift():
    """★A4 可核证据：共享 CLI 给出的指纹与生成器一致；指纹不符 ⇒ 非零退出（两跑不可比）。"""
    import subprocess  # noqa: PLC0415  （本文件其余用例不需要，就近导入）

    manifest = generate_matrix.MANIFEST_PATH
    if not manifest.is_file():
        pytest.skip("matrix54_manifest.json 不存在（先跑 tests/e2e/generate_matrix.py）")
    script = Path(generate_matrix.__file__).resolve().parent / "gpu_assignment.py"

    def run(*extra: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, str(script), "check", "--manifest", str(manifest), "--quiet", *extra],
            capture_output=True,
            text=True,
            check=False,
        )

    completed = run()
    assert completed.returncode == 0, completed.stderr
    fingerprint = completed.stdout.strip().split("指纹(sha256)=")[-1]
    expected = generate_matrix.gpu_assignment_fingerprint(
        (tier["tier_id"], tier["gpus"]) for tier in generate_matrix.build_ledger()
    )
    assert fingerprint == expected

    assert run("--expect-fingerprint", expected).returncode == 0
    drifted = run("--expect-fingerprint", "0" * 64)
    assert drifted.returncode == 1
    assert "指纹不一致" in drifted.stderr

    # 对 canonical 清单试算覆盖值：指纹必须等于"真按该覆盖生成"的那份清单的指纹。
    overridden = generate_matrix.build_ledger(generate_matrix.parse_override("T001=4;T002=4,5"))
    override_fingerprint = generate_matrix.gpu_assignment_fingerprint(
        (tier["tier_id"], tier["gpus"]) for tier in overridden
    )
    simulated = run("--spec", "T001=4;T002=4,5")
    assert simulated.returncode == 0, simulated.stderr
    assert simulated.stdout.strip().endswith(f"指纹(sha256)={override_fingerprint}")
    assert "覆盖档数=2" in simulated.stdout


# ═══════════════════════════════════════════════════════════════════════════════
# 容器内 torch 探测证据链（A3 转正路径）——**跨模块契约**的防呆（§1.2 / §2.1）
#
# 生产端：tests/e2e/generate_matrix.py::render_probe_script（落到 $RUN_DIR/torch_probe.py，
#         entry.sh 在收口前调用它）。
# 消费端：scripts/collect_results.py::read_torch_probe（判定逻辑的**唯一**实现）。
# 两端共享一条契约（schema / 字段 / sha256 防陈旧算法）。本区块做三件事：
#   ① 契约字面量不漂移（直接 import 消费端常量比对）；
#   ② 生成物（runner / entry.sh）里探测步骤存在且**顺序正确**；
#   ③ **往返**：真实跑一遍生成出来的探测脚本 ⇒ 产物必须被消费端采信；畸形产物必须被拒。
# ═══════════════════════════════════════════════════════════════════════════════

_COLLECTOR_PATH = Path(__file__).resolve().parents[2] / "scripts" / "collect_results.py"


def _load_collector():
    """加载``collect_results``（消费端）。已加载则复用，避免重复 exec 副作用。"""
    if "collect_results" not in sys.modules:
        spec = importlib.util.spec_from_file_location("collect_results", _COLLECTOR_PATH)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules["collect_results"] = module
        spec.loader.exec_module(module)
    return sys.modules["collect_results"]


#: 假 torch：**只**实现探测脚本用到的最小面（``__version__`` + ``load``）。
#: 为什么造假的而不是要求真 torch：探测脚本的契约是"用 torch.load 反序列化"，而
#: 契约测试要验证的是**落盘格式与顺序**、不是 torch 的行为；宿主（开发机/228）
#: 都**没有** torch，用真 torch 会让这条用例永远 skip（等于没有防呆）。
#: 假 torch 让"反序列化成功 / 失败"两种形态都能精确构造（§2.1 契约即防呆）。
_FAKE_TORCH = """__version__ = "2.11.0+cu130-fake"


class _Tensor:
    def __init__(self, size):
        self.shape = (size,)


class _StateDict(dict):
    pass


def load(path, map_location=None, weights_only=False):
    with open(path, "rb") as handle:
        raw = handle.read()
    if raw.startswith(b"BAD"):
        raise RuntimeError("fake: corrupted checkpoint")
    return {"state_dict": {"w0": _Tensor(4), "w1": _Tensor(4)}, "epoch": 4}
"""


def _stage_rank_files(run_dir: Path, tier: str = "T010") -> list[Path]:
    """按**生产布局**造 native 终态权重：``<run_dir>/<Tier>/final/rank_*.pt``。

    该布局不是猜的：``training.output_dir = /out/<Tier>``（容器内）+ ``/out`` 绑定到
    ``$RUN_DIR`` ⇒ 宿主侧 ``<RUN_DIR>/<Tier>/final/``（嵌套两层档号）。
    这正是 ``collect_results.find_output_dirs`` 的 ``run_dir / tier_id`` 候选，
    因此 ``relpath = "<Tier>/final/<name>"``（见报告 §③ 的实测确认）。
    """
    final = run_dir / tier / "final"
    final.mkdir(parents=True, exist_ok=True)
    files = []
    for rank in (0, 1):
        path = final / f"rank_{rank:05d}_tp_00_pp_00.pt"
        path.write_bytes(f"fake-weight-bytes-rank-{rank}\n".encode())
        files.append(path)
    return files


def _run_generated_probe(tmp_path: Path, run_dir: Path, tier: str = "T010"):
    """**真跑**生成出来的探测脚本（用假 torch），返回 (CompletedProcess, 探测文件路径)。"""
    import os  # noqa: PLC0415
    import subprocess  # noqa: PLC0415

    script = tmp_path / "torch_probe.py"
    script.write_text(generate_matrix.render_probe_script(), encoding="utf-8")
    fake_root = tmp_path / "fake_torch_root"
    (fake_root / "torch").mkdir(parents=True, exist_ok=True)
    (fake_root / "torch" / "__init__.py").write_text(_FAKE_TORCH, encoding="utf-8")
    env = {**os.environ, "PYTHONPATH": str(fake_root)}
    completed = subprocess.run(
        [sys.executable, str(script), "--run-dir", str(run_dir), "--tier", tier],
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )
    return completed, run_dir / generate_matrix.TORCH_PROBE_FILENAME


def test_torch_probe_contract_matches_collector():
    """★跨模块契约字面量必须逐字一致（改一处必须同步另一处）。"""
    collector = _load_collector()
    assert generate_matrix.TORCH_PROBE_FILENAME == collector.TORCH_PROBE_FILENAME
    assert generate_matrix.TORCH_PROBE_SCHEMA == collector.TORCH_PROBE_SCHEMA


def test_generated_probe_script_parses_and_declares_the_contract_literals():
    """生成出来的探测脚本必须是合法 Python，且 schema/文件名取自单一真相源。"""
    import ast  # noqa: PLC0415

    text = generate_matrix.render_probe_script()
    ast.parse(text)  # 语法非法 ⇒ 这里先炸，不会等到上机才发现
    assert f'PROBE_SCHEMA = "{generate_matrix.TORCH_PROBE_SCHEMA}"' in text
    assert f'PROBE_FILENAME = "{generate_matrix.TORCH_PROBE_FILENAME}"' in text
    # 契约的防呆要点必须写在脚本里（给后来改的人看）：三条硬约束与"失败不写文件"。
    assert "写完权重再算 hash" in text
    assert "不写探测文件" in text


def test_runner_contains_probe_step_in_the_required_order():
    """★顺序证明（文本断言范式，与 retention 段哨兵同款）：

    ckpt 保留块（会 rm -rf 中间段 ckpt） ⇒ 探测（算 hash + 落盘） ⇒ exit 收口。
    顺序颠倒的后果：保留块在算完 hash 之后删文件 ⇒ collector 复查时文件已不在 /
    sha256 不符 ⇒ **整体拒采信**（契约 §6.1 硬约束 1 与 2）。
    """
    runner = generate_matrix.render_runner()
    assert generate_matrix.TORCH_PROBE_SCRIPT_CONTAINER_PATH in runner
    marker = 'cat > "$RUN_DIR/torch_probe.py"'
    # ★ 关键：内层解释器**必须**是宿主侧展开（裸 `$CONTAINER_PY`），与 795/796/807 三处
    #   兄弟调用同款。写成转义的 `\$CONTAINER_PY` 会把展开推迟到容器内，而 CONTAINER_PY
    #   只在宿主侧定义（runner 第 100 行）⇒ 容器内空串 ⇒ 该行退化成空命令名 + 参数
    #   （真机症状：`行 229: : 未找到命令`），再被 `|| true` 静默吞掉 ⇒
    #   torch_probe.json 从未产出、A3 永远拿不到强证据。本断言就是这条缺陷的哨兵。
    probe_call = f'"$CONTAINER_PY" {generate_matrix.TORCH_PROBE_SCRIPT_CONTAINER_PATH}'
    assert probe_call in runner
    assert f'"\\$CONTAINER_PY" {generate_matrix.TORCH_PROBE_SCRIPT_CONTAINER_PATH}' not in runner
    # 保留块的删除动作（中间段 ckpt）必须出现在探测之前。
    delete_index = runner.index("[ckpt-retention] DELETE")
    probe_index = runner.index(probe_call)
    exit_index = runner.index('exit "\\$RC"')
    assert delete_index < probe_index < exit_index, (delete_index, probe_index, exit_index)
    # 探测脚本的写出也必须在调用之前（同源产出：不写出 ⇒ 容器内 127，探测形同虚设）。
    assert runner.index(marker) < probe_index
    # ★ 探测的失败不得改变 RC：调用行必须带显式兜底（脚本缺失/解释器缺失也不改训练结论）。
    lines = runner.splitlines()
    probe_lines = [line for line in lines if probe_call in line]
    assert len(probe_lines) == 1, probe_lines
    probe_line_index = lines.index(probe_lines[0])
    # 调用共三行（解释器行 + 参数行 + 兜底行）：前两行以续行反斜杠结尾，兜底行以 `|| echo` 收口。
    assert lines[probe_line_index].rstrip().endswith("\\"), lines[probe_line_index]
    assert lines[probe_line_index + 1].rstrip().endswith("\\"), lines[probe_line_index + 1]
    fallback = lines[probe_line_index + 2]
    assert "|| echo" in fallback, fallback
    # ★ 兜底必须留**显式痕迹**（本次真机缺陷正是被静默吞掉的）：SKIPPED 字样必须可见，
    #   且只能是 echo（返回 0）——不得出现 exit，否则会把探测失败篡改成训练失败（改结果）。
    assert "SKIPPED" in fallback, fallback
    assert "exit" not in fallback.split("||")[1], fallback
    # relpath 由运行期实测布局推出 ⇒ 探测脚本必须打印选中目录（§2.2 显式即防呆）。
    assert "selected_dir" in runner


# ── ★ P17（2026-09-23）：full-param 档的探测**不得空转**（虚假通过）──────────────────
#
# 缺陷形态（228 实测）：native 的 full-param ckpt **同时**写**空的** ``lora_state_dict``
# （只收 ``lora_`` 名字）与真权重 ``full_param_state_dict``；旧探针按"第一个 dict"取值 ⇒
# 命中空字典 ⇒ ``state_dict_kind=lora_state_dict``、``tensors=0`` 却 ``all_ok=True``
# ⇒ **A3 虚假通过**（无法发现 ckpt 损坏/截断）。本区块锁死三件事：
#   ① 键要选对（**空字典不算命中**）且 ``tensors > 0``；
#   ② 缺层 / 整层全零 / 非有限 / 与 ckpt 自述 manifest 不一致 ⇒ 判不通过；
#   ③ **LoRA 档的既有语义不变**（lora-only ckpt 仍通过）。
# 测试手法与既有探测区块同源：**真跑生成出来的脚本**，用一个只实现最小面的假 torch。

_PROBE_FUNCS: dict | None = None


def _probe_functions() -> dict:
    """把生成出来的探测脚本 ``exec`` 进命名空间以拿到其纯函数（不跑 main、不需要 torch）。"""
    global _PROBE_FUNCS
    if _PROBE_FUNCS is None:
        namespace: dict = {"__name__": "graspo_torch_probe_under_test"}
        exec(compile(generate_matrix.render_probe_script(), "<torch_probe.py>", "exec"), namespace)
        _PROBE_FUNCS = namespace
    return _PROBE_FUNCS


#: 假 torch（P17 版）：``load`` 返回由环境变量 ``FAKE_TORCH_PAYLOAD`` 描述的载荷；
#: 张量替身支持审计用到的 ``shape``/``numel()``/``abs().max()`` 与"是否有限"标记。
_FAKE_TORCH_P17 = """import json, os

__version__ = "fake-p17"


class _T:
    def __init__(self, shape, value, finite):
        self.shape = tuple(shape)
        self._value = value
        self._finite = finite

    def numel(self):
        total = 1
        for dim in self.shape:
            total *= dim
        return total

    def abs(self):
        return self

    def max(self):
        return self._value


class _B:
    def __init__(self, ok):
        self._ok = ok

    def all(self):
        return self._ok


def isfinite(tensor):
    return _B(bool(getattr(tensor, "_finite", True)))


def load(path, map_location=None, weights_only=False):
    spec = json.loads(os.environ["FAKE_TORCH_PAYLOAD"])

    def build(node):
        if isinstance(node, dict) and node.get("__tensor__"):
            return _T(node["shape"], node["value"], node["finite"])
        if isinstance(node, dict):
            return {key: build(value) for key, value in node.items()}
        return node

    return build(spec)
"""


def _tensor_spec(shape=(4,), value=1.0, finite=True) -> dict:
    return {"__tensor__": True, "shape": list(shape), "value": value, "finite": finite}


def _run_p17_probe(tmp_path: Path, run_dir: Path, tier: str, payload: dict) -> dict:
    """真跑生成出来的探测脚本（假 torch），返回解析后的探测产物。"""
    import os  # noqa: PLC0415
    import subprocess  # noqa: PLC0415

    script = tmp_path / "torch_probe_p17.py"
    script.write_text(generate_matrix.render_probe_script(), encoding="utf-8")
    fake_root = tmp_path / "fake_torch_p17"
    (fake_root / "torch").mkdir(parents=True, exist_ok=True)
    (fake_root / "torch" / "__init__.py").write_text(_FAKE_TORCH_P17, encoding="utf-8")
    env = {**os.environ, "PYTHONPATH": str(fake_root), "FAKE_TORCH_PAYLOAD": json.dumps(payload)}
    completed = subprocess.run(
        [sys.executable, str(script), "--run-dir", str(run_dir), "--tier", tier],
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )
    assert completed.returncode == 0, completed.stderr
    probe = run_dir / generate_matrix.TORCH_PROBE_FILENAME
    assert probe.exists(), completed.stdout + completed.stderr
    return json.loads(probe.read_text(encoding="utf-8"))


def _stage_full_param_ckpt(tmp_path: Path, tier: str, payload: dict, manifest: dict | None) -> Path:
    run_dir = tmp_path / "run"
    final = run_dir / tier / "final"
    final.mkdir(parents=True, exist_ok=True)
    (final / "rank_00000_tp_00_pp_00.pt").write_bytes(b"fake")
    if manifest is not None:
        (final / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return run_dir


_FULL_MANIFEST = {"tuner_type": "full", "placement": {"local_layer_indices": [0, 1]}, "pp_size": 1}


def test_p17_healthy_full_param_ckpt_selects_the_weight_bearing_key(tmp_path: Path):
    """★ 正例：full-param ckpt 必须**选到** ``full_param_state_dict`` 且 ``tensors > 0``。"""
    payload = {
        "lora_state_dict": {},
        "full_param_state_dict": {
            "layers.0.a.weight": _tensor_spec(),
            "layers.1.a.weight": _tensor_spec(),
        },
    }
    run_dir = _stage_full_param_ckpt(tmp_path, "T017", payload, _FULL_MANIFEST)
    probe = _run_p17_probe(tmp_path, run_dir, "T017", payload)
    entry = probe["checked"][0]
    assert probe["all_ok"] is True, entry
    assert entry["state_dict_kind"] == "full_param_state_dict"
    assert entry["tensors"] == 2
    assert entry["layers_found"] == [0, 1]
    assert entry["total_elements"] == 8


def test_p17_empty_state_dicts_no_longer_pass_vacuously(tmp_path: Path):
    """★ 回归锁（本缺陷本体）：两个键都是空字典 ⇒ **必须判不通过**（旧实现 all_ok=True）。"""
    payload = {"lora_state_dict": {}, "full_param_state_dict": {}}
    run_dir = _stage_full_param_ckpt(tmp_path, "T017", payload, _FULL_MANIFEST)
    probe = _run_p17_probe(tmp_path, run_dir, "T017", payload)
    entry = probe["checked"][0]
    assert probe["all_ok"] is False
    assert entry["tensors"] == 0
    assert any(reason.startswith("no_tensors") for reason in entry["audit_reasons"]), entry


def test_p17_missing_layer_is_caught(tmp_path: Path):
    """★ 判别力：少一整层（截断）⇒ 必须被 ``missing_layers`` 抓住。"""
    payload = {
        "lora_state_dict": {},
        "full_param_state_dict": {"layers.0.a.weight": _tensor_spec()},
    }
    run_dir = _stage_full_param_ckpt(tmp_path, "T017", payload, _FULL_MANIFEST)
    probe = _run_p17_probe(tmp_path, run_dir, "T017", payload)
    entry = probe["checked"][0]
    assert probe["all_ok"] is False
    assert any(reason.startswith("missing_layers") for reason in entry["audit_reasons"]), entry


def test_p17_zeroed_layer_is_caught(tmp_path: Path):
    """★ 判别力：整层权重被清零 ⇒ 必须判不通过（单层层面可确证）。"""
    payload = {
        "lora_state_dict": {},
        "full_param_state_dict": {
            "layers.0.a.weight": _tensor_spec(value=1.0),
            "layers.1.a.weight": _tensor_spec(value=0.0),
        },
    }
    run_dir = _stage_full_param_ckpt(tmp_path, "T017", payload, _FULL_MANIFEST)
    probe = _run_p17_probe(tmp_path, run_dir, "T017", payload)
    entry = probe["checked"][0]
    assert probe["all_ok"] is False
    assert entry["zero_layers"] == [1], entry


def test_p17_nonfinite_tensor_is_caught(tmp_path: Path):
    """★ 判别力：张量含 NaN/Inf ⇒ 必须判不通过。"""
    payload = {
        "lora_state_dict": {},
        "full_param_state_dict": {
            "layers.0.a.weight": _tensor_spec(finite=False),
            "layers.1.a.weight": _tensor_spec(),
        },
    }
    run_dir = _stage_full_param_ckpt(tmp_path, "T017", payload, _FULL_MANIFEST)
    probe = _run_p17_probe(tmp_path, run_dir, "T017", payload)
    entry = probe["checked"][0]
    assert probe["all_ok"] is False
    assert any(reason.startswith("nonfinite") for reason in entry["audit_reasons"]), entry


def test_p17_lora_checkpoint_semantics_are_preserved(tmp_path: Path):
    """★ 只增不改：lora-only ckpt（manifest 声明 lora）仍然**通过**。"""
    payload = {
        "lora_state_dict": {"layers.0.a.lora_A": _tensor_spec((2,))},
        "full_param_state_dict": {},
    }
    run_dir = _stage_full_param_ckpt(tmp_path, "T017", payload, {"tuner_type": "lora"})
    probe = _run_p17_probe(tmp_path, run_dir, "T017", payload)
    entry = probe["checked"][0]
    assert probe["all_ok"] is True, entry
    assert entry["state_dict_kind"] == "lora_state_dict"
    assert entry["tensors"] == 1


def test_p17_tuner_type_mismatch_is_caught(tmp_path: Path):
    """★ 自述不一致：manifest 写 ``tuner_type=full`` 但权重只在 lora 键里 ⇒ 判不通过。"""
    payload = {
        "lora_state_dict": {"layers.0.a.lora_A": _tensor_spec((2,))},
        "full_param_state_dict": {},
    }
    run_dir = _stage_full_param_ckpt(tmp_path, "T017", payload, _FULL_MANIFEST)
    probe = _run_p17_probe(tmp_path, run_dir, "T017", payload)
    entry = probe["checked"][0]
    assert probe["all_ok"] is False
    assert any(reason.startswith("tuner_type_mismatch") for reason in entry["audit_reasons"]), entry


def test_p17_multirank_ckpt_uses_its_own_rank_placement(tmp_path: Path):
    """★ A 修（2026-09-23 回归）：pp>1 时每个 rank 只持**自己那段**层 ⇒ 必须用 payload 自带的
    ``placement`` 校验，**不得**拿同一份 sibling manifest 去套所有 rank（旧行为 ⇒ 假失败）。"""
    payload = {
        "lora_state_dict": {},
        "full_param_state_dict": {
            "layers.0.a.weight": _tensor_spec(),
            "layers.1.a.weight": _tensor_spec(),
        },
        "placement": {"local_layer_indices": [0, 1], "pp_size": 4, "pp_rank": 3},
    }
    # sibling manifest 描述的是**另一个 rank** 的层区间（0..8）⇒ 旧代码会误判 missing_layers
    manifest = {
        "tuner_type": "full",
        "pp_size": 4,
        "world_size": 4,
        "placement": {"local_layer_indices": list(range(9))},
    }
    run_dir = _stage_full_param_ckpt(tmp_path, "T018", payload, manifest)
    probe = _run_p17_probe(tmp_path, run_dir, "T018", payload)
    entry = probe["checked"][0]
    assert probe["all_ok"] is True, entry
    assert entry["layers_expected"] == [0, 1]
    assert entry["layer_check_skipped"] is None


def test_p17_global_placement_vs_local_state_keys(tmp_path: Path):
    """★ A 修 v2（真 fixture 形态）：pp>1 的 placement 用**全局**层号（如 rank1 = 9..17），
    而 state_dict 键用**局部**层号（0..8）⇒ 覆盖检查必须按"应有层数"归一，**不得**假失败。"""
    payload = {
        "lora_state_dict": {},
        "full_param_state_dict": {
            f"layers.{i}.a.weight": _tensor_spec() for i in range(9)
        },
        "placement": {"local_layer_indices": list(range(9, 18)), "pp_size": 4, "pp_rank": 1},
    }
    manifest = {"tuner_type": "full", "pp_size": 4, "world_size": 4,
                "placement": {"local_layer_indices": list(range(9))}}
    run_dir = _stage_full_param_ckpt(tmp_path, "T018", payload, manifest)
    probe = _run_p17_probe(tmp_path, run_dir, "T018", payload)
    entry = probe["checked"][0]
    assert probe["all_ok"] is True, entry
    assert entry["layers_expected"] == list(range(9, 18))   # 原样保留（全局，供人读）
    assert entry["layers_expected_local"] == list(range(9))  # 归一后的局部口径
    assert entry["missing_layers"] == []


def test_p17_missing_layer_within_the_ranks_own_range_still_fails(tmp_path: Path):
    """★ 判别力**不许**因 A 修而变弱：该 rank 自己的层区间内缺层 ⇒ 仍判失败。"""
    payload = {
        "lora_state_dict": {},
        "full_param_state_dict": {"layers.0.a.weight": _tensor_spec()},
        "placement": {"local_layer_indices": [0, 1], "pp_size": 4, "pp_rank": 2},
    }
    manifest = {
        "tuner_type": "full",
        "pp_size": 4,
        "world_size": 4,
        "placement": {"local_layer_indices": list(range(9))},
    }
    run_dir = _stage_full_param_ckpt(tmp_path, "T018", payload, manifest)
    probe = _run_p17_probe(tmp_path, run_dir, "T018", payload)
    entry = probe["checked"][0]
    assert probe["all_ok"] is False
    assert any(reason.startswith("missing_layers") for reason in entry["audit_reasons"]), entry
    assert entry["missing_layers"] == [1]


def test_p17_skip_is_explicit_when_no_per_rank_range(tmp_path: Path):
    """拿不到 per-rank 层区间 ⇒ **显式**跳过（可见标记），而不是静默变弱或假失败。"""
    payload = {
        "lora_state_dict": {},
        "full_param_state_dict": {"layers.0.a.weight": _tensor_spec()},
    }
    manifest = {
        "tuner_type": "full",
        "pp_size": 4,
        "world_size": 4,
        "placement": {"local_layer_indices": list(range(9))},
    }
    run_dir = _stage_full_param_ckpt(tmp_path, "T018", payload, manifest)
    probe = _run_p17_probe(tmp_path, run_dir, "T018", payload)
    entry = probe["checked"][0]
    assert probe["all_ok"] is True, entry
    assert entry["layer_check_skipped"], entry
    assert entry["layers_expected"] == []


def test_p17_single_rank_semantics_unchanged(tmp_path: Path):
    """pp=1（sibling manifest 即本 rank）⇒ 语义与修前一致：用它校验层覆盖。"""
    payload = {
        "lora_state_dict": {},
        "full_param_state_dict": {"layers.0.a.weight": _tensor_spec()},
    }
    run_dir = _stage_full_param_ckpt(tmp_path, "T018", payload, _FULL_MANIFEST)
    probe = _run_p17_probe(tmp_path, run_dir, "T018", payload)
    entry = probe["checked"][0]
    assert probe["all_ok"] is False  # manifest 声明 [0,1]，只给了 0 ⇒ 缺 1
    assert entry["missing_layers"] == [1]
    assert entry["layer_check_skipped"] is None


def test_p17_probe_functions_skip_unknowable_dimensions():
    """边界（§2.3）：**判不了**的维度不得判 fail（无 manifest ⇒ 跳过层覆盖；张量不可读 ⇒ 跳过）。"""
    funcs = _probe_functions()
    state = {"layers.0.a.weight": object(), "layers.9.b.weight": object()}
    audit = funcs["_audit_state_dict"](state, {}, "full_param_state_dict")
    assert audit["audit_ok"] is True, audit
    assert audit["layers_expected"] == []
    assert audit["total_elements"] == 0  # 判不了 ⇒ 0，但不因此 fail


def test_p17_flatten_skips_empty_dicts_and_reports_which_were_empty():
    """边界：空字典不算命中；全空时把"哪些键存在但为空"写进 kind 便于定位。"""
    funcs = _probe_functions()
    flatten = funcs["_flatten_state_dict"]
    state, kind = flatten({"lora_state_dict": {}, "full_param_state_dict": {"a": 1}})
    assert (state, kind) == ({"a": 1}, "full_param_state_dict")
    state, kind = flatten({"lora_state_dict": {}, "full_param_state_dict": {}})
    assert state == {}
    assert kind.startswith("empty_state_dicts:")
    assert "lora_state_dict" in kind and "full_param_state_dict" in kind
    state, kind = flatten({"w": type("T", (), {"shape": (2,)})()})
    assert kind == "top_level_tensors" and state


def _expand_entry_sh(runner_text: str, arena: Path) -> str:
    """把生成的 runner 里那段 entry.sh heredoc **真的展开一遍**，返回 entry.sh 正文。

    为什么必须真展开而不能只做文本断言（§2.2 显式即防呆）：entry.sh 的语义由**两级展开**
    决定——生成器 f-string 一次、runner 的不加引号 heredoc 再一次。只读 run_matrix54.sh
    的源码看不出"容器里最终会执行什么"，本缺陷正藏在这一层（`\\$CONTAINER_PY` 与
    `$CONTAINER_PY` 在源文件里只差一个反斜杠，语义却差一次展开）。这里用 bash 真跑那一段
    heredoc（不碰 docker / GPU / 网络），得到与上机时**逐字相同**的 entry.sh。
    """
    import os  # noqa: PLC0415
    import subprocess  # noqa: PLC0415

    lines = runner_text.splitlines(keepends=True)
    start = next(
        i for i, line in enumerate(lines) if line.startswith('cat > "$RUN_DIR/entry.sh" <<ENTRY')
    )
    end = next(i for i in range(start + 1, len(lines)) if lines[i].rstrip("\n") == "ENTRY")
    script = arena / "emit_entry.sh"
    script.write_text("".join(lines[start : end + 1]), encoding="utf-8")
    # 宿主侧变量的取值：CONTAINER_PY 从 runner 自己的**唯一真相源**那行解析（§1.4），
    # 不在这里另写一个默认值（否则就是第二个解释器路径来源）。
    definition = re.search(
        r'^CONTAINER_PY="\$\{CONTAINER_PYTHON:-([A-Za-z0-9_./-]+)\}"$', runner_text, re.M
    )
    assert definition is not None, "runner 里找不到 CONTAINER_PY 的定义行（单一真相源被改）"
    env = {
        **os.environ,
        "RUN_DIR": str(arena),
        "CONTAINER_PY": definition.group(1),
        "CONFIG": "samples/configs/matrix54/placeholder.yaml",
        "NPROC": "4",
        "PORT": "29500",
        "RUN_USER_DESC": "1000:1000(placeholder)",
        "TRAIN_PATH": "/elam-v5/data/train.jsonl",
    }
    completed = subprocess.run(
        ["bash", str(script)], capture_output=True, text=True, check=False, env=env, timeout=60
    )
    assert completed.returncode == 0, completed.stderr
    return (arena / "entry.sh").read_text(encoding="utf-8")


def test_entry_sh_probe_call_uses_the_host_side_interpreter(tmp_path):
    """★★端到端展开哨兵：entry.sh 里 torch_probe 那行的解释器**非空**且等于宿主侧取值。

    这是"修前必失败、修后必通过"的用例——修前 entry.sh 该行是字面量
    `"$CONTAINER_PY" /out/torch_probe.py`（容器内未定义 ⇒ 空命令名），断言必然失败。
    反向对照：同一 heredoc 里三处兄弟调用必须得到**同一个**解释器字面量（证明语义）。
    """
    runner = generate_matrix.render_runner()
    entry = _expand_entry_sh(runner, tmp_path)
    non_comment = [
        line for line in entry.splitlines() if line.strip() and not line.lstrip().startswith("#")
    ]
    # ① 解释器**非空**（空解释器 ⇒ 空命令名 ⇒ `行 229: : 未找到命令`，本缺陷的症状）。
    assert not any(
        line.startswith(": /out/torch_probe.py") or line.startswith('"" /out/torch_probe.py')
        for line in non_comment
    ), [line for line in non_comment if "/out/torch_probe.py" in line]
    # ② 探测行的解释器部分逐字等于宿主侧 CONTAINER_PY 取值（python，非空）。
    probe_lines = [line for line in non_comment if "/out/torch_probe.py" in line]
    assert len(probe_lines) == 1, probe_lines
    assert probe_lines[0].startswith('"python" /out/torch_probe.py'), probe_lines[0]
    # ③ 与兄弟调用同款：preflight / 显存采样也必须拿到同一个解释器字面量（同语义对照）。
    assert '"python" /out/preflight_media.py ' in entry
    # ④ 容器内不得残留任何对宿主侧 CONTAINER_PY 的引用（转义写法必然留下它）。
    assert "CONTAINER_PY" not in "\n".join(non_comment)
    # ⑤ 兜底行仍在，且带显式痕迹（探测失败不被静默吞掉）。
    joined = "\n".join(non_comment)
    assert "[torch-probe] SKIPPED" in joined


def test_probe_round_trip_is_accepted_by_the_collector(tmp_path):
    """★★往返（防合成谬误）：真跑生成出来的探测脚本 ⇒ 产物必须被消费端**采信**为强证据。"""
    run_dir = tmp_path / "runs" / "T010"
    _stage_rank_files(run_dir)
    completed, probe_path = _run_generated_probe(tmp_path, run_dir)
    assert completed.returncode == 0, completed.stderr
    assert probe_path.is_file(), completed.stdout + completed.stderr
    payload = json.loads(probe_path.read_text(encoding="utf-8"))
    assert payload["schema"] == generate_matrix.TORCH_PROBE_SCHEMA
    # relpath 必须是**相对 run_dir 的 POSIX 路径**，且指向真实存在的权重文件。
    assert payload["checked"], payload
    for entry in payload["checked"]:
        assert not entry["relpath"].startswith("/"), entry
        assert not entry["relpath"].startswith("./"), entry
        assert (run_dir / entry["relpath"]).is_file(), entry
        assert entry["sha256"] == entry["sha256"].lower()
    assert payload["all_ok"] is True
    assert payload["selected_dir"] == "T010/final", payload["selected_dir"]

    collector = _load_collector()
    evidence = collector.read_torch_probe(run_dir)
    assert evidence is not None, "collector 完全没有读到探测文件"
    assert evidence.value is True, evidence
    assert evidence.source == collector.RELOAD_SOURCE_CONTAINER_PROBE, evidence


def test_probe_all_ok_false_is_a_strong_failure_not_a_gap(tmp_path):
    """★合成谬误的另一面：``all_ok=false`` 必须被采信为**强证据的实质失败**（False），
    而不是被降级成"取证缺口"（None）——那会把"ckpt 载不进"记成不可判定。"""
    run_dir = tmp_path / "runs" / "T010"
    files = _stage_rank_files(run_dir)
    files[0].write_bytes(b"BAD-corrupted-payload")  # 假 torch 见 BAD 前缀即抛异常
    completed, probe_path = _run_generated_probe(tmp_path, run_dir)
    assert completed.returncode == 0, completed.stderr
    payload = json.loads(probe_path.read_text(encoding="utf-8"))
    assert payload["all_ok"] is False
    collector = _load_collector()
    evidence = collector.read_torch_probe(run_dir)
    assert evidence is not None and evidence.value is False, evidence
    assert evidence.source == collector.RELOAD_SOURCE_CONTAINER_PROBE, evidence


@pytest.mark.parametrize(
    ("mutation", "needle"),
    [
        (lambda payload: payload.update(schema="graspo.torch_probe.v2"), "schema"),
        (lambda payload: payload.update(all_ok="true"), "all_ok"),
        (lambda payload: payload.update(checked=[]), "checked"),
        (lambda payload: payload["checked"][0].update(sha256="0" * 64), "sha256"),
        (lambda payload: payload["checked"][0].update(relpath="T010/final/nope.pt"), "不存在"),
    ],
)
def test_malformed_probe_is_rejected_with_a_visible_reason(tmp_path, mutation, needle):
    """★畸形产物必须被**拒采信**且原因可见（绝不静默放行，也不静默丢弃原因）。"""
    run_dir = tmp_path / "runs" / "T010"
    _stage_rank_files(run_dir)
    completed, probe_path = _run_generated_probe(tmp_path, run_dir)
    assert completed.returncode == 0, completed.stderr
    payload = json.loads(probe_path.read_text(encoding="utf-8"))
    mutation(payload)
    probe_path.write_text(json.dumps(payload), encoding="utf-8")

    evidence = _load_collector().read_torch_probe(run_dir)
    assert evidence is not None, "拒采信必须是「文件在但不可采信」，而不是当作没有文件"
    assert evidence.value is None, evidence
    assert needle in evidence.detail, evidence.detail


def test_probe_absent_returns_none_so_the_weak_path_still_runs(tmp_path):
    """★退路不变：**没有**探测文件 ⇒ 返回 None（正常退路），由宿主 torch/弱证据路径继续。"""
    run_dir = tmp_path / "runs" / "T010"
    run_dir.mkdir(parents=True)
    assert _load_collector().read_torch_probe(run_dir) is None


def test_a2_fallback_failure_is_recorded_not_folded_into_no_change(tmp_path, monkeypatch):
    """★★A2 兜底语义（指挥官裁定 (b)：**记账后继续**，2026-09-21）。

    改前：``except (RuntimeError, OSError, AttributeError): continue`` ⇒ 载不进 ckpt 被
    **静默吞掉**，最终 detail 只有一句笼统的"都没有权重变化证据"（无法区分文件坏了 /
    torch 版本不兼容 / 文件不存在）。

    改后：失败原因逐条累积进 ``fallback_notes`` 并并入最终 detail；**判定不变**——
    仍返回 ``value is None``（取证缺口），**绝不**返回 ``False``。

    为什么**不得**返回 ``False``：``False`` 的语义是"权重未变化"= "训练没生效"，
    这是最坏的假失败（环境/兼容性问题被记成训练失败）。
    """
    collector = _load_collector()
    run_dir = tmp_path / "runs" / "T010"
    _stage_rank_files(run_dir)
    (run_dir / "T010" / "config.yaml").write_text("training: {}\n", encoding="utf-8")
    output_dirs = collector.find_output_dirs(run_dir, "T010")
    assert output_dirs, "测试前置：产物根必须能被定位"

    class _ExplodingTorch:
        @staticmethod
        def load(*_args, **_kwargs):  # noqa: ANN002, ANN003
            raise OSError("fake: unreadable archive")

    monkeypatch.setattr(collector, "_require_torch", lambda: _ExplodingTorch)

    evidence = collector.extract_weight_changed(output_dirs, "lora", None)
    assert evidence.value is None, (
        f"兜底失败**不得**被记成「权重未变化」(False)——那是最坏的假失败；得到 {evidence!r}"
    )
    assert evidence.source is None, evidence
    # 原因必须可见（记账不吞掉）：异常类型 + 文件名 + 走的是哪条兜底路径。
    assert "兜底路径" in evidence.detail, evidence.detail
    assert "rank_00000_tp_00_pp_00.pt" in evidence.detail, evidence.detail
    assert "OSError" in evidence.detail, evidence.detail


def test_a2_fallback_skip_reason_is_recorded_when_host_has_no_torch(tmp_path, monkeypatch):
    """兜底的另一种"不可得"也要记账：宿主无 torch ⇒ 明说，而不是一句"都没有证据"。"""
    collector = _load_collector()
    run_dir = tmp_path / "runs" / "T010"
    _stage_rank_files(run_dir)
    (run_dir / "T010" / "config.yaml").write_text("training: {}\n", encoding="utf-8")
    output_dirs = collector.find_output_dirs(run_dir, "T010")
    monkeypatch.setattr(collector, "_require_torch", lambda: None)

    evidence = collector.extract_weight_changed(output_dirs, "lora", None)
    assert evidence.value is None, evidence
    assert "宿主无 torch" in evidence.detail, evidence.detail


def test_a2_fallback_still_fails_closed_when_no_evidence_at_all(tmp_path, monkeypatch):
    """★不得削弱 fail-closed：真取不到**任何**证据仍须判否（``value is None`` = 缺口）。"""
    collector = _load_collector()
    run_dir = tmp_path / "runs" / "T010"
    (run_dir / "T010" / "final").mkdir(parents=True)  # 有目录、无 rank_*.pt、无指标 JSON
    (run_dir / "T010" / "config.yaml").write_text("training: {}\n", encoding="utf-8")
    output_dirs = collector.find_output_dirs(run_dir, "T010")
    monkeypatch.setattr(collector, "_require_torch", lambda: None)

    evidence = collector.extract_weight_changed(output_dirs, "lora", None)
    assert evidence.value is None, evidence
    # 正式判据不可得这一条本身也必须入台账（改前是静默 continue 到兜底）。
    assert "lora_norm_delta" in evidence.detail, evidence.detail


#: 假 nvidia-smi：宿主侧锁卡守卫会走**真**解析器，替身必须交回它认识的读数格式。
#: 只回答 0 号卡（本用例只跑 T010 一卡档）。
_FAKE_NVIDIA_SMI = """#!/bin/bash
args="$*"
case "$args" in
  "-L") echo "GPU 0: NVIDIA A800-SXM4-80GB (UUID: GPU-fake000)"; exit 0;;
esac
case "$args" in
  *--query-gpu=index,memory.used,memory.total,utilization.gpu*)
    echo "0, 4, 81920, 0"; exit 0;;
  *--query-gpu=index,uuid*) echo "0, GPU-fake000, 4, 81916, 81920, 0, 40, 100.0"; exit 0;;
esac
exit 0
"""


def _render_probe_call_lines() -> tuple[str, str]:
    """从生成出来的 entry.sh 正文里抽出探测调用的**调用行 + 续行**（原样，不改字符）。

    这是"生成物真的含这一步"的直接证据；调用内容另由
    :func:`_execute_probe_call` 真跑（证明它真的能被调用、真的落盘）。
    """
    runner = generate_matrix.RUNNER_PATH
    if not runner.is_file():
        pytest.skip("run_matrix54.sh 不存在（先跑 tests/e2e/generate_matrix.py）")
    lines = runner.read_text(encoding="utf-8").splitlines()
    marker = generate_matrix.TORCH_PROBE_SCRIPT_CONTAINER_PATH
    index = next(
        position for position, line in enumerate(lines) if marker in line and "CONTAINER_PY" in line
    )
    call, continuation = lines[index], lines[index + 1]
    assert continuation.strip().startswith("--run-dir"), continuation
    return call, continuation


def _execute_probe_call(tmp_path: Path, run_dir: Path, tier: str = "T010"):
    """真跑"探测那一步"（**语义等价于 entry.sh 里那两行**，只是把容器内 /out 换成宿主 run_dir）。

    为什么不整条 entry.sh：`preflight_media.py` / `graspo record-gpu-memory` / `torchrun`
    都需要真 torch + 真模型（开发机没有）⇒ 整条链路的用例在开发机上只能 skip（等于没有
    防呆）。这里验证的是**本包新增的那一步**：调用形态、落盘、以及"失败不改 RC"。
    """
    import os  # noqa: PLC0415
    import subprocess  # noqa: PLC0415

    _call, _continuation = _render_probe_call_lines()  # 断言生成物里确实有这一步
    probe_py = run_dir / "torch_probe.py"
    probe_py.write_text(generate_matrix.render_probe_script(), encoding="utf-8")
    fake_root = tmp_path / "fake_torch_root"
    (fake_root / "torch").mkdir(parents=True, exist_ok=True)
    (fake_root / "torch" / "__init__.py").write_text(_FAKE_TORCH, encoding="utf-8")

    container_bin = tmp_path / "container_bin"
    container_bin.mkdir(exist_ok=True)
    (container_bin / "python").write_text(
        f'#!/bin/bash\nexec "{sys.executable}" "$@"\n', encoding="utf-8"
    )
    (container_bin / "python").chmod(0o755)

    # 等价于 entry.sh 的两行：调用探测 + 单点收口（`exit "$RC"`，探测不参与 RC）。
    script = tmp_path / "probe_step.sh"
    # entry.sh 的对应赋值是 CONTAINER_PY="${CONTAINER_PYTHON:-python}"（runner 里定义）。
    script.write_text(
        'set -o pipefail\nCONTAINER_PY="${CONTAINER_PYTHON:-python}"\n'
        f'"$CONTAINER_PY" {probe_py} \\\n'
        f'    --run-dir {run_dir} --tier "$TIER" || true\n'
        'RC=0\nexit "$RC"\n',
        encoding="utf-8",
    )
    env = {
        **os.environ,
        "TIER": tier,
        "PATH": f"{container_bin}{os.pathsep}{os.environ.get('PATH', '')}",
        "CONTAINER_PYTHON": str(container_bin / "python"),
        "PYTHONPATH": str(fake_root),
    }
    completed = subprocess.run(
        ["bash", str(script)], capture_output=True, text=True, check=False, env=env
    )
    return completed, run_dir


def test_entry_sh_really_runs_the_probe(tmp_path):
    """★★端到端：生成出来的 entry.sh **真的**会调用探测脚本，产物落进 run_dir，
    且被 collector 采信为强证据（生产端 → 消费端贯通）。

    比文本断言强在哪：文本断言只证明"字符串在文件里"；本用例真跑那一步，
    证明"真的被调用、真的按生产布局落盘、真的被消费端采信"。
    """
    run_dir = tmp_path / "runs" / "T010"
    _stage_rank_files(run_dir, "T010")
    completed, run_dir = _execute_probe_call(tmp_path, run_dir)
    probe_path = run_dir / generate_matrix.TORCH_PROBE_FILENAME
    assert probe_path.is_file(), (
        f"探测文件没落盘（探测没被调用 / 调用失败）：{completed.stdout[-2000:]}"
        f"{completed.stderr[-2000:]}"
    )
    recorded = json.loads(probe_path.read_text(encoding="utf-8"))
    assert recorded["schema"] == generate_matrix.TORCH_PROBE_SCHEMA
    assert recorded["all_ok"] is True, recorded
    assert recorded["selected_dir"] == "T010/final", recorded
    collector = _load_collector()
    evidence = collector.read_torch_probe(run_dir)
    assert evidence is not None and evidence.value is True, evidence
    assert evidence.source == collector.RELOAD_SOURCE_CONTAINER_PROBE, evidence


def test_entry_sh_probe_failure_does_not_change_training_rc(tmp_path):
    """★失败不得影响训练结论：权重不存在（探测必然失败）时，收口 rc 仍是 0。

    探测是**收尾步骤**，它的失败只允许产生"没有可采信证据"这一后果，**不允许**改变训练
    rc 的语义（既有优先级：训练失败 137/124 > 保留失败 4 > 成功 0）。

    ★ P21 起语义变更（本包核心）：探测"没验成"必须**显式落盘为不可判定载荷**
    （``all_ok=false`` + ``undecidable=true`` + ``layout_status=skipped_layout_unknown``
    + 具名 ``reasons``），而不是旧行为"什么都不写"——旧行为让消费端退回读 safetensors
    头部的**弱证据**，把"没验"读成"验过"（28 个 ms-swift 档 A3 假绿）。
    所以旧断言"探测失败时不得写出探测文件"**正是那个缺陷的编码**，本包随之改成
    "必须写出显式不可判定、且绝不标成通过"。
    """
    import shutil  # noqa: PLC0415

    run_dir = tmp_path / "runs" / "T010"
    _stage_rank_files(run_dir, "T010")
    completed, run_dir = _execute_probe_call(tmp_path, run_dir)
    assert completed.returncode == 0, completed.stderr[-2000:]
    assert (run_dir / generate_matrix.TORCH_PROBE_FILENAME).is_file()

    # 删掉训练产物（模拟"探测找不到权重"）后重跑同一步：rc 仍必须是 0，且落不可判定载荷。
    shutil.rmtree(run_dir / "T010")
    (run_dir / generate_matrix.TORCH_PROBE_FILENAME).unlink()
    import os  # noqa: PLC0415
    import subprocess  # noqa: PLC0415

    container_bin = tmp_path / "container_bin"
    env = {
        **os.environ,
        "TIER": "T010",
        "PATH": f"{container_bin}{os.pathsep}{os.environ.get('PATH', '')}",
        "CONTAINER_PYTHON": str(container_bin / "python"),
        "PYTHONPATH": str(tmp_path / "fake_torch_root"),
    }
    completed_again = subprocess.run(
        ["bash", str(tmp_path / "probe_step.sh")],
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )
    assert completed_again.returncode == 0, (
        f"探测失败改变了 rc（应为 0）：{completed_again.returncode}"
        f"stdout={completed_again.stdout[-1000:]} stderr={completed_again.stderr[-1000:]}"
    )
    assert "SKIP" in completed_again.stdout, completed_again.stdout
    probe_path = run_dir / generate_matrix.TORCH_PROBE_FILENAME
    assert probe_path.is_file(), (
        "P21：探测『没验成』必须**显式落盘**为不可判定载荷，"
        "而不是旧行为『什么都不写』（那正是被读成通过的灰区）"
    )
    recorded = json.loads(probe_path.read_text(encoding="utf-8"))
    assert recorded["all_ok"] is False, recorded
    assert recorded["undecidable"] is True, recorded
    assert recorded["layout_status"] == "skipped_layout_unknown", recorded
    assert recorded["reasons"], recorded
    assert recorded["reasons"][0].startswith("skipped_layout_unknown"), recorded
    assert recorded["checked"] == [], recorded


# ── 步数口径的**两个量**（AF1 §⑥ 方案 S1；2026-09-21）────────────────────────
#
# **为什么有这一节**：`expected_optimizer_steps`（"计划步数"，供 ckpt `save_steps`）
# 与服务门槛/披露的"**可产出步数上限**"是**两个量**。旧实现只有一个量、且对
# `GRASPO × native` 用的是"样本 ÷ 卡数"的**朴素均分**，遗漏了 replay 阈值与
# 每 epoch 一次的 force flush ⇒ 9 档声明偏高（AF1 诊断）。
# 本节把"**45 档新旧两量逐档一致**"变成**机器判据**——这既是兼容性证据，
# 也是公式正确性的交叉验证。

_NATIVE_GRASPO_NINE = (
    "T028",
    "T029",
    "T030",
    "T034",
    "T035",
    "T036",
    "T040",
    "T041",
    "T042",
)


def test_reachable_equals_plan_for_every_tier_except_native_graspo():
    """★ **45 档逐档相等；native GRASPO 那 9 档"计划 ≥ 可达"且 `save_steps == 计划`**。

    这条断言的意义有三层：
      ① **兼容性**：45 档的既有声明一字未动（ckpt 保留策略、其它判据都不受影响）；
      ② **公式交叉验证**：两条独立实现的路径在 45 档上给出同一答案 ⇒ 公式没有把
         "别族"顺手改错（AF1 已核 + 本包机核）；
      ③ **裁定锁定**（2026-09-21）：native GRASPO 的计划值（供 `save_steps`）**必须 ≥**
         可达上限（门槛判定用），且 `save_steps` **必须等于计划值**——**不许**把计划值
         压到可达上限（那会让 native 每档多落一份 `step_<N>` 重 ckpt，见
         `test_save_steps_equals_the_declared_total_for_every_tier`）。
    """
    tiers = generate_matrix.build_ledger()
    assert len(tiers) == 54
    nine = {t["tier_id"] for t in tiers if generate_matrix._is_native_graspo(t)}
    assert nine == set(_NATIVE_GRASPO_NINE), nine  # 判据本身：正好 9 档

    checked = 0
    for tier in tiers:
        if tier["tier_id"] in nine:
            continue
        assert generate_matrix.expected_optimizer_steps_reachable(
            tier
        ) == generate_matrix.expected_optimizer_steps(tier), tier["tier_id"]
        checked += 1
    assert checked == 45

    # ★ ③ 那 9 档：计划 ≥ 可达（可严格大于），且 `save_steps` 走计划、不走可达。
    for tier in tiers:
        if tier["tier_id"] not in nine:
            continue
        plan = generate_matrix.expected_optimizer_steps(tier)
        reachable = generate_matrix.expected_optimizer_steps_reachable(tier)
        assert plan >= reachable, tier["tier_id"]
        assert generate_matrix.checkpoint_save_steps(tier) == plan, tier["tier_id"]
        # 9 档的赋值口袋：计划 = 朴素均分；可达 = 阈值 + force flush
        assert plan == generate_matrix._planned_save_steps(tier), tier["tier_id"]
        assert reachable == generate_matrix.expected_optimizer_steps_per_epoch(tier), tier[
            "tier_id"
        ]


def test_native_graspo_steps_follow_threshold_and_force_flush():
    """native GRASPO 的每 epoch 上限 = `floor(floor(subset/cards)/Q) + 1`（AF1 推导）。

    真机锚点：`T030`（4 卡）自然跑完 `exit_code=0`、`optimizer_steps=1`、
    `epoch_summary` = `samples_seen=5 / completions=40 / optimized_steps=0`
    （40 < 阈值 `queue×group=64` ⇒ 阈值 0 次触发，唯一一步来自 epoch 末 force flush）。
    """
    tiers = {tier["tier_id"]: tier for tier in generate_matrix.build_ledger()}
    q = generate_matrix.NATIVE_ROLLOUT_QUEUE_BATCH_SIZE
    g = generate_matrix.ROLLOUT_GROUP_SIZE
    assert (q, g) == (8, 8), "本用例锁的是默认口径；默认值变了要连带更新证据"

    # 1 卡：20 条/rank ⇒ floor(20/8) = 2 次阈值 + 1 flush = 3
    for tier_id in ("T028", "T034", "T040"):
        tier = tiers[tier_id]
        assert generate_matrix.native_graspo_steps_per_epoch(tier) == 3, tier_id
        assert generate_matrix.expected_optimizer_steps_reachable(tier) == 3, tier_id
    # 2 卡：10 条/rank ⇒ floor(10/8) = 1 + 1 = 2
    for tier_id in ("T029", "T035", "T041"):
        tier = tiers[tier_id]
        assert generate_matrix.native_graspo_steps_per_epoch(tier) == 2, tier_id
    # 4 卡：5 条/rank ⇒ 5×8 = 40 < 64 ⇒ 阈值 0 次 + 1 flush = **1**（结构上最多 1 步）
    for tier_id in ("T030", "T036", "T042"):
        tier = tiers[tier_id]
        assert generate_matrix.native_graspo_steps_per_epoch(tier) == 1, tier_id
        assert generate_matrix.expected_optimizer_steps_reachable(tier) == 1, tier_id
        # ★ 真机锚点：T030 实测 1 步 ⇒ 公式与真机**精确命中**
        assert 5 * g < q * g, "40 条 experience < 阈值 64 ⇒ 阈值永不触发（本档的关键依据）"


def test_step_gate_applicability_matches_the_formula_for_every_tier():
    """生成期标记 == 公式重算（防手改 / 防漂移），且 9 档与 45 档分开。"""
    threshold = generate_matrix.FORMAL_GATE_MIN_OPTIMIZER_STEPS
    assert threshold == 5, "★ 本包不动门槛值；改了这条要连带更新 AF1 工位报告"
    ledger = generate_matrix.build_ledger()
    manifest = generate_matrix.build_manifest(ledger)
    by_id = {t["tier_id"]: t for t in ledger}
    na_ids = set()
    for entry in manifest["tiers"]:
        tier = by_id[entry["tier_id"]]
        marker, _ = generate_matrix.step_gate_applicability(tier, min_optimizer_steps=threshold)
        assert entry["acceptance"]["formal_gate"]["step_gate_applicability"] == marker
        if marker == generate_matrix.STEP_GATE_NOT_APPLICABLE_STRUCTURAL:
            na_ids.add(entry["tier_id"])
    assert na_ids == set(_NATIVE_GRASPO_NINE)
    # 审计块与逐档标记同源（不是第二份真相）
    audit_ids = {t["tier_id"] for t in manifest["step_gate_audit"]["structural_limited_tiers"]}
    assert audit_ids == na_ids


def test_step_gate_audit_separates_impossible_from_needs_more_epochs():
    """★ 两类必须分开标（AF1 要点 2）：4 卡恒不可达 vs 多跑 epoch 可达。"""
    manifest = generate_matrix.build_manifest(generate_matrix.build_ledger())
    kinds = {
        t["tier_id"]: t["kind"] for t in manifest["step_gate_audit"]["structural_limited_tiers"]
    }
    assert kinds == {
        "T028": "needs_more_epochs",
        "T029": "needs_more_epochs",
        "T030": "structural_impossible",
        "T034": "needs_more_epochs",
        "T035": "needs_more_epochs",
        "T036": "structural_impossible",
        "T040": "needs_more_epochs",
        "T041": "needs_more_epochs",
        "T042": "structural_impossible",
    }


def test_step_gate_audit_fails_closed_on_a_hand_edited_marker():
    """★ 生成期 fail-closed：把某档的标记手改成 `applicable` ⇒ 审计**当场拒绝**。

    这条是"让错误不可能发生"（§2.3）：结构上不可达的档**无法静默进正式台账**。
    """
    import copy  # noqa: PLC0415  （仅测试用）

    manifest = generate_matrix.build_manifest(generate_matrix.build_ledger())
    tampered = copy.deepcopy(manifest["tiers"])
    target = next(e for e in tampered if e["tier_id"] == "T030")
    target["acceptance"]["formal_gate"]["step_gate_applicability"] = "applicable"
    with pytest.raises(ValueError, match="T030"):
        generate_matrix._step_gate_audit(tampered)


def test_min_train_samples_and_min_epochs_are_declared_orphans():
    """★ 登记口径债（AF1 §⑧-5 / 本包 §⑧）：`min_train_samples` / `min_epochs` **无读取方**。

    它们进 manifest，但全仓没有任何判据读它们 ⇒ 读者会把它们当成"已生效的门槛"。
    本用例用 **ast** 机核"**没有 Load 语义的读取**"，而不是 grep 文本——
    因为 `result_judge.ORPHAN_FORMAL_GATE_FIELDS` 这个**登记常量**本身就含这两个名字
    （那是"声明它们是 orphan"，不是"使用它们"，两者必须能分开，§2.2）。
    """
    import ast  # noqa: PLC0415

    manifest = generate_matrix.build_manifest(generate_matrix.build_ledger())
    gate = manifest["tiers"][0]["acceptance"]["formal_gate"]
    # 两个键仍在清单里（本包**不静默删**口径债；删了才是掩盖）
    assert "min_train_samples" in gate and "min_epochs" in gate

    repo = generate_matrix.PROJECT_ROOT
    targets = (
        ("src/graspo/core/result_judge.py", "判定器"),
        ("src/graspo/core/result_judge.py", "判定器"),
    )
    orphan_names = ("min_epochs", "min_train_samples")
    for rel in ("src/graspo/core/result_judge.py",):
        tree = ast.parse((repo / rel).read_text(encoding="utf-8"))
        loaded: list[tuple[int, str]] = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.Constant) or not isinstance(node.value, str):
                continue
            if node.value not in orphan_names:
                continue
            # `ctx` 只有 Name/Attribute/Subscript 等才有；常量出现在元组字面量里时
            # 其父节点是 Tuple（无 ctx）⇒ 那种"登记声明"不算读取。
            if isinstance(getattr(node, "ctx", None), ast.Load):
                loaded.append((node.lineno, node.value))
        assert not loaded, f"{rel} 读取了 orphan 门槛字段：{loaded}"

    # 采集层同样不得读它们（否则就是"静默生效的第二处门槛"）
    for rel in ("scripts/collect_results.py",):
        src = (repo / rel).read_text(encoding="utf-8")
        assert "min_train_samples" not in src
        assert "min_epochs" not in src


# ═══════════════════════════════════════════════════════════════════════════════
# 确定性开关的**环境变量守卫**透传（GRASPO_DETERMINISM）
#
# 判据（本区块的机核）：默认关（宿主不设该变量）⇒ 生成物里的行为**逐字不变**；
# 设了 ⇒ entry.sh 在**容器内运行时**追加 `--determinism-spec`，并**显式打印**原值。
# 为什么判据在运行时而不是生成期：生成期读宿主环境会让生成物随宿主环境漂移，
# 而"生成物是确定版本的函数"是 §14.4 单向流动的前提。
# ═══════════════════════════════════════════════════════════════════════════════


def _rendered_runner_text() -> str:
    return generate_matrix.RUNNER_PATH.read_text(encoding="utf-8")


def _entry_sh_heredoc_text() -> str:
    """entry.sh 的 heredoc 正文（逐字，未展开）——容器内真正执行的那段。"""
    text = _rendered_runner_text()
    start = text.index('cat > "$RUN_DIR/entry.sh" <<ENTRY')
    end = text.index("\nENTRY\n", start)
    return text[start:end]


def test_runner_guard_variable_is_the_declared_single_source():
    """守卫变量名只在生成器的常量里定义一次（§1.4），生成物与它同源。"""
    assert generate_matrix.DETERMINISM_ENV == "GRASPO_DETERMINISM"
    assert generate_matrix.DETERMINISM_ENV in _rendered_runner_text()
    assert generate_matrix.DETERMINISM_ENV in _entry_sh_heredoc_text()


def test_default_off_guard_passes_no_determinism_argument():
    """★默认关机核（静态）：无值 ⇒ 参数数组为空，torchrun 命令行**不追加**任何参数。

    `"${DETERMINISM_ARGS[@]}"` 在空数组上的展开结果是**零个词**（与直接不写该行等价），
    因此这一行在默认关时的命令语义与不引入本开关时逐字相同。这里用真 bash 展开验证，
    而不是靠"我认为它等价"——空数组展开是 shell 语义里最容易想当然的一处。
    """
    import shutil  # noqa: PLC0415
    import subprocess  # noqa: PLC0415

    bash = shutil.which("bash")
    if bash is None:
        pytest.skip("本机没有 bash，无法验证空数组展开语义")
    script = (
        "DETERMINISM_ARGS=()\n"
        'set -- --config CFG "${DETERMINISM_ARGS[@]}"\n'
        'printf "%s\\n" "$@"\n'
        'printf "count=%s\\n" "$#"\n'
    )
    completed = subprocess.run([bash, "-c", script], capture_output=True, text=True, check=False)
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.splitlines() == ["--config", "CFG", "count=2"]

    # 容器内看到的环境变量必须**真的未设置**（不是空串）：`-e VAR=空串` 与不设不同。
    assert "-e GRASPO_DETERMINISM" in _rendered_runner_text()
    assert '-e "GRASPO_DETERMINISM=' not in _rendered_runner_text()


def test_guard_appends_spec_only_when_the_variable_is_non_empty():
    """★开启机核（静态）：非空 ⇒ 追加 `--determinism-spec`，且把**原值逐字**传出。"""
    body = _entry_sh_heredoc_text()
    assert r'if [ -n "\${GRASPO_DETERMINISM:-}" ]; then' in body
    assert r'DETERMINISM_ARGS+=(--determinism-spec "\${GRASPO_DETERMINISM}")' in body
    # 命令行真的用上了这个数组（否则守卫只是装饰）。
    assert r'"\${DETERMINISM_ARGS[@]}"' in body
    # 原值必须打印（§2.2 显式：dry-run 与真跑都要能看见生效的开关）。
    assert r"GRASPO_DETERMINISM=\${GRASPO_DETERMINISM}" in body


def test_runner_dry_run_prints_determinism_state_both_ways():
    """dry-run 必须显式打印守卫变量的当前状态（设了/没设两条路都要有输出）。"""
    text = _rendered_runner_text()
    assert "[dry-run] determinism（守卫变量 GRASPO_DETERMINISM" in text
    assert "已设置 ⇒ 容器内将透传 --determinism-spec（原值逐字）" in text
    assert "未设置 ⇒ 不透传 --determinism-spec（默认关" in text


def test_generator_dry_run_reports_the_guard_variable(tmp_path, monkeypatch, capsys):
    """生成器自身的 `--dry-run` 也要报（默认关必须显式可见，§2.2）。"""
    monkeypatch.delenv(generate_matrix.DETERMINISM_ENV, raising=False)
    generate_matrix.print_determinism_passthrough(dry_run=True)
    off = capsys.readouterr().out
    assert "[dry-run] 确定性开关（GRASPO_DETERMINISM）未设置" in off
    assert "默认关" in off

    monkeypatch.setenv(generate_matrix.DETERMINISM_ENV, '{"enabled":true}')
    generate_matrix.print_determinism_passthrough(dry_run=True)
    on = capsys.readouterr().out
    assert "[dry-run] 确定性开关（GRASPO_DETERMINISM）已设置" in on
    # 原值逐字（生成器不解析、不转述任何确定性字段）。
    assert '{"enabled":true}' in on


def test_guard_does_not_weaken_existing_guards():
    """★透传不得削弱任何既有守卫：逐个具名断言（§2.3 边界校验；不靠"看起来还在"）。"""
    text = _rendered_runner_text()
    required = (
        # 锁卡 / 空闲断言
        "gpu_lock_guard",
        "GRASPO_SKIP_IDLE_ASSERT",
        # monitor-link（GPU 采样的运行链路前置校验）
        "FATAL(runtime-link)",
        # ckpt 保留
        "retain_single_checkpoint",
        # 退出码单点收口
        'exit "$(cat "$RUN_DIR/exit_code")"',
        # torch_probe 顺序（脚本重载探测必须先于训练）
        "torch_probe.py",
        # 缓存根预检
        "FATAL(cache-root)",
        # 产物属主（--user）
        '"${RUN_USER_ARGS[@]}"',
    )
    for needle in required:
        assert needle in text, f"既有守卫被削弱/删除：{needle}"
    # 退出码断言不得被改写：worker rc 仍逐字写进 exit_code。
    assert 'echo "$?" > "$RUN_DIR/exit_code"' in text
    # torch_probe 的调用顺序：探测必须出现在 torchrun 之前。
    assert text.index("torch_probe.py") < text.index("-m graspo.cli.train_worker")
