"""54 档**生成配置** ↔ 配置模型（`GraspoConfig`）的契约测试（不依赖 torch / PyYAML）。

**为什么需要它**：`tests/e2e/test_generate_matrix.py` 校验的是**生成器自身**的口径
（计数、指纹、可行性格挡）。但"生成器产出的配置**真的能被配置模型接受**"是另一件事——
生成器完全可能安静地产出一份 schema 不认识的配置，直到上机 `GraspoConfig.from_yaml`
才炸（那正是 CPT/OPD 在 `WP-X3` 之前的状态：`train_method: cpt` 会被 Literal 拒绝）。

**本文件把两者接起来**：对 54 档逐档 `build_config(tier)` → `GraspoConfig.model_validate`
——这是"**可表达**"最直接的判据，且**不需要 YAML 解析器**（本机没有 PyYAML）。

此外还覆盖：
- CPT/OPD 的算法段（`pretrain` / `distill`）**确实进了配置**且取值正确；
- 渲染出的 YAML 文本里**确实出现**这些段（渲染器不丢字段）；
- `cpt|opd + native` 这类非法组合**仍然被拒**（配置层没有为了放行而放松校验）。
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

from graspo.core.schema import GraspoConfig

_REPO_ROOT = Path(__file__).resolve().parents[2]

_GENERATOR_PATH = _REPO_ROOT / "tests" / "e2e" / "generate_matrix.py"
_spec = importlib.util.spec_from_file_location("generate_matrix_for_contract", _GENERATOR_PATH)
assert _spec is not None and _spec.loader is not None
generate_matrix = importlib.util.module_from_spec(_spec)
sys.modules["generate_matrix_for_contract"] = generate_matrix
_spec.loader.exec_module(generate_matrix)

TIERS = generate_matrix.build_ledger()
_BY_ID = {str(tier["tier_id"]): tier for tier in TIERS}


@pytest.mark.parametrize("tier_id", [str(tier["tier_id"]) for tier in TIERS])
def test_generated_config_is_accepted_by_the_schema(tier_id: str) -> None:
    """★ 每一档的生成配置都必须被 `GraspoConfig` 接受（`extra="forbid"` 下的真实校验）。"""
    tier = _BY_ID[tier_id]
    config = GraspoConfig.model_validate(generate_matrix.build_config(tier))
    assert config.train_method == generate_matrix.ALGORITHM_TO_TRAIN_METHOD[str(tier["algorithm"])]
    assert config.backend == ("msswift" if tier["backend"] == "ms-swift" else "native")


@pytest.mark.parametrize("tier_id", ["T001", "T004", "T007"])
def test_cpt_configs_carry_the_pretrain_section(tier_id: str) -> None:
    """CPT 档必须显式带 `pretrain` 段（配置仍是产物的唯一描述，§10.1）。"""
    config = GraspoConfig.model_validate(generate_matrix.build_config(_BY_ID[tier_id]))
    assert config.train_method == "cpt"
    assert config.pretrain.loss_scale == "all"
    assert config.pretrain.use_chat_template is False


@pytest.mark.parametrize("tier_id", ["T046", "T049", "T052"])
def test_opd_configs_carry_a_concrete_teacher_pair(tier_id: str) -> None:
    """★ OPD 档必须带**具体**的教师路径，且教师/学生关系被**显式**记录。

    `T046`/`T049`（9B 学生）是独立教师；`T052`（27B 学生）与教师常量同名 ⇒
    落到 ms-swift 的**自蒸馏**模式。两种都必须写明关系，**不得**靠"两个路径同不同"
    让后人自己猜（自蒸馏与独立教师是两种跑法）。
    """
    config = GraspoConfig.model_validate(generate_matrix.build_config(_BY_ID[tier_id]))
    assert config.train_method == "opd"
    assert config.distill.teacher_model_path == generate_matrix.MODELS["27B"]["path"], tier_id
    assert config.distill.offload_teacher_model is True, tier_id

    relation = generate_matrix.opd_teacher_relation(_BY_ID[tier_id])
    expected = "self-distillation" if tier_id == "T052" else "independent"
    assert relation == expected, (tier_id, relation)
    if relation == "self-distillation":
        # 教师与学生同名 ⇒ 必须有一条显式说明，不能是"悄悄同名"
        note = generate_matrix._algorithm_note(_BY_ID[tier_id]) or ""
        assert "自蒸馏" in note, tier_id


def test_every_opd_config_header_states_the_teacher_student_relation() -> None:
    """★ 9 条 OPD 配置的头部**都必须**写明教师/学生关系（指挥官 2026-09-19 裁定 2 第 ③ 点）。

    为什么每档都要写：教师是**用户拍板的常量**（27B），学生随台账变；两者同名时
    配置看起来像"教师路径写错了"。头部必须让读者**一眼**看出这是不是笔误。
    """
    opd_ids = [tid for tid, t in _BY_ID.items() if t["algorithm"] == "OPD"]
    assert len(opd_ids) == 9

    for tier_id in opd_ids:
        tier = _BY_ID[tier_id]
        text = generate_matrix.render_config_yaml(
            tier, *generate_matrix.classify_expressibility(tier)
        )
        relation = generate_matrix.opd_teacher_relation(tier)
        assert f"教师/学生关系: **{relation}**" in text, tier_id
        # 指向 manifest 的逐档字段，便于读者核对
        assert "opd_teacher_relation" in text, tier_id
        if relation == "self-distillation":
            assert "不是笔误" in text, tier_id


def test_opd_manifest_records_the_teacher_relation_and_the_open_question() -> None:
    """★ 27B OPD 档（T052–T054）的教师/学生歧义必须**进 manifest**，而不是被消解掉。"""
    manifest = generate_matrix.build_manifest(TIERS)
    entries = {entry["tier_id"]: entry for entry in manifest["tiers"]}

    assert entries["T046"]["opd_teacher_relation"] == "independent"
    for tier_id in ("T052", "T053", "T054"):
        assert entries[tier_id]["opd_teacher_relation"] == "self-distillation", tier_id

    opd_teacher = manifest["feasibility_model"]["opd_teacher"]
    assert opd_teacher["teacher_model_path"] == generate_matrix.MODELS["27B"]["path"]
    assert opd_teacher["measured"] is False
    assert "待确认" in opd_teacher["open_question"]


def test_rendered_yaml_keeps_the_algorithm_sections() -> None:
    """渲染器不得吞掉 `pretrain` / `distill` 段（本机用文本断言，不需 YAML 解析器）。"""
    cpt_text = generate_matrix.render_config_yaml(
        _BY_ID["T001"], *generate_matrix.classify_expressibility(_BY_ID["T001"])
    )
    assert "\npretrain:\n" in "\n" + cpt_text
    assert "  loss_scale: all" in cpt_text
    assert "  use_chat_template: false" in cpt_text

    opd_text = generate_matrix.render_config_yaml(
        _BY_ID["T046"], *generate_matrix.classify_expressibility(_BY_ID["T046"])
    )
    assert "\ndistill:\n" in "\n" + opd_text
    assert f"  teacher_model_path: {generate_matrix.MODELS['27B']['path']}" in opd_text
    assert "  offload_teacher_model: true" in opd_text


def test_no_tier_still_produces_a_blocked_stub_for_cpt_or_opd() -> None:
    """★ CPT/OPD 的 18 档都必须有 `.yaml` 产物，不能再有 `.blocked.md`。"""
    blocked = [
        str(tier["tier_id"])
        for tier in TIERS
        if generate_matrix.classify_expressibility(tier)[0] == "blocked"
    ]
    # ★ 订正（2026-10-17，单处）：**期望集合 `["T016", "T034"]` → `[]`**。
    #   事实来源：`generate_matrix.classify_expressibility()`（本文件所测生成器）的 docstring
    #   逐字登记「native 全参 1 卡（T016/T034）**已于 2026-09-23 解除 blocked**」——改走
    #   `native.offload_optimizer_state`，真机实测 `exit=0`／**50/50 步**／宿主峰值 38.18 GiB
    #   （无 offload 对照为 79.24 GiB OOM）；收口提交 = 4 卡线 `36d6959`／`8222190`
    #   （`fix(matrix): move 4-card full-param tiers off AutoTP, enable native 1-card offload`）。
    #   故现行取值为 `unverified`、`blocked` 集合确实为**空**；T016(SFT)/T034(GRASPO) **不属**
    #   CPT/OPD 18 档，下一循环「CPT/OPD 不得 blocked」仍被完整保留（断言只收敛、不放松）。
    #   病史：该陈旧期望已由先前窗口归因并登记（.local 工作日志：期望 `["T016","T034"]` 实得 `[]`）。
    assert blocked == [], blocked
    for tier_id, tier in _BY_ID.items():
        if tier["algorithm"] in {"CPT", "OPD"}:
            assert tier_id not in blocked


def test_config_layer_still_rejects_illegal_cpt_opd_combinations() -> None:
    """★ 放行 CPT/OPD 的同时**没有放松**校验：`cpt|opd + native` 仍被拒。

    这是"从拒绝变为接受"与"非法组合仍拒绝"两句话的后者——必须同时成立。
    """
    with pytest.raises(ValueError, match="not supported on backend"):
        GraspoConfig.model_validate({"train_method": "cpt", "backend": "native"})
    with pytest.raises(ValueError, match="not supported on backend"):
        GraspoConfig.model_validate(
            {
                "train_method": "opd",
                "backend": "native",
                "distill": {"teacher_model_path": generate_matrix.MODELS["27B"]["path"]},
            }
        )
    # OPD 缺教师仍拒
    with pytest.raises(ValueError, match="teacher_model_path"):
        GraspoConfig.model_validate({"train_method": "opd", "backend": "msswift"})
    # 未知算法名仍拒
    with pytest.raises(ValidationError):
        GraspoConfig.model_validate({"train_method": "dpo", "backend": "msswift"})
