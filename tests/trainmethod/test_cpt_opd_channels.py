"""CPT / OPD 通道的**契约测试**（不依赖 torch，本机可跑；宪法 §1.3 计算与设施分离）。

断言的是"接线的 argv / 行形态 / 路由**真含**各自需要的参数"，不是"跑通了训练"：

1. **配置层**：``train_method`` 对 ``cpt``/``opd`` 从**拒绝**变为**接受**；非法组合
   （native 后端、教师缺失、拼错的算法名）**仍然拒绝**；新字段遵守 §2.2。
2. **路由层**：四种训练方法各自走到**预期**的注册表/工厂，且**只有一处**路由真相源。
3. **CPT 契约**：argv 真含 ``--use_chat_template`` / ``--loss_scale``；数据集行是
   ms-swift 官方预训练形态；多模态块保留（"所有训练都支持多模态"）。
4. **OPD 契约**：argv 真含 ``--rlhf_type gkd`` + 教师参数（27B→9B 对可配）；不带
   GRPO 的奖励/``num_iterations`` 等无消费者参数。
5. **不被破坏**：SFT / GRASPO 的 LoRA 与全量两条路径的 argv 与本改动前同形。

判据来源：ms-swift 4.5.3 源码与官方文档（见 ``_config_mapping.py`` 的注释索引），
以及配置层的 fail-closed 断言（CPT · native / OPD · native 均不支持）。
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from graspo.core.discovery import resolve_backend_builder
from graspo.core.schema import (
    TRAIN_METHOD_BACKENDS,
    DistillConfig,
    GraspoConfig,
    MsSwiftConfig,
    PretrainConfig,
    validate_train_method_combination,
)
from graspo.flow.msswift._config_mapping import (
    DEFAULT_CPT_LOSS_SCALE,
    graspo_to_ms_swift_argv,
)
from graspo.flow.msswift.dataset import (
    build_cpt_rows,
    build_opd_rows,
)

STUDENT = "/models/Qwen3.5-9B"
TEACHER = "/models/Qwen3.8-27B"

#: ARD v3 形态样本（与 ``tests/flow/msswift/test_msswift_wiring.py`` 同形）；
#: 末轮 user 内含一个图像块 ⇒ 覆盖"多模态通路"这一范围约束（§6）。
_ARD_SAMPLE = {
    "id": "sample-1",
    "source": "unit-test",
    "data_source": "ard_multi",
    "schema_version": "3.0.0",
    "messages": [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": "../images/frame.jpg"},
                {"type": "text", "text": "这张图里的读数是多少？"},
            ],
        }
    ],
    "targets": [{"id": None, "output": {"content": '{"answer": "42"}', "reasoning": None}}],
}


def _write_ard(path: Path) -> str:
    path.write_text(json.dumps(_ARD_SAMPLE, ensure_ascii=False) + "\n", encoding="utf-8")
    return str(path)


def _image_file(tmp_path: Path) -> Path:
    images = tmp_path / "images"
    images.mkdir(exist_ok=True)
    frame = images / "frame.jpg"
    frame.write_bytes(b"\xff\xd8\xff\xd9")  # 内容无关紧要：本测试不加载图像
    return frame


def _cfg(tmp_path: Path, **overrides: object) -> GraspoConfig:
    data: dict[str, object] = {
        "train_method": "cpt",
        "backend": "msswift",
        "model": {"model_path": STUDENT},
        "data": {"train_path": _write_ard(tmp_path / "ard.jsonl")},
        "training": {
            "output_dir": str(tmp_path / "out"),
            "run_name": "unit",
            "overwrite_output_dir": True,
            "max_new_tokens": 256,
        },
    }
    data.update(overrides)
    return GraspoConfig.model_validate(data)


def _opd_cfg(tmp_path: Path, **overrides: object) -> GraspoConfig:
    data: dict[str, object] = {
        "train_method": "opd",
        "distill": {"teacher_model_path": TEACHER},
    }
    data.update(overrides)
    return _cfg(tmp_path, **data)


def _argv(config: GraspoConfig, stage: str, **extra: object) -> list[str]:
    return graspo_to_ms_swift_argv(
        config, stage=stage, dataset_path="d.jsonl", output_dir="o", **extra
    )


def _value_of(argv: list[str], flag: str) -> str | None:
    if flag not in argv:
        return None
    index = argv.index(flag)
    return argv[index + 1] if index + 1 < len(argv) else None


# ── ① 配置层：cpt/opd 从"拒绝"变为"接受"，非法组合仍拒绝 ────────────────────


def test_schema_literal_accepts_cpt_and_opd():
    """`train_method` 的取值集合已含 cpt/opd（本工作包的核心解封项）。"""
    assert "cpt" in TRAIN_METHOD_BACKENDS
    assert "opd" in TRAIN_METHOD_BACKENDS
    assert GraspoConfig.model_validate({"train_method": "cpt", "backend": "msswift"})
    assert GraspoConfig.model_validate(
        {
            "train_method": "opd",
            "backend": "msswift",
            "distill": {"teacher_model_path": TEACHER},
        }
    )


def test_schema_rejects_train_method_outside_enum():
    """枚举之外的算法名仍被拒（`Literal` 未放松）。"""
    with pytest.raises(ValidationError, match="train_method"):
        GraspoConfig.model_validate({"train_method": "dpo", "backend": "msswift"})


def test_schema_rejects_cpt_and_opd_on_native():
    """★ 非法组合仍拒绝：CPT/OPD 在 native 侧是 `⛔ 不支持`（能力矩阵 §4）。"""
    with pytest.raises(ValueError, match="not supported on backend"):
        GraspoConfig.model_validate({"train_method": "cpt", "backend": "native"})
    with pytest.raises(ValueError, match="not supported on backend"):
        GraspoConfig.model_validate(
            {
                "train_method": "opd",
                "backend": "native",
                "distill": {"teacher_model_path": TEACHER},
            }
        )


def test_schema_rejects_opd_without_teacher_model():
    """OPD 的教师必须**具体**：空/未给 ⇒ 拒绝（不留"教师待定"）。"""
    with pytest.raises(ValueError, match="teacher_model_path"):
        GraspoConfig.model_validate({"train_method": "opd", "backend": "msswift"})
    with pytest.raises(ValueError, match="teacher_model_path"):
        GraspoConfig.model_validate(
            {"train_method": "opd", "backend": "msswift", "distill": {"teacher_model_path": "   "}}
        )


def test_validate_helper_is_pure_and_total():
    """纯函数可直接单测：未知 train_method 也走**明确的拒绝**而不是静默兜底。"""
    validate_train_method_combination(
        train_method="opd", backend="msswift", distill_teacher_model_path=TEACHER
    )
    with pytest.raises(ValueError, match="train_method must be one of"):
        validate_train_method_combination(
            train_method="dpo", backend="msswift", distill_teacher_model_path=None
        )


def test_new_sections_forbid_unknown_fields():
    """`extra="forbid"` 未被放松（宪法 §2.3 边界校验即防呆）。"""
    with pytest.raises(ValidationError):
        GraspoConfig.model_validate(
            {"train_method": "cpt", "backend": "msswift", "pretrain": {"loss_scales": "all"}}
        )
    with pytest.raises(ValidationError):
        GraspoConfig.model_validate(
            {
                "train_method": "opd",
                "backend": "msswift",
                "distill": {"teacher_model_path": TEACHER, "teacher_modle": "x"},
            }
        )


def test_new_optional_fields_use_annotated_none_defaults():
    """§2.2：默认值为 None 的字段必须标注 ``X | None``；不出现非 None 哨兵默认值。"""
    for model in (PretrainConfig, DistillConfig):
        for name, field in model.model_fields.items():
            if field.default is None:
                annotation = str(field.annotation)
                assert "None" in annotation, (
                    f"{model.__name__}.{name} defaults to None but is annotated {annotation!r}"
                )
            else:
                # 本工作包新增的字段**全部**是 `X | None = None`；出现别的一律是回归。
                raise AssertionError(
                    f"{model.__name__}.{name} must default to None (got {field.default!r})"
                )


def test_existing_train_methods_still_accepted():
    """回归：既有 `graspo` / `sft` 组合不受影响。"""
    assert GraspoConfig.model_validate({"train_method": "graspo"}).train_method == "graspo"
    assert (
        GraspoConfig.model_validate({"train_method": "sft", "backend": "msswift"}).train_method
        == "sft"
    )


# ── ①b 教师来源二选一：本地路径 **或** 外部服务 URL（2026-09-30 主席裁定路线 B）──
#
# 约束语义未放宽：OPD 仍必须有**具体**教师来源（不留"教师待定"），只是形态由
# "仅路径"扩展为"路径或服务 URL"。四种组合逐一覆盖（每个一个最小用例）。

#: 外部教师服务地址（`distill.teacher_model_server`）；形态见 ms-swift
#: `swift/rlhf_trainers/gkd_helpers.py::parse_teacher_model_server`（单 URL 或多教师 JSON 列表）。
TEACHER_SERVER = "http://127.0.0.1:16889"


def test_teacher_model_server_defaults_to_disabled(tmp_path: Path):
    """★ 新字段默认"未启用"（§2.2）：默认 ``None``，且未给时不透传 `--teacher_model_server`。"""
    assert DistillConfig().teacher_model_server is None
    argv = _argv(_opd_cfg(tmp_path), "opd")
    assert "--teacher_model_server" not in argv
    assert _value_of(argv, "--teacher_model") == TEACHER  # 既有"仅路径"路径行为不变


def test_teacher_source_path_only_is_accepted(tmp_path: Path):
    """组合 1/4：**仅路径** ⇒ 合法，argv 只带 `--teacher_model`（向后兼容）。"""
    argv = _argv(_opd_cfg(tmp_path), "opd")
    assert _value_of(argv, "--teacher_model") == TEACHER
    assert "--teacher_model_server" not in argv


def test_teacher_source_server_only_is_accepted(tmp_path: Path):
    """组合 2/4：**仅服务 URL** ⇒ 合法，argv 只带 `--teacher_model_server`。

    这是本工作包的解封项：不加宽 OPD 闸门的话该字段永远不可达（死配置，§7.2/§18.1）。
    """
    config = _opd_cfg(
        tmp_path, distill={"teacher_model_server": TEACHER_SERVER, "gkd_logits_topk": 64}
    )
    argv = _argv(config, "opd")
    assert _value_of(argv, "--teacher_model_server") == TEACHER_SERVER
    assert "--teacher_model" not in argv, "两者互斥 ⇒ 不得同时透传"
    assert _value_of(argv, "--gkd_logits_topk") == "64"


def test_teacher_model_server_requires_gkd_logits_topk(tmp_path: Path):
    """★ 走外部教师服务但**没配 top-k** ⇒ 配置加载即拒绝（§2.3 早失败防呆）。

    上游 ms-swift 的教师 API 只回 top-k logprobs ⇒ 该组合是**确定性非法**
    （`swift/arguments/rlhf_args.py:762-765` 无条件 raise）。判据不等到后端启动才给，
    而是在配置边界就拦下，且错误信息指到"服务端路径必须配 top-k"。
    """
    with pytest.raises(ValueError, match="gkd_logits_topk"):
        GraspoConfig.model_validate(
            {
                "train_method": "opd",
                "backend": "msswift",
                "distill": {"teacher_model_server": TEACHER_SERVER},
            }
        )


def test_teacher_source_both_given_is_rejected(tmp_path: Path):
    """组合 3/4：**两者同时给** ⇒ 配置加载即拒绝（§2.3 fail-closed；与 ms-swift 同义）。"""
    with pytest.raises(ValueError, match="mutually exclusive"):
        GraspoConfig.model_validate(
            {
                "train_method": "opd",
                "backend": "msswift",
                "distill": {
                    "teacher_model_path": TEACHER,
                    "teacher_model_server": TEACHER_SERVER,
                },
            }
        )


def test_teacher_source_both_absent_is_rejected(tmp_path: Path):
    """组合 4/4：**都不给** ⇒ 仍拒绝（不放宽成"教师待定"）。"""
    with pytest.raises(ValueError, match="teacher_model_path"):
        GraspoConfig.model_validate({"train_method": "opd", "backend": "msswift"})


# ── ② 路由层：四条路由各走预期注册表，且只有一处真相源 ──────────────────────


_ROUTE_EXPECTATIONS = {
    "graspo": ("graspo.backends", "graspo.flow.msswift.trainer", "create_msswift_trainer"),
    "sft": ("graspo.sft_backends", "graspo.flow.msswift.sft_trainer", "create_msswift_sft_trainer"),
    "cpt": ("graspo.cpt_backends", "graspo.flow.msswift.cpt_trainer", "create_msswift_cpt_trainer"),
    "opd": ("graspo.opd_backends", "graspo.flow.msswift.opd_trainer", "create_msswift_opd_trainer"),
}


@pytest.mark.parametrize("train_method", sorted(_ROUTE_EXPECTATIONS))
def test_each_train_method_routes_to_its_own_builder(train_method: str):
    """★ 四条路由各自走到**预期**的 builder（不是"都掉进 RL 注册表"）。"""
    import graspo.core.discovery as discovery

    group, module_name, attr_name = _ROUTE_EXPECTATIONS[train_method]
    assert discovery._REGISTRY_BY_TRAIN_METHOD[train_method] == group

    builder = resolve_backend_builder("msswift", train_method=train_method)
    assert builder.__module__ == module_name
    assert builder.__name__ == attr_name


def test_routing_table_is_the_single_source_of_truth():
    """★ 后端选择只有一个真相源：路由表 + 各注册表，且四组互不重复。"""
    import graspo.core.discovery as discovery

    registry = discovery._REGISTRY_BY_TRAIN_METHOD
    assert set(registry) == {"graspo", "sft", "cpt", "opd"}
    # 四组名互不相同 ⇒ 不存在"两个 train_method 共用一张表"的隐性耦合
    assert len(set(registry.values())) == len(registry)
    # 每组都在 _DEV_FALLBACKS 里有实现（生产由 entry_points 接管，见 discovery 文档）
    for group in registry.values():
        assert group in discovery._DEV_FALLBACKS


def test_cpt_and_opd_are_ms_swift_only_routes():
    """CPT/OPD 只有 msswift 一个后端实现 ⇒ native 请求被明确拒绝，不静默兜底。"""
    for train_method in ("cpt", "opd"):
        with pytest.raises(ValueError, match="No cpt trainer|No opd trainer"):
            resolve_backend_builder("native", train_method=train_method)


def test_unknown_train_method_is_rejected_not_routed_to_rl():
    """★ 未知算法名**不再**静默落到 RL 注册表（旧行为是 '非 sft 即 RL'）。"""
    with pytest.raises(ValueError, match="Unknown train_method"):
        resolve_backend_builder("msswift", train_method="dpo")
    with pytest.raises(ValueError, match="Unknown train_method"):
        resolve_backend_builder("msswift", train_method="cpt ")


def test_existing_registries_unchanged():
    """回归：`graspo.backends` / `graspo.sft_backends` 的既有条目一个不少。"""
    import graspo.core.discovery as discovery

    assert {"native", "msswift"} <= set(discovery._DEV_FALLBACKS["graspo.backends"])
    assert {"native", "msswift"} <= set(discovery._DEV_FALLBACKS["graspo.sft_backends"])


# ── ③ CPT 契约：argv 真含所需参数；LoRA / 全量两条路径都在 ────────────────────


def test_cpt_argv_carries_pretrain_parameters(tmp_path: Path):
    """★ CPT 的 argv 真含 `swift pt` 的两条定义性参数。"""
    argv = _argv(_cfg(tmp_path), "cpt")
    assert _value_of(argv, "--use_chat_template") == "false"
    assert _value_of(argv, "--loss_scale") == DEFAULT_CPT_LOSS_SCALE == "all"
    assert _value_of(argv, "--dataset") == "d.jsonl"
    assert _value_of(argv, "--max_length") == "2048"


def test_cpt_argv_honours_explicit_pretrain_overrides(tmp_path: Path):
    """显式值与 None 语义不同（§2.2）：给了就照给。"""
    config = _cfg(tmp_path, pretrain={"loss_scale": "last_round", "use_chat_template": True})
    argv = _argv(config, "cpt")
    assert _value_of(argv, "--loss_scale") == "last_round"
    assert _value_of(argv, "--use_chat_template") == "true"


def test_cpt_argv_has_no_rl_only_parameters(tmp_path: Path):
    """CPT 不得携带 RL/OPD 专有参数（ms-swift 的 `PretrainArguments` 没有它们）。"""
    argv = _argv(_cfg(tmp_path), "cpt")
    for flag in ("--rlhf_type", "--reward_funcs", "--max_completion_length", "--num_iterations"):
        assert flag not in argv, f"CPT argv must not contain {flag}"


@pytest.mark.parametrize("tuner", ["lora", "full"])
def test_cpt_supports_lora_and_full(tmp_path: Path, tuner: str):
    """★ 9 档 CPT 里 LoRA 3 档 + 全量 3 档（9B）都必须可表达。"""
    argv = _argv(_cfg(tmp_path, tuner_type=tuner), "cpt")
    assert _value_of(argv, "--tuner_type") == tuner
    if tuner == "lora":
        assert _value_of(argv, "--lora_rank") == "16"
        assert "--freeze_vit" not in argv
    else:
        assert _value_of(argv, "--freeze_vit") == "false"
        assert _value_of(argv, "--freeze_aligner") == "false"
        assert "--lora_rank" not in argv  # 全参下 LoRA 参数是"假配置"，必须不出现


def _stub_sample(messages: list[dict], targets: list[dict] | None = None):
    """graspo ``Sample`` 的**结构替身**（只含两个构建器真正读的属性）。

    为什么不走 ``load_graspo_samples``：本机无 torch，而它的实现链
    （``graspo.ripple.data``）在模块顶层 ``import torch``。行形态是 ``build_*_rows``
    的职责，不是 loader 的职责——因此用替身把被测边界收紧到"行形态"本身
    （宪法 §1.3 计算与设施分离；loader 自身由容器内测试覆盖）。
    """
    return SimpleNamespace(messages=messages, targets=targets if targets is not None else [])


def _cpt_sample():
    return _stub_sample(
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": "images/frame.jpg"},
                    {"type": "text", "text": "这张图里的读数是多少？"},
                ],
            }
        ],
        # graspo **原生** target 形态：``output.content`` 是 dict（ARD 里的 str 形态
        # 会先被适配器转成这个形态再进入构建器）。
        targets=[{"id": None, "output": {"content": {"text": "42"}, "reasoning": None}}],
    )


def test_cpt_rows_use_official_pretrain_shape(tmp_path: Path):
    """CPT 行 = ms-swift 官方预训练形态：单条 assistant 消息承载纯文本。"""
    config = _cfg(tmp_path)
    rows = build_cpt_rows([_cpt_sample()], config=config)
    assert len(rows) == 1
    messages = rows[0]["messages"]
    assert len(messages) == 1 and messages[0]["role"] == "assistant"
    blocks = messages[0]["content"]
    assert isinstance(blocks, list)
    text = "".join(b.get("text", "") for b in blocks if b.get("type") == "text")
    assert "这张图里的读数是多少？" in text
    assert "42" in text  # 首条 target 的答案文本一并进入续训文本


def test_cpt_rows_keep_multimodal_blocks(tmp_path: Path):
    """★ 范围约束（§6）：CPT 也必须在**多模态**通路下可跑 ⇒ 图像块必须保留。

    保留的判据：块留在 content 里，ms-swift 的
    ``StdTemplateInputs.remove_messages_media`` 才抽得出 ``images`` 列。
    """
    config = _cfg(tmp_path)
    blocks = build_cpt_rows([_cpt_sample()], config=config)[0]["messages"][0]["content"]
    images = [b for b in blocks if b.get("type") == "image"]
    assert images, "CPT row dropped the image block; multimodal CPT would silently train text-only"
    assert images[0].get("image"), "image block must keep its path under the key ms-swift reads"


def test_cpt_rows_reject_empty_text(tmp_path: Path):
    """拒绝产出空行（"训练"在纯 padding 上没有意义 ⇒ 边界校验 fail-closed）。"""
    config = _cfg(tmp_path)
    empty = _stub_sample(messages=[{"role": "user", "content": "   "}])
    with pytest.raises(ValueError, match="no plain text"):
        build_cpt_rows([empty], config=config)


def test_prepare_dataset_resolves_media_to_absolute_paths(tmp_path: Path, monkeypatch):
    """多模态通路：相对图像路径在写盘前解析为绝对路径（ms-swift 按 cwd 打开图像）。

    只替换 loader（它是设施入口），**不替换** ``resolve_sample_media_paths`` 与行构建器
    ——被测的正是"写盘前把相对路径变成绝对路径"这一步。
    """
    import graspo.flow.msswift.dataset as dataset

    frame = _image_file(tmp_path)
    monkeypatch.setattr(dataset, "load_graspo_samples", lambda _path: [_cpt_sample()])
    config = _cfg(tmp_path)
    out = dataset.prepare_ms_swift_dataset(config, stage="cpt", work_dir=tmp_path / "work")
    row = json.loads(Path(out).read_text(encoding="utf-8").splitlines()[0])
    image = next(b for b in row["messages"][0]["content"] if b["type"] == "image")
    assert Path(image["image"]).is_absolute(), image
    assert Path(image["image"]).name == frame.name


@pytest.mark.parametrize("stage", ["sft", "rlhf", "cpt", "opd"])
def test_prepare_dataset_stage_dispatch(tmp_path: Path, monkeypatch, stage: str):
    """四种 stage 各有自己的行构建器，产物文件名带 stage（不串味）。"""
    import graspo.flow.msswift.dataset as dataset

    monkeypatch.setattr(dataset, "load_graspo_samples", lambda _path: [_cpt_sample()])
    if stage == "opd":
        config = _opd_cfg(tmp_path)
    elif stage == "cpt":
        config = _cfg(tmp_path)
    else:
        config = _cfg(tmp_path, train_method="sft" if stage == "sft" else "graspo")
    out = Path(dataset.prepare_ms_swift_dataset(config, stage=stage, work_dir=tmp_path / "work"))
    assert out.name == f"ms_swift_{stage}.jsonl"
    row = json.loads(out.read_text(encoding="utf-8").splitlines()[0])
    assert "messages" in row
    if stage == "rlhf":
        assert "targets" in row
    else:
        assert "targets" not in row


def test_prepare_dataset_stage_dispatch_is_closed(tmp_path: Path, monkeypatch):
    """未知 stage 明确拒绝（不静默退回某个既有形态）。"""
    import graspo.flow.msswift.dataset as dataset

    monkeypatch.setattr(dataset, "load_graspo_samples", lambda _path: [_cpt_sample()])
    config = _cfg(tmp_path)
    with pytest.raises(ValueError, match="stage must be one of"):
        dataset.prepare_ms_swift_dataset(config, stage="pretrain", work_dir=tmp_path / "work")


# ── ④ OPD 契约：argv 真含 GKD + 教师参数；不带无消费者参数 ────────────────────


def test_opd_argv_carries_gkd_and_teacher_pair(tmp_path: Path):
    """★ OPD 的 argv 真含 `--rlhf_type gkd` 与教师/学生对（27B → 9B）。"""
    argv = _argv(_opd_cfg(tmp_path), "opd")
    assert _value_of(argv, "--rlhf_type") == "gkd"
    assert _value_of(argv, "--model") == STUDENT
    assert _value_of(argv, "--teacher_model") == TEACHER
    assert _value_of(argv, "--model") != _value_of(argv, "--teacher_model")
    assert _value_of(argv, "--max_completion_length") == "256"  # 学生现场采样有长度上限


def test_opd_argv_carries_optional_teacher_tuning_knobs(tmp_path: Path):
    """可选教师/蒸馏旋钮：给了就透传，字段名与 ms-swift 逐字对应。"""
    config = _opd_cfg(
        tmp_path,
        distill={
            "teacher_model_path": TEACHER,
            "teacher_adapters": ["/adapters/t1"],
            "teacher_deepspeed": "zero3",
            "offload_teacher_model": True,
            "gkd_logits_topk": 64,
            "lmbda": 0.5,
            "sft_alpha": 0.1,
            "beta": 0.0,
        },
    )
    argv = _argv(config, "opd")
    assert _value_of(argv, "--teacher_adapters") == "/adapters/t1"
    assert _value_of(argv, "--teacher_deepspeed") == "zero3"
    assert _value_of(argv, "--offload_teacher_model") == "true"
    assert _value_of(argv, "--gkd_logits_topk") == "64"
    assert _value_of(argv, "--lmbda") == "0.5"
    assert _value_of(argv, "--sft_alpha") == "0.1"
    assert _value_of(argv, "--beta") == "0.0"


def test_opd_does_not_invent_beta_or_num_iterations(tmp_path: Path):
    """★ 不发明取值（§2.2 / §6.1）：未提供 `beta` 就不透传；GKD 不吃 `num_iterations`。"""
    argv = _argv(_opd_cfg(tmp_path), "opd")
    assert "--beta" not in argv, "beta 未提供时应交 ms-swift 默认值（GKD 默认 0.5=JSD）"
    assert "--num_iterations" not in argv, "num_iterations 只有 GRPO 消费；GKD 阶段不得透传"


def test_opd_argv_has_no_grpo_reward_wiring(tmp_path: Path):
    """OPD 的监督信号来自教师 logits ⇒ 不得带 GRPO 的奖励/rollout 参数。"""
    argv = _argv(_opd_cfg(tmp_path), "opd")
    assert "--reward_funcs" not in argv
    assert "--num_generations" not in argv
    assert "--steps_per_generation" not in argv
    assert "--remove_unused_columns" not in argv
    assert _value_of(argv, "--rlhf_type") != "grpo"


@pytest.mark.parametrize("tuner", ["lora", "full"])
def test_opd_supports_lora_and_full(tmp_path: Path, tuner: str):
    """★ 9 档 OPD 里 LoRA 6 档 + 全量 3 档都必须可表达。"""
    argv = _argv(_opd_cfg(tmp_path, tuner_type=tuner), "opd")
    assert _value_of(argv, "--tuner_type") == tuner


def test_opd_rows_are_prompt_only(tmp_path: Path):
    """OPD 行 = 纯提示词：不带 `targets` 奖励列（那是 GRPO 的消费者）。"""
    config = _opd_cfg(tmp_path)
    rows = build_opd_rows([_cpt_sample()], config=config)
    assert len(rows) == 1
    assert "targets" not in rows[0]
    assert rows[0]["messages"][-1]["role"] == "user"


# ── ⑤ 回归：既有 SFT / GRASPO 的 LoRA 与全量路径不被破坏 ─────────────────────


@pytest.mark.parametrize("stage", ["sft", "rlhf"])
@pytest.mark.parametrize("tuner", ["lora", "full"])
def test_existing_channels_unbroken(tmp_path: Path, stage: str, tuner: str):
    """★ 既有两条算法通道 × 两种参数化模式：关键参数形状不变。"""
    train_method = "sft" if stage == "sft" else "graspo"
    argv = _argv(_cfg(tmp_path, train_method=train_method, tuner_type=tuner), stage)
    assert _value_of(argv, "--tuner_type") == tuner
    assert _value_of(argv, "--dataset") == "d.jsonl"
    if tuner == "lora":
        assert _value_of(argv, "--lora_rank") == "16"
    else:
        assert _value_of(argv, "--freeze_vit") == "false"
        assert _value_of(argv, "--freeze_aligner") == "false"
    if stage == "rlhf":
        assert _value_of(argv, "--rlhf_type") == "grpo"
        assert _value_of(argv, "--num_generations") == "8"
        assert _value_of(argv, "--beta") == "0.0"
        assert _value_of(argv, "--num_iterations") == "1"
    else:
        assert "--rlhf_type" not in argv
        assert _value_of(argv, "--max_completion_length") is None


def test_stage_enum_rejects_unknown_stage(tmp_path: Path):
    """stage 的封闭取值未被放松。"""
    with pytest.raises(ValueError, match="stage must be"):
        _argv(_cfg(tmp_path), "pretrain")


# ── ⑦ 配置模板同步（宪法 §7.3 模板即文档；本机无 PyYAML，故用缩进扫描）──────


def _top_level_block(text: str, key: str) -> str:
    """取 YAML 顶层段 ``key:`` 的正文（到下一个零缩进的行首为止）。

    本机没有 PyYAML（容器内才有），而"模板覆盖 schema 全部字段"是仓库自带防呆测试
    （``tests/core/test_schema.py::test_config_example_covers_all_schema_fields``）的判据。
    这里用**缩进扫描**给出同一判据的本机可跑版本——不引入 YAML 解析依赖，也不放宽断言
    （找不到段时直接让断言失败）。
    """
    lines = text.splitlines()
    start = None
    for index, line in enumerate(lines):
        if line.startswith(f"{key}:"):
            start = index + 1
            break
    assert start is not None, f"config_example.yaml is missing top-level section {key!r}"
    body: list[str] = []
    for line in lines[start:]:
        if line and not line[0].isspace() and not line.startswith("#"):
            break
        body.append(line)
    return "\n".join(body)


def _has_key(block: str, name: str) -> bool:
    """段正文里是否存在键 ``name``（允许缩进——YAML 嵌套键不是零缩进的）。"""
    return re.search(rf"^[ \t]*{re.escape(name)}\s*:", block, re.MULTILINE) is not None


@pytest.mark.parametrize(
    ("section", "model"),
    [("pretrain", PretrainConfig), ("distill", DistillConfig)],
)
def test_main_template_covers_new_sections(section: str, model: type) -> None:
    """★ 新增 schema 字段必须同步进主模板（否则容器内模板覆盖测试变红）。"""
    text = Path("samples/configs/config_example.yaml").read_text(encoding="utf-8")
    block = _top_level_block(text, section)
    missing = sorted(name for name in model.model_fields if not _has_key(block, name))
    assert not missing, f"config_example.yaml `{section}` section missing schema fields: {missing}"


def test_main_template_documents_the_widened_train_method():
    """模板必须把新取值写出来（模板即文档：用户看模板就知道 cpt/opd 存在）。"""
    text = Path("samples/configs/config_example.yaml").read_text(encoding="utf-8")
    assert re.search(r"^train_method:.*\bcpt\b.*\bopd\b", text, re.MULTILINE)


def test_msswift_template_still_covers_ms_swift_fields() -> None:
    """★ msswift 专属模板按其**自己的字段集合边界**校验（本包未给 `MsSwiftConfig` 加字段）。

    这一条同时是对指挥官裁定 3 的答复：本工作包**没有**动 ``MsSwiftConfig``，
    因此 ``config_example_msswift.yaml`` **不需要**改动；该测试把这一点变成可核事实
    ——它覆盖 ``MsSwiftConfig`` 的**直属字段**（与仓库自带的那条容器内测试同判据）。
    """
    text = Path("samples/configs/config_example_msswift.yaml").read_text(encoding="utf-8")
    block = _top_level_block(text, "msswift")
    missing = sorted(name for name in MsSwiftConfig.model_fields if not _has_key(block, name))
    assert not missing, f"config_example_msswift.yaml missing msswift fields: {missing}"


# ── ⑥ 工厂契约：构造不导入重量级依赖；执行时不静默降级 ───────────────────────


@pytest.mark.parametrize("train_method", ["cpt", "opd"])
def test_factories_return_trainer_with_train_contract(tmp_path: Path, train_method: str):
    """四条通道统一形状契约：`factory(config, selection) -> 含 train(smoke) 的对象`。"""
    config = _cfg(tmp_path, train_method=train_method, distill={"teacher_model_path": TEACHER})
    trainer = resolve_backend_builder("msswift", train_method=train_method)(config, None)
    assert callable(getattr(trainer, "train", None))
    assert "torch" not in repr(trainer)  # pragma: no cover - 仅防止 repr 泄漏重依赖


@pytest.mark.parametrize("train_method", ["cpt", "opd"])
def test_missing_ms_swift_fails_closed(tmp_path: Path, train_method: str):
    """★ 前置条件不是"伪造通过"：ms-swift 不在时 `train()` 抛 RuntimeError，不静默降级。

    本机（无 ms-swift）正好是这条负向路径的真实执行环境。
    """
    import importlib.util

    if importlib.util.find_spec("swift") is not None:  # pragma: no cover - 容器内
        pytest.skip("ms-swift is importable here; this negative case needs it absent")
    config = _cfg(tmp_path, train_method=train_method, distill={"teacher_model_path": TEACHER})
    trainer = resolve_backend_builder("msswift", train_method=train_method)(config, None)
    with pytest.raises(RuntimeError, match="ms-swift"):
        trainer.train(smoke=True)
