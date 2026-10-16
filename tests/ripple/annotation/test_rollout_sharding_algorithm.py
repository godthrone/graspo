"""★ rollout 分卡 · 算法层（``ripple/``）的机核测试 —— 每条都带**阳性对照**。

对应设计件 `report.md` §10 的 **A-1（dp=1 逐位不变 / 默认路径）**、**A-5（``μ_f`` 与会合前
整组一致 ＋ ``n`` 整数严格）**、**A-10（``ripple/`` 的 diff 只允许两处）**。

**被测对象**：``graspo.ripple.annotation.advantages`` 的四个函数
:func:`group_field_stats` / :func:`merge_field_stats` / :func:`field_means` /
:func:`compute_group_advantages`（新增的 ``field_mean_override`` 参数）。

**红线**：本文件**不验证**任何新的算法语义 —— 它验证的是
「新增的 ``(Σraw, n)`` 两量合并路径 == 单卡整组的同一多重集均值」，
以及「``field_mean_override=None`` 时与原式**逐位相同**」（A-10 的唯一可比口径）。
"""

from __future__ import annotations

import pytest

from graspo.ripple.annotation.advantages import (
    compute_group_advantages,
    field_means,
    group_field_stats,
    merge_field_stats,
    value_spans,
)
from graspo.ripple.annotation.char_tag import CharTag
from graspo.ripple.annotation.labeler import AnnotationInput, annotate

ROBOT = (
    "<tool_call>\n<function=robot_atomic_control>\n<parameter=action_type>\n"
    "{}\n</parameter>\n<parameter=angle_deg>\n{}\n</parameter>\n</function>\n</tool_call>"
)

TARGETS = [
    {
        "output": {
            "tool_calls": [
                {
                    "name": "robot_atomic_control",
                    "arguments": {"action_type": "move", "angle_deg": 30},
                }
            ]
        }
    }
]


class FakeTokenizer:
    """每字符一个 token，offset ``(i, i+1)`` —— 字符级标注与 token 一一对应。"""

    def __init__(self, eos_token_id: int = 2):
        self.eos_token_id = eos_token_id

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        ids = [ord(c) % 1000 for c in text]
        if add_special_tokens:
            ids.append(self.eos_token_id)
        return ids

    def __call__(self, text: str, return_offsets_mapping: bool = False, **kwargs):
        ids = self.encode(text)
        offsets = [(i, i + 1) for i in range(len(text))]

        class FakeEncoding:
            def __init__(self, input_ids, offset_mapping):
                self.input_ids = input_ids
                self.offset_mapping = offset_mapping

            def get(self, key, default=None):
                return getattr(self, key, default)

            def __getitem__(self, key):
                return getattr(self, key)

        return FakeEncoding(ids, offsets)


@pytest.fixture()
def tokenizer() -> FakeTokenizer:
    return FakeTokenizer()


def _annotate(texts: list[str], tok: FakeTokenizer) -> list:
    return [
        annotate(
            AnnotationInput(
                completion_text=text,
                targets=TARGETS,
                tokenizer=tok,
                format_type="tool_call",
            )
        )
        for text in texts
    ]


#: 8 条 completion：前 4 条 action_type 全对（raw=1.0），后 4 条里有 1 条错、3 条对
#: ⇒ ``action_type`` 的整组 ``n = 8``、``Σraw = 7.0`` ⇒ ``μ = 0.875``。
#: ``angle_deg`` 部分是「**漏发字段**」的构造素材（见 A-5 的 ``n`` 严格用例）。
def _group_texts() -> list[str]:
    good = ROBOT.format("move", 30)
    bad = ROBOT.format("walk", 30)
    return [good, good, good, good, bad, good, good, good]


# ══════════════════════════════════════════════════════════════════════════════
# A-1 / A-10：`field_mean_override=None` ⇒ 与原式**逐位相同**
# ══════════════════════════════════════════════════════════════════════════════

def _legacy_mu(completions, annotations, targets, tok, format_type, numeric_tolerance):
    """改动前的 μ 算术（原样抄写，仅作**对拍基准**，⛔ 不进生产路径）。"""
    tags_list = [ann.tags for ann in annotations]
    field_scores: dict[str, list[float]] = {}
    for text, tags, ann in zip(completions, tags_list, annotations, strict=True):
        from graspo.ripple.annotation.advantages import field_score, gt_value_for_field

        for s, e, f in value_spans(text, tags, ann.fields):
            gt = gt_value_for_field(targets, f)
            raw = field_score(text[s:e], gt, numeric_tolerance) if gt is not None else 0.0
            field_scores.setdefault(f, []).append(raw)
    return {f: sum(scores) / len(scores) for f, scores in field_scores.items()}


def test_dp1_default_path_mu_is_bit_identical_to_legacy_arithmetic(tokenizer):
    """★ A-1/A-10：``field_mean_override=None`` 的 μ 与改动前**逐位相同**。

    对拍：``field_means(group_field_stats(...))`` vs **原样抄写**的旧算术
    （同一批 raw、同一加序）。判据是 **float 严格 ``==``**（不是容差）。
    """
    texts = _group_texts()
    anns = _annotate(texts, tokenizer)
    legacy = _legacy_mu(texts, anns, TARGETS, tokenizer, "tool_call", 0.2)
    new = field_means(
        group_field_stats(
            completions=texts,
            annotations=anns,
            targets=TARGETS,
            tokenizer=tokenizer,
            format_type="tool_call",
            numeric_tolerance=0.2,
        )
    )
    assert new == legacy  # ★ float 严格相等（逐位）
    assert set(new) == set(legacy)
    for field, value in legacy.items():
        assert new[field] == value, (field, new[field], value)


def test_dp1_default_path_advantages_are_bit_identical(tokenizer):
    """★ A-1：``field_mean_override=None`` 与显式注入**本地 μ** 的输出逐位相同。

    这条同时证明「``field_mean_override`` 的默认分支 == 注入本地算出的 μ」
    ⇒ 新参数**不改变** dp=1 路径。
    """
    texts = _group_texts()
    anns = _annotate(texts, tokenizer)
    kwargs = dict(
        completions=texts,
        annotations=anns,
        targets=TARGETS,
        tokenizer=tokenizer,
        format_type="tool_call",
        numeric_tolerance=0.2,
    )
    default = compute_group_advantages(**kwargs)
    local_mu = field_means(
        group_field_stats(
            completions=texts,
            annotations=anns,
            targets=TARGETS,
            tokenizer=tokenizer,
            format_type="tool_call",
            numeric_tolerance=0.2,
        )
    )
    explicit = compute_group_advantages(**kwargs, field_mean_override=local_mu)
    assert default == explicit  # float 严格相等
    assert len(default) == len(texts)


# ══════════════════════════════════════════════════════════════════════════════
# A-5：跨卡 (Σraw, n) 合并 == 单卡整组；`n` **整数严格**
# ══════════════════════════════════════════════════════════════════════════════

def test_merged_field_stats_equal_singleton_over_all_slices(tokenizer):
    """★ A-5：`m=2/4` 的切片统计**按槽位序**合并后 == 单卡整组的 `(Σraw, n)`。"""
    texts = _group_texts()
    anns = _annotate(texts, tokenizer)
    common = dict(
        targets=TARGETS, tokenizer=tokenizer, format_type="tool_call", numeric_tolerance=0.2
    )
    whole = group_field_stats(completions=texts, annotations=anns, **common)

    for m in (2, 4):
        rpr = len(texts) // m
        per_rank = [
            group_field_stats(
                completions=texts[r * rpr : (r + 1) * rpr],
                annotations=anns[r * rpr : (r + 1) * rpr],
                **common,
            )
            for r in range(m)
        ]
        merged = merge_field_stats(per_rank)
        # ★ `n` 必须**整数严格相等**（定义域判据，容差吞不掉）
        assert {f: n for f, (_, n) in merged.items()} == {
            f: n for f, (_, n) in whole.items()
        }, f"m={m} 的 n 不严格相等"
        # Σraw 与 μ 也一致（同加序 ⇒ 本例可严格相等；生产按容差判）
        for field, (raw_sum, _) in whole.items():
            assert merged[field][0] == raw_sum, f"m={m} field={field} 的 Σraw 不等"
        assert field_means(merged) == field_means(whole)


def test_positive_control_dropping_one_completion_makes_n_differ(tokenizer):
    """★ A-5 阳性对照：**漏一条 completion** ⇒ `n` 必须不等（且 μ 偏移）。

    ★ 这是"容差吞不掉的真错"：若判据写成对 μ 的相对容差，漏一条会被吞掉；
    所以 `n` 必须**单独严判**。
    """
    texts = _group_texts()
    anns = _annotate(texts, tokenizer)
    common = dict(
        targets=TARGETS, tokenizer=tokenizer, format_type="tool_call", numeric_tolerance=0.2
    )
    whole = group_field_stats(completions=texts, annotations=anns, **common)
    dropped = group_field_stats(completions=texts[:-1], annotations=anns[:-1], **common)

    field = "action_type"
    assert whole[field][1] == 8
    assert dropped[field][1] == 7  # ★ 整数严格不等 ⇒ 该错误**必须**被抓住
    assert dropped[field][1] != whole[field][1]
    # μ 也变了（但幅度小到 1e-2 级容差可能吞掉 ⇒ 正是 `n` 严判的理由）
    assert field_means(dropped)[field] != field_means(whole)[field]


def test_positive_control_unmerged_local_slice_gives_wrong_mu(tokenizer):
    """★ A-5 阳性对照：**不做会合**（只用本地 `G/m` 条）⇒ μ **必须**不等。

    这条证明「合并不可省」：`m=2` 时 rank0 只看前 4 条
    （前 4 条 `action_type` 全对 ⇒ 本地 μ=1.0，整组 μ=0.875）。
    """
    texts = _group_texts()
    anns = _annotate(texts, tokenizer)
    common = dict(
        targets=TARGETS, tokenizer=tokenizer, format_type="tool_call", numeric_tolerance=0.2
    )
    whole_mu = field_means(group_field_stats(completions=texts, annotations=anns, **common))
    local_mu = field_means(
        group_field_stats(completions=texts[:4], annotations=anns[:4], **common)
    )
    assert whole_mu["action_type"] == pytest.approx(0.875)
    assert local_mu["action_type"] == pytest.approx(1.0)
    assert local_mu["action_type"] != whole_mu["action_type"]  # ★ 必须不等


def test_positive_control_advantage_differs_when_using_local_mu(tokenizer):
    """★ A-5/(c)：注入**本地切片** μ 时，V token 的 advantage **必须**与整组不同。"""
    texts = _group_texts()
    anns = _annotate(texts, tokenizer)
    common = dict(
        completions=texts,
        annotations=anns,
        targets=TARGETS,
        tokenizer=tokenizer,
        format_type="tool_call",
        numeric_tolerance=0.2,
    )
    whole_mu = field_means(group_field_stats(**{k: v for k, v in common.items()}))
    local_mu = field_means(
        group_field_stats(
            completions=texts[:4],
            annotations=anns[:4],
            targets=TARGETS,
            tokenizer=tokenizer,
            format_type="tool_call",
            numeric_tolerance=0.2,
        )
    )
    whole_adv = compute_group_advantages(**common, field_mean_override=whole_mu)
    local_adv = compute_group_advantages(**common, field_mean_override=local_mu)
    assert whole_adv != local_adv  # ★ 会合不可省（阳性对照）
    # 差值恰为 μ 之差（逐 token 平移）
    delta = whole_mu["action_type"] - local_mu["action_type"]
    assert delta == pytest.approx(-0.125)


# ══════════════════════════════════════════════════════════════════════════════
# n == 0 的字段：不进 μ 定义域（与单卡 `mu.get(field, 0.0)` 一致）
# ══════════════════════════════════════════════════════════════════════════════

def test_fields_with_zero_occurrences_are_not_in_means():
    """★ 定义域判据：``n == 0`` 的字段**不进** ``μ_f``（与 ``mu.get(...,0.0)`` 同效）。"""
    assert field_means({}) == {}
    assert field_means({"a": (0.0, 0)}) == {}  # n == 0 ⇒ 不进定义域
    assert field_means({"a": (2.0, 4)}) == {"a": 0.5}


def test_merge_field_stats_sums_by_given_order_and_keeps_n_integral():
    """★ `merge_field_stats` 按**给定顺序**相加，`n` 保持整数（不引入浮点）。"""
    a = {"f": (1.0, 3)}
    b = {"f": (2.0, 4)}
    c = {"f": (4.0, 1)}
    assert merge_field_stats([a, b, c])["f"] == (7.0, 8)
    merged = merge_field_stats([a, b, c])["f"]
    assert isinstance(merged[1], int)
    assert isinstance(merge_field_stats([{"x": (0.0, 2)}])["x"][1], int)


def test_merge_field_stats_union_of_fields():
    """跨卡字段集合取并（某字段只在某张卡上出现是合法情形）。"""
    merged = merge_field_stats([{"a": (1.0, 1)}, {"b": (2.0, 2)}])
    assert merged == {"a": (1.0, 1), "b": (2.0, 2)}


# ══════════════════════════════════════════════════════════════════════════════
# 参数校验（fail-closed）
# ══════════════════════════════════════════════════════════════════════════════

def test_group_field_stats_rejects_bad_format_type(tokenizer):
    with pytest.raises(ValueError, match="format_type"):
        group_field_stats(
            completions=["x"],
            annotations=_annotate(["x"], tokenizer),
            targets=TARGETS,
            tokenizer=tokenizer,
            format_type="yaml",
        )


def test_group_field_stats_rejects_misaligned_inputs(tokenizer):
    anns = _annotate(["x", "y"], tokenizer)
    with pytest.raises(ValueError, match="aligned"):
        group_field_stats(
            completions=["x"],
            annotations=anns,
            targets=TARGETS,
            tokenizer=tokenizer,
            format_type="tool_call",
        )
    with pytest.raises(ValueError, match="non-empty"):
        group_field_stats(
            completions=[],
            annotations=[],
            targets=TARGETS,
            tokenizer=tokenizer,
            format_type="tool_call",
        )


def test_group_field_stats_rejects_misaligned_truncation(tokenizer):
    anns = _annotate(["x", "y"], tokenizer)
    with pytest.raises(ValueError, match="truncated_by_max"):
        group_field_stats(
            completions=["x", "y"],
            annotations=anns,
            targets=TARGETS,
            tokenizer=tokenizer,
            format_type="tool_call",
            truncated_by_max=[False],
        )


def test_char_tag_enum_still_has_expected_members():
    """防呆：本文件依赖的标签语义未被改动（S/V/E 三态）。"""
    assert {CharTag.STRUCTURE, CharTag.VALUE, CharTag.ERROR} <= set(CharTag)
