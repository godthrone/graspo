"""P3 判据：**组内全对的结构位置**不进梯度（advantage 置 0）。

负责人裁定（唯一判据）：

    「对于一个结构 token 来说，如果他是对的，那么他一定对应 ground truth 里面的某个结构
    字符，对吧？如果这个结构字符对应的 rollout 中每个字符都对，那么这个 token 统一不再进入
    训练……对 ground truth 里面的所有结构字符做个统计，对于那些全对的结构字符，就从最终的
    梯度优化中去掉这些 token 样本。」

实现口径（``advantages.all_correct_structure_chars``）：每组取各条 completion 的 S 字符数的
**最小值 k**，前 k 个 S 字符覆盖的 token 置 0。理由见该函数 docstring（标注器的 S 是严格
前缀：「首个不相等字符标 E，其后全 D」，所以 S 之后的字符不可能被判定为「写对」）。

本文件钉三个层次：

1. ``test_common_prefix_masks_only_group_wide_correct_structure`` —— 合成组（FakeTokenizer），
   钉「取最小值」这件事本身：前缀按**最短的那条**截断，前缀之后的结构 token 仍是 +1.0。
2. 真实数据三个读数（``samples_r1.json`` 800 条 + ``subsets-v2/M016.jsonl`` 的 targets）：
   每条恰好 5 个 token 被置 0、合计 4000；``tool_ok=False`` 的 496 条 Σadv 由 **+3.00** 变
   **−2.00**；Σadv > 0 的条数由 **800** 变 **304**。
   ★ 这三个数**不随 B1 修复（工具名写对 ⇒ V）变化**：100 组里每组都有工具名写错的 rollout，
   它们的 S 字符数恰好 22 ⇒ ``k`` 恒为 22、mask 的字符恒是 ``<tool_call>``+换行+``<function=``；
   工具名写对的那 304 条 S 计数 94→84（不计名字）但不是最小值，抬不动 ``k``。
3. B1 回归（第 3 节）：工具名是**值**，P3 的结构 mask 不得伸进工具名。

⛔ 期望值不用被测实现算：masked token 由「标签是 S 且 advantage 恰为 0」反推（P3 之前
S 恒为 +1.0，只有被 mask 才会是 0），前缀长度用模板常量算出（22 字符）。
⛔ 不动 ``action_mask``（loss 是「序列内 masked_mean、再按 batch 求均值」，缩小 mask 会把
其余 token 的梯度放大 1.21×）——本文件只验证 advantage 取值。
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import pytest

from graspo.ripple.annotation.advantages import apply_tail_eos, compute_group_advantages
from graspo.ripple.annotation.char_tag import CharTag
from graspo.ripple.annotation.labeler import AnnotationInput, annotate
from graspo.ripple.annotation.tokenize import char_to_token_labels

ROOT = Path(__file__).resolve().parents[3]
SAMPLES = ROOT / ".local/hb-workspace/20261118-134125-sft-start-rollout/evidence/samples_r1.json"
SUBSET = ROOT / "subsets-v2/M016.jsonl"
TOKENIZER = ROOT / ".local/p0p1p2-verify/tok"

#: 组内公共前缀的字符数：``<tool_call>``(11) + ``\n``(1) + ``<function=``(10)。独立算出，
#: 不用实现里的 S 字符统计。
COMMON_PREFIX_CHARS = len("<tool_call>") + 1 + len("<function=")

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
                    "arguments": {"action_type": "逆时针旋转", "angle_deg": 16.9},
                }
            ]
        }
    }
]


class FakeTokenizer:
    """每字符一个 token，offset ``(i, i+1)``（与既有 advantage 测试同一模式）。"""

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return [ord(char) % 1000 for char in text]

    def __call__(self, text: str, return_offsets_mapping: bool = False, **kwargs):
        offsets = [(i, i + 1) for i in range(len(text))]

        class FakeEncoding:
            def __init__(self, input_ids, offset_mapping):
                self.input_ids = input_ids
                self.offset_mapping = offset_mapping

            def get(self, key, default=None):
                return getattr(self, key, default)

        return FakeEncoding(self.encode(text), offsets)


def _annotate(completion: str):
    return annotate(
        AnnotationInput(
            completion_text=completion, targets=TARGETS, tokenizer=None, format_type="tool_call"
        )
    )


# ══════════════════════════════════════════════════════════════════════════════
# 1. 「取最小值」这件事本身（合成组，快）
# ══════════════════════════════════════════════════════════════════════════════


def test_common_prefix_masks_only_group_wide_correct_structure() -> None:
    """前缀按**组内最短**的那条截断：前缀内 S→0，前缀外 S 仍 +1.0，E 仍 −1.0。

    组：A 结构全对；B 在 ``</parameter>`` 写成 ``</parametr>``（首个不匹配字符标 E、
    其后全 D）。B 的 S 字符只到 E 之前 ⇒ 组内公共前缀 = B 的 S 长度 ⇒ A 上超出该处的
    结构 token 保持 +1.0（那里 A 写对但 B 没走到，不构成「组内全对」）。
    """
    good = ROBOT.format("逆时针旋转", "16.9")
    typo = good.replace("</parameter>", "</parametr>", 1)
    error_position = typo.find("</parametr>") + len("</paramet")
    assert typo[error_position] == "r", "构造前提：首个不匹配字符是 r"
    annotations = [_annotate(good), _annotate(typo)]

    advice = compute_group_advantages(
        completions=[good, typo],
        annotations=annotations,
        targets=TARGETS,
        tokenizer=FakeTokenizer(),
        format_type="tool_call",
    )
    # 独立算出「组内全对的结构字符」：typo 的 S 字符恰好就是两条共享的正确前缀
    # （标注器从首个不匹配字符起整段 drop），good 在同一前缀上的 S 字符下标与之相同。
    tags_good = apply_tail_eos(good, annotations[0].tags, "tool_call")
    tags_typo = apply_tail_eos(typo, annotations[1].tags, "tool_call")
    common = [i for i, tag in enumerate(tags_typo) if tag is CharTag.STRUCTURE]
    beyond = [i for i, tag in enumerate(tags_good) if tag is CharTag.STRUCTURE and i not in common]
    assert common and beyond, "构造前提：两条既有公共前缀，good 也有超出公共前缀的结构字符"

    # FakeTokenizer 下字符下标 == token 下标
    assert [advice[0][i] for i in common] == [0.0] * len(common), "组内全对的结构必须置 0"
    assert [advice[1][i] for i in common] == [0.0] * len(common)
    assert [advice[0][i] for i in beyond] == [1.0] * len(beyond), (
        "前缀之外：good 写对但 typo 没走到 ⇒ 不是「组内全对」，保持 +1.0"
    )
    assert advice[1][error_position] == -1.0, "E 仍是 −1.0（P3 不碰 E）"


# ══════════════════════════════════════════════════════════════════════════════
# 2. 真实 rollout 数据的三个读数（数据 / tokenizer 是 .local 下的本机产物）
# ══════════════════════════════════════════════════════════════════════════════


@pytest.fixture(scope="module")
def rollout_rows() -> list[dict]:
    """800 条真实 rollout 逐条的 (Σadv, 被 mask 的 token 数, 被 mask 的字符区间, tool_ok)。"""
    if not (SAMPLES.is_file() and SUBSET.is_file() and TOKENIZER.is_dir()):
        pytest.skip("本机缺少真实 rollout 数据或 tokenizer（.local 下），跳过数值复现")
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(TOKENIZER))
    targets_by_id = {
        row["id"]: row["targets"]
        for row in (json.loads(line) for line in SUBSET.open(encoding="utf-8") if line.strip())
    }
    groups: dict[str, list[dict]] = defaultdict(list)
    for sample in json.load(SAMPLES.open(encoding="utf-8"))["samples"]:
        groups[sample["sample_id"]].append(sample)

    rows: list[dict] = []
    for sample_id, group in groups.items():
        group = sorted(group, key=lambda sample: sample["index"])
        targets = targets_by_id[sample_id]
        texts = [sample["completion"] for sample in group]
        annotations = [
            annotate(AnnotationInput(text, targets, None, "tool_call")) for text in texts
        ]
        ragged = compute_group_advantages(
            completions=texts,
            annotations=annotations,
            targets=targets,
            tokenizer=tokenizer,
            format_type="tool_call",
            numeric_tolerance=0.2,
        )
        for sample, text, annotation, advantage in zip(
            group, texts, annotations, ragged, strict=True
        ):
            tags = apply_tail_eos(text, annotation.tags, "tool_call")
            labels = char_to_token_labels(text, tags, annotation.fields, tokenizer)
            offsets = tokenizer(text, return_offsets_mapping=True)["offset_mapping"]
            # P3 之前 S 恒为 +1.0 ⇒ 「标签是 S 且 advantage 为 0」只可能是被 mask 的
            assert len(labels) == len(advantage), "token 数与 advantage 长度必须对齐"
            masked_chars: list[int] = []
            for label, value in zip(labels, advantage, strict=True):
                if label.tag is CharTag.STRUCTURE and value == 0.0:
                    start, end = offsets[label.token_index]
                    masked_chars.extend(range(start, end))
                elif label.tag is CharTag.STRUCTURE:
                    assert value == 1.0, f"S token 只能是 0（全对）或 +1.0，实测 {value}"
                elif label.tag is CharTag.ERROR:
                    assert value == -1.0, f"E token 必须 −1.0，实测 {value}"
            rows.append(
                {
                    "masked_count": sum(
                        1
                        for label, value in zip(labels, advantage, strict=True)
                        if label.tag is CharTag.STRUCTURE and value == 0.0
                    ),
                    "masked_chars": sorted(masked_chars),
                    "adv_sum": sum(advantage),
                    "tool_ok": sample["pred_tool"] == sample["gt_tool"],
                }
            )
    return rows


def test_each_completion_loses_exactly_the_common_prefix(rollout_rows: list[dict]) -> None:
    """800 条各恰好 5 个 token 被置 0，合计 4000；被置 0 的字符集恰是 22 字符的公共前缀。"""
    assert len(rollout_rows) == 800
    assert {row["masked_count"] for row in rollout_rows} == {5}, "每条必须恰好 5 个 token 被置 0"
    assert sum(row["masked_count"] for row in rollout_rows) == 4000
    expected_chars = list(range(COMMON_PREFIX_CHARS))
    assert all(row["masked_chars"] == expected_chars for row in rollout_rows), (
        "被置 0 的字符必须恰是 `<tool_call>\\n<function=`（前 22 个字符）"
    )


def test_wrong_tool_name_rollouts_go_from_plus_three_to_minus_two(
    rollout_rows: list[dict],
) -> None:
    """工具名写错的 496 条：Σadv **+3.00 → −2.00**（P3 之前那 +5 全部来自公共前缀的 +1）。"""
    wrong = [row for row in rollout_rows if not row["tool_ok"]]
    assert len(wrong) == 496
    before = sum(row["adv_sum"] + row["masked_count"] for row in wrong)
    after = sum(row["adv_sum"] for row in wrong)
    assert before / len(wrong) == pytest.approx(3.0), "改前：这 496 条的 Σadv 均值"
    assert after / len(wrong) == pytest.approx(-2.0), "改后：这 496 条的 Σadv 均值"


def test_positive_advantage_count_drops_from_800_to_304(rollout_rows: list[dict]) -> None:
    """Σadv > 0 的条数 **800 → 304**（304 = 工具名全对的条数）。"""
    positive_after = sum(1 for row in rollout_rows if row["adv_sum"] > 0)
    positive_before = sum(1 for row in rollout_rows if row["adv_sum"] + row["masked_count"] > 0)
    assert positive_before == 800
    assert positive_after == 304
    assert sum(1 for row in rollout_rows if row["tool_ok"]) == 304


# ══════════════════════════════════════════════════════════════════════════════
# 3. B1 阻断回归：工具名是**值**，P3 的结构 mask 不得伸进工具名
# ══════════════════════════════════════════════════════════════════════════════
#
# 划分口径（负责人定的）：**写错 = 结构**（JSON/XML 语法、括号、引号、字段名拼写、参数名）
# ｜**选错 = 值**（**工具名**、参数值）。因此「组内全对的结构位置」这一集合里不可能有
# 工具名字符——工具名对错由值域评分表达（+1.0 / −1.0），与结构 mask 无关。
#
# 修复前缺陷：工具名**写对**时被逐字符标 S（``tool_call_labeler`` 的 P1 只改了「写错」那支），
# 于是当一组里的 rollout 工具名全写对、或组内最短那条的名字前缀较长时，``k`` 会伸进工具名
# 内部，把**写对的工具名**也 mask 成 0.0。本批 100 组数据里没有一组工具名全对，所以实测影响
# 为 0，但这是潜伏越界（下一条用例就是审查者给的反例）。

#: `<tool_call>`(11) + `\n`(1) + `<function=`(10) = 22：工具名前最后一个结构字符 + 1
NAME_START = len("<tool_call>") + 1 + len("<function=")


def test_tool_name_written_right_is_not_masked_even_with_a_long_name_prefix_peer() -> None:
    """★ B1 反例：组 = [工具名全对, 工具名写成 ``robot_atomic_controX``]。

    修复前：好那条的工具名 20 个字符全是 S，坏那条的 S 只有 ``22 + 18``（名字公共前缀）
    ⇒ ``k = 40`` ⇒ 好那条的**工具名前 18 个字符**落在「全对前缀」里 ⇒ advantage 被置 0.0。
    修复后：好那条的工具名是 ``VALUE(__tool_name__)`` ⇒ 不计入 S 字符数 ⇒ 不进 mask，
    仍是 ``+1.0``（名字写对给正分，但走值域，不走结构域）。
    """
    good = ROBOT.format("逆时针旋转", "16.9")
    typo = good.replace("robot_atomic_control", "robot_atomic_controX")
    name_end = NAME_START + len("robot_atomic_control")
    annotations = [_annotate(good), _annotate(typo)]

    advice = compute_group_advantages(
        completions=[good, typo],
        annotations=annotations,
        targets=TARGETS,
        tokenizer=FakeTokenizer(),
        format_type="tool_call",
    )
    # FakeTokenizer 下字符下标 == token 下标
    name_advantages = [advice[0][position] for position in range(NAME_START, name_end)]
    assert name_advantages == [1.0] * len(name_advantages), (
        f"工具名写对 ⇒ 每个字符 +1.0，不得被 P3 的结构 mask 吃掉；实测 {name_advantages}"
    )
    masked = {position for position, value in enumerate(advice[0]) if value == 0.0}
    assert not masked & set(range(NAME_START, name_end)), (
        f"被 mask 的字符不得落在工具名内（masked={sorted(masked)}）"
    )
    # 对照：写错那条**写错的字符**仍是值域 −1.0（P1 口径不变）
    assert advice[1][name_end - 1] == -1.0, "`robot_atomic_controX` 的 X 是值错 −1.0"


def test_all_names_correct_group_keeps_the_name_positive() -> None:
    """组内 8 条工具名**全对**（本批真实数据里不存在的形态）：名字仍 +1.0，不进 mask。

    这是 B1 的一般形态——k 由全对组自己的 S 字符数决定（= 整条结构字符数），修复前会把
    名字段一起吞掉。结构字符本身照 P3 规则全置 0（组内全对），名字**不在**该集合里。
    """
    completion = ROBOT.format("逆时针旋转", "16.9")
    advice = compute_group_advantages(
        completions=[completion] * 8,
        annotations=[_annotate(completion) for _ in range(8)],
        targets=TARGETS,
        tokenizer=FakeTokenizer(),
        format_type="tool_call",
    )[0]
    name_end = NAME_START + len("robot_atomic_control")
    masked = {position for position, value in enumerate(advice) if value == 0.0}
    assert not masked & set(range(NAME_START, name_end)), (
        f"工具名不得被 mask（masked 前缀 {sorted(masked)[:30]}…）"
    )
    assert [advice[position] for position in range(NAME_START, name_end)] == [1.0] * (
        name_end - NAME_START
    )
    # 结构字符（含 `<function=`(10) + 名字后的 `>`）仍是「组内全对 ⇒ 0」
    assert advice[COMMON_PREFIX_CHARS - 1] == 0.0, "<function= 的最后一个字符是结构"
    assert advice[name_end] == 0.0, "名字后的 > 是结构"
