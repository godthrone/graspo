"""阻断 A 的负向用例（ms-swift 侧 GRASPO 工具调用奖励路径）。

判据：同一个「内容完全正确」的 tool-call completion，在 ms-swift 奖励适配器上
必须拿到非零 reward，且一个组内「对/错」两条 completion 必须产生 reward 方差
（`has_reward_variance` 为真）。

修复前：`GraspoMsSwiftReward.__call__` 无条件走 `GraspoReward.score()`（文本路径，
只认 `targets[*].output.content`）⇒ tool-call 目标下 `check_targets` 为空 ⇒
`_max_reward([]) == 0.0` ⇒ `normalized_reward` 恒 0.0 ⇒ 全组 reward 方差为 0
⇒ `classify_group` 判 RETRY → INVALID ⇒ 零 advantage ⇒ 零梯度。

本文件不需要 GPU / ms-swift / torch，直接走仓内真实代码路径。
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from graspo.flow.msswift.reward import GraspoMsSwiftReward
from graspo.ripple.group_decision import classify_group, has_reward_variance

# ── ELAM V5 真实 targets 形态：只有 tool_calls，没有 content ────────────────

TOOL_CALL_TARGETS = [
    {
        "id": "primary",
        "output": {
            "tool_calls": [{"name": "extend_arm", "arguments": {"action_type": "伸长手臂"}}]
        },
    }
]

# 跨模型标准 JSON tool call 形态（ms-swift 采样出的模型输出）
CORRECT_COMPLETION = (
    "<tool_call>\n"
    '{"name": "extend_arm", "arguments": {"action_type": "伸长手臂"}}\n'
    "</tool_call>"
)
WRONG_COMPLETION = (
    "<tool_call>\n"
    '{"name": "rotate_arm", "arguments": {"action_type": "顺时针旋转"}}\n'
    "</tool_call>"
)
PROSE_COMPLETION = "我需要把手臂伸长一点。"

_TARGETS_COLUMN = [json.dumps(TOOL_CALL_TARGETS, ensure_ascii=False)]


def _score(completions: list[str], targets: list[Any] | None = None) -> list[float]:
    orm = GraspoMsSwiftReward(args=None)
    column = _TARGETS_COLUMN * len(completions) if targets is None else targets
    return list(orm(completions, targets=column))


# ── 用例 1：正确 tool call 必须非零（修复前 = 0.0，本用例真失败）────────────


def test_correct_tool_call_completion_gets_nonzero_reward() -> None:
    scores = _score([CORRECT_COMPLETION])
    assert scores[0] > 0.0, (
        f"内容完全正确的 tool-call completion 拿到 reward={scores[0]!r}；"
        "tool-call 档恒零奖励 ⇒ 训练信号恒为零（这正是 T031 的实测现象）"
    )


# ── 用例 2：组内对/错必须产生 reward 方差（修复前 = 无方差）─────────────────


def test_group_reward_variance_is_nonzero() -> None:
    group = [CORRECT_COMPLETION] * 4 + [PROSE_COMPLETION] * 4
    scores = _score(group)
    assert has_reward_variance(scores), (
        f"组内 reward 无方差：{scores} ⇒ classify_group 必然判 RETRY/INVALID，"
        "整组零 advantage"
    )


# ── 用例 3：正确 vs 错误动作，reward 必须能区分 ────────────────────────────


def test_correct_beats_wrong_action() -> None:
    correct, wrong = _score([CORRECT_COMPLETION, WRONG_COMPLETION])
    assert correct > wrong, f"正确动作 reward={correct!r} 未高于错误动作 reward={wrong!r}"


# ── 用例 4：端到端判据 —— 可训练组必须被判为 should_train ───────────────────


def test_correct_groups_are_classified_as_trainable() -> None:
    group = [CORRECT_COMPLETION] * 4 + [PROSE_COMPLETION] * 4
    scores = _score(group)
    # content_scores 口径与 msswift trainer 一致：有结构标注 = 1.0
    content_scores = [1.0 if "<tool_call>" in text else 0.0 for text in
                      [CORRECT_COMPLETION] * 4 + [PROSE_COMPLETION] * 4]
    decision = classify_group(
        scores,
        content_scores,
        retry_count=0,
        rollout_max_retries=3,
        perfect_skip_reward_threshold=1.0,
        best_completion_has_parse_error=False,
        reject_unparseable_groups=True,
    )
    assert decision.should_train, (
        f"含正确 tool-call 的组被判为 {decision.decision.value}（rewards={scores}）"
        "⇒ 不会注入 advantage，零梯度"
    )


# ── 用例 5：纯文本（content）档语义不变 —— 回归保护 ────────────────────────


def test_content_targets_path_unchanged() -> None:
    content_targets = [{"id": "primary", "output": {"content": {"action_type": "伸长手臂"}}}]
    completion = "```json\n" + json.dumps({"action_type": "伸长手臂"}, ensure_ascii=False) + "\n```"
    scores = _score([completion], [json.dumps(content_targets, ensure_ascii=False)])
    assert scores[0] > 1.0, (
        f"content 档回归被破坏：reward={scores[0]!r}（修复前实测为 1.0043478260869565）"
    )


# ── 用例 6：缺 targets 列仍然 fail-closed（不许静默给 0 分）────────────────


def test_missing_targets_column_still_raises() -> None:
    orm = GraspoMsSwiftReward(args=None)
    with pytest.raises(ValueError, match="targets"):
        orm([CORRECT_COMPLETION])


# ── 用例 7：不可解析输出不得被静默判为可训练（防"为改而改"引入假信号）──────


def test_unparseable_group_not_trainable() -> None:
    scores = _score([PROSE_COMPLETION] * 8)
    content_scores = [0.0] * 8
    decision = classify_group(
        scores,
        content_scores,
        retry_count=0,
        rollout_max_retries=3,
        perfect_skip_reward_threshold=1.0,
        best_completion_has_parse_error=True,
        reject_unparseable_groups=True,
    )
    assert not decision.should_train, (
        f"不可解析组被判为 {decision.decision.value} ⇒ 引入噪声训练信号"
    )
