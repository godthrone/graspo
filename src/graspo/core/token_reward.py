"""GRASPO-Ripple: Token-level reward computation.

Ripple replaces completion-level GRPO rewards with per-token reward vectors.
This ensures format tokens (tool-call XML markers, think tags) naturally
have zero advantage when all completions in a group share the same format,
preventing the gradual gradient drift that causes catastrophic forgetting.

Two-path logic per completion:
1. Format correct (parsed successfully) → format tokens = 1.0, content tokens = field scores
2. Format broken → token-by-token comparison with ground truth format
"""

from __future__ import annotations

import logging
from typing import Any

from graspo.core.compare import dict_compare_score, leaf_compare_score
from graspo.core.completion import ParsedCompletion
from graspo.core.data import _format_xml_param_value

_log = logging.getLogger(__name__)


def compute_token_rewards(
    generated_token_ids: list[int],
    completion_text: str,
    parsed: ParsedCompletion,
    targets: list[dict[str, Any]],
    tokenizer: Any,
    *,
    check_list_order: bool = False,
    numeric_tolerance: float = 0.2,
) -> list[float]:
    """Compute per-token rewards for a single completion.

    Args:
        generated_token_ids: Token IDs of the generated portion (after prompt).
        completion_text: Decoded text of the generated portion.
        parsed: Parsed completion from tool-call parser.
        targets: Normalized target dicts (from ``normalize_targets``).
        tokenizer: HuggingFace tokenizer with ``encode(return_offsets_mapping=True)``.
        check_list_order: Whether list order matters in dict comparison.
        numeric_tolerance: Relative error dead zone for numeric field scoring.

    Returns:
        List of floats, one per generated token.  Length == len(generated_token_ids).
    """
    format_valid = _is_format_valid(parsed, targets)

    if format_valid:
        return _rewards_format_correct(
            generated_token_ids,
            completion_text,
            parsed,
            targets,
            tokenizer,
            check_list_order=check_list_order,
            numeric_tolerance=numeric_tolerance,
        )
    else:
        return _rewards_format_broken(
            generated_token_ids, targets, tokenizer
        )


# ── Format validity ────────────────────────────────────────────────────────────


def _is_format_valid(parsed: ParsedCompletion, targets: list[dict[str, Any]]) -> bool:
    """Check if the completion format is valid for content scoring.

    Format is valid when: parser succeeded, tool calls were extracted, and
    the number of tool calls is not excessive compared to targets.
    """
    if parsed.parse_errors:
        return False
    if not parsed.tool_calls:
        return False
    max_tc = max(
        (len(t.get("output", {}).get("tool_calls", [])) for t in targets), default=0
    )
    if len(parsed.tool_calls) > max_tc:
        return False
    return True


# ── Path 1: Format correct ─────────────────────────────────────────────────────


def _rewards_format_correct(
    generated_token_ids: list[int],
    completion_text: str,
    parsed: ParsedCompletion,
    targets: list[dict[str, Any]],
    tokenizer: Any,
    *,
    check_list_order: bool = False,
    numeric_tolerance: float = 0.2,
) -> list[float]:
    """Format is correct.  Format tokens = 1.0, content tokens = field scores.

    Uses tokenizer offset mapping to identify which token positions correspond
    to content (parameter values).  All other positions are format tokens.
    """
    encoding = tokenizer(completion_text, return_offsets_mapping=True)
    offsets: list[tuple[int, int]] = encoding.offset_mapping  # type: ignore[assignment]

    content_spans = _identify_content_spans(parsed, completion_text, targets)
    field_scores = _compute_field_scores(
        parsed, targets, check_list_order=check_list_order, numeric_tolerance=numeric_tolerance
    )

    num_generated = len(generated_token_ids)
    rewards = [0.0] * num_generated
    tok_idx = 0  # index into encoding tokens

    for gen_idx in range(num_generated):
        if tok_idx >= len(offsets):
            # Completion longer than what offsets cover
            rewards[gen_idx] = 0.0
            continue

        char_start, char_end = offsets[tok_idx]
        field_name = _span_to_field(char_start, char_end, content_spans)

        if field_name is not None:
            # Content token: use field score
            rewards[gen_idx] = field_scores.get(field_name, 0.0)
        else:
            # Format token: correct format = 1.0
            rewards[gen_idx] = 1.0

        tok_idx += 1

    # Remaining generated positions (beyond encoding length) → 0
    return rewards


def _identify_content_spans(
    parsed: ParsedCompletion,
    completion_text: str,
    targets: list[dict[str, Any]],
) -> list[tuple[int, int, str]]:
    """Identify (char_start, char_end, field_name) spans for parameter values.

    For tool-call completions, parses the Qwen XML format to find parameter
    value spans.  Returns a list of (start, end, field_key) tuples.
    """
    spans: list[tuple[int, int, str]] = []

    # For each parsed tool call, find the matching target and identify
    # parameter value spans in the completion text.
    for tc in parsed.tool_calls:
        _find_tool_call_value_spans(tc, completion_text, targets, spans)

    return spans


def _find_tool_call_value_spans(
    tc: dict[str, Any],
    completion_text: str,
    targets: list[dict[str, Any]],
    spans: list[tuple[int, int, str]],
) -> None:
    """Find parameter value character spans for a single tool call."""
    import re

    fn_name = tc.get("name", "")
    fn_args = tc.get("arguments", {}) if isinstance(tc.get("arguments"), dict) else {}

    # Find the tool_call block for this function name in the completion text
    pattern = re.compile(
        rf"<function={re.escape(fn_name)}>(.*?)</function>", re.DOTALL
    )
    for match in pattern.finditer(completion_text):
        body = match.group(1)
        body_start = match.start(1)
        # For each parameter in the arguments, find its value span
        for pname, pvalue in fn_args.items():
            param_pattern = re.compile(
                rf"<parameter={re.escape(pname)}>\s*(.*?)\s*</parameter>", re.DOTALL
            )
            param_match = param_pattern.search(body)
            if param_match:
                value_text = param_match.group(1)
                value_start = body_start + param_match.start(1)
                value_end = value_start + len(value_text)
                field_key = f"{fn_name}.{pname}"
                spans.append((value_start, value_end, field_key))


def _span_to_field(
    char_start: int, char_end: int, content_spans: list[tuple[int, int, str]]
) -> str | None:
    """Return the field name if this character span falls within a content span."""
    for cs_start, cs_end, field_name in content_spans:
        if char_start >= cs_start and char_end <= cs_end:
            return field_name
        # Also check partial overlap
        if char_start < cs_end and char_end > cs_start:
            return field_name
    return None


def _compute_field_scores(
    parsed: ParsedCompletion,
    targets: list[dict[str, Any]],
    *,
    check_list_order: bool = False,
    numeric_tolerance: float = 0.2,
) -> dict[str, float]:
    """Compute field-level scores by matching parsed tool calls against targets.

    Returns a dict mapping field_key (e.g. "robot_atomic_control.action_type") → score.
    """
    from graspo.core.compare import CompareResult

    if not parsed.tool_calls or not targets:
        return {}

    # Use the existing dict_compare_score for content scoring.
    # Wrap parsed tool calls into the same format as targets expect.
    checked = {"tool_calls": parsed.tool_calls}
    best_scores: dict[str, float] = {}
    best_dcs = -1.0
    best_target_idx = -1

    for idx, target in enumerate(targets):
        calls = target.get("output", {}).get("tool_calls")
        if not isinstance(calls, list):
            continue
        result: CompareResult = dict_compare_score(
            checked=checked,
            target={"tool_calls": calls},
            check_list_order=check_list_order,
            numeric_tolerance=numeric_tolerance,
        )
        if result.dcs > best_dcs:
            best_dcs = result.dcs
            best_target_idx = idx

    # For the best-matching target, assign per-field scores.
    if best_target_idx >= 0 and best_target_idx < len(targets):
        best_target = targets[best_target_idx]
        tgt_calls = best_target.get("output", {}).get("tool_calls", [])
        for ptc in parsed.tool_calls:
            fn_name = ptc.get("name", "")
            ptc_args = ptc.get("arguments", {}) if isinstance(ptc.get("arguments"), dict) else {}
            for tgt_tc in tgt_calls:
                if tgt_tc.get("name") != fn_name:
                    continue
                tgt_args = tgt_tc.get("arguments", {})
                for pname, pvalue in ptc_args.items():
                    tgt_value = tgt_args.get(pname)
                    field_key = f"{fn_name}.{pname}"
                    if tgt_value is not None:
                        score = leaf_compare_score(pvalue, tgt_value, numeric_tolerance=numeric_tolerance)
                        best_scores[field_key] = float(score)
                    else:
                        best_scores[field_key] = 0.0

    return best_scores


# ── Path 2: Format broken ──────────────────────────────────────────────────────


def _rewards_format_broken(
    generated_token_ids: list[int],
    targets: list[dict[str, Any]],
    tokenizer: Any,
) -> list[float]:
    """Format is broken.  Compare token-by-token with ground truth format.

    The ground truth is reconstructed from the targets using the deterministic
    Qwen XML tool-call format.  Each token position gets 1.0 for match, 0.0 for
    mismatch.  Extra generated tokens get 0.0.
    """
    gt_ids = _build_ground_truth_token_ids(targets, tokenizer)
    return [
        1.0 if i < len(gt_ids) and gen_id == gt_ids[i] else 0.0
        for i, gen_id in enumerate(generated_token_ids)
    ]


def _build_ground_truth_token_ids(
    targets: list[dict[str, Any]], tokenizer: Any
) -> list[int]:
    """Reconstruct the ground truth token sequence from targets.

    Builds the deterministic Qwen XML format from the first target's tool calls.
    """
    if not targets:
        return []

    # Use the first target with tool calls
    target = targets[0]
    calls = target.get("output", {}).get("tool_calls")
    if not isinstance(calls, list) or not calls:
        return []

    parts: list[str] = []
    for tc in calls:
        fn_name = tc.get("name", "")
        fn_args = tc.get("arguments", {}) if isinstance(tc.get("arguments"), dict) else {}
        parts.append("<tool_call>")
        parts.append(f"<function={fn_name}>")
        for pname, pvalue in fn_args.items():
            parts.append(f"<parameter={pname}>")
            parts.append(_format_xml_param_value(pvalue))
            parts.append(f"</parameter>")
        parts.append("</function>")
        parts.append("</tool_call>")

    gt_text = "\n".join(parts)
    return tokenizer.encode(gt_text, add_special_tokens=False)


# ── Token-level advantage (group normalization) ─────────────────────────────────


def compute_token_advantages(
    token_rewards: "list[list[float]]",  # (B, L) — ragged or padded
    *,
    eps: float = 1e-8,
) -> "list[list[float]]":
    """Compute token-level GRPO advantages from per-token rewards.

    For each token position t, computes:
        mean[t] = mean of rewards at position t across the group
        std[t]  = std of rewards at position t across the group
        raw_i[t] = (r_i[t] - mean[t]) / (std[t] + eps)

    Advantages are symmetrically clamped: negative advantage magnitude cannot
    exceed the maximum positive advantage in the group.  This prevents the
    v0.11.0 bug (broken completions with 7:1 negative advantage) while
    preserving corrective gradient signal for format errors that v0.11.1's
    non-negative clamping threw away.

    Positions beyond a completion's length are excluded from mean/std
    computation.  For positions that exist in some completions but not
    others, only the existing completions participate in the statistics.

    Args:
        token_rewards: List of per-completion reward lists.  Each inner list
                       may have different length (ragged).
        eps: Small constant for numerical stability.

    Returns:
        List of advantage lists, same ragged shape as input.
    """
    if not token_rewards:
        return []

    # Find max length
    max_len = max(len(r) for r in token_rewards)
    group_size = len(token_rewards)

    advantages: list[list[float]] = [[] for _ in range(group_size)]

    for t in range(max_len):
        # Collect rewards at this position across all completions that have it
        pos_rewards = [
            token_rewards[i][t]
            for i in range(group_size)
            if t < len(token_rewards[i])
        ]
        if len(pos_rewards) < 2:
            # Not enough data for meaningful statistics
            mean = pos_rewards[0] if pos_rewards else 0.0
            std = 0.0
        else:
            mean = sum(pos_rewards) / len(pos_rewards)
            variance = sum((r - mean) ** 2 for r in pos_rewards) / (len(pos_rewards) - 1)
            std = variance ** 0.5

        # Compute raw advantages for this position, then clamp symmetrically:
        # negative advantage magnitude cannot exceed the max positive advantage.
        # This prevents a few broken completions from dominating the gradient
        # (v0.11.0 bug) while preserving corrective signal for format errors
        # (v0.11.1 shortcoming).
        raw = []
        for i in range(group_size):
            if t < len(token_rewards[i]):
                raw.append((token_rewards[i][t] - mean) / (std + eps) if (std + eps) > 0 else 0.0)
            else:
                raw.append(0.0)

        max_pos = max((a for a in raw if a > 0), default=0.0)
        clamped = [max(a, -max_pos) if max_pos > 0 else 0.0 for a in raw]
        for i in range(group_size):
            advantages[i].append(clamped[i])

    return advantages
