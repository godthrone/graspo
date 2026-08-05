"""GRASPO-Ripple: Token-level reward computation.

Ripple replaces completion-level GRPO rewards with per-token reward vectors.
Format tokens (tool-call XML markers) receive 0.0 for clean completions
(format already correct — no gradient needed) and -1.0 for broken completions
(format wrong — focus on format correction).  Content tokens receive
field-level advantages: per-field scores are z-scored across clean completions
and distributed to the corresponding tokens, so content quality is compared
at the semantic field level rather than raw token position.

Two-path logic per completion:
1. Format correct (parsed successfully) → format tokens = 0.0, content tokens = field scores
2. Format broken → format tokens = -1.0, content tokens = 0.0 (untrustworthy)
"""

import logging
from typing import Any

from graspo.ripple.parsing.completion import ParsedCompletion
from graspo.ripple.parsing.xml import format_xml_param_value
from graspo.ripple.reward.compare import dict_compare_score, leaf_compare_score

_log = logging.getLogger(__name__)


# ── Public API ────────────────────────────────────────────────────────────────────


def compute_token_rewards(
    generated_token_ids: list[int],
    completion_text: str,
    parsed: ParsedCompletion,
    targets: list[dict[str, Any]],
    tokenizer: Any,
    *,
    check_list_order: bool = False,
    numeric_tolerance: float = 0.2,
) -> tuple[list[float], list[bool], list[str | None]]:
    """Compute per-token rewards with format/content labels.

    Args:
        generated_token_ids: Token IDs of the generated portion (after prompt).
        completion_text: Decoded text of the generated portion.
        parsed: Parsed completion from tool-call parser.
        targets: Normalized target dicts (from ``normalize_targets``).
        tokenizer: HuggingFace tokenizer with ``encode(return_offsets_mapping=True)``.
        check_list_order: Whether list order matters in dict comparison.
        numeric_tolerance: Relative error dead zone for numeric field scoring.

    Returns:
        Tuple of (rewards, is_format, field_keys) — three parallel lists of
        equal length (== len(generated_token_ids)).

        - *rewards*: per-token float in [0, 1].
        - *is_format*: True for format tokens, False for content tokens.
        - *field_keys*: the field key (e.g. "fn.param") for content tokens,
          ``None`` for format tokens.
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
        return _rewards_format_broken(generated_token_ids, targets, tokenizer)


# ── Format validity ────────────────────────────────────────────────────────────────


def _is_format_valid(parsed: ParsedCompletion, targets: list[dict[str, Any]]) -> bool:
    """Check if the completion format is valid for content scoring.

    Format is valid when: parser succeeded, tool calls were extracted, and
    the number of tool calls is not excessive compared to targets.
    """
    if parsed.parse_errors:
        return False
    if not parsed.tool_calls:
        return False
    max_tc = max((len(t.get("output", {}).get("tool_calls", [])) for t in targets), default=0)
    if len(parsed.tool_calls) > max_tc:
        return False
    return True


# ── Path 1: Format correct ─────────────────────────────────────────────────────────


def _rewards_format_correct(
    generated_token_ids: list[int],
    completion_text: str,
    parsed: ParsedCompletion,
    targets: list[dict[str, Any]],
    tokenizer: Any,
    *,
    check_list_order: bool = False,
    numeric_tolerance: float = 0.2,
) -> tuple[list[float], list[bool], list[str | None]]:
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
    is_format = [True] * num_generated
    field_keys: list[str | None] = [None] * num_generated
    tok_idx = 0  # index into encoding tokens

    for gen_idx in range(num_generated):
        if tok_idx >= len(offsets):
            # Completion longer than what offsets cover
            rewards[gen_idx] = 0.0
            is_format[gen_idx] = True
            field_keys[gen_idx] = None
            continue

        char_start, char_end = offsets[tok_idx]
        field_name = _span_to_field(char_start, char_end, content_spans)

        if field_name is not None:
            # Content token: use field score
            rewards[gen_idx] = field_scores.get(field_name, 0.0)
            is_format[gen_idx] = False
            field_keys[gen_idx] = field_name
        else:
            # Format token: correct format = 1.0
            rewards[gen_idx] = 1.0
            is_format[gen_idx] = True
            field_keys[gen_idx] = None

        tok_idx += 1

    # Remaining generated positions (beyond encoding length) → 0, format
    return rewards, is_format, field_keys


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
    pattern = re.compile(rf"<function={re.escape(fn_name)}>(.*?)</function>", re.DOTALL)
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
    from graspo.ripple.reward.compare import CompareResult

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
                        score = leaf_compare_score(
                            pvalue, tgt_value, numeric_tolerance=numeric_tolerance
                        )
                        best_scores[field_key] = float(score)
                    else:
                        best_scores[field_key] = 0.0

    return best_scores


# ── Path 2: Format broken ──────────────────────────────────────────────────────────


def _rewards_format_broken(
    generated_token_ids: list[int],
    targets: list[dict[str, Any]],
    tokenizer: Any,
) -> tuple[list[float], list[bool], list[str | None]]:
    """Format is broken.  Compare token-by-token with ground truth format.

    The ground truth is reconstructed from the targets using the deterministic
    Qwen XML tool-call format.  Each token position gets 1.0 for match, 0.0 for
    mismatch.  Extra generated tokens get 0.0.

    All tokens are marked as format tokens — there is no meaningful content
    extraction from a broken-format completion.
    """
    gt_ids = _build_ground_truth_token_ids(targets, tokenizer)
    num_gen = len(generated_token_ids)
    rewards = [
        1.0 if i < len(gt_ids) and gen_id == gt_ids[i] else 0.0
        for i, gen_id in enumerate(generated_token_ids)
    ]
    is_format = [True] * num_gen
    field_keys: list[str | None] = [None] * num_gen
    return rewards, is_format, field_keys


def _build_ground_truth_token_ids(targets: list[dict[str, Any]], tokenizer: Any) -> list[int]:
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
            parts.append(format_xml_param_value(pvalue))
            parts.append("</parameter>")
        parts.append("</function>")
        parts.append("</tool_call>")

    gt_text = "\n".join(parts)
    return tokenizer.encode(gt_text, add_special_tokens=False)


# ── Token-level advantage ──────────────────────────────────────────────────────────


def compute_token_advantages(
    token_rewards: list[list[float]],
    is_format_masks: list[list[bool]],
    field_keys: list[list[str | None]],
    *,
    eps: float = 1e-8,
) -> list[list[float]]:
    """Compute token-level advantages with format/content separation (v0.19.0).

    **Format tokens** for clean completions receive 0.0 — format is already
    correct, no gradient needed.  Broken completions receive -1.0 on format
    tokens to focus learning on format correction.

    **Content tokens** for clean completions receive ``cs - mean(cs)`` — the
    deviation from the group mean.  This is a simple, scale-honest signal:
    each 0.01 of content_score difference produces exactly 0.01 of advantage,
    without artificial amplification from std normalization.  Broken
    completions receive 0.0 on content tokens (untrustworthy).

    Prior to v0.19.0, content advantage was ``(cs - mean) / std`` (z-score),
    which amplified small differences by up to 12× when within-group variance
    was low.  The raw difference preserves the correct direction while keeping
    per-token gradient magnitudes honest and comparable across groups.

    Args:
        token_rewards: Ragged per-completion reward lists.
        is_format_masks: Parallel ragged list; True for format tokens.
        field_keys: Parallel ragged list; field key string for content tokens,
                    ``None`` for format tokens.
        eps: Small constant for numerical stability.

    Returns:
        List of advantage lists, same ragged shape as input.
    """
    if not token_rewards:
        return []

    group_size = len(token_rewards)
    advantages: list[list[float]] = [[] for _ in range(group_size)]

    # --- Step 1: compute field-level advantages for content tokens ---

    # Determine which completions are format-clean.
    # A completion is clean if it passed format validation: either it has
    # content tokens (is_format=False) from successful XML parsing, or all
    # its tokens are format tokens with high reward (the entire output is
    # format tokens and they're correct).
    is_clean: list[bool] = []
    for i in range(group_size):
        num_tokens = min(len(is_format_masks[i]), len(token_rewards[i]))
        if num_tokens == 0:
            is_clean.append(False)
            continue
        has_content = any(not is_format_masks[i][t] for t in range(num_tokens))
        if has_content:
            is_clean.append(True)
        else:
            # All format tokens — check if they're correct (reward > 0.5).
            clean_format_count = sum(
                1 for t in range(num_tokens) if is_format_masks[i][t] and token_rewards[i][t] > 0.5
            )
            is_clean.append(clean_format_count >= num_tokens * 0.5)

    # Collect per-field scores from clean completions.
    field_scores_by_key: dict[str, list[tuple[int, float]]] = {}
    for i in range(group_size):
        if not is_clean[i]:
            continue
        seen: set[str] = set()
        for t in range(min(len(token_rewards[i]), len(field_keys[i]))):
            fk = field_keys[i][t]
            if fk is not None and fk not in seen:
                seen.add(fk)
                field_scores_by_key.setdefault(fk, []).append((i, token_rewards[i][t]))

    # Compute per-field raw difference across clean completions.
    # cs - mean(cs): honest scale, correct direction, no artificial amplification.
    field_adv: dict[str, dict[int, float]] = {}  # field_key → {completion_idx: advantage}
    for fk, entries in field_scores_by_key.items():
        scores = [e[1] for e in entries]
        indices = [e[0] for e in entries]
        if len(scores) >= 2:
            mean = sum(scores) / len(scores)
        elif len(scores) == 1:
            mean = scores[0]
        else:
            mean = 0.0
        field_adv[fk] = {}
        for idx, s in zip(indices, scores):
            field_adv[fk][idx] = s - mean

    # --- Step 2: assign advantages per token ---

    # Determine which positions are format-token positions.
    # A token position t is a "format position" if ANY clean completion
    # marks it as a format token.
    max_len_any = max(len(is_format_masks[i]) for i in range(group_size))
    is_format_pos = [False] * max_len_any
    for t in range(max_len_any):
        for i in range(group_size):
            if t < len(is_format_masks[i]) and is_clean[i]:
                if is_format_masks[i][t]:
                    is_format_pos[t] = True
                    break

    for i in range(group_size):
        for t in range(len(token_rewards[i])):
            is_fmt = is_format_masks[i][t]
            fk = field_keys[i][t] if t < len(field_keys[i]) else None

            if is_format_pos[t] and t < len(is_format_pos):
                # Format token position: clean=0.0 (already correct, no gradient),
                # broken=-1.0 (focus on format correction)
                advantages[i].append(0.0 if is_clean[i] else -1.0)
            elif fk is not None and not is_fmt:
                # Content token: look up field-level advantage
                if is_clean[i] and fk in field_adv and i in field_adv[fk]:
                    advantages[i].append(field_adv[fk][i])
                else:
                    advantages[i].append(0.0)
            else:
                # Remaining tokens (e.g. broken at a content-only position)
                advantages[i].append(0.0)

    return advantages


# ── Group-level wrapper ──────────────────────────────────────────────────────────


def compute_token_rewards_for_group(
    *,
    generation: Any,
    parsed_completions: list[Any],
    targets: Any,
    tokenizer: Any,
    reward_config: Any,
) -> tuple[list[list[float]], list[list[bool]], list[list[str | None]]]:
    """为一个 rollout group 的每个 completion 计算 token 级 reward。

    Returns:
        (token_rewards, is_format_masks, field_keys) — three ragged lists.
    """
    from graspo.ripple.reward.reward import normalize_targets

    normalized_targets = normalize_targets(targets)
    token_rewards: list[list[float]] = []
    is_format_masks: list[list[bool]] = []
    field_keys: list[list[str | None]] = []
    prompt_len = int(generation.prompt_len)

    for idx, completion_text in enumerate(generation.completions):
        parsed = parsed_completions[idx]
        gen_ids = generation.sequences[idx, prompt_len:].tolist()
        tr, is_fmt, fk = compute_token_rewards(
            generated_token_ids=gen_ids,
            completion_text=completion_text,
            parsed=parsed,
            targets=normalized_targets,
            tokenizer=tokenizer,
            check_list_order=bool(reward_config.check_list_order),
            numeric_tolerance=float(reward_config.numeric_tolerance),
        )
        token_rewards.append(tr)
        is_format_masks.append(is_fmt)
        field_keys.append(fk)

    return token_rewards, is_format_masks, field_keys
