"""标注 → per-token advantage:字段级相对化 + 末尾 EOS 惩罚。

链路(方案 token-reward-advantage-design-v0.20.md §3):

1. 字符级标注 ``AnnotationResult``(annotate 输出,S/V/T/W/E/D)
2. 末尾 EOS 惩罚:结构不完整(截断/缺闭合/纯乱码 = "没写完")且非 max_len 硬截断
   → 末尾非空白字符标 E(-1.0),语义"在这里结束是错的"
3. token 级映射 ``char_to_token_labels``(offset_mapping,特殊 token 跳过)
4. 字段级 raw:V span 与 GT 比较(``leaf_compare_score`` / 数组存在性匹配),
   同一字段的所有 V token 共享字段分
5. ``μ_f`` = 组内 per-field 均值;``adv(V) = raw − μ_f``(n=1 安全零)
6. adv:``S→+1.0 / V→raw−μ_f / T·W·D→0 / E→−1.0``

纯函数、零设施依赖;tokenizer 只用于 offset_mapping。

## ★ rollout 分卡（2026-10-16）带来的**唯一**算法层改动

``μ_f`` 是组内**唯一**的跨条耦合量（V-span 提取 / ``field_score`` / 末尾 EOS / token 映射
全部**逐条闭合**）。rollout 分卡下每卡只持 ``G/m`` 条，故：

- 本模块新增 :func:`group_field_stats`（per-field ``(Σraw, n)`` 的**唯一**计算入口）、
  :func:`merge_field_stats`（按槽位序逐卡相加）、:func:`field_means`（``(Σ,n) → μ_f``）；
- :func:`compute_group_advantages` 新增**可选** ``field_mean_override``（默认 ``None``）。
  ``None`` ⇒ 走原式（**dp=1 路径逐位不变**）；给定 ⇒ **不使用本地 μ**，改用注入的整组 ``μ_f``。

⛔ **算法公式一字未改**：``adv(V) = raw − μ_f``、``S/V/T·W·D/E`` 四个取值、
``group_field_stats`` 里的 raw 打分与定义域判定，全部与改动前**同式**。
flow 层**不得**重写一遍 μ 的算术（那违反 §1.4 单一真相源）。
"""

from typing import Any

from graspo.ripple.annotation.char_tag import CharTag
from graspo.ripple.annotation.labeler import AnnotationResult
from graspo.ripple.annotation.tokenize import char_to_token_labels
from graspo.ripple.reward.compare import leaf_compare_score

_OPEN_TAG = "<tool_call>"
_CLOSE_TAG = "</tool_call>"
_FENCE = "```json"


def structure_incomplete(completion: str, format_type: str) -> bool:
    """结构是否不完整(截断/缺闭合/无结构)。

    - tool_call:未出现闭合标签(含纯乱码——没有结构就算"没写完")
    - json:去围栏后无法解析(有结构但未闭合)
    已有 E 定位的 completion 由调用方先行排除(不在此判定)。
    """
    if format_type == "tool_call":
        return _CLOSE_TAG not in completion
    import json

    body = completion
    if _FENCE in body:
        body = body.split(_FENCE, 1)[1]
        if "```" in body:
            body = body.split("```", 1)[0]
    try:
        json.loads(body)
        return False
    except (TypeError, ValueError, json.JSONDecodeError):
        return True


def apply_tail_eos(
    completion: str,
    tags: list[CharTag],
    format_type: str,
    *,
    truncated_by_max: bool = False,
) -> list[CharTag]:
    """末尾 EOS 惩罚:结构不完整 + 非 max_len 截断 → 末尾非空白字符标 E。

    返回新的 tags 列表(不改传入列表)。已有 E 定位或 completion 为空时不处理。
    """
    result = list(tags)
    if truncated_by_max:
        return result
    if not completion or not completion.strip():
        return result
    if CharTag.ERROR in result:
        return result
    if not structure_incomplete(completion, format_type):
        return result
    tail = len(completion.rstrip()) - 1
    result[tail] = CharTag.ERROR
    return result


def value_spans(
    completion: str, tags: list[CharTag], fields: list[str | None]
) -> list[tuple[int, int, str]]:
    """从字符级标注提取连续 V 段:(start, end, field)。

    同一 field 的连续 V 字符合并为一个 span(一个值可能跨多个 token);
    不同 field 的相邻 V 段不合并(中间隔结构字符)。
    """
    spans: list[tuple[int, int, str]] = []
    n = len(completion)
    i = 0
    while i < n:
        if tags[i] == CharTag.VALUE:
            j = i
            f = fields[i]
            while j < n and tags[j] == CharTag.VALUE and fields[j] == f:
                j += 1
            spans.append((i, j, f or ""))
            i = j
        else:
            i += 1
    return spans


def _parse_value(text: str) -> object:
    """V span 文本 → 值。

    JSON 字符串值(带引号)保持字符串——全数字字符串(如 IMSI)不能被 float 误转;
    无引号文本尝试转 float(XML 数值 / JSON 数字)。
    """
    t = text.strip()
    if len(t) >= 2 and t[0] == '"' and t[-1] == '"':
        return t[1:-1]
    try:
        return float(t)
    except ValueError:
        return t


def field_score(value_text: str, gt_val: object, numeric_tolerance: float) -> float:
    """字段级打分:整个 value span 与 GT 值比较。

    GT 为 list(JSON 数组)时:值文本与任一元素比较,命中即 1.0(存在性匹配)。
    标量:``leaf_compare_score``(数字容差 / 字符串精确)。
    """
    v = _parse_value(value_text)
    if isinstance(gt_val, list):
        for elem in gt_val:
            if isinstance(elem, (int, float)) and not isinstance(elem, bool):
                if (
                    isinstance(v, float)
                    and leaf_compare_score(v, elem, numeric_tolerance=numeric_tolerance) >= 1.0
                ):
                    return 1.0
            elif v == elem:
                return 1.0
        return 0.0
    return float(leaf_compare_score(v, gt_val, numeric_tolerance=numeric_tolerance))


def gt_value_for_field(targets: list[dict[str, Any]], field: str) -> object | None:
    """按 field 名从 targets 取 GT 值(field 形如 ``fn.param`` 或 json key)。"""
    for target in targets:
        out = target.get("output") or {}
        if "tool_calls" in out:
            for tc in out["tool_calls"]:
                args = tc.get("arguments") or {}
                if field in args:
                    return args[field]
        if "content" in out:
            if field in out["content"]:
                return out["content"][field]
    return None


def group_field_stats(
    *,
    completions: list[str],
    annotations: list[AnnotationResult],
    targets: list[dict[str, Any]],
    tokenizer: Any,
    format_type: str,
    numeric_tolerance: float = 0.2,
    truncated_by_max: list[bool] | None = None,
) -> dict[str, tuple[float, int]]:
    """组内 per-field ``(Σraw, n)`` —— ``μ_f`` 的**唯一**计算入口（★ §1.4 单一真相源）。

    为什么必须单独成函数（而不是让 flow 层自己算一遍）：``μ_f`` 的定义域与 raw 打分
    都是**算法层**语义（V-span 提取、``field_score``、末尾 EOS 后处理、
    ``leaf_compare_score`` 的容差）。flow 层重写一遍 = **复制算法代码** ⇒ 双真相源，
    一旦算法口径变，切片路径会静默用旧口径（`report.md` §4-③ / X-11）。

    **跨卡合并的口径**（`report.md` §4-③ / X-02 / X-11）：
    ``μ_f`` 是"组内所有发出该字段的 completion 的 raw 的均值"，即：
    ``μ_f = Σraw / n``（``n`` = **发出该字段的 span 计数**）。
    两个量都必须跨卡合并，且 ``n`` 用**整数严格相等**判据（相对容差会吞掉
    "漏一条 completion"这类真错）。

    **加序**：本函数返回**未求和**的 ``(Σraw, n)``，跨卡按**槽位序**相加
    ⇒ 与单卡整组的加序**相同**（仍非逐位，浮点累加本身有误差）。

    Returns:
        ``{field: (Σraw, n)}``；``n > 0`` 的字段才进入 ``μ_f`` 的定义域。

    Raises:
        ValueError: ``format_type`` 非法 / ``completions`` 与 ``annotations`` 不对齐。
    """
    if format_type not in ("tool_call", "json"):
        raise ValueError(f"format_type must be 'tool_call' or 'json', got {format_type!r}")
    if not completions or len(completions) != len(annotations):
        raise ValueError("completions must be a non-empty list aligned with annotations")
    truncated = truncated_by_max or [False] * len(completions)
    if len(truncated) != len(completions):
        raise ValueError("truncated_by_max must be aligned with completions")

    tags_list = [
        apply_tail_eos(text, ann.tags, format_type, truncated_by_max=tr)
        for text, ann, tr in zip(completions, annotations, truncated, strict=True)
    ]
    totals: dict[str, float] = {}
    counts: dict[str, int] = {}
    for text, tags, ann in zip(completions, tags_list, annotations, strict=True):
        for s, e, f in value_spans(text, tags, ann.fields):
            gt = gt_value_for_field(targets, f)
            raw = field_score(text[s:e], gt, numeric_tolerance) if gt is not None else 0.0
            totals[f] = totals.get(f, 0.0) + raw
            counts[f] = counts.get(f, 0) + 1
    return {f: (totals[f], counts[f]) for f in totals}


def merge_field_stats(
    per_rank_stats: list[dict[str, tuple[float, int]]],
) -> dict[str, tuple[float, int]]:
    """按**给定顺序**（= 槽位序）逐卡相加 ``(Σraw, n)``（★ 加序确定）。

    :param per_rank_stats: 各卡的 :func:`group_field_stats` 结果，**按 rank 序**。

    Returns:
        ``{field: (Σraw, n)}`` —— 与单卡整组**同一个多重集**的 ``(Σ, n)``。
    """
    totals: dict[str, float] = {}
    counts: dict[str, int] = {}
    for stats in per_rank_stats:
        for field, (raw_sum, n) in stats.items():
            totals[field] = totals.get(field, 0.0) + float(raw_sum)
            counts[field] = counts.get(field, 0) + int(n)
    return {f: (totals[f], counts[f]) for f in totals}


def field_means(stats: dict[str, tuple[float, int]]) -> dict[str, float]:
    """``(Σraw, n)`` → ``μ_f``（``n == 0`` 的字段**不进定义域**，与单卡
    ``mu.get(field, 0.0)`` 的行为一致）。"""
    return {f: raw_sum / n for f, (raw_sum, n) in stats.items() if n > 0}


def compute_group_advantages(
    *,
    completions: list[str],
    annotations: list[AnnotationResult],
    targets: list[dict[str, Any]],
    tokenizer: Any,
    format_type: str,
    numeric_tolerance: float = 0.2,
    truncated_by_max: list[bool] | None = None,
    field_mean_override: dict[str, float] | None = None,
) -> list[list[float]]:
    """为一个 rollout group 计算 per-token advantage(ragged)。

    Args:
        completions: 组内生成区域文本(与 annotations 一一对应)。
        annotations: 每条 completion 的字符级标注(annotate 输出)。
        targets: 归一化目标(GT,结构 ``{output: {tool_calls | content}}``)。
        tokenizer: 支持 ``return_offsets_mapping`` 的 HF tokenizer。
        format_type: ``"tool_call"`` 或 ``"json"``(组内一致,由
            ``Sample.expects_tool_calls`` 决定,不在本层猜测)。
        numeric_tolerance: 数字相对误差死区(leaf_compare_score)。
        truncated_by_max: 每条 completion 是否被 max_new_tokens 硬截断
            (被外力切断 → 不标末尾 EOS;缺省全部按未截断处理)。
        field_mean_override: ★ **可选**（默认 ``None``）。给定 ``{field: μ_f}`` 时
            **不再本地算 μ**(改用注入值);``None`` ⇒ 走原式(**dp=1 路径逐位不变**)。
            rollout 分卡下由 flow 层把各卡切片的
            :func:`group_field_stats` ``all_gather`` 后**按槽位序求和**得到整组 ``μ_f``
            再注入（`report.md` §4-③ / X-11）。

    Returns:
        ragged per-token advantage,与每条 completion 的生成 token 数对齐:
        ``S→+1.0 / V→raw−μ_f / T·W·D→0 / E→−1.0``。
    """
    if format_type not in ("tool_call", "json"):
        raise ValueError(f"format_type must be 'tool_call' or 'json', got {format_type!r}")
    if not completions or len(completions) != len(annotations):
        raise ValueError("completions must be a non-empty list aligned with annotations")
    truncated = truncated_by_max or [False] * len(completions)
    if len(truncated) != len(completions):
        raise ValueError("truncated_by_max must be aligned with completions")

    # ---- 1. 末尾 EOS 后处理 + token 级标注 ----
    tags_list = [
        apply_tail_eos(text, ann.tags, format_type, truncated_by_max=tr)
        for text, ann, tr in zip(completions, annotations, truncated, strict=True)
    ]
    token_labels = [
        char_to_token_labels(text, tags, ann.fields, tokenizer)
        for text, tags, ann in zip(completions, tags_list, annotations, strict=True)
    ]

    # ---- 2. 字段级 raw:每条 completion 的 V span → 字段分 ----
    # group_spans: [completion_idx][(start, end, field, raw)]
    group_spans: list[list[tuple[int, int, str, float]]] = []
    for text, tags, ann in zip(completions, tags_list, annotations, strict=True):
        spans: list[tuple[int, int, str, float]] = []
        for s, e, f in value_spans(text, tags, ann.fields):
            gt = gt_value_for_field(targets, f)
            raw = field_score(text[s:e], gt, numeric_tolerance) if gt is not None else 0.0
            spans.append((s, e, f, raw))
        group_spans.append(spans)

    # ---- 3. μ_f:per-field 组内均值(有该字段 V 段的 completion 参与) ----
    # ★ 口径单一真相源：本地 μ 走 `field_means(group_field_stats(...))`，
    #   ⛔ 不再就地重算一遍（rollout 分卡下 flow 层用同一个函数算切片统计）。
    #   `field_mean_override is None` ⇒ 与原式**逐位相同**（同一批 raw、同一加序）。
    #   ⚠ 已知代价：本分支多付出**一次** `value_spans` + `field_score` 扫描
    #   （`group_spans` 与 `group_field_stats` 各一次）。这是"单一真相源"换来的
    #   纯 CPU 开销（`field_score` 是字符串/浮点比较，无 IO、无 torch）——
    #   ⛔ 不以"复制一份 μ 算术进本函数"来省这点开销（那正是双真相源）。
    if field_mean_override is None:
        mu = field_means(
            group_field_stats(
                completions=completions,
                annotations=annotations,
                targets=targets,
                tokenizer=tokenizer,
                format_type=format_type,
                numeric_tolerance=numeric_tolerance,
                truncated_by_max=truncated_by_max,
            )
        )
    else:
        mu = dict(field_mean_override)

    # ---- 4. per-token adv ----
    advantages: list[list[float]] = []
    for text, ann, labels, spans in zip(
        completions, annotations, token_labels, group_spans, strict=True
    ):
        enc = tokenizer(text, return_offsets_mapping=True)
        offsets = enc.get("offset_mapping")  # type: ignore[attr-defined]
        if offsets is None:  # pragma: no cover - 防御
            raise ValueError("tokenizer must return offset_mapping (return_offsets_mapping=True)")
        adv: list[float] = []
        for label in labels:
            tag = label.tag
            start, end = offsets[label.token_index]
            if tag == CharTag.STRUCTURE:
                adv.append(1.0)
            elif tag == CharTag.VALUE:
                raw = next(
                    (r for s, e, _, r in spans if start >= s and end <= e),
                    0.0,
                )
                adv.append(raw - mu.get(label.field or "", 0.0))
            elif tag == CharTag.ERROR:
                adv.append(-1.0)
            else:
                adv.append(0.0)
        advantages.append(adv)
    return advantages


def compute_ripple_advantages(
    ragged_advantages: list[list[float]],
    old_log_probs: Any,
    prompt_len: int,
) -> Any:
    """GRASPO-Ripple: align ragged per-token advantages to the log-prob tensor.

    Advantages come from the annotation-driven pipeline
    (:func:`graspo.ripple.annotation.advantages.compute_group_advantages`):
    ``S→+1.0 / V→raw−μ_f / T·W·D→0 / E→−1.0``, no group-level normalization
    (v0.20.0).

    The ragged advantages are aligned to the ``old_log_probs`` tensor shape:
    prompt positions → 0, generated region → ragged advantages, trailing
    padding → 0.

    Args:
        ragged_advantages: Per-completion advantage lists (generated region).
        old_log_probs: shape (B, seq_len-1) log-prob tensor.
        prompt_len: Prompt token count (used for alignment).

    Returns:
        shape (B, seq_len-1) per-token advantage tensor.
    """
    import torch

    batch_size = old_log_probs.shape[0]
    seq_len_m1 = old_log_probs.shape[1]

    if len(ragged_advantages) != batch_size:
        raise RuntimeError(
            f"ragged_advantages batch size {len(ragged_advantages)} != "
            f"old_log_probs batch size {batch_size}"
        )

    # Align to old_log_probs shape: prompt positions → 0, generated region →
    # ragged advantages, trailing padding → 0.
    advantages = torch.zeros(
        batch_size,
        seq_len_m1,
        dtype=old_log_probs.dtype,
        device=old_log_probs.device,
    )
    gen_start = prompt_len - 1  # first generated token index in old_log_probs
    for i in range(batch_size):
        adv = ragged_advantages[i]
        for t in range(len(adv)):
            pos = gen_start + t
            if pos < seq_len_m1:
                advantages[i, pos] = adv[t]

    return advantages
