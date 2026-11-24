"""标注 → per-token advantage:每个 token 只看自己对不对 + 末尾 EOS 惩罚。

链路(方案 token-reward-advantage-design-v0.20.md §3):

1. 字符级标注 ``AnnotationResult``(annotate 输出,S/V/T/W/E/D)
2. 末尾 EOS 惩罚:结构不完整(截断/缺闭合/纯乱码 = "没写完")且非 max_len 硬截断
   → 末尾非空白字符标 E(-1.0),语义"在这里结束是错的"
3. token 级映射 ``char_to_token_labels``(offset_mapping,特殊 token 跳过)
4. 字段级 raw:V span 与 GT 比较(``leaf_compare_score`` / 数组存在性匹配),
   同一字段的所有 V token 共享字段分
5. adv:``S→+1.0 / V→±1.0 / T·W·D→0 / E→−1.0``
   （★ 例外：组内**每条都写对**的结构位置 → ``0``，见下方 P3）

纯函数、零设施依赖;tokenizer 只用于 offset_mapping。

## ★★ P2（2026-11-18）：删掉 μ_f，值 token 改为 ±1 —— 直接后果

负责人的设计原则（唯一判据）：

    「按 token 算梯度，每个 token 有自己的梯度更新方向。第一个错误的 token 会有负梯度，
    后面的 token 就扔掉。如果这个 token 和前面的所有 token 都正确，那么这些都有正梯度。」

旧式 ``adv(V) = raw − μ_f``（``μ_f`` = 该字段的**组内均值**）是 GRPO 的组内比较，违背这条
原则，实测后果（800 条真实 rollout）：

- 全对组 ``μ_f = 1`` ⇒ 正确 token 的优势 ``1 − 1 = 0``（正确也拿不到正梯度）；
- 全错组 ``μ_f = 0`` ⇒ 错误 token 的优势 ``0 − 0 = 0``（错误也拿不到负梯度）；
- 100 个 rollout group 里 32 组的 8 条**全错**、0 组全对 ⇒ 值 token 一律 0 优势，
  只剩结构 ``+1.0`` 那一路信号。

现在值 token **只看它自己对不对**（``raw == 1.0`` ⇒ ``+1.0``，否则 ``−1.0``），
与同组其它 completion 无关（逐条闭合）。因此 ``μ_f`` / ``group_field_stats`` /
``field_means`` 这一套组内均值机制**不再参与 advantage**：函数本体保留（native 分卡
路径仍在 gather 它），但 ``compute_group_advantages`` 不再读它，
``field_mean_override`` 参数仅为调用方签名兼容而保留（传任何值都不影响输出）。

⛔ 不引入任何组内归一化（减均值 / z-score / baseline）——已被负责人明确否决。
⛔ P2 对 ``S=+1.0 / E=−1.0 / T·W·D=0`` 三类取值一字未改（P3 之后结构位置有唯一的
   例外：组内全对 ⇒ 0，见下一节）。

## ★★ P3（2026-11-23）：组内全对的结构位置不进梯度

负责人裁定（唯一判据）：

    「对于一个结构 token 来说，如果他是对的，那么他一定对应 ground truth 里面的某个结构
    字符，对吧？如果这个结构字符对应的 rollout 中每个字符都对，那么这个 token 统一不再进入
    训练……对 ground truth 里面的所有结构字符做个统计，对于那些全对的结构字符，就从最终的
    梯度优化中去掉这些 token 样本。」

「写错 = 结构 / 选错 = 值」的划分也是他定的：JSON/XML 语法、括号、引号、字段名拼写、
参数名 = 结构；工具名、参数值 = 值。

改前实测（800 条真实 rollout = 100 组 × 8 条）：每条 rollout 都无条件拿到**同一批 5 个
恒定前缀 token ×(+1) = +5** 的正优势，与答对答错无关 ⇒ 800 条 Σadv 全为正
（均值 +11.06），工具名叫错的 496 条净 **+3.00**。模型学到的是「写出模板前缀就赚 +5，
工具名叫什么无所谓」——训练塌缩的直接机制。

改后（同一批数据）：正优势条数 **800 → 304**（= 工具名全对的条数），工具名写错的 496 条
Σadv **+3.00 → −2.00**。实现在 :func:`all_correct_structure_chars`（k = 组内 S 字符数最小值）。

★ ``k`` 的定义域是**整组**（G 条），不是**本卡切片**：``dp>1`` 时每张卡只持 ``G//m`` 条，
本卡切片的最小值 ≥ 整组最小值 ⇒ 会把「本卡都写对、别的卡写短了」的结构位置也置 0
（过度 mask，dp=1 与 dp=4 读数不可比）。设施层用 :func:`group_structure_char_counts`
得到逐条结构字符数、跨卡 ``min`` 归约后经 ``group_min_structure_chars`` 传回。

⛔ 用 ``advantage = 0`` 实现，**不缩小 ``action_mask``**：loss 是「序列内 masked_mean、再按
batch 求均值」，缩小 mask 会把其余 token 的梯度整体放大 1.21×，读数无法与历史对比。
"""

import json
from typing import Any

from graspo.ripple.annotation.char_tag import CharTag
from graspo.ripple.annotation.labeler import AnnotationResult
from graspo.ripple.annotation.tokenize import char_to_token_labels
from graspo.ripple.annotation.tool_call_labeler import TOOL_NAME_FIELD
from graspo.ripple.reward.compare import leaf_compare_score

#: :func:`_json_decoded_value` 的"解不出来"哨兵（区分"解不出"与"解出 None"）。
_NOT_JSON = object()

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


def _json_decoded_value(value_text: str) -> object:
    """把值文本按 **JSON 语义**再解一次（``true``/``false``/``null``/带转义的字符串）。

    失败返回 ``_NOT_JSON``（模块级哨兵，区分"解不出来"和"解出 None"）。

    为什么需要这个第二解释（P2 暴露的旧缺陷）：标注是**字符级**的，值文本原样来自模型输出，
    于是 ``"hello\\nworld"``（字面反斜杠+n）与 ``true`` 这类字面量按"字符串精确比较"永远
    比不上 GT（``hello<换行>world`` / ``True``）。旧式 ``raw − μ_f`` 把这些误判吸收成
    ``0 − 0 = 0``（无信号，从没被发现）；P2 的 ±1 会把它们变成对**正确输出**的惩罚，
    所以必须在打分里先把 JSON 语义解释补上。
    """
    try:
        return json.loads(value_text.strip())
    except (TypeError, ValueError):
        return _NOT_JSON


def field_score(value_text: str, gt_val: object, numeric_tolerance: float) -> float:
    """字段级打分:整个 value span 与 GT 值比较。

    GT 为 list(JSON 数组)时:值文本与任一元素比较,命中即 1.0(存在性匹配)。
    标量:``leaf_compare_score``(数字容差 / 字符串精确)。

    值文本按**两种解释**取较高分：原样解析(:func:`_parse_value`)与 JSON 语义
    (:func:`_json_decoded_value`)。第二种是 P2 的必要前提——否则 ``true`` / ``"a\\nb"``
    这类合法输出会被判成"值错"而吃 −1（见 :func:`_json_decoded_value` 的说明）。
    """
    candidates = [_parse_value(value_text)]
    decoded = _json_decoded_value(value_text)
    if decoded is not _NOT_JSON:
        candidates.append(decoded)
    if isinstance(gt_val, list):
        for elem in gt_val:
            for v in candidates:
                if isinstance(elem, (int, float)) and not isinstance(elem, bool):
                    if (
                        isinstance(v, float)
                        and leaf_compare_score(v, elem, numeric_tolerance=numeric_tolerance) >= 1.0
                    ):
                        return 1.0
                elif v == elem:
                    return 1.0
        return 0.0
    return max(
        float(leaf_compare_score(v, gt_val, numeric_tolerance=numeric_tolerance))
        for v in candidates
    )


def value_advantage(raw: float) -> float:
    """字段分 ``raw`` → 值 token 的 per-token advantage：满分 ``+1.0`` / 非满分 ``−1.0``。

    **为什么需要这条映射**（P2）：``raw`` **不是**二值。字符串字段走精确比较（0/1），
    但数字字段走 :func:`leaf_compare_score`：容差死区内（相对误差 ≤ ``numeric_tolerance``）
    **恰为 1.0**，死区外按 ``1/(1+超出量)`` **连续衰减**（例：16.9 vs 40，容差 0.2 ⇒ ≈0.462）。
    若直接把连续值当 advantage，错误值的梯度会随"错多少"缩放，且永远到不了满幅 −1——
    这与负责人「值只分对/错，各 token 各有方向」的判据不符，也与结构 ±1 的量级不可比。

    因此定死一条规则（写进测试 ``test_numeric_within_tolerance_is_plus_one_beyond_is_minus_one``）：

    - ``raw >= 1.0``（含容差死区内的满分）⇒ ``+1.0``：这个值算"对"
    - ``raw < 1.0``（哪怕 0.99）⇒ ``−1.0``：这个值算"错"

    选「非满分即错」而不是「按 raw 分档」的依据：GRASPO 的 per-token 信号是**方向**信号，
    不是连续打分器；容差死区已经把"数值小误差"划进"对"，死区之外就是明确写错。
    容差本身仍由 ``numeric_tolerance`` 单一控制，本函数不引入第二个阈值。
    """
    return 1.0 if raw >= 1.0 else -1.0


def gt_value_for_field(targets: list[dict[str, Any]], field: str) -> object | None:
    """按 field 名从 targets 取 GT 值(field 形如 ``fn.param`` / json key / ``TOOL_NAME_FIELD``)。

    ``field == TOOL_NAME_FIELD`` 时取**工具名**本身（P1）：``<function=NAME>`` 的 NAME 被标注成
    一个可评分字段，GT 就是 ``targets[0].output.tool_calls[0].name``（不在 ``arguments`` 里）。
    """
    if field == TOOL_NAME_FIELD:
        for target in targets:
            out = target.get("output") or {}
            for tc in out.get("tool_calls") or []:
                if isinstance(tc.get("name"), str):
                    return tc["name"]
        return None
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


def structure_char_count(tags: list[CharTag]) -> int:
    """一条 completion 上的 S 字符数（``k`` 的唯一计数口径）。"""
    return sum(1 for tag in tags if tag is CharTag.STRUCTURE)


def group_structure_char_counts(
    *,
    completions: list[str],
    annotations: list[AnnotationResult],
    format_type: str,
    truncated_by_max: list[bool] | None = None,
) -> list[int]:
    """本卡切片各条 completion 的**结构字符数**（★ P3 的 ``k`` 的跨卡会合输入）。

    ## 为什么单独成函数

    :func:`all_correct_structure_chars` 的 ``k`` 是**组内**各条结构字符数的最小值。
    ``native`` 后端在 ``dp>1`` 时每张卡只持有本卡的 ``G//m`` 条 completion
    （``flow/backend/native/trainer/rollout.py``）⇒ 若 ``k`` 取**本卡切片**的最小值，
    必有 ``k_shard >= k_group``，于是「本卡都写对、别的卡写短了」的结构位置也被置 0
    （**过度 mask**）⇒ 同一算法在 dp=1 与 dp≥2 下行为不同、读数不可比。

    修复只在**设施层**做一件事：把本卡切片的逐条结构字符数随既有 gather 一起做一次
    ``min`` 归约，得到整组的 ``k`` 再传回 :func:`compute_group_advantages` 的
    ``group_min_structure_chars``（载荷是**整数标量**，不扩 ``gather`` 契约）。
    本函数就是那个**被会合的标量**的算法层出处 —— ⛔ 不让 flow 层自己数 S 字符
    （末尾 EOS 后处理、标注口径都是算法层语义，重写一遍即双真相源）。

    ``truncated_by_max`` 的口径与 :func:`compute_group_advantages` **同一个**
    （被 max_len 硬截断 ⇒ 不标末尾 E ⇒ 不改 S 计数），否则会合出的 ``k`` 与算法层
    实际用的计数不一致，仍会多 mask 或少 mask。

    Returns:
        与 ``completions`` 对齐的结构字符数列表（**已过** :func:`apply_tail_eos`）。

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
    return [
        structure_char_count(apply_tail_eos(text, ann.tags, format_type, truncated_by_max=tr))
        for text, ann, tr in zip(completions, annotations, truncated, strict=True)
    ]


def all_correct_structure_chars(
    tags_list: list[list[CharTag]], *, min_structure_chars: int | None = None
) -> list[set[int]]:
    """组内**每条都写对**的结构字符下标集合（P3：这些位置不进梯度）。

    ## 为什么是「各条 S 字符数的最小值」而不是「按 GT 位置身份对齐」

    标注器（``tool_call_labeler``）的 S 标记语义是**严格前缀**：它按期望 mark 序列与模型输出
    逐字符比对，字符相等标 S，**首个不相等字符标 E，其后全部 D**（工具名写错时该段标 V 后
    drop）。因此一条 completion 上的 S 字符 ⟺ 「它写对的那部分结构的前缀」——S 之后的字符
    全部没走到/写错，也就不可能被判为「写对」。

    ⚠ 工具名字符**从不计入** S 字符数（P1 + B1，2026-11-24）：写对整段 ``VALUE(__tool_name__)``
    （+1.0），写错整段 ``VALUE``（−1.0）——工具名是**值**（选错），不是结构（写错），所以
    ``k`` 伸不进工具名内部（否则会把写对的工具名 mask 成 0.0）。

    于是一个结构位置「组内每条都写对」的必要条件是：**每条都走到了这个位置**，即
    ``位置序 < min(各条的 S 字符数)``。取 ``k = min S 字符数``，前 k 个 S 字符就是组内
    **可能**全对的位置；而前 k 个 S 字符本身**必然**全对（公共前缀里的字符每条都写对了）。

    这就是用户裁定的那句「对 ground truth 里面的所有结构字符做个统计，对于那些全对的
    结构字符，就从最终的梯度优化中去掉这些 token 样本」。

    已知的近似（实测不影响本批结果，故不做严格对齐）：参数是**无序集合**匹配，参数乱序时
    「第 k 个 S 字符」在不同 rollout 上可能对应不同的 GT 单元身份。本批 800 条数据上两种口径
    结果**完全一致**（都是每条 5 个 token）；严格口径要同时改 ``tool_call_labeler`` /
    ``json_labeler`` / ``tokenize`` 并扩展跨卡 gather 契约（只允许标量+计数），代价远大于收益。

    Args:
        tags_list: 组内各条 completion 的字符标注（**已过** :func:`apply_tail_eos`）。
        min_structure_chars: 显式给定的 ``k``（★ 分卡路径：**整组**各条结构字符数的
            最小值，由设施层跨卡 ``min`` 归约后传回，见
            :func:`group_structure_char_counts`）。``None`` ⇒ 用 ``tags_list`` 自己的
            ``min``（= 单卡整组口径，dp=1 路径的行为）。

    Returns:
        与 ``tags_list`` 对齐的列表，每项是该条 completion 上「组内全对」的字符下标集合
        （即它的前 ``k`` 个 S 字符；``k = min`` 各条 S 字符数）。
    """
    if min_structure_chars is not None and min_structure_chars < 0:
        raise ValueError(f"min_structure_chars must be >= 0, got {min_structure_chars}")
    structure_counts = [structure_char_count(tags) for tags in tags_list]
    if not structure_counts:
        return []
    k = min(structure_counts) if min_structure_chars is None else int(min_structure_chars)
    prefix: list[set[int]] = []
    for tags in tags_list:
        structure_positions = [i for i, tag in enumerate(tags) if tag is CharTag.STRUCTURE]
        prefix.append(set(structure_positions[:k]))
    return prefix


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
    group_min_structure_chars: int | None = None,
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
        field_mean_override: ⚠ **自 P2 起不再参与计算**（传任何值输出都一样）。保留该参数
            只为不破坏 native 分卡调用方的签名（``flow/backend/native/trainer/rollout.py``
            仍在 gather 整组 ``(Σraw, n)`` 并传入）。值 token 已改为逐条闭合的 ±1，
            ``μ_f`` 没有可以作用的位置。
        group_min_structure_chars: P3 的 ``k``（组内各条**结构字符数**的最小值）。
            ``None`` ⇒ 用 ``completions`` 自己的 ``min``（单卡整组 = dp=1 路径的行为，
            逐位不变）。★ **``native`` 在 ``dp>1`` 时必须传整组的 ``k``**：那时本函数的
            ``completions`` 只是本卡切片（``G//m`` 条），本卡 ``min`` ≥ 整组 ``min`` ⇒
            会把「本卡都写对、别的卡写短了」的结构位置也置 0（过度 mask，dp=1 与 dp=4
            读数不可比）。取值出处见 :func:`group_structure_char_counts`。

    Returns:
        ragged per-token advantage,与每条 completion 的生成 token 数对齐:
        ``S→+1.0 / V→±1.0（值对 +1、值错 −1）/ T·W·D→0 / E→−1.0``；
        **组内全对的结构位置 → 0**（P3，见 :func:`all_correct_structure_chars`）。
    """
    if format_type not in ("tool_call", "json"):
        raise ValueError(f"format_type must be 'tool_call' or 'json', got {format_type!r}")
    if not completions or len(completions) != len(annotations):
        raise ValueError("completions must be a non-empty list aligned with annotations")
    truncated = truncated_by_max or [False] * len(completions)
    if len(truncated) != len(completions):
        raise ValueError("truncated_by_max must be aligned with completions")
    # ⚠ P2：`field_mean_override` **有意不读**（见 docstring）——保留签名只为 native 分卡
    #   调用方不崩。它唯一的用途曾是注入组内均值 μ_f，而 μ_f 已不参与 advantage。
    _ = field_mean_override

    # ---- 1. 末尾 EOS 后处理 + token 级标注 ----
    tags_list = [
        apply_tail_eos(text, ann.tags, format_type, truncated_by_max=tr)
        for text, ann, tr in zip(completions, annotations, truncated, strict=True)
    ]
    token_labels = [
        char_to_token_labels(text, tags, ann.fields, tokenizer)
        for text, tags, ann in zip(completions, tags_list, annotations, strict=True)
    ]

    # ---- 1b. 组内全对的结构位置 → 该位置的 token 不进梯度（P3）----
    # 见 :func:`all_correct_structure_chars`：k = 组内各条 S 字符数的最小值，前 k 个 S 字符
    # 组内每条都写对，这些位置**没有可学的信号**（永远 +1.0 的常数偏置），adv 置 0。
    # ★ 分卡路径（dp>1）的 k 由设施层跨卡 min 归约后传入（本卡切片只持 G//m 条）；
    #   None ⇒ 本组（= 整组）本地 min，dp=1 路径逐位不变。
    group_correct_chars = all_correct_structure_chars(
        tags_list, min_structure_chars=group_min_structure_chars
    )

    # ---- 2. 字段级 raw:每条 completion 的 V span → 字段分 ----
    # group_spans: [completion_idx][(start, end, field, raw)]
    # ★ raw = ``None`` 表示**该字段没有 GT 可比**（例：JSON 数组元素在标注里没有字段名，
    #   ``gt_value_for_field`` 取不到）。这类 token 必须**没有信号（0）**，不能算"值错"——
    #   旧式 ``raw − μ_f`` 下它们恰好是 ``0 − 0 = 0``（无信号）；P2 的 ±1 若把 None 当
    #   0.0 就会凭空给一批"无法判对错"的 token 打 −1（实测踩到：J01/J02… 等 JSON 数组元素
    #   用例——它们的 V span 没有字段名，GT 取不到）。
    group_spans: list[list[tuple[int, int, str, float | None]]] = []
    for text, tags, ann in zip(completions, tags_list, annotations, strict=True):
        spans: list[tuple[int, int, str, float | None]] = []
        for s, e, f in value_spans(text, tags, ann.fields):
            gt = gt_value_for_field(targets, f)
            raw = field_score(text[s:e], gt, numeric_tolerance) if gt is not None else None
            spans.append((s, e, f, raw))
        group_spans.append(spans)

    # ---- 3. per-token adv（★ 值 token 逐条闭合；结构 token 只看组内全对，见 1b）----
    # P2（2026-11-18）：值 token 的旧式 ``raw − μ_f`` 已删除。``μ_f`` 是组内均值，属 GRPO 的
    # 组内比较：全对组 μ=1 ⇒ 正确 token 优势 0；全错组 μ=0 ⇒ 错误 token 优势 0（真实数据里
    # 32 组全错、0 组全对 ⇒ 值 token 一律没信号）。现在每个值 token 只看它自己对不对
    # ⇒ ``raw == 1.0`` 为 +1.0、否则 −1.0（:func:`value_advantage`）。
    advantages: list[list[float]] = []
    for text, ann, labels, spans, prefix_chars in zip(
        completions, annotations, token_labels, group_spans, group_correct_chars, strict=True
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
                # P3：组内全对的结构位置置 0（不进梯度）。只有 token 覆盖的字符**全部**
                # 落在全对前缀内才算；跨出前缀的 token 后半段不在「全对」范围内，按原规则
                # 给 +1.0（少 mask 一个 token 是保守方向，绝不反过来多 mask）。
                if all(char in prefix_chars for char in range(start, end)):
                    adv.append(0.0)
                else:
                    adv.append(1.0)
            elif tag == CharTag.VALUE:
                raw = next(
                    (r for s, e, _, r in spans if start >= s and end <= e),
                    None,
                )
                if raw is None:
                    # 两种「没有字段分」的情形，都必须**没有信号**（⛔ 不臆造 ±1，
                    # 那会把"无法判对错"变成对模型的惩罚）：
                    #   a) 该字段没有 GT 可比（``gt_value_for_field`` 取不到）；
                    #   b) token 与 V span 不对齐（跨 span 边界 / 纯空白 token）。
                    adv.append(0.0)
                else:
                    adv.append(value_advantage(raw))
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
    ``S→+1.0 / V→±1.0 / T·W·D→0 / E→−1.0``（组内全对的结构位置为 0）,无组内归一化
    (v0.20.0;P2 起值 token 为逐条闭合的 ±1,不含任何组内均值;P3 起全对结构位置置 0)。

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
