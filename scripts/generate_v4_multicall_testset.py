"""生成多调用错误形态测试集 annotation_testset_v4_multicall.jsonl。

背景（2026-08-08，v21 退化根因排查）：
- v21 训练中多调用（每轮多个 <tool_call>）增长 20 倍（0.5%→9.8%），
  违反 prompt"每轮只输出一个工具调用"约束，未被压制。
- 根因：标注器 3.5 双调用检测（tool_call_labeler.py:374-381）只覆盖
  "闭合后紧邻 <tool_call>" 形态（T33/T40），真实数据 94.5% 是
  "闭合后间隔杂散文本再出现 <tool_call>"——全部漏检（第二调用 W 无 E）。

本脚本从真实 rollout（rollouts.readable.jsonl）抽取各形态样本：
- 杂散文本间隔形态：多条（tool_response/image/conclusion/summary/response/
  analysis/think/自由文本 等 100+ 种自造标签）
- 其余错误形态各 2 条：紧邻双调用 / 截断第二调用 / 病态(顺序乱) /
  首调用有错+多调用（提前 return 绕过检测）

annotation 由标注器生成（当前 v3.x 算法），correct=pending——
等待用户审核 HTML 后逐条裁定期望标注，再根据裁定修改标注算法。

用法: python scripts/generate_v4_multicall_testset.py <rollouts.jsonl> > tests/data/annotation_testset_v4_multicall.jsonl
"""

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from graspo.ripple.annotation.labeler import AnnotationInput, annotate

# ── 形态分类 ──────────────────────────────────────────────────────────────


def classify_form(text: str) -> str | None:
    """对多调用 completion 分类（返回 None 表示非多调用）。

    返回形如 ``gap:<tag>`` / ``adjacent`` / ``truncated_second`` /
    ``malformed_order`` / ``first_call_error``。``first_call_error`` 表示
    首调用自身有格式错误（提前 return 绕过 3.5 多调用检测的形态）。
    """
    if text.count("<tool_call>") <= 1:
        return None
    m = re.search(r"</tool_call>(.*?)<tool_call>", text, re.DOTALL)
    if not m:
        # 第二调用出现在首闭合前（病态）或第二调用无闭合（截断）
        if text.count("</tool_call>") < text.count("<tool_call>"):
            return "truncated_second"
        return "malformed_order"
    gap = m.group(1)
    if not gap.strip():
        return "adjacent"  # 紧邻（仅空白间隔）
    tag = re.search(r"<([a-zA-Z_]+)", gap)
    if tag:
        return f"gap:{tag.group(1)}"  # 间隔 <tag>
    return "gap:freetext"  # 间隔自由文本


def first_call_is_clean(completion: str, gt_fn: str) -> bool:
    """首调用是否完全正确（工具名 + 参数名 + 值都匹配 GT，用于纯净样本筛选）。"""
    body = re.search(r"<tool_call>(.*?)</tool_call>", completion, re.DOTALL)
    if not body:
        return False
    text = body.group(1)
    fn = re.search(r"<function=([^>\n]+)>", text)
    if not fn or fn.group(1).strip() != gt_fn:
        return False
    params = dict(re.findall(r"<parameter=([^>\s]+)>\s*\n(.*?)\n\s*</parameter>", text, re.S))
    return bool(params)


# ── ground_truth 转换（targets dict → "fn(arg=v, arg=v)" 字符串）───────────


def targets_to_gt(targets: list[dict]) -> str | None:
    """从真实 targets 提取首个 tool_call 转 v3 测试集 GT 格式。"""
    if not targets:
        return None
    out = targets[0].get("output") or {}
    calls = out.get("tool_calls") or []
    if not calls:
        return None
    call = calls[0]
    args = []
    for k, v in (call.get("arguments") or {}).items():
        if isinstance(v, float) and v.is_integer():
            v = int(v)
        args.append(f"{k}={v}")
    return f"{call['name']}({', '.join(args)})"


# ── 样本抽取 ──────────────────────────────────────────────────────────────


def extract_samples(path: str) -> dict[str, list[dict]]:
    """从真实 rollout 抽取各形态样本（去重，保留首个）。

    间隔杂散文本形态（gap:*) 优先抽"首调用完全正确"的样本，使多调用
    漏检更纯净（不被首调用自身的 E 干扰）；首调用有错的样本单独归入
    ``first_call_error`` 形态。
    """
    by_form: dict[str, list[dict]] = {}
    seen: set[tuple] = set()
    for line in Path(path).open(encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        gt = targets_to_gt(rec.get("targets"))
        if not gt:
            continue
        gt_fn = gt.split("(", 1)[0]
        for comp in rec.get("completions") or []:
            text = comp.get("completion") if isinstance(comp, dict) else str(comp)
            form = classify_form(text)
            if not form:
                continue
            # seen key 含 gt_fn：同一 completion 文本可能出现在多个 rec
            # （模型重复输出），不同 rec 的 GT 不同——不带 gt_fn 会把
            # 后出现的 clean 样本误判为重复（先出现的是 dirty）
            key = (form, text[:80], gt_fn)
            if key in seen:
                continue
            seen.add(key)
            # 首调用有错 → 独立形态（提前 return 绕过多调用检测）。
            # gap:* 与 adjacent 优先 clean（多调用漏检纯净）；truncated_second
            # 真实形态即首调用残缺（如 <function=think</function>），保留原样。
            if form in ("gap:tool_response", "gap:image", "gap:conclusion",
                        "gap:summary", "gap:response", "gap:analysis",
                        "gap:think", "gap:freetext", "adjacent"):
                if not first_call_is_clean(text, gt_fn):
                    form = "first_call_error"
            by_form.setdefault(form, []).append(
                {"completion": text, "ground_truth": gt, "epoch": rec.get("epoch")}
            )
    return by_form


def annotate_case(sample: dict, cid: str, case: str, notes: str) -> dict:
    """对样本运行标注器生成 annotation（当前算法，pending 待裁定）。"""
    result = annotate(
        AnnotationInput(
            completion_text=sample["completion"],
            targets=targets_from_gt(sample["ground_truth"]),
            tokenizer=None,
            format_type="tool_call",
            check_json_markdown=False,
            check_think=False,
        )
    )
    annotation = "".join(t.value for t in result.tags)
    e_idx = annotation.find("E")
    return {
        "id": cid,
        "type": "tool_call",
        "case": case,
        "completion": sample["completion"],
        "ground_truth": sample["ground_truth"],
        "annotation": annotation,
        "correct": "pending",
        "error_pos": str(e_idx) if e_idx >= 0 else "",
        "notes": notes,
    }


def targets_from_gt(gt: str) -> list[dict]:
    """GT 字符串 → targets（与 v3 生成脚本 make_targets 同构）。"""
    name, rest = gt.split("(", 1)
    arguments = {}
    for pair in rest.rstrip(")").strip().split(","):
        pair = pair.strip()
        if not pair:
            continue
        k, v = pair.split("=", 1)
        v = v.strip()
        try:
            arguments[k.strip()] = float(v)
        except ValueError:
            arguments[k.strip()] = v
    return [{"output": {"tool_calls": [{"name": name.strip(), "arguments": arguments}]}}]


# ── 用例构造 ──────────────────────────────────────────────────────────────

# 杂散文本间隔形态（用户要求多条，看具体情况后裁定压制方式）
GAP_FORMS = [
    "gap:tool_response",
    "gap:image",
    "gap:conclusion",
    "gap:summary",
    "gap:response",
    "gap:analysis",
    "gap:think",
    "gap:freetext",
]
# 其余错误形态（每种 2 条）
OTHER_FORMS = ["adjacent", "truncated_second", "first_call_error"]

GAP_NOTES = {
    "gap:tool_response": "间隔 <tool_response> 自造标签后第二调用（真实最大类，297 条）——第二调用首字符应 E？杂散标签如何处理待裁定",
    "gap:image": "间隔 <image> 自造标签后第二调用（样本统计）",
    "gap:conclusion": "间隔 <conclusion> 总结标签后第二调用（70 条）",
    "gap:summary": "间隔 <summary> 后第二调用（35 条）",
    "gap:response": "间隔 <response> 后第二调用（26 条）",
    "gap:analysis": "间隔 <analysis> 后第二调用（23 条）",
    "gap:think": "间隔 <think> 后第二调用（6 条）",
    "gap:freetext": "间隔自由文本后第二调用（2 条）",
}
OTHER_NOTES = {
    "adjacent": "紧邻双调用（仅空白间隔，T33 同构但来自真实数据）——第二调用首字符 E 其后 D",
    "truncated_second": "第二调用被截断（无 </tool_call> 闭合）——regex 抓不到、reward 层也看不见的盲区形态",
    "first_call_error": "首调用自身有格式错误 + 尾部多调用——标注器提前 return 绕过 3.5 多调用检测",
}


def build_cases(by_form: dict[str, list[dict]]) -> list[dict]:
    cases: list[dict] = []
    idx = 0

    def add(form: str, sample: dict, label: str, notes: str) -> None:
        nonlocal idx
        idx += 1
        cases.append(annotate_case(sample, f"M{idx:02d}", label, notes))

    # 杂散文本形态：每条取 2 个不同样本（若只有 1 个则取 1）
    for form in GAP_FORMS:
        samples = by_form.get(form, [])[:2]
        for i, sample in enumerate(samples):
            add(
                form,
                sample,
                f"间隔{form[4:]} 多调用({i + 1})",
                GAP_NOTES[form],
            )
    # 其余形态：每条 2 条
    for form in OTHER_FORMS:
        samples = by_form.get(form, [])[:2]
        for i, sample in enumerate(samples):
            add(
                form,
                sample,
                f"{form} 多调用({i + 1})",
                OTHER_NOTES[form],
            )
    return cases


def main() -> None:
    if len(sys.argv) < 2:
        print("用法: python scripts/generate_v4_multicall_testset.py <rollouts.jsonl>")
        sys.exit(1)
    by_form = extract_samples(sys.argv[1])
    print(f"# 形态分布: { {k: len(v) for k, v in sorted(by_form.items())} }", file=sys.stderr)
    missing = [f for f in GAP_FORMS + OTHER_FORMS if not by_form.get(f)]
    if missing:
        print(f"# 警告: 无样本的形态: {missing}", file=sys.stderr)
    for case in build_cases(by_form):
        print(json.dumps(case, ensure_ascii=False))


if __name__ == "__main__":
    main()
