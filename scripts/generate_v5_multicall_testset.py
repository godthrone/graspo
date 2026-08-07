"""生成多调用单元测试数据集 annotation_testset_v5_multicall.jsonl（2026-08-08）。

从 v4（19 条，真实 rollout 抽取）精选 5 条代表性样本，按用户裁定的
**新标注规则**标注期望：

    用户裁定："思考的废话应该在工具调用前面；工具调用完成之后，所有的
    多余字符应该直接标 E 后面跟 D。工具之后没有思考，只可能是乱码，
    要压制而不是放弃。"（2026-08-08，另确认千问模板 </tool_call> 后
    无换行规定——1312 条合法 completion 的 </tool_call> 后均为空串，
    故闭合后**第一个字符（含换行）直接 E + 其后全 D**，一刀切）

新规则（取代 v3.x 零散裁定）：
1. tool_call 闭合标签（</tool_call>）之后**第一个字符**（含换行符）标 E，
   其后全部 D——无论尾随的是第二个 <tool_call>、自造杂散标签
   （<tool_response>/<conclusion> 等）、还是自然语言/换行。
   T36（闭合后自然语言尾随 W）被本规则推翻。
2. 首调用内容按现有语义正常标注（结构 S、值 V）——首调用正确则 E 与
   首调用无关，仅打在闭合后。
3. 首调用自身残缺/工具名错误：E 标在残缺处/错误处（用户确认 v4 正确），
   其后全 D——尾部内容被 E 后全 D 覆盖，无需额外处理。

5 条样本（每条代表一个错误形态）：
- M01 tool_response 夹 JSON（间隔形态 + 自造标签）→ 新规则
- M02 conclusion 夹自然语言（间隔形态 + 文本夹层）→ 新规则
- M03 紧邻双调用（闭合后紧跟第二个 <tool_call>）→ 新规则（E 移到闭合后首个换行）
- M04 截断第二调用（首调用残缺）→ v4 已正确，保留
- M05 首调用工具名错 + 尾部多调用 → v4 已正确，保留

标注生成方式：标注器基础输出（首调用 S/V 正常）+ 新规则覆盖
（闭合后第一个字符 E + 其后全 D）。标注数据集不入库，等待用户审核
后由独立 agent 核验再置 correct=yes。

用法: python scripts/generate_v5_multicall_testset.py > tests/data/annotation_testset_v5_multicall.jsonl
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from graspo.ripple.annotation.labeler import AnnotationInput, annotate

V4_PATH = (
    Path(__file__).resolve().parents[1]
    / "tests" / "data" / "annotation_testset_v4_multicall.jsonl"
)
TOOL_CALL_CLOSE = "</tool_call>"


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


def annotate_new_rule(completion: str, gt: str) -> tuple[str, int]:
    """新规则：标注器基础输出 + 闭合后第一个字符 E + 其后全 D。"""
    result = annotate(
        AnnotationInput(
            completion_text=completion,
            targets=targets_from_gt(gt),
            tokenizer=None,
            format_type="tool_call",
            check_json_markdown=False,
            check_think=False,
        )
    )
    tags = list(t.value for t in result.tags)
    close_pos = completion.find(TOOL_CALL_CLOSE)
    if close_pos < 0:
        return "".join(tags), tags.index("E") if "E" in tags else -1
    after = close_pos + len(TOOL_CALL_CLOSE)
    if after < len(completion):
        tags[after] = "E"
        for i in range(after + 1, len(completion)):
            tags[i] = "D"
        return "".join(tags), after
    return "".join(tags), tags.index("E") if "E" in tags else -1


def load_v4() -> dict[str, dict]:
    """v4 → {case: record}。"""
    out = {}
    for line in V4_PATH.open(encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        out[rec["case"]] = rec
    return out


def main() -> None:
    v4 = load_v4()
    # (v4 case_key, 新 id, 新 case 名, 是否按新规则重标)
    picks = [
        ("间隔tool_response 多调用(1)", "M01", "间隔<tool_response>夹JSON后第二调用", True),
        ("间隔conclusion 多调用(1)", "M02", "间隔<conclusion>夹自然语言后第二调用", True),
        ("adjacent 多调用(1)", "M03", "紧邻双调用（仅空白间隔）", True),
        ("truncated_second 多调用(1)", "M04", "截断第二调用（首调用残缺）", False),
        ("first_call_error 多调用(1)", "M05", "首调用工具名错+尾部多调用", False),
    ]
    for case_key, cid, label, re_annotate in picks:
        rec = v4.get(case_key)
        if not rec:
            print(f"# 警告: v4 缺 {case_key}", file=sys.stderr)
            continue
        if re_annotate:
            annotation, e_pos = annotate_new_rule(rec["completion"], rec["ground_truth"])
            e_pos_str = str(e_pos) if e_pos >= 0 else ""
            notes = (
                f"{rec.get('notes', '')} [v5 新规则: </tool_call> 后第一个字符 E + 其后全 D，"
                f"一刀切压制（千问模板无尾部换行规定）；首调用内容正常 S/V]"
            )
        else:
            annotation = rec["annotation"]
            e_pos_str = rec["error_pos"]
            notes = f"{rec.get('notes', '')} [v5: 用户确认 v4 标注正确，保留]"
        out = dict(rec)
        out.update(
            {
                "id": cid,
                "case": label,
                "annotation": annotation,
                "correct": "pending",
                "error_pos": e_pos_str,
                "notes": notes,
            }
        )
        print(json.dumps(out, ensure_ascii=False))


if __name__ == "__main__":
    main()
