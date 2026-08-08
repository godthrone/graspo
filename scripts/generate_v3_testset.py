"""生成 annotation_testset_v3.jsonl（v3.0，2026-08-07）。

v2 → v3 变更（与 v14 数据工具拆分同步）：
- T 系列全部换成 v14 双工具格式（rotate_arm / extend_arm，按 GT action_type 映射）；
  T30（get_weather 跨函数）保留原样
- 标注语义更新（标注器 v3.0 修复，用户裁定）：
  a) 多余/未知参数 → `<parameter=` S + 参数名首字符 E（原 E 落在 p/`<` 处）
  b) 双调用（闭合后第二个 <tool_call>）→ 第二个调用首字符 E（v20 毒药形态）
- 新增 T33-T36 双工具场景（毒药双调用 / 工具名错 / 跨工具参数错 / 尾随回归）
- J 系列 21 条不动

用法: python scripts/generate_v3_testset.py > tests/data/annotation_testset_v3.jsonl
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from graspo.ripple.annotation.labeler import AnnotationInput, annotate

# 转换源 v2 文件已于 2026-08-07 删除（v3 为唯一数据源）——本脚本为历史追溯/
# 重建参考：如重建，先从 git 恢复 v2（git show <commit>:tests/data/annotation_testset_v2.jsonl）
V2_PATH = Path(__file__).resolve().parents[1] / "tests" / "data" / "annotation_testset_v2.jsonl"

_ROTATION_ACTIONS = {"逆时针旋转", "顺时针旋转"}


def tool_for_gt(ground_truth: str) -> str | None:
    """按 GT 的 action_type 判定 v14 工具名（无 action_type 时按参数名判定）。"""
    if not ground_truth.startswith("robot_atomic_control("):
        return None  # 非 robot_atomic_control 用例（T30 get_weather）
    for pair in ground_truth.split("(")[1].split(","):
        pair = pair.strip()
        if pair.startswith("action_type="):
            action = pair.split("=", 1)[1].strip().strip("'\"")
            if action in _ROTATION_ACTIONS:
                return "rotate_arm"
            return "extend_arm"
    # 缺 action_type 的 GT（T27）：按参数名判定——angle_deg 属旋转、distance_cm 属伸缩
    if "angle_deg" in ground_truth:
        return "rotate_arm"
    if "distance_cm" in ground_truth:
        return "extend_arm"
    return None


def convert_t_case(case: dict) -> dict:
    """T 系列：robot_atomic_control → v14 工具名（completion + ground_truth）。"""
    gt = case["ground_truth"]
    tool = tool_for_gt(gt)
    if tool is None:
        return dict(case)
    out = dict(case)
    out["completion"] = case["completion"].replace("robot_atomic_control", tool)
    out["ground_truth"] = gt.replace("robot_atomic_control", tool)
    out["notes"] = (case.get("notes", "") + f" [v3: 工具 {tool}]").strip()
    return out


def make_targets(format_type: str, gt: str) -> list[dict]:
    if format_type == "tool_call":
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
    try:
        content = json.loads(gt)
    except (TypeError, ValueError, json.JSONDecodeError):
        content = {}
    return [{"output": {"content": content}}]


def annotate_case(case: dict) -> dict:
    """对一条用例运行标注器（v3.0 语义），生成 annotation 字段。"""
    cid = case["id"]
    no_fence = {"J02", "J18", "J20", "J21", "J22"}
    think_cases = {"T11", "T24", "T25"}
    result = annotate(
        AnnotationInput(
            completion_text=case["completion"],
            targets=make_targets(case["type"], case["ground_truth"]),
            tokenizer=None,
            format_type=case["type"],
            check_json_markdown=cid not in no_fence,
            check_think=cid in think_cases,
        )
    )
    annotation = "".join(t.value for t in result.tags)
    e_idx = annotation.find("E")
    case["annotation"] = annotation
    case["error_pos"] = str(e_idx) if e_idx >= 0 else ""
    # 标注数据集不能由标注器自动生成后直接入库（8/6 教训）：
    # pending 状态等待独立 agent 规则核验，通过后置 yes
    case["correct"] = "pending"
    return case


# ── 新增双工具 case（v3.0）──
NEW_CASES = [
    {
        "id": "T33",
        "type": "tool_call",
        "case": "双完整调用(毒药形态)",
        "completion": (
            "<tool_call>\n<function=rotate_arm>\n<parameter=action_type>\n逆时针旋转\n"
            "</parameter>\n<parameter=angle_deg>\n15\n</parameter>\n</function>\n</tool_call>\n"
            "<tool_call>\n<function=extend_arm>\n<parameter=action_type>\n伸长手臂\n"
            "</parameter>\n<parameter=distance_cm>\n20\n</parameter>\n</function>\n</tool_call>"
        ),
        "ground_truth": "rotate_arm(action_type=逆时针旋转, angle_deg=40.0)",
        "notes": "v20 毒药形态（S9-1 同构）：完整首调用 + 尾部第二完整调用违反"
        "'每轮只输出一个工具调用'约束——第二个 <tool_call> 首字符 E 其后 D",
    },
    {
        "id": "T34",
        "type": "tool_call",
        "case": "工具名错(rotate→extend)",
        "completion": (
            "<tool_call>\n<function=extend_arm>\n<parameter=action_type>\n逆时针旋转\n"
            "</parameter>\n<parameter=angle_deg>\n15\n</parameter>\n</function>\n</tool_call>"
        ),
        "ground_truth": "rotate_arm(action_type=逆时针旋转, angle_deg=40.0)",
        "notes": "工具名错误 = 格式错误（v3.0 裁定）：E 于 extend_arm 的 e（函数名首字符）",
    },
    {
        "id": "T35",
        "type": "tool_call",
        "case": "跨工具参数错(rotate_arm用distance_cm)",
        "completion": (
            "<tool_call>\n<function=rotate_arm>\n<parameter=action_type>\n逆时针旋转\n"
            "</parameter>\n<parameter=distance_cm>\n25\n</parameter>\n</function>\n</tool_call>"
        ),
        "ground_truth": "rotate_arm(action_type=逆时针旋转, angle_deg=40.0)",
        "notes": "参数与工具强绑定（v14 原则）：rotate_arm 用 distance_cm = 多余参数"
        " → `<parameter=` S + 参数名首字符 E（d）",
    },
    {
        "id": "T36",
        "type": "tool_call",
        "case": "闭合后自然语言尾随(回归)",
        "completion": (
            "<tool_call>\n<function=rotate_arm>\n<parameter=action_type>\n逆时针旋转\n"
            "</parameter>\n<parameter=angle_deg>\n40.0\n</parameter>\n</function>\n</tool_call>\n"
            "好的完成了"
        ),
        "ground_truth": "rotate_arm(action_type=逆时针旋转, angle_deg=40.0)",
        "notes": "T17 回归：闭合后自然语言尾随仍 W 不标 E（与 T33 双调用"
        "的区别：非 <tool_call> 开头）",
    },
]


def main() -> None:
    rows = [
        json.loads(line)
        for line in V2_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    out = []
    for case in rows:
        if case["type"] == "tool_call":
            out.append(convert_t_case(case))
        else:
            out.append(dict(case))  # J 系列原样
    out.extend(NEW_CASES)
    for case in out:
        annotate_case(case)
    # id 排序输出（T 在前 J 在后，保持 v2 顺序）
    out.sort(key=lambda c: c["id"])
    for case in out:
        print(json.dumps(case, ensure_ascii=False))
    n_t = sum(1 for c in out if c["type"] == "tool_call")
    n_j = len(out) - n_t
    print(f"# {len(out)} 条（T {n_t} + J {n_j}）", file=sys.stderr)


if __name__ == "__main__":
    main()
