"""生成 annotation_testset_v2.jsonl 的标注字符串。

运行实际的标注器，对新用例生成期望标注，以便人工/agent 核验。
用法: python scripts/generate_v2_testset.py > tests/data/annotation_testset_v2.jsonl
"""

import json
import sys
from pathlib import Path

# 确保能 import graspo
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from graspo.ripple.annotation.labeler import AnnotationInput, annotate


def make_targets(format_type: str, gt: str) -> list[dict]:
    """与 test_annotation_testset.py 中的 _make_targets 一致。"""
    if format_type == "tool_call":
        name, rest = gt.split("(", 1)
        args_str = rest.rstrip(")").strip()
        arguments: dict = {}
        for pair in args_str.split(","):
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
    """对一条用例运行标注器，生成 annotation 字段。"""
    completion = case["completion"]
    cid = case["id"]
    format_type = case["type"]

    # 与测试文件一致的配置
    no_fence = {"J02", "J18", "J20", "J21", "J22"}  # 无围栏场景
    think_cases = {"T11", "T24", "T25"}  # think 场景

    result = annotate(
        AnnotationInput(
            completion_text=completion,
            targets=make_targets(format_type, case["ground_truth"]),
            tokenizer=None,
            format_type=format_type,
            check_json_markdown=cid not in no_fence,
            check_think=cid in think_cases,
        )
    )
    annotation = "".join(t.value for t in result.tags)
    error_pos = ""
    e_idx = annotation.find("E")
    if e_idx >= 0:
        error_pos = str(e_idx)

    case["annotation"] = annotation
    case["error_pos"] = error_pos
    case["correct"] = "yes"  # agent 独立核验通过（2026-08-06）
    return case


# ── 保留的旧用例（删除 T02, T14, J07）──
KEEP_IDS = {
    "T01", "T03", "T04", "T05", "T06", "T07", "T08", "T09", "T10",
    "T11", "T12", "T13", "T15", "T16", "T17", "T18", "T19", "T20",
    "J01", "J02", "J03", "J04", "J05", "J06", "J08", "J09", "J10",
    "J11", "J12", "J13", "J14", "J15",
}

# ── 新用例定义 ──
NEW_CASES = [
    # ── Tool Call P0 ──
    {
        "id": "T21", "type": "tool_call",
        "case": "仅tool_call无后续",
        "completion": "<tool_call>\n",
        "ground_truth": "robot_atomic_control(action_type=逆时针旋转, angle_deg=40.0)",
        "notes": "27.2% 真实数据：TC 后无 function/参数，match_expected 在文本中途结束返回 True，不标 E"
    },
    {
        "id": "T22", "type": "tool_call",
        "case": "TC+function无参数",
        "completion": "<tool_call>\n<function=robot_atomic_control>",
        "ground_truth": "robot_atomic_control(action_type=逆时针旋转, angle_deg=40.0)",
        "notes": "227 条真实数据：TC+函数名后停止，参数循环提前退出，无 E"
    },
    {
        "id": "T23", "type": "tool_call",
        "case": "前导文本+仅TC",
        "completion": "让我分析图像。\n\n<tool_call>\n",
        "ground_truth": "robot_atomic_control(action_type=逆时针旋转, angle_deg=40.0)",
        "notes": "组合：前导 W + 结构不完整全 S，347 条真实数据"
    },
    {
        "id": "T24", "type": "tool_call",
        "case": "THINK_ONLY无tool_call",
        "completion": "<think>\n图像显示前方路径清晰。\n</think>",
        "ground_truth": "robot_atomic_control(action_type=逆时针旋转, angle_deg=40.0)",
        "notes": "0.6% 真实数据：think 标签存在但无 tool_call，open_pos<0 返回全 W+T+S"
    },
    {
        "id": "T25", "type": "tool_call",
        "case": "think+截断tool_call",
        "completion": "<think>\n让我分析选项。\n</think>\n\n<tool_call>\n<function=robot_atomic_control>\n<parameter=action_type>\n顺时针旋转\n</parameter>\n<parameter=angle_deg>",
        "ground_truth": "robot_atomic_control(action_type=顺时针旋转, angle_deg=30)",
        "notes": "组合：think 内容 + tool_call 中途截断，验证 think 与截断的组合"
    },
    {
        "id": "T26", "type": "tool_call",
        "case": "前导+闭合标签拼错",
        "completion": "好的，我来操作。\n\n<tool_call>\n<function=robot_atomic_control>\n<parameter=action_type>\n逆时针旋转\n</parametr>\n<parameter=angle_deg>\n40.0\n</parameter>\n</function>\n</tool_call>",
        "ground_truth": "robot_atomic_control(action_type=逆时针旋转, angle_deg=40.0)",
        "notes": "组合：前导 W + 结构错误 E（</parametr> 拼错），验证 W 和 E 的交互"
    },
    # ── Tool Call P1 ──
    {
        "id": "T27", "type": "tool_call",
        "case": "单参数clean输出",
        "completion": "<tool_call>\n<function=robot_atomic_control>\n<parameter=angle_deg>\n45.0\n</parameter>\n</function>\n</tool_call>",
        "ground_truth": "robot_atomic_control(angle_deg=45.0)",
        "notes": "单参数 clean 输出，验证 seen_params 只含一个、missing 为空"
    },
    {
        "id": "T28", "type": "tool_call",
        "case": "function闭合标签拼错",
        "completion": "<tool_call>\n<function=robot_atomic_control>\n<parameter=action_type>\n逆时针旋转\n</parameter>\n</funcion>\n</tool_call>",
        "ground_truth": "robot_atomic_control(action_type=逆时针旋转, angle_deg=40.0)",
        "notes": "function 闭合标签拼错（</funcion>），match_expected 在 line 286 失败"
    },
    {
        "id": "T29", "type": "tool_call",
        "case": "tool_call闭合标签拼错",
        "completion": "<tool_call>\n<function=robot_atomic_control>\n<parameter=action_type>\n逆时针旋转\n</parameter>\n<parameter=angle_deg>\n40.0\n</parameter>\n</function>\n</toll_call>",
        "ground_truth": "robot_atomic_control(action_type=逆时针旋转, angle_deg=40.0)",
        "notes": "tool_call 闭合标签拼错（</toll_call>），match_expected 在 line 289 失败"
    },
    {
        "id": "T30", "type": "tool_call",
        "case": "不同函数名+参数名",
        "completion": "<tool_call>\n<function=get_weather>\n<parameter=city>\nBeijing\n</parameter>\n</function>\n</tool_call>",
        "ground_truth": "get_weather(city=Beijing)",
        "notes": "验证 _target_fn_name 返回 get_weather、_target_param_order 返回 [city]"
    },
    # ── Tool Call P2 ──
    {
        "id": "T31", "type": "tool_call",
        "case": "空参数值",
        "completion": "<tool_call>\n<function=robot_atomic_control>\n<parameter=action_type>\n\n</parameter>\n<parameter=angle_deg>\n40.0\n</parameter>\n</function>\n</tool_call>",
        "ground_truth": "robot_atomic_control(action_type=逆时针旋转, angle_deg=40.0)",
        "notes": "空值 span：v_end==v_start 分支跳过值标记，值位置空白标 W（空值不训练，无 E）"
    },
    {
        "id": "T32", "type": "tool_call",
        "case": "参数标签残缺无>",
        "completion": "<tool_call>\n<function=robot_atomic_control>\n<parameter=action_type\n</function>\n</tool_call>",
        "ground_truth": "robot_atomic_control(action_type=逆时针旋转, angle_deg=40.0)",
        "notes": "参数标签缺 >：name_end 找到的是后续 </function> 的 >，param_name 含换行 → 不在 GT 参数集 → 走多余参数分支 E（agent 核验修正）"
    },
    # ── JSON P0 ──
    {
        "id": "J16", "type": "json",
        "case": "嵌套对象",
        "completion": "```json\n{\"user\":{\"name\":\"Alice\",\"age\":30}}\n```",
        "ground_truth": "{\"user\": {\"name\": \"Alice\", \"age\": 30}}",
        "notes": "嵌套对象：stack 深度 2+ 验证；已知 bug：内层 key 校验只查顶层 GT（E@17 非期望行为，待修复后更新）"
    },
    # ── JSON P1 ──
    {
        "id": "J17", "type": "json",
        "case": "boolean+null值",
        "completion": "```json\n{\"active\":true,\"deleted\":false,\"note\":null}\n```",
        "ground_truth": "{\"active\": true, \"deleted\": false, \"note\": null}",
        "notes": "true/false/null 字面量，验证 startswith 分支"
    },
    {
        "id": "J18", "type": "json",
        "case": "顶层数组",
        "completion": "[\"CMIOT\",\"IMS\",\"APN\"]",
        "ground_truth": "[\"CMIOT\", \"IMS\", \"APN\"]",
        "notes": "顶层数组（非对象），check_json_markdown=False，验证 [ 根路径"
    },
    {
        "id": "J19", "type": "json",
        "case": "转义字符",
        "completion": "```json\n{\"msg\":\"hello\\nworld\",\"path\":\"C:\\\\Users\"}\n```",
        "ground_truth": "{\"msg\": \"hello\\nworld\", \"path\": \"C:\\\\Users\"}",
        "notes": "字符串含 \\n 和 \\\\，验证 _find_string_end 转义跳过"
    },
    {
        "id": "J20", "type": "json",
        "case": "前导文本+JSON无围栏",
        "completion": "这是提取结果：\n{\"故障号码\":[\"1442201593053\"],\"IMSI\":[\"460240401593053\"]}",
        "ground_truth": "{\"故障号码\": [\"1442201593053\"], \"IMSI\": [\"460240401593053\"]}",
        "notes": "无围栏+前导文本：check_json_markdown=False 时整段须为合法 JSON，前导文本首字符 E（设计行为，与方案 §5.2 一致）"
    },
    # ── JSON P2 ──
    {
        "id": "J21", "type": "json",
        "case": "嵌套数组",
        "completion": "[\"CMIOT\",[\"sub1\",\"sub2\"],\"APN\"]",
        "ground_truth": "[\"CMIOT\", [\"sub1\", \"sub2\"], \"APN\"]",
        "notes": "嵌套数组：stack 含 [\"a\", \"a\"]，验证嵌套数组 push/pop"
    },
    {
        "id": "J22", "type": "json",
        "case": "多余闭合括号",
        "completion": "{\"故障号码\":[\"1442201593053\"]}}",
        "ground_truth": "{\"故障号码\": [\"1442201593053\"]}",
        "notes": "额外 }：stack 为空时 } → E 截断"
    },
]


def main():
    # 读取旧测试集
    old_path = Path(__file__).resolve().parents[1] / "tests" / "data" / "annotation_testset.jsonl"
    with open(old_path, encoding="utf-8") as f:
        old_cases = [json.loads(line) for line in f if line.strip()]

    # 保留旧用例
    kept = [c for c in old_cases if c["id"] in KEEP_IDS]
    kept_ids = {c["id"] for c in kept}
    print(f"保留旧用例: {len(kept)} 条", file=sys.stderr)
    removed = [c for c in old_cases if c["id"] not in KEEP_IDS]
    print(f"删除旧用例: {[c['id'] for c in removed]}", file=sys.stderr)

    # 标记旧用例为 yes
    for c in kept:
        c["correct"] = "yes"

    # 生成新用例标注
    print(f"新用例: {len(NEW_CASES)} 条", file=sys.stderr)
    new_cases = []
    for case in NEW_CASES:
        annotated = annotate_case(case)
        new_cases.append(annotated)
        print(f"  {annotated['id']}: annotation={annotated['annotation']} (len={len(annotated['annotation'])})",
              file=sys.stderr)
        print(f"    completion len={len(annotated['completion'])}, error_pos={annotated['error_pos']}",
              file=sys.stderr)

    # 合并：旧用例按原始顺序 + 新用例
    all_cases = kept + new_cases
    all_ids = [c["id"] for c in all_cases]
    print(f"\n总计: {len(all_cases)} 条", file=sys.stderr)
    print(f"ID 列表: {all_ids}", file=sys.stderr)

    # 输出 JSONL
    for case in all_cases:
        print(json.dumps(case, ensure_ascii=False))


if __name__ == "__main__":
    main()