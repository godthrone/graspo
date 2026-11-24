#!/usr/bin/env python3
"""合成长序列的 **token 实测器** —— 用 ms-swift 真 collator 量出真实 token 数（不上机亦可跑）。

**职责边界**

- **本模块**：用**本仓自己的适配点**（``graspo.flow.backend.msswift.dataset.build_ms_swift_messages``）
  + **ms-swift 自己的 Template / 真 collator**，量出
  ① T013 子集里**实测最短**的基样下标；② 不同复制份数 N 下的**实际 token 数**（标定点）；
  ③ 已生成档的**实际 token 数**（复核"目标长度真的达到了"）。
- **不负责**：构造合成长序列（``build_synth_seq.py``）、上机训练（``run_synth.sh``）。

**为什么必须实测（本项目的口径纪律：算式不是实测）**

``data.max_prompt_length`` 只是 ms-swift ``--max_length``（**截断上限、不 pad**）。真实长度由
数据内容 + chat template + **图像 token 预算**（ms-swift 的 ``image_max_token_num``，
本档实测 ``Setting image_max_token_num: 16384``）+ 截断策略共同决定 ⇒ 没有任何算式能替代实测。
本脚本用 batch=1 的**真 collator**（无 padding ⇒ ``input_ids`` 的宽度就是该样本真实长度），
并且**关闭截断**（``max_length=10^9``）——否则量到的是"被裁到上限的长度"，不是真实长度。

**三种模式**

| 模式 | 作用 | 产物 |
|---|---|---|
| ``--find-min`` | 量源子集每条的**真实 token 数**，报出最短那条的下标 | stdout（含逐条读数） |
| ``--calibrate`` | 在**临时文件**上构造 N = 1, 2, 4 份的合成行并实测其 token 数 ⇒ 标定点
| ``--out`` 的 JSON（半成品，不含最终档） |
| （默认） | 量 ``--subset`` 指向的 JSONL 每条的**真实 token 数** ⇒ 档复核 | ``--out`` 的 JSON |

**标定的两步流程（为什么不是一步）**

``build_synth_seq.py`` 的产物文件名固定为 ``<tier-id>.jsonl``（``run_matrix54.sh`` 的路径一致性
强制），所以"每个档一个 JSONL"在**同一变体的同一目录**里只留最后一个档。因此：

1. **标定步**：本脚本用 ``--calibrate`` 在**临时文件**里自造 N 个复制份数的样本、实测其长度，
   得到 ``(copies, tokens)`` 实测点 ⇒ ``measure_synth_calibration.json``；
2. **生成步**：``build_synth_seq.py`` 用这些**实测点**插值/外推定 N，逐档落盘；
3. **复核步**：本脚本（默认模式）逐档实测 ``<tier-id>.jsonl``，产出
   ``measured_tokens.json`` —— **这就是"目标长度真的达到了"的判据**。

**用法（容器内；在远端机上经 runner 的镜像跑，或本机有 ms-swift 环境时直接跑）**

    # ① 找最短基样并标定（含 N=1,2,4 三个实测点）
    python3 measure_synth_tokens.py --source /work/elam-v5/subsets/T013.jsonl \\
        --model /models/Qwen3.5-9B --calibrate --out /out/measure_synth_calibration.json

    # ② 复核某一档（或逐档，见 run_synth.sh 的 verify 阶段）
    python3 measure_synth_tokens.py --subset /out/synth/T013.jsonl \\
        --model /models/Qwen3.5-9B --out /out/measured_tokens.json

**依赖**：ms-swift（``swift``）+ ``torch`` + 本仓 ``graspo``（即容器内环境）。
**注意**：``load_model=False`` ⇒ **不加载权重、不占 GPU 显存**；但仍需模型目录含
tokenizer/processor/template 元信息（远端机上 ``/models/Qwen3.5-9B`` 满足）。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

#: 标定用的复制份数点（覆盖小 N，用于拟合"份数 → 长度"的线性关系）。
CALIBRATION_COPIES = [1, 2, 4]

#: 关闭截断的量法：max_length 取 10^9（否则量到的是被裁到上限的长度）。
NO_TRUNCATION = 10**9


def _load_encoder(model: str, max_length: int = NO_TRUNCATION):
    """构造 ms-swift 真编码链路（Template + 真 collator），**不加载权重**。

    与 ``task-r3-lenramp/rig/probe_seq_len.py``（已被验证的量法）逐字同源：
    同一 ``get_model_processor`` / ``get_template`` / ``template.data_collator`` 入口。
    """
    import torch
    from swift.model import get_model_processor
    from swift.template import get_template, get_template_meta

    # 与已验证的量法（task-r3-lenramp/rig/probe_seq_len.py）逐字同源：同一入口、同一 dtype。
    # `load_model=False` ⇒ 只拿 processor（tokenizer/图像处理器 + template 元信息），
    # **不加载权重**。
    _model, processor = get_model_processor(model, torch_dtype=torch.bfloat16, load_model=False)
    template_meta = get_template_meta(processor.model_info, processor.model_meta)
    template = get_template(
        processor,
        template_type=template_meta.template_type,
        max_length=max_length,
        truncation_strategy="delete",
    )
    template.init_processor(processor)
    return template


def _encode_len(template, messages: list, collator) -> int:
    """把 messages 过模板 + 真 collator（batch=1）⇒ 返回真实 token 数。

    batch=1 是本量的关键：``data_collator`` 在 ``padding_to`` 不传时把 batch pad 到
    **batch 内最长样本**；batch=1 ⇒ **无 padding**，张量宽度即该样本真实长度。
    """
    fed = template.encode({"messages": messages})
    batch = collator([fed]) if isinstance(fed, dict) else collator([fed[0]])
    return int(batch["input_ids"].shape[1])


def _iter_records(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            yield line_no, json.loads(line)


def _measure_source(path: Path, template, collator) -> list[dict]:
    """量源 JSONL 每条的**真实** token 数（走本仓自己的适配点）。"""
    from graspo.flow.backend.msswift.dataset import build_ms_swift_messages

    # 相对媒体路径按 JSONL 父目录解析（项目锚点口径，与训练时一致）。
    os.chdir(str(path.parent.resolve()))
    results: list[dict] = []
    for line_no, raw in _iter_records(path):
        entry: dict = {"line_no": line_no, "id": raw.get("id")}
        try:
            messages = build_ms_swift_messages(raw["messages"])
            entry["tokens"] = _encode_len(template, messages, collator)
            entry["image_blocks"] = _count_media(raw["messages"])
        except Exception as exc:  # noqa: BLE001 - 只读探针，任何异常都要看得见
            entry["tokens"] = None
            entry["error"] = f"{type(exc).__name__}: {str(exc)[:200]}"
        results.append(entry)
    return results


def _count_media(messages: list) -> int:
    total = 0
    for message in messages:
        if not isinstance(message, dict):
            continue
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for block in content:
            if isinstance(block, dict) and str(block.get("type") or "").lower() in {
                "image",
                "image_url",
                "video",
                "video_url",
            }:
                total += 1
    return total


def _summarize(results: list[dict]) -> dict:
    values = sorted(entry["tokens"] for entry in results if entry.get("tokens"))
    if not values:
        return {"samples": len(results), "measurable": 0}
    return {
        "samples": len(results),
        "measurable": len(values),
        "min": values[0],
        "median": values[len(values) // 2],
        "max": values[-1],
    }


def _shortest(results: list[dict]) -> dict:
    usable = [entry for entry in results if entry.get("tokens")]
    if not usable:
        raise RuntimeError("没有任何可测量的样本（全部报错？）")
    return min(usable, key=lambda entry: entry["tokens"])


def run_find_min(args: argparse.Namespace) -> int:
    source = Path(args.source).resolve()
    template = _load_encoder(args.model)
    collator = template.data_collator
    results = _measure_source(source, template, collator)
    summary = _summarize(results)
    best = _shortest(results)
    print(f"source={source}")
    print(f"summary={json.dumps(summary, ensure_ascii=False)}")
    print(
        f"shortest: line_no={best['line_no']} id={best.get('id')} "
        f"tokens={best['tokens']} image_blocks={best.get('image_blocks')}"
    )
    print("per_sample =", [entry.get("tokens") for entry in results])
    print(
        f"⚠ line_no 是 **1-based**；build_synth_seq.py 的 --base-index 是 **0-based**，"
        f"⇒ --base-index {best['line_no'] - 1}"
    )
    if args.out:
        Path(args.out).write_text(
            json.dumps(
                {
                    "mode": "find-min",
                    "source": str(source),
                    "measured": True,
                    "summary": summary,
                    "shortest": best,
                    "per_sample": results,
                },
                ensure_ascii=False,
                indent=1,
            ),
            encoding="utf-8",
        )
    return 0


def run_calibrate(args: argparse.Namespace) -> int:
    """在**临时文件**上构造 N = CALIBRATION_COPIES 个合成行并实测 ⇒ 标定点。"""
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from build_synth_seq import build_synth_record  # 单一构造点，不复制第二套逻辑

    source = Path(args.source).resolve()
    template = _load_encoder(args.model)
    collator = template.data_collator
    from graspo.flow.backend.msswift.dataset import build_ms_swift_messages

    results = _measure_source(source, template, collator)
    summary = _summarize(results)
    best = _shortest(results)
    base_index = best["line_no"] - 1  # 1-based → 0-based
    print(f"source={source}")
    print(f"summary={json.dumps(summary, ensure_ascii=False)}")
    print(f"base(shortest): index0={base_index} id={best.get('id')} tokens={best['tokens']}")

    rows = [raw for _, raw in _iter_records(source)]
    base = rows[base_index]

    # 临时目录里量：媒体相对路径要能解析 ⇒ cwd 设到源 JSONL 父目录。
    # 本步**不落盘**（合成行直接在内存里过模板），因此不会在磁盘留下任何中间 JSONL。
    os.chdir(str(source.parent.resolve()))
    points: list[dict] = []
    for copies in CALIBRATION_COPIES:
        record, provenance = build_synth_record(
            base,
            copies=copies,
            variant=args.variant,
            assistant_content=args.assistant_content,
        )
        try:
            messages = build_ms_swift_messages(record["messages"])
            tokens = _encode_len(template, messages, collator)
            points.append(
                {
                    "copies": copies,
                    "tokens": tokens,
                    "media_blocks": provenance["output_media_block_count"],
                }
            )
            print(
                f"  copies={copies:4d} tokens={tokens:8d} "
                f"media_blocks={provenance['output_media_block_count']}"
            )
        except Exception as exc:  # noqa: BLE001
            print(f"  copies={copies:4d} ERROR {type(exc).__name__}: {str(exc)[:200]}")

    if len(points) < 2:
        print("FATAL: 有效实测点 < 2，标定不可用（需要至少两个 copies 点做插值）", file=sys.stderr)
        return 2

    # 线性拟合的斜率（tokens per copy）也一并登记，便于人工核对是否线性。
    if len(points) >= 2:
        first, last = points[0], points[-1]
        slope = (last["tokens"] - first["tokens"]) / (last["copies"] - first["copies"])
        print(
            f"measured slope = {slope:.1f} tokens per copy "
            f"（线性性核对：若各点偏离直线过多，请改用报告里的插值法逐点处理）"
        )
    else:
        slope = None

    doc = {
        "mode": "calibrate",
        "measured": True,
        "measured_by": "measure_synth_tokens.py（ms-swift 真 collator，batch=1，无截断）",
        "source": str(source),
        "model": args.model,
        "variant": args.variant,
        "base_index": base_index,
        "base_id": base.get("id"),
        "base_tokens_measured": best["tokens"],
        "source_length_summary": summary,
        "points": points,
        "slope_tokens_per_copy": slope,
        "no_truncation_max_length": NO_TRUNCATION,
        "caveat": (
            "★ 真实长度分布：T013 子集实测 min/median/max = 1941/1943/2001；"
            "本标定只把'最短那条'放大，不代表原始分布。"
        ),
    }
    if args.out:
        Path(args.out).write_text(json.dumps(doc, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"wrote {args.out}")
    return 0


def run_measure(args: argparse.Namespace) -> int:
    """复核：量 ``--subset`` 每条的**真实** token 数（每档 1 行 ⇒ 报露出该档长度）。"""
    subset = Path(args.subset).resolve()
    template = _load_encoder(args.model)
    collator = template.data_collator
    results = _measure_source(subset, template, collator)
    summary = _summarize(results)
    print(f"subset={subset}")
    print(f"summary={json.dumps(summary, ensure_ascii=False)}")
    for entry in results:
        print(
            f"  line {entry['line_no']:3d}: id={entry.get('id')} "
            f"tokens={entry.get('tokens')} image_blocks={entry.get('image_blocks')}"
        )
    doc = {
        "mode": "measure",
        "measured": True,
        "measured_by": "measure_synth_tokens.py（ms-swift 真 collator，batch=1，无截断）",
        "subset": str(subset),
        "model": args.model,
        "summary": summary,
        "per_sample": results,
    }
    if args.out:
        Path(args.out).write_text(json.dumps(doc, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"wrote {args.out}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="合成长序列 token 实测器（ms-swift 真 collator）")
    parser.add_argument("--source", type=Path, help="源 T013 子集 JSONL（find-min / calibrate 用）")
    parser.add_argument("--subset", type=Path, help="要复核的 JSONL（默认模式用）")
    parser.add_argument("--model", default="/models/Qwen3.5-9B", help="模型目录（只读 processor）")
    parser.add_argument(
        "--variant",
        default="v1-image-copy",
        help="标定用的变体（须与生成时一致！两个变体不可混用一份标定）",
    )
    parser.add_argument("--assistant-content", default="OK", help="最小 assistant 轮内容")
    parser.add_argument("--calibrate", action="store_true", help="标定模式（含 find-min）")
    parser.add_argument("--find-min", action="store_true", help="只找最短基样")
    parser.add_argument("--out", type=Path, help="JSON 产物路径")
    args = parser.parse_args(argv)

    if args.calibrate:
        if not args.source:
            parser.error("--calibrate 需要 --source")
        return run_calibrate(args)
    if args.find_min:
        if not args.source:
            parser.error("--find-min 需要 --source")
        return run_find_min(args)
    if not args.subset:
        parser.error("默认模式需要 --subset（或用 --calibrate / --find-min）")
    return run_measure(args)


if __name__ == "__main__":
    raise SystemExit(main())
