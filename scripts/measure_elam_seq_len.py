#!/usr/bin/env python3
"""真实测量 ELAM 数据集单条样本的 `input_ids` 长度（只加载 processor，不加载权重、不训练）。

职责（宪法 §1.6 文件边界声明）
--------------------------------
本文件只做一件事：把 JSONL 数据集逐条喂给**真实的多模态 processor**，量出
`input_ids` 的真实长度，并输出结构化 JSON（逐条长度 + 分布统计 + 分量分解 + 截断判定）。
它**不读训练配置、不改数据集、不加载模型权重、不做任何训练/推理**。

为什么必须"真实测"，而不能套用默认值或估算
--------------------------------------------
1. **图像 token 数不是常数，取决于分辨率 × processor 行为。** Qwen3-VL 系列在
   `smart_resize` 之后按 `patch_size=16` / `merge_size=2` 切开，
   单图图像 token 数 = `(H/16/2) * (W/16/2)`。同一张 1280×720 的图，
   上游是否给了像素预算，结果可以差一个数量级。
2. **本项目当前的真实口径是"不限制像素" ⇒ 图像会被上采样、token 数偏高。**
   `src/graspo/core/schema.py:712` 的 `max_pixels: int | None = None` 语义是
   **不透传**（注释逐字："None = 不透传（交 ms-swift 默认值 = 不限制）"），
   且档位 YAML 无一设置 `max_pixels`。
   ⇒ 实测 1280×720 被 `smart_resize` 放大到 **2560×1408**、grid `[1,44,80]`、
   **880 token/图**——这是当前的**真实训练口径**，不是理论值。
3. **文字/模板/工具定义的分量也不可忽略**：同一 mini 集的实测分解为
   2 图 1760 + 文本/模板/工具 620 = **2380**（纯文本 616）。
   只算图像会低估约 26%，据此选上下文会选错。
4. **默认值会骗人。** 若按"名义上下文 16384"或"720p≈几百 token"的直觉估算，
   会得出"2048 够用"的错误结论；实测是 **100/100 条在 2048 下全部被截断**。
   ⇒ 上下文选型必须以实测为准（宪法 §1.4 单一真相源：长度这个事实只有一个来源 = 真 processor）。

典型用法
--------
    # mini 集全量（100 条）实测，JSON 落盘到 .local/（产物不入 git，宪法 §16.1）
    python3 scripts/measure_elam_seq_len.py \\
        --jsonl   .local/mini-dataset/mini/mini-short-mm-train.jsonl \\
        --model-dir /path/to/Qwen3.5-9B \\
        --out     .local/hb-workspace/<ts>/task-x/artifacts/measure_result.json

    # 只跑前 3 条做冒烟（不上机也需要 GPU 以外的依赖：transformers + torch + PIL）
    python3 scripts/measure_elam_seq_len.py --jsonl <jsonl> --model-dir <dir> --limit 3

    # 把输入钉死到确切的字节（推荐；见下方"已知的 hash 争议"）
    ... --expect-sha256 594d41b0b8af3dc77cfad45768eca116c3eb1e2314ce33b19ac3e12052669cad

**已知的 hash 争议（务必按实测值，不要照抄历史数字）**：mini 数据集
（`.local/mini-dataset/mini/mini-short-mm-train.jsonl`，100 行 / 208822 B）的
**权威 sha256 = `594d41b0b8af3dc77cfad45768eca116c3eb1e2314ce33b19ac3e12052669cad`**
（两个独立来源一致：① 磁盘文件 `sha256sum`；② `task-d2-token-measure` 的机器产物
`artifacts/measure_result.json` 的 `mini_sha256_actual`）。
历史文档/工作包里流传的 `...cfdc...` 是**被误抄的 `expected` 值**——该包的产物
`mini_sha256_expected` 恰为 `...cfdc...`，而其 `notes` 里逐字记着
`SHA256_MISMATCH: actual != expected given in briefing`。用 `--expect-sha256` 钉住
实测值，可以让"这次测量的长度到底对应哪份字节"变成可机核的事实。

退出码
------
    0 = 成功（全部样本测得）
    2 = 用法/环境错误（缺依赖、路径不存在、`--expect-sha256` 不匹配、图像缺失且未授权跳过）
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
from pathlib import Path
from typing import Any

# `scripts/` 不是包（无 `__init__.py`），但本脚本既要能"直接执行"，又要能被 importlib
# 按文件路径加载；自插脚本目录是同时满足两种加载方式的唯一稳妥写法（§2.2 显式依赖）。
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from errors import UsageError  # noqa: E402  （必须在上面的 sys.path 注入之后）

DEFAULT_TRUNCATION_LEVELS = (1024, 2048, 4096)
HISTOGRAM_BIN_WIDTH = 128


# --------------------------------------------------------------------------- 输入


def sha256_of(path: Path) -> str:
    """流式求 sha256（不整份读进内存；大 JSONL 也安全）。"""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_samples(jsonl_path: Path, limit: int | None) -> list[dict[str, Any]]:
    """逐行读 JSONL。`limit=None` 表示全量；空行跳过但不静默吞掉坏行。"""
    samples: list[dict[str, Any]] = []
    with jsonl_path.open(encoding="utf-8") as handle:
        for lineno, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                samples.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise UsageError(f"{jsonl_path}:{lineno} 不是合法 JSON：{exc}") from exc
            if limit is not None and len(samples) >= limit:
                break
    if not samples:
        raise UsageError(f"{jsonl_path}: 没有任何样本（空文件或全为空白行）")
    return samples


def to_hf_messages(
    sample: dict[str, Any], warnings: list[str]
) -> tuple[list[dict], list[str], Any]:
    """把数据集行转成 HF 多模态消息，并返回（消息, 图像相对路径, tools）。

    **逐字透传**图像路径与文本，不做任何改写——路径按数据集约定相对 `--image-root` 解析。
    未知的 content 类型不静默丢弃，记一条 warning（宪法 §3.2 透明退路）。
    """
    messages: list[dict] = []
    image_paths: list[str] = []
    for message in sample.get("messages", []):
        role = message.get("role")
        content = message.get("content")
        if isinstance(content, str):
            messages.append({"role": role, "content": [{"type": "text", "text": content}]})
            continue
        parts: list[dict] = []
        for item in content or []:
            item_type = item.get("type")
            if item_type == "image":
                parts.append({"type": "image"})
                image_paths.append(item["image"])
            elif item_type == "text":
                parts.append({"type": "text", "text": item["text"]})
            else:
                warnings.append(f"id={sample.get('id')} 未知 content 类型 {item_type!r}（已跳过）")
        messages.append({"role": role, "content": parts})
    return messages, image_paths, sample.get("tools")


def resolve_images(image_paths: list[str], image_root: Path) -> list[Path]:
    """把数据集里的相对路径解析为磁盘路径。

    数据集约定（见 `scripts/build_mini_elam_subset.py`）：图像字段形如
    `../images/xxx.jpg`，**相对数据根目录**解析；mini 文件落在与 `images/` 同级的
    子目录中，故默认 `--image-root` = JSONL 所在目录，`../images/` 恰好命中。
    """
    resolved: list[Path] = []
    for raw in image_paths:
        candidate = Path(raw)
        if candidate.is_absolute():
            resolved.append(candidate)
        else:
            resolved.append((image_root / candidate).resolve())
    return resolved


# --------------------------------------------------------------------------- 测量


def load_processor(model_dir: Path, max_pixels: int | None) -> tuple[Any, str, dict[str, Any]]:
    """**只**加载 processor（tokenizer + image processor），绝不加载模型权重。

    返回（processor, 加载方式描述, 模型 config 摘要）。
    """
    try:
        import transformers
    except ImportError as exc:  # pragma: no cover - 环境相关
        raise UsageError(
            "缺少 `transformers`。本工具必须在有 transformers + torch + pillow 的环境里运行"
            "（例如项目容器 `graspo-msswift:4.5.3`）；本机缺依赖时只能做静态检查。"
        ) from exc

    config_path = model_dir / "config.json"
    if not config_path.is_file():
        raise UsageError(f"模型目录缺少 config.json：{model_dir}")
    model_config = json.loads(config_path.read_text(encoding="utf-8"))

    # 显式传 max_pixels（None ⇒ 不透传，与 schema.py:712 的语义保持同一口径）。
    # 加载方式逐条记录进 JSON，避免"到底传没传"变成口头约定（§2.2 显式即防呆）。
    attempts: list[tuple[str, dict[str, Any]]] = []
    if max_pixels is None:
        attempts = [("Qwen3VLProcessor", {}), ("AutoProcessor", {})]
    else:
        attempts = [
            ("Qwen3VLProcessor", {"max_pixels": max_pixels}),
            ("AutoProcessor", {"max_pixels": max_pixels}),
        ]
    errors: list[str] = []
    for class_name, kwargs in attempts:
        processor_class = getattr(transformers, class_name, None)
        if processor_class is None:
            errors.append(f"{class_name} 不存在于 transformers {transformers.__version__}")
            continue
        try:
            processor = processor_class.from_pretrained(str(model_dir), **kwargs)
        except Exception as exc:  # noqa: BLE001 - 逐个候选降级，全部失败才报错
            errors.append(f"{class_name}({kwargs}) 失败：{type(exc).__name__}: {exc}")
            continue
        summary = {
            "processor_load_call": f"{class_name}.from_pretrained(**{kwargs})",
            "processor_class": type(processor).__name__,
            "transformers_version": transformers.__version__,
            "max_pixels_passed": max_pixels,
            "weights_loaded_any": False,
            "failed_attempts": errors,
        }
        return processor, summary["processor_load_call"], summary
    raise UsageError("无法加载任何 processor：" + "；".join(errors))


def _as_ids(raw: Any) -> list[int]:
    """把 `input_ids`（tensor / 嵌套 list / 单层 list）统一成 `list[int]`。"""
    if hasattr(raw, "tolist"):
        raw = raw.tolist()
    if raw and isinstance(raw[0], list):
        raw = raw[0]
    return [int(token) for token in raw]


def _shape_of(value: Any) -> list[int] | None:
    if value is None:
        return None
    if hasattr(value, "shape"):
        return [int(dim) for dim in value.shape]
    return [len(value)]


def measure_one(
    processor: Any,
    messages: list[dict],
    tools: Any,
    images: list[Path],
    image_token_id: int,
) -> dict[str, Any]:
    """复刻真实 collator 的编码路径：chat template → processor（文本 + 图像）。"""
    from PIL import Image  # 局部导入：仅真正测量时才需要

    template_kwargs: dict[str, Any] = {"tokenize": False, "add_generation_prompt": True}
    if tools:
        template_kwargs["tools"] = tools
    rendered = processor.apply_chat_template([messages], **template_kwargs)
    if isinstance(rendered, list):
        rendered = rendered[0]

    call_kwargs: dict[str, Any] = {"tokenize": True, "return_dict": True}
    if images:
        pil_images = [Image.open(path).convert("RGB") for path in images]
        inputs = processor(text=[rendered], images=pil_images, **call_kwargs)
    else:
        inputs = processor(text=[rendered], **call_kwargs)

    ids = _as_ids(inputs["input_ids"])
    image_tokens = sum(1 for token in ids if token == image_token_id)
    return {
        "total_input_ids": len(ids),
        "image_tokens": image_tokens,
        "non_image_tokens": len(ids) - image_tokens,
        "n_images": len(images),
        "image_grid_thw": inputs.get("image_grid_thw").tolist()
        if hasattr(inputs.get("image_grid_thw"), "tolist")
        else inputs.get("image_grid_thw"),
        "pixel_values_shape": _shape_of(inputs.get("pixel_values")),
        "rendered_chars": len(rendered),
        "rendered_prompt_prefix": rendered[:600],
    }


def build_breakdown(
    processor: Any, sample: dict[str, Any], image_root: Path, image_token_id: int
) -> dict[str, Any]:
    """分解一条样本：纯文本 / 单图 / 双图，用来回答"图像占多少、文本占多少"。"""
    warnings: list[str] = []
    messages, paths, tools = to_hf_messages(sample, warnings)
    resolved = resolve_images(paths, image_root)
    out: dict[str, Any] = {
        "sample_id": sample.get("id"),
        "image_paths": paths,
        "text_only": measure_one(processor, messages, tools, [], image_token_id),
        "full": measure_one(processor, messages, tools, resolved, image_token_id),
    }
    for index, path in enumerate(resolved, 1):
        out[f"only_image_{index}"] = measure_one(
            processor, messages, tools, [path], image_token_id
        )
    full = out["full"]
    text_only = out["text_only"]
    out["derived"] = {
        "image_tokens_per_image": [
            out[f"only_image_{index}"]["image_tokens"] for index in range(1, len(resolved) + 1)
        ],
        "image_tokens_total": full["image_tokens"],
        "non_image_tokens_with_images": full["non_image_tokens"],
        "text_only_total": text_only["total_input_ids"],
        "vision_boundary_overhead": full["total_input_ids"]
        - (text_only["total_input_ids"] + full["image_tokens"]),
    }
    return out


def summarise(
    lengths: list[int], levels: tuple[int, ...], per_sample: list[dict]
) -> dict[str, Any]:
    """分布统计 + 直方图 + 逐截断档判定（全部由逐条实测长度派生）。"""
    ordered = sorted(lengths)
    stats = {
        "count": len(ordered),
        "min": ordered[0],
        "p50": int(statistics.median(ordered)),
        "max": ordered[-1],
        "mean": round(statistics.fmean(ordered), 2),
        "stdev": round(statistics.stdev(ordered), 2) if len(ordered) > 1 else 0.0,
        "unique_lengths": sorted(set(ordered)),
    }
    stats["p90"] = ordered[min(len(ordered) - 1, max(0, round(0.90 * (len(ordered) - 1))))]
    stats["p99"] = ordered[min(len(ordered) - 1, max(0, round(0.99 * (len(ordered) - 1))))]

    bins: dict[int, int] = {}
    for value in ordered:
        start = (value // HISTOGRAM_BIN_WIDTH) * HISTOGRAM_BIN_WIDTH
        bins[start] = bins.get(start, 0) + 1
    histogram = [
        {"bin_start": start, "bin_end": start + HISTOGRAM_BIN_WIDTH - 1, "count": bins[start]}
        for start in sorted(bins)
    ]

    truncation = []
    for level in levels:
        over = [row for row in per_sample if row.get("ok") and row["total_input_ids"] > level]
        truncation.append(
            {
                "context": level,
                "would_truncate_count": len(over),
                "fits_count": len(ordered) - len(over),
                "truncation_rate_pct": round(100.0 * len(over) / len(ordered), 1),
                "max_overrun_tokens": max(
                    (row["total_input_ids"] - level for row in over), default=0
                ),
                "first_failing_samples": [
                    {"line": row["line"], "id": row.get("id"), "total": row["total_input_ids"]}
                    for row in over[:5]
                ],
            }
        )
    return {
        "length_stats": stats,
        "histogram_bin_width": HISTOGRAM_BIN_WIDTH,
        "histogram": histogram,
        "truncation": truncation,
        "theoretical_min_context": ordered[-1],
    }


# --------------------------------------------------------------------------- CLI


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="measure_elam_seq_len.py",
        description=(
            "只加载 processor 真实测量 ELAM JSONL 的 input_ids 长度（结构化 JSON 输出）。"
            "不加载模型权重、不训练。"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--jsonl", required=True, type=Path, help="输入数据集 JSONL（逐行一个样本）")
    parser.add_argument("--model-dir", required=True, type=Path, help="模型目录（只用于加载 processor）")
    parser.add_argument(
        "--limit", type=int, default=None, help="只测前 N 条（默认全量）；冒烟时用 3"
    )
    parser.add_argument(
        "--image-root",
        type=Path,
        default=None,
        help="图像相对路径的解析基准（默认 = JSONL 所在目录；数据集约定 ../images/ 相对数据根）",
    )
    parser.add_argument(
        "--max-pixels",
        type=int,
        default=None,
        help=(
            "传给 processor 的像素预算。默认不传（与 schema.py:712 的 `None = 不透传` 同口径）；"
            "若要复核「限定像素」的影响，显式给值（例如 1048576）"
        ),
    )
    parser.add_argument(
        "--truncation-levels",
        default=",".join(str(level) for level in DEFAULT_TRUNCATION_LEVELS),
        help="逗号分隔的候选上下文长度（默认 1024,2048,4096）",
    )
    parser.add_argument(
        "--expect-sha256",
        default=None,
        help="期望的 JSONL sha256；不匹配即 fail-closed（把测量钉到确切的输入字节）",
    )
    parser.add_argument(
        "--skip-missing-images",
        action="store_true",
        help=(
            "图像缺失时跳过该条并记 WARNING（透明退路，默认关闭 = fail-closed）。"
            "跳过会改变结论，只在明确接受时才打开。"
        ),
    )
    parser.add_argument("--out", type=Path, default=None, help="JSON 输出路径（默认打印到 stdout）")
    parser.add_argument("--indent", type=int, default=2, help="JSON 缩进（默认 2）")
    return parser


def parse_levels(raw: str) -> tuple[int, ...]:
    levels: list[int] = []
    for piece in raw.split(","):
        piece = piece.strip()
        if not piece:
            continue
        try:
            levels.append(int(piece))
        except ValueError as exc:
            raise UsageError(f"--truncation-levels 含非整数：{piece!r}") from exc
    if not levels:
        raise UsageError("--truncation-levels 为空")
    return tuple(sorted(set(levels)))


def run(args: argparse.Namespace) -> dict[str, Any]:
    # ① 参数校验优先：只用标准库，廉价且能立刻失败（不把重依赖的导入放在前面挡路）
    jsonl_path: Path = args.jsonl
    if not jsonl_path.is_file():
        raise UsageError(f"--jsonl 不存在：{jsonl_path}")
    model_dir: Path = args.model_dir
    if not model_dir.is_dir():
        raise UsageError(f"--model-dir 不存在：{model_dir}")
    if args.limit is not None and args.limit <= 0:
        raise UsageError("--limit 必须为正整数")

    actual_sha = sha256_of(jsonl_path)
    if args.expect_sha256 is not None and actual_sha != args.expect_sha256:
        raise UsageError(
            f"sha256 不匹配（fail-closed）：期望 {args.expect_sha256}，实测 {actual_sha}"
            f"（{jsonl_path}）——测量对象不是你以为的那份字节，拒绝产出结论"
        )

    image_root: Path = args.image_root or jsonl_path.parent
    levels = parse_levels(args.truncation_levels)

    # ② 依赖门：本工具必须在有 transformers + torch + pillow 的环境里运行
    try:
        import PIL  # noqa: F401 - 提前失败，避免跑一半才发现缺依赖
    except ImportError as exc:
        raise UsageError(
            "缺少 `pillow`。本工具必须在有 transformers + torch + pillow 的环境里运行"
            "（例如项目容器 `graspo-msswift:4.5.3`）；本机缺依赖时只能做静态检查。"
        ) from exc

    samples = load_samples(jsonl_path, args.limit)
    warnings: list[str] = []

    processor, load_call, load_summary = load_processor(model_dir, args.max_pixels)
    image_token_id = json.loads((model_dir / "config.json").read_text(encoding="utf-8")).get(
        "image_token_id"
    )
    if image_token_id is None:
        raise UsageError(f"{model_dir}/config.json 缺 image_token_id，无法精确计数图像 token")

    result: dict[str, Any] = {
        "status": "started",
        "jsonl_path": str(jsonl_path),
        "jsonl_sha256": actual_sha,
        "jsonl_bytes": jsonl_path.stat().st_size,
        "jsonl_lines_measured": len(samples),
        "limit": args.limit,
        "image_root": str(image_root),
        "model_dir": str(model_dir),
        "image_token_id": image_token_id,
        "truncation_levels": list(levels),
        **load_summary,
        "processor_load_call_used": load_call,
        "notes": [],
    }

    per_sample: list[dict[str, Any]] = []
    for line_no, sample in enumerate(samples, 1):
        messages, image_paths, tools = to_hf_messages(sample, warnings)
        resolved = resolve_images(image_paths, image_root)
        missing = [str(path) for path in resolved if not path.is_file()]
        row: dict[str, Any] = {
            "line": line_no,
            "id": sample.get("id"),
            "n_images": len(resolved),
            "image_paths": image_paths,
        }
        if missing:
            if not args.skip_missing_images:
                raise UsageError(
                    f"第 {line_no} 条（id={sample.get('id')}）缺图像：{missing[:3]}"
                    f"（解析基准 {image_root}）——图像缺失会让长度偏小；"
                    "确认要跳过时请显式加 --skip-missing-images"
                )
            row["ok"] = False
            row["error"] = f"missing_images: {missing[:3]}"
            warnings.append(f"line {line_no}（id={sample.get('id')}）图像缺失，已按授权跳过")
            per_sample.append(row)
            continue
        try:
            row.update(measure_one(processor, messages, tools, resolved, image_token_id))
            row["ok"] = True
        except Exception as exc:  # noqa: BLE001 - 单条失败不掩盖，记账后继续
            row["ok"] = False
            row["error"] = f"{type(exc).__name__}: {exc}"
            warnings.append(f"line {line_no}（id={sample.get('id')}）测量失败：{row['error']}")
        per_sample.append(row)
        if line_no % 25 == 0:
            print(f"[progress] {line_no}/{len(samples)} 条已测", file=sys.stderr, flush=True)

    ok_lengths = [row["total_input_ids"] for row in per_sample if row.get("ok")]
    if not ok_lengths:
        raise UsageError("没有任何样本测量成功，拒绝产出空结论")
    result["n_ok"] = len(ok_lengths)
    result["n_failed"] = len(per_sample) - len(ok_lengths)
    result["per_sample"] = per_sample
    result.update(summarise(ok_lengths, levels, per_sample))

    image_token_counts = [row["image_tokens"] for row in per_sample if row.get("ok")]
    # 每图 token 数按各条自己的图数归一（不假设"恒为 2 张"，宪法 §2.2 显式即防呆）
    per_image = sorted(
        {
            row["image_tokens"] // row["n_images"]
            for row in per_sample
            if row.get("ok") and row["n_images"]
        }
    )
    result["image_tokens_summary"] = {
        "min": min(image_token_counts),
        "max": max(image_token_counts),
        "mean": round(statistics.fmean(image_token_counts), 2),
        "unique_values": sorted(set(image_token_counts)),
        "tokens_per_image": per_image,
    }
    non_image_counts = [row["non_image_tokens"] for row in per_sample if row.get("ok")]
    result["non_image_tokens_summary"] = {
        "min": min(non_image_counts),
        "max": max(non_image_counts),
        "mean": round(statistics.fmean(non_image_counts), 2),
        "unique_values": sorted(set(non_image_counts)),
    }

    try:
        result["breakdown"] = build_breakdown(processor, samples[0], image_root, image_token_id)
    except Exception as exc:  # noqa: BLE001 - 分解失败不应毁掉主测量
        warnings.append(f"breakdown 失败：{type(exc).__name__}: {exc}")
        result["breakdown"] = None

    result["notes"] = warnings
    result["status"] = "ok"
    return result


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        result = run(args)
    except UsageError as exc:
        print(f"[用法/环境错误] {exc}", file=sys.stderr)
        return 2
    payload = json.dumps(result, ensure_ascii=False, indent=args.indent)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(payload + "\n", encoding="utf-8")
        print(f"[done] 已写出 {args.out}", file=sys.stderr)
    else:
        print(payload)
    return 0


if __name__ == "__main__":
    sys.exit(main())
