#!/usr/bin/env python3
"""构建 ELAM V5「最短多模态」mini 训练子集（用于实测矩阵跑通，不用于效果评测）。

用途
----
从 ELAM V5 balanced 训练集（`data/train.jsonl`，全程只读）中挑出**最短的 N 条**
多模态样本，产出逐字节原样的 mini JSONL，供跑批脚本这类 runner 在容器内做
「跑通实测」——把数据量压到最小以缩短单档验证时间。

排序规则
--------
`(图像张数升序, 全部文本字符总数升序, id 升序)`

- 图像张数：遍历 `messages[*].content[*]`，统计 `{"type": "image"}` 的个数。
- 文本字符总数：遍历 `messages[*].content[*]`，对 `{"type": "text"}` 的 `text`
  求 `len()` 后求和（Python `len()` = Unicode 码点数）。
- `id` 仅作**确定性 tiebreak**：保证同源同参数下产出逐字节一致（幂等），
  不代表任何"更短"的语义。

推断依据（【推断】，非实测）
--------------------------
源集全部图像均为 720p（1280x720）。在同分辨率、同图像处理器（processor）下，
**单张图的图像 token 数与样本内容无关，只与分辨率有关**。因此"整条样本的
视觉 token 总量"单调于图像张数，故可以 `图像张数 → 文本字符数` 作为
"最短"的**代理排序**，无需本机 tokenizer。
> 本机无 transformers / torch / tokenizer，真实 token 数无法在本机获得，
> 必须在容器内用镜像自带 tokenizer 实测（见 `mini/measure-tokens.md`）。

命名约定（防呆，宪法 §2）
--------------------------
输出文件名**必须带 `mini-short-mm-` 前缀**（默认 `mini-short-mm-train.jsonl`），
用于与源数据 `train.jsonl` **物理区分**——两者用途完全不同（mini 只用于矩阵跑通，
源数据用于真实训练与效果评测）。若只靠目录区分、靠人记得住，迟早会把 mini 当源数据跑，
或把源数据当 mini 跑。命名即防线：一眼能看出它不是原始数据。
`mini` = 小集；`short-mm` = 最短多模态（shortest multimodal）；`train` = 训练用。

前提：图像路径是相对路径
------------------------
记录内的图像字段形如 `{"type": "image", "image": "../images/xxx.jpg"}`，
**相对 `data/` 目录**解析。因此本脚本**逐字原样透传**源行，绝不改写任何字段——
尤其不把 `../images/...` 改写成绝对路径或 `images/...`，否则容器内解析会失败。
同理，mini 文件应落在与 `images/` **同级**的数据根子目录（如 `<ELAM_HOST>/mini/`），
使 `../images/` 恰好解析到 `<ELAM_HOST>/images/`。

幂等性
------
无时间戳、无随机、无字典序依赖（`json.loads` → 只读字段，落盘用原始行文本）。
同源文件 + 同 `--count` 重复运行产出逐字节一致。

边界
----
- 源文件只读，脚本不写、不删、不移动任何源数据。
- 不硬编码本机路径：`--src` / `--out` / `--count` 均可覆盖。

用法
----
    python3 scripts/build_mini_elam_subset.py \\
        --src  <ELAM_HOST>/data/train.jsonl \\
        --out  <ELAM_HOST>/mini-dataset/mini-short-mm-train.jsonl \\
        --count 100

不传 `--src` 时回落到环境变量 `ELAM_TRAIN_JSONL`（仅作本机/容器路径注入点，
不是配置真相源，见宪法 §7.1；路径不存在时脚本以非零码退出并给出提示）。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

DEFAULT_COUNT = 100
SRC_ENV_VAR = "ELAM_TRAIN_JSONL"


def count_images_and_chars(record: dict) -> tuple[int, int]:
    """返回 `(图像张数, 文本字符总数)`。

    只读遍历 `messages[*].content[*]`：`type == "image"` 计张数，
    `type == "text"` 累加 `len(text)`。结构异常时按 0 处理，不抛异常中断整批。
    """
    n_images = 0
    n_chars = 0
    for message in record.get("messages") or []:
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for item in content:
            if not isinstance(item, dict):
                continue
            item_type = item.get("type")
            if item_type == "image":
                n_images += 1
            elif item_type == "text":
                text = item.get("text")
                if isinstance(text, str):
                    n_chars += len(text)
    return n_images, n_chars


def rank_key(record: dict, raw_line: str) -> tuple[int, int, str]:
    """排序键：图像张数 → 文本字符数 → id（确定性 tiebreak）。"""
    n_images, n_chars = count_images_and_chars(record)
    record_id = record.get("id")
    if not isinstance(record_id, str):
        # 无 id 时用行内容做确定性兜底，避免依赖输入顺序。
        record_id = raw_line
    return (n_images, n_chars, record_id)


def load_candidates(src: Path) -> list[tuple[tuple[int, int, str], str]]:
    """读源 JSONL，返回 `[(排序键, 原始行文本), ...]`。源文件全程只读。"""
    candidates: list[tuple[tuple[int, int, str], str]] = []
    with src.open("r", encoding="utf-8") as handle:
        for lineno, raw_line in enumerate(handle, start=1):
            stripped = raw_line.rstrip("\n").rstrip("\r")
            if not stripped:
                continue
            try:
                record = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise SystemExit(f"FATAL: {src}:{lineno} JSON 解析失败: {exc}") from exc
            candidates.append((rank_key(record, stripped), stripped))
    return candidates


def build(src: Path, out: Path, count: int) -> dict:
    """生成 mini 子集并返回统计字典。"""
    if not src.is_file():
        raise SystemExit(
            f"FATAL: 源文件不存在: {src}\n"
            f"       请用 --src 指定 ELAM V5 的 data/train.jsonl，"
            f"或设置环境变量 {SRC_ENV_VAR}。"
        )

    candidates = load_candidates(src)
    total = len(candidates)
    if count > total:
        raise SystemExit(f"FATAL: --count {count} 大于源总条数 {total}。")

    candidates.sort(key=lambda pair: pair[0])
    selected = candidates[:count]

    out.parent.mkdir(parents=True, exist_ok=True)
    # 逐字原样写出，不 re-serialize（避免键序/空格/ensure_ascii 造成任何改写）。
    with out.open("w", encoding="utf-8", newline="\n") as handle:
        for _, raw_line in selected:
            handle.write(raw_line)
            handle.write("\n")

    return {
        "src": str(src),
        "out": str(out),
        "total": total,
        "count": count,
        "selected": selected,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="构建 ELAM V5 最短多模态 mini 训练子集（只读源数据，逐字节原样透传）。"
    )
    parser.add_argument(
        "--src",
        default=os.environ.get(SRC_ENV_VAR),
        help=f"源 train.jsonl 路径（只读）。默认取环境变量 {SRC_ENV_VAR}。",
    )
    parser.add_argument(
        "--out",
        default="mini/mini-short-mm-train.jsonl",
        help=(
            "输出 mini JSONL 路径（默认 mini/mini-short-mm-train.jsonl，相对当前目录）。"
            "⚠ 文件名必须带 `mini-short-mm-` 前缀，用于与源数据 train.jsonl 物理区分（防呆，宪法 §2）。"
        ),
    )
    parser.add_argument(
        "--count",
        type=int,
        default=DEFAULT_COUNT,
        help=f"入选条数（默认 {DEFAULT_COUNT}）。",
    )
    args = parser.parse_args(argv)

    if not args.src:
        raise SystemExit(
            f"FATAL: 未指定 --src，且环境变量 {SRC_ENV_VAR} 未设置。\n"
            f"       示例: --src <ELAM_HOST>/data/train.jsonl"
        )

    src = Path(args.src).expanduser().resolve()
    out = Path(args.out).expanduser().resolve()

    # 防呆（宪法 §2.4）：拒绝把输出写回源文件，避免自毁数据源。
    if out == src:
        raise SystemExit(f"FATAL: --out 与 --src 相同（{src}），拒绝覆盖源数据。")

    stats = build(src, out, args.count)

    image_hist: dict[int, int] = {}
    char_counts: list[int] = []
    for key, _ in stats["selected"]:
        n_images, n_chars, _ = key
        image_hist[n_images] = image_hist.get(n_images, 0) + 1
        char_counts.append(n_chars)

    print(f"源文件总条数 : {stats['total']}")
    print(f"入选条数     : {stats['count']}")
    print(f"输出         : {stats['out']}")
    print(f"图像张数分布 : {dict(sorted(image_hist.items()))}")
    if char_counts:
        print(
            f"文本字符     : min={min(char_counts)} max={max(char_counts)} "
            f"sum={sum(char_counts)}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
