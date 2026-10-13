#!/usr/bin/env python3
"""起跑/评测护栏（§7 形态谓词 / §8 产物新鲜度 / §10 27B Mamba 并发）。

三条护栏都是 228 实测事故的制度化（见 docs/run-discipline.md §7/§8/§10）：

§7  评测配置的 ``checkpoint_path`` 必须是**评测容器内可达**路径，且**谓词须按形态分支**：
    全参/HF ⇒ config.json ∨ model-*.safetensors；PEFT adapter ⇒ adapter_config.json ∨
    adapter_model.safetensors；native rank 分片 ⇒ rank_*.pt ∨ manifest.json。

§8  合并/导出目录**必须为空或本批新建**；⛔ 不得以"存在产物文件"判成功。
    pre：导出前目录不存在（或为空）；post：导出后存在 ∧ 含权重 ∧ 创建时刻 ≥ 本批起跑。

§10 服务 27B（Mamba）档必须显式 ``--max-num-seqs`` 且 ≤ 可用 Mamba cache blocks。

用法（可独立跑，也可被 launcher/评测脚本调用）：
    python3 scripts/run_guards.py form   --checkpoint <dir>                 # §7
    python3 scripts/run_guards.py fresh  --dir <dir> --phase pre|post --since <epoch>
    python3 scripts/run_guards.py mamba  --max-num-seqs 192 --blocks 232
退出码：0 通过；1 失败（fail-closed）；2 用法错误。
"""

from __future__ import annotations

import argparse
import os
import sys

#: 形态 → 该形态下的"可信标志文件"（任一命中即认为形态成立）
FORM_MARKERS: dict[str, tuple[str, ...]] = {
    "full": ("config.json", "model.safetensors", "model.safetensors.index.json"),
    "adapter": ("adapter_config.json", "adapter_model.safetensors"),
    "native_rank": ("manifest.json",),
}


def detect_form_and_validate(checkpoint_dir: str) -> tuple[str, bool, str]:
    """§7：返回 (形态, 是否通过, 说明)。全参/ adapter / native rank 三种形态各按自己的谓词判。"""
    if not checkpoint_dir or not os.path.isdir(checkpoint_dir):
        return ("none", False, f"checkpoint 目录不存在：{checkpoint_dir}")
    # PEFT adapter 优先（adapter 目录里通常**没有** config.json）
    if any(os.path.exists(os.path.join(checkpoint_dir, m)) for m in FORM_MARKERS["adapter"]):
        return ("adapter", True, "adapter 形态：命中 adapter_config.json / adapter_model.safetensors")
    if any(os.path.exists(os.path.join(checkpoint_dir, m)) for m in FORM_MARKERS["full"]):
        return ("full", True, "全参/HF 形态：命中 config.json / model*.safetensors")
    # 分片权重：model-00001-of-00018.safetensors（merged-hf 的真实形态）
    if any(f.startswith("model-") and f.endswith(".safetensors") and "-of-" in f for f in os.listdir(checkpoint_dir)):
        return ("full", True, "全参/HF 分片形态：命中 model-*-of-*.safetensors")
    if any(os.path.exists(os.path.join(checkpoint_dir, m)) for m in FORM_MARKERS["native_rank"]):
        return ("native_rank", True, "native 形态：命中 manifest.json")
    for pat in ("rank_",):
        if any(f.startswith(pat) and f.endswith(".pt") for f in os.listdir(checkpoint_dir)):
            return ("native_rank", True, "native 形态：命中 rank_*.pt")
    return ("unknown", False, f"目录存在但无任何形态标志文件（{sorted(os.listdir(checkpoint_dir))[:5]}）")


def check_fresh(directory: str, phase: str, since_epoch: float | None = None) -> tuple[bool, str]:
    """§8：pre = 目录必须不存在或为空；post = 存在 ∧ 含权重 ∧ 创建时刻 ≥ since。"""
    exists = os.path.exists(directory)
    if phase == "pre":
        if not exists:
            return (True, "§8(pre) 通过：目录在导出前不存在（将为本批新建）")
        entries = os.listdir(directory)
        if not entries:
            return (True, "§8(pre) 通过：目录存在但为空")
        return (False, f"§8(pre) 失败：目录在导出前已存在且非空（陈旧产物）：{entries[:5]}")
    if phase == "post":
        if not exists:
            return (False, "§8(post) 失败：导出后目录不存在（本批未产出）")
        weights = [f for f in os.listdir(directory) if f.endswith((".safetensors", ".pt"))]
        if not weights:
            return (False, "§8(post) 失败：导出后目录无权重文件（本批未产出）")
        if since_epoch is not None:
            ctime = os.stat(directory).st_ctime
            if ctime < since_epoch:
                return (False, f"§8(post) 失败：目录创建时刻 {ctime} < 本批起跑 {since_epoch}（疑似陈旧）")
        return (True, f"§8(post) 通过：本批新建且含权重（{len(weights)} 个）")
    return (False, f"未知 phase：{phase}")


def check_mamba(max_num_seqs: int, blocks: int) -> tuple[bool, str]:
    """§10：max_num_seqs 必须 ≤ 可用 Mamba cache blocks。"""
    if max_num_seqs <= blocks:
        return (True, f"§10 通过：max_num_seqs={max_num_seqs} ≤ blocks={blocks}")
    return (False, f"§10 失败：max_num_seqs={max_num_seqs} > 可用 Mamba cache blocks={blocks}（会 Engine core initialization failed）")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="起跑/评测护栏（§7/§8/§10）")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_form = sub.add_parser("form", help="§7 形态谓词")
    p_form.add_argument("--checkpoint", required=True)

    p_fresh = sub.add_parser("fresh", help="§8 产物新鲜度")
    p_fresh.add_argument("--dir", required=True)
    p_fresh.add_argument("--phase", choices=("pre", "post"), required=True)
    p_fresh.add_argument("--since", type=float, default=None)

    p_mamba = sub.add_parser("mamba", help="§10 Mamba 并发")
    p_mamba.add_argument("--max-num-seqs", type=int, required=True)
    p_mamba.add_argument("--blocks", type=int, required=True)

    args = parser.parse_args(argv)
    if args.cmd == "form":
        form, ok, msg = detect_form_and_validate(args.checkpoint)
        print(f"[{form}] {msg}")
    elif args.cmd == "fresh":
        ok, msg = check_fresh(args.dir, args.phase, args.since)
        print(msg)
    else:
        ok, msg = check_mamba(args.max_num_seqs, args.blocks)
        print(msg)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
