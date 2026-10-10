#!/usr/bin/env python3
"""matrix54 **FSDP 变体**（`samples/configs/matrix54-fsdp/`）生成器 —— 阶段1：4 卡档 DeepSpeed → FSDP2。

## 为什么是独立脚本（而不是改 `generate_matrix.py` / `generate_matrix_v2.py`）

* `samples/configs/matrix54-v2/` 是**已冻结**的定稿（指挥官 2026-09-23 冻结，总指纹
  `7a230bad…`）。本轮是**新的一轮**（基础＝FSDP 接线），因此
  **既不覆盖也不手改** 冻结目录，而是从它派生一个**新的、可复核的变体目录**。
* 与 v2 生成器同一套做法（§ 先例）：**只读**上游 yaml 文本，做**显式枚举、逐条断言**的
  文本变换，写到一个新目录；变换之外**逐字节不变**（脚本内 assert）。
  不做 `yaml.load/dump` 往返——那会重排键序、丢注释，让"改了哪几行"无法逐行核对。

## 本变体的**全部**改动（每档 4 条，逐条断言恰好命中 1 次）

| # | 性质 | 改动 |
|---|---|---|
| T1 | **语义（唯一一处）** | `msswift.deepspeed: zero2` → `msswift.fsdp: samples/configs/fsdp/fsdp2_full_state_dict.json` |
| T2 | 注释 | 第 9 行 `# 显存可行性:` 里的 `ms-swift 全参 zero2（…）` 算式**替换为"待重算"说明**——换后端后旧算式已过期，**不编造新数字**（§2.2 不让文档说假话） |
| T3 | 注释 | `# 配置版本:` → 标明本目录是 fsdp 变体、基线是 v2 |
| T4 | 注释 | `# 生成器:` → 指向本脚本；并加一行写明**授权来源** |

★ 为什么 T2 用"待重算"而不是给新算式：本阶段只改分片后端，
`generate_matrix.py::estimate_config_per_card_gib` **尚未加 fsdp 分支**（该函数目前只有
deepspeed 三分支 + 一个"全参全压一张卡"的兜底）。若在此处手写一个算式，就会出现
"两个地方各写一份显存口径"（§1.4 违反）且无法被生成器校验。⇒ 如实标"待重算"，
把重算留给 manifest 的 `feasibility_model`（见 `tests/e2e/matrix54_manifest_fsdp.json`）。

用法::

    .venv/bin/python tests/e2e/generate_matrix_fsdp.py            # 生成 + 打印改动表
    .venv/bin/python tests/e2e/generate_matrix_fsdp.py --check     # 只校验落盘内容与重算一致
    .venv/bin/python tests/e2e/generate_matrix_fsdp.py --emit-dir <DIR>   # 额外落 fingerprint/changes
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

SRC_DIR = ROOT / "samples/configs/matrix54-v2"
DST_DIR = ROOT / "samples/configs/matrix54-fsdp"

#: 阶段1 = 4 卡的 4 档（原 `zero2`）；阶段2 = 1/2 卡的 8 档（原 `zero2_offload`，需 CPU offload）
PHASE1_TIERS = ("T006", "T021", "T039", "T051")
PHASE2_TIERS = ("T004", "T005", "T019", "T020", "T037", "T038", "T049", "T050")

#: `msswift.fsdp` 的取值 = **仓库内相对路径**。
#: 容器内 `docker run … -w /workspace/graspo`（`tests/e2e/run_matrix54.sh`），且
#: ms-swift `swift/utils/utils.py::json_parse_to_dict` 用 `os.path.exists(value)` 判"是路径"，
#: **相对路径按容器 CWD 解析** ⇒ 该值在容器内可读。用相对路径是为了不把
#: `/workspace/graspo` 这个挂载点写死进配置。
FSDP_CONFIG_RELPATH = "samples/configs/fsdp/fsdp2_full_state_dict.json"           # 阶段1（4 卡）
FSDP_CONFIG_RELPATH_OFFLOAD = "samples/configs/fsdp/fsdp2_full_state_dict_offload.json"  # 阶段2（1/2 卡）

#: 冻结指纹守卫：阶段1 的 4 档已落位并双向 sha 复核过，**后续任何改动都必须让它变红**
#: （不是"记得别改"，而是"改了就跑不过"）。新增阶段2 不得影响这 4 个文件。
PHASE1_FROZEN_TOTAL_FINGERPRINT = "c78780c48f4d78426ca5fc99df413c3c428bd8a8f3e68ce276ba5f7037717586"

def transforms_for(phase: int) -> tuple[tuple[str, str, str, str], ...]:
    """按阶段给出 4 条变换。**每条都断言恰好命中 1 次**；两阶段的唯一语义差异是
    「原 deepspeed 值」与「目标 fsdp 配置（是否 offload）」以及随之的注释口径。"""
    if phase == 1:
        old_ds, new_fsdp = "  deepspeed: zero2\n", f"  fsdp: {FSDP_CONFIG_RELPATH}\n"
        old_est = (
            "ms-swift 全参 zero2（纯 DP4，**未开 AutoTP**）：权重 2 + (梯度+优化器) 10/4 B/param；"
            "激活/其他预算 12 GiB"
        )
        new_est = (
            "ms-swift 全参 **FSDP2**（`full_shard auto_wrap`，= ZeRO-3 等价；纯 DP4，无 TP/PP）："
            "★ **本变体只改分片后端，显存算式待重算**——旧 zero2 数字已随本变体废弃，"
            "不在此处编造新数字；重算口径见 `tests/e2e/matrix54_manifest_fsdp.json` 的 "
            "`feasibility_model`"
        )
    elif phase == 2:
        old_ds, new_fsdp = "  deepspeed: zero2_offload\n", f"  fsdp: {FSDP_CONFIG_RELPATH_OFFLOAD}\n"
        old_est = (
            "ms-swift 全参 zero2_offload：权重 2/1 + 梯度 2 B/param（优化器态在 CPU）；"
            "激活/其他预算 12 GiB"
        )
        new_est = (
            "ms-swift 全参 **FSDP2 + CPU offload**（`full_shard auto_wrap` + `CPUOffloadPolicy`；"
            "1/2 卡下 FSDP 分片面不够 ⇒ 必须 offload）：★ **本变体只改分片后端，显存算式待重算**"
            "——旧 zero2_offload 的 45.5 GiB 已随本变体废弃，不在此处编造新数字；"
            "★ 吞吐亦未知（CPUOffloadPolicy 比 zero2_offload 多搬参数与梯度），以冒烟实测为准"
        )
    else:
        raise SystemExit(f"FATAL: 未知阶段 {phase}")
    return (
        ("T1 语义", "semantic", old_ds, new_fsdp),
        ("T2 注释", "comment", old_est, new_est),
        (
            "T3 注释",
            "comment",
            "# 配置版本: v2（v1 基线 = samples/configs/matrix54/；本目录为冻结的 v2 定稿）",
            "# 配置版本: fsdp-v1（基线 = 冻结的 samples/configs/matrix54-v2/；本目录是 DeepSpeed→FSDP2 变体）",
        ),
        (
            "T4 注释",
            "comment",
            # ★ 上游（samples/configs/matrix54-v2/*.yaml:3）的生成器声明在 B4/§(3) 改为
            # 「冻结件 + v1 基线 + v2 补丁件退役留档」；本变换必须逐字跟随，否则
            # `--check` 会报「上游 yaml 已漂移」（fail-closed，不做模糊匹配）。
            "# 生成器: 冻结件（不可由仓库重算）；v1 基线 = tests/e2e/generate_matrix.py；"
            "v2 逐档补丁件 tests/e2e/generate_matrix_v2.py 已退役留档 "
            ".local/repo-moved-out/tests/e2e/；手改无效",
            (
                "# 生成器: tests/e2e/generate_matrix_fsdp.py（手改无效，请改生成器）\n"
                "# 授权: 指挥官 2026-09-27 裁定「推进 FSDP2 两阶段方案，先做 4 卡 4 档纯配置直换」"
                "（用户 2026-09-22 拍板 DeepSpeed 彻底废弃）"
                if phase == 1 else
                "# 生成器: tests/e2e/generate_matrix_fsdp.py（手改无效，请改生成器）\n"
                "# 授权: 指挥官 2026-09-27 裁定「推进 FSDP2 两阶段方案」（本档属阶段2：1/2 卡 + CPU offload）"
                "（用户 2026-09-22 拍板 DeepSpeed 彻底废弃）"
            ),
        ),
    )


def assert_fsdp_config_is_full_state_dict() -> None:
    """★ fail-closed：`msswift.fsdp` 指向的配置**必须**是 FULL_STATE_DICT。

    理由（实测，见 report.md §2）：SHARDED_STATE_DICT 下 transformers 的 `save_model`
    整体跳过 HF 落盘（`transformers/trainer.py:3797`）⇒ checkpoint 里没有 `model.safetensors`
    ⇒ 我方 collector 找不到权重（`scripts/collect_results.py:933-938/957`）⇒ A4 指纹恒 None。
    这条断言把"改错了要等冒烟才暴露"提前成"生成期就拒绝"（§2.3 边界校验）。
    """
    import json as _json
    for rel in (FSDP_CONFIG_RELPATH, FSDP_CONFIG_RELPATH_OFFLOAD):
        path = ROOT / rel
        if not path.is_file():
            raise SystemExit(f"FATAL: fsdp 配置不存在：{rel}")
        doc = _json.loads(path.read_text(encoding="utf-8"))
        got = doc.get("fsdp_config", {}).get("state_dict_type")
        if got != "FULL_STATE_DICT":
            raise SystemExit(
                f"FATAL: {rel} 的 state_dict_type={got!r}，必须是 'FULL_STATE_DICT' —— "
                "否则 checkpoint 内不含 HF 权重、A4 指纹恒 None（见 report.md §2 实测）"
            )
        if doc.get("fsdp_config", {}).get("fsdp_version") != 2:
            raise SystemExit(f"FATAL: {rel} 必须 fsdp_version=2（FSDP1 在 RL 通道被上游拒绝）")


_HEADER_HINT = "本档相对 samples/configs/matrix54-v2/ 的改动：仅 T1（语义：deepspeed→fsdp）+ T2/T3/T4（注释）"


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def render(tier: str, phase: int) -> tuple[bytes, list[dict[str, object]]]:
    """读冻结 v2 yaml → 逐条变换 → 返回 (新文本, 逐条改动记录)。"""
    src = SRC_DIR / f"{tier}.yaml"
    text = src.read_text(encoding="utf-8")
    records: list[dict[str, object]] = []
    for name, kind, old, new in transforms_for(phase):
        hits = text.count(old)
        if hits != 1:
            raise SystemExit(
                f"FATAL {tier}: 变换 {name} 期望命中 1 次，实际 {hits} 次 —— "
                f"上游 yaml 已漂移，本生成器必须显式更新（不做模糊匹配）"
            )
        text = text.replace(old, new)
        records.append({"tier": tier, "name": name, "kind": kind, "hits": hits})
    # 兼容性断言：变换后不得残留 deepspeed 字段、且必须有 fsdp 字段
    if "\n  deepspeed:" in text:
        raise SystemExit(f"FATAL {tier}: 变换后仍残留 msswift.deepspeed 行")
    want = FSDP_CONFIG_RELPATH if phase == 1 else FSDP_CONFIG_RELPATH_OFFLOAD
    if f"\n  fsdp: {want}\n" not in text:
        raise SystemExit(f"FATAL {tier}: 变换后缺 msswift.fsdp 行（phase={phase}）")
    return text.encode("utf-8"), records


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="只校验落盘内容与重算一致，不写文件")
    parser.add_argument("--emit-dir", default=None, help="额外落 fingerprint.txt / changes.md 的目录")
    parser.add_argument("--phase", choices=["1", "2", "all"], default="all",
                        help="阶段1 = 4 卡 4 档；阶段2 = 1/2 卡 8 档；默认两阶段")
    args = parser.parse_args()

    assert_fsdp_config_is_full_state_dict()

    phases: list[int] = [1, 2] if args.phase == "all" else [int(args.phase)]
    tiers_of = {1: PHASE1_TIERS, 2: PHASE2_TIERS}
    total_hex_of: dict[int, str] = {}
    blobs_all: dict[str, bytes] = {}
    rows_all: list[tuple[int, str, str, str]] = []

    for phase in phases:
        out_blobs: dict[str, bytes] = {}
        for tier in tiers_of[phase]:
            new_bytes, records = render(tier, phase)
            out_blobs[f"{tier}.yaml"] = new_bytes
            blobs_all[f"{tier}.yaml"] = new_bytes
            before = sha256_bytes((SRC_DIR / f"{tier}.yaml").read_bytes())
            after = sha256_bytes(new_bytes)
            rows_all.append((phase, tier, before, after))
            for rec in records:
                print(f"[phase{phase}][{rec['tier']}] {rec['name']:<8} {rec['kind']:<8} 命中 {rec['hits']} 次")
        total = hashlib.sha256()
        for name in sorted(out_blobs):
            total.update(f"{sha256_bytes(out_blobs[name])}  {name}\n".encode())
        total_hex_of[phase] = total.hexdigest()
        # 阶段1 冻结守卫（阶段1 单跑或 all 跑都要过）
        if phase == 1 and total_hex_of[1] != PHASE1_FROZEN_TOTAL_FINGERPRINT:
            raise SystemExit(
                f"FATAL: 阶段1 总指纹漂移！期望 {PHASE1_FROZEN_TOTAL_FINGERPRINT}，实得 {total_hex_of[1]}\n"
                "       阶段1 已落位并双向 sha 复核；本次改动不得影响那 4 个文件（新阶段应只增加文件）。"
            )

    print()
    print("| 阶段 | 档 | before sha256 (matrix54-v2) | after sha256 (matrix54-fsdp) |")
    print("|---|---|---|---|")
    for phase, tier, before, after in rows_all:
        print(f"| {phase} | {tier} | `{before}` | `{after}` |")
    print()
    for phase in phases:
        print(f"阶段{phase} 总指纹（{len(tiers_of[phase])} 档，算法同 v2 冻结）: {total_hex_of[phase]}")

    if args.check:
        bad = []
        for name, blob in blobs_all.items():
            path = DST_DIR / name
            if not path.is_file():
                bad.append(f"{name}: 缺失")
            elif path.read_bytes() != blob:
                bad.append(f"{name}: 内容与重算不一致")
        if bad:
            print("\nCHECK FAILED:" + "\n".join(bad), file=sys.stderr)
            return 1
        print("\nCHECK OK：落盘内容与重算逐字节一致")
        return 0

    DST_DIR.mkdir(parents=True, exist_ok=True)
    for name, blob in blobs_all.items():
        (DST_DIR / name).write_bytes(blob)
    print(f"\n已写 {len(blobs_all)} 个文件到 {DST_DIR.relative_to(ROOT)}/")

    if args.emit_dir:
        emit = Path(args.emit_dir)
        emit.mkdir(parents=True, exist_ok=True)
        lines = [
            "# matrix54-fsdp 改动清单与冻结指纹（阶段1 + 阶段2）", "",
            "> 阶段1 = 4 卡 4 档（原 `zero2`，FSDP2 无 offload）；阶段2 = 1/2 卡 8 档（原 `zero2_offload`，FSDP2 + CPU offload）",
            "", "| 阶段 | 总指纹 |", "|---|---|",
        ]
        for phase in phases:
            lines.append(f"| {phase} | `{total_hex_of[phase]}` |")
        lines += ["", "| 阶段 | 档 | before sha256 | after sha256 |", "|---|---|---|---|"]
        lines += [f"| {ph} | {t} | `{b}` | `{a}` |" for ph, t, b, a in rows_all]
        lines += ["", "## 两阶段的逐条变换（生成器断言恰好命中 1 次）", ""]
        for phase in phases:
            for name, kind, old, new in transforms_for(phase):
                lines += [f"### 阶段{phase} · {name}（{kind}）", "", "```diff",
                          f"- {old.strip()}", f"+ {new.strip()}", "```", ""]
        (emit / "changes.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
        (emit / "fingerprint.txt").write_text(
            "配置版本: fsdp-v1（阶段1 = 4 卡 4 档；阶段2 = 1/2 卡 8 档）\n"
            f"目录: {DST_DIR.relative_to(ROOT)}/\n"
            f"文件数: {len(blobs_all)}\n"
            "总指纹算法: 按文件名排序 → 拼接每行 '<sha256>  <name>\\n' → 取 sha256\n"
            + "".join(f"阶段{ph} 总指纹: {total_hex_of[ph]}\n" for ph in phases)
            + f"阶段1 冻结守卫值: {PHASE1_FROZEN_TOTAL_FINGERPRINT}\n"
            f"fsdp 配置（阶段1）: {FSDP_CONFIG_RELPATH}\n"
            f"fsdp 配置（阶段2）: {FSDP_CONFIG_RELPATH_OFFLOAD}\n"
            f"基线（只读、未改动）: {SRC_DIR.relative_to(ROOT)}/\n"
            "授权: 指挥官 2026-09-27「推进 FSDP2 两阶段方案」；用户 2026-09-22 拍板 DeepSpeed 彻底废弃\n",
            encoding="utf-8",
        )
        print(f"已写 fingerprint.txt / changes.md 到 {emit}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
