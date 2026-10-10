#!/usr/bin/env python3
"""matrix54 **FSDP 变体清单**（`tests/e2e/matrix54_manifest_fsdp.json`）——只读 v2 冻结清单派生。

## 为什么是独立脚本 + 独立文件

* `tests/e2e/matrix54_manifest_source.json` 是已冻结接线的清单（r1–r4 轮在用），**不改它**；
  本轮是新的一轮（基础＝FSDP 接线）⇒ 派生 `…_fsdp.json`，与 v2 并存。
* 派生方式与配置生成器同源：**只读**基线 → **显式枚举、逐条断言**的改动 → 写新文件。
  改动之外**深度相等**（脚本内 assert），所以"改了哪几条"可逐条核对。

## 覆盖范围

* **阶段1（4 档）**：T006/T021/T039/T051 —— 4 卡，原 `zero2` → FSDP2（无 offload）
* **阶段2（8 档）**：T004/T005/T019/T020/T037/T038/T049/T050 —— 1/2 卡，原 `zero2_offload` → FSDP2 + CPU offload
  （★ 阶段2 的可行性仍待 T037 冒烟：`use_vllm=false` 的 rollout 路径上游没有 FSDP 专用代码）

## 本变体的**全部**改动（逐条断言命中）

| # | 位置 | 改动 | 理由 |
|---|---|---|---|
| M1 | 12 档 `tiers[*].config` | `samples/configs/matrix54-v2/<T>.yaml` → `samples/configs/matrix54-fsdp/<T>.yaml` | 指向 FSDP 变体配置 |
| M2 | 12 档 `feasibility.recipe.msswift` | `deepspeed: zero2[_offload]` → `fsdp: samples/configs/fsdp/fsdp2_full_state_dict[_offload].json` | 与 M1 同源（§1.4 配方只有一处） |
| M3 | 12 档 `feasibility.estimate` | 去掉 zero2/zero2_offload 的 49.7 / 45.5 GiB 算式，改为"**待重算** + 保守性不变式 / 必须实测项" | 只改后端、未重跑估算器 ⇒ 不复用旧数字、不编新数字 |
| M4 | `runtime.image` | `graspo-msswift:4.5.3` → `graspo:v0.28.11-cu130fix-v2-nfc3`（经 nfc1→nfc2→nfc3） | ①③ 的读数只在打过补丁的镜像里存在；② **12 档必须同一镜像**（否则"连续 3 次"的口径单元不成立）；③ nfc3 是 rollout 修复后的唯一可用镜像 |
| M5 | 顶层 provenance | 新增 `config_version/supersedes/fsdp_migration` | 留痕：基线 sha256 + 授权来源 + ③ 计数语义 + Megatron 不可用 |
| M6 | **全部 54 档** `tiers[*].gpus` | 源清单逐档 `gpus`（源生成器卡池映射）→ `1卡[7]/2卡[6,7]/4卡[4,5,6,7]` | ★ 不换就**跑不起来**：runner 锁卡守卫只允许 `{4,5,6,7}`，实测 `run_matrix54.sh T006 --dry-run` 直接 fail-closed 拒绝。<br>★ **B8（2026-10-11）**：源清单自 2026-10-05 起**已自带** `{4,5,6,7}` 卡池映射，旧的"v2 基线应为 `[0]/[0,1]/[0,1,2,3]`"前置断言随之失效（`--check` 曾因此 `FATAL T001`）⇒ 本变体不再假定源卡池形态，只记录源值（供反向探针逐档还原）并钉定自身卡池 |

用法::

    .venv/bin/python tests/e2e/generate_matrix_manifest_fsdp.py            # 生成 + 打印改动表
    .venv/bin/python tests/e2e/generate_matrix_manifest_fsdp.py --check     # 只校验
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "tests/e2e/matrix54_manifest_source.json"
OUT = ROOT / "tests/e2e/matrix54_manifest_fsdp.json"

PHASE1_TIERS = ("T006", "T021", "T039", "T051")                                  # 4 卡，原 zero2
PHASE2_TIERS = ("T004", "T005", "T019", "T020", "T037", "T038", "T049", "T050")   # 1/2 卡，原 zero2_offload
TIERS = PHASE1_TIERS + PHASE2_TIERS
TIER_PHASE = {**{t: 1 for t in PHASE1_TIERS}, **{t: 2 for t in PHASE2_TIERS}}

FSDP_CONFIG_RELPATH = "samples/configs/fsdp/fsdp2_full_state_dict.json"
FSDP_CONFIG_RELPATH_OFFLOAD = "samples/configs/fsdp/fsdp2_full_state_dict_offload.json"
OLD_DS_BY_PHASE = {1: "zero2", 2: "zero2_offload"}
FSDP_CONF_BY_PHASE = {1: FSDP_CONFIG_RELPATH, 2: FSDP_CONFIG_RELPATH_OFFLOAD}
# 2026-09-27：镜像链条 nfc1 → nfc2 → **nfc3**（三者必须逐档统一，否则"连续 3 次"的口径单元不成立）
#   · nfc1：只有"非有限梯度计数"，经独立裁决不足以支撑判据③；
#   · nfc2：补齐"逐步 + 原始 loss/梯度"证据（完整 ③ instrumentation）；
#   · **nfc3**（本值）：修掉 rollout 的 `inference_mode` 与 FSDP2 参数簿记的冲突
#     （`swift/infer_engine/transformers_engine.py:511` inference_mode→no_grad，依据上游自注
#     "Ensure template._post_encode has no gradient"）——**它是唯一能跑通 6 个 rollout 档
#     （T037/38/39 GRASPO、T049/50/51 OPD）的镜像**；CPT/SFT 不受该改动影响。
FSDP_IMAGE = "graspo:v0.28.11-cu130fix-v2-nfc3"

#: M6 卡池映射（当前 runner 锁卡守卫只允许 {4,5,6,7}，用户 2026-09-24 批准 A）
GPU_POOL_BY_CARDS = {1: [7], 2: [6, 7], 4: [4, 5, 6, 7]}

#: M3 要替换掉的旧算式（逐字取自 v2 清单；两阶段各一条）
CANON_ZERO2_4CARD = (
    "估算每卡 49.7 GiB（预算 78 GiB）⇒ 可行：ms-swift 全参 zero2（纯 DP4，**未开 AutoTP**）："
    "权重 2 + (梯度+优化器) 10/4 B/param；激活/其他预算 12 GiB"
)
CANON_ZERO2_OFFLOAD = (
    "估算每卡 45.5 GiB（预算 78 GiB）⇒ 可行：ms-swift 全参 zero2_offload："
    "权重 2/1 + 梯度 2 B/param（优化器态在 CPU）；激活/其他预算 12 GiB"
)
CANON_BY_PHASE = {1: CANON_ZERO2_4CARD, 2: CANON_ZERO2_OFFLOAD}

NEW_WHOLE = {
    1: (
        "★ **待重算（后端已由 zero2 改为 FSDP2）**：本变体只改分片后端，未重跑 "
        "`estimate_config_per_card_gib`（该函数目前**没有** fsdp 分支）⇒ **不给出新的每卡数字**，"
        "旧 zero2 的 49.7 GiB 已随本变体废弃。保守依据（**不变式，不是实测**）："
        "`full_shard` + `reshard_after_forward` 把**参数 + 梯度 + 优化器态**都按 DP 摊，"
        "而 zero2 只摊**梯度 + 优化器态** ⇒ 每卡常驻 ≤ 旧值 ⇒ verdict 沿用 `feasible` 是**保守**的；"
        "实测每卡峰值由冒烟回填（`gpu_memory.jsonl`）。"
    ),
    2: (
        "★ **待重算（后端已由 zero2_offload 改为 FSDP2 + CPU offload）**：本变体只改分片后端，"
        "未重跑 `estimate_config_per_card_gib`（无 fsdp 分支）⇒ **不给出新的每卡数字**，"
        "旧 zero2_offload 的 45.5 GiB 已随本变体废弃。两点必须实测（不得凭机制推断）："
        "① 1 卡下 FSDP **零分片**，只能靠 `CPUOffloadPolicy` 顶，9B 全参是否装得下；"
        "② `CPUOffloadPolicy` 比 zero2_offload 多搬**参数与梯度** ⇒ 吞吐可能显著下降。"
        "verdict 沿用 `feasible` 属**未验证的保守假设**，须冒烟回填后再改判定。"
    ),
}

NEW_STUDENT_SUFFIX = {
    1: (
        "学生侧算式：★ **待重算（后端已由 zero2 改为 FSDP2）** —— 口径同 T006 的说明；"
        "`full_shard` 摊参数 + 梯度 + 优化器态 ⊇ zero2 只摊梯度 + 优化器态 ⇒ 每卡常驻 ≤ 旧值。"
    ),
    2: (
        "学生侧算式：★ **待重算（后端已由 zero2_offload 改为 FSDP2 + CPU offload）** —— "
        "口径同 T004 的说明；1 卡零分片 + 参数/梯度也进 CPU 的吞吐代价必须实测。"
    ),
}


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _patch_estimate(tier: str, estimate: str) -> tuple[str, str]:
    """M3：按该档 estimate 的**实际形态**与该档阶段做确定替换（不接受模糊匹配）。"""
    phase = TIER_PHASE[tier]
    canon = CANON_BY_PHASE[phase]
    if estimate == canon:
        return NEW_WHOLE[phase], f"整条替换（阶段{phase} 学生侧单算式形态）"
    if estimate.endswith(canon):
        return estimate[: -len(canon)] + NEW_STUDENT_SUFFIX[phase], \
            f"替换尾部学生侧算式（阶段{phase}；保留教师侧未测算说明）"
    raise SystemExit(f"FATAL {tier}: feasibility.estimate 形态与预期不符，生成器必须显式更新：{estimate[:120]!r}")


def build() -> tuple[dict, list[dict[str, object]]]:
    base = json.loads(BASE.read_text(encoding="utf-8"))
    doc = copy.deepcopy(base)
    changes: list[dict[str, object]] = []

    by_id = {t["tier_id"]: t for t in doc["tiers"]}
    for tier in TIERS:
        phase = TIER_PHASE[tier]
        node = by_id.get(tier)
        if node is None:
            raise SystemExit(f"FATAL: 基线清单里没有 {tier}")

        old_cfg = node["config"]
        if old_cfg != f"samples/configs/matrix54-v2/{tier}.yaml":
            raise SystemExit(f"FATAL {tier}: config 期望 matrix54-v2，实际 {old_cfg!r}")
        node["config"] = f"samples/configs/matrix54-fsdp/{tier}.yaml"
        changes.append({"tier": tier, "id": "M1", "field": "tiers[*].config",
                        "before": old_cfg, "after": node["config"]})

        mss = node["feasibility"].get("recipe", {}).get("msswift")
        want_ds = OLD_DS_BY_PHASE[phase]
        if mss is None or mss.get("deepspeed") != want_ds:
            raise SystemExit(f"FATAL {tier}: feasibility.recipe.msswift 期望 deepspeed={want_ds}，实际 {mss!r}")
        mss.pop("deepspeed")
        mss["fsdp"] = FSDP_CONF_BY_PHASE[phase]
        changes.append({"tier": tier, "id": "M2", "field": "feasibility.recipe.msswift.deepspeed→fsdp",
                        "before": want_ds, "after": FSDP_CONF_BY_PHASE[phase]})

        old_est = node["feasibility"]["estimate"]
        new_est, how = _patch_estimate(tier, old_est)
        node["feasibility"]["estimate"] = new_est
        changes.append({"tier": tier, "id": "M3", "field": "feasibility.estimate", "before": how,
                        "after": "待重算 + 保守性不变式 / 必须实测项"})

    # M6：卡池 —— 全部 54 档显式钉到本变体的确定性卡池。
    # ★ B8（2026-10-11）：源清单 `matrix54_manifest_source.json` 自 2026-10-05 起已自带
    #   {4,5,6,7} 卡池映射（源生成器 GPU_POOL_BY_CARDS）；旧的"v2 基线应为 [0]/[0,1]/[0,1,2,3]"
    #   前置断言随源清单演进失效（`--check` 因此 FATAL T001）。本变体**不再假定源卡池形态**：
    #   ① 逐档记录源值（供反向探针还原）；② 钉定自身卡池。卡池是**调度字段**（不是训练语义）。
    for node in doc["tiers"]:
        cards = int(node["cards"])
        if cards not in GPU_POOL_BY_CARDS:
            raise SystemExit(f"FATAL {node['tier_id']}: 未预期的 cards={cards}")
        base_gpus = list(node["gpus"])
        if len(base_gpus) != cards or len(set(base_gpus)) != cards:
            raise SystemExit(
                f"FATAL {node['tier_id']}: 源清单 gpus={base_gpus} 与 cards={cards} 不自洽")
        node["gpus"] = list(GPU_POOL_BY_CARDS[cards])
    changes.append({"tier": "全部 54 档", "id": "M6", "field": "tiers[*].gpus",
                    "before": "源清单逐档 gpus（源生成器卡池映射；已逐档记入反向探针）",
                    "after": "1卡[7] / 2卡[6,7] / 4卡[4,5,6,7]（本变体钉定的允许卡池）"})

    old_image = doc["runtime"]["image"]
    if old_image != "graspo-msswift:4.5.3":
        raise SystemExit(f"FATAL: runtime.image 期望 graspo-msswift:4.5.3，实际 {old_image!r}")
    doc["runtime"]["image"] = FSDP_IMAGE
    changes.append({"tier": "-", "id": "M4", "field": "runtime.image",
                    "before": old_image, "after": FSDP_IMAGE})

    doc["config_version"] = "fsdp-v1"
    doc["supersedes"] = {
        "relationship": "v2 清单（matrix54_manifest_source.json）**未被改动**；本文件是它的 FSDP 变体派生版",
        "base_manifest": BASE.relative_to(ROOT).as_posix(),
        "base_manifest_sha256": sha256_file(BASE),
        "note": "v2 的 v1 关系（supersedes.base_manifest）予以保留，不覆盖",
        "base_supersedes_kept": base.get("supersedes"),
    }
    doc["fsdp_migration"] = {
        "round": "fsdp-phase1+2",
        "phase1": f"4 卡 {len(PHASE1_TIERS)} 档（{'/'.join(PHASE1_TIERS)}）zero2 → FSDP2（无 offload）",
        "phase2": f"1/2 卡 {len(PHASE2_TIERS)} 档（{'/'.join(PHASE2_TIERS)}）zero2_offload → FSDP2 + CPU offload",
        "phase2_gate": "★ 阶段2 的可行性待 T037 冒烟：`use_vllm=false` 的 rollout 路径上游没有 FSDP 专用代码"
                       "（swift/rlhf_trainers/rollout_mixin.py:376-380）。T037 通过前，阶段2 的 8 档不得当作已验证。",
        "authorization": "指挥官 2026-09-27 裁定「推进 FSDP2 两阶段方案」；用户 2026-09-22 拍板 DeepSpeed 彻底废弃",
        "config_generator": "tests/e2e/generate_matrix_fsdp.py",
        "hard_requirement": "msswift.fsdp 指向的配置必须 state_dict_type=FULL_STATE_DICT，"
                            "否则 checkpoint 内不含 HF 权重 ⇒ A4 指纹恒 None（证据见 report.md §2）",
        "count_semantics": "★ 判据③计数（nonfinite_grad_step_count）在 FSDP 下**只能读作 零/非零**："
                           "grad_norm 已是跨 rank 归约后的全局量，补丁随后的 all_reduce(SUM) 会把同一事件"
                           "按 rank 各计一次 ⇒ 数值被放大 world_size 倍。A7 只判 ==0，故零当且仅零未被破坏。",
        "megatron": "★ 不作备选：实际运行镜像（本文件 runtime.image）缺 transformer_engine/apex ⇒ "
                    "Megatron 通道不可用。FSDP2 是唯一可行路径。",
        "runs_root": "必须用新产根（不覆盖 runs/v2-r1..r4）。历史：nfc2 冒烟 = runs/v2-fsdp-smoke2；本轮起改为方案 A（大卷）：T037 单档 = ${GRASPO_RUNS_ROOT}/v2-t037-nfc3，12 档 ×3 = ${GRASPO_RUNS_ROOT}/v2-fsdp-r{1,2,3}（/ 仅 3.5T，3 跑次约 5.4TB 装不下）。★ `${GRASPO_RUNS_ROOT}` 为**占位符**（本字段是 provenance 说明，不是 runner 读取的路径；runner 的产根来自环境变量 RUN_ROOT，见 run_matrix54.sh）",
        "runtime_image_id": "sha256:ffb19b78d9a75178188e8085a6419c4b1abca5e452a2116caa1a838044922d28",
        "runtime_image_note": "★ 12 档**统一用 runtime.image/runtime_image_id 所指的 nfc3**（口径单元一致性）。字段名已由 `nfc2_image_id` 改为**中性名** `runtime_image_id`，避免下次再出现「名字与值不符」。镜像链：nfc1（计数器不足）→ nfc2（补 ③ instrumentation）→ nfc3（rollout inference_mode→no_grad，唯一能跑通 6 个 rollout 档）。nfc3 的层前缀证据：nfc2 的 48 层是 nfc3 的 50 层的前缀。",
        "superseded_nfc1": "nfc1 跑次经独立裁决（review-r1r2r3/c3-implication-review.md）判定不足以支撑判据③：①钩子从不读 loss（实测 loss=inf 而 count=0）②两键不含步数、未计 hook 调用次数（grad 全 None 时 clip 返回 0.0 且权重逐位不变）③9/21 档的 rlhf 循环未核实 ⇒ 由 nfc2 取代",
    }
    changes.append({"tier": "-", "id": "M5", "field": "config_version/supersedes/fsdp_migration",
                    "before": base.get("config_version"), "after": "fsdp-v1 + provenance"})

    # 不改动断言：除已枚举路径外，其余与基线深度相等。
    # ★ 注意：`config_version` / `supersedes` 在基线里**本来就存在**，必须**还原成基线值**
    #   （不是 pop——pop 会把基线里也有的键删掉，造成假差异）。
    probe = copy.deepcopy(doc)
    probe["config_version"] = base["config_version"]
    probe["supersedes"] = copy.deepcopy(base["supersedes"])
    probe.pop("fsdp_migration", None)
    probe["runtime"]["image"] = old_image
    for tier in TIERS:
        phase = TIER_PHASE[tier]
        n = next(t for t in probe["tiers"] if t["tier_id"] == tier)
        n["config"] = f"samples/configs/matrix54-v2/{tier}.yaml"
        m = n["feasibility"]["recipe"]["msswift"]
        m.pop("fsdp")
        m["deepspeed"] = OLD_DS_BY_PHASE[phase]
        n["feasibility"]["estimate"] = next(
            t for t in base["tiers"] if t["tier_id"] == tier)["feasibility"]["estimate"]
    # M6 作用于**全部 54 档**，还原也必须覆盖全部档（逐档还原为**源清单**的 gpus 值）
    base_gpus_by_id = {t["tier_id"]: t["gpus"] for t in base["tiers"]}
    for n in probe["tiers"]:
        n["gpus"] = list(base_gpus_by_id[n["tier_id"]])

    if probe != base:
        raise SystemExit("FATAL: 除已枚举的 M1-M6 之外还有其它差异 —— 生成器必须显式列举")
    return doc, changes


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    doc, changes = build()
    payload = json.dumps(doc, ensure_ascii=False, indent=2, sort_keys=False) + "\n"

    print(f"基线 {BASE.relative_to(ROOT)}  sha256={sha256_file(BASE)}")
    print(f"派生 sha256(内存中内容)          ={hashlib.sha256(payload.encode()).hexdigest()}")
    print()
    print("| # | 档 | 字段 | before | after |")
    print("|---|---|---|---|---|")
    for c in changes:
        print(f"| {c['id']} | {c['tier']} | `{c['field']}` | {str(c['before'])[:60]} | {str(c['after'])[:60]} |")

    if args.check:
        if not OUT.is_file():
            print("\nCHECK FAILED: 目标不存在", file=sys.stderr)
            return 1
        if OUT.read_text(encoding="utf-8") != payload:
            print("\nCHECK FAILED: 落盘内容与重算不一致", file=sys.stderr)
            return 1
        print("\nCHECK OK：落盘内容与重算逐字节一致")
        return 0

    OUT.write_text(payload, encoding="utf-8")
    print(f"\n已写 {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
