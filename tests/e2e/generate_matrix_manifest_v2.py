#!/usr/bin/env python3
"""matrix54 **配置 v2 运行清单**（v2 manifest）生成器 —— 只新增，不改 v1。

为什么是独立脚本 + 独立产物（§6.1 简单优先 / §1.4 单一真相源）：
  · ``tests/e2e/matrix54_manifest.json`` 是 v1 的冻结产物，**必须逐字节不变**；
    本脚本**只读**它（作为 v1 台账的单一真相源），把它按 v2 规则**改写出一份新清单**
    ``tests/e2e/matrix54_manifest_v2.json``。v1 生成器一行未动。
  · 改写面**只有四类**，逐条显式、逐条可核：
      ① ``config``：``samples/configs/matrix54/T0NN.yaml`` → ``samples/configs/matrix54-v2/T0NN.yaml``
         （runner 实际读取的就是这个字段，见 ``run_matrix54.sh`` 的 ``CONFIG="${CONFIG_STATUS%%|*}"``）
      ② ``data.subset_size``：9 档 ``GRASPO × native`` → ``48 × 卡数``（48/96/192）；
         其余 45 档**照 v1**（CPT/SFT=100、GRASPO/OPD=20）。3 个 not_applicable 档保持 v1。
      ③ ``data.train_path`` / 步数两字段 / ``acceptance.formal_gate.*``：按 v2 真值重算
         （``expected_optimizer_steps_per_epoch`` / ``_reachable`` 用**真实步进语义**，
         不是 v1 的朴素均分；见 :func:`v2_native_steps_per_epoch`）
      ④ ``data.source`` / ``data.source_jsonl``（**新增字段**）：逐档标注取数来源与命令，
         供 ``rig/run_matrix54_v2source.sh`` 做**单档进程作用域**的读取源切换。
         既有字段一个都没删、没改语义 ⇒ 老消费者（``collect_results.py``）不受影响。

★ 步数真值（逐条有代码依据，见 ``docs/capability-matrix.html`` §1 与
  ``.local/hb-workspace/20260923-analysis/task-config-freeze-v2/changes.md`` §1）::

      T = floor(subset_size / dp_size)          # native 只按 DP 分片；PP 档 dp_size=1 ⇒ 不分片
      每 epoch 步数 = ceil(T / Q) = (T + Q - 1) // Q        Q = rollout_queue_batch_size = 8

  v2 子集 48×卡数 ⇒ 每卡 prompt 数 P = 48（PP 档例外：dp_size=1 ⇒ T=96/192）::

      T028 T029 T030 T034 T040 T041 T042  →  T = 48  →  6 步
      T035（dp=1, pp=2）                  →  T = 96  → 12 步
      T036（dp=1, pp=4）                  →  T = 192 → 24 步
      全部 ≥ 硬前提 5 步 ⇒ step 门槛对 54 档**全部适用**（v2 无 structural_limited 档）。

用法::

    .venv/bin/python tests/e2e/generate_matrix_manifest_v2.py            # 写 v2 清单
    .venv/bin/python tests/e2e/generate_matrix_manifest_v2.py --check    # 只校验已落盘的 v2 清单
    .venv/bin/python tests/e2e/generate_matrix_manifest_v2.py --print-steps
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import re
import sys
from pathlib import Path

_HERE = Path(__file__).resolve()
PROJECT_ROOT = _HERE.parents[2]  # tests/e2e/x.py → tests/e2e → tests → <repo>
V1_GENERATOR_PATH = _HERE.parent / "generate_matrix.py"
V1_MANIFEST_PATH = _HERE.parent / "matrix54_manifest.json"
V2_MANIFEST_PATH = _HERE.parent / "matrix54_manifest_v2.json"
V2_CONFIG_DIR_REL = "samples/configs/matrix54-v2"
V1_CONFIG_DIR_REL = "samples/configs/matrix54"

#: v2 子集目录名（容器内 ``/work/elam-v5/subsets-v2`` 的叶子名）。
V2_SUBSET_DIRNAME = "subsets-v2"
#: 228 上**三个互不重叠**的宿主路径（运行期环境信息；都可从命令行覆盖）：
#:   · 原始数据根（只读源）—— ``data/train.jsonl`` / ``mini-dataset/`` / ``images/``
#:   · 预生成冻结子集目录（只读）—— ``T0NN.jsonl``
#:   · 运行产物根（可写，runner 的 ``RUN_ROOT``）—— ``<attempt_root>/<T0NN>/...``
#: ★ 三者在 228 上**刻意不嵌套**：冻结件永不落在运行产物根之内 ⇒
#:   runner 在运行产物根里做任何清理都不会碰到冻结件（§1.4/§2.4）。
DEFAULT_HOST_DATA_ROOT = "/home/<user>/<dataset-volume>"
DEFAULT_PREGEN_SUBSET_DIR = "/home/<user>/<RUN_ROOT_NAME>/subsets-v2"
DEFAULT_ATTEMPT_ROOT = "<RUN_ROOT>/v2"

#: 9 档「GRASPO × native」的 v2 子集条数（= 48 × 卡数）。
V2_SUBSET_SIZE_BY_TIER: dict[str, int] = {
    "T028": 48, "T029": 96, "T030": 192,
    "T034": 48, "T035": 96, "T036": 192,
    "T040": 48, "T041": 96, "T042": 192,
}
#: 3 个 not_applicable 档（v1 已定，v2 不改；不产出可跑配置）。
NOT_APPLICABLE_TIERS: tuple[str, ...] = ("T052", "T053", "T054")
#: 来源名（进 ``data.source``；也是 ``rig/run_matrix54_v2source.sh`` 的判据字面量）。
SOURCE_FORMAL = "formal"
SOURCE_MINI = "mini"

Q = 8  # 与 v1 生成器同源（下面用断言核对，不另抄常量）
MINI_LINE_COUNT = 100
FORMAL_LINE_COUNT = 6378
MIN_OPTIMIZER_STEPS = 5


def load_v1_module():
    """只读加载 v1 生成器（拿常量与 fail-closed 校验函数，不触发任何写盘）。"""
    spec = importlib.util.spec_from_file_location("_manifest_v2_v1_generator", V1_GENERATOR_PATH)
    if spec is None or spec.loader is None:  # pragma: no cover
        raise RuntimeError(f"无法加载 v1 生成器：{V1_GENERATOR_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


gm = load_v1_module()

#: 单一真相源核对：Q / 门槛 / epoch 数都从 v1 生成器取，不在本脚本另立一份。
assert int(gm.NATIVE_ROLLOUT_QUEUE_BATCH_SIZE) == Q, "Q 与 v1 生成器不一致"
assert int(gm.FORMAL_GATE_MIN_OPTIMIZER_STEPS) == MIN_OPTIMIZER_STEPS, "门槛与 v1 不一致"
assert int(gm.ELAM_TRAIN_COUNT) == FORMAL_LINE_COUNT, "正式集行数与 v1 不一致"


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def v2_native_steps_per_epoch(subset_size: int, dp_size: int, q: int = Q) -> int:
    """native GRASPO 每 epoch **真实**步数 = ``ceil(T/Q)``，``T = floor(subset/dp_size)``。

    与 v1 生成器 :func:`generate_matrix.native_graspo_steps_per_epoch` 的
    ``floor(P/Q) + 1`` 的差别：后者是**上界**，当 ``P mod Q == 0`` 时会多算 1
    （epoch 末 force flush 遇上空 buffer 直接返回 False）。v2 的 48/96/192 全是
    8 的整数倍，两者**必然不同**（v1 式会算出 7/13/25），故 v2 必须用真实式。
    """
    prompts = max(0, int(subset_size)) // max(1, int(dp_size))
    if prompts <= 0:
        return 0
    return (prompts + max(1, int(q)) - 1) // max(1, int(q))


def read_yaml_scalar(path: Path, key: str) -> str:
    """从档配置里读 ``  <key>: <value>``（读不到即 fail-closed，不猜默认值）。"""
    hits = re.findall(rf"^\s*{re.escape(key)}:\s*(\S+)\s*$", path.read_text(encoding="utf-8"), re.M)
    if len(hits) != 1:
        raise AssertionError(f"{path}: 字段 {key!r} 出现 {len(hits)} 次（期望 1）")
    return hits[0]


def v2_dp_size(tier_id: str) -> int:
    """该档 native 的 ``dp_size``（从 **v2 档配置**读，单一真相源）。"""
    return int(read_yaml_scalar(PROJECT_ROOT / V2_CONFIG_DIR_REL / f"{tier_id}.yaml", "dp_size"))


def is_native_graspo(entry: dict) -> bool:
    return str(entry["backend"]) == "native" and str(entry["algorithm"]) == "GRASPO"


def patch_tier(entry: dict, data_root: str, pregen_dir: str) -> dict:
    """把 v1 的单档条目改写成 v2 条目（就地改写一份深拷贝）。"""
    tier_id = str(entry["tier_id"])
    out = copy.deepcopy(entry)
    is_na = tier_id in NOT_APPLICABLE_TIERS

    # ① config 口径
    if out.get("config"):
        if Path(out["config"]).parent.as_posix() != V1_CONFIG_DIR_REL:
            raise AssertionError(f"{tier_id}: v1 config 目录异常：{out['config']}")
        out["config"] = f"{V2_CONFIG_DIR_REL}/{tier_id}.yaml"

    # ② 子集条数
    v1_subset = int(out["data"]["subset_size"])
    if is_na:
        subset = v1_subset  # not_applicable：不跑，保持 v1（不假装有 v2 子集）
        source = None
    elif tier_id in V2_SUBSET_SIZE_BY_TIER:
        subset = V2_SUBSET_SIZE_BY_TIER[tier_id]
        source = SOURCE_FORMAL
    else:
        subset = v1_subset
        source = SOURCE_MINI

    # ③ 数据源逐档标注（逐档具名；取数命令与条数都写死在这一档里）
    src_rel = (
        "/work/elam-v5/data/train.jsonl" if source == SOURCE_FORMAL
        else "/work/elam-v5/mini-dataset/mini-short-mm-train.jsonl"
    ) if source else None
    host_src = (
        f"{data_root}/data/train.jsonl" if source == SOURCE_FORMAL
        else f"{data_root}/mini-dataset/mini-short-mm-train.jsonl"
    ) if source else None
    host_subset = f"{pregen_dir}/{tier_id}.jsonl"

    data = dict(out["data"])
    data["subset_size"] = subset
    if is_na:
        data["train_path"] = out["data"]["train_path"]  # v1 原值（该档不跑）
    else:
        data["train_path"] = f"/work/elam-v5/{V2_SUBSET_DIRNAME}/{tier_id}.jsonl"
    data["source"] = source
    # ★ source_jsonl = **原始源文件**（不是 subsets-v2 冻结件）：runner 的
    #   `head -n $SUBSET <源>` 本身就把"前 N 条"再算一遍并落 <RUN_ROOT>/<T>/subsets/。
    #   让 head 的输入是**只读的原始源**（而非冻结子集）是**防呆**：即使有人把 RUN_ROOT
    #   设成数据根、导致 head 的输出路径恰好等于输入路径（shell 先清空输出再读输入 ⇒
    #   0 行），被清空的也只是"重算得到的那份"，永不可能清空 228 上的冻结件或原始数据。
    data["source_jsonl"] = host_src
    data["source_jsonl_container"] = src_rel
    data["extract_command"] = (
        None if is_na else f"head -n {subset} {host_src} > {host_subset}"
    )
    data["source_rule"] = (
        None if is_na else (
            f"subsets-v2（v2 冻结子集）：本档取 **{source}** 源前 {subset} 条"
            + ("（= 48 × 卡数）" if source == SOURCE_FORMAL else "（沿用 v1 算法下限）")
        )
    )
    data["subset_file_pregenerated"] = not is_na
    out["data"] = data

    # ④ 步数两字段：native GRASPO 走真实式；其余 45 档与 v1 同式同值（v1 已按真机校准）
    if is_native_graspo(out):
        dp_size = v2_dp_size(tier_id)
        per_epoch = v2_native_steps_per_epoch(subset, dp_size)
        out["step_formula_v2"] = (
            f"T = floor({subset}/{dp_size}) = {subset // dp_size}；"
            f"每 epoch = ceil(T/Q) = ceil({subset // dp_size}/{Q}) = {per_epoch}"
            f"（Q={Q}；dp_size={dp_size}；PP 档 dp_size=1 ⇒ 不分片）"
        )
    else:
        per_epoch = int(out["expected_optimizer_steps_per_epoch"])
        out["step_formula_v2"] = (
            "非 native GRASPO：沿用 v1 口径（子集/卡数与 epoch 数均未变）"
            f" = {per_epoch}"
        )
    reachable = per_epoch * int(gm.MATRIX_MAX_EPOCHS)

    out["expected_optimizer_steps_per_epoch"] = per_epoch
    out["expected_optimizer_steps_reachable"] = reachable
    # 计划步数（ckpt 保留口径）：native GRASPO 的 v2 计划值 = 该档 v2 档配置的 save_steps
    # （由 generate_matrix_v2.py 按 v1 同式算出；本脚本**只读**它，不另算一份）。
    if is_native_graspo(out):
        out["checkpoint_save_steps"] = int(
            read_yaml_scalar(PROJECT_ROOT / V2_CONFIG_DIR_REL / f"{tier_id}.yaml", "save_steps")
        )
        out["expected_optimizer_steps"] = out["checkpoint_save_steps"]

    # ⑤ formal_gate：门槛值与 v1 一致（不放宽），只重算标记与 basis
    gate = dict(out["acceptance"]["formal_gate"])
    gate["min_optimizer_steps"] = MIN_OPTIMIZER_STEPS
    gate["min_epochs"] = int(gm.MATRIX_MAX_EPOCHS)
    gate["step_gate_applicability"] = (
        gm.STEP_GATE_APPLICABLE
        if reachable >= MIN_OPTIMIZER_STEPS
        else gm.STEP_GATE_NOT_APPLICABLE_STRUCTURAL
    )
    gate["step_gate_basis"] = (
        f"expected_optimizer_steps_reachable={reachable}；per_epoch={per_epoch}；"
        f"max_epochs={gm.MATRIX_MAX_EPOCHS}；"
        + (
            f"可产出步数上限={reachable} ≥ 门槛 {MIN_OPTIMIZER_STEPS} ⇒ 门槛照常判"
            if reachable >= MIN_OPTIMIZER_STEPS
            else (
                f"该档可产出步数上限={reachable} < 门槛 {MIN_OPTIMIZER_STEPS} ⇒ "
                "步数门槛对本题不适用，判「⚠ 口径不可测」而不是「❌ 失败」"
            )
        )
    )
    gate["min_train_samples"] = 20 if str(out["algorithm"]) in {"GRASPO", "OPD"} else 100
    out["acceptance"] = {**copy.deepcopy(out["acceptance"]), "formal_gate": gate}
    return out


def build_v2_audit(entries: list[dict]) -> dict:
    """v2 的 step 门槛审计节（形状与 v1 兼容，内容按 v2 真值）。

    ★ 保留与 v1 **同名**的键（``threshold`` / ``rule`` / ``structural_limited_tiers`` …），
      避免任何读清单的旧代码 KeyError；v2 下 ``structural_limited_tiers`` 应为**空**。
    """
    limited: list[dict] = []
    for entry in entries:
        gate = entry["acceptance"]["formal_gate"]
        marker = gate["step_gate_applicability"]
        reachable = int(entry["expected_optimizer_steps_reachable"])
        # fail-closed：标记与公式必须一致（防手改 / 防漂移）
        expected_marker = (
            gm.STEP_GATE_NOT_APPLICABLE_STRUCTURAL
            if reachable < MIN_OPTIMIZER_STEPS
            else gm.STEP_GATE_APPLICABLE
        )
        if marker != expected_marker:
            raise AssertionError(
                f"{entry['tier_id']}: step 门槛标记={marker} 与公式重算的 {expected_marker} "
                f"不一致（reachable={reachable}，门槛={MIN_OPTIMIZER_STEPS}）⇒ 生成期 fail-closed"
            )
        if expected_marker == gm.STEP_GATE_NOT_APPLICABLE_STRUCTURAL:
            limited.append(
                {
                    "tier_id": entry["tier_id"],
                    "algorithm": entry["algorithm"],
                    "backend": entry["backend"],
                    "cards": entry["cards"],
                    "subset_size": entry["data"]["subset_size"],
                    "per_epoch": entry["expected_optimizer_steps_per_epoch"],
                    "max_epochs": int(gm.MATRIX_MAX_EPOCHS),
                    "reachable": reachable,
                    "min_optimizer_steps": MIN_OPTIMIZER_STEPS,
                    "derivation": entry.get("step_formula_v2", ""),
                }
            )
    return {
        "threshold": MIN_OPTIMIZER_STEPS,
        "threshold_source": "docs/capability-matrix.html §1「正式记录的硬前提」（v2 不放宽）",
        "rule": (
            "reachable >= threshold ⇒ 门槛照常判；reachable < threshold ⇒ 判「⚠ 口径不可测」"
            "（不是 ❌ 失败，也不是 ✅ 通过）"
        ),
        "structural_limited_tiers": limited,
        "structural_limited_note": (
            "v2 下应为**空列表**：9 档 GRASPO×native 的子集已抬到 48×卡数（每 epoch 6/12/24 步），"
            "其余 45 档口径未变且本就 ≥5 步 ⇒ 54 档的 step 门槛**全部适用**。"
            if not limited else
            "★ 仍有结构受限档（v2 未达预期，需复核子集/卡数/公式）——见上方条目。"
        ),
    }


def build_v2_manifest(
    v1_manifest: dict, data_root: str, pregen_dir: str, attempt_root: str,
    v1_manifest_sha: str, config_fingerprint: str,
) -> dict:
    v1_tiers = v1_manifest["tiers"]
    entries = [patch_tier(t, data_root, pregen_dir) for t in v1_tiers]

    # 顶层：深拷贝 v1 → 只改版本/来源/数据源/审计节，其余（counts/models/feasibility_model…）
    # 逐字节保留（counts 不随 v2 变：档数、算法、卡数、后端、模式、适用性都没变）。
    out = copy.deepcopy(v1_manifest)
    out["source"] = "docs/capability-matrix.html §1/§6/§7（v2 定稿；配置列/数据列 = HTML）"
    out["config_version"] = "v2"
    out["supersedes"] = {
        "relationship": "v1 清单（matrix54_manifest.json）**未被改动**；本文件是它的 v2 派生版",
        "base_manifest": "tests/e2e/matrix54_manifest.json",
        "base_manifest_sha256": v1_manifest_sha,
        "config_v2_fingerprint": config_fingerprint,
        "generator": "tests/e2e/generate_matrix_manifest_v2.py",
        "config_v2_dir": V2_CONFIG_DIR_REL,
        "subset_dir_container": f"/work/elam-v5/{V2_SUBSET_DIRNAME}",
        "subset_dir_host": pregen_dir,
        "host_data_root": data_root,
        "attempt_root": attempt_root,
    }
    out["changes_vs_v1"] = {
        "config": f"{V1_CONFIG_DIR_REL}/T0NN.yaml → {V2_CONFIG_DIR_REL}/T0NN.yaml（51 个可跑档）",
        "subset_size": (
            "9 档 GRASPO×native：20 → 48×卡数（48/96/192）；其余 45 档不变（CPT/SFT=100、GRASPO/OPD=20）"
        ),
        "train_path": f"/work/elam-v5/subsets/ → /work/elam-v5/{V2_SUBSET_DIRNAME}/（51 个可跑档）",
        "steps": (
            "expected_optimizer_steps_per_epoch/_reachable 按**真实步进语义**重算："
            "native GRASPO = ceil(floor(subset/dp_size)/Q)"
        ),
        "new_fields": [
            "config_version", "supersedes", "changes_vs_v1", "data_sources", "checkpoint_retention_v2",
            "per-tier data.source / data.source_jsonl / data.source_jsonl_container / "
            "data.extract_command / data.source_rule / data.subset_file_pregenerated / step_formula_v2",
        ],
        "unchanged": (
            "counts / models / feasibility_model / runtime.checkpoint_retention / 每档的 "
            "model*/algorithm/mode/backend/cards/gpus/status/feasibility/acceptance.criteria"
        ),
    }
    # 顶层 data 段：v1 的算法级口径保留（向后兼容），另加 v2 的逐档口径。
    top_data = copy.deepcopy(out["data"])
    top_data["subset_dir_v1"] = top_data.get("subset_dir")
    top_data["subset_dir"] = f"/work/elam-v5/{V2_SUBSET_DIRNAME}"
    top_data["subset_dir_by_version"] = {
        "v1": "/work/elam-v5/subsets",
        "v2": f"/work/elam-v5/{V2_SUBSET_DIRNAME}",
    }
    top_data["subset_size_by_tier_v2"] = dict(V2_SUBSET_SIZE_BY_TIER)
    top_data["subset_size_v2_rule"] = (
        "9 档 GRASPO×native = 48 × 卡数（48/96/192）；其余 45 档沿用 v1 算法下限"
        "（仍在 subset_size_by_algorithm 里，逐字节未改）"
    )
    top_data["source_by_tier_v2"] = {
        e["tier_id"]: e["data"].get("source") for e in entries
    }
    top_data["source_switch_rule"] = (
        "★ 严禁全局 export 切换：mini 集不是正式集头部，全局切换会让 45 档拿到错的 100 条。"
        "逐档切换见顶层 data_sources.runner_wiring"
    )
    # v1 的 train_source="mini" 是**单源**口径，v2 是**逐档混合** ⇒ 另立字段，
    # 不改 v1 字段（改它会让老读者以为整批都从 mini 取数）。
    top_data["train_source_v2"] = "hybrid（逐档：9 档 formal + 45 档 mini）"
    top_data["train_source_v1_field_note"] = (
        "上方 train_source/train_source_env_var/*_note 是 v1 单源口径的**原样保留**（向后兼容），"
        "**不代表 v2 的实际取数**；v2 以每档 data.source / data.source_jsonl 为准。"
    )
    out["data"] = top_data
    # 数据源节：把"逐档来源"和"怎么用"写在一起（自包含）
    n_formal = sum(1 for e in entries if e["data"].get("source") == SOURCE_FORMAL)
    n_mini = sum(1 for e in entries if e["data"].get("source") == SOURCE_MINI)
    out["data_sources"] = {
        "rule": (
            "★ 逐档、**进程作用域**切换：不得用全局 export 一次性切源（mini 集不是正式集头部，"
            "全局切换会让 45 档拿到错的 100 条）。用法见 launcher"
        ),
        "sources": {
            SOURCE_FORMAL: {
                "container": "/work/elam-v5/data/train.jsonl",
                "host_relative_under_data_root": "data/train.jsonl",
                "line_count": FORMAL_LINE_COUNT,
                "tiers": sorted(e["tier_id"] for e in entries if e["data"].get("source") == SOURCE_FORMAL),
            },
            SOURCE_MINI: {
                "container": "/work/elam-v5/mini-dataset/mini-short-mm-train.jsonl",
                "host_relative_under_data_root": "mini-dataset/mini-short-mm-train.jsonl",
                "line_count": MINI_LINE_COUNT,
                "note": "不是正式集头部（按「最短多模态」挑的）⇒ 只能逐档切换，禁止全局切换",
                "tiers": sorted(e["tier_id"] for e in entries if e["data"].get("source") == SOURCE_MINI),
            },
        },
        "tier_count": {SOURCE_FORMAL: n_formal, SOURCE_MINI: n_mini, "not_applicable": len(NOT_APPLICABLE_TIERS)},
        "pregenerated_subsets": {
            "host_dir": pregen_dir,
            "container_dir": f"/work/elam-v5/{V2_SUBSET_DIRNAME}",
            "generator": "tests/e2e/make_subsets_v2.py",
            "manifest": f"{pregen_dir}/subsets-v2.manifest.json",
            "not_nested_under_attempt_root": not pregen_dir.startswith(attempt_root.rstrip("/") + "/"),
            "note": (
                "冻结预生成件（逐档具名、逐档校验行数+sha256）；真跑时 runner 仍会按 "
                "data.subset_size 从该档源（data.source_jsonl）head 出 "
                "<RUN_ROOT>/<T>/subsets/<T>.jsonl 作为**运行时副本**，两者内容等价。"
            ),
            "why_source_is_raw_not_frozen": (
                "data.source_jsonl 指向**原始源**（data/train.jsonl / mini-...jsonl）而非 subsets-v2 冻结件："
                "head 的输出路径若与输入路径相同（误把 RUN_ROOT 指向数据根）会被 shell 清空 ⇒ "
                "用原始源做输入时，最坏情况只是重算失败，绝不清空冻结件。"
            ),
        },
        "host_paths": {
            "data_root": data_root,
            "pregen_subset_dir": pregen_dir,
            "attempt_root_recommended": attempt_root,
            "rule": "三者互不嵌套；attempt_root 是 runner 的 RUN_ROOT（可写），pregen_dir 只读冻结件",
        },
        "runner_wiring": {
            "launcher": "rig/run_matrix54_v2source.sh",
            "how": (
                "launcher 读本清单的 tiers[].data.source_jsonl，对**单个档**设 "
                "GRASPO_ELAM_MINI_JSONL=<该档源> 后 exec run_matrix54.sh ⇒ 逐档进程作用域、"
                "退出即消失、零跨档残留。runner 与 driver 均未改。"
            ),
            "driver_pointing": (
                "driver 的 --manifest <path> 直接指向本文件（或 export GRASPO_RUNNER_MANIFEST=<path>）；"
                "driver 只透传该变量，不改 driver 代码。"
            ),
        },
    }
    # ckpt 保留：语义与 v1 一致，只把 v2 的逐档事实写清楚
    out["checkpoint_retention_v2"] = {
        "policy": v1_manifest["runtime"]["checkpoint_retention"]["policy"],
        "keep_per_tier": 1,
        "semantics": "与 v1 逐条一致（每档只留一份 final/；native 只认非空 final/）",
        "v2_facts": {
            "native_graspo_9_tiers": {
                "save_steps_planned": 48,
                "reachable_steps": {
                    "T028": 6, "T029": 6, "T030": 6, "T034": 6, "T040": 6, "T041": 6, "T042": 6,
                    "T035": 12, "T036": 24,
                },
                "why_only_final": (
                    "native 中间段叫 step_<N>、runner 的清理范围只有 checkpoint-*；v2 的 "
                    "save_steps=48 > 可达步数（6/12/24）⇒ 计划值不触发、只落终态 final/，"
                    "与 v1 的刻意设计同源（见 runtime.checkpoint_retention.how 的 native 例外段）。"
                ),
            },
            "other_45_tiers": "与 v1 逐值相同（未改子集/卡数/epoch）",
        },
    }
    out["step_gate_audit"] = build_v2_audit(entries)
    out["tiers"] = entries
    return out


def check(manifest_path: Path) -> int:
    """只读校验：v2 清单的字段自洽性（不依赖网络/服务器）。

    配置 v2 目录、冻结子集目录都是**可选**交叉核对：本机看不到就**打印跳过原因**
    （不静默通过），看到就逐档核对。
    """
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    entries = manifest["tiers"]
    config_dir = PROJECT_ROOT / V2_CONFIG_DIR_REL
    have_configs = config_dir.is_dir()
    problems: list[str] = []
    if len(entries) != 54:
        problems.append(f"档数 {len(entries)} != 54")
    if manifest.get("config_version") != "v2":
        problems.append("config_version != v2")
    for e in entries:
        tid = e["tier_id"]
        if e.get("config") and not str(e["config"]).startswith(V2_CONFIG_DIR_REL):
            problems.append(f"{tid}: config 不在 {V2_CONFIG_DIR_REL}")
        per = int(e["expected_optimizer_steps_per_epoch"])
        reach = int(e["expected_optimizer_steps_reachable"])
        if reach != per * int(gm.MATRIX_MAX_EPOCHS):
            problems.append(f"{tid}: reachable({reach}) != per_epoch({per}) × max_epochs")
        if reach < MIN_OPTIMIZER_STEPS:
            problems.append(f"{tid}: reachable={reach} < 门槛 {MIN_OPTIMIZER_STEPS}")
        if tid in V2_SUBSET_SIZE_BY_TIER:
            want = V2_SUBSET_SIZE_BY_TIER[tid]
            if int(e["data"]["subset_size"]) != want:
                problems.append(f"{tid}: subset_size != {want}")
            if e["data"].get("source") != SOURCE_FORMAL:
                problems.append(f"{tid}: source != formal")
            if have_configs:
                dp = v2_dp_size(tid)
                want_steps = v2_native_steps_per_epoch(want, dp)
                if per != want_steps:
                    problems.append(f"{tid}: per_epoch {per} != 公式值 {want_steps}（dp={dp}）")
            elif not re.search(rf"= {per}\b", str(e.get("step_formula_v2", ""))):
                problems.append(f"{tid}: step_formula_v2 与 per_epoch={per} 不自洽")
        elif tid not in NOT_APPLICABLE_TIERS:
            if e["data"].get("source") != SOURCE_MINI:
                problems.append(f"{tid}: source != mini")
        if not e.get("config") and tid not in NOT_APPLICABLE_TIERS:
            problems.append(f"{tid}: 可跑档却缺 config")
    if manifest["step_gate_audit"]["structural_limited_tiers"]:
        problems.append("v2 仍存在 structural_limited 档")

    print(
        "配置 v2 交叉核对："
        + (f"已逐档复算（{config_dir}）" if have_configs
           else f"**跳过**（本机看不到 {config_dir}；改由清单内的 step_formula_v2 自洽性核对）")
    )
    # 冻结子集交叉核对（能看到才做）
    subset_dir = Path(manifest["data_sources"]["pregenerated_subsets"]["host_dir"])
    if subset_dir.is_dir():
        for e in entries:
            if not e["data"].get("subset_file_pregenerated"):
                continue
            p = subset_dir / f"{e['tier_id']}.jsonl"
            if not p.is_file():
                problems.append(f"{e['tier_id']}: 冻结子集缺失 {p}")
                continue
            lines = p.read_bytes().count(b"\n")
            if lines != int(e["data"]["subset_size"]):
                problems.append(
                    f"{e['tier_id']}: 冻结子集行数 {lines} != subset_size {e['data']['subset_size']}"
                )
        print(f"冻结子集核对：{subset_dir}（已逐档核对行数）")
    else:
        print(f"冻结子集核对：**跳过**（本机看不到 {subset_dir}；以 228 上的 subsets-v2.manifest.json 为准）")

    if problems:
        for p in problems:
            print(f"MISMATCH: {p}")
        return 1
    print(f"check OK: 54 档；formal={sum(1 for e in entries if e['data'].get('source')=='formal')} "
          f"mini={sum(1 for e in entries if e['data'].get('source')=='mini')}；"
          "全部 reachable ≥ 5")
    return 0


def print_steps(v1_manifest: dict, pregen_dir: str = "") -> None:
    print(f"Q={Q} 门槛={MIN_OPTIMIZER_STEPS} epoch={gm.MATRIX_MAX_EPOCHS}  公式=ceil(floor(subset/dp)/Q)")
    for tid in sorted(V2_SUBSET_SIZE_BY_TIER):
        size = V2_SUBSET_SIZE_BY_TIER[tid]
        dp = v2_dp_size(tid)
        steps = v2_native_steps_per_epoch(size, dp)
        v1e = next(t for t in v1_manifest["tiers"] if t["tier_id"] == tid)
        print(
            f"  {tid}: subset {v1e['data']['subset_size']}→{size}  dp={dp}  "
            f"T={size // dp}  steps={steps}  (v1 朴素/上界式={v1e['expected_optimizer_steps_per_epoch']})"
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="matrix54 v2 运行清单生成/校验（不跑训练）")
    parser.add_argument("--out", type=Path, default=V2_MANIFEST_PATH, help="v2 清单落点")
    parser.add_argument("--check", action="store_true", help="只校验已落盘的 v2 清单")
    parser.add_argument("--print-steps", action="store_true", help="只打印步数对照，不写文件")
    parser.add_argument("--host-data-root", default=DEFAULT_HOST_DATA_ROOT,
                        help="228 上的 ELAM V5 原始数据根（data/、mini-dataset/、images/）")
    parser.add_argument("--pregen-subset-dir", default=DEFAULT_PREGEN_SUBSET_DIR,
                        help="228 上的 v2 冻结子集目录（其下 T0NN.jsonl）")
    parser.add_argument("--attempt-root", default=DEFAULT_ATTEMPT_ROOT,
                        help="runner 的 RUN_ROOT 推荐值（运行产物根；须与上面两者不嵌套）")
    parser.add_argument(
        "--config-fingerprint", default=None,
        help="配置 v2 总指纹（默认从 task-config-freeze-v2/fingerprint.txt 读；读不到则记 unknown）",
    )
    args = parser.parse_args(argv)

    if args.check:
        return check(args.out)

    v1_bytes = V1_MANIFEST_PATH.read_bytes()
    v1_manifest_sha = sha256_bytes(v1_bytes)
    v1_manifest = json.loads(v1_bytes.decode("utf-8"))

    if args.print_steps:
        print_steps(v1_manifest, args.pregen_subset_dir)
        return 0

    fingerprint = args.config_fingerprint
    if not fingerprint:
        fp_file = (
            PROJECT_ROOT / ".local" / "hb-workspace" / "20260923-analysis"
            / "task-config-freeze-v2" / "fingerprint.txt"
        )
        if fp_file.is_file():
            m = re.search(r"总指纹:\s*([0-9a-f]{64})", fp_file.read_text(encoding="utf-8"))
            fingerprint = m.group(1) if m else "unknown"
        else:
            fingerprint = "unknown"

    manifest = build_v2_manifest(
        v1_manifest, args.host_data_root, args.pregen_subset_dir, args.attempt_root,
        v1_manifest_sha, fingerprint,
    )
    text = json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"
    args.out.write_text(text, encoding="utf-8")
    rc = check(args.out)
    print(f"v2 清单已写出：{args.out}（sha256={sha256_bytes(text.encode())}）")
    print(f"  v1 基线 sha256={v1_manifest_sha}（v1 清单字节未动）")
    print(f"  配置 v2 总指纹={fingerprint}")
    return rc


if __name__ == "__main__":
    sys.exit(main())
