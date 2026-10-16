#!/usr/bin/env python3
"""matrix54 **配置 v2 运行清单**（v2 manifest）生成器 —— 只新增，不改 v1。

为什么是独立脚本 + 独立产物（§6.1 简单优先 / §1.4 单一真相源）：
  · ``tests/e2e/matrix54_manifest.json`` 是 v1 的冻结产物，**必须逐字节不变**；
    本脚本**只读**它（作为 v1 台账的单一真相源），把它按 v2 规则**改写出一份新清单**
    ``tests/e2e/matrix54_manifest_source.json``。v1 生成器一行未动。
  · 改写面**只有四类**，逐条显式、逐条可核：
      ① ``config``：``samples/configs/matrix54/T0NN.yaml`` → ``samples/configs/matrix54-generated/T0NN.yaml``
         （runner 实际读取的就是这个字段，见 ``run_matrix54.sh`` 的 ``CONFIG="${CONFIG_STATUS%%|*}"``）。
         ★ **2026-10-16 改指**（消「跑次用的配置 ≠ 清单声明的配置」）：跑次树逐字引用的是
         ``matrix54-generated/``（= ``tools/matrix/gen_configs.py`` 从
         ``docs/capability-matrix.data.json`` 生成、经实测与矩阵声明 lr **逐档一致**的现行世代）；
         改指前清单指向 ``matrix54-v2/``，而它仍是旧 lr/epoch 口径（**21 档与矩阵声明不一致**）。
         ★ **2026-10-16 步数/检查点口径改指**（消「清单声明的步数 ≠ 跑次实际」）：``checkpoint_save_steps``
         与 ``expected_optimizer_steps_reachable`` 的 epoch 数**一律从该档 ``config`` 所指的
         ``matrix54-generated/`` yaml 现读**（``training.save_steps`` / ``training.max_epochs``）。
         改指前：epoch 数走 **v1 冻结期常量** ``MATRIX_MAX_EPOCHS=1`` ⇒ 全 54 档 ``reachable``
         **低估 2–8 倍**（T010 清单 100 vs 真值 400）；``checkpoint_save_steps`` 则分两路
         （native 读**已退役**的 ``matrix54-v2/``、其余走 v1 透传）⇒ 与跑次 yaml **51/51 不一致**。
         ``matrix54-v2/`` 现**只**用于读 ``dp_size``（native 分片口径），⛔ 不再作步数/保存口径来源。
      ② ``data.subset_size``：9 档 ``GRASPO × native`` → ``48 × 卡数``（48/96/192）；
         其余 45 档**照 v1**（CPT/SFT=100、GRASPO/OPD=20）。3 个 not_applicable 档保持 v1。
      ③ ``data.train_path`` / 步数两字段 / ``acceptance.formal_gate.*``：按 v2 真值重算
         （``expected_optimizer_steps_per_epoch`` / ``_reachable`` 用**真实步进语义**，
         不是 v1 的朴素均分；见 :func:`v2_native_steps_per_epoch`）
      ④ ``data.source`` / ``data.source_jsonl``（**新增字段**）：逐档标注取数来源与命令，
         供 ``tests/e2e/run_matrix54_source.sh`` 做**单档进程作用域**的读取源切换。
         既有字段一个都没删、没改语义 ⇒ 老消费者（``collect_results.py``）不受影响。

★ 步数真值（逐条有代码依据，见 ``docs/capability-matrix.html`` §1 与
  ``.local/hb-workspace/20260923-analysis/task-config-freeze-v2/changes.md`` §1）::

      T = floor(subset_size / dp_size)          # native 只按 DP 分片；PP 档 dp_size=1 ⇒ 不分片
      每 epoch 步数 = ceil(T / Q) = (T + Q - 1) // Q        Q = rollout_queue_batch_size = 8

  v2 子集 48×卡数 ⇒ 每卡 prompt 数 P = 48（PP 档例外：dp_size=1 ⇒ T=96/192）::

      ★ 每 epoch 步数**不随 epoch 数变**；""整个 run 可达""= 每 epoch 步数 × **该档 yaml 的
        ``training.max_epochs``**（SFT 4／CPT/OPD 2／GRASPO 8，2026-10-15 主席裁定；唯一真相源）。

      T028 T029 T030 T034 T040 T041 T042  →  T = 48  →  6 步
      T035（dp=1, pp=2）                  →  T = 96  → 12 步
      T036（dp=1, pp=4）                  →  T = 192 → 24 步
      全部 ≥ 硬前提 5 步 ⇒ step 门槛对 54 档**全部适用**（v2 无 structural_limited 档）。

用法::

    .venv/bin/python tests/e2e/generate_matrix_manifest_source.py            # 写 v2 清单
    .venv/bin/python tests/e2e/generate_matrix_manifest_source.py --check    # 只校验已落盘的 v2 清单
    .venv/bin/python tests/e2e/generate_matrix_manifest_source.py --print-steps
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
V2_MANIFEST_PATH = _HERE.parent / "matrix54_manifest_source.json"

# ★ 生成链防丢失（2026-10-16）：判定实现与 v1 生成器/测试共用同一份（§1.4 单一真相源）。
sys.path.insert(0, str(_HERE.parent))
from regen_field_guard import (  # noqa: E402  （sys.path 就位之后导入）
    guard_regeneration_keeps_on_disk_fields,
    matches_any,
)
from gpu_assignment import (  # noqa: E402  （§1.4 单一真相源：锁卡集合只有这一份定义）
    GPU_ALLOWED,
    GPU_FORBIDDEN_PRODUCTION,
)

#: ★ **跑次实际读取的档配置目录**（= 矩阵生成件）：由 ``tools/matrix/gen_configs.py`` 从
#:   ``docs/capability-matrix.data.json`` 确定性生成 ⇒ 可复现、且与矩阵声明逐档一致。
#:   ``tiers[*].config`` 一律指这里（⛔ 不再指 ``matrix54-v2/``，那是旧 lr/epoch 世代）。
CONFIG_DIR_REL = "samples/configs/matrix54-generated"
#: ★ **native ``dp_size`` 口径来源目录**（v2 冻结历史，**只读**）：**只**读 ``dp_size``。
#:   ★ 2026-10-16：``save_steps`` **不再**从这里读（它已退役，T028 实测 48 vs 现行 49）——
#:   步数/保存口径一律从 ``CONFIG_DIR_REL``（= 跑次实际读的 yaml）现读。
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
#: ★ §15.1：清单**只存占位符**，不存宿主绝对路径（A 类/公开仓库禁止内网路径）。
#: 运行期由 `tests/e2e/run_matrix54_source.sh` 用宿主环境展开 ``${VAR}``；
#: 本地需要真实路径时用 ``--host-data-root/--pregen-subset-dir/--attempt-root`` 显式覆盖
#: （覆盖值会原样写进清单 ⇒ 仅供本机/228 临时使用，⛔ 不得提交真实宿主路径）。
DEFAULT_HOST_DATA_ROOT = "${GRASPO_ELAM_DATA_ROOT}"
DEFAULT_PREGEN_SUBSET_DIR = "${GRASPO_PREGEN_SUBSET_DIR}"
DEFAULT_ATTEMPT_ROOT = "${GRASPO_ATTEMPT_ROOT}"

#: 9 档「GRASPO × native」的 v2 子集条数（= 48 × 卡数）。
V2_SUBSET_SIZE_BY_TIER: dict[str, int] = {
    "T028": 48, "T029": 96, "T030": 192,
    "T034": 48, "T035": 96, "T036": 192,
    "T040": 72, "T041": 144, "T042": 288,
}
#: ★ 声明式例外（2026-10-02 批准）：例外档的「每卡 prompt 数」。
#: 历史 parity 声明（v2 配置生成器 tests/e2e/generate_matrix_v2.py 已退役留档 .local/repo-moved-out/tests/e2e/，不可再核）：本常量为在役唯一 P_TARGET_OVERRIDE 定义。
P_TARGET_OVERRIDE: dict[str, int] = {"T040": 72, "T041": 72, "T042": 72}

#: ★ 计划步数锚定覆盖（2026-10-02 批准）：这三档的计划步数固定为 6，不随子集放大变 9。
#: 历史 parity 声明（同上，generate_matrix_v2.py 已退役留档）：本常量为在役唯一 PLANNED_STEPS_OVERRIDE 定义。
PLANNED_STEPS_OVERRIDE: dict[str, int] = {"T040": 6, "T041": 6, "T042": 6}
#: 3 个 not_applicable 档（v1 已定，v2 不改；不产出可跑配置）。
NOT_APPLICABLE_TIERS: tuple[str, ...] = ("T052", "T053", "T054")
#: 来源名（进 ``data.source``；也是 ``tests/e2e/run_matrix54_source.sh`` 的判据字面量）。
SOURCE_FORMAL = "formal"
SOURCE_MINI = "mini"

Q = 8  # 与 v1 生成器同源（下面用断言核对，不另抄常量）
MINI_LINE_COUNT = 100
FORMAL_LINE_COUNT = 6378
MIN_OPTIMIZER_STEPS = 5

#: ★ **A2 的最低 epoch 门槛**（= 1）：语义是"至少跑完几个 epoch"，⛔ **不是**该档的计划 epoch 数。
#:   与 v1 冻结常量 ``gm.MATRIX_MAX_EPOCHS``（=1）**解耦**（值相同 ⇒ 行为逐字不变）。
#:   为什么必须解耦（2026-10-16）：epoch 数只能有**一个**真相源 = 跑次档配置 yaml 的
#:   ``training.max_epochs``（= SFT 4／CPT/OPD 2／GRASPO 8）。若这里继续引用 v1 常量，
#:   "epoch 的第二个来源"就还在，日后仍会有人拿它去算步数（本次全 54 档 `reachable`
#:   低估 2–8 倍正是这么来的）。
MIN_EPOCHS_GATE = 1


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


# ═══════════════════════════════════════════════════════════════════════════════
# ★★ 18 档口径扩面（2026-10-16 主席裁定）的**单一真相源**（修正轮 1 / 🔴B1a）
#
# 病史（复核报告 B1a，实测）：本清单的 18 档 ``subset_size`` / 步数 **只存在于在盘文件里**，
#   生成器没有任何输入或表编码它们 ⇒ 任何人走「声明的唯一写路径」重生成，18 档就
#   ``subset 100→20``、步数 ÷5，而 ``guard_regeneration_keeps_on_disk_fields()`` 只防
#   **字段路径丢失**、``check()`` 对非 native 档又不重算 ⇒ **rc=0 且仍打印 `check OK`**。
#
# 修法（⛔ 不是"加豁免/白名单变绿"，而是把真值编码进生成链 ＋ 让值回退 fail-closed）：
#   · **声明口径**唯一真相源 = ``tests/e2e/make_subsets.py::MINI_SUBSET_TIERS_DECLARED``
#     （18 档 = 100；其中 3 档具名豁免未落地，见 ``DECLARED_NOT_LANDED``）；
#   · **在盘落地真值** = ``make_subsets.MINI_TIERS``（15 档 = 100、T052–54 = 20）；
#   · 两者**只读加载**，⛔ 不在本文件另抄一份常量（§1.4）。
# ═══════════════════════════════════════════════════════════════════════════════
MAKE_SUBSETS_PATH = PROJECT_ROOT / "tests/e2e" / "make_subsets.py"


def load_make_subsets_module():
    """只读加载子集生成器（拿声明口径表与在盘真值表；不触发任何写盘）。"""
    spec = importlib.util.spec_from_file_location("_manifest_v2_make_subsets", MAKE_SUBSETS_PATH)
    if spec is None or spec.loader is None:  # pragma: no cover
        raise RuntimeError(f"无法加载子集生成器：{MAKE_SUBSETS_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ms = load_make_subsets_module()

#: ★ 声明口径表（18 档 = 100），来自唯一真相源。
MINI_DECLARED_SUBSET_SIZE: dict[str, int] = dict(ms.MINI_SUBSET_TIERS_DECLARED)
#: ★ 在盘落地真值表（15 档 = 100、T052–54 = 20）。
MINI_LANDED_SUBSET_SIZE: dict[str, int] = dict(ms.MINI_TIERS)
#: ★ 具名豁免（声明 100、数据未落地）。
MINI_DECLARED_NOT_LANDED: dict[str, str] = dict(ms.DECLARED_NOT_LANDED)
#: 生成器**只读**判据用的「仓内冻结子集目录」（宿主机路径是运行时占位符，见 `resolve_subset_check_dir`）。
REPO_SUBSET_DIR_REL = V2_SUBSET_DIRNAME

#: v1 清单的**逐档 `data.subset_size`**（只读；供 `mini_tier_subset_size` 的「非声明面」下限核对，
#: 以及 `check()` 的现算复核）。★ 读一次、离线复用，⛔ 不每次重读文件（避免半截编辑时的竞态）。
V1_SUBSET_SIZE_BY_TIER: dict[str, int] = {
    str(t["tier_id"]): int(t["data"]["subset_size"])
    for t in json.loads(V1_MANIFEST_PATH.read_text(encoding="utf-8"))["tiers"]
}

#: v1 清单的**逐档 `expected_optimizer_steps_per_epoch`**（只读）：`check()` 用来复核
#: 非 native 档的 `per_epoch` 是否等于「v1 值 × 口径比率」（修正轮 2 / 🔴B-1）。
V1_PER_EPOCH_BY_TIER: dict[str, int] = {
    str(t["tier_id"]): int(t["expected_optimizer_steps_per_epoch"])
    for t in json.loads(V1_MANIFEST_PATH.read_text(encoding="utf-8"))["tiers"]
}

#: ★ 已具名登记的**既有**（⛔ 非本轮引入）`per_epoch` 漂移档 —— 第二轮全局审查 W-7，
#: 属"原包步数口径对齐只改了矩阵侧"，**须单独裁定**（⛔ 本轮不修）。
#: ★ 该表**不是静默白名单**：`cross_check_manifest_against_matrix()` 会断言这些档**确实仍在漂**，
#:   一旦被修好（不再漂）⇒ 立即报「豁免表陈旧」（fail-closed），必须删出。
KNOWN_PRE_EXISTING_PER_EPOCH_DRIFT: tuple[str, ...] = ("T017", "T018")

assert set(MINI_DECLARED_NOT_LANDED) <= set(MINI_DECLARED_SUBSET_SIZE), "豁免档必须在声明口径表里"
assert all(v == 100 for v in MINI_DECLARED_SUBSET_SIZE.values()), "18 档声明口径应为 100"


# ═══════════════════════════════════════════════════════════════════════════════
# ★★ 生成链防丢失（2026-10-16）：post-fill 注册表 + 「重生成不得移除在盘字段」守卫 ★★
#
# 病史：在盘 ``matrix54_manifest_source.json`` 是**跑次台账**，跑后会回填逐档
#   ``status`` / ``matrix_conclusion`` / ``availability`` / ``blocked_reason`` / ``live_evidence`` /
#   ``retest`` 等字段（manifest-authority 包实测：跑一次默认生成器就会把这些状态冲掉，
#   故当时被迫改用外科式改指）。整体重生成**静默丢真值**正是本守卫要挡的。
#
# 语义（与 ``tests/e2e/generate_matrix.py`` 同一份注册表的**下游继承**）：
#   · 本生成器的输出 = 深拷贝 v1 清单 + 显式改写，故 v1 清单里登记过的后填字段会**原样继承**；
#     注册表直接取 ``gm.POST_FILL_EXTRA_PATTERNS``（⛔ 不在这里另抄一份，避免两处漂移）。
#   · 生成前先比对**在盘旧文件**：新内容若比它少任何字段路径 ⇒ **fail-closed、不写盘、逐条列出**。
#   · 确需例外 ⇒ 进 ``MANIFEST_ALLOWED_FIELD_LOSS``（显式具名，用到即打印）。
POST_FILL_EXTRA_PATTERNS: tuple[str, ...] = tuple(getattr(gm, "POST_FILL_EXTRA_PATTERNS", ()))
MANIFEST_ALLOWED_FIELD_LOSS: tuple[str, ...] = ()


def is_post_fill_extra(path: str) -> bool:
    """该字段路径是否已登记为「只在盘出现、生成器不产出」的后填字段。"""
    return matches_any(path, POST_FILL_EXTRA_PATTERNS)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def config_dir_total_fingerprint(config_dir: Path) -> str:
    """配置目录**总指纹**（v2 冻结点算法，逐字）：按文件名排序 → 拼接每行
    ``<sha256>  <name>\\n`` → 取 sha256。

    ★ 默认**现算**，不回读 `.local/` 下的历史 `fingerprint.txt`：那种回读在文件陈旧时
    会把过期指纹静默写回清单（真实发生过：默认路径读的是 `7a230bad…`，而目录实测已是
    `066ca99a…`，见 `tests/e2e/matrix54_manifest_source.json` 的 `config_v2_fingerprint`）。

    fail-closed：目录不存在 / 为空即抛错——⛔ 不返回 ``"unknown"`` 之类占位值，
    否则陈旧指纹会以"看起来正常"的形式落盘。
    """
    if not config_dir.is_dir():
        raise FileNotFoundError(f"配置目录不存在，无法现算总指纹：{config_dir}")
    entries = sorted(
        (path.name, sha256_bytes(path.read_bytes()))
        for path in config_dir.iterdir()
        if path.is_file()
    )
    if not entries:
        raise AssertionError(f"配置目录为空，无法现算总指纹：{config_dir}")
    digest = hashlib.sha256()
    for name, sha in entries:
        digest.update(f"{sha}  {name}\n".encode())
    return digest.hexdigest()


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


def run_config_scalar(tier_id: str, key: str) -> "int | None":
    """★ 从**跑次实际读取**的档配置（``CONFIG_DIR_REL``）读一个标量；无该档 yaml ⇒ ``None``。

    ★ 为什么必须是这个目录（§1.4「跑次用的口径 ⇒ 清单声明的口径」）：``tiers[*].config``
      指的就是这里，runner 起跑读的也是这里。清单里的步数/检查点口径若从**别处**取，
      就会出现"清单声明 ≠ 跑次实际"（本次修复的正是这个病）。
    ★ 只允许**这一个**取值入口：本函数是 epoch 数与 ``save_steps`` 的唯一读取点，
      ⛔ 不许在别处再写一遍目录拼接（否则又造出第二真相源）。
    """
    path = PROJECT_ROOT / CONFIG_DIR_REL / f"{tier_id}.yaml"
    if not path.is_file():
        return None
    return int(read_yaml_scalar(path, key))


def run_config_epochs(tier_id: str) -> "int | None":
    """该档**跑次实际使用**的 epoch 数（= yaml 的 ``training.max_epochs``）。

    ★★ 这是 epoch 数的**唯一真相源**（§1.4）。病史（2026-10-16，本包修复）：
      原先 ``reachable`` 走 ``gm.MATRIX_MAX_EPOCHS``——那是 **v1 冻结期常量（=1）**，
      而 2026-10-15 主席把 epoch 口径改为 SFT=4／CPT/OPD=2／GRASPO=8 后，矩阵与 yaml
      都跟上了，**只有本生成器没跟** ⇒ 全 54 档 ``expected_optimizer_steps_reachable``
      被**低估 2–8 倍**（T010：清单 100 vs 真值 400）。
    ★ 读不到（该档无 yaml）⇒ 返回 ``None``，由调用方按**具名豁免**处置（⛔ 不猜、⛔ 不退回 1）。
    """
    return run_config_scalar(tier_id, "max_epochs")


#: ★ 无 yaml 档的**具名豁免**（epoch 口径不适用）：这 3 档 ``not_applicable``（当前无配方）
#:   ⇒ **三个 config 目录里都没有它们的 yaml**（实测 matrix54/ · matrix54-v2/ ·
#:   matrix54-generated/ 三处皆无）⇒ "按 yaml 现算"这条口径对它们**无定义**。
#:   ⇒ 生成器**原样保留其 v1 口径值**（⛔ 不发明、⛔ 不引入第二份 epoch 表），
#:   且三向机检对它们**具名豁免 ＋ 反向断言**（豁免表必须仍恰好等于本集合）。
#:   ★ 它们从不被跑 ⇒ 三键对它们**无任何消费方**（无 run ⇒ 无 ``RunEvidence``）⇒ 无害。
NOT_APPLICABLE_NO_CONFIG: tuple[str, ...] = tuple(NOT_APPLICABLE_TIERS)


def is_native_graspo(entry: dict) -> bool:
    return str(entry["backend"]) == "native" and str(entry["algorithm"]) == "GRASPO"


#: ★ 卡池确定性映射（**2026-10-16 起改为只含合法卡**；合法性 = :data:`GPU_ALLOWED` −
#:   :data:`GPU_FORBIDDEN_PRODUCTION` − 主席禁用集 ``{0,2}``）。
#: ★ 旧值 ``4卡->[4,5,6,7]`` **含两张生产锁卡**，过去只靠 :func:`locked_card_violation`
#:   的「恰等于整卡池即登记不 raise」豁免才通过 ⇒ 该豁免已按主席令**收窄**（池自身非法即 raise）。
#: 1卡->[4]｜2卡->[4,5]｜3卡->[4,5,6]｜4卡->[4,5,6,7]
GPU_POOL_BY_CARDS: dict[int, list[int]] = {1: [4], 2: [4, 5], 3: [4, 5], 4: [1, 3, 4, 5]}

#: ★ 按档的**显式卡位覆盖表**（项目准许的**资源调度例外**；⛔ 不改卡数）。
#: ★★ **2026-10-16 撤销唯一一条例外（T041 `[2,3]`）** —— 该值是 **2026-10-09 的一次性调度决策**
#:    （原文：「4卡最好是相邻的…把4-7给四卡」「0-3是先到先得」⇒ T041 挪到 0–3 内当时唯一可用的
#:    相邻空闲对 `[2,3]`，因为「GPU1 被他人 `rrc-runtime` 占用」）。它描述的是**当时的空闲状态**，
#:    不是设计约束；而 **GPU2 是第三方账号占用中的常驻邻居卡**（`docs/card-pool-contract.md:20-22`
#:    规则 4「邻居资源只读不占…只观测、不排队、不干预」）⇒ 该值把邻居的卡写进了我方声明。
#:    ⇒ 撤销后 T041 走默认映射 :data:`GPU_POOL_BY_CARDS`\[2\] = ``[4,5]``。
#: ★ `cards`（卡数）**保持 2 不变**：卡数属基础列不得改动；本表只改"用哪几张卡"。
#: ★ 未列出的档一律走 :data:`GPU_POOL_BY_CARDS`（上面那行映射**逐字未变**）。
TIER_GPU_OVERRIDE: dict[str, list[int]] = {
    # ★ 2026-10-16 车道重出（合法卡集 {1,3,4,5}；卡数逐档不变）：
    #   {1} = T031 关键路径（7.64 h，独占）｜{3,5} = 唯一 2 卡通道｜
    #   {4} = 共享卡，只放 ≤0.6 h 的 9B LoRA 1 卡档（⛔ 不上 27B/全参）｜{3} = 1 卡溢出
    "T031": [1],
    "T001": [4],
    "T010": [4],
    "T013": [4],
    "T028": [4],
    "T016": [3],
    "T022": [3],
    "T040": [3],
    "T002": [3, 5],
    "T011": [3, 5],
    "T014": [3, 5],
    "T017": [3, 5],
    "T023": [3, 5],
    "T029": [3, 5],
    "T032": [3, 5],
    "T041": [3, 5],
}

#: ★ **邻居/他人常驻卡**（⛔ 我方声明里绝不可出现）：GPU2 = 第三方账号占用中的常驻邻居卡
#:   （★ §15 去标识：本件为正式件，第三方**具名账号一律角色化表述**，⛔ 不写账号字面量）
#:   （`docs/card-pool-contract.md` 规则 4）。这是一份**具名**清单：新增邻居卡必须显式登记。
NEIGHBOR_CARDS: tuple[int, ...] = (2,)

#: ★ **具名锁卡集合**（任一档 `gpus` 都不得含其中任何一张）：
#:   生产锁卡 :data:`GPU_FORBIDDEN_PRODUCTION` ∪ 邻居卡 :data:`NEIGHBOR_CARDS`。
#:   ★ 另有**越界 leg**：任何**不在** :data:`GPU_ALLOWED` 内的卡同样被拒（见 :func:`locked_card_hits`）
#:   —— 越界集的基数无界（8, 99, …），故不枚举进本常量，而由 :data:`GPU_ALLOWED` 成员判定。
LOCKED_CARDS: tuple[int, ...] = tuple(sorted(
    set(GPU_FORBIDDEN_PRODUCTION) | set(NEIGHBOR_CARDS)
))

#: ★ 本世代卡池与项目锁卡契约的**已知冲突登记**（只登记、⛔ 不擅自改池；由指挥统一处置）。
#:   键 = 卡池 tuple；值 = 冲突的锁卡 tuple。生成时打印，并写入清单顶层
#:   ``gpu_pool_contract_conflicts``（机读、可审计），⛔ 不是静默放行。
POOL_CONTRACT_CONFLICTS: dict[tuple[int, ...], tuple[int, ...]] = {}
#: 生成期逐档的池冲突命中（供 main() 汇总打印）。
_uncaptured_pool_conflicts: list[tuple[str, tuple[int, ...], tuple[int, ...]]] = []


def locked_card_hits(gpus: "list[int] | tuple[int, ...]") -> tuple[int, ...]:
    """返回 ``gpus`` 里命中的**锁卡**（升序、去重）；空 tuple = 通过。

    命中 = 具名锁卡（:data:`LOCKED_CARDS` = 邻居卡 ∪ 生产锁卡）**或**越界卡（∉ :data:`GPU_ALLOWED`）。
    """
    out_of_range = {int(g) for g in gpus if int(g) not in set(GPU_ALLOWED)}
    return tuple(sorted(({int(g) for g in gpus} & set(LOCKED_CARDS)) | out_of_range))


def locked_card_violation(
    tier_id: str, gpus: "list[int] | tuple[int, ...]", cards: int
) -> "str | None":
    """★ 通用锁卡断言（fail-closed）：**任何档**的 ``gpus`` 都不得含被锁的卡。

    判据（三条，全部具名）：
      ① 不得含 :data:`NEIGHBOR_CARDS`（邻居常驻卡 ⇒ 占别人的卡）；
      ② 不得含 :data:`GPU_FORBIDDEN_PRODUCTION`（生产锁卡）；
      ③ 不得含任何**不在** :data:`GPU_ALLOWED` 内的卡（越界卡）。

    ★ 唯一例外（**具名、逐次打印、写入清单**）：当该值**恰好等于本世代卡池**
    :data:`GPU_POOL_BY_CARDS`\\[cards\\] 时，冲突来自**卡池本身**（2026-10-05 主席令「八卡全开」
    定的 ``{4,5,6,7}``），不是本档的调度选择 ⇒ **登记进 :data:`POOL_CONTRACT_CONFLICTS` 与
    ``_uncaptured_pool_conflicts``，不 raise**（⛔ 本生成器不擅自改池）。
    任何**其它**形态一旦命中 ⇒ 直接 raise（这正是能抓到 T041 `[2,3]` 的那条断言）。

    返回 None = 通过；否则返回拒绝理由（供测试与自证直接调用）。
    """
    hits = locked_card_hits(gpus)
    if not hits:
        return None
    norm = tuple(int(g) for g in gpus)
    pool = tuple(GPU_POOL_BY_CARDS.get(int(cards), ()))
    if norm == pool and not locked_card_hits(pool):
        # ★ 2026-10-16 收窄：豁免**只对自身合法的卡池**生效。旧码 `norm == pool` 单独成立即放过，
        #   使 ``[4,5,6,7]``（含两张生产锁卡 GPU6/7）长期"合法"通过 —— 这个洞已堵：
        #   池自身命中锁卡 ⇒ 继续往下走 raise（⛔ 不再登记了事）。
        POOL_CONTRACT_CONFLICTS[pool] = hits
        _uncaptured_pool_conflicts.append((tier_id, pool, hits))
        return None
    names = []
    for g in hits:
        if g in NEIGHBOR_CARDS:
            names.append(f"GPU{g}=邻居常驻卡（NEIGHBOR_CARDS）")
        elif g in GPU_FORBIDDEN_PRODUCTION:
            names.append(f"GPU{g}=生产锁卡（GPU_FORBIDDEN_PRODUCTION）")
        else:
            names.append(f"GPU{g}=越界卡（不在 GPU_ALLOWED={tuple(GPU_ALLOWED)} 内）")
    return (
        f"{tier_id}: gpus={list(gpus)} 含被锁的卡 ⇒ {'；'.join(names)}"
        f"（本世代卡池为 {list(pool)}）"
    )


#: ★★ **主席禁用集**（2026-10-16）：GPU0 = 他人占用中；GPU2 = 邻居常驻卡。
#:   ⛔ 与「当前空闲与否」无关 —— 它是**策略级**禁用清单，⛔ 不得因"此刻空着"而放宽。
CHAIRMAN_BANNED_CARDS: tuple[int, ...] = (0, 2)


def legal_card_set() -> tuple[int, ...]:
    """★ 合法卡集（**单一入口**）= ``GPU_ALLOWED`` − ``GPU_FORBIDDEN_PRODUCTION`` − 主席禁用集。

    2026-10-16 实测 = ``(1, 3, 4, 5)``。⛔ 生产锁卡 ``(6,7)`` 与邻居卡 ``(2,)`` 不在内。
    """
    return tuple(sorted(set(GPU_ALLOWED) - set(GPU_FORBIDDEN_PRODUCTION) - set(CHAIRMAN_BANNED_CARDS)))


def non_legal_card_violation(tier_id: str, gpus: "list[int] | tuple[int, ...]") -> "str | None":
    """★ 合法卡集断言（fail-closed）：``gpus`` 必须 ⊆ :func:`legal_card_set`。

    ★ 这条**专抓**旧「池等价豁免」放过的形态：旧 4 卡池 ``[4,5,6,7]`` 含生产锁卡 GPU6/7，
    过去因 `norm == pool` 被登记了事、**从未 raise**；本断言看的是**合法集**，与卡池无关。
    """
    bad = tuple(sorted({int(g) for g in gpus} - set(legal_card_set())))
    if not bad:
        return None
    return (f"{tier_id}: gpus={list(gpus)} 含非合法卡 {list(bad)}"
            f"（合法集 = {list(legal_card_set())}；禁 = 生产锁卡 {list(GPU_FORBIDDEN_PRODUCTION)} ∪ "
            f"邻卡 {list(NEIGHBOR_CARDS)} ∪ 主席禁用 {list(CHAIRMAN_BANNED_CARDS)}）")


def mini_tier_subset_size(tier_id: str, v1_subset: int | None = None) -> int:
    """★ mini 档（非 formal / 非 not_applicable）的 v2 子集条数 —— **唯一入口**（🔴B1a）。

    取数顺序（任一步不成立即 **fail-closed**，⛔ **绝不回落 v1 的 20**）：

    ① 该档在 :data:`MINI_DECLARED_SUBSET_SIZE`（= ``make_subsets.MINI_SUBSET_TIERS_DECLARED``）
       ⇒ 声明口径（18 档 = 100）；
    ② 与 :data:`MINI_LANDED_SUBSET_SIZE`（= ``make_subsets.MINI_TIERS``，在盘真值）对照：
       已落地档两者必须**相等**；豁免档（:data:`MINI_DECLARED_NOT_LANDED`）允许在盘值更小，
       但**必须在豁免表里有具名理由**；
    ③ 若仓内有该档冻结子集文件（``subsets-v2/<T>.jsonl``）⇒ 行数必须等于**在盘真值**；
    ④ 既不在声明表、也不在 v1 可核范围 ⇒ **raise**（陌生档 ⇒ 不猜）。
    """
    landed = MINI_LANDED_SUBSET_SIZE.get(tier_id)
    declared = MINI_DECLARED_SUBSET_SIZE.get(tier_id)
    if declared is not None:
        if tier_id in MINI_DECLARED_NOT_LANDED:
            if landed is None:
                raise AssertionError(f"{tier_id}: 声明口径豁免档却缺在盘真值表条目")
            if landed >= declared:
                raise AssertionError(
                    f"{tier_id}: 已登记为「声明未落地」但在盘真值 {landed} ≥ 声明 {declared}"
                    f" ⇒ 豁免表陈旧，应删出（fail-closed）"
                )
        else:
            if landed != declared:
                raise AssertionError(
                    f"{tier_id}: 声明口径 {declared} != 在盘落地真值 {landed}"
                    f"（make_subsets.MINI_SUBSET_TIERS_DECLARED vs MINI_TIERS）⇒ "
                    f"⛔ 拒绝生成（值回退 fail-closed；请同步两表或登记豁免）"
                )
        f = PROJECT_ROOT / REPO_SUBSET_DIR_REL / f"{tier_id}.jsonl"
        if f.is_file():
            lines = f.read_bytes().count(b"\n")
            if lines != landed:
                raise AssertionError(
                    f"{tier_id}: 冻结子集在盘 {lines} 行 != make_subsets.MINI_TIERS[{tier_id}]={landed}"
                    f" ⇒ ⛔ 拒绝生成（声明与产物不同值）"
                )
        # ★ 返回**声明口径**（= 清单 `data.subset_size` 声明的条数）：豁免档的冻结件虽然只有 20 行，
        #   但清单声明的就是 100（"声明未落地"这一事实由 `subset_file_pregenerated=False` ＋
        #   账本 `declared_not_landed` ＋ 交叉核对的具名豁免共同承担，⛔ 不是靠把声明值改小）。
        return int(declared)
    # 非声明面：以 v1 为下限（改前既有行为），但**必须**能由 v1 核出，⛔ 不用魔法默认值
    if v1_subset is None:
        raise AssertionError(f"{tier_id}: 非声明面档缺 v1 subset_size ⇒ 拒绝猜默认值")
    if landed is not None and int(landed) != int(v1_subset):
        raise AssertionError(
            f"{tier_id}: 非声明面档 v1 subset_size={v1_subset} != 在盘真值 {landed} ⇒ 拒绝生成"
        )
    return int(v1_subset)


def caliber_subset_ratio(tier_id: str, v1_subset: int, subset: int, is_na: bool) -> int:
    """★★ **口径扩张比率** —— 步数/ckpt 间隔随子集扩面同步的**唯一入口**（修正轮 2 / 🔴B-1）。

    病史（第二轮全局审查 B-1，本人机算确认）：修正轮 1 让本文件的**非 native 分支透传 v1**
    （`per_epoch = int(out["expected_optimizer_steps_per_epoch"])`），于是把在盘的
    `per_epoch/reachable` 由 **800/800 回退成 160/160**（原包 `merge_manifest_18.py` 的既定意图是
    `subset_size/per_epoch/reachable/checkpoint_save_steps` **四者一律 ×5**，其 docstring 与
    `SCALED_KEYS` 逐字可查），而 `docs/capability-matrix.data.json` 的 `per_epoch_steps` 仍是 800
    ⇒ 18 档"每 epoch 计划步数"在矩阵与 runner 权威清单里**两个值**（§1.4 破 ＋ ① 步数门被实质放宽）。

    比率取值（★ 与"在盘落地子集"解耦，⛔ 不混用）：
      · **声明口径档**（`MINI_DECLARED_SUBSET_SIZE`，18 档）⇒ `声明口径 // v1 声明口径`。
        ★ **含 3 档 `not_applicable` 豁免档**：它们的在盘 `subset_size` 仍是 v1 的 20，但**声明口径 = 100**
        （`declared_subset_size`），原包对 18 档一律 ×5，矩阵侧 `per_epoch_steps` 也是 100
        ⇒ 比率**必须取声明口径**（取落地值会得到 1，把这三档落回 20/10/5，与矩阵不符）。
      · 其余档 ⇒ `subset // v1_subset`（CPT/SFT 100→100 = 1）。
    非整数倍 ⇒ **raise**（⛔ 不静默取整、不静默豁免）。
    """
    if tier_id in MINI_DECLARED_SUBSET_SIZE:
        numer = int(MINI_DECLARED_SUBSET_SIZE[tier_id])
        basis = f"声明口径 {numer}"
    else:
        numer = int(v1_subset)
        basis = f"在盘子集 {numer}"
    v1s = int(v1_subset)
    if v1s <= 0:
        raise AssertionError(f"{tier_id}: v1 subset_size={v1s} 非法 ⇒ 拒绝算比率")
    if numer % v1s:
        raise AssertionError(
            f"{tier_id}: {basis} / v1 {v1s} 非整数倍 ⇒ ⛔ 拒绝按比率改步数（需人工裁定）"
        )
    return numer // v1s


def resolve_subset_check_dir(declared: str, repo: Path | None = None) -> "tuple[Path | None, str]":
    """解析「冻结子集交叉核对」的真实目录；返回 ``(目录或 None, 不能解析的具名原因)``。

    ★★ 病史（复核报告 B1b / §2.3）：``check()`` 原先直接 ``Path(manifest.host_dir).is_dir()``，
      而在盘 ``host_dir`` = **未展开字面量** ``${GRASPO_PREGEN_SUBSET_DIR}`` ⇒ 恒 False ⇒
      **每次都静默跳过**整套行数核对（7 条 MISMATCH 就是这样漏掉的）。
    ★ 现在**三级解析**（⛔ 不许静默跳过；解析不到 ⇒ 调用方必须按"未核对"处理并 fail-closed）：
      ① 清单里是**真实路径**且存在 ⇒ 用它（228 现场）；
      ② 仓内 ``subsets-v2/``（冻结件的**唯一真相源副本**）存在 ⇒ 用它（本机常态）；
      ③ 都不存在 ⇒ 返回 ``None`` ＋ 具名原因。
    """
    repo = repo or PROJECT_ROOT
    s = str(declared or "")
    if s and not s.startswith("${"):
        p = Path(s)
        if p.is_dir():
            return p, ""
    in_repo = repo / REPO_SUBSET_DIR_REL
    if in_repo.is_dir():
        return in_repo, ""
    return None, (f"清单 host_dir={declared!r}（未展开占位符/不可达）且仓内 {REPO_SUBSET_DIR_REL}/ 不存在")


def patch_tier(entry: dict, data_root: str, pregen_dir: str) -> dict:
    """把 v1 的单档条目改写成 v2 条目（就地改写一份深拷贝）。"""
    tier_id = str(entry["tier_id"])
    out = copy.deepcopy(entry)
    # ★ 卡池确定性重映射（2026-10-05）：v1 清单的 gpus 是 0–3 口径、本生成器原样透传
    #   ⇒ 不加此步，任何一次重生成都会把运行期清单的卡池回退成 0–3。
    _cards = int(out.get("cards") or 0)
    if _cards not in GPU_POOL_BY_CARDS:
        raise AssertionError(f"{tier_id}: cards={_cards} 不在 1–4，卡池无法确定性映射")
    if tier_id in TIER_GPU_OVERRIDE:
        # ★ 显式覆盖（资源调度例外）也 fail-closed：卡数错配 / 重复卡一律拒绝，⛔ 不将就。
        _override = [int(g) for g in TIER_GPU_OVERRIDE[tier_id]]
        if len(_override) != _cards:
            raise AssertionError(
                f"{tier_id}: 覆盖卡位 {_override} 卡数 {len(_override)} ≠ cards={_cards}"
            )
        if len(set(_override)) != len(_override):
            raise AssertionError(f"{tier_id}: 覆盖卡位 {_override} 含重复卡")
        # ★★ 通用断言：**显式卡位例外必须落在「合法卡集」内**。
        #    这就是「陈旧调度值」这一类缺陷的根因判据：例外描述的是**当时的空闲状态**，
        #    一旦锁卡策略换代，旧例外就指向了不该占的卡（T041 `[2,3]` 即此形态）⇒ 必须显式重建。
        #    ★ 2026-10-16：判据域从 `GPU_POOL_BY_CARDS[cards]`（**默认值**）改为
        #      `legal_card_set()`（= GPU_ALLOWED − 生产锁卡 − 主席禁用 {0,2}）—— 因为车道重出后
        #      同一卡数会有**多条车道**（如 2 卡档分属 `[3,5]` 与 `[4,5]`），默认值不再是全体上界。
        #      ★ 保护强度**只增不减**：旧域漏掉的 `[4,5,6,7]`（含生产锁卡）在此被拒。
        _pool = set(legal_card_set())
        _outside = sorted(set(_override) - _pool)
        if _outside:
            raise AssertionError(
                f"{tier_id}: 覆盖卡位 {_override} 含合法卡集外的卡 {_outside}"
                f"（合法集 legal_card_set()={sorted(_pool)}）⇒ 拒绝（陈旧例外须显式重建）"
            )
        out["gpus"] = _override
    else:
        out["gpus"] = list(GPU_POOL_BY_CARDS[_cards])
    # ★★ 通用锁卡断言（2026-10-16 新增）：**全部 54 档**一律过 —— 不得含邻居卡 / 生产锁卡 / 越界卡。
    #    这就是当初能抓到「T041 声明了邻居 GPU2」的那条断言；⛔ 撤销例外时不得把它一起删掉。
    #    ★ 卡池自身的已知冲突（如 {4,5,6,7} ∩ {6,7}）走具名登记分支（不 raise），见 locked_card_violation。
    _lv = locked_card_violation(tier_id, out["gpus"], _cards)
    if _lv:
        raise AssertionError(f"锁卡断言失败 ⇒ {_lv}")
    if len(out["gpus"]) != _cards:
        raise AssertionError(
            f"{tier_id}: gpus={out['gpus']} 个数 {len(out['gpus'])} ≠ cards={_cards}（§12 同源）"
        )
    is_na = tier_id in NOT_APPLICABLE_TIERS

    # ① config 口径
    if out.get("config"):
        if Path(out["config"]).parent.as_posix() != V1_CONFIG_DIR_REL:
            raise AssertionError(f"{tier_id}: v1 config 目录异常：{out['config']}")
        out["config"] = f"{CONFIG_DIR_REL}/{tier_id}.yaml"

    # ② 子集条数（★ 修正轮 1：18 档扩面**由唯一真相源现算**，⛔ 不再回落 v1 的 20）
    v1_subset = int(out["data"]["subset_size"])
    if is_na:
        subset = v1_subset  # not_applicable：不跑，保持 v1（不假装有 v2 子集）
        source = None
    elif tier_id in V2_SUBSET_SIZE_BY_TIER:
        subset = V2_SUBSET_SIZE_BY_TIER[tier_id]
        source = SOURCE_FORMAL
    else:
        subset = mini_tier_subset_size(tier_id, v1_subset)
        source = SOURCE_MINI
    # ★★ 口径扩张比率（修正轮 2 / 🔴B-1）：**声明口径档一律适用（含 3 档 `not_applicable` 豁免档）**。
    #   · formal 档（`V2_SUBSET_SIZE_BY_TIER`，9 档 native）的步数走真实式（`v2_native_steps_per_epoch`），
    #     ckpt 间隔由 v2 档配置决定 ⇒ **不按比率**；
    #   · 其余档（含 18 档声明口径档）⇒ 走 :func:`caliber_subset_ratio`（唯一入口）。
    #   ⛔ 旧实现只在 `not is_na` 时算比率 ⇒ 3 档豁免档漏掉，且非 native 分支根本没用它（B-1 根因）。
    caliber_ratio = 1
    if tier_id not in V2_SUBSET_SIZE_BY_TIER:
        caliber_ratio = caliber_subset_ratio(tier_id, v1_subset, subset, is_na)

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
        # ★★ 修正轮 1（🔴B1b 的 3 档具名豁免）：`not_applicable` 档的 `subset_size` **保持 v1 的 20**
        #    （⛔ 不假装有 v2 冻结件、⛔ 不为它伪造数据），但"**声明口径 = 100 未落地**"这一事实
        #    **机读登记**在本字段（唯一真相源 = `make_subsets.MINI_SUBSET_TIERS_DECLARED`），
        #    与 `subsets-v2.manifest.json::tiers[].declared_not_landed` 同源同义。
        data["declared_subset_size"] = int(MINI_DECLARED_SUBSET_SIZE[tier_id])
        data["declared_not_landed"] = {
            "declared": int(MINI_DECLARED_SUBSET_SIZE[tier_id]),
            "landed": int(v1_subset),
            "reason": MINI_DECLARED_NOT_LANDED[tier_id],
            "authority": "make_subsets.MINI_SUBSET_TIERS_DECLARED vs MINI_TIERS 的差额",
        }
    else:
        data.pop("declared_subset_size", None)
        data.pop("declared_not_landed", None)
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
            + (
                f"【★ 锚定覆盖为 {PLANNED_STEPS_OVERRIDE[tier_id]}；未锚定公式值为 {per_epoch}】"
                if tier_id in PLANNED_STEPS_OVERRIDE else ""
            )
        )
    else:
        # ★★★ 修正轮 2（🔴B-1 根因）：**必须乘口径比率**。
        #     旧实现 `per_epoch = int(out["expected_optimizer_steps_per_epoch"])` 是**纯透传 v1**
        #     ⇒ 18 档声明口径扩面（20→100）后 `per_epoch` 停在 20 行口径的 160/80/40/20/10/5，
        #     而 `docs` 矩阵的 `per_epoch_steps` = 800/400/200/100/50/25 ⇒ §1.4 双真相源 ＋
        #     `scripts/collect_results.py:1846-1858,2348` 的 ① 步数门被实质放宽 5 倍。
        _v1_per = int(out["expected_optimizer_steps_per_epoch"])
        per_epoch = _v1_per * int(caliber_ratio)
        out["step_formula_v2"] = (
            "非 native GRASPO：v1 口径 × **口径扩张比率**"
            f"（v1={_v1_per} × {caliber_ratio} = {per_epoch}）"
            + (
                "；★ 本档属「声明口径扩面」⇒ 比率 = 声明口径 ÷ v1 声明口径"
                if caliber_ratio != 1 else "；比率 = 1（子集与 v1 同口径）"
            )
        )
    # ★★★ 2026-10-16 修复（🔴「reachable 低估 2–8 倍」）：epoch 数改从**跑次实际使用的
    #     档配置 yaml** 现算（`run_config_epochs`，单一真相源），⛔ 不再用 `gm.MATRIX_MAX_EPOCHS`。
    #     病史：v1 冻结期常量 = 1，而 2026-10-15 主席把 epoch 口径改为 SFT=4／CPT/OPD=2／
    #     GRASPO=8 ⇒ 矩阵与 yaml 都跟上了、**只有本生成器没跟** ⇒ 全 54 档
    #     `expected_optimizer_steps_reachable` = per_epoch × 1（T010：100 vs 真值 400）。
    #     后果（`result_judge.resolve_step_gate_applicability`）：低估使"结构上够不着门槛"
    #     更易成立 ⇒ A2 门槛退出闸门、改按每 epoch 步数判 ⇒ 是**软化**方向。当前 54 档
    #     因最低值 6 ≥ 门槛 5 而未触发（实测 0/54 判定变化），属**潜伏**缺陷。
    epochs = run_config_epochs(tier_id)
    assert epochs is not None or tier_id in NOT_APPLICABLE_NO_CONFIG, (
        f"{tier_id}: 读不到跑次档配置的 max_epochs，且该档不在无 yaml 具名豁免 "
        f"{list(NOT_APPLICABLE_NO_CONFIG)} 内 ⇒ fail-closed（⛔ 不猜、⛔ 不退回 MATRIX_MAX_EPOCHS）"
    )
    if epochs is not None:
        assert epochs >= 1, f"{tier_id}: yaml max_epochs={epochs} 非正整数"
        reachable = per_epoch * int(epochs)
    else:
        # 无 yaml 档：**原样保留 v1 口径**（= per_epoch × 1），具名豁免、机检反向断言兜住。
        reachable = per_epoch * int(gm.MATRIX_MAX_EPOCHS)
    # ★ 锚定覆盖（2026-10-02 批准）：仅这三档
    if tier_id in PLANNED_STEPS_OVERRIDE:
        anchored = int(PLANNED_STEPS_OVERRIDE[tier_id])
        assert anchored >= MIN_OPTIMIZER_STEPS, (
            f"{tier_id}: 锚定步数 {anchored} < 门槛 {MIN_OPTIMIZER_STEPS}"
        )
        per_epoch = anchored
        reachable = anchored * int(epochs) if epochs is not None \
            else anchored * int(gm.MATRIX_MAX_EPOCHS)

    out["expected_optimizer_steps_per_epoch"] = per_epoch
    # ★ 键名语义 ≠ 实际语义时必须就地加注（2026-10-02 通则；本块随清单同行、机可读）
    out["key_semantics"] = {
        "expected_optimizer_steps_per_epoch": (
            "每 epoch 的**计划优化步数**（① 的比较基准）。本轮 T040/T041/T042 = 6（锚定覆盖，"
            "不随子集放大而变；未锚定的公式值为 9，见 step_formula_v2 的【★ 锚定覆盖…】尾注）。"
        ),
        "expected_optimizer_steps_reachable": (
            "整个 run 的可达步数上限 = `expected_optimizer_steps_per_epoch` × "
            "**该档跑次档配置 yaml 的 `training.max_epochs`**（= 该键的唯一真相源，"
            "见 `step_gate_basis` 与本档 `config` 所指文件；⛔ 不再用 v1 冻结期常量）。"
            "当前值见本键本身（⛔ 此处不写死数字：写死的例子会随 epoch 口径变更而变成假话）。"
        ),
        "checkpoint_save_steps": (
            "**ckpt 保存间隔**（= 该档**跑次档配置 yaml** 的 `training.save_steps`，"
            "逐档从该 yaml 现读、⛔ 不再从已退役的 " + V2_CONFIG_DIR_REL + "/ 读）。"
            "⛔ **不是**计划优化步数 —— 计划优化步数见 `expected_optimizer_steps_per_epoch`。"
            "（不变量：save_steps 必须 > 可达步数，否则会落中间 step_<N> ckpt、破坏「每档只留一份」。）"
        ),
        "expected_optimizer_steps": (
            "**历史键名**：语义 = `checkpoint_save_steps`（保存间隔），⛔ **不是**计划优化步数。"
            "保留旧名仅为不破坏既有消费方；新代码请读 `expected_optimizer_steps_per_epoch`。"
        ),
        "data.subset_size": (
            "训练子集**条数**（= 每卡 prompt 数 × 卡数；本轮三档 72/144/288 ⇒ P 恒 72）。"
            "⛔ 不是步数。"
        ),
        "acceptance.formal_gate.min_optimizer_steps": (
            "A2 **正式门门槛**（= 5，本轮未动）。与①（= 6）是两个数、都保留：A2 = 正式门，"
            "① = 「实际步数 ≥ 计划步数」的完整性指标（更严）。"
        ),
    }
    out["expected_optimizer_steps_reachable"] = reachable
    # ★★★ 2026-10-16 修复（🔴「css 与跑次 yaml 矛盾」）：**全 54 档一律**从**跑次实际使用**的
    #     档配置 yaml 现读 `save_steps`（⛔ 不再分 native/非 native 两条路，也⛔ 不再从**已退役**的
    #     V2_CONFIG_DIR_REL 读）。病史：native 档从 matrix54-v2 读（T028：退役 48 vs 现行 49），
    #     其余档走 `v1 值 × caliber_ratio`（T010：100 vs yaml 401）⇒ 实测 **51/51 有 yaml 的档
    #     全部与跑次 yaml 不一致**，而该键的 `key_semantics` 逐字声明它就是"该档 yaml 的
    #     training.save_steps" ⇒ 清单**自相矛盾**。改由 yaml 现读后，该声明**由构造成立**，
    #     且与矩阵 `save_steps`（= derived_steps + 1）自动三向一致（实测 51/51）。
    _css = run_config_scalar(tier_id, "save_steps")
    if _css is not None:
        out["checkpoint_save_steps"] = int(_css)
    else:
        # 无 yaml 档（具名豁免 NOT_APPLICABLE_NO_CONFIG）：原样保留 v1 口径 × 口径比率。
        # ⛔ 不发明数值；机检对其具名豁免 ＋ 反向断言。
        assert tier_id in NOT_APPLICABLE_NO_CONFIG, (
            f"{tier_id}: 读不到跑次档配置的 save_steps，且不在无 yaml 具名豁免 "
            f"{list(NOT_APPLICABLE_NO_CONFIG)} 内 ⇒ fail-closed"
        )
        out["checkpoint_save_steps"] = int(out["checkpoint_save_steps"]) * int(caliber_ratio)
    # ★★ 清单自带 `key_semantics.expected_optimizer_steps` 逐字声明：该键语义 = `checkpoint_save_steps`
    #    ⇒ 本不变量**对全 54 档无条件成立**（HEAD 清单实测 54/54 相等；本包手工并入曾只改一半
    #    ⇒ 18 档违反）。⛔ 不再把它限制在 native 分支里。
    out["expected_optimizer_steps"] = out["checkpoint_save_steps"]

    # ⑤ formal_gate：门槛值与 v1 一致（不放宽），只重算标记与 basis
    gate = dict(out["acceptance"]["formal_gate"])
    gate["min_optimizer_steps"] = MIN_OPTIMIZER_STEPS
    # ★ 2026-10-16：`min_epochs` 与 v1 冻结常量**解耦**（值不变、仍是 1 ⇒ 行为逐字不变）。
    #   它语义是「**最少**要跑完几个 epoch」（门槛的下限），⛔ **不是**该档的计划 epoch 数 ——
    #   后者是 `training.max_epochs`（= 4/2/8，见 `run_config_epochs`）。原先误用
    #   `gm.MATRIX_MAX_EPOCHS`（=1）当来源，既掩盖了"它只是下限"，又让 epoch 出现第二个来源。
    gate["min_epochs"] = MIN_EPOCHS_GATE
    gate["step_gate_applicability"] = (
        gm.STEP_GATE_APPLICABLE
        if reachable >= MIN_OPTIMIZER_STEPS
        else gm.STEP_GATE_NOT_APPLICABLE_STRUCTURAL
    )
    gate["step_gate_basis"] = (
        f"expected_optimizer_steps_reachable={reachable}；per_epoch={per_epoch}；"
        f"max_epochs={epochs if epochs is not None else '—（无档配置 yaml，具名豁免）'}；"
        + ("【★ 锚定：本档计划步数按 2026-10-02 批准的覆盖值固定，不随子集放大而变】"
           if tier_id in PLANNED_STEPS_OVERRIDE else "")
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
                    "max_epochs": run_config_epochs(entry["tier_id"]),
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
    v1_manifest_sha: str, config_fingerprint: str, config_dir_fingerprint: str = "",
) -> dict:
    v1_tiers = v1_manifest["tiers"]
    POOL_CONTRACT_CONFLICTS.clear()
    _uncaptured_pool_conflicts.clear()
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
        "generator": "tests/e2e/generate_matrix_manifest_source.py",
        "config_v2_dir": V2_CONFIG_DIR_REL,
        "config_dir": CONFIG_DIR_REL,
        "config_dir_fingerprint": config_dir_fingerprint,
        "config_dir_note": (
            "★ 跑次实际读取的档配置目录 = " + CONFIG_DIR_REL + "（矩阵生成件；由 tools/matrix/gen_configs.py "
            "从 docs/capability-matrix.data.json 生成，与矩阵声明逐档一致）。"
            "★ " + V2_CONFIG_DIR_REL + " 仅作 **native ``dp_size`` 口径来源**，⛔ 不再是跑次配置来源。"
            "★ 2026-10-16：``save_steps`` / epoch 数（``training.max_epochs``）**只读本目录**"
            "（= 跑次实际读的 yaml，单一真相源）；改指前 epoch 数走 v1 常量 ⇒ ``reachable`` 低估 2–8 倍。"
        ),
        "subset_dir_container": f"/work/elam-v5/{V2_SUBSET_DIRNAME}",
        "subset_dir_host": pregen_dir,
        "host_data_root": data_root,
        "attempt_root": attempt_root,
    }
    out["changes_vs_v1"] = {
        "config": f"{V1_CONFIG_DIR_REL}/T0NN.yaml → {CONFIG_DIR_REL}/T0NN.yaml（51 个可跑档）"
                  "｜★ 2026-10-16：步数/保存口径**同指**本目录（yaml 的 training.max_epochs / "
                  f"save_steps 即唯一真相源）；{V2_CONFIG_DIR_REL}/ 现只供 native dp_size",
        "subset_size": (
            "① 9 档 GRASPO×native：20 → 48×卡数（48/96/192）；"
            "② ★ **2026-10-16 主席裁定「18 个 20 行档放大到 100 行」**：原 18 个 20 行档的"
            "**声明口径 = 100**（唯一真相源 = `tests/e2e/make_subsets.py::MINI_SUBSET_TIERS_DECLARED`），"
            "其中 **15 档已在盘落地**（`MINI_TIERS` 同步为 100、冻结件实做 ×100），"
            "**3 档具名豁免未落地**（`T052–T054`：`not_applicable` ⇒ 无生成件、⛔ 不伪造；见 "
            "`make_subsets.DECLARED_NOT_LANDED` 与 `subsets-v2.manifest.json::tiers[].declared_not_landed`）；"
            "③ 其余 CPT/SFT 27 档 = 100（不变）"
        ),
        "train_path": f"/work/elam-v5/subsets/ → /work/elam-v5/{V2_SUBSET_DIRNAME}/（51 个可跑档）",
        "steps": (
            "expected_optimizer_steps_per_epoch/_reachable 按**真实步进语义**重算："
            "native GRASPO = ceil(floor(subset/dp_size)/Q)；"
            "★ 2026-10-16：`_reachable` = per_epoch × **该档 yaml 的 training.max_epochs**"
            "（改前误用 v1 常量 1 ⇒ 低估 2–8 倍）；`checkpoint_save_steps` = 该 yaml 的 "
            "training.save_steps（改前 native 读已退役目录、其余走 v1 透传 ⇒ 51/51 与跑次 yaml 不一致）"
        ),
        "new_fields": [
            "config_version", "supersedes", "changes_vs_v1", "data_sources", "checkpoint_retention_v2",
            "per-tier data.source / data.source_jsonl / data.source_jsonl_container / "
            "data.extract_command / data.source_rule / data.subset_file_pregenerated / step_formula_v2",
        ],
        "gpus": (
            "★ 2026-10-16：**撤销 T041 的陈旧卡位例外** `[2,3]`（2026-10-09 的一次性调度值，"
            "把第三方账号占用中的常驻邻居卡 GPU2 写进了我方声明）⇒ T041 走默认映射 "
            "`GPU_POOL_BY_CARDS[2] = [4,5]`。★ 其余 53 档 `gpus` **逐档逐字未变**（仍 = 本世代卡池）。"
        ),
        "unchanged": (
            "counts / models / feasibility_model / runtime.checkpoint_retention / 每档的 "
            "model*/algorithm/mode/backend/cards/status/feasibility/acceptance.criteria"
            "（★ `gpus` 见上条：仅 T041 一档改）"
        ),
    }
    # ★★ 卡池/锁卡契约的**已知冲突登记**（只登记、⛔ 不擅自改池；由指挥统一处置）：
    #    机读、可审计 —— 后人不必靠读注释去发现「本世代 4 卡池 {4,5,6,7} 含生产锁卡 6/7」。
    out["gpu_pool_contract_conflicts"] = {
        "locked_cards": list(LOCKED_CARDS),
        "gpu_allowed": list(GPU_ALLOWED),
        "gpu_forbidden_production": list(GPU_FORBIDDEN_PRODUCTION),
        "neighbor_cards": list(NEIGHBOR_CARDS),
        "assertion": (
            "locked_card_violation()：任何档 gpus 不得含 locked_cards 中任一张；"
            "唯一例外 = 该值恰等于本世代卡池 GPU_POOL_BY_CARDS[cards]（登记于此，不 raise）"
        ),
        "conflicts": [
            {
                "pool": list(pool),
                "locked_hits": list(hits),
                "tiers": sorted(t for t, p, _h in _uncaptured_pool_conflicts if p == pool),
            }
            for pool, hits in sorted(POOL_CONTRACT_CONFLICTS.items())
        ],
        "note": (
            "★ 上列冲突是**卡池定义**（2026-10-05 主席令「八卡全开」定的 {4,5,6,7}）与"
            "`gpu_assignment.GPU_FORBIDDEN_PRODUCTION=(6,7)` 之间的矛盾，**不是本生成器的调度选择**"
            "⇒ 本生成器只登记、⛔ 不擅自改池。邻居卡 GPU2 已随 T041 例外撤销而从全部声明中清零。"
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
            "generator": "tests/e2e/make_subsets.py",
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
            "launcher": "tests/e2e/run_matrix54_source.sh",
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


def derived_state_regression(new_doc, out_path) -> list[str]:
    """★★ **派生状态**守卫（修正轮 2 / 🟡W-1b）：`--out` 覆盖**已存在的跑次台账**时，
    重生成不得改写任何 tier 的 `status`。

    病史（第二轮全局审查 W-1）：在盘清单的 `status` 是**跑次派生态**（`ready`/`blocked`/`running`/
    `not_applicable`，由 `backfill_matrix.py` 一族回填），而重生成从 v1 深拷贝 ⇒ 会把 **45 档**
    改写成静态 `judged`（实测 `ready→judged` 21、`unverified→judged` 8、`unverified→blocked` 5…）。
    ⇒ "声明的写路径"会**静默翻转派生状态**（§1.4/§2.4）。`caliber_value_regression()` 只管口径字段，
    覆盖不到这一面，故单列本守卫。

    语义（fail-closed，⛔ 不是"警告"）：
      · `out_path` **不存在** ⇒ 首次生成，没有可比对象 ⇒ 通过（⛔ 不阻塞正常通路）；
      · 存在 ⇒ 逐档比 `status`（在盘 vs 新内容），**任一不同即返回具名条目**，调用方**拒绝写盘**。
    """
    out_path = Path(out_path)
    if not out_path.is_file():
        return []
    on_disk = json.loads(out_path.read_text(encoding="utf-8"))
    od = {str(t.get("tier_id")): t for t in on_disk.get("tiers", [])}
    drift: list[str] = []
    for t in new_doc.get("tiers", []):
        tid = str(t.get("tier_id"))
        o = od.get(tid)
        if o is None:
            continue
        if "status" in t and "status" in o and str(t["status"]) != str(o["status"]):
            drift.append(f"{tid}.status: 在盘 {o['status']!r} → 重生成 {t['status']!r}")
    return drift


def caliber_value_regression(new_doc, out_path) -> list[str]:
    """★★ **值回退**守卫（修正轮 1 / 🔴B1a 的 fail-closed 面）：在盘 vs 新内容的**口径值**必须一致。

    病史（复核报告 B1a）：`guard_regeneration_keeps_on_disk_fields()` 只防**字段路径丢失**
    （实测 `only_disk=0` 却仍把 18 档 `subset_size` 100 写回 20）⇒ 值回退**静默**发生且
    `check()` 照样打印 `check OK`。本函数把"值"也纳入 fail-closed。

    比对面 = **口径关键字段**（⛔ 不是全部字段：在盘清单是跑次台账，含生成器之外的 post-fill
    字段与派生 status，逐字段全比会误伤 —— 见工位报告 §B1a 的实测归因）：
      · `data.subset_size` / `data.source` / `data.extract_command` / `data.source_rule`
      · `expected_optimizer_steps` / `..._per_epoch` / `..._reachable` / `checkpoint_save_steps`
    判据：**在盘值 == 新内容值**。不等 ⇒ 返回具名条目（调用方**拒绝写盘**）。
    ★ 首次生成（在盘不存在）没有可比对象 ⇒ 直接通过。
    """
    out_path = Path(out_path)
    if not out_path.is_file():
        return []
    on_disk = json.loads(out_path.read_text(encoding="utf-8"))
    od = {str(e["tier_id"]): e for e in on_disk.get("tiers", [])}
    drift: list[str] = []
    for e in new_doc.get("tiers", []):
        tid = str(e["tier_id"])
        o = od.get(tid)
        if o is None:
            continue
        for k in ("expected_optimizer_steps", "expected_optimizer_steps_per_epoch",
                  "expected_optimizer_steps_reachable", "checkpoint_save_steps"):
            if k in e and k in o and str(e[k]) != str(o[k]):
                drift.append(f"{tid}.{k}: 在盘 {o[k]} → 重生成 {e[k]}")
        for k in ("subset_size", "source", "extract_command", "source_rule"):
            if k in e.get("data", {}) and k in o.get("data", {}) \
                    and str(e["data"][k]) != str(o["data"][k]):
                drift.append(f"{tid}.data.{k}: 在盘 {str(o['data'][k])[:50]} → 重生成 {str(e['data'][k])[:50]}")
    return drift


def cross_check_manifest_against_matrix(
    manifest_path: "Path | str | None" = None,
    matrix_path: "Path | str | None" = None,
    exempt_drift: "tuple[str, ...] | None" = None,
    exempt_no_config: "tuple[str, ...] | None" = None,
) -> list[str]:
    """★★ **跨件不变量机检**（修正轮 2 / 🔴B-1 ＋ 2026-10-16 **扩为三向一致**）：全 54 档

        I-a  `manifest.expected_optimizer_steps_per_epoch == docs.rows[].per_epoch_steps`
        I-b  `manifest.expected_optimizer_steps_reachable   == docs.rows[].derived_steps`
        I-c  `manifest.checkpoint_save_steps == docs.rows[].save_steps == <该档 yaml>.training.save_steps`
        I-d  `manifest.expected_optimizer_steps == manifest.checkpoint_save_steps`（同清单内）
        I-e  `manifest.expected_optimizer_steps_reachable >= MIN_OPTIMIZER_STEPS`（**重投闸门**：
             修正前后都不得有档跨过 `reachable < 门槛` 这条线 ⇒ "先跑后收集"才成立）

    为什么必须跨件（⛔ 不能只比"清单 ↔ 重生成"）：B-1 之所以漏网，正是因为
    `caliber_value_regression()` 只比**清单 ↔ 重生成**（两者同为 160，自洽地错），
    而**真值在 `docs/capability-matrix.data.json`**（`per_epoch_steps` = 800）。
    ⇒ 这条机检把"矩阵 ↔ runner 权威清单"这一对**唯一真相源**钉在一起（§1.4）。

    ★★ **为什么 2026-10-16 要扩到 I-b/I-c**：原实现只读矩阵行的 `per_epoch_steps` **一个字段**
      ⇒ 凡需 `max_epochs` 才能算出的量**都不在它的定义域内**，而 `max_epochs` **只在矩阵行**
      （清单里没有这个字段）⇒ `expected_optimizer_steps_reachable`（低估 2–8 倍）与
      `checkpoint_save_steps`（与跑次 yaml 51/51 不一致）**结构上看不到**。矩阵行**已有**
      `max_epochs` / `derived_steps` / `save_steps` ⇒ 三向对拍本来就可做，只是没写。
    ★ I-c 的第三向 = **档位 yaml**：它是**跑次实际读**的那一份（`tiers[*].config`）；只比
      "清单 ↔ 矩阵"会漏掉"矩阵对、yaml 错"的情形。

    ★ **豁免不是静默白名单**（两处，均带反向断言）：
      · `exempt_drift`（默认 :data:`KNOWN_PRE_EXISTING_PER_EPOCH_DRIFT` = W-7 的 T017/T018，
        ⛔ 本轮不修）里的档 **必须确实仍在漂**；不漂了 ⇒ 报「豁免表陈旧」。
      · `exempt_no_config`（默认 :data:`NOT_APPLICABLE_NO_CONFIG` = T052–T054）里的档
        **必须确实没有档配置 yaml**；有了 ⇒ 报「豁免表陈旧」（逼下一轮改走 yaml 现算）。
    """
    mpath = Path(manifest_path) if manifest_path is not None else V2_MANIFEST_PATH
    dpath = Path(matrix_path) if matrix_path is not None else (
        PROJECT_ROOT / "docs" / "capability-matrix.data.json"
    )
    exempt = KNOWN_PRE_EXISTING_PER_EPOCH_DRIFT if exempt_drift is None else tuple(exempt_drift)
    exempt_nc = (NOT_APPLICABLE_NO_CONFIG if exempt_no_config is None else tuple(exempt_no_config))
    if not mpath.is_file():
        return [f"清单不在盘：{mpath}"]
    if not dpath.is_file():
        return [f"矩阵正式件不在盘：{dpath}（⛔ 跨件不变量无法核对 ⇒ fail-closed）"]
    man = json.loads(mpath.read_text(encoding="utf-8"))
    rows = {str(r.get("tier")): r for r in json.loads(dpath.read_text(encoding="utf-8"))["rows"]}
    problems: list[str] = []
    drifted: set[str] = set()
    no_config_seen: set[str] = set()
    for e in man.get("tiers", []):
        tid = str(e["tier_id"])
        r = rows.get(tid)
        if r is None:
            problems.append(f"{tid}: 矩阵正式件里没有该档 ⇒ 无法跨件核对")
            continue
        # ── I-a：per_epoch（B-1 原有）────────────────────────────────────────
        want = r.get("per_epoch_steps")
        if want is None:
            problems.append(f"{tid}: 矩阵行缺 per_epoch_steps ⇒ 无法跨件核对")
            continue
        got = int(e["expected_optimizer_steps_per_epoch"])
        if got != int(want):
            drifted.add(tid)
            if tid not in exempt:
                problems.append(
                    f"{tid}: manifest.per_epoch={got} != matrix.per_epoch_steps={want}"
                    f"（§1.4 单一真相源破 ⇒ ① 步数门基准被改）"
                )
        # ── I-b：reachable == matrix.derived_steps（2026-10-16 新增）──────────
        der = r.get("derived_steps")
        reach = int(e["expected_optimizer_steps_reachable"])
        if run_config_epochs(tid) is None:
            # ★ 无档配置 yaml 档（not_applicable，⛔ 无配方 ⇒ 无跑次）：**本不变量的定义域之外**
            #   —— I-b/I-c 讲的是"清单声明 == 跑次实际"，而它们**没有跑次配置**可比。
            #   ⇒ 具名豁免（`exempt_nc`，集合必须**恰好**等于实测无 yaml 的档），
            #     且**仍**要求它自洽：`applicable` 必须为假 ＋ `reachable == per_epoch × epochs`
            #     （生成器对它们原样保留 v1 口径，epochs 走具名常量） ＋ `eos == css`。
            if bool(e.get("applicable")):
                problems.append(
                    f"{tid}: 无档配置 yaml 却标 `applicable=True` ⇒ 豁免前提破（fail-closed）"
                )
            if reach != got * int(gm.MATRIX_MAX_EPOCHS):
                problems.append(
                    f"{tid}: 无 yaml 豁免档自洽性破：reachable={reach} != "
                    f"per_epoch({got}) × {int(gm.MATRIX_MAX_EPOCHS)}（v1 冻结口径）"
                )
        elif der is None:
            problems.append(f"{tid}: 矩阵行缺 derived_steps ⇒ I-b 无法核对（fail-closed）")
        else:
            if reach != int(der):
                if tid in exempt:
                    # W-7 档：manifest.per_epoch 本身在漂 ⇒ reachable 必然跟着漂；
                    # 仍要求它严格等于 manifest.per_epoch × yaml.max_epochs（自洽即可）。
                    ep = r.get("max_epochs")
                    if ep is None or reach != got * int(ep):
                        problems.append(
                            f"{tid}: 豁免档 I-b 自洽性破：reachable={reach} != "
                            f"per_epoch({got}) × matrix.max_epochs({ep})"
                        )
                else:
                    problems.append(
                        f"{tid}: manifest.reachable={reach} != matrix.derived_steps={der}"
                        f"（epoch 口径分叉 ⇒ A2「结构可达性」前置判定输入被低估/高估）"
                    )
        # ── I-c：css == matrix.save_steps == yaml.training.save_steps ─────────
        mss = r.get("save_steps")
        css = int(e["checkpoint_save_steps"])
        yaml_cfg = run_config_scalar(tid, "save_steps")
        if yaml_cfg is None:
            no_config_seen.add(tid)
            if tid not in exempt_nc:
                problems.append(
                    f"{tid}: 该档无档配置 yaml（读不到 training.save_steps）且不在无 yaml 具名豁免 "
                    f"{list(exempt_nc)} 内 ⇒ I-c 无法核对（fail-closed）"
                )
        elif mss is None:
            problems.append(f"{tid}: 矩阵行缺 save_steps ⇒ I-c 无法核对（fail-closed）")
        else:
            if css != int(yaml_cfg):
                problems.append(
                    f"{tid}: manifest.checkpoint_save_steps={css} != "
                    f"{CONFIG_DIR_REL}/{tid}.yaml 的 training.save_steps={yaml_cfg}"
                    f"（清单声明 ≠ 跑次实际）"
                )
            if int(mss) != int(yaml_cfg):
                problems.append(
                    f"{tid}: matrix.save_steps={mss} != "
                    f"{CONFIG_DIR_REL}/{tid}.yaml 的 training.save_steps={yaml_cfg}"
                    f"（矩阵生成件与跑次 yaml 分叉 ⇒ 需重生成 yaml）"
                )
        # ── I-d：清单内不变量 eos == css ─────────────────────────────────────
        if "expected_optimizer_steps" in e and int(e["expected_optimizer_steps"]) != css:
            problems.append(
                f"{tid}: manifest.expected_optimizer_steps={e['expected_optimizer_steps']} != "
                f"checkpoint_save_steps={css}（同清单内不变量破）"
            )
        # ── I-e：重投闸门（修正前后都不得有档跨过 reachable < 门槛）────────────
        #   ★ 只对**可跑档**（有 yaml）成立：无 yaml 档 not_applicable ⇒ 从不产生 RunEvidence
        #     ⇒ 那个"第三态"分支对它们永远不会执行（其 v1 口径值也不参与任何判定）。
        reach_v = int(e["expected_optimizer_steps_reachable"])
        if run_config_epochs(tid) is not None and reach_v < MIN_OPTIMIZER_STEPS:
            problems.append(
                f"{tid}: reachable={reach_v} < 门槛 {MIN_OPTIMIZER_STEPS} ⇒ **该档会触发 A2 第三态**"
                f"（门槛退出闸门、改按每 epoch 步数判）⇒ 「先跑、事后用修正清单收集」**不成立**，"
                f"必须重跑/重判。修 epoch 口径前请先处理本档。"
            )
    # ★ 反向断言 ①：具名豁免必须仍然"确实在漂"，否则豁免表陈旧（⛔ 不许留静默白名单）
    stale = sorted(set(exempt) - drifted)
    if stale:
        problems.append(
            f"豁免表陈旧：{stale} 已不再漂移 ⇒ 必须从 KNOWN_PRE_EXISTING_PER_EPOCH_DRIFT 删出"
            f"（本表只允许承载**确实仍在漂**的既有档）"
        )
    # ★ 反向断言 ②：无 yaml 豁免表必须**恰好**等于"矩阵里无档配置 yaml"的档集合
    if no_config_seen != set(exempt_nc):
        problems.append(
            f"无 yaml 豁免表陈旧：实测无 yaml 的档={sorted(no_config_seen)}，"
            f"具名豁免表={sorted(exempt_nc)} ⇒ 两者必须相等（有了 yaml 就该改走 yaml 现算）"
        )
    return problems


def epoch_single_source_probe(entries: list[dict], data_root: str = "",
                             pregen_dir: str = "") -> list[str]:
    """★★ **epoch 单真相源** 机检（2026-10-16）：把 v1 冻结常量 ``gm.MATRIX_MAX_EPOCHS``
    **扰动成哨兵值**后逐档重建，断言**除无 yaml 具名豁免档外**，任何档的步数/保存/注记
    **都不得改变**。

    为什么这条必须有（⛔ 光"把值改对"不够）：本次全 54 档 ``expected_optimizer_steps_reachable``
    低估 2–8 倍的**根因**就是"epoch 存在第二个来源（``gm.MATRIX_MAX_EPOCHS``）被拿去算步数"。
    只把值改对、不测"那条路已断" ⇒ 日后有人再拿它算一次，同样的病会静默复发。
    ★ 这条**能失败**：把 :func:`patch_tier` 里的 ``run_config_epochs()`` 换回
      ``int(gm.MATRIX_MAX_EPOCHS)`` ⇒ 本探针立刻红（除豁免档外全档变化）。
    """
    sentinel = int(gm.MATRIX_MAX_EPOCHS) + 5
    watched = (
        "expected_optimizer_steps_per_epoch", "expected_optimizer_steps_reachable",
        "expected_optimizer_steps", "checkpoint_save_steps", "step_formula_v2",
    )
    real = [patch_tier(e, data_root, pregen_dir) for e in entries]
    saved = gm.MATRIX_MAX_EPOCHS
    try:
        gm.MATRIX_MAX_EPOCHS = sentinel
        probed = [patch_tier(e, data_root, pregen_dir) for e in entries]
    finally:
        gm.MATRIX_MAX_EPOCHS = saved
    problems: list[str] = []
    for a, b in zip(real, probed):
        tid = str(a["tier_id"])
        hits = [k for k in watched if a.get(k) != b.get(k)]
        ga = (a.get("acceptance") or {}).get("formal_gate") or {}
        gb = (b.get("acceptance") or {}).get("formal_gate") or {}
        if ga.get("step_gate_basis") != gb.get("step_gate_basis"):
            hits.append("acceptance.formal_gate.step_gate_basis")
        if tid in NOT_APPLICABLE_NO_CONFIG:
            continue  # 具名豁免：这 3 档本就（且只）用 v1 常量，⛔ 不判否
        if hits:
            problems.append(
                f"{tid}: 扰动 v1 常量 MATRIX_MAX_EPOCHS → {sentinel} 后字段变化 {hits}"
                f" ⇒ epoch 仍有**第二个来源**参与步数/保存口径（⛔ 必须走 run_config_epochs）"
            )
    return problems


def v1_entries_for_probe(manifest_path: Path) -> list[dict]:
    """探针用的 v1 条目集（与生成路径**逐字同源**：v1 清单的 tiers，⛔ 不用 v2 件反推）。"""
    return json.loads(V1_MANIFEST_PATH.read_text(encoding="utf-8"))["tiers"]


def check(manifest_path: Path, allow_unverified_subset_check: bool = False,
          subset_dir_override: "str | Path | None" = None) -> int:
    """只读校验：v2 清单的字段自洽性（不依赖网络/服务器）。

    配置 v2 目录、冻结子集目录都是**可选**交叉核对：本机看不到就**打印跳过原因**
    （不静默通过），看到就逐档核对。
    """
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    entries = manifest["tiers"]
    config_dir = PROJECT_ROOT / V2_CONFIG_DIR_REL
    have_configs = config_dir.is_dir()
    run_config_dir = PROJECT_ROOT / CONFIG_DIR_REL
    have_run_configs = run_config_dir.is_dir()
    problems: list[str] = []
    # ★★ 合法卡集机检（2026-10-16 新增）：对**在盘清单**逐档断言 gpus ⊆ legal_card_set()。
    for _t in entries:
        _v = non_legal_card_violation(str(_t.get("tier_id") or "?"), _t.get("gpus") or [])
        if _v:
            problems.append("合法卡集：" + _v)
    declared_anchor_diffs: list[str] = []
    if len(entries) != 54:
        problems.append(f"档数 {len(entries)} != 54")
    if manifest.get("config_version") != "v2":
        problems.append("config_version != v2")
    for e in entries:
        tid = e["tier_id"]
        if e.get("config") and not str(e["config"]).startswith(CONFIG_DIR_REL):
            problems.append(f"{tid}: config 不在 {CONFIG_DIR_REL}（跑次实际读取的档配置目录）")
        # ★ 跑次配置存在性核对（看不到目录即跳过，不静默通过）：清单指哪一份就必须有哪一份。
        if have_run_configs and e.get("config"):
            if not (PROJECT_ROOT / str(e["config"])).is_file():
                problems.append(f"{tid}: 清单声明的 config 文件不存在：{e['config']}")
        per = int(e["expected_optimizer_steps_per_epoch"])
        reach = int(e["expected_optimizer_steps_reachable"])
        # ★ 2026-10-16：epoch 数从**跑次档配置 yaml** 现算（单一真相源），⛔ 不用 v1 冻结常量。
        _ep = run_config_epochs(tid)
        if _ep is None:
            if tid not in NOT_APPLICABLE_NO_CONFIG:
                problems.append(f"{tid}: 读不到跑次档配置 yaml 的 max_epochs（且不在无 yaml 具名豁免内）")
        elif reach != per * int(_ep):
            problems.append(
                f"{tid}: reachable({reach}) != per_epoch({per}) × max_epochs({_ep})（yaml 现算）"
            )
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
                    if tid in PLANNED_STEPS_OVERRIDE and per == int(PLANNED_STEPS_OVERRIDE[tid]):
                        # ★ 已声明的锚定差异（不判否，但**必须打印**，偏离可见）
                        declared_anchor_diffs.append(
                            f"{tid}: per_epoch {per}（锚定覆盖）!= 未锚定公式值 {want_steps}（dp={dp}）"
                        )
                    else:
                        problems.append(f"{tid}: per_epoch {per} != 公式值 {want_steps}（dp={dp}）")
            elif not re.search(rf"= {per}\b", str(e.get("step_formula_v2", ""))):
                problems.append(f"{tid}: step_formula_v2 与 per_epoch={per} 不自洽")
        elif tid not in NOT_APPLICABLE_TIERS:
            if e["data"].get("source") != SOURCE_MINI:
                problems.append(f"{tid}: source != mini")
            # ★★ 修正轮 1（🔴B1a）：mini 档的 `subset_size` 必须**逐档等于**唯一真相源现算值
            #    ⇒ 值回退（100→20）当场报 MISMATCH、非 0 退出（⛔ 不再"check OK 却已回退"）。
            v1_subset = (V1_SUBSET_SIZE_BY_TIER.get(tid))
            try:
                want_mini = mini_tier_subset_size(tid, v1_subset)
            except AssertionError as exc:
                problems.append(f"{tid}: 子集条数唯一入口 fail-closed ⇒ {exc}")
            else:
                if int(e["data"]["subset_size"]) != int(want_mini):
                    problems.append(
                        f"{tid}: subset_size {e['data']['subset_size']} != 唯一真相源现算值 {want_mini}"
                        f"（⛔ 值回退）"
                    )
        # ★★ 修正轮 2（🔴B-1）：**非 native 档的 `per_epoch` 必须 = v1 值 × 口径比率**
        #    ⇒ 旧实现纯透传 v1（把 800 回退成 160）当场报 MISMATCH、非 0 退出。
        if tid not in V2_SUBSET_SIZE_BY_TIER:
            _v1_per = V1_PER_EPOCH_BY_TIER.get(tid)
            if _v1_per is None:
                problems.append(f"{tid}: 非 native 档但 v1 清单无 per_epoch ⇒ 无法复核（fail-closed）")
            else:
                try:
                    _ratio = caliber_subset_ratio(
                        tid, V1_SUBSET_SIZE_BY_TIER[tid], int(e["data"]["subset_size"]),
                        tid in NOT_APPLICABLE_TIERS,
                    )
                except AssertionError as exc:
                    problems.append(f"{tid}: 口径比率唯一入口 fail-closed ⇒ {exc}")
                else:
                    _want_per = int(_v1_per) * int(_ratio)
                    if per != _want_per:
                        problems.append(
                            f"{tid}: per_epoch {per} != v1({_v1_per}) × 口径比率({_ratio}) = {_want_per}"
                            f"（⛔ 计划步数回退；① 步数门会被实质放宽）"
                        )
        # ★★ 修正轮 1（🔴B1c）：清单自带 `key_semantics` 声明
        #    `expected_optimizer_steps` 语义 = `checkpoint_save_steps` ⇒ 逐档必须相等。
        if "checkpoint_save_steps" in e and "expected_optimizer_steps" in e:
            if int(e["expected_optimizer_steps"]) != int(e["checkpoint_save_steps"]):
                problems.append(
                    f"{tid}: expected_optimizer_steps {e['expected_optimizer_steps']} != "
                    f"checkpoint_save_steps {e['checkpoint_save_steps']}"
                    f"（违反清单自带 key_semantics 不变量）"
                )
        if not e.get("config") and tid not in NOT_APPLICABLE_TIERS:
            problems.append(f"{tid}: 可跑档却缺 config")
    if manifest["step_gate_audit"]["structural_limited_tiers"]:
        problems.append("v2 仍存在 structural_limited 档")

    # ★★ 通用锁卡断言（对**在盘清单**复核，⛔ 不只信生成期）：任何档 gpus 不得含邻居卡 / 生产锁卡 / 越界卡。
    #    唯一例外 = 该值恰等于本世代卡池（卡池自身与锁卡契约的矛盾 ⇒ 只登记）。
    _pool_hits: list[tuple[str, tuple[int, ...]]] = []
    for e in entries:
        _gpus = e.get("gpus") or []
        _hits = locked_card_hits(_gpus)
        if not _hits:
            continue
        _pool = tuple(GPU_POOL_BY_CARDS.get(int(e["cards"]), ()))
        if tuple(int(g) for g in _gpus) == _pool:
            _pool_hits.append((e["tier_id"], _hits))
        else:
            problems.append(
                f"{e['tier_id']}: gpus={_gpus} 含锁卡 {list(_hits)}（⛔ 不等于本世代卡池 {list(_pool)} "
                f"⇒ 拒绝；锁卡集合 LOCKED_CARDS={list(LOCKED_CARDS)}）"
            )
    if _pool_hits:
        _pools = sorted({tuple(int(g) for g in (next(x for x in entries if x["tier_id"] == t)["gpus"] or []))
                         for t, _h in _pool_hits})
        print(
            f"LOCKED-CARD-POOL-CONFLICT（已登记、不判否）: {len(_pool_hits)} 档的 gpus == 本世代卡池，"
            f"而该卡池与 GPU_FORBIDDEN_PRODUCTION={list(GPU_FORBIDDEN_PRODUCTION)} 相交 ⇒ "
            f"池={[list(p) for p in _pools]}；档号={sorted(t for t, _h in _pool_hits)}"
        )
    print(
        f"锁卡断言：LOCKED_CARDS={list(LOCKED_CARDS)}（邻居 {list(NEIGHBOR_CARDS)} ∪ 生产锁卡 "
        f"{list(GPU_FORBIDDEN_PRODUCTION)} ∪ 越界）——{len(entries)} 档逐档已核；"
        f"含邻居卡 GPU2 的档 = {sum(1 for e in entries if 2 in (e.get('gpus') or []))} 档"
    )

    print(
        "配置 v2 交叉核对："
        + (f"已逐档复算（{config_dir}）" if have_configs
           else f"**跳过**（本机看不到 {config_dir}；改由清单内的 step_formula_v2 自洽性核对）")
    )
    print(
        "跑次配置交叉核对："
        + (f"已逐档核对存在性（{run_config_dir}）" if have_run_configs
           else f"**跳过**（本机看不到 {run_config_dir}）")
    )
    # ★★ 冻结子集交叉核对（修正轮 1 · 🔴B1b 根因）：**三态、⛔ 不许静默跳过**。
    #    病史：原先 `Path(host_dir).is_dir()` 在盘值为**未展开字面量** ⇒ 恒 False ⇒
    #    每次只打印「跳过」并继续，7 条行数 MISMATCH 因此从未被发现，而报告仍引 `check OK`。
    #    现在：① 真实目录 ⇒ 逐档真核对；② 仓内 `subsets-v2/` ⇒ 用它真核对（本机常态）；
    #          ③ 都拿不到 ⇒ 打印 **UNVERIFIED** 并把"未覆盖范围"计入 problems（fail-closed，
    #             除非调用方**显式**传 `--allow-unverified-subset-check` —— 显式即会打印）。
    subset_dir, why_unresolved = resolve_subset_check_dir(
        str(subset_dir_override or manifest["data_sources"]["pregenerated_subsets"]["host_dir"])
    )
    subset_ok = subset_exempt = 0
    unverified_scope: list[str] = []
    if subset_dir is not None:
        for e in entries:
            tid = e["tier_id"]
            if not e["data"].get("subset_file_pregenerated"):
                # ★ 逐档具名豁免（⛔ 不是静默跳过）：打印档号与理由，且**不计入已核对数**。
                subset_exempt += 1
                unverified_scope.append(
                    f"{tid}: subset_file_pregenerated=False（未落地/不适用 ⇒ 本项不核对）"
                )
                continue
            p = subset_dir / f"{tid}.jsonl"
            if not p.is_file():
                problems.append(f"{tid}: 冻结子集缺失 {p}")
                continue
            lines = p.read_bytes().count(b"\n")
            if lines != int(e["data"]["subset_size"]):
                problems.append(
                    f"{tid}: 冻结子集行数 {lines} != subset_size {e['data']['subset_size']}"
                )
            else:
                subset_ok += 1
        print(
            f"冻结子集核对：{subset_dir}（**已逐档核对行数**：OK {subset_ok} 档；"
            f"具名豁免 {subset_exempt} 档＝{sorted(x.split(':')[0] for x in unverified_scope)}）"
        )
    else:
        print(f"冻结子集核对：**UNVERIFIED（未核对）** —— {why_unresolved}")
        unverified_scope = [
            f"{e['tier_id']}: 未核对（{why_unresolved}）" for e in entries
        ]
        if not allow_unverified_subset_check:
            problems.append(
                f"冻结子集核对 UNVERIFIED（{why_unresolved}）⇒ 本清单的"
                f"「声明 subset_size vs 冻结件行数」这一面**未被覆盖**，"
                f"⛔ 不得据此宣称 check OK；如需临时放行请显式加 --allow-unverified-subset-check"
            )
    if unverified_scope:
        print(f"UNVERIFIED-SCOPE（未覆盖范围，逐档具名；共 {len(unverified_scope)} 档）:")
        for x in unverified_scope:
            print(f"  · {x}")

    # ★★ 修正轮 2（🔴B-1 ＋ 审查建议 S2）：**跨件不变量机检** ——
    #    `manifest.per_epoch == docs.per_epoch_steps`（全 54 档；具名豁免 T017/T018 = W-7 既有，
    #    且豁免必须"确实仍在漂"，否则报陈旧 ⇒ 逼下一轮删出）。
    cross = cross_check_manifest_against_matrix(manifest_path)
    # ★★ epoch 单真相源（2026-10-16）：扰动 v1 常量 ⇒ 除具名豁免档外不得有任何变化。
    cross = list(cross) + epoch_single_source_probe(
        v1_entries_for_probe(manifest_path), "", ""
    )
    if cross:
        problems.extend(f"[跨件] {x}" for x in cross)
    else:
        print(
            "跨件不变量核对（三向）：manifest.per_epoch == docs.per_epoch_steps（52/52）"
            "｜manifest.reachable == docs.derived_steps ｜manifest.checkpoint_save_steps "
            "== docs.save_steps == 档位 yaml 的 training.save_steps"
            f"（具名豁免 {len(KNOWN_PRE_EXISTING_PER_EPOCH_DRIFT)} 档 = "
            f"{list(KNOWN_PRE_EXISTING_PER_EPOCH_DRIFT)}，W-7 既有、⛔ 本轮不修；已验明确实在漂）"
        )
    for d in declared_anchor_diffs:
        print(f"DECLARED-ANCHOR: {d}（已声明的锚定差异，不判否）")
    if problems:
        for p in problems:
            print(f"MISMATCH: {p}")
        return 1
    if declared_anchor_diffs:
        print(f"锚定差异（已声明）: {len(declared_anchor_diffs)} 档")
    print(f"check OK: 54 档；formal={sum(1 for e in entries if e['data'].get('source')=='formal')} "
          f"mini={sum(1 for e in entries if e['data'].get('source')=='mini')}；"
          "全部 reachable ≥ 5")
    return 0


def print_steps(v1_manifest: dict, pregen_dir: str = "") -> None:
    print(f"Q={Q} 门槛={MIN_OPTIMIZER_STEPS}  公式=ceil(floor(subset/dp)/Q)"
          f"   epoch 逐档读自 {CONFIG_DIR_REL}/<T>.yaml 的 training.max_epochs（单一真相源）")
    for tid in sorted(V2_SUBSET_SIZE_BY_TIER):
        size = V2_SUBSET_SIZE_BY_TIER[tid]
        dp = v2_dp_size(tid)
        steps = v2_native_steps_per_epoch(size, dp)
        v1e = next(t for t in v1_manifest["tiers"] if t["tier_id"] == tid)
        _ep = run_config_epochs(tid)
        print(
            f"  {tid}: subset {v1e['data']['subset_size']}→{size}  dp={dp}  "
            f"T={size // dp}  steps={steps}  epoch={_ep}  reachable={steps * int(_ep or 0)}  "
            f"(v1 朴素/上界式={v1e['expected_optimizer_steps_per_epoch']})"
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="matrix54 v2 运行清单生成/校验（不跑训练）")
    parser.add_argument("--out", type=Path, default=V2_MANIFEST_PATH, help="v2 清单落点")
    parser.add_argument("--check", action="store_true", help="只校验已落盘的 v2 清单")
    parser.add_argument("--print-steps", action="store_true", help="只打印步数对照，不写文件")
    parser.add_argument(
        "--allow-unverified-subset-check", action="store_true",
        help="★ 显式放行「冻结子集目录解析不到 ⇒ 未核对」的 fail-closed（默认不放行；用到即打印）",
    )
    parser.add_argument("--host-data-root", default=DEFAULT_HOST_DATA_ROOT,
                        help="228 上的 ELAM V5 原始数据根（data/、mini-dataset/、images/）")
    parser.add_argument("--pregen-subset-dir", default=DEFAULT_PREGEN_SUBSET_DIR,
                        help="228 上的 v2 冻结子集目录（其下 T0NN.jsonl）")
    parser.add_argument("--attempt-root", default=DEFAULT_ATTEMPT_ROOT,
                        help="runner 的 RUN_ROOT 推荐值（运行产物根；须与上面两者不嵌套）")
    parser.add_argument(
        "--config-fingerprint", default=None,
        help="**步数口径来源**目录总指纹（默认按冻结点算法对 samples/configs/matrix54-v2/ **现算**）",
    )
    args = parser.parse_args(argv)

    if args.check:
        _ov = args.pregen_subset_dir if args.pregen_subset_dir != DEFAULT_PREGEN_SUBSET_DIR else None
        return check(args.out, allow_unverified_subset_check=args.allow_unverified_subset_check,
                     subset_dir_override=_ov)

    v1_bytes = V1_MANIFEST_PATH.read_bytes()
    v1_manifest_sha = sha256_bytes(v1_bytes)
    v1_manifest = json.loads(v1_bytes.decode("utf-8"))

    if args.print_steps:
        print_steps(v1_manifest, args.pregen_subset_dir)
        return 0

    # ★ 默认**现算**（不再回读外部指纹文件）：外部文件会陈旧，回读旧值会把过期指纹
    # 静默写回清单字段（§1.4 单一真相源 / §2.2 不让文档说假话）。算法逐字 = v2 冻结点口径。
    fingerprint = args.config_fingerprint or config_dir_total_fingerprint(
        PROJECT_ROOT / V2_CONFIG_DIR_REL
    )
    run_fingerprint = config_dir_total_fingerprint(PROJECT_ROOT / CONFIG_DIR_REL)

    manifest = build_v2_manifest(
        v1_manifest, args.host_data_root, args.pregen_subset_dir, args.attempt_root,
        v1_manifest_sha, fingerprint, run_fingerprint,
    )
    # ★★ 生成链防丢失（fail-closed，**在写盘之前**）：重生成不得移除在盘清单已有的字段。
    lost = guard_regeneration_keeps_on_disk_fields(
        manifest,
        args.out,
        post_fill_patterns=POST_FILL_EXTRA_PATTERNS,
        allowed_drop=MANIFEST_ALLOWED_FIELD_LOSS,
        what="v2 运行清单 matrix54_manifest_source.json",
    )
    if lost:
        print(
            f"FATAL: 重生成会从在盘清单移除 {len(lost)} 条字段路径（前 5 条：{lost[:5]}）"
            f"⇒ 拒绝写盘 {args.out}（fail-closed）。",
            file=sys.stderr,
        )
        return 1
    # ★★ 修正轮 1（🔴B1a 的 fail-closed 面，⛔ 不是豁免/白名单变绿）：**值回退**守卫。
    #    病史：`guard_regeneration_keeps_on_disk_fields()` 只防"字段路径丢失"，**不防值回退**
    #    ⇒ 生成器把在盘 18 档 `subset_size=100` 写回 20 时 rc=0、且 `check()` 照样 OK。
    #    现在：写盘前对**口径关键字段**逐档做值一致性断言 —— 新内容的值必须 == **唯一真相源现算值**
    #    （同一套 `mini_tier_subset_size()` / `expected_optimizer_steps` 不变量）；
    #    在盘值若与现算值不同，要么是**在盘漂移**（如上一轮的手工并入漏改一半）⇒ 打印具名清单并**拒绝写盘**。
    drift = caliber_value_regression(manifest, args.out)
    if drift:
        print(
            f"FATAL: 重生成相对在盘清单存在 {len(drift)} 处**口径值不一致** ⇒ 拒绝写盘 {args.out}"
            f"（fail-closed；⛔ 不许静默回退 100→20 / 步数 ÷5）。前 8 条：",
            file=sys.stderr,
        )
        for d in drift[:8]:
            print(f"  VALUE-DRIFT {d}", file=sys.stderr)
        print(
            "  处置：① 口径变更请改 `make_subsets.MINI_SUBSET_TIERS_DECLARED` / `MINI_TIERS`（唯一真相源）；"
            "② 若在盘件才是真值 ⇒ 先查清为何与本生成器现算不一致；⛔ 不得加例外放行。",
            file=sys.stderr,
        )
        return 1
    # ★★ 修正轮 2（🟡W-1b）：**派生状态**守卫 —— 覆盖已存在的跑次台账时，⛔ 不得改写任何 tier 的 `status`。
    state_drift = derived_state_regression(manifest, args.out)
    if state_drift:
        print(
            f"FATAL: 重生成会把在盘清单的 {len(state_drift)} 档**派生 status** 改写"
            f"（跑次台账真值 ⇒ 只许外科式并入）⇒ 拒绝写盘 {args.out}。前 8 条：",
            file=sys.stderr,
        )
        for d in state_drift[:8]:
            print(f"  STATE-DRIFT {d}", file=sys.stderr)
        print(
            "  处置：① 整体重生成**不是**本清单的合法写路径（在盘含跑次回填字段）；"
            "② 口径/字段改动请走数据级外科式并入器；⛔ 不得加例外放行。",
            file=sys.stderr,
        )
        return 1
    text = json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"
    args.out.write_text(text, encoding="utf-8")
    rc = check(args.out, allow_unverified_subset_check=args.allow_unverified_subset_check)
    print(f"v2 清单已写出：{args.out}（sha256={sha256_bytes(text.encode())}）")
    print(f"  v1 基线 sha256={v1_manifest_sha}（v1 清单字节未动）")
    print(f"  步数口径来源 v2 总指纹={fingerprint}")
    print(f"  跑次配置目录 {CONFIG_DIR_REL} 总指纹={run_fingerprint}")
    return rc


if __name__ == "__main__":
    sys.exit(main())
