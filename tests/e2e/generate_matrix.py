#!/usr/bin/env python3
"""GRASPO 本期 54 档测试矩阵生成器（重写版，替代旧的全排列生成器）。

职责：把 `docs/capability-matrix.md` §7 的 54 档实测台账（T001–T054）逐档
翻译成「配置骨架 + 运行清单」，供后续 GPU 实验按档执行。生成物：

  - `samples/configs/matrix54/T###.yaml` —— 每档一个配置骨架
  - `samples/configs/matrix54/T###.blocked.md` —— 当前给不出配方档位的阻塞说明
  - `tests/e2e/matrix54_manifest.json` —— 运行清单（44+ 字段/档，含 GPU 集合、
    验收门槛、可表达性状态、**显存可行性三态**），是结果收集器 `scripts/collect_results.py` 的输入
  - `tests/e2e/run_matrix54.sh` —— 批量执行脚本骨架（自带锁卡守卫与可信采样）

**两条正交的口径（不要混读）**：

- **可表达性**（`ready` / `unverified` / `blocked`）答"**配置能不能被接受、能不能生成**"。
  CPT / OPD 已于 2026-09-19（`WP-X3`）解除 `blocked`：`train_method` 枚举含 `cpt`/`opd`，
  各有专用注册表与 ms-swift 映射（`swift pt` / GKD）。
- **显存可行性**（`feasible` / `infeasible` / `unmeasured` / `blocked`）答"**能不能声称装得下**"。
  `unmeasured` = 确定性算式通过、但存在**未测算**的额外显存消费者（目前仅 OPD 的教师，
  P-25）⇒ **不得写成「可行」**，等上机冒烟回填。

**★ 「可表达」≠「能跑通」≠「显存可行」**：三者都不互相蕴含。

**口径（用户拍板，权威）**：只用 1/2/4 卡，每次最多 4 卡且仅取 GPU0-5；
4 卡首选 {0,1,2,3}；上下文长度由递增加长法实测得出、不预设档位。

**为什么是重写而不是改造旧生成器**：旧 `generate_matrix.py` 的口径已废——
数学全排列出 56 个 yaml、镜像名 `graspo:0.29.0`（不存在）、`WORLD_SIZES=[1,2,4]`
全排列、无锁卡守卫、与 54 档台账不对应。旧口径的任何残留都会把"能力上限"
测错，因此按宪法 §18.1 重写。

用法：
    python3 tests/e2e/generate_matrix.py --dry-run --assert-count 54   # 只自检不落盘
    python3 tests/e2e/generate_matrix.py                              # 生成全部产物（默认取 mini 数据集）
    python3 tests/e2e/generate_matrix.py --train-source full          # 切回源数据 data/train.jsonl
    python3 tests/e2e/generate_matrix.py --gpu-override 'T001=4;T010=4,5'   # 卡集合覆盖（逐档映射）
    python3 tests/e2e/generate_matrix.py --gpu-override 'T001=4' \
        --manifest-out .local/matrix54_manifest.gpu-alt.json          # 生成变体清单，不动 canonical

**卡集合覆盖点（唯一一处）**：默认 `GPU_SETS` 把 10 个 1 卡档全压在 GPU0 ⇒ 只能串行。
`--gpu-override` 给出「档 → 卡集合」显式映射（无通配符），**只改用哪几张卡、不改卡数**
（不符即 fail-closed）；卡号 ⊆ GPU0–5，GPU6/7（生产 vLLM）出现即拒绝；跨 NUMA 只提示。
不传该参数 ⇒ manifest/runner 与旧版**逐字节相同**。运行期换卡集合用既有
`GRASPO_RUNNER_MANIFEST` 指向另一份清单（runner 零改动、档位 YAML 零改动）。

**训练子集的取数来源（本生成器是唯一真相源）**：默认取 **mini 数据集**
（`<ELAM_HOST>/mini-dataset/mini-short-mm-train.jsonl`，100 行、sha256 记入 manifest 的
`data.mini_dataset`）——它只用于「矩阵跑通」阶段压时间/压成本。**效果评测必须用全量 test 集
（`data/test.jsonl`，702 行），不得用 mini 集的跑通结果替代效果结论。**
⚠ **不要为了"顺手修正"去改 `data.train_path`**：该字段只约束**生成位置**，不约束**读取源**；
每档的 `<Tier>.jsonl` 仍生成到原 `$RUN_DIR/subsets/`，runner 的断言
`TRAIN_PATH == dirname(TRAIN_PATH)/<Tier>.jsonl` 与图像根反推
`dirname(dirname(train_path))/images` 都依赖它 —— 改配置反而会打断这条链。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

#: 卡集合校验/覆盖的唯一实现（§1.4 单一真相源）：生成期与运行期共用同一个模块，
#: 不在生成器里另写一套判据。本模块**只依赖 stdlib**，不破坏生成器的 stdlib-only 约束。
sys.path.insert(0, str(Path(__file__).resolve().parent))
from gpu_assignment import (  # noqa: E402  （必须在 sys.path 就位之后导入）
    GPU_ALLOWED,
    GPU_FORBIDDEN_PRODUCTION,
    MAX_CARDS_PER_JOB,
    GpuAssignmentError,
    check_gpus,
    describe_default_sets,
    format_assignment_line,
    gpu_assignment_fingerprint,
    parse_override,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = PROJECT_ROOT / "samples" / "configs" / "matrix54"
MANIFEST_PATH = PROJECT_ROOT / "tests" / "e2e" / "matrix54_manifest.json"
RUNNER_PATH = PROJECT_ROOT / "tests" / "e2e" / "run_matrix54.sh"

#: `record-gpu-memory` 采样产物的容器内落点（**单一真相源**，§1.4）。
#: 容器内 /out 由 runner 绑定到宿主 <RUN_ROOT>/<T###>（挂载表 `"$RUN_DIR|/out|rw"`），
#: 因此 /out/gpu 就是宿主采样摘要 `gpu_memory_summary.json` 的落点，由
#: `scripts/collect_results.py::_read_host_sample_peak_memory` 读入 `host_sample_peak_gib`
#: 旁路 —— **不是** §7 峰值列的口径（该列为后端二分口径，见使用点 4）。
#: 采集口径（宪法 §10.1：产物位置必须能被 config 描述）与取证链**逐字不变**。
#:
#: ★ 本常量是"显存摘要落在哪"的**唯一字面量**。四个使用点必须与它一致，任何漂移都是
#:   取证链断裂（机核见 `tests/e2e/test_run_matrix54_runner.py` 的
#:   `test_gpu_evidence_dir_is_one_anchor_across_producer_runner_and_collector`）：
#:     1) 生成物档配置 `gpu_monitor.output_dir`（build_config 直接引用本常量）；
#:     2) runner 的前置校验（把档配置读到的值与本常量比对，fail-closed）；
#:     3) 容器内 entry.sh 的 `GPU_EVIDENCE_DIR` + 收口段断言/缺口登记（生成期注入）；
#:     4) `scripts/collect_results.py::_read_host_sample_peak_memory` 读的 `<run_dir>/gpu/`：
#:        宿主采样摘要读入 `host_sample_peak_gib` 旁路，**不是** §7 峰值列的口径——该列为
#:        **后端二分口径**：native 档读 `rank_metrics` 的 rank0 `max_allocated_mib`；
#:        ms-swift 档读其自报 `memory(GiB)`（`max_memory_reserved`）。
#:
#: ★ 为什么必须是 `/out/gpu`（RUN_DIR 直下）而**不能**是 `<training.output_dir>/gpu`：
#:   矩阵档一律 `overwrite_output_dir: true`，训练进程启动时对**非空**的
#:   `<training.output_dir>` 执行 `shutil.rmtree(out)`
#:   （`src/graspo/flow/lora/lora_io.py::prepare_output_dir`）。采样器先于 torchrun
#:   启动，若落点在该子树内，则采样器刚建好的目录会被整棵删掉，首个 `append_jsonl`
#:   即 ENOENT——228 真机实测首错签名：`gpu_monitor.py:621 ... '/out/T010/gpu/gpu_memory.jsonl'`
#:   （2026-09-20，task-e6-nan-verdict 真跑日志）。落点放在 RUN_DIR 直下即与训练
#:   输出目录**解耦**，不依赖"谁先谁后"（§2 防呆）。
GPU_MONITOR_CONTAINER_DIR = "/out/gpu"

#: 采样间隔（秒）。与 W3 之前 CLI 上的 `--interval-sec 2` **逐值相同**：间隔决定
#: 产物里的样本条数 ⇒ 属"参与决定产物的参数"，按 §10.1 必须进 config、不得回 CLI。
GPU_MONITOR_INTERVAL_SEC = 2.0

#: 容器内 HOME / 缓存根（**单一真相源**，§1.4）。
#:
#: ★ 为什么需要它：`docker run --user <宿主 uid>:<gid>` 让容器进程**不是 root**，而镜像里
#:   root 的 HOME（`/root`）对这个 uid **不可写** ⇒ 任何写 `~/.cache` 的库都 PermissionError
#:   （HF datasets/transformers 缓存、deepspeed 的 torch 扩展构建、triton/inductor 编译缓存
#:   全在这个口径下）。因此把 HOME 显式指到一个**人人可写**的容器内目录（`/tmp` 是 1777；
#:   容器 `--rm` ⇒ 退出即消失，不留任何宿主残渣）。
#: ★ 为什么不指到 `/out`：`/out` 是**产物目录**，宿主侧的运维/清理动作按"产物"看待它
#:   （U5「跑完即删」）——在产物树里混一个不属于产物的 HOME，等于给清理动作留暗坑。
#: ★ 唯一字面量在本常量；所有使用点（DOCKER_ARGS 的 `-e`、entry.sh 的预检、dry-run 打印）
#:   都由 `CONTAINER_CACHE_ROOTS` 生成，测试断言它在生成物里只作为该常量出现一次（单源防漂移）。
CONTAINER_HOME_DIR = "/tmp/graspo-runner-home"

#: 容器内**可写缓存/状态根**的清单（**单一真相源**，§1.4）。
#:
#: 每项 = `(环境变量名, 相对 CONTAINER_HOME_DIR 的子路径)`；`"."` 表示 HOME 本身。
#:
#: ★ 为什么必须有一张表：容器以 `--user <宿主 uid>:<gid>` 运行后，镜像里任何"写死到
#:   root 可写路径"的缓存根都会在非 root 下变成 `PermissionError`，而训练在**数据集装载
#:   阶段**就会写它们。`043c776` 补了 `HOME`/`XDG_CACHE_HOME`/`TORCHINDUCTOR_CACHE_DIR`/
#:   `TRITON_CACHE_DIR`，**漏了 `MODELSCOPE_CACHE`**（228 实测确证）：镜像 ENV 把它写死成
#:   `/mnt/workspace/.cache/modelscope/hub`，而 `/mnt` 在镜像里是 `root:root 0755`
#:   （FHS 标准目录；`docker/Dockerfile*` 里**没有**任何 `mkdir /mnt/workspace`）⇒ 非 root
#:   建不出 `/mnt/workspace`。ms-swift 的 `swift/dataset/loader.py:57` 正是拿
#:   `modelscope.hub.utils.utils.get_cache_dir()`（读 `MODELSCOPE_CACHE`）拼出 `datasets` 的
#:   `cache_dir` ⇒ `datasets/builder.py` 的 `os.makedirs` 抛 `PermissionError: '/mnt/workspace'`，
#:   rank0 exit 1、其余 rank 被 SIGTERM；宿主侧只看到 `[ckpt-retention] FATAL: 未找到任何
#:   checkpoint-*`（因为训练**一步都没跑**）。
#: ★ 为什么用表而不是散落的 `-e` 行：清单**只有一处**，三个使用点全部由它生成 ——
#:   ① `docker run` 的 `-e` 注入；② `entry.sh` 的可写性预检；③ dry-run 与真跑的显式打印。
#:   散落的写法正是本条回归的成因（补了三个、漏了第四个），表把"再漏一个"变成不可能。
#: ★ 为什么全部挂在 `CONTAINER_HOME_DIR` 之下：`043c776` 已把 HOME 建在 `/tmp` 下的一个
#:   可写根（`/tmp` 是 1777，容器 `--rm` ⇒ 退出即消失；字面量只允许出现在那个常量里——
#:   测试断言它在生成器里只出现一次）。缓存根**沿用同一个根**，不另造第二个可写根（§1.4）；
#:   也**不放进 `/out`**（那是产物树，U5「跑完即删」会撞上非产物目录）。
#: ★ 为什么只列这五个（不列 `HF_HOME`/`TORCH_HOME`/`TRANSFORMERS_CACHE` 等）：镜像**没有**
#:   声明它们 ⇒ 它们的默认值本来就是 `~/.cache/...`，在 HOME 已指向可写根后天然可写；
#:   显式列出来只是重复一遍 HOME 的效果（本包不扩范围）。镜像里**声明过**的路径型变量只有
#:   两个：本表的 `MODELSCOPE_CACHE` 与 `NVM_DIR=/root/.nvm`（后者是 node/nvm 的，训练不用，
#:   故不入表——判据见 report §②）。
CONTAINER_CACHE_ROOTS: tuple[tuple[str, str], ...] = (
    ("HOME", "."),
    ("XDG_CACHE_HOME", ".cache"),
    # ★ 本次修的正主：镜像 ENV 把它指向 root-only 的 `/mnt/workspace/...`。
    #   显式覆盖到 HOME 之下后，modelscope/ms-swift 的一切落盘（`get_cache_dir()` 的所有
    #   调用点：`datasets` 缓存、`datasets/map_cache`、packing `tmp`、跨进程 `lockers`、
    #   权重 `offload_cache`、hub `files`/`_github`）都落在可写根内。
    ("MODELSCOPE_CACHE", ".cache/modelscope/hub"),
    ("TORCHINDUCTOR_CACHE_DIR", ".torchinductor"),
    ("TRITON_CACHE_DIR", ".triton"),
)


def container_cache_root_pairs() -> list[tuple[str, str]]:
    """把 :data:`CONTAINER_CACHE_ROOTS` 解析成 `(变量名, 容器内绝对路径)`。

    **唯一**解析点（§1.4）：`docker run` 的 `-e`、`entry.sh` 的预检清单、dry-run 的打印
    三处都只消费本函数的输出，任何一处都不再自己拼路径。
    """
    return [
        (name, CONTAINER_HOME_DIR if sub == "." else f"{CONTAINER_HOME_DIR}/{sub}")
        for name, sub in CONTAINER_CACHE_ROOTS
    ]


def render_cache_roots_array() -> str:
    """把清单渲染成 shell 数组正文（`"VAR=/path"` 逐行，不含 `NAME=(` / `)` 两行）。

    生成物里出现两份（runner 与 entry.sh）——它们**同源**，不是两份真相：
    两者都由本函数产出，测试断言二者逐字节相同。
    """
    return "\n".join(f'    "{name}={path}"' for name, path in container_cache_root_pairs())


#: 容器内 `USER` / `LOGNAME` 的**中性**取值（**单一真相源**，§1.4）。
#:
#: ★ 为什么必须显式给：以宿主的裸 uid 运行时，镜像的 `/etc/passwd` 里**没有这个 uid**，
#:   则 `pwd.getpwuid(os.getuid())` 抛 KeyError；`getpass.getuser()`（很多库启动时打环境
#:   横幅都会调：ms-swift / deepspeed / wandb / pip 都有）先看 `LOGNAME`/`USER` 环境变量，
#:   两者都没有才落到 `pwd` ⇒ 不显式给就可能**在训练开始前**炸掉。
#: ★ 为什么用中性常量而不是宿主账号名：宿主账号名属环境信息（宪法 §15.1/§16），注入容器后
#:   会随 stdout/logging 产物落到共享机上；中性值与宿主身份解耦。
CONTAINER_USER_NAME = "graspo"

#: 容器内 torch 探测证据的**文件名**与 **schema 标识**（**单一真相源**，§1.4）。
#:
#: ★★★ 这是一条**跨模块契约**：``scripts/collect_results.py`` 的
#:   ``TORCH_PROBE_FILENAME`` / ``TORCH_PROBE_SCHEMA`` 是这条契约的**消费端**，
#:   本处是**生产端**。两边必须逐字一致——**改一处必须同步另一处**；不一致的后果是
#:   探测证据被判"schema 不匹配 ⇒ 不采信"，A3 退回弱证据路径（环境性伪否复发）。
#:   防漂移由 ``tests/e2e/test_generate_matrix.py`` 的
#:   ``test_torch_probe_contract_matches_collector`` 断言（直接 import collector 的常量比对）。
#:
#: 为什么生成器不直接 import collector 的常量：generator 全程 stdlib-only（见文件头），
#: 而 collector 的导入链会拉起 graspo 包；生成期多一个重依赖入口不划算。改用"同一字面量
#: + 生成期/测试期双向断言"的防呆方式（§2.1 契约即防呆）。
TORCH_PROBE_FILENAME = "torch_probe.json"
TORCH_PROBE_SCHEMA = "graspo.torch_probe.v1"

#: 探测证据在**容器内**的落点（= 宿主 ``<RUN_ROOT>/<T###>/torch_probe.json``）。
#: 为什么是 RUN_DIR 直下而不是训练输出目录里：与 ``GPU_MONITOR_CONTAINER_DIR`` 同因——
#: ``training.output_dir``（= ``/out/<T###>``）会被训练侧 ``overwrite_output_dir`` 的
#: ``shutil.rmtree`` 清理，而 ``collect_results.read_torch_probe`` 读的正是
#: ``<run_dir>/torch_probe.json``（run_dir = 宿主 ``<RUN_ROOT>/<T###>``）⇒ 同源于运行目录。
TORCH_PROBE_CONTAINER_PATH = "/out/torch_probe.json"

#: 探测脚本在**容器内**的落点。它由 runner 生成到 ``$RUN_DIR/torch_probe.py``，
#: 而 ``$RUN_DIR`` 已整体绑定到容器 ``/out`` ⇒ 无需新增挂载项（不存在"在只读挂载下
#: 建 mountpoint"的问题，§10.1 挂载表不因此变化）。
TORCH_PROBE_SCRIPT_CONTAINER_PATH = "/out/torch_probe.py"

#: 被探测权重的**相对 run_dir 定位秩**（**唯一**定义处；契约 §6.1 硬约束 3）。
#: 逐项尝试、取第一个有 ``rank_*.pt`` 的目录，``relpath`` 一律由**运行期实测的目录**
#: 相对 run_dir 推出（POSIX），不写死字面量：
#:   1) ``<T###>/final`` —— native 权威布局（容器 ``/out/<T###>/final`` = 宿主
#:      ``<RUN_ROOT>/<T###>/<T###>/final``；真机核对见 task-j2-torch-probe 报告 §③）；
#:   2) ``final`` —— 兼容"已经位于 run_dir 直下"的布局；
#:   3) ``find -maxdepth 3 -name final`` —— 兜底发现（先按 posixpath 排序取第一个有 rank_*.pt 的）。
#: 探测脚本会把**实际选中**的目录与 relpath 打进 stdout（§2.2 显式即防呆），
#: 因此第一档上机就能当场核对布局假设，不靠猜。
TORCH_PROBE_RUN_DIR = "/out"

# ── 台账口径常量（单一真相源，§1.4）────────────────────────────────────────
CARDS: tuple[int, ...] = (1, 2, 4)
EXPECTED_TOTAL = 54
EXPECTED_BY_ALGORITHM = {"CPT": 9, "SFT": 18, "GRASPO": 18, "OPD": 9}
EXPECTED_BY_CARDS = {1: 18, 2: 18, 4: 18}
EXPECTED_BY_BACKEND = {"ms-swift": 36, "native": 18}
EXPECTED_BY_MODE = {"LoRA": 36, "全量": 18}

#: 4 卡首选 {0,1,2,3}（唯一全部位于 NUMA0）；GPU6/7 为生产卡，永不出现。
#  —— 这是**默认值**：不设覆盖时逐字生效（向后兼容）。
GPU_SETS: dict[int, tuple[int, ...]] = {1: (0,), 2: (0, 1), 4: (0, 1, 2, 3)}

#: ★ 卡集合**覆盖点**（本工程唯一一处；判据/解析在 `tests/e2e/gpu_assignment.py`）。
#:
#: 动机：默认把 10 个 1 卡档全压在 GPU0 ⇒ 只能串行，且 GPU0 被他人占用即整批开不了工。
#: 用法（生成期，逐档显式映射、无通配符）：
#:     python3 tests/e2e/generate_matrix.py --gpu-override 'T001=4;T010=4,5'
#: 未在覆盖值里出现的档 ⇒ 沿用 `GPU_SETS`（并在 stdout 显式打印来源）。
#: 覆盖只改**用哪几张**，**不改卡数**（不符即 fail-closed，见 `gpu_assignment.check_gpus`）。
#: 运行期换卡集合：用既有环境变量 `GRASPO_RUNNER_MANIFEST` 指向另一份清单
#: （runner 只从清单读卡集合 ⇒ 不改 runner、不改档位 YAML；实际用哪几张卡由守卫写进运行产物）。
#: A4「同档两跑必须同卡位」：卡集合只由清单给出，两次运行引用同一份清单即可；
#: `gpu_assignment_fingerprint` 给出映射指纹，生成时打印、可用
#: `python3 tests/e2e/gpu_assignment.py check --manifest <清单> --expect-fingerprint <指纹>` 复核。
GPU_OVERRIDE_CLI_FLAG = "--gpu-override"

# ── 模型权重：运行链路的一环，必须显式挂载（不能只写容器内路径就了事）──────
# **宿主机上的模型根目录属于环境信息**（§15.1/§16），不写进 tracked 文件：
# 运行时由 runner 从环境变量 ``GRASPO_MODELS_HOST_ROOT`` 取（见 .local/ 下的本地配置），
# 只读挂载到容器内 ``/models``。生成器只记录**容器内**路径；"模型从哪来"这一环
# 由 runner 的 fail-closed 前置断言守住（缺失即拒绝启动，不会掉到"数据问题"分类）。
MODELS_HOST_ROOT_ENV = "GRASPO_MODELS_HOST_ROOT"
MODELS_CONTAINER_ROOT = "/models"

#: 容器内目录名（= 宿主侧相对模型根的目录名）；显示名是用户给定的专名，不含内网信息。
_MODEL_DIR_NAMES: dict[str, str] = {"9B": "Qwen3.5-9B", "27B": "Qwen3.8-27B"}


def models_host_dir_env(model: str) -> str:
    """该档模型目录的宿主侧**可选**覆盖环境变量名。

    默认取 ``$GRASPO_MODELS_HOST_ROOT/<容器内目录名>``；仅当某档模型不在同一
    模型根下时，才用 ``GRASPO_9B_HOST_DIR`` / ``GRASPO_27B_HOST_DIR`` 单点覆盖。
    """
    return f"GRASPO_{model.upper()}_HOST_DIR"


MODELS: dict[str, dict[str, Any]] = {
    size: {
        "name": dirname,
        "path": f"{MODELS_CONTAINER_ROOT}/{dirname}",
        # ★ 视觉塔能力（本矩阵两个基座的**事实**，供 `lora_target_preset` 做条件选择）。
        #   Qwen3.5-9B / Qwen3.8-27B 都是多模态基座：native 侧
        #   `native_qwen_lora_available_targets` 会给出 `visual.*` 可用目标，
        #   运行期预检也按 `image_token_id` 判定"模型有视觉塔"。
        #   写成**显式字段**（不从模型名 `Qwen3.5` 猜）：将来换纯文本基座时改这一处。
        "vision": True,
    }
    for size, dirname in _MODEL_DIR_NAMES.items()
}


def has_vision(size: str) -> bool:
    """该档基座是否具备视觉塔（纯数据，不猜模型名）。"""
    return bool(MODELS[str(size)].get("vision"))


def data_has_media(algorithm: str) -> bool:
    """该算法的训练子集是否含媒体（图像）。

    ELAM V5 的 train/test 全部含图（jsonl 里写 `"image": "../images/…"`），矩阵子集按
    算法取前 N 条，因此**每个算法**的子集都含图——这里显式声明该事实。
    将来若引入纯文本子集，改这一处即可。
    """
    return bool(DATA_HAS_MEDIA_BY_ALGORITHM.get(str(algorithm), False))


def lora_target_preset(tier: dict[str, Any]) -> str:
    """LoRA 档的 ``lora.target_preset``：**条件选择**，不无条件写死。

    判据：**模型有视觉塔 且 该档数据含图** ⇒ ``vision_common``；否则 ``language_safe``。

    - ``vision_common``（`core/lora.py`）含 merger + blocks 的视觉模式 ⇒ native 侧
      `_replace_visual_lora_modules` 真把视觉塔换成可训 LoRA，运行期预检的第 3 步
      （"visual LoRA 参数必须已注册且 requires_grad"）才能通过。
      【实测证据】T028 修复前失败态：`language_safe` ⇒ 0 个可训视觉参数 ⇒
      `RuntimeError: multimodal preflight failed: no trainable visual parameters found`；
      换 `vision_common` ⇒ **同一段代码**产出 28 个可训视觉参数
      （见 `tests/flow/lora/test_visual_lora_trainable.py`）。
    - **"只训语言塔"仍是合法组合**：模型无视觉塔、或数据是纯文本时条件不成立，
      仍下发 ``language_safe``——不得无条件改成视觉预设。

    注：``lora.target_preset`` 是 **native 侧模块名预设**；ms-swift 后端用
    regex/``all-linear`` 的 ``target_modules``，配置映射会在启动时记一条"该字段在
    msswift 后端不生效"的 WARNING（``native_only_fields``）。是否 native 不影响本判据
    ——判据只表达"这档该不该训视觉塔"这一事实，两个后端各有自己的表达方式。
    """
    if has_vision(str(tier["model"])) and data_has_media(str(tier["algorithm"])):
        return "vision_common"
    return "language_safe"


# ── 训练数据：ELAM V5（只读确认过结构，非占位符）────────────────────────────
# 【实测确证】结构（数据根目录下）：
#   data/train.jsonl    （6378 行）
#   data/test.jsonl     （ 702 行）
#   images/*.jpg        （15954 个）
# 图像在 jsonl 里写作 `"image": "../images/xxx_left_eye.jpg"`（**相对 data 文件所在目录**），
# 由 grasp 的 resolve_messages_media_paths 以 (data_dir / path).resolve() 展开为绝对路径。
# 因此容器内约定：整棵数据目录挂到 /work/elam-v5，
#   data_dir = /work/elam-v5/data，`../images` → /work/elam-v5/images。✓
#
# **宿主机上的数据根目录属于环境信息**（§16），不写进 tracked 文件：
# 运行时由 runner 从环境变量 `GRASPO_ELAM_HOST_ROOT` 取（见 .local/ 下的本地配置）。
ELAM_HOST_ROOT_ENV = "GRASPO_ELAM_HOST_ROOT"
ELAM_CONTAINER_ROOT = "/work/elam-v5"
ELAM_TRAIN_JSONL = f"{ELAM_CONTAINER_ROOT}/data/train.jsonl"
ELAM_TEST_JSONL = f"{ELAM_CONTAINER_ROOT}/data/test.jsonl"
ELAM_SUBSET_DIR = f"{ELAM_CONTAINER_ROOT}/subsets"
ELAM_TRAIN_COUNT = 6378
ELAM_TEST_COUNT = 702
ELAM_IMAGE_COUNT = 15954

# ── mini 数据集：「矩阵跑通」阶段的取数来源（**默认**），与源数据物理隔离 ──────────
# 用户硬约束（数据防呆）：不同用途的数据**物理隔离**、靠**命名与路径**防误用；
# mini 文件名自带 `mini-short-mm-` 前缀、落独立目录 `mini-dataset/`，
# **不得改回 `train.jsonl`**（否则与源数据同名混放，防呆失效）。
# 落点口径与源数据一致：**相对于宿主数据根**（`$ELAM_HOST`，运行时由
# `GRASPO_ELAM_HOST_ROOT` 注入），因此这里只写相对位置，不写任何宿主绝对路径（§15.1/§16）。
ELAM_MINI_JSONL_RELATIVE = "mini-dataset/mini-short-mm-train.jsonl"
#: 取数来源的**运行时覆盖点**（runner 读它）。`full` 侧复用既有的 `ELAM_HOST` 口径，
#: 不另造第二套真相源；mini 侧因落点不在容器根下，需要一个显式路径变量（§2.2 显式即防呆）。
ELAM_MINI_JSONL_ENV = "GRASPO_ELAM_MINI_JSONL"
#: 生成期读取**开发机本地镜像**算 sha256/行数（与 228 上那份同源、已核 sha256 一致）：
#: manifest 里的摘要必须**实测算出**，不得硬编码（硬编码的摘要等于没有摘要）。
#: 注意本地 staging 目录比 228 落点多一层 `mini/`（见 `.local/mini-dataset/mini/upload-228.md`）。
LOCAL_MINI_JSONL = (
    PROJECT_ROOT / ".local" / "mini-dataset" / "mini" / "mini-short-mm-train.jsonl"
)

#: 每档训练子集大小：按 §6 门槛取下限即可（用户已定"尽量省资源"，不整集跑）。
#: SFT ≥100 条、RL ≥20 条（GRASPO 属 RL）。
SUBSET_SIZE_BY_ALGORITHM: dict[str, int] = {"CPT": 100, "SFT": 100, "GRASPO": 20, "OPD": 20}

#: 每个算法的子集**是否含媒体（图像）**。ELAM V5 的 train.jsonl 每行都带
#: `"image": "../images/…"`，子集是按算法取前 N 条 ⇒ 四个算法的子集都含图。
#: 显式声明（而不是"因为名字里有 ELAM 所以当然有图"）：`lora_target_preset` 靠它
#: 决定要不要把视觉塔纳入可训目标，判据必须是可读的（宪法 §2.2 显式即防呆）。
DATA_HAS_MEDIA_BY_ALGORITHM: dict[str, bool] = {
    "CPT": True,
    "SFT": True,
    "GRASPO": True,
    "OPD": True,
}

#: ⚠️ 已知数据问题，**必须显式标注、不得静默忽略**（评测口径由评测链路负责，
#: 本工程只负责不掩盖）：train/test 样本 id 不重叠，但图像去重存在交集。
DATA_INTEGRITY_CAVEAT = (
    "ELAM V5 train/test 样本 id 不重叠；但**图像去重交集 752 张**（占测试图像 38.5%），"
    "且 **169/702 个测试样本的图像集是某训练样本的子集**。训练集与评测集「同源不重叠」"
    "的口径在图像层面不成立，评测结论须在对齐该口径后再下判断。"
)

INITIAL_MAX_PROMPT_LENGTH = 8192
TIMEOUT_SEC = 7200

#: 已作废口径——生成物中出现任何一条即自检失败（防"把框架 bug 固化成能力上限"）。
#:
#: ★ `allocator 主口径` / `主口径摘要`（2026-09-21 追加）：`gpu_memory_summary.json`
#: 是**宿主侧 nvidia-smi 采样旁路摘要**（喂 `host_sample_peak_gib`），既**不是**
#: 容器内 allocator（那是 `rank_metrics` 的 `memory.max_allocated_mib`），也**不是**
#: §7「实测每卡峰值显存」列的口径（该列为后端二分口径，唯一映射见
#: `src/graspo/core/result_judge.py::PEAK_MEMORY_CALIBER_BY_BACKEND`）。
#: 旧措辞把两个量混成一个，会持续误导读者（§1.4 单一真相源 / §2.2 显式即防呆）。
FORBIDDEN_PATTERNS: tuple[str, ...] = (
    "2卡96K",
    "FSDP2 权重分片",
    "95格",
    "graspo:0.29.0",
    "WORLD_SIZES",
    "DOCKER_IMAGE",
    "allocator 主口径",
    "主口径摘要",
)

#: 全参档位的可表达性说明：配置字段已存在（`tuner_type: full`），但能力本身
#: 是本期的 ⚠️ 开发目标，必须实测后才能标 ✅。
FULL_MODE_NOTE = (
    "全参路径由 `tuner_type: full` 表达（native 与 ms-swift 共用同一开关）。"
    "该能力当前 ⚠️ 未验证（capability-matrix §4「全量 · native / ms-swift」），"
    "且 native 全参只支持 PP 分片（tp_size>1 或 dp_size>1 会被配置校验 fail-closed 拒绝），"
    "因此本档 native 布局固定为 pp_size=卡数、dp=tp=1。"
)

# ── 显存可行性**三态**判定（进 manifest 的 `feasibility.verdict`，显式优于布尔二值）──
# 为什么不是布尔：CPT/OPD 之前，"估算过预算"与"没得估"两件事被压进同一个 None；
# OPD 引入了第三种情况——**确定性算式通过，但存在未测算的额外显存消费者（教师）**。
# 混进 True 就是"为凑数放行"，混进 False 就是"把没测过的说成跑不了"。
VERDICT_FEASIBLE = "feasible"  # 确定性算式 ≤ 预算，且无未测算的额外消费者
VERDICT_INFEASIBLE = "infeasible"  # 确定性算式已超预算 ⇒ 拒绝生成
VERDICT_UNMEASURED = "unmeasured"  # 算式通过，但有未测算的额外消费者 ⇒ 不得声称"可行"
VERDICT_BLOCKED = "blocked"  # 当前给不出配方（能力未落地 / 无分片手段）
#: ★ 合法终态（用户授权）：该档"确实不适用"——**不存在用户想要的那种配置**，不是"还没做"。
#: 与 `blocked` 的区别：`blocked` = 现在给不出配方、能力落地后可能给得出；
#: `not_applicable` = **给不出也不该给**（用户已拍板的口径下不存在该档的合法形态）。
VERDICT_NOT_APPLICABLE = "not_applicable"

ALGORITHM_TO_TRAIN_METHOD = {"CPT": "cpt", "SFT": "sft", "GRASPO": "graspo", "OPD": "opd"}

#: ★ RLHF（GRASPO/GRPO）会把训练子集**重复**这么多次再进训练循环：ms-swift 的
#: ``RepeatSampler(mini_repeat_count=num_generations)``，而
#: ``num_generations = training.rollout_group_size``（``msswift/_config_mapping.py`` 的
#: ``stage == "rlhf"`` 分支）。
#: **单一真相源 = ``src/graspo/core/schema.py`` 的 ``rollout_group_size`` 默认值**；
#: 矩阵档位配置**不覆盖**该字段 ⇒ 运行时用的就是这个默认值。
#: 这里只能写常量（生成器不导入 ``graspo.core.schema``：那条 import 会拉起
#: pydantic + ripple→torch 的重依赖，生成器必须保持 stdlib-only）。
#: 防漂移由 ``tests/e2e/test_generate_matrix.py`` 的**源码文本级 parity 测试**兜住：
#: schema 默认值一改，测试立刻判红。
ROLLOUT_GROUP_SIZE = 8

#: ★ native GRASPO 侧的 **rollout 队列批大小** ``rollout_queue_batch_size``。
#: **单一真相源 = ``src/graspo/core/schema.py`` 的同名默认值**（矩阵档位配置**不覆盖**
#: 它 ⇒ 运行时用的就是这个默认值）。与 :data:`ROLLOUT_GROUP_SIZE` 同理，生成器必须
#: 保持 stdlib-only，只能写常量；防漂移由 ``tests/e2e/test_generate_matrix.py`` 的
#: **源码文本级 parity 测试**兜住（schema 默认值一改即判红）。
NATIVE_ROLLOUT_QUEUE_BATCH_SIZE = 8

#: ★ **replay buffer 阈值**（native GRASPO 每个 optimizer step 的触发条件）。
#: **不是新造的口径**：它与 ``src/graspo/core/schema.py`` 的
#: ``replay_buffer_optimize_threshold`` 属性**同式同值**——该属性返回
#: ``rollout_queue_batch_size × rollout_group_size``（``schema.py:389-390``），
#: 消费点是 ``flow/trainer/optimize.py`` 的 ``len(replay_buffer) >= threshold``。
#: 生成器不导入 schema（见上）⇒ 就地按同一算式展开；parity 由源码文本级测试兜住。
NATIVE_REPLAY_BUFFER_OPTIMIZE_THRESHOLD = (
    NATIVE_ROLLOUT_QUEUE_BATCH_SIZE * ROLLOUT_GROUP_SIZE
)

#: ★ CPT / OPD 的**通道**已落地（`WP-X3`，2026-09-19）：`train_method` 枚举含 `cpt`/`opd`，
#: 配置层对这两个算法**接受**、对 `cpt|opd + native` **fail-closed 拒绝**；路由表
#: （`core/discovery._REGISTRY_BY_TRAIN_METHOD`）把它们分别落到 `graspo.cpt_backends` /
#: `graspo.opd_backends`。⇒ 本生成器**不再**把这两类档判为 `blocked`。
#: 依据：`docs/capability-matrix.md` §4（CPT · ms-swift = ⚠️ 未验证；OPD · ms-swift = ⚠️ 未验证；
#: 两者 native 侧 = ⛔ 不支持）。
CPT_OPD_UNBLOCKED_NOTE = (
    "CPT/OPD 的**配置通道**已由 WP-X3 落地（train_method 枚举 + 专用注册表 + ms-swift "
    "pretrain/GKD 映射）。因此本档**可表达**。注意两种口径不要混："
    "**「可表达」= 配置能被接受、能生成、能启动**；**「跑通」= 真机跑出 A1–A6 证据**。"
    "本档当前只到前者。"
)

#: CPT（继续预训练）走了 ms-swift 的 `swift pt` 通道（= `swift sft --use_chat_template false
#: --loss_scale all`）。数据集形态是**纯文本续训行**，由
#: `flow/msswift/dataset.py::build_cpt_rows` 从 ELAM V5 形态的样本摊平得到。
CPT_NOTE = (
    CPT_OPD_UNBLOCKED_NOTE + " 数据形态：ms-swift 官方预训练行（单条 assistant 纯文本 + 媒体块）。"
    "**★ 语料口径（不得误读）**：本档用的是 **ELAM V5 形态**的指令数据摊平后的纯文本，"
    "目的是**跑通通道**，**不是「CPT 语料已就绪」**——本期没有预训练语料，"
    "语料问题按用户 2026-09-18 拍板登记为「回头再说」，本生成器不造语料。"
)

#: OPD（on-policy 蒸馏）走了 ms-swift 的 **GKD** 通道（`--rlhf_type gkd`），
#: 教师是同一训练进程内的**独立冻结模型**（`--teacher_model`），学生现场采样、
#: 教师现场打分（逐 token 稠密信号）。教师/学生对由用户 2026-09-18 拍板：
#: 教师 = `Qwen3.8-27B`、学生 = `Qwen3.5-9B`。
OPD_NOTE = (
    CPT_OPD_UNBLOCKED_NOTE + " 算法路径：GKD（`--rlhf_type gkd`）+ 独立冻结教师。"
    "**★ 教师侧显存未测算（P-25）**：教师模型在训练进程内**额外占一份显存**，"
    "本生成器没有教师侧的实测数字 ⇒ 所有 OPD 档的可行性判定为"
    f"**「待上机测量」（{VERDICT_UNMEASURED}）**，**不得**预先写成「可行」。"
)

#: OPD 教师模型在台账里的键（教师 = 27B，学生 = 台账的 `model`）。
OPD_TEACHER_MODEL = "27B"

#: ★ OPD 的**学生固定为 9B**（用户拍板，2026-09-19）：
#: 用户的原话是"OPD 要的是**教师 27B → 学生 9B**（省显存，价值高）"⇒ OPD 这一条通道的
#: **学生恒为 9B**，教师恒为 27B 常量。台账 §7 的 `model` 列在 graspo 里处处是
#: "**被训练的学生**"（`model.model_path`）⇒ 台账里那三行 **27B OPD 档**（`T052`–`T054`）
#: 若按现行口径读作"学生 = 27B"，就需要 **27B 学生 + 27B 教师**：教师权重**实测 50.10 GiB**
#: （上机 probe，bf16），再加 27B 学生权重（≈52 GiB）
#: ⇒ **权重常驻就 ≈100 GiB**，与用户"省显存"的意图**正好相反**。
#: ⇒ **不存在用户想要的那种配置** ⇒ 这三档判 **`not_applicable`（确实不适用）**，
#: 理由见 :data:`OPD_NOT_APPLICABLE_NOTE`。这是用户授权的**合法终态**
#: （目标验收口径明文："「失败」与「确实不适用」是合法产出"），**不是 blocked（还没做）**。
OPD_STUDENT_MODEL = "9B"

OPD_NOT_APPLICABLE_NOTE = (
    "★ 确实不适用（not_applicable）：用户拍板 **OPD 教师 = 27B 常量、学生 = 9B**（省显存，价值高）"
    "⇒ OPD 通道的学生**固定 9B**。台账该档的 `model` 列在 graspo 里是**被训练的学生**，"
    "按现行口径读作 27B 学生 ⇒ 需要 **27B 学生 + 27B 教师**：教师 bf16 权重**实测 50.10 GiB**"
    "（上机 probe），再加 27B 学生权重（≈52 GiB）⇒ **权重常驻 ≈100 GiB**，"
    "与用户「省显存」的意图正好相反。**因此用户想要的那种配置不存在** ⇒ 本档不适用。"
    "★ 这是**合法终态**（验收口径明文：「失败」与「确实不适用」都是合法产出），"
    "**不是「blocked（当前给不出配方、将来可能给得出）」**，也不是「跑失败了」。"
)


def opd_not_applicable(tier: dict[str, Any]) -> tuple[bool, str | None]:
    """该档是否属于"**确实不适用**"（用户已拍板的口径下不存在合法配置）。

    单一真相源（§1.4）：配置头部、manifest 的 ``applicability``、可行性与可表达性判定
    都从这里取值，不各判一次。
    """
    if str(tier["algorithm"]) != "OPD" or str(tier["model"]) != "27B":
        return False, None
    return True, OPD_NOT_APPLICABLE_NOTE


def opd_teacher_relation(tier: dict[str, Any]) -> str | None:
    """OPD 档的教师/学生关系：``independent`` / ``self-distillation`` / ``not-applicable``。

    - 非 OPD 档 ⇒ ``None``（"关系"这个概念对它们不适用，不编造一个值）；
    - OPD 但不适用档 ⇒ ``not-applicable``（拒绝按"自蒸馏"渲染）；
    - 其余 OPD ⇒ 比较学生与教师路径：同名 ``self-distillation``，否则 ``independent``。

    单一真相源：配置头部注释、manifest、报告都引用本函数，不各判一次（§1.4）。
    """
    if str(tier["algorithm"]) != "OPD":
        return None
    not_applicable, _ = opd_not_applicable(tier)
    if not_applicable:
        return "not-applicable"
    student = MODELS[str(tier["model"])]["path"]
    teacher = MODELS[OPD_TEACHER_MODEL]["path"]
    return "self-distillation" if student == teacher else "independent"


# ── 档位台账构造 ────────────────────────────────────────────────────────────


def _block(algorithm: str, backend: str, model: str, mode: str) -> list[dict[str, Any]]:
    """按固定顺序生成一个 (算法, 后端, 模型, 模式) 块：卡数 1 → 2 → 4。"""
    return [
        {"algorithm": algorithm, "backend": backend, "model": model, "mode": mode, "cards": c}
        for c in CARDS
    ]


def apply_gpu_override(
    tiers: list[dict[str, Any]],
    override: Mapping[str, Sequence[int]] | None = None,
) -> list[str]:
    """把「档 → 卡集合」覆盖值应用到每档 ``gpus``，返回被覆盖的档号（升序）。

    **fail-closed**（§2.3）：未知档号、卡数不符、含 GPU6/7、越界、重复、空集合
    一律抛 :class:`GpuAssignmentError` —— 拒绝生成，不产出"将就"的清单。
    **未覆盖的档同样过守卫**（默认值也必须合法），因此这条判据对全部 54 档生效。
    覆盖只改 ``gpus``，**不碰** ``cards`` 与任何其它字段。
    """
    override = override or {}
    known = {tier["tier_id"] for tier in tiers}
    unknown = sorted(set(override) - known)
    if unknown:
        raise GpuAssignmentError(
            f"{GPU_OVERRIDE_CLI_FLAG} 含未知档号 {unknown}；合法档号见台账（T001–T054），"
            f"写错的档号不会被静默忽略"
        )
    overridden: list[str] = []
    for tier in tiers:
        tier_id = str(tier["tier_id"])
        cards = int(tier["cards"])
        if tier_id in override:
            check_gpus(tier_id, override[tier_id], cards)  # 先校验，再落值
            tier["gpus"] = [int(gpu) for gpu in override[tier_id]]
            overridden.append(tier_id)
        else:
            check_gpus(tier_id, tier["gpus"], cards)
    return sorted(overridden)


def build_ledger(
    gpu_override: Mapping[str, Sequence[int]] | None = None,
) -> list[dict[str, Any]]:
    """按 §7 台账的逐行顺序生成 54 档（T001 → T054）。

    `gpu_override`（档 → 卡集合）**只改每档用哪几张卡**；不给时行为与旧版逐字相同
    （``GPU_SETS``），并同样过 `gpu_assignment.check_gpus` 范围守卫。
    """
    tiers: list[dict[str, Any]] = []

    # CPT 9 = 9B(LoRA,全量)×3卡 + 27B(LoRA)×3卡
    tiers += _block("CPT", "ms-swift", "9B", "LoRA")
    tiers += _block("CPT", "ms-swift", "9B", "全量")
    tiers += _block("CPT", "ms-swift", "27B", "LoRA")
    # SFT 18 = 9B(LoRA native, LoRA ms-swift, 全量 native, 全量 ms-swift)
    #          + 27B(LoRA native, LoRA ms-swift)
    tiers += _block("SFT", "native", "9B", "LoRA")
    tiers += _block("SFT", "ms-swift", "9B", "LoRA")
    tiers += _block("SFT", "native", "9B", "全量")
    tiers += _block("SFT", "ms-swift", "9B", "全量")
    tiers += _block("SFT", "native", "27B", "LoRA")
    tiers += _block("SFT", "ms-swift", "27B", "LoRA")
    # GRASPO 18 = 与 SFT 同构
    tiers += _block("GRASPO", "native", "9B", "LoRA")
    tiers += _block("GRASPO", "ms-swift", "9B", "LoRA")
    tiers += _block("GRASPO", "native", "9B", "全量")
    tiers += _block("GRASPO", "ms-swift", "9B", "全量")
    tiers += _block("GRASPO", "native", "27B", "LoRA")
    tiers += _block("GRASPO", "ms-swift", "27B", "LoRA")
    # OPD 9 = 与 CPT 同构
    tiers += _block("OPD", "ms-swift", "9B", "LoRA")
    tiers += _block("OPD", "ms-swift", "9B", "全量")
    tiers += _block("OPD", "ms-swift", "27B", "LoRA")

    for index, tier in enumerate(tiers, start=1):
        tier["tier_id"] = f"T{index:03d}"
        tier["gpus"] = list(GPU_SETS[tier["cards"]])
    # 覆盖点（唯一一处）：不覆盖时只做范围守卫（默认值也必须合法）；覆盖时改成映射里的卡。
    apply_gpu_override(tiers, gpu_override)
    return tiers


def classify_expressibility(tier: dict[str, Any]) -> tuple[str, str | None]:
    """判定该档当前能否被配置模型表达，返回 (status, reason)。

    - ``ready``：配置可表达，常规路径（LoRA）；
    - ``unverified``：配置可表达，但对应能力格当前 ⚠️ 未验证（全量）；
    - ``blocked``：配置模型/映射当前**无法表达**；或虽可表达，但按当前口径
      估算给不出可靠配方（如 native 全参 1 卡，见 ``BLOCKED_REASONS`` 的
      诚实表述——那是**保守判定，不是已证的必然 OOM**）。
    - ``not_applicable``：**确实不适用**——用户已拍板的口径下**不存在**该档的合法配置
      （现有唯一来源：台账的 27B OPD 档 `T052`–`T054`，见 :func:`opd_not_applicable`）。
      这是**合法终态**，不是 ``blocked``（"还没做"），也不产出可跑配置。

    **CPT / OPD 已于 2026-09-19（`WP-X3`）解除 ``blocked``**：`train_method` 枚举
    含 `cpt`/`opd`、各有专用注册表与 ms-swift 映射（pretrain / GKD）。
    它们按**模式**落 ``ready``（LoRA）/ ``unverified``（全量），与 SFT/GRASPO 同口径。

    **注意 ``可表达`` ≠ ``跑通``，也 ≠ ``显存可行``**：显存那一维由
    :func:`feasibility_verdict` 的三态判定给（CPT/OPD 的通道可行性见 ``CPT_NOTE`` /
    ``OPD_NOTE``；OPD 因教师侧未测算恒为 ``unmeasured``，但 ``not_applicable`` 档
    连"待测量"都不成立——它们不适用）。
    """
    # ★ 先判"确实不适用"：一个**不存在合法形态**的档，不该走后面任何一条"能跑/待测"的路径。
    not_applicable, na_reason = opd_not_applicable(tier)
    if not_applicable:
        return "not_applicable", na_reason
    # native 全参只有 PP 分片这一种手段：1 卡 ⇒ 无分片（无 offload）⇒ 保守判 blocked。
    # 理由不写「必然 OOM」——见 BLOCKED_REASONS["native_full_1card"] 的诚实表述。
    if tier["mode"] == "全量" and tier["backend"] == "native" and int(tier["cards"]) == 1:
        return "blocked", BLOCKED_REASONS["native_full_1card"]
    if tier["mode"] == "全量":
        return "unverified", FULL_MODE_NOTE
    return "ready", None


# ── 配置骨架 ────────────────────────────────────────────────────────────────


def _learning_rate(algorithm: str) -> float:
    return 5.0e-5 if algorithm == "SFT" else 5.0e-6


def subset_path(tier_id: str) -> str:
    """该档训练子集在容器内的固定路径（runner 负责按 manifest 生成）。"""
    return f"{ELAM_SUBSET_DIR}/{tier_id}.jsonl"


def subset_size(algorithm: str) -> int:
    """该档训练子集大小（§6 门槛下限；省资源不整集跑）。"""
    return SUBSET_SIZE_BY_ALGORITHM[algorithm]


# ── 训练子集的取数来源（mini 默认 / full 可切回）──────────────────────────────
#: 合法取值：`mini`（默认，跑通用） / `full`（源数据 `data/train.jsonl`，对照与回退用）。
TRAIN_SOURCE_MINI = "mini"
TRAIN_SOURCE_FULL = "full"
TRAIN_SOURCES: tuple[str, ...] = (TRAIN_SOURCE_MINI, TRAIN_SOURCE_FULL)

#: 跑通阶段**必须**说明的一句话：mini 不是效果口径。
MINI_PURPOSE_NOTE = (
    "mini 数据集只用于「矩阵跑通」（压时间/压成本），**效果评测必须用全量 test 集**"
    f"（{ELAM_TEST_JSONL}，{ELAM_TEST_COUNT} 行）；mini 的跑通结果**不得**当作效果结论。"
)


def resolve_train_source(train_source: str | None = None) -> str:
    """把 `None`（未显式给出）解析为默认来源 `mini`。

    默认行为必须**可被一句话说清**：不传参 ⇒ 生成「读 mini 数据集」的 runner。
    显式切回源数据：`--train-source full`（或对 runner 直接
    `export {ELAM_MINI_JSONL_ENV}=$ELAM_HOST/data/train.jsonl`，见 `render_runner` 注释）。
    """
    if train_source is None:
        return TRAIN_SOURCE_MINI
    if train_source not in TRAIN_SOURCES:
        raise ValueError(
            f"未知的 train_source={train_source!r}；只接受 {TRAIN_SOURCES}（默认 {TRAIN_SOURCE_MINI}）"
        )
    return train_source


def local_mini_jsonl_digest() -> tuple[str, int]:
    """实测算出开发机本地 mini 镜像的 `(sha256, 行数)`。

    **fail-closed**：拿不到摘要就拒绝生成——manifest 里记一份编造的摘要比不记更糟
    （§2.3 边界校验、§2.2 显式即防呆）。mini 与 228 上那份同源（已核 sha256 一致）。
    """
    if not LOCAL_MINI_JSONL.is_file():
        raise FileNotFoundError(
            f"mini 数据集本地镜像不存在：{LOCAL_MINI_JSONL}（生成期需要它算 sha256/行数）；"
            f"先跑 scripts/build_mini_elam_subset.py，或用 --train-source full 切回源数据"
        )
    digest = hashlib.sha256()
    lines = 0
    with LOCAL_MINI_JSONL.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
            lines += chunk.count(b"\n")
    return digest.hexdigest(), lines


def assert_mini_dataset_is_big_enough(line_count: int) -> None:
    """mini 必须装得下**最大**档位子集，否则跑通会在 `subset too small` 处假失败。"""
    largest = max(SUBSET_SIZE_BY_ALGORITHM.values())
    if line_count < largest:
        raise ValueError(
            f"mini 数据集只有 {line_count} 行 < 最大档位子集 {largest} 行 ⇒ "
            f"runner 会在 `FATAL: subset too small` 处失败（假失败，与被测能力无关）"
        )


def mini_dataset_manifest_fields(train_source: str) -> dict[str, Any]:
    """manifest `data` 段的**新增**字段：本档取数来源 + mini 摘要/行数。

    ⚠ 既有字段（`train_jsonl` / `test_jsonl` / `subset_dir` / `counts` …）与每档
    `data.train_path`(**生成位置**, 不是读取源) / `full_train_jsonl` **逐字不变**。
    """
    digest, lines = local_mini_jsonl_digest()
    assert_mini_dataset_is_big_enough(lines)
    return {
        "train_source": train_source,
        "train_source_env_var": ELAM_MINI_JSONL_ENV,
        "mini_dataset": {
            # ⚠ 键名不得含 "host"：`test_manifest_has_no_host_identity_and_declares_model_mount`
            #   只放行 `*_env_var` / `*_default` / `*_note` 三种含 host 的键名（那是不变量，
            #   不为本包放宽）。这里是"相对数据根的位置"，值本身不含任何宿主信息。
            "relative_path_under_data_root": ELAM_MINI_JSONL_RELATIVE,
            "jsonl_sha256": digest,
            "line_count": lines,
            "purpose": MINI_PURPOSE_NOTE,
            "isolation": (
                "文件名带 `mini-short-mm-` 前缀且落独立目录 `mini-dataset/`，与源数据 "
                "`data/train.jsonl` **物理隔离**（用户硬约束：靠命名与路径防误用；"
                "源数据只读；禁止混放同名文件）"
            ),
        },
        "train_path_not_the_source_note": (
            "每档 `data.train_path` 只约束**生成位置**（<RUN_ROOT>/<Tier>/subsets/<Tier>.jsonl），"
            "**不约束读取源**；runner 的 `TRAIN_PATH == dirname(TRAIN_PATH)/<Tier>.jsonl` 断言与"
            "图像根反推 `dirname(dirname(train_path))/images` 都依赖它 ⇒ **不要改配置来换数据源**，"
            "换源只切本段 `train_source`"
        ),
    }


def _repeats_dataset_by_rollout_group(tier: dict[str, Any]) -> bool:
    """该档是否走 ms-swift 的 **RLHF/GRPO** 路径（= 唯一会把数据集按 G 重复的路径）。

    **逐路径核实，不做一刀切**（2026-09-20，真机证据见下）：
      · **ms-swift**：``flow/msswift/_config_mapping.py`` 只有 ``stage == "rlhf"`` 分支
        下发 ``--num_generations`` / ``--generation_batch_size`` / ``--steps_per_generation``，
        该 stage 只被 ``msswift/trainer.py``（GRASPO 训练器）使用
        （``sft_trainer`` / ``cpt_trainer`` / ``opd_trainer`` 各传
        ``"sft"`` / ``"cpt"`` / ``"opd"``）
        ⇒ 判据 = ``train_method == "graspo"`` **且 backend 走 ms-swift**。
      · **native 的 GRASPO 档不重复**：``flow/trainer/trainer.py`` 按
        ``rollout_queue_batch_size × rollout_group_size`` 归组，**每个 optimizer step
        消费固定条数 prompt**；真机 ``T028``（native/9B/1卡）的 ``events.jsonl`` 逐字：
        ``samples_seen`` 8 → 16、``samples_total`` 20、``optimizer_steps_per_rank`` 1
        ⇒ 一个 epoch 只有约 3 步。**给 native 档乘 G 会高 8 倍**，故必须排除。
      · **SFT / CPT / OPD(GKD) 不重复**：各 stage 分支都不下发 ``--num_generations``；
        真机逐档相符（``T025``/``T026``/``T027`` = 100/50/25，``T047``/``T048`` = 10/5）。
    """
    return (
        str(tier["backend"]) == "ms-swift"
        and ALGORITHM_TO_TRAIN_METHOD[str(tier["algorithm"])] == "graspo"
    )


def _is_native_graspo(tier: dict[str, Any]) -> bool:
    """该档是否走 **native 训练器的 GRASPO** 路径。

    判据含 **backend**（不只是算法），依据与 :func:`_repeats_dataset_by_rollout_group`
    同源：``flow/trainer/trainer.py`` 是 native 侧唯一的 GRASPO 训练器，它**不重复数据集**，
    而是按 ``rollout_queue_batch_size × rollout_group_size`` 归组、**阈值触发才出一步**
    （见 :func:`native_graspo_steps_per_epoch`）。ms-swift 侧走的是另一条路径
    （``RepeatSampler`` 重复数据集），**不得**共用本判据。
    """
    return (
        str(tier["backend"]) == "native"
        and ALGORITHM_TO_TRAIN_METHOD[str(tier["algorithm"])] == "graspo"
    )


# ── 步数口径：**两个量，各自只有一处真相源**（AF1 §⑥ 方案 S1，2026-09-21）──
#
# 为什么必须分开：这两个量服务**两件不同的事**，混成一个字段就必然有一件事是错的。
#
#   · 「计划步数」:func:`expected_optimizer_steps` —— 给 **ckpt 保留策略**用
#     （``training.save_steps`` ⇒ 每档只落一份 ckpt）。它回答"我打算隔多少步存一次",
#     **不回答**"这一档最多能跑出几步"。
#   · 「可产出步数上限」:func:`expected_optimizer_steps_reachable` —— 给**门槛判定**
#     与**披露**用。它必须按该档**训练器的真实步进语义**算，否则会把"结构上跑不出
#     这么多步"误报成"这档训练失败"。
#
# 实测依据（AF1 诊断，`task-af1-step-gate/report.md`）：9 档 ``GRASPO × native``
# 用旧的朴素均分（``subset_size ÷ cards``）声明步数，而 native 的真实步进由
# **replay 阈值（``queue×group``）+ 每 epoch 一次的 force flush** 决定 ⇒
# ``T028/T029/T030`` 声明 20/10/5，实际上限只有 3/2/1/epoch。真机锚点：``T030``
# 自然跑完 ``exit_code=0``、``optimizer_steps=1``、``epoch_summary`` 为
# ``samples_seen=5 / completions=40 / optimized_steps=0``（40 < 64 ⇒ 阈值 0 次触发，
# 唯一一步来自 epoch 末 force flush）。
# 其余 45 档两条口径**逐档一致**（AF1 已核 + 本包机核，见工位报告 §④）。


#: 矩阵档位的 ``training.max_epochs``（**唯一真相源**；:func:`build_config` 与
#: :func:`expected_optimizer_steps_reachable` 都读它，口径不可能分叉）。
#: 为什么是 1：矩阵跑通阶段按"每档只跑一个 epoch"定预算（AF1 §④：本问题在
#: ``max_epochs=1`` 下才有那 9 档的不自洽；AF1 方案 S5 若要改 epoch 数须用户拍板）。
MATRIX_MAX_EPOCHS = 1


def native_graspo_steps_per_epoch(tier: dict[str, Any]) -> int:
    """``GRASPO × native`` 档**每 epoch 可产出的 optimizer 步数上限**。

    公式（AF1 §② 推导，逐条都有代码依据）::

        每 rank prompt 数 P = floor(subset_size / cards)        # DP 分片
        阈值触发步数        = floor(P / rollout_queue_batch_size)
        epoch 末 force flush = +1                               # 每 epoch 恰好一次
        ⇒ 每 epoch 上限 = floor(P / Q) + 1

    代码依据（**函数名/字段名为锚，行号会漂移**）：

      · DP 分片：``flow/trainer/trainer.py`` 的
        ``epoch_samples = epoch_samples[adapter.dp_rank :: adapter.dp_size]``；
      · 每条 prompt 追加 ``G`` 条 experience、且非 trainable 的 prompt **不进** buffer：
        ``flow/trainer/rollout.py`` 的 ``_append_experiences`` / 三条 early-return；
      · 阈值 = ``rollout_queue_batch_size × rollout_group_size``：
        ``schema.py`` 的 ``replay_buffer_optimize_threshold``，消费在
        ``flow/trainer/optimize.py`` 的 ``len(replay_buffer) >= threshold``
        ⇒ 触发 ``floor(P×G / (Q×G)) = floor(P/Q)`` 次；
      · epoch 末无条件 force：``flow/trainer/trainer.py`` 的
        ``self._maybe_optimize(epoch=..., force=True)``；``optimize.py`` 的
        ``local_wants_train = (force and len(replay_buffer) > 0) or ...``
        ⇒ **只要 buffer 非空就恰好 +1 步**（与 buffer 里剩多少条无关）。

    ★ **这是上限，不是保证**：若某些 prompt 被判 ``retry``/``invalid``/``perfect_skip``
    （不进 buffer），实际步数可能**更低**（AF1 §⑧-2）。门槛判定必须按"上限"比，
    否则会把"上限之下"的正常 run 判成异常。
    """
    cards = max(1, int(tier["cards"]))
    prompts_per_rank = subset_size(str(tier["algorithm"])) // cards
    return prompts_per_rank // NATIVE_ROLLOUT_QUEUE_BATCH_SIZE + 1


def expected_optimizer_steps_per_epoch(tier: dict[str, Any]) -> int:
    """该档**每 epoch** 可产出的 optimizer 步数上限（三条路径**显式分派**，不搞一刀切）。

    ====================  =========================================  ==============
    路径                  每 epoch 上限                               真机校准
    ====================  =========================================  ==============
    SFT / CPT / OPD      ``floor(subset_size / cards)``              T010/11/12
    GRASPO × ms-swift    ``floor(subset_size × G / cards)``          T031/32/33
    GRASPO × native      ``native_graspo_steps_per_epoch``（阈值+flush）T028/T030
    ====================  =========================================  ==============

    前两条与 :func:`_planned_save_steps` **同式同值**（下面的实现**真的调用**
    同一段计算，不是抄一遍公式 ⇒ §1.4 单一真相源）；只有 native GRASPO 走第三条
    专用公式。**本函数与 :func:`expected_optimizer_steps`（计划值）是两个量**：
    它们在 45 档上逐档相等，在 native GRASPO 那 9 档上**刻意不等**（计划值偏大，
    见 :func:`checkpoint_save_steps`）——`save_steps` 走计划值，门槛判定走本函数。
    """
    if _is_native_graspo(tier):
        return max(1, native_graspo_steps_per_epoch(tier))
    return expected_optimizer_steps(tier)


def expected_optimizer_steps_reachable(tier: dict[str, Any]) -> int:
    """该档**整个 run 可产出的 optimizer 步数上限** = 每 epoch 上限 × ``max_epochs``。

    ``max_epochs`` 由 :data:`MATRIX_MAX_EPOCHS` **唯一决定**（所有档统一，见
    :func:`build_config` 的 ``training.max_epochs``）——**这里不另造第二个 epoch 真相源**。

    这个量是**门槛判定**的输入（``acceptance.formal_gate.expected_optimizer_steps_reachable``
    ⇒ ``result_judge.RunEvidence.expected_optimizer_steps_reachable``）：
    当 ``reachable < min_optimizer_steps`` 时，"步数少于门槛"这件事**结构上必然发生**，
    判据必须把它与"真的没训练"分开（见 ``result_judge`` 的
    :data:`~graspo.core.result_judge.STEP_GATE_NOT_APPLICABLE_MARKER`）。
    """
    return expected_optimizer_steps_per_epoch(tier) * MATRIX_MAX_EPOCHS


# ── 生成期对照（**fail-closed**，§2.3 边界校验即防呆）──────────────────────


def step_gate_applicability(
    tier: dict[str, Any], *, min_optimizer_steps: int
) -> tuple[str, str]:
    """该档的 **step 门槛是否适用** ⇒ ``(标记, 理由)``，写进 manifest 的
    ``acceptance.formal_gate.step_gate_applicability``。

    三值（**显式**，§2.2 显式即防呆；不用布尔是因为"未知"与"不适用"必须可区分）：

    ==============================  ========================================
    标记                             含义
    ==============================  ========================================
    ``applicable``                   ``reachable >= min_optimizer_steps``
                                     ⇒ 门槛**照常判**（现有判据一字不改）
    ``not_applicable_structural``    ``reachable < min_optimizer_steps`` 且
                                     ``reachable >= 1`` ⇒ 该档**无论跑多久都达不到
                                     门槛**，判据返回"⚠ 口径不可测"（**不是** ❌ 失败，
                                     也**不是** ✅ 通过）
    ``review_required``              ``reachable < 1``：**生成器不会产出这种档**
                                     （见 :func:`assert_step_gate_is_satisfiable`），
                                     出现即说明生成期防线被绕过 ⇒ 判据 fail-closed
    ==============================  ========================================

    ★ 这个字段**只标记事实，不放宽任何要求**：门槛值 ``min_optimizer_steps`` 一个都不动，
    ``reachable >= 门槛`` 的档仍然按原判据一字不改地判。
    """
    reachable = expected_optimizer_steps_reachable(tier)
    if reachable < 1:
        return STEP_GATE_REVIEW_REQUIRED, f"可产出步数上限={reachable} < 1（非法，生成期防线被绕过）"
    if reachable < min_optimizer_steps:
        note = (
            f"该档可产出步数上限={reachable} < 门槛 {min_optimizer_steps}"
            f"（{'结构上不可能达标' if MATRIX_MAX_EPOCHS == 1 else '在 max_epochs=' + str(MATRIX_MAX_EPOCHS) + ' 下达不到'}）"
            "⇒ 步数门槛对本题不适用，判「⚠ 口径不可测」而不是「❌ 失败」"
        )
        return STEP_GATE_NOT_APPLICABLE_STRUCTURAL, note
    return STEP_GATE_APPLICABLE, (
        f"可产出步数上限={reachable} ≥ 门槛 {min_optimizer_steps} ⇒ 门槛照常判"
        + (
            ""
            if reachable >= min_optimizer_steps * 2
            else f"（余量不足：上限 {reachable} 仅比门槛高 {reachable - min_optimizer_steps} 步）"
        )
    )


#: step 门槛适用性的三值（唯一真相源；判定侧按**同一字符串**识别，见
#: ``result_judge.STEP_GATE_NOT_APPLICABLE_MARKER``）。
STEP_GATE_APPLICABLE = "applicable"
STEP_GATE_NOT_APPLICABLE_STRUCTURAL = "not_applicable_structural"
STEP_GATE_REVIEW_REQUIRED = "review_required"

#: **正式记录门槛的 optimizer step 下限**（``docs/capability-matrix.md`` §6「正式记录门槛」：
#: 每档 ≥1 完整 epoch 且 ≥5 optimizer step）。
#: ★ **AF1/指挥官裁定：这个值一个都不动**（2026-09-21）。它不是"按档折算"的软门槛——
#: 它要拦的是"压根没训练"这个实质问题；结构上跑不到它的档走**显式的第三态**
#: （``step_gate_applicability == not_applicable_structural``），**不是**把门槛降下来。
#: 判定侧的缺省值真相源是 ``result_judge.MIN_OPTIMIZER_STEPS``（同值 5）；本常量只保证
#: 生成器写进 manifest 的值与"门槛行"一致（防"文档说 5、清单写 1"这类分裂）。
FORMAL_GATE_MIN_OPTIMIZER_STEPS = 5


def assert_step_gate_is_recorded(tier: dict[str, Any], *, min_optimizer_steps: int) -> None:
    """**生成期 fail-closed 断言**（§2.3）：结构上不可能达标的档**必须**被显式标记。

    它挡的是这个具体事故：一份"静默"的 manifest 把 ``T030``（4 卡 native GRASPO，
    上限 1 步）与 ``T010``（SFT 1 卡，上限 100 步）在门槛面前**写成同一回事** ⇒
    下一批跑完，T030 被 A2 判 ❌，读者读成"4 卡 native GRASPO 能力不行"。
    （AF1 诊断的正是这个"即将发生的误判"。）

    为什么要有它而不是"靠人记得看报告"：防呆设计的原则是**让人根本没法犯错**——
    生成器如果产出一份"上限 < 门槛、却标 applicable"的 manifest，必须**当场炸**，
    而不是等一批 GPU 跑完再被台账误读（§2.3 边界校验即防呆）。
    """
    marker, note = step_gate_applicability(tier, min_optimizer_steps=min_optimizer_steps)
    if marker == STEP_GATE_REVIEW_REQUIRED:
        raise ValueError(
            f"档 {tier['tier_id']}：可产出步数上限 "
            f"{expected_optimizer_steps_reachable(tier)} < 1 —— 生成器不得产出这种档"
            f"（{note}）"
        )


# ── ckpt 保留策略（2026-09-19，用户已授权；单一真相源）────────────────────────


def _planned_save_steps(tier: dict[str, Any]) -> int:
    """该档的**计划保存间隔**（= 计划总优化步数，朴素均分口径）——**唯一定义处**。

    ``= max(1, 实际参与优化的样本数 ÷ 卡数)``，**向下取整**。

    ★ **RLHF（GRASPO/GRPO）档的总样本数 = 子集 × ``rollout_group_size``**：
    ms-swift 的 ``RepeatSampler(mini_repeat_count=G)`` 会把每条 prompt **重复 G 次**
    再进训练循环（``G`` = ``training.rollout_group_size``，默认
    :data:`ROLLOUT_GROUP_SIZE`；矩阵档位配置**不覆盖**它）。
    **2026-09-20 修复**：本函数原先漏掉这一乘数 ⇒ GRASPO 档的"计划步数"低 ``G`` 倍
    （``T031``/``T043`` 声明 20，真机 ``global_step/max_steps`` 跑到 **160/160** = 20 × 8）。

    **哪些档要乘**（**不搞一刀切**，逐路径核实）：只有走 ms-swift
    ``stage == "rlhf"`` 的档才重复数据集——见
    :func:`_repeats_dataset_by_rollout_group`（判据含 **backend**）。
    **不乘**的三类都有真机证据：
      · SFT / CPT / OPD(GKD)：``T025``(1卡)=100、``T026``(2卡)=50、``T027``(4卡)=25、
        ``T047``(2卡)=10、``T048``(4卡)=5；
      · **native 的 GRASPO 档**（``T028``–``T030`` / ``T034``–``T036`` / ``T040``–``T042``，
        共 9 档）：native 训练器按 ``rollout_queue_batch_size × rollout_group_size`` 归组、
        **每个 optimizer step 消费固定条数 prompt**（``T028`` 的 ``events.jsonl`` 逐字：
        ``samples_seen`` 8 → 16、``samples_total`` 20、``optimizer_steps_per_rank`` 1
        ⇒ 一个 epoch 约 3 步）⇒ 给它们乘 G 会**高 8 倍**。
        （★ native 档的声明值因此仍偏大：它不读 ``save_steps``、只落 ``final/`` ⇒
        该项对 native 是**纯信息字段**；是否另修见工位报告"额外发现"。）

    **实测校准（ms-swift GRASPO，1 卡）**：``T031``/``T043`` 真机跑到 160/160，且保留块日志
    显示真落了 8 份 ckpt（``checkpoint-{20,40,…,160}``）⇒ 修之前 ``save_steps``=20
    **每个 run 落 8 份**，与"每档只落一份"的设计相悖，靠容器内保留块删掉 7 份。
    （**2/4 卡 ms-swift GRASPO 档无真机实测**——站内不存在这类 run；80 / 40 是按
    ``总样本 ÷ 卡数`` 摊分的**推断**，见工位报告"未核实"登记。）

    依据（**实测口径，不是猜测**）：矩阵档每卡 micro batch = 1、
    ``gradient_accumulation_micro_batches`` = 1 ⇒ 每卡每步 1 个样本、全局步数 = 每卡步数。
    上机实测对得上：``T013``（1 卡/100 条）= 100 步、``T014``（2 卡/100 条）= 50 步、
    ``T015``（4 卡/100 条）= 25 步（``task-r3-retest2`` 的 ``Train: N%|k/k`` 与
    ``last_model_checkpoint: checkpoint-<k>``）。

    **取向下取整**（不是向上）：它保证 ``save_steps ≤ 实际步数`` ⇒ **至少落一份 ckpt**。
    若取向上而实际步数不足（数据末端被 drop 的情形），就会一份都不落 ⇒ A3 无证据可判。

    ★ **本函数是"计划值"，不是"可达上限"**：它按朴素均分算出"打算隔多少步存一次"，
    **不回答**"这一档最多能跑出几步"——后者的真相源是
    :func:`expected_optimizer_steps_reachable`。两者**只在 native GRASPO 那 9 档上不等**，
    且那是**刻意**的（见 :func:`checkpoint_save_steps` 的 docstring：native 的中间段落在
    ``step_<N>``，保留块只清 ``checkpoint-*`` ⇒ 计划值偏大反而**正是**"每档只留一份"）。

    ★ **单一真相源（§1.4）**：本函数是"样本数 ÷ 卡数（×G）"这条朴素算式的**唯一定义处**；
    :func:`expected_optimizer_steps` 与 :func:`expected_optimizer_steps_per_epoch`
    的非 native 分支都**调用本函数**，不另抄一遍公式。
    """
    cards = max(1, int(tier["cards"]))
    samples = subset_size(str(tier["algorithm"]))
    if _repeats_dataset_by_rollout_group(tier):
        samples *= ROLLOUT_GROUP_SIZE
    return max(1, samples // cards)


def expected_optimizer_steps(tier: dict[str, Any]) -> int:
    """该档的**计划步数**（清单字段 ``tiers[*].expected_optimizer_steps``）。

    = :func:`_planned_save_steps`（朴素均分，向下取整）。它是**历史字段**，与 ckpt 保留策略
    的配置侧同源；**没有任何判据消费它**——门槛判定读的是
    :func:`expected_optimizer_steps_reachable` / :func:`expected_optimizer_steps_per_epoch`
    （经 ``scripts/collect_results.py`` 的 ``extract_expected_optimizer_steps``）。

    ★ **两个量，各只有一个真相源**（AF1 §⑥ 方案 S1）：

    ============================  ====================================  ==================
    量                             含义                                  消费者
    ============================  ====================================  ==================
    ``expected_optimizer_steps``   计划步数（朴素均分，可偏大）           仅清单披露
    ``..._per_epoch``              每 epoch 可产出步数上限（真实调度）    门槛口径 / 披露
    ``..._reachable``              整个 run 可产出步数上限                A2 第三态
    ============================  ====================================  ==================

    ★ **对 native GRASPO 的 9 档，本函数刻意保留朴素均分的偏大值**（``T028``=20、``T030``=5，
    而可达上限只有 3 / 1）：native 的中间段落在 ``step_<N>``（``optimize.py``），容器内保留块
    只清 ``checkpoint-*``、对 native 只承认 ``final/``（``generate_matrix.py`` 的
    ``retain_single_checkpoint``）⇒ ``save_steps`` 偏大 ⇒ **不触发中间段** ⇒ 恰好只留
    ``final/`` 一份。反之把计划值压到可达上限会真落 ``step_<N>`` + ``final/`` = **2 份重 ckpt**
    （``T034/T035/T036`` 是全参 9B，代价最大）——那是缺陷，不是修复。
    （2026-09-21 指挥官裁定：回退 ``5c6bde0`` 把计划值改成可达上限的那一改；依据
    ``task-ah1-savesteps-lock`` §③/§⑤-5.2：保留策略**显式承认** ``final/``，旧值无害。）
    """
    return _planned_save_steps(tier)


def checkpoint_save_steps(tier: dict[str, Any]) -> int:
    """该档的 ``training.save_steps``：**计划保存间隔**（⇒ 每档只落一份 ckpt）。

    ★ **2026-09-21 解耦**（指挥官裁定，`task-ah1-savesteps-lock` §⑤-5.2）：本函数**不再**
    ``return expected_optimizer_steps(tier)``，而是**直接**走朴素均分口径
    :func:`_planned_save_steps`。理由：``expected_optimizer_steps`` 一族带有
    ``_is_native_graspo`` 的**A2 侧分派**（native 按 replay 阈值 + force flush 算真实调度），
    而"保存间隔"是**保留策略侧**的独立问题；两者绑在一起时，任何一次对 A2 口径的修正都会
    顺手改掉 ``save_steps``——``5c6bde0`` 就是这么把 8 档 ``save_steps`` 从 20/10/5
    改成 3/2/1 的。解耦后**保存间隔只由计划口径决定，永不随 A2 分派漂移**。

    ★ 这里的"计划步数"含 `rollout_group_size` 乘数（2026-09-20 修复，见
    :func:`_planned_save_steps`）：修之前 GRASPO 档拿到的 `save_steps` 只有真总步数的 1/8
    ⇒ 每个 run 真落 **8 份** ckpt（`T031`/`T043` 实测 `checkpoint-{20..160}`），"每档只落一份"
    实际上全靠容器内保留块删掉 7 份。修后 ms-swift GRASPO 的 `save_steps` = 真总步数
    ⇒ 保存恰好发生在最后一步（1 卡 GRASPO = 160）。

    ★ **native GRASPO 那 9 档是刻意的例外**：它们的计划值（``T028``=20、``T030``=5）**大于**
    可达上限（3 / 1）⇒ ``save_steps`` **永不触发** ⇒ native 不落中间段、只落终态 ``final/``。
    这正是保留策略要的形态：容器内保留块**只清 ``checkpoint-*``**，而 native 的中间段叫
    ``step_<N>``（``flow/trainer/optimize.py``）——一旦计划值压到可达上限，每档会真落
    ``step_<N>`` + ``final/`` = **2 份重 ckpt**，与 ``keep_per_tier = 1`` 冲突，且
    ``T034/T035/T036`` 是全参 9B，代价最大。⇒ 计划值偏大**不是缺陷**（旧值无害）。

    ★ 为什么是"总步数"而不是 0/负数：``save_steps <= 0`` 在映射层会落到
    ``save_strategy: epoch`` 分支（``msswift/_config_mapping.py``），那是**另一种落盘形态**，
    不是"不落盘"。要"每档只落一份"，正确做法是让保存间隔等于总步数。

    ★ **A3 判据不受影响**：A3 = "checkpoint 能否被重新加载"，判据实现
    （``scripts/collect_results.py`` 的 ``find_checkpoint_dirs()`` /
    ``extract_checkpoint_reloadable()``）只要求**存在至少一份**可重载 ckpt；native 的那一份是
    ``final/``（``NATIVE_CHECKPOINT_DIRNAME = "final"``，``rglob("final")``），
    保留块也**显式承认**非空 ``final/``（``KEEP …（native 终态产物）`` + ``保留 ckpt 数=1``；
    真机锚点 ``task-e1-smoke-t010`` 的 ``KEEP /out/T010/final`` + ``ckpt_retention.state=ok``）
    ——**没有**"必须有多少份"的要求。所以"只落 ``final/``"完全满足 A3/A3 保留策略；
    而"每步一份"浪费的是磁盘，不是判据。
    """
    return _planned_save_steps(tier)


# ── 显存可行性模型（生成期断言，§2.3 边界校验即防呆）────────────────────────
# **为什么要有这个**：配置能生成 ≠ 能跑。上一轮的坑就是"安静地产出一个跑起来才爆的
# 配置"，把错误推迟到花掉 GPU 时间之后才暴露。这里在生成期用保守算式估算每卡需求，
# 超过单卡可用显存就直接拒绝生成并报原因。
#
# 算式（保守，单位 bytes/param；9B=9.0e9，27B=27.0e9）：
#   - bf16 权重          2  （两个后端一致）
#   - bf16 梯度          2  （两个后端一致）
#   - 优化器态(m+v)      fp32=8 / bf16=4 —— **按后端分别假设，见下**
#   - LoRA：基座冻结 ⇒ 只算权重 2 B/param（适配器参数量忽略，远小于基座）
#   - ZeRO-2：梯度 + 优化器按卡数摊 ⇒ 2 + (2+优化器)/N
#   - ZeRO-2 + CPU offload：优化器态在宿主内存 ⇒ 2 + 2（梯度全量）
#   - AutoTP=T：权重再按 T 摊 ⇒ 权重 2/T
#   - native 全参：只支持 PP 分片 ⇒ 权重/梯度/优化器都按 PP=卡数 摊 ⇒ k/N
# 激活值用一个名义常量（上下文长度由递增加长法实测得出、不预设档位）。
PARAMS_BILLION = {"9B": 9.0, "27B": 27.0}
_BF16 = 2.0
_GRAD = 2.0

#: ── 常驻口径「单一真相源」（§1.4）：**按后端分别假设，不混用一个 k** ──────────
#: 此前 native 与 ms-swift 共用 k=12（fp32 优化器态），于是 native 1 卡被算成
#: 112.6 GiB ⇒ 写成「必然 OOM」；而 native 侧的实际实现口径是 k=8（bf16 Adam，
#: 见 task-i2-fullparam §3.1：`torch.optim.AdamW` 用 `zeros_like(param)` 建 m/v，
#: dtype 跟随 bf16 参数 ⇒ 常数 70.1 GiB、余量 ≈10 GiB）。两处口径不一致，且
#: **两侧的 k 都是假设、不是实测**。本生成器现在的统一口径是：
#:   - native 全参  : k = 8  B/param（bf16 Adam 假设，来源 task-i2-fullparam §3.1）
#:   - ms-swift 全参: k = 12 B/param（DeepSpeed fp32 优化器态假设，保守）
#:   - LoRA         : 2 B/param（基座冻结）
#: 每个假设都写进 manifest 的 `feasibility_model.assumptions`，便于事后核账。
_ADAM_FP32 = 8.0
_ADAM_BF16 = 4.0
#: native 侧实际实现口径（bf16 矩估计）。**这是假设**：若工程上改为 fp32 矩估计
#: （k=12），native 1 卡的估算会升到 105 GiB+（算式级必爆）。
NATIVE_OPTIMIZER_BYTES_PER_PARAM = _BF16 + _GRAD + _ADAM_BF16
#: ms-swift / DeepSpeed 侧保守口径（fp32 优化器态）。
MSSWIFT_OPTIMIZER_BYTES_PER_PARAM = _BF16 + _GRAD + _ADAM_FP32
ACTIVATION_ALLOWANCE_GIB = 12.0
#: 单卡 80 GiB，留出余量后允许的上限。
CARD_BUDGET_GIB = 78.0

NATIVE_ACCOUNTING_ASSUMPTION = (
    "native 全参 k=8 B/param —— 假设 `torch.optim.AdamW` 的 m/v 矩估计跟随 bf16 参数"
    "（`zeros_like(param)`，无 fp32 master）。来源：task-i2-fullparam §3.1。"
    "**该假设未经上机实测**：若改为 fp32 矩估计（k=12）则 1 卡估算升至 105 GiB+。"
)
MSSWIFT_ACCOUNTING_ASSUMPTION = (
    "ms-swift/DeepSpeed 全参 k=12 B/param —— 保守假设 fp32 优化器态（m+v = 8）。"
    "**该假设未经上机实测，方向偏保守。**"
)


def _native_static_gib(params: float) -> float:
    """native 全参 1 卡（PP=1，无分片）的常驻显存估算（GiB）。"""
    return params * NATIVE_OPTIMIZER_BYTES_PER_PARAM / (1024.0**3)


#: 当前不可表达的原因（每条都指向能力矩阵里的 ⚠️ 格）。
#: native 1 卡那条**不写「必然 OOM」**——算式依赖未经实测的 bf16 Adam 假设，
#: 且余量只有约 10 GiB，只能如实说「判定不可靠、本轮不承诺」。
#:
#: ★ ``cpt`` / ``opd`` 两条已于 2026-09-19 删除（`WP-X3` 落地了配置通道）：
#: 旧文案「`train_method` 的 Literal 只有 {'graspo','sft'}」**已是过期事实**，
#: 保留它就会把已解决的问题永远钉在「不可表达」上。留痕见 ``CPT_NOTE`` / ``OPD_NOTE``。
BLOCKED_REASONS: dict[str, str] = {
    "native_full_1card": (
        "native 全参当前只支持 PP 分片、且无 offload ⇒ 1 卡无任何分片手段。"
        f"按 native 实际实现口径（k={NATIVE_OPTIMIZER_BYTES_PER_PARAM:g} B/param，"
        "bf16 Adam 假设）9B 常驻 ≈"
        f"{_native_static_gib(PARAMS_BILLION['9B'] * 1.0e9):.1f} GiB，"
        f"加激活/其他预算 {ACTIVATION_ALLOWANCE_GIB:g} GiB ⇒ ≈"
        f"{_native_static_gib(PARAMS_BILLION['9B'] * 1.0e9) + ACTIVATION_ALLOWANCE_GIB:.1f} GiB"
        f" > 预算 {CARD_BUDGET_GIB:g} GiB，余量仅约 10 GiB。"
        "**该结论依赖 bf16 Adam 假设、未经上机实测，判定不可靠 ⇒ 本轮不承诺**；"
        "若工程上改为 fp32 矩估计则算式级必爆。"
        "需 native 侧实现全参 offload、改用 ≥2 卡走 PP、或先用 1 卡边界取证档实测后再定。"
    ),
}


def ms_swift_full_recipe(cards: int) -> dict[str, Any]:
    """ms-swift 全参的可行配方（≤4 卡；依据全量入口工作包的方案）。

    - 1 / 2 卡 → ``zero2_offload``（优化器态换出到 CPU，否则单卡常驻超预算）
    - 4 卡 → ``zero2`` + ``deepspeed_autotp_size: 4``（ZeRO-2 + AutoTP 权重分片）
    """
    if cards <= 2:
        return {"deepspeed": "zero2_offload"}
    return {"deepspeed": "zero2", "deepspeed_autotp_size": 4}


def estimate_config_per_card_gib(tier: dict[str, Any], config: dict[str, Any]) -> tuple[float, str]:
    """按**实际生成的配置**估算每卡显存需求，返回 (GiB, 算式依据)。

    校验的是"我们真正要产出的配置"而不是"我们以为的配方"——否则生成器漏配
    deepspeed 时断言照样会通过（这正是上一轮的坑）。
    """
    params = PARAMS_BILLION[str(tier["model"])] * 1.0e9
    cards = int(tier["cards"])
    backend = str(tier["backend"])
    mode = str(tier["mode"])

    if mode == "LoRA":
        bytes_per_param = _BF16
        basis = f"LoRA：基座冻结 ⇒ 权重 {_BF16:g} B/param"
    elif backend == "native":
        pp_size = int(config.get("native", {}).get("pp_size", 1) or 1)
        bytes_per_param = NATIVE_OPTIMIZER_BYTES_PER_PARAM / pp_size
        basis = (
            f"native 全参 PP 分片（k={NATIVE_OPTIMIZER_BYTES_PER_PARAM:g} B/param，"
            f"bf16 Adam 假设）："
            f"({_BF16:g}+{_GRAD:g}+{_ADAM_BF16:g})/{pp_size} B/param"
            + ("" if pp_size > 1 else "（无分片）")
        )
    else:
        msswift = config.get("msswift", {})
        deepspeed = msswift.get("deepspeed")
        autotp = float(msswift.get("deepspeed_autotp_size") or 1)
        if deepspeed == "zero2_offload":
            weight = _BF16 / autotp
            sharded = _GRAD  # 优化器态 offload 到 CPU，梯度仍全量
            basis = (
                f"ms-swift 全参 zero2_offload：权重 {_BF16:g}/{autotp:g}"
                f" + 梯度 {_GRAD:g} B/param（优化器态在 CPU）"
            )
        elif deepspeed in {"zero2", "zero3", "zero2_offload", "zero3_offload"}:
            weight = _BF16 / autotp
            sharded = (_GRAD + _ADAM_FP32) / cards
            basis = (
                f"ms-swift 全参 {deepspeed}+AutoTP{autotp:g}：权重 {weight:g}"
                f" + (梯度+优化器) {_GRAD + _ADAM_FP32:g}/{cards} B/param"
            )
        else:
            # 没配任何分片/offload：全参 ⇒ 权重+梯度+优化器全量压在一张卡上。
            weight = _BF16
            sharded = _GRAD + _ADAM_FP32
            basis = (
                f"ms-swift 全参**未配 deepspeed/offload**：权重 {_BF16:g} + 梯度 {_GRAD:g}"
                f" + 优化器 {_ADAM_FP32:g} = "
                f"{MSSWIFT_OPTIMIZER_BYTES_PER_PARAM:g} B/param 全量压单卡"
            )
        bytes_per_param = weight + sharded

    gib = params * bytes_per_param / (1024.0**3) + ACTIVATION_ALLOWANCE_GIB
    return gib, f"{basis}；激活/其他预算 {ACTIVATION_ALLOWANCE_GIB:g} GiB"


def estimate_per_card_gib(tier: dict[str, Any]) -> tuple[float, str]:
    """按该档的规范配置估算每卡显存需求。"""
    return estimate_config_per_card_gib(tier, build_config(tier))


#: 教师权重常驻量（GiB，bf16 权重 2 B/param）——**只用来说明"不换出为什么不行"**，
#: 不是教师侧显存需求（教师前向的工作集未测算，见 ``OPD_TEACHER_UNMEASURED_NOTE``）。
def _teacher_weights_gib() -> float:
    params = PARAMS_BILLION[OPD_TEACHER_MODEL] * 1.0e9
    return params * _BF16 / (1024.0**3)


OPD_TEACHER_UNMEASURED_NOTE = (
    "★ **待上机测量（P-25）**：OPD 的教师模型（Qwen3.8-27B）在训练进程内**额外占一份显存**"
    "（每个 rank 一份），本生成器**没有**教师侧的实测数字 ⇒ **不得**判定为「可行」。"
    "配方已显式设 `distill.offload_teacher_model: true`（教师权重常驻 CPU、前向时换入），"
    "但教师**前向时的 GPU 工作集**仍是未测算量 ⇒ 本档标 "
    f"`{VERDICT_UNMEASURED}`，由**上机冒烟**（建议先跑 `T046`）回填实测后再改判定。"
    "参考算式（教师**不**换出时）：27B bf16 权重 ≈ "
    f"{_teacher_weights_gib():.1f} GiB/卡，与学生侧常驻相加在任何卡数下都超预算 ⇒ "
    "「不换出」不构成可用配方（这是权重常驻的算术事实，不是实测结论）。"
)


def feasibility_verdict(tier: dict[str, Any]) -> tuple[str, str]:
    """该档的显存可行性**三态判定**（单一真相源，§1.4），返回 (verdict, detail)。

    判定顺序即语义（先确定性、后未测算）：

    1. ``not_applicable``：该档**确实不适用**（用户拍板口径下不存在合法配置）——
       不参与任何"可行性"判断（对不存在的东西问"装不装得下"没有意义）。
    2. ``blocked``：该档当前给不出配方（能力未落地 / 无分片手段）——由
       :func:`classify_expressibility` 决定，这里不做二次判断。
    3. ``infeasible``：**确定性算式**（学生侧常驻 + 配方里显式给出的分片/offload）
       已超单卡预算 ⇒ 生成期断言**拒绝生成**并报出算式。
    4. ``unmeasured``：确定性算式通过，但存在**未测算的额外显存消费者**
       （目前唯一来源：OPD 的教师）⇒ **不得写「可行」**，标「待上机测量」（P-25）。
    5. ``feasible``：确定性算式通过，且无未测算的额外消费者。

    **为什么第 3 步不能并进第 4 步**：把"没量过的教师"当成"不存在"，就是拿未实测的
    假设去支撑一个「可行」结论（跟踪文档 `D-23`：推定值不得入账）。
    **为什么不能并进第 2 步**：算式通过就说它"必然 OOM"，同样是把假设伪装成结论。
    """
    # ★ "确实不适用"最先判：它既不是 infeasible（没说它放不下），也不是 unmeasured
    #   （它连"待上机测量"都不成立——没有该测量的配置）。
    not_applicable, na_reason = opd_not_applicable(tier)
    if not_applicable:
        return VERDICT_NOT_APPLICABLE, na_reason or ""

    status, _ = classify_expressibility(tier)
    if status == "blocked":
        return VERDICT_BLOCKED, "blocked：当前给不出可行配方（原因见 status_reason）"

    gib, basis = estimate_per_card_gib(tier)
    within_budget = gib <= CARD_BUDGET_GIB
    verdict_text = "可行" if within_budget else "估算超预算 ⇒ 判定不可靠（假设未实测）"
    detail = f"估算每卡 {gib:.1f} GiB（预算 {CARD_BUDGET_GIB:g} GiB）⇒ {verdict_text}：{basis}"
    if not within_budget:
        return VERDICT_INFEASIBLE, detail
    if str(tier["algorithm"]) == "OPD":
        # 先堵一个后门：`unmeasured` **不得**成为"任何 OPD 配置都放行"的借口。
        # 教师**不换出**时，其 bf16 权重是每卡常驻的——这是权重常驻的**算术事实**
        # （不是实测结论），与学生侧相加必然超预算 ⇒ 确定性判定为 infeasible。
        distill = build_config(tier).get("distill") or {}
        if distill.get("offload_teacher_model") is not True:
            teacher_gib = _teacher_weights_gib()
            return VERDICT_INFEASIBLE, (
                "OPD 教师**未换出**（`distill.offload_teacher_model` 不是 true）："
                f"教师 27B bf16 权重 ≈ {teacher_gib:.1f} GiB/卡是**常驻**的，"
                f"与学生侧估算 {gib:.1f} GiB 相加 ≈ {gib + teacher_gib:.1f} GiB > "
                f"预算 {CARD_BUDGET_GIB:g} GiB ⇒ 拒绝生成。"
                "（依据是权重常驻的算术事实，不是上机实测；教师**换出后**的 GPU 工作集"
                "仍是未测算量，见 OPD_TEACHER_UNMEASURED_NOTE。）"
            )
        # 学生侧确定性算式通过、教师已换出，但教师前向的 GPU 工作集未测算
        # ⇒ 只能标"待上机测量"。
        return VERDICT_UNMEASURED, f"{OPD_TEACHER_UNMEASURED_NOTE}；学生侧算式：{detail}"
    return VERDICT_FEASIBLE, detail


def feasibility(tier: dict[str, Any]) -> tuple[bool, str]:
    """兼容视图：只有 ``feasible`` 才为 ``True``。

    ★ **``False`` 不等于「必然 OOM」，也不等于「不可生成」**——它只表示
    **「不得声称可行」**。"超预算"与"未测算"必须用 :func:`feasibility_verdict` 区分。
    """
    verdict, detail = feasibility_verdict(tier)
    return verdict == VERDICT_FEASIBLE, detail


def feasibility_for(tier: dict[str, Any]) -> tuple[bool | None, str]:
    """给 blocked / unmeasured / not_applicable 档返回 ``None``（**均不得声称可行**）。

    三者用 ``feasibility_verdict`` 的字符串区分（``blocked`` / ``unmeasured`` /
    ``not_applicable``），manifest 里三者都带显式 ``verdict`` 字段——不靠 ``None`` 猜。
    """
    verdict, detail = feasibility_verdict(tier)
    if verdict in (VERDICT_BLOCKED, VERDICT_UNMEASURED, VERDICT_NOT_APPLICABLE):
        return None, detail
    return verdict == VERDICT_FEASIBLE, detail


def build_config(tier: dict[str, Any]) -> dict[str, Any]:
    """构造一档的配置骨架（只使用 schema 中确实存在的字段）。"""
    model = MODELS[str(tier["model"])]
    tier_id = str(tier["tier_id"])
    cards = int(tier["cards"])
    backend = str(tier["backend"])
    algorithm = str(tier["algorithm"])

    config: dict[str, Any] = {
        "train_method": ALGORITHM_TO_TRAIN_METHOD[algorithm],
        "backend": "msswift" if backend == "ms-swift" else "native",
        "tuner_type": "full" if tier["mode"] == "全量" else "lora",
        "model": {
            "model_path": model["path"],
            "torch_dtype": "bfloat16",
            "gradient_checkpointing": True,
        },
        "data": {
            # 本档训练子集：由 run_matrix54.sh 从 ELAM V5 data/train.jsonl 取前 N 条生成。
            # 路径固定 ⇒ config 仍是产物的唯一描述（§10.1）；N 记录在 manifest 里。
            # 图像相对路径 `../images/...` 由该路径的父目录解析到 /work/elam-v5/images。
            "train_path": subset_path(tier_id),
            "max_prompt_length": INITIAL_MAX_PROMPT_LENGTH,
        },
        "training": {
            "output_dir": f"/out/{tier_id}",
            "run_name": tier_id,
            "overwrite_output_dir": True,
            "seed": 42,
            # ★ 唯一真相源 = `MATRIX_MAX_EPOCHS`：`expected_optimizer_steps_reachable`
            #   按同一个常量算"可产出步数上限" ⇒ 两处口径不可能分叉（§1.4）。
            "max_epochs": MATRIX_MAX_EPOCHS,
            "learning_rate": _learning_rate(algorithm),
            "gradient_accumulation_micro_batches": 1,
            # epoch 末 checkpoint 关掉：全参档**不受** `save_steps` 控制的落盘（native trainer
            # 的 final、ms-swift 的 epoch 末）也要挡一道；容器内另有兜底保留步骤。
            "save_checkpoint_every_epoch": False,
            # ★ ckpt 保留策略（2026-09-19，用户已授权）：`save_steps` = 该档**计划总步数**
            #   ⇒ 每档只落一份 ckpt。原值 `1` ⇒ 每 step 一份 ⇒ 单档 100 份、≈25 GB/档
            #   （228 实测，磁盘 96%）。口径与 A3 论证见 `checkpoint_save_steps()`。
            #   ★ native GRASPO 那 9 档的计划值**大于**可达上限（20/10/5 vs 3/2/1）⇒
            #   `save_steps` 不触发 ⇒ 只落终态 `final/`（保留策略显式承认它）。这是刻意的：
            #   native 中间段叫 `step_<N>`、保留块只清 `checkpoint-*`，压到可达上限反而会
            #   每档多留一份重 ckpt（与 `keep_per_tier=1` 冲突）。详见该函数 docstring。
            #   **不能**设 0/负：那会落到 `save_strategy: epoch`（另一种落盘形态）。
            "save_steps": checkpoint_save_steps(tier),
        },
    }

    if tier["mode"] == "LoRA":
        # LoRA 参数档位：r=8/alpha=16 与既有 ms-swift 冒烟口径一致（可再调）。
        # ★ `target_preset` 是**条件选择**：模型有视觉塔 且 该档数据含图 ⇒ 视觉预设，
        #   否则语言预设（详见 `lora_target_preset` 的 docstring 与其实测证据）。
        config["lora"] = {
            "r": 8,
            "alpha": 16,
            "dropout": 0.0,
            "target_preset": lora_target_preset(tier),
        }
    # 全参档位不写 lora 段：`tuner_type: full` 与 lora.adapter_path 互斥，
    # 配置校验会在加载时拒绝两者并存（schema.validate_tuner_type_combination）。

    if backend == "native":
        if tier["mode"] == "全量" and cards > 1:
            # native 全参只支持 PP 分片；DP/TP 会被校验 fail-closed 拒绝。
            config["native"] = {
                "tp_size": 1,
                "dp_size": 1,
                "pp_size": cards,
                "micro_batch_size": 1,
            }
        else:
            config["native"] = {
                "tp_size": 1,
                "dp_size": cards,
                "pp_size": 1,
                "micro_batch_size": 1,
            }
    else:
        config["msswift"] = {
            "nproc_per_node": cards,
            "per_device_train_batch_size": 1,
            "sequence_parallel_size": 1,
            "use_vllm": False,
        }
        if tier["mode"] == "全量":
            # 全参必须给可行配方，否则 1 卡估算超预算（生成期断言会兜底）。
            config["msswift"].update(ms_swift_full_recipe(cards))

    if algorithm == "GRASPO":
        config["reward"] = {"kind": "graspo"}
    if algorithm == "CPT":
        # CPT 的算法级配置是**后端中立**段（`core/schema.py::PretrainConfig`），
        # 由 `_config_mapping` 的 `stage="cpt"` 分支映射成 `swift pt` 的两条等价参数。
        # 这里**显式写出**（而不是靠 schema 默认值），因为"配置仍是产物的唯一描述"
        # （§10.1）：读档位配置即可回答"这档是不是按 CPT 语义在跑"。
        config["pretrain"] = {"loss_scale": "all", "use_chat_template": False}
    if algorithm == "OPD":
        # OPD 的教师/学生对是**后端中立**段（`core/schema.py::DistillConfig`）；
        # 学生 = 上面的 `model.model_path`（不另设字段），教师显式给出（配置层强制非空）。
        # `offload_teacher_model: true` 是**起始配方**：教师 27B 权重常驻 CPU，
        # 否则其 bf16 权重（≈50.3 GiB/卡）与学生侧常驻相加在任何卡数下都超预算。
        # ★ 这只是"起跑姿势"，不是"可行性结论"——教师前向的 GPU 工作集未测算，
        #   全部 OPD 档的 verdict 恒为 UNMEASURED（见 OPD_TEACHER_UNMEASURED_NOTE）。
        config["distill"] = {
            "teacher_model_path": MODELS[OPD_TEACHER_MODEL]["path"],
            "offload_teacher_model": True,
        }
    # ── 运维监控段（`graspo record-gpu-memory` 用；**与训练产物无关**）──────────
    # 为什么写进档配置（而不是由 runner 现编一份）：`record-gpu-memory` 自 W3 起
    # **配置驱动**（宪法 §10.1）——`output_dir` / `tag` / `interval_sec` 全部决定产物，
    # CLI 上已无对应选项。而"这档的产物落在哪"本来就是**同一份档位配置**里的既有
    # 事实（见上面的 `training.output_dir`）；把监控段也写进同一个文件，"这一档跑
    # 什么、产物落在哪、采样打什么标签"就只有**一个**真相源（§1.4）。若改由 runner
    # 现编一份，同一条事实就有两份文档，迟早漂移。
    #
    # ★ 落点**必须**由 `GPU_MONITOR_CONTAINER_DIR` 唯一决定（= 容器内 `/out/gpu`
    #   = 宿主 `<RUN_ROOT>/<T###>/gpu/`），**不能**用 `<training.output_dir>/gpu`：
    #
    #   2026-09-20 真机确证的缺陷（228 T010，两次真跑同一签名）：矩阵档一律
    #   `overwrite_output_dir: true`，训练进程启动时
    #   `lora_io.py::prepare_output_dir(out, overwrite=True)` 对**非空**的
    #   `<training.output_dir>`（= `/out/T010`）执行 `shutil.rmtree(out)`；而采样器
    #   先于 torchrun 起、刚在 `/out/T010/gpu` 建好目录并 touch 了 jsonl ⇒ 整棵被删
    #   ⇒ 采样器首个 `append_jsonl` 即 ENOENT（首错签名 `gpu_monitor.py:621` +
    #   `'/out/T010/gpu/gpu_memory.jsonl'`），摘要与"取证缺口登记"一并消失。
    #   ⇒ 把落点与训练输出目录**解耦**（落在 RUN_DIR 直下），而不是靠"谁先谁后"的运气。
    #   同时这条路径正是 runner 断言、缺口登记与 `collect_results._read_host_sample_peak_memory`
    #   一直在读的那条：宿主采样摘要读入 `host_sample_peak_gib` 旁路，**不是** §7 峰值列的
    #   口径（该列为**后端二分口径**：native 档读 `rank_metrics` 的 rank0 `max_allocated_mib`；
    #   ms-swift 档读其自报 `memory(GiB)`，即 `max_memory_reserved`）—— 修完生产端与消费端
    #   重新同源。
    #   ★ 改这里会断取证链：改前先读 scripts/collect_results.py::_read_peak_memory。
    config["gpu_monitor"] = {
        "output_dir": GPU_MONITOR_CONTAINER_DIR,
        "tag": tier_id,
        "interval_sec": GPU_MONITOR_INTERVAL_SEC,
    }
    return config


def _yaml_scalar(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str):
        return value
    return str(value)


def opd_relation_header(tier: dict[str, Any]) -> str:
    """OPD 档**一眼可辨**的教师/学生关系声明（**每档都有**，不只在自蒸馏档）。

    ★ 为什么每档都要写：教师是**用户拍板的常量**（27B），学生随台账变；两者同名时
    配置看起来像"教师路径写错了"。把关系写进**每一条** OPD 配置的头部，
    读配置的人（或 AI）不必去翻 manifest 就知道**这不是笔误**。
    """
    relation = opd_teacher_relation(tier)
    student = MODELS[str(tier["model"])]["name"]
    teacher = MODELS[OPD_TEACHER_MODEL]["name"]
    pointer = (
        "逐档关系见 manifest 的 `feasibility_model.opd_teacher.relation_by_tier`"
        "（逐档字段名 `opd_teacher_relation`）。"
    )
    if relation == "self-distillation":
        return (
            f"★ 教师/学生关系: **self-distillation** —— 本档**教师 = 学生**（均为 {teacher}），"
            "**这不是笔误**：台账该档的 `model` 列 = **学生**（graspo 处处如此），"
            f"而教师是用户拍板的**常量** {teacher}，两者同名即自蒸馏"
            "（ms-swift 明确支持的合法跑法：LoRA 用 `disable_adapter()` 取教师 logits，"
            "不额外加载模型）。" + pointer
        )
    if relation == "not-applicable":
        # 防御性分支：`not_applicable` 档**不产出 YAML**（由 classify_expressibility 拦在
        # 渲染之前）。留这个分支是为了"即使被误渲染，也绝不写成自蒸馏/独立教师"——
        # 那种输出会让人以为这档有配方。
        return (
            f"★ 教师/学生关系: **not-applicable** —— 本档（{student} 学生 × {teacher} 教师常量）"
            "**确实不适用**（见 OPD_NOT_APPLICABLE_NOTE）；本档不应产出可跑配置。" + pointer
        )
    return (
        f"★ 教师/学生关系: **independent** —— 学生 {student} ≠ 教师 {teacher}"
        "（用户拍板常量）。" + pointer
    )


def _algorithm_note(tier: dict[str, Any]) -> str | None:
    """该档位的算法口径说明（写进档位配置头部；``None`` = 无额外口径需要声明）。

    单一真相源：CPT / OPD 的说明各只有一份（``CPT_NOTE`` / ``OPD_NOTE``），
    配置头部、报告都引用它，不各写一遍（§1.4）。OPD 额外带上**逐档**的教师/学生关系
    （``opd_relation_header``）——自蒸馏与独立教师是**两种不同的跑法**，必须显式。
    """
    algorithm = str(tier["algorithm"])
    note = {"CPT": CPT_NOTE, "OPD": OPD_NOTE}.get(algorithm)
    if algorithm == "OPD" and note:
        note = f"{note} {opd_relation_header(tier)}"
    return note


def render_config_yaml(tier: dict[str, Any], status: str, reason: str | None) -> str:
    """把配置 dict 渲染成带口径注释的 YAML 文本。"""
    tier_id = str(tier["tier_id"])
    verdict, verdict_detail = feasibility_verdict(tier)
    header = [
        f"# {tier_id} | {tier['model']} | {tier['algorithm']} | {tier['mode']} | "
        f"{tier['backend']} | {tier['cards']}卡",
        "# 生成器: tests/e2e/generate_matrix.py（手改无效，请改生成器）",
        "# 口径: 只用 1/2/4 卡；GPU 集合取自 {0,1,2,3}；上下文长度"
        "由递增加长法实测得出，不预设档位。",
        f"# 数据: ELAM V5 训练子集前 {subset_size(str(tier['algorithm']))} 条（宿主数据根目录经 "
        f"{ELAM_HOST_ROOT_ENV} 注入，见 .local/）；子集由 run_matrix54.sh 生成，"
        f"图像经 ../images 解析。",
        f"# ⚠️ 数据口径告警: {DATA_INTEGRITY_CAVEAT}",
        f"# 可表达性: {status}",
        # 显存可行性必须**带 verdict 标签**：`unmeasured` 与 `infeasible` 的文案完全不同，
        # 只看一句人话会被读混（"不得声称可行" vs "估算超预算"）。
        f"# 显存可行性: [{verdict}] {verdict_detail}",
    ]
    if algorithm_note := _algorithm_note(tier):
        header.append(f"# 算法口径: {algorithm_note}")
    if reason:
        header.append(f"# 注意: {reason}")
    lines: list[str] = list(header)

    config = build_config(tier)
    for section, value in config.items():
        if isinstance(value, dict):
            lines.append(f"{section}:")
            for key, item in value.items():
                if key == "max_prompt_length":
                    lines.append(f"  {key}: {_yaml_scalar(item)}")
                    continue
                lines.append(f"  {key}: {_yaml_scalar(item)}")
        else:
            lines.append(f"{section}: {_yaml_scalar(value)}")
    return "\n".join(lines) + "\n"


def render_blocked_stub(tier: dict[str, Any], reason: str) -> str:
    """不可表达档位的阻塞说明（不产出一个"看似能跑"的错误配置）。"""
    return (
        f"# {tier['tier_id']} 当前不可执行（blocked）\n\n"
        f"档位: {tier['model']} | {tier['algorithm']} | {tier['mode']} | "
        f"{tier['backend']} | {tier['cards']}卡\n\n"
        f"阻塞原因: {reason}\n\n"
        "处置: 该档所需能力是本期 14 个 ⚠️ 能力格的开发目标；能力落地并通过验证后，\n"
        "     由 tests/e2e/generate_matrix.py 重新生成本档配置再执行。\n"
    )


def render_not_applicable_stub(tier: dict[str, Any], reason: str) -> str:
    """**确实不适用**档位的说明（不产出一个"看似能跑"的错误配置）。

    ★ 与 :func:`render_blocked_stub` 的区别**必须写在纸面上**：`blocked` 是
    "现在给不出配方、能力落地后可再生成"；`not_applicable` 是
    "**用户已拍板的口径下不存在该档的合法形态**"——它**不是**待办，不会被
    "能力落地"解除。把两者写成同一段文字，就会有人把它当成待办去排期。
    """
    return (
        f"# {tier['tier_id']} 确实不适用（not_applicable）—— 合法终态，不产出可跑配置\n\n"
        f"档位: {tier['model']} | {tier['algorithm']} | {tier['mode']} | "
        f"{tier['backend']} | {tier['cards']}卡\n\n"
        f"不适用的理由: {reason}\n\n"
        "定性: **确实不适用**（用户已拍板的口径下不存在该档的合法配置）——\n"
        "      **不是** `blocked`（尚未给出配方、将来可能做得了），"
        "**不是** `❌ 失败`（跑过没通过）。\n"
        "      「失败」与「确实不适用」都是验收口径明文允许的**合法产出**。\n"
        "处置: 无需排期、无需上机；台账侧按 `⛔ 不适用` 登记，理由引用本节。\n"
    )


# ── 运行清单 ────────────────────────────────────────────────────────────────


def gpu_assignment_record(spec: str, tiers: list[dict[str, Any]]) -> dict[str, Any]:
    """本档卡集合的**覆盖记录**（只在给定覆盖时写进 manifest，默认清单逐字节不变）。

    只记录 ``tiers[].gpus`` 装不下的信息（**不复制**逐档卡集合，避免第二真相源）：
    覆盖值本身、来源、边界常量、映射指纹与 A4 口径。
    """
    pairs = [(str(tier["tier_id"]), tier["gpus"]) for tier in tiers]
    return {
        "source": "override",
        "override_spec": spec,
        "override_cli_flag": GPU_OVERRIDE_CLI_FLAG,
        "default_by_cards": {str(cards): list(gpus) for cards, gpus in sorted(GPU_SETS.items())},
        "allowed_gpus": list(GPU_ALLOWED),
        "forbidden_production_gpus": list(GPU_FORBIDDEN_PRODUCTION),
        "max_cards_per_job": MAX_CARDS_PER_JOB,
        "numa_nodes": {"0": [0, 1, 2, 3], "1": [4, 5, 6, 7]},
        "fingerprint_sha256": gpu_assignment_fingerprint(pairs),
        "fingerprint_note": (
            "档 → 卡集合映射（tiers[].gpus）的 sha256。A4「同档两跑必须同卡位」："
            "两次运行必须引用同一份清单；复核命令 "
            "`python3 tests/e2e/gpu_assignment.py check --manifest <清单> "
            "--expect-fingerprint <指纹>`，指纹不一致即两跑不可比。"
        ),
        "runtime_override_note": (
            "运行期换卡集合：`GRASPO_RUNNER_MANIFEST=<另一份清单>`（既有覆盖点）。"
            "runner 只从清单读 tiers[].gpus ⇒ 不改 runner、不改档位 YAML；"
            "本次实际用哪几张卡由守卫写进运行产物（`[gpu-guard] OK: "
            "NVIDIA_VISIBLE_DEVICES=…`）。"
        ),
    }


def build_manifest(
    tiers: list[dict[str, Any]],
    train_source: str | None = None,
    gpu_assignment: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """构造运行清单（结果收集器的输入）。

    `train_source` 决定 runner 的**训练子集读取源**（默认 `mini`，见 `resolve_train_source`）；
    它**不**改变任何既有字段与每档 `data.train_path`（生成位置）。
    `gpu_assignment`（覆盖记录）为 ``None`` 时**不新增任何字段** —— 默认清单逐字节不变。
    """
    resolved_train_source = resolve_train_source(train_source)
    entries = []
    for tier in tiers:
        status, reason = classify_expressibility(tier)
        entry: dict[str, Any] = {
            "tier_id": tier["tier_id"],
            "model": tier["model"],
            "model_name": MODELS[str(tier["model"])]["name"],
            "model_path": MODELS[str(tier["model"])]["path"],
            "algorithm": tier["algorithm"],
            "mode": tier["mode"],
            "backend": tier["backend"],
            "cards": tier["cards"],
            "gpus": tier["gpus"],
            "status": status,
            #: 宿主侧模型目录的两级来源（环境信息，值只在 .local/）：
            #: 默认 ``$GRASPO_MODELS_HOST_ROOT/<容器内目录名>``，逐档可用
            #: ``GRASPO_9B_HOST_DIR`` / ``GRASPO_27B_HOST_DIR`` 单点覆盖。
            "model_host_root_env_var": MODELS_HOST_ROOT_ENV,
            "model_host_dir_env_var": models_host_dir_env(str(tier["model"])),
            "model_host_dir_default": (
                f"<{MODELS_HOST_ROOT_ENV}>/{_MODEL_DIR_NAMES[str(tier['model'])]}"
            ),
        }
        if reason is not None:
            entry["status_reason"] = reason
        # ★ 适用性（2026-09-19，指挥官裁定 A）：与"可表达性"正交的一维。
        #   `applicable: false` = **确实不适用**（用户拍板口径下不存在合法配置），
        #   是**合法终态**；只有 `true` 的档才谈"能不能跑 / 装不装得下"。
        applicable = status != "not_applicable"
        entry["applicable"] = applicable
        if not applicable:
            entry["applicability_reason"] = reason
        if str(tier["algorithm"]) == "OPD":
            # 教师/学生关系必须进 manifest：独立教师与自蒸馏是两种跑法，
            # 事后只看 "teacher_model_path" 无法区分（同名的档）。
            # ★ `T052`–`T054` 现为 `not-applicable`（不再按自蒸馏渲染）。
            entry["opd_teacher_relation"] = opd_teacher_relation(tier)
        if status in ("blocked", "not_applicable"):
            entry["config"] = None
            # 不适用档不产出"可跑配置"，也不产出 blocked 的"能力落地后再生成"说明——
            # 后者会被读成"这档将来能跑"。理由进 `applicability_reason`。
            entry["blocked_doc"] = (
                f"samples/configs/matrix54/{tier['tier_id']}.blocked.md"
                if status == "blocked"
                else None
            )
        else:
            entry["config"] = f"samples/configs/matrix54/{tier['tier_id']}.yaml"
        entry["data"] = {
            "train_path": subset_path(str(tier["tier_id"])),
            "subset_size": subset_size(str(tier["algorithm"])),
            "full_train_jsonl": ELAM_TRAIN_JSONL,
            "caveat": DATA_INTEGRITY_CAVEAT,
        }
        # 期望总优化步数 = `training.save_steps`（ckpt 保留策略，见 runtime.checkpoint_retention）
        entry["expected_optimizer_steps"] = expected_optimizer_steps(tier)
        entry["checkpoint_save_steps"] = checkpoint_save_steps(tier)
        # ★ 步数口径的**第二个量**（AF1 §⑥ 方案 S1；`expected_optimizer_steps` 保留原义
        #   给 ckpt `save_steps` 用，**不覆盖**）：每 epoch 上限 + 整个 run 的上限。
        #   公式只在 `native_graspo_steps_per_epoch` / `expected_optimizer_steps_per_epoch`
        #   一处定义，这里只是把结果落盘（§1.4 单一真相源）。
        entry["expected_optimizer_steps_per_epoch"] = expected_optimizer_steps_per_epoch(tier)
        entry["expected_optimizer_steps_reachable"] = expected_optimizer_steps_reachable(tier)
        feasible, estimate = feasibility_for(tier)
        verdict, _ = feasibility_verdict(tier)
        entry["feasibility"] = {
            # ★ `verdict` 是显式四态（+infeasible 共五值），`feasible` 只保留向后兼容语义：
            #   feasible ⇒ True；infeasible ⇒ False；unmeasured / blocked / not_applicable
            #   ⇒ None（**不得声称可行**）。不要用 `feasible is None` 区分原因 —— 读 `verdict`。
            "verdict": verdict,
            "feasible": feasible,
            "requires_measurement": verdict == VERDICT_UNMEASURED,
            "estimate": estimate,
            "recipe": (
                {"native": build_config(tier).get("native")}
                if tier["backend"] == "native"
                else {"msswift": build_config(tier).get("msswift")}
            )
            if verdict not in (VERDICT_BLOCKED, VERDICT_NOT_APPLICABLE)
            else None,
        }
        entry["acceptance"] = {
            "formal_gate": {
                "min_epochs": 1,
                "min_optimizer_steps": FORMAL_GATE_MIN_OPTIMIZER_STEPS,
                "min_train_samples": 20 if tier["algorithm"] in {"GRASPO", "OPD"} else 100,
                # ★ **生成期显式标记**（AF1 §⑥ 方案 S4，§2.2 显式即防呆）：该档的 step
                #   门槛**是否适用**。让"结构上不可能达标"的档**无法静默进正式台账**——
                #   否则下一批跑完，T030 会被 A2 判 ❌，而读者会读成"能力不行"。
                #   ★ 它**不放宽任何要求**：门槛值 min_optimizer_steps 一个字都没动；
                #     标记为 applicable 的档照原判据判（见 step_gate_applicability）。
                "step_gate_applicability": step_gate_applicability(
                    tier, min_optimizer_steps=FORMAL_GATE_MIN_OPTIMIZER_STEPS
                )[0],
                "step_gate_basis": (
                    f"expected_optimizer_steps_reachable={expected_optimizer_steps_reachable(tier)}"
                    f"；per_epoch={expected_optimizer_steps_per_epoch(tier)}"
                    f"；max_epochs={MATRIX_MAX_EPOCHS}；"
                    + step_gate_applicability(
                        tier, min_optimizer_steps=FORMAL_GATE_MIN_OPTIMIZER_STEPS
                    )[1]
                ),
            },
            "criteria": ["A1", "A2", "A3", "A4", "A5", "A6"],
        }
        entries.append(entry)

    manifest: dict[str, Any] = {
        "schema_version": 1,
        "source": "docs/capability-matrix.md §6/§7",
        "counts": {
            "total": len(entries),
            "by_algorithm": _count_by(entries, "algorithm"),
            "by_cards": {str(k): v for k, v in sorted(_count_by_cards(entries).items())},
            "by_backend": _count_by(entries, "backend"),
            "by_mode": _count_by(entries, "mode"),
            "by_status": _count_by(entries, "status"),
            "by_applicability": _count_by(
                [
                    {"applicable": "applicable" if e["applicable"] else "not_applicable"}
                    for e in entries
                ],
                "applicable",
            ),
            "by_feasibility_verdict": _count_by(
                [entry["feasibility"] for entry in entries], "verdict"
            ),
        },
        "runtime": {
            "image": "graspo-msswift:4.5.3",
            "timeout_sec": TIMEOUT_SEC,
            "initial_max_prompt_length": INITIAL_MAX_PROMPT_LENGTH,
            # ── ckpt 保留策略（2026-09-19，用户已授权；主机侧 rig 照此执行，不另立口径）──
            # 原本 `save_steps: 1` ⇒ 每 step 一份 ⇒ ≈25 GB/档（228 实测，磁盘 96%）。
            # 现策略：每档只留**一份** ckpt（A3 判据只需一份可重载的 ckpt）。
            "checkpoint_retention": {
                "policy": "keep-latest-single-checkpoint",
                "keep_per_tier": 1,
                "how": (
                    "① 配置侧：`training.save_steps` = 该档**计划总步数**（见每档 "
                    "`checkpoint_save_steps`），=「每档只落一份」；"
                    "★ 计划步数=「实际参与优化的样本数 ÷ 卡数」，而 **RLHF（GRASPO/GRPO）档的"
                    "样本数 = 子集 × `training.rollout_group_size`**（ms-swift 的 "
                    "`RepeatSampler(mini_repeat_count=G)` 会按 G 重复数据集；矩阵档不覆盖该字段，"
                    "取 schema 默认值 8）⇒ `T031`/`T043` 真机 `global_step/max_steps` = 160/160。"
                    "**此项 2026-09-20 修复**：修前 GRASPO 档的 `save_steps` 低了 8 倍、每个 run "
                    "真落 8 份 ckpt，只剩一份实际全靠②；"
                    "★ **native GRASPO 的例外**：其计划值（20/10/5）**大于**可达上限（3/2/1）"
                    "⇒ `save_steps` 不触发 ⇒ 只落终态 `final/`，由②的 native 分支显式承认"
                    "（`KEEP …（native 终态产物）`）。这是刻意的：native 中间段叫 `step_<N>`、"
                    "②的清理范围只有 `checkpoint-*`，把计划值压到可达上限反而每档多留一份重 ckpt；"
                    "② runner 侧：容器内落盘后立即逐项目具名删除旧 ckpt（`[ckpt-retention]` "
                    "日志行），并断言至少还剩一份（否则报运行链路错误，不静默通过）。"
                ),
                "keep": [
                    "最新一份 `checkpoint-<step>/`（A2 权重变化证据 + A3 重载证据）",
                    "轻量证据一律保留：`stdout.log`、`logging.jsonl`、`args.json`、"
                    "`trainer_state.json`、`gpu/` 读数、`subsets/`、`exit_code`、"
                    "`ckpt_retention.state`",
                ],
                "delete": [
                    "同一档内除最新一份以外的 `checkpoint-*/`（逐项具名，不用通配符展开做操作数）",
                ],
                "a3_unaffected": (
                    "A3 = checkpoint 能否重载；`scripts/collect_results.py` 的 "
                    "`find_checkpoint_dirs()` 只要求**存在 ≥1 份**可解析的 `checkpoint-<step>/`，"
                    "没有份数要求 ⇒ 留一份即满足 A3。"
                ),
                "blocked_by_env": (
                    "checkpoint 由容器内 root 落盘，宿主普通用户删不掉（228 实测：`sudo` 需密码）"
                    "⇒ 删除只能发生在容器内，故实现落在 runner 的 entry.sh。"
                ),
            },
        },
        "models": {
            "container_root": MODELS_CONTAINER_ROOT,
            "host_root_env_var": MODELS_HOST_ROOT_ENV,
            "host_root_note": (
                "宿主模型根目录属环境信息（§15.1/§16），由运行时环境变量注入，"
                "不入 tracked 文件；runner 只读挂载到容器内 " + MODELS_CONTAINER_ROOT + "。"
                "缺失时 runner fail-closed 拒绝启动（运行链路错误，不是数据问题）。"
            ),
            "overrides": {
                size: {
                    "container_path": MODELS[size]["path"],
                    "host_dir_env_var": models_host_dir_env(size),
                    "host_dir_default": f"<{MODELS_HOST_ROOT_ENV}>/{_MODEL_DIR_NAMES[size]}",
                }
                for size in MODELS
            },
            "mount_in_runner": (
                f'-v "$MODELS_ROOT:{MODELS_CONTAINER_ROOT}:ro"（container_root 只读挂载）'
            ),
        },
        "feasibility_model": {
            "card_budget_gib": CARD_BUDGET_GIB,
            "activation_allowance_gib": ACTIVATION_ALLOWANCE_GIB,
            "bf16_param_bytes": _BF16,
            "grad_bytes": _GRAD,
            "adam_bytes": {
                "native": _ADAM_BF16,
                "ms-swift": _ADAM_FP32,
                "note": (
                    "按后端分别假设（统一口径，§1.4）：native k=8 B/param（bf16 矩估计）；"
                    "ms-swift k=12 B/param（DeepSpeed fp32 优化器态，保守）。"
                ),
            },
            "bytes_per_param_by_backend": {
                "native_full": NATIVE_OPTIMIZER_BYTES_PER_PARAM,
                "msswift_full": MSSWIFT_OPTIMIZER_BYTES_PER_PARAM,
                "lora": _BF16,
            },
            "assumptions": {
                "native_full": NATIVE_ACCOUNTING_ASSUMPTION,
                "msswift_full": MSSWIFT_ACCOUNTING_ASSUMPTION,
                "measured": False,
                "note": (
                    "两侧的优化器态精度都是**假设**、不是实测值；因此「估算超预算」只能表述为"
                    "『判定不可靠』，**不得写成『必然 OOM』**。"
                ),
            },
            "rules": [
                "LoRA：基座冻结 ⇒ 权重 2 B/param",
                "ms-swift 全参 zero2_offload（1/2 卡）：权重 2 + 梯度 2（优化器态在 CPU）",
                "ms-swift 全参 zero2+AutoTP4（4 卡）：权重 2/4 + (梯度+优化器) 10/4",
                "native 全参：仅 PP 分片 ⇒ k/卡数（k=8，bf16 Adam 假设）；"
                "1 卡无分片 ⇒ 估算超预算、判定不可靠（标 blocked，理由为诚实表述而非『必然 OOM』）",
                "CPT：与学生侧同一套学生侧算式（算法不改变常驻口径）；全量档沿用 ms-swift 全参配方",
                "OPD：学生侧算式同上；**教师侧不参与算式**——教师（27B）显存是本生成器"
                "**未测算**的量 ⇒ 判定恒为 unmeasured（不得声称可行，P-25）",
            ],
            "verdicts": {
                "values": [
                    VERDICT_FEASIBLE,
                    VERDICT_INFEASIBLE,
                    VERDICT_UNMEASURED,
                    VERDICT_BLOCKED,
                ],
                "meaning": {
                    VERDICT_FEASIBLE: "确定性算式 ≤ 预算且无未测算的额外消费者",
                    VERDICT_INFEASIBLE: "确定性算式已超预算 ⇒ 拒绝生成",
                    VERDICT_UNMEASURED: "算式通过但有未测算的额外消费者 ⇒ 不得声称可行，待上机测量",
                    VERDICT_BLOCKED: "当前给不出配方（能力未落地 / 无分片手段）",
                    VERDICT_NOT_APPLICABLE: (
                        "确实不适用——用户拍板口径下不存在该档的合法配置 ⇒ 合法终态，"
                        "不产出可跑配置、不参与可行性判断"
                    ),
                },
                "note": (
                    "`feasibility.feasible` 是向后兼容的布尔视图：True/False/None。"
                    "`None` 同时覆盖 unmeasured / blocked / not_applicable —— "
                    "**必须读 `verdict` 区分**，不得用 `feasible is None` 推断原因。"
                ),
            },
            "opd_teacher": {
                "teacher_model": MODELS[OPD_TEACHER_MODEL]["name"],
                "teacher_model_path": MODELS[OPD_TEACHER_MODEL]["path"],
                "student_model": (
                    f"OPD 学生**固定 {OPD_STUDENT_MODEL}**（用户拍板：省显存，价值高）；"
                    "教师 = 27B 常量。台账 `model` 列在 graspo 里 = 被训练的学生。"
                ),
                "teacher_weights_gib_bf16": round(_teacher_weights_gib(), 1),
                "teacher_weights_gib_bf16_measured": 50.1,
                "teacher_weights_measurement": (
                    "上机 probe 实测（PyTorch allocator `max_memory_allocated`，bf16，"
                    "26.896 B 参数）⇒ 50.10 GiB；出处 task-cpt-opd / task-r3-retest2 报告。"
                ),
                "measured": False,
                "unmeasured_note": OPD_TEACHER_UNMEASURED_NOTE,
                "not_applicable_note": OPD_NOT_APPLICABLE_NOTE,
                "not_applicable_tiers": [
                    str(tier["tier_id"]) for tier in tiers if opd_not_applicable(tier)[0]
                ],
                "relation_by_tier": {
                    str(tier["tier_id"]): opd_teacher_relation(tier)
                    for tier in tiers
                    if str(tier["algorithm"]) == "OPD"
                },
                "decision": (
                    "★ 已由指挥官裁定并落地（2026-09-19）：`T052`–`T054`（台账的 27B OPD 档）"
                    "由原口径 A（27B→27B 自蒸馏）改判为 **`not_applicable`（确实不适用）**。"
                    "理由：用户拍板 **OPD 教师 = 27B 常量、学生 = 9B**（省显存）⇒ OPD 学生固定 9B；"
                    "27B 行若作为学生则需 27B 学生 + 27B 教师，教师权重实测 50.10 GiB、"
                    "加 27B 学生权重 ≈100 GiB 权重常驻，与用户意图相反 ⇒ **确实不适用**。"
                    "`T046`–`T051`（9B OPD）保持 **independent（独立教师）**关系不变。"
                    "这是**合法终态**（验收口径明文：「失败」与「确实不适用」都是合法产出）。"
                ),
            },
            "generation_gate": (
                "assert_memory_feasible：确定性算式超预算（infeasible）即拒绝生成"
                "（不允许安静产出不可靠配置）；unmeasured 档**允许生成但不声称可行**"
                "（未测算的量只能靠上机量出来），并在 stdout 打出待测清单；"
                "`not_applicable` 档**不产出可跑配置**（确实不适用、无待测量）"
            ),
        },
        "data": {
            "source": "ELAM V5 balanced（与开发机副本 md5 一致）",
            "host_root_env_var": ELAM_HOST_ROOT_ENV,
            "host_root_note": (
                "宿主数据根目录属环境信息（§16），由运行时环境变量注入，不入 tracked 文件"
            ),
            "container_root": ELAM_CONTAINER_ROOT,
            "train_jsonl": ELAM_TRAIN_JSONL,
            "test_jsonl": ELAM_TEST_JSONL,
            "subset_dir": ELAM_SUBSET_DIR,
            "counts": {
                "train": ELAM_TRAIN_COUNT,
                "test": ELAM_TEST_COUNT,
                "images": ELAM_IMAGE_COUNT,
            },
            "image_reference_style": '"image": "../images/<name>.jpg"（相对 data 文件父目录）',
            "subset_size_by_algorithm": dict(SUBSET_SIZE_BY_ALGORITHM),
            "subset_rationale": "按 §6 门槛取下限（SFT ≥100 / RL ≥20），不整集跑（省资源）",
            "integrity_caveat": DATA_INTEGRITY_CAVEAT,
            # ↓↓↓ 新增（本包）：本档「训练子集**取数来源**」= mini 数据集 + 实测算出的摘要/行数。
            #     既有字段（含每档 `data.train_path` / `full_train_jsonl`）**一个都没动**。
            #     读取源与 `data.train_path` 是两件事：后者只约束**生成位置**。
            **mini_dataset_manifest_fields(resolved_train_source),
        },
    }
    # 覆盖记录**只在给定覆盖时出现**：不设覆盖 ⇒ 本节缺失 ⇒ 清单与旧版逐字节相同。
    if gpu_assignment is not None:
        manifest["gpu_assignment"] = gpu_assignment
    # ★ **生成期 fail-closed + 显式登记**（§2.3 边界校验即防呆 / §2.2 显式即防呆）：
    #   ① 逐档复核"写进清单的标记"与"公式重算的标记"一致（防手改/防漂移）；
    #   ② 把"结构上达不到门槛"的档**集中列出来**（含公式代入值）⇒ 读清单的人
    #      不可能漏看"这是上限所致，不是能力不足"。
    #   ★ **必须排在 `tiers` 之前**：`build_manifest` 的返回字面量里 `tiers` 是最后一个键，
    #     而 `tests/e2e/test_generate_matrix.py::test_default_manifest_has_no_gpu_assignment_section`
    #     把"`tiers` 是最后一个键"当成"没有顺手加尾部字段"的机器判据（防呆装置，不改）。
    manifest["step_gate_audit"] = _step_gate_audit(entries)
    manifest["tiers"] = entries
    return manifest


def _step_gate_audit(entries: list[dict[str, Any]]) -> dict[str, Any]:
    """逐档复核 step 门槛标记 + 登记"结构上达不到门槛"的档（生成期 fail-closed）。

    为什么要有它：AF1 诊断的核心风险是"9 档 native GRASPO 的步数低于门槛这件事，
    会被台账读者读成'这档能力不行'"。要让这种误读**不可能发生**，光在判据里加第三态
    还不够——**清单本身**必须把这个事实写在明面上（§2.2 显式即防呆：不靠读者去推）。
    """
    limited: list[dict[str, Any]] = []
    for entry in entries:
        gate = entry.get("acceptance", {}).get("formal_gate", {})
        marker = gate.get("step_gate_applicability")
        reachable = entry.get("expected_optimizer_steps_reachable")
        threshold = gate.get("min_optimizer_steps")
        # ① 标记与公式必须一致（否则说明清单被手改过 ⇒ 当场炸，不静默出清单）
        if marker == STEP_GATE_APPLICABLE and (
            not isinstance(reachable, int) or not isinstance(threshold, int)
        ):
            raise ValueError(
                f"档 {entry['tier_id']}：step 门槛标记={marker}，但 reachable/threshold "
                f"不是整数（{reachable!r}/{threshold!r}）⇒ 清单自相矛盾"
            )
        if isinstance(reachable, int) and isinstance(threshold, int):
            expected_marker = (
                STEP_GATE_NOT_APPLICABLE_STRUCTURAL
                if reachable < threshold
                else STEP_GATE_APPLICABLE
            )
            if marker != expected_marker:
                raise ValueError(
                    f"档 {entry['tier_id']}：step 门槛标记={marker} 与公式重算的 "
                    f"{expected_marker} 不一致（reachable={reachable}，threshold={threshold}）"
                    "⇒ 生成期 fail-closed"
                )
            if expected_marker == STEP_GATE_NOT_APPLICABLE_STRUCTURAL:
                limited.append(
                    {
                        "tier_id": entry["tier_id"],
                        "algorithm": entry["algorithm"],
                        "backend": entry["backend"],
                        "cards": entry["cards"],
                        "subset_size": entry["data"]["subset_size"],
                        "per_epoch": entry["expected_optimizer_steps_per_epoch"],
                        "max_epochs": MATRIX_MAX_EPOCHS,
                        "reachable": reachable,
                        "min_optimizer_steps": threshold,
                        "derivation": (
                            f"每 epoch 上限 = floor(floor({entry['data']['subset_size']}/"
                            f"{entry['cards']})/{NATIVE_ROLLOUT_QUEUE_BATCH_SIZE}) + 1 = "
                            f"{entry['expected_optimizer_steps_per_epoch']}；"
                            f"× max_epochs({MATRIX_MAX_EPOCHS}) = {reachable}"
                            f"  < 门槛 {threshold}"
                        ),
                        # 「结构上不可能」与「多跑 epoch 可达」是**两类**，必须分开标（AF1 要点 2）
                        "kind": (
                            "structural_impossible"
                            if entry["expected_optimizer_steps_per_epoch"] < threshold
                            and entry["algorithm"] == "GRASPO"
                            and entry["backend"] == "native"
                            and entry["data"]["subset_size"] // entry["cards"]
                            < NATIVE_ROLLOUT_QUEUE_BATCH_SIZE
                            else "needs_more_epochs"
                        ),
                    }
                )
    return {
        "threshold": FORMAL_GATE_MIN_OPTIMIZER_STEPS,
        "threshold_source": "docs/capability-matrix.md §6「正式记录门槛」（AF1/指挥官裁定：不放宽）",
        "rule": (
            "reachable >= threshold ⇒ 门槛照常判；reachable < threshold ⇒ 判「⚠ 口径不可测」"
            "（不是 ❌ 失败，也不是 ✅ 通过）"
        ),
        "structural_limited_tiers": limited,
        "structural_limited_note": (
            "这些档的 step 门槛**不是能力失败**，是『子集取数 × 训练器步进语义』算出来的"
            "上限所致。其中 kind=structural_impossible 的档**把 epoch 调大也达不到门槛**"
            "（阈值 queue×group 在 4 卡下永不可触发）；kind=needs_more_epochs 的档"
            "在 max_epochs 提高后可达（属目标/预算层，须用户拍板，本生成器不擅自改）。"
        ),
    }


def _count_by(entries: list[dict[str, Any]], field: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for entry in entries:
        counts[str(entry[field])] = counts.get(str(entry[field]), 0) + 1
    return counts


def _count_by_cards(entries: list[dict[str, Any]]) -> dict[int, int]:
    counts: dict[int, int] = {}
    for entry in entries:
        cards = int(entry["cards"])
        counts[cards] = counts.get(cards, 0) + 1
    return counts


# ── 自检 ────────────────────────────────────────────────────────────────────


def assert_ledger(tiers: list[dict[str, Any]], expected_total: int) -> None:
    """结构自检：逐档对应台账，计数与 §6/§7 完全一致。"""
    assert len(tiers) == expected_total, f"expected {expected_total} tiers, got {len(tiers)}"
    ids = [tier["tier_id"] for tier in tiers]
    expected_ids = [f"T{index:03d}" for index in range(1, expected_total + 1)]
    assert ids == expected_ids, "tier ids must be contiguous T001..T{expected_total}"

    by_algorithm = _count_by(tiers, "algorithm")
    assert by_algorithm == EXPECTED_BY_ALGORITHM, f"algorithm counts: {by_algorithm}"
    assert _count_by_cards(tiers) == EXPECTED_BY_CARDS, "card counts mismatch"
    assert _count_by(tiers, "backend") == EXPECTED_BY_BACKEND, "backend counts mismatch"
    assert _count_by(tiers, "mode") == EXPECTED_BY_MODE, "mode counts mismatch"

    # 逐档指纹：抽 5 个关键档位，锁死 §7 的行序（防止"数量对、顺序错"）。
    fingerprints = {
        "T001": ("9B", "CPT", "LoRA", "ms-swift", 1),
        "T010": ("9B", "SFT", "LoRA", "native", 1),
        "T021": ("9B", "SFT", "全量", "ms-swift", 4),
        "T028": ("9B", "GRASPO", "LoRA", "native", 1),
        "T046": ("9B", "OPD", "LoRA", "ms-swift", 1),
        "T054": ("27B", "OPD", "LoRA", "ms-swift", 4),
    }
    by_id = {tier["tier_id"]: tier for tier in tiers}
    for tier_id, expected in fingerprints.items():
        tier = by_id[tier_id]
        actual = (
            tier["model"],
            tier["algorithm"],
            tier["mode"],
            tier["backend"],
            tier["cards"],
        )
        assert actual == expected, f"{tier_id}: expected {expected}, got {actual}"

    # GPU 集合守卫（**加强版**，逐档跑 `gpu_assignment.check_gpus` 这一条实现）：
    # 卡数必须等于该档 cards；卡号 ⊆ {0..5}；不得含 GPU6/7（生产 vLLM）；≤ 4 卡；去重；非空。
    # 旧版写死 `max(gpus) <= 3`，与本次新增的覆盖点（允许把档挪到 GPU4/5 避让共享机他人占用）
    # 冲突，因此改为**显式允许集合** {0..5} —— 6/7 与越界卡仍然 fail-closed（未削弱边界）。
    for tier in tiers:
        gpus = tier["gpus"]
        check_gpus(str(tier["tier_id"]), gpus, int(tier["cards"]))
    assert GPU_FORBIDDEN_PRODUCTION == (6, 7) and MAX_CARDS_PER_JOB == 4
    assert GPU_ALLOWED == (0, 1, 2, 3, 4, 5)


def assert_no_legacy_terms(rendered: list[tuple[str, str]]) -> None:
    """生成物中不得出现已作废口径；也不得有顶层 `r:` 键。"""
    for name, text in rendered:
        for pattern in FORBIDDEN_PATTERNS:
            assert pattern not in text, f"{name}: forbidden legacy term {pattern!r}"
        for line in text.splitlines():
            assert not line.startswith("r:"), f"{name}: illegal top-level `r:` key"


def assert_expressibility(tiers: list[dict[str, Any]]) -> dict[str, int]:
    """统计可表达性，并断言各状态之和等于总数。

    ★ 四态（2026-09-19 起）：`not_applicable` 是**确实不适用**的合法终态
    （现有唯一来源 `T052`–`T054`）——它与 `blocked` 分开计数，否则
    "还没做"与"不该做"会被压成同一个数（§1.4 单一真相源要求可回答"为什么"）。
    """
    counts: dict[str, int] = {
        "ready": 0,
        "unverified": 0,
        "blocked": 0,
        "not_applicable": 0,
    }
    for tier in tiers:
        status, _ = classify_expressibility(tier)
        counts[status] += 1
    assert sum(counts.values()) == EXPECTED_TOTAL, counts
    return counts


# ── 生成期显存可行性断言（§2.3 边界校验即防呆）──────────────────────────────


def assert_memory_feasible(tiers: list[dict[str, Any]]) -> dict[str, tuple[str, str]]:
    """对每个**可执行**档做保守显存估算；任一档确定性算式超预算即**拒绝生成**。

    这是本轮补齐的关键防线：配置能生成 ≠ 能跑。宁可在生成期失败并报出算式，
    也不要在花掉 GPU 时间之后才发现不可靠。

    **四态口径（单一真相源 = :func:`feasibility_verdict`）**：

    - ``infeasible`` ⇒ **拒绝生成**（报出算式）；
    - ``unmeasured`` ⇒ **允许生成**，但**不声称可行**：把"待上机测量"打到 stdout，
      由 manifest 的 ``feasibility.requires_measurement`` 承接（★ 允许生成是**刻意的**：
      未测算的量只能靠上机量出来，"因为没量过所以不生成"会让它永远量不了）；
    - ``feasible`` ⇒ 通过；
    - ``blocked`` ⇒ 不参与断言（已知给不出配方，由 ``classify_expressibility`` 说明原因）；
    - ``not_applicable`` ⇒ **不参与断言**（确实不适用：既不是"放不下"也不是"待测量"，
      由 ``classify_expressibility`` / manifest 的 ``applicability`` 说明原因）。

    Returns:
        ``{tier_id: (verdict, detail)}``——第一项是**字符串四态**，不是布尔
        （布尔会把 "未测算" / "超预算" / "不适用" 压成同一个 False）。
    """
    verdicts: dict[str, tuple[str, str]] = {}
    violations: list[str] = []
    pending_measurement: list[str] = []
    for tier in tiers:
        verdict, detail = feasibility_verdict(tier)
        if verdict in (VERDICT_BLOCKED, VERDICT_NOT_APPLICABLE):
            continue
        verdicts[str(tier["tier_id"])] = (verdict, detail)
        if verdict == VERDICT_INFEASIBLE:
            violations.append(
                f"{tier['tier_id']} ({tier['algorithm']}/{tier['mode']}/"
                f"{tier['backend']}/{tier['cards']}卡): {detail}"
            )
        elif verdict == VERDICT_UNMEASURED:
            pending_measurement.append(f"{tier['tier_id']} ({tier['algorithm']})")
    if violations:
        raise AssertionError(
            "生成期显存可行性断言失败——拒绝生成确定性算式已超预算的配置：\n  - "
            + "\n  - ".join(violations)
        )
    if pending_measurement:
        print(
            "⚠️  待上机测量（未测算的额外显存消费者：OPD 教师，见 P-25）"
            f"——共 {len(pending_measurement)} 档，**不得**在此之前写成「可行」：\n  - "
            + "\n  - ".join(pending_measurement)
        )
    return verdicts


def feasibility_table(tiers: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """每个档位的配方 + 可行性自检结果（写进 manifest / 报告）。"""
    rows: list[dict[str, Any]] = []
    for tier in tiers:
        status, reason = classify_expressibility(tier)
        verdict, detail = feasibility_verdict(tier)
        ok, _ = feasibility_for(tier)
        config = build_config(tier) if status != "blocked" else {}
        recipe: dict[str, Any] = {}
        if status != "blocked":
            if tier["backend"] == "native":
                recipe = {"native": config.get("native")}
            else:
                recipe = {"msswift": config.get("msswift")}
            for section in ("pretrain", "distill"):
                if section in config:
                    recipe[section] = config[section]
        rows.append(
            {
                "tier_id": tier["tier_id"],
                "algorithm": tier["algorithm"],
                "mode": tier["mode"],
                "backend": tier["backend"],
                "cards": tier["cards"],
                "status": status,
                "verdict": verdict,
                "recipe": recipe,
                "feasible": ok,
                "estimate": detail,
                "status_reason": reason,
            }
        )
    return rows


# ── 容器内 torch 探测脚本（A3 转正路径的**生产端**）────────────────────────────
#
# 背景（环境性伪否，本段要根治的）：宿主 228 **没有 torch** ⇒ collector 对 native 档
# 只能拿到弱证据（``structural_only:zip_crc``），A3「checkpoint 可重载」判**取证缺口**
# ⇒ 18 个 native 档拿不到完整结论。但**跑训练的镜像自带 torch**（``graspo-msswift:4.5.3``，
# 实测 torch 2.11.0+cu130）⇒ 证据明明可得，只是没人把它落盘。
#
# 本段把探测结果落成 ``<run_dir>/torch_probe.json``，契约（schema / 字段 / sha256 防陈旧
# 算法 / 拒采信行为 / 判定映射）见 ``scripts/collect_results.py::read_torch_probe``；
# **判定逻辑只在 collector 里**（§1.4 单一真相源），本脚本只负责"在**有 torch 的地方**
# 取证据并落盘"，不承担判定。
#
# ★ 为什么是**独立脚本 + 运行期实测布局**而不是把路径写死：
#   ``checked[].relpath`` 必须是**相对 run_dir 的 POSIX 路径**。容器内 ``training.output_dir
#   = /out/<T###>`` ⇒ native 权重在 ``/out/<T###>/final/``，即宿主
#   ``<RUN_ROOT>/<T###>/<T###>/final/``（嵌套两层档号）。这个"从 run_dir 到权重"的相对
#   路径写错就会被 collector **拒采信**（文件不存在 ⇒ 整体退回弱证据）。因此脚本按
#   :data:`TORCH_PROBE_DIR_CANDIDATES` 的秩**实测**目录，relpath 由实测结果推出，
#   并把选中项打进 stdout（§2.2 显式即防呆）——第一档上机即可当场核对布局假设。
#
# ★ 失败不得影响训练结论（§3.4 退路与防线之分）：
#   本脚本**任何**失败路径都以 ``exit 0`` 结束，且**不写**探测文件 ⇒ collector 退回
#   弱证据路径（A3 仍是取证缺口，绝不放行）。探测失败**不是**训练失败，绝不改
#   ``exit_code`` 的训练语义（既有优先级：训练失败 137/124 > 保留失败 4 > 成功 0）。
PROBE_HEADER = '''#!/usr/bin/env python3
"""容器内 torch 重载探测（A3「checkpoint 可重载」的证据生产端）。

由 tests/e2e/generate_matrix.py::render_probe_script 生成到 $RUN_DIR/torch_probe.py
（容器内 /out/torch_probe.py）——**手改无效**，改生成器后重生成。

职责边界（§1.1 一事一责）：只做一件事——在**有 torch 的容器里**对 native 终态权重
做一次真实 torch.load 探测，把结果落成 <run_dir>/torch_probe.json。
**不做判定**：判定逻辑只在 scripts/collect_results.py::read_torch_probe（§1.4）。

契约（与 collector 同一条契约，改一处必须同步另一处）：
    {"schema": "<schema>", "torch": "<version>", "all_ok": true,
     "checked": [{"relpath": "<relpath>", "ok": true, "sha256": "<64 hex>", "tensors": 882}]}

三条硬约束（写错 ⇒ 证据被拒采信）：
    1) **写完权重再算 hash**：本脚本在训练结束、保留策略执行完之后才跑；
    2) **算完 hash 后不得再改动该文件**：本脚本只读不写权重，且它是收口前最后一个
       接触产物的步骤；
    3) relpath 一律**相对 run_dir 的 POSIX 路径**（不用绝对路径、不加 ./ 前缀）。

任何失败（无 torch / 找不到权重 / 读写异常）⇒ 打印原因、**不写探测文件**、exit 0。
"""
'''

PROBE_BODY = '''

import argparse
import hashlib
import json
import os
import posixpath
import sys

PROBE_SCHEMA = "{schema}"
PROBE_FILENAME = "{filename}"


def _print(*parts: object) -> None:
    print("[torch-probe]", *parts, flush=True)


def _sha256_and_size(path: str) -> tuple[str, int]:
    digest = hashlib.sha256()
    total = 0
    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
            total += len(chunk)
    return digest.hexdigest(), total


def _json_safe(value: object) -> object:
    if isinstance(value, dict):
        return {{str(key): _json_safe(item) for key, item in value.items()}}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def _flatten_state_dict(loaded: object) -> tuple[dict[str, object], str]:
    """把 torch.load 的返回值归一成「张量名 -> 张量」的扁平视图（只用于**计数与审计**）。"""
    if not isinstance(loaded, dict):
        return {{}}, "returned:" + type(loaded).__name__
    state = loaded.get("state_dict")
    if isinstance(state, dict):
        return state, "state_dict"
    lora = loaded.get("lora_state_dict")
    if isinstance(lora, dict):
        return lora, "lora_state_dict"
    tensors = {{key: value for key, value in loaded.items() if hasattr(value, "shape")}}
    if tensors:
        return tensors, "top_level_tensors"
    return {{}}, "top_level_other"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="torch probe for A3 evidence")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--tier", required=True)
    parser.add_argument("--max-files", type=int, default=4)
    args = parser.parse_args(argv)

    run_dir = os.path.abspath(args.run_dir)
    probe_path = os.path.join(run_dir, PROBE_FILENAME)
    try:
        import torch
    except Exception as exc:
        _print("SKIP: 容器内没有可用的 torch（" + type(exc).__name__ + ": " + str(exc) + "）")
        _print("  契约要求：不写探测文件 ⇒ collector 退回弱证据路径（A3 不因探测失败而假通过）")
        return 0
    torch_version = str(getattr(torch, "__version__", "unknown"))

    candidates = [
        os.path.join(run_dir, args.tier, "final"),
        os.path.join(run_dir, "final"),
    ]
    for current, dirnames, _ in os.walk(run_dir):
        dirnames.sort()
        for name in list(dirnames):
            if name == "final":
                candidates.append(os.path.join(current, name))
    unique: list[str] = []
    for candidate in candidates:
        if candidate not in unique:
            unique.append(candidate)
    checked_dir: str | None = None
    rank_files: list[str] = []
    for candidate in unique:
        if not os.path.isdir(candidate):
            continue
        found = sorted(
            os.path.join(candidate, name)
            for name in os.listdir(candidate)
            if name.startswith("rank_") and name.endswith(".pt")
        )
        if found:
            checked_dir = candidate
            rank_files = found
            break
    if checked_dir is None:
        _print("SKIP: 在 run_dir=" + run_dir + " 下找不到任何含 rank_*.pt 的 final/ 目录")
        _print("  候选目录：" + repr(unique))
        _print("  契约要求：不写探测文件 ⇒ A3 退回取证缺口（绝不放行）")
        return 0

    checked: list[dict[str, object]] = []
    all_ok = True
    for path in rank_files[: max(0, args.max_files)]:
        relpath = posixpath.join(*os.path.relpath(path, run_dir).split(os.sep))
        entry: dict[str, object] = {{"relpath": relpath, "ok": False, "sha256": None, "tensors": None}}
        digest: str | None = None
        size = 0
        try:
            digest, size = _sha256_and_size(path)
        except OSError as exc:
            entry["error"] = type(exc).__name__ + ": " + str(exc)
        else:
            entry["sha256"] = digest
            try:
                loaded = torch.load(path, map_location="cpu", weights_only=False)
            except Exception as exc:
                entry["ok"] = False
                entry["error"] = "torch.load: " + type(exc).__name__ + ": " + str(exc)
            else:
                state, kind = _flatten_state_dict(loaded)
                entry["ok"] = True
                entry["tensors"] = len(state)
                entry["state_dict_kind"] = kind
        if not entry["ok"]:
            all_ok = False
        checked.append(entry)
        _print("FILE relpath=" + relpath + " bytes=" + str(size) + " sha256=" + str(digest)
               + " ok=" + str(entry["ok"]) + " tensors=" + str(entry.get("tensors"))
               + (" error=" + str(entry.get("error")) if entry.get("error") else ""))

    if not checked:
        _print("SKIP: max-files 为 0 ⇒ 无可探测文件；不写探测文件")
        return 0

    payload = {{
        "schema": PROBE_SCHEMA,
        "torch": torch_version,
        "all_ok": bool(all_ok),
        "checked": checked,
        "selected_dir": os.path.relpath(checked_dir, run_dir),
        "checked_rank_files": len(rank_files),
        "probed_rank_files": len(checked),
        "state_dict_audit": [_json_safe(entry) for entry in checked],
    }}
    try:
        with open(probe_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\\n")
    except OSError as exc:
        _print("ERROR: 写 " + probe_path + " 失败：" + type(exc).__name__ + ": " + str(exc))
        _print("  契约要求：证据未落盘 ⇒ A3 退回取证缺口（绝不放行）")
        return 0
    _print("WROTE " + probe_path + " schema=" + PROBE_SCHEMA + " torch=" + torch_version
           + " all_ok=" + str(all_ok) + " checked=" + str(len(checked))
           + " selected_dir=" + str(payload["selected_dir"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
'''


def render_probe_script() -> str:
    """渲染容器内 torch 探测脚本（**唯一**生成入口，§1.4）。

    为什么用"常量拼接"而不是 f-string：生成物是 **Python 源码**，密集使用 ``{}``
    （字典/集合解析）⇒ 放进 f-string 必须逐个转义，可读性与可维护性都差（§2.2）。
    契约字面量（schema / 文件名）从上面的单一真相源常量注入，其余照抄。
    """
    return PROBE_HEADER + PROBE_BODY.format(
        schema=TORCH_PROBE_SCHEMA,
        filename=TORCH_PROBE_FILENAME,
    )


# ── 运行脚本骨架 ────────────────────────────────────────────────────────────


def render_runner(train_source: str | None = None) -> str:
    """生成批量执行脚本骨架（锁卡守卫 + 可信采样 + 数据子集 + 逐档记录）。

    `train_source` 只决定**训练子集读哪份 JSONL**（默认 `mini`），
    不影响 `data.train_path`、挂载目的地与任何断言。
    """
    resolve_train_source(train_source)
    # 容器内 torch 探测脚本（A3 转正路径的生产端）：在本函数内渲染一次，避免"渲染两次
    # 得到两份不同内容"的双真相源风险（§1.4）。契约字面量来自模块级常量。
    probe_script = render_probe_script()
    # 容器内**可写缓存/状态根**清单的 shell 数组正文：由 CONTAINER_CACHE_ROOTS 渲染，
    # runner 与 entry.sh 两处同源复用（避免 f-string 里再拼路径 / 再写字面量）。
    cache_roots_array = render_cache_roots_array()
    return f"""#!/bin/bash
# GRASPO 54 档批量执行骨架 —— 由 tests/e2e/generate_matrix.py 生成（手改无效）。
#
# 用法: bash tests/e2e/run_matrix54.sh <T###> [--dry-run]
#   - 每档：宿主侧锁卡守卫 → **模型挂载前置断言** → 生成训练子集 → docker run
#           （单一路径锁卡，只认 NVIDIA_VISIBLE_DEVICES）→ 容器内可信采样 + torchrun 训练
#   - 产物落 <RUN_ROOT>/<T###>/：exit_code, stdout.log, gpu/, subsets/, <T###>/（训练输出）
#     产物**属主 = 宿主跑批用户**（容器以 `--user <宿主 uid>:<gid>` 运行）⇒ 跑完后宿主侧
#     可直接 `rm`，不需要任何提权（PG-13 / U5「跑完即删」）。生效用户在 --dry-run 里可见。
#   - **必须先导出宿主数据根目录**：export {ELAM_HOST_ROOT_ENV}=<宿主 ELAM V5 数据根目录>
#   - **必须先导出宿主模型根目录**：export {MODELS_HOST_ROOT_ENV}=<宿主模型根目录>
#     该目录只读挂载到容器内 {MODELS_CONTAINER_ROOT}（模型**不在镜像里，必须挂**）。
#     个别档模型不在同一根下时，可用 `GRASPO_9B_HOST_DIR` / `GRASPO_27B_HOST_DIR` 单点覆盖。
#     （上面两者都是环境信息，见 .local/ 本地配置；不写进 tracked 文件，宪法 §16）
#
# 数据（**挂载目的地由配置反推，不硬编码**）:
#   每档按 §6 门槛取下限生成训练子集 -> 宿主 <RUN_ROOT>/<T###>/subsets/<T###>.jsonl，
#   只读挂到**该档配置 data.train_path 的所在目录**；图像根只读挂到**由 train_path
#   反推出来的同级 images 目录**。两者都从 manifest 的 data.train_path 用 dirname 推出来，
#   而不是写死的常量（单一真相源，见脚本 1) 段）。
#
#   ⚠ 为什么不再用"整棵数据根 + 其下嵌套子集"（实测确证的运行链路缺陷，rc=125）：
#     旧命令 `-v "$ELAM_HOST:{ELAM_CONTAINER_ROOT}:ro"` +
#     `-v "$RUN_DIR/subsets:{ELAM_SUBSET_DIR}:ro"` 把子挂载点放在一个**只读**绑定挂载之下，
#     dockerd 必须在只读文件系统上创建嵌套 mountpoint：
#     `mkdirat .../subsets: read-only file system` ⇒ `docker run` 返回 **125，一档都起不来**。
#     现方案让所有挂载目标**彼此互不嵌套**（子集目录与图像目录是兄弟，父目录留在容器
#     可写层里），从结构上消除"在只读挂载下建 mountpoint"这一步。
#     该不变量在启动前由 `assert_mount_targets_not_under_readonly` 断言（fail-closed，
#     给出可操作报错，而不是让 docker 报一个语义模糊的 125）。
#
# 模型: 每档配置里的 `model.model_path` 都指向容器内 {MODELS_CONTAINER_ROOT}/<模型目录名>；
#       runner 从 manifest 读出该路径并把它解析回宿主路径做存在性断言。
# ⚠️ 数据口径告警: {DATA_INTEGRITY_CAVEAT}
#
# 数据**读取源**（与上面"挂载目的地"是两件事，别混读）:
#   默认读 **mini 数据集**：`$ELAM_HOST/{ELAM_MINI_JSONL_RELATIVE}`
#   （文件名带 `mini-short-mm-` 前缀、落独立目录 `mini-dataset/`，与源数据
#    `data/train.jsonl` **物理隔离**——用户硬约束：靠命名与路径防误用，源数据只读，
#   禁止混放同名文件）。它**只用于「矩阵跑通」**（压时间/压成本）：
#   效果评测必须用**全量 test 集** {ELAM_TEST_JSONL}（{ELAM_TEST_COUNT} 行），
#   mini 的跑通结果不得当效果结论。
#   · 切回源数据（对照/回退）：`export {ELAM_MINI_JSONL_ENV}=$ELAM_HOST/data/train.jsonl`
#   · ⚠ **不要改 `data.train_path` 来换数据源**：该字段只约束**生成位置**
#     （<RUN_ROOT>/<Tier>/subsets/<Tier>.jsonl）。下面 1) 段的两条断言
#     （`TRAIN_PATH == dirname(TRAIN_PATH)/<Tier>.jsonl` 与图像根反推
#     `dirname(dirname(train_path))/images`）依赖它；改动它会直接打断整条链。
#
# 红线：只用 GPU0-5、每次最多 4 卡、GPU6/7 永不触碰。本脚本不自动执行矩阵。
set -uo pipefail

# ── 挂载结构不变量（fail-closed，纯 argv 分析，不需要 docker）──────────────────
# 唯一真相源：**任何挂载目标都不得位于另一个只读（ro）挂载目标之下**。
# 原因（实测确证，rc=125）：父挂载只读时 dockerd 无法在其内部创建嵌套 mountpoint，
# `mkdirat ...: read-only file system` ⇒ docker 返回 125，容器根本没起来。
# 本函数把这条"运行链路缺陷"提前变成可操作报错；参数是 `宿主源|容器目标|模式` 三元组。
#   - 目标规范化为无尾斜杠；
#   - 只读父目标下的任何挂载都判违规（含相等的情况——同一目标挂两次也是错误）。
# 允许测试单独 source 进来调用（见 GRASPO_RUNNER_LIB_ONLY）。
assert_mount_targets_not_under_readonly() {{
    local -a specs=("$@")
    local i j src dst mode src2 dst2 mode2
    for ((i = 0; i < ${{#specs[@]}}; i++)); do
        IFS='|' read -r src dst mode <<< "${{specs[$i]}}"
        for ((j = 0; j < ${{#specs[@]}}; j++)); do
            [ "$i" -eq "$j" ] && continue
            IFS='|' read -r src2 dst2 mode2 <<< "${{specs[$j]}}"
            [ "$mode2" = "ro" ] || continue
            case "${{dst}}/" in
                "${{dst2}}/"*)
                    echo "FATAL(runtime-link): 挂载目标 '$dst' 位于只读挂载 '$dst2' 之下 ——" >&2
                    echo "  dockerd 无法在只读文件系统上创建嵌套 mountpoint，会以 rc=125 失败。" >&2
                    echo "  修法：让挂载目标彼此互不嵌套（例如把子目录与它的父目录分开挂到同一层的兄弟位置）。" >&2
                    return 6
                    ;;
            esac
        done
    done
    return 0
}}

# 测试用的库模式：只定义上面的断言函数，不执行主流程（不读 manifest、不起容器）。
if [ "${{GRASPO_RUNNER_LIB_ONLY:-0}}" = "1" ]; then
    return 0 2>/dev/null || exit 0
fi

TIER="${{1:?usage: run_matrix54.sh <T###> [--dry-run]}}"
MODE="${{2:-run}}"
ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
IMAGE="${{GRASPO_IMAGE:-graspo-msswift:4.5.3}}"
RUN_ROOT="${{RUN_ROOT:-$ROOT_DIR/.local/matrix54-runs}}"
ELAM_HOST="${{{ELAM_HOST_ROOT_ENV}:-}}"
ELAM_MINI_PATH="${{{ELAM_MINI_JSONL_ENV}:-$ELAM_HOST/{ELAM_MINI_JSONL_RELATIVE}}}"
MODELS_ROOT="${{{MODELS_HOST_ROOT_ENV}:-}}"
MANIFEST="${{GRASPO_RUNNER_MANIFEST:-$ROOT_DIR/tests/e2e/matrix54_manifest.json}}"
# ↑ 清单路径可被 `GRASPO_RUNNER_MANIFEST` 覆盖：给"逐长度递增批"这类需要**单档清单**
#   的场景用（rig 只换清单、不换 runner 逻辑）；默认路径与行为逐字不变。
#   本行与 `run_matrix54.sh` 的同一行**逐字对应**（由 rig/verify_runner_parity.py 校验）。
PYBIN="${{PYTHON:-python3}}"
CONTAINER_PY="${{CONTAINER_PYTHON:-python}}"

# ── 产物属主：容器以**宿主跑批用户**运行（唯一真相源，§1.4）──────────────────────
# 缺陷（PG-13 / U5「产物跑完了测好了就删」执行不了）：镜像**没有 USER 指令** ⇒ 容器默认
# uid 0 运行 ⇒ 容器写出的产物一律 root:root，而宿主跑批账号对 root 文件**无 unlink 权限**
# ⇒ `rm` 报 Permission denied、`rmdir` 失败、228 上 `sudo -n` 要密码（228 实测：
# task-f2-resync-verify §⑨ 共 20 项 / 49,133,569 B ≈ 47 MiB 删不掉；task-e4-reclaim 旧树
# 522.7 MB 同因整树残留）。共享机上残留只增不减，而 228 磁盘已被顶到 100% 两次。
# ★ 修法是**防呆**（§2），不是"事后提权去删"（那治标不治本，且需 root 权限）：
#   让产物**从一开始**就归宿主跑批用户 —— 容器进程以宿主 uid:gid 运行。
# ★ 单一真相源（§1.4）：uid/gid 只在这里解析**一次**，DOCKER_ARGS 只引用 $RUN_UID/$RUN_GID；
#   容器内 HOME/缓存的落点、`USER`/`LOGNAME` 的中性值都是模块级常量
#   （generate_matrix.py::CONTAINER_HOME_DIR / CONTAINER_USER_NAME），不在本脚本里另写字面量。
# ★ 显式即防呆（§2.2）：dry-run 与真跑都打印生效用户；取不到合法 uid/gid ⇒ **fail-closed**
#   （exit 7），绝不静默回落成 root —— 静默回落 = 复现本条要根治的缺陷。
# ★ 回退（透明退路 §3.2，需显式声明且留痕）：GRASPO_RUN_AS_ROOT=1 ⇒ 仍以 root 运行，
#   但 stderr 打印 WARNING 并写 $RUN_ROOT/skipped_guards.log（可事后审计）。真实上机不得设置。
RUN_UID="${{GRASPO_RUN_UID:-$(id -u)}}"
RUN_GID="${{GRASPO_RUN_GID:-$(id -g)}}"
case "$RUN_UID" in ''|*[!0-9]*) RUN_UID="" ;; esac
case "$RUN_GID" in ''|*[!0-9]*) RUN_GID="" ;; esac
if [ -z "$RUN_UID" ] || [ -z "$RUN_GID" ]; then
    echo "FATAL(runtime-link): 无法确定宿主 uid/gid，拒绝启动 ——" >&2
    echo "  GRASPO_RUN_UID='${{GRASPO_RUN_UID:-}}' GRASPO_RUN_GID='${{GRASPO_RUN_GID:-}}'（须为非负整数）" >&2
    echo "  为什么 fail-closed：没有正确的 uid/gid 就只能以 root 跑容器，产物会属 root:root、" >&2
    echo "  宿主账号事后删不掉（PG-13 / U5「跑完即删」执行不了）——这正是不允许静默回落的原因。" >&2
    echo "  修法：确保 id -u / id -g 可用，或用 GRASPO_RUN_UID/GRASPO_RUN_GID 显式指定。" >&2
    exit 7
fi
if [ "${{GRASPO_RUN_AS_ROOT:-0}}" = "1" ]; then
    RUN_USER_ARGS=(--user "0:0")
    RUN_USER_DESC="0:0(root)"
    echo "WARNING(透明退路 §3.2): GRASPO_RUN_AS_ROOT=1 ⇒ 本次容器以 **root** 运行，产物将属 root:root，" >&2
    echo "  宿主普通用户事后**删不掉**（U5「跑完即删」执行不了、共享机残留不可回收）。真实上机不得设置本变量。" >&2
    mkdir -p "$RUN_ROOT"
    printf '%s run_as_root=1 tier=%s image=%s uid=%s gid=%s\\n' \\
        "$(date -Is 2>/dev/null || date)" "$TIER" "$IMAGE" "$RUN_UID" "$RUN_GID" \\
        >> "$RUN_ROOT/skipped_guards.log"
else
    RUN_USER_ARGS=(--user "$RUN_UID:$RUN_GID")
    RUN_USER_DESC="$RUN_UID:$RUN_GID"
fi

# ── 容器内**可写缓存/状态根**（**单一真相源**，§1.4）────────────────────────────
# 缺陷（228 实测确证，本包修的回归）：`043c776` 把容器改成以宿主 uid 运行，但只补了
# HOME/XDG_CACHE_HOME/TORCHINDUCTOR_CACHE_DIR/TRITON_CACHE_DIR，**漏了 MODELSCOPE_CACHE**。
# 镜像 ENV 把它写死成 `/mnt/workspace/.cache/modelscope/hub`，而 `/mnt` 在镜像里是
# `root:root 0755`（FHS 标准目录；docker/Dockerfile* 里没有任何 mkdir /mnt/workspace）
# ⇒ 非 root 建不出 `/mnt/workspace`。ms-swift 的 `swift/dataset/loader.py:57` 拿
# `modelscope...get_cache_dir()`（读 MODELSCOPE_CACHE）当 `datasets` 的 cache_dir
# ⇒ `datasets/builder.py` 的 os.makedirs 抛 `PermissionError: '/mnt/workspace'`：
# 训练**在数据集装载阶段**就死，宿主侧只看到 `[ckpt-retention] FATAL: 未找到任何 checkpoint-*`
# （因为训练一步都没跑）。临时绕过是手工派生镜像 `..-p1cachefix`（mkdir + chmod 1777），
# 那要求每台机器各自维护一个派生镜像——本段把它换成**干净镜像开箱可用**的正规修法。
#
# ★ 清单的**唯一真相源**在生成器（tests/e2e/generate_matrix.py::CONTAINER_CACHE_ROOTS）；
#   下面三处全部由它派生，本脚本不写任何路径字面量：
#     ① 本数组 → `docker run` 的 `-e` 注入（见下方 DOCKER_ARGS 段）；
#     ② entry.sh 的可写性预检（同源渲染，测试断言两份数组逐字节相同）；
#     ③ print_cache_roots：真跑与 `--dry-run` 都能看见**实际注入**了哪些根（§2.2 显式即防呆）。
# ★ 为什么缓存根全挂在 HOME 之下：`043c776` 已把 HOME 建到 /tmp（1777、容器 --rm 即消失）；
#   沿用同一个可写根，不另造第二个（§1.4），也不放进 /out（产物树，见 U5「跑完即删」）。
CONTAINER_CACHE_ROOTS=(
{cache_roots_array}
)
# ★ 显式即防呆（§2.2）：缓存根落到哪，决定 PermissionError 会不会在训练开始前出现 ——
#   不能只在出错后才知道。真跑与 dry-run 都调用本函数打印实际清单。
print_cache_roots() {{
    printf '[cache-roots] %s\\n' "${{CONTAINER_CACHE_ROOTS[@]}}"
}}

# 0) 从运行清单取出该档的 GPU / 配置 / 卡数 / 子集大小 / 容器内模型路径。
# 交接格式是**单行 JSON**：旧的"read 多个变量 < <(python …)"在 stdout 是管道时
# 会受 Python 块缓冲影响，字段可能错位（潜在隐患，本轮改为显式 JSON 交接）。
TIER_JSON=$("$PYBIN" - "$MANIFEST" "$TIER" <<'PY'
import json, sys
manifest = json.load(open(sys.argv[1], encoding="utf-8"))
tier = next(t for t in manifest["tiers"] if t["tier_id"] == sys.argv[2])
data = tier.get("data") or {{}}
print(json.dumps({{
    "gpus": ",".join(str(g) for g in tier["gpus"]),
    "config": tier.get("config"),
    "status": tier["status"],
    "cards": tier["cards"],
    "subset_size": data.get("subset_size") or 0,
    "train_path": data.get("train_path") or "-",
    "model_path": tier.get("model_path") or "-",
    "model_env": tier.get("model_host_dir_env_var") or "-",
    "verdict": (tier.get("feasibility") or {{}}).get("verdict") or "-",
}}, ensure_ascii=False))
PY
) || exit 5

read -r GPUS CONFIG_STATUS NPROC SUBSET TRAINPATH MODEL_PATH MODEL_ENV VERDICT \\
  < <("$PYBIN" - "$TIER_JSON" <<'PY'
import json, sys
data = json.loads(sys.argv[1])
print(
    data["gpus"],
    (data["config"] or "-") + "|" + data["status"],
    data["cards"],
    data["subset_size"],
    data["train_path"],
    data["model_path"],
    data["model_env"],
    data["verdict"],
)
PY
)
CONFIG="${{CONFIG_STATUS%%|*}}"
TIER_STATUS="${{CONFIG_STATUS#*|}}"

# 0a) **不可表达（blocked）档必须先于一切放行路径被拒**，包括 --dry-run。
# 顺序即语义：dry-run 是"预检"入口，预检把 blocked 档报成 rc=0 会让批处理
# 把不可跑的档当成可跑（历史缺陷 F-6）。因此这条检查排在 dry-run 分支**之前**。
if [ "$CONFIG" = "-" ]; then
    echo "FATAL: $TIER 当前不可表达（blocked），见 samples/configs/matrix54/$TIER.blocked.md" >&2
    exit 2
fi

# 0a-ter) **显存采样链路的前置校验（宿主侧，fail-closed）** —— W3 连带失效的修复。
#
#    record-gpu-memory 自 W3 起是**配置驱动**命令（§10.1）：gpu_monitor 段的
#    output_dir / tag / interval_sec 决定产物位置与内容，CLI 上已无
#    --output-dir / --tag / --interval-sec。
#    因此这里**不再自己拼参数**，只做三件事：
#      ① 从档配置读出 gpu_monitor.output_dir / tag / interval_sec（单一真相源）；
#      ② 把它与取证链锚点 {GPU_MONITOR_CONTAINER_DIR}（生成期注入，见 generate_matrix.py::
#         GPU_MONITOR_CONTAINER_DIR）比对，并**反向**断言它**不落在** training.output_dir 之内
#         —— 落在里面就会被训练侧的 overwrite（shutil.rmtree 整棵目录）删掉（228 实测）；
#      ③ 校不过就拒绝启动，绝不"先跑起来再说"。
#    为什么必须从档配置读、而不是在这里另写一份：产物位置与档号标签已经写在
#    **档配置自身**里；在这里再写一遍，同一条事实就有两份描述，迟早漂移（§1.4）。
#
#    容器内 /out 由下面的 -v "$RUN_DIR:/out" 绑定到宿主 <RUN_ROOT>/<T###>
#    （RUN_DIR = <RUN_ROOT>/<T###>），因此容器内 {GPU_MONITOR_CONTAINER_DIR}
#    就是宿主 <RUN_ROOT>/<T###>/gpu/ ——
#    正是**宿主采样摘要** gpu_memory_summary.json 的落点，由
#    collect_results.py::_read_host_sample_peak_memory 读入 `host_sample_peak_gib` 旁路。
#    ★ 它**不是** §7「实测每卡峰值显存」列的口径（该列现为**二分口径**：native 档读
#      rank0 allocator `max_allocated`、ms-swift 档读 `max_memory_reserved`，见 §9.1）；
#      宿主采样在共享机上会被同租户作业污染，一律不得混入该列。
#
#    ★ 读法只用 grep/sed（**不引 PyYAML**）：本脚本在宿主上跑，宿主 python 不一定
#      装了 PyYAML（它只在镜像里保证有）。档配置的 gpu_monitor 段由 generate_matrix.py
#      生成，形状固定（段名行 + 每行两空格缩进的 KEY: VALUE），故按形状读；读不到即
#      fail-closed —— 绝不"读不到就当默认值"。
#
#    ★ 三条不变量（改档配置生成逻辑或改本段之前，先读 scripts/collect_results.py）：
#       a. tag == 档号：摘要里的 tag 与档号一致 ⇒ 读数可归属到档；
#       b. 产物落点 == {GPU_MONITOR_CONTAINER_DIR} == 宿主 <RUN_ROOT>/<T###>/gpu/，
#          且**不在** <training.output_dir> 之内：取证链不断、训练清洗不伤证据；
#       c. interval_sec == {GPU_MONITOR_INTERVAL_SEC:g}：与旧 CLI 口径逐值相同（样本条数不变）。
TIER_CONFIG_PATH="$ROOT_DIR/$CONFIG"
# 取 gpu_monitor 段内某字段的标量值（不存在则输出空）。
read_gpu_monitor_field() {{
    sed -n '/^gpu_monitor:$/,/^[^ #]/p' "$TIER_CONFIG_PATH" | sed -n "s/^  $1:[[:space:]]*\\([^[:space:]#]*\\).*/\\1/p" | head -n 1
}}
# 取 training 段的 output_dir（用于交叉校验落点）。
read_training_output_dir() {{
    sed -n '/^training:$/,/^[^ #]/p' "$TIER_CONFIG_PATH" | sed -n 's/^  output_dir:[[:space:]]*\\([^[:space:]#]*\\).*/\\1/p' | head -n 1
}}
GPU_MONITOR_OUTPUT_DIR="$(read_gpu_monitor_field output_dir)"
GPU_MONITOR_TAG="$(read_gpu_monitor_field tag)"
GPU_MONITOR_INTERVAL_SEC="$(read_gpu_monitor_field interval_sec)"
TRAINING_OUTPUT_DIR="$(read_training_output_dir)"

monitor_problems=""
if [ -z "$CONFIG" ] || [ ! -f "$TIER_CONFIG_PATH" ]; then
    monitor_problems="${{monitor_problems}}档配置不可读：$TIER_CONFIG_PATH;"
fi
if [ -z "$GPU_MONITOR_OUTPUT_DIR" ]; then
    monitor_problems="${{monitor_problems}}gpu_monitor.output_dir 为空;"
fi
if [ "$GPU_MONITOR_TAG" != "$TIER" ]; then
    monitor_problems="${{monitor_problems}}gpu_monitor.tag='$GPU_MONITOR_TAG' 与档号 '$TIER' 不一致;"
fi
if [ -z "$GPU_MONITOR_INTERVAL_SEC" ]; then
    monitor_problems="${{monitor_problems}}gpu_monitor.interval_sec 缺失;"
fi
if [ -z "$TRAINING_OUTPUT_DIR" ]; then
    monitor_problems="${{monitor_problems}}training.output_dir 为空;"
elif [ "$GPU_MONITOR_OUTPUT_DIR" != "{GPU_MONITOR_CONTAINER_DIR}" ]; then
    monitor_problems="${{monitor_problems}}gpu_monitor.output_dir='$GPU_MONITOR_OUTPUT_DIR' 与取证链锚点 {GPU_MONITOR_CONTAINER_DIR} 不一致;"
fi
# ★ 反向不变量（2026-09-20 真机确证的缺陷，228 两次真跑同一签名）：落点**不得**落在
#   training.output_dir 之内 —— 训练侧 `overwrite_output_dir: true` 会对非空的该目录
#   执行 `shutil.rmtree`，把采样器刚建好的 gpu/ 连 jsonl 一起删掉（首错
#   `gpu_monitor.py:621 append_jsonl ... '/out/T010/gpu/gpu_memory.jsonl'`）。
#   这是**防线**（§2.3）：违法即 fail-closed，不给"跑起来再说"的机会。
case "$GPU_MONITOR_OUTPUT_DIR" in
    "$TRAINING_OUTPUT_DIR"/*)
        monitor_problems="${{monitor_problems}}gpu_monitor.output_dir='$GPU_MONITOR_OUTPUT_DIR' 落在训练输出目录 '$TRAINING_OUTPUT_DIR' 之内 —— 训练侧 overwrite（shutil.rmtree 整棵目录）会删掉采样产物;" ;;
esac
case "$GPU_MONITOR_OUTPUT_DIR" in
    /out/*) ;;
    *) monitor_problems="${{monitor_problems}}gpu_monitor.output_dir='$GPU_MONITOR_OUTPUT_DIR' 不在容器内 /out/ 之下;" ;;
esac
if [ -n "$monitor_problems" ]; then
    echo "FATAL(monitor-link): 档 $TIER 的显存采样链路前置校验未通过 ——" >&2
    printf '%s\n' "$monitor_problems" | tr ';' '\n' | sed '/^$/d; s/^/  - /' >&2
    echo "  修法：改 tests/e2e/generate_matrix.py::build_config 的 gpu_monitor 段并重新生成" >&2
    echo "  档配置（手改档配置无效——该文件由生成器写出）。" >&2
    exit 5
fi
# 宿主侧落点：把容器内 <output_dir> 映射回宿主（容器内 /out = 宿主 <RUN_ROOT>/<T###>）。
RUN_GPU_DIR="$RUN_ROOT/$TIER${{GPU_MONITOR_OUTPUT_DIR#/out}}"
echo "[monitor-link] OK: tier=$TIER tag=$GPU_MONITOR_TAG interval_sec=$GPU_MONITOR_INTERVAL_SEC"
echo "[monitor-link]   容器内 $GPU_MONITOR_OUTPUT_DIR -> 宿主 $RUN_GPU_DIR"

# 0a-bis) **待上机测量**档（OPD：教师侧显存未测算，P-25）：允许跑（跑就是为了量出来），
# 但必须显式提示——并把"教师侧峰值显存"记进产物，否则这一档的可行性永远无据。
# 判据：manifest 的 `feasibility.verdict == unmeasured`（不是"没配方"，两者语义不同）。
if [ "$VERDICT" = "unmeasured" ]; then
    echo "WARN(待上机测量): $TIER 的额外显存消费者（OPD 教师）**未测算** ——" >&2
    echo "     本次跑的产物里必须单独记录**教师侧**与**学生侧**峰值显存，回填后才能写「可行」。" >&2
    echo "     参考：教师 27B bf16 权重 ≈ 50.3 GiB/卡（不换出时）；本档配方已 offload_teacher_model=true。" >&2
fi

# 0b) 环境前置：数据根与模型根都必须显式给出（两者都是环境信息，见 .local/）。
if [ -z "$ELAM_HOST" ]; then
    echo "FATAL(runtime-link): 数据未挂载 —— 请先 export {ELAM_HOST_ROOT_ENV}=<宿主 ELAM V5 数据根目录>" >&2
    exit 3
fi
if [ -z "$MODELS_ROOT" ]; then
    echo "FATAL(runtime-link): 模型未挂载 —— 请先 export {MODELS_HOST_ROOT_ENV}=<宿主模型根目录>" >&2
    echo "  容器内 {MODELS_CONTAINER_ROOT} 的宿主来源；模型不在镜像里，必须挂载。" >&2
    exit 4
fi

# 2) 宿主侧锁卡守卫（fail-closed；与容器内 train_worker 是同一实现）
"$PYBIN" "$ROOT_DIR/scripts/gpu_lock_guard.py" --visible "$GPUS" || exit 1

# 2a) 宿主侧**目标卡实测空闲断言**（F-10，fail-closed）：逐卡实测
#     memory.used ≤64 MiB 且 utilization.gpu ≤5%；任一卡被占即拒绝启动。
#     本轮实战里这道断言拦下过含被第三方占用的 GPU3 的 4 卡目标集。
#     宁等不抢：不 kill 他人进程，改选实测空闲的卡或等它空下来。
#     GRASPO_SKIP_IDLE_ASSERT=1 仅供无 GPU 的脚手架自检（如假 docker 的 dry-run 回归），
#     真实上机**不得**设置——跳过即失去这道防线，属显式降级。
if [ "${{GRASPO_SKIP_IDLE_ASSERT:-0}}" != "1" ]; then
    "$PYBIN" "$ROOT_DIR/scripts/gpu_idle_assert.py" --visible "$GPUS" || exit 1
else
    # 透明退路（宪法 §3.2）：跳过防线必须留痕，不能只在日志里"没有输出"。
    # 两处痕迹：① stderr 的显式 WARNING；② **环境指纹/产物** `skipped_guards.log`，
    # 使"这次运行的 F-10 防线被关过"可事后审计。
    echo "WARNING(透明退路 §3.2): GRASPO_SKIP_IDLE_ASSERT=1 ⇒ 本次运行**已跳过** F-10 目标卡空闲断言（GPU ${{GPUS}} 未做实测空闲校验）；真实上机不得设置本变量。" >&2
    mkdir -p "$RUN_ROOT"
    printf '%s skip_idle_assert=1 tier=%s gpus=%s image=%s\\n' \\
        "$(date -Is 2>/dev/null || date)" "$TIER" "$GPUS" "$IMAGE" \\
        >> "$RUN_ROOT/skipped_guards.log"
fi

# 2b) **模型挂载 fail-closed 前置断言**（与数据侧同一模式）。
# 为什么必须在 docker run **之前**：模型不在获批镜像里，配置却指向容器内
# {MODELS_CONTAINER_ROOT}/<模型目录名>；若缺失，容器会以"找不到模型目录"退出，
# 而 collect_results.py 的日志分类器会把 FileNotFoundError/No such file or directory
# 归成「数据问题」且 counts_toward_max_context=False ⇒ 真因（运行链路缺模型）
# 会被记成一次普通失败。这里先拒绝启动并把原因标成**运行链路错误**。
MODEL_DIR_NAME="${{MODEL_PATH#{MODELS_CONTAINER_ROOT}/}}"
if [ "$MODEL_PATH" != "{MODELS_CONTAINER_ROOT}/$MODEL_DIR_NAME" ] || [ -z "$MODEL_DIR_NAME" ]; then
    echo "FATAL(runtime-link): 模型未挂载 —— 配置的 model_path 不在 {MODELS_CONTAINER_ROOT}/ 下：$MODEL_PATH" >&2
    exit 4
fi
MODEL_HOST_DIR="${{!MODEL_ENV:-$MODELS_ROOT/$MODEL_DIR_NAME}}"
if [ ! -d "$MODEL_HOST_DIR" ]; then
    echo "FATAL(runtime-link): 模型未挂载 —— 宿主模型目录不存在：$MODEL_HOST_DIR" >&2
    echo "  期望目录名：$MODEL_DIR_NAME（容器内路径 $MODEL_PATH；模型不在镜像里，必须挂载）" >&2
    echo "  模型根 {MODELS_HOST_ROOT_ENV}=$MODELS_ROOT；可用 $MODEL_ENV=<该档模型目录> 单点覆盖。" >&2
    echo "  ⚠ 这是**运行链路错误**，不是数据问题；修好挂载后再跑，不要记为数据问题。" >&2
    exit 4
fi

# 1) **由配置自身反推挂载目的地**（单一真相源，不硬编码目的地）：
#    相对媒体路径的锚点口径（native 与 ms-swift **共用同一条**）是"训练 JSONL 的父目录"
#    （native: flow/trainer/sft_trainer.py 的 Path(train_path).parent；
#      ms-swift: flow/msswift/dataset.py. 两处说的是同一件事，本脚本只认这一条）。
#    ELAM V5 的样本把图像写作 `../images/<file>.jpg` ⇒ 容器内必须让
#    `dirname(train_path)/../images` 可达。因此：
#      TRAIN_PATH_DIR_CONTAINER = dirname(train_path)          ← 子集的宿主目录挂到这里
#      IMAGE_DIR_CONTAINER      = dirname(dirname(train_path))/images  ← 图像根挂到这里
#    布局若变（如 train_path 移到 <根>/data/v2/xxx.jsonl），这两处**自动跟随**，
#    不需要改本脚本——这就是"不再断裂"的原因；另有容器内媒体预检兜底（见 4) 段）。
TRAIN_PATH="$TRAINPATH"
TRAIN_PATH_DIR_CONTAINER="$(dirname "$TRAIN_PATH")"
IMAGE_DIR_CONTAINER="$(dirname "$TRAIN_PATH_DIR_CONTAINER")/images"
if [ "$TRAIN_PATH" != "$TRAIN_PATH_DIR_CONTAINER/$TIER.jsonl" ]; then
    echo "FATAL(runtime-link): 数据子集路径与配置不一致 ——" >&2
    echo "  配置 data.train_path=$TRAIN_PATH，但本档生成的子集文件名是 $TIER.jsonl" >&2
    echo "  （runner 只生成 <RUN_ROOT>/$TIER/subsets/$TIER.jsonl）。" >&2
    echo "  ⚠ 这是**运行链路错误**：容器里打开的数据文件与实际产物不是同一个。" >&2
    exit 3
fi

# 1b) **媒体锚点可达性（宿主侧，fail-closed）**：图像根缺失时立刻拒绝启动。
#     若不拦，训练会以 FileNotFoundError / PIL.UnidentifiedImageError 退出，
#     被 collect_results.py 的日志分类器记成「数据问题」，真因（运行链路缺图像根）被掩盖。
#     ⚠ 不用符号链接兜底：符号链接在容器内指向宿主路径，必然断开（r3 实测踩过）。
if [ ! -d "$ELAM_HOST/images" ]; then
    echo "FATAL(runtime-link): 媒体锚点不可达 —— 宿主图像根不存在：$ELAM_HOST/images" >&2
    echo "  锚点口径：相对媒体路径相对训练 JSONL 的父目录解析 ⇒ 容器内需要" >&2
    echo "  <dirname(train_path)>/../images = $IMAGE_DIR_CONTAINER" >&2
    echo "  宿主来源：$ELAM_HOST/images（环境变量 {ELAM_HOST_ROOT_ENV}）。" >&2
    echo "  ⚠ 这是**运行链路错误**，不是数据问题；补齐图像根后再跑。" >&2
    exit 3
fi

if [ "$MODE" = "--dry-run" ]; then
    echo "[dry-run] $TIER gpus=$GPUS nproc=$NPROC status=$TIER_STATUS image=$IMAGE"
    # ★ 生效用户必须**在 dry-run 里就可见**（§2.2 显式即防呆）：产物属主是"跑完即删"
    #   （PG-13 / U5）能否执行的前提，而它由下面 docker run 的 --user 决定 ⇒ 预检就得能看见。
    echo "[dry-run] user=$RUN_USER_DESC home={CONTAINER_HOME_DIR}（容器进程以该 uid:gid 运行 ⇒ 产物属主 = 宿主跑批用户）"
    # ★ 缓存根清单**必须在 dry-run 里就可见**（§2.2 显式即防呆）：镜像里写死到 root 可写
    #   路径的缓存根（MODELSCOPE_CACHE=/mnt/workspace/...）正是"训练在数据集装载阶段就死"
    #   的根因 ⇒ 预检就得能看见它被指到了哪里，而不是等 rank0 PermissionError 再回头查。
    echo "[dry-run] cache-roots（将逐个以 -e 注入容器；清单唯一真相源见 generate_matrix.py::CONTAINER_CACHE_ROOTS）:"
    print_cache_roots
    echo "[dry-run] config=$CONFIG train_subset=$SUBSET 条 -> $TRAINPATH"
    echo "[dry-run] models=$MODELS_ROOT:$MODEL_DIR_NAME -> {MODELS_CONTAINER_ROOT}（只读）"
    echo "[dry-run] subsets=$RUN_ROOT/$TIER/subsets -> $TRAIN_PATH_DIR_CONTAINER（只读）"
    echo "[dry-run] images=$ELAM_HOST/images -> $IMAGE_DIR_CONTAINER（只读；媒体锚点）"
    exit 0
fi

RUN_DIR="$RUN_ROOT/$TIER"
mkdir -p "$RUN_DIR/subsets"

# 3) 生成训练子集：只取门槛下限（SFT ≥100 / RL ≥20），不整集跑。
# 3a) **读取源**（← 与 `data.train_path` 无关）：默认 mini 数据集（跑通用），
#     可用 `{ELAM_MINI_JSONL_ENV}` 覆盖；切回源数据见文件头「数据」段注释。
if [ ! -f "$ELAM_MINI_PATH" ]; then
    echo "FATAL: 训练子集读取源不存在：$ELAM_MINI_PATH" >&2
    echo "  默认取 mini 数据集 $ELAM_HOST/{ELAM_MINI_JSONL_RELATIVE}（只用于跑通）；" >&2
    echo "  要改回源数据：export {ELAM_MINI_JSONL_ENV}=$ELAM_HOST/data/train.jsonl" >&2
    exit 3
fi
head -n "$SUBSET" "$ELAM_MINI_PATH" > "$RUN_DIR/subsets/$TIER.jsonl"
LINES=$(wc -l < "$RUN_DIR/subsets/$TIER.jsonl")
if [ "$LINES" -lt "$SUBSET" ]; then
    echo "FATAL: subset too small: $LINES < $SUBSET" >&2
    exit 3
fi

# 4) 容器内媒体预检脚本（写在运行目录里，经 /out 只读进容器；训练前先跑）。
#    它是**断言**不是第二套解析器：只验证"训练器将要打开的绝对路径确实存在"。
#    锚点口径与两个后端共用同一条：训练 JSONL 的父目录。
cat > "$RUN_DIR/preflight_media.py" <<'PY'
#!/usr/bin/env python3
# 媒体锚点可达性前置断言（容器内，训练之前；只读、不改写任何路径）。
#
# 锚点口径（全项目单一真相源，native 与 ms-swift 共用同一条）：
#     训练 JSONL 里的相对媒体路径，相对于该 JSONL 文件所在目录解析。
# 本脚本验证"训练器将要打开的绝对路径在容器内确实存在"，因此布局漂移会在训练
# 启动前以可操作报错 fail-closed；否则 ms-swift / native 会在数据准备阶段抛
# FileNotFoundError / PIL.UnidentifiedImageError，被日志分类器记成「数据问题」，
# 把运行链路缺口的真因掩盖掉。

import json
import os
import sys

MEDIA_KEYS = ("image", "path", "url", "video")
MEDIA_TYPES = ("image", "image_url", "video", "video_url")
MAX_EXAMPLES = 5


def is_relative(value):
    return not value.startswith(("http://", "https://", "/", "data:"))


def iter_media_paths(raw):
    # 产出该样本里的媒体路径（读的是训练侧同一个字段：messages[*].content[*]）。
    messages = raw.get("messages")
    if not isinstance(messages, list):
        return
    for message in messages:
        if not isinstance(message, dict):
            continue
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for block in content:
            if not isinstance(block, dict):
                continue
            if str(block.get("type") or "").lower() not in MEDIA_TYPES:
                continue
            for key in MEDIA_KEYS:
                value = block.get(key)
                if isinstance(value, str):
                    yield value


def main():
    train_path = sys.argv[1]
    if not os.path.isabs(train_path):
        train_path = os.path.join(os.getcwd(), train_path)
    anchor = os.path.dirname(train_path)
    checked = 0
    blocks_found = 0
    missing = []
    shape_mismatch = 0
    with open(train_path, "r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                raw = json.loads(line)
            except ValueError as exc:
                print("[preflight] FATAL(runtime-link): %s:%d 不是合法 JSON: %s" % (train_path, line_no, exc))
                return 3
            if not isinstance(raw, dict):
                continue
            found_here = 0
            for value in iter_media_paths(raw):
                found_here += 1
                if not is_relative(value):
                    # 绝对路径 / http(s) / data: 不属本预检范围（训练侧同样跳过）。
                    continue
                checked += 1
                resolved = os.path.join(anchor, value)
                if not os.path.exists(resolved):
                    if len(missing) < MAX_EXAMPLES:
                        missing.append("%s:%d -> %s" % (train_path, line_no, resolved))
            blocks_found += found_here
            if found_here == 0 and '"image"' in line:
                # 文本里出现 image 字段却按契约解析不到任何媒体块 ⇒ 形状变了。
                # 不做静默放行：预检若"什么都没查"就等于没有防线。
                shape_mismatch += 1
    if missing:
        print("[preflight] FATAL(runtime-link): 媒体锚点不可达 —— 相对媒体路径解析后不存在。")
        print("[preflight]   锚点口径：相对路径相对训练 JSONL 的父目录解析；anchor=%s" % anchor)
        print("[preflight]   共检查 %d 条相对媒体路径，缺失示例：" % checked)
        for item in missing:
            print("[preflight]     %s" % item)
        print("[preflight]   这是运行链路错误（挂载/布局），不是数据问题。")
        print("[preflight]   修法：确认图像根被挂到 dirname(dirname(train_path))/images。")
        return 3
    if shape_mismatch:
        print("[preflight] FATAL(runtime-link): 预检解析器与数据形状不一致 ——")
        print("[preflight]   %d 行含 'image' 字段，但按 messages[*].content[*] 契约解析不到任何媒体块。" % shape_mismatch)
        print("[preflight]   不做静默放行（否则预检形同虚设）：请同步本脚本与数据契约。")
        return 3
    print("[preflight] media anchor OK: %d 条相对媒体路径在 %s 下全部可达" % (checked, anchor))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
PY

PORT=$(( 29500 + RANDOM % 400 ))
# ★ 容器内 torch 重载探测脚本（A3 转正路径的**生产端**）：与 entry.sh 同源产出到
#   $RUN_DIR/torch_probe.py（$RUN_DIR 已整体绑定到容器 /out ⇒ 容器内即
#   {TORCH_PROBE_SCRIPT_CONTAINER_PATH}，**无需新增挂载项**）。内容由
#   tests/e2e/generate_matrix.py::render_probe_script 渲染（契约字面量取自
#   TORCH_PROBE_SCHEMA / TORCH_PROBE_FILENAME 单一真相源）。
cat > "$RUN_DIR/torch_probe.py" <<'TORCH_PROBE_PY'
{probe_script}
TORCH_PROBE_PY
cat > "$RUN_DIR/entry.sh" <<ENTRY
set -o pipefail
# ★ 显存取证目录（容器内**唯一**落点，§1.4）：与档配置 gpu_monitor.output_dir、runner 的
#   前置校验、collect_results 读的 <run_dir>/gpu/ 同源 —— 唯一字面量在
#   tests/e2e/generate_matrix.py::GPU_MONITOR_CONTAINER_DIR，生成期注入下面这一行。
#   ★ 为什么必须先建出来：收口段的三处写盘点（gpu_summary.state ×2、
#     EVIDENCE_GAP_gpu_memory_summary）都在这个目录下；目录不存在时那些登记会 ENOENT，
#     "取证缺口"连痕迹都留不下（228 实测 entry.sh 行 197/200：没有那个文件或目录）。
#   ★ 为什么落点在 RUN_DIR 直下而不在训练输出目录里：训练侧 overwrite_output_dir 会
#     shutil.rmtree 整棵 training.output_dir（228 实测把采样器刚建好的 gpu/ 删掉）。
#   ⚠ 写法约束：本段是**不加引号的 heredoc 正文** ⇒ 引用内层变量一律写 \\$VAR（外层不
#     展开、留给容器内展开），注释里同样不得出现裸美元符/裸反引号（见下方哨兵测试）。
GPU_EVIDENCE_DIR="{GPU_MONITOR_CONTAINER_DIR}"
mkdir -p "\\$GPU_EVIDENCE_DIR"
# ★ 容器内**可写缓存/状态根**：先逐个建好，再逐个**预检可写性**（§2.3 边界校验即防呆）。
#   清单与本文件顶部的 CONTAINER_CACHE_ROOTS 段**同源**（两者都由生成器的
#   tests/e2e/generate_matrix.py::CONTAINER_CACHE_ROOTS 渲染，测试断言逐字节相同）。
#   为什么必须显式建：容器进程以**宿主 uid** 运行（见 runner 的「产物属主」段），
#   镜像里 root 的 HOME 与镜像 ENV 的 MODELSCOPE_CACHE（=/mnt/workspace/.cache/modelscope/hub，
#   而 /mnt 是 root:root 0755）对这个 uid 都**不可写** ⇒ HF datasets / modelscope 数据集缓存 /
#   triton·inductor 编译缓存在**训练开始前**就 PermissionError（228 实测：rank0 exit 1，
#   宿主侧只看到"未找到任何 checkpoint-*"，因为训练一步都没跑）。
#   ★ 预检的判据（本包要的"具体是哪个路径不可写"）：训练之前就把**变量名 + 路径**指名道姓
#     地报出来（FATAL(cache-root)），而不是让 dataloader 在深处抛一个不含变量名的
#     PermissionError；随后 fail-closed（exit 8），绝不静默放行到训练。
CONTAINER_CACHE_ROOTS=(
{cache_roots_array}
)
for CACHE_KV in "\\${{CONTAINER_CACHE_ROOTS[@]}}"; do
    CACHE_DIR="\\${{CACHE_KV#*=}}"
    if ! mkdir -p "\\$CACHE_DIR" 2>/dev/null || [ ! -w "\\$CACHE_DIR" ]; then
        echo "FATAL(cache-root): 容器内缓存根不可写: \\$CACHE_KV" >&2
        echo "  容器进程的 --user = $RUN_USER_DESC；HOME=\\$HOME" >&2
        echo "  为什么 fail-closed：缓存根不可写 ⇒ 数据集/编译缓存落盘时 PermissionError，" >&2
        echo "  那是「训练还没开始」的假失败（产物里只会看到「未找到任何 checkpoint-*」）。" >&2
        echo "  修法：该变量须由 runner 的 CONTAINER_CACHE_ROOTS 指向宿主用户可写的目录。" >&2
        exit 8
    fi
done
"$CONTAINER_PY" /out/preflight_media.py "$TRAIN_PATH" || exit 3
"$CONTAINER_PY" -m graspo record-gpu-memory --idle-only || exit 1
# ★ 显存采样：**配置驱动**（宪法 §10.1；W3 起 CLI 上已无 --output-dir/--tag/--interval-sec，
#   旧写法会 argparse 报错 rc=2）。本档配置（与训练用的是**同一份** YAML）自带
#   gpu_monitor 段：output_dir=/out/gpu（= 上文 GPU_EVIDENCE_DIR = 取证链锚点
#   GPU_MONITOR_CONTAINER_DIR；**不在** training.output_dir 之内，故训练侧 overwrite
#   的 rmtree 删不到它）、tag=<档号>、interval_sec=2 —— 产物落点与档号
#   标签仍由同一份档位配置描述，不在这里另写一份（§1.4 单一真相源）。
#   ⚠ 写法约束（heredoc 展开语义，见 task-ckpt-retention-fix 报告）：外层 heredoc 分隔符
#     不带引号 ⇒ 外层先展开一次、容器内再展开一次。CONFIG 是宿主 runner 的变量（容器内
#     不存在），必须留给**外层**展开，故保持裸美元符；写成反斜杠转义会把字面量 CONFIG
#     留给内层，容器内展开为空 ⇒ 命令 fail-closed（rc=2）。
"$CONTAINER_PY" -m graspo record-gpu-memory --config "/workspace/graspo/$CONFIG" &
SAMPLER=\\$!
torchrun --standalone --nproc_per_node="$NPROC" --master_port="$PORT" \\
  -m graspo.cli.train_worker --config "/workspace/graspo/$CONFIG" > /out/stdout.log 2>&1
RC=\\$?
# ★ 可注入的 worker 退出码探针（默认 0 ⇒ 不参与判定，行为与不设时逐字相同）：
#   只在容器内注入 GRASPO_FAKE_WORKER_RC 时生效，供"exit_code 不得假报 0"的负向用例
#   在**假容器**下真跑整条 entry.sh（不需要 GPU/docker）。默认值 0 表示"不覆盖 torchrun
#   的真实退出码"——§2 显式即防呆：不设即无副作用，且不会把真实 rc 悄悄改掉。
if [ -n "\\${{GRASPO_FAKE_WORKER_RC:-}}" ]; then RC="\\$GRASPO_FAKE_WORKER_RC"; fi
# 产物放开读权限（**保留**；改用宿主 uid 运行后它从"补丁"降级为"兜底"）：
# 历史缺陷：容器以 root 运行 ⇒ ms-swift/HF 默认落 root:0600 ⇒ 宿主侧的
# collect_results.py（普通用户）**打不开** checkpoint 权重 ⇒ A2/A3 读不到证据
# （228 实测踩过：A3 一度被记成"checkpoint 无法重新加载"，属方向性错误）。
# 现在容器以**宿主 uid:gid** 运行（见 runner 的「产物属主」段）⇒ 产物属主已经正确、
# 读权限也天然成立；本行留着覆盖两类残余情形：① 运行前目录里已有的**他人/root 旧产物**
# （对它们 chmod 会 EPERM，故只把 stderr 丢弃、不因它们报错）；② umask 偏严的环境。
# 产物必须可被宿主侧的收集/复现链路读取（§6）；chmod 只改元数据、不碰内容。
chmod -R a+rX /out 2>/dev/null || echo "WARN: chmod -R a+rX /out failed — 宿主侧可能读不到产物" >&2
# ── ckpt 保留策略（落盘后立即执行，§2.4 具名清理；**通配符只用于 find 的 -name 匹配，不作为删除操作数**）──
# 为什么在容器内做：保留动作必须在**权重刚落盘、还没被宿主侧收集**的窗口内完成，
# 且要与训练进程用**同一套** find/mtime 视角（§1.4）。容器以宿主 uid 运行后，
# 宿主侧其实也删得掉中间段 ckpt；但放在这里仍是"落盘后立即收缩"的唯一窗口。
# （历史背景：原实现依赖"容器内是 root ⇒ 只有这里删得掉"——宿主普通用户对 root:root
#   文件无 unlink 权限，228 实测 sudo 需密码。该依赖已随 --user 修复消除，逻辑不变。）
# ★ 写法约束（防呆）：本块**只使用内层 shell 自己的变量**（\\$1/\\$@/函数内局部量），
#   不读写外层 runner 的任何变量，也不内联任何函数调用。这样"外层展开"与"内层展开"
#   在语义上重合，不会因为外层 set -u 误判内层变量（228/本机都踩过这个坑）。
#   ★ 唯一的例外是档号 \\$TIER：它由**外层 docker run -e "\\$TIER" 显式注入**容器，
#     不是"恰好未定义⇒空串"的巧合（那种隐性依赖是脆弱设计，§2 显式即防呆）。
retain_single_checkpoint() {{
    local output_root="\\${{1:-}}"
    local -a all=() ckpts=()
    local path step candidate="" newest_path="" newest_step="" deleted=0 keep_count=0

    if [ ! -d "\\$output_root" ]; then
        return 0
    fi
    mapfile -t all < <(find "\\$output_root" -type d -name 'checkpoint-*' -print 2>/dev/null)
    if [ "\\${{#all[@]}}" -eq 0 ]; then
        # ★ native 后端的产物形态**不是** checkpoint-*，而是终态目录 final/。
        #   权威定义（§1.4 单一真相源，不在这里另立一套；注意本段是 heredoc 正文，
        #   照防呆约束**不写裸反引号/裸美元符**）：
        #     · scripts/collect_results.py 的 NATIVE_CHECKPOINT_DIRNAME = "final"
        #       （find_checkpoint_dirs 对 native 就用 rglob("final") 找）；
        #     · src/graspo/flow/trainer/trainer.py 的 _save_checkpoint(output_dir / "final")。
        #   native 只落**一份**终态 ⇒ **没有"中间段"要删除**，但它**确实有产物** ⇒
        #   不得记 missing（那是"假失败"：训练成功却 rc=4，会污染台账，
        #   见 2026-09-20 复核 N2）。
        #   ★ 只承认**含至少一个文件**的 final/：空目录 = 产物根本没落盘 ⇒
        #     仍然走下面的 missing 硬失败（三态语义**不放松**）。
        local -a finals=() final_files=()
        mapfile -t finals < <(find "\\$output_root" -maxdepth 2 -type d -name 'final' -print 2>/dev/null)
        for candidate in "\\${{finals[@]}}"; do
            mapfile -t final_files < <(find "\\$candidate" -type f -print -quit 2>/dev/null)
            if [ "\\${{#final_files[@]}}" -gt 0 ]; then
                echo "[ckpt-retention] KEEP   \\$candidate（native 终态产物；本档无 checkpoint-* 中间段可清理）"
                echo "[ckpt-retention] 保留 ckpt 数=1 删除数=0"
                return 0
            fi
        done
        echo "[ckpt-retention] FATAL: 未找到任何 checkpoint-* 目录（output_root=\\$output_root）——" >&2
        echo "[ckpt-retention]   A3（checkpoint 重载）将无证据可判。这是**运行链路错误**，" >&2
        echo "[ckpt-retention]   不是训练失败：请检查 save_steps 是否大于实际优化步数。" >&2
        return 4
    fi
    for path in "\\${{all[@]}}"; do
        if [ -f "\\$path/trainer_state.json" ]; then
            ckpts+=("\\$path")
        else
            echo "[ckpt-retention] WARN: 跳过（缺 trainer_state.json）\\$path" >&2
        fi
    done
    if [ "\\${{#ckpts[@]}}" -eq 0 ]; then
        ckpts=("\\${{all[@]}}")
    fi
    # 只留**最新一份**（每档一份：manifest runtime.checkpoint_retention.keep_per_tier）。
    # ★ 2026-09-20 修复"跨 run 误删**本轮最新**"：原判据**只比步数**
    #   （[ "\\$step" -gt "\\$newest_step" ]，**严格大于**）。同一档**第二次跑**时，
    #   上一轮 v0-*/checkpoint-160 与本轮 v1-*/checkpoint-160 步数相同 ⇒
    #   keeper 完全由 find 的 readdir 顺序决定（ext4 dir_index / overlayfs / tmpfs 各不相同）；
    #   顺序不利时**本轮最新**被当"中间段"删除（228 实测 T043 run2：v1-*/checkpoint-160
    #   消失、last-checkpoint 断链 ⇒ 本轮的 A2/A3 证据丢失，而 ckpt_retention.state 仍报 ok）。
    #   本机复现与负向用例见工位 task-retention-xrun/evidence/。
    #   现判据 = **三元全序**，与 find 的输出顺序**无关**（同一份产物在任何文件系统上结论一致）：
    #     ① mtime 大者胜 —— "最近被写入"的那份 = **本轮**训练最后落盘的那份
    #        （同一 run 内步数单调递增 ⇒ 与"步数最大"同解；跨 run 时它正确表达"本轮优先"）；
    #     ② mtime 平手 ⇒ 步数大者胜；③ 仍平手 ⇒ 路径字典序大者胜（v1-* > v0-*）。
    # ⚠ 本块是**不加引号的 heredoc 正文**（外层 set -uo pipefail）⇒ 本段的注释里
    #   不得出现裸反引号（外层会当命令替换真的执行）或裸美元符（外层会当变量展开）；
    #   哨兵测试见 test_run_matrix54_runner.py 的 retention 段哨兵。
    for path in "\\${{ckpts[@]}}"; do
        step="\\${{path##*-}}"
        case "\\$step" in
            ''|*[!0-9]*) continue ;;
        esac
        if [ -z "\\$newest_path" ]; then
            newest_path="\\$path"; newest_step="\\$step"
        elif [ "\\$path" -nt "\\$newest_path" ]; then
            newest_path="\\$path"; newest_step="\\$step"
        elif [ ! "\\$newest_path" -nt "\\$path" ]; then
            if [ "\\$step" -gt "\\$newest_step" ]; then
                newest_path="\\$path"; newest_step="\\$step"
            elif [ "\\$step" -eq "\\$newest_step" ] && [ "\\$path" \\> "\\$newest_path" ]; then
                newest_path="\\$path"; newest_step="\\$step"
            fi
        fi
    done
    if [ -z "\\$newest_path" ]; then
        echo "[ckpt-retention] WARN: ckpt 目录名不含步数，跳过保留动作（不猜、不删）" >&2
        return 0
    fi
    for path in "\\${{ckpts[@]}}"; do
        if [ "\\$path" = "\\$newest_path" ]; then
            keep_count=\\$((keep_count + 1))
            echo "[ckpt-retention] KEEP   \\$path"
            continue
        fi
        echo "[ckpt-retention] DELETE \\$path（中间段 ckpt；A3 只需最新一份）"
        if ! rm -rf -- "\\$path"; then
            echo "[ckpt-retention] WARN: rm 失败 \\$path" >&2
        fi
        deleted=\\$((deleted + 1))
    done
    echo "[ckpt-retention] 保留 ckpt 数=\\$keep_count 删除数=\\$deleted"
    return 0
}}
OUTPUT_ROOT="/out/\\$TIER"
CKPT_STATE="none"
if [ -d "\\$OUTPUT_ROOT" ]; then
    if retain_single_checkpoint "\\$OUTPUT_ROOT"; then
        CKPT_STATE="ok"
    else
        CKPT_STATE="missing"
        echo "[ckpt-retention] FATAL: 保留动作未完成（见上方 FATAL/WARN）——" >&2
        echo "[ckpt-retention]   训练产物目录存在但**无一份可留的 checkpoint**；A3 无证据可判。" >&2
        echo "[ckpt-retention]   这是**运行链路错误**，不是训练失败：请检查 save_steps 是否大于实际优化步数。" >&2
        printf '%s\\n' "\\$CKPT_STATE" > /out/ckpt_retention.state
        # 训练失败（RC≠0）时保留原 rc，绝不掩盖训练失败；训练成功却无 ckpt 才是链路错误。
        if [ "\\$RC" = "0" ]; then RC=4; fi
    fi
else
    # ★ 目录不存在**不得静默跳过**（保留策略悄悄失效 ⇒ 磁盘迟早爆）。
    #   分两种情形：
    #     · 训练本身失败（RC≠0）：目录本就不会存在 ⇒ 记 state=none 并**保留原 rc**，
    #       不掩盖训练失败（这是唯一合法情形）。
    #     · 训练成功（RC=0）却无产物目录 ⇒ **产物通道/档号定位坏了** ⇒ FATAL + RC=4。
    CKPT_STATE="none"
    echo "[ckpt-retention] FATAL: 训练产物目录不存在（output_root=\\$OUTPUT_ROOT，RC=\\$RC）——" >&2
    echo "[ckpt-retention]   保留策略无法执行。若 RC≠0 则是训练失败（按原 rc 返回，不掩盖）；" >&2
    echo "[ckpt-retention]   若 RC=0 则说明**产物目录没落出来或档号定位错误**（TIER=\\${{TIER:-<未设置>}}），" >&2
    echo "[ckpt-retention]   这是运行链路错误，不是训练失败。" >&2
    printf '%s\\n' "\\$CKPT_STATE" > /out/ckpt_retention.state
    if [ "\\$RC" = "0" ]; then RC=4; fi
fi
printf '%s\\n' "\\$CKPT_STATE" > /out/ckpt_retention.state
# ══ A3 转正：容器内 torch 重载探测落盘（**收口前最后一个接触权重的步骤**）══
#   背景（本段要根治的环境性伪否）：宿主 228 **没有 torch** ⇒ collector 对 native 档
#   只能拿弱证据（structural_only:zip_crc），A3「checkpoint 可重载」判取证缺口。
#   但训练镜像自带 torch（graspo-msswift:4.5.3 实测 torch 2.11.0+cu130）⇒ 证据可得。
#   ★ 为什么**必须**插在 ckpt 保留块**之后**、收口之前（顺序是判据的一部分）：
#     ① 保留块会 rm -rf 中间段 checkpoint（checkpoint-*）。若探测先算 sha256、保留块
#        后删文件，则 collector 复查时 relpath 指向的文件已消失 / 集合已变 ⇒ **整体拒采信**
#        （契约 §6.1 硬约束 1：写完权重再算 hash；硬约束 2：算完 hash 后不得再改动文件）。
#     ② native 档保留块**只读不删**（识别到非空 final/ 即 KEEP 并 return 0）⇒ 在它之后
#        算 hash 与"写完权重再算 hash"等价，且此后本脚本只读权重、只写探测文件本身。
#     ③ 探测**必须**在 exit 收口之前：它是最后一次机会读 /out（容器退出 = 权重不可再读）。
#   ★ 为什么排在 gpu 采样器收尾**之前**也无妨：两者互不接触权重（采样器只写 gpu/ 摘要），
#     顺序对探测证据无影响；此处紧跟保留块可让"最后一次写产物"与"算 hash"相邻。
#   ★ 失败**不得**影响训练结论（§3.4 退路与防线之分）：探测脚本任何失败路径都 exit 0，
#     且失败时**不写**探测文件 ⇒ collector 退回弱证据路径（A3 仍是取证缺口，绝不放行）。
#     || true 是显式兜底（脚本不存在 / 解释器缺失时也不得改 RC）；RC 只由训练/保留决定。
#   ★ 探测脚本由生成器**同源产出**到 \\$RUN_DIR/torch_probe.py（\\$RUN_DIR 已绑定到
#     容器 /out）⇒ 不新增挂载项；契约字面量（schema / 文件名）取自
#     generate_matrix.py::TORCH_PROBE_SCHEMA / TORCH_PROBE_FILENAME（唯一真相源 §1.4）。
#   ★★ 写法约束（**2026-09-21 真机缺陷修复**；与上方 795/796/807 三处兄弟调用**逐字同款**）：
#     裸美元符 CONTAINER_PY（外层展开、烤定成字面量 "python" 落到 entry.sh），
#     **不得**转义成反斜杠美元符。原因：CONTAINER_PY 的唯一真相源在**宿主侧**
#     （本脚本第 100 行 = generate_matrix.py::CONTAINER_PY 段），它**不经** docker run 的 -e
#     注入容器 ⇒ 一旦转义，展开推迟到容器内、变量未定义 ⇒ 空串 ⇒
#     该行退化成「空命令名 + /out/torch_probe.py 参数」（228 实测 行 229: : 未找到命令）
#     ⇒ 被行尾 || true 静默吞掉 ⇒ torch_probe.json **全部从未产出**、stdout [torch-probe] 命中 0。
#     （仅 \\$TIER 这类**由 docker run -e 显式注入**的容器内变量才该转义——见下方收口段。
#     本段是 heredoc 正文：注释里同样不得出现裸反引号/裸美元符，否则外层会真的执行它们。）
#   ★★ 兜底必须留**显式痕迹**（2026-09-21 防呆改进）：|| true 只负责"不改 RC"，
#     它**不能**也不该阻止我们看见失败——本次真机缺陷正是被它静默吞掉的。
#     故兜底分支追加一条 SKIPPED 到 stderr（随**容器 stdout/stderr** 直达跑批终端/批日志，
#     人在台账外能看见）。⚠ 落点提醒：runner 的 docker run **未重定向**，<run>/stdout.log
#     是容器内 torchrun 那一行的重定向（本行不在其中）⇒ 找痕迹请到跑批终端/批日志里找。
#     且 echo 自身返回 0 ⇒ RC 仍只由训练/保留决定（fail-closed 语义一点没削弱：探测没跑成
#     ⇒ 不写 torch_probe.json ⇒ collector 依旧退回弱证据、A3 依旧是取证缺口，绝不放行）。
#     不写 EVIDENCE_GAP_torch_probe 文件：collector 没有读它的路径 ⇒ 那是个没人消费的死标记
#     （§18.1 不留负债）；且一个"看起来像证据"的旁路文件容易被误当成已取证（§2.3）。
"$CONTAINER_PY" {TORCH_PROBE_SCRIPT_CONTAINER_PATH} \\\\
    --run-dir {TORCH_PROBE_RUN_DIR} --tier "\\$TIER" \\\\
    || echo "[torch-probe] SKIPPED: 探测未执行成功（解释器/脚本缺失，或探测自身失败）——" \\
            "不写 torch_probe.json ⇒ A3 退回弱证据路径（取证缺口），绝不放行；RC 不受影响" >&2
# ★ 轻量证据（stdout.log / logging.jsonl / args.json / trainer_state.json / gpu 读数）
#   **一律不删**——"删权重不删证据"（宪法 §16 + 验收锚点必须是产物证据）。
# ══ 收口前必须做：**先让容器内采样器把摘要落盘，再退出**（2026-09-19 缺陷修复）══
#   ⚠ 本段同样位于**不加引号的 heredoc 正文**：注释里不得写裸反引号（外层会当命令替换
#     真的执行）或裸美元符（外层会当变量展开）；要写字面量请用 \\$ 转义。见下方哨兵测试。
#   缺陷：原实现把采样器丢在后台（SAMPLER=\\$!）后跑 torchrun，训练结束**直接 exit**。
#   容器一退出，后台进程**收不到 SIGTERM**（直接随容器消失）⇒ record_gpu_memory 的
#   finally 块（**摘要的唯一落盘点**，含 memory_used_mib_peak）**从不执行** ⇒
#   宿主采样摘要（读入 host_sample_peak_gib 旁路——**不是** §7 峰值列的口径，该列自
#   2026-09-21 起为二分口径、见 §9.1）**取证缺口**：228 实测全批
#   <run>/gpu/gpu_memory_summary.json 缺失 ⇒ collect_results._read_host_sample_peak_memory
#   拿到 None、该旁路缺值（task-r3-bulk §⑦-7）。
#   修法：**显式** SIGTERM + **有界等待**（只等本次 run 的采样器，零 kill 他人进程）：
#     · SIGTERM ⇒ 触发 record_gpu_memory 的 finally ⇒ 摘要落盘；
#     · 有界（≤30 s）⇒ 采样器卡死也不得拖住容器退出（§3.1 同效退路：只改代价不改结果）；
#     · 过后断言 \\$GPU_EVIDENCE_DIR/gpu_memory_summary.json **确实存在**，否则写
#       \\$GPU_EVIDENCE_DIR/EVIDENCE_GAP_gpu_memory_summary **显式登记取证缺口**（§2.3 边界校验：
#       读不到不得静默当作已取证）；★ 断言**不动 RC**，绝不掩盖训练结果。
#   ★ 取证目录**不再写死字面量**（2026-09-20 第二真相源清理）：全部引用文件头定义的
#     \\$GPU_EVIDENCE_DIR（= 档配置 gpu_monitor.output_dir = 生成期注入的
#     GPU_MONITOR_CONTAINER_DIR）。旧写法把取证目录路径写死 7 处，与档配置分叉：
#     断言按旧路径永远找不到摘要、且缺口登记自身因目录不存在而 ENOENT
#     （228 实测 entry.sh 行 197/200），属 §1.4 双真相源缺陷。
wait_for_gpu_summary() {{
    local pid="\\${{1:-}}" waited=0
    if [ -z "\\$pid" ]; then
        echo "[gpu-summary] WARN: 采样器 pid 为空 —— 无法等待，取证可能不完整" >&2
    else
        kill -TERM "\\$pid" 2>/dev/null || true
        while kill -0 "\\$pid" 2>/dev/null; do
            if [ "\\$waited" -ge 60 ]; then
                echo "[gpu-summary] WARN: 采样器 30s 内未退出（pid=\\$pid）⇒ 不再等（只改代价，不改判定）" >&2
                break
            fi
            sleep 0.5
            waited=\\$((waited + 1))
        done
        wait "\\$pid" 2>/dev/null || true
    fi
    # ★ 宿主采样旁路取证缺口断言（缺失 ⇒ 显式登记，不静默）
    #   本文件是**宿主侧 nvidia-smi 采样摘要**，只喂台账**旁路字段**
    #   host_sample_peak_gib（+ host_sample_peak_gap_mib 诊断列）——
    #   **不是** §7「实测每卡峰值显存」列的口径，也**不是**容器内 allocator
    #   （那是另一个量：native 档取 rank_metrics 的 memory.max_allocated_mib，
    #   由 transformer_adapter._emit_rank_memory_event 落盘）。§7 峰值列自
    #   2026-09-21 起为**后端二分口径**，唯一映射见
    #   src/graspo/core/result_judge.py 的 PEAK_MEMORY_CALIBER_BY_BACKEND
    #   （native 档 = rank0 max_allocated；ms-swift 档 = 自报 max_memory_reserved）。
    #   ★ 旁路缺失仍必须登记：它连诊断列一起缺，不得静默当作已取证（§2.2 / §2.3）。
    if [ -f "\\$GPU_EVIDENCE_DIR/gpu_memory_summary.json" ]; then
        printf '%s\\n' "present" > "\\$GPU_EVIDENCE_DIR/gpu_summary.state"
    else
        printf '%s\\n' \\\\
            "MISSING: \\$GPU_EVIDENCE_DIR/gpu_memory_summary.json（宿主采样旁路摘要缺失；不是容器内 allocator，也不是 §7 峰值列口径——该列口径见 result_judge.PEAK_MEMORY_CALIBER_BY_BACKEND）" \\\\
            > "\\$GPU_EVIDENCE_DIR/EVIDENCE_GAP_gpu_memory_summary"
        printf '%s\\n' "missing" > "\\$GPU_EVIDENCE_DIR/gpu_summary.state"
        echo "[gpu-summary] EVIDENCE-GAP: 宿主采样旁路摘要缺失（旁路字段 host_sample_peak_gib；不是 §7 峰值列口径——该列口径见 result_judge.PEAK_MEMORY_CALIBER_BY_BACKEND）—— 已写" \\\\
             "\\$GPU_EVIDENCE_DIR/EVIDENCE_GAP_gpu_memory_summary（**不得**当成已取证）" >&2
    fi
    return 0
}}
wait_for_gpu_summary "\\$SAMPLER"
# ══ 收口：把容器的退出码**逐字**交还给 torchrun 的退出码（原缺陷的修复点）══
#   原实现捕获了 RC=\\$?（= torchrun 退出码）却**从未 exit "\\$RC"**；本脚本最后一条
#   命令是上面的 printf（成功 ⇒ 0），于是**容器退 0**：worker 真失败（rc=1/137/124）
#   也被宿主记成 exit_code=0（228 实测：T028/T031 stdout 里 exitcode: 1，宿主记 0）。
#   判据由"最后一条命令是否成功"变成"torchrun 真的成功了吗"（§2.3 边界校验不靠巧合）。
#   ★ 优先级明确且互不掩盖（三段各自只写自己的失败语义，收口处统一表达）：
#       1) 训练失败（RC≠0，可能是 OOM=137 / 超时=124 / 预检 rc=1/3）⇒ **原样交还**，
#          绝不被保留策略的 4 覆盖（保住真因，§3.1 退路不改变结果）；
#       2) 训练成功而保留策略失败（上面两处 RC=0 分支 ⇒ RC=4）⇒ 交还 4
#          （"没产物/档号错"是运行链路错误，不得静默成功）；
#       3) 两者都成功 ⇒ 0。
#   块自身不再 exit：由这里单点收口（避免两处 exit 语义分叉，§1.4）。
exit "\\$RC"

ENTRY

# 5) **挂载表（唯一真相源）**：三个数据/代码/模型来源 + 运行目录。
#    目的地全部互不嵌套：子集目录与图像目录是兄弟，它们的父目录留在容器可写层里，
#    因此不存在"在只读绑定挂载之下创建 mountpoint"这一步（旧方案 rc=125 的根因）。
#    图像根的目的地由 train_path 反推（1) 段）——与锚点口径同源，布局变化自动跟随。
MOUNT_SPECS=(
    "$ROOT_DIR|/workspace/graspo|ro"
    "$MODELS_ROOT|{MODELS_CONTAINER_ROOT}|ro"
    "$ELAM_HOST/images|$IMAGE_DIR_CONTAINER|ro"
    "$RUN_DIR/subsets|$TRAIN_PATH_DIR_CONTAINER|ro"
    "$RUN_DIR|/out|rw"
    "$RUN_DIR/entry.sh|/entry.sh|ro"
)
assert_mount_targets_not_under_readonly "${{MOUNT_SPECS[@]}}" || exit 6

# 锁卡采用**单一路径**：宿主侧先由 gpu_lock_guard 断言，容器只认
# NVIDIA_VISIBLE_DEVICES（与容器内 train_worker 的守卫同源）。
# 不同时用 `--gpus`——两者可能互相覆盖，语义有歧义。
DOCKER_ARGS=(
    --rm --runtime=nvidia
    # ★ 产物属主（PG-13 / U5「跑完即删」）：容器以**宿主跑批用户**运行 ⇒ 容器写出的
    #   产物属主 = 宿主用户 ⇒ 事后宿主侧直接 `rm` 得掉，不需要任何提权。
    #   值只在上面「产物属主」段解析一次（$RUN_UID/$RUN_GID），这里只引用，不重算。
    #   ⚠ 顺序无关，但必须落在 image 之前（是 docker run 的选项，不是容器命令的一部分）。
    "${{RUN_USER_ARGS[@]}}"
    # 共享内存（**≥2 卡必需**）：torchrun 多 rank 按 fd 共享 CPU 张量，docker 默认
    # /dev/shm 仅 64 MiB ⇒ 多卡启动即 `unable to allocate shared memory(shm)` /
    # `Resource temporarily unavailable (11)`（228 实测：4 个 rank 同抛，Train: 0%）。
    # 与 run.sh / tests/e2e/run_matrix.sh 的既有约定逐字一致（§1.4 单一真相源）。
    --ipc=host --shm-size=16g
    -e "NVIDIA_VISIBLE_DEVICES=$GPUS"
    # 档号**显式**进容器：容器内入口脚本（entry.sh）的 ckpt 保留块要按档号定位
    # `<RUN_ROOT>/<T###>`（= 容器内 `/out/<T###>`）。不靠"内层恰好未定义⇒空串"这种
    # 隐性巧合——那是脆弱设计（§2 显式即防呆）。与上面 NVIDIA_VISIBLE_DEVICES 同范式。
    -e "TIER=$TIER"
    -e PYTHONPATH=/workspace/graspo/src
    -e HF_HUB_OFFLINE=1
    -e TOKENIZERS_PARALLELISM=false
    # ★ 容器内**可写缓存/状态根**（清单见上面的 CONTAINER_CACHE_ROOTS 段，§1.4 单一真相源）：
    #   非 root 进程既写不了镜像里 root 的 /root，也写不了镜像 ENV 写死的
    #   MODELSCOPE_CACHE=/mnt/workspace/...（`/mnt` 是 root:root 0755）⇒ 必须逐个显式指到
    #   可写根，否则 modelscope 数据集缓存 / HF 缓存 / triton·inductor 编译缓存在
    #   **训练开始前**就 PermissionError（228 实测：rank0 exit 1，训练一步没跑）。
    #   ★ 逐项由数组展开，本处**不写任何路径字面量** —— 根治"补了三个、漏了第四个"的漂移。
    #   ★ USER/LOGNAME（见 generate_matrix.py::CONTAINER_USER_NAME 的单一真相源）：
    #   镜像 /etc/passwd 里没有宿主的这个 uid ⇒ `getpass.getuser()` 落到 pwd.getpwuid 会
    #   KeyError（ms-swift/deepspeed/wandb 启动横幅都调它）。给中性值兜住，同时避免把
    #   宿主账号名注入容器产物（§15.1/§16：账号名属环境信息）。
    -e "USER={CONTAINER_USER_NAME}"
    -e "LOGNAME={CONTAINER_USER_NAME}"
)
# ★ 可写缓存/状态根逐个 `-e` 注入（清单与上面 CONTAINER_CACHE_ROOTS 同源，§1.4）：
#   展开成 `-e VAR=/path`，docker run 处不再出现任何缓存根字面量。
for CACHE_KV in "${{CONTAINER_CACHE_ROOTS[@]}}"; do
    DOCKER_ARGS+=(-e "$CACHE_KV")
done
for SPEC in "${{MOUNT_SPECS[@]}}"; do
    IFS='|' read -r SRC DST MODE <<< "$SPEC"
    if [ "$MODE" = "ro" ]; then
        DOCKER_ARGS+=(-v "$SRC:$DST:ro")
    else
        DOCKER_ARGS+=(-v "$SRC:$DST")
    fi
done
# ★ 显式打印**实际注入**的缓存根清单（§2.2）：与 dry-run 用的是同一个函数、同一份清单，
#   保证"预检看见的"和"真跑注入的"不可能不一致。
print_cache_roots
docker run "${{DOCKER_ARGS[@]}}" -w /workspace/graspo "$IMAGE" bash /entry.sh
echo "$?" > "$RUN_DIR/exit_code"
exit "$(cat "$RUN_DIR/exit_code")"
"""  # noqa: E501  模板内含 bash 续行/长行：逐行硬换行会改写生成的 shell 语义，改由 test_run_matrix54_runner.py 的 65 个用例把关


# ── 主流程 ──────────────────────────────────────────────────────────────────


def render_all_tiers(tiers: list[dict[str, Any]]) -> list[tuple[str, str]]:
    """按可表达性把每档渲染成"要落盘的文件"（文件名, 内容）。

    单一真相源（§1.4）：`--dry-run` 自检与正式生成**走同一条渲染路径**——否则
    "自检通过、生成出来的东西不一样"，自检就是假自检（这正是本轮把两处重复循环
    抽成一个函数的原因）。

    - ``blocked`` ⇒ ``T###.blocked.md``（能力落地后可再生成，配 `.blocked` 说明）；
    - ``not_applicable`` ⇒ ``T###.not_applicable.md``（**确实不适用**的合法终态；
      **不产出任何可跑配置**，理由必须可读）；
    - 其余 ⇒ ``T###.yaml``。
    """
    rendered: list[tuple[str, str]] = []
    for tier in tiers:
        status, reason = classify_expressibility(tier)
        tier_id = str(tier["tier_id"])
        if status == "blocked":
            rendered.append((f"{tier_id}.blocked.md", render_blocked_stub(tier, reason or "")))
        elif status == "not_applicable":
            rendered.append(
                (f"{tier_id}.not_applicable.md", render_not_applicable_stub(tier, reason or ""))
            )
        else:
            rendered.append((f"{tier_id}.yaml", render_config_yaml(tier, status, reason)))
    return rendered


def generate(
    expected_total: int,
    train_source: str | None = None,
    gpu_override: Mapping[str, Sequence[int]] | None = None,
    gpu_override_spec: str | None = None,
    manifest_out: Path | None = None,
) -> dict[str, Any]:
    """生成全部产物。

    `gpu_override`（档 → 卡集合）覆盖**只改每档 ``gpus``**，不改 ``cards``、
    不改档位 YAML、不改 runner（runner 只从清单读卡集合）。
    `manifest_out` 指定清单落点（默认 `tests/e2e/matrix54_manifest.json`）——给
    "不覆盖 canonical 清单、生成一份变体清单再让 `GRASPO_RUNNER_MANIFEST` 指过去"用。
    """
    resolved_train_source = resolve_train_source(train_source)
    tiers = build_ledger(gpu_override)
    assert_ledger(tiers, expected_total)
    expressibility = assert_expressibility(tiers)
    # 生成期显存可行性断言：任一可执行档估算超预算 ⇒ 拒绝生成并报算式。
    assert_memory_feasible(tiers)

    rendered = render_all_tiers(tiers)
    assert_no_legacy_terms(rendered)

    record = gpu_assignment_record(gpu_override_spec, tiers) if gpu_override else None
    manifest = build_manifest(tiers, resolved_train_source, gpu_assignment=record)
    manifest_text = json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"
    runner_text = render_runner(resolved_train_source)
    assert_no_legacy_terms([("manifest", manifest_text), ("runner", runner_text)])

    if CONFIG_DIR.exists():
        shutil.rmtree(CONFIG_DIR)
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    for name, text in rendered:
        (CONFIG_DIR / name).write_text(text, encoding="utf-8")
    manifest_path = MANIFEST_PATH if manifest_out is None else Path(manifest_out)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(manifest_text, encoding="utf-8")
    RUNNER_PATH.write_text(runner_text, encoding="utf-8")
    RUNNER_PATH.chmod(0o755)
    write_feasibility_table(tiers)

    return {
        "tiers": tiers,
        "manifest": manifest,
        "expressibility": expressibility,
        "manifest_path": manifest_path,
        "gpu_override": dict(gpu_override or {}),
    }


def write_feasibility_table(tiers: list[dict[str, Any]]) -> Path:
    """把"每档配方 + 可行性自检结果"落成一份可读表格（工位证据）。"""
    rows = feasibility_table(tiers)
    lines = [
        "| 档位 | 算法 | 模式 | 后端 | 卡数 | 状态 | 可行性判定 | 配方 | 可行性自检 |",
        "|---|---|---|---|:--:|---|---|---|---|",
    ]
    for row in rows:
        recipe = json.dumps(row["recipe"], ensure_ascii=False) if row["recipe"] else "—"
        lines.append(
            f"| {row['tier_id']} | {row['algorithm']} | {row['mode']} | {row['backend']} "
            f"| {row['cards']} | {row['status']} | {row['verdict']} "
            f"| {recipe} | {row['estimate']} |"
        )
    path = PROJECT_ROOT / ".local" / "matrix54_feasibility.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def print_summary(result: dict[str, Any]) -> None:
    manifest = result["manifest"]
    counts = manifest["counts"]
    print(f"档位总数: {counts['total']}")
    print(f"  算法: {counts['by_algorithm']}")
    print(f"  卡数: {counts['by_cards']}")
    print(f"  后端: {counts['by_backend']}")
    print(f"  模式: {counts['by_mode']}")
    print(f"  可表达性: {result['expressibility']}")
    print(f"  显存可行性: {counts['by_feasibility_verdict']}")
    # 【必须打印】读取源必须对操作者可见：默认行为要能"一句话说清"（否则换源是隐式的）。
    print_data_source(manifest["data"])
    print_gpu_assignment(result["tiers"], result.get("gpu_override") or {})
    print(f"配置目录: {CONFIG_DIR}")
    print(f"运行清单: {result.get('manifest_path', MANIFEST_PATH)}")
    print(f"执行骨架: {RUNNER_PATH}")


def print_gpu_assignment(
    tiers: list[dict[str, Any]],
    override: Mapping[str, Sequence[int]],
    *,
    dry_run: bool = False,
) -> None:
    """打出生效卡集合与**来源**（默认 / 覆盖）——默认行为必须显式可见、覆盖必须可审计。

    逐档打印被覆盖的档（本档用哪几张卡 + 卡数 + NUMA 提示）；未覆盖的档按卡数汇总。
    末尾打印映射指纹（A4「同档两跑同卡位」的可核证据）。
    """
    prefix = "[dry-run] " if dry_run else ""
    overridden = {str(tier_id): tuple(gpus) for tier_id, gpus in override.items()}
    if overridden:
        rendered_spec = ";".join(
            f"{tier_id}={','.join(str(gpu) for gpu in gpus)}"
            for tier_id, gpus in sorted(overridden.items())
        )
        source_label = f"覆盖（{GPU_OVERRIDE_CLI_FLAG}='{rendered_spec}'）"
    else:
        source_label = "默认（GPU_SETS；未给 --gpu-override）"
    print(f"{prefix}卡集合来源: {source_label}")
    print(f"{prefix}卡集合默认（按卡数）: {describe_default_sets(GPU_SETS)}")
    if overridden:
        by_id = {str(tier["tier_id"]): tier for tier in tiers}
        for tier_id in sorted(overridden):
            tier = by_id[tier_id]
            line = format_assignment_line(
                tier_id, tier["gpus"], int(tier["cards"]), "override"
            )
            print(f"{prefix}  {line}")
        print(
            f"{prefix}  其余 {len(tiers) - len(overridden)} 档沿用上面的按卡数默认集合"
            f"（逐档值见清单 tiers[].gpus）"
        )
    fingerprint = gpu_assignment_fingerprint(
        (str(tier["tier_id"]), tier["gpus"]) for tier in tiers
    )
    print(
        f"{prefix}卡集合指纹(sha256): {fingerprint}；复核命令 "
        f"`python3 tests/e2e/gpu_assignment.py check --manifest <清单> "
        f"--expect-fingerprint {fingerprint}`"
    )
    print(
        f"{prefix}运行期卡集合来源: runner 只从清单读 tiers[].gpus；换卡集合用 "
        f"GRASPO_RUNNER_MANIFEST 指向另一份清单（实际用哪几张卡由守卫写进运行产物）"
    )


def print_data_source(data: dict[str, Any], *, dry_run: bool = False) -> None:
    """打出「本次生成的 runner 从哪份 JSONL 取训练子集」——默认值必须显式可见。"""
    mini = data["mini_dataset"]
    prefix = "[dry-run] " if dry_run else ""
    print(
        f"{prefix}训练子集读取源: {data['train_source']} "
        f"（覆盖点 {data['train_source_env_var']}；"
        f"默认 = <{ELAM_HOST_ROOT_ENV}>/{mini['relative_path_under_data_root']}，"
        f"sha256={mini['jsonl_sha256']}，{mini['line_count']} 行）"
    )
    if data["train_source"] != TRAIN_SOURCE_MINI:
        print(f"{prefix}⚠ 非默认来源：效果口径请确认（{MINI_PURPOSE_NOTE}）")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="GRASPO 54-tier matrix generator.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run all self-checks without writing any file.",
    )
    parser.add_argument(
        "--assert-count",
        type=int,
        default=EXPECTED_TOTAL,
        help="Expected tier count asserted by --dry-run (default: 54).",
    )
    parser.add_argument(
        "--train-source",
        choices=TRAIN_SOURCES,
        default=None,
        help=(
            "训练子集的读取源：mini（默认；矩阵跑通用）| full（源数据 <ELAM_HOST>/data/train.jsonl，"
            "对照与回退用）。不影响任何档位配置与 data.train_path。"
        ),
    )
    parser.add_argument(
        GPU_OVERRIDE_CLI_FLAG,
        default=None,
        help=(
            "卡集合覆盖值（**唯一覆盖点**，逐档显式映射、无通配符）："
            "'T001=4;T010=4,5'。只改每档用哪几张卡，**不改卡数**（不符即拒绝生成）；"
            "卡号必须 ⊆{0..5}（GPU6/7 是生产 vLLM，出现即拒绝）。"
            "未出现的档沿用默认 GPU_SETS。不传 ⇒ 产物与旧版逐字节相同。"
        ),
    )
    parser.add_argument(
        "--manifest-out",
        default=None,
        help=(
            "清单落点（默认 tests/e2e/matrix54_manifest.json）。给『生成变体清单、"
            "不动 canonical 清单』用：随后 GRASPO_RUNNER_MANIFEST=<该清单> 即可按新卡集合跑批。"
        ),
    )
    return parser


def run(args: argparse.Namespace) -> int:
    """执行已解析的命令行（覆盖值在这里统一解析一次，dry-run 与真生成共用）。"""
    train_source = resolve_train_source(args.train_source)
    gpu_override = parse_override(args.gpu_override) if args.gpu_override is not None else None

    if args.dry_run:
        tiers = build_ledger(gpu_override)
        assert_ledger(tiers, args.assert_count)
        expressibility = assert_expressibility(tiers)
        # 生成期显存可行性断言也必须在 dry-run 里跑（自检不含它就是假自检）。
        assert_memory_feasible(tiers)
        rendered = render_all_tiers(tiers)
        assert_no_legacy_terms(rendered)
        verdict_counts: dict[str, int] = {}
        for tier in tiers:
            verdict, _ = feasibility_verdict(tier)
            verdict_counts[verdict] = verdict_counts.get(verdict, 0) + 1
        print(
            f"dry-run OK: {len(tiers)} tiers, expressibility={expressibility}, "
            f"feasibility_verdict={verdict_counts}"
        )
        # dry-run 也打读取源与摘要：默认行为必须可见（真生成前先看清）。
        print_data_source(build_manifest(tiers, train_source)["data"], dry_run=True)
        print_gpu_assignment(tiers, gpu_override or {}, dry_run=True)
        print(
            "说明：`infeasible` 档已被拒绝生成；`unmeasured` 档**允许生成但不声称可行**"
            "（未测算的量只能靠上机量出来，见上方待测量清单）；`blocked` 档不产出配置；"
            "`not_applicable` 档**确实不适用**（合法终态）——既不产出配置，也不参与可行性判断。"
        )
        return 0

    result = generate(
        args.assert_count,
        train_source,
        gpu_override=gpu_override,
        gpu_override_spec=args.gpu_override,
        manifest_out=Path(args.manifest_out) if args.manifest_out else None,
    )
    print_summary(result)
    return 0


def main(argv: list[str] | None = None) -> int:
    """CLI 入口：卡集合违规给出**可操作报错**并 fail-closed（不落任何产物）。"""
    args = build_arg_parser().parse_args(argv)
    try:
        return run(args)
    except GpuAssignmentError as exc:
        print(f"FATAL(gpu-assignment): {exc}", file=sys.stderr)
        print("  用法: --gpu-override 'T001=4;T002=4,5'（逐档映射，无通配符）", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
