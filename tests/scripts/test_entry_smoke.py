#!/usr/bin/env python3
"""入口脚本**冒烟全覆盖**（B4/§(1) 常设检查）。

## 为什么需要它

B3 的冒烟抓出 `scripts/gpu_lock_guard.py` 自 P0 起加载错模块（`run.sh:114` 直接
`AttributeError` 拒起），而当时**三个闸门都没抓到**——说明闸门对"宿主侧入口/启动
脚本"有覆盖漏洞：单测只覆盖被 import 的函数，对拍台只覆盖 Ripple 数值，边界检查
只覆盖 import 图。本文件补这一层。

## 判据（逐脚本，可复跑）

- **Python 入口**：能 `--help` 且 `rc=0`；无 argparse 的入口退化为**语法/编译冒烟**；
- **Shell 入口**：`bash -n` 语法检查；可 `--help` 者跑一次并期望 `rc=0`；可
  `--dry-run` 者跑一次，接受"有据的 fail-closed"（环境根未注入时按设计拒起）；
- **输出**：`候选 N（py X / sh Y），坏了 B 个` + 逐个原因；`B != 0` 即测试失败。

## 发现规则与排除理由（★ 新的入口脚本不会被静默漏掉）

候选由 `git ls-files`（**已跟踪**）∪ `git ls-files --others --exclude-standard`
（**未跟踪但不被 `.gitignore` 忽略**）**自动发现**：所有 `*.sh`、`scripts/*.py`、
`docker/*.py`、`tools/*.py`、`tests/e2e/{,rig_synth/}*.py`（去掉 pytest 测试件）。
★ 只扫已跟踪集合是本检查器此前的**结构性盲区**：新落的入口脚本在 `git add` 之前是
`??`（未跟踪），不在候选集里 ⇒ 断言无从触发 ⇒ 门禁在"提交前"通过、"提交后"才失败
（实测事件：`tests/e2e/regen_field_guard.py` 随 U4 `a1e53b1` 入库时漏登记）。补上
未跟踪段后，"新增入口未登记"在提交前即可被拦。发现到的每个路径
必须在本文件的 `POLICY` 或 `EXCLUSIONS` 里有**具名**条目，否则测试**当场失败**
（fail-closed：新增入口必须显式登记冒烟方式，不得漏检）。

排除（`EXCLUSIONS`，附理由）：
- `tests/**/test_*.py`：pytest 测试件，由单测闸门覆盖，不是"启动脚本"；
- `scripts/errors.py`：纯模块（无 `__main__`、无 CLI），被单测 import；
- `tools/gate_common.py`：门禁库（无 `__main__`），由 6 个 gate 入口间接覆盖；
- `src/**`：库代码，由边界闸门 + 单测覆盖；
- `tests/e2e/rig_synth/test_*.py`：同上，pytest 测试件。

不作为"常设冒烟对象"但**已在 POLICY 里具名**的（附理由，均仍做 `bash -n` /
语法检查）：
- `docker/build.sh` / `docker/proxy-env.sh`：构建期脚本，跑一次会真去 build/需要
  BuildKit 环境；语法检查 + 构建门禁（`check_env_versions.py`）已覆盖其核心；
- `scripts/eval_serve_vllm.sh`：会起 vLLM 并占 GPU（红线：不占 GPU）；
- `tests/e2e/run_matrix.sh`：位置参数为 output_dir，跑一次会真跑矩阵；
- `tests/e2e/rig_synth/verify_build.sh`：需要注入宿主 rig 根，跑一次会 verify 真实产物。
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
TIMEOUT_SEC = 180

# ── 冒烟模式 ────────────────────────────────────────────────────────────────
HELP = "python-help"  # python <path> --help ⇒ rc==0
COMPILE = "python-compile"  # 只编译（无 argparse 的入口）⇒ rc==0
RUN0 = "python-run"  # 直接跑（忽略 --help 的入口）⇒ rc==0
SH_HELP = "shell-help"  # bash -n + bash <path> --help ⇒ rc==0
SH_SYNTAX = "shell-syntax"  # bash -n ⇒ rc==0
SH_FAILCLOSED = "shell-failclosed"  # bash -n + 跑一次，接受有据的 fail-closed


@dataclass(frozen=True)
class Spec:
    mode: str
    reason: str
    argv: tuple[str, ...] = ()
    #: None = 必须是"有据的 fail-closed"（即 rc ≠ 0）；否则必须落在该集合内。
    expect_rc: tuple[int, ...] | None = (0,)
    expect_in: str | None = None


#: 入口/启动脚本 → 冒烟方式（**每个候选都必须在此**）。
POLICY: dict[str, Spec] = {
    "scripts/run_guards.py": Spec(HELP, "起跑/评测护栏 CLI（§7 形态谓词 / §8 新鲜度 / §10 Mamba），argparse 子命令"),
    "scripts/gen_tier_json.py": Spec(HELP, "由两侧 eval_report 生成 task-matrix-runs/<TIER>.json（落盘自检：delta_pp 同源 / evidence_paths 非空）"),
    "scripts/gen_eval.py": Spec(
        HELP,
        "M 档评测脚本生成器（2026-10-22 起 argparse：--tier/--gpu/--port/--blocks/--no-ssh/--dry-run；"
        "真实渲染需活 checkpoint 与空闲卡，故常设冒烟只做 --help；形态现算的行为由"
        "`--dry-run --no-ssh` 自检覆盖，见工作包 B 报告 T4-5）",
    ),
    # ★ 补登记（2026-10-25 评测入口缺件轮）：`$EVAL_DIR/run-eval.sh` 的**唯一产出者** ——
    #   编排链在评测步以 `bash "$EV/run-eval.sh" "$T" "$EV" <TIER> after|base <cfg>`
    #   调用它（`scripts/gen_eval.py` 两处 / `scripts/launch_pipeline.py` 两处），而此前**没有产出者**
    #   ⇒ 评测步 rc=127、判据④ 无法取证。本件 = 模板 + 幂等安装（sha 记录）+ check；
    #   `install` 缺省只打印计划（零写入），`--apply` 才落盘；跑一次会写 `$EVAL_DIR` ⇒
    #   常设冒烟只做 `--help`（rc=0、零写入）。落位件本身的行为由
    #   `tests/scripts/test_eval_runner_install.py` 覆盖（幂等/不覆盖异内容/缺产物 fail-closed）。
    "scripts/eval_runner.py": Spec(
        HELP,
        "评测入口 `$EVAL_DIR/run-eval.sh` 的唯一产出者（模板渲染 + 幂等安装 + sha 记录 + check）；"
        "install 缺省只打印计划、--apply 才写 → 冒烟只做 --help",
    ),
    # ★ 新增（2026-10-28 · D23/N3b）：读数件合成步 —— 此前是**编排层手工步**（漏补即后续
    #   取证链 `FAIL: readings.json 缺失或不可解析`，M013/M018 人工补过）；现在由
    #   `tools/matrix/fill_one.py` 在读数派生之前自动调用。无参数只打印用法（rc=2），
    #   故常设冒烟只做 `--help`（rc=0、零写入）。
    "scripts/mk_readings.py": Spec(
        HELP,
        "readings.raw.json + exit_code/elapsed_sec ⇒ readings.json 的唯一合成步（幂等；D23）",
    ),
    "scripts/make_result_json.py": Spec(
        HELP,
        "跑次读数 ⇒ 回填口径结果 json（matrix-run/v1）的**唯一产出者**；落盘前调"
        "`tools/matrix/backfill_matrix.py::validate_run` 自检",
    ),
    "scripts/record_checks.py": Spec(
        HELP,
        "判据③④ 的证据产出者（审查第 2 轮 🟡-A）：从评测报告实测算 ④、按算法分级产出 ③"
        "（GRASPO 须签署逐条不变式证据），写进 `<run-dir>/readings.json` 的 checks{}；"
        "缺证据不写键（回填器 fail-closed）；argparse CLI ⇒ --help 冒烟",
    ),
    # ── 仓库根 / 宿主启动链 ────────────────────────────────────────────────
    "run.sh": Spec(SH_HELP, "训练启动主入口；支持 --help（宿主引导层稳定契约）"),
    # ── scripts/*.py（§10.2 制度化脚本，B3 已登记稳定子命令）────────────────
    "scripts/build_mini_elam_subset.py": Spec(HELP, "宿主侧数据工具；argparse CLI"),
    "scripts/check_env_versions.py": Spec(HELP, "镜像/构建期版本门禁入口"),
    "scripts/collect_results.py": Spec(HELP, "矩阵结果收集入口"),
    "scripts/convert_targets_format.py": Spec(
        COMPILE, "无 argparse 的位置参数 CLI（argv 直读）⇒ 语法/可加载冒烟"
    ),
    "scripts/env_snapshot.py": Spec(HELP, "环境快照记录入口"),
    "scripts/gpu_idle_assert.py": Spec(HELP, "F-10 空闲断言入口（ticket ⑤ 的 UUT）"),
    "scripts/gpu_lock_guard.py": Spec(HELP, "run.sh:114 锁卡守卫（B3 修过 P0 后损坏）"),
    "scripts/measure_elam_seq_len.py": Spec(HELP, "ELAM 序列长度测量入口"),
    # ── tools/matrix/ 文档工具链入口（功能矩阵 SSOT 的产出/校验链）────────
    "tools/matrix/render_matrix.py": Spec(HELP, "矩阵 HTML 的唯一渲染通道（模板+数据⇒HTML）"),
    "tools/matrix/verify_matrix.py": Spec(HELP, "矩阵执法：重渲染+逐字节比对+结构/内容/脱敏断言"),
    "tools/matrix/build_official.py": Spec(HELP, "正式机读件（脱敏）构建入口"),
    "tools/matrix/build_data.py": Spec(HELP, "内部数据构建入口（列定义+统一口径+逐档参数）"),
    "tools/matrix/extract_params.py": Spec(HELP, "逐档参数抽取入口"),
    "tools/matrix/extract_spec.py": Spec(HELP, "逐档规格抽取入口（含冲突/缺项登记）"),
    "tools/matrix/backfill_matrix.py": Spec(HELP, "跑次结果回填入口（增量/fail-closed）"),
    "tools/matrix/run_self_proof.py": Spec(HELP, "三组自证入口（空数据/假数据/手改 1 字符）"),
    "tools/matrix/scan_matrix_leaks.py": Spec(HELP, "矩阵脱敏扫描（§17.7 禁止项：宿主路径/内网/机器名/人名/账号）"),
    "tools/matrix/enumerate_matrix.py": Spec(
        HELP,
        "档位枚举脚本：全量组合 → 按显存可行性剔除 → 生成 124 档矩阵数据件；确定性可复跑。"
        "★ 本件不是渲染链入口（渲染 = render_matrix.py），此处只做语法/CLI 冒烟（--help ⇒ rc=0）",
    ),
    "tools/matrix/gen_configs.py": Spec(HELP, "矩阵数据 → 测试 yaml 生成器（确定性；禁手改 yaml）"),
    # ★ 补登记（2026-10-23「基准件唯一 + 按轮副本」轮）：矩阵链**路径唯一真相源**（宪法 §1.4）——
    #   基准件 / 按轮副本的命名与路径、基准件冻结 sha、权限约定，并 `--sh` 导出一组 `PATHS_*` 赋值供 bash
    #   侧 `eval`（各脚本 ⛔ 不再各写一份路径字面量）。**纯读**：`--help` / 人读一览 / `--sh` 均零写入；
    #   唯一写面是显式 `--set-human-view-path`（只改**副本**数据件 `meta.authority.human_view_path` 那一行）。
    "tools/matrix/paths.py": Spec(
        HELP,
        "矩阵链路径唯一真相源（基准件/按轮副本命名与路径 + 基准件冻结 sha + 权限约定 + `--sh` 导出）；"
        "纯读，唯一写面是显式 `--set-human-view-path`（只改副本数据件一行）",
    ),
    # ★ 补登记（2026-10-23 同批）：**开新一轮**入口（非交互 · 幂等）—— ① 校验基准件 sha256 与 444
    #   ⇒ ② `cp -p` 建当日人读副本 644 ⇒ ③ 建当日数据副本 644 且 `human_view_path` 指向该副本 ⇒
    #   ④ 打印下一步命令；副本已存在则**只校验不覆盖**（同参重跑 rc=0、零字节变化）。
    #   ⛔ 永不写基准件、⛔ 不做任何 git 操作、⛔ 不跑训练不占卡。跑一次会真建副本（写 docs/）⇒
    #   冒烟只做 `bash -n` + `--help`（打印用法、rc=0、零写入）。
    "tools/matrix/new_round.sh": Spec(
        SH_HELP,
        "开新一轮入口（非交互·幂等）：校验基准件 sha+权限 ⇒ 派生当日副本对（人读件/数据件 644，"
        "数据件 human_view_path 指向副本）；⛔ 不写基准件、⛔ 无 git 操作。跑一次会真建副本 ⇒ 冒烟只做 --help",
    ),
    # ★ 补登记（2026-10-25 报告定稿冻结轮）：**填表写范围硬闸门**（主席令「填写报告的脚本只能填写
    #   第四章表格里面的实验内容数据，不能修改其它任何地方」）。两份 HTML（原版 vs 填表后）做纯结构
    #   diff：任何差异必须落在「第 4 章矩阵表 tbody · M 档行 · 实验结果列（= 数据件 data_keys 现读）
    #   · 纯文本」面内，否则 rc=2 并列违规清单；已接进回填链（不过即回滚本次写入，§22）。
    #   **纯读**（不写任何文件）⇒ 冒烟跑 `--help`（打印用法、rc=0）。
    "tools/matrix/check_fill_scope.py": Spec(
        HELP,
        "填表写范围硬闸门（§22）：原版 vs 填表后两份 HTML 纯结构 diff，差异只许落在第 4 章实验数据"
        "单元格文本；否则 rc=2 + 违规清单（文件:行:列 + 片段）；纯读，已接进回填链（不过即回滚）",
    ),
    # ★ 补登记（2026-10-22 基线轮）：68 档开跑前就绪闸门 —— **十条**口径（档号集合 / LoRA 三参+目标整块 /
    #   子集 size 与在盘 sha / 全参必分片 / native 开关 / DeepSpeed 禁用 + R1-3 交叉断言 ⑦身份同源 /
    #   ⑧并行形状×卡数 / ⑨全参真分片 / ⑩步数子集不滞后）；argparse CLI ⇒ --help 冒烟。
    "tools/matrix/check_68_readiness.py": Spec(
        HELP,
        "68 档开跑前就绪闸门（十条口径，全 fail-closed；⑦–⑩ 为配置↔数据件/派生件交叉断言）；"
        "argparse CLI（--config-dir/--data/--manifest/--subset-dir/--runner-manifest）",
    ),
    # ★ 补登记（2026-10-22 基线轮）：68 档语料切换工具 —— argparse CLI，--source-jsonl/--expect-sha256
    #   必填且无默认值（无参不可能误落盘）；默认 --dry-run 只出计划不写盘，--apply 才写（带备份/四道自检/
    #   失败回滚）。入库前本件是 `??` 未跟踪，由本检查器的「未跟踪段」发现 ⇒ 提交前即拦下未登记。
    #   跑一次 --apply 会真改语料 ⇒ 冒烟只做 --help（rc=0，不落盘）。
    "tools/matrix/switch_corpus.py": Spec(
        HELP,
        "语料切换工具：默认 --dry-run 只出计划不写盘，--apply 才落盘且带备份/自检/回滚",
    ),
    # ★ 补登记（2026-10-24 判据③ 轮）：GRASPO 逐档签署件生成器 —— 判据③「按《算法说明》逐条判断」
    #   的**唯一产出者**（8 条条文从算法说明第 9 节现解析 / 档集合从 matrix68 清单现算 / 证据必须外部给）。
    #   默认只出计划不写盘，--apply 才落盘；缺证据、有条目 verdict=false、档号不属于 GRASPO 集合、
    #   证据坐标不可核 ⇒ rc=2 且**零写入**。跑一次 --apply 会写 samples/configs/matrix68-checks/ ⇒ 冒烟只做 --help。
    "tools/matrix/sign_graspo_invariants.py": Spec(
        HELP,
        "GRASPO 判据③ 签署件生成器（第 9 节 8 条不变式逐条取证；缺证据即拒签 rc=2、零写入）",
    ),
    # ★ 新增（2026-10-27 · D23）：**取证/回填链的唯一入口**（收编 A800 侧在版控之外的
    #   `work/b09/mk_evidence.sh`）。原来那份不在版控里的脚本直连 `record_checks.py` +
    #   `make_result_json.py`，**绕过 fill_one** ⇒ 链首「读数件合成」被漏 ⇒ M013/M018 人工补、
    #   M008 盘上 `readings.json` 是空壳（`exit_code=99`）把**跑成功**的档判失败（实测三例）。
    #   收编后链 = ⓪ `fill_one.py::ensure_readings`（单一真相源）→ ① `record_checks.py` → ② `fill_one.sh`。
    #   常设冒烟只做 `bash -n` + `--help`（rc=0、零写入）：真跑需要跑次目录与当轮副本；
    #   链序/幂等/失败可见/`--dry-run` 不落盘由 `tests/tools/test_mk_evidence_single_chain.py` 覆盖。
    "tools/matrix/mk_evidence.sh": Spec(
        SH_HELP,
        "取证/回填链唯一入口（⓪读数件合成 → ①record_checks → ②fill_one.sh，⛔ 无第二份派生调用）；"
        "--help 打印用法 rc=0、零写入；真跑需跑次目录 ⇒ 冒烟只做 --help",
    ),
    # ★ 补登记（2026-10-22 基线轮）：回填防呆 wrapper —— 钉死「带日期副本对」与 .local 台账，指向仓内
    #   原版（数据件或人读件）一律 rc=2 拒起。跑一次会真回填（写数据件/HTML）⇒ 冒烟只做 --help 与 bash -n。
    "tools/matrix/backfill_dated.sh": Spec(
        SH_HELP,
        "回填防呆 wrapper：--data-in/--html-out/--template/--ledger-out 钉死为带日期副本对与 .local 台账；"
        "指向仓内原版（数据件或人读件）rc=2 拒起。跑一次会真回填 ⇒ 冒烟只做 --help（打印用法、rc=0）",
    ),
    # ★ 补登记（2026-10-22 基线轮）：回填隔离的收尾自检（**8 步**，只读）。**无 --help、不解析参数**：
    #   无参运行即真跑 8 步（原版 sha 复验 → 原版对/副本对两次 verify → 两态 sha 判定 → docs/ 面
    #   git status 核对 → 泄漏扫描 → 临时件残留 → 原版两件权限 444），只写 .local 读数台账，
    #   rc=0 = ALL PASS。⑧ 是 R1-4 新增的物理只读闸门（实测 644 ⇒ rc=1，444 ⇒ rc=0）。
    #   故按「真实无参行为」登记：argv=() 且期望 rc=0 + 输出含 ALL PASS（⛔ 不接受 fail-closed）。
    "tools/matrix/verify_dated.sh": Spec(
        SH_FAILCLOSED,
        "回填隔离收尾自检（8 步只读：原版 sha 复验→原版对/副本对两次 verify→两态 sha 判定→"
        "docs/ 面 git status→泄漏扫描→临时件残留→原版两件权限 444）；无 --help 与参数解析 ⇒ 无参真跑一次，"
        "期望 rc=0 且输出 ALL PASS（只写 .local 读数台账，不写 docs/）",
        argv=(),
        expect_rc=(0,),
        expect_in="ALL PASS",
    ),
    # ★ 补登记（2026-10-22 基线轮 R1-4）：原版两件的**物理只读**加固（对应 19:20 事故：无 --dry-run 的
    #   直调回填改写了冻结件）。幂等、只做 chmod 444 + 打印 sha；跑一次会改权限位 ⇒ 冒烟只做 bash -n，
    #   不实际改权限（权限是否 444 由 verify_dated.sh ⑧ 在常设自检里查）。
    "tools/matrix/freeze_originals.sh": Spec(
        SH_SYNTAX, "原版两件 chmod 444 幂等加固脚本（只改权限位 + 打印 sha）；冒烟只做 bash -n"
    ),
    # ★ 补登记（2026-10-25，随 `a7a5690` 入库时漏登记，由本检查器的「未跟踪段」在提交前拦下）：
    #   OPD 档**外部教师服务**起服脚本 —— 把 27B 的 SFT/LoRA/dp1 产物以 vLLM 起成 OpenAI 兼容服务，
    #   供训练侧 `distill.teacher_model_server` 指向（教师**另卡**部署 ⇒ 教师占卡**不计入**矩阵该档
    #   `cards`：矩阵卡数只算被训练的模型，见 docs/run-discipline.md §21）。
    #   ⛔ 默认 dry-run：只打印将要执行的 docker 命令（零外部调用、不起容器、不探网、不查卡）；
    #   `--apply` 才真起容器（会占教师卡 ⇒ 红线：不占 GPU）⇒ 常设冒烟只做 `bash -n` + `--help`
    #   （打印用法、rc=0、零写入）。
    "tools/matrix/serve_teacher.sh": Spec(
        SH_HELP,
        "OPD 外部教师起服脚本（27B SFT/LoRA/dp1 产物 → vLLM/OpenAI 兼容服务；教师占卡不计入该档 cards）："
        "默认 dry-run 只打印 docker 命令，`--apply` 才真起容器并占教师卡（红线：不占 GPU）⇒ 冒烟只做 bash -n + --help",
    ),
    # ── shell 工具/运维入口 ────────────────────────────────────────────────
    "scripts/eval_cleanup.sh": Spec(SH_HELP, "评测清理入口；支持 -h/--help ⇒ rc=0"),
    "scripts/sync_deploy_tree.sh": Spec(SH_HELP, "§11：仓内 scripts/*.py 幂等同步到部署树 + 逐件冒烟（失败即非零退出）"),
    "scripts/eval_export_checkpoint.sh": Spec(SH_HELP, "评测导出入口；-h 显式 exit 0"),
    "scripts/eval_serve_vllm.sh": Spec(
        SH_SYNTAX, "容器生命周期脚本，跑一次会起 vLLM 并占 GPU（红线：不占 GPU）"
    ),
    "docker/build.sh": Spec(SH_SYNTAX, "构建期脚本，跑一次会真 build 镜像（本轮 ⛔ 不跑构建）"),
    "docker/proxy-env.sh": Spec(SH_SYNTAX, "被 build.sh source 的代理注入片段，无独立 CLI"),
    "docker/gate_script.py": Spec(
        COMPILE, "构建期双向诚实门禁（Dockerfile RUN 调用），非镜像环境跑必 fail-closed"
    ),
    # ── tests/e2e 生成器/启动器（宿主侧实验链入口）──────────────────────────
    # ★ 补登记（2026-10-13）：跑次侧新增起跑 launcher —— 六条断言（树指纹/实测空闲/排期/叠跑闸/镜像+单源/dry-run）。
    #   依赖真实树 + docker + 卡 ⇒ 无副作用下只能做 `bash -n` 语法冒烟（SH_SYNTAX）。
    "tests/e2e/launch_matrix54.sh": Spec(
        SH_SYNTAX, "矩阵起跑 launcher（六条断言 → nohup 起跑 + 巡检）；需真实树/docker/卡 ⇒ 语法冒烟"
    ),
    # ★ 补登记（2026-10-13）：矩阵侧新增"产物级"校验入口 —— 比较运行 config 与部署树 yaml。
    #   运行时需真实 runs 产物 ⇒ 无副作用下只能做 `bash -n`/语法冒烟（SH_SYNTAX 等价：py_compile 已由入口自身保证）。
    "tools/matrix/check_run_config.py": Spec(
        HELP, "运行 config ↔ 部署树 yaml 的产物级校验（不一致即 FAIL-CLOSED）"
    ),
    # ★ 补登记（2026-10-14）：里程碑记录「冻结件」节的**机读生成器**（只读打印 / `--inject` 注入）。
    #   无参运行 ⇒ rc==0 且输出含三件 sha 行 ⇒ 用 HELP 模式做常设冒烟（⛔ 不注入任何文件）。
    "tools/matrix/freeze_table.py": Spec(
        HELP, "里程碑「冻结件」节机读生成器（值来自实物 sha256＋字节；支持 --inject）"
    ),
    # ★ 补登记（2026-10-16）：生成链「防丢失」共享守卫 —— 上一批 U4 commit `a1e53b1` 新增时
    #   仍是**未跟踪**文件，`git ls-files` 发现不到 ⇒ 提交后才发现漏登记（详见 tools/matrix/README.md）。
    #   本件是**库模块**（无 `__main__`/无 CLI，被 3 个清单生成器与 `test_regen_field_guard.py` import）
    #   ⇒ 无 `--help` 可冒烟，按 POLICY 既有先例（无 argparse 的件）取 **COMPILE** 模式做语法/可编译冒烟。
    "tests/e2e/regen_field_guard.py": Spec(
        COMPILE,
        "生成链防丢失共享守卫库（重生成不得移除在盘字段；仅 stdlib）；由 3 个清单生成器 import",
    ),
    "tests/e2e/generate_matrix.py": Spec(HELP, "54 档矩阵生成器（v1 基线）"),
    "tests/e2e/generate_matrix_fsdp.py": Spec(HELP, "matrix54-fsdp 变体生成器"),
    "tests/e2e/generate_matrix_manifest_fsdp.py": Spec(HELP, "fsdp 清单生成器"),
    "tests/e2e/generate_matrix_manifest_source.py": Spec(HELP, "v2 清单生成器"),
    "tests/e2e/gpu_assignment.py": Spec(HELP, "卡位分配校验入口"),
    "tests/e2e/make_subsets.py": Spec(HELP, "subsets-v2 冻结件生成器（B1 并入 E-49/E-50）"),
    "tests/e2e/pp_rollout_smoke.py": Spec(HELP, "PP rollout 冒烟入口"),
    "tests/e2e/run_matrix54.sh": Spec(
        SH_FAILCLOSED,
        "矩阵主启动器；--dry-run 在环境根未注入时按 §15.1/§16 设计 fail-closed",
        argv=("T001", "--dry-run"),
        expect_rc=None,
        expect_in="GRASPO_ELAM_HOST_ROOT",
    ),
    "tests/e2e/generate_matrix68_manifest_source.py": Spec(
        HELP, "68 档（M001–M068）运行清单生成器：数据件 + 68 份 yaml + 子集现读；含 --verify 在盘自证入口"
    ),
    "tests/e2e/run_matrix68.sh": Spec(
        SH_FAILCLOSED,
        "68 档 M 档 runner；`--dry-run` 只打印命令与判据、⛔ 不启动 docker/torchrun（无需注入环境根 ⇒ rc=0）",
        argv=("--only", "M001", "--dry-run"),
        expect_rc=(0,),
        expect_in="\\[dry-run\\] M001",
    ),
    "tests/e2e/run_matrix54_source.sh": Spec(
        SH_FAILCLOSED,
        "v2source 逐档 launcher；--dry-run 同样要求宿主根注入，缺失即拒起",
        argv=("T028", "--dry-run"),
        expect_rc=None,
        expect_in="GRASPO_ELAM_DATA_ROOT",
    ),
    "tests/e2e/run_matrix.sh": Spec(
        SH_SYNTAX, "旧矩阵 runner；位置参数 output_dir，跑一次会真跑矩阵"
    ),
    "tests/e2e/rig_synth/build_synth_seq.py": Spec(HELP, "合成长序列基线构建器"),
    "tests/e2e/rig_synth/emit_configs.py": Spec(HELP, "合成长序列配套配置生成器"),
    "tests/e2e/rig_synth/emit_manifests.py": Spec(HELP, "合成长序列清单生成器"),
    "tests/e2e/rig_synth/measure_synth_tokens.py": Spec(HELP, "合成长序列 token 测量入口"),
    "tests/e2e/rig_synth/verify_synth_artifacts.py": Spec(HELP, "合成长序列产物校验入口"),
    "tests/e2e/rig_synth/run_synth.sh": Spec(
        SH_FAILCLOSED,
        "合成长序列 runner；无 --variant 时按设计拒起（race-free 参数守卫）",
        argv=(),
        expect_rc=None,
        expect_in="必须给 --variant",
    ),
    "tests/e2e/rig_synth/verify_build.sh": Spec(
        SH_SYNTAX, "需要注入宿主 rig 根，跑一次会 verify 真实产物（本机无该根）"
    ),
    # ── tools/ 门禁入口 ────────────────────────────────────────────────────
    # 注：这 6 个 gate 的 `-h/--help` 曾因 run_gate 把 SystemExit(0) 吞成 rc=2 而报错退出，
    # 已被本轮的入口冒烟抓出并修（tools/gate_common.py）。
    "tools/gate_commit_identity.py": Spec(HELP, "§15.1 提交身份门禁入口"),
    "tools/gate_content.py": Spec(HELP, "§15.1 内容门禁入口"),
    "tools/gate_filetypes.py": Spec(HELP, "§15.1 文件类型门禁入口"),
    "tools/gate_pii.py": Spec(HELP, "§15.1 PII 门禁入口"),
    "tools/gate_secrets.py": Spec(HELP, "§15.1 凭据门禁入口"),
    "tools/gate_negative_demo.py": Spec(
        RUN0, "门禁负向演示（忽略参数直接跑；全部用例必须产生真实失败）"
    ),
    "tools/readme_mirror_check.py": Spec(HELP, "§17.1 README 双语镜像检查入口"),
}

#: 具名排除（附理由；自动发现到的、落在此表的路径不参与冒烟）。
EXCLUSIONS: dict[str, str] = {
    "scripts/launch_pipeline.py": "纯模块（无 CLI/无 __main__ 入口），被单测 import；启动链的判定函数由 launcher/评测脚本调用",
    "scripts/errors.py": "纯模块（无 __main__/无 CLI），被单测 import，不是启动脚本",
    "scripts/step_readings.py": "纯模块（无 __main__/无 CLI）：步数读数的唯一真相源，被 make_result_json/collect_results import",
    "tools/check_env.py": "★ 依赖校验入口：自身即门禁脚本，缺依赖时非零退出（不易做常设冒烟）",
    "tools/gate_common.py": "门禁库（无 __main__），由 6 个 gate 入口间接覆盖",
}


def _ls_files(*extra: str) -> list[str]:
    """跑一次 `git ls-files`（`extra` 见调用点），返回相对仓库根的 posix 路径。"""
    return subprocess.run(
        ["git", "ls-files", *extra], cwd=REPO, capture_output=True, text=True, check=True
    ).stdout.splitlines()


def _discover() -> list[str]:
    """自动发现入口/启动脚本候选（见模块 docstring 的发现规则）。

    候选集 = **已跟踪**（`git ls-files`）∪ **未跟踪但不被 `.gitignore` 忽略**
    （`git ls-files --others --exclude-standard`）。

    ★ 只取前者是本检查器此前的**结构性盲区**：新落的入口脚本在 `git add` 之前
    是 `??`（未跟踪），不在索引里 ⇒ 候选集里没有它 ⇒ `uncovered` 断言无从触发
    ⇒ 门禁在"提交前"通过、"提交后"失败（实测事件：`tests/e2e/regen_field_guard.py`
    随 U4 `a1e53b1` 入库时漏登记，提交包内 `pytest tests/scripts/` = 147 passed，
    提交后同一条 gate = 1 failed）。补上未跟踪段后，该盲区闭合。
    """
    tracked = _ls_files()
    untracked = _ls_files("--others", "--exclude-standard")
    out: list[str] = []
    for path in [*tracked, *untracked]:
        if path.endswith(".sh"):
            out.append(path)
        elif path.endswith(".py") and (
            path.startswith("scripts/")
            or path.startswith("docker/")
            or path.startswith("tools/")
            or (
                path.startswith("tests/e2e/")
                and Path(path).name != "__init__.py"
                and not Path(path).name.startswith("test_")
            )
        ):
            out.append(path)
    return sorted(set(out))


def _env_for(spec: Spec) -> dict[str, str]:
    env = dict(os.environ)
    if spec.mode == SH_FAILCLOSED:
        for key in (
            "GRASPO_ELAM_HOST_ROOT",
            "GRASPO_ELAM_DATA_ROOT",
            "GRASPO_PREGEN_SUBSET_DIR",
            "GRASPO_ATTEMPT_ROOT",
            "RUN_ROOT",
            "NVIDIA_VISIBLE_DEVICES",
        ):
            env.pop(key, None)
    return env


def _run(path: str, spec: Spec) -> subprocess.CompletedProcess[str]:
    py = sys.executable
    if spec.mode == COMPILE:
        code = (
            "import pathlib,sys;"
            "compile(pathlib.Path(sys.argv[1]).read_text(encoding='utf-8'),sys.argv[1],'exec')"
        )
        argv = [py, "-c", code, path]
    elif spec.mode in (HELP, RUN0):
        argv = [py, path, *(("--help",) if spec.mode == HELP else spec.argv)]
    elif spec.mode == SH_SYNTAX:
        return subprocess.run(
            ["bash", "-n", path], cwd=REPO, capture_output=True, text=True, check=False
        )
    elif spec.mode in (SH_HELP, SH_FAILCLOSED):
        syntax = subprocess.run(
            ["bash", "-n", path], cwd=REPO, capture_output=True, text=True, check=False
        )
        if syntax.returncode != 0:
            return syntax
        if spec.mode == SH_HELP:
            argv = ["bash", path, "--help"]
        else:
            argv = ["bash", path, *spec.argv]
    else:  # pragma: no cover - 配置错误直接炸
        raise AssertionError(f"未知冒烟模式 {spec.mode!r}（{path}）")
    return subprocess.run(
        argv,
        cwd=REPO,
        capture_output=True,
        text=True,
        check=False,
        env=_env_for(spec),
        timeout=TIMEOUT_SEC,
    )


def _verdict(path: str, spec: Spec, proc: subprocess.CompletedProcess[str]) -> str | None:
    """返回 None = 通过；否则返回人类可读的失败原因。"""
    if spec.expect_rc is None:
        if proc.returncode == 0:
            return "rc=0（期望有据的 fail-closed，即 rc≠0）"
    elif proc.returncode not in spec.expect_rc:
        got = proc.returncode
        if got < 0:
            got = f"信号 {-got}"
        tail = (proc.stderr or proc.stdout or "").strip().splitlines()
        where = tail[-1] if tail else "(无输出)"
        return f"rc={got}（期望 {'/'.join(map(str, spec.expect_rc))}）；最后一行：{where[:200]}"
    if spec.expect_in and not re.search(spec.expect_in, proc.stderr + proc.stdout):
        return f"rc={proc.returncode} 但输出未含 fail-closed 证据 {spec.expect_in!r}"
    if spec.mode in (HELP, SH_HELP):
        blob = (proc.stdout + proc.stderr).lower()
        if not re.search(r"usage|用法", blob):
            return "rc=0 但输出里没有 'usage'/'用法'（可能没有真的走到 --help）"
    return None


def smoke() -> tuple[list[tuple[str, Spec, str | None, int]], list[str]]:
    """跑全部候选，返回 (逐项结果, 未登记候选)。"""
    discovered = _discover()
    uncovered = [p for p in discovered if p not in POLICY and p not in EXCLUSIONS]
    results = []
    for path in discovered:
        spec = POLICY.get(path)
        if spec is None:
            continue
        proc = _run(path, spec)
        results.append((path, spec, _verdict(path, spec, proc), proc.returncode))
    return results, uncovered


def render(results, uncovered) -> tuple[str, int]:
    lines = ["", "=" * 78, "入口脚本冒烟全覆盖（B4/§(1)）", "=" * 78]
    broken = [r for r in results if r[2] is not None]
    py = sum(1 for _, s, _, _ in results if s.mode.startswith("python"))
    sh = len(results) - py
    for path, spec, why, rc in results:
        mark = "OK  " if why is None else "BROKEN"
        lines.append(f"[{mark}] {path:<52} {spec.mode:<18} rc={rc}")
        if why is not None:
            lines.append(f"         ↳ {why}")
    lines.append("-" * 78)
    lines.append(
        f"候选 {len(results)} 个（python {py} / shell {sh}），"
        f"坏了 {len(broken)} 个，未具名登记 {len(uncovered)} 个"
    )
    for path in uncovered:
        lines.append(f"  ✗ 未登记候选：{path}（新增入口必须写进 POLICY 或 EXCLUSIONS）")
    lines.append("=" * 78)
    return "\n".join(lines), len(broken) + len(uncovered)


def main() -> int:
    results, uncovered = smoke()
    text, bad = render(results, uncovered)
    print(text)
    return 1 if bad else 0


def test_report_baseline_is_frozen() -> None:
    """★ §22 报告冻结断言（2026-10-25 主席令）：基准件 sha256 / 字节 / 权限 444 **任一**不符 ⇒ 红。

    为什么放在"入口脚本"闸门里：改报告的**唯一**合法通道是「模板 + 数据件 + 渲染」（`render_matrix.py`，
    需主席批准），其余一切写基准件的路都是违规 —— 这条断言是那条通道之外的兜底（物理只读 + 内容指纹）。
    ★ 唯一实现在 `tools/matrix/paths.py::check_baseline_frozen()`（⛔ 本文件不抄 sha/字节/权限字面量）。
    """
    sys.path.insert(0, str(REPO / "tools" / "matrix"))
    import paths as matrix_paths

    bad = matrix_paths.check_baseline_frozen()
    assert bad == [], (
        f"报告基准件冻结被破坏（冻结日 {matrix_paths.FROZEN_DATE}）：{bad}\n"
        f"  依据：{matrix_paths.FROZEN_AUTHORITY}\n"
        "  ⇒ ⛔ 无主席许可不得修改报告；结果只写当日副本（new_round.sh --date YYYYMMDD）。"
    )
    # 一次性把"三条判据各自能红"的判据也钉住：任一不符都必须进 bad 清单（⛔ 不是"看着像对"）
    facts = matrix_paths.baseline_freeze_facts()
    assert set(facts["expect"]) == {"sha256", "bytes", "mode"}, "冻结三断言的键集合被改（⛔ 少一条就不再能红）"


def test_entry_scripts_smoke_all_covered_and_broken_zero() -> None:
    """★ 常设闸门：入口脚本全冒烟 + 坏了 0 个 + 无未登记候选。"""
    results, uncovered = smoke()
    text, bad = render(results, uncovered)
    print(text)
    assert not uncovered, f"发现未具名登记的入口脚本：{uncovered}"
    broken = [(p, why) for p, _, why, _ in results if why is not None]
    assert not broken, f"入口脚本冒烟有 {len(broken)} 个坏了：{broken}"


if __name__ == "__main__":
    raise SystemExit(main())
