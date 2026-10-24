#!/usr/bin/env python3
"""`$EVAL_DIR/run-eval.sh`（评测入口）的**唯一产出者** —— 生成 / 安装 / 校验，仅此一处。

## 为什么有本文件（缺件实测）

编排链在**评测步**要求 `$EVAL_DIR/run-eval.sh` 存在并被这样调用（三处调用方，参数序完全一致）：

  · `scripts/gen_eval.py:252`（after）/ `:267`（base）——
    `bash "$EV/run-eval.sh" "$T" "$EV" {tier} after|base "$EV/eval-{lower}-after|base.yaml"`
  · `scripts/launch_pipeline.py:299`（after）/ `:308`（base）——同上（`EVAL_SCRIPT_TEMPLATE` 内）

而仓内**只有调用方、没有产出者**，`$EVAL_DIR` 里也没有该文件 ⇒ 生成的 `run-<tier>-eval.sh`
跑到评测步必然 `bash: …/run-eval.sh: No such file or directory`（rc=127），整档判据④（vLLM 推理）
无法取证。本文件补的就是这一个缺件，且**只补一份**（⛔ 不另立第二套实现、⛔ 不往生成脚本里内联）。

## 契约（唯一真相源；机核断言见 `tests/scripts/test_eval_runner_install.py`）

| 项 | 值 |
|---|---|
| 落点 | `<EVAL_DIR>/run-eval.sh`（宿主路径，0755）+ `<EVAL_DIR>/run-eval.sh.meta.json`（sha 记录） |
| 调用 | `bash run-eval.sh <TREE> <EVALDIR> <TIER> <ROLE> <EVALCFG>` |
| 入参语义 | `TREE` 部署树根（挂 `/workspace/graspo:ro`，另 `$TREE/runs → /runs:ro`）；`EVALDIR` 评测根（挂 `/evalout`）；`TIER` M 档号（只做容器名/日志）；`ROLE` ∈ `after|base|teacher`（= 报告子目录口径，与 `scripts/record_checks.py:102/108` 同源）；`EVALCFG` 评测 yaml（必须位于 `EVALDIR` 内，容器内以 `/evalout/<basename>` 引用） |
| 产物 | `<EVALDIR>/.local/reports/<ROLE>/eval_report.json`（`graspo eval run` 写）+ `<EVALDIR>/evidence/eval-<ROLE>-stdout.log`（本脚本 tee） |
| 退出码 | 0 成功/自检通过；1 用法错；4 **待训练产物缺失**（模型/检查点目录不在盘，fail-closed）；5 配置/落点口径不符（`output_dir` 不是本 role 的独立目录、报告目录非空、`EVALCFG` 不在 `EVALDIR` 内）；6 vLLM 端点未就绪；7 评测命令 rc=0 但报告未产出；其他 = `graspo eval run` 的真实 rc 原样透传 |
| 自检 | `run-eval.sh -h`|`--help`（用法，rc 0）；`--check-only [--self] [--no-endpoint] <5 参>`（静态自检 + 打印将执行的命令行，rc 0 = 可跑） |

## 单一真相源（⛔ 不复制评测逻辑）

- **镜像名**：渲染时从 `scripts/gen_eval.py` 的 `IMG` / `BASEIMG` 取（运行期可被
  `GRASPO_EVAL_IMG` / `GRASPO_EVAL_BASEIMG` 覆写）——本文件不写第二份镜像字面量；
- **评测客户端**：仍是既有 `graspo eval run --eval-config …`（容器内 CLI），本脚本只做
  参数/落点/产物三件事，⛔ 不重算任何判据、不碰阈值；
- **vLLM 起服**：归 `scripts/eval_serve_vllm.sh`（唯一入口）——本脚本只**指向**它（端点未就绪时
  给出该脚本的调用配方），⛔ 不自己起容器（起服要占卡，属调用方脚本职责）；
- **宿主根**：`GRASPO_MODEL_HOST_ROOT` / `GRASPO_ELAM_V5_DIR` 与 `scripts/launch_pipeline.py` 的
  `MODEL_HOST_ROOT` / `ELAM_HOST_DIR` **同一规则**（缺省 `$HOME/models` / `$HOME/elam-v5-balanced`），
  §15.1：源码里不留任何宿主路径或内网地址字面量（一律运行期取值）。

## 唯一路径 vs 被禁的第二条

- ★ **唯一路径** = 本文件渲染出的**通用** `run-eval.sh`（`<TREE> <EVALDIR> <TIER> <ROLE> <EVALCFG>`），
  由 `scripts/gen_eval.py` 的评测前置自动安装到 `$EVAL_DIR`（幂等、sha 记录）。
- ⛔ **禁用**：为每个档各生成一份 `$EVAL_DIR/run-eval.sh`（档号内联进去）。理由是调用方
  （`gen_eval.py:252/267`、`launch_pipeline.py:299/308`）**始终显式传档号**，一对多只会让
  `$EVAL_DIR` 里出现互相覆盖的同名件；本文件的存在就是为了让"每档一份"这条路不可能被走回来
  （`install()` 遇到异内容即拒写，须显式 `--force` 且留备份）。

## 用法

```
python3 scripts/eval_runner.py --help
python3 scripts/eval_runner.py install --eval-dir <EVAL_DIR>                 # 只打印计划（零写入）
python3 scripts/eval_runner.py install --eval-dir <EVAL_DIR> --apply         # 落位（幂等；异内容拒写）
python3 scripts/eval_runner.py install --eval-dir <EVAL_DIR> --apply --force # 覆盖（先备份旧件）
python3 scripts/eval_runner.py check   --eval-dir <EVAL_DIR>                 # 验 sha/权限/语法，rc 0=就绪
python3 scripts/eval_runner.py render                                       # 打印脚本全文（供审查/对拍）
python3 scripts/eval_runner.py show    --eval-dir <EVAL_DIR>                 # 打印将执行的评测命令行
```
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:                                     # 作为脚本直跑 / 被 import 都成立
    sys.path.insert(0, str(HERE))

ROOT = HERE.parent

#: 评测入口文件名（落点 = `<EVAL_DIR>/<RUNNER_NAME>`；调用方按此名拼路径）。
RUNNER_NAME = "run-eval.sh"
#: sha 记录的落点（与脚本同目录，机器可读：模板版本 + sha256 + 字节数 + 安装时刻）。
META_NAME = "run-eval.sh.meta.json"
#: 模板版本：模板有语义改动时必须 +1（`check()` 会把版本不符算作"需重新安装"）。
TEMPLATE_VERSION = "1"
#: CLI 允许的 role（= 报告子目录口径；与 `scripts/record_checks.py:102/108` 的
#: `AFTER_REPORT_REL` / `TEACHER_REPORT_REL` 同一族，⛔ 不另立命名）。
ROLES = ("after", "base", "teacher")

#: 退出码表（脚本与本文件共用同一份常量，测试逐条断言）。
RC = {
    "ok": 0,
    "usage": 1,
    "blocked": 2,          # 安装被拒（异内容 / sha 不符）—— 只在 install/check 侧用
    "no_artifact": 4,      # 待训练产物缺失（fail-closed）
    "cfg": 5,              # 配置 / 落点口径不符 / 报告目录非空
    "endpoint": 6,         # vLLM 端点未就绪
    "no_report": 7,        # 命令 rc=0 但报告未产出（无证据的成功）
}

#: §15.1 机密：本文件与渲染产物**都不得出现**的宿主/内网形态（测试逐条扫）。
#: ★ 刻意只写**通用形状**（具体账号/主机名的清单由 `.local/` 外置，见
#:   `tools/matrix/scan_matrix_leaks.py` 的分工）——检测器自身不得成为泄漏源。
DENY_PATTERNS = (
    r"/data/[a-z0-9_]+/", r"/home/[A-Za-z0-9_.-]+/", r"/mnt/[a-z]/",
    r"\b10\.\d{1,3}\.\d{1,3}\.\d{1,3}\b", r"\b192\.168\.\d{1,3}\.\d{1,3}\b",
    r"\b172\.(1[6-9]|2\d|3[01])\.\d{1,3}\.\d{1,3}\b",
)

#: `run-eval.sh` 模板（**唯一一份**；占位符用 `%%NAME%%`，避开 bash 的 `${…}` 花括号）。
TEMPLATE = r"""#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# run-eval.sh —— 评测客户端执行器（评测入口契约的唯一实现）
#
#   ★ 本文件由 `scripts/eval_runner.py` 渲染（模板在该文件里，是唯一真相源）。
#     ⛔ 不得手改：`python3 scripts/eval_runner.py check --eval-dir <EVAL_DIR>` 会因 sha 不符拒跑；
#        要改行为请改模板后重新 install（`install --eval-dir <EVAL_DIR> --apply --force`，旧件自动备份）。
#     模板版本 = %%TEMPLATE_VERSION%%
#
# 用法：
#   bash run-eval.sh <TREE> <EVALDIR> <TIER> <ROLE> <EVALCFG>
#   bash run-eval.sh -h | --help
#   bash run-eval.sh --check-only [--self] [--no-endpoint] <TREE> <EVALDIR> <TIER> <ROLE> <EVALCFG>
#
# 入参（调用方 = `scripts/gen_eval.py` 渲染出的 run-<tier>-eval.sh；三处调用参数序一致）：
#   TREE     部署树根（宿主路径）⇒ 挂为 /workspace/graspo:ro，且 $TREE/runs ⇒ /runs:ro
#   EVALDIR  评测根（宿主路径）  ⇒ 挂为 /evalout
#   TIER     M 档号（只用于容器名与日志；口径由 EVALCFG 决定）
#   ROLE     after | base | teacher（= 报告子目录口径）
#   EVALCFG  评测 yaml（宿主路径，必须位于 EVALDIR 内）
#
# 产物：<EVALDIR>/.local/reports/<ROLE>/eval_report.json
#       <EVALDIR>/evidence/eval-<ROLE>-stdout.log
#
# 退出码：
#   0  成功 / 自检通过
#   1  用法错误
#   4  待训练产物缺失（模型或检查点目录不在盘）—— fail-closed，⛔ 不用空目录糊过去
#   5  配置/落点口径不符（output_dir 不是本 ROLE 的独立目录、报告目录非空、EVALCFG 不在 EVALDIR 内）
#   6  vLLM 端点未就绪（起服归 scripts/eval_serve_vllm.sh；本脚本不自己起容器）
#   7  评测命令 rc=0 但报告未产出（无证据的成功 ⇒ 不算成功）
#   其他 = `graspo eval run` 的真实 rc 原样透传
# ─────────────────────────────────────────────────────────────────────────────
set -uo pipefail

IMAGE_RUN="%%IMAGE_RUN%%"
IMAGE_BASE="%%IMAGE_BASE%%"
GENERATOR="scripts/eval_runner.py"
TEMPLATE_VERSION="%%TEMPLATE_VERSION%%"

RC_USAGE=1
RC_NO_ARTIFACT=4
RC_CFG=5
RC_ENDPOINT=6
RC_NO_REPORT=7

say() { printf '%s %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*"; }
die() { local rc="$1"; shift; say "FATAL: $*"; exit "$rc"; }

usage() {
  cat <<'USAGE'
用法：
  bash run-eval.sh <TREE> <EVALDIR> <TIER> <ROLE> <EVALCFG>
  bash run-eval.sh -h | --help
  bash run-eval.sh --check-only [--self] [--no-endpoint] <TREE> <EVALDIR> <TIER> <ROLE> <EVALCFG>

TREE     部署树根（宿主路径）⇒ /workspace/graspo:ro ＋ $TREE/runs ⇒ /runs:ro
EVALDIR  评测根（宿主路径）  ⇒ /evalout
TIER     M 档号（只用于容器名/日志）
ROLE     after | base | teacher（= 报告子目录口径）
EVALCFG  评测 yaml（宿主路径，必须位于 EVALDIR 内）

产物：<EVALDIR>/.local/reports/<ROLE>/eval_report.json ＋ <EVALDIR>/evidence/eval-<ROLE>-stdout.log
退出码：0 成功/自检通过；1 用法；4 待训练产物缺失；5 配置/落点不符；6 vLLM 未就绪；
        7 报告未产出；其他 = grasp eval run 的真实 rc
USAGE
}

MODE=run
SELF_ONLY=0
CHECK_ENDPOINT=1
POS=()
for arg in "$@"; do
  case "$arg" in
    -h|--help) usage; exit 0 ;;
    --check-only) MODE=check ;;
    --self) SELF_ONLY=1 ;;
    --no-endpoint) CHECK_ENDPOINT=0 ;;
    -*) say "FATAL: 未知选项 $arg（-h 看用法）"; exit "$RC_USAGE" ;;
    *) POS+=("$arg") ;;
  esac
done

# ── 自检：脚本自身完整性（--self 且无位置参数时不碰环境）──────────────────
self_check() {
  local bad=0
  [ -n "$IMAGE_RUN" ] || { say "  ✘ IMAGE_RUN 为空（模板渲染失败）"; bad=1; }
  [ -n "$IMAGE_BASE" ] || { say "  ✘ IMAGE_BASE 为空（模板渲染失败）"; bad=1; }
  case "$IMAGE_RUN" in *:*) : ;; *) say "  ✘ IMAGE_RUN=$IMAGE_RUN 不含 tag"; bad=1 ;; esac
  [ -n "$TEMPLATE_VERSION" ] || { say "  ✘ TEMPLATE_VERSION 为空"; bad=1; }
  bash -n "$0" || { say "  ✘ bash -n 未过"; bad=1; }
  say "  镜像 run=$IMAGE_RUN base=$IMAGE_BASE ；模板版本=$TEMPLATE_VERSION ；生成器=$GENERATOR"
  say "  ROLE 允许集：after base teacher（= .local/reports/<ROLE>/eval_report.json）"
  [ "$bad" -eq 0 ] || return 1
  return 0
}

if [ "$SELF_ONLY" -eq 1 ] || [ "$MODE" = check ]; then
  if [ "${#POS[@]}" -eq 0 ]; then
    say "=== run-eval.sh 自检（--self，仅脚本自身，不碰模型/卡/端点）==="
    self_check || die "$RC_USAGE" "自检未过 ⇒ 脚本件损坏，重新 install"
    say "SELF-CHECK PASS"
    exit 0
  fi
fi

# ── 1. 位置参数域 ───────────────────────────────────────────────────────────
[ "${#POS[@]}" -eq 5 ] || die "$RC_USAGE" "需要 5 个位置参数（TREE EVALDIR TIER ROLE EVALCFG），实得 ${#POS[@]}；-h 看用法"
T="${POS[0]}"; EV="${POS[1]}"; TIER="${POS[2]}"; ROLE="${POS[3]}"; CFG="${POS[4]}"
for pair in "TREE=$T" "EVALDIR=$EV" "TIER=$TIER" "ROLE=$ROLE" "EVALCFG=$CFG"; do
  [ -n "${pair#*=}" ] || die "$RC_USAGE" "参数为空：${pair%%=*}（fail-closed，⛔ 不回落成默认值）"
done
case "$ROLE" in
  after|base|teacher) : ;;
  *) die "$RC_USAGE" "ROLE=$ROLE 不在允许集 {after base teacher}" ;;
esac
[ -d "$T" ] || die "$RC_CFG" "TREE 不是目录：$T"
[ -d "$EV" ] || die "$RC_CFG" "EVALDIR 不是目录：$EV（评测前置：先 python3 scripts/eval_runner.py install --eval-dir $EV --apply）"
[ -f "$CFG" ] || die "$RC_CFG" "评测配置不存在：$CFG（先跑 scripts/gen_eval.py 生成 eval-<tier>-<role>.yaml）"
EV_ABS="$(cd "$EV" && pwd)"
CFG_ABS="$(cd "$(dirname "$CFG")" && pwd)/$(basename "$CFG")"
[ "$(dirname "$CFG_ABS")" = "$EV_ABS" ] \
  || die "$RC_CFG" "评测配置必须在 EVALDIR 之内（容器里按 /evalout/<basename> 引用）：$CFG_ABS 不在 $EV_ABS 下"

# ── 2. 配置口径（output_dir / role 目录 / 模型目录）──────────────────────────
cfg_get() { sed -n "s/^$1:[[:space:]]*//p" "$CFG" | head -1 | sed 's/[[:space:]]*$//' | sed 's/^["'\'']//; s/["'\'']$//'; }

OUT_DIR="$(cfg_get output_dir)"
CKPT="$(cfg_get checkpoint_path)"
BASEM="$(cfg_get base_model_path)"
ENDPOINT="$(cfg_get endpoint)"
WANT_OUT="/evalout/.local/reports/$ROLE"

[ -n "$OUT_DIR" ] || die "$RC_CFG" "评测配置缺 output_dir：$CFG_ABS"
[ "$OUT_DIR" = "$WANT_OUT" ] \
  || die "$RC_CFG" "output_dir=$OUT_DIR ≠ $WANT_OUT（ROLE=$ROLE）⇒ 拒跑（防两侧相撞/串档）"
[ -n "$ENDPOINT" ] || die "$RC_CFG" "评测配置缺 endpoint：$CFG_ABS"

# 容器内路径 → 宿主路径（**与 scripts/launch_pipeline.py 的 MODEL_HOST_ROOT / ELAM_HOST_DIR 同一规则**；
# §15.1：缺省值一律运行期取值，⛔ 本脚本不含宿主路径字面量）
host_map() {
  local p="$1"
  case "$p" in
    /models/*) printf '%s/%s' "${GRASPO_MODEL_HOST_ROOT:-$HOME/models}" "${p#/models/}" ;;
    /runs/*)   printf '%s/%s' "${T%/}/runs" "${p#/runs/}" ;;
    /evalout/*) printf '%s/%s' "$EV_ABS" "${p#/evalout/}" ;;
    *) printf '%s' "$p" ;;
  esac
}

check_artifact() {   # $1=字段名 $2=配置值
  local field="$1" val="$2" host
  case "$val" in
    ""|null|"~") say "  $field = <null>（本 role 不适用，跳过）"; return 0 ;;
  esac
  case "$val" in
    /*) : ;;
    *) die "$RC_NO_ARTIFACT" "待训练产物：$field=$val 不是绝对路径 ⇒ 评测不可跑（fail-closed）" ;;
  esac
  host="$(host_map "$val")"
  if [ -d "$host" ]; then
    say "  $field = $val → 宿主 $host [在盘]"
    [ -n "$(ls -A "$host" 2>/dev/null)" ] || say "    ⚠ 该目录为空（可能只是占位）：$host"
    return 0
  fi
  die "$RC_NO_ARTIFACT" "待训练产物缺失：$field=$val（宿主 $host 不存在）⇒ 训练尚未产出可评测的目录，本档评测不可跑；⛔ 不用空目录糊过去（那会让评测跑出无意义读数）"
}

say "=== $TIER eval role=$ROLE（MODE=$MODE）==="
say "TREE=$T EVALDIR=$EV_ABS CFG=$CFG_ABS"
say "--- 2.1 评测配置落点 ---"
say "  output_dir = $OUT_DIR（= 本 ROLE 独立目录 ✓）"
say "  endpoint   = $ENDPOINT"
say "--- 2.2 可评测产物（fail-closed：缺 ⇒ rc=$RC_NO_ARTIFACT）---"
check_artifact checkpoint_path "$CKPT"
check_artifact base_model_path "$BASEM"

# 报告目录：防覆盖（真跑前必须空/不存在）
OUT_HOST="$EV_ABS/.local/reports/$ROLE"
if [ -n "$(ls -A "$OUT_HOST" 2>/dev/null)" ]; then
  die "$RC_CFG" "报告目录已存在且非空 ⇒ 拒跑（防覆盖上一轮证据）：$OUT_HOST（如需重跑请先归档/移走该目录）"
fi

# ── 3. 命令行（打印 + 执行）──────────────────────────────────────────────────
mkdir -p "$EV_ABS/evidence"
STDOUT_LOG="$EV_ABS/evidence/eval-$ROLE-stdout.log"
TS="$(date -u +%s)"
ELAM_HOST="${GRASPO_ELAM_V5_DIR:-$HOME/elam-v5-balanced}"
CMD=(docker run --rm --network=host --name "${TIER,,}-eval-$ROLE-$TS"
     -e "GRASPO_IMG=$IMAGE_RUN" -e "GRASPO_BASEIMG=$IMAGE_BASE"
     -v "$T:/workspace/graspo:ro" -v "$T/runs:/runs:ro" -v "$EV_ABS:/evalout"
     -v "$ELAM_HOST:/work/elam-v5:ro"
     -w /workspace/graspo -e "PYTHONPATH=/workspace/graspo/src" "$IMAGE_RUN"
     graspo eval run --eval-config "/evalout/$(basename "$CFG_ABS")")

render_cmd() {   # 供人读的命令行（执行时用数组，⛔ 不 eval 字符串）
  local out="" w
  for w in "$@"; do
    case "$w" in
      *[[:space:]]*) out+=" \"$w\"" ;;
      *) out+=" $w" ;;
    esac
  done
  printf '%s' "${out# }"
}
say "--- 3.1 将执行的评测命令 ---"
say "  $(render_cmd "${CMD[@]}")"
say "  stdout 双份留证 ⇒ $STDOUT_LOG"
say "  （vLLM 起服不在本脚本职责内：由调用方 run-<tier>-eval.sh 起；手动起服用 scripts/eval_serve_vllm.sh）"

# 端点就绪（vLLM 未起 ⇒ 明确报错，⛔ 不静默撞连接错误）
if [ "$CHECK_ENDPOINT" -eq 1 ]; then
  if ! command -v curl >/dev/null 2>&1; then
    die "$RC_ENDPOINT" "本机无 curl ⇒ 无法确认 vLLM 端点 $ENDPOINT 就绪（fail-closed）"
  fi
  if ! curl -sf -m 5 "$ENDPOINT/v1/models" 2>/dev/null | grep -q graspo-eval; then
    cat >&2 <<EOF
FATAL: vLLM 端点未就绪（$ENDPOINT/v1/models 未返回 served_model_name=graspo-eval）
  起服归 scripts/eval_serve_vllm.sh（唯一入口），例如：
    EVAL_MODEL_DIR=<宿主模型目录> EVAL_GPUS=<显式卡集> EVAL_PORT=<端口> bash scripts/eval_serve_vllm.sh
  ⛔ 本脚本不自己起容器（起服占卡，属调用方脚本职责；也避免与调用方抢卡）
EOF
    exit "$RC_ENDPOINT"
  fi
  say "  vLLM 端点就绪 ✓"
fi

if [ "$MODE" = check ]; then
  if [ "$CHECK_ENDPOINT" -eq 1 ]; then
    say "CHECK PASS（静态自检全过：落点/配置/产物/端点齐备；未执行评测、未起容器）"
  else
    say "CHECK PASS（静态自检全过：落点/配置/产物（--no-endpoint：未查端点）；未执行评测、未起容器）"
  fi
  exit 0
fi

# ── 4. 执行 + 证据断言 ─────────────────────────────────────────────────────
mkdir -p "$EV_ABS/.local/reports/$ROLE"
"${CMD[@]}" 2>&1 | tee "$STDOUT_LOG"
RC_EVAL=${PIPESTATUS[0]}
say "eval rc=$RC_EVAL"
REPORT="$OUT_HOST/eval_report.json"
if [ ! -f "$REPORT" ]; then
  if [ "$RC_EVAL" -eq 0 ]; then
    die "$RC_NO_REPORT" "评测命令 rc=0 但报告未产出 ⇒ 无证据的成功，不算成功（fail-closed）：$REPORT"
  fi
  die "$RC_EVAL" "评测命令 rc=$RC_EVAL 且报告未产出：$REPORT（stdout 见 $STDOUT_LOG）"
fi
say "report OK: $REPORT ($(stat -c%s "$REPORT") B)"
say "=== $TIER eval role=$ROLE DONE rc=$RC_EVAL ==="
exit "$RC_EVAL"
"""


@dataclass(frozen=True)
class InstallResult:
    """`install()` 的结果（`status` 决定调用方怎么打印/怎么 fail-closed）。"""

    status: str          # installed | unchanged | would-install | blocked | forced
    path: Path
    sha256: str
    meta_path: Path
    message: str


def _images() -> tuple[str, str]:
    """镜像名**只**从 `scripts/gen_eval.py` 取（⛔ 本文件不写第二份字面量）。"""
    import gen_eval as ge  # noqa: PLC0415  （同目录；模块级只读 env，无副作用）

    return str(ge.IMG), str(ge.BASEIMG)


def render(*, images: tuple[str, str] | None = None) -> str:
    """渲染 `run-eval.sh` 全文（纯函数：不写盘、不碰环境）。"""
    img_run, img_base = images or _images()
    text = TEMPLATE
    for name, val in (("%%IMAGE_RUN%%", img_run), ("%%IMAGE_BASE%%", img_base),
                      ("%%TEMPLATE_VERSION%%", TEMPLATE_VERSION)):
        if name not in TEMPLATE:
            raise RuntimeError(f"模板缺占位符 {name} ⇒ 模板损坏")
        text = text.replace(name, val)
    # ⚠ 只查"未替换的占位符形态"（模板里的 `${pair%%=*}` 这类 bash 参数展开合法含有 `%%`）
    leftover = re.findall(r"%%[A-Z][A-Z0-9_]*%%", text)
    if leftover:
        raise RuntimeError(f"渲染后仍有未替换的占位符 {leftover} ⇒ 模板损坏")
    if not img_run or not img_base:
        raise RuntimeError("镜像名为空（scripts/gen_eval.py 的 IMG/BASEIMG）⇒ fail-closed")
    return text


def sha256_of(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def meta_path_of(eval_dir: Path) -> Path:
    return Path(eval_dir) / META_NAME


def runner_path_of(eval_dir: Path) -> Path:
    return Path(eval_dir) / RUNNER_NAME


def _now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def install(eval_dir: str | Path, *, apply: bool = False, force: bool = False,
            backup: bool = True) -> InstallResult:
    """把渲染出的 `run-eval.sh` 幂等装到 `eval_dir`（★ 唯一落位通道）。

    幂等与防呆（宪法 §2.3/§2.4）：
      · 目标不存在          ⇒ 写入（0755）+ 写 meta ⇒ `installed`；
      · 目标存在且**同内容** ⇒ 不动内容（必要时补 0755/补 meta）⇒ `unchanged`；
      · 目标存在且**异内容** ⇒ **拒写**（`status="blocked"`，`apply=False` 时也算计划里可见的阻断）；
        确需覆盖须显式 `force=True`（先把旧件备份成 `run-eval.sh.bak.<sha8>`）⇒ `forced`。
    ⛔ 永不静默覆盖：`$EVAL_DIR/run-eval.sh` 是**唯一**评测入口，被另一份实现顶替会让"契约唯一"失效。
    """
    ev = Path(eval_dir)
    if not ev.is_dir():
        raise FileNotFoundError(f"EVAL_DIR 不存在（先建目录）：{ev}")
    text = render()
    data = text.encode("utf-8")
    sha = sha256_of(data)
    dst = runner_path_of(ev)
    meta = meta_path_of(ev)
    meta_doc = {
        "schema": "graspo-eval-runner/1",
        "name": RUNNER_NAME,
        "template_version": TEMPLATE_VERSION,
        "sha256": sha,
        "bytes": len(data),
        "generator": "scripts/eval_runner.py",
        "contract": "bash run-eval.sh <TREE> <EVALDIR> <TIER> <ROLE> <EVALCFG>",
        "installed_at": _now(),
    }

    if dst.exists():
        old = dst.read_bytes()
        old_sha = sha256_of(old)
        if old_sha == sha:
            if apply:
                dst.chmod(0o755)
                _write_meta(meta, meta_doc)
            return InstallResult("unchanged", dst, sha, meta,
                                 f"{dst} 已存在且内容一致（sha256={sha[:16]}…）⇒ 不覆盖")
        if not apply:
            return InstallResult("blocked", dst, sha, meta,
                                 f"{dst} 已存在但内容不同（现有 {old_sha[:16]}… ≠ 模板 {sha[:16]}…）"
                                 f"⇒ 需 --force 才覆盖（⛔ 不静默覆盖唯一评测入口）")
        if not force:
            return InstallResult("blocked", dst, sha, meta,
                                 f"{dst} 已存在但内容不同（现有 {old_sha[:16]}… ≠ 模板 {sha[:16]}…）"
                                 f"⇒ 拒写；确认要覆盖请加 --force（旧件会备份为 {RUNNER_NAME}.bak.{old_sha[:8]}）")
        if backup:
            dst.replace(dst.with_name(f"{RUNNER_NAME}.bak.{old_sha[:8]}"))
        dst.write_bytes(data)
        dst.chmod(0o755)
        _write_meta(meta, meta_doc)
        return InstallResult("forced", dst, sha, meta,
                             f"{dst} 已覆盖（旧 sha {old_sha[:16]}… → 新 sha {sha[:16]}…，旧件已备份）")

    if not apply:
        return InstallResult("would-install", dst, sha, meta,
                             f"将写入 {dst}（sha256={sha[:16]}…，0755）＋ {meta.name}")
    dst.write_bytes(data)
    dst.chmod(0o755)
    _write_meta(meta, meta_doc)
    return InstallResult("installed", dst, sha, meta,
                         f"已写入 {dst}（sha256={sha[:16]}…，0755）＋ {meta.name}")


def _write_meta(meta: Path, doc: dict) -> None:
    meta.write_text(json.dumps(doc, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def check(eval_dir: str | Path) -> tuple[bool, list[str], dict]:
    """校验落位件是否就绪（sha / 模板版本 / 权限 / `bash -n`）。返回 `(ok, 逐条消息, meta)`。"""
    ev = Path(eval_dir)
    msgs: list[str] = []
    dst = runner_path_of(ev)
    meta = meta_path_of(ev)
    ok = True
    if not ev.is_dir():
        return (False, [f"✘ EVAL_DIR 不存在：{ev}"], {})
    if not dst.is_file():
        return (False, [f"✘ 缺 {dst}（评测入口未落位）⇒ python3 {Path(__file__).name} install "
                        f"--eval-dir {ev} --apply"], {})
    text = render()
    sha = sha256_of(text.encode("utf-8"))
    have = sha256_of(dst.read_bytes())
    msgs.append(f"{'✓' if have == sha else '✘'} sha256：现有 {have[:16]}… / 模板 {sha[:16]}…")
    ok = ok and have == sha
    if meta.is_file():
        try:
            doc = json.loads(meta.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            doc = {}
            msgs.append(f"✘ {meta.name} 不是合法 json")
            ok = False
        if doc.get("sha256") and doc["sha256"] != have:
            msgs.append(f"✘ {meta.name} 记录的 sha 与在盘件不符（{str(doc.get('sha256'))[:16]}…）")
            ok = False
        if doc.get("template_version") and doc["template_version"] != TEMPLATE_VERSION:
            msgs.append(f"✘ 模板版本不符（记录 {doc['template_version']} / 现 {TEMPLATE_VERSION}）⇒ 需重新安装")
            ok = False
    else:
        msgs.append(f"✘ 缺 sha 记录 {meta.name} ⇒ 无法核「落位件即模板产物」")
        ok = False
        doc = {}
    mode = dst.stat().st_mode & 0o777
    msgs.append(f"{'✓' if mode == 0o755 else '✘'} 权限 {oct(mode)}（期望 0o755）")
    ok = ok and mode == 0o755
    p = subprocess.run(["bash", "-n", str(dst)], capture_output=True, text=True)
    msgs.append(f"{'✓' if p.returncode == 0 else '✘'} bash -n rc={p.returncode}")
    ok = ok and p.returncode == 0
    p2 = subprocess.run(["bash", str(dst), "--check-only", "--self"], capture_output=True, text=True)
    msgs.append(f"{'✓' if p2.returncode == 0 else '✘'} 脚本自检 --self rc={p2.returncode}")
    ok = ok and p2.returncode == 0
    return (ok, msgs, doc)


def show_command(eval_dir: str | Path) -> list[str]:
    """打印"落位件会执行的评测命令形态"（静态，不执行；不含任何宿主字面量）。"""
    dst = runner_path_of(Path(eval_dir))
    return [f"{dst} <TREE> <EVALDIR> <TIER> <ROLE> <EVALCFG>",
            f"  即：bash {dst} <部署树根> {Path(eval_dir)} <M0nn> after|base <EVAL_DIR>/eval-<tier>-<role>.yaml"]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="生成/安装/校验 <EVAL_DIR>/run-eval.sh（评测入口契约的唯一产出者）",
        epilog="落位后由 scripts/gen_eval.py 生成的 run-<tier>-eval.sh 以 "
               "`bash <EVAL_DIR>/run-eval.sh <TREE> <EVALDIR> <TIER> <ROLE> <EVALCFG>` 调用。")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_inst = sub.add_parser("install", help="把 run-eval.sh 落到 <EVAL_DIR>（默认只打印计划，--apply 才写）")
    p_inst.add_argument("--eval-dir", required=True, help="评测根（$EVAL_DIR，宿主路径）")
    p_inst.add_argument("--apply", action="store_true", help="真写盘（缺省只打印计划，零写入）")
    p_inst.add_argument("--force", action="store_true", help="内容不一致时覆盖（旧件自动备份）")
    p_inst.add_argument("--no-backup", action="store_true", help="--force 覆盖时不备份（不推荐）")

    p_chk = sub.add_parser("check", help="校验落位件（sha/版本/权限/bash -n/自检）")
    p_chk.add_argument("--eval-dir", required=True)

    p_render = sub.add_parser("render", help="打印 run-eval.sh 全文（供审查/对拍，零写入）")
    p_render.add_argument("--out", default=None, help="可选：写到该文件（缺省 stdout）")

    p_show = sub.add_parser("show", help="打印落位件将执行的评测命令形态")
    p_show.add_argument("--eval-dir", required=True)

    args = ap.parse_args(argv)

    if args.cmd == "render":
        text = render()
        if args.out:
            Path(args.out).write_text(text, encoding="utf-8")
            print(f"WROTE {args.out}")
        else:
            sys.stdout.write(text)
        return RC["ok"]

    if args.cmd == "show":
        for line in show_command(args.eval_dir):
            print(line)
        return RC["ok"]

    if args.cmd == "install":
        try:
            res = install(args.eval_dir, apply=args.apply, force=args.force,
                          backup=not args.no_backup)
        except FileNotFoundError as exc:
            print(f"FATAL: {exc}", file=sys.stderr)
            return RC["blocked"]
        print(f"[{res.status}] {res.message}")
        if res.status == "blocked":
            return RC["blocked"]
        if res.status in ("installed", "forced", "unchanged"):
            print(f"meta → {res.meta_path}")
            print(f"下一步：bash {res.path} --check-only --self")
        return RC["ok"]

    ok, msgs, doc = check(args.eval_dir)
    print(f"=== {args.eval_dir}/{RUNNER_NAME} 校验 ===")
    for m in msgs:
        print("  " + m)
    if doc:
        print(f"  记录：模板版本 {doc.get('template_version')} / sha256 {str(doc.get('sha256'))[:16]}…"
              f" / {doc.get('bytes')} B / 安装于 {doc.get('installed_at')}")
    print("CHECK PASS（落位件即模板产物、可跑）" if ok else
          f"CHECK FAIL（rc={RC['blocked']}）⇒ 重新 install")
    return RC["ok"] if ok else RC["blocked"]


if __name__ == "__main__":
    sys.exit(main())
