#!/usr/bin/env bash
# run_matrix54_source.sh —— **逐档**数据源接线层（逐档跑批用；不改 runner、不改 driver）
#
# ★ 迁址说明（2026-10-10，宪法合规化 B1）：本文件原为 `rig/run_matrix54_v2source.sh`。
#   `rig/` 整树按裁定 1 退役到 `.local/repo-moved-out/`，但本文件是**全仓唯一的逐档数据源
#   启动器** ⇒ 作为唯一例外保留，迁到 `tests/e2e/` 并去掉 `_v2` 版本后缀（§18.1）；
#   同时把清单默认路径同步为 `tests/e2e/matrix54_manifest_source.json`。
#   原过渡期核对件 `rig/verify_runner_parity.py` 已随 rig/ 退役 ⇒ 本层不再做该核对。
#
# 问题（为什么必须有这一层）
# --------------------------
# `tests/e2e/run_matrix54.sh` 生成训练子集时只有**一个**读取源：
#     ELAM_MINI_PATH="${GRASPO_ELAM_MINI_JSONL:-$ELAM_HOST/mini-dataset/mini-short-mm-train.jsonl}"
#     head -n "$SUBSET" "$ELAM_MINI_PATH" > "$RUN_DIR/subsets/$TIER.jsonl"
# 而 v2 是**逐档混合**：9 档 GRASPO×native 必须取 **ELAM V5 正式集**前 48/96/192 条
# （T040/T041/T042 按 E-49 例外 = 72/144/288），
# 其余 45 档必须取 **mini 冒烟集**前 100 条。mini 集**不是**正式集头部（按「最短多模态」
# 挑的）⇒ **不能**用一次全局 `export` 切换（会让 45 档拿到错的 100 条）。
#
# 解法（宪法 §6.1 最小、只新增）
# -----------------------------
# 本脚本对**单个档**读出逐档清单里的 `tiers[].data.source_jsonl`，把它设进
# `GRASPO_ELAM_MINI_JSONL` 后 `exec` 调 `run_matrix54.sh`。
#   · 进程作用域：变量随 `exec` 进入子进程，**本进程退出即消失** ⇒ 零跨档残留；
#   · 每档一次 exec ⇒ 同一时刻只有一个档在跑，不存在"另一档继承了错源"；
#   · runner / driver / generate_matrix.py **一行未改**（不碰 tracked 生成物与锁文件）。
#
# 用法（与 run_matrix54.sh 同形；env 变量原样透传）
# -----------------------------
#     export GRASPO_ELAM_HOST_ROOT=<远端机上的 ELAM V5 数据根>
#     export GRASPO_MODELS_HOST_ROOT=<远端机上的模型根>
#     # ★ §15.1：清单只存 ${VAR} 占位符，不存宿主绝对路径 ⇒ 运行前须设：
#     export GRASPO_ELAM_DATA_ROOT=<远端机上的 ELAM V5 原始数据根>
#     export GRASPO_PREGEN_SUBSET_DIR=<冻结子集目录>   # 仅清单展示用；本层不读
#     export GRASPO_ATTEMPT_ROOT=<运行产物根>          # 仅清单展示用；本层不读
#     export GRASPO_RUNNER_MANIFEST=<清单路径>   # 不设则默认 tests/e2e/matrix54_manifest_source.json
#     export RUN_ROOT=<可写运行产物根，如 <实验根>/runs/v2>
#     bash tests/e2e/run_matrix54_source.sh T028            # 真跑
#     bash tests/e2e/run_matrix54_source.sh T028 --dry-run  # 预检（同样会打印生效读取源）
#
# 直跑 runner（绕过本层）时的行为**逐字不变**——本层是纯新增的入口，不是替换。
#
# 纪律
# ----
#   · 只读清单、只设一个环境变量、只 exec；**不写任何文件、不删任何东西**；
#   · 读不到清单/档/字段 ⇒ **warn 后退回 runner 的既有默认源**（= 与直跑逐字同行为），
#     绝不静默用错源，也绝不因此拒绝启动（兼容 v1 清单直接喂进来的用法）；
#   · 清单里的 `${VAR}` 用宿主环境展开；未设的变量**原样保留**，由下面的 `-f` 检查
#     fail-closed 报错并提示该 export 哪个变量（不静默退回错源）；
#   · 唯一 fail-closed：源文件**行数 < 该档 subset_size** ⇒ 直接退出（否则 runner 会在
#     容器起来之后再报 "subset too small"，白烧一次 GPU 起停）。
set -uo pipefail

TIER="${1:?usage: run_matrix54_source.sh <T###> [--dry-run]}"
shift $(( $# > 0 ? 1 : 0 ))

# 仓库根：优先用调用方传入的（runner 快照会把 $0 换到 /tmp，故显式支持）
# ★ 本文件在 tests/e2e/ 下 ⇒ 相对位置推导要上溯 **两级** 才是仓库根（原 rig/ 版只上一级）。
ROOT_DIR="${GRASPO_REPO_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}"
RUNNER="$ROOT_DIR/tests/e2e/run_matrix54.sh"

if [ ! -f "$RUNNER" ]; then
    echo "FATAL(v2source): 找不到 runner：$RUNNER" >&2
    echo "  修法：export GRASPO_REPO_ROOT=<仓库根>（默认从本脚本位置推 <repo>/tests/e2e/../..）" >&2
    exit 3
fi

# 与 runner 同一优先级读清单路径（不设 = 跑 v1 的默认清单，行为逐字同 runner）
MANIFEST="${GRASPO_RUNNER_MANIFEST:-$ROOT_DIR/tests/e2e/matrix54_manifest_source.json}"

# 需要 python3 解析 JSON；没有就退回 runner 默认源（与直跑同行为）
# ⚠ 宿主侧 python 一律带 PYTHONDONTWRITEBYTECODE=1：装配树纪律 = 与基准 commit 逐字节等价，
#   在树内写 __pycache__/*.pyc 会让 `git ls-tree -r <commit>` 逐文件比对（873 文件）失效。
PYBIN="${PYTHON:-python3}"
if ! command -v "$PYBIN" >/dev/null 2>&1; then
    echo "WARN(v2source): 找不到 $PYBIN ⇒ 不覆盖读取源，按 runner 既有默认源继续" >&2
    exec bash "$RUNNER" "$TIER" "$@"
fi

# ── 解析该档的 data.source_jsonl + data.subset_size（stdout 单行，第三个 token = rc）──
PARSED="$(
PYTHONDONTWRITEBYTECODE=1 "$PYBIN" - "$MANIFEST" "$TIER" <<'PY' 2>/dev/null
import json, os, re, sys
try:
    with open(sys.argv[1], encoding="utf-8") as handle:
        manifest = json.load(handle)
except Exception:
    print("-\t0\t2")          # 清单读不到 ⇒ rc=2（退回默认源）
    sys.exit(0)
tier = next((t for t in manifest.get("tiers", []) if t.get("tier_id") == sys.argv[2]), None)
if tier is None:
    print("-\t0\t3")          # 档不在清单里 ⇒ rc=3
    sys.exit(0)
data = tier.get("data") or {}
src = data.get("source_jsonl") or "-"

def _expand(value):
    """清单只存 ${VAR} 占位符（§15.1）⇒ 用宿主环境展开。

    未设的变量**原样保留**（不猜、不填默认）⇒ 下面的 `-f` 检查会 fail-closed
    并提示该 export 哪个变量，绝不静默退回错源。
    """
    def repl(m):
        name = m.group("braced") or m.group("plain")
        return os.environ.get(name, m.group(0))
    return re.sub(
        r"\$\{(?P<braced>[A-Za-z_][A-Za-z0-9_]*)\}|\$(?P<plain>[A-Za-z_][A-Za-z0-9_]*)",
        repl, value,
    )

src = _expand(src)
subset = int(data.get("subset_size") or 0)
print(f"{src}\t{subset}\t0")
PY
)"
PARSED_RC=$?

SOURCE_JSONL="$(printf '%s' "$PARSED" | cut -f1)"
SUBSET_SIZE="$(printf '%s' "$PARSED" | cut -f2)"
LOOKUP_RC="$(printf '%s' "$PARSED" | cut -f3)"
[ -n "$LOOKUP_RC" ] || LOOKUP_RC="2"

if [ "$PARSED_RC" != "0" ] || [ "$LOOKUP_RC" != "0" ] || [ "$SOURCE_JSONL" = "-" ] || [ -z "$SOURCE_JSONL" ]; then
    case "$LOOKUP_RC" in
        3) echo "WARN(v2source): 清单 $MANIFEST 里没有档 $TIER ⇒ 按 runner 既有默认源继续" >&2 ;;
        *) echo "WARN(v2source): 无法从 $MANIFEST 读出 $TIER 的 data.source_jsonl（rc=$LOOKUP_RC）" >&2
           echo "                   ⇒ 按 runner 既有默认源继续（= 与直跑 runner 逐字同行为）" >&2 ;;
    esac
    exec bash "$RUNNER" "$TIER" "$@"
fi

if [ ! -f "$SOURCE_JSONL" ]; then
    echo "FATAL(v2source): 该档源文件不存在：$SOURCE_JSONL" >&2
    case "$SOURCE_JSONL" in
        *'${'*)
            echo "  提示：清单里存的是 \${VAR} 占位符（§15.1）⇒ 请先 export 对应宿主根" >&2
            echo "        常用：GRASPO_ELAM_DATA_ROOT / GRASPO_PREGEN_SUBSET_DIR / GRASPO_ATTEMPT_ROOT" >&2
            ;;
    esac
    echo "  档=$TIER subset_size=$SUBSET_SIZE（清单 $MANIFEST）" >&2
    exit 3
fi
[ -r "$SOURCE_JSONL" ] || { echo "FATAL(v2source): 该档源文件不可读：$SOURCE_JSONL" >&2; exit 3; }

# 行数预检（唯一 fail-closed）：源不够 ⇒ runner 也会报 subset too small，但那时容器已经起停一次
if [ -n "$SUBSET_SIZE" ] && [ "$SUBSET_SIZE" != "0" ]; then
    SOURCE_LINES="$(wc -l < "$SOURCE_JSONL" | tr -d ' ')"
    if [ "$SOURCE_LINES" -lt "$SUBSET_SIZE" ]; then
        echo "FATAL(v2source): 源行数不足 —— $SOURCE_JSONL 只有 $SOURCE_LINES 行 < 该档 subset_size=$SUBSET_SIZE" >&2
        echo "  （$TIER：正式集档需 ≥192 条；mini 档需 ≥100 条）" >&2
        exit 3
    fi
else
    SOURCE_LINES="?"
fi

# ★ 显式即防呆（§2.2）：生效的读取源必须在 **--dry-run 与真跑**都可见，
#   否则"源切没生效"与"源切了但跑的还是旧源"在日志里长得一模一样。
echo "[v2source] tier=$TIER source_jsonl=$SOURCE_JSONL（$SOURCE_LINES 行）"
echo "[v2source] subset_size=$SUBSET_SIZE ⇒ runner 将执行：head -n $SUBSET_SIZE $SOURCE_JSONL > <RUN_ROOT>/$TIER/subsets/$TIER.jsonl"
echo "[v2source] GRASPO_ELAM_MINI_JSONL 仅在本进程（exec 后即消失，不影响任何其他档）"

# ★ 过渡期核对件 rig/verify_runner_parity.py 已随 rig/ 整树退役（2026-10-10，B1 裁定 1）
#   ⇒ 本层不再做该核对；此处显式留注，避免"静默少了一道核对"被当成没发生。

# ── 逐档进程作用域切换读取源，然后 exec runner（argv 原样透传）──────────────
export GRASPO_ELAM_MINI_JSONL="$SOURCE_JSONL"
exec bash "$RUNNER" "$TIER" "$@"
