#!/bin/bash
# GRASPO **68 档（M001–M068）批量执行入口** —— 由 `tests/e2e/matrix68_manifest.json` **现读**驱动。
#
# 用法:
#   bash tests/e2e/run_matrix68.sh                                  # 按清单顺序跑全部 68 档
#   bash tests/e2e/run_matrix68.sh --only M001                      # 只跑一档
#   bash tests/e2e/run_matrix68.sh --only M001,M011,M031 --dry-run   # 逐档只打印命令/闸门/判据
#   bash tests/e2e/run_matrix68.sh --run-root /data/…/r1 --dry-run
#   bash tools/matrix/backfill_dated.sh --runs-dir <run-root>/collect   # 结果回填入口（扁平目录）
#
# 与 `run_matrix54.sh` 的关系（⛔ 本文件不修改 T 版 runner）：
#   · 结构照搬（宿主 uid 运行 / 挂载不变量断言 / 三闸门 / 子集落地 / entry.sh / exit_code 落盘）；
#   · **档位来源换成 M 档清单**：`cards` / `subset_size` / `train_path` / `config` / 步数门槛
#     逐档从 `tests/e2e/matrix68_manifest.json` 现读，⛔ 本脚本**不写死任何档号或卡集**。
#
# 每档判据的**唯一来源**是清单（= 数据件 + 在盘 yaml 现读的派生结果），本脚本只负责：
#   ① 子集落地并断言（行数 == subset_size ∧ sha256 == 清单 subset_sha256）；
#   ② 起跑前三道闸门（cards-busy → 择卡 → gpu_lock_guard → gpu_idle_assert）；
#   ③ 起一次训练（docker run 单容器），把 exit_code / 墙钟 / 显存峰值落成**原始读数**；
#   ④ 跑前/跑后环境快照 + `--diff`（rc=3 ⇒ 本轮标不可比）；
#   ⑤ 交回填口径的结果 json（由 `scripts/make_result_json.py` 产出，⛔ 本脚本不算 status）。
#
# 产物（`<run-root>/<档号>/`）：
#   tier.json                 该档从清单现读的全部字段（审计：跑了什么就是清单里的什么）
#   subsets/<档号>.jsonl      本档训练子集（只读挂进容器）
#   entry.sh                  容器内入口（缓存根预检 → 媒体预检 → 采集 → torchrun）
#   stdout.log                torchrun 的全部输出（宿主侧唯一日志落点）
#   gpu/ps.txt ins.txt        占卡台账原始件（cards-busy 的入参）
#   gpu/peak_mem.json         宿主侧显存采样峰值（拿不到 ⇒ 如实写 null）
#   exit_code                 容器退出码（**原样**，不解释）
#   readings.json             **原始实测读数**（墙钟/退出码/峰值显存/证据路径；缺项写 null）
#                             ★ 判据③④ 的结论也落在这里的 `checks{}`（唯一产出者 =
#                               `scripts/record_checks.py`，评测后执行；缺证据则不写该键）
#   env-snapshot-{before,after}.json / env-diff.json   环境快照与冻结维度比对
#   <档号>.json               回填器口径的结果 json（`status` 由回填器现算，本链不写）
#   collect/<档号>.json       同上结果的**扁平副本**（= 回填入口；回填器 `--runs-dir` 只吃扁平目录）
#
# 环境（都在宿主侧、属环境信息 §16，⛔ 不写进 tracked 文件）：
#   export GRASPO_ELAM_HOST_ROOT=<宿主 ELAM V5 数据根>     （必需：图像根从这里挂）
#   export GRASPO_MODELS_HOST_ROOT=<宿主模型根>            （必需：只读挂到容器内 /models）
#   export GRASPO_IMAGE=<训练镜像 tag>                    （默认见 IMAGE 行）
#   GRASPO_GPU_POSITION_MIB  本档每卡头寸（默认 49152 = 48 GiB）
#   GRASPO_RUN_TIMEOUT       单档训练墙钟上限（秒；默认 7200）
#   GRASPO_GPU_SAMPLER_GRACE_SEC  训练进程返回后**等显存采样器收尾**的上限（秒；默认 30）。
#                            到点仍在跑 ⇒ SIGTERM/SIGKILL 后强回收（⛔ 不改 exit_code）。
#                            与上面那些"环境信息"不同，它**不改变任何读数口径**，只改收尾代价。
#   GRASPO_EVAL_DIR          评测根（可选）：若其下已有 `.local/reports/after/eval_report.json`，
#                            本档跑完就当场产出判据③④ 证据并复跑适配器；未跑评测则只打印后续命令
#   GRASPO_ALGO_CHECKS_DIR   GRASPO 算法检查签署件目录。**只有显式设了才生效**：非空 ⇒ 作为
#                            `--algo-checks-dir` 传给 record_checks（显式目录**不回退**）；
#                            未设 ⇒ 不传该参数，record_checks 按缺省查 <run-dir>/algo-checks、
#                            缺件回退仓内 samples/configs/matrix68-checks/（⛔ 不再默认传一个
#                            从不存在的 <run-root>/algo-checks，那会把回退关掉 ⇒ 判据③ 恒待测试）
#   GRASPO_PUBLIC_ALGOS      SFT/CPT/OPD 判据③ 的公开算法签署件（默认仓内
#                            samples/configs/matrix68-checks/public-algos.json；缺件 ⇒ 不写该键）
#
# 红线：卡 0–7 先到先得、⛔ 不预留/不抢占他人（被占即顺延，⛔ 绝不 kill 他人进程/停他人容器）；
#       评测与训练分卡；本脚本**不自动跑评测**（评测链见 `scripts/gen_eval.py`）。
#       判据③④ 的证据**只能来自实测/签署**：缺证据 ⇒ 不写 `readings.json.checks` ⇒ 回填器
#       fail-closed（该档保持「待测试」）；⛔ 不许手填 true 让流程变绿（见
#       `docs/run-discipline.md` 的「判据③④ 的判定与证据」一节）。
set -uo pipefail

# ══ 挂载结构不变量（fail-closed，纯 argv 分析，不需要 docker）══════════════════════
# 唯一真相源：**任何挂载目标都不得位于另一个只读（ro）挂载目标之下**。
# 原因（run_matrix54.sh 实测确证，rc=125）：父挂载只读时 dockerd 无法在其内部创建嵌套
# mountpoint（`mkdirat …: read-only file system`）⇒ docker 返回 125，容器根本没起来。
# 参数是 `宿主源|容器目标|模式` 三元组；允许测试单独 source 进来调用。
assert_mount_targets_not_under_readonly() {
    local -a specs=("$@")
    local i j src dst mode src2 dst2 mode2
    for ((i = 0; i < ${#specs[@]}; i++)); do
        IFS='|' read -r src dst mode <<< "${specs[$i]}"
        for ((j = 0; j < ${#specs[@]}; j++)); do
            [ "$i" -eq "$j" ] && continue
            IFS='|' read -r src2 dst2 mode2 <<< "${specs[$j]}"
            [ "$mode2" = "ro" ] || continue
            case "${dst}/" in
                "${dst2}/"*)
                    echo "FATAL(runtime-link): 挂载目标 '$dst' 位于只读挂载 '$dst2' 之下 ——" >&2
                    echo "  dockerd 无法在只读文件系统上创建嵌套 mountpoint，会以 rc=125 失败。" >&2
                    echo "  修法：让挂载目标彼此互不嵌套。" >&2
                    return 6
                    ;;
            esac
        done
    done
    return 0
}

# 测试用的库模式：只定义上面的断言函数，不执行主流程（不读清单、不起容器）。
if [ "${GRASPO_RUNNER_LIB_ONLY:-0}" = "1" ]; then
    return 0 2>/dev/null || exit 0
fi

# ══ 参数 ════════════════════════════════════════════════════════════════════════
ONLY=""
DRY_RUN=0
MANIFEST=""
RUN_ROOT=""
while [ $# -gt 0 ]; do
    case "$1" in
        --only)        ONLY="${2:?--only 需要档号（M0nn[,M0nn…]）}"; shift 2 ;;
        --only=*)      ONLY="${1#*=}"; shift ;;
        --dry-run)     DRY_RUN=1; shift ;;
        --run-root)    RUN_ROOT="${2:?--run-root 需要目录}"; shift 2 ;;
        --run-root=*)  RUN_ROOT="${1#*=}"; shift ;;
        --manifest)    MANIFEST="${2:?--manifest 需要路径}"; shift 2 ;;
        --manifest=*)  MANIFEST="${1#*=}"; shift ;;
        # ★ --help 打印**整个文件头**（到 `set -uo pipefail` 为止）：原先写死 `2,40p`，
        #   头部一长（如新增判据③④ 渠道说明）就会把「环境」段截断。
        -h|--help)     awk 'NR>1 && /^set -uo pipefail$/{exit} NR>1' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) echo "FATAL: 未知参数 $1（见 --help）" >&2; exit 2 ;;
    esac
done

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYBIN="${PYTHON:-$ROOT_DIR/.venv/bin/python}"
[ -x "$PYBIN" ] || PYBIN="python3"
MANIFEST="${MANIFEST:-$ROOT_DIR/tests/e2e/matrix68_manifest.json}"
CONFIG_DIR_REL="samples/configs/matrix68"
RUN_ROOT="${RUN_ROOT:-$ROOT_DIR/.local/matrix68-runs}"
IMAGE="${GRASPO_IMAGE:-graspo-msswift:4.5.3}"
CONTAINER_PY="${CONTAINER_PYTHON:-python}"
POSITION_MIB="${GRASPO_GPU_POSITION_MIB:-49152}"
TIMEOUT_SEC="${GRASPO_RUN_TIMEOUT:-7200}"
ELAM_HOST="${GRASPO_ELAM_HOST_ROOT:-}"
MODELS_ROOT="${GRASPO_MODELS_HOST_ROOT:-}"
EVAL_DIR="${GRASPO_EVAL_DIR:-}"
GPU_POOL_CSV="0,1,2,3,4,5,6,7"
# ★ 只有**显式**设了 GRASPO_ALGO_CHECKS_DIR 才把 `--algo-checks-dir` 传给 record_checks：
#   record_checks 的语义是"显式给 ⇒ 只认该目录、不回退"，而 `<run-root>/algo-checks` 从来不存在
#   ⇒ 旧默认值等于把回退关掉，判据③ 每档都缺证据（M016/M052 都是事后手工重跑救回来的）。
#   缺省留空 ⇒ record_checks 走既有回退（仓内 samples/configs/matrix68-checks/）。
ALGO_CHECKS_DIR="${GRASPO_ALGO_CHECKS_DIR:-}"
export PYTHONDONTWRITEBYTECODE=1

say() { printf '%s %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*"; }
fatal() { echo "FATAL: $*" >&2; }

# ══ 宿主侧显存采样器的**有界回收**（2026-10-26 修 D1 的宿主半边）════════════════════
# 为什么必须有：宿主采样器是 `while :; do nvidia-smi …; sleep 2; done` 的**无界**子 shell，
# 旧代码 `kill "$SAMPLER_PID"; wait "$SAMPLER_PID"` 里的 `wait` 没有上限 —— 采样器一旦
# 卡在 nvidia-smi 上（驱动忙/被复位），runner 就与「容器内采样器挂死」一样停不下来，
# 最后被外层超时记成整档失败。M016 的 exit_code=137 是**容器内**那一半造成的（见
# entry.sh 收口段的注释），这里是**同一缺陷的宿主半边**：两半都无界，任何一半卡住都够呛。
# 口径：SIGTERM ⇒ 以 GRASPO_GPU_SAMPLER_GRACE_SEC 为上限等它自己退 ⇒ 到点 SIGKILL ⇒
#       wait 回收。★ 只对本脚本自己起的那个 pid 动手（⛔ 不 pkill、⛔ 不碰他人进程）：
#       卡号台账里的他人进程由闸门 B/C 判，**不在这里杀**。
# ★ 实现要点：用"看门狗 + `wait`"而不是轮询 `kill -0`——子进程退出后会短暂处于僵尸态，
#   `kill -0` 对僵尸**仍然成功**，轮询法会把每档都拖满整个 grace 才返回。
reap_sampler_pid() {
    local pid="${1:-}" grace="${GRASPO_GPU_SAMPLER_GRACE_SEC:-30}" watchdog
    if [ -z "$pid" ]; then
        return 0
    fi
    kill -0 "$pid" 2>/dev/null || { wait "$pid" 2>/dev/null || true; return 0; }
    case "$grace" in ''|*[!0-9]*) grace=30 ;; esac
    [ "$grace" -gt 0 ] || grace=30
    kill -TERM "$pid" 2>/dev/null || true
    (
        sleep "$grace"
        echo "[gpu-summary] WARN: 宿主采样器 ${grace}s 内未退出（pid=$pid）⇒ SIGKILL（只改代价，不改判定）" >&2
        kill -KILL "$pid" 2>/dev/null || true
    ) &
    watchdog=$!
    wait "$pid" 2>/dev/null || true
    kill -TERM "$watchdog" 2>/dev/null || true
    wait "$watchdog" 2>/dev/null || true
    return 0
}

[ -f "$MANIFEST" ] || { fatal "清单不存在：$MANIFEST（先跑 tests/e2e/generate_matrix68_manifest_source.py）"; exit 2; }

# ══ 从清单现读：档号表 ══════════════════════════════════════════════════════════
# ★ 单一解析点：bash 不自己解析 yaml，也不猜任何档位字段 —— 全部字段由清单给出。
manifest_tiers() {
    "$PYBIN" - "$MANIFEST" <<'PY'
import json, sys
m = json.load(open(sys.argv[1], encoding="utf-8"))
for t in m["tiers"]:
    print(t["tier_id"])
PY
}

tier_json() {
    "$PYBIN" - "$MANIFEST" "$1" <<'PY'
import json, sys
m = json.load(open(sys.argv[1], encoding="utf-8"))
hit = [t for t in m["tiers"] if t["tier_id"] == sys.argv[2]]
if not hit:
    raise SystemExit(f"FATAL: 清单里没有档号 {sys.argv[2]}")
t = hit[0]
d = {
    "cards": t["cards"],
    "config": t["config"],
    "subset_size": t["data"]["subset_size"],
    "subset_sha256": t["data"]["subset_sha256"],
    "host_relative_path": t["data"]["host_relative_path"],
    "train_path": t["data"]["train_path"],
    "model_path": t["model_path"],
    "model_env": t["model_host_dir_env_var"],
    "nproc": t["parallel_components"]["nproc_per_node"],
    "max_epochs": t["max_epochs"],
    "planned_steps": t["planned_steps"],
    "per_epoch_steps": t["expected_optimizer_steps_per_epoch"],
    "derived_steps": t["expected_optimizer_steps"],
    "step_gate_threshold": t["step_gate"]["threshold"],
    "algorithm": t["algorithm"],
    "mode": t["mode"],
    "tuner_type": t["tuner_type"],
    "backend": t["backend"],
    "lr": t["lr"],
    "parallel": t["parallel"],
    "train_method": t["train_method"],
}
print(json.dumps(d, ensure_ascii=False))
PY
}

# ══ 档号集合（--only 或全清单）══════════════════════════════════════════════════
ALL_TIERS="$(manifest_tiers)" || exit 5
if [ -n "$ONLY" ]; then
    TARGETS=$(printf '%s' "$ONLY" | tr ',' ' ')
    for t in $TARGETS; do
        if ! printf '%s\n' "$ALL_TIERS" | grep -qx -- "$t"; then
            fatal "--only 指定了清单里不存在的档号：$t"; exit 2
        fi
    done
else
    TARGETS="$ALL_TIERS"
fi
N_TARGETS=$(printf '%s\n' $TARGETS | grep -c . || true)

# ══ 产物属主（容器以宿主跑批用户运行）════════════════════════════════════════════
# 为什么：镜像没有 USER 指令 ⇒ 默认 uid 0 ⇒ 产物 root:root ⇒ 宿主账号删不掉（共享机残留）。
RUN_UID="${GRASPO_RUN_UID:-$(id -u)}"
RUN_GID="${GRASPO_RUN_GID:-$(id -g)}"
case "$RUN_UID" in ''|*[!0-9]*) RUN_UID="" ;; esac
case "$RUN_GID" in ''|*[!0-9]*) RUN_GID="" ;; esac
if [ -z "$RUN_UID" ] || [ -z "$RUN_GID" ]; then
    fatal "无法确定宿主 uid/gid ⇒ 拒绝启动（没有正确 uid 就只能以 root 跑，产物事后删不掉）"
    echo "  GRASPO_RUN_UID='${GRASPO_RUN_UID:-}' GRASPO_RUN_GID='${GRASPO_RUN_GID:-}'（须为非负整数）" >&2
    exit 7
fi
RUN_USER_DESC="$RUN_UID:$RUN_GID"
RUN_USER_ARGS=(--user "$RUN_UID:$RUN_GID")

# ══ 容器内可写缓存/状态根（单一清单，只在这里出现一次）══════════════════════════
# 为什么必须显式指：镜像 ENV 把 MODELSCOPE_CACHE 写死到 /mnt/workspace/…（root:root 0755），
# 容器以宿主 uid 运行时建不出 ⇒ datasets/modelscope 在**训练开始前**就 PermissionError。
CONTAINER_CACHE_ROOTS=(
    "HOME=/tmp/graspo-runner-home"
    "XDG_CACHE_HOME=/tmp/graspo-runner-home/.cache"
    "MODELSCOPE_CACHE=/tmp/graspo-runner-home/.cache/modelscope/hub"
    "TORCHINDUCTOR_CACHE_DIR=/tmp/graspo-runner-home/.torchinductor"
    "TRITON_CACHE_DIR=/tmp/graspo-runner-home/.triton"
)

# ══ 单档执行 ════════════════════════════════════════════════════════════════════
# 返回码：0 = 该档成功；非 0 = 该档失败（批量继续跑下一档，不因一档失败而中断整批）。
run_tier() {
    local TIER="$1" TJ
    TJ="$(tier_json "$TIER")" || { fatal "$TIER: 清单读取失败"; return 5; }
    local CARDS CONFIG SUBSET SUBSET_SHA HOST_REL TRAINPATH MODEL_PATH MODEL_ENV NPROC
    local MAX_EPOCHS PLANNED PER_EPOCH DERIVED GATE ALGO MODE TUNER BACKEND LR PARALLEL TRAIN_METHOD
    # 用 python 把 JSON 摊成一行固定顺序的字段（值里不含空格：路径与数字；比 eval 安全）
    # ★ 用 `|` 作分隔符（不是空格）：`parallel` 字段本身含空格（如 `nproc_per_node=1（1 卡）`），
    #   按空格 read 会把字段整体错位 —— 这是"字段错位"族错误的经典形态，此处从结构上消除。
    IFS='|' read -r CARDS CONFIG SUBSET SUBSET_SHA HOST_REL TRAINPATH MODEL_PATH MODEL_ENV NPROC \
        MAX_EPOCHS PLANNED PER_EPOCH DERIVED GATE ALGO MODE TUNER BACKEND LR PARALLEL \
        TRAIN_METHOD \
        < <("$PYBIN" - "$TJ" <<'TIERFIELDS'
import json, sys
d = json.loads(sys.argv[1])
print("|".join(str(d[k]) for k in (
    "cards", "config", "subset_size", "subset_sha256", "host_relative_path", "train_path",
    "model_path", "model_env", "nproc", "max_epochs", "planned_steps", "per_epoch_steps",
    "derived_steps", "step_gate_threshold", "algorithm", "mode", "tuner_type", "backend", "lr",
    "parallel", "train_method")))
TIERFIELDS
        )
    if [ -z "${CARDS:-}" ] || [ -z "${CONFIG:-}" ]; then
        fatal "$TIER: 清单字段解析为空（cards/config）"; return 5
    fi

    local RUN_DIR="$RUN_ROOT/$TIER"
    local TRAIN_PATH_DIR_CONTAINER IMAGE_DIR_CONTAINER TRAIN_PATH
    TRAIN_PATH="$TRAINPATH"
    TRAIN_PATH_DIR_CONTAINER="$(dirname "$TRAIN_PATH")"
    IMAGE_DIR_CONTAINER="$(dirname "$TRAIN_PATH_DIR_CONTAINER")/images"

    # ── 0) 档位自洽（清单即判据；⛔ 这里不二次解释，只做"跑的东西 = 清单写的东西"断言）──
    if [ "$TRAIN_PATH" != "$TRAIN_PATH_DIR_CONTAINER/$TIER.jsonl" ]; then
        fatal "$TIER: 清单 train_path=$TRAIN_PATH 的文件名 ≠ $TIER.jsonl（运行链路错误）"; return 3
    fi
    [ -f "$ROOT_DIR/$CONFIG" ] || { fatal "$TIER: 档配置不存在：$ROOT_DIR/$CONFIG"; return 3; }
    [ -f "$ROOT_DIR/$HOST_REL" ] || { fatal "$TIER: 清单登记的训练子集不存在：$ROOT_DIR/$HOST_REL"; return 3; }

    if [ "$DRY_RUN" = "1" ]; then
        echo "[dry-run] $TIER cards=$CARDS nproc=$NPROC backend=$BACKEND $MODE/$TUNER algo=$ALGO lr=$LR"
        echo "[dry-run]   配置=$CONFIG（max_epochs=$MAX_EPOCHS save_steps=$((${DERIVED}+1))）"
        echo "[dry-run]   子集=$HOST_REL → 容器 $TRAIN_PATH（$SUBSET 行；sha256=${SUBSET_SHA:0:12}…）"
        echo "[dry-run]   模型=$MODEL_PATH（宿主 \$$MODEL_ENV）image=$IMAGE 超时=${TIMEOUT_SEC}s"
        echo "[dry-run]   并行=$PARALLEL；tp/pp/dp 与 nproc=$NPROC 已由清单断言（A6）"
        echo "[dry-run]   判据①：exit_code==0 ∧ steps_actual==$DERIVED（= max_epochs $MAX_EPOCHS × per_epoch $PER_EPOCH）∧ 检查点可加载 ∧ 无 NaN/Inf"
        echo "[dry-run]   步数门槛：threshold=$GATE reachable=$DERIVED planned=$PLANNED per_epoch=$PER_EPOCH"
        echo "[dry-run]   判据②：由 tools/matrix/backfill_matrix.py 按数据件现算（⛔ 本链不写 status）"
        echo "[dry-run]   判据③④ 渠道：评测后跑 $PYBIN scripts/record_checks.py --tier $TIER --run-dir $RUN_DIR --eval-dir ${EVAL_DIR:-<EVAL_DIR>}${ALGO_CHECKS_DIR:+ --algo-checks-dir $ALGO_CHECKS_DIR}  ⇒ 写 readings.json 的 checks{真/假}"
        echo "[dry-run]     · ④ 取 ${EVAL_DIR:-<EVAL_DIR>}/.local/reports/after/eval_report.json 实测：样本数==$SUBSET、请求失败 0、逐样本可解析条数==$SUBSET、温度 0"
        echo "[dry-run]     · ③ GRASPO 档须有 $TIER.json：缺省查 <run-dir>/algo-checks，缺件回退仓内 samples/configs/matrix68-checks/（<档号>.json / graspo-<档号>.json）；仅当显式设了 GRASPO_ALGO_CHECKS_DIR 才传 --algo-checks-dir，显式目录**不回退**。第 9 节 8 条不变式须逐条 + 检查人/时间签署；SFT/CPT/OPD 必须有公开算法签署件 samples/configs/matrix68-checks/public-algos.json（损失/标签掩码规则及 OPD 教师信号规则，带 file:line + 签署人/时间）"
        echo "[dry-run]     · 缺 ③④ 证据 ⇒ 不写该键 ⇒ 回填器 fail-closed（保持「待测试」）；⛔ 不得手填 true 凑通过"
        echo "[dry-run]     · 档号护栏：readings.json 的 tier 必须 == --tier（record_checks / make_result_json 各自核，不符 rc=2）"
        if [ "$ALGO" = "OPD" ]; then
            echo "[dry-run]   判据② OPD 教师读数（需前置：教师侧评测报告）：${EVAL_DIR:-<EVAL_DIR>}/.local/reports/teacher/eval_report.json"
            echo "[dry-run]     · 前提：教师档（27B / SFT / LoRA / dp=1，本矩阵 M041）先训练完成并导出 merged-hf；缺 ⇒ 本档判据② 不可判定（⛔ 不填默认值）"
            echo "[dry-run]     · 产出：$PYBIN scripts/gen_eval.py --tier $TIER --role teacher --gpu <评测卡> --port <端口> [--blocks <实测>]  ⇒ 教师侧评测脚本（与两侧同一份 $SUBSET 条 / 温度 0 / 同一判分函数）"
            echo "[dry-run]     · 取数：上述教师报告 summary.accuracy ⇒ readings.json 的 teacher_after_acc ⇒ 结果 json 的 eval.teacher_after_acc ⇒ 判据②（教师 − 学生未训练 ≥ 教师线，阈值由数据件现读）"
            echo "[dry-run]     · 缺教师报告 ⇒ 不写 teacher_after_acc ⇒ 回填 rc=1「判据② 缺证据」⇒ 该档保持「待测试」"
        fi
        # ★ 闸门：**命令 + 判据都打印，但一条都不执行**（dry-run 必须零副作用、零训练启动）
        echo "[dry-run]   闸门 1/3: $PYBIN scripts/run_guards.py cards-busy --ps-file $RUN_DIR/gpu/ps.txt --inspect-file $RUN_DIR/gpu/ins.txt   # 判据：我方容器已占/已预约的卡（含 created/restarting）必须排除"
        echo "[dry-run]   闸门 2/3: $PYBIN scripts/gpu_lock_guard.py --visible <本档实取 $CARDS 张>   # 判据：卡号 ⊆ 0–7、无重复、非空"
        echo "[dry-run]   闸门 3/3: $PYBIN scripts/gpu_idle_assert.py --visible <同上> --position-mib $POSITION_MIB   # 判据：剩额 ≥ $POSITION_MIB MiB ∧ util==0，连续 3 帧"
        echo "[dry-run]   环境快照: $PYBIN scripts/env_snapshot.py --run-root $RUN_ROOT --out $RUN_DIR/env-snapshot-before.json --phase before --config-dir $CONFIG_DIR_REL --image $IMAGE"
        echo "[dry-run]   训练: docker run --rm --runtime=nvidia --user $RUN_USER_DESC --ipc=host --shm-size=16g -e NVIDIA_VISIBLE_DEVICES=<同上> … $IMAGE bash /entry.sh（timeout ${TIMEOUT_SEC}s）"
        echo "[dry-run]   结果: $PYBIN scripts/make_result_json.py --tier $TIER --run-dir $RUN_DIR --manifest tests/e2e/matrix68_manifest.json --out $RUN_DIR/$TIER.json --collect-dir $RUN_ROOT/collect"
        echo "[dry-run]   回填入口: bash tools/matrix/backfill_dated.sh --runs-dir $RUN_ROOT/collect   # 扁平 <档号>.json；⛔ 不要指 $RUN_ROOT 本身（嵌套目录回填器不认）"
        return 0
    fi

    mkdir -p "$RUN_DIR/subsets" "$RUN_DIR/gpu" || { fatal "$TIER: 建不出跑次目录 $RUN_DIR"; return 3; }
    printf '%s\n' "$TJ" > "$RUN_DIR/tier.json"

    # ── 1) 子集落地（单一真相源 = 仓库内冻结子集；断言行数 + sha256）────────────────
    cp -f "$ROOT_DIR/$HOST_REL" "$RUN_DIR/subsets/$TIER.jsonl" || { fatal "$TIER: 子集落地失败"; return 3; }
    local LINES SHA_GOT
    LINES=$(wc -l < "$RUN_DIR/subsets/$TIER.jsonl")
    if [ "$LINES" -ne "$SUBSET" ]; then
        fatal "$TIER: 落地子集行数 $LINES ≠ 清单 subset_size $SUBSET"; return 3
    fi
    SHA_GOT=$(sha256sum "$RUN_DIR/subsets/$TIER.jsonl" | awk '{print $1}')
    if [ "$SHA_GOT" != "$SUBSET_SHA" ]; then
        fatal "$TIER: 落地子集 sha256=$SHA_GOT ≠ 清单 $SUBSET_SHA（子集在盘漂移）"; return 3
    fi

    # ── 2) 环境前置（数据/模型都是环境信息，缺即 fail-closed）─────────────────────
    if [ -z "$ELAM_HOST" ]; then
        fatal "$TIER: 数据未挂载 —— export GRASPO_ELAM_HOST_ROOT=<宿主 ELAM V5 数据根>"; return 3
    fi
    if [ -z "$MODELS_ROOT" ]; then
        fatal "$TIER: 模型未挂载 —— export GRASPO_MODELS_HOST_ROOT=<宿主模型根>"; return 4
    fi
    local MODEL_DIR_NAME="${MODEL_PATH#/models/}"
    if [ "$MODEL_PATH" != "/models/$MODEL_DIR_NAME" ] || [ -z "$MODEL_DIR_NAME" ]; then
        fatal "$TIER: 配置 model_path=$MODEL_PATH 不在 /models/ 下（运行链路错误）"; return 4
    fi
    local MODEL_HOST_DIR="${!MODEL_ENV:-$MODELS_ROOT/$MODEL_DIR_NAME}"
    [ -d "$MODEL_HOST_DIR" ] || { fatal "$TIER: 宿主模型目录不存在：$MODEL_HOST_DIR"; return 4; }
    [ -d "$ELAM_HOST/images" ] || {
        fatal "$TIER: 媒体锚点不可达 —— 宿主图像根不存在：$ELAM_HOST/images"; return 3; }

    # ── 3) 闸门 A：§17 占卡台账（我方容器已占/已预约的卡必须排除）─────────────────
    local STAMP_MS
    STAMP_MS="$(date +%s)"
    "$PYBIN" "$ROOT_DIR/scripts/env_snapshot.py" --run-root "$RUN_ROOT" \
        --out "$RUN_DIR/env-snapshot-before.json" --phase before \
        --config-dir "$CONFIG_DIR_REL" --image "$IMAGE" >/dev/null 2>&1 \
        || say "[env-snapshot] WARN: before 快照采集失败（不阻断跑批，但该轮可比性无法自证）"

    if command -v docker >/dev/null 2>&1; then
        docker ps --format '{{.Names}}' > "$RUN_DIR/gpu/ps.txt" 2>/dev/null || : > "$RUN_DIR/gpu/ps.txt"
        if [ -s "$RUN_DIR/gpu/ps.txt" ]; then
            # shellcheck disable=SC2046
            docker inspect $(cat "$RUN_DIR/gpu/ps.txt") > "$RUN_DIR/gpu/ins.txt" 2>/dev/null \
                || : > "$RUN_DIR/gpu/ins.txt"
        else
            : > "$RUN_DIR/gpu/ins.txt"
        fi
        local BUSY_OUT BUSY_CSV
        if ! BUSY_OUT="$("$PYBIN" "$ROOT_DIR/scripts/run_guards.py" cards-busy \
                --ps-file "$RUN_DIR/gpu/ps.txt" --inspect-file "$RUN_DIR/gpu/ins.txt")"; then
            fatal "$TIER: 闸门 A（cards-busy）失败 ⇒ 该档跳过（不抢占他人）"; return 1
        fi
        BUSY_CSV=$(printf '%s\n' "$BUSY_OUT" | sed -n 's/^BUSY_CARDS=//p')
        say "[gate-A cards-busy] $TIER BUSY_CARDS=${BUSY_CSV:-<空>}"
    else
        fatal "$TIER: 无 docker 命令 ⇒ 无法取占卡台账（该轮不可跑）"; return 3
    fi

    # ── 4) 择卡（从实测空闲里现取 cards 张；⛔ 不读清单 gpus 的候选窗口）─────────────
    # 判据（§13 头寸 + 「不抢占他人」）：卡 ∈ 0–7 ∧ 不在占卡台账 ∧ 剩余 ≥ 本档头寸 ∧ util==0。
    local PICKED
    if ! PICKED="$("$PYBIN" - "$BUSY_CSV" "$CARDS" "$POSITION_MIB" "$GPU_POOL_CSV" <<'PY'
import subprocess, sys
busy = {g for g in (sys.argv[1] or "").split(",") if g.strip()}
need = int(sys.argv[2]); position = int(sys.argv[3]); pool = sys.argv[4].split(",")
try:
    out = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.free,utilization.gpu",
                          "--format=csv,noheader,nounits"],
                         capture_output=True, text=True, check=True).stdout
except Exception as exc:                                   # noqa: BLE001
    raise SystemExit(f"FATAL: nvidia-smi 取不到读数 ⇒ 无法实测择卡（{exc}）")
rows = []
for line in out.strip().splitlines():
    parts = [p.strip() for p in line.split(",")]
    if len(parts) < 3:
        continue
    rows.append((parts[0], int(float(parts[1])), int(float(parts[2]))))
by_idx = {r[0]: r for r in rows}
picked, rejected = [], []
for g in pool:
    if g in busy:
        rejected.append(f"{g}(§17 我方占用)"); continue
    if g not in by_idx:
        rejected.append(f"{g}(无读数)"); continue
    _, free, util = by_idx[g]
    if util != 0:
        rejected.append(f"{g}(util={util}%)"); continue
    if free < position:
        rejected.append(f"{g}(剩余 {free} < 头寸 {position} MiB)"); continue
    picked.append(g)
    if len(picked) == need:
        break
if len(picked) != need:
    raise SystemExit(f"FATAL: 需 {need} 张空闲卡，实测仅 {len(picked)} 张；被排除：{rejected} ⇒ 顺延，⛔ 不硬上")
print(",".join(picked))
PY
)"; then
        fatal "$TIER: 择卡失败（空闲不足/读数取不到）⇒ 该档跳过，不抢占他人"; return 1
    fi
    local GPUS="$PICKED"
    if [ "$(printf '%s' "$GPUS" | tr ',' '\n' | grep -c .)" -ne "$CARDS" ]; then
        fatal "$TIER: §12 断言失败 —— len(卡集)=$GPUS 与 cards=$CARDS 不一致"; return 1
    fi
    say "[pick] $TIER cards=$CARDS → GPUS=$GPUS（剩余 ≥ ${POSITION_MIB} MiB 且 util==0）"

    # ── 5) 闸门 B/C：卡号合法性 → 逐卡实测空闲（真闸门）────────────────────────────
    if ! "$PYBIN" "$ROOT_DIR/scripts/gpu_lock_guard.py" --visible "$GPUS"; then
        fatal "$TIER: 闸门 B（gpu_lock_guard）失败 ⇒ 该档跳过"; return 1
    fi
    if ! "$PYBIN" "$ROOT_DIR/scripts/gpu_idle_assert.py" --visible "$GPUS" --position-mib "$POSITION_MIB"; then
        fatal "$TIER: 闸门 C（gpu_idle_assert）失败 ⇒ 该档跳过（不抢占他人）"; return 1
    fi

    # ── 6) entry.sh（容器内入口）：缓存根可写 → 媒体预检 → 显存采集 → torchrun ──────
    local PORT=$(( ( (RANDOM % 2000) + 29500 ) ))
    {
        cat <<ENTRY_HEAD
set -o pipefail
GPU_EVIDENCE_DIR="/out/gpu"
mkdir -p "\$GPU_EVIDENCE_DIR"
CONTAINER_CACHE_ROOTS=(
ENTRY_HEAD
        local kv
        for kv in "${CONTAINER_CACHE_ROOTS[@]}"; do printf '    "%s"\n' "$kv"; done
        cat <<'ENTRY_MID'
)
for CACHE_KV in "${CONTAINER_CACHE_ROOTS[@]}"; do
    CACHE_DIR="${CACHE_KV#*=}"
    if ! mkdir -p "$CACHE_DIR" 2>/dev/null || [ ! -w "$CACHE_DIR" ]; then
        echo "FATAL(cache-root): 容器内缓存根不可写: $CACHE_KV" >&2
        echo "  为什么 fail-closed：缓存根不可写 ⇒ 数据集/编译缓存 PermissionError ⇒ 训练一步没跑就被记成失败。" >&2
        exit 8
    fi
done
ENTRY_MID
        # 媒体锚点预检（宿主侧渲染本档 train_path；容器内断言媒体可达）
        cat <<ENTRY_PREFLIGHT
"$CONTAINER_PY" - "$TRAIN_PATH" <<'PY'
import json, sys
from pathlib import Path
train = Path(sys.argv[1])
base = train.parent
missing, checked = 0, 0
for line in train.read_text(encoding="utf-8").splitlines():
    if not line.strip():
        continue
    rec = json.loads(line)
    cand = []
    for k in ("image", "path", "url", "video"):
        v = rec.get(k)
        if isinstance(v, str):
            cand.append(v)
    for k in ("images", "image_url", "video_url"):
        v = rec.get(k)
        if isinstance(v, list):
            cand.extend([x for x in v if isinstance(x, str)])
        elif isinstance(v, dict) and isinstance(v.get("url"), str):
            cand.append(v["url"])
    for v in cand:
        if v.startswith(("http://", "https://", "/", "data:")):
            continue
        checked += 1
        if not (base / v).exists():
            missing += 1
            if missing <= 5:
                print(f"[media-preflight] MISSING {(base / v)}", file=sys.stderr)
print(f"[media-preflight] checked={checked} missing={missing} base={base}")
raise SystemExit(1 if missing else 0)
PY
[ \$? -eq 0 ] || exit 3
ENTRY_PREFLIGHT
        cat <<ENTRY_TAIL
# ══ 收口①：训练进程一返回就**有界**回收容器内的显存采样器（2026-10-26 修 D1）══
#   缺陷（M016 实证）：采样器 graspo record-gpu-memory 是**无界循环**（训练多长采多长），
#   旧代码这里是 wait "\$SAMPLER" —— 无上限地等它 ⇒ entry.sh 永不返回 ⇒ 容器不退出 ⇒
#   外层 timeout 57600s 把 docker CLI 杀掉。runs/M016 因此记 exit_code=137、
#   elapsed=57720（= 57600 + kill-after 120），而 stdout.log 末行是
#   End time of running main: 2026-10-26 00:10:10、步数 6400/6400、8/8 epoch
#   —— 训练**已经**按主流程正常返回，成功被记成超时失败（外层超时本该只保护训练）。
#   ⚠ 比台账更糟（2026-10-26 实测）：被杀的只是 docker CLI，**容器没退** ——
#   docker ps 里 graspo-M016-… 至今 Up 18h、采样器进程仍在跑；泄漏的容器一直占着
#   那张卡的「容器声明」，后续档位的闸门 A 会把它判成 BUSY（M052 因此只能换卡）。
#   所以本段的收口不是"省几秒"，是让容器真的能退出。
#   口径（只动代价、不动判定）：
#     · SIGTERM 是 record_gpu_memory 的**优雅停止**信号（它在 finally 里落
#       gpu_memory_summary.json），所以先 TERM 再等，摘要不丢；
#     · 上限 = GRASPO_GPU_SAMPLER_GRACE_SEC（默认 30s，⛔ 不进任何判据），到点 SIGKILL；
#     · 最后 wait 回收，容器退出时不留孤儿；
#     · ★ 退出码**只取 torchrun 的 RC**：采样器死不死、摘要落没落盘都不许改变训练结论
#       （摘要缺失只登记 EVIDENCE_GAP，见下一条 if）。
#   ★ 只对自己起的那个 pid（SAMPLER=\$!）动手：⛔ 不 pkill、⛔ 不碰他人进程/容器。
#   ★ 实现用「看门狗 + wait」而不是轮询 kill -0：子进程退出后会短暂是僵尸态，
#     kill -0 对僵尸**仍然成功**，轮询法会把每一档都拖满整个 grace 才返回。
wait_for_gpu_summary() {
    local pid="\${1:-}" grace="\${GRASPO_GPU_SAMPLER_GRACE_SEC:-30}" watchdog
    if [ -z "\$pid" ]; then
        echo "[gpu-summary] WARN: 采样器 pid 为空 —— 无法等待，取证可能不完整" >&2
        return 0
    fi
    case "\$grace" in ''|*[!0-9]*) grace=30 ;; esac
    [ "\$grace" -gt 0 ] || grace=30
    kill -TERM "\$pid" 2>/dev/null || true
    (
        sleep "\$grace"
        echo "[gpu-summary] WARN: 采样器 \${grace}s 内未退出（pid=\$pid）⇒ SIGKILL（只改代价，不改判定）" >&2
        kill -KILL "\$pid" 2>/dev/null || true
    ) &
    watchdog=\$!
    wait "\$pid" 2>/dev/null || true
    kill -TERM "\$watchdog" 2>/dev/null || true
    wait "\$watchdog" 2>/dev/null || true
    return 0
}
"$CONTAINER_PY" -m graspo record-gpu-memory --config "/workspace/graspo/$CONFIG" &
SAMPLER=\$!
torchrun --standalone --nproc_per_node="$NPROC" --master_port="$PORT" \\
  -m graspo.cli.train_worker --config "/workspace/graspo/$CONFIG" > /out/stdout.log 2>&1
RC=\$?
wait_for_gpu_summary "\$SAMPLER"
if [ -f "\$GPU_EVIDENCE_DIR/gpu_memory_summary.json" ]; then
    printf '%s\n' present > "\$GPU_EVIDENCE_DIR/gpu_summary.state"
else
    printf '%s\n' "MISSING: gpu_memory_summary.json（训练侧采样缺失；峰值显存列另由宿主侧采样补）" \\
        > "\$GPU_EVIDENCE_DIR/EVIDENCE_GAP_gpu_memory_summary"
    printf '%s\n' missing > "\$GPU_EVIDENCE_DIR/gpu_summary.state"
fi
echo "\$RC" > /out/exit_code
exit "\$RC"
ENTRY_TAIL
    } > "$RUN_DIR/entry.sh"

    # ── 7) 挂载表（目的地全部互不嵌套）→ 不变量断言 ────────────────────────────────
    local -a MOUNT_SPECS=(
        "$ROOT_DIR|/workspace/graspo|ro"
        "$MODELS_ROOT|/models|ro"
        "$ELAM_HOST/images|$IMAGE_DIR_CONTAINER|ro"
        "$RUN_DIR/subsets|$TRAIN_PATH_DIR_CONTAINER|ro"
        "$RUN_DIR|/out|rw"
        "$RUN_DIR/entry.sh|/entry.sh|ro"
    )
    assert_mount_targets_not_under_readonly "${MOUNT_SPECS[@]}" || return 6

    local -a DOCKER_ARGS=(
        --rm --runtime=nvidia
        "${RUN_USER_ARGS[@]}"
        --ipc=host --shm-size=16g
        -e "NVIDIA_VISIBLE_DEVICES=$GPUS"
        -e "TIER=$TIER"
        -e PYTHONPATH=/workspace/graspo/src
        -e HF_HUB_OFFLINE=1
        -e TOKENIZERS_PARALLELISM=false
        -e USER=graspo
        -e LOGNAME=graspo
        --name "graspo-$TIER-$(date +%s)"
    )
    local CACHE_KV SPEC SRC DST MODE_S
    for CACHE_KV in "${CONTAINER_CACHE_ROOTS[@]}"; do DOCKER_ARGS+=(-e "$CACHE_KV"); done
    for SPEC in "${MOUNT_SPECS[@]}"; do
        IFS='|' read -r SRC DST MODE_S <<< "$SPEC"
        if [ "$MODE_S" = "ro" ]; then DOCKER_ARGS+=(-v "$SRC:$DST:ro"); else DOCKER_ARGS+=(-v "$SRC:$DST"); fi
    done

    # ── 8) 宿主侧显存采样（旁路；拿不到就如实记 null）──────────────────────────────
    local SAMPLE_LOG="$RUN_DIR/gpu/host_samples.txt"
    : > "$SAMPLE_LOG"
    local SAMPLER_PID=""
    if command -v nvidia-smi >/dev/null 2>&1; then
        (
            while :; do
                printf '%s ' "$(date +%s)"
                nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits 2>/dev/null \
                    | tr '\n' ';'
                printf '\n'
                sleep 2
            done
        ) >> "$SAMPLE_LOG" 2>/dev/null &
        SAMPLER_PID=$!
    fi

    # ── 9) 起跑（单容器；超时即 kill 自己起的那个容器，⛔ 不用 pkill）────────────────
    local T_START T_END RC
    T_START=$(date +%s)
    say "[start] $TIER GPUS=$GPUS image=$IMAGE timeout=${TIMEOUT_SEC}s"
    timeout --signal=TERM --kill-after=120 "$TIMEOUT_SEC" \
        docker run "${DOCKER_ARGS[@]}" -w /workspace/graspo "$IMAGE" bash /entry.sh
    RC=$?
    T_END=$(date +%s)
    # 训练 docker 进程一返回就**有界**回收宿主采样器（D1 的宿主半边；⛔ 不改 RC，
    # 也⛔ 不把这段耗时算进 elapsed —— elapsed 的起止在上一行已经定死）。
    reap_sampler_pid "$SAMPLER_PID"
    echo "$RC" > "$RUN_DIR/exit_code"
    say "[end] $TIER exit_code=$RC elapsed=$((T_END - T_START))s"

    # ── 10) 峰值显存（宿主侧采样 + 训练侧摘要，逐档如实落盘）──────────────────────
    "$PYBIN" - "$SAMPLE_LOG" "$GPUS" "$RUN_DIR/gpu/peak_mem.json" <<'PY'
import json, sys
path, gpus, out = sys.argv[1], sys.argv[2].split(","), sys.argv[3]
peak = {}
for line in open(path, encoding="utf-8", errors="replace"):
    parts = line.strip().split(" ", 1)
    if len(parts) < 2:
        continue
    for item in parts[1].split(";"):
        item = item.strip()
        if not item or "," not in item:
            continue
        idx, used = [x.strip() for x in item.split(",")[:2]]
        try:
            used = float(used)
        except ValueError:
            continue
        peak[idx] = max(peak.get(idx, 0.0), used)
sel = {g: peak.get(g) for g in gpus}
doc = {
    "caliber": "宿主侧 nvidia-smi memory.used 采样峰值（MiB，每 2 s 一帧）；非训练侧 allocator 读数",
    "gpus": gpus,
    "peak_used_mib_by_gpu": sel,
    "peak_used_mib": (max(v for v in sel.values() if v is not None)
                      if any(v is not None for v in sel.values()) else None),
    "measured": any(v is not None for v in sel.values()),
}
open(out, "w", encoding="utf-8").write(json.dumps(doc, ensure_ascii=False, indent=1) + "\n")
print(f"[peak-mem] {doc['peak_used_mib']} MiB（measured={doc['measured']}）")
PY

    # ── 11) 跑后环境快照 + 冻结维度比对（rc=3 ⇒ 该轮**不可比**，如实标记）────────────
    local COMPARABLE="unknown" DIFF_RC=0
    "$PYBIN" "$ROOT_DIR/scripts/env_snapshot.py" --run-root "$RUN_ROOT" \
        --out "$RUN_DIR/env-snapshot-after.json" --phase after \
        --config-dir "$CONFIG_DIR_REL" --image "$IMAGE" >/dev/null 2>&1 || true
    if [ -f "$RUN_DIR/env-snapshot-before.json" ] && [ -f "$RUN_DIR/env-snapshot-after.json" ]; then
        "$PYBIN" "$ROOT_DIR/scripts/env_snapshot.py" --diff "$RUN_DIR/env-snapshot-before.json" \
            "$RUN_DIR/env-snapshot-after.json" > "$RUN_DIR/env-diff.txt" 2>&1
        DIFF_RC=$?
        case "$DIFF_RC" in
            0) COMPARABLE="true" ;;
            3) COMPARABLE="false"; say "[env-diff] $TIER 冻结维度发生变化 ⇒ 本轮**不可比**（⛔ 不悄悄重跑）" ;;
            *) COMPARABLE="unknown" ;;
        esac
    fi

    # ── 12) 原始读数落盘（readings.json = **只记实测**；缺项写 null，⛔ 不估算）────────
    #   ★ 为什么由 runner 落：`elapsed`/`throughput` 全仓无产出脚本（链路侦察 B3），
    #     墙钟只有 runner 自己在起止两侧测得准；adapter 只负责把它换算进回填口径。
    local ELAPSED_SEC=$((T_END - T_START))
    printf '%s\n' "$ELAPSED_SEC" > "$RUN_DIR/elapsed_sec"
    "$PYBIN" - "$RUN_DIR" "$TIER" "$RC" "$ELAPSED_SEC" "$COMPARABLE" "$DIFF_RC" "$GPUS" \
        "$CONFIG" "$IMAGE" "$TIMEOUT_SEC" <<'READINGS'
import json, sys
run_dir, tier, rc, elapsed, comparable, diff_rc, gpus, config, image, timeout = sys.argv[1:11]
doc = {
    "tier": tier,
    "exit_code": int(rc),
    "elapsed_sec": float(elapsed),
    "comparable": (None if comparable == "unknown" else comparable == "true"),
    "env_diff_rc": int(diff_rc),
    "gpus": gpus.split(","),
    "config": config,
    "image": image,
    "timeout_sec": int(timeout),
    "command": f"bash tests/e2e/run_matrix68.sh --only {tier}",
    "note": ("elapsed_sec = runner 起止墙钟差；peak_mem 见 gpu/peak_mem.json；"
             "steps_actual 由 adapter 从 stdout.log 的 global_step 取最大值"),
}
open(run_dir + "/readings.json", "w", encoding="utf-8").write(
    json.dumps(doc, ensure_ascii=False, indent=1) + "\n")
READINGS

    # ── 13) 原始读数 → 回填口径结果 json（adapter 内跑回填器校验器自检）──────────────
    local RESULT_RC=0
    #   ★ 同时落一份**扁平副本**到 <run-root>/collect/（= 回填入口）：回填器 `--runs-dir` 只吃
    #     扁平目录，嵌套的 <run-root>/<档号>/<档号>.json 它不认（审查 🟡-3）。结构校验不过（rc=1）
    #     时 adapter 不写扁平副本 —— ⛔ 不让不合契约的件进回填目录。
    local RESULT_ARGS=(--tier "$TIER" --run-dir "$RUN_DIR" --manifest "$MANIFEST"
                       --out "$RUN_DIR/$TIER.json" --collect-dir "$RUN_ROOT/collect")
    [ -n "$EVAL_DIR" ] && RESULT_ARGS+=(--eval-dir "$EVAL_DIR")
    "$PYBIN" "$ROOT_DIR/scripts/make_result_json.py" "${RESULT_ARGS[@]}" || RESULT_RC=$?
    if [ "$RESULT_RC" -ne 0 ]; then
        say "[result-json] $TIER 结果 json 未达回填口径（rc=$RESULT_RC）—— 见 $RUN_DIR/$TIER.json 的 note"
    fi

    # ── 13.5) 判据③④：评测产物已就位 ⇒ 当场产出证据并复跑适配器（唯一渠道）──────────
    #   ★ 为什么必须在这里补：runner **不自动跑评测**（评测链 = `scripts/gen_eval.py`），而判据③④
    #     的证据只能用评测报告/算法检查签署件算出来 ⇒ 已有报告就当场算；没有就只打印后续命令，
    #     ⛔ 绝不伪造证据、⛔ 绝不手填 true（审查第 2 轮 🟡-A 的落点）。
    if [ -n "$EVAL_DIR" ] && [ -f "$EVAL_DIR/.local/reports/after/eval_report.json" ]; then
        say "[checks] $TIER 评测报告已就位 ⇒ 产出判据③④ 证据（写 readings.json 的 checks）"
        # ★ OPD 档另有判据② 的教师读数前提（唯一来源 = 教师侧评测报告；缺 ⇒ 不写键 ⇒ fail-closed）
        if [ "$ALGO" = "OPD" ] && [ ! -f "$EVAL_DIR/.local/reports/teacher/eval_report.json" ]; then
            say "[checks] $TIER 是 OPD 档但缺教师侧评测报告（$EVAL_DIR/.local/reports/teacher/eval_report.json）⇒ 判据② 的 teacher_after_acc 无实测证据（⛔ 不填默认值）"
            say "[checks] $TIER 补法：$PYBIN scripts/gen_eval.py --tier $TIER --role teacher --gpu <评测卡> --port <端口>；顺序约束：教师档（27B / SFT / LoRA / dp=1，本矩阵 M041）必须先于 OPD 档完成并导出 merged-hf"
        fi
        local CHECKS_RC=0
        # ★ 缺省**不传** `--algo-checks-dir`（缺陷根因：旧代码恒传 `<run-root>/algo-checks`，而该目录
        #   从来不存在；record_checks 对显式目录"只认不存在的目录、不回退仓内签署件" ⇒ 判据③ 每档
        #   都缺证据 ⇒ 每档事后手工重跑才能救回）。不传 ⇒ record_checks 走既有回退（仓内
        #   samples/configs/matrix68-checks/）。显式设了 GRASPO_ALGO_CHECKS_DIR 才传：显式即不回退
        #   （⛔ 防止偷偷用仓内旧签署件顶替跑次侧**已存在**的签署件）。
        local CHECKS_ARGS=(--tier "$TIER" --run-dir "$RUN_DIR" --eval-dir "$EVAL_DIR"
                           --manifest "$MANIFEST")
        [ -n "$ALGO_CHECKS_DIR" ] && CHECKS_ARGS+=(--algo-checks-dir "$ALGO_CHECKS_DIR")
        "$PYBIN" "$ROOT_DIR/scripts/record_checks.py" "${CHECKS_ARGS[@]}" || CHECKS_RC=$?
        if [ "$CHECKS_RC" -ne 0 ]; then
            say "[checks] $TIER 判据②③④ 证据不全（rc=$CHECKS_RC）⇒ 回填时该档 fail-closed；补齐证据后重跑 record_checks"
        fi
        RESULT_RC=0
        "$PYBIN" "$ROOT_DIR/scripts/make_result_json.py" "${RESULT_ARGS[@]}" || RESULT_RC=$?
        say "[checks] $TIER 结果 json 已按判据②（教师）③④ 证据复算（rc=$RESULT_RC）"
    else
        say "[checks] $TIER 判据③④ 待评测后执行：$PYBIN scripts/record_checks.py --tier $TIER --run-dir $RUN_DIR --eval-dir <EVAL_DIR>${ALGO_CHECKS_DIR:+ --algo-checks-dir $ALGO_CHECKS_DIR}，再复跑 make_result_json（缺证据 ⇒ 回填器 fail-closed）"
        [ "$ALGO" = "OPD" ] && say "[checks] $TIER 另需教师侧评测报告（判据②）：$PYBIN scripts/gen_eval.py --tier $TIER --role teacher …（教师档先于 OPD 档完成；缺 ⇒ 判据② 不可判定）"
    fi
    # 该档"成功"= 训练退出码 0（判据① 的部分事实）；四条判据的合取由回填器现算，⛔ 本脚本不下结论
    if [ "$RC" -ne 0 ]; then
        say "[tier-failed] $TIER exit_code=$RC（该档记失败，批量继续）"
        return 1
    fi
    return 0
}

# ══ 批量主流程 ══════════════════════════════════════════════════════════════════
mkdir -p "$RUN_ROOT" 2>/dev/null || true
say "矩阵 68 档 runner：清单=$MANIFEST 目标=$N_TARGETS 档 run_root=$RUN_ROOT dry_run=$DRY_RUN"
if [ "$DRY_RUN" = "1" ]; then
    say "dry-run 模式：只打印每档将执行的命令与判据；⛔ 不调用 docker / torchrun / nvidia-smi"
fi

FAILED=""
OK=""
for TIER in $TARGETS; do
    if run_tier "$TIER"; then
        OK="$OK $TIER"
    else
        FAILED="$FAILED $TIER"
    fi
done

say "汇总：成功=$(printf '%s\n' $OK | grep -c . || true) 档；失败=$(printf '%s\n' $FAILED | grep -c . || true) 档"
[ -n "$FAILED" ] && say "失败档号：$FAILED"
[ "$DRY_RUN" = "1" ] && { say "dry-run 结束（零训练启动）"; exit 0; }
[ -n "$FAILED" ] && exit 1
exit 0
