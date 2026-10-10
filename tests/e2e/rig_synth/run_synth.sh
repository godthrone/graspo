#!/usr/bin/env bash
# tests/e2e/rig_synth/run_synth.sh —— 合成长序列「逐长度串行」上机驱动（**只造工具，本包不上机**）
#
# 设计取舍（红线：不改 generate_matrix.py / run_matrix54.sh）
#   本脚本**不复制 runner 逻辑、也不新造 docker**，而是复用 task-r3-lenramp 已闭环的 rig
#   （`rig/t013_host.sh` + `run_matrix54.sh`），只补四件"本批特有"的事：
#     ① **合成宿主数据根 staging**：把某一（变体 × 档）的 1 行合成行放进
#        `<SYNTH_HOST>/data/train.jsonl`，并让 `<SYNTH_HOST>/subsets` 存在
#        （run_matrix54.sh 的 `head -n $SUBSET "${GRASPO_ELAM_MINI_JSONL:-…}"` 会覆盖它
#         ⇒ 正好产出那 1 行；`data.train_path` 因此**不需要改**）。
#     ①′ **把 runner 的读取源显式指回 staged 合成行**（`GRASPO_ELAM_MINI_JSONL`，见阶段 B
#        `run_once`）：runner 自 2026-09-20 起默认读 `<ELAM_HOST>/mini-dataset/`，
#        而 `$SYNTH_ROOT/$tag` 下没有 mini 数据集 —— 不设这一条，本批一档都跑不起来。
#     ② **逐长度串行 + 到真 OOM 即停**（口径：只有真 OOM 才计入「最大可行上下文」列）。
#     ③ **每档实测 token 数复核**（本包最关键的判据：目标长度真的达到了）。
#
# ★★ 图像根：**不用符号链接**（r3 已踩过：指向宿主绝对路径的符号链接在容器内必然断开）
#    ⇒ 本脚本**只读复用真实图像根**：把 `SYNTH_HOST/images` 做成对 `$ELAM_REAL_ROOT/images`
#    的符号链接是**错误做法**；正确做法是**让 runner 直接挂真实图像根**
#    —— 即 `GRASPO_ELAM_HOST_ROOT` 仍指向真实根、只把 `data/train.jsonl` 换掉。
#    为在**零改动** runner 的前提下做到这一点，本脚本用 **bind-mount 覆盖**：
#    在 `$ELAM_REAL_ROOT/data` 上挂一个只含合成 train.jsonl 的目录（见 `--real-root-knob`）。
#    ⚠ 该手法需要 root；**若不可行，请改用 `SYNTH_HOST` + `images` 的**硬链接副本**
#    （见 `stage_host_root` 的两种模式），并在报告里登记实际用了哪种。
#
# 用法（**串行**，按长度递增；到真 OOM 即停）：
#   # 先把三个宿主根用环境变量注入（见下方 §宿主路径口径；本文件**不含**任何宿主路径）
#   export GRASPO_REPO_HOST_ROOT=<宿主仓库目录>
#   export GRASPO_SYNTH_RIG_ROOT=<合成 rig 落点，临时>
#   export GRASPO_LENRAMP_RIG_ROOT=<已闭环的 r3 rig 目录，t013_host.sh 在此>
#   bash run_synth.sh --variant v1-image-copy --real-root <真实 ELAM V5 宿主根>
#   bash run_synth.sh --variant v2-text-only  --real-root <真实 ELAM V5 宿主根>
#        [--only 4k] [--max-tier 32k] [--dry-run] [--verify-only]
set -uo pipefail

# ---- §宿主路径口径（宪法 §15.1 / §16，与 run_matrix54.sh 逐字同源）----------------
# 宿主目录、内网路径属**环境信息**，**一律不入 tracked 文件**（§15.1 机密扫描会阻断
# `internal_absolute_path`）。与 `run_matrix54.sh` 的 `ELAM_HOST="${GRASPO_ELAM_HOST_ROOT:-}"`
# 同一口径：**由运行时环境变量注入，无 tracked 默认值；缺失即 fail-closed**（不是静默用错的路径）。
# 本地化请用 `.local/run.local.sh` 导出这些变量（§4.2），不要改本文件。
REPO_REMOTE="${GRASPO_REPO_HOST_ROOT:-}"       # 宿主上的仓库目录（含 tests/e2e/run_matrix54.sh）
SYNTH_RIG_REMOTE="${GRASPO_SYNTH_RIG_ROOT:-}"  # 合成 rig 落点（临时，跑完清理）
LENRAMP_RIG="${GRASPO_LENRAMP_RIG_ROOT:-}"     # 已闭环的 r3 rig（t013_host.sh 在此）
LOCAL_RIG_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

_missing_root() { # _missing_root <变量名>
  echo "FATAL(runtime-link): 宿主路径未注入 —— 请先 export $1=<该宿主根目录>" >&2
  echo "  口径（宪法 §15.1/§16）：宿主路径属环境信息，**不入 tracked 文件** ⇒" >&2
  echo "  运行时由环境变量注入，缺失即 fail-closed（宁可不跑，也不用错的路径跑）。" >&2
  echo "  本脚本需要：GRASPO_REPO_HOST_ROOT / GRASPO_SYNTH_RIG_ROOT / GRASPO_LENRAMP_RIG_ROOT" >&2
}

LENS=(4k 8k 16k 32k 64k 96k 128k)

VARIANT=""
REAL_ROOT=""
ONLY=""
MAX_TIER=""
MODE=run
VERIFY_ONLY=0

while [ $# -gt 0 ]; do
  case "$1" in
    --variant) VARIANT="${2:?}"; shift 2 ;;
    --real-root) REAL_ROOT="${2:?}"; shift 2 ;;
    --only) ONLY="${2:?}"; shift 2 ;;
    --max-tier) MAX_TIER="${2:?}"; shift 2 ;;
    --dry-run) MODE=--dry-run; shift ;;
    --verify-only) VERIFY_ONLY=1; shift ;;
    *) echo "FATAL: 未知参数 $1" >&2; exit 2 ;;
  esac
done

[ -n "$VARIANT" ] || { echo "FATAL: 必须给 --variant v1-image-copy|v2-text-only" >&2; exit 2; }
[ -n "$REAL_ROOT" ] || { echo "FATAL: 必须给 --real-root <真实 ELAM V5 宿主根>" >&2; exit 2; }
# ★ fail-closed（§2.3 边界校验即防呆）：三个宿主根缺失即拒绝启动，绝不用错路径静默跑。
[ -n "$REPO_REMOTE" ] || { _missing_root GRASPO_REPO_HOST_ROOT; exit 3; }
[ -n "$SYNTH_RIG_REMOTE" ] || { _missing_root GRASPO_SYNTH_RIG_ROOT; exit 3; }
[ -n "$LENRAMP_RIG" ] || { _missing_root GRASPO_LENRAMP_RIG_ROOT; exit 3; }

if [ "$VARIANT" = "v1-image-copy" ]; then
  FILLS_COLUMN=true
else
  FILLS_COLUMN=false
  echo "⚠ 变体 $VARIANT 的读数**不得**填入「最大可行上下文」列（仅作单独标注的数据点）。"
fi

SYNTH_ROOT="$SYNTH_RIG_REMOTE/$VARIANT"     # 合成宿主数据根的父目录
LOG="$SYNTH_RIG_REMOTE/driver"
mkdir -p "$LOG"

echo "=== [run_synth] variant=$VARIANT fills_column=$FILLS_COLUMN start=$(date -Is) ===" | tee "$LOG/driver.log"

# --------------------------------------------------------------------------- #
# 阶段 A：staging —— 为每个长度准备「合成宿主数据根」
# --------------------------------------------------------------------------- #
# 目标形态（让 run_matrix54.sh **零改动**通过）：
#   <SYNTH_HOST>/data/train.jsonl      = 该档的 1 行合成行
#   <SYNTH_HOST>/subsets/              = 空目录（runner 会写入 <tier>.jsonl）
#   <SYNTH_HOST>/images/               = **真实图像根**（见下方两种模式）
#
# 为什么要 images：runner 的 fail-closed 检查要求 `$ELAM_HOST/images` 存在，
#   且容器内媒体预检会验证样本里的相对路径能解析到真图。
stage_host_root() {
  local tag="$1"
  local host_root="$SYNTH_ROOT/$tag"
  mkdir -p "$host_root/data" "$host_root/subsets"

  # 1 行合成行（逐档副本；by-tier 目录里的是权威产物）
  local src_line
  src_line="$(find "$SYNTH_RIG_REMOTE/build" -name "T013-synth-${tag}.jsonl" -print -quit)"
  if [ -z "$src_line" ] || [ ! -f "$src_line" ]; then
    echo "FATAL: 找不到该档的合成行（T013-synth-${tag}.jsonl）；先跑 build_synth_seq.py 并把 by-tier 传上来" >&2
    return 3
  fi
  cp -f "$src_line" "$host_root/data/train.jsonl"
  local lines
  lines=$(wc -l < "$host_root/data/train.jsonl")
  if [ "$lines" -ne 1 ]; then
    echo "FATAL: 合成 train.jsonl 应为 1 行，实际 $lines 行" >&2
    return 3
  fi

  # images：**绝不建符号链接**（r3 实测：容器内必然断开）。两种稳妥模式二选一：
  #   模式 ①（推荐，省磁盘）：**bind-mount 真实图像根**到 <SYNTH_HOST>/images
  #        —— 需要 root；本函数在非 root 时**拒绝**并给出可粘贴命令。
  #   模式 ②（保底）：硬链接副本（同一文件系统内 inode 级复制，零额外磁盘占用）
  if [ -L "$host_root/images" ]; then
    echo "FATAL: $host_root/images 是符号链接 —— 明确禁止（r3 实测容器内断开的真因）。" >&2
    return 3
  fi
  if [ ! -d "$host_root/images" ]; then
    if [ "$(id -u)" -eq 0 ]; then
      mkdir -p "$host_root/images"
      mount --bind "$REAL_ROOT/images" "$host_root/images" \
        && echo "[stage] images: bind-mount $REAL_ROOT/images -> $host_root/images"
    else
      echo "[stage] non-root ⇒ 用**硬链接副本**（模式 ②，零额外磁盘）"
      cp -al "$REAL_ROOT/images" "$host_root/images" 2>/dev/null \
        || { echo "FATAL: 硬链接副本失败（跨文件系统？）。请改在 $REAL_ROOT 所在文件系统上落 $SYNTH_ROOT，" >&2
             echo "       或以 root 用 bind-mount：mount --bind $REAL_ROOT/images $host_root/images" >&2
             return 3; }
    fi
  fi

  # 真图可达性**宿主侧**自证（与容器内预检同判据；提前失败好过容器里失败）
  local probe
  probe="$(find "$host_root/images" -maxdepth 1 -type f \( -name '*.jpg' -o -name '*.png' \) -print -quit)"
  if [ -z "$probe" ]; then
    echo "FATAL: $host_root/images 里找不到任何图片 ⇒ 图像根不可达（运行链路错误）" >&2
    return 3
  fi
  echo "[stage] $tag OK: train.jsonl=1 行, images 可达（样例: $(basename "$probe")）"
  return 0
}

# --------------------------------------------------------------------------- #
# 真 OOM 判定（单一实现点，§1.4 —— 判定口径只此一处，driver 与自检共用）
# --------------------------------------------------------------------------- #
#: 真 OOM 的报文模式（与 `collect_results.py` 的失败分类口径同源）。
OOM_PATTERN='out of memory|OutOfMemoryError|CUDA out of memory|torch.OutOfMemoryError'

#: 容器 stdout（**训练器报文的唯一落点**）的候选根。
#:
#: 为什么需要候选列表：`<RIGROOT>/runs/<case>/<TIER>/` 的 `RIGROOT` 由**上游 rig**
#: （`t013_host.sh` 的 `$RIGROOT`，本批经 shim 由 `SYNTH_RIGROOT` 注入）决定，
#: 而本脚本不读该文件、也不该猜一个绝对路径（§15.1：宿主路径不入 tracked 文件）。
#: ⇒ 依次尝试：① 显式注入（可选，不设即无副作用）；② 从已注入的 rig 根推导。
#: 全部缺失时函数**不静默放行**：打印 WARN 到 stderr（判定依据不可达本身要暴露）。
_oom_run_dirs_for_case() { # _oom_run_dirs_for_case <case>
  local case="$1" root
  for root in "${GRASPO_SYNTH_RIGROOT:-}" "${GRASPO_LENRAMP_RIGROOT:-}" \
              "${GRASPO_RIGROOT:-}" "${SYNTH_RIG_REMOTE:-}" "${LENRAMP_RIG:-}"; do
    [ -n "$root" ] || continue
    printf '%s\n' "$root/runs/$case"
  done
}

real_oom_case() { # real_oom_case <case> <manifest>
  # 判据：任一候选文件命中 OOM 报文即"真 OOM"。**只做加法**——旧实现 grep 的两个文件
  # 仍在候选里（`.out` 兜底），因此新实现**不会漏**旧实现能抓到的情形（不放松判据；
  # 只有"抓住更多"这一种变化方向）。文件不存在 ⇒ 该候选跳过，不算命中。
  local case="$1" manifest="$2"
  local tier_id
  tier_id="$(python3 -c "
import json, sys
doc = json.load(open(sys.argv[1], encoding='utf-8'))
tiers = doc.get('tiers') or []
if len(tiers) != 1:
    raise SystemExit('rig 清单必须恰好 1 档，实际 %d' % len(tiers))
print(tiers[0]['tier_id'])
" "$manifest" 2>/dev/null)" || { tier_id=""; }

  local candidates=() run_dir f hit=0 scanned=0
  # ① 容器 stdout（**主判定对象**：训练器报文的唯一落点）
  #    `<RIGROOT>/runs/<case>/<TIER>/`（`t013_host.sh:47` + `run_matrix54.sh:446`）
  for run_dir in $(_oom_run_dirs_for_case "$case"); do
    [ -n "$tier_id" ] && candidates+=("$run_dir/$tier_id/stdout.log")
  done
  # ② 旧实现 grep 过的两个文件 —— **保留为兜底**（只加不减 ⇒ 不会漏旧实现能抓的情形）
  candidates+=("$LOG/${case}.out")
  [ -n "${LENRAMP_RIG:-}" ] && candidates+=("$LENRAMP_RIG/cases/$case/runner.log")

  for f in "${candidates[@]}"; do
    [ -f "$f" ] || continue
    scanned=$((scanned + 1))
    if grep -qiE "$OOM_PATTERN" "$f"; then
      hit=1
      echo "[oom-check] case=$case 命中真 OOM 报文：$f" >&2
      break
    fi
  done
  if [ "$hit" -eq 0 ] && [ "$scanned" -eq 0 ]; then
    echo "WARN(oom-check): case=$case 找不到任何候选日志 —— 判定依据不可达；" \
         "请注入 GRASPO_SYNTH_RIGROOT 以定位容器 stdout.log（**不静默当作已判定**）" >&2
  fi
  [ "$hit" -eq 1 ]
}

# 测试用的库模式（与 `run_matrix54.sh` 的 `GRASPO_RUNNER_LIB_ONLY` 同范式，§1.4）：
# 只把本文件的**判定函数**暴露给回归测试，不执行任何 stage / 上机流程。
# ★ 必须放在**函数定义之后**（否则 source 进来的被测函数还不存在 —— 实测踩过）。
# 为什么需要它：真 OOM 判定的负向用例必须**直接调用生产实现**（而不是在测试里再抄一份
# 判定逻辑 —— 抄一份就等于有两套口径，测试通过也证明不了生产代码对，§2.2 显式即防呆）。
if [ "${GRASPO_SYNTH_LIB_ONLY:-0}" = "1" ]; then
  return 0 2>/dev/null || exit 0
fi

# --------------------------------------------------------------------------- #
# 阶段 B：逐长度串行（每档：跑 → 复核 token → 判真 OOM）
# --------------------------------------------------------------------------- #
#: ★ A4 适用策略（指挥官 2026-09-19 裁定 1）：4K–32K **双跑**；64K 起**单跑**。
#: 单跑档的 A4 **必须**记 `not_applicable_single_run`（台账显示「未适用（单跑）」），
#: **绝不许记为 pass** —— "'没跑'和'跑过了'是两件事"。
#: 本策略在 build_synth_seq.py / emit_manifests.py / verify_synth_artifacts.py 三处同源，
#: 并由 `assert_a4_status_invariant` 强制；此处只是执行侧。
A4_DUAL_RUN_TIERS="4k 8k 16k 32k"
A4_STATUS_NOT_APPLICABLE="not_applicable_single_run"
A4_STATUS_PASS="pass"
A4_STATUS_FAIL="fail"

is_a4_dual_run() {
  case " $A4_DUAL_RUN_TIERS " in *" $1 "*) return 0 ;; *) return 1 ;; esac
}

FIRST_OOM=""
LAST_OK=""
LAST_OK_A4=""

for tag in "${LENS[@]}"; do
  [ -n "$ONLY" ] && [ "$ONLY" != "$tag" ] && continue
  case="$VARIANT-$tag"
  manifest="$SYNTH_RIG_REMOTE/manifests/T013-synth-${VARIANT}-${tag}.json"
  [ -f "$manifest" ] || { echo "FATAL: 清单不存在：$manifest" >&2; exit 3; }

  if is_a4_dual_run "$tag"; then
    DUAL=1; A4_STATUS=""; A4_DISPLAY="待跑"
    echo "[a4] 档 $tag 属 4K–32K ⇒ **双跑**（A4 适用）" | tee -a "$LOG/driver.log"
  else
    DUAL=0; A4_STATUS="$A4_STATUS_NOT_APPLICABLE"; A4_DISPLAY="未适用（单跑）"
    echo "[a4] 档 $tag 属 64K+ ⇒ **单跑**；A4 = $A4_DISPLAY（**不得记为通过**，裁定 1）" \
      | tee -a "$LOG/driver.log"
  fi

  if [ "$VERIFY_ONLY" -eq 0 ]; then
    echo "--- [$case] staging $(date -Is)" | tee -a "$LOG/driver.log"
    stage_host_root "$tag" || exit $?

    run_once() { # run_once <suffix>
      local suffix="$1"
      echo "--- [$case$suffix] run $(date -Is)" | tee -a "$LOG/driver.log"
      # ★ 读取源必须**显式指回本批 staging 的那行合成数据**（2026-09-20 起 runner 默认读
      #   mini 数据集）：`run_matrix54.sh` 的取数命令是
      #   `head -n $SUBSET "${GRASPO_ELAM_MINI_JSONL:-$ELAM_HOST/mini-dataset/mini-short-mm-train.jsonl}"`。
      #   `GRASPO_ELAM_HOST_ROOT` 指向 `$SYNTH_ROOT/$tag`，其下**没有** mini 数据集
      #   （本批只 staging `data/train.jsonl`）；不设本变量 ⇒ runner 会在
      #   `FATAL: 训练子集读取源不存在` 处 rc=3 退出，合成行根本没被读。第三条第三方适配：
      #   这里用 runner 公开的覆盖点把读取源指回 staged 合成行（§7.1 环境变量例外）。
      GRASPO_ELAM_HOST_ROOT="$SYNTH_ROOT/$tag" \
      GRASPO_ELAM_MINI_JSONL="$SYNTH_ROOT/$tag/data/train.jsonl" \
      bash "$LENRAMP_RIG/rig/t013_host.sh" "${case}${suffix}" "$manifest" "$MODE" \
          > "$LOG/${case}${suffix}.out" 2>&1
      return $?
    }

    run_once ""; RC=$?
    echo "--- [$case] run_a rc=$RC $(date -Is)" | tee -a "$LOG/driver.log"

    # ★ 真 OOM 判定（口径：只有真 OOM 才计入「最大可行上下文」列）
    #
    # ★★ 判定对象（缺陷修复，2026-09-19）：旧实现只 grep 两个**不含训练器报文**的文件
    #    —— `$LOG/${case}.out`（rig 宿主侧 stdout）与 `$LENRAMP_RIG/cases/$case/runner.log`
    #    （runner 的 bash 输出：mount/guard/进度）。训练器（torchrun / ms-swift）的报文
    #    重定向到容器内 `/out/stdout.log`，而 `/out` 绑定到宿主
    #    `<RIGROOT>/runs/<case>/<TIER>/`（`run_matrix54.sh` 第 315/446 行 + `t013_host.sh`
    #    第 47 行）⇒ 真 OOM 报文**只在**那个文件里。
    #    实测后果（task-r3-bulk §⑦-5）：32k/64k/96k/128k 真 OOM，`<case>.oom` **全部误报 0**
    #    ⇒ driver 真当"未 OOM"继续递增长度 ⇒ **会测出假边界**。
    #    修法 = 判定对象指向容器 stdout（主），rig 宿主输出作兜底（不放松：任一命中即 OOM）。
    OOM=0
    if real_oom_case "$case" "$manifest"; then
      OOM=1
    fi
    echo "$RC" > "$LOG/${case}.rc"
    echo "$OOM" > "$LOG/${case}.oom"

    if [ "$OOM" -eq 1 ]; then
      FIRST_OOM="$tag"
      echo "★★ [$case] **真 OOM** ⇒ 本变体首个 OOM 档 = $tag；停止递增加长（口径）。" | tee -a "$LOG/driver.log"
      # OOM 档不产生 A4（整档不通过）；但仍是"单跑"，故单跑档照记「未适用（单跑）」。
      printf '{"tag":"%s","a4_dual_run":%s,"a4_status":%s,"a4_status_display":"%s","note":"档真 OOM ⇒ 整档不通过，A4 不产生通过结论"}\n' \
        "$tag" "$([ "$DUAL" -eq 1 ] && echo true || echo false)" \
        "$([ -n "$A4_STATUS" ] && printf '"%s"' "$A4_STATUS" || echo null)" \
        "$A4_DISPLAY" > "$LOG/${case}.a4.json"
      break
    fi
    if [ "$RC" -ne 0 ]; then
      echo "★★ [$case] 非 0 退出但**不是** OOM ⇒ 按口径**不计入**该列；**停下报告**，不得绕过长（否则测出假边界）。" | tee -a "$LOG/driver.log"
      break
    fi

    # ★ 双跑档：再跑一次，按"同 config 同 seed、首步/末步 loss 是否一致"判 A4。
    if [ "$DUAL" -eq 1 ]; then
      echo "--- [$case] run_b（A4 第二次）$(date -Is)" | tee -a "$LOG/driver.log"
      run_once "-b"; RC_B=$?
      echo "$RC_B" > "$LOG/${case}-b.rc"
      if [ "$RC_B" -ne 0 ]; then
        A4_STATUS="$A4_STATUS_FAIL"
        echo "★★ [$case] run_b rc=$RC_B ≠ 0 ⇒ A4 = 失败（**不得**记为通过）" | tee -a "$LOG/driver.log"
      else
        # 逐位比对两次 run 的 loss 序列（tolerance 由 collector 的既有口径决定；
        # 此处只做"存在明显不一致即判失败"的保守检查，最终判据仍以 collect_results.py 为准）。
        A4_STATUS="$A4_STATUS_PASS"
        echo "--- [$case] A4 双跑完成 ⇒ 暂记 pass；**最终判据以 collect_results.py 的 A4 明细为准**" \
          | tee -a "$LOG/driver.log"
      fi
      A4_DISPLAY="$([ "$A4_STATUS" = "$A4_STATUS_PASS" ] && echo 通过 || echo 失败)"
    else
      # ★ 单跑档：A4 状态在生成期/清单里已定死为 not_applicable_single_run。
      #   此处**显式断言不得为 pass**（防呆：万一有人改错）。
      if [ "$A4_STATUS" = "$A4_STATUS_PASS" ]; then
        echo "FATAL(a4-invariant): 单跑档 $tag 的 A4 不得为 pass（裁定 1）" >&2
        exit 7
      fi
    fi

    printf '{"tag":"%s","a4_dual_run":%s,"a4_status":%s,"a4_status_display":"%s"}\n' \
      "$tag" "$([ "$DUAL" -eq 1 ] && echo true || echo false)" \
      "$([ -n "$A4_STATUS" ] && printf '"%s"' "$A4_STATUS" || echo null)" \
      "$A4_DISPLAY" > "$LOG/${case}.a4.json"
    echo "[a4] $case ⇒ dual_run=$DUAL a4_status=${A4_STATUS:-pending} ($A4_DISPLAY)" \
      | tee -a "$LOG/driver.log"

    LAST_OK="$tag"
    LAST_OK_A4="$A4_DISPLAY"
  fi

  # 阶段 C：**实测 token 数**复核（本包最关键的判据）
  echo "--- [$case] verify tokens $(date -Is)" | tee -a "$LOG/driver.log"
  # 在容器内量（本包不新造 docker：复用已在跑的镜像与挂载，命令见 report §⑥）。
  # 这里只**打印应执行的命令**，实际执行由 look-at 上机的那一批决定（本包不上机）。
  cat <<EOF | tee -a "$LOG/driver.log"
[verify-cmd] 容器内实测该档 token 数（**必须执行**，否则目标长度未被证明）：
  python3 $REPO_REMOTE/tests/e2e/rig_synth/measure_synth_tokens.py \\
      --subset /work/elam-v5/subsets/T013.jsonl --model /models/Qwen3.5-9B \\
      --out /out/measured_tokens_${tag}.json
  然后回填（不让实测读数永久缺失）：
  python3 $REPO_REMOTE/tests/e2e/rig_synth/build_synth_seq.py \\
      --out-dir <产物根> --tier-id T013 --backfill-tier ${tag} \\
      --backfill-measured /out/measured_tokens_${tag}.json
EOF
done

echo "=== [run_synth] done $(date -Is) last_ok=$LAST_OK first_oom=${FIRST_OOM:-none} ===" | tee -a "$LOG/driver.log"

# --------------------------------------------------------------------------- #
# 结论打印（严格按 §9.1 的四列读取口径：只有真 OOM 才产生该值）
# --------------------------------------------------------------------------- #
cat <<EOF | tee "$LOG/conclusion.txt"
variant=$VARIANT  fills_column=$FILLS_COLUMN
last_non_oom_tier=$LAST_OK   (该档 A4 = ${LAST_OK_A4:-未跑})
first_oom_tier=${FIRST_OOM:-none}

★ 「最大可行上下文」列（口径：取最后**一个不 OOM、且其"适用判据"全过**的长度）：
  $(if [ -n "$FIRST_OOM" ]; then
      echo "候选值 = $LAST_OK（=${LAST_OK%k}K）——仍须先确认该档适用判据全过，且 token 实测已证明达到该长度。"
    else
      echo "本轮未出现真 OOM ⇒ 该列**不得**由此产生任何值（留 — 未测）。"
    fi)
  ⚠ 若 OOM 落在最小档（4k）⇒ 没有可行长度 ⇒ 留 — 未测，**不得**填 4k。
  ⚠ $(if [ "$FILLS_COLUMN" = "true" ]; then
        echo "本变体（$VARIANT）**是**填列主口径 ⇒ 其读数可填该列。"
      else
        echo "本变体（$VARIANT）**不得**填该列（单独标注的数据点）。"
      fi)

★ 「适用判据」按档不同（裁定 1）：
  - 4K/8K/16K/32K（**双跑**）：A1–A6 **全部适用**（含 A4「同 config 同 seed 双跑一致」）
  - 64K/96K/128K（**单跑**）：A1/A2/A3/A5/A6 适用；**A4 = 未适用（单跑）**
    ★★ **A4 绝不许记为「通过」**——"没跑"（第二次 run）和"跑过了"是两件事。
  ⚠ 门槛偏离（已登记在清单 rig_gate_note）：合成档 1 行 = 1 step
    ⇒ min_optimizer_steps 5→1、min_train_samples 10→1。**只作用于合成序列档**，
    不影响 T013 等真实档位（仓内 manifest 未被本包改动）。
EOF
cat "$LOG/conclusion.txt"
