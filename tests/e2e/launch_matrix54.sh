#!/usr/bin/env bash
# batch2 起跑器：T010(SFT) / T028(GRASPO RL)，9B 单卡 native，官方基底+薄包装镜像
# 六条断言（树指纹 / 实测空闲 / 排期 / 叠跑闸 / 镜像+单源 / dry-run）→ nohup 起跑 + 巡检器
set -uo pipefail
TREE="${GRASPO_TREE:?need GRASPO_TREE}"
ENVF="${GRASPO_ENVF:?need GRASPO_ENVF}"
TIER="${GRASPO_TIER:?need GRASPO_TIER}"
GPU="${GRASPO_GPU:?need GRASPO_GPU}"
# ★ 暂存根（内含 sched_window_guard.py）：**由环境变量驱动，无内部默认值**。
#   2026-10-17 去内部化：原先回落到宿主内部绝对路径（含真实账号段）⇒ 违反 §15.1
#   「内网路径 / 真实账号不得出现在已跟踪文件里」，且门禁 gate_content 恒红。
#   现与上方 TREE/ENVF/TIER/GPU/FP_EXPECT **同一 fail-closed 形态**：未设即明确
#   报错退出，⛔ 不静默回落到任何内部值。
STAGE="${GRASPO_STAGE:?need GRASPO_STAGE}"
IMG=${GRASPO_IMG:-graspo:v0.28.11-cu130fix-v2-deepspeed-20261012-run}
IMGFP=${GRASPO_IMGFP:-f83c67d24ba1}
BASEIMG=${GRASPO_BASEIMG:-graspo:v0.28.11-cu130fix-v2-deepspeed-20261012}
BASEFP=${GRASPO_BASEFP:-d3985d02a633}
FP_EXPECT="${GRASPO_FP_EXPECT:?need GRASPO_FP_EXPECT}"
SCHED_SEC="${GRASPO_SCHED_SEC:-172800}"
ST="${GRASPO_STATUS_DIR:-$TREE/status-$TIER}"
PYBIN_OR_PYTHON="$(command -v python3)"
LOG="$ST/launch.log"; mkdir -p "$ST"
say(){ printf '%s %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" | tee -a "$LOG"; }
die(){ say "FATAL: $*"; say "===== 未起跑（fail-closed）====="; exit 1; }

say "===================== launch-batch2 START tier=$TIER gpu=$GPU ====================="
set -a; source "$ENVF"; set +a
say "REPO=$GRASPO_REPO_ROOT MANIFEST=$GRASPO_RUNNER_MANIFEST"
[ -d "$GRASPO_REPO_ROOT" ] || die "树不存在"

say "--- [assert 0] 代码身份（树指纹）---"
FP_GOT=$(cd "$TREE" && find . -type f ! -name 'ASSEMBLED-TREE.sha256' ! -path './runs*' ! -path './status*' ! -path '*/__pycache__/*' -print0 | xargs -0 sha256sum | LC_ALL=C sort | sha256sum | awk '{print $1}')
say "  树指纹=$FP_GOT"; say "  期望=$FP_EXPECT"
[ "$FP_GOT" = "$FP_EXPECT" ] || die "树指纹不符 ⇒ 拒跑"
say "  ASSERT_OK_CODE_IDENTITY（HEAD a674b4f tree + manifest 两档对齐生成 yaml）"

say "--- [assert 1] 卡实测（★ 逐卡循环：每张卡各自校验 剩余≥头寸 ∧ util 连续 3 帧 0）---"
export GRASPO_ALLOWED_GPU_INDICES="0,1,2,3,4,5,6,7"
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$PYTHONPATH" python3 "$TREE/scripts/gpu_lock_guard.py" --visible "$GPU" 2>&1 | tee -a "$LOG"
[ "${PIPESTATUS[0]}" -eq 0 ] || die "gpu_lock_guard 拒绝"
# ★ 逐卡断言：交给仓内**已固化且有单测**的实现 scripts/gpu_idle_assert.py
#   （默认连续采样 3 帧、间隔 12 s、帧帧满足；**任一**目标卡 剩余 < 本档头寸 或 util != 0 ⇒ rc=1 ⇒ 拒跑）。
#   ⛔ 不再手写单卡解析：旧写法对 `nvidia-smi -i 3,4` 会拿到两行 ⇒
#      `[: 81153\n81153: integer expression expected` ⇒ FATAL（228 实测）；
#      而绕法"只传一张卡做预检"会让**多卡档只验一张卡**，与"当场复测每张卡"的纪律不符。
#   头寸默认 48 GiB（与旧硬编码值一致），可用 GRASPO_GPU_POSITION_MIB 覆盖 —— 语义未变。
PYTHONDONTWRITEBYTECODE=1 "$PYBIN_OR_PYTHON" "$TREE/scripts/gpu_idle_assert.py" --visible "$GPU" 2>&1 | tee -a "$LOG"
[ "${PIPESTATUS[0]}" -eq 0 ] || die "目标卡实测空闲断言拒绝（逐卡：剩余 ≥ 头寸 ∧ util 连续 3 帧 0）"
say "  ASSERT_OK_GPU_IDLE_PER_CARD（GPU=$GPU；逐卡 3 帧）"

say "--- [assert 2] 排期守卫（★ 禁窗已按裁定解除）---"
# ★ 逐字理由（指挥官 2026-10-12 转达主席确认）：
#   「该窗（06:00–07:00 UTC）的唯一存在理由是基座每日 06:08–06:52 更换 NVIDIA 栈、
#    历史上已致跑次 CUDA 崩溃；主席已确认【已禁用 NVIDIA 自动更新】，故该窗已无对象 ⇒
#    ★ 禁窗解除；★ 若日后重新启用自动更新，本窗须立即恢复。」
# ★ 为什么不能"传一个空窗"给守卫：守卫要求 window-end > window-start（不支持跨零点），
#   而任何非零窗在多日跑次上都必然重叠 ⇒ 无法用参数表达"无禁窗"。
#   ⇒ 故本处**显式跳过**该守卫，并把"已解除 + 恢复条件"写进日志与报告。
#   ⛔ 只跳过"排期窗"这一项；⛔ 未改 idle 断言 / 树指纹 / 镜像断言 / 叠跑闸。
if [ "${GRASPO_SCHED_WINDOW_DISABLED:-}" = "1" ]; then
    say "  禁窗已解除（NVIDIA 自动更新已禁用，主席 2026-10-12 确认）⇒ SKIP 排期窗守卫"
    say "  ★ 恢复条件：若日后重新启用自动更新，必须立即恢复 06:00–07:00 UTC 禁窗"
    say "  ASSERT_OK_SCHED_WINDOW_DISABLED_BY_RULING"
else
    PYTHONDONTWRITEBYTECODE=1 python3 "$STAGE/sched_window_guard.py" --duration-sec "$SCHED_SEC" --label "$TIER" 2>&1 | tee -a "$LOG"
    [ "${PIPESTATUS[0]}" -eq 0 ] || die "排期守卫拒跑"
    say "  ASSERT_OK_SCHED_WINDOW"
fi

say "--- [assert 3] 叠跑闸（并行允许；只挡非我方容器）---"
# ★ 修正（2026-10-12）：本闸的**唯一目的**是"不与争用资源的他人作业并发"；
#   而 228 是共享机、第三方容器可能随时存在（如 vla-6dof 系），**其存在本身不等于争用**。
#   ⇒ 判据改为：★ 我方(graspo:*)一律允许共存；★ 非我方容器**只警告不拒跑**（真实争用由
#     卡空闲断言 util==0% + 运行期抽测/退让 覆盖；⛔ 永不 kill/停他人容器）。
for c in $(docker ps -q); do
  nm=$(docker inspect "$c" --format '{{.Name}}' | sed 's#^/##'); im=$(docker inspect "$c" --format '{{.Config.Image}}')
  case "$im" in graspo:*) say "  我方：$nm ($im) ⇒ 允许共存";; *) say "  ⚠ 非我方容器（存在但不动）：$nm ($im)";; esac
done
say "  ASSERT_OK_NO_FOREIGN_CONTAINER"

say "--- [assert 4] 镜像断言（官方基底 + 运行薄包装；F5 处置）---"
IID=$(docker image inspect "$IMG" --format '{{.Id}}' | sed 's/^sha256://'); say "  wrapper id=$IID (expect $IMGFP)"
case "$IID" in "$IMGFP"*) : ;; *) die "wrapper id 不符" ;; esac
BID=$(docker image inspect "$BASEIMG" --format '{{.Id}}' | sed 's/^sha256://'); say "  base(官方) id=$BID (expect $BASEFP)"
case "$BID" in "$BASEFP"*) : ;; *) die "base id 不符" ;; esac
[ "${GRASPO_IMAGE:-}" = "$IMG" ] || die "GRASPO_IMAGE($GRASPO_IMAGE) != $IMG ⇒ 单源不符"
say "  ASSERT_OK_IMAGE（基底=交付件 双锚之一；包装=运行便利）"

say "--- [precheck] dry-run ---"
( cd "$TREE" && bash tests/e2e/run_matrix54_source.sh "$TIER" --dry-run ) > "$ST/dryrun.txt" 2>&1
DRC=$?; say "  dry-run rc=$DRC"
[ "$DRC" -eq 0 ] || { tail -20 "$ST/dryrun.txt" | tee -a "$LOG"; die "dry-run 未通过"; }
grep -E "dry-run\] $TIER" "$ST/dryrun.txt" | tee -a "$LOG"

if [ -f "$RUN_ROOT/$TIER/exit_code" ]; then mv "$RUN_ROOT/$TIER/exit_code" "$RUN_ROOT/$TIER/exit_code.stale.$(date -u +%s)"; say "  已归档残留 exit_code"; fi
say "--- nohup 起跑 ---"
cd "$TREE" || die cd
nohup setsid bash tests/e2e/run_matrix54_source.sh "$TIER" > "$ST/train.$TIER.stdout.log" 2>&1 &
say "  pid=$! start=$(date -u +%Y-%m-%dT%H:%M:%SZ) stdout=$ST/train.$TIER.stdout.log"
echo "$!" > "$ST/RUNPID"
say "===================== launch-batch2 DONE tier=$TIER ====================="
