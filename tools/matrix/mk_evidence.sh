#!/usr/bin/env bash
# mk_evidence.sh — **取证/回填链的唯一入口**（A800 侧 `work/b09/mk_evidence.sh` 的收编版）。
#
#   用法: bash tools/matrix/mk_evidence.sh <M档号> [--dry-run]
#   例:   bash tools/matrix/mk_evidence.sh M008 --dry-run   # 只排练，不落任何文件
#         bash tools/matrix/mk_evidence.sh M008            # 真跑取证链（幂等）
#
# ★★ 为什么有本文件（D23 缺陷的根因，2026-10-27）：
#   A800 侧那份 `mk_evidence.sh` **不在版控里**，它自己直连
#   `scripts/record_checks.py` + `scripts/make_result_json.py`，**绕过 `fill_one.py`**
#   ⇒ `fill_one.py` 链首的「读数件合成（`scripts/mk_readings.py`）」这一步在 A800 上
#   **根本没被调用**。后果三次实测：M013 / M018 各人工补一次（`mk_evidence` rc=2，
#   `FAIL: readings.json 缺失或不可解析`）；M008 盘上的 `readings.json` 是**空壳**
#   （`exit_code=99` / 其余全 null），而同一目录 `readings.raw.json` 写着 `exit_code=0`
#   ⇒ 判据① 会把**跑成功**的档按"退出码 99"判失败。
#
# ★ 收编口径（宪法 §1.4 单一真相源 —— 本文件**不另立任何一步**）：
#   ⓪ 读数件合成 = `tools/matrix/fill_one.py::ensure_readings()`（**同一个函数**，⛔ 不另写一份
#      skip 判定 / argv）；幂等、失败可见、`--dry-run` ⇒ 传 `--check` **不落盘**；
#   ① `checks{}` / `teacher_after_acc` 的唯一产出者 = `scripts/record_checks.py`（fill_one 没有这步）；
#   ② 读数派生 + 四条判据现算 + §22 写范围闸门 + 幂等落盘 = **`tools/matrix/fill_one.sh` 一条命令**
#      （⛔ 本文件不再自己调 `make_result_json.py`：那就是"第二份通道"，正是本轮缺陷的成因）。
#   ⇒ 于是「走 mk_evidence」与「走 fill_one」得到的 `readings.json` / 证据集**逐字节相同**
#     （不是靠对拍维护，是靠**同一条链**成立；实证见工位 `DELIVERY.md` 的离线演练）。
#
# ★ `--dry-run`：⓪ 只算不写（`--check`）、① record_checks 是写盘步 ⇒ **跳过并明示**、
#   ② 透传给 `fill_one.sh --dry-run`（它自带"不写任何文件"保证）。全程零落盘。
#
# ★ 退出码（与 `fill_one.sh` 同语义，⛔ 不新造）：0 成功/幂等短路；2 用法或档号非法
#   /当轮副本缺失；3 证据不足（含**读数件合成失败**，该档保持「待测试」，未写任何文件）；
#   4 闸门失败（已回滚）；5 基准件三锁不成立；9 解释器环境不足（需进容器）。
#
# ★ 环境变量（缺省即生产口径；⛔ 只为"换现场"，不为"换口径"）：
#   GRASPO_RUN_ROOT / GRASPO_EVAL_ROOT / GRASPO_ROUND_DATE  原样透传（见 fill_one.sh）
#   GRASPO_EVIDENCE_EVAL_DIR   喂 `record_checks.py --eval-dir`；缺省 = `${GRASPO_EVAL_ROOT}/<档号>`（在盘才用）
#   GRASPO_EVIDENCE_STEP_FILL_ONE     ② 的委托目标（缺省 = 仓内 `tools/matrix/fill_one.sh`；单测注入口）
#   GRASPO_EVIDENCE_STEP_RECORD_CHECKS ① 的产出者（缺省 = 仓内 `scripts/record_checks.py`；单测注入口）
#   GRASPO_EVIDENCE_ROOTFS / _MOUNT_SPEC / _RUN_IN_ROOTFS / _ROOTFS_WORKDIR
#       非空 ⇒ 先用 `unshare` 把自身送进 rootfs 再跑（A800 原件的信封，缺省关 ⇒ 就在当前环境跑）
#       A800 现场取值：`<主机侧数据根>/rootfs/graspo` · `<主机侧数据根>/work/b09/mnt-verify-b09.json`
#       · `<主机侧数据根>/run_in_rootfs.py` · `/workspace/graspo`（rootfs 内 `GRASPO_RUN_ROOT=/runs`、
#       `GRASPO_EVAL_ROOT=/evalout` 由调用方按现场给，本文件不硬编码）
set -uo pipefail

REPO="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
PY="$REPO/.venv/bin/python"
[[ -x "$PY" ]] || PY="$(command -v python3 || true)"
[[ -n "${PY:-}" ]] || { printf '⛔ 找不到 python 解释器\n' >&2; exit 1; }

FILL_ONE="${GRASPO_EVIDENCE_STEP_FILL_ONE:-$REPO/tools/matrix/fill_one.sh}"
RECORD_CHECKS="${GRASPO_EVIDENCE_STEP_RECORD_CHECKS:-$REPO/scripts/record_checks.py}"

usage() {
  sed -n '2,12p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
}

# ── 参数 ────────────────────────────────────────────────────────────────────
TIER=""
DRY=0
for a in "$@"; do
  case "$a" in
    -h|--help) usage; exit 0 ;;
    --dry-run) DRY=1 ;;
    -*) printf '⛔ 未识别参数 %s（见 --help）\n' "$a" >&2; exit 2 ;;
    *) [[ -z "$TIER" ]] || { printf '⛔ 只接一个档号（已有 %s，又给 %s）\n' "$TIER" "$a" >&2; exit 2; }
       TIER="$a" ;;
  esac
done
if [[ -z "$TIER" ]]; then usage; exit 2; fi
if [[ ! "$TIER" =~ ^M[0-9]{3}$ ]]; then
  printf '⛔ 拒绝：非法档号 %s（须形如 M001…M068）⇒ 未做任何事（rc=2）\n' "$TIER" >&2; exit 2
fi

echo "→ 取证链 $TIER｜dry-run=$([[ $DRY == 1 ]] && echo 是 || echo 否)｜链: mk_readings(⓪) → record_checks(①) → fill_one(②)"

# ── 可选：rootfs 信封（A800 原件的能力；缺省关）───────────────────────────────
if [[ -n "${GRASPO_EVIDENCE_ROOTFS:-}" && "${GRASPO_EVIDENCE_IN_ROOTFS:-0}" != "1" ]]; then
  RIM="${GRASPO_EVIDENCE_RUN_IN_ROOTFS:?需要 GRASPO_EVIDENCE_RUN_IN_ROOTFS（rootfs 内 python）}"
  MS="${GRASPO_EVIDENCE_MOUNT_SPEC:?需要 GRASPO_EVIDENCE_MOUNT_SPEC（挂载规格 json）}"
  WD="${GRASPO_EVIDENCE_ROOTFS_WORKDIR:-/workspace/graspo}"
  echo "→ 送进 rootfs 再跑：$GRASPO_EVIDENCE_ROOTFS（挂载规格 $MS）"
  set -- "$TIER"; [[ $DRY == 1 ]] && set -- "$TIER" --dry-run
  exec env ROOTFS="$GRASPO_EVIDENCE_ROOTFS" MOUNT_SPEC="$MS" GRASPO_EVIDENCE_IN_ROOTFS=1 \
       "${PY}" "$RIM" bash -c "cd '$WD' || exit 7; bash tools/matrix/mk_evidence.sh $*"
fi

RUN_ROOT="${GRASPO_RUN_ROOT:-$REPO/runs}"
RDIR="$RUN_ROOT/$TIER"
if [[ ! -d "$RDIR" ]]; then
  echo "⛔ 缺跑次目录 $RDIR ⇒ 该档保持「待测试」，**未写任何文件**（rc=3）"
  exit 3
fi
echo "→ 跑次证据 $RDIR"

# ── ⓪ 读数件合成（唯一实现 = fill_one.py::ensure_readings，⛔ 不另写一份）──────────
readings_step() {
  "$PY" - "$REPO" "$RDIR" "$DRY" <<'PY'
import pathlib, sys
repo, rdir, dry = sys.argv[1], pathlib.Path(sys.argv[2]), sys.argv[3] == "1"
sys.path.insert(0, str(pathlib.Path(repo) / "tools" / "matrix"))
import fill_one                                    # 唯一真相源：同一条链的同一个函数
rc, out = fill_one.ensure_readings(rdir, dry_run=dry)
sys.stdout.write(out)
if out and not out.endswith("\n"):
    sys.stdout.write("\n")
sys.exit(rc)
PY
}
mk_out="$(readings_step)"; mk_rc=$?
echo "→ ⓪ 读数件合成 mk_readings.py rc=$mk_rc"
[[ -n "$mk_out" ]] && printf '%s\n' "$mk_out"
if (( mk_rc != 0 )); then
  # 失败可见：原样输出已在上面贴出（⛔ 不吞、不拿空值兜底）⇒ 拒绝进入取证链
  echo "⛔ 拒绝取证：$TIER 的读数件合成失败（mk_readings rc=$mk_rc）⇒ 该档保持「待测试」，**未写任何文件**（rc=3）"
  exit 3
fi

# ── ① checks{}/teacher_after_acc 的唯一产出者（fill_one 不含此步）─────────────────
if (( DRY == 1 )); then
  echo "→ ① record_checks.py **跳过**（它是写盘步：--dry-run 全程不落盘）"
else
  ev_args=()
  EV_DIR="${GRASPO_EVIDENCE_EVAL_DIR:-${GRASPO_EVAL_ROOT:+$GRASPO_EVAL_ROOT/$TIER}}"
  [[ -n "${EV_DIR:-}" && -d "$EV_DIR" ]] && ev_args=(--eval-dir "$EV_DIR")
  echo "→ ① record_checks $TIER（checks{} 唯一产出者）"
  rc_out="$("$PY" "$RECORD_CHECKS" --tier "$TIER" --run-dir "$RDIR" "${ev_args[@]}" 2>&1)"
  rc1=$?
  [[ -n "$rc_out" ]] && printf '%s\n' "$rc_out" | sed 's/^/  /'
  echo "→ ① record_checks rc=$rc1"
  # ⛔ 不因它非零就中断（与原件同口径：checks 缺证据时由回填器 fail-closed 判 unknown）
  (( rc1 != 0 )) && echo "  ⚠ checks 证据未写全（rc=$rc1）⇒ 该档判据③④ 将 fail-closed（保持「待测试」）"
fi

# ── ② 派生 + 判据 + 闸门 + 落盘（唯一通道 = fill_one.sh，⛔ 不另写一份）───────────
f_args=("$TIER"); (( DRY == 1 )) && f_args+=("--dry-run")
echo "→ ② 委托 $FILL_ONE ${f_args[*]}"
bash "$FILL_ONE" "${f_args[@]}"
f_rc=$?
echo "[mk_evidence] $TIER rc=$f_rc"
exit "$f_rc"
