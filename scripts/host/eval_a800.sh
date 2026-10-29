#!/bin/bash
# 档级评测（chroot 版，无 docker）：eval_a800.sh <TIER> <FORM:adapter|full> <CKPT> <SERVE_GPUS> [SIDE:both|after]
# CKPT = 容器内路径（/runs/...）; SERVE_GPUS 如 0,1
#
# ⚠️ 主机侧 harness：**只在评测主机上直接执行**（自身不进容器）。
# 两个主机相关量做成可覆盖，默认值 = 原先硬编码值 ⇒ 行为零变化：
#   GRASPO_EVAL_HOST_ROOT  主机侧数据根（默认 /data/remote）
#   GRASPO_EVAL_PORT       vLLM 端口（默认 8103，与 gen_eval --port 同源）
set -uo pipefail
HOST_ROOT="${GRASPO_EVAL_HOST_ROOT:-/data/remote}"
TIER="${1:?}"; FORM="${2:?}"; CK="${3:?}"; GPUS="${4:-0,1}"; SIDE="${5:-both}"
RT=$HOST_ROOT/rootfs/graspo; RV=$HOST_ROOT/rootfs/vllm
MS=$HOST_ROOT/mnt-eval.json; EV=$HOST_ROOT/eval/$TIER
PORT="${GRASPO_EVAL_PORT:-8103}"; LOWER=$(echo "$TIER" | tr 'A-Z' 'a-z')
MD=/evalout/merged-hf-$LOWER
MDH=$HOST_ROOT/eval/merged-hf-$LOWER
export GRASPO_EVAL_DIR=/evalout/$TIER
mkdir -p "$EV/evidence" "$EV/.local/reports/after" "$EV/.local/reports/base"
LOG="$EV/eval.log"; say(){ printf '%s %s\n' "$(date -u +%FT%TZ)" "$*" | tee -a "$LOG"; }
TRAIN(){  # 在训练 rootfs 里跑
  ROOTFS=$RT MOUNT_SPEC=$MS CUDA_VISIBLE_DEVICES="${CV:-$GPUS}" \
    unshare -r -m -p -f --mount-proc --kill-child python3 $HOST_ROOT/run_in_rootfs.py "$@"
}
VSERVE_BG(){ # $1=模型目录(容器路径)  在 vllm rootfs 里后台起服务
  ROOTFS=$RV MOUNT_SPEC=$MS CUDA_VISIBLE_DEVICES="$GPUS" \
    TRITON_LIBCUDA_PATH=/usr/lib/x86_64-linux-gnu \
    setsid nohup unshare -r -m -p -f --mount-proc --kill-child python3 $HOST_ROOT/run_in_rootfs.py \
      python3 -m vllm.entrypoints.openai.api_server --model "$1" --served-model-name graspo-eval \
      --port $PORT --max-model-len 8192 --gpu-memory-utilization 0.85 --trust-remote-code \
      --enable-auto-tool-choice --tool-call-parser qwen3_xml >> "$2" 2>&1 < /dev/null &
  echo $!
}
wait_ready(){ for i in $(seq 1 120); do
    if curl -s -m 3 "http://127.0.0.1:$PORT/v1/models" 2>/dev/null | grep -q graspo-eval; then return 0; fi; sleep 10; done; return 1; }
stop_vllm(){ pkill -KILL -f "vllm.entrypoints.openai.api_server --model" 2>/dev/null; sleep 3; }

rm -rf $HOST_ROOT/eval/.local/reports
say "=== $TIER eval START form=$FORM ckpt=$CK gpus=$GPUS side=$SIDE"
# 1) 口径生成（gen_eval 现算 caliber / §7 / §10）
TRAIN python3 /workspace/graspo/scripts/gen_eval.py --tier "$TIER" --gpu "$GPUS" --port $PORT \
  --no-ssh --form "$FORM" --checkpoint "$CK" --tmp-dir /evalout/$TIER >> "$LOG" 2>&1
say "gen_eval rc=$?"
sed -i "s#^checkpoint_path:.*#checkpoint_path: $MD#" "$HOST_ROOT/eval/$TIER/eval-$LOWER-after.yaml" 2>/dev/null || true
AFTER_Y=/evalout/$TIER/eval-$LOWER-after.yaml; BASE_Y=/evalout/$TIER/eval-$LOWER-base.yaml
MP=$(grep -m1 "^base_model_path:" "$HOST_ROOT/eval/$TIER/eval-$LOWER-after.yaml" | awk "{print \$2}")
say "base_model=$MP"
# 2) after 侧：LoRA 先 merge
if [ "$FORM" = adapter ]; then
  rm -rf "$MDH"
  cat > "$EV/export.yaml" <<Y
dataset_path: /work/elam-v5/subsets-v2/$TIER.jsonl
output_dir: /evalout/.local/reports/after
base_model_path: $MP
checkpoint_path: $CK
merged_output_dir: $MD
gpus: "$GPUS"
role: after
Y
  CV=$GPUS TRAIN /usr/local/bin/graspo eval export --eval-config "/evalout/$TIER/export.yaml" >> "$EV/evidence/02-export.txt" 2>&1
  say "export rc=$? → $(du -sh "$MDH" 2>/dev/null | cut -f1)"
  SERVE=$MD
else
  say "form=full ⇒ 零 merge"
  SERVE=$CK
fi
[ -f "$EV/export.yaml" ] && LORA_MERGED=true || LORA_MERGED=false
[ "$FORM" = full ] && LORA_MERGED=na
# 3) after 服务 + 评测
stop_vllm; VSERVE_BG "$SERVE" "$EV/evidence/vllm-after.log" >/dev/null
if wait_ready; then
  say "vLLM after READY"
  CV="$GPUS" TRAIN /usr/local/bin/graspo eval run --eval-config "$AFTER_Y" > "$EV/evidence/03-eval-after.txt" 2>&1
  say "eval after rc=$?"
else
  say "★ vLLM after 未就绪"; tail -5 "$EV/evidence/vllm-after.log" >> "$LOG"; stop_vllm; exit 8
fi
[ "$SIDE" = after ] && { stop_vllm; say "=== $TIER AFTER-ONLY DONE"; exit 0; }
stop_vllm
# 4) base 服务 + 评测（base 侧 = 底座模型）
VSERVE_BG "$MP" "$EV/evidence/vllm-base.log" >/dev/null
if wait_ready; then
  say "vLLM base READY"
  CV="$GPUS" TRAIN /usr/local/bin/graspo eval run --eval-config "$BASE_Y" > "$EV/evidence/04-eval-base.txt" 2>&1
  say "eval base rc=$?"
else
  say "★ vLLM base 未就绪"; tail -5 "$EV/evidence/vllm-base.log" >> "$LOG"; stop_vllm; exit 9
fi
stop_vllm
# 5) 报告副本先行：delta 读的是 $EV/.local/reports/ 下的副本，**必须先拷再算**
#    工具修复 2026-10-29：原顺序把 delta 放在拷贝之前 ⇒ 必然 not found ⇒ delta rc=1。
#    只挪顺序，delta 的计算逻辑与 judge4 口径**均未改动**。
mkdir -p "$EV/.local/reports"; cp -a $HOST_ROOT/eval/.local/reports/. "$EV/.local/reports/" 2>/dev/null
# 6) delta
TRAIN /usr/local/bin/graspo eval delta --before /evalout/$TIER/.local/reports/base/eval_report.json \
  --after /evalout/$TIER/.local/reports/after/eval_report.json --eval-config "$AFTER_Y" \
  > "$EV/evidence/05-delta.txt" 2>&1
say "delta rc=$?"
# 7) judge4.json（字段名按指挥官指定）
python3 - "$EV" "$TIER" "$LORA_MERGED" <<'PY'
import json, sys, os
ev, tier, merged = sys.argv[1], sys.argv[2], sys.argv[3]
evr = ev + "/.local/reports"
def load(p):
    try: return json.load(open(p))
    except Exception: return None
a = load(f"{evr}/after/eval_report.json")
b = load(f"{evr}/base/eval_report.json")
def pick(d, *names):
    for n in names:
        if isinstance(d, dict) and n in d and d[n] is not None: return d[n]
    for k, v in (d or {}).items():
        if isinstance(v, dict):
            r = pick(v, *names)
            if r is not None: return r
    return None
out = {"tier": tier, "lora_merged": merged, "served_by_vllm": bool(a is not None),
       "after_acc": pick(a, "accuracy", "acc", "overall_acc", "score"),
       "base_acc": pick(b, "accuracy", "acc", "overall_acc", "score"),
       "parse_rate": pick(a, "parse_rate", "parseable_rate"),
       "parse_rate_base": pick(b, "parse_rate", "parseable_rate")}
if out["after_acc"] is not None and out["base_acc"] is not None:
    out["delta_pp"] = round((out["after_acc"] - out["base_acc"]) * 100, 2)
else: out["delta_pp"] = None
json.dump(out, open(f"{ev}/judge4.json", "w"), ensure_ascii=False, indent=2)
print(json.dumps(out, ensure_ascii=False))
PY
say "=== $TIER EVAL DONE"
