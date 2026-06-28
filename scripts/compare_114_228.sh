#!/usr/bin/env bash
# Compare 114 (graspoflow 0.8.0, H100) vs 228 (native-tp 0.7.0, A800) — by epoch
set -euo pipefail

echo "============================================"
echo "  GRASPO 对比 (Epoch): 114 (0.8.0) vs 228 (0.7.0)"
echo "  $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================"

# Fetch status
ssh -p 22022 zhangzy@10.1.252.114 "docker ps --filter name=graspo_tp4_graspoflow --format '{{.Status}}' 2>/dev/null || echo 'NOT_RUNNING'" 2>/dev/null > /tmp/status_114.txt
ssh -p 22022 zhangzy@10.1.251.228 "docker ps --filter name=graspo_tp4_longrun --format '{{.Status}}' 2>/dev/null || echo 'NOT_RUNNING'" 2>/dev/null > /tmp/status_228.txt

echo "114 status: $(cat /tmp/status_114.txt)"
echo "228 status: $(cat /tmp/status_228.txt)"

# Fetch epoch data
ssh -p 22022 zhangzy@10.1.252.114 "docker logs graspo_tp4_graspoflow 2>&1 | grep 'epoch_summary'" 2>/dev/null > /tmp/epochs_114.jsonl &
ssh -p 22022 zhangzy@10.1.251.228 "docker logs graspo_tp4_longrun 2>&1 | grep 'epoch_summary'" 2>/dev/null > /tmp/epochs_228.jsonl &
ssh -p 22022 zhangzy@10.1.252.114 "docker logs graspo_tp4_graspoflow 2>&1 | grep 'train_step' | tail -1" 2>/dev/null > /tmp/step_114.jsonl &
ssh -p 22022 zhangzy@10.1.251.228 "docker logs graspo_tp4_longrun 2>&1 | grep 'train_step' | tail -1" 2>/dev/null > /tmp/step_228.jsonl &
wait

python3 << 'PYEOF'
import json

def parse_epochs(path):
    epochs = {}
    try:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                    e = d.get("epoch", {})
                    ep = e.get("epoch", "?")
                    epochs[ep] = e
                except:
                    pass
    except FileNotFoundError:
        pass
    return epochs

def parse_step(path):
    try:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    return json.loads(line)
    except:
        pass
    return None

epochs_114 = parse_epochs("/tmp/epochs_114.jsonl")
epochs_228 = parse_epochs("/tmp/epochs_228.jsonl")
step_114 = parse_step("/tmp/step_114.jsonl")
step_228 = parse_step("/tmp/step_228.jsonl")

# Progress info
print()
if step_114:
    e = step_114["epoch"]
    print(f"114 progress: epoch {e['epoch']} samples {e['samples_seen']}/{e['samples_total']} step {step_114['step']}")
else:
    print("114: no train_step yet")
if step_228:
    e = step_228["epoch"]
    print(f"228 progress: epoch {e['epoch']} samples {e['samples_seen']}/{e['samples_total']} step {step_228['step']}")
else:
    print("228: no train_step yet")

# Epoch comparison
common = sorted(set(epochs_114.keys()) & set(epochs_228.keys()))
print(f"\n{'='*80}")
if not common:
    print("No common epochs yet — 114 needs to complete its first epoch.")
else:
    print(f"Common epochs: {min(common)}..{max(common)} ({len(common)} epochs)")
    header = f"{'Epoch':>6} | {'114 reward':>10} | {'228 reward':>10} | {'114 content':>10} | {'228 content':>10} | {'114 retry':>9} | {'228 retry':>9} | {'114 train':>9} | {'228 train':>9}"
    sep = f"{'':->6}-+-{'':->10}-+-{'':->10}-+-{'':->10}-+-{'':->10}-+-{'':->9}-+-{'':->9}-+-{'':->9}-+-{'':->9}"
    print(header)
    print(sep)
    for ep in common:
        e114 = epochs_114[ep]; e228 = epochs_228[ep]
        ra114 = e114.get("decisions",{}).get("rollout_attempts",{})
        ra228 = e228.get("decisions",{}).get("rollout_attempts",{})
        tr114 = e114.get("decisions",{}).get("trainable",{})
        tr228 = e228.get("decisions",{}).get("trainable",{})
        print(f"{ep:>6} | {e114.get('reward_mean',0):10.4f} | {e228.get('reward_mean',0):10.4f} | "
              f"{e114.get('content_mean',0):10.4f} | {e228.get('content_mean',0):10.4f} | "
              f"{ra114.get('retry',0):>9} | {ra228.get('retry',0):>9} | "
              f"{tr114.get('total',0):>9} | {tr228.get('total',0):>9}")

    # Latest common epoch detail
    latest = max(common)
    print(f"\n--- Latest common epoch {latest} detail ---")
    e114 = epochs_114[latest]; e228 = epochs_228[latest]
    for name in ["reward_mean","content_mean","best_reward"]:
        print(f"  {name:<20}  114: {e114.get(name,0):.4f}    228: {e228.get(name,0):.4f}")
    d114 = e114.get("decisions",{}); d228 = e228.get("decisions",{})
    for name in ["rollout_attempts.retry","terminal.perfect_skip","terminal.invalid","trainable.total","trainable.max_correct"]:
        parts = name.split(".")
        v114 = d114.get(parts[0],{}).get(parts[1],0) if len(parts)==2 else d114.get(name,0)
        v228 = d228.get(parts[0],{}).get(parts[1],0) if len(parts)==2 else d228.get(name,0)
        print(f"  {name:<20}  114: {v114:<10}  228: {v228:<10}")

# Health check
with open("/tmp/status_114.txt") as f:
    status_114 = f.read().strip()
if "NOT_RUNNING" in status_114 or "Exited" in status_114:
    print(f"\n*** WARNING: 114 training STOPPED! Status: {status_114} ***")
PYEOF

echo ""
echo "============================================"