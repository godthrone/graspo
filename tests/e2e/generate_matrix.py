#!/usr/bin/env python3
"""GRASPO 并行测试矩阵生成器。

数学全排列：对所有 world_size ∈ {1, 2, 4}，枚举所有正整数三元组 (tp, dp, pp)
满足 tp × dp × pp = world_size。SP ∈ {on, off} 仅在 tp ≥ 2 时两态。

覆盖 RL (graspo) 和 SFT 两种训练方法。

输出：
  - samples/configs/matrix/*.yaml（56 个 config：28 RL + 28 SFT）
  - tests/e2e/run_matrix.sh（批量执行脚本）
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = PROJECT_ROOT / "samples" / "configs" / "matrix"
SCRIPT_PATH = PROJECT_ROOT / "tests" / "e2e" / "run_matrix.sh"

MODELS: dict[str, dict[str, str]] = {
    "9B": {
        "model_path": "/data/user/models/Qwen3.5-9B",
        "model_name": "Qwen3.5-9B",
    },
    "27B": {
        "model_path": "/data/user/vllm/Qwen3.8-27B",
        "model_name": "Qwen3.8-27B",
    },
}

TRAIN_METHODS = ["rl", "sft"]
WORLD_SIZES = [1, 2, 4]
TRAIN_PATH = "samples/data/tool_call_mm/train.jsonl"
TIMEOUT_SEC = 1200
DOCKER_IMAGE = "graspo:0.29.0"
GPU_DEVICES = "0,1,2,3"  # GPUs 0-3 on 121


def factor_triples(w: int) -> list[tuple[int, int, int]]:
    """Return all (tp, dp, pp) positive integer triples with tp*dp*pp == w."""
    result = []
    for tp in range(1, w + 1):
        for dp in range(1, w + 1):
            if w % (tp * dp) == 0:
                pp = w // (tp * dp)
                result.append((tp, dp, pp))
    return result


def _build_training_config(name: str, method: str) -> dict[str, Any]:
    """Build the training section, which differs between RL and SFT."""
    common = {
        "output_dir": "/workspace/outputs",
        "run_name": name,
        "seed": 42,
        "max_epochs": 1,
        "gradient_accumulation_micro_batches": 1,
        "weight_decay": 0.01,
        "max_grad_norm": 1.0,
        "save_steps": -1,
        "save_checkpoint_every_epoch": False,
        "lr_scheduler": {
            "type": "constant",
            "warmup_steps": 0,
            "min_lr_ratio": 0.0,
            "decay_steps": 0,
        },
    }

    if method == "rl":
        return {
            **common,
            "learning_rate": 5.0e-6,
            "rollout_group_size": 2,
            "rollout_queue_batch_size": 2,
            "rollout_max_retries": 2,
            "policy_ratio_clip_eps": 0.2,
            "max_new_tokens": 64,
            "temperature": 1.0,
            "top_p": 1.0,
            "perfect_skip_reward_threshold": 2.0,
            "reject_unparseable_groups": False,
        }
    else:  # sft
        return {
            **common,
            "learning_rate": 5.0e-5,
        }


def _build_reward_config(method: str) -> dict[str, Any] | None:
    """RL needs reward config; SFT does not."""
    if method == "rl":
        return {
            "kind": "graspo",
            "check_think": False,
            "check_json_markdown": True,
            "check_list_order": False,
            "marker_reward_weight": 10.0,
            "content_reward_weight": 100.0,
            "anti_useless_str_reward_weight": 1.0,
            "anti_useless_str_half_reward_len": 100,
            "numeric_tolerance": 0.2,
        }
    return None


def generate_config(
    model_key: str,
    method: str,
    tp: int,
    dp: int,
    pp: int,
    sp: bool,
    world_size: int,
    config_path: Path,
) -> None:
    """Generate a single e2e test config YAML."""
    model = MODELS[model_key]
    name = f"{model_key.lower()}_{method}_w{world_size}_tp{tp}_dp{dp}_pp{pp}_sp{1 if sp else 0}"

    placement = "auto"
    if tp >= 2 and pp == 1:
        placement = "qwen3_tp"

    train_method = "graspo" if method == "rl" else "sft"

    config: dict[str, Any] = {
        "train_method": train_method,
        "backend": "native",
        "model": {
            "model_path": model["model_path"],
            "trust_remote_code": True,
            "torch_dtype": "bfloat16",
            "gradient_checkpointing": True,
            "chat_template_kwargs": {"enable_thinking": False},
        },
        "data": {
            "train_path": TRAIN_PATH,
            "max_prompt_length": 8192,
        },
        "lora": {
            "r": 16,
            "alpha": 32,
            "dropout": 0.05,
            "adapter_path": None,
            "target_preset": "language_safe",
            "target_modules": ["language_all_linear", "vision_common"],
        },
        "native": {
            "tp_size": tp,
            "dp_size": dp,
            "pp_size": pp,
            "micro_batch_size": 1,
            "dp_replicate_lora": True,
        },
        "training": _build_training_config(name, method),
        "launch": None,
    }

    reward = _build_reward_config(method)
    if reward is not None:
        config["reward"] = reward

    if placement != "auto":
        config["native"]["placement_strategy"] = placement
    if sp:
        config["native"]["sequence_parallel"] = True

    # Write YAML manually for clean formatting
    config_path.parent.mkdir(parents=True, exist_ok=True)
    method_label = "RL" if method == "rl" else "SFT"
    with open(config_path, "w") as f:
        f.write(f"# GRASPO e2e matrix [{method_label}]: {model_key} w={world_size} TP={tp} DP={dp} PP={pp} SP={'on' if sp else 'off'}\n")
        _write_yaml(f, config, 0)


def _write_yaml(f, obj, indent: int, key: str | None = None) -> None:
    """Write a Python object as clean YAML."""
    prefix = "  " * indent

    if obj is None:
        if key is not None:
            f.write(f"{'  ' * (indent - 1)}{key}: null\n")
        else:
            f.write(f"{prefix}null\n")
        return
    elif isinstance(obj, bool):
        if key is not None:
            f.write(f"{'  ' * (indent - 1)}{key}: {str(obj).lower()}\n")
        else:
            f.write(f"{prefix}{str(obj).lower()}\n")
        return
    elif isinstance(obj, (int, float)):
        if key is not None:
            f.write(f"{'  ' * (indent - 1)}{key}: {obj}\n")
        else:
            f.write(f"{prefix}{obj}\n")
        return
    elif isinstance(obj, str):
        if key is not None:
            if obj == "":
                f.write(f'{'  ' * (indent - 1)}{key}: ""\n')
            else:
                f.write(f"{'  ' * (indent - 1)}{key}: {obj}\n")
        else:
            if obj == "":
                f.write(f'{prefix}""\n')
            else:
                f.write(f"{prefix}{obj}\n")
        return
    elif isinstance(obj, dict):
        if key is not None:
            f.write(f"{'  ' * (indent - 1)}{key}:\n")
        for k, v in obj.items():
            _write_yaml(f, v, indent + 1, k)
    elif isinstance(obj, list):
        if key is not None:
            f.write(f"{'  ' * (indent - 1)}{key}:\n")
        for item in obj:
            f.write(f"{prefix}  -")
            if isinstance(item, (dict, list)):
                f.write("\n")
                _write_yaml(f, item, indent + 2, None)
            elif item is None:
                f.write(" null\n")
            elif isinstance(item, bool):
                f.write(f" {str(item).lower()}\n")
            elif isinstance(item, str):
                f.write(f" {item}\n")
            else:
                f.write(f" {item}\n")


def generate_run_script(tests: list[dict]) -> None:
    """Generate tests/e2e/run_matrix.sh.

    Usage: bash run_matrix.sh <output_dir>

    - output_dir: 必填，测试输出根目录（如 /data/user/e2e-results）
    - 每个测试在 <output_dir>/<test_name>/ 下独立保存日志和训练产物
    - 汇总摘要写入 <output_dir>/summary.txt
    """
    gpu_map = {1: "0", 2: "0,1", 4: "0,1,2,3"}

    lines = [
        "#!/bin/bash",
        "# GRASPO e2e matrix runner — auto-generated by generate_matrix.py",
        "# Usage: bash run_matrix.sh <output_dir>",
        "# No set -e: single test failure does not stop the matrix",
        "",
        "set -u  # fail on undefined variables",
        "",
        'if [ $# -lt 1 ]; then',
        '  echo "Usage: $0 <output_dir>"',
        '  echo "  output_dir: directory to store per-test logs and outputs"',
        '  echo "  Example: $0 /data/user/e2e-results"',
        "  exit 1",
        "fi",
        "",
        'OUTDIR="$(realpath "$1")"',
        'SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"',
        'PROJ_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"',
        "",
        f"IMG={DOCKER_IMAGE}",
        f"MDIR9=/data/user/models",
        f"MDIR27=/data/user/vllm",
        'SDIR="$PROJ_DIR/samples"',
        'MOUNTS="-v $MDIR9:/data/user/models -v $MDIR27:/data/user/vllm -v $SDIR:/workspace/graspo/samples"',
        f"TIMEOUT={TIMEOUT_SEC}",
        "",
        'mkdir -p "$OUTDIR"',
        "",
        'SUMMARY="$OUTDIR/summary.txt"',
        'echo "=== GRASPO E2E Matrix $(date) ===" | tee "$SUMMARY"',
        'echo "Image: $IMG" | tee -a "$SUMMARY"',
        'echo "Timeout: $TIMEOUT s per test" | tee -a "$SUMMARY"',
        'echo "Output: $OUTDIR" | tee -a "$SUMMARY"',
        f'echo "Tests: {len(tests)}" | tee -a "$SUMMARY"',
        'echo "" | tee -a "$SUMMARY"',
        "",
        "PASS=0; FAIL=0; TIME=0",
        "",
        "run_one() {",
        '  local name="$1"; local cfg="$2"; local gpus="$3"',
        "",
        '  local test_dir="$OUTDIR/$name"',
        '  mkdir -p "$test_dir"',
        '  local log="$test_dir/log.txt"',
        "",
        '  echo "[$(date +%H:%M:%S)] ▶ $name" | tee -a "$SUMMARY"',
        "",
        '  docker rm -f "e2e-${name}" 2>/dev/null || true',
        "",
        '  local nccl_env=""',
        '  if [ "$gpus" != "0" ]; then',
        '    nccl_env="-e NCCL_P2P_DISABLE=1"',
        "  fi",
        "",
        '  local output_mount="-v $test_dir:/workspace/outputs"',
        "",
        "  timeout $TIMEOUT docker run --rm --name \"e2e-${name}\" \\",
        '    --gpus "\\\"device=${gpus}\\\"" \\',
        "    --ipc=host --shm-size=16g \\",
        "    -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \\",
        "    $nccl_env \\",
        "    $MOUNTS \\",
        '    $output_mount \\',
        '    $IMG launch --config "$cfg" > "$log" 2>&1',
        "  local rc=$?",
        "",
        '  docker rm -f "e2e-${name}" 2>/dev/null || true',
        "",
        "  if [ $rc -eq 0 ]; then",
        '    echo "  ✅ $name PASS" | tee -a "$SUMMARY"',
        "    PASS=$((PASS + 1))",
        "  elif [ $rc -eq 124 ]; then",
        '    echo "  ⏰ $name TIMEOUT" | tee -a "$SUMMARY"',
        "    TIME=$((TIME + 1))",
        "  else",
        '    echo "  ❌ $name FAIL (rc=$rc)" | tee -a "$SUMMARY"',
        "    FAIL=$((FAIL + 1))",
        '    echo "  Last 5 log lines:" | tee -a "$SUMMARY"',
        '    grep -v "MEM rank" "$log" 2>/dev/null | tail -5 | sed "s/^/    /" | tee -a "$SUMMARY"',
        "  fi",
        "}",
        "",
    ]

    # Group tests by method then model
    for method, method_label in [("rl", "RL"), ("sft", "SFT")]:
        method_tests = [t for t in tests if t["method"] == method]
        lines.append(f"echo '' | tee -a \"$SUMMARY\"")
        lines.append(f"echo '===== {method_label} ({len(method_tests)} tests) =====' | tee -a \"$SUMMARY\"")

        for model_key in ["9B", "27B"]:
            model_tests = [t for t in method_tests if t["model"] == model_key]
            lines.append(f"echo '' | tee -a \"$SUMMARY\"")
            lines.append(f"echo '--- {model_key} ({len(model_tests)} tests) ---' | tee -a \"$SUMMARY\"")

            for t in model_tests:
                gpus = gpu_map[t["world_size"]]
                lines.append(
                    f'run_one {t["name"]} "samples/configs/matrix/{t["name"]}.yaml" "{gpus}"'
                )

    lines.extend([
        "",
        'echo "" | tee -a "$SUMMARY"',
        'echo "=== DONE $(date) ===" | tee -a "$SUMMARY"',
        'echo "PASS=$PASS FAIL=$FAIL TIMEOUT=$TIME" | tee -a "$SUMMARY"',
    ])

    SCRIPT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(SCRIPT_PATH, "w") as f:
        f.write("\n".join(lines) + "\n")
    SCRIPT_PATH.chmod(0o755)


def main() -> None:
    # Clean config dir
    shutil.rmtree(CONFIG_DIR, ignore_errors=True)
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)

    tests: list[dict] = []

    for method in TRAIN_METHODS:
        for model_key in MODELS:
            for w in WORLD_SIZES:
                triples = factor_triples(w)
                for tp, dp, pp in triples:
                    sp_options = [False, True] if tp >= 2 else [False]
                    for sp in sp_options:
                        name = f"{model_key.lower()}_{method}_w{w}_tp{tp}_dp{dp}_pp{pp}_sp{1 if sp else 0}"
                        config_name = f"{name}.yaml"
                        config_path = CONFIG_DIR / config_name

                        generate_config(model_key, method, tp, dp, pp, sp, w, config_path)

                        tests.append({
                            "name": name,
                            "method": method,
                            "model": model_key,
                            "world_size": w,
                            "tp": tp,
                            "dp": dp,
                            "pp": pp,
                            "sp": sp,
                            "config": f"samples/configs/matrix/{config_name}",
                        })

    generate_run_script(tests)

    # Print summary
    print(f"Generated {len(tests)} configs → {CONFIG_DIR}")
    print(f"Run script → {SCRIPT_PATH}")
    print()

    for method in TRAIN_METHODS:
        count = sum(1 for t in tests if t["method"] == method)
        print(f"  {method.upper()}: {count} tests")

    print(f"\n  Total: {len(tests)} tests (14 per model × 2 models × 2 methods)")

    # Print matrix table
    print("\nMatrix:")
    print(f"{'Method':<6} {'Model':<4} {'W':<2} {'TP':<3} {'DP':<3} {'PP':<3} {'SP':<4} {'Name'}")
    print("-" * 60)
    for t in tests:
        sp_str = "on" if t["sp"] else "off"
        print(f"{t['method']:<6} {t['model']:<4} {t['world_size']:<2} {t['tp']:<3} {t['dp']:<3} {t['pp']:<3} {sp_str:<4} {t['name']}")


if __name__ == "__main__":
    main()