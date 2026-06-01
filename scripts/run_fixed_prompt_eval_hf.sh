#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH="${MODEL_PATH:?Set MODEL_PATH to the base Hugging Face model path.}"
PROMPTS_PATH="${PROMPTS_PATH:?Set PROMPTS_PATH to fixed_eval_prompts.jsonl.}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/fixed-eval}"
ADAPTER_PATHS="${ADAPTER_PATHS:-}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-512}"
TORCH_DTYPE="${TORCH_DTYPE:-bfloat16}"

mkdir -p "${OUTPUT_DIR}"

python3 - "${MODEL_PATH}" "${PROMPTS_PATH}" "${OUTPUT_DIR}" "${ADAPTER_PATHS}" "${MAX_NEW_TOKENS}" "${TORCH_DTYPE}" <<'PY'
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_path = sys.argv[1]
prompts_path = Path(sys.argv[2])
output_dir = Path(sys.argv[3])
adapter_paths = [item for item in sys.argv[4].split(",") if item]
max_new_tokens = int(sys.argv[5])
torch_dtype_name = sys.argv[6]
torch_dtype = torch.bfloat16 if torch_dtype_name in {"bf16", "bfloat16"} else torch.float16

records = [json.loads(line) for line in prompts_path.read_text(encoding="utf-8").splitlines() if line.strip()]
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token


def render_prompt(record: dict) -> str:
    messages = record.get("messages")
    if messages and hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return str(record.get("prompt", ""))


def generate(model, label: str) -> None:
    output_path = output_dir / f"{label}.jsonl"
    with output_path.open("w", encoding="utf-8") as handle:
        for record in records:
            prompt = render_prompt(record)
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            with torch.no_grad():
                output_ids = model.generate(
                    **inputs,
                    do_sample=False,
                    max_new_tokens=max_new_tokens,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
            completion_ids = output_ids[0, inputs["input_ids"].shape[1] :]
            completion = tokenizer.decode(completion_ids, skip_special_tokens=True)
            handle.write(json.dumps({"id": record.get("id"), "completion": completion}, ensure_ascii=False) + "\n")
    print(f"Wrote {output_path}")


base_model = AutoModelForCausalLM.from_pretrained(
    model_path,
    trust_remote_code=True,
    torch_dtype=torch_dtype,
    device_map="auto",
)
base_model.eval()
generate(base_model, "base")

if adapter_paths:
    try:
        from peft import PeftModel
    except Exception as exc:
        print(f"PEFT is not available; adapter eval skipped: {exc}")
        raise SystemExit(0)

    for idx, adapter_path in enumerate(adapter_paths, start=1):
        adapter_dir = Path(adapter_path)
        label = adapter_dir.name or f"adapter_{idx}"
        if not (adapter_dir / "adapter_config.json").exists():
            print(f"Skipping {adapter_path}: not a Hugging Face PEFT adapter directory")
            continue
        model = PeftModel.from_pretrained(base_model, adapter_path)
        model.eval()
        generate(model, label)
        base_model.disable_adapter()
PY
