#!/usr/bin/env python3
"""E2E generation comparison: GRASPO (with fix) vs HF, greedy decode, 50 tokens."""
import json, torch
from pathlib import Path

device = torch.device("cuda:0")
torch.manual_seed(42)

# ---- Load both models ----
from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration
proc = AutoProcessor.from_pretrained("/workspace/models/Qwen3.5-9B", trust_remote_code=True, local_files_only=True)
hf = Qwen3_5ForConditionalGeneration.from_pretrained(
    "/workspace/models/Qwen3.5-9B", torch_dtype=torch.bfloat16,
    trust_remote_code=True, local_files_only=True,
).to(device).eval()

from graspo.backends.native_tp.models.qwen.modeling_hybrid import Qwen35HybridTextModel
from graspo.backends.native_tp.models.qwen.modeling import load_native_qwen_config
from graspo.backends.native_tp.tensor_utils import SafetensorIndex
from graspo.backends.native_tp.placement import build_placement_plan

native_cfg = load_native_qwen_config(Path("/workspace/models/Qwen3.5-9B"))
loader = SafetensorIndex(Path("/workspace/models/Qwen3.5-9B"))
types = list(getattr(native_cfg, "layer_types", []) or [])
placement = build_placement_plan(
    strategy="qwen3_tp", model_family=native_cfg.family,
    num_hidden_layers=int(native_cfg.num_hidden_layers),
    tp_size=1, pp_size=1, tp_rank=0, pp_rank=0, layer_types=types,
)
g = Qwen35HybridTextModel(
    hf_config=native_cfg, loader=loader,
    tp_rank=0, tp_size=1, placement=placement,
    lora_r=0, lora_alpha=1, lora_dropout=0.0,
    lora_targets=set(), gradient_checkpointing=False,
    torch_dtype=torch.bfloat16, device=device,
).eval()

# ---- Load samples ----
with open("/workspace/data/data/elam_graspo_train.jsonl") as f:
    samples = [json.loads(line) for line in f if line.strip()][:10]

print(f"Testing {len(samples)} samples with greedy decode (T=0), max 50 new tokens\n")

total_tokens = 0
total_matches = 0
all_match_samples = 0

eos_id = proc.tokenizer.eos_token_id

for si, s in enumerate(samples):
    msgs = []
    for m in s["messages"]:
        c = m.get("content", "")
        if isinstance(c, list):
            nc = []
            for item in c:
                if isinstance(item, dict) and item.get("type") == "image":
                    img_name = Path(item["image"]).name
                    nc.append({"type": "image", "image": f"/workspace/images/{img_name}"})
                else:
                    nc.append(item)
            msgs.append({"role": m["role"], "content": nc})
        else:
            msgs.append({"role": m["role"], "content": c})

    kwargs = {"tokenize": True, "add_generation_prompt": True, "return_dict": True,
              "return_tensors": "pt", "enable_thinking": False}
    if s.get("tools"):
        kwargs["tools"] = s["tools"]
    inputs = proc.apply_chat_template(msgs, **kwargs)
    ids = inputs["input_ids"].to(device)
    attn = inputs["attention_mask"].to(device)
    mm = {}
    for k in ("pixel_values", "image_grid_thw", "mm_token_type_ids"):
        v = inputs.get(k)
        if v is not None and v.numel() > 0:
            mm[k] = v.to(device)

    # ---- HF greedy decode ----
    with torch.no_grad():
        hf_out = hf.generate(
            ids, attention_mask=attn, **{k: v for k, v in mm.items()},
            max_new_tokens=50, do_sample=False, pad_token_id=0,
            eos_token_id=eos_id,
        )
    hf_tokens = hf_out[0, ids.shape[1]:].tolist()

    # ---- GRASPO greedy decode ----
    g_tokens = []
    past_kv = None
    current_ids = ids
    current_mask = attn
    current_mm = mm  # Only used for prefill
    for step in range(50):
        with torch.no_grad():
            logits, past_kv = g(
                current_ids,
                attention_mask=current_mask,
                past_key_values=past_kv,
                multimodal_inputs=current_mm,
                use_cache=True,
            )
        next_logits = logits[:, -1, :].float()
        next_token = next_logits.argmax(dim=-1).item()
        g_tokens.append(next_token)

        # Prepare for next step
        current_ids = torch.tensor([[next_token]], device=device)
        current_mask = torch.cat([
            current_mask,
            torch.ones(current_mask.shape[0], 1, dtype=current_mask.dtype, device=device)
        ], dim=1)
        current_mm = None  # No multimodal inputs for decode steps

        if next_token == eos_id:
            break

    # Compare
    match_count = 0
    first_mismatch = -1
    for i in range(min(len(hf_tokens), len(g_tokens))):
        if hf_tokens[i] == g_tokens[i]:
            match_count += 1
        elif first_mismatch < 0:
            first_mismatch = i

    total_tokens += min(len(hf_tokens), len(g_tokens))
    total_matches += match_count
    all_match = hf_tokens == g_tokens[:len(hf_tokens)] and len(hf_tokens) == len(g_tokens)
    if all_match:
        all_match_samples += 1

    status = "✅ MATCH" if all_match else f"❌ MISMATCH at token {first_mismatch}"
    print(f"  Sample {si+1}: {match_count}/{min(len(hf_tokens), len(g_tokens))} tokens match, "
          f"HF={len(hf_tokens)} G={len(g_tokens)} {status}")

print(f"\n{'='*50}")
print(f"Summary: {all_match_samples}/{len(samples)} samples match perfectly")
print(f"Token-level: {total_matches}/{total_tokens} match")
if all_match_samples == len(samples):
    print("*** ALL SAMPLES MATCH — VISUAL TOWER FIX CONFIRMED! ***")
else:
    print("Some samples have mismatches — may be other issues.")
