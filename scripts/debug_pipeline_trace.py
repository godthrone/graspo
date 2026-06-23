#!/usr/bin/env python3
"""Instrument adapter pipeline at every step, compare with HF on same input."""

import json, torch
from pathlib import Path

torch.manual_seed(42)
device = torch.device("cuda:0")

# === Load HF model ===
from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration
print("Loading HF...", flush=True)
proc = AutoProcessor.from_pretrained(
    "/workspace/models/Qwen3.5-9B", trust_remote_code=True, local_files_only=True,
)
hf_model = Qwen3_5ForConditionalGeneration.from_pretrained(
    "/workspace/models/Qwen3.5-9B", torch_dtype=torch.bfloat16,
    trust_remote_code=True, local_files_only=True,
).to(device).eval()

# === Load GRASPO adapter ===
from graspo.core.schema import GraspoConfig
config = GraspoConfig.from_dict({
    "backend": "native-tp",
    "model": {"model_path": "/workspace/models/Qwen3.5-9B", "trust_remote_code": True,
               "torch_dtype": "bfloat16", "chat_template_kwargs": {"enable_thinking": False}},
    "data": {"train_path": "/workspace/data/data/elam_graspo_train.jsonl", "max_prompt_length": 2048},
    "training": {"rollout_group_size": 1, "max_new_tokens": 64,
                  "temperature": 0.0, "top_p": 1.0,
                  "optimize_prompt_batch_size": 1, "optimize_times_per_step": 1},
    "lora": {"r": 16, "alpha": 32, "dropout": 0.05, "target_preset": "language_safe"},
    "backend_config": {"native_tp": {"tp_size": 1, "pp_size": 1,
                       "placement_strategy": "qwen3_tp", "forward_batch_size": 1,
                       "use_kv_cache_for_rollout": True,
                       "empty_cache_after_rollout_split": False}},
})
from graspo.backends.native_tp.models.qwen.adapter import QwenNativeTPAdapter
print("Loading GRASPO adapter...", flush=True)
adapter = QwenNativeTPAdapter(config)
adapter.setup()

# === Load sample ===
with open("/workspace/data/data/elam_graspo_train.jsonl") as f:
    raw_samples = [json.loads(l) for l in f]
sample = raw_samples[5]  # Use sample 5 (known to diverge)

# === Build KNOWN-GOOD inputs (matching manual model test) ===
msgs = []
for m in sample["messages"]:
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
tools = sample.get("tools")
if tools:
    kwargs["tools"] = tools
good_inputs = proc.apply_chat_template(msgs, **kwargs)
good_ids = good_inputs["input_ids"].to(device)
good_attn = good_inputs["attention_mask"].to(device)
good_mm = {}
for k in ("pixel_values", "image_grid_thw", "mm_token_type_ids"):
    v = good_inputs.get(k)
    if v is not None and len(v) > 0:
        good_mm[k] = v.to(device)

print(f"\n=== KNOWN-GOOD inputs ===", flush=True)
print(f"  input_ids shape: {good_ids.shape}", flush=True)
print(f"  attn sum: {good_attn.sum().item()}", flush=True)
print(f"  pixel_values: {good_mm.get('pixel_values', 'N/A')}", flush=True)
print(f"  image_grid_thw: {good_mm.get('image_grid_thw', 'N/A')}", flush=True)

# === HF generate with logits ===
hf_kw = {k: v for k, v in good_mm.items() if v is not None}
with torch.no_grad():
    hf_gen = hf_model.generate(
        input_ids=good_ids, attention_mask=good_attn,
        max_new_tokens=64, do_sample=False, use_cache=True,
        pad_token_id=proc.tokenizer.eos_token_id,
        return_dict_in_generate=True, output_logits=True,
        **hf_kw,
    )
hf_tokens = hf_gen.sequences[0, good_ids.shape[1]:].tolist()
print(f"\n=== HF generate ===", flush=True)
print(f"  Generated {len(hf_tokens)} tokens", flush=True)
print(f"  First 15: {[proc.tokenizer.decode([t]) for t in hf_tokens[:15]]}", flush=True)

# === GRASPO model directly (known-good reference) ===
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
g_model = Qwen35HybridTextModel(
    hf_config=native_cfg, loader=loader,
    tp_rank=0, tp_size=1, placement=placement,
    lora_r=16, lora_alpha=32, lora_dropout=0.05,
    lora_targets={"language.full_attn.q_proj", "language.full_attn.v_proj",
                   "language.linear_attn.q_proj", "language.linear_attn.v_proj"},
    gradient_checkpointing=False, torch_dtype=torch.bfloat16, device=device,
).eval()

print(f"\n=== GRASPO model directly ===", flush=True)
g_pkv = None
g_seqs = good_ids.clone()
g_attn = good_attn.clone()
g_tokens = []
for step in range(64):
    with torch.no_grad():
        if step == 0:
            logits, g_pkv = g_model(
                input_ids=g_seqs, attention_mask=g_attn,
                multimodal_inputs=good_mm, use_cache=True,
            )
        else:
            logits, g_pkv = g_model(
                input_ids=g_seqs, attention_mask=g_attn,
                past_key_values=g_pkv, use_cache=True,
            )
    t = int(logits[0, -1, :].argmax().item())
    g_tokens.append(t)
    # Compare with HF at each step
    if step < len(hf_tokens) and t != hf_tokens[step]:
        print(f"  DIVERGE at step {step}: HF={proc.tokenizer.decode([hf_tokens[step]])!r} G={proc.tokenizer.decode([t])!r}", flush=True)
        break
    g_seqs = torch.tensor([[t]], dtype=torch.long, device=device)
    g_attn = torch.cat([g_attn, torch.ones(1, 1, dtype=g_attn.dtype, device=device)], dim=1)
else:
    print(f"  All {min(len(hf_tokens), len(g_tokens))} tokens match HF", flush=True)

# === NOW: Instrument adapter pipeline ===
print(f"\n=== ADAPTER PIPELINE TRACE ===", flush=True)

# 1. Monkey-patch _encode_multimodal_rows
orig_encode = adapter._encode_multimodal_rows
def traced_encode(rows, **kw):
    result = orig_encode(rows, **kw)
    # result is a dict with keys: input_ids, attention_mask, pixel_values, etc.
    print(f"[_encode_multimodal_rows]", flush=True)
    for k, v in result.items():
        if isinstance(v, torch.Tensor):
            print(f"  {k}: shape={list(v.shape)} dtype={v.dtype}", flush=True)
        elif isinstance(v, list):
            print(f"  {k}: len={len(v)}", flush=True)
        else:
            print(f"  {k}: {v}", flush=True)
    # Compare with known-good
    if "input_ids" in result:
        adapter_ids = result["input_ids"]
        print(f"  input_ids match known-good: {torch.equal(adapter_ids.cpu(), good_ids.cpu())}")
        if not torch.equal(adapter_ids.cpu(), good_ids.cpu()):
            for i in range(min(adapter_ids.shape[1], good_ids.shape[1])):
                if adapter_ids[0, i] != good_ids[0, i]:
                    print(f"    First diff at pos {i}: adapter={adapter_ids[0,i].item()} known={good_ids[0,i].item()}")
                    break
    if "pixel_values" in result:
        pv_adapter = result["pixel_values"]
        pv_known = good_mm.get("pixel_values")
        if pv_known is not None:
            pv_match = torch.equal(pv_adapter.cpu(), pv_known.cpu())
            print(f"  pixel_values match: {pv_match}")
    return result
adapter._encode_multimodal_rows = traced_encode

# 2. Monkey-patch _generate_multimodal_with_kv_cache
orig_gen_kv = adapter._generate_multimodal_with_kv_cache
def traced_gen_kv(*args, **kwargs):
    sequences = kwargs.get("sequences")
    attention_mask = kwargs.get("attention_mask")
    multimodal_inputs = kwargs.get("multimodal_inputs")
    temperature = kwargs.get("temperature", 1.0)
    max_new_tokens = kwargs.get("max_new_tokens", 64)

    batch = sequences.shape[0] if sequences is not None else "?"
    seq_len = sequences.shape[1] if sequences is not None else "?"
    print(f"\n[_generate_multimodal_with_kv_cache]", flush=True)
    print(f"  sequences: shape={sequences.shape}, batch={batch}, seq_len={seq_len}", flush=True)
    print(f"  attention_mask: shape={attention_mask.shape}, sum={attention_mask.sum().item()}", flush=True)
    print(f"  multimodal_inputs: {list(multimodal_inputs.keys()) if multimodal_inputs else 'None'}", flush=True)
    # Compare with known-good
    if sequences is not None:
        print(f"  sequences match known-good: {torch.equal(sequences.cpu(), good_ids.cpu())}")
    if attention_mask is not None:
        print(f"  attention_mask match known-good: {torch.equal(attention_mask.cpu(), good_attn.cpu())}")

    # Run the original and capture intermediate values via additional patching
    return orig_gen_kv(*args, **kwargs)
adapter._generate_multimodal_with_kv_cache = traced_gen_kv

# 3. Monkey-patch the MODEL's forward to capture the actual call
orig_model_fwd = adapter.model.forward
fwd_calls = []
def traced_model_fwd(*args, **kwargs):
    ids = kwargs.get("input_ids", args[0] if args else None)
    attn = kwargs.get("attention_mask", args[1] if len(args) > 1 else None)
    mm = kwargs.get("multimodal_inputs")
    pkv = kwargs.get("past_key_values")
    step = len(fwd_calls)
    fwd_calls.append({"step": step})
    if step < 3:  # Log first few calls
        print(f"\n  [model.forward call {step}]", flush=True)
        if ids is not None:
            print(f"    input_ids shape: {ids.shape}, first tok: {ids[0,0].item() if ids.shape[1]>0 else '?'}", flush=True)
        if attn is not None:
            print(f"    attn shape: {attn.shape}, sum: {attn.sum().item()}", flush=True)
        print(f"    multimodal_inputs: {list(mm.keys()) if mm else 'None'}", flush=True)
        print(f"    past_key_values: {'present' if pkv else 'None'}", flush=True)
    return orig_model_fwd(*args, **kwargs)
adapter.model.forward = traced_model_fwd

# === Run adapter generation ===
from graspo.core.data import load_jsonl
samples = load_jsonl(config.data.train_path)
s = samples[5]

print(f"\n=== Running adapter.generate_sample_groups ===", flush=True)
gen = adapter.generate_sample_groups(
    samples=[s], rollout_group_size=1, max_new_tokens=64,
    max_prompt_length=2048, temperature=0.0, top_p=1.0,
    chat_template_kwargs={"enable_thinking": False},
)
adapter_text = gen[0].completions[0]

print(f"\n=== COMPARISON ===", flush=True)
hf_text = proc.tokenizer.decode(hf_tokens)
print(f"HF:      {hf_text[:150]}", flush=True)
print(f"G_model: {proc.tokenizer.decode(g_tokens)[:150]}", flush=True)
print(f"Adapter: {adapter_text[:150]}", flush=True)

# Parse all three
from graspo.backends.native_tp.tool_parser import parse_qwen_tool_completion
hf_p = parse_qwen_tool_completion(hf_text, tools=s.tools)
g_p = parse_qwen_tool_completion(proc.tokenizer.decode(g_tokens), tools=s.tools)
a_p = parse_qwen_tool_completion(adapter_text, tools=s.tools)
print(f"\nParse errors: HF={len(hf_p.parse_errors)}, G_model={len(g_p.parse_errors)}, Adapter={len(a_p.parse_errors)}", flush=True)
if a_p.parse_errors:
    print(f"Adapter errors: {a_p.parse_errors}", flush=True)
