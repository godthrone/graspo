#!/usr/bin/env python3
"""Verify inv_freq hypothesis: compare GRASPO vs HF visual tower, swap inv_freq, recheck."""
import torch
from pathlib import Path

device = torch.device("cuda:0")
torch.manual_seed(42)

# ---- Load HF visual tower ----
from transformers import Qwen3_5ForConditionalGeneration
hf = Qwen3_5ForConditionalGeneration.from_pretrained(
    "/workspace/models/Qwen3.5-9B", torch_dtype=torch.bfloat16,
    trust_remote_code=True, local_files_only=True,
).to(device).eval()

hf_visual = hf.model.visual

# ---- Load GRASPO visual tower ----
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
g_visual = g.visual

# ---- Part 1: Compare ALL parameters ----
print("=" * 60)
print("PART 1: Full parameter comparison")
print("=" * 60)

hf_params = dict(hf_visual.named_parameters())
g_params = dict(g_visual.named_parameters())

hf_keys = set(hf_params.keys())
g_keys = set(g_params.keys())
print(f"HF param keys: {len(hf_keys)}, G param keys: {len(g_keys)}")
print(f"Key sets equal: {hf_keys == g_keys}")

only_hf = hf_keys - g_keys
only_g = g_keys - hf_keys
if only_hf:
    print(f"Only in HF: {sorted(only_hf)[:20]}")
if only_g:
    print(f"Only in G:  {sorted(only_g)[:20]}")

param_mismatches = []
for name in sorted(hf_keys & g_keys):
    hf_w = hf_params[name]
    g_w = g_params[name]
    if hf_w.shape != g_w.shape:
        param_mismatches.append((name, "SHAPE", float('nan')))
        continue
    diff = (hf_w.float() - g_w.float()).abs().max().item()
    equal = torch.equal(hf_w, g_w)
    if diff > 1e-7 or not equal:
        param_mismatches.append((name, diff, equal))

print(f"\nParameter mismatches: {len(param_mismatches)}")
for name, diff, equal in param_mismatches[:20]:
    flag = "" if equal else " NOT_EQUAL"
    print(f"  {name}: maxdiff={diff:.10f}{flag}")

# ---- Part 2: Compare ALL buffers ----
print(f"\n{'=' * 60}")
print("PART 2: Full buffer comparison (includes inv_freq)")
print("=" * 60)

hf_bufs = dict(hf_visual.named_buffers())
g_bufs = dict(g_visual.named_buffers())

hf_buf_keys = set(hf_bufs.keys())
g_buf_keys = set(g_bufs.keys())
print(f"HF buffer keys: {len(hf_buf_keys)}, G buffer keys: {len(g_buf_keys)}")
print(f"Key sets equal: {hf_buf_keys == g_buf_keys}")

only_hf_b = hf_buf_keys - g_buf_keys
only_g_b = g_buf_keys - hf_buf_keys
if only_hf_b:
    print(f"Only in HF buffers: {sorted(only_hf_b)}")
if only_g_b:
    print(f"Only in G buffers:  {sorted(only_g_b)}")

for name in sorted(hf_buf_keys | g_buf_keys):
    hf_b = hf_bufs.get(name)
    g_b = g_bufs.get(name)
    if hf_b is None:
        print(f"  {name}: MISSING in HF")
        continue
    if g_b is None:
        print(f"  {name}: MISSING in G")
        continue
    if hf_b.shape != g_b.shape:
        print(f"  {name}: SHAPE MISMATCH HF={tuple(hf_b.shape)} G={tuple(g_b.shape)}")
        continue
    diff = (hf_b.float() - g_b.float()).abs().max().item()
    equal = torch.equal(hf_b, g_b)
    marker = ""
    if not equal or diff > 1e-7:
        marker = " *** MISMATCH ***"
    print(f"  {name}: shape={tuple(hf_b.shape)} maxdiff={diff:.10f} equal={equal}{marker}")

    # For inv_freq specifically, print full values
    if "inv_freq" in name.lower():
        print(f"    HF inv_freq: {hf_b.float().tolist()}")
        print(f"    G  inv_freq: {g_b.float().tolist()}")
        print(f"    per-element diff: {[(hf_b.float()[i] - g_b.float()[i]).item() for i in range(len(hf_b))]}")

# ---- Part 3: Visual forward comparison ----
print(f"\n{'=' * 60}")
print("PART 3: Visual forward comparison")
print("=" * 60)

import json
from transformers import AutoProcessor
proc = AutoProcessor.from_pretrained("/workspace/models/Qwen3.5-9B", trust_remote_code=True, local_files_only=True)
with open("/workspace/data/data/elam_graspo_train.jsonl") as f:
    s = json.loads(f.readline())

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
pv = inputs["pixel_values"].to(device)
gt = inputs["image_grid_thw"].to(device)

with torch.no_grad():
    hf_out = hf_visual(pv, grid_thw=gt).last_hidden_state
    g_out = g_visual.forward(pv, grid_thw=gt).last_hidden_state
    vdiff = (hf_out.float() - g_out.float()).abs()
    print(f"HF output: shape={hf_out.shape} norm={hf_out.float().norm():.4f}")
    print(f"G  output: shape={g_out.shape} norm={g_out.float().norm():.4f}")
    print(f"Maxdiff: {vdiff.max().item():.6f}")
    print(f"Meandiff: {vdiff.mean().item():.6f}")

# ---- Part 4: inv_freq SWAP test ----
print(f"\n{'=' * 60}")
print("PART 4: inv_freq SWAP test — copy HF inv_freq into GRASPO visual")
print("=" * 60)

# Find the rotary_pos_emb module in both and swap inv_freq
def find_rotary_modules(visual):
    result = {}
    for name, mod in visual.named_modules():
        if hasattr(mod, 'inv_freq'):
            result[name] = mod
    return result

hf_rotary = find_rotary_modules(hf_visual)
g_rotary = find_rotary_modules(g_visual)
print(f"HF rotary modules: {list(hf_rotary.keys())}")
print(f"G  rotary modules: {list(g_rotary.keys())}")

# Copy HF inv_freq → G for each rotary module
for name in hf_rotary:
    if name in g_rotary:
        hf_inv = hf_rotary[name].inv_freq.data
        g_rotary[name].inv_freq.data.copy_(hf_inv)
        print(f"  Copied inv_freq for {name}")

# Re-run visual forward with swapped inv_freq
with torch.no_grad():
    g_out_fixed = g_visual.forward(pv, grid_thw=gt).last_hidden_state
    vdiff_fixed = (hf_out.float() - g_out_fixed.float()).abs()
    print(f"G output (fixed): norm={g_out_fixed.float().norm():.4f}")
    print(f"Maxdiff after fix: {vdiff_fixed.max().item():.6f}")
    print(f"Meandiff after fix: {vdiff_fixed.mean().item():.6f}")

    # Compare to before-fix
    improvement = (vdiff.max().item() - vdiff_fixed.max().item())
    print(f"\nImprovement in maxdiff: {improvement:.6f}")
    if vdiff_fixed.max().item() < 1e-3:
        print("*** inv_freq FIX CONFIRMED: visual output now matches HF! ***")
    else:
        print("*** inv_freq is NOT the (only) issue — visual output still differs ***")

# ---- Part 5: Full weight comparison (just to be sure) ----
print(f"\n{'=' * 60}")
print("PART 5: Check for any other non-weight state (buffers besides inv_freq)")
print("=" * 60)
# Count total elements that differ
total_diff_elems = 0
for name in hf_buf_keys & g_buf_keys:
    if "inv_freq" in name.lower():
        continue  # already checked
    diff = (hf_bufs[name].float() - g_bufs[name].float()).abs().max().item()
    if diff > 1e-7:
        print(f"  Non-inv_freq buffer diff: {name} maxdiff={diff:.6f}")
        total_diff_elems += 1
if total_diff_elems == 0:
    print("  All non-inv_freq buffers match exactly.")
