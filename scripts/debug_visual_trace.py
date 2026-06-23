#!/usr/bin/env python3
"""Debug: compare vision configs and trace the visual forward path."""
import torch
from pathlib import Path

device = torch.device("cuda:0")
torch.manual_seed(42)

# ---- Load HF ----
from transformers import Qwen3_5ForConditionalGeneration
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5VisionConfig

hf = Qwen3_5ForConditionalGeneration.from_pretrained(
    "/workspace/models/Qwen3.5-9B", torch_dtype=torch.bfloat16,
    trust_remote_code=True, local_files_only=True,
).to(device).eval()

hf_vision_config = hf.config.vision_config

# ---- Load GRASPO ----
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

# ---- Compare vision configs ----
print("=" * 60)
print("Vision Config Comparison")
print("=" * 60)
g_vision_dict = dict(getattr(native_cfg, "vision_config", {}) or {})
g_vision_config = g.visual.config if hasattr(g.visual, 'config') else None

print(f"HF vision_config type: {type(hf_vision_config).__name__}")
print(f"G  vision_config type: {type(g_vision_config).__name__}")

# Compare all config attributes
hf_vc = hf_vision_config.to_dict() if hasattr(hf_vision_config, 'to_dict') else vars(hf_vision_config)
g_vc = g_vision_config.to_dict() if hasattr(g_vision_config, 'to_dict') else g_vision_dict

all_keys = sorted(set(list(hf_vc.keys()) + list(g_vc.keys())))
print(f"\nConfig key diffs:")
for k in all_keys:
    hf_v = hf_vc.get(k, "<MISSING>")
    g_v = g_vc.get(k, "<MISSING>")
    if hf_v != g_v:
        print(f"  {k}: HF={hf_v} G={g_v}")
    else:
        print(f"  {k}: {hf_v} (match)")

# ---- Compare visual module structure ----
print(f"\n{'=' * 60}")
print("Module structure comparison")
print("=" * 60)

def module_tree(module, prefix=""):
    names = []
    for name, child in module.named_children():
        full = f"{prefix}.{name}" if prefix else name
        cls = type(child).__name__
        names.append((full, cls))
        names.extend(module_tree(child, full))
    return names

hf_tree = module_tree(hf.model.visual)
g_tree = module_tree(g.visual)

hf_set = set(name for name, _ in hf_tree)
g_set = set(name for name, _ in g_tree)

if hf_set != g_set:
    only_hf = hf_set - g_set
    only_g = g_set - hf_set
    if only_hf:
        print(f"Only in HF: {sorted(only_hf)[:10]}")
    if only_g:
        print(f"Only in G:  {sorted(only_g)[:10]}")

# Compare module types
print("\nModule type diffs:")
hf_map = dict(hf_tree)
g_map = dict(g_tree)
type_diffs = 0
for name in sorted(hf_set & g_set):
    if hf_map[name] != g_map[name]:
        print(f"  {name}: HF={hf_map[name]} G={g_map[name]}")
        type_diffs += 1
if type_diffs == 0:
    print("  All module types match!")

# ---- Check attention implementation ----
print(f"\n{'=' * 60}")
print("Attention implementation check")
print("=" * 60)

hf_attn_impl = getattr(hf_vision_config, '_attn_implementation', None) or getattr(hf.config, '_attn_implementation', None)
g_attn_impl = getattr(g.visual.config, '_attn_implementation', None) if hasattr(g.visual, 'config') else None

print(f"HF _attn_implementation: {hf_attn_impl}")
print(f"G  _attn_implementation: {g_attn_impl}")

# Check if SDPA is being used
print(f"\nHF model dtype: {hf.dtype}")
print(f"G visual dtype: {next(g.visual.parameters()).dtype}")

# ---- Detailed visual output analysis ----
print(f"\n{'=' * 60}")
print("Detailed visual output analysis")
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

print(f"pixel_values: shape={pv.shape}, dtype={pv.dtype}, min={pv.min():.4f}, max={pv.max():.4f}")

with torch.no_grad():
    hf_out = hf.model.visual(pv, grid_thw=gt).last_hidden_state
    # Explicit cast to bf16 for GRASPO input
    pv_bf16 = pv.to(torch.bfloat16)

    # Try: HF with bf16 input
    hf_out_bf16 = hf.model.visual(pv_bf16, grid_thw=gt).last_hidden_state
    print(f"\nHF(f32 in) vs HF(bf16 in): maxdiff={(hf_out.float() - hf_out_bf16.float()).abs().max():.4f}")

    # GRASPO: try both f32 and bf16 input
    g_out = g.visual.forward(pv, grid_thw=gt).last_hidden_state
    g_out_bf16 = g.visual.forward(pv_bf16, grid_thw=gt).last_hidden_state
    print(f"G(f32 in)  vs G(bf16 in):  maxdiff={(g_out.float() - g_out_bf16.float()).abs().max():.4f}")

    # Compare norm per output token
    hf_norms = hf_out_bf16.float().norm(dim=1)
    g_norms = g_out_bf16.float().norm(dim=1)
    norm_diff = (hf_norms - g_norms).abs()
    print(f"\nPer-token norm diff: max={norm_diff.max():.4f}, mean={norm_diff.mean():.4f}")

    # Find the most divergent token
    worst_idx = norm_diff.argmax().item()
    print(f"Worst token index: {worst_idx}")
    print(f"  HF norm: {hf_norms[worst_idx]:.4f}, G norm: {g_norms[worst_idx]:.4f}")

    # Element-wise comparison at worst token
    hf_vals = hf_out_bf16[worst_idx, :10].float().tolist()
    g_vals = g_out_bf16[worst_idx, :10].float().tolist()
    print(f"  HF first 10 dims: {[f'{v:.4f}' for v in hf_vals]}")
    print(f"  G  first 10 dims: {[f'{v:.4f}' for v in g_vals]}")

    # Per-dimension analysis
    per_dim_diff = (hf_out_bf16.float() - g_out_bf16.float()).abs().max(dim=0).values
    worst_dim = per_dim_diff.argmax().item()
    print(f"\nWorst dimension: {worst_dim}, maxdiff={per_dim_diff[worst_dim]:.4f}")

    # Check: is the difference worse at the beginning or end?
    first_half = hf_out_bf16[:10, :].float()
    first_half_g = g_out_bf16[:10, :].float()
    last_half = hf_out_bf16[-10:, :].float()
    last_half_g = g_out_bf16[-10:, :].float()
    print(f"First 10 tokens maxdiff: {(first_half - first_half_g).abs().max():.4f}")
    print(f"Last 10 tokens maxdiff:  {(last_half - last_half_g).abs().max():.4f}")

    # Check pooler output (after merger)
    hf_full = hf.model.visual(pv_bf16, grid_thw=gt)
    g_full = g.visual.forward(pv_bf16, grid_thw=gt)
    pooler_hf = hf_full.pooler_output
    pooler_g = g_full.pooler_output if hasattr(g_full, 'pooler_output') else g_full[1]
    print(f"\nPooler output:")
    print(f"  HF pooler: shape={pooler_hf.shape} norm={pooler_hf.float().norm():.4f}")
    print(f"  G  pooler: shape={pooler_g.shape} norm={pooler_g.float().norm():.4f}")
    print(f"  Pooler maxdiff: {(pooler_hf.float() - pooler_g.float()).abs().max():.4f}")

    # Check first layer output
    if hasattr(hf.model.visual, 'blocks') and hasattr(g.visual, 'blocks'):
        # Try to capture first block output
        hf_block0_out = {}
        g_block0_out = {}

        h_hf = hf.model.visual.blocks[0].register_forward_hook(
            lambda m, inp, out: hf_block0_out.update(
                {"val": (out[0] if isinstance(out, tuple) else out).detach()}
            )
        )
        h_g = g.visual.blocks[0].register_forward_hook(
            lambda m, inp, out: g_block0_out.update(
                {"val": (out[0] if isinstance(out, tuple) else out).detach()}
            )
        )

        with torch.no_grad():
            _ = hf.model.visual(pv_bf16, grid_thw=gt)
            _ = g.visual.forward(pv_bf16, grid_thw=gt)

        h_hf.remove()
        h_g.remove()

        if "val" in hf_block0_out and "val" in g_block0_out:
            hf_b0 = hf_block0_out["val"]
            g_b0 = g_block0_out["val"]
            block_diff = (hf_b0.float() - g_b0.float()).abs().max()
            print(f"\nBlock 0 output maxdiff: {block_diff:.4f}")
            print(f"  HF block 0 norm: {hf_b0.float().norm():.4f}")
            print(f"  G  block 0 norm: {g_b0.float().norm():.4f}")
