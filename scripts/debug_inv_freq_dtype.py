#!/usr/bin/env python3
"""Verify inv_freq dtype hypothesis and test proper fix."""
import torch
from pathlib import Path

device = torch.device("cuda:0")
torch.manual_seed(42)

# ---- Load both models ----
from transformers import Qwen3_5ForConditionalGeneration
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

# ---- Check inv_freq dtype ----
print("=== inv_freq dtype check ===")
hf_inv_freq = None
g_inv_freq = None
for name, buf in hf.model.visual.named_buffers():
    if "inv_freq" in name:
        hf_inv_freq = buf
        print(f"HF {name}: dtype={buf.dtype}, device={buf.device}, values={buf.float().tolist()}")
for name, buf in g.visual.named_buffers():
    if "inv_freq" in name:
        g_inv_freq = buf
        print(f"G  {name}: dtype={buf.dtype}, device={buf.device}, values={buf.float().tolist()}")

# ---- Fix 1: Change buffer dtype to float32 and copy HF values ----
print("\n=== Fix test: change buffer dtype to float32 + copy HF values ===")
for name, mod in g.visual.named_modules():
    if hasattr(mod, 'inv_freq'):
        # Save old
        old_inv = mod.inv_freq.data.clone()
        # Create float32 buffer
        new_inv = hf_inv_freq.data.float().clone()
        # Replace
        mod.register_buffer("inv_freq", new_inv, persistent=False)
        print(f"  {name}: dtype was {old_inv.dtype}, now {mod.inv_freq.dtype}")
        print(f"    Old: {old_inv.float().tolist()}")
        print(f"    New: {mod.inv_freq.float().tolist()}")

# ---- Re-run visual comparison ----
print("\n=== Visual output comparison after fix ===")
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
pv_bf16 = pv.to(torch.bfloat16)

with torch.no_grad():
    # HF reference
    hf_out = hf.model.visual(pv_bf16, grid_thw=gt)
    hf_last = hf_out.last_hidden_state
    hf_pooler = hf_out.pooler_output

    # GRASPO fixed
    g_out = g.visual.forward(pv_bf16, grid_thw=gt)
    g_last = g_out.last_hidden_state if hasattr(g_out, 'last_hidden_state') else g_out[0]
    g_pooler = g_out.pooler_output if hasattr(g_out, 'pooler_output') else g_out[1]

    last_diff = (hf_last.float() - g_last.float()).abs()
    pooler_diff = (hf_pooler.float() - g_pooler.float()).abs()

    print(f"Last hidden state maxdiff: {last_diff.max().item():.6f}")
    print(f"Last hidden state meandiff: {last_diff.mean().item():.6f}")
    print(f"Pooler output maxdiff: {pooler_diff.max().item():.6f}")
    print(f"Pooler output meandiff: {pooler_diff.mean().item():.6f}")

    if pooler_diff.max().item() < 1e-3:
        print("\n*** SUCCESS: Pooler output matches HF after dtype fix! ***")
    elif pooler_diff.max().item() < 0.1:
        print(f"\nPooler diff significantly reduced but not zero.")
    else:
        print(f"\nPooler still differs. inv_freq dtype is not the only issue.")

    # Compare block 0 after fix
    hf_b0 = {}
    g_b0 = {}
    h1 = hf.model.visual.blocks[0].register_forward_hook(
        lambda m, i, o: hf_b0.update({"v": (o[0] if isinstance(o, tuple) else o).detach()})
    )
    h2 = g.visual.blocks[0].register_forward_hook(
        lambda m, i, o: g_b0.update({"v": (o[0] if isinstance(o, tuple) else o).detach()})
    )
    _ = hf.model.visual(pv_bf16, grid_thw=gt)
    _ = g.visual.forward(pv_bf16, grid_thw=gt)
    h1.remove()
    h2.remove()
    if "v" in hf_b0 and "v" in g_b0:
        b0_diff = (hf_b0["v"].float() - g_b0["v"].float()).abs().max()
        print(f"Block 0 maxdiff after fix: {b0_diff:.6f} (was 0.0625)")
