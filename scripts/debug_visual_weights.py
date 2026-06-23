"""Compare GRASPO vs HF visual tower weights."""
import torch
from pathlib import Path

device = torch.device("cuda:0")

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

# Compare visual tower parameters
hf_params = dict(hf.model.visual.named_parameters())
g_params = dict(g.visual.named_parameters())

print(f"HF visual params: {len(hf_params)} keys")
print(f"G  visual params: {len(g_params)} keys")
print(f"Key sets match: {set(hf_params.keys()) == set(g_params.keys())}")

if set(hf_params.keys()) != set(g_params.keys()):
    only_hf = set(hf_params.keys()) - set(g_params.keys())
    only_g = set(g_params.keys()) - set(hf_params.keys())
    print(f"  Only in HF: {sorted(only_hf)[:10]}")
    print(f"  Only in G:  {sorted(only_g)[:10]}")

# Show all parameter diffs
print("\nPer-parameter comparison:")
all_match = True
for name in sorted(hf_params.keys())[:20]:
    hf_w = hf_params[name].data
    g_w = g_params[name].data if name in g_params else None
    if g_w is None:
        print(f"  {name}: MISSING in GRASPO")
        all_match = False
        continue
    if hf_w.shape != g_w.shape:
        print(f"  {name}: SHAPE MISMATCH HF={tuple(hf_w.shape)} G={tuple(g_w.shape)}")
        all_match = False
        continue
    diff = (hf_w.float() - g_w.float()).abs()
    maxd = diff.max().item()
    equal = torch.equal(hf_w, g_w)
    if maxd > 0.001 or not equal:
        all_match = False
    print(f"  {name}: shapes HF={tuple(hf_w.shape)} G={tuple(g_w.shape)} maxdiff={maxd:.6f} equal={equal}")

# Also check buffers (non-parameter state like running_mean etc)
hf_bufs = dict(hf.model.visual.named_buffers())
g_bufs = dict(g.visual.named_buffers())
if hf_bufs:
    print(f"\nBuffer comparison ({len(hf_bufs)} buffers):")
    for name in sorted(hf_bufs.keys())[:10]:
        hf_b = hf_bufs[name]
        g_b = g_bufs[name] if name in g_bufs else None
        if g_b is None:
            print(f"  {name}: MISSING in GRASPO")
            continue
        if hf_b.shape != g_b.shape:
            print(f"  {name}: SHAPE MISMATCH")
            continue
        diff = (hf_b.float() - g_b.float()).abs()
        print(f"  {name}: maxdiff={diff.max().item():.6f}")

if all_match:
    print("\nALL VISUAL WEIGHTS MATCH — bug is in forward path (dtype/preprocessing)")
else:
    print("\nVISUAL WEIGHT MISMATCH — bug is in weight loading")

# Also: test with explicit dtype casting
print("\n=== Test: explicit dtype ===")
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
if s.get("tools"): kwargs["tools"] = s["tools"]
inputs = proc.apply_chat_template(msgs, **kwargs)
pv = inputs["pixel_values"].to(device)
gt = inputs["image_grid_thw"].to(device)

print(f"pixel_values dtype from processor: {pv.dtype}")  # float32
print(f"HF model dtype: {hf.dtype}")  # bfloat16
print(f"GRASPO model dtype: {g.visual.merger.linear_fc1.weight.dtype}")  # bfloat16

# Test: cast pv to bfloat16 before visual
with torch.no_grad():
    hf_out = hf.model.visual(pv, grid_thw=gt).last_hidden_state
    g_out_orig = g.visual.forward(pv, grid_thw=gt).last_hidden_state

    # Try with bfloat16 input
    pv_bf16 = pv.to(torch.bfloat16)
    g_out_bf16 = g.visual.forward(pv_bf16, grid_thw=gt).last_hidden_state
    hf_out_bf16 = hf.model.visual(pv_bf16, grid_thw=gt).last_hidden_state

    print(f"\nHF out (float32 in):   norm={hf_out.float().norm().item():.2f}")
    print(f"G  out (float32 in):   norm={g_out_orig.float().norm().item():.2f}")
    print(f"G  out (bfloat16 in):  norm={g_out_bf16.float().norm().item():.2f}")
    print(f"HF out (bfloat16 in):  norm={hf_out_bf16.float().norm().item():.2f}")

    print(f"\nG(bf16) vs HF(bf16) maxdiff: {(g_out_bf16.float() - hf_out_bf16.float()).abs().max().item():.2f}")
    print(f"G(bf16) vs HF(f32)  maxdiff: {(g_out_bf16.float() - hf_out.float()).abs().max().item():.2f}")
