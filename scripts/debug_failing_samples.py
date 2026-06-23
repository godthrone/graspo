#!/usr/bin/env python3
"""Investigate failing samples: compare per-layer hidden states after visual fix."""
import json, torch
from pathlib import Path

device = torch.device("cuda:0")
torch.manual_seed(42)

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

eos_id = proc.tokenizer.eos_token_id

# Load sample 5 (the first failing one)
with open("/workspace/data/data/elam_graspo_train.jsonl") as f:
    samples = [json.loads(line) for line in f if line.strip()]

# Check if sample 5 and 6 have same content
s5 = samples[4]
s6 = samples[5]
print(f"Sample 5 first message: {str(s5['messages'][0].get('content',''))[:100]}")
print(f"Sample 6 first message: {str(s6['messages'][0].get('content',''))[:100]}")
print(f"Sample 5 has tools: {bool(s5.get('tools'))}, Sample 6: {bool(s6.get('tools'))}")

# Use sample 5
s = s5
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

print(f"\nInput: ids={ids.shape}, n_image_tokens={(ids == hf.config.image_token_id).sum().item()}")

# ---- Compare prefill hidden states per layer ----
print("\n=== Per-layer hidden state comparison (prefill) ===")
# Hook HF layers
hf_hidden = {}
def make_hf_hook(idx):
    def hook(module, input, output):
        hf_hidden[idx] = output[0].detach() if isinstance(output, tuple) else output.detach()
    return hook

# Hook GRASPO layers
g_hidden = {}
def make_g_hook(idx):
    def hook(module, input, output):
        g_hidden[idx] = output[0].detach() if isinstance(output, tuple) else output.detach()
    return hook

hf_hooks = []
g_hooks = []
for i, (hf_layer, g_layer) in enumerate(zip(hf.model.language_model.layers, g.layers)):
    hf_hooks.append(hf_layer.register_forward_hook(make_hf_hook(i)))
    g_hooks.append(g_layer.register_forward_hook(make_g_hook(i)))

with torch.no_grad():
    hf_out = hf.model(
        ids, attention_mask=attn,
        pixel_values=mm.get("pixel_values"), image_grid_thw=mm.get("image_grid_thw"),
        use_cache=False,
    )
    g_out = g(ids, attention_mask=attn, multimodal_inputs=mm, use_cache=False)

for h in hf_hooks:
    h.remove()
for h in g_hooks:
    h.remove()

# Compare
print(f"{'Layer':>6} {'maxdiff':>12} {'HF norm':>12} {'G norm':>12}")
for i in range(len(hf.model.language_model.layers)):
    if i in hf_hidden and i in g_hidden:
        diff = (hf_hidden[i].float() - g_hidden[i].float()).abs()
        print(f"  L{i:<4} {diff.max().item():>12.6f} {hf_hidden[i].float().norm():>12.4f} {g_hidden[i].float().norm():>12.4f}")

# Compare logits at step 31 (before the divergence at step 32)
print(f"\n=== Logit comparison at prefill output ===")
hf_logits = hf.lm_head(hf_out[0])
g_logits = g.lm_head(g_out[0] if isinstance(g_out, tuple) else g_out)

last_pos = attn.sum(dim=1).long() - 1
hf_last = hf_logits[0, last_pos[0], :].float()
g_last = g_logits[0, last_pos[0], :].float()
print(f"Logits maxdiff: {(hf_last - g_last).abs().max().item():.6f}")
print(f"HF top-5: {hf_last.topk(5).indices.tolist()} values={hf_last.topk(5).values.tolist()}")
print(f"G  top-5: {g_last.topk(5).indices.tolist()} values={g_last.topk(5).values.tolist()}")

# Compare argmax
hf_argmax = hf_last.argmax().item()
g_argmax = g_last.argmax().item()
print(f"HF argmax: {hf_argmax} G argmax: {g_argmax} {'MATCH' if hf_argmax == g_argmax else 'MISMATCH'}")

# Check if it's a near-tie
top2 = hf_last.topk(2)
print(f"HF top-2 margin: {top2.values[0].item() - top2.values[1].item():.6f}")
