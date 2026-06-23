"""Compare per-layer hidden states between GRASPO and HF."""
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

# Get HF hidden states by patching
hf_hidden_states = {}
def _hf_hook(name):
    def hook(module, input, output):
        if isinstance(output, tuple):
            hf_hidden_states[name] = output[0].detach()
        else:
            hf_hidden_states[name] = output.detach()
    return hook

# Register hooks on HF decoder layers
for i, layer in enumerate(hf.model.language_model.layers):
    layer.register_forward_hook(_hf_hook(f"L{i}"))

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
    lora_r=16, lora_alpha=32, lora_dropout=0.05,
    lora_targets={"language.full_attn.q_proj", "language.full_attn.v_proj",
                   "language.linear_attn.q_proj", "language.linear_attn.v_proj"},
    gradient_checkpointing=False, torch_dtype=torch.bfloat16, device=device,
).eval()

# Patch GRASPO layers to capture hidden states
g_hidden_states = {}
orig_forwards = {}
for i, layer in enumerate(g.layers):
    orig_fwd = layer.forward
    orig_forwards[i] = orig_fwd
    def _make_g_hook(idx):
        def g_hook(*args, **kwargs):
            result = orig_forwards[idx](*args, **kwargs)
            if isinstance(result, tuple):
                g_hidden_states[f"L{idx}"] = result[0].detach()
            else:
                g_hidden_states[f"L{idx}"] = result.detach()
            return result
        return g_hook
    layer.forward = _make_g_hook(i)

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
ids = inputs["input_ids"].to(device)
attn = inputs["attention_mask"].to(device)

mm = {}
for k in ("pixel_values", "image_grid_thw", "mm_token_type_ids"):
    v = inputs.get(k)
    if v is not None and len(v) > 0:
        mm[k] = v.to(device)
hf_kw = {k: v for k, v in mm.items() if v is not None}

# Run prefill on both
with torch.no_grad():
    hf_out = hf(input_ids=ids, attention_mask=attn, **hf_kw, use_cache=True)
    g_logits, g_pkv = g(input_ids=ids, attention_mask=attn, multimodal_inputs=mm, use_cache=True)

print("=== Per-layer hidden state comparison ===")
for i in range(32):
    hf_h = hf_hidden_states.get(f"L{i}")
    g_h = g_hidden_states.get(f"L{i}")
    if hf_h is None or g_h is None:
        print(f"  L{i:2d}: MISSING")
        continue
    lt = types[i] if types else "?"
    diff = (hf_h.float() - g_h.float().to(device)).abs()
    cos_sim = torch.nn.functional.cosine_similarity(
        hf_h.float().view(-1), g_h.float().to(device).view(-1), dim=0
    ).item()
    print(f"  L{i:2d}({lt:16s}): norm HF={hf_h.float().norm().item():.2f} G={g_h.float().norm().item():.2f} maxdiff={diff.max().item():.4f} cos_sim={cos_sim:.6f}")
