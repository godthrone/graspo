"""Compare GRASPO vs HF per-layer KV + RoPE + first-token logits."""
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
    lora_r=16, lora_alpha=32, lora_dropout=0.05,
    lora_targets={"language.full_attn.q_proj", "language.full_attn.v_proj",
                   "language.linear_attn.q_proj", "language.linear_attn.v_proj"},
    gradient_checkpointing=False, torch_dtype=torch.bfloat16, device=device,
).eval()

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

# Prefill both models
with torch.no_grad():
    hf_out = hf(input_ids=ids, attention_mask=attn, **hf_kw, use_cache=True)
    g_logits, g_pkv = g(input_ids=ids, attention_mask=attn, multimodal_inputs=mm, use_cache=True)

hf_pkv = hf_out.past_key_values
hf_logits = hf_out.logits

# Per-layer KV comparison
print("=== Per-layer attention KV diff ===")
for i in range(len(g_pkv)):
    hf_layer = hf_pkv.layers[i]
    has_kv = hasattr(hf_layer, "keys") and hasattr(hf_layer, "values")
    if not has_kv:
        continue

    hf_k = hf_layer.keys
    hf_v = hf_layer.values
    g_k = g_pkv[i][0]
    g_v = g_pkv[i][1]

    lt = types[i] if types else "?"
    hf_kn = hf_k.float().norm().item()
    g_kn = g_k.float().norm().item()
    hf_vn = hf_v.float().norm().item()
    g_vn = g_v.float().norm().item()
    kd = (hf_k.float() - g_k.float()).abs()
    vd = (hf_v.float() - g_v.float()).abs()

    print(f"  L{i:2d}({lt:16s}): k_norm HF={hf_kn:10.2f} G={g_kn:10.2f} k_maxdiff={kd.max().item():.2f} | v_norm HF={hf_vn:10.2f} G={g_vn:10.2f} v_maxdiff={vd.max().item():.2f}")

# Compare position IDs
print("\n=== Position IDs ===")
g_pos_ids = g.compute_multimodal_position_ids(
    input_ids=ids, attention_mask=attn, multimodal_inputs=mm,
    past_key_values=None, query_len=ids.shape[1],
)
print(f"GRASPO position_ids: shape={g_pos_ids.shape} ndim={g_pos_ids.ndim}")
print(f"  first cols (dim 0): {g_pos_ids[0, 0, :8].tolist()}")
print(f"  last cols  (dim 0): {g_pos_ids[0, 0, -8:].tolist()}")
print(f"  rope_deltas: {g.rope_deltas}")

# Compare first-token logits
print("\n=== First token logits ===")
actual_len = attn[0].sum().item() - 1
hf_first = hf_logits[0, actual_len]
g_first = g_logits[0, actual_len]
cos_sim = torch.nn.functional.cosine_similarity(hf_first.float(), g_first.float(), dim=0).item()
print(f"cos_sim: {cos_sim:.6f}")
print(f"HF top-10 tokens: {hf_first.topk(10).indices.tolist()}")
print(f"G  top-10 tokens: {g_first.topk(10).indices.tolist()}")
print(f"HF top-10 values: {[f'{v:.2f}' for v in hf_first.topk(10).values.tolist()]}")
print(f"G  top-10 values: {[f'{v:.2f}' for v in g_first.topk(10).values.tolist()]}")
