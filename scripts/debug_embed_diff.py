"""Compare embedding+visual outputs between GRASPO and HF."""
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

# Inject hooks to capture HF embedding output
hf_embed_out = {}
def _hf_hook(module, input, output):
    hf_embed_out["output"] = output.detach()

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

# Print input info
print(f"input_ids shape: {ids.shape}")
print(f"image_token_id from config: {hf.config.image_token_id}")
n_img_tokens = (ids == hf.config.image_token_id).sum().item()
print(f"Number of image placeholder tokens: {n_img_tokens}")

# Compare visual tower outputs directly
print("\n=== Visual tower comparison ===")
with torch.no_grad():
    hf_visual_out = hf.model.visual(mm["pixel_values"], grid_thw=mm["image_grid_thw"]).last_hidden_state
    print(f"HF visual output shape: {hf_visual_out.shape}, dtype={hf_visual_out.dtype}")

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

with torch.no_grad():
    g_visual_out = g.visual.forward(mm["pixel_values"], grid_thw=mm["image_grid_thw"]).last_hidden_state
    print(f"G visual output shape: {g_visual_out.shape}, dtype={g_visual_out.dtype}")
    vdiff = (hf_visual_out.float() - g_visual_out.float()).abs()
    print(f"Visual output maxdiff: {vdiff.max().item():.4f}")

# Compare embedding outputs (after image feature insertion)
print("\n=== Embedding output comparison ===")
# HF: hook the embed_tokens output then manually trace the visual insertion
hf_emb = hf.model.language_model.embed_tokens
with torch.no_grad():
    hf_text_emb = hf_emb(ids)
    print(f"HF text embed shape: {hf_text_emb.shape}")

g_text_emb = g.embed_tokens(ids)
print(f"G text embed shape: {g_text_emb.shape}")
emb_diff = (hf_text_emb.float() - g_text_emb.float()).abs()
print(f"Text embed maxdiff: {emb_diff.max().item():.6f}")

# Now check after visual insertion
with torch.no_grad():
    g_embed_out = g.embed_inputs(ids, multimodal_inputs=mm)
    print(f"G embed+visual output shape: {g_embed_out.shape}")

# HF: manually replicate the visual insertion
image_token_id = hf.config.image_token_id
image_mask = ids.eq(image_token_id).unsqueeze(-1).expand_as(hf_text_emb)
hf_embed_out_val = hf_text_emb.masked_scatter(image_mask, hf_visual_out.to(hf_text_emb.dtype))
print(f"HF embed+visual output shape: {hf_embed_out_val.shape}")

embed_vis_diff = (hf_embed_out_val.float() - g_embed_out.float()).abs()
print(f"Embed+visual maxdiff: {embed_vis_diff.max().item():.4f}")

# Also compare the visual features element-by-element
if n_img_tokens > 0:
    img_positions = (ids == image_token_id).nonzero(as_tuple=True)[1]
    print(f"\nImage positions sample: {img_positions[:5].tolist()}")
    for pos in img_positions[:3].tolist():
        hf_val = hf_embed_out_val[0, pos, :5].float().tolist()
        g_val = g_embed_out[0, pos, :5].float().tolist()
        print(f"  pos {pos}: HF={[f'{x:.4f}' for x in hf_val]} G={[f'{x:.4f}' for x in g_val]}")
