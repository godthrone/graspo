#!/usr/bin/env python3
"""extract_params.py — 从权威件 HTML + 档位配置抽取 54 档「主要训练参数」原始值。

用途：为 matrix-data.json 提供：
  - 档位身份与计划参数（来自权威件 HTML）：tier/model/algo/mode/backend/cards/parallelism/
    lora_rank/seq_len_cap/micro_batch/grad_accum/max_epochs/subset_size/planned_steps/data_source；
  - 训练配置（来自 samples/configs/matrix54/T0NN.yaml）：lr/seed/plen/精度/GC/LoRA r·alpha·dropout·target/
    数据路径/epoch/save_steps/并行尺寸/关键开关（pretrain / reward / distill）。

边界：只读；只写本工位 evidence/ 下的 JSON。
用法：
  <repo-root>/.venv/bin/python extract_params.py \
      --html <repo-root>/.local/repo-moved-out/docs/capability-matrix-20261006-r3.html \
      --config-dir <repo-root>/samples/configs/matrix54 \
      --out evidence/params-raw.json
"""
import argparse, html, json, re, hashlib, sys
from pathlib import Path

SIMPLE = ["tier", "model", "algo", "mode", "backend", "cards", "parallelism",
          "lora_rank", "seq_len_cap", "micro_batch", "grad_accum", "max_epochs",
          "subset_size", "planned_steps", "data_source"]

TD_RE = re.compile(r'<td\b[^>]*\bdata-col="([^"]+)"[^>]*>(.*?)</td>', re.S)
TR_RE = re.compile(r'<tr\b[^>]*>(.*?)</tr>', re.S)
TAG_RE = re.compile(r'<[^>]+>')

# yaml 里要抽的全部键（section, yaml 键 -> 扁平输出名）。取不到记 None（⇒ 数据里写 '—'，不编造）。
YAML_KEYS = [
    ("__top__", "train_method", "train_method"),
    ("__top__", "backend", "backend"),
    ("__top__", "tuner_type", "tuner_type"),
    ("model", "model_path", "model_path"),
    ("model", "torch_dtype", "torch_dtype"),
    ("model", "gradient_checkpointing", "gradient_checkpointing"),
    ("data", "train_path", "train_path"),
    ("data", "max_prompt_length", "max_prompt_length"),
    ("training", "seed", "seed"),
    ("training", "max_epochs", "max_epochs_cfg"),
    ("training", "learning_rate", "learning_rate"),
    ("training", "gradient_accumulation_micro_batches", "grad_accum_cfg"),
    ("training", "save_steps", "save_steps"),
    ("lora", "r", "lora_r"),
    ("lora", "alpha", "lora_alpha"),
    ("lora", "dropout", "lora_dropout"),
    ("lora", "target_preset", "lora_target"),
    ("msswift", "nproc_per_node", "msswift_nproc"),
    ("msswift", "nnodes", "msswift_nnodes"),
    ("msswift", "deepspeed", "msswift_deepspeed"),
    ("msswift", "per_device_train_batch_size", "msswift_per_device_bs"),
    ("msswift", "sequence_parallel_size", "msswift_seq_parallel"),
    ("msswift", "use_vllm", "msswift_use_vllm"),
    ("native", "tp_size", "native_tp"),
    ("native", "dp_size", "native_dp"),
    ("native", "pp_size", "native_pp"),
    ("native", "micro_batch_size", "native_micro_bs"),
    ("native", "allow_unverified_pp_rollout", "native_allow_unverified_pp_rollout"),
    ("pretrain", "loss_scale", "pretrain_loss_scale"),
    ("pretrain", "use_chat_template", "pretrain_use_chat_template"),
    ("reward", "kind", "reward_kind"),
    ("distill", "teacher_model_path", "distill_teacher"),
    ("distill", "offload_teacher_model", "distill_offload_teacher"),
]


def strip_tags(s):
    return html.unescape(TAG_RE.sub("", s)).strip()


def parse_html(path):
    raw = Path(path).read_text(encoding="utf-8")
    matrix = raw.split('<table class="matrix">', 1)[1].split("</table>", 1)[0]
    body = matrix.split("<tbody>", 1)[1].split("</tbody>", 1)[0]
    rows = []
    for tr in TR_RE.findall(body):
        cells = {}
        for key, val in TD_RE.findall(tr):
            cells[key] = strip_tags(val)
        if cells:
            rows.append(cells)
    return raw, rows


def parse_yaml_lite(path):
    """零依赖按 section 抽取主要训练参数字段；缺失记 None（不编造）。"""
    text = Path(path).read_text(encoding="utf-8") if Path(path).exists() else ""
    flat = {out: None for _, _, out in YAML_KEYS}
    sec = "__top__"
    for line in text.splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        m = re.match(r'^([A-Za-z_][A-Za-z0-9_]*):\s*(.*)$', line)
        if m:                       # 顶层 section（或顶层键 train_method / backend / tuner_type）
            sec = m.group(1)
            key, val = m.group(1), m.group(2).strip()
            for s, k, out in YAML_KEYS:
                if s == "__top__" and k == key and val != "":
                    flat[out] = val
            continue
        m2 = re.match(r'^\s+([A-Za-z_][A-Za-z0-9_]*):\s*(.*)$', line)
        if m2:
            key, val = m2.group(1), m2.group(2).strip()
            for s, k, out in YAML_KEYS:
                if s == sec and k == key and val != "":
                    flat[out] = val
    return flat


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--html", required=True)
    ap.add_argument("--config-dir", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    raw, rows = parse_html(a.html)
    src_sha = hashlib.sha256(raw.encode("utf-8")).hexdigest()
    config_dir = Path(a.config_dir)

    tiers = []
    for cells in rows:
        tier = cells.get("tier", "")
        if not re.fullmatch(r"T\d{3}", tier):
            continue
        rec = {"tier": tier, "from_html": {k: cells.get(k, "") for k in SIMPLE}}
        cfg = config_dir / f"{tier}.yaml"
        rec["config_path"] = str(cfg)
        rec["config_present"] = cfg.exists()
        rec["from_config"] = parse_yaml_lite(cfg)
        tiers.append(rec)

    doc = {
        "schema": "params-raw/v2",
        "authoritative_html": str(Path(a.html).resolve()),
        "authoritative_html_sha256": src_sha,
        "config_dir": str(config_dir.resolve()),
        "row_count": len(tiers),
        "config_present_count": sum(1 for t in tiers if t["config_present"]),
        "yaml_fields": sorted({out for _, _, out in YAML_KEYS}),
        "tiers": tiers,
    }
    outp = Path(a.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(doc, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"rows={len(tiers)} config_present={doc['config_present_count']} src_sha256={src_sha}")
    print(f"wrote {outp}")


if __name__ == "__main__":
    sys.exit(main())
