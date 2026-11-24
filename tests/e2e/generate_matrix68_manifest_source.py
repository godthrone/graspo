#!/usr/bin/env python3
"""68 档（M001–M068）运行清单生成器 —— 从**数据件 + 68 份 yaml + 训练子集**现读，⛔ 不手抄常量。

为什么有这件（工作包 B / 2026-10-22 第一轮基线）：
  定稿文档与数据件已切到 **68 档 M001–M068**（native 36 / ms-swift 32），而运行清单仍是
  旧 **54 档 T 档号**体系（`tests/e2e/matrix54_manifest_source.json`、`matrix54_manifest.json`）。
  runner 按清单 `tier_id` 取 `config` / `subset_size`（`tests/e2e/run_matrix54.sh`）⇒
  **68 档一档也起不来**。本文件产出 M 档的两份清单（源 / 被消费），schema 逐键对齐 T 版。

与 T 版生成器（`tests/e2e/generate_matrix_manifest_source.py`）的关系：
  · **它是只读参考**（本包不改它）；本文件是**新增的 M 档件**，不引用 T 档号、不写 T 档常量；
  · 本文件**不重算步数**：`per_epoch_steps` / `derived_steps` / `planned_steps` 一律取数据件
    该行现值，`max_epochs` / `save_steps` 一律取**在盘 yaml**；两者的乘积关系当场断言
    （数据件派生 vs 配置现读，任一处漂移即 rc=1）。

三份输入（全部**现读**，⛔ 本文件不含任何档号/卡集/阈值/步数常量）：
  ① `--data`（默认 `docs/capability-matrix.data.json`）= 唯一真相源：档号集合、算法、形态、
     `cards`、并行维度、`subset_size`、步数三件、`lr`、`applicable`、四条判据与阈值；
  ② `--config-dir`（默认 `samples/configs/matrix68`）= 68 份 yaml（`max_epochs` / `save_steps` /
     `model_path` / `data.train_path` / `launch.nproc_per_node` / `native.*` / `msswift.*` / `lora.*`）；
  ③ `--subset-root`（默认 `subsets-v2`）= 68 份训练子集（**行数必须 == `subset_size`**）。

一致性断言（任一条不成立 ⇒ **rc=1 并打印档号**，⛔ 不静默跳过）：
  A1 档数 == `--expect-count`（默认 68）；A2 档号集合 == 数据件 `applicable` 行集合；
  A3 每档 `config` 文件存在；A4 每档训练子集存在且**行数 == `subset_size`**；
  A5 `len(卡集) == cards`（卡号唯一、⊆ 0–7、`cards > 0`）；
  A6 yaml 与数据件**身份同源**：`launch.nproc_per_node`/`msswift.nproc_per_node` == `cards`、
    `tp×pp×dp == cards`、`model`/`backend`/`tuner_type`/`train_method`/`lr`/`model_path` 逐项一致；
  A7 步数门槛：`max_epochs`（yaml）× `per_epoch_steps`（数据件）== `derived_steps`（数据件）；
  A8 yaml 的 `data.train_path` == 数据件 `train_path` 解析出的容器路径（同一份文件）；
  A9 产物**不得出现 `deepspeed:`**（DeepSpeed 已禁用；一旦回填即静默下发 ⇒ fail-closed）。

幂等（`--check-determinism`）：同输入两跑**逐字节一致**（产物内不含时间戳/绝对宿主路径）。

用法：
    .venv/bin/python tests/e2e/generate_matrix68_manifest_source.py            # 生成两件
    .venv/bin/python tests/e2e/generate_matrix68_manifest_source.py --check-determinism
    .venv/bin/python tests/e2e/generate_matrix68_manifest_source.py --print-tier M001   # 单行 JSON（供 runner 现读）
    .venv/bin/python tests/e2e/generate_matrix68_manifest_source.py --list-tiers
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import sys
from pathlib import Path

FAIL = 1

ROOT = Path(__file__).resolve().parents[2]

#: 卡池（远端机：GPU 0–7 先到先得、不预留、单作业不限卡数）。**卡号集合**是环境/裁决事实，
#: 不是档位常量；档位数一律从数据件现读。
GPU_POOL = tuple(range(8))

#: 后端名映射（数据件用展示名，yaml 用枚举名）。只是**名字映射**，不含任何档位信息。
BACKEND_TO_YAML = {"ms-swift": "msswift", "native": "native"}
YAML_TO_BACKEND = {v: k for k, v in BACKEND_TO_YAML.items()}

#: 数据件 `train_method` ⇒ 判据② 的冻结口径键（与 `meta.criteria_caliber.threshold.by_method` 同名）。
#: 只是**键名字符串**，⛔ 不含阈值数值（阈值一律从数据件现读）。
CRITERIA_METHOD_KEY = {"sft": "sft", "graspo": "graspo", "cpt": "cpt", "opd": "opd"}


# ── 小工具 ───────────────────────────────────────────────────────────────────
def die(msg: str, tiers: list[str] | None = None) -> "NoReturn":       # noqa: F821
    print(f"FAIL: {msg}", file=sys.stderr)
    if tiers:
        print(f"FAIL: 涉及档号 {tiers}", file=sys.stderr)
    raise SystemExit(FAIL)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def fingerprint_files(paths: list[Path]) -> str:
    """目录总指纹（逐文件 relpath + sha256 排序后聚合；与 `scripts/env_snapshot.py` 同手法）。"""
    items = []
    for p in sorted(paths):
        items.append(f"{p.name}\t{sha256_file(p)}")
    return sha256_bytes("\n".join(items).encode())


def load_yaml(p: Path) -> dict:
    try:
        import yaml
    except ImportError:                                   # pragma: no cover
        die("缺少 PyYAML：本脚本必须**现读** yaml 才能取 max_epochs/save_steps（⛔ 不手抄常量）")
    return yaml.safe_load(p.read_text(encoding="utf-8")) or {}


def line_count(p: Path) -> int:
    with p.open("rb") as fh:
        return sum(1 for _ in fh)


def as_int(v, what: str, tier: str) -> int:
    try:
        return int(str(v).strip())
    except (TypeError, ValueError):
        die(f"{tier}: {what} 不是整数（实际 {v!r}）", [tier])


def rel(p: Path) -> str:
    """仓库相对路径（不在仓内时用绝对路径 —— /tmp 夹具自测用，⛔ 不影响正式产物）。"""
    try:
        return str(Path(p).resolve().relative_to(ROOT))
    except ValueError:
        return str(Path(p))


def same_number(a, b) -> bool:
    """数值相等（两处都能解析成数时按数值比，否则按去空白字符串比）。"""
    try:
        return abs(float(str(a)) - float(str(b))) <= 1e-12 * max(1.0, abs(float(str(b))))
    except (TypeError, ValueError):
        return str(a).strip() == str(b).strip()


def require(cond: bool, msg: str, tiers: list[str]) -> None:
    if not cond:
        die(msg, tiers)


# ── 数据件读取 ───────────────────────────────────────────────────────────────
def load_data(path: Path) -> tuple[dict, list[dict], dict]:
    """返回 (整个数据件, 适用行按档号排序, meta.criteria_caliber)。"""
    if not path.is_file():
        die(f"数据件不存在：{path}")
    doc = json.loads(path.read_text(encoding="utf-8"))
    rows = doc.get("rows") or []
    if not isinstance(rows, list) or not rows:
        die(f"数据件没有 rows：{path}")
    applicable = sorted([r for r in rows if r.get("applicable") is True], key=lambda r: r["tier"])
    if not applicable:
        die(f"数据件里没有 applicable=true 的行：{path}")
    caliber = ((doc.get("meta") or {}).get("criteria_caliber")) or {}
    if not caliber:
        die("数据件缺 meta.criteria_caliber（四条判据口径的唯一真相源）")
    return doc, applicable, caliber


def threshold_block(caliber: dict) -> dict:
    th = caliber.get("threshold") or {}
    by_method = th.get("by_method") or {}
    if not by_method:
        die("数据件缺 meta.criteria_caliber.threshold.by_method（判据② 逐算法通过线）")
    return th


def per_method_pass_line(th: dict, method: str) -> tuple[str, str]:
    """判据② 该算法的通过线**展示文本**（⛔ 不解析成数值、不另存第二份口径）。

    数值判定归回填器（`tools/matrix/backfill_matrix.py::frozen_map_from_caliber`）；
    本清单只带"口径文本 + 来源指针"，避免制造第二份阈值副本。
    """
    entry = (th.get("by_method") or {}).get(method)
    if not isinstance(entry, dict):
        die(f"数据件 threshold.by_method 缺 {method!r} 的条目")
    crit = str(entry.get("criterion") or "")
    val = entry.get("value")
    teacher = entry.get("teacher")
    text = f"{crit} {val}" if crit else json.dumps(entry, ensure_ascii=False)
    if teacher:
        text += f"；{teacher}"
    return text, crit


def criteria_summary(caliber: dict) -> list[dict]:
    """四条判据（标题 + 正文）——**只出现一次**（顶层），逐档只引用 id。"""
    pcs = caliber.get("pass_criteria") or []
    if len(pcs) != 4:
        die(f"数据件 criteria_caliber.pass_criteria 应为四条，实际 {len(pcs)} 条")
    out = []
    for i, c in enumerate(pcs, 1):
        out.append({"id": f"C{i}", "title": c.get("title") or "", "text": c.get("text") or "",
                    "from_threshold": bool(c.get("from_threshold"))})
    return out


# ── 单档渲染 ─────────────────────────────────────────────────────────────────
def candidate_cards(index: int, cards: int) -> list[int]:
    """确定性**候选窗口**（仅用于 schema 对齐与审计；真实择卡由 runner 起跑前动态取）。

    ⛔ 语义边界：本清单**不预留**卡（远端机卡池先到先得、不抢占他人）。runner 读的是 `cards`
    （张数），起跑前从**逐卡实测空闲**的卡里现取同样多张并断言 `len(picked) == cards`。
    """
    start = (index * max(cards, 1)) % len(GPU_POOL)
    return [GPU_POOL[(start + k) % len(GPU_POOL)] for k in range(cards)]


def build_tier(index: int, row: dict, *, cfg_dir: Path, subset_root: Path,
               th: dict, gate_default: int) -> dict:
    tier = str(row["tier"])
    errs_tier = [tier]

    # ── ① 数据件侧（唯一真相源）─────────────────────────────────────────────
    cards = as_int(row.get("cards"), "cards", tier)
    require(cards > 0, f"{tier}: cards 必须 > 0（实际 {cards}）", errs_tier)
    model = str(row.get("model") or "")
    algo = str(row.get("algo") or "")
    mode = str(row.get("mode") or "")
    backend = str(row.get("backend") or "")
    tuner_type = str(row.get("tuner_type") or "")
    train_method = str(row.get("train_method") or "")
    lr_declared = str(row.get("lr") or "")
    subset_size = as_int(row.get("subset_size"), "subset_size", tier)
    per_epoch_steps = as_int(row.get("per_epoch_steps"), "per_epoch_steps", tier)
    derived_steps = as_int(row.get("derived_steps"), "derived_steps", tier)
    planned_steps = as_int(row.get("planned_steps"), "planned_steps", tier)
    tp, pp, dp = (as_int(row.get(k), k, tier) for k in ("tp", "pp", "dp"))
    train_rel = str(row.get("train_path") or "")

    require(backend in BACKEND_TO_YAML, f"{tier}: backend={backend!r} 不认识（应为 "
                                        f"{sorted(BACKEND_TO_YAML)}）", errs_tier)
    require(train_method in CRITERIA_METHOD_KEY,
            f"{tier}: train_method={train_method!r} 不是 {sorted(CRITERIA_METHOD_KEY)} 之一", errs_tier)
    require(train_rel.endswith(f"/{tier}.jsonl"),
            f"{tier}: 数据件 train_path={train_rel!r} 的文件名与档号不一致", errs_tier)
    require(tp * pp * dp == cards,
            f"{tier}: tp×pp×dp={tp}×{pp}×{dp}={tp * pp * dp} ≠ cards={cards}（并行维度与卡数不同源）",
            errs_tier)

    # ── ② 训练子集（A4：存在 + 行数 == subset_size）────────────────────────
    subset_path = (subset_root / f"{tier}.jsonl")
    require(subset_path.is_file(),
            f"{tier}: 训练子集不存在：{subset_path}（依赖工作包 A 产出的 subsets-v2/M###.jsonl）",
            errs_tier)
    lines = line_count(subset_path)
    require(lines == subset_size,
            f"{tier}: 训练子集行数 {lines} ≠ 数据件 subset_size {subset_size}（{subset_path}）", errs_tier)

    # ── ③ 档配置（A3/A6/A7/A8/A9：全部**现读** yaml）──────────────────────
    cfg_path = cfg_dir / f"{tier}.yaml"
    require(cfg_path.is_file(),
            f"{tier}: 档配置不存在：{cfg_path}（依赖工作包 A 产出的 samples/configs/matrix68/M###.yaml）",
            errs_tier)
    cfg = load_yaml(cfg_path)
    training = cfg.get("training") or {}
    data_sec = cfg.get("data") or {}
    model_sec = cfg.get("model") or {}

    max_epochs = as_int(training.get("max_epochs"), "training.max_epochs", tier)
    save_steps = as_int(training.get("save_steps"), "training.save_steps", tier)
    yaml_lr = str(training.get("learning_rate") or "")
    model_path = str(model_sec.get("model_path") or "")
    yaml_train_path = str(data_sec.get("train_path") or "")
    yaml_backend = str(cfg.get("backend") or "")
    yaml_tuner = str(cfg.get("tuner_type") or "")
    yaml_train_method = str(cfg.get("train_method") or "")

    # A6 身份同源
    require(yaml_backend == BACKEND_TO_YAML[backend],
            f"{tier}: yaml backend={yaml_backend!r} 与数据件 backend={backend!r} 不同源", errs_tier)
    require(yaml_tuner == tuner_type,
            f"{tier}: yaml tuner_type={yaml_tuner!r} ≠ 数据件 {tuner_type!r}", errs_tier)
    require(yaml_train_method == train_method,
            f"{tier}: yaml train_method={yaml_train_method!r} ≠ 数据件 {train_method!r}", errs_tier)
    # ★ 学习率按**数值**比（yaml 把 `2.0e-05` 读成浮点后 str() 是 `2e-05`；比字符串会误判）
    require(same_number(yaml_lr, lr_declared),
            f"{tier}: yaml learning_rate={yaml_lr!r} ≠ 数据件 lr={lr_declared!r}", errs_tier)
    require(Path(model_path).name == str(row.get("model_path") or ""),
            f"{tier}: yaml model_path={model_path!r} 的目录名 ≠ 数据件 model_path="
            f"{row.get('model_path')!r}", errs_tier)

    nproc = cfg_nproc(cfg, backend, tier)
    require(nproc == cards,
            f"{tier}: yaml nproc_per_node={nproc} ≠ cards={cards}（§12：卡数与进程数不得各改一半）",
            errs_tier)
    # ★ 并行维度只对 **native** 要求在 yaml 里落 tp/pp/dp；ms-swift 的并行由
    #   `msswift.nproc_per_node`（+ FSDP2 分片）承载，其 tp/pp 恒 1、dp 恒 == 卡数
    #   （数据件该行 `parallel_scheme` 自述"tp/pp 语义由 FSDP2 分片承载（Megatron 未接线）"）。
    if backend == "native":
        yaml_dims = (as_int((cfg.get("native") or {}).get("tp_size", 1), "native.tp_size", tier),
                     as_int((cfg.get("native") or {}).get("pp_size", 1), "native.pp_size", tier),
                     as_int((cfg.get("native") or {}).get("dp_size", 1), "native.dp_size", tier))
        require(yaml_dims == (tp, pp, dp),
                f"{tier}: yaml 并行维度 {yaml_dims} ≠ 数据件 (tp,pp,dp)={(tp, pp, dp)}", errs_tier)
    else:
        require((tp, pp) == (1, 1) and dp == cards,
                f"{tier}: ms-swift 档的数据件并行应为 tp=1/pp=1/dp==cards，实际 "
                f"tp={tp}/pp={pp}/dp={dp}/cards={cards}", errs_tier)
        require("native" not in cfg,
                f"{tier}: ms-swift 档的 yaml 不得带 `native:` 段（并行语义由 msswift 承载）", errs_tier)

    # A7 步数门槛：max_epochs(yaml) × per_epoch_steps(数据件) == derived_steps(数据件)
    recomputed = max_epochs * per_epoch_steps
    require(recomputed == derived_steps,
            f"{tier}: 步数不同源 —— max_epochs(yaml)={max_epochs} × per_epoch_steps(数据件)="
            f"{per_epoch_steps} = {recomputed} ≠ derived_steps(数据件)={derived_steps}", errs_tier)
    require(save_steps == derived_steps + 1,
            f"{tier}: yaml save_steps={save_steps} 与 derived_steps+1={derived_steps + 1} 不同源"
            "（口径=每档只落一份检查点 ⇒ save_steps 取计划总步数 +1）", errs_tier)

    # A8 train_path 同源
    require(yaml_train_path != "", f"{tier}: yaml 缺 data.train_path", errs_tier)
    require(Path(yaml_train_path).name == f"{tier}.jsonl",
            f"{tier}: yaml data.train_path={yaml_train_path!r} 的文件名 ≠ {tier}.jsonl", errs_tier)
    require(Path(yaml_train_path).parent.name == subset_root.name,
            f"{tier}: yaml data.train_path={yaml_train_path!r} 的父目录 ≠ {subset_root.name}"
            "（与数据件 train_path 的落点不同源）", errs_tier)

    # A9 DeepSpeed 必须缺席
    require("deepspeed" not in cfg,
            f"{tier}: yaml 出现 `deepspeed:` —— DeepSpeed 已禁用（2026-09 起全部档走 native/FSDP2），"
            "一旦下发即静默改配方 ⇒ fail-closed", errs_tier)

    # ── ④ 卡集（A5）────────────────────────────────────────────────────────
    gpus = candidate_cards(index, cards)
    require(len(gpus) == cards, f"{tier}: len(卡集)={len(gpus)} ≠ cards={cards}", errs_tier)
    require(len(set(gpus)) == len(gpus), f"{tier}: 卡集有重复卡号 {gpus}", errs_tier)
    require(all(g in GPU_POOL for g in gpus), f"{tier}: 卡集 {gpus} 越出卡池 {list(GPU_POOL)}", errs_tier)

    # ── ⑤ 判据口径（从数据件现读）─────────────────────────────────────────
    method_key = CRITERIA_METHOD_KEY[train_method]
    pass_line_text, pass_line_crit = per_method_pass_line(th, method_key)
    gate_threshold = step_gate_threshold(row, gate_default)
    reachable = derived_steps
    gate_applicable = reachable >= gate_threshold

    lora = None
    if tuner_type == "lora":
        lora_sec = cfg.get("lora") or {}
        require(bool(lora_sec), f"{tier}: tuner_type=lora 但 yaml 无 lora 段", errs_tier)
        lora = {k: lora_sec.get(k) for k in ("r", "alpha", "dropout", "target_modules")}

    tier_doc = collections.OrderedDict()
    tier_doc["tier_id"] = tier
    tier_doc["model"] = model
    tier_doc["model_name"] = Path(model_path).name
    tier_doc["model_path"] = model_path
    tier_doc["algorithm"] = algo
    tier_doc["train_method"] = train_method
    tier_doc["mode"] = mode
    tier_doc["tuner_type"] = tuner_type
    tier_doc["backend"] = backend
    tier_doc["cards"] = cards
    tier_doc["gpus"] = gpus
    tier_doc["gpus_semantics"] = (
        "确定性**候选窗口**（仅用于 schema 对齐与审计）：远端机卡池 GPU0–7 先到先得、不预留、"
        "不抢占他人 ⇒ 真实择卡由 runner 起跑前按 §12/§13 从**逐卡实测空闲**现取，"
        "只读本档 cards（张数）并断言 len(picked)==cards。⛔ 本字段不为任何卡做预留。"
    )
    tier_doc["status"] = "ready"
    tier_doc["model_host_root_env_var"] = "GRASPO_MODELS_HOST_ROOT"
    tier_doc["model_host_dir_env_var"] = ("GRASPO_27B_HOST_DIR" if model == "27B"
                                          else "GRASPO_9B_HOST_DIR")
    tier_doc["model_host_dir_default"] = f"<GRASPO_MODELS_HOST_ROOT>/{Path(model_path).name}"
    tier_doc["applicable"] = True
    tier_doc["config"] = rel(cfg_path)
    tier_doc["data"] = collections.OrderedDict((
        ("train_path", yaml_train_path),                      # 容器内路径（= yaml data.train_path）
        ("subset_size", subset_size),
        ("host_relative_path", rel(subset_path)),
        ("subset_sha256", sha256_file(subset_path)),
        ("subset_lines", lines),
        # ★ 与 T 版清单同名键（`data.full_train_jsonl` / `data.caveat`）：M 档的**训练源**就是
        #   该档自己的 100 行子集（fit 口径），全量训练集只作口径说明 ⇒ 路径由容器根派生。
        ("full_train_jsonl", f"{container_data_root_of(yaml_train_path)}/data/train.jsonl"),
        ("caveat", "数据集级图像层重叠口径见顶层 data.dataset_caveat（本档不另抄）"),
    ))
    tier_doc["parallel"] = str(row.get("parallel") or "")
    tier_doc["parallel_dims"] = str(row.get("parallel_dims") or "")
    tier_doc["parallel_components"] = collections.OrderedDict(
        (("nproc_per_node", nproc), ("tp", tp), ("pp", pp), ("dp", dp)))
    tier_doc["lr"] = lr_declared
    tier_doc["max_epochs"] = max_epochs
    tier_doc["checkpoint_save_steps"] = save_steps
    tier_doc["expected_optimizer_steps"] = derived_steps        # 计划总步数（判据① 的分母）
    tier_doc["expected_optimizer_steps_per_epoch"] = per_epoch_steps
    tier_doc["expected_optimizer_steps_reachable"] = reachable
    tier_doc["planned_steps"] = planned_steps
    tier_doc["step_formula"] = (
        f"总训练步数 = max_epochs(yaml, {max_epochs}) × 每 epoch 步数(数据件, {per_epoch_steps}) "
        f"= {derived_steps}；每 epoch 步数 = ⌈训练样本数({subset_size}) ÷ 总卡数({cards})⌉ "
        f"= {per_epoch_steps}（★ 两处独立派生，A7 当场断言相等）"
    )
    tier_doc["key_semantics"] = collections.OrderedDict((
        ("expected_optimizer_steps", "判据① 的分母（计划总训练步数），= max_epochs × 每 epoch 步数"),
        ("expected_optimizer_steps_per_epoch", "每 epoch 的计划优化步数 = ⌈subset_size ÷ cards⌉"),
        ("planned_steps", "数据件该行自述的计划步数（与 derived 同一口径，供审计对照）"),
        ("checkpoint_save_steps", "yaml 的 training.save_steps（= 计划总步数 + 1，每档只落一份）"),
        ("gpus", "确定性候选窗口（仅为 schema 对齐/审计）；真实择卡由 runner 起跑前动态取"),
    ))
    tier_doc["step_gate"] = collections.OrderedDict((
        ("threshold", gate_threshold),
        ("applicable", gate_applicable),
        ("reachable", reachable),
        ("basis", f"reachable={reachable} ≥ 门槛 {gate_threshold} ⇒ 门槛照常判"
                  if gate_applicable else
                  f"reachable={reachable} < 门槛 {gate_threshold} ⇒ ⚠ 口径不可测（⛔ 不是 ❌ 失败）"),
    ))
    tier_doc["acceptance"] = collections.OrderedDict((
        ("criteria_ref", "见顶层 criteria（四条判据正文的唯一副本）"),
        ("criteria_ids", [f"C{i}" for i in range(1, 5)]),
        ("method_key", method_key),
        ("pass_line", pass_line_text),
        ("pass_line_criterion", pass_line_crit),
        ("threshold_source", "docs/capability-matrix.data.json → "
                             "meta.criteria_caliber.threshold.by_method"),
        ("all_four_must_hold", True),
    ))
    tier_doc["feasibility"] = collections.OrderedDict((
        ("verdict", "unmeasured"),
        ("measured", False),
        ("note", "本机未上机：显存/可行性以**在盘配置**声明为准，实测读数待跑批回填"),
        ("recipe", declared_recipe(cfg, backend)),
    ))
    if lora is not None:
        tier_doc["lora"] = lora
    # 后端段原样带上（gen_eval 的 §10 判据、runner 的 nproc 校验都从这里取，⛔ 不各自解析 yaml）
    for sec in ("native", "msswift"):
        if isinstance(cfg.get(sec), dict):
            tier_doc[sec] = cfg[sec]
    tier_doc["switches"] = str(row.get("switches") or "")
    tier_doc["deepspeed"] = "—"
    return tier_doc


def cfg_nproc(cfg: dict, backend: str, tier: str) -> int:
    """进程数：ms-swift 看 `msswift.nproc_per_node`，native 看 `launch.nproc_per_node`。"""
    if backend == "ms-swift":
        sec = cfg.get("msswift") or {}
        if "nproc_per_node" in sec:
            return as_int(sec["nproc_per_node"], "msswift.nproc_per_node", tier)
    sec = cfg.get("launch") or {}
    if "nproc_per_node" in sec:
        return as_int(sec["nproc_per_node"], "launch.nproc_per_node", tier)
    die(f"{tier}: yaml 里既无 msswift.nproc_per_node 也无 launch.nproc_per_node ⇒ 无法断言卡数同源",
        [tier])


def declared_recipe(cfg: dict, backend: str) -> dict:
    """把配置里**已下发**的开关登记回清单（供 `gen_configs --check-declared` 逐键对齐）。

    ★ 只登记"确实写在 yaml 里"的键，⛔ 不预填任何默认值 —— 清单因此不会比配置更乐观。
    """
    if backend == "native":
        nat = cfg.get("native") or {}
        keys = ("offload_optimizer_state", "sequence_parallel", "tp_size", "pp_size", "dp_size",
                "micro_batch_size")
        return {"native": {k: nat[k] for k in keys if k in nat}}
    msw = cfg.get("msswift") or {}
    keys = ("nproc_per_node", "nnodes", "per_device_train_batch_size", "sequence_parallel_size",
            "use_vllm", "fsdp")
    return {"msswift": {k: msw[k] for k in keys if k in msw}}


#: 步数门槛（"正式记录的硬前提"：至少 N 个优化步）。★ 该数值**不在**数据件里
#: （数据件只落 `derived_steps` / `per_epoch_steps` / `planned_steps`）⇒ 本文件是它的
#: **唯一一处**声明，并可由 `--step-gate-threshold` 覆盖；⛔ 不散落到 runner / 清单 / 文档副本。
#: 口径见报告 §1「正式记录的硬前提」（min_epochs=1 / min_optimizer_steps=5 / min_train_samples=100）。
DEFAULT_STEP_GATE_THRESHOLD = 5


def step_gate_threshold(row: dict, default: int) -> int:
    """步数门槛：优先取数据件该行显式值（若将来落进数据件），否则用唯一默认值。"""
    for key in ("min_optimizer_steps", "step_gate_threshold"):
        if key in row and str(row[key]).strip() not in ("", "—"):
            return int(str(row[key]).strip())
    return default


# ── 顶层文档 ─────────────────────────────────────────────────────────────────
def build_manifest(*, data_path: Path, cfg_dir: Path, subset_root: Path,
                   expect_count: int, gate_threshold: int = DEFAULT_STEP_GATE_THRESHOLD) -> tuple[dict, dict]:
    doc, applicable, caliber = load_data(data_path)
    criteria = criteria_summary(caliber)
    th = threshold_block(caliber)

    tiers = [build_tier(i, row, cfg_dir=cfg_dir, subset_root=subset_root, th=th,
                        gate_default=gate_threshold)
             for i, row in enumerate(applicable)]

    # ── A1/A2：档数与档号集合 ────────────────────────────────────────────────
    tier_ids = [t["tier_id"] for t in tiers]
    require(len(tier_ids) == expect_count,
            f"档数 {len(tier_ids)} ≠ 标称 {expect_count}", tier_ids)
    require(len(set(tier_ids)) == len(tier_ids), "档号有重复", tier_ids)
    missing = sorted({r["tier"] for r in doc["rows"] if r.get("applicable") is True}
                     - set(tier_ids))
    require(not missing, f"数据件 applicable 行未进清单：{missing}", missing)

    data_root = container_data_root(tiers[0]["data"]["train_path"])
    for t in tiers:                                    # 全档同一数据根（否则挂载面会分叉）
        require(container_data_root(t["data"]["train_path"]) == data_root,
                f"{t['tier_id']}: 数据根 {container_data_root(t['data']['train_path'])!r} "
                f"与首档 {data_root!r} 不同源", [t["tier_id"]])

    def count(key: str) -> dict:
        return dict(sorted(collections.Counter(str(t.get(key)) for t in tiers).items()))

    counts = collections.OrderedDict((
        ("total", len(tiers)),
        ("by_algorithm", count("algorithm")),
        ("by_cards", dict(sorted(collections.Counter(t["cards"] for t in tiers).items()))),
        ("by_backend", count("backend")),
        ("by_mode", count("mode")),
        ("by_tuner_type", count("tuner_type")),
        ("by_model", count("model")),
        ("by_status", count("status")),
        ("by_applicability", {"applicable": len(tiers), "not_applicable": 0}),
        ("by_subset_size", dict(sorted(collections.Counter(t["data"]["subset_size"]
                                                           for t in tiers).items()))),
    ))

    cfg_files = [cfg_dir / f"{t['tier_id']}.yaml" for t in tiers]
    subset_files = [subset_root / f"{t['tier_id']}.jsonl" for t in tiers]

    head = collections.OrderedDict()
    head["schema_version"] = 1
    head["manifest_version"] = "matrix68/v1"
    head["source"] = ("docs/capability-matrix.data.json（唯一真相源：档号/算法/形态/卡数/并行/"
                      "subset_size/步数三件/lr/四条判据与阈值）"
                      "+ samples/configs/matrix68/*.yaml（现读 max_epochs/save_steps/模型与并行）"
                      "+ subsets-v2/M###.jsonl（现读行数）")
    head["generator"] = "tests/e2e/generate_matrix68_manifest_source.py"
    head["input_fingerprints"] = collections.OrderedDict((
        ("data_sha256", sha256_file(data_path)),
        ("data_path", rel(data_path)),
        ("config_dir", rel(cfg_dir)),
        ("config_dir_fingerprint_sha256", fingerprint_files(cfg_files)),
        ("subset_dir", rel(subset_root)),
        ("subset_dir_fingerprint_sha256", fingerprint_files(subset_files)),
    ))
    head["counts"] = counts
    head["criteria"] = collections.OrderedDict((
        ("all_four_must_hold", True),
        ("items", criteria),
        ("threshold_source", "docs/capability-matrix.data.json → "
                             "meta.criteria_caliber.threshold"),
        ("threshold_status", (th.get("status") or "")),
        ("note", "判据② 的逐算法通过线以该指针为唯一来源；本清单只带展示文本（⛔ 不带第二份数值），"
                 "判定归 tools/matrix/backfill_matrix.py 现算"),
    ))
    head["card_assignment"] = collections.OrderedDict((
        ("policy", "dynamic"),
        ("ruling", "远端机 GPU 0–7 先到先得、不预留、不抢占他人、单作业不限卡数"),
        ("gpu_pool", list(GPU_POOL)),
        ("how", "runner 逐档起跑前串三道闸门（run_guards cards-busy → gpu_lock_guard --visible → "
                "gpu_idle_assert --visible，服务档按 util×总显存算头寸），从实测空闲卡现取 cards 张，"
                "并断言 len(picked)==cards；实跑后回验登记进占卡台账（§17.3）"),
        ("assertion", "A5：len(清单 gpus) == cards ∧ 卡号唯一 ∧ ⊆ 卡池；真实择卡以起跑时实测为准"),
        ("tier_gpus_field", "仅为 schema 对齐与审计的确定性候选窗口，⛔ 不是预留"),
    ))
    head["runtime"] = collections.OrderedDict((
        ("image_env_var", "GRASPO_IMAGE"),
        ("timeout_sec", None),
        ("initial_max_prompt_length", initial_max_prompt_length(cfg_files[0])),
        ("note", "timeout_sec 由 runner 的 `--timeout` 或 GRASPO_RUN_TIMEOUT 显式给出（⛔ 不在此编造）"),
    ))
    head["models"] = collections.OrderedDict((
        ("container_root", "/models"),
        ("host_root_env_var", "GRASPO_MODELS_HOST_ROOT"),
        ("host_root_note", "宿主模型根目录属环境信息（§15.1/§16），运行时由环境变量注入，不入 tracked 文件"),
        ("overrides", model_overrides(tiers)),
        ("mount_in_runner", "-v \"$MODELS_ROOT:/models:ro\"（container_root 只读挂载）"),
    ))
    head["data"] = collections.OrderedDict((
        ("container_root", data_root),
        ("host_root_env_var", "GRASPO_ELAM_HOST_ROOT"),
        ("host_root_note", "宿主数据根目录属环境信息（§16），运行时由环境变量注入，不入 tracked 文件"),
        ("subset_dir_container", f"{data_root}/{subset_root.name}"),
        ("subset_dir_host_relative", subset_root.name),
        ("train_jsonl", f"{data_root}/data/train.jsonl"),
        ("test_jsonl", f"{data_root}/data/test.jsonl"),
        ("full_train_jsonl", f"{data_root}/data/train.jsonl"),
        ("dataset_caveat", dataset_caveat()),
        ("image_reference_style", "\"image\": \"../images/<name>.jpg\"（相对数据文件父目录）"),
        ("image_dir_container", f"{data_root}/images"),
        ("subset_size_policy", "逐档取数据件该行 subset_size（⛔ 本清单不统一改写）；行数断言 A4"),
        ("train_path_not_the_source_note",
         "每档 `data.train_path` 只约束**生成位置**（<RUN_ROOT>/<Tier>/subsets/<Tier>.jsonl）；"
         "runner 的 `TRAIN_PATH == dirname(TRAIN_PATH)/<Tier>.jsonl` 断言与图像根反推都依赖它 ⇒ "
         "⛔ 不要改配置来换数据源"),
    ))
    head["step_gate"] = collections.OrderedDict((
        ("threshold_default", gate_threshold),
        ("threshold_source", "报告 §1「正式记录的硬前提」（min_epochs=1 / min_optimizer_steps=5 / "
                             "min_train_samples=100）；★ 数据件未落该数值 ⇒ 唯一声明点在本生成器的 "
                             "DEFAULT_STEP_GATE_THRESHOLD（可用 --step-gate-threshold 覆盖）"),
        ("rule", "reachable ≥ threshold ⇒ 门槛照常判；reachable < threshold ⇒ 判「⚠ 口径不可测」"
                 "（不是 ❌ 失败，也不是 ✅ 通过）"),
        ("structural_limited_tiers", [t["tier_id"] for t in tiers
                                      if not t["step_gate"]["applicable"]]),
    ))

    manifest = collections.OrderedDict()
    manifest.update(head)
    manifest["tiers"] = tiers
    # 源清单 = 全量 + 生成期自证；消费清单 = runner 逐档现读的同一份内容（同字节，便于审计）
    manifest["self_check"] = collections.OrderedDict((
        ("assertions", ["A1 档数 == 标称", "A2 档号集合 == 数据件 applicable 行集合",
                        "A3 每档 config 存在", "A4 每档训练子集存在且行数 == subset_size",
                        "A5 len(卡集) == cards", "A6 yaml 与数据件身份同源",
                        "A7 max_epochs(yaml)×per_epoch_steps(数据件) == derived_steps",
                        "A8 yaml train_path 与数据件同源", "A9 产物无 deepspeed"]),
        ("all_passed", True),
    ))
    return manifest, manifest


#: 数据集级口径（ELAM V5 train/test **图像层**重叠）的唯一在库副本在 T 版源清单里。
#: ★ 这里**只引用不复制**：复制会变成第三份口径副本，改一处忘一处就会说谎。
T_SOURCE_MANIFEST = ROOT / "tests" / "e2e" / "matrix54_manifest_source.json"


def dataset_caveat() -> str:
    """数据集级 caveat（图像层重叠）—— 从唯一副本读取；取不到 ⇒ 显式说明取自哪里。"""
    doc = read_json(T_SOURCE_MANIFEST) or {}
    text = ((doc.get("data") or {}).get("integrity_caveat") or "").strip()
    if text:
        return text
    return ("见 tests/e2e/matrix54_manifest_source.json → data.integrity_caveat"
            "（ELAM V5 train/test 图像层重叠口径；本清单不另抄一份）")


def read_json(path: Path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def container_data_root_of(train_path_container: str) -> str:
    return str(Path(train_path_container).parent.parent)


def container_data_root(train_path_container: str) -> str:
    """`<根>/<subset_dir>/<tier>.jsonl` ⇒ `<根>`（不写死任何绝对路径）。"""
    p = Path(train_path_container)
    return str(p.parent.parent)


def initial_max_prompt_length(cfg_path: Path) -> int | None:
    cfg = load_yaml(cfg_path)
    v = ((cfg.get("data") or {}).get("max_prompt_length"))
    return int(v) if v is not None else None


def model_overrides(tiers: list[dict]) -> dict:
    out = collections.OrderedDict()
    for t in tiers:
        out.setdefault(t["model"], collections.OrderedDict((
            ("container_path", t["model_path"]),
            ("host_dir_env_var", t["model_host_dir_env_var"]),
            ("host_dir_default", t["model_host_dir_default"]),
        )))
    return out


# ── 自证：校验**在盘清单**（阳性对照入口）──────────────────────────────────────
def verify_manifest(path: Path, *, data_path: Path, cfg_dir: Path, subset_root: Path,
                    expect_count: int, gate_threshold: int) -> int:
    """校验一份**在盘清单**是否仍然自洽（并与其输入可重建结果一致）。

    为什么需要它（而不是只靠"重新生成一遍"）：runner 消费的是**在盘文件**。文件被改坏
    （`cards` 与卡集不符、`train_path` 指向不存在的文件、`config` 指错……）时，"重新生成"
    会把改坏的地方**覆盖掉**，看不见问题；`--verify` 则**就在那份文件上**逐条断言，
    与 runner 起跑前看到的是同一个字节。任一条不成立 ⇒ rc=1 并打印档号。
    """
    if not path.is_file():
        die(f"待校验清单不存在：{path}")
    on_disk = json.loads(path.read_text(encoding="utf-8"))
    tiers = on_disk.get("tiers") or []
    if not tiers:
        die(f"清单没有 tiers：{path}")

    bad: dict[str, list[str]] = {}

    def flag(tier: str, msg: str) -> None:
        bad.setdefault(str(tier), []).append(msg)

    _, applicable, _ = load_data(data_path)
    want_tiers = {str(r["tier"]) for r in applicable}
    got_tiers = {str(t.get("tier_id")) for t in tiers}

    # A1/A2（在盘文件本身）
    if len(tiers) != expect_count:
        flag("<清单>", f"档数 {len(tiers)} ≠ 标称 {expect_count}")
    if got_tiers != want_tiers:
        for t in sorted(want_tiers - got_tiers):
            flag(t, "数据件 applicable 行在清单里缺失")
        for t in sorted(got_tiers - want_tiers):
            flag(t, "清单含数据件里不 applicable 的档号")
    rows_by_tier = {str(r["tier"]): r for r in applicable}

    # 逐档结构断言（与 build_tier 同口径，但**读在盘文件**）
    for t in tiers:
        tier = str(t.get("tier_id"))
        cards = t.get("cards")
        gpus = t.get("gpus") or []
        if isinstance(cards, bool) or not isinstance(cards, int) or cards <= 0:
            flag(tier, f"cards 非法：{cards!r}")
        else:
            if len(gpus) != cards:                                   # A5
                flag(tier, f"len(卡集)={len(gpus)} ≠ cards={cards}（A5）")
            if len(set(gpus)) != len(gpus):
                flag(tier, f"卡集有重复卡号：{gpus}（A5）")
            if any(g not in GPU_POOL for g in gpus):
                flag(tier, f"卡集 {gpus} 越出卡池 {list(GPU_POOL)}（A5）")
            comps = t.get("parallel_components") or {}
            prod = 1
            for k in ("tp", "pp", "dp"):
                try:
                    prod *= int(comps.get(k, 1))
                except (TypeError, ValueError):
                    prod = -1
            if prod != cards:
                flag(tier, f"tp×pp×dp={prod} ≠ cards={cards}（A6）")
        # A3 配置存在
        cfg_rel = t.get("config") or ""
        if not cfg_rel or not (ROOT / cfg_rel).is_file():
            flag(tier, f"config 不存在：{cfg_rel!r}（A3）")
        # A4 训练子集存在 + 行数 == subset_size
        subset_size = (t.get("data") or {}).get("subset_size")
        host_rel = (t.get("data") or {}).get("host_relative_path") or ""
        sp = ROOT / host_rel if host_rel else None
        if sp is None or not sp.is_file():
            flag(tier, f"训练子集不存在：{host_rel!r}（A4）")
        else:
            try:
                lines = line_count(sp)
            except OSError:
                lines = -1
            if lines != subset_size:
                flag(tier, f"训练子集行数 {lines} ≠ subset_size {subset_size}（A4）")
        # A7 步数门槛同源（max_epochs × per_epoch == derived）
        try:
            recomputed = int(t["max_epochs"]) * int(t["expected_optimizer_steps_per_epoch"])
            if recomputed != int(t["expected_optimizer_steps"]):
                flag(tier, f"max_epochs×per_epoch={recomputed} ≠ "
                           f"expected_optimizer_steps={t['expected_optimizer_steps']}（A7）")
        except (KeyError, TypeError, ValueError):
            flag(tier, "步数门槛字段缺失或非数值（A7）")
        # 与数据件身份一致（A6 的数据件侧）
        row = rows_by_tier.get(tier)
        if row is not None:
            for mk, dk in (("algorithm", "algo"), ("mode", "mode"), ("backend", "backend"),
                           ("model", "model"), ("tuner_type", "tuner_type")):
                if str(t.get(mk)) != str(row.get(dk)):
                    flag(tier, f"{mk}={t.get(mk)!r} ≠ 数据件 {dk}={row.get(dk)!r}（A6）")

    # 与"同输入重建结果"逐字节比（catch 任何字段级漂移，含被手改的注释性字段）
    rebuilt = dumps(build_manifest(data_path=data_path, cfg_dir=cfg_dir, subset_root=subset_root,
                                   expect_count=expect_count, gate_threshold=gate_threshold)[0])
    if dumps(on_disk) != rebuilt:
        drift = []
        rt = {str(x["tier_id"]): x for x in json.loads(rebuilt)["tiers"]}
        for t in tiers:
            tier = str(t.get("tier_id"))
            if rt.get(tier) != t:
                drift.append(tier)
        if not drift:
            drift = ["<顶层字段/元信息>"]
        drift = [d for d in drift if d not in bad]
        for d in drift:
            flag(d, "与同输入重建结果不一致（清单已被就地改动 ⇒ 请重新生成）")

    if bad:
        print(f"FAIL: 清单 {path} 未通过自证（{len(bad)} 处）：", file=sys.stderr)
        for tier in sorted(bad):
            for msg in bad[tier]:
                print(f"  - {tier}: {msg}", file=sys.stderr)
        print(f"FAIL: 涉及档号 {sorted(bad)}", file=sys.stderr)
        return FAIL
    print(f"VERIFY PASS：{path} 自证通过（{len(tiers)} 档；A1–A9 在盘复核 + 与重建结果逐字节一致）")
    return 0


# ── 输出 ─────────────────────────────────────────────────────────────────────
def dumps(doc: dict) -> str:
    return json.dumps(doc, ensure_ascii=False, indent=1) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="68 档（M001–M068）运行清单生成器：数据件 + yaml + 子集现读")
    ap.add_argument("--data", default=str(ROOT / "docs" / "capability-matrix.data.json"))
    ap.add_argument("--config-dir", default=str(ROOT / "samples" / "configs" / "matrix68"))
    ap.add_argument("--subset-root", default=str(ROOT / "subsets-v2"))
    ap.add_argument("--out-source", default=str(ROOT / "tests" / "e2e" / "matrix68_manifest_source.json"))
    ap.add_argument("--out-manifest", default=str(ROOT / "tests" / "e2e" / "matrix68_manifest.json"))
    ap.add_argument("--expect-count", type=int, default=68, help="标称档数（默认 68）")
    ap.add_argument("--step-gate-threshold", type=int, default=DEFAULT_STEP_GATE_THRESHOLD,
                    help="步数门槛（默认 5；报告 §1 硬前提。★ 唯一声明点，⛔ 别在别处再抄一份）")
    ap.add_argument("--check-determinism", action="store_true",
                    help="同输入重算两遍，断言逐字节一致（不落盘）")
    ap.add_argument("--print-tier", default=None, metavar="M0nn",
                    help="打印该档的单行 JSON（供 runner 现读；⛔ 不在 bash 里二次解析清单）")
    ap.add_argument("--list-tiers", action="store_true", help="逐行打印档号（供 shell 循环）")
    ap.add_argument("--verify", default=None, metavar="PATH",
                    help="校验一份**在盘清单**（阳性对照入口）：A1–A9 在盘复核 + 与重建结果逐字节一致")
    ap.add_argument("--allow-write", action="store_true", default=True,
                    help=argparse.SUPPRESS)   # 保留位：本件只写 --out-source/--out-manifest
    args = ap.parse_args(argv)

    data_path, cfg_dir, subset_root = Path(args.data), Path(args.config_dir), Path(args.subset_root)
    if args.verify:
        return verify_manifest(Path(args.verify), data_path=data_path, cfg_dir=cfg_dir,
                               subset_root=subset_root, expect_count=args.expect_count,
                               gate_threshold=args.step_gate_threshold)
    manifest, _ = build_manifest(data_path=data_path, cfg_dir=cfg_dir, subset_root=subset_root,
                                 expect_count=args.expect_count,
                                 gate_threshold=args.step_gate_threshold)

    if args.list_tiers:
        for t in manifest["tiers"]:
            print(t["tier_id"])
        return 0
    if args.print_tier:
        hit = [t for t in manifest["tiers"] if t["tier_id"] == args.print_tier]
        if not hit:
            die(f"--print-tier {args.print_tier}：清单里没有该档", [args.print_tier])
        print(json.dumps(hit[0], ensure_ascii=False))
        return 0

    text = dumps(manifest)
    if args.check_determinism:
        manifest2, _ = build_manifest(data_path=data_path, cfg_dir=cfg_dir, subset_root=subset_root,
                                      expect_count=args.expect_count,
                                      gate_threshold=args.step_gate_threshold)
        text2 = dumps(manifest2)
        if text != text2:
            die("幂等自证失败：同输入两跑不逐字节一致")
        print(f"DETERMINISM PASS（{len(manifest['tiers'])} 档，{len(text.encode())} B，"
              f"sha256={sha256_bytes(text.encode())}）")
        return 0

    for out in (Path(args.out_source), Path(args.out_manifest)):
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8")
        print(f"wrote {rel(out)}  tiers={len(manifest['tiers'])} "
              f"bytes={len(text.encode())} sha256={sha256_bytes(text.encode())}")
    print("ASSERTIONS PASS：A1–A9 全过（"
          + "、".join(f"{k}={v}" for k, v in manifest["counts"].items()
                      if k in ("total", "by_backend", "by_model", "by_subset_size")) + "）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
