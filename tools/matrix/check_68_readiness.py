#!/usr/bin/env python3
"""check_68_readiness.py — 68 档（M001–M068）**开跑前就绪闸门**：把定稿口径变成可回归断言。

为什么需要本文件：现行闸门没有一条能判「配置符合定稿文档第 3 章口径」——
`GraspoConfig.from_yaml` 是 pydantic 全量校验但对 68 个生成件**全 PASS**
（它不校验 r/alpha、子集条数、全参必分片、native SP）；`gen_configs --check-declared`
的键是旧 54 档 `T0xx` 档号 ⇒ 对 M 档**空转**。⇒ 开跑前缺一道「口径级」机检。

六条断言（逐条对应本轮六项必改，全部 fail-closed）：

  ① **档号集合**：`--config-dir` 下恰 68 个 `M0nn.yaml`，且档号集合 == 数据件 `applicable`
     行集合（⛔ 多一个、少一个都不行；目录里也不许混入非 M 档号文件）。
  ② **LoRA 三参 + 目标整块**：`tuner_type == lora` 的 52 档 —— `lora.r/alpha/dropout`
     必须恰为 `32 / 64 / 0.05`，且 `lora.target_modules` 含 `in_proj_qkv`（**整块**，
     ⛔ 不许拆成 `in_proj_q/k/v` 三段）。
  ③ **子集**：逐档 `subset_size == 100`（数据件行 + 登记清单两处同值），
     `data.train_path` 指向的**在盘实体存在**、行数 == 100、`sha256` == 清单登记值。
     ★ 解析口径：`train_path` 是**容器内路径**（`<data_root>/subsets-v2/M0nn.jsonl`），
     本闸门只取尾部 `subsets-v2/<tier>.jsonl` 在 `--subset-dir`（默认仓根 `subsets-v2/`）里核，
     ⇒ 宿主侧无需挂载 `/work/elam-v5` 也能机检。
  ④ **全参必分片**：`tuner_type == full` ∧ `backend == msswift` ⇒ 必带 `msswift.fsdp`；
     且**卡数 ≤ 2 时必须取 `_offload` 配方**（4 卡取无 offload 配方）。DeepSpeed 已禁用
     ⇒ 全参只能靠 FSDP2 摊优化器态（依据 `samples/configs/fsdp/*` 的 `_phase2_only` 分工）。
  ⑤ **native 并行开关**：`backend == native` ∧ `tp_size > 1` ⇒ `native.sequence_parallel == true`；
     `backend == native` ∧ 全参 ∧ 卡数 == 1 ⇒ `native.offload_optimizer_state == true`。
  ⑥ **DeepSpeed 禁用**：任何产物出现 `deepspeed:` ⇒ 失败（主席 2026-09-27 裁定 / 用户 2026-09-22 拍板）。

阳性对照（必须实测，⛔ 不是"看着像会失败"）::

    # 把一份配置改坏到临时副本 ⇒ 期望 rc=1
    cp -r samples/configs/matrix68 /tmp/bad68 && sed -i 's/^  r: 32$/  r: 128/' /tmp/bad68/M001.yaml
    python3 tools/matrix/check_68_readiness.py --config-dir /tmp/bad68   # ⇒ rc=1

用法::

    python3 tools/matrix/check_68_readiness.py \
        [--config-dir samples/configs/matrix68] [--data docs/capability-matrix.data.json] \
        [--manifest subsets-v2/subsets-v2.manifest.json] [--subset-dir subsets-v2] [--quiet]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

#: 期望档数（68 档 = M001–M068）。
EXPECTED_TIERS = tuple(f"M{i:03d}" for i in range(1, 69))
#: 每档训练样本数（定稿口径：全部档位共用同一份 100 条）。
EXPECTED_SUBSET_SIZE = 100
#: LoRA 三参（定稿口径，人读件第 3 章）。
EXPECTED_LORA = {"r": 32, "alpha": 64, "dropout": 0.05}
#: LoRA 目标必须包含的**整块**模块名（⛔ 不是拆开的三段 q/k/v）。
REQUIRED_TARGET_TOKEN = "in_proj_qkv"
#: FSDP2 配方文件名（1/2 卡 → 开 CPU offload；4 卡 → 无 offload）。
FSDP_OFFLOAD_RECIPE = "fsdp2_full_state_dict_offload.json"
FSDP_PLAIN_RECIPE = "fsdp2_full_state_dict.json"

DEFAULTS = {
    "config_dir": "samples/configs/matrix68",
    "data": "docs/capability-matrix.data.json",
    "manifest": "subsets-v2/subsets-v2.manifest.json",
    "subset_dir": "subsets-v2",
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _i(v):
    try:
        return int(v)
    except Exception:
        return None


def load_yaml(path: Path) -> dict:
    import yaml
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def manifest_m_entries(path: Path) -> dict:
    """读登记清单 ⇒ `{tier: entry}`（只取 M 档条目）。★ fail-closed：清单缺失即拒判。"""
    if not path.is_file():
        raise SystemExit(f"FATAL: 登记清单不存在：{path}（⛔ 不静默跳过 ③ 的 sha 核对）")
    doc = json.loads(path.read_text(encoding="utf-8"))
    out = {}
    for t in doc.get("tiers") or []:
        tid = str(t.get("tier_id") or t.get("tier") or "")
        if tid.startswith("M"):
            out[tid] = t
    return out


def resolve_subset_path(train_path: str, subset_dir: Path):
    """把配置里的**容器内** `train_path` 解析成宿主侧在盘路径。

    只管一种形态：尾部必须是 `subsets-v2/<tier>.jsonl`（档号身份），
    宿主侧实体在 `--subset-dir` 下同名 —— 于是**不需要** `/work/elam-v5` 挂载。
    """
    parts = [p for p in str(train_path).replace("\\", "/").split("/") if p]
    if len(parts) < 2 or parts[-2] != "subsets-v2":
        return None, f"train_path 尾部不是 subsets-v2/<tier>.jsonl（{train_path}）"
    return subset_dir / parts[-1], None


def check(config_dir: Path, data_path: Path, manifest_path: Path, subset_dir: Path) -> int:
    failures: list[tuple[str, str]] = []          # (判据编号, 说明)
    notes: list[str] = []

    # ── 前置：数据件 ──────────────────────────────────────────────────────
    if not data_path.is_file():
        raise SystemExit(f"FATAL: 数据件不存在：{data_path}")
    data = json.loads(data_path.read_text(encoding="utf-8"))
    rows = {str(r["tier"]): r for r in (data.get("rows") or []) if r.get("applicable")}

    # ── 前置：配置目录 ────────────────────────────────────────────────────
    if not config_dir.is_dir():
        raise SystemExit(f"FATAL: 配置目录不存在：{config_dir}")
    files = sorted(p for p in config_dir.iterdir() if p.is_file())
    yamls = sorted(p for p in files if p.suffix in (".yaml", ".yml"))
    others = [p.name for p in files if p.suffix not in (".yaml", ".yml")]
    got_tiers = {p.stem for p in yamls}

    # ── ① 档号集合 ────────────────────────────────────────────────────────
    if others:
        failures.append(("①档号集合", f"{config_dir} 混入非 yaml 文件（⛔ 目录只放档号配置）：{others}"))
    non_m = sorted(t for t in got_tiers if not re.fullmatch(r"M\d{3}", t))
    if non_m:
        failures.append(("①档号集合", f"{config_dir} 含非 M 档号文件：{non_m}"))
    if len(yamls) != len(EXPECTED_TIERS):
        failures.append(("①档号集合", f"配置数 {len(yamls)} != 期望 {len(EXPECTED_TIERS)}"))
    missing = sorted(set(EXPECTED_TIERS) - got_tiers)
    extra = sorted(got_tiers - set(EXPECTED_TIERS))
    if missing:
        failures.append(("①档号集合", f"缺档号 {len(missing)} 个：{missing[:10]}"))
    if extra:
        failures.append(("①档号集合", f"多档号 {len(extra)} 个：{extra[:10]}"))
    if got_tiers != set(rows):
        only_cfg = sorted(got_tiers - set(rows))
        only_data = sorted(set(rows) - got_tiers)
        failures.append(("①档号集合", f"配置档号集合 != 数据件 applicable 行集合"
                                       f"（仅配置有 {only_cfg[:10]}；仅数据件有 {only_data[:10]}）"))

    # 逐档加载（只加载**存在且档号合法**的档，避免缺档时的重复噪声）
    cfgs = {}
    for st in sorted(got_tiers & set(rows)):
        p = config_dir / f"{st}.yaml"
        if not p.is_file():
            p = config_dir / f"{st}.yml"
        try:
            cfgs[st] = load_yaml(p)
        except Exception as exc:                      # noqa: BLE001 - 逐档如实报告
            failures.append(("①档号集合", f"{st}: yaml 不可解析：{str(exc)[:100]}"))

    try:
        m_entries = manifest_m_entries(manifest_path)
    except SystemExit as exc:
        print(str(exc), file=sys.stderr)
        return 2

    n_lora = n_msfull = n_nsp = n_noff = n_cpt = 0
    lora_tiers, msfull_tiers, nsp_tiers, noff_tiers, cpt_tiers = [], [], [], [], []

    for tier in sorted(cfgs):
        cfg = cfgs[tier]
        row = rows[tier]
        tuner = str(cfg.get("tuner_type") or "")
        backend = str(cfg.get("backend") or "")
        cards = _i(row.get("cards") or row.get("cards_total"))
        tp = _i(cfg.get("native", {}).get("tp_size")) if isinstance(cfg.get("native"), dict) else None
        tp = tp if tp is not None else _i(row.get("tp_size"))

        # ── ② LoRA 52 档：三参 + 目标整块 ─────────────────────────────────
        if tuner == "lora":
            n_lora += 1
            lora_tiers.append(tier)
            block = cfg.get("lora")
            if not isinstance(block, dict):
                failures.append(("②LoRA三参", f"{tier}: 缺 `lora` 段"))
            else:
                bad = {k: block.get(k) for k, v in EXPECTED_LORA.items()
                       if block.get(k) is None or abs(float(block.get(k)) - v) > 1e-9}
                if bad:
                    failures.append(("②LoRA三参", f"{tier}: lora 三参不符定稿口径 "
                                                   f"{EXPECTED_LORA}，实为 {bad}"))
                tm = block.get("target_modules")
                if not tm or REQUIRED_TARGET_TOKEN not in str(tm):
                    failures.append(("②LoRA三参", f"{tier}: target_modules 不含整块 "
                                                   f"`{REQUIRED_TARGET_TOKEN}`（⛔ 不得拆成 q/k/v 三段）"))

        # ── ③ 子集：subset_size==100 + 实体在盘 + sha 与登记值一致 ─────────
        row_sub = _i(row.get("subset_size"))
        if row_sub != EXPECTED_SUBSET_SIZE:
            failures.append(("③子集", f"{tier}: 数据件 subset_size={row.get('subset_size')} "
                                      f"!= {EXPECTED_SUBSET_SIZE}"))
        ent = m_entries.get(tier)
        if ent is None:
            failures.append(("③子集", f"{tier}: 登记清单无该档 M 条目（无法核 sha256）"))
        else:
            if _i(ent.get("subset_size")) != EXPECTED_SUBSET_SIZE:
                failures.append(("③子集", f"{tier}: 清单 subset_size={ent.get('subset_size')} "
                                          f"!= {EXPECTED_SUBSET_SIZE}"))
            if _i(ent.get("lines")) != EXPECTED_SUBSET_SIZE:
                failures.append(("③子集", f"{tier}: 清单 lines={ent.get('lines')} "
                                          f"!= {EXPECTED_SUBSET_SIZE}"))
        train_path = str((cfg.get("data") or {}).get("train_path") or "")
        if not train_path:
            failures.append(("③子集", f"{tier}: 缺 `data.train_path`"))
        else:
            ent_path, why = resolve_subset_path(train_path, subset_dir)
            if ent_path is None:
                failures.append(("③子集", f"{tier}: {why}"))
            elif ent_path.name != f"{tier}.jsonl":
                failures.append(("③子集", f"{tier}: train_path 指向 {ent_path.name}"
                                          f"（⛔ 档号与文件不一致）"))
            elif not ent_path.is_file():
                failures.append(("③子集", f"{tier}: 在盘实体不存在：{ent_path}"))
            else:
                n_lines = ent_path.read_bytes().count(b"\n")
                if n_lines != EXPECTED_SUBSET_SIZE:
                    failures.append(("③子集", f"{tier}: 实体行数 {n_lines} != {EXPECTED_SUBSET_SIZE}"))
                digest = sha256_file(ent_path)
                if ent is not None and digest != str(ent.get("sha256")):
                    failures.append(("③子集", f"{tier}: 实体 sha256={digest[:16]}… != "
                                              f"清单登记 {str(ent.get('sha256'))[:16]}…"))

        # ── ④ 全参必分片（ms-swift）────────────────────────────────────────
        if tuner == "full" and backend == "msswift":
            n_msfull += 1
            msfull_tiers.append(tier)
            fsdp = str((cfg.get("msswift") or {}).get("fsdp") or "")
            want = FSDP_OFFLOAD_RECIPE if (cards is not None and cards <= 2) else FSDP_PLAIN_RECIPE
            if not fsdp:
                failures.append(("④全参必分片", f"{tier}: ms-swift 全参缺 `msswift.fsdp`"
                                                f"（DeepSpeed 已禁用 ⇒ 必须走 FSDP2）"))
            elif not fsdp.endswith(want):
                failures.append(("④全参必分片", f"{tier}: cards={cards} 应取 `{want}`，实为 `{fsdp}`"))

        # ── ⑤ native 并行开关 ──────────────────────────────────────────────
        if backend == "native" and tp is not None and tp > 1:
            n_nsp += 1
            nsp_tiers.append(tier)
            if (cfg.get("native") or {}).get("sequence_parallel") is not True:
                failures.append(("⑤native开关", f"{tier}: tp_size={tp}>1 但 "
                                                f"`native.sequence_parallel` 不为 true"))
        if backend == "native" and tuner == "full" and cards == 1:
            n_noff += 1
            noff_tiers.append(tier)
            if (cfg.get("native") or {}).get("offload_optimizer_state") is not True:
                failures.append(("⑤native开关", f"{tier}: 1 卡全参但 "
                                                f"`native.offload_optimizer_state` 不为 true"
                                                f"（不开则单卡显存放不下）"))

        # ── CPT 显式语义开关（本轮第 ⑥ 项，属②同族的防呆机检，非硬失败项）──
        if str(cfg.get("train_method") or "") == "cpt":
            n_cpt += 1
            cpt_tiers.append(tier)
            pt = cfg.get("pretrain") or {}
            if pt.get("loss_scale") != "all" or pt.get("use_chat_template") is not False:
                failures.append(("②LoRA三参·CPT附项",
                                 f"{tier}: CPT 未显式写 pretrain.loss_scale=all / "
                                 f"use_chat_template=false（实为 {pt}）"))

        # ── ⑥ 任何产物含 deepspeed: ⇒ 失败 ────────────────────────────────
        raw = (config_dir / f"{tier}.yaml").read_text(encoding="utf-8")
        if re.search(r"^\s*deepspeed\s*:", raw, re.M):
            failures.append(("⑥DeepSpeed禁用", f"{tier}: 产物含 `deepspeed:`"
                                               f"（DeepSpeed 已禁用）"))

    # ── 报告 ──────────────────────────────────────────────────────────────
    checks = {
        "①档号集合": f"{len(yamls)} 个配置 / 数据件 applicable {len(rows)} 行",
        "②LoRA三参": f"{n_lora} 档 lora（r/alpha/dropout=32/64/0.05 + in_proj_qkv 整块）+ CPT {n_cpt} 档",
        "③子集": f"subset_size=100 + 在盘实体 sha256 核对",
        "④全参必分片": f"{n_msfull} 档 ms-swift 全参（FSDP2 配方按卡数分工）",
        "⑤native开关": f"SP {n_nsp} 档（tp>1）/ 1 卡全参 offload {n_noff} 档",
        "⑥DeepSpeed禁用": "产物不得出现 `deepspeed:`",
    }
    print("=== 68 档就绪闸门（check_68_readiness）===")
    print(f"  config-dir = {config_dir}")
    print(f"  data       = {data_path}")
    print(f"  manifest   = {manifest_path}")
    print(f"  subset-dir = {subset_dir}")
    for cid, desc in checks.items():
        got = [f for f in failures if f[0].startswith(cid[:2])]
        print(f"  [{'PASS' if not got else 'FAIL'}] {cid}：{desc}")
    print(f"\n  命中档号：lora={n_lora} ms-swift全参={n_msfull} nativeSP={n_nsp} "
          f"native1卡offload={n_noff} cpt={n_cpt}")
    for label, lst in (("ms-swift全参", msfull_tiers), ("native SP(tp>1)", nsp_tiers),
                       ("native 1卡全参 offload", noff_tiers), ("CPT", cpt_tiers)):
        notes.append(f"    {label}: {lst}")
    print("\n".join(notes))

    if failures:
        print(f"\nREADINESS FAIL：{len(failures)} 处不符定稿口径")
        for cid, msg in failures:
            print(f"  - [{cid}] {msg}")
        return 1
    print("\nREADINESS PASS（68 档 × 六条口径）")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="68 档开跑前就绪闸门（六条口径机检，fail-closed）")
    ap.add_argument("--config-dir", default=DEFAULTS["config_dir"], type=Path,
                    help="68 档配置目录（默认新目录 samples/configs/matrix68）")
    ap.add_argument("--data", default=DEFAULTS["data"], type=Path, help="矩阵数据件（唯一可编辑源头）")
    ap.add_argument("--manifest", default=DEFAULTS["manifest"], type=Path,
                    help="子集登记清单（逐档 subset_size/lines/sha256/path）")
    ap.add_argument("--subset-dir", default=DEFAULTS["subset_dir"], type=Path,
                    help="宿主侧子集实体目录（对应容器内 <data_root>/subsets-v2）")
    a = ap.parse_args(argv)
    return check(a.config_dir, a.data, a.manifest, a.subset_dir)


if __name__ == "__main__":
    sys.exit(main())
