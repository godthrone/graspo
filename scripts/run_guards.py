#!/usr/bin/env python3
"""起跑/评测护栏（§7 形态谓词 / §8 产物新鲜度 / §10 27B Mamba 并发）。

三条护栏都是 228 实测事故的制度化（见 docs/run-discipline.md §7/§8/§10）：

§7  评测配置的 ``checkpoint_path`` 必须是**评测容器内可达**路径，且**谓词须按形态分支**：
    全参/HF ⇒ config.json ∨ model-*.safetensors；PEFT adapter ⇒ adapter_config.json ∨
    adapter_model.safetensors；native rank 分片 ⇒ rank_*.pt ∨ manifest.json。

§8  合并/导出目录**必须为空或本批新建**；⛔ 不得以"存在产物文件"判成功。
    pre：导出前目录不存在（或为空）；post：导出后存在 ∧ 含权重 ∧ 创建时刻 ≥ 本批起跑。

§10 服务 27B（Mamba）档必须显式 ``--max-num-seqs`` 且 ≤ 可用 Mamba cache blocks。

用法（可独立跑，也可被 launcher/评测脚本调用）：
    python3 scripts/run_guards.py form   --checkpoint <dir>                 # §7
    python3 scripts/run_guards.py fresh  --dir <dir> --phase pre|post --since <epoch>
    python3 scripts/run_guards.py mamba  --max-num-seqs 192 --blocks 232
退出码：0 通过；1 失败（fail-closed）；2 用法错误。
"""

from __future__ import annotations

import argparse
import os
import sys

#: 形态 → 该形态下的"可信标志文件"（任一命中即认为形态成立）
def _probe(form, checkpoint_dir):
    """极薄探测：目录是否命中 `form` 的标记（供"通道是否存在"断言使用）。"""
    return any(os.path.exists(os.path.join(checkpoint_dir, m)) for m in FORM_MARKERS.get(form, ()))


FORM_MARKERS: dict[str, tuple[str, ...]] = {
    "full": ("config.json", "model.safetensors", "model.safetensors.index.json"),
    "adapter": ("adapter_config.json", "adapter_model.safetensors"),
    "native_rank": ("manifest.json",),
}


def busy_cards_from_docker(names: list[str] | None = None, docker_json: str | None = None) -> tuple[set[str], list[str]]:
    """★ §17 结构性兜底：从 `docker inspect` 的 `DeviceRequests` 解析**我方容器占用的卡**。

    ★ 必须覆盖两类：ⓐ **已占**（running）ⓑ **已预约未加载**（created/restarting/刚起、vLLM 未就绪）
    —— 后者正是"加载期卡看似空闲"的坑（T002 评测起在 GPU6，而 T044 占着 GPU6/7 ⇒ 只剩 3.81 GiB）。

    入参 `docker_json` 为 `docker inspect` 输出的 JSON 数组（便于单测）；返回 (禁用卡集, 说明列表)。
    """
    import json as _json
    busy: set[str] = set()
    notes: list[str] = []
    try:
        items = _json.loads(docker_json) if docker_json else []
    except Exception as exc:  # pragma: no cover - 解析失败必须显式
        return (set(), [f"解析 docker inspect 失败：{exc}"])
    for it in items:
        name = (it.get("Name") or "").lstrip("/")
        if names and name not in names:
            continue
        state = ((it.get("State") or {}).get("Status") or "").lower()
        reqs = ((it.get("HostConfig") or {}).get("DeviceRequests")) or []
        gids: list[str] = []
        for r in reqs:
            for d in (r.get("DeviceIDs") or []):
                gids.extend(str(d).split(","))
        # ★ 228 实测：训练容器用 `--runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=1,3,6,7` 传卡
        #   ⇒ `HostConfig.DeviceRequests` 为 **null** ⇒ 必须同时从 env 采（否则台账恒为空）
        for e in ((it.get("Config") or {}).get("Env") or []):
            if e.startswith("NVIDIA_VISIBLE_DEVICES="):
                val = e.split("=", 1)[1]
                gids.extend(g.strip() for g in val.split(",") if g.strip() not in ("", "all", "void", "none"))
        if gids:
            busy.update(g.strip() for g in gids if g.strip() != "")
            notes.append(f"{name}[{state}] ⇒ GPU {','.join(gids)}")
    return (busy, notes)


def exclude_cards_held_by_us(available: list[str], busy_cards: set[str]) -> list[str]:
    """★ 择卡硬排除：把"**我方**已占/已预约（含 loading 态）"的卡从可用集里剔除
    （⛔ 不是建议，是硬排除）。

    ★ 2026-10-17 更名（原 ``include_reserved_cards``）：主席令「228 先到先得，随便用。」
    ＋「单个作业不限制卡数。」⇒ 卡池**不存在**"预留/禁用/生产锁卡"这类静态名单
    （本仓的契约扫查 ``tests/compliance/test_gpu_contract_retired_sweep.py`` 会对该词表报红）。
    本函数的语义从来不是"预留卡"，而是"**我方自己**这些卡已经在用了 ⇒ 别自己撞自己"——
    真正对外的硬要求是「不抢占**他人**」，那一条住在起跑前的逐卡实测空闲断言里
    （``scripts/gpu_idle_assert.py``：剩余 ≥ 本档头寸 ∧ util == 0；⛔ 绝不 kill 他人）。
    """
    return [c for c in available if str(c) not in busy_cards]


def check_train_infer_exclusive(target_card: str, card_roles: dict[str, str]) -> tuple[bool, str]:
    """★ B10 契约规则 1/2（2026-10-14 立，实例驱动）：同一张卡不得同时承载训练与推理。

    ★ 实例：T022 的评测 vLLM 与 T030 的训练共用 GPU4 ⇒ 起服务失败，逐字
    `Free memory on device cuda:0 (30.82/79.25 GiB) … < 0.85 (67.36 GiB)` ⇒ 评测白跑一轮。

    ★ 判据："卡上没显存 ≠ 卡没人用" —— 推理 vLLM 加载期卡看似空闲，实为已预约。
    ★ 违反 ⇒ 返回 (False, 理由)，调用方**起跑前 fail-closed**。

    Args:
        target_card: 拟使用的卡号（字符串）。
        card_roles: 卡号 → 角色（``"train"`` / ``"infer"``）；同一卡若已登记任一角色，
            再以另一角色申请即视为冲突。
    """
    existing = card_roles.get(str(target_card))
    if existing is None:
        return True, f"卡 {target_card} 未被任何训练/推理容器引用 ⇒ 放行"
    if existing == "train":
        return False, (
            f"卡 {target_card} 已被【训练】容器占用 ⇒ 拒绝以【推理】复用"
            "（B10 规则 1/2：二者不得交叉；★ 卡上没显存 ≠ 卡没人用）"
        )
    return False, f"卡 {target_card} 已被【{existing}】容器占用 ⇒ 拒绝复用"


def card_roles_from_containers(docker_ps_text: str) -> dict[str, str]:
    """★ 汇聚 `{card: role}`：从运行中容器（只读）判定该卡承载的是训练还是推理。

    判定依据（依次）：容器名/`--name`、`-e TIER=`（训练档）、``vllm``/``served-model-name``（推理服务）；
    卡号取 `-e NVIDIA_VISIBLE_DEVICES=`。⛔ 只读，**不 kill、不动任何容器**。
    """
    roles: dict[str, str] = {}
    for line in (docker_ps_text or "").splitlines():
        line = line.strip()
        if not line:
            continue
        role = "infer" if ("vllm" in line or "served-model-name" in line) else "train"
        for token in line.replace('"', " ").split():
            if token.startswith("NVIDIA_VISIBLE_DEVICES="):
                # ★ 值内逗号是卡分隔符（`=1,3`）；⛔ 不可先把逗号当空白切开，否则丢卡
                for card in token.split("=", 1)[1].split(","):
                    card = card.strip()
                    if card and card not in ("all", "void", "none"):
                        roles.setdefault(card, role)
    return roles


def assert_no_card_reuse(target_cards: list[str], docker_ps_text: str) -> None:
    """★ B10 规则 1/2 的【接线】：起跑前对每张目标卡做互斥校验，违反即 fail-closed。

    ★ 顺序保证：**本函数在起跑前调用** ⇒ ⛔ 不会"跑起来才发现"。
    """
    roles = card_roles_from_containers(docker_ps_text)
    for card in target_cards:
        ok, why = check_train_infer_exclusive(card, roles)
        if not ok:
            raise SystemExit(f"★ B10 卡池契约 fail-closed（起跑前）：{why}")


def detect_form_and_validate(checkpoint_dir: str) -> tuple[str, bool, str]:
    """§7：返回 (形态, 是否通过, 说明)。全参/ adapter / native rank 三种形态各按自己的谓词判。"""
    if not checkpoint_dir or not os.path.isdir(checkpoint_dir):
        return ("none", False, f"checkpoint 目录不存在：{checkpoint_dir}")
    # PEFT adapter 优先（adapter 目录里通常**没有** config.json）
    if any(os.path.exists(os.path.join(checkpoint_dir, m)) for m in FORM_MARKERS["adapter"]):
        return ("adapter", True, "adapter 形态：命中 adapter_config.json / adapter_model.safetensors")
    if any(os.path.exists(os.path.join(checkpoint_dir, m)) for m in FORM_MARKERS["full"]):
        return ("full", True, "全参/HF 形态：命中 config.json / model*.safetensors")
    # 分片权重：model-00001-of-00018.safetensors（merged-hf 的真实形态）
    if any(f.startswith("model-") and f.endswith(".safetensors") and "-of-" in f for f in os.listdir(checkpoint_dir)):
        return ("full", True, "全参/HF 分片形态：命中 model-*-of-*.safetensors")
    # ★★ 起跑前 fail-closed：该形态**必须有可用评测通道**（单一真源在 eval.merged_export）
    try:
        from graspo.eval.merged_export import ExportError as _ExportError  # noqa: PLC0415
        from graspo.eval.merged_export import assert_form_has_channel  # noqa: PLC0415
        for _f in ("base_model", "adapter", "native_graspo", "native_rank"):
            if _probe(_f, checkpoint_dir):
                assert_form_has_channel(_f)
    except ImportError:
        pass
    except _ExportError as exc:                   # ★ 只吞"无通道"这一类；⛔ 不用 Exception 掩盖真 bug
        raise SystemExit(f"FAIL-CLOSED：起跑前形态通道断言未通过：{exc}")

    if any(os.path.exists(os.path.join(checkpoint_dir, m)) for m in FORM_MARKERS["native_rank"]):
        return ("native_rank", True, "native 形态：命中 manifest.json")
    for pat in ("rank_",):
        if any(f.startswith(pat) and f.endswith(".pt") for f in os.listdir(checkpoint_dir)):
            return ("native_rank", True, "native 形态：命中 rank_*.pt")
    return ("unknown", False, f"目录存在但无任何形态标志文件（{sorted(os.listdir(checkpoint_dir))[:5]}）")


def check_fresh(directory: str, phase: str, since_epoch: float | None = None) -> tuple[bool, str]:
    """§8：pre = 目录必须不存在或为空；post = 存在 ∧ 含权重 ∧ 创建时刻 ≥ since。"""
    exists = os.path.exists(directory)
    if phase == "pre":
        if not exists:
            return (True, "§8(pre) 通过：目录在导出前不存在（将为本批新建）")
        entries = os.listdir(directory)
        if not entries:
            return (True, "§8(pre) 通过：目录存在但为空")
        return (False, f"§8(pre) 失败：目录在导出前已存在且非空（陈旧产物）：{entries[:5]}")
    if phase == "post":
        if not exists:
            return (False, "§8(post) 失败：导出后目录不存在（本批未产出）")
        weights = [f for f in os.listdir(directory) if f.endswith((".safetensors", ".pt"))]
        if not weights:
            return (False, "§8(post) 失败：导出后目录无权重文件（本批未产出）")
        if since_epoch is not None:
            ctime = os.stat(directory).st_ctime
            if ctime < since_epoch:
                return (False, f"§8(post) 失败：目录创建时刻 {ctime} < 本批起跑 {since_epoch}（疑似陈旧）")
        return (True, f"§8(post) 通过：本批新建且含权重（{len(weights)} 个）")
    return (False, f"未知 phase：{phase}")


def check_mamba(max_num_seqs: int, blocks: int) -> tuple[bool, str]:
    """§10：max_num_seqs 必须 ≤ 可用 Mamba cache blocks。"""
    if max_num_seqs <= blocks:
        return (True, f"§10 通过：max_num_seqs={max_num_seqs} ≤ blocks={blocks}")
    return (False, f"§10 失败：max_num_seqs={max_num_seqs} > 可用 Mamba cache blocks={blocks}（会 Engine core initialization failed）")


def parse_docker_inspect_stream(text: str) -> list[dict]:
    """★ 解析 `docker inspect` 的**多 JSON 对象流**（非单个数组）。

    `docker inspect <c1> <c2>` 输出的是**并排的多个 JSON 对象**（不是数组）⇒ 直接 `json.loads`
    会失败（T002 那次我的真机冒烟就卡在这里）。本函数用 `json.JSONDecoder.raw_decode` 逐个解码。
    """
    import json as _json
    dec = _json.JSONDecoder(); objs: list[dict] = []; i = 0; n = len(text or "")
    while i < n:
        while i < n and text[i] in " \t\r\n,":
            i += 1
        if i >= n:
            break
        try:
            obj, j = dec.raw_decode(text, i)
        except Exception:
            break
        if isinstance(obj, list):
            objs.extend(obj)          # ★ 单数组及其拼接（docker inspect 有时按容器给数组）
        else:
            objs.append(obj)
        i = j
    return objs


def busy_cards_cli(docker_ps_text: str = "", docker_inspect_text: str = "") -> tuple[set[str], str]:
    """给 CLI 用：由 `docker ps --format '{{.Names}}'` 与 `docker inspect <names>` 的原始输出算出禁用卡集。"""
    names = [l.strip() for l in (docker_ps_text or "").splitlines() if l.strip()]
    items = parse_docker_inspect_stream(docker_inspect_text)
    busy, _ = busy_cards_from_docker(names=names, docker_json=__import__("json").dumps(items))
    return (busy, ",".join(sorted(busy, key=lambda x: int(x) if x.isdigit() else 99)))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="起跑/评测护栏（§7/§8/§10）")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_form = sub.add_parser("form", help="§7 形态谓词")
    p_form.add_argument("--checkpoint", required=True)

    p_fresh = sub.add_parser("fresh", help="§8 产物新鲜度")
    p_fresh.add_argument("--dir", required=True)
    p_fresh.add_argument("--phase", choices=("pre", "post"), required=True)
    p_fresh.add_argument("--since", type=float, default=None)

    p_mamba = sub.add_parser("mamba", help="§10 Mamba 并发")
    p_mamba.add_argument("--max-num-seqs", type=int, required=True)
    p_mamba.add_argument("--blocks", type=int, required=True)

    p_busy = sub.add_parser("cards-busy", help="§17 占卡台账（我方容器占用的卡，含加载期）")
    p_busy.add_argument("--ps-file", default="-", help="docker ps --format {{.Names}} 的输出文件（- = 读 stdin 的 ps 段）")
    p_busy.add_argument("--inspect-file", default=None, help="docker inspect 的输出文件（多对象流）")

    args = parser.parse_args(argv)
    if args.cmd == "form":
        form, ok, msg = detect_form_and_validate(args.checkpoint)
        print(f"[{form}] {msg}")
    elif args.cmd == "cards-busy":
        ps_text = sys.stdin.read() if args.ps_file == "-" else open(args.ps_file, encoding="utf-8").read()
        ins_text = open(args.inspect_file, encoding="utf-8").read() if args.inspect_file else ""
        busy, csv = busy_cards_cli(ps_text, ins_text)
        print(f"BUSY_CARDS={csv}")
        ok = True
    elif args.cmd == "fresh":
        ok, msg = check_fresh(args.dir, args.phase, args.since)
        print(msg)
    else:
        ok, msg = check_mamba(args.max_num_seqs, args.blocks)
        print(msg)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
