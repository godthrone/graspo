#!/usr/bin/env python3
"""gen_configs.py — 由矩阵数据（唯一可编辑源头）确定性生成各档测试 yaml。

★ 方向固定：**矩阵数据 → yaml（产物）**。⛔ yaml 不得手工编辑。
★ 生成器**不读时钟 / 环境 / 网络 / 人机交互**；同一 (数据, 参数) ⇒ 同一 yaml 字节。
★ 矩阵数据的路径为符号路径（已脱敏）：真实前缀由命令行参数 `--model-root` / `--data-root` 注入，
   因此生成结果只依赖 (数据, 参数)，可复现。

用法：
  python3 tools/matrix/gen_configs.py --data docs/capability-matrix.data.json \
      --out-dir samples/configs/matrix54-generated --only T010,T028
  确定性/幂等自证：
  python3 tools/matrix/gen_configs.py --data <data.json> --out-dir <dir> --check-determinism
  与既有冻结 yaml 的差异归因：
  python3 tools/matrix/gen_configs.py --data <data.json> --out-dir <dir> --check-dir samples/configs/matrix54-v2
"""
import argparse
import json
import re
import sys
from pathlib import Path

HEADER = ("# 本文件由 tools/matrix/gen_configs.py 从 docs/capability-matrix.data.json 自动生成，"
          "⛔ 禁止手工编辑；请改数据后重新生成。\n")


def opd_teacher_missing(text):
    """★ 断言辅助（可单测）：OPD 档是否缺教师（`teacher_model_path` 非空 或 显式 `teacher_model_server`）。

    返回 True = **缺**（应 fail-closed）。
    """
    if "train_method: opd" not in text:
        return False
    import re as _re
    m = _re.search(r"^\s+teacher_model_path:\s*(\S+)\s*$", text, _re.M)
    srv = _re.search(r"^\s+teacher_model_server:\s*(\S+)\s*$", text, _re.M)
    ok_path = bool(m) and m.group(1) not in ('""', "''", "~", "null")
    ok_srv = bool(srv) and srv.group(1) not in ('""', "''", "~", "null")
    return not (ok_path or ok_srv)


def opd_cards_shortfall(teacher, cards):
    """★ OPD 卡数断言（**数据驱动**）：manifest 若声明了 `min_cards`（P-25 实测回填后），则 `cards` 不得低于它。

    返回 None = 通过；否则返回拒绝理由。★ manifest **未声明**时不做限制（其规则明载教师侧"未测算"、
    判定恒为 unmeasured，禁止凭空编造卡数）。
    """
    if not teacher or not teacher.get("min_cards"):
        return None
    try:
        c = int(cards or 0)
    except Exception:
        return f"cards 不可解析（{cards!r}）"
    mc = int(teacher["min_cards"])
    if c < mc:
        return f"OPD 卡数不足：cards={c} < feasibility_model.opd_teacher.min_cards={mc}"
    return None


def opd_pending_unmarked(teacher, row):
    """★ 未定态断言：`measured` 非 True 且 `min_cards` 未声明 ⇒ 该档**必须带未定标记**（`cards_pending`），
    ⛔ 不允许"看起来已定"地生成（否则会被误当可跑）。返回 True = 缺标记（应 fail-closed）。
    """
    if not teacher:
        return False
    if row.get("availability") == "blocked":
        return False          # ★ blocked 档由 blocked 断言负责（它们本就不得起跑）
    if teacher.get("measured") is True and teacher.get("min_cards"):
        why = opd_cards_shortfall(teacher, row.get("cards"))
        return bool(why) is False and False        # 已定态：由卡数断言负责
    return not bool(row.get("cards_pending"))


def blocked_unmarked(row):
    """★ blocked 档必须有 `blocked_reason`（起跑侧据此拒跑）；缺理由 ⇒ fail-closed。"""
    if row.get("availability") != "blocked":
        return False
    return not bool(str(row.get("blocked_reason") or "").strip())


def _gen_self_test():
    cases = [
        ("① OPD 有 teacher_model_path ⇒ 放行", not opd_teacher_missing("train_method: opd\n  teacher_model_path: /models/Qwen3.8-27B\n")),
        ("② ★ 负例：OPD 缺 teacher ⇒ **必须判缺**（fail-closed）", opd_teacher_missing("train_method: opd\nlora:\n  r: 8\n")),
        ("③ ★ 负例：OPD 空值 teacher ⇒ **必须判缺**", opd_teacher_missing("train_method: opd\n  teacher_model_path:\n")),
        ("④ 非 OPD 档 ⇒ 不判缺", not opd_teacher_missing("train_method: sft\n")),
        ("⑤ 卡数断言：min_cards=2 且 cards=1 ⇒ **必须拒**",
         opd_cards_shortfall({"min_cards": 2}, 1) is not None),
        ("⑥ 卡数断言：min_cards=2 且 cards=2 ⇒ 放行", opd_cards_shortfall({"min_cards": 2}, 2) is None),
        ("⑦ ★ 卡数断言：manifest 未声明 min_cards ⇒ **不放宽也不臆造**（不限制）",
         opd_cards_shortfall({"measured": False}, 1) is None),
        ("⑧ ★ 未定态：实测未完成且未声明 min_cards ⇒ 缺 `cards_pending` **必须判缺**（fail-closed）",
         opd_pending_unmarked({"measured": "partial", "min_cards": None}, {"cards": 4}) is True),
        ("⑩ ★ blocked 档缺 `blocked_reason` ⇒ **必须判缺**（fail-closed）",
         blocked_unmarked({"availability": "blocked"}) is True),
        ("⑪ blocked 档带理由 ⇒ 放行（但仍不得起跑）",
         blocked_unmarked({"availability": "blocked", "blocked_reason": "P-25 实测不可行"}) is False),
        ("⑨ 未定态：带 `cards_pending` ⇒ 放行",
         opd_pending_unmarked({"measured": "partial", "min_cards": None}, {"cards": 4, "cards_pending": True}) is False),
    ]
    ok = True
    for name, good in cases:
        print(f"  [{'PASS' if good else 'FAIL'}] {name}")
        ok = ok and good
    print(f"GEN-SELF-TEST {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


def _load_scalars(path):
    """极简 YAML 标量加载（覆盖本生成器输出的形态）：单引号=字面量（'' -> '），双引号=转义，plain=字面量。

    ★ 验收以**加载值**为准（⛔ 不看文件文本写了几个反斜杠）。
    """
    out = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        m = re.match(r"^\s*-\s+(.*\S)\s*$", line)
        if not m:
            continue
        v = m.group(1)
        if v.startswith("'") and v.endswith("'"):
            out.append(v[1:-1].replace("''", "'"))
        elif v.startswith('"') and v.endswith('"'):
            out.append(v[1:-1].encode().decode("unicode_escape"))
        else:
            out.append(v)
    return out


def uniform_map(data):
    return {it["key"]: it["value"] for it in (data.get("uniform") or {}).get("items", [])}


def yaml_of(row, uni, model_root, data_root, lora_backends=None, teacher=None):
    """渲染一个档位的 yaml 文本（确定性：固定键序、无时钟/环境）。"""
    tier = row["tier"]
    tuner = row.get("tuner_type")
    backend = row.get("backend")
    L = [HEADER]
    L.append(f"train_method: {row.get('train_method')}\n")
    L.append(f"backend: {'msswift' if backend == 'ms-swift' else backend}\n")
    L.append(f"tuner_type: {tuner}\n")

    L.append("model:\n")
    L.append(f"  model_path: {model_root.rstrip('/')}/{row.get('model_path')}\n")
    L.append(f"  torch_dtype: {uni.get('torch_dtype')}\n")
    L.append(f"  gradient_checkpointing: {uni.get('gradient_checkpointing')}\n")

    L.append("data:\n")
    L.append(f"  train_path: {data_root.rstrip('/')}/{row.get('train_path')}\n")
    L.append(f"  max_prompt_length: {uni.get('max_prompt_length')}\n")

    L.append("training:\n")
    L.append(f"  output_dir: /out/{tier}\n")
    L.append(f"  run_name: {tier}\n")
    L.append("  overwrite_output_dir: true\n")
    L.append(f"  seed: {uni.get('seed')}\n")
    L.append(f"  max_epochs: {row.get('max_epochs')}\n")
    L.append(f"  learning_rate: {row.get('lr')}\n")
    L.append(f"  gradient_accumulation_micro_batches: {uni.get('gradient_accumulation_micro_batches')}\n")
    L.append(f"  save_checkpoint_every_epoch: {uni.get('save_checkpoint_every_epoch')}\n")
    L.append(f"  save_steps: {row.get('save_steps')}\n")

    # ★ OPD：教师身份是**已 pin 的事实**（manifest `feasibility_model.opd_teacher`）⇒ 必须落进 yaml。
    #   ★ 配置值 = **容器内路径**（`/models/Qwen3.8-27B`）；**挂载源用宿主路径**（起跑侧决定）——两者别混。
    if str(row.get("train_method") or "").lower() == "opd" and teacher:
        L.append("distill:\n")
        _parts = teacher.get("path_parts") or []
        _cp = "/" + "/".join(str(x) for x in _parts if x) if _parts else ""
        L.append(f"  teacher_model_path: {_cp}\n")
        if teacher.get("offload"):
            L.append("  offload_teacher_model: true\n")
    if row.get("train_method") == "graspo" and uni.get("reward.kind"):
        L.append("reward:\n")
        L.append(f"  kind: {uni.get('reward.kind')}\n")

    if tuner == "lora":
        # ★ 按 backend 分流：native 用 preset 名；ms-swift 不认 preset 名（另一套语汇）⇒ 用同覆盖集的显式模块名列表
        lb = (lora_backends or {})
        if backend == "ms-swift":
            expr = (lb.get("msswift") or {}).get("expression") or []
        else:
            expr = (lb.get("native") or {}).get("expression") or []
        if not expr:
            expr = json.loads(row.get("target_modules") or "null") or []
        L.append("lora:\n")
        L.append(f"  r: {uni.get('lora_r')}\n")
        L.append(f"  alpha: {uni.get('lora_alpha')}\n")
        L.append(f"  dropout: {uni.get('lora_dropout')}\n")
        L.append("  target_modules:\n")
        for t in expr:
            ent = str(t)
            # 正则/含特殊字符的条目加单引号，避免 YAML 误读（内部无单引号）
            ent = f"'{ent}'" if any(c in ent for c in "^|*\\:") else ent
            L.append(f"    - {ent}\n")

    if backend == "native":
        L.append("native:\n")
        L.append(f"  tp_size: {row.get('tp_size')}\n")
        L.append(f"  dp_size: {row.get('dp_size')}\n")
        L.append(f"  pp_size: {row.get('pp_size')}\n")
        L.append(f"  micro_batch_size: {uni.get('micro_batch_size')}\n")
        if row.get("allow_unverified_pp_rollout") not in (None, "", "—"):
            L.append(f"  allow_unverified_pp_rollout: {row.get('allow_unverified_pp_rollout')}\n")
    elif backend == "ms-swift":
        nn = row.get("nnodes")
        nn = 1 if nn in (None, "", "—") else nn
        L.append("msswift:\n")
        L.append(f"  nproc_per_node: {row.get('nproc_per_node')}\n")
        L.append(f"  nnodes: {nn}\n")
        L.append(f"  per_device_train_batch_size: {uni.get('micro_batch_size')}\n")
        L.append(f"  sequence_parallel_size: {uni.get('sequence_parallel_size')}\n")
        L.append(f"  use_vllm: {uni.get('use_vllm')}\n")
        if row.get("deepspeed") not in (None, "", "—"):
            L.append(f"  deepspeed: {row.get('deepspeed')}\n")
    if uni.get("gpu_monitor.interval_sec"):
        L.append("gpu_monitor:\n")
        L.append("  output_dir: /out/gpu\n")
        L.append(f"  tag: {tier}\n")
        L.append(f"  interval_sec: {uni.get('gpu_monitor.interval_sec')}\n")
    return "".join(L)


def main():
    ap = argparse.ArgumentParser(description="矩阵数据 → 测试 yaml（确定性生成）")
    ap.add_argument("--data", default="docs/capability-matrix.data.json")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--model-root", default="/models")
    ap.add_argument("--data-root", default="/work/elam-v5")
    ap.add_argument("--only", default=None, help="逗号分隔档号；缺省=全部 applicable 档")
    ap.add_argument("--check-determinism", action="store_true", help="生成两次并逐字节比对（幂等自证）")
    ap.add_argument("--check-dir", default=None, help="与既有冻结 yaml 目录做差异归因")
    ap.add_argument("--self-test", action="store_true", help="OPD 教师断言自检（含缺值/空值负例）")
    ap.add_argument("--verify-loadable", action="store_true",
                    help="按【加载值】验收：正则项加载后须为单反斜杠、可编译、结构符合收紧版")
    a = ap.parse_args()

    if getattr(a, "self_test", False):
        return _gen_self_test()

    data = json.loads(Path(a.data).read_text(encoding="utf-8"))
    uni = uniform_map(data)
    teacher = (data.get("meta") or {}).get("distill_teacher")
    lora_backends = (data.get("meta") or {}).get("lora_target_backends") or {}
    only = set(x.strip() for x in a.only.split(",")) if a.only else None
    rows = [r for r in data["rows"] if r.get("applicable") and (only is None or r["tier"] in only)]
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    written = {}
    for r in rows:
        text = yaml_of(r, uni, a.model_root, a.data_root, lora_backends, teacher)
        p = out / f"{r['tier']}.yaml"
        p.write_text(text, encoding="utf-8")
        written[r["tier"]] = p
        print(f"wrote {p}  max_epochs={r.get('max_epochs')}  derived_steps={r.get('derived_steps')}  save_steps={r.get('save_steps')}")

    if a.verify_loadable:
        bad = []
        for tier, pth in sorted(written.items()):
            for v in _load_scalars(pth):
                if "language_model" not in v and "visual" not in v:
                    continue
                if "\\\\" in v:                      # 加载值出现双反斜杠 ⇒ 过度转义
                    bad.append((tier, "加载值含双反斜杠（过度转义）"))
                try:
                    re.compile(v)
                except re.error as e:
                    bad.append((tier, f"正则不可编译: {e}"))
                if "visual\\.(blocks\\.\\d+" not in v and "language_model" in v:
                    bad.append((tier, "未采用收紧版视觉枚举（仍为过宽写法）"))
        print(f"LOADABLE {'PASS' if not bad else 'FAIL'}（{len(written)} 档；问题 {bad}）")
        if bad:
            return 1

    _bl = [r["tier"] for r in rows if blocked_unmarked(r)]
    if _bl:
        print(f"FAIL-CLOSED：以下档标 blocked 却无理由 ⇒ 拒绝产出：{_bl}")
        return 1

    _unmarked = [r["tier"] for r in rows
                 if str(r.get("train_method") or "").lower() == "opd" and opd_pending_unmarked(teacher, r)]
    if _unmarked:
        print(f"FAIL-CLOSED：以下 OPD 档处于「教师卡数未定」却缺 cards_pending 标记 ⇒ 拒绝产出：{_unmarked}")
        return 1

    _short = []
    for r in rows:
        if str(r.get("train_method") or "").lower() == "opd":
            why = opd_cards_shortfall(teacher, r.get("cards") or r.get("cards_total"))
            if why:
                _short.append((r["tier"], why))
    if _short:
        print(f"FAIL-CLOSED：OPD 卡数低于 manifest 声明 ⇒ 拒绝产出：{_short}")
        return 1

    _no_teacher = [t for t, pth in sorted(written.items()) if opd_teacher_missing(Path(pth).read_text(encoding="utf-8"))]
    if _no_teacher:
        print(f"FAIL-CLOSED：以下 OPD 档缺教师（teacher_model_path 空 且无 teacher_model_server）⇒ 拒绝产出：{_no_teacher}")
        return 1

    if a.check_determinism:
        bad = []
        for r in rows:
            again = yaml_of(r, uni, a.model_root, a.data_root, lora_backends, teacher)
            if again != written[r["tier"]].read_text(encoding="utf-8"):
                bad.append(r["tier"])
        print(f"DETERMINISM {'PASS' if not bad else 'FAIL'}（{len(rows)} 档；不一致 {bad}）")
        if bad:
            return 1

    if a.check_dir:
        base = Path(a.check_dir)
        print(f"\n=== 与既有冻结 yaml 的差异归因（{base}）===")
        for tier, p in sorted(written.items()):
            old = base / f"{tier}.yaml"
            if not old.exists():
                print(f"  {tier}: 旧目录无此档")
                continue
            d_old = flatten(old.read_text(encoding="utf-8"))
            d_new = flatten(p.read_text(encoding="utf-8"))
            keys = sorted(set(d_old) | set(d_new))
            rows_diff = [(k, d_old.get(k, "(缺)"), d_new.get(k, "(缺)")) for k in keys if d_old.get(k) != d_new.get(k)]
            print(f"  {tier}: 字段级差异 {len(rows_diff)} 处")
            for k, o, n in rows_diff:
                print(f"      {k}: {o!r} -> {n!r}   【{attribute(k)}】")
    return 0


def flatten(text):
    """把 yaml 文本摊平成 {key.path: value}（列表项合并为逗号串）；忽略注释与空行。"""
    d, stack = {}, []
    for line in text.splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        indent = len(line) - len(line.lstrip())
        body = line.strip()
        if body.startswith("- "):
            key = ".".join(x[1] for x in stack)
            d[key] = (d.get(key, "") + "," + body[2:].strip()).strip(",")
            continue
        k, _, v = body.partition(":")
        while stack and stack[-1][0] >= indent:
            stack.pop()
        path = ".".join([x[1] for x in stack] + [k])
        if v.strip() == "":
            stack.append((indent, k))
        else:
            d[path] = v.strip()
    return d


#: 差异归因表（逐条对应已登记的口径变更）
ATTRIBUTION = [
    ("lora.target_modules", "LoRA 恢复口径：单 preset → 列表 [language_all_linear, vision_common]（358 实例；122 为签名/模式条目数）"),
    ("lora.target_preset", "LoRA 恢复口径：旧单 preset 已被 target_modules 列表取代"),
    ("lora.r", "LoRA 恢复口径：r 8 → 128"),
    ("lora.alpha", "LoRA 恢复口径：alpha 16 → 256"),
    ("lora.dropout", "LoRA 恢复口径：dropout 0.0 → 0.05"),
    ("training.learning_rate", "lr：LoRA 走恢复值 1.0e-05（旧值为偏离值）"),
    ("training.max_epochs", "epoch 口径（主席定案）：SFT/CPT/OPD=2；GRASPO(RL)=5"),
    ("training.save_steps", "save_steps 重算：= 推导步数 + 1（取消人为步数上限）"),
    ("gpu_monitor", "生成器未覆盖的既有字段（gpu_monitor.*）：矩阵数据未收录 ⇒ 本次生成不含"),
    ("reward", "内容不变（kind=graspo），仅块序前移（生成器固定键序）"),
]


def attribute(key):
    for pref, why in ATTRIBUTION:
        if key == pref or key.startswith(pref + "."):
            return why
    return "未归因（需人工确认）"


if __name__ == "__main__":
    sys.exit(main())
