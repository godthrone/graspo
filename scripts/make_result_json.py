#!/usr/bin/env python3
"""跑次读数 ⇒ **回填口径**的结果 json（`matrix-run/v1`）—— 唯一产出者。

为什么有这件（工作包 B / 2026-10-22，链路侦察断点 B1+B3）：
  · `scripts/gen_tier_json.py` 的产物**结构上过不了回填器**：它写死 `status`（回填器见 `status`
    一律 rc=1 拒收）、缺 `run_result`/`steps_actual`/`elapsed`/`note`、评测三件在**顶层**
    （回填器要 `eval{}`）、`algo_checks`/`inference` 是**自由文本**（回填器要 `checks{bool,bool}`）。
  · `elapsed` / `throughput` 全仓**无产出脚本**，`peak_mem` 与台账里的 `peak_memory_gib` 名字也不对。
  ⇒ 没有一件脚本能产出"可回填"的 json。本文件是那个**薄适配器**：把
    ① runner 的原始实测读数（`<run-dir>/readings.json` / `exit_code` / `elapsed_sec` / `gpu/peak_mem.json`）、
    ② 清单身份（`tests/e2e/matrix68_manifest.json`）+ 数据件该行（**逐项对齐回填器的身份比对**）、
    ③ 可选的评测读数（`--eval-dir` 的 `eval_report.json`，或 `gen_tier_json.py` 的产物）
    组装成回填器入参契约的字段。

契约（逐字段对照 `tools/matrix/backfill_matrix.py`）：
  顶层必填 `schema/tier/params/run_result/steps_actual/elapsed/peak_mem/throughput/effect/note/run_path/provenance`；
  ⛔ **绝对不写 `status`**（是否通过由回填器按四条判据现算）；
  `run_result{exit_code:int, epochs_completed:int|null, epochs_planned:int|null,
  final_ckpt_loadable:bool|null, nan_or_inf_detected:bool|null}`；
  ★ 判据①（2026-10-23 对齐第 2 章「跑满 Epoch」）= 退出码 0 ∧ `epochs_completed ≥ epochs_planned`
  ∧ 检查点**真可加载** ∧ 无 NaN/Inf；`steps_actual` 仍必填但**只作台账**，⛔ 不作通过条件。
    · `epochs_*` 取自实测：`trainer_state.json` 的 `epoch` / `events.jsonl`(epoch_summary) /
      `logging.jsonl`；**取不到写 `null`**（回填器判 unknown ⇒ fail-closed），⛔ 不用步数换算。
    · `final_ckpt_loadable` = **真加载校验**（safetensors 头逐张量偏移 / `.bin`·`.pt` 的
      `torch.load(map_location="meta")` 元数据 / 权重索引逐分片 / LoRA adapter 齐件）；
      某格式无法安全校验 ⇒ `null` + 原因（⛔ 不当 True）。
    · `nan_or_inf_detected` = stdout ＋ 全部 `*.jsonl` 旁路 ＋ `trainer_state` 三处证据源里
      命中 metric 非有限值或 `skipped_nonfinite > 0`；三处都读不到 ⇒ `null`。
  6 个数据列必须"写满"（⛔ 空/`—`/null 会被拒收）；
  `eval{after_acc,base_acc,delta_pp[,teacher_after_acc]}`（判据②）、
  `checks{algo_checks_passed:bool, inference_parseable:bool}`（判据③④）——
  **缺证据就不写这两个键**（⛔ 不用 false/null 冒充"判过了"），回填器会把对应判据判成
  `unknown` 并 fail-closed。
  ★ 判据③④ 的**唯一产出者与渠道**（审查第 2 轮 🟡-A 的落点）：评测跑完后先执行
  `scripts/record_checks.py`，它从 `<EVAL_DIR>/.local/reports/after/eval_report.json` 实测计算
  ④（样本数/请求失败数/逐样本可解析条数/温度 0），并按算法分级产出 ③（SFT/CPT/OPD 结构化置位 +
  依据串；GRASPO 必须有签署的逐条不变式证据）⇒ 写进 `<run-dir>/readings.json` 的
  `checks` 与 `checks_evidence`。本文件只**消费**那份结论（读 `readings.json`），⛔ 不自己判、
  ⛔ 不补 true。缺键时本文件判 rc=4（结构合法但证据不全），回填器保持「待测试」。
  成文渠道与字段格式见 `docs/run-discipline.md` 的「判据③④ 的判定与证据」一节。
  ★ 判据②（**仅 OPD 档**）的教师读数 `eval.teacher_after_acc`：唯一渠道 = `<run-dir>/readings.json`
  的顶层 `teacher_after_acc`（产出者同上，从 `<EVAL_DIR>/.local/reports/teacher/eval_report.json`
  按与学生侧同一口径实测）。本文件只**透传**，⛔ 不自己算、⛔ 缺就不写（OPD 档缺 ⇒ rc=4，
  回填器 `judge_effect` 判 unknown ⇒ fail-closed 保持「待测试」）。见 `docs/run-discipline.md` §20。

落盘前**必经回填器自己的校验器**（动态 import `backfill_matrix.validate_run`）自检：
字段/身份/口径任何一条不过 ⇒ **不宣告成功**，rc 非 0，并在 stdout 逐条列出问题。
缺读数（如没跑评测、日志缺失）⇒ 该字段写 `null` 并在 `note` 里写明**具体缺哪一项、为什么**
（⛔ 不估算、不填充数）；这种 json 会被回填器拒收 —— 这是**设计如此**（fail-closed）。

用法：
    # ① 评测跑完 ⇒ 先产出判据③④ 证据（唯一产出者；写 <run-root>/M001/readings.json）
    python3 scripts/record_checks.py --tier M001 --run-dir <run-root>/M001 \\
        --eval-dir <EV> [--algo-checks-dir <DIR>]
    # ② 再生成回填口径的结果 json（本文件只消费 readings.json 的 checks{}）
    python3 scripts/make_result_json.py --tier M001 --run-dir <run-root>/M001 \\
        --out <run-root>/M001/M001.json [--eval-dir <EV>] [--tier-json <gen_tier_json 产物>] \\
        [--manifest tests/e2e/matrix68_manifest.json] [--data docs/capability-matrix.data.json] \\
        --collect-dir <run-root>/collect
    # ⇒ 缺 ③④ 证据时 rc=4（结果 json 照落盘供审计，但回填器会 fail-closed 保持「待测试」）

★ **回填入口 = `<run-root>/collect`**（扁平 `<档号>.json`）：`--runs-dir` 只吃扁平目录，
  而 runner 的跑次归档在 `<run-root>/<档号>/<档号>.json`（嵌套，回填器不认）。
  ⇒ runner 每档都带 `--collect-dir <run-root>/collect`；回填命令：
        bash tools/matrix/backfill_dated.sh --runs-dir <run-root>/collect
  `--collect-dir` 的落盘时机：结构校验通过后（含"证据不全"的 rc=4 件）才写扁平副本，
  结构不合法（rc=1）不写 —— ⛔ 不让不合契约的件进回填目录。

★ **6 个实验数据列只放"读数/结论"，不放过程说明**：`peak_mem` 列只有 `<数值> GiB`（或
  `null`）；测量方式、证据来源一律写进 `note` 列，且用中性措辞 —— 该列会随回填进入人读件，
  ⛔ 不得出现内部路径、内部工位名、内部行话（人读件的脱敏/禁词由机核现验）。

退出码：0 = 已通过回填器校验器（可回填）；4 = 结构合法但**证据不全**（已落盘、供审计，
        回填器会拒收）；2 = 用法/输入错误；1 = 校验器判不通过（字段/身份口径错）。
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import re
import sys
import zipfile
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "matrix-run/v1"

#: 检查点权重文件名（与 `scripts/collect_results.py::FINAL_CKPT_WEIGHT_FILENAMES` 同一张表：
#: 覆盖 native 分片 / HF 分片 / ms-swift peft adapter 三种布局）。
CKPT_WEIGHT_NAMES = (
    "adapter_model.safetensors", "adapter_model.bin",
    "model.safetensors", "pytorch_model.bin", "pytorch_model.bin.index.json",
    "model.safetensors.index.json",
)
CKPT_WEIGHT_GLOBS = ("model-*.safetensors", "rank_*.pt", "*.safetensors")

#: 「实跑步数」的读取口径 —— **四种实盘形态**（2026-10-26 修 D2；M016 真实日志驱动，
#: fixture 见 `tests/scripts/fixtures/m016_stdout_excerpt.log`）：
#:   ① `'global_step/max_steps': '6400/6400'` —— ms-swift 指标 dict。**值是带引号的字符串**
#:      （单/双引号皆可）；旧正则 `[:=]\s*(\d+)` 卡在值前面那个 `'` 上 ⇒ M016 的
#:      stdout.log 里 1282 行**一条都认不出** ⇒ `steps_actual=None` ⇒ 回填器 fail-closed，
#:      训练已成功的档进不了表。
#:   ② tqdm 进度行 `Train: 100%|██████████| 6400/6400 [14:27:53<00:00, 4.95s/it]`
#:      （锚成"竖线 + 计数 + 方括号"的形状：⛔ 不裸扫 `\d+/\d+`，否则日期/速率会假阳性）。
#:   ③ `'global_step': '200'` / `"global_step": 200` —— 同一 dict 家族 / JSON 家族
#:      （值同样是"字符串或数字"；`logging.jsonl` 末行即 `"global_step": 6400`）。
#:   ④ `trainer_state.json` 的 `"global_step": 6400`（由 :func:`parse_steps_evidence`
#:      在跑次目录里逐件读取 —— 与 ③ 共用 `GLOBAL_STEP_RE`）。
#:   ★ 既有语义不变：`global_step/max_steps` 取**分子**（= 当前步）；多源并存时取**最大**。
GLOBAL_STEP_RE = re.compile(r"['\"]?global_step['\"]?\s*[:=]\s*['\"]?(\d+)")
GLOBAL_STEP_SLASH_RE = re.compile(
    r"global_step\s*/\s*max_steps['\"]?\s*[:=]\s*['\"]?(\d+)(?:\s*/\s*(\d+))?")
#: ② tqdm 进度条里的「已跑/总数」。**必须锚住训练条**（`Train:` 标签 + `|计数 [` 框线）：
#:   日志里还有下载/存盘等**无关进度条**（M016 实测 stdout.log 里有 7724 处 `|N/M [`，
#:   其中绝大多数来自 `Train:` 之外）——裸扫 `|\s*\d+/\d+\s*\[` 会把无关计数当步数，
#:   而口径是"多源取最大" ⇒ 一个偏大的假读数会**静默污染** steps_actual。
#:   ⛔ 无标签的 HF 风格条（`  5%|▌ | 320/6400 [`）不取：那种情形有 ①②③④ 里的
#:   `global_step` / `trainer_state.json` 兜底，宁缺勿错（fail-closed）。
TQDM_STEP_RE = re.compile(
    r"(?:^|[\r\n])\s*(?:Train|Training):[^\r\n]*?\|\s*(\d+)\s*/\s*(\d+)\s*\[")
#: 同一读数出现在多处（且数值相等）时，**取更权威的形态**记 provenance：
#: 显式 `global_step/max_steps` dict > `trainer_state.json` 的 `global_step` > tqdm 进度条。
_STEP_KIND_RANK = {"global_step/max_steps": 3, "trainer_state.global_step": 2,
                   "global_step": 2, "tqdm_progress": 1}
#: ★ NaN/Inf 的唯一判定正则（判据① 的第四项）：**metric 字段取非有限值**才算命中。
#:   `loss` / `grad_norm` / `reward` / `logits` × `nan|-inf|inf|infinity`（大小写不敏感）。
#:   ⛔ 不裸扫 `nan`/`inf` 单词：日志里的 "infinite"、路径名会假阳性。
NANINF_RE = re.compile(r"(?i)['\"]?(loss|grad_norm|reward|logits)['\"]?\s*[:=]\s*"
                       r"(nan|-?inf(?:inity)?)\b")
#: ★ 被跳过的非有限梯度步数（native 旁路 `skipped_nonfinite` / ms-swift
#:   `nonfinite_grad_step_count`）：**> 0 才算命中**（字段出现在盘上且为 0 是"正面证据"，
#:   ⛔ 不得按"字段出现过"判异常）。
NANINF_SKIP_RE = re.compile(
    r"['\"]?(skipped_nonfinite|nonfinite_grad_step_count)['\"]?\s*[:=]\s*(-?\d+)")

#: ★ 判据① 的 epoch 读数容差（进度读数是浮点，"8.0 epoch 完成"不能因 7.999999 判成 7）。
EPOCH_TOL = 1e-9

DATA_COLUMNS = ("peak_mem", "steps_actual", "elapsed", "throughput", "effect", "note")


# ── 小工具 ───────────────────────────────────────────────────────────────────
def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:                      # pragma: no cover
        raise SystemExit(f"FATAL: 无法加载 {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def read_json(path: Path, default=None):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return default


def _basis_ids() -> tuple[str, str]:
    """判据③ 两个**依据标识**的唯一真相源 = `scripts/record_checks.py`。

    ⛔ 不在此另写一份字符串：产出端（record_checks）与消费端（本文件的备注列）认的必须是
    同一个 id，否则"有没有签署"两边会不一致。导入失败 ⇒ 返回空串 ⇒ 备注列按
    "证据不完整"处理（fail-safe，⛔ 不猜成"已通过"）。
    """
    try:
        if str(ROOT / "scripts") not in sys.path:
            sys.path.insert(0, str(ROOT / "scripts"))
        import record_checks  # noqa: PLC0415

        return record_checks.GRASPO_ALGO_BASIS, record_checks.PUBLIC_ALGO_BASIS
    except Exception:  # noqa: BLE001 - 任何导入失败都退化为"认不出签署"
        return "", ""


def read_int_file(path: Path) -> int | None:
    try:
        return int(path.read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
        return None


def read_float_file(path: Path) -> float | None:
    try:
        return float(path.read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
        return None


def _iter_jsonl(path: Path):
    """逐行读 jsonl（生成 `(行号, 对象)`）：**行号是证据坐标**（报告要能指到哪一行）。"""
    try:
        with path.open("r", encoding="utf-8", errors="replace") as fh:
            for lineno, line in enumerate(fh, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(obj, dict):
                    yield lineno, obj
    except OSError:                                           # pragma: no cover - 权限/竞态
        return


# ── 证据抽取（全部来自**实测**产物；抽不到 ⇒ None + 原因）───────────────────────
def find_checkpoint_dirs(run_dir: Path) -> list[Path]:
    """发现本档的检查点目录（`checkpoint-*` / `final`），按步号降序。"""
    found = [p for p in run_dir.rglob("checkpoint-*") if p.is_dir()]
    found += [p for p in run_dir.rglob("final") if p.is_dir()]

    def step_of(p: Path) -> int:
        tail = p.name.rsplit("-", 1)[-1]
        return int(tail) if tail.isdigit() else 10 ** 9

    return sorted(set(found), key=step_of, reverse=True)


class _LayoutError(Exception):
    """safetensors 头部不可用（异常消息即人读原因）。"""


def _safetensors_layout(path: Path) -> tuple[dict[str, tuple[int, int]], int, int]:
    """**纯 stdlib** 解析 safetensors 布局 ⇒ `({张量名: (绝对起, 绝对止)}, 数据区起点, 文件大小)`。

    格式 = `[8 B 小端头长度 N][N B JSON 头][权重数据区]`；头里每个张量带 `data_offsets`。
    逐张量校验"偏移落在数据区内"，越界 ⇒ :class:`_LayoutError`。⛔ 不读权重数据。
    **解析只此一处**（§1.4）：`_safetensors_probe`（结构自洽）与 `_lora_nonzero_probe`
    （内容非零）都从这里取偏移，⛔ 不各抄一份。
    """
    try:
        size = path.stat().st_size
    except OSError as exc:                                    # pragma: no cover - 权限/竞态
        raise _LayoutError(f"{path.name} 不可读：{exc!r}") from exc
    if size < 9:
        raise _LayoutError(f"{path.name} 只有 {size} 字节"
                           "（装不下头长度 + JSON 头 ⇒ 不是合法 safetensors）")
    try:
        with path.open("rb") as fh:
            n = int.from_bytes(fh.read(8), "little")
            if n <= 0 or 8 + n > size:
                raise _LayoutError(f"{path.name} 头长度 {n} 与文件大小 {size} 不符")
            raw = fh.read(n)
    except OSError as exc:                                    # pragma: no cover
        raise _LayoutError(f"{path.name} 不可读：{exc!r}") from exc
    try:
        hdr = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise _LayoutError(f"{path.name} 的 safetensors 头不是 JSON") from exc
    if not isinstance(hdr, dict):
        raise _LayoutError(f"{path.name} 的 safetensors 头不是对象")
    tensors = {k: v for k, v in hdr.items() if k != "__metadata__"}
    if not tensors:
        raise _LayoutError(f"{path.name} 的 safetensors 头里没有任何张量")
    data_start = 8 + n
    body = size - data_start
    index: dict[str, tuple[int, int]] = {}
    for name, meta in tensors.items():
        off = meta.get("data_offsets") if isinstance(meta, dict) else None
        if (not isinstance(off, list) or len(off) != 2
                or not all(isinstance(x, int) and not isinstance(x, bool) for x in off)):
            raise _LayoutError(f"{path.name} 张量 {name} 的 data_offsets 不合法：{off!r}")
        b, e = off
        if not (0 <= b <= e <= body):
            raise _LayoutError(f"{path.name} 张量 {name} 的偏移 [{b},{e}] 越出数据区"
                               f"（{body} B）")
        index[name] = (data_start + b, data_start + e)
    return index, data_start, size


def _safetensors_probe(path: Path) -> tuple[bool | None, str, list[str]]:
    """**真解析** safetensors 头（⛔ 不读权重数据、⛔ 不依赖 safetensors/torch 库）。

    据此可判"这个文件是不是一份完整的 safetensors"：头长度自洽 ∧ 头是 JSON ∧ 有张量 ∧
    每个张量的偏移都落在数据区内。7 字节的垃圾文件在这里第一道就判 False。
    """
    try:
        index, data_start, size = _safetensors_layout(path)
    except _LayoutError as exc:
        return False, str(exc), []
    return (True, f"{path.name}：{len(index)} 个张量 / 头 {data_start - 8} B / "
                  f"数据区 {size - data_start} B 全部自洽", list(index))


def _lora_nonzero_probe(path: Path) -> tuple[bool | None, str]:
    """LoRA 权重是否**真被更新**：`adapter_model.safetensors` 里 `lora_B` 张量有非零字节。

    为什么口径如此：peft / ms-swift 的 `lora_B` 初值**恒为 0**（初始化即零矩阵），训练一步
    没走则整个 `lora_B` 逐字节全零。⇒ "存在非零字节"即"优化器真的写过权重"，是**内容级**
    证据（判据① 要求"算法机制真实生效、不是空转"），⛔ 不用文件大小/时间戳这类弱指纹顶替。
    按偏移分块读、**命中即停**。返回 `None` = 读不到 / 无 lora_B 张量（取证缺口，⛔ 不当 False）。
    """
    try:
        index, _start, _size = _safetensors_layout(path)
    except _LayoutError as exc:
        return None, f"{path.name} 布局不可解析 ⇒ 无法核 lora_B 非零（{exc}）"
    keys = [k for k in index if "lora_b" in k.lower()]
    if not keys:
        return None, f"{path.name} 里没有 lora_B 张量（实测 {len(index)} 个张量名）"
    chunk = 1 << 20
    seen_nonzero = False
    try:
        with path.open("rb") as fh:
            for key in keys:
                begin, end = index[key]
                fh.seek(begin)
                remaining = end - begin
                while remaining > 0:
                    buf = fh.read(min(chunk, remaining))
                    if not buf:
                        return None, f"{path.name} 的 {key} 读数提前结束（文件被截断？）"
                    if buf != b"\x00" * len(buf):
                        seen_nonzero = True
                        break
                    remaining -= len(buf)
                if seen_nonzero:
                    break
    except OSError as exc:                                    # pragma: no cover
        return None, f"{path.name} 不可读：{exc!r}"
    if seen_nonzero:
        return True, (f"{path.name}：{len(keys)} 个 lora_B 张量里实测到非零权重"
                      "（LoRA B 初值恒为 0 ⇒ 权重真被优化器写过）")
    return False, (f"{path.name}：{len(keys)} 个 lora_B 张量**逐字节全零**"
                   "⇒ 训练未真正更新权重（机制未生效）")


def _torch_meta_probe(path: Path) -> tuple[bool | None, str, list[str]]:
    """`.bin` / `.pt` 分片的元数据校验（⛔ 不物化权重、⛔ 不上 GPU）。

    两道：① `torch.save` 的 zip 归档结构（`data.pkl` 存在）；② 有 torch 时再
    `torch.load(map_location="meta", weights_only=True, mmap=True)` 把元数据解析出来。
    ② 因为 `weights_only=True` ⇒ 只还原张量/基础类型，不执行 pickle 任意代码；
    `map_location="meta"` ⇒ 只建元数据、不分配存储。取不到 torch ⇒ **null**（fail-closed）。
    """
    try:
        size = path.stat().st_size
    except OSError as exc:                                    # pragma: no cover
        return False, f"{path.name} 不可读：{exc!r}", []
    if size < 8:
        return False, f"{path.name} 只有 {size} 字节（装不下 torch.save 归档头）", []
    zip_why = None
    try:
        with zipfile.ZipFile(path) as zf:
            names = zf.namelist()
        if not any(n == "data.pkl" or n.endswith("/data.pkl") for n in names):
            zip_why = f"{path.name} 是 zip 归档但缺 data.pkl（不是 torch.save 产物）"
    except zipfile.BadZipFile as exc:
        zip_why = f"{path.name} 不是 zip 归档（{exc}）"
    except OSError as exc:                                    # pragma: no cover
        return False, f"{path.name} 不可读：{exc!r}", []
    try:
        import torch  # noqa: PLC0415 - 可选依赖，缺则 fail-closed
    except Exception:                                         # noqa: BLE001
        return None, (f"{path.name}：torch 不可用 ⇒ 无法解析元数据"
                      + (f"（{zip_why}）" if zip_why else "")), []
    try:
        obj = torch.load(str(path), map_location="meta", weights_only=True, mmap=True)
    except Exception as exc:                                  # noqa: BLE001
        # 旧版（非 zip）序列化不支持 mmap ⇒ 只在这一种情况下退一步再试一次（仍不物化权重）
        if "mmap" not in str(exc):
            return False, f"{path.name} 元数据解析失败：{exc!r}", []
        try:
            obj = torch.load(str(path), map_location="meta", weights_only=True)
        except Exception as exc2:                             # noqa: BLE001
            return False, f"{path.name} 元数据解析失败：{exc2!r}", []
    if not isinstance(obj, dict) or not obj:
        return False, f"{path.name} 的元数据不是非空字典（实际 {type(obj).__name__}）", []
    return True, (f"{path.name}：元数据 {len(obj)} 项可解析（meta 加载，未物化权重"
                  + ("；非 zip 旧序列化" if zip_why else "") + "）"), list(obj)


def _probe_weight_file(path: Path) -> tuple[bool | None, str, list[str]]:
    low = path.name.lower()
    if low.endswith(".safetensors"):
        return _safetensors_probe(path)
    if low.endswith((".bin", ".pt", ".pth")):
        return _torch_meta_probe(path)
    return None, f"{path.name}：未知权重格式（无法安全校验）", []


def _probe_index(index: Path) -> tuple[bool | None, str]:
    """权重索引（`*.index.json`）⇒ 逐分片**打开并解析**（⛔ 不整模型加载）。"""
    doc = read_json(index, None)
    if not isinstance(doc, dict):
        return False, f"{index.name} 不可解析或不是 JSON 对象"
    weight_map = doc.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        return False, f"{index.name} 缺 weight_map（权重索引不完整）"
    shards = sorted({str(v) for v in weight_map.values()})
    if not shards:
        return False, f"{index.name} 的 weight_map 里没有任何分片"
    verdicts: list[bool | None] = []
    notes: list[str] = []
    for shard in shards:
        sp = index.parent / shard
        if not sp.is_file():
            return False, f"{index.name} 引用的分片 {shard} 不存在"
        if sp.stat().st_size <= 0:
            return False, f"分片 {shard} 是空文件"
        ok, why, _keys = _probe_weight_file(sp)
        verdicts.append(ok)
        notes.append(why)
        if ok is False:
            return False, why
    if any(v is None for v in verdicts):
        return None, "；".join(n for v, n in zip(verdicts, notes) if v is None)
    return True, (f"{index.name}：weight_map {len(weight_map)} 项 ⇒ {len(shards)} 个分片逐一可解析"
                  f"（{notes[0]}）")


def inspect_checkpoint(path: Path) -> tuple[bool | None, str]:
    """检查点"**真可加载**"的唯一判定（判据① 第三项）。

    口径（2026-10-23 收紧，取代旧的"目录里有非空文件即 True"）：
      · PEFT/LoRA adapter 布局 ⇒ `adapter_config.json` 可解析 ＋ adapter 权重可解析且
        **含 LoRA 张量名**；
      · 有 `*.index.json` ⇒ 读 `weight_map`，逐分片真解析（分片缺失/空/头坏 ⇒ False）；
      · 直接权重件（HF 单件 / native `rank_*.pt` / `model-*.safetensors` 分片）⇒ 逐件真解析；
      · 某格式无法安全校验（未知后缀 / 无 torch）⇒ **None**（fail-closed，⛔ 不许当 True）。
    ⛔ 不下载/不加载整模型到显存：safetensors 只读头，`.bin`/`.pt` 走 meta 加载。
    """
    # ── ① PEFT/LoRA adapter 布局 ──────────────────────────────────────────
    cfg = path / "adapter_config.json"
    adapter_weights = [path / n for n in ("adapter_model.safetensors", "adapter_model.bin")
                       if (path / n).is_file() and (path / n).stat().st_size > 0]
    if cfg.is_file() or adapter_weights:
        conf = read_json(cfg, None)
        if not isinstance(conf, dict) or not conf:
            return False, ("adapter_config.json " +
                           ("不可解析" if conf is None else "为空") + "（LoRA adapter 不合契约）")
        if not adapter_weights:
            return False, ("有 adapter_config.json 但缺 adapter 权重"
                           "（adapter_model.safetensors/.bin）")
        keys: list[str] = []
        for w in adapter_weights:
            ok, why, ks = _probe_weight_file(w)
            if ok is not True:
                return (False if ok is False else None), why
            keys.extend(ks)
        if not any("lora" in k.lower() for k in keys):
            return False, f"adapter 权重里没有 LoRA 张量名（实测 {len(keys)} 个张量名）"
        return True, (f"PEFT adapter：adapter_config.json 可解析（r={conf.get('r')}）＋ "
                      f"{adapter_weights[0].name} 的 {len(keys)} 个张量名可解析且含 LoRA 权重")

    # ── ② 权重索引：分片齐不齐由 weight_map 说了算 ─────────────────────────
    for name in ("model.safetensors.index.json", "pytorch_model.bin.index.json"):
        idx = path / name
        if idx.is_file():
            return _probe_index(idx)

    # ── ③ 直接权重件（HF 单件 / native 分片）：文件名表与 glob 一律取本文件顶部的
    #       `CKPT_WEIGHT_NAMES` / `CKPT_WEIGHT_GLOBS`（§1.4：⛔ 不在函数里另抄一份）
    files: list[Path] = [path / n for n in CKPT_WEIGHT_NAMES
                         if not n.endswith(".index.json") and (path / n).is_file()]
    for pattern in CKPT_WEIGHT_GLOBS:
        files += sorted(p for p in path.glob(pattern) if p.is_file())
    files = [p for p in dict.fromkeys(files) if p.stat().st_size > 0]
    if not files:
        return False, ("检查点目录内没有非空权重文件"
                       "（model.safetensors / *.index.json / rank_*.pt 皆无）")
    unknown: list[str] = []
    for f in files:
        ok, why, _keys = _probe_weight_file(f)
        if ok is False:
            return False, why
        if ok is None:
            unknown.append(why)
    if unknown:
        return None, "；".join(unknown)
    return True, f"{len(files)} 个权重文件逐一可解析（{files[0].name}：头/元数据自洽）"


def ckpt_loadable(run_dir: Path) -> tuple[bool | None, str, str | None]:
    """最终检查点是否**真可加载**（= :func:`inspect_checkpoint` 的结论 + 定位结果）。"""
    dirs = find_checkpoint_dirs(run_dir)
    if not dirs:
        return None, "未找到任何 checkpoint-*/ 或 final/ 目录（证据缺失）", None
    latest = dirs[0]
    ok, why = inspect_checkpoint(latest)
    return ok, f"最新检查点 {latest.name}：{why}", str(latest)


def _scan_steps_text(text: str, source: str) -> list[dict]:
    """在一段文本里逐行找步数读数 ⇒ 候选列表（`value` / `line` / `kind` / `raw`）。

    逐行（而不是整篇 `finditer`）的唯一原因：provenance 要报**哪一行**（审计可复现）。
    三种形态共用一条"预筛"（行里既没有 `global_step` 也没有 tqdm 竖线就跳过），
    否则 19 MB / 19 万行的真日志会被三份正则各扫一遍。
    """
    out: list[dict] = []
    # ⚠ 只按 `\n` 切行（⛔ `splitlines()`）：tqdm 用 `\r` 刷新进度帧，`splitlines()` 会把
    #   同一**物理行**切成几十份 ⇒ provenance 行号与 grep/sed 口径不符、肉眼复核不到。
    for lineno, line in enumerate(text.split("\n"), 1):
        if "global_step" in line:
            for m in GLOBAL_STEP_SLASH_RE.finditer(line):
                out.append({"value": int(m.group(1)), "line": lineno,
                            "kind": "global_step/max_steps", "raw": m.group(0)[:60],
                            "max_steps": int(m.group(2)) if m.group(2) else None,
                            "source": source})
            for m in GLOBAL_STEP_RE.finditer(line):
                out.append({"value": int(m.group(1)), "line": lineno,
                            "kind": "global_step", "raw": m.group(0)[:60],
                            "source": source})
        elif "Train:" in line or "Training:" in line:
            for m in TQDM_STEP_RE.finditer(line):
                out.append({"value": int(m.group(1)), "line": lineno,
                            "kind": "tqdm_progress", "raw": m.group(0)[:60],
                            "source": source})
    return out


def _steps_from_stdout(stdout_log: Path) -> tuple[int | None, str, list[dict]]:
    """:func:`parse_steps` 的实现体（带 provenance）。"""
    if not stdout_log.is_file():
        return None, "训练日志缺失 ⇒ 无法取实跑步数", []
    # ⚠ 按**字节**读再解码（⛔ `read_text()`）：文本模式会把 `\r` 规范化成 `\n`，
    #   行号就与 grep/sed 口径不符（provenance 里的行号必须能被人直接复核）。
    text = stdout_log.read_bytes().decode("utf-8", "replace")
    found = _scan_steps_text(text, stdout_log.name)
    if not found:
        return None, "训练日志里没有步数读数（后端未打印/训练未进入优化循环）", []
    best = max(found, key=lambda c: (c["value"], _STEP_KIND_RANK.get(c["kind"], 0)))
    origin = f"{best['source']}:{best['line']}（形态 {best['kind']}，原文 {best['raw'].strip()!r}）"
    return best["value"], f"训练日志的最大步数读数 = {best['value']}（取自 {origin}）", [best]


def parse_steps(stdout_log: Path) -> tuple[int | None, str]:
    """实跑优化步数：日志里 `global_step` 的**最大值**（与台账同一口径）。

    认四种实盘形态（见 `GLOBAL_STEP_*` / `TQDM_STEP_*` 的注释）。返回值里的 `why`
    **带 provenance**（哪个文件第几行、哪个形态、原文），审计不必再翻日志。
    """
    value, why, _prov = _steps_from_stdout(stdout_log)
    return value, why


def parse_steps_evidence(run_dir: Path) -> tuple[int | None, str, list[dict]]:
    """实跑步数 + **多源 provenance**：`stdout.log` ∪ 跑次目录下全部 `trainer_state.json`。

    ★ 为什么多源（2026-10-26 D2）：ms-swift/HF 的 `trainer_state.json` 末态就有
      `"global_step": 6400`（M016 实证），而 stdout 只打印 `'global_step/max_steps': '6400/6400'`；
      单看 stdout 会在"日志被截断/后端不打印指标 dict"时判 None。多源并存时取**最大**，
      并把每个候选的坐标写进 `provenance.steps_evidence`（⛔ 不取第一个、不猜）。
    """
    candidates: list[dict] = []
    value, why, prov = _steps_from_stdout(run_dir / "stdout.log")
    candidates += prov
    for state in sorted(run_dir.rglob("trainer_state.json")):
        try:
            doc = json.loads(state.read_text(encoding="utf-8", errors="replace"))
        except (OSError, json.JSONDecodeError):
            continue
        gs = doc.get("global_step") if isinstance(doc, dict) else None
        if isinstance(gs, int) and not isinstance(gs, bool):
            candidates.append({"value": gs, "line": None, "kind": "trainer_state.global_step",
                               "raw": f"global_step={gs}",
                               "source": str(state.relative_to(run_dir))})
    if not candidates:
        return value, why, []
    best = max(candidates, key=lambda c: (c["value"], _STEP_KIND_RANK.get(c["kind"], 0)))
    where = (f"{best['source']}:{best['line']}" if best.get("line") is not None
             else f"{best['source']}")
    return best["value"], (f"实跑步数 = {best['value']}（多源取最大，取自 {where}，"
                           f"形态 {best['kind']}）"), candidates


def _naninf_sources(run_dir: Path) -> list[Path]:
    """NaN/Inf 的**全部证据源**（判据① 第四项）：stdout + 所有 jsonl 旁路 + trainer_state。

    为什么不再只看 stdout（2026-10-23 断点）：native/ms-swift 的数值异常会先落进
    `events.jsonl` / `rank_metrics*.jsonl` 的 metric 字段与 `skipped_nonfinite`，
    stdout 只是人读摘要；只扫 stdout 会把"已经发生过非有限跳步"判成"无异常"。
    """
    out: list[Path] = []
    stdout_log = run_dir / "stdout.log"
    if stdout_log.is_file():
        out.append(stdout_log)
    out += sorted(p for p in run_dir.rglob("*.jsonl") if p.is_file())
    out += sorted(p for p in run_dir.rglob("trainer_state.json") if p.is_file())
    return out


def detect_nan_inf(run_dir: Path) -> tuple[bool | None, str, list[Path]]:
    """NaN/Inf = 「events/日志里出现 nan/inf/skipped_nonfinite 三者任一」。

    证据源与正则（⛔ 不裸扫单词，避免 "infinite"/路径名假阳性）：
      · `NANINF_RE`：metric 字段（loss/grad_norm/reward/logits）取 nan/±inf；
      · `NANINF_SKIP_RE`：`skipped_nonfinite` / `nonfinite_grad_step_count` **> 0**。
    三处（stdout / jsonl 旁路 / trainer_state）**都读不到** ⇒ `None`（fail-closed）；
    读得到且零命中 ⇒ `False`（有据的"无异常"）。
    """
    sources = _naninf_sources(run_dir)
    if not sources:
        return None, ("训练目录下读不到任何数值证据"
                      "（stdout.log / *.jsonl / trainer_state.json 均缺失）⇒ 无法判 NaN/Inf"), []
    hits: list[str] = []
    skip_seen = 0
    for p in sources:
        try:
            with p.open("r", encoding="utf-8", errors="replace") as fh:
                for lineno, line in enumerate(fh, 1):
                    m = NANINF_RE.search(line)
                    if m:
                        hits.append(f"{p.name}:{lineno} {m.group(0)[:40]!r}")
                        break
                    m2 = NANINF_SKIP_RE.search(line)
                    if m2:
                        skip_seen += 1
                        if int(m2.group(2)) > 0:
                            hits.append(f"{p.name}:{lineno} {m2.group(0)[:40]!r}")
                            break
        except OSError:                                       # pragma: no cover - 权限/竞态
            continue
    if hits:
        return True, f"数值异常命中：{hits[0]}（共 {len(hits)} 处）", sources
    return False, (f"扫描 {len(sources)} 个证据源（stdout / jsonl 旁路 / trainer_state）未出现 "
                   f"NaN/Inf；非有限跳步计数字段出现 {skip_seen} 次且均 ≤ 0"), sources


def epochs_from_run(run_dir: Path) -> tuple[int | None, str, dict | None]:
    """判据① 的 **epoch 读数**（跑完了几轮）。

    取值优先级（逐条给"哪一行/哪个字段"的依据；⛔ 取不到就 None，⛔ 不拿步数换算）：
      ① `trainer_state.json`（native/HF/ms-swift 同形）的 `epoch` 浮点 × `global_step` 定序；
      ② `events.jsonl` 的 `epoch_summary`：`epoch`（0 起） + `epoch_cumulative.progress`；
      ③ ms-swift `logging.jsonl` 的 `epoch` 字段。
    完成轮数 = `floor(读数 + EPOCH_TOL)`（进度是浮点，满轮不能因 7.999999 判成 7）。
    """
    best = None
    for p in sorted(run_dir.rglob("trainer_state.json")):
        doc = read_json(p, None)
        if not isinstance(doc, dict):
            continue
        ep = doc.get("epoch")
        if isinstance(ep, bool) or not isinstance(ep, (int, float)):
            continue
        gs = doc.get("global_step")
        gs = gs if isinstance(gs, int) and not isinstance(gs, bool) else -1
        cand = (float(ep), gs, p, doc.get("num_train_epochs"))
        if best is None or (cand[1], cand[0]) > (best[1], best[0]):
            best = cand                      # ★ 取"步号最大"的那份状态（多 checkpoint 时）
    if best is not None:
        ep, gs, p, nte = best
        done = int(math.floor(ep + EPOCH_TOL))
        why = (f"trainer_state.json 的 epoch={ep:g}（global_step={gs}"
               f"{f'，num_train_epochs={nte}' if isinstance(nte, (int, float)) else ''}）"
               f" ⇒ 已完成 {done} 个 epoch")
        return done, why, {"source": str(p), "field": "epoch", "line": None,
                           "raw": ep, "num_train_epochs": nte}

    best_ev = None
    for p in sorted(run_dir.rglob("events.jsonl")):
        for lineno, obj in _iter_jsonl(p):
            if obj.get("event") != "epoch_summary":
                continue
            ep = obj.get("epoch")
            prog = (obj.get("epoch_cumulative") or {}).get("progress")
            if isinstance(ep, bool) or not isinstance(ep, int):
                continue
            if isinstance(prog, bool) or not isinstance(prog, (int, float)):
                continue
            v = float(ep) + float(prog)
            if best_ev is None or v > best_ev[0]:
                best_ev = (v, p, lineno, ep, float(prog))
    if best_ev is not None:
        v, p, lineno, ep, prog = best_ev
        done = int(math.floor(v + EPOCH_TOL))
        why = (f"events.jsonl 第 {lineno} 行 epoch_summary：epoch={ep} + progress={prog:g}"
               f" ⇒ 已完成 {done} 个 epoch")
        return done, why, {"source": str(p), "field": "epoch_summary", "line": lineno,
                           "raw": v, "epoch_index": ep, "progress": prog}

    best_lg = None
    for p in sorted(run_dir.rglob("logging.jsonl")):
        for lineno, obj in _iter_jsonl(p):
            ep = obj.get("epoch")
            if isinstance(ep, bool) or not isinstance(ep, (int, float)):
                continue
            if best_lg is None or float(ep) > best_lg[0]:
                best_lg = (float(ep), p, lineno)
    if best_lg is not None:
        ep, p, lineno = best_lg
        done = int(math.floor(ep + EPOCH_TOL))
        why = f"logging.jsonl 第 {lineno} 行 epoch={ep:g} ⇒ 已完成 {done} 个 epoch"
        return done, why, {"source": str(p), "field": "epoch", "line": lineno, "raw": ep}

    return None, ("训练目录下读不到 epoch 读数（无 trainer_state.json 的 epoch / 无 "
                  "events.jsonl 的 epoch_summary / 无 logging.jsonl 的 epoch 字段）"), None


def loss_series_from_run(run_dir: Path) -> tuple[list[float], str, dict | None]:
    """训练 loss 序列（判据② 的 CPT 支：**唯一**取法）。

    来源与 native/ms-swift 的实际落点一一对应：
      · `trainer_state.json` 的 `log_history[*].loss`（ms-swift/HF 同形 ⇒ CPT 主源）；
      · `events.jsonl` 的 `train_step` 的 `optimize.loss_mean`（native）；
      · ms-swift `logging.jsonl` 的 `loss`。
    ⛔ 取不到 ⇒ 空序列（调用方写 null ⇒ 回填器判 unknown，fail-closed）。
    """
    best = None
    for p in sorted(run_dir.rglob("trainer_state.json")):
        doc = read_json(p, None)
        if not isinstance(doc, dict):
            continue
        hist = doc.get("log_history")
        if not isinstance(hist, list):
            continue
        series = [float(r["loss"]) for r in hist
                  if isinstance(r, dict) and isinstance(r.get("loss"), (int, float))
                  and not isinstance(r.get("loss"), bool)]
        gs = doc.get("global_step")
        gs = gs if isinstance(gs, int) and not isinstance(gs, bool) else -1
        if series and (best is None or gs > best[0]):
            best = (gs, series, p, "trainer_state.json 的 log_history[*].loss")
    if best is not None:
        _gs, series, p, field = best
        return series, f"{field}（{len(series)} 个点）", {"source": str(p), "field": field}

    series, p = [], None
    for q in sorted(run_dir.rglob("events.jsonl")):
        cur = []
        for _lineno, obj in _iter_jsonl(q):
            if obj.get("event") != "train_step":
                continue
            v = (obj.get("optimize") or {}).get("loss_mean")
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                cur.append(float(v))
        if len(cur) > len(series):
            series, p = cur, q
    if series and p is not None:
        return series, f"events.jsonl 的 train_step.optimize.loss_mean（{len(series)} 个点）", {
            "source": str(p), "field": "train_step.optimize.loss_mean"}

    series, p = [], None
    for q in sorted(run_dir.rglob("logging.jsonl")):
        cur = []
        for _lineno, obj in _iter_jsonl(q):
            v = obj.get("loss")
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                cur.append(float(v))
        if len(cur) > len(series):
            series, p = cur, q
    if series and p is not None:
        return (series, f"logging.jsonl 的 loss（{len(series)} 个点）",
                {"source": str(p), "field": "loss"})

    return [], ("训练目录下读不到 loss 序列（trainer_state.log_history / "
                "events.train_step.optimize / logging.jsonl 均无）"), None


def loss_curve(series: list[float], why: str, evidence: dict | None) -> dict | None:
    """`eval.loss_curve` 的结构（判据② CPT 支的唯一载体）：首/末/最小 + **逐步序列**。

    为什么带 `series`：CPT 的判定口径（`result_judge.judge_cpt_loss`）= **首尾窗口均值**
    严格下降，"只给首末两点"无法复算该口径。⛔ 本文件不另立阈值 ⇒ 必须把序列交给判定器。
    """
    if not series:
        return None
    return {"n": len(series),
            "first": series[0],
            "last": series[-1],
            "min": min(series),
            "series": series,
            "source": why,
            "evidence": evidence or {}}


def ledger_record(ledger: Path | None, tier: str) -> dict | None:
    """从 `scripts/collect_results.py` 的台账（`ledger.jsonl`）里取该档记录（若有）。"""
    if ledger is None or not ledger.is_file():
        return None
    for line in ledger.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        if str(rec.get("tier_id")) == tier:
            return rec
    return None


def peak_mem_text(run_dir: Path, ledger: Path | None = None, tier: str = "") -> tuple[str | None, str]:
    """峰值显存列。**列值只写数值读数**（`<数值> GiB`；无读数 ⇒ None），⛔ 不写任何解释性文字。

    为什么（修正包 R1-1，审查 🔴-1）：本列会随回填落进人读件的「实验数据」表，机核会扫正文
    —— 解释性文字（测量方式/来源/内部路径）一旦写进本列就会被人读件闸门拒收。
    ⇒ 测量方式与来源作为**第二个返回值**由调用方写进 `note` 列（中性措辞）。

    取值优先级（§1.4 单一真相源）：
    ① `collect_results.py` 台账的 `peak_memory_gib`（该列测量方式由
       `graspo.core.result_judge.PEAK_MEMORY_CALIBER_BY_BACKEND` 按后端唯一决定，
       ⛔ 本文件不另立口径）；
    ② 否则退回 runner 的**宿主侧采样**（`<run-dir>/gpu/peak_mem.json`）。
    """
    rec = ledger_record(ledger, tier) if ledger else None
    if rec:
        gib = rec.get("peak_memory_gib")
        if isinstance(gib, (int, float)):
            return f"{float(gib):.2f} GiB", "矩阵台账的峰值显存读数（测量方式按后端唯一确定）"
    doc = read_json(run_dir / "gpu" / "peak_mem.json", {}) or {}
    used = doc.get("peak_used_mib")
    if used is None:
        return None, "宿主侧显存采样无读数"
    gpus = ",".join(str(g) for g in doc.get("gpus") or [])
    return f"{float(used) / 1024.0:.2f} GiB", f"宿主侧采样的显存占用峰值 = {used} MiB（GPU {gpus}）"


# ── 评测读数（判据②）────────────────────────────────────────────────────────
def eval_block(eval_dir: Path | None, tier_json: Path | None) -> tuple[dict | None, str | None, str]:
    """返回 (eval{}, effect 文本, 说明)。⛔ 两处都取不到 ⇒ (None, None, 原因)。"""
    if eval_dir:
        after = read_json(eval_dir / ".local" / "reports" / "after" / "eval_report.json")
        base = read_json(eval_dir / ".local" / "reports" / "base" / "eval_report.json")
        if isinstance(after, dict) and isinstance(base, dict):
            a = _acc(after)
            b = _acc(base)
            if a is not None and b is not None:
                ev = {"after_acc": round(a, 6), "base_acc": round(b, 6),
                      "delta_pp": round((a - b) * 100, 4)}
                return (ev,
                        (f"after acc={a * 100:.4f}%；base acc={b * 100:.4f}%；"
                         f"Δ = {(a - b) * 100:+.4f} pp"),
                        "取自评测报告（eval_report.json）的 after / base 准确率")
    if tier_json:
        doc = read_json(tier_json, {}) or {}
        a, b = doc.get("after_acc"), doc.get("base_acc")
        if isinstance(a, (int, float)) and isinstance(b, (int, float)):
            ev = {"after_acc": float(a), "base_acc": float(b),
                  "delta_pp": round((float(a) - float(b)) * 100, 4)}
            return ev, str(doc.get("effect") or ""), "取自评测汇总产物的 after / base 准确率"
    return None, None, ("本档没有评测读数（未运行评测，或 after / base 两侧评测报告不齐；"
                        "判据② 不可判定，回填器会 fail-closed）")


def _acc(report: dict) -> float | None:
    """取准确率比例。★ 兼容评测产物契约：`eval_report.json` 的聚合结论在 `summary{}` 下
    （`src/graspo/eval/schema.py::EvalSummary`），不是顶层；顶层形态留给 `gen_tier_json` 那类
    扁平产物。⛔ 缺这两种形态之一 ⇒ 返回 None（判据② 由回填器 fail-closed，⛔ 不估算）。"""
    if isinstance(report.get("summary"), dict):
        report = report["summary"]
    for key in ("accuracy", "accuracy_percent", "acc"):
        v = report.get(key)
        if isinstance(v, (int, float)):
            v = float(v)
            return v / 100.0 if key == "accuracy_percent" or v > 1.0 else v
    n_correct, n_total = report.get("n_correct"), report.get("n_total")
    if isinstance(n_correct, int) and isinstance(n_total, int) and n_total > 0:
        return n_correct / n_total
    return None


# ── D1 超时伪影的**证据化重派生**（2026-10-26；显式开关，默认关）──────────────────
#: 训练**真的跑完**的正面证据：ms-swift 在 main 返回时打印的那一行（M016 实测逐字）。
FINISH_MARKER_RE = re.compile(
    r"End time of running main:\s*(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})")
#: trainer 自报的墙钟（`'train_runtime': '5.207e+04'`）—— 只作**交叉校验**，不作 elapsed 口径。
TRAIN_RUNTIME_RE = re.compile(r"['\"]train_runtime['\"]\s*:\s*['\"]?([0-9.eE+-]+)")
#: runner 自己的起跑时刻（`2026-10-25T09:41:51Z [start] …`，写在 `<run-root>/<档号>.console.log`）。
RUNNER_START_RE = re.compile(r"^\s*(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})Z\s+\[start\]")
#: 「被外层 timeout 杀掉」的两个退出码（124 = TERM 生效；137 = 128+9 = kill-after 的 SIGKILL）。
TIMEOUT_EXIT_CODES = (124, 137)


def timeout_artifact_evidence(*, run_dir: Path, exit_code: int | None,
                              elapsed_raw: float | None, steps_actual: int | None,
                              steps_planned: int | None, epochs_done: int | None,
                              epochs_planned: int | None, loadable: bool | None,
                              nan_inf: bool | None) -> dict:
    """D1 的**事后取证**：判断 `exit_code∈{124,137}` 是"训练失败"还是"外层超时伪影"。

    ★ 为什么需要（M016 实证）：D1 的挂死让容器活到外层 `timeout` 上限才被杀 ⇒ 训练**已经**
      按 main 正常返回，台账里却是 `exit_code=137 / elapsed=57720`（超时 57600+kill-after 120）。
      旧链没有任何"重派生"通道，于是这一档永远进不了表。
    ★ 判据（**全部**满足才算伪影；任一不满足 ⇒ `applies=False`，一个字段都不动）：
      ① 退出码 ∈ {124, 137} **且** 墙钟 ≥ 该档 timeout（两者都读自 readings.json）；
      ② stdout.log 有 `End time of running main:` —— torchrun 的主函数**已返回**；
      ③ 步数跑满：最后一次 `global_step/max_steps` 读数 n/n，或实跑步数 ≥ 清单计划步数；
      ④ `epochs_completed ≥ epochs_planned`（既有 epoch 证据）；
      ⑤ `final_ckpt_loadable is True`；⑥ `nan_or_inf_detected is False`。
      ⇒ 训练真实状态 = 成功（`exit_code=0`），`elapsed` 改取"runner 起跑 → main 返回"的
      墙钟差（**排除 D1 的空等**），并用 trainer 自报 `train_runtime` 做交叉校验
      （torchrun 起得比 runner 晚 ⇒ `train_runtime ≤ 派生 elapsed ≤ 原始 elapsed`）。
    ★ ⛔ 不是"手改成 0"：每个条件都在返回值里带坐标（文件:行 + 原文），
      审计可以逐条复核；条件不全 ⇒ `applies=False` + `why`（fail-closed，不改任何字段）。
    """
    ev: dict = {
        "applies": False,
        "rule": ("exit_code∈{124,137} ∧ elapsed≥timeout ∧ stdout 有 End time of running main "
                 "∧ 步数跑满 ∧ epochs_completed≥planned ∧ ckpt 可加载 ∧ 无 NaN/Inf"),
        "exit_code_raw": exit_code,
        "exit_code_derived": None,
        "elapsed_raw": elapsed_raw,
        "elapsed_derived": None,
        "conditions": {},
        "why": "",
    }
    cond = ev["conditions"]

    readings = read_json(run_dir / "readings.json", {}) or {}
    timeout_sec = readings.get("timeout_sec")
    timeout_sec = float(timeout_sec) if isinstance(timeout_sec, (int, float)) else None
    cond["raw_is_timeout_code"] = {
        "ok": exit_code in TIMEOUT_EXIT_CODES,
        "detail": f"readings.json/exit_code 的原始退出码 = {exit_code}"
                  f"（外层 timeout 的产物形如 {list(TIMEOUT_EXIT_CODES)}）"}
    cond["elapsed_reached_timeout"] = {
        "ok": (timeout_sec is not None and elapsed_raw is not None
               and float(elapsed_raw) >= timeout_sec),
        "detail": f"elapsed={elapsed_raw} timeout_sec={timeout_sec}"}

    # ② ③ 同一次扫描 stdout.log（20 MB 真日志只读一遍）
    finish: dict | None = None
    last_slash: dict | None = None
    train_runtime: float | None = None
    stdout_log = run_dir / "stdout.log"
    if stdout_log.is_file():
        for lineno, line in enumerate(
                stdout_log.read_bytes().decode("utf-8", "replace").split("\n"), 1):
            # ⚠ 三个判据必须**各自独立**地看同一行：ms-swift 把 `train_runtime` 与
            #   `global_step/max_steps` 打在**同一个物理行**里（M016 stdout.log:190379），
            #   用 elif 会把后者吞掉 ⇒ 交叉校验拿不到 trainer 自报墙钟。
            if "End time of running main" in line:
                m = FINISH_MARKER_RE.search(line)
                if m:
                    finish = {"line": lineno, "timestamp": m.group(1), "raw": line.strip()[:120]}
            if "global_step/max_steps" in line:
                m = GLOBAL_STEP_SLASH_RE.search(line)
                if m:
                    last_slash = {"line": lineno,
                                  "pair": [int(m.group(1)),
                                           int(m.group(2)) if m.group(2) else None]}
            if "train_runtime" in line:
                m = TRAIN_RUNTIME_RE.search(line)
                if m:
                    try:
                        train_runtime = float(m.group(1))
                    except ValueError:
                        pass
    cond["trainer_main_returned"] = {
        "ok": finish is not None,
        "detail": (f"stdout.log:{finish['line']} {finish['raw']!r}" if finish
                   else "stdout.log 里没有 `End time of running main`（主函数没返回）")}
    steps_full = bool(
        (last_slash and last_slash["pair"][1] is not None
         and last_slash["pair"][0] == last_slash["pair"][1])
        or (isinstance(steps_actual, int) and isinstance(steps_planned, int)
            and steps_actual >= steps_planned))
    cond["steps_completed"] = {
        "ok": steps_full,
        "detail": (f"stdout.log:{last_slash['line']} 末次读数 {last_slash['pair'][0]}/"
                   f"{last_slash['pair'][1]}" if last_slash else "stdout.log 无步数读数")
                  + f"；steps_actual={steps_actual} planned={steps_planned}"}
    cond["epochs_completed"] = {
        "ok": (epochs_done is not None and epochs_planned is not None
               and epochs_done >= epochs_planned),
        "detail": f"epochs_completed={epochs_done} epochs_planned={epochs_planned}"}
    cond["final_ckpt_loadable"] = {"ok": loadable is True,
                                   "detail": f"final_ckpt_loadable={loadable!r}"}
    cond["nan_or_inf_free"] = {"ok": nan_inf is False,
                               "detail": f"nan_or_inf_detected={nan_inf!r}"}
    if finish is not None:
        ev["finish_marker"] = {"source": "stdout.log", **finish}
    if last_slash is not None:
        ev["last_step_reading"] = {"source": "stdout.log", **last_slash}
    ev["train_runtime_sec"] = train_runtime

    if not all(c["ok"] for c in cond.values()):
        ev["why"] = ("不构成超时伪影（条件不全 ⇒ 一个字段都不改，fail-closed）："
                     + "；".join(k for k, c in cond.items() if not c["ok"]))
        return ev

    # ── 派生 elapsed：runner 起跑 → main 返回（= 训练真实墙钟，排除 D1 的空等）──────────
    start_ts = None
    console = run_dir.parent / f"{run_dir.name}.console.log"
    if console.is_file():
        for line in console.read_text(encoding="utf-8", errors="replace").splitlines():
            m = RUNNER_START_RE.match(line)
            if m:
                start_ts = m.group(1)
                ev["runner_start"] = {"source": console.name, "timestamp": m.group(1),
                                      "raw": line.strip()[:120]}
                break
    if start_ts is None:
        ev["why"] = ("超时伪影成立，但缺 runner 起跑时刻（<run-root>/<档号>.console.log 的 "
                     "`[start]` 行）⇒ elapsed 不予派生（fail-closed 保留原值读数），退出码照派生")
        ev["exit_code_derived"] = 0
        ev["applies"] = True
        return ev
    try:
        t0 = datetime.strptime(start_ts, "%Y-%m-%dT%H:%M:%S")
        t1 = datetime.strptime(finish["timestamp"], "%Y-%m-%d %H:%M:%S")
    except ValueError:
        ev["why"] = "起跑/结束时刻解析失败 ⇒ elapsed 不予派生（保留原值读数）"
        ev["exit_code_derived"] = 0
        ev["applies"] = True
        return ev
    derived = (t1 - t0).total_seconds()
    # 交叉校验：容器的本地时间 = UTC（M016 实证：console.log 的 Z 时刻与 stdout.log 的
    # End time / 宿主 mtime 同一坐标），再叠一条与 trainer 自报墙钟的上下界校验。
    sane = 0 < derived <= float(elapsed_raw)
    if train_runtime is not None:
        sane = sane and train_runtime <= derived + 1.0
    ev["elapsed_derived"] = round(derived, 1) if sane else None
    ev["elapsed_cross_check"] = {
        "t_start": start_ts, "t_trainer_end": finish["timestamp"],
        "train_runtime_sec": train_runtime,
        "detail": (f"{finish['timestamp']} − {start_ts} = {derived:.0f}s；"
                   + (f"trainer 自报 train_runtime={train_runtime:.0f}s（须 ≤ 派生值）；"
                      if train_runtime is not None else "trainer 自报 train_runtime 读不到；")
                   + f"原始 elapsed={elapsed_raw}s（含 D1 空等）")}
    if not sane:
        ev["why"] = "派生 elapsed 未通过交叉校验 ⇒ 不采用（保留原值），退出码照派生"
    ev["exit_code_derived"] = 0
    ev["applies"] = True
    return ev


#: 「运维按授权停掉僵尸容器」的退出码（128+SIGTERM=143 / 128+SIGKILL=137）。
OPERATOR_STOP_EXIT_CODES = (137, 143)

#: 人工停容器的事实留痕文件名（跑次目录下；唯一名字，⛔ 不另设目录）。
OPERATOR_STOP_ATTESTATION = "operator-stop.json"

#: 墙钟与 trainer 自报 `train_runtime` 的允许相对差（两条独立读数互校；超出即不认）。
OPERATOR_STOP_WALLCLOCK_TOLERANCE = 0.01


def read_operator_stop_attestation(run_dir: Path, tier: str, exit_code: int | None) -> dict:
    """读并校验"人工停容器"的事实留痕 ⇒ `{ok, detail, raw}`。

    留痕必须**可机器核对**（⛔ 不接受自由文本）：容器名须是 `graspo-<档号>-<id>` 形态、
    `docker stop` 的 rc 须为 0、`finished_at` 须可解析、容器观测到的 `ExitCode` 须等于
    本跑次记录的退出码 —— 否则"人工停"这件事在本档不成立。
    """
    path = run_dir / OPERATOR_STOP_ATTESTATION
    doc = read_json(path, None)
    if not isinstance(doc, dict):
        return {"ok": False, "raw": None,
                "detail": f"缺人工停容器留痕（{path.name} 不存在或不是 JSON 对象）"}
    raw = {
        "source": str(path), "container": doc.get("container"),
        "stop_command": doc.get("stop_command"), "stop_rc": doc.get("stop_rc"),
        "started_at_utc": doc.get("started_at_utc"),
        "stopped_at_utc": doc.get("stopped_at_utc"),
        "finished_at_utc": doc.get("finished_at_utc"),
        "state_exit_code": doc.get("state_exit_code"),
        "state_status": doc.get("state_status"), "state_running": doc.get("state_running"),
        "oom_killed": doc.get("oom_killed"), "error": doc.get("error"),
        "recorded_at_utc": doc.get("recorded_at_utc"),
        "authorization": doc.get("authorization"),
        "raw_inspect": doc.get("raw_inspect"),
    }
    problems = []
    name = doc.get("container")
    if not isinstance(name, str) or not name.startswith(f"graspo-{tier}-"):
        problems.append(f"container={name!r} 不是本档容器（须 graspo-{tier}-<id>）")
    if doc.get("stop_rc") != 0:
        problems.append(f"stop_rc={doc.get('stop_rc')!r}（docker stop 必须 rc=0）")
    if not isinstance(doc.get("finished_at_utc"), str) or not doc["finished_at_utc"].strip():
        problems.append("缺 finished_at_utc（容器终止时刻）")
    if doc.get("state_exit_code") != exit_code:
        problems.append(f"留痕里 state_exit_code={doc.get('state_exit_code')!r} ≠ 跑次退出码 "
                        f"{exit_code!r}")
    if doc.get("state_running") is not False:
        problems.append(f"state_running={doc.get('state_running')!r}（须为 false）")
    if not problems:
        try:
            datetime.fromisoformat(str(doc["finished_at_utc"]).replace("Z", "+00:00"))
        except ValueError:
            problems.append(f"finished_at_utc 不可解析：{doc['finished_at_utc']!r}")
    if problems:
        return {"ok": False, "raw": raw, "detail": "；".join(problems)}
    return {"ok": True, "raw": raw,
            "detail": (f"{name}：docker stop rc=0，容器 {doc['finished_at_utc']} 终止"
                       f"（ExitCode={exit_code}）")}


def operator_stop_evidence(*, run_dir: Path, tier: str, exit_code: int | None,
                           elapsed_raw: float | None, steps_actual: int | None,
                           steps_planned: int | None, epochs_done: int | None,
                           epochs_planned: int | None, loadable: bool | None,
                           ckpt_path: str | None, nan_inf: bool | None) -> dict:
    """D1 的**第二种**事后取证：容器由运维按授权停止（而非被外层 timeout 杀掉）。

    ★ 与 :func:`timeout_artifact_evidence`（`--recover-timeout-artifact`）**并列、互不影响**：
      那条通道的判据一个字节都没改；本函数只处理"墙钟没到外层 timeout、人主动停了僵尸容器"
      这一种形态（M052 实证：主流程 18:22:30 返回，容器挂在 `wait "$SAMPLER"` 上，运维
      18:27:46 停掉）。两条通道都默认关，各自**全部**条件成立才动，任一不成立 ⇒ `applies=False`。

    ★ 判据（**全部**满足才算伪影；每条都带坐标，审计可逐条复核）：
      ① 原始退出码 ∈ {137, 143}（SIGKILL / SIGTERM）；
      ② stdout.log 有 `End time of running main:`（主流程已返回）；
      ③ 步数跑满：末次读数 n/n 或实跑步数 ≥ 计划步数；
      ④ `epochs_completed ≥ epochs_planned`；
      ⑤ `final_ckpt_loadable is True`；
      ⑥ LoRA 权重**内容级**非全零（`lora_B` 逐字节扫描；⛔ 不用大小/时间戳）；
      ⑦ `nan_or_inf_detected is False`；
      ⑧ 墙钟与 trainer 自报 `train_runtime` 相对差 ≤ 1%（两条独立读数互校）；
      ⑨ **有人工停容器的事实留痕**（见 :func:`read_operator_stop_attestation`）。
      ⇒ 训练真实状态 = 成功（`exit_code` 派 0），`elapsed` **原样保留**（本条通道不假定
      墙钟被空等污染；`train_runtime` 互校保证它就是真实训练墙钟）。
    """
    ev: dict = {
        "applies": False,
        "rule": ("exit_code∈{137,143} ∧ 主流程已返回 ∧ 步数跑满 ∧ epoch 跑满 ∧ ckpt 可加载 "
                 "∧ LoRA 权重非全零 ∧ 无 NaN/Inf ∧ |墙钟−train_runtime|/train_runtime ≤ 1% "
                 "∧ 有人工停容器留痕"),
        "exit_code_raw": exit_code,
        "exit_code_derived": None,
        "elapsed_raw": elapsed_raw,
        "conditions": {},
        "why": "",
    }
    cond = ev["conditions"]
    cond["raw_is_operator_stop_code"] = {
        "ok": exit_code in OPERATOR_STOP_EXIT_CODES,
        "detail": f"readings.json/exit_code 的原始退出码 = {exit_code}（人工 docker stop 的产物"
                  f"形如 {list(OPERATOR_STOP_EXIT_CODES)}）"}

    finish: dict | None = None
    last_slash: dict | None = None
    train_runtime: float | None = None
    stdout_log = run_dir / "stdout.log"
    if stdout_log.is_file():
        for lineno, line in enumerate(
                stdout_log.read_bytes().decode("utf-8", "replace").split("\n"), 1):
            if "End time of running main" in line:
                m = FINISH_MARKER_RE.search(line)
                if m:
                    finish = {"line": lineno, "timestamp": m.group(1), "raw": line.strip()[:120]}
            if "global_step/max_steps" in line:
                m = GLOBAL_STEP_SLASH_RE.search(line)
                if m:
                    last_slash = {"line": lineno,
                                  "pair": [int(m.group(1)),
                                           int(m.group(2)) if m.group(2) else None]}
            if "train_runtime" in line:
                m = TRAIN_RUNTIME_RE.search(line)
                if m:
                    try:
                        train_runtime = float(m.group(1))
                    except ValueError:
                        pass
    cond["trainer_main_returned"] = {
        "ok": finish is not None,
        "detail": (f"stdout.log:{finish['line']} {finish['raw']!r}" if finish
                   else "stdout.log 里没有 `End time of running main`（主函数没返回）")}
    steps_full = bool(
        (last_slash and last_slash["pair"][1] is not None
         and last_slash["pair"][0] == last_slash["pair"][1])
        or (isinstance(steps_actual, int) and isinstance(steps_planned, int)
            and steps_actual >= steps_planned))
    cond["steps_completed"] = {
        "ok": steps_full,
        "detail": (f"stdout.log:{last_slash['line']} 末次读数 {last_slash['pair'][0]}/"
                   f"{last_slash['pair'][1]}" if last_slash else "stdout.log 无步数读数")
                  + f"；steps_actual={steps_actual} planned={steps_planned}"}
    cond["epochs_completed"] = {
        "ok": (epochs_done is not None and epochs_planned is not None
               and epochs_done >= epochs_planned),
        "detail": f"epochs_completed={epochs_done} epochs_planned={epochs_planned}"}
    cond["final_ckpt_loadable"] = {"ok": loadable is True,
                                   "detail": f"final_ckpt_loadable={loadable!r}"}

    lora_ok: bool | None = None
    lora_why = "缺检查点路径 ⇒ 无法核 LoRA 权重内容"
    if ckpt_path:
        weights = Path(ckpt_path) / "adapter_model.safetensors"
        if weights.is_file():
            lora_ok, lora_why = _lora_nonzero_probe(weights)
        else:
            lora_why = f"{weights} 不存在（非 PEFT adapter 布局）"
    cond["lora_weights_nonzero"] = {"ok": lora_ok is True, "detail": lora_why}

    cond["nan_or_inf_free"] = {"ok": nan_inf is False,
                               "detail": f"nan_or_inf_detected={nan_inf!r}"}

    rel_diff: float | None = None
    if (isinstance(elapsed_raw, (int, float)) and isinstance(train_runtime, (int, float))
            and float(train_runtime) > 0):
        rel_diff = abs(float(elapsed_raw) - float(train_runtime)) / float(train_runtime)
    cond["wallclock_matches_train_runtime"] = {
        "ok": rel_diff is not None and rel_diff <= OPERATOR_STOP_WALLCLOCK_TOLERANCE,
        "detail": (f"elapsed={elapsed_raw} 与 trainer 自报 train_runtime={train_runtime} 的"
                   f"相对差 = {(rel_diff * 100 if rel_diff is not None else float('nan')):.2f}%"
                   f"（阈值 {OPERATOR_STOP_WALLCLOCK_TOLERANCE:.0%}）")}

    att = read_operator_stop_attestation(run_dir, tier, exit_code)
    cond["operator_stop_attested"] = {"ok": att["ok"], "detail": att["detail"]}
    if att["raw"] is not None:
        ev["operator_stop"] = att["raw"]
    if finish is not None:
        ev["finish_marker"] = {"source": "stdout.log", **finish}
    if last_slash is not None:
        ev["last_step_reading"] = {"source": "stdout.log", **last_slash}
    ev["train_runtime_sec"] = train_runtime
    ev["wallclock_rel_diff"] = None if rel_diff is None else round(rel_diff, 6)

    if not all(c["ok"] for c in cond.values()):
        ev["why"] = ("不构成「运维按授权停容器」伪影（条件不全 ⇒ 一个字段都不改，fail-closed）："
                     + "；".join(k for k, c in cond.items() if not c["ok"]))
        return ev
    ev["exit_code_derived"] = 0
    ev["applies"] = True
    return ev


def _operator_stop_note(ev: dict) -> str:
    """note 列的人读句（⛔ 不含内部路径；数值全部取自证据）。

    「原始退出码 X；主流程于 <时刻> 返回后容器挂起（D1，entry.sh 无界等待采样器），
      由运维按授权于 <时刻> 停止；墙钟 N s 与 trainer 自报 M s 差 K%；步数/epoch/ckpt/NaN 证据全过」
    """
    finish = (ev.get("finish_marker") or {}).get("timestamp")
    att = ev.get("operator_stop") or {}
    stopped = att.get("finished_at_utc")
    if isinstance(stopped, str) and "T" in stopped:
        stopped = stopped.replace("T", " ")[:19]
    elapsed = ev.get("elapsed_raw")
    runtime = ev.get("train_runtime_sec")
    rel = ev.get("wallclock_rel_diff")

    def _sci(sec: float) -> str:
        """109100.0 → '1.091e5'（与 trainer 自报同量级的紧凑写法）。"""
        return f"{float(sec):.4g}".replace("e+0", "e").replace("e+", "e")

    bits = [f"原始退出码 {ev.get('exit_code_raw')}",
            f"主流程于 {finish} 返回后容器挂起（D1，entry.sh 无界等待采样器）",
            f"由运维按授权于 {stopped} 停止"]
    if isinstance(elapsed, (int, float)) and isinstance(runtime, (int, float)):
        bits.append(f"墙钟 {float(elapsed):.0f} s 与 trainer 自报 {_sci(runtime)} s "
                    f"差 {(rel or 0) * 100:.1f}%")
    bits.append("步数/epoch/ckpt/NaN 证据全过")
    return "退出码按证据重派生（运维按授权停容器）：" + "；".join(bits)


# ── D9 崩溃档的**证据化入表**（2026-10-27；显式开关，默认关）──────────────────────
#: 异常行：**行首无缩进**的 `<名字>Error: <消息>`（Python traceback 的末行形态）。
#: ⛔ 不匹配缩进的续行（`    ^^^^`）与 `Traceback (most recent call last):`。
CRASH_EXC_RE = re.compile(r"^([A-Za-z_][A-Za-z0-9_.]*(?:Error|Exception|Interrupt|Fault))\s*:(.*)$")
#: traceback 帧：`  File "<文件>", line <行号>, in <函数>`
CRASH_FRAME_RE = re.compile(r'^\s*File "([^"]+)", line (\d+)')


def crashed_tier_evidence(*, run_dir: Path, tier: str, exit_code: int | None,
                          steps_actual: int | None, ckpt_dirs: list[Path],
                          readings: dict) -> dict:
    """D9 的**事后取证**：这档是不是「起跑即崩、一步优化都没跑」⇒ 可以按崩溃如实入表。

    ★ 为什么需要（M011 实证，228 实测 2026-10-25）：native 全参档在 `_load_model` 就崩
      （`model_adapter.py:61` → `core/lora.py:170`），`exit_code=1`、`stdout.log` 有逐字的
      `ValueError: lora.target_modules is required…`、无任何 checkpoint、无 `trainer_state.json`。
      旧链的三条「仅失败」通道全不通（回填器无开关 / `gen_tier_json` 取值域自相矛盾 /
      本文件缺 `steps_actual`/`throughput` 即判 rc=4）⇒ **实测崩溃、证据齐全的档一格写不进去**，
      台账永远显示"待测试"、误导决策。
    ★ 判据（**四条全部**成立才算；任一不成立 ⇒ `applies=False` + `why`，一个字段都不改）：
      ① 退出码是整数、**≠ 0**、且 **∉ {124,137}**（超时/被外层杀掉属 D1 的两条既有通道，
         ⛔ 不许抢它们的活）；
      ② `stdout.log` 里能**定位到报错原文**：末条"消息非空"的异常行 + 它上面最近的
         `File "…", line N` 帧 ⇒ 记下日志行号与源码 file:line（审计可逐条复核）；
      ③ **没有任何有效训练产物**：无 `checkpoint-*`/`final` 目录、无 `trainer_state.json`
         （或有但 `global_step == 0`）、且**没有**任何步数读数（`steps_actual` 为 None 或 0）；
         ⛔ 「跑了一半 / 超时但已训练若干步」一律不成立 —— 那属另一条通道；
      ④ 该档 `tier.json` 在且能解析、`readings.json` 的 `tier` 与本档一致 ⇒ 能证明"跑的是这个档"。
    ★ ⛔ 本函数**不派生任何读数、不放宽任何判据**：它只把「退出码≠0 且一步没跑」这件已经发生
      的事实变成可入表的证据；最终状态仍由回填器按第 2 章四条判据算出（必然是 `不通过`）。
    """
    ev: dict = {
        "applies": False,
        "rule": ("exit_code≠0 且 ∉{124,137} ∧ stdout.log 可定位报错原文与行号 ∧ "
                 "无任何训练产物（无 ckpt / 无 trainer_state 或 global_step==0 / 无步数读数）∧ "
                 "tier.json 在且 readings.tier 一致"),
        "exit_code_raw": exit_code,
        "error": None,
        "artifact_probe": {},
        "tier_proof": {},
        "conditions": {},
        "explained_gaps": [],
        "why": "",
    }
    cond = ev["conditions"]

    cond["exit_code_is_error"] = {
        "ok": (isinstance(exit_code, int) and not isinstance(exit_code, bool)
               and exit_code != 0 and exit_code not in TIMEOUT_EXIT_CODES),
        "detail": (f"exit_code={exit_code!r}（崩溃档要求：整数、≠0、且不是外层超时的 "
                   f"{list(TIMEOUT_EXIT_CODES)} —— 那两个码属 D1 的超时/停容器通道）"),
    }

    # ② 同一遍扫描 stdout.log，取**末条消息非空的异常行** + 它上面最近的 traceback 帧
    error: dict | None = None
    stdout_log = run_dir / "stdout.log"
    if stdout_log.is_file():
        lines = stdout_log.read_bytes().decode("utf-8", "replace").split("\n")
        for idx, line in enumerate(lines):
            mo = CRASH_EXC_RE.match(line)
            if not mo or not mo.group(2).strip():
                continue                     # 空消息的包装异常（`ChildFailedError:`）不作报错原文
            frame = None
            for j in range(idx - 1, -1, -1):
                mf = CRASH_FRAME_RE.match(lines[j])
                if mf:
                    frame = {"file": mf.group(1), "line": int(mf.group(2)), "log_line": j + 1}
                    break
                if CRASH_EXC_RE.match(lines[j]):
                    break                    # 撞上上一条异常 ⇒ 不在同一个 traceback 块里
            error = {"log_line": idx + 1, "text": line.rstrip(),
                     "exception": mo.group(1), "frame": frame}
    cond["error_located"] = {
        "ok": bool(error and error.get("frame")),
        "detail": (f"stdout.log:{error['log_line']} {error['text'][:120]!r}；"
                   f"帧 = {error['frame']['file']}:{error['frame']['line']}"
                   if error and error.get("frame") else
                   ("stdout.log 里没有可定位的异常行+行号（缺 `File \"…\", line N` 帧）"
                    if error else "stdout.log 里没有 `<异常名>: <消息>` 形态的报错行")),
    }
    if error is not None:
        ev["error"] = error

    # ③ 没有任何有效训练产物
    state_steps: list[dict] = []
    for p in sorted(run_dir.rglob("trainer_state.json")):
        doc = read_json(p, None)
        if isinstance(doc, dict) and isinstance(doc.get("global_step"), int) \
                and not isinstance(doc.get("global_step"), bool):
            state_steps.append({"source": str(p), "global_step": doc["global_step"]})
    zero_or_none = (steps_actual is None
                    or (isinstance(steps_actual, int) and not isinstance(steps_actual, bool)
                        and steps_actual == 0))
    no_training = (not ckpt_dirs) and all(s["global_step"] == 0 for s in state_steps) and zero_or_none
    ev["artifact_probe"] = {
        "checkpoint_dirs": [str(p) for p in ckpt_dirs],
        "trainer_state_global_steps": state_steps,
        "steps_actual": steps_actual,
    }
    cond["no_training_artifact"] = {
        "ok": no_training,
        "detail": (f"checkpoint/final 目录 {len(ckpt_dirs)} 个；trainer_state 的 global_step="
                   f"{[s['global_step'] for s in state_steps] or '无 trainer_state.json'}；"
                   f"steps_actual={steps_actual!r}（崩溃档要求：三者都指向「一步没跑」）"),
    }

    # ④ 能证明"跑的是这个档"
    tj = run_dir / "tier.json"
    tj_doc = read_json(tj, None) if tj.is_file() else None
    identity_keys = ("config", "algorithm", "tuner_type", "backend", "nproc", "max_epochs")
    tj_ok = isinstance(tj_doc, dict) and any(k in tj_doc for k in identity_keys)
    rtier = readings.get("tier")
    ev["tier_proof"] = {"tier_json": str(tj) if tj.is_file() else None,
                        "tier_json_keys": sorted(k for k in identity_keys if isinstance(tj_doc, dict)
                                                 and k in tj_doc),
                        "readings_tier": rtier}
    cond["tier_proven"] = {
        "ok": bool(tj_ok and str(rtier or "").strip() == tier),
        "detail": (f"tier.json {'在' if tj.is_file() else '缺'}"
                   f"（身份键 {ev['tier_proof']['tier_json_keys']}）；"
                   f"readings.tier={rtier!r} / 本档 {tier!r}"),
    }

    if not all(c["ok"] for c in cond.values()):
        ev["why"] = ("不构成证据齐全的崩溃档（条件不全 ⇒ 一个字段都不改，fail-closed）："
                     + "；".join(f"{k}（{c['detail']}）" for k, c in cond.items() if not c["ok"]))
        return ev

    ev["applies"] = True
    return ev


def _crash_note(ev: dict, exit_code: int) -> str:
    """note 列首句（**逐字**模板，D9 的验收句）。

    「训练崩溃，未产生有效训练：exit_code=<原值>，<报错首行原文>（stdout.log:<行号>）」
    """
    err = ev.get("error") or {}
    frame = err.get("frame") or {}
    return (f"训练崩溃，未产生有效训练：exit_code={exit_code}，{err.get('text')}"
            f"（stdout.log:{err.get('log_line')}）"
            + (f"；同一条 traceback 的源码帧 = {frame.get('file')}:{frame.get('line')}"
               if frame else ""))


# ── 组装 ─────────────────────────────────────────────────────────────────────
def build_doc(*, tier: str, run_dir: Path, manifest: dict, data_rows: dict,
              eval_dir: Path | None, tier_json: Path | None, note_extra: str,
              ledger: Path | None = None,
              recover_timeout_artifact: bool = False,
              recover_operator_stop: bool = False,
              recover_crashed_tier: bool = False) -> tuple[dict, dict]:
    """返回 (结果 json, 诊断信息{schema_errors, evidence_gaps})。"""
    mt = next((t for t in manifest["tiers"] if t["tier_id"] == tier), None)
    if mt is None:
        raise SystemExit(f"FATAL: 清单里没有档号 {tier}")
    row = data_rows.get(tier)
    if row is None:
        raise SystemExit(f"FATAL: 数据件里没有档号 {tier}")

    # ── 身份：清单与数据件必须同源（⛔ 不同源就别写 json，先修数据）──────────────
    ident_problems = []
    for mk, dk in (("algorithm", "algo"), ("mode", "mode"), ("backend", "backend"),
                   ("model", "model"), ("lr", "lr"), ("tuner_type", "tuner_type")):
        if str(mt.get(mk)).strip() != str(row.get(dk)).strip():
            try:
                same = abs(float(str(mt.get(mk))) - float(str(row.get(dk)))) < 1e-15
            except (TypeError, ValueError):
                same = False
            if not same:
                ident_problems.append(f"清单 {mk}={mt.get(mk)!r} ≠ 数据件 {dk}={row.get(dk)!r}")
    if ident_problems:
        raise SystemExit("FATAL: 清单与数据件身份不同源 ⇒ ⛔ 不产出结果 json：" + "；".join(ident_problems))

    gaps: list[str] = []
    readings = read_json(run_dir / "readings.json", {}) or {}

    # ── 判据① 的结构化事实 ────────────────────────────────────────────────
    exit_code = readings.get("exit_code")
    if exit_code is None:
        exit_code = read_int_file(run_dir / "exit_code")
    if isinstance(exit_code, bool) or not isinstance(exit_code, int):
        gaps.append("缺跑次退出码（跑次目录下无退出码读数）")
        exit_code = None

    steps_actual = readings.get("steps_actual")
    steps_why = "读自跑次读数文件"
    steps_prov: list[dict] = []
    if not isinstance(steps_actual, int) or isinstance(steps_actual, bool):
        steps_actual, steps_why, steps_prov = parse_steps_evidence(run_dir)
        if steps_actual is None:
            gaps.append(f"缺实跑步数（{steps_why}）")

    # ── 判据① 的 **Epoch 口径**（2026-10-23：跑满 epoch 才叫"跑完"，步数只作台账）────
    #   `epochs_planned` = 清单该档 `max_epochs`（配置真相源）；`epochs_completed` 只从
    #   trainer_state.json / events.jsonl(epoch_summary) / logging.jsonl 的**实测**读数取。
    #   ⛔ 取不到 ⇒ 写 null（回填器判 unknown ⇒ fail-closed），⛔ 绝不用步数换算。
    epochs_planned = mt.get("max_epochs")
    epochs_planned = int(epochs_planned) if isinstance(epochs_planned, int) and not isinstance(
        epochs_planned, bool) else None
    if epochs_planned is None:
        gaps.append(f"缺计划轮数（清单该档 max_epochs={mt.get('max_epochs')!r} 不是整数）")
    epochs_done, epochs_why, epochs_ev = epochs_from_run(run_dir)
    if epochs_done is None:
        gaps.append(f"缺 epoch 读数（{epochs_why}）⇒ 判据① 不作通过判定（⛔ 不回落步数）")

    loadable, loadable_why, ckpt_path = ckpt_loadable(run_dir)
    if loadable is None:
        gaps.append(f"缺最终检查点可加载性结论（{loadable_why}）")

    nan_inf, naninf_why, naninf_sources = detect_nan_inf(run_dir)
    if nan_inf is None:
        gaps.append(f"缺 NaN/Inf 检测结论（{naninf_why}）")

    # ── D1 超时伪影的**事后取证**（2026-10-26）：只在显式开关下生效，默认路径逐字不变 ─────
    #   ⛔ 默认关的理由：D1 修好后新跑次不该再出现"训练跑完却被超时杀掉"，本通道只用于
    #      **已落盘的老跑次**（如 M016）按证据重派生；不设开关 ⇒ 行为与修复前完全一致。
    artifact: dict | None = None
    if recover_timeout_artifact:
        artifact = timeout_artifact_evidence(
            run_dir=run_dir, exit_code=exit_code,
            elapsed_raw=(readings.get("elapsed_sec")
                         if readings.get("elapsed_sec") is not None
                         else read_float_file(run_dir / "elapsed_sec")),
            steps_actual=steps_actual,
            steps_planned=mt.get("expected_optimizer_steps", mt.get("planned_steps")),
            epochs_done=epochs_done, epochs_planned=epochs_planned,
            loadable=loadable, nan_inf=nan_inf)
        if artifact["applies"] and artifact.get("exit_code_derived") is not None:
            exit_code = artifact["exit_code_derived"]

    # ── D1 伪影的**第二种**事后取证（2026-10-26）：容器由运维按授权停止 ─────────────────
    #   与上一条**并列**、互不影响：判据各自独立、都从**原始** exit_code 起算（⛔ 不拿上一条的
    #   派升值当输入）。两条通道的条件互斥（上一条要求 elapsed ≥ 外层 timeout，本条要求
    #   墙钟与 trainer 自报 train_runtime 相差 ≤1%）⇒ 同一跑次最多只有一条成立。
    operator_stop: dict | None = None
    if recover_operator_stop:
        operator_stop = operator_stop_evidence(
            run_dir=run_dir, tier=tier, exit_code=exit_code,
            elapsed_raw=(readings.get("elapsed_sec")
                         if readings.get("elapsed_sec") is not None
                         else read_float_file(run_dir / "elapsed_sec")),
            steps_actual=steps_actual,
            steps_planned=mt.get("expected_optimizer_steps", mt.get("planned_steps")),
            epochs_done=epochs_done, epochs_planned=epochs_planned,
            loadable=loadable, ckpt_path=ckpt_path, nan_inf=nan_inf)
        if operator_stop["applies"] and operator_stop.get("exit_code_derived") is not None:
            exit_code = operator_stop["exit_code_derived"]

    # ── D9 崩溃档的**事后取证**（2026-10-27）：只在显式开关下生效，默认路径逐字不变 ─────
    #   ★ 与上面两条 D1 通道**并列**、互不影响：这条只认「退出码≠0 且一步没跑」的档
    #     （exit_code ∉ {124,137}，见 crashed_tier_evidence 的条件①）；D1 的两条通道各自
    #     要求"训练真的跑完了"的正面证据 ⇒ 同一跑次不可能同时成立。
    #   ★ 为什么需要它：实测崩溃、证据齐全的档此前一格写不进去（缺 steps/throughput 即 rc=4），
    #     台账只能显示"待测试"。本通道**不改判据**：它只把「一步没跑」这件事写成 0 步 / 0 轮 /
    #     无检查点，最终状态由回填器按判据① 算出（退出码≠0 ⇒ `不通过`）。
    crash: dict | None = None
    if recover_crashed_tier:
        crash = crashed_tier_evidence(
            run_dir=run_dir, tier=tier, exit_code=exit_code, steps_actual=steps_actual,
            ckpt_dirs=find_checkpoint_dirs(run_dir), readings=readings)
        if crash["applies"]:
            # ⛔ 退出码保持**原值**（判据① 靠它判"有据且不过"）；只把"一步没跑"写成 0。
            steps_actual, steps_why = 0, (
                f"崩溃前未进入优化循环（stdout.log:{crash['error']['log_line']} 的报错先于"
                "任何步数读数；无 checkpoint / 无 trainer_state 的 global_step 读数）")
            steps_prov = [{"source": "stdout.log", "kind": "no_step_reading",
                           "line": crash["error"]["log_line"], "value": 0,
                           "raw": crash["error"]["text"][:120]}]
            epochs_done, epochs_why = 0, (
                "训练崩溃于第 0 步（无 trainer_state.json 的 epoch 读数、无 events/logging 的 "
                "epoch 字段）⇒ 完成 0 轮")
            epochs_ev = {"source": "crashed_tier_evidence", "field": "none", "line": None,
                         "raw": 0}
            loadable, loadable_why = False, (
                "无任何 checkpoint-*/ 或 final/ 目录（崩溃发生在产出检查点之前）⇒ "
                "不存在可加载的最终检查点")

    # ── 三个数值/文本列 ───────────────────────────────────────────────────
    elapsed = readings.get("elapsed_sec")
    if elapsed is None:
        elapsed = read_float_file(run_dir / "elapsed_sec")
    if artifact is not None and artifact.get("elapsed_derived") is not None:
        elapsed = artifact["elapsed_derived"]
    if elapsed is None:
        gaps.append("缺训练墙钟秒数（跑次未落墙钟读数）")
    peak_mem, peak_why = peak_mem_text(run_dir, ledger, tier)
    if peak_mem is None:
        gaps.append(f"缺 peak_mem（{peak_why}）")

    throughput = None
    if crash is not None and crash["applies"]:
        throughput = (f"不适用（训练崩溃，未进入优化循环：{steps_actual} 步 / "
                      f"{float(elapsed):.1f} s 墙钟）" if elapsed else
                      "不适用（训练崩溃，未进入优化循环）")
    elif isinstance(steps_actual, int) and elapsed:
        throughput = (f"{steps_actual / float(elapsed):.3f} 步/s"
                      f"（{steps_actual} 步 / {float(elapsed):.1f} s；样本吞吐 "
                      f"{mt['data']['subset_size'] * mt['max_epochs'] / float(elapsed):.3f} 样本/s）")
    else:
        gaps.append("缺 throughput（步数与墙钟不齐）")

    ev, ev_effect, ev_why = eval_block(eval_dir, tier_json)
    if ev is None:
        gaps.append(f"effect/判据②：{ev_why}")

    # ── 判据②（OPD）：教师读数 `eval.teacher_after_acc`（★ 唯一渠道 = readings.json）─────────
    #   ★ 产出者 = `scripts/record_checks.py`（从 `<EVAL_DIR>/.local/reports/teacher/eval_report.json`
    #     按与学生侧同一口径实测：同一份 100 条 / 温度 0 / 同一判分函数；见
    #     `docs/run-discipline.md` 的「OPD 教师侧评测」一节）。
    #   ⛔ 缺读数就**不写**这个键（回填器 judge_effect 判 unknown ⇒ fail-closed 保持「待测试」），
    #     绝不填 0、绝不拿学生读数冒充教师读数、绝不因为"非 OPD 档"给默认值。
    algo = str(row.get("algo") or "").upper()
    teacher_acc = readings.get("teacher_after_acc")
    if not isinstance(teacher_acc, (int, float)) or not (0.0 <= float(teacher_acc) <= 1.0):
        teacher_acc = None
    if algo == "OPD":
        if teacher_acc is None:
            gaps.append("判据② 缺证据：OPD 需教师在训练数据集上的完全正确率"
                        "（eval.teacher_after_acc；先跑教师侧评测，见 run-discipline「OPD 教师侧评测」）"
                        " ⇒ 不作通过判定")
        elif ev is not None:
            ev["teacher_after_acc"] = round(float(teacher_acc), 6)

    # ── 判据③④ 的结构化结论（★ 唯一渠道：`<run-dir>/readings.json` 的 `checks`）──────────
    #   ★ 产出者 = `scripts/record_checks.py`（评测跑完后一条命令，从评测报告与算法检查证据现算；
    #     见 `docs/run-discipline.md` 的「判据③④ 的判定与证据」一节）。
    #   ⛔ 缺键就**不写**这两个键（回填器据此判 unknown ⇒ fail-closed 保持「待测试」），
    #     绝不写 false/null 冒充"判过了"，更不硬编 true。
    checks: dict[str, bool] = {}
    checks_ev = readings.get("checks_evidence") if isinstance(readings.get("checks_evidence"), dict) else {}
    raw_checks = readings.get("checks")
    if isinstance(raw_checks, dict):
        checks = {k: raw_checks[k] for k in ("algo_checks_passed", "inference_parseable")
                  if isinstance(raw_checks.get(k), bool)}
    for key, cid, why in (
        ("algo_checks_passed", "判据③",
         "算法正确性检查缺证据（GRASPO 档须逐档签署第 9 节 8 条不变式；"
         "SFT/CPT/OPD 须逐类公开算法签署件（损失/标签掩码规则））"),
        ("inference_parseable", "判据④", "评测报告缺证据（未跑评测或报告不齐）"),
    ):
        if key not in checks:
            gaps.append(f"{cid} 缺证据：{why} ⇒ 不作通过判定")

    # ── 判据②（**CPT 支**）：训练 loss 正常下降（★ 判定器在回填器，本文件只产证据）──────
    #   ★ 为什么 CPT 必须另有 loss 证据：第 2 章判据② 对 CPT 的定义是「训练 loss 正常下降」
    #     （SFT/GRASPO/OPD 看 Δ）。⛔ 此前这条**没有任何判定代码**，"无 loss 读数 + after=base"
    #     会被静默判通过。⇒ 本文件产出 `eval.loss_curve`（首/末/最小 + 逐步序列），回填器
    #     `judge_effect` 用 `result_judge.judge_cpt_loss` 的同一口径（首尾窗口均值严格下降）判定；
    #     取不到序列 ⇒ 不写键 ⇒ 回填器判 unknown（fail-closed）。
    loss_curve_doc = None
    loss_curve_why = ""
    if algo == "CPT":
        _series, loss_curve_why, _series_ev = loss_series_from_run(run_dir)
        loss_curve_doc = loss_curve(_series, loss_curve_why, _series_ev)
        if loss_curve_doc is None:
            gaps.append(f"判据② 缺证据：CPT 需训练 loss 序列（{loss_curve_why}）⇒ 不作通过判定")
        else:
            ev = ev if ev is not None else {}
            ev["loss_curve"] = loss_curve_doc

    # ── ★ 以下 6 列会随回填进入人读件的「实验数据」表 ⇒ 只放读数/结论，措辞必须中性 ────
    #   （⛔ 内部路径 / 内部工位名 / 内部行话一律不进这 6 列；证据路径留在 provenance 里供审计，
    #    provenance 不进人读件。）
    # ★ `effect` 特例：回填器要求"6 列写满"（空/null 一律拒收），而"本档未跑评测"是一个
    #   **有据的结论**（不是缺数据）⇒ 按既有 `gen_tier_json --no-eval` 的同一写法如实声明
    #   "不可评"，⛔ 不填任何效果数字。数值类列（elapsed/throughput/peak_mem/steps_actual）
    #   缺读数时仍写 null —— 那些是**实测值**，编不出来。
    effect = ev_effect if ev_effect else ("不可评：本档没有评测读数（未运行评测，或 after / base "
                                         "两侧评测报告不齐）")
    if crash is not None and crash["applies"]:
        # ★ 崩溃档的"不可评"真因 = 训练崩溃（不是"忘了跑评测"），如实写清；⛔ 不填任何效果数字。
        effect = ("不可评：训练崩溃，未产生任何训练产物与评测读数（崩溃归因见 note 列首句）")
    if loss_curve_doc is not None:
        effect += (f"；训练 loss {loss_curve_doc['first']:.4g} → {loss_curve_doc['last']:.4g}"
                   f"（最小 {loss_curve_doc['min']:.4g}，共 {loss_curve_doc['n']} 个点）")
    note_parts = []
    if crash is not None and crash["applies"]:
        # ★ D9 的验收句（逐字模板）：「训练崩溃，未产生有效训练：exit_code=<原值>，<报错首行原文>
        #   （stdout.log:<行号>）」—— 放在 note 首位，人读件一眼看到真因。
        note_parts.append(_crash_note(crash, exit_code))
    if note_extra:
        note_parts.append(note_extra)
    note_parts.append(f"退出码 {exit_code}" if exit_code is not None else "退出码缺（未取到跑次退出码）")
    note_parts.append(f"实跑步数 {steps_actual}（{steps_why}；计划 {mt['expected_optimizer_steps']} 步）"
                      if isinstance(steps_actual, int) else
                      f"计划 {mt['expected_optimizer_steps']} 步，实跑未知")
    note_parts.append(
        f"完成轮数 {epochs_done}/{epochs_planned} 个 epoch（{epochs_why}）"
        if isinstance(epochs_done, int) and epochs_planned is not None else
        f"计划 {epochs_planned if epochs_planned is not None else '未知'} 个 epoch，实跑轮数未知"
        f"（{epochs_why}）")
    note_parts.append(f"检查点：{loadable_why}")
    note_parts.append(f"数值异常：{naninf_why}")
    note_parts.append(f"峰值显存：{peak_why}")
    note_parts.append(f"效果读数：{ev_why}")
    if algo == "CPT":
        note_parts.append(f"训练 loss 序列：{loss_curve_why}"
                          if loss_curve_doc is not None
                          else f"训练 loss 序列缺（{loss_curve_why}）")
    note_parts.extend(_teacher_note(algo, ev))
    note_parts.extend(_checks_note(checks, checks_ev, row))
    if artifact is not None:
        if artifact["applies"]:
            _d = artifact.get("elapsed_derived")
            note_parts.append(
                f"退出码按证据重派生：原始 {artifact['exit_code_raw']} → "
                f"{artifact.get('exit_code_derived')}（训练已按主流程返回：见步骤跑满、"
                f"轮数跑满、检查点可加载、无 NaN/Inf 四项实测）"
                + (f"；墙钟改取「起跑 → 主流程返回」= {_d} s"
                   f"（原始 {artifact.get('elapsed_raw')} s 含收尾空等）" if _d else ""))
        else:
            note_parts.append(f"退出码按证据重派生未成立（保留原始读数）：{artifact['why']}")
    if operator_stop is not None:
        if operator_stop["applies"]:
            note_parts.append(_operator_stop_note(operator_stop))
        else:
            note_parts.append(f"退出码按「运维停容器」证据重派生未成立（保留原始读数）："
                              f"{operator_stop['why']}")
    comparable = readings.get("comparable")
    if comparable is False:
        note_parts.append("环境快照冻结维度发生变化 ⇒ 本档读数不可比（不得悄悄重跑）")
    if crash is not None and crash["applies"]:
        # ★ 每一条"缺 XX 读数"都是"训练崩溃"的直接后果 ⇒ 收进证据里留痕，不再让本文件自判
        #   "证据不全"（rc=4）。⛔ 这不是放宽门槛：放行后回填器按判据① 判的是"退出码≠0 +
        #   0/N 轮 + 检查点不可加载" ⇒ 状态必然是 `不通过`，不可能变成 `通过`。
        crash["explained_gaps"] = list(gaps)
        gaps = []
    if gaps:
        note_parts.append("证据缺口（不满足回填条件，不补数）：" + "；".join(gaps))

    doc = {
        "schema": SCHEMA,
        "tier": tier,
        "params": {
            "model": str(row.get("model") or ""),
            "algo": str(row.get("algo") or ""),
            "mode": str(row.get("mode") or ""),
            "backend": str(row.get("backend") or ""),
            "parallel": str(row.get("parallel") or ""),
            "lr": str(row.get("lr") or ""),
            # ★ 身份/语义键一律**逐字取数据件**（回填器按语义归一化后逐项比对；自己拼串会错位）
            "lora_target": str(row.get("lora_target") or ""),
            "data_subset": str(row.get("data_subset") or ""),
            "switches": str(row.get("switches") or ""),
            # 审计用（回填器不比对未知键）：
            "data_subset_detail": f"{mt['data']['subset_size']} 条（{mt['data']['host_relative_path']}）",
            # 审计用（回填器不比对未知键）：
            "tuner_type": mt["tuner_type"],
            "nproc_per_node": mt["parallel_components"]["nproc_per_node"],
            "tp": mt["parallel_components"]["tp"],
            "pp": mt["parallel_components"]["pp"],
            "dp": mt["parallel_components"]["dp"],
            "max_epochs": mt["max_epochs"],
        },
        "run_result": {
            "exit_code": exit_code,
            # ★ 判据① 的**新口径分母/分子**（Epoch）：缺读数写 null（回填器判 unknown，fail-closed）
            "epochs_completed": epochs_done,
            "epochs_planned": epochs_planned,
            "final_ckpt_loadable": loadable,
            "nan_or_inf_detected": nan_inf,
        },
        "steps_actual": steps_actual,
        "elapsed": elapsed,
        "peak_mem": peak_mem,
        "throughput": throughput,
        "effect": effect,
        "note": "；".join(note_parts),
        "run_path": str(ckpt_path or run_dir),
        "provenance": {
            "produced_by": "scripts/make_result_json.py（工作包 B / task-manifest）",
            "command": readings.get("command") or f"bash tests/e2e/run_matrix68.sh --only {tier}",
            "evidence_paths": [str(p) for p in _evidence_paths(
                run_dir, ckpt_path, checks_ev, _extra_evidence(epochs_ev, naninf_sources,
                                                              loss_curve_doc))],
            # 判据① 的 epoch 读数坐标（哪个文件、哪个字段、第几行；审计用，不进人读件）
            "epoch_evidence": epochs_ev,
            # 实跑步数的读数坐标（哪个文件第几行、哪种形态、多源取最大的依据；审计用）
            "steps_evidence": steps_prov,
            # 判据③④ 的证据链（有据才落；审计用，不进人读件）
            "checks_evidence": checks_ev,
        },
    }
    # D1 超时伪影的重派生取证（只在显式开关下出现；审计用，不进人读件）
    if artifact is not None:
        doc["provenance"]["timeout_artifact_evidence"] = artifact
    # D1 的第二条通道（人工按授权停僵尸容器）的取证；同样只在显式开关下出现、审计用
    if operator_stop is not None:
        doc["provenance"]["operator_stop_evidence"] = operator_stop
    # D9 崩溃档的取证（只在显式开关下出现；含四条判据的 file:line 坐标 + 被解释掉的证据缺口）
    if crash is not None:
        doc["provenance"]["crashed_tier_evidence"] = crash
    if str(row.get("target_modules") or "").strip():
        doc["params"]["target_modules"] = str(row["target_modules"])
    if ev is not None:
        doc["eval"] = ev
    # 判据③④：只有真的有结构化结论时才写（⛔ 不写 false/null 冒充"判过了"）
    if checks:
        doc["checks"] = checks
    return doc, {"evidence_gaps": gaps}


def _teacher_note(algo: str, ev: dict | None) -> list[str]:
    """判据② 的 OPD 教师读数**备注列**文字（中性措辞：⛔ 无内部路径/工位名/过程行话）。

    只写"规则 + 可核数"：教师 = 27B / SFT / LoRA / dp=1 档产物，教师线由回填器按数据件现算
    （⛔ 本文件不写阈值数值）。
    """
    if algo != "OPD":
        return []
    t = (ev or {}).get("teacher_after_acc")
    b = (ev or {}).get("base_acc")
    if not isinstance(t, (int, float)):
        return ["判据②：教师侧读数缺（未跑教师侧评测）⇒ 判据② 不可判定"]
    txt = f"判据②：教师在训练数据集上的完全正确率 {float(t) * 100:.2f}%"
    if isinstance(b, (int, float)):
        txt += f"；教师 − 学生未训练模型 = {(float(t) - float(b)) * 100:+.2f} pp"
    return [txt]


def _algo_signing(row_algo: str, detail: dict) -> dict:
    """判据③ 的**签署信息**（从 `checks_evidence.algo.detail` 现读，⛔ 不由布尔反推）。

    为什么必须看证据：备注列会进人读发布件。此前只要 `algo_checks_passed is True` 就印
    「按《GRASPO 算法说明》第 9 节逐条算法检查，全部不变式通过」——档号错配时**一份签署件都
    没有**也照印（审查第 3 轮 🔴-2 的不实陈述）。⇒ 现在只把证据自己声明的
    `basis_id` + `checker` + `checked_at` 搬进文案；三者不齐 ⇒ 只写"证据不完整"，
    ⛔ 不出现任何"已通过检查"字样。
    """
    basis = str(detail.get("basis_id") or "")
    checker = str(detail.get("checker") or "").strip()
    at = str(detail.get("checked_at") or "").strip()
    graspo_basis, public_basis = _basis_ids()
    if graspo_basis and basis == graspo_basis and checker and at:
        return {"kind": "graspo", "checker": checker, "checked_at": at}
    if public_basis and basis == public_basis and checker and at:
        return {"kind": "public", "checker": checker, "checked_at": at,
                "method": str(detail.get("method") or row_algo or "").lower()}
    return {"kind": None, "basis_id": basis or None, "checker": checker or None,
            "checked_at": at or None}


def _checks_note(checks: dict, checks_ev: dict, row: dict) -> list[str]:
    """判据③④ 的**备注列**文字（中性措辞：⛔ 无内部路径/工位名/过程行话）。

    只写"结论 + 可核数 + 签署人/时间"，不写证据路径（路径在 `provenance.checks_evidence`
    里供审计）。★ 判据③ 的措辞**只能**由证据自身的声明生成（见 :func:`_algo_signing`）。
    """
    out: list[str] = []
    algo = str(row.get("algo") or "").upper()
    alg_ev = (checks_ev.get("algo_checks_passed")
              if isinstance(checks_ev.get("algo_checks_passed"), dict) else {})
    detail = alg_ev.get("detail") if isinstance(alg_ev.get("detail"), dict) else {}
    if checks.get("algo_checks_passed") is True:
        sign = _algo_signing(algo, detail)
        if sign["kind"] == "graspo":
            out.append(f"判据③：按《GRASPO 算法说明》第 9 节逐条算法检查，全部不变式通过"
                       f"（签署人 {sign['checker']}，{sign['checked_at']}）")
        elif sign["kind"] == "public":
            out.append(f"判据③：{row.get('algo') or '本算法'} 按 public-algos 签署件核对损失函数与"
                       f"标签掩码处理规则"
                       f"{'（含教师信号规则）' if sign.get('method') == 'opd' else ''}"
                       f"，结论通过（签署人 {sign['checker']}，{sign['checked_at']}）")
        else:
            # ★ 没有签署留痕 ⇒ 不作"已通过检查"的陈述（⛔ 发布件不得自证）
            out.append("判据③：算法正确性检查证据缺签署人/时间，本节不作结论陈述")
    elif checks.get("algo_checks_passed") is False:
        out.append("判据③：算法正确性检查未通过")
    inf_ev = (checks_ev.get("inference_parseable")
              if isinstance(checks_ev.get("inference_parseable"), dict) else {})
    inf = inf_ev.get("detail") if isinstance(inf_ev.get("detail"), dict) else {}
    if checks.get("inference_parseable") is True:
        out.append(f"判据④：训练数据集 {inf.get('parse_ok')}/{inf.get('expected')} 条推理输出可解析"
                   f"（vLLM 温度 {inf.get('temperature')}）")
    elif checks.get("inference_parseable") is False:
        out.append(f"判据④：推理结果不满足可解析要求"
                   f"（可解析 {inf.get('parse_ok')}/{inf.get('expected')} 条）")
    return out


def _extra_evidence(epochs_ev: dict | None, naninf_sources: list[Path] | None,
                    loss_curve_doc: dict | None) -> list[Path]:
    """判据①（epoch / NaN 证据源）与判据② CPT（loss 序列）的证据原件 —— 在盘才登记。"""
    out: list[Path] = []
    for ev in (epochs_ev, (loss_curve_doc or {}).get("evidence")):
        src = (ev or {}).get("source") if isinstance(ev, dict) else None
        if isinstance(src, str) and Path(src).is_file():
            out.append(Path(src))
    for p in naninf_sources or []:
        if Path(p).is_file():
            out.append(Path(p))
    return out


def _evidence_paths(run_dir: Path, ckpt_path: str | None,
                    checks_ev: dict | None = None,
                    extra: list[Path] | None = None) -> list[Path]:
    out = []
    for rel in ("stdout.log", "exit_code", "gpu/peak_mem.json", "tier.json",
                "env-snapshot-before.json", "env-snapshot-after.json", "env-diff.txt"):
        p = run_dir / rel
        if p.exists():
            out.append(p)
    if ckpt_path:
        out.append(Path(ckpt_path))
    # 判据③④ 的证据原件（评测报告 / 算法检查签署件）—— 存在于盘才登记
    for key in ("algo_checks_passed", "inference_parseable", "teacher_after_acc"):
        src = ((checks_ev or {}).get(key) or {}).get("detail", {}).get("source")
        if isinstance(src, str) and Path(src).exists() and Path(src) not in out:
            out.append(Path(src))
    for p in extra or []:                     # 判据①（epoch / NaN）/ 判据② CPT（loss）的原件
        if p.exists() and p not in out:
            out.append(p)
    if not out:
        out = [run_dir]
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="跑次读数 ⇒ 回填口径结果 json（matrix-run/v1）")
    ap.add_argument("--tier", required=True)
    ap.add_argument("--run-dir", required=True,
                    help="跑次目录（内含 readings.json / stdout.log / gpu/…；判据③④ 结论取自 "
                         "readings.json 的 checks{} —— 由 scripts/record_checks.py 产出）")
    ap.add_argument("--out", default=None, help="默认 <run-dir>/<tier>.json")
    ap.add_argument("--manifest", default=str(ROOT / "tests" / "e2e" / "matrix68_manifest.json"))
    ap.add_argument("--data", default=str(ROOT / "docs" / "capability-matrix.data.json"))
    ap.add_argument("--eval-dir", default=None, help="评测根（含 .local/reports/{after,base}/eval_report.json）")
    ap.add_argument("--tier-json", default=None, help="scripts/gen_tier_json.py 的产物（含 after/base/delta）")
    ap.add_argument("--ledger", default=None,
                    help="collect_results.py 的 ledger.jsonl（提供则峰值显存用矩阵冻结口径 "
                         "peak_memory_gib，否则用宿主侧采样并显式标注口径）")
    ap.add_argument("--note", default="", help="附加备注（写进 note 列）")
    ap.add_argument("--recover-operator-stopped-container", action="store_true",
                    help="D1 事后取证之二（默认关）：把「训练已按主流程返回、容器却挂在收尾等待上、"
                         "由运维按授权 docker stop」的老跑次按证据重派生 exit_code（判据见 "
                         "operator_stop_evidence 的 docstring；条件不全 ⇒ 一个字段都不改）。"
                         "⛔ 与 --recover-timeout-artifact 并列、互不影响，两条都默认关。")
    ap.add_argument("--recover-timeout-artifact", action="store_true",
                    help="D1 事后取证（默认关）：把「训练已按主流程返回、却被外层 timeout 杀掉」"
                         "的老跑次按证据重派生 exit_code/elapsed（判据见 "
                         "timeout_artifact_evidence 的 docstring；条件不全 ⇒ 一个字段都不改）。"
                         "⛔ 不用它替代正常跑次，也不手改任何读数。")
    ap.add_argument("--recover-crashed-tier", action="store_true",
                    help="D9 崩溃档入表（默认关）：把「实测崩溃、一步优化都没跑、证据齐全」的老跑次"
                         "按证据如实入表（判据见 crashed_tier_evidence 的 docstring：退出码≠0 且 "
                         "∉{124,137} ∧ stdout.log 可定位报错原文与行号 ∧ 无任何训练产物 ∧ "
                         "tier.json 在）。⛔ 不改判据、不改读数：只把「一步没跑」写成 0 步/0 轮/"
                         "无检查点，最终状态由回填器算出（必然是「不通过」）；条件不全 ⇒ 一个字段都不改。")
    ap.add_argument("--collect-dir", default=None,
                    help="回填入口：额外落一份扁平副本 <dir>/<tier>.json（runner 传 <run-root>/collect；"
                         "回填器 `--runs-dir` 只吃扁平目录、且拒 stray）")
    args = ap.parse_args(argv)

    run_dir = Path(args.run_dir)
    if not run_dir.is_dir():
        print(f"FAIL: --run-dir 不存在：{run_dir}")
        return 2
    # ── ★ 档号护栏（审查第 3 轮 🔴-2）──────────────────────────────────────────
    #   `--tier` 与 `--run-dir` 错配时，本文件会把**别的档**的读数与 checks 装进本档的结果 json，
    #   回填器照判「通过」、备注列还会印出不实的"逐条检查通过"。⇒ readings.json 的 `tier`
    #   必须存在且与 `--tier` 逐字相同；⛔ 不自动纠正、不覆盖、不"取跑次目录名兜底"。
    readings = read_json(run_dir / "readings.json", {}) or {}
    rtier = readings.get("tier")
    if not isinstance(rtier, str) or not rtier.strip():
        print(f"FAIL: readings.json 缺 tier 字段（--tier={args.tier}，readings.tier={rtier!r}）"
              "⇒ 无法证明该跑次目录属于本档；⛔ 不自动纠正、不覆盖")
        return 2
    if rtier.strip() != args.tier.strip():
        print(f"FAIL: 档号错配：readings.json tier={rtier.strip()} ≠ --tier={args.tier.strip()}"
              "（⛔ 不自动纠正/不覆盖；请改用与该跑次一致的 --tier，或换 --run-dir）")
        return 2
    manifest = read_json(Path(args.manifest))
    if not isinstance(manifest, dict) or not manifest.get("tiers"):
        print(f"FAIL: 清单读不出 tiers：{args.manifest}")
        return 2
    matrix = read_json(Path(args.data))
    if not isinstance(matrix, dict) or not matrix.get("rows"):
        print(f"FAIL: 数据件读不出 rows：{args.data}")
        return 2
    data_rows = {r["tier"]: r for r in matrix["rows"]}

    doc, diag = build_doc(tier=args.tier, run_dir=run_dir, manifest=manifest,
                          data_rows=data_rows, eval_dir=Path(args.eval_dir) if args.eval_dir else None,
                          tier_json=Path(args.tier_json) if args.tier_json else None,
                          note_extra=args.note,
                          ledger=Path(args.ledger) if args.ledger else None,
                          recover_timeout_artifact=args.recover_timeout_artifact,
                          recover_operator_stop=args.recover_operator_stopped_container,
                          recover_crashed_tier=args.recover_crashed_tier)

    out = Path(args.out) if args.out else run_dir / f"{args.tier}.json"
    text = json.dumps(doc, ensure_ascii=False, indent=1) + "\n"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    print(f"wrote {out}  bytes={len(text.encode())}")

    # ── 用**回填器自己的校验器**自检（★ 契约不靠本文件自述）────────────────────
    bf = load_module(ROOT / "tools" / "matrix" / "backfill_matrix.py", "_backfill_contract")
    errs = bf.validate_run(doc, args.tier, data_rows)
    if errs:
        print("FAIL: 回填器校验器判不通过（这些字段/身份口径不对）：")
        for e in errs:
            print(f"  - {e}")
        print("  ⇒ 结果 json 已落盘供审计，但**不得**当作可回填件（rc=1）")
        return 1

    if args.collect_dir:
        cdir = Path(args.collect_dir)
        cdir.mkdir(parents=True, exist_ok=True)
        (cdir / f"{args.tier}.json").write_text(text, encoding="utf-8")
        print(f"collected {cdir / (args.tier + '.json')}（扁平目录，供 backfill_matrix --runs-dir）")

    if diag["evidence_gaps"]:
        print("WARN: 结构合法但证据不全（回填器会在判据层 fail-closed 拒收；⛔ 不补数）:")
        for g in diag["evidence_gaps"]:
            print(f"  - {g}")
        return 4

    print(f"OK: 通过回填器校验器（tier={args.tier} steps_actual={doc['steps_actual']} "
          f"elapsed={doc['elapsed']} peak_mem={str(doc['peak_mem'])[:40]}…）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
