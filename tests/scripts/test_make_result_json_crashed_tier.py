"""`scripts/make_result_json.py` 的 **D9 崩溃档入表**通道：实测崩溃、证据齐全的档如实落 `不通过`。

## 为什么有这条通道（实证来源，⛔ 不是凭空造的 fixture）

M011（9B / SFT / 全参 / native / 单卡，228 实测 2026-10-25，在役树
`<tree>/runs/M011/`）：

* `exit_code` = `1`（`runs/M011/exit_code` 逐字）、`elapsed_sec` = `334.0`；
* `stdout.log` 第 20 行是**逐字**的
  `ValueError: lora.target_modules is required (…)`，最近的一帧是第 18 行
  `File "/workspace/graspo/src/graspo/core/lora.py", line 170, in resolve_lora_target_modules`
  —— 即 N1：全参档仍去解析 LoRA 目标（`model_adapter.py:61` 早于 `full_param` 定义）；
* **没有任何训练产物**：无 `checkpoint-*` / `final`、无 `trainer_state.json`
  ⇒ 崩溃发生在 `_load_model`，一步优化都没跑（⛔ 不是"跑了一半"）；
* `tier.json` 在（`algorithm=SFT / tuner_type=full / backend=native / nproc=1`）⇒ 能证明跑的是这档。

官方三条「仅失败」通道此前**全不通**：`backfill_matrix.py` 没有开关；
`gen_tier_json.py` 的 `MODE_ALLOWED` 与 `MODE_BY_TUNER` 自相矛盾；本脚本缺 `steps_actual`/
`throughput` 时自判"证据不全"（rc=4）⇒ **实测崩溃、证据齐全的档一格写不进去**，台账只能显示
"待测试"。

## 红线（本模块逐条钉住）

* 开关 **默认关**：不传 `recover_crashed_tier` ⇒ 结果 json **逐字节**与修复前相同
  （golden = `fixtures/m011_crash_default_doc.json`，由**修复前**的代码生成）。
* 四条证据门槛**全部**成立才放行；任一不成立 ⇒ `applies=False`、一个字段都不改、rc≠0。
* ⛔ 绝不给"跑了一半 / 超时被杀"的档用这条通道：`steps_actual>0`、有可加载 checkpoint、
  或 `exit_code ∈ {124,137}`（属 D1 的两条通道）⇒ 一律拒绝。
* ⛔ 不改判据：本通道只让「退出码≠0 且一步没跑」这一事实进入判据①，最终状态由回填器算出。
"""

import json
import pathlib
import sys

import pytest

_ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "scripts"))
sys.path.insert(0, str(_ROOT / "tools" / "matrix"))
import make_result_json as m  # noqa: E402
import note_style  # noqa: E402
import backfill_matrix as bf  # noqa: E402

_MANIFEST = _ROOT / "tests" / "e2e" / "matrix68_manifest.json"
_DATA = _ROOT / "docs" / "capability-matrix.data.json"
_GOLDEN = _ROOT / "tests" / "scripts" / "fixtures" / "m011_crash_default_doc.json"
#: ★ D29 重基线前的同一 golden（修复前代码产出、原样留档）—— 机核「只许 note 一段变」的对照件
_BEFORE = _ROOT / "tests" / "scripts" / "fixtures" / "m011_crash_default_doc.BEFORE-D29.json"
_RUNDIR_PLACEHOLDER = "<RUNDIR>"

#: M011 的 `runs/M011/stdout.log` **逐字**原文（228 在役树，52 行；此处保留前 20 行 +
#: 收尾的 torchrun 失败块 —— 判据要的"报错原文 + 行号"全部在这 20 行里，行号与真件一致）。
M011_STDOUT = """[train-worker] GPU lock OK: NVIDIA_VISIBLE_DEVICES=0
Traceback (most recent call last):
  File "<frozen runpy>", line 198, in _run_module_as_main
  File "<frozen runpy>", line 88, in _run_code
  File "/workspace/graspo/src/graspo/cli/train_worker.py", line 147, in <module>
    main()
  File "/workspace/graspo/src/graspo/cli/train_worker.py", line 143, in main
    builder(config, selection).train(smoke=args.smoke)
  File "/workspace/graspo/src/graspo/flow/backend/native/trainer/sft_trainer.py", line 154, in train
    self.runtime.setup()
  File "/workspace/graspo/src/graspo/flow/backend/native/runtime.py", line 270, in setup
    self._adapter.setup()
  File "/workspace/graspo/src/graspo/flow/backend/native/transformer_model_adapter.py", line 286, in setup
    self._load_model(hf_config, model_path)
  File "/workspace/graspo/src/graspo/flow/backend/native/models/qwen35_36/model_adapter.py", line 61, in _load_model
    lora_targets = resolve_lora_target_modules(
                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/workspace/graspo/src/graspo/core/lora.py", line 170, in resolve_lora_target_modules
    raise ValueError(
ValueError: lora.target_modules is required (a single real HF module-name regex; both backends share it). lora.target_preset no longer selects LoRA targets — it is kept only so pre-existing config files keep parsing. Set lora.target_modules to the shared regex '^(?:model\\.)?language_model\\..*\\.(q_proj|k_proj|v_proj|o_proj|in_proj_z|out_proj|in_proj_qkv|gate_proj|up_proj|down_proj)$|^(?:model\\.)?visual\\.(blocks\\.\\d+\\.(attn\\.(qkv|proj)|mlp\\.linear_fc1|mlp\\.linear_fc2)|merger\\.linear_fc1|merger\\.linear_fc2)$' or an explicit regex.
E1025 10:34:02.421000 58 torch/distributed/elastic/multiprocessing/api.py:986] failed (exitcode: 1) local_rank: 0 (pid: 194) of binary: /usr/bin/python3
"""

#: 真件里那条报错原文（判据要求 note 里**逐字**出现）
M011_ERROR = (
    "ValueError: lora.target_modules is required (a single real HF module-name regex; both "
    "backends share it). lora.target_preset no longer selects LoRA targets — it is kept only "
    "so pre-existing config files keep parsing. Set lora.target_modules to the shared regex "
    "'^(?:model\\.)?language_model\\..*\\.(q_proj|k_proj|v_proj|o_proj|in_proj_z|out_proj|"
    "in_proj_qkv|gate_proj|up_proj|down_proj)$|^(?:model\\.)?visual\\.(blocks\\.\\d+\\.("
    "attn\\.(qkv|proj)|mlp\\.linear_fc1|mlp\\.linear_fc2)|merger\\.linear_fc1|merger\\.linear_fc2)$' "
    "or an explicit regex."
)


def _fake_crash_run(tmp_path, *, exit_code=1, elapsed=334.0, stdout=None, tier="M011",
                    tier_json=True, checkpoint=False, trainer_state=False):
    """造一个"M011 形态"的跑次目录：起跑即崩、一步没跑、无任何训练产物。"""
    run = tmp_path / tier
    run.mkdir()
    (run / "stdout.log").write_text(stdout if stdout is not None else M011_STDOUT, encoding="utf-8")
    if exit_code is not None:
        (run / "exit_code").write_text(f"{exit_code}\n", encoding="utf-8")
    if elapsed is not None:
        (run / "elapsed_sec").write_text(f"{elapsed}\n", encoding="utf-8")
    readings = {"tier": tier, "elapsed_sec": elapsed,
                "config": f"samples/configs/matrix68/{tier}.yaml",
                "image": "graspo:v0.28.11-cu130fix-v2-msgspec-20261012-run",
                "timeout_sec": 57600,
                "command": f"bash tests/e2e/run_matrix68.sh --only {tier}",
                "note": "elapsed_sec = runner 起止墙钟差；peak_mem 见 gpu/peak_mem.json"}
    if exit_code is not None:
        readings["exit_code"] = exit_code
    (run / "readings.json").write_text(json.dumps(readings, indent=1), encoding="utf-8")
    gpu = run / "gpu"
    gpu.mkdir()
    (gpu / "peak_mem.json").write_text(json.dumps({
        "caliber": "宿主侧 nvidia-smi memory.used 采样峰值（MiB，每 2 s 一帧）",
        "gpus": ["3"], "peak_used_mib_by_gpu": {"3": 425.0},
        "peak_used_mib": 425.0, "measured": True}, indent=1), encoding="utf-8")
    if tier_json:
        (run / "tier.json").write_text(json.dumps({
            "cards": 1, "config": f"samples/configs/matrix68/{tier}.yaml", "subset_size": 100,
            "nproc": 1, "max_epochs": 4, "planned_steps": 100, "derived_steps": 400,
            "algorithm": "SFT", "mode": "全参", "tuner_type": "full", "backend": "native",
            "lr": "5.0e-06", "parallel": "tp=1 pp=1 dp=1（1 卡）", "train_method": "sft"},
            ensure_ascii=False), encoding="utf-8")
    if checkpoint or trainer_state:
        ck = run / tier / "v0-20261025-103300" / "checkpoint-200"
        ck.mkdir(parents=True)
        (ck / "adapter_config.json").write_text(json.dumps({"r": 32}), encoding="utf-8")
        (ck / "trainer_state.json").write_text(
            json.dumps({"global_step": 200, "epoch": 1.0, "max_steps": 400}), encoding="utf-8")
    return run


def _manifest_and_rows():
    if not _MANIFEST.is_file() or not _DATA.is_file():
        pytest.skip("仓内清单/数据件不在（只跑单元层）")
    manifest = json.loads(_MANIFEST.read_text(encoding="utf-8"))
    matrix = json.loads(_DATA.read_text(encoding="utf-8"))
    return manifest, {r["tier"]: r for r in matrix["rows"]}


def _build(run, **flags):
    manifest, rows = _manifest_and_rows()
    doc, diag = m.build_doc(tier="M011", run_dir=run, manifest=manifest, data_rows=rows,
                            eval_dir=None, tier_json=None, note_extra="", **flags)
    return doc, diag


# ── ① 证据全 ⇒ 放行，且 note 里**逐字**有报错原文与日志行号 ────────────────────
def test_crash_tier_admitted_with_full_evidence(tmp_path):
    run = _fake_crash_run(tmp_path)
    doc, diag = _build(run, recover_crashed_tier=True)
    ev = doc["provenance"]["crashed_tier_evidence"]
    assert ev["applies"] is True, ev["why"]

    # ⛔ 原退出码一个字都不改（判据① 靠它判"有据且不过"）
    assert doc["run_result"]["exit_code"] == 1
    # 一步没跑 = 0 步 / 0 轮（证据化，不是占位）
    assert doc["steps_actual"] == 0 and doc["run_result"]["epochs_completed"] == 0
    assert doc["run_result"]["final_ckpt_loadable"] is False
    # 证据缺口全部由"训练崩溃"这件事实解释 ⇒ 不再自判"证据不全"（rc=4）
    assert diag["evidence_gaps"] == [], diag["evidence_gaps"]

    # ★ E1（2026-10-27 主席裁定）改了**展示口径**：备注列只说人话 —— 报错只留
    #   「异常类名 ＋ 正文前 ≤40 字…」，⛔ 不再把整段报错原文/正则/路径/日志坐标抄进表格。
    note = doc["note"]
    assert note.startswith("训练崩溃，未产生有效训练：ValueError: lora.target_modules is required"), note
    assert "…" in note, f"超长报错必须截断并留省略号：{note}"
    assert "由于" not in note and "(" not in note, f"报错摘要里不得带英文帮助/括号原文：{note}"
    # ⛔ 备注里不许出现工程日志形态（坐标 / 绝对路径 / file:line / 正则原文）
    assert not note_style.has_noise(note), note
    assert len(note) <= note_style.NOTE_LIMIT, f"{len(note)} 字超上限：{note}"
    # ★ 诚实没有丢：**完整报错原文与日志行号仍在证据件里**（句句可核）
    assert M011_ERROR in ev["error"]["text"], "证据件里没有逐字的报错原文"
    assert ev["error"]["log_line"] == 20, ev["error"]
    assert "stdout.log:20" not in note, "日志坐标不该进备注"
    # 证据坐标：日志行号 + 源码帧（lora.py:170）
    assert ev["error"]["log_line"] == 20
    assert ev["error"]["frame"]["file"].endswith("core/lora.py")
    assert ev["error"]["frame"]["line"] == 170


# ── ② 回填器按四季判据把这档判成 `不通过`（状态由回填器算，⛔ 不是本文件自报） ──
def test_backfiller_judges_it_as_not_passed(tmp_path):
    run = _fake_crash_run(tmp_path)
    doc, _ = _build(run, recover_crashed_tier=True)
    _manifest, rows = _manifest_and_rows()
    fm = bf.frozen_map_from_caliber(
        json.loads(_DATA.read_text(encoding="utf-8"))["meta"]["criteria_caliber"]["threshold"])
    status, fails, _unks = bf.judge_row(doc, rows["M011"], fm, None)
    assert status == "不通过", (status, fails)
    assert any("退出码 1 ≠ 0" in f for f in fails), fails
    # ⛔ 结果 json 不得自带 status（口径：状态只能由回填器算）
    assert bf.validate_run(doc, "M011", rows) == [], bf.validate_run(doc, "M011", rows)


# ── ③ 缺 exit_code ⇒ 拒（fail-closed，一个字段都不改） ────────────────────────
def test_missing_exit_code_rejected(tmp_path):
    run = _fake_crash_run(tmp_path, exit_code=None)
    doc, diag = _build(run, recover_crashed_tier=True)
    ev = doc["provenance"]["crashed_tier_evidence"]
    assert ev["applies"] is False
    assert "exit_code" in ev["why"], ev["why"]
    assert doc["steps_actual"] is None, "通道未成立却改了读数"
    assert "训练崩溃，未产生有效训练" not in doc["note"]
    assert diag["evidence_gaps"], "未成立时仍必须报证据缺口（不能假装齐备）"


# ── ④ 缺报错行号 ⇒ 拒（日志里只有异常文本、没有 `File "…", line N` 帧） ────────
def test_missing_error_line_number_rejected(tmp_path):
    run = _fake_crash_run(
        tmp_path, stdout="[train-worker] GPU lock OK\nValueError: boom（没有 traceback 帧）\n")
    doc, diag = _build(run, recover_crashed_tier=True)
    ev = doc["provenance"]["crashed_tier_evidence"]
    assert ev["applies"] is False
    assert "行号" in ev["why"] or "帧" in ev["why"], ev["why"]
    assert doc["steps_actual"] is None
    assert diag["evidence_gaps"]


# ── ⑤ 已有 checkpoint / 已训练若干步 ⇒ 拒（那是"跑了一半"，属另一条通道） ────────
@pytest.mark.parametrize("kwargs", [{"checkpoint": True}, {"trainer_state": True}])
def test_existing_training_artifact_rejected(tmp_path, kwargs):
    run = _fake_crash_run(tmp_path, **kwargs)
    doc, _diag = _build(run, recover_crashed_tier=True)
    ev = doc["provenance"]["crashed_tier_evidence"]
    assert ev["applies"] is False
    assert "训练产物" in ev["why"] or "checkpoint" in ev["why"], ev["why"]
    # ⛔ 通道不成立时，读数保持原样（这里是 trainer_state 的 global_step=200，不是被抹成 0）
    assert doc["steps_actual"] == 200, doc["steps_actual"]


# ── ⑥ 超时码（124/137）不得走崩溃通道（那是 D1 的两条既有通道的地盘） ────────────
@pytest.mark.parametrize("code", [124, 137])
def test_timeout_exit_codes_never_use_crash_channel(tmp_path, code):
    run = _fake_crash_run(tmp_path, exit_code=code)
    doc, _diag = _build(run, recover_crashed_tier=True)
    ev = doc["provenance"]["crashed_tier_evidence"]
    assert ev["applies"] is False, ev["why"]
    assert doc["run_result"]["exit_code"] == code


# ── ⑦ 开关默认关 ⇒ 结果 json 与**修复前**逐字节相同（golden 由修复前代码产出） ────
#   ★ 2026-10-27 D29 重基线：主席裁定备注列删掉「计划 N 步」从句、并清掉内部记号（`⛔`），
#     这两处**只改人读文案**（`note` 一个字段）；golden 随之更新，**除 note 外逐字节未动**
#     （前后 diff 与原因见工位 `.local/hb-workspace/20261022-baseline-r1/task-fix-mkevidence/DELIVERY.md`）。
def _normalised_json(doc, run):
    text = json.dumps(doc, ensure_ascii=False, indent=1) + "\n"
    return text.replace(str(run), _RUNDIR_PLACEHOLDER)


def test_default_off_is_byte_identical_to_pre_fix_output(tmp_path):
    run = _fake_crash_run(tmp_path)
    doc, diag = _build(run)                       # ⛔ 不传开关
    assert "crashed_tier_evidence" not in doc["provenance"]
    assert doc["steps_actual"] is None and doc["run_result"]["final_ckpt_loadable"] is None
    assert "训练崩溃，未产生有效训练" not in doc["note"]
    assert diag["evidence_gaps"], "默认关时证据缺口必须照旧（rc=4）"
    if not _GOLDEN.is_file():                     # pragma: no cover - golden 固定入库
        pytest.skip(f"缺 golden：{_GOLDEN}")
    assert _normalised_json(doc, run) == _GOLDEN.read_text(encoding="utf-8"), \
        "默认关时结果 json 与修复前不是逐字节相同 ⇒ 破坏了『未设开关时行为不变』的承诺"


# ── ⑦b D29 重基线**只许** note 一段变（主席 D3 机核：防止顺手放宽别处）──────────────
def test_golden_differs_from_pre_d29_baseline_only_in_note():
    """golden 与「D29 前基线」（`…BEFORE-D29.json`，修复前代码产出并留档）除 note 行外逐字节相同。"""
    before = _BEFORE.read_text(encoding="utf-8")
    now = _GOLDEN.read_text(encoding="utf-8")

    def note_line(text):
        found = [ln for ln in text.splitlines() if ln.lstrip().startswith('"note"')]
        assert len(found) == 1, found
        return found[0]

    lines = now.splitlines()
    idx = next(i for i, ln in enumerate(lines) if ln.lstrip().startswith('"note"'))
    lines[idx] = note_line(before)
    rebuilt = "\n".join(lines) + ("\n" if now.endswith("\n") else "")
    assert rebuilt == before, "golden 与 D29 前基线**除 note 行外**出现了字节差异（⛔ 不许顺手放宽别处）"
    assert note_line(now) != note_line(before), "note 应当随 D29 变化（否则重基线是空操作）"


# ── ⑧ 命令行入口：带开关 rc=0（可入表）；不带开关 rc=4（证据不全，保持待测试） ────
def test_main_exit_codes(tmp_path, capsys):
    run = _fake_crash_run(tmp_path)
    out = tmp_path / "M011.json"
    assert m.main(["--tier", "M011", "--run-dir", str(run), "--out", str(out),
                   "--recover-crashed-tier"]) == 0
    assert json.loads(out.read_text(encoding="utf-8"))["steps_actual"] == 0
    capsys.readouterr()
    # 不传开关 ⇒ 旧链照旧拒收（本 fixture 缺 steps_actual/throughput ⇒ 校验器 rc=1）；总之 ≠ 0
    assert m.main(["--tier", "M011", "--run-dir", str(run), "--out", str(out)]) != 0


# ── ⑨ `[rankN]: ` 前缀形态（torchrun `--tee` ⇒ 每个 worker 的每一行都带前缀）──────────
# ★ 为什么要这一组：**同样的事实，只因日志形态不同就入不了表**。实测 11 档（M004 / M044 / M066
#   与 A800 的 M005 / M006 / M009 / M015 / M028 / M045 / M046 / M051）的报错原文**只出现在**
#   `[rank0]: XxxError: …` 这种行里（`--tee` 前缀），而定位器原先锚定行首 `^` ⇒ 全部定位失败
#   ⇒ fill_one 一律 rc=3、台账只能显示"待测试"。下面这份 fixture 照 M066 真件的形态写：
#   先一段带前缀的 rank 侧 traceback（末行 ImportError），后面是 torchrun launcher 自己的
#   **不带前缀**的 traceback（末行 `ChildFailedError:` 消息为空 ⇒ 不得当作报错原文）。
M066_STDOUT_RANK_PREFIXED = """[rank0]: Traceback (most recent call last):
[rank0]:   File "/workspace/graspo/src/graspo/cli/train_worker.py", line 147, in <module>
[rank0]:     main()
[rank0]:   File "/workspace/graspo/src/graspo/flow/backend/ms_swift/opd_teacher.py", line 88, in build_teacher
[rank0]:     import vllm
[rank0]:   File "/workspace/graspo/src/graspo/flow/backend/ms_swift/vllm_utils.py", line 41, in <module>
[rank0]:     raise ImportError('vLLM is not installed. Please install it with `pip install vllm`.')
[rank0]: ImportError: vLLM is not installed. Please install it with `pip install vllm`.
[rank1]: Traceback (most recent call last):
[rank1]:   File "/workspace/graspo/src/graspo/flow/backend/ms_swift/vllm_utils.py", line 41, in <module>
[rank1]: ImportError: vLLM is not installed. Please install it with `pip install vllm`.
E1025 09:35:14.884000 10 torch/distributed/elastic/multiprocessing/api.py:1012] Sending process 145 closing signal SIGTERM
Traceback (most recent call last):
  File "/usr/local/bin/torchrun", line 6, in <module>
    sys.exit(main())
  File "/usr/local/lib/python3.12/dist-packages/torch/distributed/elastic/multiprocessing/api.py", line 86, in _terminate_process_handler
    raise SignalException(f"Process {os.getpid()} got signal: {sigval}", sigval=sigval)
torch.distributed.elastic.multiprocessing.errors.ChildFailedError:
"""


def test_rank_prefixed_traceback_is_located(tmp_path):
    """`[rankN]: XxxError: …` 形态必须与不带前缀的形态**同等**可定位（否则 11 档入不了表）。"""
    run = _fake_crash_run(tmp_path, stdout=M066_STDOUT_RANK_PREFIXED, elapsed=374.0)
    doc, diag = _build(run, recover_crashed_tier=True)
    ev = doc["provenance"]["crashed_tier_evidence"]
    assert ev["applies"] is True, ev["why"]
    # 认的是**最后一条消息非空的**异常行：前缀行里的 ImportError（⛔ 不是空消息的 ChildFailedError）。
    # 本 fixture 有 rank0/rank1 两条同样的 ImportError ⇒ 末条 = 第 11 行（rank1），口径与不带前缀时一致。
    assert ev["error"]["exception"] == "ImportError", ev["error"]
    assert ev["error"]["log_line"] == 11, ev["error"]
    assert ev["error"]["frame"]["file"].endswith("vllm_utils.py"), ev["error"]
    assert ev["error"]["frame"]["line"] == 41, ev["error"]
    # 前缀只是日志形态，**不得**进人读件的备注
    assert doc["steps_actual"] == 0 and doc["run_result"]["epochs_completed"] == 0
    note = doc["note"]
    assert note.startswith("训练崩溃，未产生有效训练：ImportError: vLLM is not installed"), note
    assert "[rank" not in note, f"rank 前缀不该进备注：{note}"
    assert not note_style.has_noise(note), note
    assert len(note) <= note_style.NOTE_LIMIT, f"{len(note)} 字超上限：{note}"
    assert diag["evidence_gaps"] == [], diag["evidence_gaps"]


def test_rank_prefix_alone_does_not_admit_without_evidence(tmp_path):
    """只加前缀容忍**不放松门槛**：有产物（跑到第 200 步）⇒ 依然拒收（⛔ 不给"跑了一半"用）。"""
    run = _fake_crash_run(tmp_path, stdout=M066_STDOUT_RANK_PREFIXED, checkpoint=True)
    doc, _ = _build(run, recover_crashed_tier=True)
    ev = doc["provenance"]["crashed_tier_evidence"]
    assert ev["applies"] is False, ev
    assert "no_training_artifact" in ev["why"], ev["why"]


def test_unprefixed_form_unchanged_by_prefix_tolerance():
    """不带前缀的原有形态逐条不变（M011 那条 must-not-break 的底线）。"""
    lines = M011_STDOUT.split("\n")
    hit = [ln for ln in lines if m.CRASH_EXC_RE.match(ln)]
    assert hit and hit[-1].startswith("ValueError:"), hit
    frame = [ln for ln in lines if m.CRASH_FRAME_RE.match(ln)]
    assert frame and frame[-1].strip().startswith('File "/workspace/graspo/src/graspo/core/lora.py"'), frame
    # 缩进续行 / `Traceback …` 仍不得被当异常行
    assert not m.CRASH_EXC_RE.match("    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^")
    assert not m.CRASH_EXC_RE.match("Traceback (most recent call last):")
