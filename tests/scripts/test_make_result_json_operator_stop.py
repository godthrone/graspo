"""`scripts/make_result_json.py` 的**第二种** D1 事后取证：容器由运维按授权停止。

## 为什么有这条通道（实证来源，⛔ 不是凭空造的 fixture）

M052（27B / GRASPO / LoRA / ms-swift / 单卡，228 实测）：

* 训练 **正常跑完**：`stdout.log` 第 190368 行
  `[INFO:swift] End time of running main: 2026-10-26 18:22:30.749935`；
  `global_step/max_steps = 6400/6400`；`trainer_state.json` `epoch=8.0`；
  `train_runtime=1.091e+05`；checkpoint-6400 的 `adapter_model.safetensors`
  **1020 个张量**（510 lora_A + 510 lora_B）逐个 data_offsets 自洽、`lora_B` 非全零。
* 但容器**不收尾**：`entry.sh:57` 是 `wait "$SAMPLER"`（无界等待采样器，D1），主流程返回后
  容器一直 Up 且不占 GPU。运维按授权 `docker stop graspo-M052-1792929815`（rc=0，
  `FinishedAt=2026-10-26T18:27:46Z`，`ExitCode=137`）⇒ 跑次目录记下 `exit_code=137`。
* 于是 `--recover-timeout-artifact`（第一条通道）**正确地不认**：它的 ① 要求
  `elapsed ≥ 外层 timeout`，而这里 elapsed=109451 s < timeout=129600 s —— 容器不是被外层
  超时杀掉的。⇒ 补这条并列通道，判据换成"墙钟与 trainer 自报 `train_runtime` 互校（≤1%）"
  加"必须有人工停容器的事实留痕"。

## 红线（本模块逐条钉住）

* 两条通道**并列、互不影响**，都**默认关**：不传开关 ⇒ 结果 json 逐字节不变。
* 任一条件不成立 ⇒ `applies=False` 且 `exit_code_derived is None`（一个字段都不改）。
* ⛔ 不填任何数值：派生出来的 `exit_code=0` 是"九条证据同时成立"的结论，不是常量。
"""

import json
import pathlib
import struct
import sys

import pytest

_ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "scripts"))
import make_result_json as m  # noqa: E402

_MANIFEST = _ROOT / "tests" / "e2e" / "matrix68_manifest.json"
_DATA = _ROOT / "docs" / "capability-matrix.data.json"


# ── 造一份**形态合法**的 safetensors（头 + 数据区；lora_B 可取全零/非零） ──────
def _write_safetensors(path: pathlib.Path, *, lora_b_nonzero: bool) -> None:
    tensors = {}
    blobs = []
    offset = 0
    for name, payload in (
        ("base_model.model.x.lora_A.weight", b"\x01\x02" * 8),
        ("base_model.model.x.lora_B.weight", (b"\x03\x04" * 8) if lora_b_nonzero else bytes(16)),
    ):
        tensors[name] = {"dtype": "BF16", "shape": [8, 2],
                         "data_offsets": [offset, offset + len(payload)]}
        offset += len(payload)
        blobs.append(payload)
    header = json.dumps(tensors, separators=(",", ":")).encode("utf-8")
    path.write_bytes(struct.pack("<Q", len(header)) + header + b"".join(blobs))


def _fake_run(tmp_path, *, exit_code=137, elapsed=109451.0, train_runtime=1.091e5,
              finish=True, steps=6400, steps_pair=None, epochs=(8, 8), lora_nonzero=True,
              attest=True, stop_rc=0, state_exit_code=None):
    """造一个"M052 形态"的跑次目录：主流程返回、容器被运维停掉。"""
    run = tmp_path / "M052"
    run.mkdir()
    pair = steps_pair or f"{steps}/{steps}"
    metrics = (f"{{'train_runtime': {train_runtime!r}, 'global_step/max_steps': "
               f"'{pair}', 'epoch': '{epochs[0]}'}}")
    lines = ["[INFO:swift] begin", f"{{'global_step/max_steps': '{pair}'}}"]
    if finish:
        lines.append("[INFO:swift] End time of running main: 2026-10-26 18:22:30.749935")
    lines.append(metrics)
    (run / "stdout.log").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (run / "readings.json").write_text(json.dumps(
        {"tier": "M052", "exit_code": exit_code, "elapsed_sec": elapsed,
         "timeout_sec": 129600}), encoding="utf-8")
    (run / "exit_code").write_text(f"{exit_code}\n", encoding="utf-8")
    (run / "elapsed_sec").write_text(f"{elapsed}\n", encoding="utf-8")

    ck = run / "M052" / "v0-20261025-120345" / "checkpoint-6400"
    ck.mkdir(parents=True)
    (ck / "adapter_config.json").write_text(json.dumps({"r": 32, "peft_type": "LORA"}),
                                            encoding="utf-8")
    _write_safetensors(ck / "adapter_model.safetensors", lora_b_nonzero=lora_nonzero)
    (ck / "trainer_state.json").write_text(json.dumps(
        {"global_step": steps, "epoch": float(epochs[0]), "num_train_epochs": epochs[1],
         "max_steps": steps}), encoding="utf-8")
    if attest:
        (run / "operator-stop.json").write_text(json.dumps({
            "container": "graspo-M052-1792929815",
            "stop_command": "docker stop graspo-M052-1792929815",
            "stop_rc": stop_rc,
            "started_at_utc": "2026-10-25T12:03:36.056748957Z",
            "stopped_at_utc": "2026-10-26T18:27:36Z",
            "finished_at_utc": "2026-10-26T18:27:46.22656667Z",
            "state_exit_code": exit_code if state_exit_code is None else state_exit_code,
            "state_status": "removing",
            "state_running": False,
            "oom_killed": False,
            "error": "",
            "recorded_at_utc": "2026-10-26T18:28:00Z",
            "authorization": "指挥官授权：只停这一个容器名",
            "raw_inspect": "Status=removing Running=false ExitCode=137 "
                           "FinishedAt=2026-10-26T18:27:46.22656667Z",
        }, indent=1), encoding="utf-8")
    return run


def _evidence(run, **over):
    kw = dict(tier="M052", exit_code=137, elapsed_raw=109451.0, steps_actual=6400,
              steps_planned=800, epochs_done=8, epochs_planned=8, loadable=True,
              ckpt_path=str(next(run.rglob("checkpoint-6400"))), nan_inf=False)
    kw.update(over)
    return m.operator_stop_evidence(run_dir=run, **kw)


# ── ① 条件全过 ⇒ 派 0，墙钟**不动**（它不是空等出来的） ─────────────────────
def test_operator_stop_derives_success_only_with_full_evidence(tmp_path):
    run = _fake_run(tmp_path)
    ev = _evidence(run)
    assert ev["applies"] is True, ev["why"]
    assert ev["exit_code_raw"] == 137 and ev["exit_code_derived"] == 0, ev
    assert ev["elapsed_raw"] == 109451.0, "本条通道不假定墙钟被空等污染 ⇒ 不得改 elapsed"
    assert "elapsed_derived" not in ev or ev["elapsed_derived"] is None, ev
    assert all(c["ok"] for c in ev["conditions"].values()), ev["conditions"]
    assert ev["wallclock_rel_diff"] <= m.OPERATOR_STOP_WALLCLOCK_TOLERANCE, ev
    # 人工停容器的事实必须逐项留痕（容器名 / stop rc / FinishedAt）
    assert ev["operator_stop"]["container"] == "graspo-M052-1792929815", ev["operator_stop"]
    assert ev["operator_stop"]["stop_rc"] == 0
    assert ev["finish_marker"]["timestamp"] == "2026-10-26 18:22:30"


def test_operator_stop_note_states_every_fact_verbatim(tmp_path):
    """note 列必须逐字说清"停容器 + 墙钟互校"两件事（⛔ 不许含糊成"已通过"）。"""
    ev = _evidence(_fake_run(tmp_path))
    note = m._operator_stop_note(ev)
    for token in ("原始退出码 137", "18:22:30", "18:27:46", "运维按授权",
                  "109451 s", "1.091e5 s", "0.3%", "步数/epoch/ckpt/NaN"):
        assert token in note, (token, note)
    assert "不通过" not in note


# ── ② fail-closed：任一条件不成立 ⇒ 一个字段都不改 ────────────────────────────
@pytest.mark.parametrize(
    "over, broken",
    [
        ({}, "trainer_main_returned"),                      # 主流程没返回 ⇒ 真失败
        ({"exit_code": 1}, "raw_is_operator_stop_code"),    # 普通失败码不认
        ({"exit_code": 143}, None),                        # SIGTERM 也认（单独断言）
        ({"loadable": None}, "final_ckpt_loadable"),
        ({"loadable": False}, "final_ckpt_loadable"),
        ({"nan_inf": True}, "nan_or_inf_free"),
        ({"nan_inf": None}, "nan_or_inf_free"),
        ({"epochs_done": 7}, "epochs_completed"),
        ({"steps_actual": 100, "steps_planned": 800, "_run": {"steps": 100,
                                                              "steps_pair": "100/6400"}},
         "steps_completed"),
        ({"elapsed_raw": 1.3e5}, "wallclock_matches_train_runtime"),  # 墙钟含 ~19% 空等
    ],
)
def test_operator_stop_never_fires_without_evidence(tmp_path, over, broken):
    over = dict(over)
    run_kw = over.pop("_run", {})
    run = _fake_run(tmp_path, exit_code=over.get("exit_code", 137),
                    finish=("trainer_main_returned" not in (broken or "")), **run_kw)
    ev = _evidence(run, **over)
    if broken is None:                                      # 143 应认
        assert ev["applies"] is True, ev["why"]
        return
    assert ev["applies"] is False, ev
    assert ev["conditions"][broken]["ok"] is False, ev["conditions"]
    assert ev["exit_code_derived"] is None, "不成立时仍给派生退出码 ⇒ 会掩盖真失败"


@pytest.mark.parametrize(
    "attest_kwargs, broken_fragment",
    [
        ({"attest": False}, "缺人工停容器留痕"),
        ({"stop_rc": 1}, "stop_rc"),
        ({"state_exit_code": 0}, "state_exit_code"),
    ],
)
def test_operator_stop_requires_machine_checkable_attestation(tmp_path, attest_kwargs,
                                                              broken_fragment):
    """没有**可机器核对**的人工停容器留痕 ⇒ 不认（⛔ 自由文本不行）。"""
    run = _fake_run(tmp_path, **attest_kwargs)
    ev = _evidence(run)
    assert ev["applies"] is False, ev
    assert ev["conditions"]["operator_stop_attested"]["ok"] is False
    assert broken_fragment in ev["conditions"]["operator_stop_attested"]["detail"]


def test_operator_stop_rejects_all_zero_lora_b(tmp_path):
    """LoRA B 逐字节全零 = 训练一步没走 ⇒ 内容级证据不成立，⛔ 不许派成功。"""
    run = _fake_run(tmp_path, lora_nonzero=False)
    ev = _evidence(run)
    assert ev["applies"] is False, ev
    assert ev["conditions"]["lora_weights_nonzero"]["ok"] is False
    assert "逐字节全零" in ev["conditions"]["lora_weights_nonzero"]["detail"]


def test_lora_nonzero_probe_reads_real_bytes(tmp_path):
    """直接钉住内容级探针本身（非全零 ⇒ True；全零 ⇒ False；无 lora_B ⇒ None）。"""
    nz = tmp_path / "nz.safetensors"
    _write_safetensors(nz, lora_b_nonzero=True)
    ok, why = m._lora_nonzero_probe(nz)
    assert ok is True, why
    z = tmp_path / "z.safetensors"
    _write_safetensors(z, lora_b_nonzero=False)
    ok, why = m._lora_nonzero_probe(z)
    assert ok is False, why
    plain = tmp_path / "plain.safetensors"
    _write_safetensors(plain, lora_b_nonzero=True)
    raw = plain.read_bytes()
    plain.write_bytes(raw.replace(b"lora_B", b"lora_C"))
    ok, why = m._lora_nonzero_probe(plain)
    assert ok is None and "没有 lora_B" in why, why


# ── ③ 开关默认关：不传开关 ⇒ 结果 json 逐字节不变（含 exit_code 仍 137） ────────
def _real_manifest_and_rows():
    if not _MANIFEST.is_file() or not _DATA.is_file():
        pytest.skip("仓内清单/数据件不在（只跑单元层）")
    manifest = json.loads(_MANIFEST.read_text(encoding="utf-8"))
    matrix = json.loads(_DATA.read_text(encoding="utf-8"))
    return manifest, {r["tier"]: r for r in matrix["rows"]}


def _build(run, **flags):
    manifest, rows = _real_manifest_and_rows()
    doc, _diag = m.build_doc(tier="M052", run_dir=run, manifest=manifest, data_rows=rows,
                             eval_dir=None, tier_json=None, note_extra="", **flags)
    return doc


def test_switch_is_off_by_default(tmp_path):
    doc = _build(_fake_run(tmp_path))
    assert doc["run_result"]["exit_code"] == 137, "默认关时不得动读数"
    assert "operator_stop_evidence" not in doc["provenance"]
    assert "timeout_artifact_evidence" not in doc["provenance"]
    assert "退出码按证据重派生" not in doc["note"]


def test_switch_on_derives_zero_and_keeps_evidence(tmp_path):
    doc = _build(_fake_run(tmp_path), recover_operator_stop=True)
    assert doc["run_result"]["exit_code"] == 0, doc["run_result"]
    ev = doc["provenance"]["operator_stop_evidence"]
    assert ev["applies"] is True and ev["exit_code_raw"] == 137
    # ★ E1：备注只说结论（「退出码 0（按证据重派生，原始 137）」），
    #   "运维按授权停容器 / 墙钟互校"的细节留在证据件里（`_operator_stop_note` 仍逐字保留该文案）
    assert "退出码 0（按证据重派生，原始 137）" in doc["note"], doc["note"]
    assert doc["provenance"]["operator_stop_evidence"]["applies"] is True
    # 墙钟不改：本条通道的 elapsed 就是真实训练墙钟
    assert doc["elapsed"] == pytest.approx(109451.0)


# ── ④ 与 `--recover-timeout-artifact` 并存、互不影响 ─────────────────────────
def test_two_switches_coexist_without_interference(tmp_path):
    """同一跑次：timeout 通道不成立（墙钟没到 timeout）⇒ 单独开它不改；两条同开仍由
    operator 通道派 0，且两段证据各自独立落 provenance（⛔ 不互相吞）。"""
    run = _fake_run(tmp_path)
    only_timeout = _build(run, recover_timeout_artifact=True)
    assert only_timeout["run_result"]["exit_code"] == 137, "超时通道条件不成立却改了读数"
    assert only_timeout["provenance"]["timeout_artifact_evidence"]["applies"] is False

    both = _build(run, recover_timeout_artifact=True, recover_operator_stop=True)
    assert both["run_result"]["exit_code"] == 0
    assert both["provenance"]["timeout_artifact_evidence"]["applies"] is False
    assert both["provenance"]["operator_stop_evidence"]["applies"] is True

    only_op = _build(run, recover_operator_stop=True)
    assert only_op["provenance"]["operator_stop_evidence"]["applies"] is True
    assert "timeout_artifact_evidence" not in only_op["provenance"]
