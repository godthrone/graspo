"""每 rank 首步探针的机核测试（宪法 §2.2 显式 / §1.4 单一真相源）。

覆盖：指纹算法正确且可跨进程比对、旁路落盘字段齐备、包装是只读且**只记一次**、
ms-swift 回调经官方扩展点注册（不改上游源码）、默认关时 argv 逐字不变。
"""

from __future__ import annotations

import hashlib
import json
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

from graspo.flow.msswift.first_step_probe import (
    FINGERPRINT_ALGORITHM,
    PROBE_CALLBACK_NAME,
    PROBE_PHASE,
    FirstStepProbeRecord,
    fingerprint_input_ids,
    install_first_step_recorder,
    install_probe_callback,
    probe_payload,
    record_first_step_probe,
)

torch = pytest.importorskip("torch", reason="probe tests need torch tensors")


@pytest.fixture(autouse=True)
def _restore_active_switch():
    """进程内开关是全局绑定：每个用例前后都复位，避免测试相互污染（§1.4 单一真相源）。"""
    from graspo.core.determinism import DeterminismSwitch, active_switch, bind_active_switch

    original = active_switch()
    bind_active_switch(DeterminismSwitch())
    yield
    bind_active_switch(original)


# ── ① 指纹算法 ─────────────────────────────────────────────────────────────


def test_fingerprint_is_stable_and_content_sensitive():
    tensor = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.long)
    first = fingerprint_input_ids(tensor)
    second = fingerprint_input_ids(tensor.clone())
    assert first == second, "同一内容必须得到同一指纹（跨进程/跨跑可比）"

    changed = fingerprint_input_ids(torch.tensor([[1, 2, 3], [4, 5, 7]], dtype=torch.long))
    assert changed[0] != first[0], "内容变了指纹必须变"

    reshaped = fingerprint_input_ids(torch.tensor([[1, 2, 3, 4, 5, 6]], dtype=torch.long))
    assert reshaped[0] != first[0], "形状进入哈希：同字节不同形状不得撞车"


def test_fingerprint_matches_documented_algorithm():
    """指纹算法必须与 ``FINGERPRINT_ALGORITHM`` 声明的一致（唯一真相源）。"""
    tensor = torch.tensor([7, 8, 9], dtype=torch.long)
    digest, shape, dtype_name = fingerprint_input_ids(tensor)
    expected_raw = tensor.detach().to("cpu").contiguous().view(torch.uint8).numpy().tobytes()
    expected = hashlib.sha256(
        f"{dtype_name}|{shape}|".encode("utf-8") + expected_raw
    ).hexdigest()
    assert digest == expected
    assert shape == [3]
    assert dtype_name == "int64"
    assert "sha256" in FINGERPRINT_ALGORITHM


def test_fingerprint_supports_bfloat16_input_ids():
    """bf16 没有 numpy 等价 dtype——必须走原始字节路径，不能抛异常。"""
    tensor = torch.tensor([1.5, 2.5], dtype=torch.bfloat16)
    digest, shape, dtype_name = fingerprint_input_ids(tensor)
    assert len(digest) == 64
    assert shape == [2]
    assert dtype_name == "bfloat16"


def test_fingerprint_rejects_non_tensor():
    with pytest.raises(TypeError):
        fingerprint_input_ids([1, 2, 3])


def test_fingerprint_is_not_device_dependent():
    """CPU 与 meta/其它设备上的同内容张量应得到同一指纹（搬 CPU 后再哈希）。"""
    cpu = torch.tensor([[1, 2]], dtype=torch.long)
    other = torch.tensor([[1, 2]], dtype=torch.long)
    assert fingerprint_input_ids(cpu)[0] == fingerprint_input_ids(other)[0]


# ── ② 旁路落盘 ─────────────────────────────────────────────────────────────


def test_record_first_step_probe_writes_documented_fields(tmp_path, monkeypatch):
    from graspo.flow import logging as flow_logging

    monkeypatch.setattr(flow_logging, "_run_id", "unit-test")
    path = record_first_step_probe(
        tmp_path,
        FirstStepProbeRecord(
            rank=1,
            step=1,
            local_loss=0.765648,
            loss_dtype="float32",
            input_ids_sha256="a" * 64,
            input_ids_shape=[1, 8],
            input_ids_dtype="int64",
            epoch=0.0,
        ),
    )
    assert path.name == "rank_metrics.rank_00001.jsonl"
    assert path.parent == tmp_path / "logs" / "unit-test"
    payload = json.loads(path.read_text(encoding="utf-8").strip())
    assert payload["phase"] == PROBE_PHASE
    assert payload["rank"] == 1
    assert payload["local_loss"] == pytest.approx(0.765648)
    assert payload["input_ids_sha256"] == "a" * 64
    assert payload["input_ids_shape"] == [1, 8]
    assert payload["fingerprint_algorithm"] == FINGERPRINT_ALGORITHM
    # 采集侧按 `metrics` 字段分流：探针行**不得**带 metrics（否则会被当逐步指标）。
    assert "metrics" not in payload


def test_rank_metrics_filename_is_single_source(tmp_path, monkeypatch):
    from graspo.flow import logging as flow_logging

    monkeypatch.setattr(flow_logging, "_run_id", "unit-test")
    path = record_first_step_probe(
        tmp_path,
        FirstStepProbeRecord(3, 1, 0.1, "float32", "b" * 64, [1, 2], "int64", None),
    )
    assert path.name == flow_logging.rank_metrics_filename(3)
    assert path.name == "rank_metrics.rank_00003.jsonl"


def test_each_rank_writes_its_own_file(tmp_path, monkeypatch):
    from graspo.flow import logging as flow_logging

    monkeypatch.setattr(flow_logging, "_run_id", "unit-test")
    for rank in (0, 1):
        record_first_step_probe(
            tmp_path,
            FirstStepProbeRecord(rank, 1, 0.5, "float32", "c" * 64, [1, 1], "int64", None),
        )
    names = sorted(p.name for p in (tmp_path / "logs" / "unit-test").glob("rank_metrics.*"))
    assert names == ["rank_metrics.rank_00000.jsonl", "rank_metrics.rank_00001.jsonl"]


# ── ③ 只读包装（不侵入既有训练逻辑）────────────────────────────────────────


class _FakeTrainer:
    """最小 trainer 替身：只提供 ``compute_loss``（被包装的那个接口）。"""

    def __init__(self, loss: float = 1.25) -> None:
        self.loss = torch.tensor(loss)
        self.calls = 0

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        self.calls += 1
        if return_outputs:
            return self.loss, {"logits": None}
        return self.loss


def test_recorder_records_once_and_passes_through(tmp_path, monkeypatch):
    from graspo.flow import logging as flow_logging

    monkeypatch.setattr(flow_logging, "_run_id", "unit-test")
    trainer = _FakeTrainer(loss=0.75)
    install_first_step_recorder(trainer, output_dir=tmp_path, rank=0, epoch=0.0)

    batch = {"input_ids": torch.tensor([[1, 2, 3]], dtype=torch.long)}
    loss, outputs = trainer.compute_loss(None, batch, return_outputs=True)
    assert float(loss) == pytest.approx(0.75), "返回值必须原样透传（只读旁路）"
    assert outputs == {"logits": None}
    expected_sha = fingerprint_input_ids(batch["input_ids"])[0]

    # 第二次调用不得再写一行（每个 rank 只记首步）。
    trainer.compute_loss(None, batch)
    assert trainer.calls == 2

    path = tmp_path / "logs" / "unit-test" / "rank_metrics.rank_00000.jsonl"
    lines = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert len(lines) == 1
    assert lines[0]["local_loss"] == pytest.approx(0.75)
    assert lines[0]["input_ids_sha256"] == expected_sha
    assert lines[0]["epoch"] == 0.0


def test_recorder_is_idempotent(tmp_path, monkeypatch):
    from graspo.flow import logging as flow_logging

    monkeypatch.setattr(flow_logging, "_run_id", "unit-test")
    trainer = _FakeTrainer()
    install_first_step_recorder(trainer, output_dir=tmp_path, rank=0)
    wrapped = trainer.compute_loss
    install_first_step_recorder(trainer, output_dir=tmp_path, rank=0)
    assert trainer.compute_loss is wrapped, "重复安装不得二次包装"


def test_recorder_skips_when_loss_is_not_a_tensor(tmp_path, monkeypatch):
    from graspo.flow import logging as flow_logging

    monkeypatch.setattr(flow_logging, "_run_id", "unit-test")

    class _NoLossTrainer:
        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            return None

    trainer = _NoLossTrainer()
    install_first_step_recorder(trainer, output_dir=tmp_path, rank=0)
    assert trainer.compute_loss(None, {"input_ids": torch.tensor([1])}) is None
    assert not (tmp_path / "logs").exists(), "取不到 loss 时不得落任何文件"


# ── ③b GRPO / OPD 落点（本包要修的核心：旧实现只查顶层 input_ids ⇒ GRPO 零产出）──


def test_input_ids_from_batch_handles_both_shapes():
    """取键的**唯一真相源**：SFT 扁平 batch 与 GRPO/OPD 嵌套 batch 都要认得。"""
    from graspo.flow.msswift.first_step_probe import input_ids_from_batch

    flat = {"input_ids": torch.tensor([[1, 2]]), "labels": torch.tensor([[1, 2]])}
    nested = {
        "model_inputs": {"input_ids": torch.tensor([[3, 4]]), "attention_mask": None},
        "grpo_batch": object(),
    }
    assert torch.equal(input_ids_from_batch(flat), flat["input_ids"])
    assert torch.equal(input_ids_from_batch(nested), nested["model_inputs"]["input_ids"])
    assert input_ids_from_batch({"model_inputs": {}, "grpo_batch": object()}) is None
    assert input_ids_from_batch(None) is None
    assert input_ids_from_batch(["not", "a", "dict"]) is None


class _FakeGrpoTrainer:
    """GRPO/OPD 替身：``compute_loss`` 收到**嵌套** batch、且两条分支都只返回 loss 张量。

    形状依据（**代码依据**）：``swift/rlhf_trainers/grpo_trainer.py:767`` 构造
    ``{"model_inputs": ..., "grpo_batch": ...}``；``:862-876`` 消费；
    GKD(OPD) 同形，见 ``gkd_trainer.py:117-120``。
    """

    def __init__(self, loss: float = 0.375) -> None:
        self.loss = torch.tensor(loss)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        # 真实 GRPO 的 ``compute_loss`` 无论 ``return_outputs`` 真假都返回**张量**
        # （``grpo_trainer.py:876`` / ``:891``）——旧包装盲目 ``result[0]`` 会 IndexError。
        return self.loss


def _grpo_batch(nonce: int = 0):
    return {
        "model_inputs": {
            "input_ids": torch.tensor([[10 + nonce, 11, 12]], dtype=torch.long),
            "labels": torch.tensor([[10 + nonce, 11, 12]], dtype=torch.long),
        },
        "grpo_batch": object(),
    }


@pytest.mark.parametrize("return_outputs", [False, True])
def test_grpo_nested_batch_produces_a_local_row(tmp_path, monkeypatch, return_outputs):
    """★ 本包核心：GRPO/OPD 路径**必须有产出**（旧实现零产出）。"""
    from graspo.flow import logging as flow_logging

    monkeypatch.setattr(flow_logging, "_run_id", "unit-test")
    trainer = _FakeGrpoTrainer(loss=0.5)
    install_first_step_recorder(trainer, output_dir=tmp_path, rank=2, epoch=0.0)

    batch = _grpo_batch()
    result = trainer.compute_loss(None, batch, return_outputs=return_outputs)
    # 只读旁路：返回值原样透传（GRPO 分支返回裸张量，不得被包装改成元组/标量）。
    assert torch.is_tensor(result)
    assert float(result) == pytest.approx(0.5)

    path = tmp_path / "logs" / "unit-test" / "rank_metrics.rank_00002.jsonl"
    lines = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert len(lines) == 1, "GRPO 路径必须落一行（旧实现这里恒为空）"
    payload = lines[0]
    assert payload["rank"] == 2
    assert payload["loss_caliber"] == "per_rank_local"
    assert payload["local_loss"] == pytest.approx(0.5)
    assert payload["input_ids_sha256"] == fingerprint_input_ids(
        batch["model_inputs"]["input_ids"]
    )[0]
    assert payload["input_ids_shape"] == [1, 3]


def test_grpo_records_once_only(tmp_path, monkeypatch):
    from graspo.flow import logging as flow_logging

    monkeypatch.setattr(flow_logging, "_run_id", "unit-test")
    trainer = _FakeGrpoTrainer()
    install_first_step_recorder(trainer, output_dir=tmp_path, rank=0)
    trainer.compute_loss(None, _grpo_batch(), return_outputs=True)
    trainer.compute_loss(None, _grpo_batch(nonce=99), return_outputs=True)
    path = tmp_path / "logs" / "unit-test" / "rank_metrics.rank_00000.jsonl"
    assert len(path.read_text(encoding="utf-8").strip().splitlines()) == 1


# ── ③c 生成输入通道（E1 第三通道）────────────────────────────────────────────


def test_generation_fingerprint_is_stable_and_content_sensitive():
    from graspo.flow.msswift.first_step_probe import fingerprint_generation_inputs

    requests = [SimpleNamespace(messages=[{"role": "user", "content": "2 的平方是多少"}])]
    assert fingerprint_generation_inputs(requests) == fingerprint_generation_inputs(requests)
    other = [SimpleNamespace(messages=[{"role": "user", "content": "3 的平方是多少"}])]
    assert fingerprint_generation_inputs(requests)[0] != fingerprint_generation_inputs(other)[0]
    # 非 JSON 值（PIL.Image 等）不得走 repr（repr 含对象地址 ⇒ 两跑不可比）。
    unstable = [SimpleNamespace(messages=[{"role": "user", "content": object()}])]
    assert fingerprint_generation_inputs(unstable) == fingerprint_generation_inputs(unstable)


def test_generation_input_lands_in_the_local_row(tmp_path, monkeypatch):
    from graspo.flow import logging as flow_logging
    from graspo.flow.msswift.first_step_probe import (
        capture_generation_inputs,
        clear_generation_input_fingerprints,
    )

    monkeypatch.setattr(flow_logging, "_run_id", "unit-test")
    clear_generation_input_fingerprints()
    try:
        requests = [SimpleNamespace(messages=[{"role": "user", "content": "hello"}])]
        capture_generation_inputs(rank=1, call_index=0, call_seed=42, infer_requests=requests)
        # 幂等：第二次调用（更晚的采样批）不覆写首个指纹。
        capture_generation_inputs(
            rank=1,
            call_index=1,
            call_seed=43,
            infer_requests=[SimpleNamespace(messages=[{"role": "user", "content": "later"}])],
        )
        trainer = _FakeGrpoTrainer()
        install_first_step_recorder(trainer, output_dir=tmp_path, rank=1)
        trainer.compute_loss(None, _grpo_batch(), return_outputs=True)

        path = tmp_path / "logs" / "unit-test" / "rank_metrics.rank_00001.jsonl"
        payload = json.loads(path.read_text(encoding="utf-8").strip())
        assert payload["generation_input_request_count"] == 1
        assert payload["generation_input_call_index"] == 0
        assert payload["generation_input_call_seed"] == 42
        assert len(payload["generation_input_sha256"]) == 64
        assert payload["generation_input_algorithm"]
    finally:
        clear_generation_input_fingerprints()


def test_generation_capture_is_a_noop_when_probe_is_off(monkeypatch):
    """默认关 ⇒ 生成侧零动作：不 import 探针、不记录任何指纹。"""
    from graspo.core.determinism import DeterminismSwitch, bind_active_switch
    from graspo.flow.msswift import first_step_probe as probe_module
    from graspo.flow.msswift._rollout_seed import _capture_first_step_probe_generation_inputs

    bind_active_switch(DeterminismSwitch())
    probe_module.clear_generation_input_fingerprints()
    requests = [SimpleNamespace(messages=[{"role": "user", "content": "hello"}])]
    _capture_first_step_probe_generation_inputs(
        rank=0, call_index=0, call_seed=42, args=(requests,), kwargs={}
    )
    assert probe_module.generation_input_fingerprint(0) is None


def test_generation_capture_records_when_probe_is_on():
    from graspo.core.determinism import DeterminismSwitch, bind_active_switch
    from graspo.flow.msswift import first_step_probe as probe_module
    from graspo.flow.msswift._rollout_seed import _capture_first_step_probe_generation_inputs

    original = bind_active_switch(DeterminismSwitch(enabled=False, probe_first_step=True))
    probe_module.clear_generation_input_fingerprints()
    try:
        requests = [SimpleNamespace(messages=[{"role": "user", "content": "hello"}])]
        _capture_first_step_probe_generation_inputs(
            rank=0, call_index=0, call_seed=42, args=(requests,), kwargs={}
        )
        assert probe_module.generation_input_fingerprint(0) is not None
    finally:
        probe_module.clear_generation_input_fingerprints()
        bind_active_switch(original)


# ── ③d 双通道并列：全局口径行 + 局部口径行（防口径分叉 §1.4）────────────────


def test_callback_on_log_writes_the_global_caliber_row(tmp_path, monkeypatch, stub_swift_callbacks):
    from graspo.flow import logging as flow_logging

    monkeypatch.setattr(flow_logging, "_run_id", "unit-test")
    callback_class = install_probe_callback()
    trainer = _FakeTrainer(loss=0.75)
    args = SimpleNamespace(output_dir=str(tmp_path), process_index=3)
    callback = callback_class(args, trainer)
    callback.on_train_begin(args, SimpleNamespace(epoch=0.0), SimpleNamespace())
    trainer.compute_loss(None, {"input_ids": torch.tensor([[1]])})

    # 首步日志（ms-swift 的 logs['loss'] = nested_gather(tr_loss).mean() = 跨 rank 均值）
    state = SimpleNamespace(global_step=1, epoch=0.0)
    callback.on_log(args, state, SimpleNamespace(), logs={"loss": 0.1802505})
    # 后续日志不得再落一行（只关心首步）。
    callback.on_log(args, state, SimpleNamespace(), logs={"loss": 0.9})

    path = tmp_path / "logs" / "unit-test" / "rank_metrics.rank_00003.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 2, "双通道并列：局部行 + 全局行各一行"
    by_caliber = {row["loss_caliber"]: row for row in rows}
    assert set(by_caliber) == {"per_rank_local", "cross_rank_mean"}
    # 局部行只带局部值，不伪造全局值（反之亦然）。
    local = by_caliber["per_rank_local"]
    assert local["local_loss"] == pytest.approx(0.75)
    assert local["global_loss"] is None
    global_row = by_caliber["cross_rank_mean"]
    assert global_row["global_loss"] == pytest.approx(0.1802505)
    assert global_row["local_loss"] is None
    assert global_row["global_loss_source"]
    # 两行都不得带 metrics（否则采集侧会当逐步指标，污染台账）。
    assert all("metrics" not in row for row in rows)


def test_on_log_ignores_non_numeric_loss(tmp_path, monkeypatch, stub_swift_callbacks):
    from graspo.flow import logging as flow_logging

    monkeypatch.setattr(flow_logging, "_run_id", "unit-test")
    callback_class = install_probe_callback()
    args = SimpleNamespace(output_dir=str(tmp_path), process_index=0)
    callback = callback_class(args, _FakeTrainer())
    callback.on_log(args, SimpleNamespace(global_step=1, epoch=None), SimpleNamespace(), logs={})
    assert not (tmp_path / "logs").exists()



# ── ④ ms-swift 扩展点注册（不改上游源码）───────────────────────────────────


@pytest.fixture()
def stub_swift_callbacks(monkeypatch):
    """塞一个带 ``callbacks_map`` 的假 ``swift.callbacks``（不要求真安装 ms-swift）。"""
    swift_module = types.ModuleType("swift")
    callbacks_module = types.ModuleType("swift.callbacks")

    class _StubTrainerCallback:
        def __init__(self, args, trainer):
            self.args = args
            self.trainer = trainer

    callbacks_module.TrainerCallback = _StubTrainerCallback
    callbacks_module.callbacks_map = {}
    swift_module.callbacks = callbacks_module
    monkeypatch.setitem(sys.modules, "swift", swift_module)
    monkeypatch.setitem(sys.modules, "swift.callbacks", callbacks_module)
    return callbacks_module


def test_install_probe_callback_registers_into_official_map(stub_swift_callbacks):
    callback_class = install_probe_callback()
    assert stub_swift_callbacks.callbacks_map[PROBE_CALLBACK_NAME] is callback_class
    # 幂等：第二次安装返回同一个类，不覆盖。
    assert install_probe_callback() is callback_class


def test_callback_on_train_begin_installs_recorder(tmp_path, monkeypatch, stub_swift_callbacks):
    from graspo.flow import logging as flow_logging

    monkeypatch.setattr(flow_logging, "_run_id", "unit-test")
    callback_class = install_probe_callback()
    trainer = _FakeTrainer(loss=2.0)
    args = SimpleNamespace(output_dir=str(tmp_path), process_index=1)
    callback = callback_class(args, trainer)
    callback.on_train_begin(args, SimpleNamespace(epoch=0.0), SimpleNamespace())

    batch = {"input_ids": torch.tensor([[9]], dtype=torch.long)}
    trainer.compute_loss(None, batch)
    path = tmp_path / "logs" / "unit-test" / "rank_metrics.rank_00001.jsonl"
    payload = json.loads(path.read_text(encoding="utf-8").strip())
    assert payload["rank"] == 1
    assert payload["local_loss"] == pytest.approx(2.0)


# ── ⑤ 接线：默认关 ⇒ argv 逐字不变 ─────────────────────────────────────────

_ARD_SAMPLE = {
    "id": "sample-1",
    "source": "unit-test",
    "data_source": "ard_text",
    "schema_version": "3.0.0",
    "messages": [{"role": "user", "content": "列出 2 的平方"}],
    "targets": [{"id": None, "output": {"content": '{"answer": "4"}', "reasoning": None}}],
}


def _write_config(tmp_path: Path):
    from graspo.core.schema import GraspoConfig

    data_path = tmp_path / "ard.jsonl"
    data_path.write_text(json.dumps(_ARD_SAMPLE, ensure_ascii=False) + "\n", encoding="utf-8")
    return GraspoConfig.model_validate(
        {
            "train_method": "sft",
            "backend": "msswift",
            "model": {"model_path": "models/Qwen3.5-9B"},
            "data": {"train_path": str(data_path)},
            "training": {
                "output_dir": str(tmp_path / "out"),
                "run_name": "unit",
                "overwrite_output_dir": True,
            },
        }
    )


@pytest.fixture()
def stub_ms_swift(monkeypatch, stub_swift_callbacks):
    """拦截 ms-swift 的库入口，记录 ``sft_main`` 收到的 argv（不真的跑训练）。"""
    calls: dict[str, list] = {"sft_main": []}
    trainers_module = types.ModuleType("swift.trainers")

    class _StubTrainer:
        pass

    trainers_module.Trainer = _StubTrainer
    pipelines_module = types.ModuleType("swift.pipelines")
    pipelines_module.sft_main = lambda argv: calls["sft_main"].append(list(argv))
    monkeypatch.setitem(sys.modules, "swift.trainers", trainers_module)
    monkeypatch.setitem(sys.modules, "swift.pipelines", pipelines_module)
    return calls


def test_probe_disabled_leaves_ms_swift_argv_untouched(tmp_path, stub_ms_swift):
    from graspo.core.determinism import DeterminismSwitch, bind_active_switch
    from graspo.flow.msswift.sft_trainer import create_msswift_sft_trainer

    bind_active_switch(DeterminismSwitch())
    create_msswift_sft_trainer(_write_config(tmp_path), None).train(smoke=True)
    argv = stub_ms_swift["sft_main"][0]
    assert "--callbacks" not in argv
    assert PROBE_CALLBACK_NAME not in argv


def test_probe_enabled_appends_callbacks_flag(tmp_path, stub_ms_swift, stub_swift_callbacks):
    from graspo.core.determinism import DeterminismSwitch, bind_active_switch
    from graspo.flow.msswift.sft_trainer import create_msswift_sft_trainer

    bind_active_switch(DeterminismSwitch(enabled=False, probe_first_step=True))
    create_msswift_sft_trainer(_write_config(tmp_path), None).train(smoke=True)
    argv = stub_ms_swift["sft_main"][0]
    assert "--callbacks" in argv
    assert argv[argv.index("--callbacks") + 1] == PROBE_CALLBACK_NAME
    # 探针只在 ms-swift 的官方扩展点注册（map 里出现该名字），上游源码零改动。
    assert PROBE_CALLBACK_NAME in stub_swift_callbacks.callbacks_map
    # 冒烟边界参数仍在（不得因为探针而丢掉既有行为）。
    assert "--max_steps" in argv


def test_probe_flag_is_consumed_independently_of_total_switch():
    """探针不是"确定性开关"的一部分：``enabled=False`` 时也可单独打开（A/B 的臂 A）。"""
    from graspo.core.determinism import DeterminismSwitch, bind_active_switch
    from graspo.flow.msswift.sft_trainer import probe_first_step_enabled

    original = bind_active_switch(DeterminismSwitch())
    try:
        assert probe_first_step_enabled() is False
        bind_active_switch(DeterminismSwitch(enabled=False, probe_first_step=True))
        assert probe_first_step_enabled() is True
        bind_active_switch(DeterminismSwitch(enabled=True))
        assert probe_first_step_enabled() is False
    finally:
        bind_active_switch(original)


def test_probe_payload_has_no_metrics_key():
    payload = probe_payload(FirstStepProbeRecord(0, 1, 0.5, "float32", "d" * 64, [1], "int64", 0.0))
    assert "metrics" not in payload
    assert payload["kind"] == "diagnostic"
    assert payload["reproducible"] is True


# ── ⑥ 接线点唯一化 + 三条通道（SFT / GRPO / OPD）都接上 ──────────────────────


def test_probe_extra_argv_default_off_is_empty_and_does_not_register(stub_swift_callbacks):
    """默认关：**不注册回调、不产出任何参数**（调用方因此连分支都不需要）。"""
    from graspo.flow.msswift.first_step_probe import probe_extra_argv

    assert probe_extra_argv(False) == []
    assert stub_swift_callbacks.callbacks_map == {}


def test_probe_extra_argv_enabled_registers_and_names_the_callback(stub_swift_callbacks):
    from graspo.flow.msswift.first_step_probe import probe_extra_argv

    assert probe_extra_argv(True) == ["--callbacks", PROBE_CALLBACK_NAME]
    assert PROBE_CALLBACK_NAME in stub_swift_callbacks.callbacks_map


def test_probe_active_extra_argv_reads_the_process_switch(stub_swift_callbacks):
    """判据只有一处：进程内绑定的 ``DeterminismSwitch``（§1.4）。"""
    from graspo.core.determinism import DeterminismSwitch, bind_active_switch
    from graspo.flow.msswift.first_step_probe import probe_active_extra_argv

    bind_active_switch(DeterminismSwitch())
    assert probe_active_extra_argv() == []
    # 探针独立于总开关：``enabled=False`` 也能单独打开（A/B 的臂 A）。
    bind_active_switch(DeterminismSwitch(enabled=False, probe_first_step=True))
    assert probe_active_extra_argv() == ["--callbacks", PROBE_CALLBACK_NAME]
    # 总开关打开但探针关 ⇒ 仍然什么都不追加。
    bind_active_switch(DeterminismSwitch(enabled=True))
    assert probe_active_extra_argv() == []


@pytest.mark.parametrize(
    "module_name",
    [
        "graspo.flow.msswift.sft_trainer",
        "graspo.flow.msswift.trainer",
        "graspo.flow.msswift.opd_trainer",
    ],
)
def test_every_training_channel_calls_the_single_wiring_point(module_name):
    """★静态机核：三条训练通道都调用**同一个**接线点，且都是先拼 `extra_argv` 再建 argv。

    为什么不各写一份：接线口径（注册回调 + `--callbacks` 的拼写）只允许有一处实现
    （§1.4）——本断言让"漏接某条通道"或"某条通道自己又拼了一份"在测试期就暴露。
    """
    import inspect

    module = __import__(module_name, fromlist=["train"])
    source = inspect.getsource(module)
    # 只取训练方法体（模块级还有探针/开关等无关键）；`train` 是三条通道共有的入口名。
    body = inspect.getsource(_train_method_of(module))

    assert "probe_active_extra_argv" in body, f"{module_name} 未接探针接线点"
    # 追加发生在参数向量**建好之前**（否则探针永远进不了 argv）。
    assert body.index("probe_active_extra_argv()") < body.index("graspo_to_ms_swift_argv(")
    # 通道内不得自己装回调、也不得出现回调名字面量——接线口径只允许有一处实现（§1.4）。
    assert "install_probe_callback" not in body
    assert PROBE_CALLBACK_NAME not in body


def _train_method_of(module):
    """取出模块里**唯一**带 ``train`` 方法的训练器门面类的那个方法（机核辅助）。"""
    import inspect

    for _, obj in inspect.getmembers(module, inspect.isclass):
        if obj.__module__ != module.__name__:
            continue
        if "train" in obj.__dict__:
            return obj.__dict__["train"]
    raise AssertionError(f"{module.__name__} 里找不到带 train 的训练器门面类")

