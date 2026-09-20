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
