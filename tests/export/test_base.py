"""模板方法契约（T13/T14）：公开流程不可被覆盖；钩子调用顺序被锁定。"""

from __future__ import annotations

from pathlib import Path

import pytest
from helpers import make_export_config

from graspo.export.base import Exporter
from graspo.export.schema import (
    CheckpointClassification,
    CheckpointForm,
    ExportTarget,
    ShardPlan,
)


def test_subclass_cannot_override_the_public_export_method():
    """T13：覆盖公开模板方法 ⇒ 类定义当场报错（不是运行期才发现流程被改写）。"""
    with pytest.raises(TypeError, match="must not override"):

        class Bad(Exporter):  # noqa: D401 - intentional violation
            form = CheckpointForm.NATIVE_LORA_RANK
            supported_targets = frozenset({ExportTarget.PEFT_ADAPTER})

            def export(self, config, classification, *, base_model_path=None):  # type: ignore[override]
                return None


class _RecordingExporter(Exporter):
    form = CheckpointForm.HF_BASE_MODEL
    supported_targets = frozenset({ExportTarget.MERGED_HF})

    def __init__(self, calls: list[str]) -> None:
        self.calls = calls

    @classmethod
    def _matches(cls, root: Path):
        return True, ("stub",)

    def _plan(self, config, classification, *, base_model_path=None):
        self.calls.append("plan")
        return ShardPlan(
            form=self.form,
            target=ExportTarget.MERGED_HF,
            checkpoint_name="stub",
            source_dir=str(classification.path),
            base_model_path=base_model_path or "models/base",
        )

    def _assert_plan_is_sound(self, plan):
        self.calls.append("assert_plan")
        super()._assert_plan_is_sound(plan)

    def _load(self, plan):
        self.calls.append("load")
        from graspo.export.base import LoadedShards

        return LoadedShards(payloads=[], shard_refs=[], paths=[])

    def _assert_shards_are_sound(self, plan, loaded):
        self.calls.append("assert_shards")
        return super()._assert_shards_are_sound(plan, loaded)

    def _assemble(self, loaded, plan):
        self.calls.append("assemble")
        return {"tensor": 1}

    def _prepare_destination(self, plan, config, classification):
        self.calls.append("prepare_destination")
        return super()._prepare_destination(plan, config, classification)

    def _write(self, destination, assembled, plan):
        self.calls.append("write")
        (destination / "config.json").write_text("{}", encoding="utf-8")
        return ("config.json",)

    def _verify_produced_state(self, destination, plan, assembled, written):
        self.calls.append("verify")
        return len(assembled)


def test_template_method_call_order_is_locked(tmp_path):
    calls: list[str] = []
    exporter = _RecordingExporter(calls)
    checkpoint = tmp_path / "ckpt"
    checkpoint.mkdir()
    classification = CheckpointClassification(
        form=CheckpointForm.HF_BASE_MODEL, path=str(checkpoint), evidence=("stub",)
    )
    config = make_export_config(checkpoint, tmp_path / "out2", fmt="merged-hf")
    exporter.export(config, classification, base_model_path="models/base")

    assert calls == [
        "plan",
        "assert_plan",
        "load",
        "assert_shards",
        "assemble",
        "prepare_destination",
        "write",
        "verify",
    ]
