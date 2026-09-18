"""``graspo.eval.orchestrator`` 的单测：链路编排的目标解析与产物契约。

编排里真正需要 GPU/网络的部分（发请求、合并权重）本轮不上机，因此这里测的是
**判定与产物**这两块：要服务哪个目录、报告里必须有哪些字段。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from graspo.eval.criteria import ACCURACY_CRITERIA_VERSION
from graspo.eval.guard import GpuGuardError
from graspo.eval.merged_export import CheckpointKind
from graspo.eval.orchestrator import (
    EVAL_REPORT_FILENAME,
    OrchestrationError,
    load_eval_dataset,
    resolve_eval_target,
    run_evaluation,
    write_report,
)
from graspo.eval.schema import EvalModel, EvalSummary


def _write(path: Path, payload: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(payload, encoding="utf-8")


def _base_model(root: Path) -> Path:
    _write(root / "config.json", "{}")
    (root / "model.safetensors").write_bytes(b"\x00")
    return root


def _peft_adapter(root: Path) -> Path:
    _write(root / "adapter_config.json", json.dumps({"base_model_name_or_path": "/models/base"}))
    (root / "adapter_model.safetensors").write_bytes(b"\x00")
    return root


def _dataset_root(tmp_path: Path) -> Path:
    image = tmp_path / "images" / "a.jpg"
    image.parent.mkdir(parents=True, exist_ok=True)
    image.write_bytes(b"\xff\xd8\xff")
    sample = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": "../images/a.jpg"},
                    {"type": "text", "text": "go"},
                ],
            }
        ],
        "targets": [
            {"output": {"tool_calls": [{"name": "rotate_arm", "arguments": {"action_type": "l"}}]}}
        ],
    }
    _write(tmp_path / "data" / "test.jsonl", json.dumps(sample, ensure_ascii=False) + "\n")
    _write(tmp_path / "data" / "train.jsonl", json.dumps(sample, ensure_ascii=False) + "\n")
    return tmp_path / "data" / "test.jsonl"


# ── resolve_eval_target ─────────────────────────────────────────────────────


def test_target_for_base_role_needs_no_checkpoint(tmp_path):
    model, ready = resolve_eval_target(
        role="base",
        served_name="base-srv",
        checkpoint_path=None,
        base_model_path="/models/base",
        export_format=None,
        merged_output_dir=None,
    )
    assert model.role == "base"
    assert ready == "/models/base"
    assert model.checkpoint_path is None


def test_non_base_role_without_checkpoint_is_rejected():
    with pytest.raises(OrchestrationError, match="requires a checkpoint_path"):
        resolve_eval_target(
            role="after",
            served_name="s",
            checkpoint_path=None,
            base_model_path="/models/base",
            export_format=None,
            merged_output_dir=None,
        )


def test_peft_checkpoint_resolves_to_merged_output_dir(tmp_path):
    adapter = _peft_adapter(tmp_path / "run" / "checkpoint-500")
    model, ready = resolve_eval_target(
        role="sft",
        served_name="sft-srv",
        checkpoint_path=str(tmp_path / "run"),
        base_model_path="/models/base",
        export_format=None,
        merged_output_dir=str(tmp_path / "merged"),
    )
    assert model.export_format == CheckpointKind.PEFT_ADAPTER.value
    # 需要合并 → 服务的目录是 merged 产物目录；返回的"就绪目录"是 adapter 本身
    assert model.path == str(tmp_path / "merged")
    assert ready == str(adapter.resolve())


def test_peft_checkpoint_without_merged_dir_is_rejected(tmp_path):
    _peft_adapter(tmp_path / "adapter")
    with pytest.raises(OrchestrationError, match="needs merging"):
        resolve_eval_target(
            role="after",
            served_name="s",
            checkpoint_path=str(tmp_path / "adapter"),
            base_model_path="/models/base",
            export_format=None,
            merged_output_dir=None,
        )


def test_export_format_contradicting_detected_kind_is_rejected(tmp_path):
    _peft_adapter(tmp_path / "adapter")
    with pytest.raises(OrchestrationError, match="contradicts the detected checkpoint kind"):
        resolve_eval_target(
            role="after",
            served_name="s",
            checkpoint_path=str(tmp_path / "adapter"),
            base_model_path="/models/base",
            export_format="graspo-native",
            merged_output_dir=str(tmp_path / "merged"),
        )


def test_native_checkpoint_is_delegated(tmp_path):
    root = tmp_path / "native"
    _write(root / "metadata.json", "{}")
    (root / "shard_0.safetensors").write_bytes(b"\x00")
    model, ready = resolve_eval_target(
        role="after",
        served_name="s",
        checkpoint_path=str(root),
        base_model_path="/models/base",
        export_format=None,
        merged_output_dir=str(tmp_path / "merged"),
    )
    assert model.export_format == CheckpointKind.GRASPO_NATIVE.value
    assert ready == str(root)


def test_checkpoint_that_is_already_a_base_model_is_served_directly(tmp_path):
    base = _base_model(tmp_path / "already-base")
    model, ready = resolve_eval_target(
        role="after",
        served_name="s",
        checkpoint_path=str(base),
        base_model_path=str(base),
        export_format=None,
        merged_output_dir=None,
    )
    assert model.export_format == CheckpointKind.BASE_MODEL.value
    assert ready == str(base)


# ── load_eval_dataset ───────────────────────────────────────────────────────


def test_load_eval_dataset_without_train_path_skips_overlap(tmp_path):
    dataset_path = _dataset_root(tmp_path)
    dataset, overlap, train_subset = load_eval_dataset(dataset_path)
    assert len(dataset.samples) == 1
    assert overlap is None  # 未计算 ≠ 算过且为零（§2.2）
    assert train_subset is None  # 没分析 ⇒ 不记训练切片标识


def test_load_eval_dataset_with_train_path_computes_overlap(tmp_path):
    dataset_path = _dataset_root(tmp_path)
    dataset, overlap, train_subset = load_eval_dataset(
        dataset_path, train_path=tmp_path / "data" / "train.jsonl"
    )
    assert overlap == {0}  # 测试样本的图像集是训练样本的子集
    assert train_subset is not None
    assert train_subset.sample_count == 1
    assert len(train_subset.sha256) == 64
    assert train_subset.path.endswith("train.jsonl")


def test_load_eval_dataset_records_train_slice_length(tmp_path):
    """训练切片长度必须记进产物——不同切片给出不同重叠集。"""
    dataset_path = _dataset_root(tmp_path)
    _, _, full = load_eval_dataset(dataset_path, train_path=tmp_path / "data" / "train.jsonl")
    _, _, sliced = load_eval_dataset(
        dataset_path, train_path=tmp_path / "data" / "train.jsonl", train_limit=1
    )
    assert full is not None and sliced is not None
    assert full.sample_count == 1 and sliced.sample_count == 1
    # sha256 是整份文件的哈希，所以切片与全量在这里相同（本 fixture 只有 1 行）；
    # 关键是 sample_count 被显式记录，供解释"为什么两次口径不同"。
    assert full.sha256 == sliced.sha256


def test_load_eval_dataset_rejects_empty_train_slice(tmp_path):
    from graspo.eval.dataset import DatasetError

    dataset_path = _dataset_root(tmp_path)
    empty = tmp_path / "data" / "train.jsonl"
    empty.write_text("", encoding="utf-8")
    with pytest.raises(DatasetError):
        load_eval_dataset(dataset_path, train_path=empty)


# ── run_evaluation 的防呆与产物 ─────────────────────────────────────────────


def test_run_evaluation_refuses_without_explicit_gpus(tmp_path):
    """锁卡防呆在编排入口就生效——连产物目录都不该被创建。"""
    dataset_path = _dataset_root(tmp_path)
    dataset, _, _ = load_eval_dataset(dataset_path)
    with pytest.raises(GpuGuardError):
        run_evaluation(
            base_url="http://127.0.0.1:1",
            model=EvalModel(served_name="s", role="base", path="/models/base"),
            dataset=dataset,
            output_dir=tmp_path / "out",
            gpus=None,
        )
    assert not (tmp_path / "out").exists()


def test_run_evaluation_refuses_production_cards_before_any_request(tmp_path):
    dataset_path = _dataset_root(tmp_path)
    dataset, _, _ = load_eval_dataset(dataset_path)
    with pytest.raises(GpuGuardError):
        run_evaluation(
            base_url="http://127.0.0.1:1",
            model=EvalModel(served_name="s", role="base", path="/models/base"),
            dataset=dataset,
            output_dir=tmp_path / "out",
            gpus="6,7",
        )
    assert not (tmp_path / "out").exists()


def test_run_evaluation_refuses_overlap_without_train_subset_metadata(tmp_path):
    """做了重叠分析却不给训练切片标识 → 拒绝：否则剔除口径的数值无法解释。"""
    dataset_path = _dataset_root(tmp_path)
    dataset, overlap, _ = load_eval_dataset(
        dataset_path, train_path=tmp_path / "data" / "train.jsonl"
    )
    assert overlap is not None
    with pytest.raises(OrchestrationError, match="no train_subset metadata"):
        run_evaluation(
            base_url="http://127.0.0.1:1",
            model=EvalModel(served_name="s", role="base", path="/models/base"),
            dataset=dataset,
            output_dir=tmp_path / "out",
            gpus="0,1",
            overlap_indices=overlap,
            train_subset=None,
        )


def test_write_report_emits_contract_fields(tmp_path):
    from graspo.eval.schema import (
        EvalCriteria,
        EvalDataset,
        EvalDecoding,
        EvalEnvironmentPointer,
        EvalReport,
        utc_now_iso,
    )

    report = EvalReport(
        run_id="base-20260918-000000",
        started_at=utc_now_iso(),
        finished_at=utc_now_iso(),
        elapsed_sec=1.5,
        model=EvalModel(served_name="s", role="base", path="/models/base"),
        dataset=EvalDataset(
            path="data/test.jsonl",
            split="test",
            sha256="a" * 64,
            sample_count_total=1,
            sample_count_requested=1,
        ),
        decoding=EvalDecoding(temperature=0.0, top_p=0.9, max_tokens=128, enable_thinking=False),
        criteria=EvalCriteria(version=ACCURACY_CRITERIA_VERSION, source="src"),
        environment=EvalEnvironmentPointer(
            visible_gpus=[0, 1], gpu_count=2, fingerprint_path="/tmp/env.json"
        ),
        summary=EvalSummary(
            sample_count_total=1,
            sample_count_valid=1,
            sample_count_error=0,
            correct=1,
            incorrect=0,
            accuracy=1.0,
            accuracy_percent=100.0,
        ),
        samples=[{"sample_index": 0, "all_right": True}],
    )
    path = write_report(report, tmp_path)
    assert path.name == EVAL_REPORT_FILENAME
    payload = json.loads(path.read_text(encoding="utf-8"))

    # 独立复现所需的最小充分信息，逐项断言（产物契约硬要求）
    for key in (
        "schema_version",
        "run_id",
        "started_at",
        "finished_at",
        "elapsed_sec",
        "model",
        "dataset",
        "decoding",
        "criteria",
        "environment",
        "summary",
        "samples",
    ):
        assert key in payload, f"report is missing {key}"
    assert payload["decoding"]["temperature"] == 0.0
    assert payload["criteria"]["version"] == ACCURACY_CRITERIA_VERSION
    assert payload["dataset"]["sha256"] == "a" * 64
    assert payload["environment"]["visible_gpus"] == [0, 1]
    assert payload["samples"]  # 逐样本明细 → 聚合结论可被独立重算
    # 重叠分析溯源：没做时必须可判别，而不是靠数值猜
    assert payload["summary"]["overlap_analysis_performed"] is False
    assert payload["train_subset"] is None
