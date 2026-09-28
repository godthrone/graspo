"""``graspo.eval.orchestrator`` 的单测：链路编排的目标解析与产物契约。

编排里真正需要 GPU/网络的部分（发请求、合并权重）本轮不上机，因此这里测的是
**判定与产物**这两块：要服务哪个目录、报告里必须有哪些字段。

★ **2026-09-28（挂死修复）**：本模块的用例**默认不触任何真 vLLM 服务**——真服务
相关路径由 :func:`_block_real_vllm_service` 这个 autouse fixture **默认 mock 掉**。
起因是一次实测挂死：``test_run_evaluation_refuses_production_cards_before_any_request``
原先靠"6/7 = 保留卡 ⇒ 守卫在发请求前就拒绝"通过；而保留集合改为**可配置**（默认
``()``）后 ``gpus="6,7"`` 不再被拒，于是它一路走到 ``wait_until_ready`` 去轮询
``http://127.0.0.1:1``，单次探测 120s × 3 重试 × 60 轮 ⇒ 整套测试在 40% 处挂死。
两条修法同时落地：①该用例显式注入"6/7 是保留卡"的部署事实（见
``_inject_deployment_fact``）；②``VllmEvalClient.wait_until_ready`` 加**总预算**
（见 ``src/graspo/eval/vllm_client.py``），并在此处默认 mock，使"意外走到网络"也只会
得到确定性的快速失败。真服务用例需**显式**覆盖这两条，禁止默默依赖真服务。
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
from graspo.eval.vllm_client import VllmEvalClient, VllmEvalError


@pytest.fixture(autouse=True)
def _inject_deployment_fact(monkeypatch):
    """显式声明的部署事实（不再依赖写死元组，见 ``core.gpu_guard`` 模块头）：
    允许 0–5、**6/7 为保留（生产）卡**——后者正是本模块两条守卫用例的前提。"""
    monkeypatch.setenv("GRASPO_ALLOWED_GPU_INDICES", "0,1,2,3,4,5")
    monkeypatch.setenv("GRASPO_RESERVED_GPU_INDICES", "6,7")


@pytest.fixture(autouse=True)
def _block_real_vllm_service(monkeypatch):
    """默认 mock：任何真发请求的路径都快速失败，**绝不挂死**（见模块 docstring）。"""

    def _refuse(self, *args, **kwargs):  # noqa: ANN001, ANN002, ANN003
        raise VllmEvalError(
            "mocked: 本模块的单测不触真 vLLM 服务（真服务用例须显式覆盖本 fixture）"
        )

    monkeypatch.setattr(VllmEvalClient, "wait_until_ready", _refuse)
    monkeypatch.setattr(VllmEvalClient, "evaluate_samples", _refuse)


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


# ── 产物落点必须显式、绝对、且落盘后必须真的在预期位置 ────────────────────────
#
# 历史事故（task-r3-effect §⑦-3）：`run_evaluation` 用 cwd 相对路径当输出路径，
# 容器里 cwd 与挂载点不一致 ⇒ `eval_report.json` 写进了意料之外的目录，未挂载时
# **静默丢失**（数字拿到了、产物没了、没有任何报错）。以下三个用例把这条堵死。


def _minimal_report(tmp_path: Path):
    from graspo.eval.schema import (
        EvalCriteria,
        EvalDataset,
        EvalDecoding,
        EvalEnvironmentPointer,
        EvalReport,
        EvalSummary,
        utc_now_iso,
    )

    return EvalReport(
        run_id="base-20260918-000000",
        started_at=utc_now_iso(),
        finished_at=utc_now_iso(),
        elapsed_sec=1.0,
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
            visible_gpus=[0], gpu_count=1, fingerprint_path="/tmp/env.json"
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


def test_run_evaluation_rejects_relative_output_dir_before_any_write(tmp_path, monkeypatch):
    """相对 output_dir ⇒ 在任何落盘之前就拒绝（不依赖 cwd 的落点）。

    这条堵的正是"产物静默丢失"：不拒绝的话，落点会随进程 cwd 漂移，
    调用方还以为产物在配置写的位置。
    """
    monkeypatch.chdir(tmp_path)
    dataset_path = _dataset_root(tmp_path)
    dataset, _, _ = load_eval_dataset(dataset_path)
    with pytest.raises(OrchestrationError, match="is not absolute"):
        run_evaluation(
            base_url="http://127.0.0.1:1",
            model=EvalModel(served_name="s", role="base", path="/models/base"),
            dataset=dataset,
            output_dir=".local/eval/runs/relative",
            gpus="0",
        )
    # 拒绝发生在 create 之前 ⇒ 相对 cwd 的位置不该被建出来
    assert not (tmp_path / ".local").exists()


def test_write_report_rejects_relative_output_dir(tmp_path, monkeypatch):
    """`write_report` 同样 fail-closed —— 不只入口检查，产物写手也要守。"""
    monkeypatch.chdir(tmp_path)
    with pytest.raises(OrchestrationError, match="is not absolute"):
        write_report(_minimal_report(tmp_path), ".local/eval/runs/relative")


def test_write_report_lands_at_the_absolute_path_it_returns(tmp_path, monkeypatch):
    """绝对路径 ⇒ 产物真的落在返回路径上，且内容非空（落点断言的反向验证）。"""
    workdir = tmp_path / "elsewhere"
    workdir.mkdir()
    monkeypatch.chdir(workdir)  # cwd 故意与产物目录无关
    out = tmp_path / ".local" / "eval" / "runs" / "abs"
    path = write_report(_minimal_report(tmp_path), out)
    assert path == out / EVAL_REPORT_FILENAME
    assert path.is_file() and path.stat().st_size > 0
    # cwd 下不该冒出任何产物
    assert not (workdir / ".local").exists()


def test_assert_file_persisted_rejects_missing_and_empty(tmp_path):
    """落点断言本身必须能真失败：不存在 / 空文件都要报错，不得静默。"""
    from graspo.eval.orchestrator import _assert_file_persisted

    with pytest.raises(OrchestrationError, match="not present"):
        _assert_file_persisted(tmp_path / "nope.json", what="eval_report.json")
    empty = tmp_path / "empty.json"
    empty.write_text("", encoding="utf-8")
    with pytest.raises(OrchestrationError, match="is empty"):
        _assert_file_persisted(empty, what="eval_report.json")
