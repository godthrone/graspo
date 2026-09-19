"""graspo.eval — 效果评测链路（ELAM V5 数据集 + vLLM，温度锁 0）。

层边界（宪法 §1.1 / §1.3）：本包是**评测域**。纯计算部分（口径判定、聚合、Δ 配对、
checkpoint 形态识别）零设施依赖、可在 CPU 上单测；设施部分（vLLM HTTP 调用、
权重重合并、锁卡采样）集中在各自的模块里。

- `criteria.py`     — 准确率口径的单一真相源（all_right = 工具名 AND 动作方向）
- `schema.py`       — 评测产物契约（pydantic）
- `dataset.py`      — ELAM V5 读取 + 图像重叠检测
- `guard.py`        — GPU 锁卡防呆 + 只采可见卡的显存采样
- `vllm_client.py`  — vLLM Chat Completions 客户端（温度锁 0）
- `evaluate.py`     — 聚合 + Δ 配对 + 阈值判定
- `merged_export.py`— checkpoint → merged-hf（补齐 ms-swift PEFT 缺口）
- `orchestrator.py` — 链路编排与产物落盘

公开 API 见 ``__all__``；外部使用者不需要关心内部文件划分（§8.6）。
"""

from graspo.eval.criteria import (
    ACCURACY_CRITERIA_SOURCE,
    ACCURACY_CRITERIA_VERSION,
    GroundTruth,
    Prediction,
    accuracy,
    extract_ground_truth,
    extract_prediction,
    is_all_right,
)
from graspo.eval.dataset import (
    Dataset,
    DatasetError,
    EvalSample,
    OverlapStats,
    attach_train_overlap,
    file_sha256,
    load_dataset,
    overlap_sample_indices,
    overlap_stats,
)
from graspo.eval.evaluate import (
    DeltaError,
    DeltaResult,
    compute_delta,
    load_report,
    summarize,
)
from graspo.eval.guard import (
    ALLOWED_GPU_INDICES,
    MAX_GPU_COUNT,
    RESERVED_GPU_INDICES,
    GpuGuardError,
    GpuPlan,
    assert_devices_free,
    parse_gpu_plan,
    repo_relative,
    resolve_gpu_plan,
    sample_device_memory_mib,
    sample_driver_version,
)
from graspo.eval.merged_export import (
    CheckpointClassification,
    CheckpointKind,
    ExportError,
    classify_checkpoint,
    existing_merged_output,
    merge_peft_checkpoint,
    normalize_language_model_targets,
    prepare_output_directory,
    resolve_peft_adapter_dir,
)
from graspo.eval.orchestrator import (
    EVAL_REPORT_FILENAME,
    OrchestrationError,
    load_eval_dataset,
    resolve_eval_target,
    run_evaluation,
    write_report,
)
from graspo.eval.schema import (
    EVAL_ARTIFACT_SCHEMA_VERSION,
    EvalCriteria,
    EvalDataset,
    EvalDecoding,
    EvalModel,
    EvalReport,
    EvalSummary,
    EvalTrainSubset,
)
from graspo.eval.vllm_client import (
    EVAL_MAX_TOKENS,
    EVAL_TEMPERATURE,
    EVAL_TOP_P,
    ClientConfig,
    SampleRecord,
    TemperatureLockError,
    VllmEvalClient,
    VllmEvalError,
    assert_request_body_locked,
    build_messages,
    build_request_body,
)

__all__ = [
    "ACCURACY_CRITERIA_SOURCE",
    "ACCURACY_CRITERIA_VERSION",
    "ALLOWED_GPU_INDICES",
    "EVAL_ARTIFACT_SCHEMA_VERSION",
    "EVAL_MAX_TOKENS",
    "EVAL_REPORT_FILENAME",
    "EVAL_TEMPERATURE",
    "EVAL_TOP_P",
    "MAX_GPU_COUNT",
    "CheckpointClassification",
    "CheckpointKind",
    "ClientConfig",
    "Dataset",
    "DatasetError",
    "DeltaError",
    "DeltaResult",
    "EvalCriteria",
    "EvalDataset",
    "EvalDecoding",
    "EvalModel",
    "EvalReport",
    "EvalSample",
    "EvalSummary",
    "EvalTrainSubset",
    "ExportError",
    "GpuGuardError",
    "GpuPlan",
    "GroundTruth",
    "OrchestrationError",
    "OverlapStats",
    "RESERVED_GPU_INDICES",
    "Prediction",
    "SampleRecord",
    "TemperatureLockError",
    "VllmEvalClient",
    "VllmEvalError",
    "accuracy",
    "assert_devices_free",
    "assert_request_body_locked",
    "attach_train_overlap",
    "build_messages",
    "build_request_body",
    "classify_checkpoint",
    "compute_delta",
    "existing_merged_output",
    "extract_ground_truth",
    "extract_prediction",
    "file_sha256",
    "is_all_right",
    "load_dataset",
    "load_eval_dataset",
    "load_report",
    "merge_peft_checkpoint",
    "normalize_language_model_targets",
    "overlap_sample_indices",
    "overlap_stats",
    "parse_gpu_plan",
    "prepare_output_directory",
    "repo_relative",
    "resolve_eval_target",
    "resolve_gpu_plan",
    "resolve_peft_adapter_dir",
    "run_evaluation",
    "sample_device_memory_mib",
    "sample_driver_version",
    "summarize",
    "write_report",
]
