"""评测链路编排：checkpoint → merged-hf → vLLM 推理 → 聚合 → 结构化产物。

**职责**：把一条链路的各环节串起来并落盘产物。它是**唯一的编排点**——
每个环节本身在各自的模块里（`merged_export` / `vllm_client` / `evaluate` /
`guard`），这里只负责顺序、前置校验、以及在正确的位置写正确的东西。

**本文件不负责**：
- 起/停 vLLM 容器（`scripts/eval_serve_vllm.sh`，容器生命周期是运维边界）
- 判定对错（`criteria.py`）、合并权重（`merged_export.py`）
- 删除任何东西（`scripts/eval_cleanup.sh` 只打印命令，交人工执行）

**链路（每一步的产物都是下一步的输入，失败即停）**

```
checkpoint              ──classify──▶ base / native / peft
   │
   ├─ base      ─────────────────────────────┐
   ├─ native    ──graspo export──▶ merged-hf ─┤
   └─ peft      ──merge_peft───▶ merged-hf ──┤
                                              ▼
                    vLLM 服务（显式锁卡，温度锁 0）
                                              │
                                         推理 test 集
                                              ▼
                              eval_report.json（自包含产物）
```

**为什么产物必须自包含**：评测结论要支撑"达标/不达标"的发布决策（宪法 §6 环境
可复现 + 工作包"产物契约"要求）。只留一个准确率数字，事后无法回答"分母是多少、
口径是哪版、温度多少、哪台机器、哪些样本错了"。因此 `eval_report.json` 里
把逐样本明细与全部上下文一并写入。
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from graspo.eval.criteria import (
    ACCURACY_CRITERIA_SOURCE,
    ACCURACY_CRITERIA_VERSION,
)
from graspo.eval.dataset import (
    Dataset,
    DatasetError,
    attach_train_overlap,
    load_dataset,
    overlap_sample_indices,
)
from graspo.eval.error import OrchestrationError
from graspo.eval.evaluate import summarize
from graspo.eval.guard import GpuPlan, resolve_gpu_plan
from graspo.eval.merged_export import (
    CheckpointClassification,
    CheckpointKind,
    ExportError,
    classify_checkpoint,
    resolve_peft_adapter_dir,
)
from graspo.eval.schema import (
    EvalCriteria,
    EvalDataset,
    EvalDecoding,
    EvalEnvironmentPointer,
    EvalModel,
    EvalReport,
    EvalRole,
    EvalTrainSubset,
    make_run_id,
    utc_now_iso,
)
from graspo.eval.vllm_client import (
    EVAL_MAX_TOKENS,
    EVAL_TEMPERATURE,
    EVAL_TOP_P,
    ClientConfig,
    VllmEvalClient,
    VllmEvalError,
)

#: 评测产物文件名（链路各环节统一用这一个名字）。
EVAL_REPORT_FILENAME = "eval_report.json"

#: 环境指纹文件名（与报告同目录）。
ENVIRONMENT_FILENAME = "environment.json"


def load_eval_dataset(
    dataset_path: str | Path,
    *,
    train_path: str | Path | None = None,
    train_limit: int | None = None,
    dataset_path_for_report: str | None = None,
    train_path_for_report: str | None = None,
) -> tuple[Dataset, set[int] | None, EvalTrainSubset | None]:
    """加载评测数据集；给了训练集就同时算出"图像重叠"样本下标。

    Args:
        dataset_path: 评测集 JSONL（如 v5 的 ``test.jsonl``）。
        train_path: 训练集 JSONL，仅用于重叠分析；``None`` 表示不分析。
        train_limit: 训练集只取前 N 条（54 档台账里的"train 前 100/20 条"）。
            ``None`` = 全量。**切片长度会被记进产物**——不同的切片会给出不同的
            重叠集，不记下来事后无法解释两次评测的口径差异。
        dataset_path_for_report: 产物里记录的数据集路径（脱敏后用）。
        train_path_for_report: 产物里记录的训练集路径（脱敏后用）。

    Returns:
        ``(dataset, overlap_indices, train_subset)``：
        - 未提供 ``train_path`` ⇒ ``overlap_indices=None``、
          ``train_subset=None``，明确表示"**未做**分析"，而不是伪装成
          "算过、重叠为零"（宪法 §2.2 None 是唯一空值语义）；
        - 提供了 ``train_path`` ⇒ ``overlap_indices`` 是**集合**（可能为空集，
          表示"做了且重叠为零"），``train_subset`` 记录切片标识与 sha256。

    Raises:
        DatasetError: 训练集为空、或 ``train_limit`` 非法。
    """
    dataset = load_dataset(dataset_path)
    if train_path is None:
        return dataset, None, None

    train_dataset = load_dataset(train_path)
    if train_limit is not None and train_limit > 0 and train_limit < len(train_dataset.samples):
        train_dataset.samples = train_dataset.samples[:train_limit]
    if not train_dataset.samples:
        raise DatasetError(f"train dataset {train_path} has no samples to analyse overlap with")

    attach_train_overlap(dataset, train_dataset)
    subset = EvalTrainSubset(
        path=train_path_for_report or str(train_dataset.path),
        # 注意：sha256 是**整份训练文件**的内容哈希（切片是文件的前缀，因此它同样
        # 锁定了切片内容）。切片长度单独记，两者合起来唯一确定本次用了哪些样本。
        sha256=train_dataset.sha256,
        sample_count=len(train_dataset.samples),
    )
    return dataset, overlap_sample_indices(dataset), subset


def _classify_or_find_adapter(
    checkpoint_path: str | Path,
    *,
    adapter_dir: str | None,
) -> CheckpointClassification:
    """识别 checkpoint 形态；ms-swift 的嵌套输出目录会被向下解析到 adapter。

    用户拿到的往往不是 ``checkpoint-500`` 本身，而是 ms-swift 的输出**根目录**
    （``<output_dir>/v3-20260901-101010/checkpoint-500``）。直接对根目录做形态识别
    会失败，于是这里多一步：识别不动时，向下有界搜索 ``adapter_config.json``，
    找到就按 PEFT 处理，并按**真实 adapter 目录**出结论（而不是把根目录当 adapter）。

    Args:
        checkpoint_path: 用户给的路径。
        adapter_dir: 显式指定的 adapter 目录（绕过搜索）。

    Returns:
        ``CheckpointClassification``。

    Raises:
        OrchestrationError: 既识别不出形态，也找不到嵌套 adapter。
    """
    if adapter_dir:
        return classify_checkpoint(adapter_dir)
    try:
        return classify_checkpoint(checkpoint_path)
    except ExportError as original:
        resolved = resolve_peft_adapter_dir(checkpoint_path)
        if resolved is None:
            raise OrchestrationError(str(original)) from None
        return classify_checkpoint(resolved)


def resolve_eval_target(
    *,
    role: EvalRole,
    served_name: str,
    checkpoint_path: str | None,
    base_model_path: str,
    export_format: str | None,
    merged_output_dir: str | None,
    adapter_dir: str | None = None,
) -> tuple[EvalModel, str]:
    """决定"要评测哪个模型目录"，并把决定记录成 ``EvalModel``。

    **本函数不做任何容器操作**——它只解析路径与形态。真正的合并
    （`merge_peft_checkpoint`）与 vLLM 启动由调用方在锁卡后执行。

    Args:
        role: 模型在本链路中的角色（base / after / sft / other）。
        served_name: 注册到 vLLM 的服务名（``--served-model-name``）。
        checkpoint_path: 训练产物路径；base 角色时为 ``None``。
        base_model_path: 底座模型路径（也是 base 角色的评测对象）。
        export_format: 产物形态提示；``None`` 表示自动识别。
        merged_output_dir: 合并产物落盘目录（需要合并时必填）。
        adapter_dir: 显式指定的 PEFT adapter 目录（绕过自动搜索）。

    Returns:
        ``(EvalModel, model_directory_ready_for_serving)``。

    Raises:
        OrchestrationError: 无法确定要服务的目录，或缺少必要的路径参数。
    """
    if checkpoint_path is None:
        if role != "base":
            raise OrchestrationError(
                f"role={role!r} requires a checkpoint_path; only role='base' may omit it"
            )
        return (
            EvalModel(
                served_name=served_name,
                role=role,
                path=base_model_path,
                base_model_path=base_model_path,
            ),
            base_model_path,
        )

    classification = _classify_or_find_adapter(checkpoint_path, adapter_dir=adapter_dir)
    if export_format and export_format != classification.kind.value:
        raise OrchestrationError(
            f"export_format={export_format!r} contradicts the detected checkpoint kind "
            f"{classification.kind.value!r} at {checkpoint_path}"
        )

    if classification.kind is CheckpointKind.BASE_MODEL:
        return (
            EvalModel(
                served_name=served_name,
                role=role,
                path=str(classification.path),
                base_model_path=base_model_path,
                checkpoint_path=str(checkpoint_path),
                export_format=classification.kind.value,
            ),
            str(classification.path),
        )

    if merged_output_dir is None:
        raise OrchestrationError(
            f"checkpoint {checkpoint_path} is {classification.kind.value} and needs merging, "
            "but no merged_output_dir was configured"
        )

    if classification.kind is CheckpointKind.PEFT_ADAPTER:
        effective_adapter = adapter_dir or str(classification.path)
        model = EvalModel(
            served_name=served_name,
            role=role,
            path=str(merged_output_dir),
            base_model_path=base_model_path,
            checkpoint_path=str(checkpoint_path),
            export_format=CheckpointKind.PEFT_ADAPTER.value,
        )
        return model, effective_adapter

    model = EvalModel(
        served_name=served_name,
        role=role,
        path=str(merged_output_dir),
        base_model_path=base_model_path,
        checkpoint_path=str(checkpoint_path),
        export_format=CheckpointKind.GRASPO_NATIVE.value,
    )
    # native 形态：交给既有 ``graspo export --config``；这里只报出需要合并，
    # 由 scripts/eval_export_checkpoint.sh 调用既有导出路径。
    return model, str(classification.path)


def _assert_file_persisted(path: Path, *, what: str) -> None:
    """断言产物**真的**落在预期位置且非空；否则显式报错，绝不静默（宪法 §2.3/§13.1）。

    为什么需要这条：``write_text`` 成功**不等于**产物在预期位置。历史实测事故
    （`task-r3-effect` §⑦-3）：`run_evaluation` 用 cwd 相对路径当输出路径，
    容器 cwd 与挂载点不一致时产物写进了意料之外的目录——数字拿到了，
    `eval_report.json` 却不在宿主机上，**没有任何报错**。评测结果的落点是留证的
    前提，落错位置必须比"没落盘"更早、更响亮地暴露。

    Args:
        path: 期望已经存在的产物文件。
        what: 用于报错文案的产物名（如 ``eval_report.json``）。

    Raises:
        OrchestrationError: 文件不存在，或存在但为空。
    """
    if not path.is_file():
        raise OrchestrationError(
            f"{what} was reported as written but is not present at {path} — the artifact "
            "did not land where the config promised. Refusing to report success "
            "(silent loss of evaluation artifacts is exactly what this guard prevents)."
        )
    if path.stat().st_size == 0:
        raise OrchestrationError(f"{what} at {path} is empty — refusing to report success")


def _resolve_artifact_dir(output_dir: str | Path) -> Path:
    """把产物目录解析成**绝对路径**；相对路径一律拒绝（fail-closed）。

    Args:
        output_dir: 配置给出的产物目录。

    Returns:
        绝对路径（尚未创建）。

    Raises:
        OrchestrationError: 路径为空或为相对路径。
    """
    if output_dir is None or str(output_dir).strip() == "":
        raise OrchestrationError(
            "output_dir is required for run_evaluation; artifacts must have a configured home"
        )
    candidate = Path(str(output_dir)).expanduser()
    if not candidate.is_absolute():
        raise OrchestrationError(
            f"output_dir={str(output_dir)!r} is not absolute; refusing to resolve it against "
            f"the process cwd ({Path.cwd()}) — a cwd-relative artifact path silently lands "
            "somewhere else (or is lost when that location is not mounted). "
            "Make the path absolute in the eval config."
        )
    return candidate


def run_evaluation(
    *,
    base_url: str,
    model: EvalModel,
    dataset: Dataset,
    output_dir: str | Path,
    gpus: str | None,
    top_p: float = EVAL_TOP_P,
    max_tokens: int = EVAL_MAX_TOKENS,
    seed: int | None = None,
    max_workers: int = 8,
    overlap_indices: set[int] | None = None,
    train_subset: EvalTrainSubset | None = None,
    environment_fingerprint: dict[str, Any] | None = None,
    dataset_split: str = "test",
    dataset_path_for_report: str | None = None,
    wait_for_service: bool = True,
) -> EvalReport:
    """跑一次完整评测并落盘 ``eval_report.json`` + ``environment.json``。

    Args:
        base_url: vLLM OpenAI 兼容端点（如 ``http://127.0.0.1:18889``）。
        model: 被评测模型的标识。
        dataset: 已加载的评测数据集。
        output_dir: 产物目录（应当位于 ``.local/`` 下，见宪法 §16）。
        gpus: **显式**卡列表字符串。``None``/空 → 拒绝启动（fail-closed）。
        top_p: 与 v3 对齐的 top_p（温度 0 时无效果）。
        max_tokens: 与 v3 对齐。
        seed: 仅作记录（温度为 0 时解码本身是确定的）。
        max_workers: 并发请求数。
        overlap_indices: 图像重叠样本下标。``None`` = **未做分析**；空集 = 做了
            但重叠集为空（两者在产物里靠 ``overlap_analysis_performed`` 区分）。
        train_subset: 本次用于重叠分析的训练子集标识（路径 + sha256 + 切片长度）。
            做了分析却没给标识时拒绝——否则事后无法解释剔除口径的数值从何而来。
        environment_fingerprint: 环境指纹（写进 ``environment.json``）。
        dataset_split: 数据集划分名（test / train / ...）。
        dataset_path_for_report: 报告里记录的数据集路径；``None`` 用 dataset.path。
        wait_for_service: 是否先轮询等服务就绪。

    Returns:
        落盘的 ``EvalReport``。

    Raises:
        OrchestrationError: 锁卡防呆失败或服务不可用。
    """
    if overlap_indices is not None and train_subset is None:
        raise OrchestrationError(
            "overlap analysis was performed but no train_subset metadata was supplied; "
            "the artifact must record which training slice drove the overlap exclusion "
            "(otherwise the excluded-accuracy number cannot be explained later)"
        )
    plan: GpuPlan = resolve_gpu_plan(gpus)
    destination = _resolve_artifact_dir(output_dir)
    destination.mkdir(parents=True, exist_ok=True)

    run_id = make_run_id(model.role)
    started_at = utc_now_iso()
    wall_started = time.monotonic()
    client = VllmEvalClient(
        ClientConfig(
            base_url=base_url,
            served_model_name=model.served_name,
            max_workers=max_workers,
        )
    )
    if wait_for_service:
        try:
            client.wait_until_ready()
        except VllmEvalError as exc:
            raise OrchestrationError(f"vLLM service not ready at {base_url}: {exc}") from exc

    records = client.evaluate_samples(dataset.samples)
    summary = summarize([record.to_dict() for record in records], overlap_indices=overlap_indices)

    fingerprint_path = destination / ENVIRONMENT_FILENAME
    fingerprint_payload: dict[str, Any] = {
        "run_id": run_id,
        "visible_gpus": plan.devices,
        "gpu_count": plan.count,
        "endpoint": base_url,
        "captured_at": utc_now_iso(),
    }
    fingerprint_payload.update(environment_fingerprint or {})
    fingerprint_path.write_text(
        json.dumps(fingerprint_payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    _assert_file_persisted(fingerprint_path, what=ENVIRONMENT_FILENAME)

    report = EvalReport(
        run_id=run_id,
        started_at=started_at,
        finished_at=utc_now_iso(),
        elapsed_sec=round(time.monotonic() - wall_started, 3),
        model=model,
        dataset=EvalDataset(
            path=dataset_path_for_report or str(dataset.path),
            split=dataset_split,
            sha256=dataset.sha256,
            sample_count_total=len(dataset.samples),
            sample_count_requested=len(dataset.samples),
        ),
        train_subset=train_subset,
        decoding=EvalDecoding(
            temperature=EVAL_TEMPERATURE,
            top_p=top_p,
            max_tokens=max_tokens,
            enable_thinking=False,
            seed=seed,
        ),
        criteria=EvalCriteria(
            version=ACCURACY_CRITERIA_VERSION,
            source=ACCURACY_CRITERIA_SOURCE,
        ),
        environment=EvalEnvironmentPointer(
            visible_gpus=list(plan.devices),
            gpu_count=plan.count,
            fingerprint_path=str(fingerprint_path),
        ),
        summary=summary,
        samples=[record.to_dict() for record in records],
    )
    write_report(report, destination)
    return report


def write_report(report: EvalReport, output_dir: str | Path) -> Path:
    """把报告落盘为 ``<output_dir>/eval_report.json``，返回路径。

    产物落点必须**绝对**且落盘后必须**真的存在**——否则显式报错（宪法 §2.3）。
    见 :func:`_assert_file_persisted` 里登记的历史静默丢失事故。
    """
    resolved = _resolve_artifact_dir(output_dir)
    destination = resolved / EVAL_REPORT_FILENAME
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(report.model_dump(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    _assert_file_persisted(destination, what=EVAL_REPORT_FILENAME)
    return destination


def resolve_peft_adapter(checkpoint_path: str | Path) -> Path | None:
    """薄封装：从 ms-swift 输出目录定位 PEFT adapter（透传 `merged_export`）。"""
    return resolve_peft_adapter_dir(checkpoint_path)
