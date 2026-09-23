"""每 rank 首步探针：只读旁路，落「首步 local loss + 首批 input_ids 指纹」。

**为什么需要它（判别实验的"天然分水岭"）**

`task-x1-sft-repro/report.md` §⑤ E1 与 `task-a3-determinism/report.md` 的结论：
多卡 SFT 同批首步 loss 差 `3.2389e-3`（1 个 bf16 ulp 量级）时，静态分析**分不出**
"输入侧"与"数值侧"。判读规则只有一条：

- 两跑的同 rank ``input_ids_sha256`` **不同** ⇒ 首批内容不同 ⇒ 根因在数据/种子链
  （**可修**，不是宪法 §6 的排除项）；
- 两跑的 sha **逐位相同**而 local loss 不同 ⇒ 同输入不同结果 ⇒ 数值/内核侧，
  转到确定性开关（``determinism.enabled``）做 A/B。

**边界（一事一责，§1.1 / §1.3）**

- **本模块**：指纹算法、旁路落盘（写 ``rank_metrics.rank_XXXXX.jsonl``，
  与 native 侧既有 per-rank 旁路**同一个文件、同一种行格式**）、ms-swift 回调的
  注册与安装（**不改 ms-swift 源码**，§1.2 对扩展开放）。
- **不负责**：判据（A4/A6 一字不改）、训练语义、采集层解析（采集层按
  ``phase`` 与 ``metrics`` 字段判定，本模块的行**不带 metrics** ⇒ 被显式跳过，
  不会污染逐步指标；见 ``scripts/collect_results.py::_read_step_metrics``）。

**旁路承诺（不侵入既有训练逻辑）**

安装点只有一处：**包装** ``trainer.compute_loss`` —— 读它的返回值与原 batch，
仅在本 rank 的**第一次**调用时写一行 JSON 并置位"已记录"，之后立即直通。
无集合通信、无全局状态、不改变张量、不改变返回值、不改变任何超参。

**GRPO / OPD 的落点为什么与 SFT 不同（真机事故，必读）**

第一版只查**顶层** ``inputs["input_ids"]``，于是：SFT（扁平 batch，``input_ids`` 与
``labels`` 同层）有产出；**GRPO/OPD 零产出**——因为 ms-swift 的 RLHF ``compute_loss``
收到的是 ``{"model_inputs": {"input_ids": ...}, "grpo_batch": ...}``（构造点
``rlhf_trainers/grpo_trainer.py:767``，消费点 ``:862-876``），顶层没有 ``input_ids``。
包装**确实被调用**，但 ``batch is None`` ⇒ 静默不落盘。

修法是把"从 batch 里取 ``input_ids``"收进**唯一真相源** :func:`input_ids_from_batch`
（两套形状都在同一处判断），而不是在每个通道里各写一份。

**三条并列通道（口径不许分叉，§1.4）**

1. ``input_ids_sha256``：训练 micro-batch 的输入指纹（SFT 与 GRPO 共用取键函数）。
2. ``generation_input_sha256``：**生成输入**的指纹（rollout 首次 ``infer`` 的
   ``infer_requests`` 内容规范化后 sha256；由 :func:`capture_generation_inputs`
   在生成侧抓取、:func:`generation_input_fingerprint` 取回）。只有它 + 通道 3 的
   sha 并列，才能把"prompt 变了（输入侧）"与"prompt 没变、补全变了（采样/数值侧）"分开。
3. 两条**并列落盘**的 loss 行：局部行（``loss_caliber="per_rank_local"``，字段
   ``local_loss``）与全局行（``loss_caliber="cross_rank_mean"``，字段 ``global_loss``，
   来自 ``trainer.log()`` 的 ``logs['loss']`` = ms-swift ``nested_gather(tr_loss).mean()``）。
   **为什么必须并列**：接入探针后，若台账只看到 per-rank 局部值就把它当首步 loss，
   口径会从"跨 rank 均值"悄悄变成"局部"（同一档两批读数不可比）。
"""

from __future__ import annotations

import dataclasses
import datetime
import hashlib
import json
import logging
import weakref
from pathlib import Path
from typing import Any

#: 旁路行的 ``phase`` 名（**唯一真相源**；已登记进
#: ``graspo.core.result_judge.DIAGNOSTIC_PHASES``，采集侧据此判"无指标可丢"）。
PROBE_PHASE = "first_step_probe"

#: ms-swift ``--callbacks`` 里使用的回调名（注册进 ``swift.callbacks.callbacks_map``）。
PROBE_CALLBACK_NAME = "graspo_first_step_probe"

#: ``input_ids`` 指纹的算法标识：写进产物，事后核对时知道这串 sha 是怎么来的（§2.2）。
FINGERPRINT_ALGORITHM = "sha256(dtype|shape|raw_uint8_le_bytes_of_contiguous_cpu_tensor)"

#: 生成输入指纹的算法标识（**与张量指纹不同**：生成输入是一批请求对象，不是张量）。
GENERATION_INPUT_ALGORITHM = (
    "sha256(utf8 of canonical json of infer_requests[].messages; "
    "non-json values replaced by '<TypeName>')"
)

#: 局部口径标识：``local_loss`` = 本 rank **未归约**的 loss（唯一真相源）。
LOCAL_LOSS_CALIBER = "per_rank_local"

#: 全局口径标识：``global_loss`` = 跨 rank 均值（与 ``logging.jsonl`` 的 ``loss`` 同口径）。
GLOBAL_LOSS_CALIBER = "cross_rank_mean"

#: 全局口径的取数来源说明（写进产物，事后知道这串数从哪来，§2.2 显式）。
GLOBAL_LOSS_SOURCE = (
    "trainer.log(logs['loss']) @first on_log "
    "(ms-swift '__maybe_log_save_evaluate' 的 nested_gather(tr_loss).mean(); "
    "与 logging.jsonl 的 'loss' 同口径)"
)


def fingerprint_input_ids(input_ids: Any) -> tuple[str, list[int], str]:
    """计算 ``input_ids`` 张量的**内容指纹**（跨进程/跨机器可比的确定性算法）。

    算法（唯一真相源 :data:`FINGERPRINT_ALGORITHM`）：把张量搬成 **CPU 上连续的**
    张量，按 dtype 的原始字节序取 ``view(uint8)`` 的字节流，再对
    ``"<dtype>|<shape>|" + bytes`` 做 sha256 —— **形状与 dtype 也进入哈希**，
    因此"同内容不同形状"不会被误判为同一次输入（§2.2 显式）。

    Args:
        input_ids: 任意 torch 张量（或已经是 CPU numpy 数组的等价物）。

    Returns:
        ``(sha256_hexdigest, shape_list, dtype_name)``。
    """
    import torch  # 延迟导入：本模块在无 torch 环境仍可导入（§1.3）

    tensor = input_ids
    if not torch.is_tensor(tensor):
        raise TypeError(f"fingerprint_input_ids expects a torch.Tensor, got {type(tensor)!r}")
    cpu = tensor.detach().to("cpu").contiguous()
    shape = [int(size) for size in cpu.shape]
    dtype_name = str(cpu.dtype).replace("torch.", "")
    if cpu.numel() == 0:
        raw = b""
    elif cpu.dtype == torch.uint8:
        raw = cpu.numpy().tobytes()
    else:
        # bf16 / fp16 / fp8 等没有 numpy 等价 dtype，按原始字节取（不改数值、不反量化）。
        raw = cpu.view(torch.uint8).numpy().tobytes()
    digest = hashlib.sha256(
        f"{dtype_name}|{shape}|".encode("utf-8") + raw
    ).hexdigest()
    return digest, shape, dtype_name


def input_ids_from_batch(inputs: Any) -> Any | None:
    """从 ``compute_loss`` 收到的 batch 里取训练 micro-batch 的 ``input_ids``（唯一真相源）。

    两套 batch 形状只在**本函数**里判断，SFT / GRPO / OPD 三条通道共用，不各写一份
    （§1.4）：

    - **SFT（扁平）**：``{"input_ids": Tensor, "labels": Tensor, ...}``
      —— ``swift/trainers/trainer.py:66-75`` 的 ``compute_loss`` 就是读 ``inputs['labels']``；
    - **GRPO / OPD（嵌套）**：``{"model_inputs": {"input_ids": Tensor, ...},
      "grpo_batch": ...}`` —— 构造点 ``rlhf_trainers/grpo_trainer.py:767``，消费点
      ``:862-876``（``model_inputs = inputs['model_inputs']``）。

    非 tensor / 两个位置都没有 ⇒ ``None``（调用方据此**不落盘**，而不是落一行空记录）。

    ★ 这是"探针在 GRPO 路径零产出"的修复点：旧实现只查顶层 ``inputs["input_ids"]``，
      GRPO 的顶层没有该键 ⇒ 静默不落盘。
    """
    import torch  # 延迟导入：本模块在无 torch 环境仍可导入（§1.3）

    if not isinstance(inputs, dict):
        return None
    nested = inputs.get("model_inputs")
    for candidate in (inputs.get("input_ids"), nested.get("input_ids") if isinstance(nested, dict) else None):
        if torch.is_tensor(candidate):
            return candidate
    return None


def _canonical_generation_value(value: Any) -> Any:
    """把生成请求里的任意值归一成**确定性可 JSON 化**的结构（防呆：不用 ``repr``）。

    ``repr`` 含对象地址（如 ``<PIL.Image.Image object at 0x7f...>``）⇒ 同一输入两跑会
    得到不同字节、指纹失去可比性。因此非 JSON 值只保留**类型名**这一稳定信息。
    """
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, dict):
        return {
            str(key): _canonical_generation_value(value[key])
            for key in sorted(value, key=str)
        }
    if isinstance(value, (list, tuple)):
        return [_canonical_generation_value(item) for item in value]
    return f"<{type(value).__name__}>"


def fingerprint_generation_inputs(infer_requests: Any) -> tuple[str, int, str]:
    """计算 rollout **生成输入**的内容指纹（算法见 :data:`GENERATION_INPUT_ALGORITHM`）。

    Args:
        infer_requests: ms-swift ``TransformersEngine.infer`` 收到的请求（可迭代或单个）；
            本函数只读它们的 ``messages`` 属性（``InferRequest.messages``，
            见 ``swift/infer_engine/protocol.py:45-95``）。

    Returns:
        ``(sha256_hexdigest, request_count, algorithm)``。
    """
    if isinstance(infer_requests, (list, tuple)):
        requests = list(infer_requests)
    else:
        requests = [infer_requests]
    messages = [
        _canonical_generation_value(getattr(request, "messages", None)) for request in requests
    ]
    payload = json.dumps(messages, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest(), len(requests), GENERATION_INPUT_ALGORITHM


@dataclasses.dataclass(frozen=True, slots=True)
class GenerationInputFingerprint:
    """本 rank **首次** rollout 的生成输入指纹（生成侧抓取 → 首步记录器取回，§1.4）。"""

    sha256: str
    request_count: int
    call_index: int
    call_seed: int
    algorithm: str


#: 每进程只保留**首次** rollout 的生成指纹（本 rank 的第一代 = 首步监督信号的输入侧）。
#: 键 = rank；一个进程只跑一个 rank，因此"首次即定"是安全的（后续调用不覆写）。
_GENERATION_INPUTS: dict[int, GenerationInputFingerprint] = {}


def capture_generation_inputs(
    *, rank: int, call_index: int, call_seed: int, infer_requests: Any
) -> GenerationInputFingerprint | None:
    """只读旁路：把本 rank **首次** rollout 的生成输入指纹记入内存（生成侧调用）。

    幂等：同一 rank 已有记录 ⇒ 直接返回既有值，不重算、不覆写（首个采样批才是首步
    监督信号的输入）。**不做集合通信、不落盘**（落盘由首步记录器统一做，避免两处写）。

    Returns:
        本次（或既有）指纹；``infer_requests`` 为空/取不到 ``messages`` 时返回 ``None``。
    """
    existing = _GENERATION_INPUTS.get(int(rank))
    if existing is not None:
        return existing
    if infer_requests is None:
        return None
    digest, count, algorithm = fingerprint_generation_inputs(infer_requests)
    fingerprint = GenerationInputFingerprint(
        sha256=digest,
        request_count=count,
        call_index=int(call_index),
        call_seed=int(call_seed),
        algorithm=algorithm,
    )
    _GENERATION_INPUTS[int(rank)] = fingerprint
    return fingerprint


def generation_input_fingerprint(rank: int) -> GenerationInputFingerprint | None:
    """取回本 rank 首次 rollout 的生成输入指纹（未抓到 ⇒ ``None``，如 SFT 无生成）。"""
    return _GENERATION_INPUTS.get(int(rank))


def clear_generation_input_fingerprints() -> None:
    """清空生成指纹缓存（**仅供测试**：进程级状态需要用例间复位）。"""
    _GENERATION_INPUTS.clear()


def _loss_value_and_dtype(loss: Any) -> tuple[float, str] | None:
    """从 compute_loss 的返回值里取"本 rank 未归约的局部 loss"（取不到 ⇒ ``None``）。"""
    import torch  # noqa: PLC0415

    candidate = loss
    if isinstance(candidate, dict):
        candidate = candidate.get("loss")
    if not torch.is_tensor(candidate):
        return None
    local = candidate.detach().to("cpu")
    return float(local), str(local.dtype).replace("torch.", "")


@dataclasses.dataclass(frozen=True, slots=True)
class FirstStepProbeRecord:
    """一行探针记录的**字段契约**（单一真相源；下游按这些字段名读）。

    Attributes:
        rank: 该记录的 rank 序号（``RANK`` 语义，来自 ms-swift ``args.process_index``）。
        step: ``global_step``（首步为 1）。
        local_loss: 该 rank **未归约**的首步局部 loss（float，CPU 化后的 python float）。
        loss_dtype: loss 张量的 dtype 名（记录精度口径，便于解释 ulp 量级）。
        input_ids_sha256: 首批 ``input_ids`` 的内容指纹（算法见
            :data:`FINGERPRINT_ALGORITHM`）。
        input_ids_shape / input_ids_dtype: 指纹的两个入参，缺失时 sha 不可复现。
        epoch: 该步所处的 epoch（ms-swift ``state.epoch``）。
        global_loss: 跨 rank 均值的首步 loss（全局口径行专用；局部行留 ``None``）。
        global_loss_source: 全局口径的取数来源说明（§2.2 显式）。
        loss_caliber: 本行 loss 的口径标识（防"局部当全局"的读数分叉，§1.4）。
        generation_input_sha256 / generation_input_algorithm /
        generation_input_request_count / generation_input_call_index /
        generation_input_call_seed: 生成输入通道（算法见
            :data:`GENERATION_INPUT_ALGORITHM`；SFT 无生成 ⇒ 全 ``None``）。
    """

    rank: int
    step: int
    local_loss: float | None = None
    loss_dtype: str | None = None
    input_ids_sha256: str | None = None
    input_ids_shape: list[int] | None = None
    input_ids_dtype: str | None = None
    epoch: float | None = None
    #: 全局口径首步 loss（跨 rank 均值；与 ``logging.jsonl`` 的 ``loss`` 同口径）。
    #: 与 ``local_loss`` **并列落盘**，由 :data:`GLOBAL_LOSS_CALIBER` 标注口径（§1.4）。
    global_loss: float | None = None
    #: 全局口径的取数来源说明（事后可核，见 :data:`GLOBAL_LOSS_SOURCE`）。
    global_loss_source: str | None = None
    #: 本行 loss 的口径标识（``per_rank_local`` / ``cross_rank_mean``）——**防口径分叉**。
    loss_caliber: str = LOCAL_LOSS_CALIBER
    #: 生成输入指纹（rollout 首次 ``infer`` 的 ``infer_requests`` 内容；SFT 无生成 ⇒ ``None``）。
    generation_input_sha256: str | None = None
    generation_input_algorithm: str | None = None
    generation_input_request_count: int | None = None
    generation_input_call_index: int | None = None
    generation_input_call_seed: int | None = None


def probe_payload(record: FirstStepProbeRecord) -> dict[str, Any]:
    """把 :class:`FirstStepProbeRecord` 渲染成旁路 JSONL 的 payload（唯一渲染点）。

    ★ **不得**出现 ``metrics`` 键：采集侧按该键把记录分流为逐步指标
    （``scripts/collect_results.py::_rank_metric_steps``）；诊断行带了它就会污染台账。
    """
    return {
        "event": PROBE_PHASE,
        "phase": PROBE_PHASE,
        "kind": "diagnostic",
        # 与显存快照不同：本行**就是**为跨跑比对而落的，因此显式声明可复现口径。
        "reproducible": True,
        "timestamp": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
        "fingerprint_algorithm": FINGERPRINT_ALGORITHM,
        "rank": int(record.rank),
        "step": int(record.step),
        # ── 双通道并列：口径由 loss_caliber 显式标注，读的人不可能把局部当全局 ──
        "loss_caliber": record.loss_caliber,
        "local_loss": None if record.local_loss is None else float(record.local_loss),
        "global_loss": None if record.global_loss is None else float(record.global_loss),
        "global_loss_source": record.global_loss_source,
        "loss_dtype": record.loss_dtype,
        "input_ids_sha256": record.input_ids_sha256,
        "input_ids_shape": None if record.input_ids_shape is None else list(record.input_ids_shape),
        "input_ids_dtype": record.input_ids_dtype,
        # ── 生成输入通道（SFT 无生成 ⇒ 全 None，不伪造）──
        "generation_input_sha256": record.generation_input_sha256,
        "generation_input_algorithm": record.generation_input_algorithm,
        "generation_input_request_count": record.generation_input_request_count,
        "generation_input_call_index": record.generation_input_call_index,
        "generation_input_call_seed": record.generation_input_call_seed,
        "epoch": record.epoch,
    }


def record_global_first_step_probe(
    output_dir: str | Path,
    *,
    rank: int,
    step: int,
    global_loss: float,
    epoch: float | None = None,
    source: str = GLOBAL_LOSS_SOURCE,
    filename: str | None = None,
) -> Path:
    """落**全局口径**首步 loss 的并列行（跨 rank 均值；局部行由记录器在 ``compute_loss`` 落）。

    两行**同一文件、同一 phase、同一字段契约**，靠 ``loss_caliber`` 区分 ——
    这样台账既拿得到"哪个 rank 变了"的局部证据，也拿得到与 ``logging.jsonl``
    可比的首步值，不会因为接入探针而让首步 loss 口径悄悄从均值变局部（§1.4）。
    """
    return record_first_step_probe(
        output_dir,
        FirstStepProbeRecord(
            rank=int(rank),
            step=int(step),
            global_loss=float(global_loss),
            epoch=epoch,
            loss_caliber=GLOBAL_LOSS_CALIBER,
            global_loss_source=source,
        ),
        filename=filename,
    )


def record_first_step_probe(
    output_dir: str | Path,
    record: FirstStepProbeRecord,
    *,
    filename: str | None = None,
) -> Path:
    """把一个 rank 的首步探针记录追加到该 rank 的 ``rank_metrics`` 旁路文件。

    Args:
        output_dir: 运行输出目录（与 native 侧 ``_emit_rank_memory_event`` 同一处）。
        record: 记录。
        filename: 覆盖文件名（测试用）；``None`` ⇒ 用
            :func:`graspo.flow.logging.rank_metrics_filename`（单一真相源）。

    Returns:
        实际写入的文件路径。
    """
    from graspo.flow.logging import append_jsonl_segment, rank_metrics_filename, run_log_dir

    name = filename or rank_metrics_filename(record.rank)
    path = run_log_dir(output_dir) / name
    append_jsonl_segment(path, probe_payload(record))
    return path


#: 已安装记录器的 trainer 实例（幂等标记；用弱引用，不给第三方对象挂属性）。
_INSTALLED_TRAINERS: weakref.WeakSet[Any] = weakref.WeakSet()


def install_first_step_recorder(
    trainer: Any,
    *,
    output_dir: str | Path,
    rank: int,
    epoch: float | None = None,
) -> None:
    """把一个 rank 的"首步记录器"安装到 **trainer 实例**上（只读旁路，包装 compute_loss）。

    为什么是包装 ``compute_loss``：ms-swift / HF 的 ``TrainerCallback`` 钩子
    （``on_step_end`` 等）**拿不到** batch 与 loss（回调签名里没有这两个参数），
    而 ``compute_loss`` 恰好在每个 micro-batch 上收到 ``inputs`` 并返回 loss ——
    它是"能同时看到输入与损失"的**唯一**稳定接口。

    包装是幂等的：同一个 trainer 实例重复调用时不再二次包装（标记用模块级
    ``WeakSet`` 记录，不往第三方对象上挂属性）。

    Args:
        trainer: ms-swift / HF ``Trainer`` 实例。
        output_dir: ms-swift 实际使用的输出目录（取 ``args.output_dir``，不另造来源）。
        rank: 该进程的 rank。
        epoch: 记录时用的 epoch（``on_train_begin`` 的 ``state.epoch``）。
    """
    if trainer in _INSTALLED_TRAINERS:
        return
    _INSTALLED_TRAINERS.add(trainer)
    original = trainer.compute_loss
    state = {"recorded": False}

    def _wrapped_compute_loss(model: Any, inputs: Any, return_outputs: bool = False, **kwargs: Any):
        result = original(model, inputs, return_outputs=return_outputs, **kwargs)
        if not state["recorded"]:
            # ``return_outputs=True`` 时 HF 的契约是 ``(loss, outputs)``；但 ms-swift 的
            # GRPO ``compute_loss`` 在两条分支上都只返回 loss 张量
            # （``rlhf_trainers/grpo_trainer.py:862-891``）⇒ 不能盲目 ``result[0]``
            # （0 维张量取下标会 ``IndexError``）。按"是不是二元组"判断，不按声明判断。
            is_pair = isinstance(result, (tuple, list))
            loss_source = result[0] if (return_outputs and is_pair) else result
            loss_info = _loss_value_and_dtype(loss_source)
            # ★ GRPO 的 ``inputs`` 是嵌套的（``model_inputs``），SFT 是扁平的 ⇒ 取键走
            #   唯一真相源 ``input_ids_from_batch``（旧实现只查顶层，GRPO 因此零产出）。
            batch = input_ids_from_batch(inputs)
            if loss_info is not None and batch is not None:
                digest, shape, dtype_name = fingerprint_input_ids(batch)
                generation = generation_input_fingerprint(int(rank))
                state["recorded"] = True
                record_first_step_probe(
                    output_dir,
                    FirstStepProbeRecord(
                        rank=int(rank),
                        step=1,
                        local_loss=loss_info[0],
                        loss_dtype=loss_info[1],
                        input_ids_sha256=digest,
                        input_ids_shape=shape,
                        input_ids_dtype=dtype_name,
                        epoch=epoch,
                        loss_caliber=LOCAL_LOSS_CALIBER,
                        generation_input_sha256=(generation.sha256 if generation else None),
                        generation_input_algorithm=(generation.algorithm if generation else None),
                        generation_input_request_count=(
                            generation.request_count if generation else None
                        ),
                        generation_input_call_index=(
                            generation.call_index if generation else None
                        ),
                        generation_input_call_seed=(generation.call_seed if generation else None),
                    ),
                )
        return result

    trainer.compute_loss = _wrapped_compute_loss


def install_probe_callback() -> Any:
    """把探针回调注册进 ms-swift 的 ``callbacks_map``，返回该回调类（幂等）。

    **为什么用回调而不是改 ms-swift**：ms-swift 的扩展点是
    ``TrainerArguments.callbacks`` + ``swift.callbacks.callbacks_map``（上游
    ``trainers/mixin.py`` 在 Trainer 构造时按名字查表）。graspo 侧注册一个新名字
    属于**对扩展开放**（§1.2），ms-swift 源码一行不改、也不 fork（§14.4 单向流动；
    第三方包不是我们的真相源）。

    注册必须发生在 ``sft_main(argv)`` 之前——ms-swift 解析参数时**不校验**
    ``--callbacks`` 的名字（只在 Trainer 构造时查表），因此只要注册在训练器构造前
    完成即可。

    Raises:
        ImportError: ms-swift 未安装（由调用方的 ``_require_ms_swift()`` 先行给出
            可操作提示；此处不吞异常，避免"探针静默没装"）。
    """
    import swift.callbacks as swift_callbacks
    from swift.callbacks import TrainerCallback

    existing = swift_callbacks.callbacks_map.get(PROBE_CALLBACK_NAME)
    if existing is not None:
        return existing

    class _GraspoFirstStepProbeCallback(TrainerCallback):
        """只读首步探针回调。

        - ``on_train_begin``：把记录器装到 trainer 上（包装 ``compute_loss``，落**局部**行）。
        - ``on_log``：首步日志落盘时，把**全局口径**（``logs['loss']`` = 跨 rank 均值）
          追加成**并列的第二行**——这样接入探针不会让首步 loss 口径从"全局均值"悄悄
          变成"局部"（§1.4 单一真相源；口径由 ``loss_caliber`` 字段自证）。
        """

        def __init__(self, args: Any, trainer: Any) -> None:
            super().__init__(args, trainer)
            self._global_recorded = False

        def on_train_begin(self, _args: Any, state: Any, control: Any, **kwargs: Any) -> None:
            output_dir = getattr(self.args, "output_dir", None)
            if output_dir is None:
                return
            rank = int(getattr(self.args, "process_index", 0) or 0)
            install_first_step_recorder(
                self.trainer,
                output_dir=output_dir,
                rank=rank,
                epoch=getattr(state, "epoch", None),
            )

        def on_log(
            self, _args: Any, state: Any, control: Any, logs: Any = None, **kwargs: Any
        ) -> None:
            """第一条日志 = 首步（ms-swift 首步必 log）⇒ 落全局口径行，之后不再落。"""
            if self._global_recorded:
                return
            loss = logs.get("loss") if isinstance(logs, dict) else None
            if isinstance(loss, bool) or not isinstance(loss, (int, float)):
                return
            output_dir = getattr(self.args, "output_dir", None)
            if output_dir is None:
                return
            self._global_recorded = True
            record_global_first_step_probe(
                output_dir,
                rank=int(getattr(self.args, "process_index", 0) or 0),
                step=int(getattr(state, "global_step", 1) or 1),
                global_loss=float(loss),
                epoch=getattr(state, "epoch", None),
            )

    swift_callbacks.callbacks_map[PROBE_CALLBACK_NAME] = _GraspoFirstStepProbeCallback
    return _GraspoFirstStepProbeCallback


def probe_extra_argv(enabled: bool) -> list[str]:
    """探针开启时**应当追加**的 ms-swift 命令行参数（关闭 ⇒ **空列表**，逐字零变化）。

    这是三条训练通道（SFT / GRPO / OPD）**共用的唯一接线点**（§1.4 单一真相源）：
    回调的注册与 ``--callbacks`` 的拼写只在这里出现一次，各通道只传自己的开关状态。
    调用方必须在拼 argv **之前**调用它——它同时完成 ``callbacks_map`` 的注册。

    Args:
        enabled: 该通道本次是否开启探针（判据 = `core.determinism.active_switch()`，
            见 :func:`probe_active_extra_argv`）。

    Returns:
        形如 ``["--callbacks", PROBE_CALLBACK_NAME]``；``enabled=False`` ⇒ ``[]``。
    """
    if not enabled:
        return []
    install_probe_callback()
    from graspo.flow.logging import rank_metrics_filename

    logging.getLogger(__name__).info(
        "graspo: per-rank first-step probe enabled (callback=%s, file=%s)",
        PROBE_CALLBACK_NAME,
        rank_metrics_filename(0),
    )
    # ★ **逐步日志密度**（2026-09-23 指挥官裁定的落地，路线 b = 运行期覆盖）：
    #   为什么放在这里：A7（训练真推进）需要"**覆盖每一步**的读数"才能自证"没有跳过
    #   优化器步"；而 ms-swift/transformers 的默认 `logging_steps=5` 只落 1/5 的步
    #   ⇒ 非 logging 步上发生的跳过**在日志里不可见**（A7 只能记「口径不可测」）。
    #   为什么这样落地最省（证据链）：①`logging_steps` 在项目自身的配置/生成器里
    #   **根本不存在**（全仓 grep 只命中 collector 注释与 transformers 库）；
    #   ②`_config_mapping.graspo_to_ms_swift_argv(..., extra_argv=...)` 是**文档写明的
    #   唯一覆盖入口**（"追加在末尾，可覆盖前面的同名参数"）；
    #   ③四条 ms-swift 通道**已在此处接线**（`extra_argv += probe_active_extra_argv()`）；
    #   ④门是 **CLI/env**（`--determinism --determinism-probe-first-step`，
    #   经 `GRASPO_DETERMINISM` → `--determinism-spec`）⇒ **不动任何生成物**
    #   （`samples/configs/matrix54/*.yaml` 是生成物，手改违规、228 禁跑生成器）。
    #   **探针关闭 ⇒ 本函数返回 `[]` ⇒ 逐字零行为变化**（默认关，既有跑次不受影响）。
    return ["--callbacks", PROBE_CALLBACK_NAME, "--logging_steps", "1"]


def probe_active_extra_argv() -> list[str]:
    """按**进程内已绑定的确定性开关**判断探针开关，返回应追加的 argv（关闭 ⇒ 空）。

    ``probe_first_step`` 的判据只有一处（``core.determinism.active_switch()``，
    §1.4）——本函数把"判据 + 接线"合成一个调用，三条训练通道因此不可能各自判错。
    """
    from graspo.core.determinism import active_switch

    return probe_extra_argv(bool(active_switch().probe_first_step))
