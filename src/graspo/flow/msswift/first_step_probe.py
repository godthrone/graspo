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
"""

from __future__ import annotations

import dataclasses
import datetime
import hashlib
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
    """

    rank: int
    step: int
    local_loss: float
    loss_dtype: str
    input_ids_sha256: str
    input_ids_shape: list[int]
    input_ids_dtype: str
    epoch: float | None


def probe_payload(record: FirstStepProbeRecord) -> dict[str, Any]:
    """把 :class:`FirstStepProbeRecord` 渲染成旁路 JSONL 的 payload（唯一渲染点）。"""
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
        "local_loss": float(record.local_loss),
        "loss_dtype": record.loss_dtype,
        "input_ids_sha256": record.input_ids_sha256,
        "input_ids_shape": list(record.input_ids_shape),
        "input_ids_dtype": record.input_ids_dtype,
        "epoch": record.epoch,
    }


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
            loss_source = result[0] if return_outputs else result
            loss_info = _loss_value_and_dtype(loss_source)
            batch = inputs.get("input_ids") if isinstance(inputs, dict) else None
            if loss_info is not None and batch is not None:
                digest, shape, dtype_name = fingerprint_input_ids(batch)
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
        """只读首步探针回调：``on_train_begin`` 时把记录器装到 trainer 上。"""

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

    swift_callbacks.callbacks_map[PROBE_CALLBACK_NAME] = _GraspoFirstStepProbeCallback
    return _GraspoFirstStepProbeCallback
