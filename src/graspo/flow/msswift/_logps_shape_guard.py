"""GRPO per-token logps 的**形状守卫**（fail-closed：只报错，不改数值）。

**职责边界**

- **本模块**：在进入 ms-swift ``GRPOTrainer._get_logps_via_local_forward`` 的两处切片以及
  ``selective_log_softmax`` 之前，判定本次前向的序列长度 ``S`` 与整批算出的
  ``logits_to_keep``（记作 ``L``）是否形状自洽；不自洽 ⇒ 抛带**六个量**与 **file:line**
  的自定义异常，把 trl 深处一个不含张量语义的 index 错提前成可读的现场报告。
- **不负责**：改数值（不 clamp、不重算 ``logits_to_keep``、不改切片、不补 pad）、不重试、
  不降级、不改变训练结果。守卫**只把崩溃换成明确报错**，语义零变化。

**为什么只需一个判据**（推导；来源为本镜像内文件，sha256 已核）

ms-swift 4.5.3 ``swift/rlhf_trainers/grpo_trainer.py:1570/1573``：

.. code-block:: python

    logits = logits[:, -(logits_to_keep + 1):-1, :]        # :1570
    input_ids_for_logps = input_ids[:, -logits_to_keep:]   # :1573
    ...
    logps = selective_log_softmax(logits, input_ids_for_logps)   # :1592（非 padding_free 分支）

令 ``S = input_ids.shape[-1]``、``L = logits_to_keep``。对 ``S ≥ 1`` 逐段展开切片：

- ``logits`` 切片行数 ``rows = L  if S ≥ L+1  else S-1``
- ``input_ids_for_logps`` 列数 ``cols = min(S, L)``

两者相等 ⟺ ``S ≥ L+1``：

- ``S = L`` 时 ``rows = L-1``、``cols = L`` —— 恰好差 1；
- ``S < L`` 时 ``rows = S-1``、``cols = S`` —— 同样差 1。

``trl.trainer.utils.selective_log_softmax``（本镜像 ``trl 0.29.1``，``utils.py:562``）
在 bf16 分支逐 batch 行做 ``row_logps.gather(dim=-1, index=row_labels)``，其契约要求
``rows == cols``；因此 ``S ≤ L`` **必然**崩在同一处，报错是

.. code-block:: text

    RuntimeError: Size does not match at dimension 0 expected index [2382, 1]
    to be no larger than self [2381, 248320] apart from dimension 1

—— 这正是 T037 的原始签名（``S = L = 2382``，词表维 248320 与 ``config.vocab_size`` 一致，
**不是**失败维）。

**该判据对两条模型前向都必要且充分**：模型是否接受 ``logits_to_keep`` 形参（
``'logits_to_keep' in self.model_kwarg_keys``）只决定返回的 logits 是 ``S`` 行还是
``min(S, L+1)`` 行；把这一层代入后 ``rows``/``cols`` 公式与结论都不变。
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "MS_SWIFT_LOGPS_SLICE_SITE",
    "MS_SWIFT_LOGITS_TO_KEEP_SITE",
    "LogpsShapeGuardError",
    "logps_slice_shapes",
    "check_logps_shapes",
]

#: 抛出点（single source of truth；错误信息逐字引用，不在别处再写一份）。
MS_SWIFT_LOGPS_SLICE_SITE = "swift/rlhf_trainers/grpo_trainer.py:1570,1573 -> :1582/:1592"
#: ``logits_to_keep`` 的计算口径。
MS_SWIFT_LOGITS_TO_KEEP_SITE = "swift/rlhf_trainers/utils.py:2170"


class LogpsShapeGuardError(RuntimeError):
    """per-token logps 的形状在进入切片/gather 前已可判定不自洽（fail-closed）。

    只报告，不修复：持有六个量 + 两个 file:line 位置 + 可复算的预测形状。
    """

    def __init__(self, message: str, *, context: dict[str, Any]) -> None:
        super().__init__(message)
        self.context = context


def logps_slice_shapes(seq_len: int, logits_to_keep: int) -> tuple[int, int]:
    """按 ms-swift 的两处切片，算出 ``(logits 行数, input_ids_for_logps 列数)``。

    纯函数（无 torch 依赖），便于单测直接覆盖边界。
    """
    if seq_len < 1:
        raise ValueError(f"seq_len must be >= 1, got {seq_len}")
    if logits_to_keep < 0:
        raise ValueError(f"logits_to_keep must be >= 0, got {logits_to_keep}")
    rows = logits_to_keep if seq_len >= logits_to_keep + 1 else seq_len - 1
    cols = min(seq_len, logits_to_keep)
    return rows, cols


def check_logps_shapes(
    *,
    input_ids: Any,
    logits_to_keep: int,
    padding_free: bool,
    is_multimodal: bool | None,
    dynamic_num_samples: bool | None,
    model: Any = None,
    site: str = MS_SWIFT_LOGPS_SLICE_SITE,
) -> None:
    """形状自洽的前置断言；不自洽 ⇒ :class:`LogpsShapeGuardError`。

    Args:
        input_ids: 本次前向的 ``input_ids``（padding_free 下是 rmpad 张量，最后一维同为
            本次前向的序列长度 ``S``）。
        logits_to_keep: 整批算出的 ``L``（``grpo_batch.logits_to_keep``）。
        padding_free / is_multimodal / dynamic_num_samples: 判定走哪条分支所需的三个量，
            只进错误信息，不参与判据。
        model: 可选；给了就顺带核词表维（索引不得超过目标维）。
        site: 抛出点的 file:line（单一真相源常量，测试可覆写）。
    """
    seq_len = int(input_ids.shape[-1])
    keep = int(logits_to_keep)
    rows, cols = logps_slice_shapes(seq_len, keep)
    context: dict[str, Any] = {
        "input_ids.shape": tuple(int(x) for x in input_ids.shape),
        "S(input_ids.shape[-1])": seq_len,
        "logits_to_keep(L)": keep,
        "predicted_logits_slice_rows": rows,
        "predicted_input_ids_for_logps_cols": cols,
        "padding_free": bool(padding_free),
        "is_multimodal": is_multimodal,
        "dynamic_num_samples": dynamic_num_samples,
        "logits_to_keep_site": MS_SWIFT_LOGITS_TO_KEEP_SITE,
        "slice_site": site,
    }
    if rows != cols:
        msg = (
            f"[graspo/logps-shape-guard] 形状不自洽：本次前向 S={seq_len} ≤ "
            f"logits_to_keep L={keep} ⇒ logits 切片 {rows} 行 / "
            f"input_ids_for_logps {cols} 列，差 {cols - rows}；下一步 "
            "selective_log_softmax 的 gather 必然抛 "
            "'Size does not match at dimension 0 ... apart from dimension 1'。\n"
            f"六个量：S={seq_len} L={keep} logits_rows={rows} index_cols={cols} "
            f"padding_free={bool(padding_free)} is_multimodal={is_multimodal} "
            f"dynamic_num_samples={dynamic_num_samples}"
            f"（另 input_ids.shape={context['input_ids.shape']}）。\n"
            f"口径与抛出点：L 由 {MS_SWIFT_LOGITS_TO_KEEP_SITE} 从**整批** labels 算出；"
            f"两处切片与 gather 在 {site}。"
        )
        raise LogpsShapeGuardError(msg, context=context)

    # 索引不得超过目标维（gather 的最后一维契约）。只在能确定词表大小时判。
    vocab_size = _model_vocab_size(model)
    if vocab_size is not None and int(input_ids.numel()) > 0:
        max_id = int(input_ids.max())
        context["max_input_id"] = max_id
        context["model_vocab_size"] = vocab_size
        if max_id >= vocab_size:
            raise LogpsShapeGuardError(
                f"[graspo/logps-shape-guard] 索引越词表维：input_ids.max()={max_id} ≥ "
                f"model vocab_size={vocab_size}（gather 的最后一维契约）。\n"
                f"口径与抛出点：{site}；logits_to_keep 口径 {MS_SWIFT_LOGITS_TO_KEEP_SITE}。",
                context=context,
            )


def _model_vocab_size(model: Any) -> int | None:
    if model is None:
        return None
    config = getattr(model, "config", None)
    text_config = getattr(config, "text_config", None)
    candidates = (getattr(text_config, "vocab_size", None), getattr(config, "vocab_size", None))
    for candidate in candidates:
        if isinstance(candidate, int) and candidate > 0:
            return candidate
    return None
