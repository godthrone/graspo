"""checkpoint 语义判定（**纯逻辑**：无 torch、无 graspo 依赖，可在无 torch 机器上单测）。

**它防的是什么**

native 的导出通道（``flow/lora/lora_io.py``）是 **LoRA 语义**：它从
``lora_state_dict`` / ``lora_tensor_metadata`` 重建 LoRA A/B 对，再把 delta 合并回
基座权重。全参（``tuner_type: full``）的 checkpoint 里：

- ``lora_state_dict`` 是**空字典**（全参模式不创建 LoRA 矩阵）；
- 训练后的权重在 ``full_param_state_dict`` 里。

于是 ``merged-hf`` 路径会算出**零个 delta**，把**基座权重原样**写进输出目录——
**导出"成功"、文件齐全、权重一帧没学到**。这是最危险的一类失败（静默产出错误产物），
必须在边界上 fail-closed（宪法 §2.3）。
"""

from __future__ import annotations

from typing import Any


def is_full_param_payload(payload: dict[str, Any]) -> bool:
    """该 checkpoint shard 是否来自全参训练。

    两个判据取或（任一成立即认定）：

    - ``tuner_type == "full"``：v0.25+ 写出的 shard 显式带该字段；
    - ``full_param_state_dict is not None``：权重载荷本身存在（防御旧/异构 shard）。

    用 ``is not None`` 而不是真值判断（宪法 §2.2）：LoRA shard 里该键存在但值为
    ``None``，空字典/``None`` 的语义必须区分清楚。
    """
    if str(payload.get("tuner_type") or "") == "full":
        return True
    return payload.get("full_param_state_dict") is not None


def require_lora_semantics(payloads: list[dict[str, Any]], *, operation: str) -> None:
    """**防呆**：全参 checkpoint 不得走 LoRA 语义的导出通道。

    Args:
        payloads: 已加载的 rank shard（每个是一次 ``torch.load`` 的结果）。
        operation: 导出格式名（仅用于错误消息，如 ``"merged-hf"``）。

    Raises:
        ValueError: 任一片段来自全参训练。

    Note:
        native 全参导出（全参权重 → HF 格式）需要**完整的 native→HF 参数名映射**
        与 PP 分片拼装，**尚未实现**。这里的拒绝是有意的，不是漏配——宁可不导出，
        也不产出"看起来成功、实质是未训练基座"的产物。
    """
    for payload in payloads:
        if is_full_param_payload(payload):
            raise ValueError(
                f"Cannot export a full-parameter (tuner_type=full) native checkpoint as "
                f"'{operation}': this export path is LoRA-only. The shard stores no LoRA "
                "tensors (they are not created in full-parameter mode), so merging would "
                "silently emit an untrained copy of the base model. "
                "Use the checkpoint for resume (load_checkpoint supports full-parameter "
                "training state), or implement a full-parameter exporter "
                "(needs a native->HF parameter-name map plus PP shard assembly)."
            )
