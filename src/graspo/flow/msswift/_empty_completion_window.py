"""没有可训练位置的 micro-batch：**形状契约修复**（集成层；不改算法语义）。

**职责边界**

- **本模块**：判定"本 micro-batch 一个可训练位置都没有"，并在这种情况下把上游 logps
  切片的输入形状修回自洽（把 completion 窗口收窄到 ``S-1``）。只改**形状**，不改数值语义。
- **不负责**：奖励、逐字符标注、有效样本判定、优势、loss 公式 —— 这些都在 ``ripple/**``，
  本轮**一行不改**。
- **与 :mod:`graspo.flow.msswift._logps_shape_guard` 的分工**：守卫是**断言**
  （fail-closed，只报错、不改数值）；本模块是**修复**（只改形状、不抛）。两者判的不是同一
  件事，因此**不重复判**：本模块处理"可识别且可安全处理"的批，守卫兜住其余一切形状退化。
  本模块**不修改** ``_logps_shape_guard.py``（该模块整体保持原样）。

**为什么需要**（T037 实测，2026-10-01）

空/格式错 response ⇒ 该样本 ``labels`` 全 ``-100`` ⇒
``swift/rlhf_trainers/utils.py:2170`` 的 ``argmax(ne(labels, -100))`` 退化为 0 ⇒
``logits_to_keep = S``；上游 ``grpo_trainer.py:1570`` 的 ``[-(L+1):-1]`` 在 ``S ≤ L`` 时给
``S-1`` 行、``:1573`` 的 ``[-L:]`` 给 ``S`` 列 ⇒ ``selective_log_softmax``（``:1592``）的
gather 抛 ``RuntimeError: Size does not match at dimension 0 ...``（T037 pass2 实测签名：``expected index [2382, 1] to be no larger than self [2381, 248320]``）。

**为什么是"收窄窗口"而不是"整批跳过"**（2026-10-01 只读追查，file:line 可核）

micro-batch 是**按下标取用**的：``grpo_trainer.py:205-210``

.. code-block:: python

    num_rollout_samples = self.args.steps_per_generation * self.template.sequence_parallel_size
    generate_every = num_rollout_samples * self.num_iterations
    if self._step % generate_every == 0 or self._buffered_inputs is None:
        self._buffered_inputs = self._generate_and_score_completions(inputs)
    micro_batch = self._buffered_inputs[self._step % num_rollout_samples]

⇒ 缓冲列表长度必须恒为 ``steps_per_generation × sp_world_size``（本档 = 8）；少一项即取到
错的 micro-batch / IndexError。且 ``_postprocess_batch`` 会**把同一份 samples 再切一次**并要求
与 ``batch_encoded_inputs`` 1:1 对位（``grpo_trainer.py:254-256`` 的 ``assert``、
``graspo`` 侧 ``trainer.py:456-461`` 的 ``RuntimeError``）。⇒ **"整批跳过"在集成层不安全**；
本模块改为"**保留该 micro-batch、把它的窗口收窄 + mask 恒 False**"：数学效果与"跳过"一致
（该批 loss 严格为 0、不产生任何梯度），但**批数与图结构都不变**。

**判据（两半缺一不可）**

1. ``logits_to_keep >= seq_len``：上游那条 slicing 契约已经退化；
2. ``completion_mask.any() is False``：该批**没有任何可训练位置**。

只有两半**同时**成立才动手 ⇒ 收窄窗口**不可能丢任何真实 token**（本来一个都没有）。
"有真实 token 却 ``L >= S``"这种**理论上不该出现**的组合**故意不修**：留给形状守卫
fail-closed —— 那说明另有实现缺陷，必须暴露而不是掩盖。
"""

from typing import Any

__all__ = ["repair_empty_completion_window"]


def repair_empty_completion_window(grpo_batch: Any, seq_len: int) -> dict[str, int] | None:
    """把"无可训练位置"的 micro-batch 的形状契约修回自洽（**只改形状**）。

    Args:
        grpo_batch: ms-swift 的 ``GRPOBatch``（或任何带 ``logits_to_keep`` /
            ``completion_mask`` 的对象）。就地修改这两个字段。
        seq_len: 本次前向的序列长度 ``S``（``model_inputs['input_ids'].shape[-1]``）。

    Returns:
        ``None`` = 没动（正常批、或判据不成立、或没有可收窄的余地）；
        否则返回一个小字典，说明改了哪两个量（便于调用点记录，便于现场取证）。
    """
    if grpo_batch is None or not isinstance(seq_len, int) or seq_len < 2:
        # seq_len < 2 时没有可收窄的余地：不在这里制造更小的形状，交给守卫 fail-closed。
        return None
    logits_to_keep = getattr(grpo_batch, "logits_to_keep", None)
    completion_mask = getattr(grpo_batch, "completion_mask", None)
    if not isinstance(logits_to_keep, int) or completion_mask is None:
        return None
    if logits_to_keep < seq_len:
        return None  # 契约本来就自洽 —— 正常批恒走这条（零赋值路径）
    if bool(completion_mask.numel()) and bool(completion_mask.any()):
        return None  # 有可训练位置却 L>=S ⇒ 不掩盖，交给守卫
    new_keep = seq_len - 1
    grpo_batch.completion_mask = completion_mask[:, -new_keep:]
    grpo_batch.logits_to_keep = new_keep
    return {
        "seq_len": seq_len,
        "logits_to_keep_before": logits_to_keep,
        "logits_to_keep_after": new_keep,
    }
