"""msswift 后端的「从 checkpoint 续训」启动前守卫（**fail-closed，绝不静默重跑**）。

**职责边界（§1.1 一事一责）**

- **本模块**：``training.resume_from_checkpoint`` 在 msswift 通道上的**启动前校验**
  ——它是"要不要真的续训"这件事的唯一边界（§1.4 单一真相源）。
- **不负责**：拼 ms-swift 参数（``_config_mapping.py`` 的 ``--resume_from_checkpoint``
  一行）、启动 ms-swift、恢复训练状态（那是 ms-swift / transformers 的训练循环）。

**为什么必须有这道门（实测缺陷，2026-10-22 A800 M003 副本）**

graspo 早就把 ``resume_from_checkpoint`` 写在配置契约里（``core/schema.py``），native
后端也真的会恢复状态；但 msswift 通道**一条 argv 都没有产出**它 ⇒ 用户给了路径，
ms-swift 收到的是 ``resume_from_checkpoint=None`` ⇒ 从 ``global_step=0`` 重跑
（实测证据：``args.json`` 里 ``resume_from_checkpoint = None``、日志
``'global_step/max_steps': '1/100'``、``learning_rate 2e-05``、epoch 0.04）。
**静默丢弃配置 = 最坏的失败形态**（§2.2 显式即防呆 / §7.2 拒绝假配置）：
14.5h 的档位会白跑一遍，用户还以为在续训。

因此这里把"给了路径却不成立"的三种形态全部变成**启动前**的可读报错：

1. 路径不存在 ⇒ :class:`FileNotFoundError`（与 native 侧
   ``native/trainer/checkpoint.py::_resume_if_requested`` 同型同文）；
2. 路径不是目录 ⇒ :class:`NotADirectoryError`；
3. 目录里没有 :data:`TRAINER_STATE_FILE` ⇒ :class:`RuntimeError`（与 native 侧
   "missing trainer_state" 的语义对齐：没有 ``trainer_state.json`` 就没有
   ``global_step``／优化器步数，transformers 无从恢复 ⇒ 只能从头训练）。

以及两条**必须并存的非法组合**（配置层，无需 IO）：

4. 与 ``training.overwrite_output_dir: true`` 同时给 ⇒ :class:`ValueError`
   —— ``overwrite`` 的语义就是"清掉输出目录"，而续训要读的目录往往正是它（或它的
   子目录）；两者并存等于"先删掉要续训的原料再续训"。
5. 与 ``lora.adapter_path`` 同时给 ⇒ :class:`ValueError`，与 native 侧
   ``native/runtime.py::validate_native_runtime_config`` **逐字同一条规则**：
   续训恢复的是完整 trainer 状态（含 optimizer/scheduler），加载 PEFT adapter 只是
   LoRA warm-start，二者语义冲突。

**为什么校验放在训练器（设施层）而不是映射层**：``_config_mapping.py`` 是"零 IO 的
纯映射"（§1.3 计算与设施分离，有测试锁定）；路径检查要碰文件系统，属于设施层。
四个训练通道（SFT / RL / CPT / OPD）在调用 ``validate_combinations`` 的同一处调用
本函数，因此**四条通道的 fail-closed 语义完全相同**。

**ms-swift 侧真的会续训吗（实测依据，ms-swift 4.5.3 + transformers 5.12.1）**

- ``swift/arguments/sft_args.py:200``：``resume_from_checkpoint`` 在 ``__post_init__``
  里 ``to_abspath(..., check_exists=True)``，``resume_only_model=False``（默认）时
  **不改** ``model``/``adapters``，权重由 trainer 侧加载；
- ``swift/pipelines/train/sft.py:237-263``：``_get_resume_checkpoint()`` 直接返回
  ``args.resume_from_checkpoint``，并 ``trainer.train(resume_checkpoint)``
  ⇒ **参数确实被消费**（这点必须查证：transformers 5.12.1 的
  ``TrainingArguments.resume_from_checkpoint`` 自称 "Not directly used by Trainer"，
  若 ms-swift 不转发就会重演同一个静默丢弃 bug）。
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

#: 「可续训 checkpoint」的判据文件（**单一真相源**，本模块与测试都引用它）。
#: transformers 的 ``Trainer`` 靠它恢复 ``global_step`` / epoch / 优化器步数：
#: ``transformers/trainer.py`` 的 ``TrainerState.load_from_json(os.path.join(
#: resume_from_checkpoint, TRAINER_STATE_NAME))``。没有它就只能从第 1 步重跑。
TRAINER_STATE_FILE = "trainer_state.json"


def validate_resume_checkpoint(config: Any) -> str | None:
    """续训请求的**启动前** fail-closed 校验；未请求续训时什么都不做。

    Args:
        config: ``GraspoConfig`` 实例（只读 ``training`` / ``lora`` 两段）。

    Returns:
        校验通过的 checkpoint 目录字符串（``None`` = 配置里没有续训请求）。

    Raises:
        ValueError: 与 ``lora.adapter_path``（native 同一条规则）或
            ``training.overwrite_output_dir: true`` 同时给。
        FileNotFoundError: 路径不存在。
        NotADirectoryError: 路径存在但不是目录。
        RuntimeError: 目录里没有 :data:`TRAINER_STATE_FILE`（不可续训）。

    Note:
        每一条报错都**点名该改哪个键**并显式声明"不会退回从头训练"——静默降级正是
        这条 bug 的原始形态（§2.3 边界校验即防呆 / §3.2 透明退路）。
    """
    checkpoint = config.training.resume_from_checkpoint
    if not checkpoint:
        return None
    path = Path(checkpoint)

    # ── 配置层非法组合（无需 IO，先判：报错最便宜、信息最直接）──────────────
    if config.lora.adapter_path:
        # 与 native 侧逐字同一条规则（native/runtime.py::validate_native_runtime_config）。
        raise ValueError(
            "training.resume_from_checkpoint and lora.adapter_path cannot both be set "
            f"(msswift backend): resume_from_checkpoint={path} "
            f"lora.adapter_path={config.lora.adapter_path}. Resuming restores the full "
            "trainer state (weights + optimizer + LR scheduler + RNG), while "
            "lora.adapter_path is only a LoRA warm-start — pick one."
        )
    if config.training.overwrite_output_dir:
        raise ValueError(
            "training.resume_from_checkpoint cannot be combined with "
            "training.overwrite_output_dir: true (msswift backend): "
            f"resume_from_checkpoint={path}. overwrite_output_dir means 'wipe the output "
            "directory first', which destroys the very directory being resumed from "
            "(or its parent). Set training.overwrite_output_dir: false when resuming."
        )

    # ── 文件系统层（不成立的路径一律启动前报错，⛔ 不退回从头训练）────────────
    if not path.exists():
        raise FileNotFoundError(
            "training.resume_from_checkpoint does not exist (msswift backend): "
            f"{path}. Refusing to start — silently ignoring it would train from step 1 "
            "again (that is exactly the bug this check exists for)."
        )
    if not path.is_dir():
        raise NotADirectoryError(
            "training.resume_from_checkpoint is not a directory (msswift backend): "
            f"{path}. A resumable checkpoint is a directory produced by a previous run "
            "(checkpoint-<step>)."
        )
    if not (path / TRAINER_STATE_FILE).is_file():
        raise RuntimeError(
            "msswift checkpoint is missing "
            f"{TRAINER_STATE_FILE} (msswift backend): {path}. Refusing to start — "
            "transformers restores global_step / optimizer steps from that file, so "
            "without it the run would silently restart from step 1. Point "
            "training.resume_from_checkpoint at a real checkpoint-<step> directory "
            "(saved with save_only_model: false)."
        )
    return str(path)
