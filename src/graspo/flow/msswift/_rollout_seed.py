"""OPD(GKD) on-policy rollout 的**可复现播种**（集成层补丁，不改 ms-swift 源码）。

**职责边界**（宪法 §1.1）

- **本模块**：在 ``(train_method="opd", backend="msswift")`` 这条路径上，保证
  **每次 on-policy rollout 生成之前**全局 torch RNG 都被钉到配置里声明的
  ``training.seed``；从而"同 config 同 seed 双跑"的首步监督信号一致。
- **不负责**：采样分布/温度的取值（那是算法语义，见下）、GKD 的 ``lmbda``
  分支、教师侧任何东西、数据集准备、参数映射。

**为什么需要它**（真因，逐字取证见工位报告 §②）

ms-swift 4.5.3 的 GKD 在 ``use_vllm=false`` 下，学生现场采样走
``TransformersEngine`` —— 而这条路径**从全局 torch RNG 取随机数**：

- ``swift/infer_engine/protocol.py:199`` 有 ``RequestConfig.seed`` 字段；
- 但它的**消费者只有 vLLM 与 lmdeploy**（``infer_engine/vllm_engine.py:343``、
  ``infer_engine/lmdeploy_engine.py:177``）；``TransformersEngine`` 全文**不读**
  ``seed``（``grep -rn seed infer_engine/`` 里没有任何
  ``infer_engine/transformers_engine.py`` 命中）；
- 而 ``rlhf_trainers/rollout_mixin.py:356`` 构造 ``RequestConfig(...)`` 时
  **根本没传 ``seed``**（该文件 ``:13`` 只 **import** ``set_seed``，全文无调用）；
- ``rlhf_trainers/rollout_mixin.py:1110`` 的 ``_get_request_config`` 只在
  ``vllm_mode == 'colocate' and vllm_tensor_parallel_size > 1`` 时填 ``seed``
  —— 本通道两者都不成立；
- 唯一的播种是 ``rlhf_trainers/grpo_trainer.py:143``
  ``set_seed(args.seed, device_specific=True)``，位置在 **trainer ``__init__``**。
  ``GKDTrainer`` 经 ``super().__init__`` 继承，所以整场训练**只播这一次**。

⇒ 生成时刻的全局 RNG 状态 = 那次播种之后、进程内**一切**消耗过随机数的东西
的结果（dataloader worker 播种、数据集/模型构造等），**不是** ``seed`` 的确定函数
⇒ 同 config 同 seed 的两跑可以不同 ⇒ A4（可复现性）结构性不可过。

**为什么这是"可复现性措施"而不是"算法语义改变"**

全程只调用 ``torch.manual_seed(seed)`` —— 只把全局 RNG 的**起点**钉成确定值：

- 不动 ``temperature`` / ``top_p`` / ``top_k`` / ``min_p`` / ``repetition_penalty``；
- 不动 ``lmbda``（on-policy 分支选择由 ``gkd_trainer.py:445-448`` 的**隔离**
  ``random.Random(seed + global_step)`` 决定，不读全局 RNG，本补丁碰不到）；
- 不动 ``num_generations`` / 批次结构 / 教师侧；
- 不强制 greedy —— 采样照旧是采样，只是"从确定的起点开始采"。

对比：把 ``temperature`` 改成 0 才是算法语义改变（on-policy 采样 → 贪心），本模块
**不做**那件事（见 ``docs``/工位报告 §③ 的论证）。

**为什么不改上游**（宪法 §1.2 对扩展开放、对修改关闭）

本补丁与 ``_rope_compat.py`` 同构：在 ms-swift 的**边界**上**作用域内**包装
（调用原函数而不是替换），退出即还原；不 fork、不 vendored、不编辑 ms-swift 源码。

**为什么在集成层"再播一次"不会破坏"每次 rollout 前只播一次"的语义**

本补丁的播种**确实落在每个 rollout 之前**，这是有意的：``seed`` 不是"每步递增"
的序列种子，而是"本次运行"的确定起点。真正需要的是**同一配置的两跑一致**，
而不是"同一次运行内每步不同"。以固定的 ``seed`` 重播 ⇒ 两跑的 rollout 序列逐字
相同（本模块的测试用"两次独立进程式消耗"复现了这一点）。

**与 ``beta`` 无关**（纪律 ``P-26``）：ms-swift GKD 的 ``beta`` 是**散度方向**，
graspo 的 ``beta`` 是 **KL 惩罚系数** —— 同名不同义。本模块一个 ``beta`` 都不碰。
"""

from __future__ import annotations

import contextlib
import inspect
import logging
from typing import Any, Iterator

logger = logging.getLogger(__name__)

#: ms-swift 里"真正从全局 RNG 取随机数"的那个 rollout 引擎方法。
#: 路径与类名逐字取自 ms-swift 4.5.3（``swift/infer_engine/transformers_engine.py:50/573``）。
_ENGINE_MODULE = "swift.infer_engine.transformers_engine"
_ENGINE_CLASS = "TransformersEngine"
_ENGINE_METHOD = "infer"


def resolve_rollout_seed(config: Any) -> int:
    """从 graspo 配置取出本次运行的播种值（**唯一真相源** = ``training.seed``）。

    与 ``_config_mapping.py:479`` 送给 ms-swift 的 ``--seed`` **是同一个值**：
    ``training.seed``。这里不发明第二个种子、不新增配置项、不读环境变量
    （宪法 §1.4 / §7.1）。

    Args:
        config: ``GraspoConfig``（或等价对象，带 ``training.seed``）。

    Returns:
        播种用的整数。

    Raises:
        ValueError: ``training.seed`` 缺失或不是整数 —— **拒绝静默通过**（§13.1）：
            播种值不确定时不能假装可复现。
    """
    training = getattr(config, "training", None)
    seed = getattr(training, "seed", None)
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise ValueError(
            "graspo rollout seeding: training.seed is required to be an int to pin the "
            f"on-policy rollout RNG (got {seed!r} from training.seed). Refusing to start, "
            "because an unseeded rollout makes same-config/same-seed reruns diverge."
        )
    return seed


def _reseed_before_rollout(seed: int) -> None:
    """把全局 torch RNG 钉到 ``seed``（每个 rollout 前调用一次）。

    ``torch`` 在此处**延迟导入**：本模块只在 ms-swift 引擎已存在（即真的要跑
    rollout）时才会走到这里，避免让"仅做链路接线/构造门面"的调用方被拖进重型导入。
    """
    import torch

    torch.manual_seed(seed)


@contextlib.contextmanager
def rollout_seed_deterministic(config: Any) -> Iterator[Any]:
    """在 ms-swift 的 rollout 生成边界上**作用域内**钉定全局 RNG。

    包装 ``swift.infer_engine.transformers_engine.TransformersEngine.infer``：
    每次调用**先** ``torch.manual_seed(seed)``，再走原方法。于是"每次 on-policy
    rollout 生成之前全局 RNG 都处于确定起点"，而生成本身（采样分布、温度、
    ``num_generations``）逐字不变。

    Args:
        config: ``GraspoConfig``（播种值取 ``training.seed``）。

    Yields:
        一个可变账本 ``dict``：``{"reseed_count": int, "required": bool}``。
        调用方在训练结束后**必须**检查 ``reseed_count > 0``：为 0 说明
        ``TransformersEngine.infer`` 一次都没被调用过 —— 那么这次运行的
        "已播种"是**假的**，必须 fail-closed（§2.3 边界校验即防呆）。

    Note:
        - 若 ms-swift 未安装 / 引擎类或方法不存在，本上下文**不做任何事**并把
          ``required`` 置 ``False``（此时连 ms-swift 都进不去，谈不上 rollout）；
          一旦类存在但方法缺失，抛 ``RuntimeError``（上游漂移必须显式暴露，
          不能静默失去播种）。
        - 包装是**幂等**的：重复进入不会叠加多层包装。
    """
    seed = resolve_rollout_seed(config)
    ledger: dict[str, Any] = {"reseed_count": 0, "required": False, "seed": seed}

    try:
        module = __import__(_ENGINE_MODULE, fromlist=[_ENGINE_CLASS])
    except ImportError:
        logger.debug(
            "graspo rollout seeding: ms-swift (%s) is not importable; skipping the patch.",
            _ENGINE_MODULE,
        )
        yield ledger
        return

    engine_cls = getattr(module, _ENGINE_CLASS, None)
    if engine_cls is None:
        logger.debug(
            "graspo rollout seeding: %s has no %s; skipping the patch.",
            _ENGINE_MODULE,
            _ENGINE_CLASS,
        )
        yield ledger
        return

    original = inspect.getattr_static(engine_cls, _ENGINE_METHOD, None)
    if original is None or not callable(original):
        raise RuntimeError(
            "graspo rollout seeding: ms-swift's "
            f"{_ENGINE_MODULE}.{_ENGINE_CLASS}.{_ENGINE_METHOD} is missing (upstream drift?). "
            "Refusing to continue: without this hook the on-policy rollout cannot be re-seeded "
            "and same-config/same-seed reruns would silently diverge again."
        )

    # 幂等：已经装过就复用既有包装（不叠第二层）。
    if getattr(original, "_graspo_rollout_seeded", False):
        ledger["required"] = True
        yield ledger
        return

    ledger["required"] = True

    def _graspo_infer(self: Any, *args: Any, **kwargs: Any) -> Any:
        _reseed_before_rollout(seed)
        ledger["reseed_count"] += 1
        return original(self, *args, **kwargs)

    # 保留描述符语义：``infer`` 是普通实例方法（非 classmethod/staticmethod），
    # 但用 ``functools.wraps`` 会丢掉可读的标记 —— 这里显式打标记 + 逐字转发签名。
    _graspo_infer.__name__ = _ENGINE_METHOD
    _graspo_infer.__doc__ = original.__doc__
    _graspo_infer._graspo_rollout_seeded = True  # type: ignore[attr-defined]
    _graspo_infer._graspo_original = original  # type: ignore[attr-defined]

    engine_cls.infer = _graspo_infer
    try:
        yield ledger
    finally:
        engine_cls.infer = original

    if ledger["reseed_count"] > 0:
        logger.info(
            "graspo rollout seeding: re-seeded the global torch RNG to training.seed=%s before "
            "%d on-policy rollout generation call(s).",
            seed,
            ledger["reseed_count"],
        )


def assert_rollout_seed_applied(ledger: dict[str, Any]) -> None:
    """训练结束后校验"播种真的接线了"（fail-closed，§2.3）。

    Args:
        ledger: ``rollout_seed_deterministic`` 产出的账本。

    Raises:
        RuntimeError: 补丁装了（``required``）但一次都没触发 —— 说明这次运行的
            可复现性**没有被实际保证**，不能当成功交付。
    """
    if ledger.get("required") and ledger.get("reseed_count", 0) == 0:
        raise RuntimeError(
            "graspo rollout seeding: the patch was installed but "
            f"{_ENGINE_CLASS}.{_ENGINE_METHOD} was never called, so no on-policy rollout was "
            "re-seeded. This run's reproducibility is NOT guaranteed; refusing to report it "
            "as seeded."
        )
