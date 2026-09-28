"""ms-swift on-policy rollout 的**可复现播种**（集成层补丁，不改 ms-swift 源码）。

**职责边界**（宪法 §1.1）

- **本模块**：在**所有**走 ms-swift on-policy 生成的 graspo 通道上
  （``backend="msswift"`` 的 ``train_method="opd"`` 与 ``train_method="graspo"``），
  保证**每次 on-policy rollout 生成之前**全局 torch RNG 都从一个**由
  ``(training.seed, 进程 rank, 该进程内的第几次推理调用)`` 唯一确定**的值起步；
  从而"同 config 同 seed 同卡位双跑"的首步监督信号一致。
- **不负责**：采样分布/温度的取值（那是算法语义，见下）、GKD 的 ``lmbda``
  分支、教师侧任何东西、数据集准备、参数映射。

**播种值为什么是派生量而不是 ``training.seed`` 本身**（真机教训，必读）

第一版接线（``219abdd``/``p1``）在每次 ``infer`` 前重播 ``training.seed`` 的**字面值**。
这样做两件事同时发生：

1. **各 rank 播同一个值** ⇒ 与上游 ``rlhf_trainers/grpo_trainer.py:143`` 的
   ``set_seed(args.seed, device_specific=True)`` **背道而驰**（后者的有效种子 =
   ``seed + process_index``，注释原文是"防止生成重复补全"）；
2. **每次调用仍从同一个起点起步**，于是"起点确定"这件事被过度满足成"每次调用同一流"。

结果（T033 真机）：4 个 rank 的补全长度被压到**同一个上限**（``completions/max_length``
恒 30.0，未打补丁时是 30–96），首行 ``frac_reward_zero_std=1.0``。
⇒ **可复现性不能靠"压掉多样性"换来。**

本模块的正解：把**唯一的播种值** ``training.seed`` 当作**基础种子**，用
:func:`derive_rollout_seed` 派生出**每次调用各不相同、每个 rank 各不相同、
但对 (seed, rank, 调用序号) 完全确定**的种子。三条性质分别对应：

- 确定性 ⇒ 同 config/seed/卡位双跑逐位相同；
- rank 区分 ⇒ 复现上游 ``device_specific=True`` 的意图，跨 rank 不再重复补全；
- 调用序号区分 ⇒ 相邻两次 rollout 不再从同一随机流起步。

**组内多样性从哪来**（这是本方案的关键，不是副作用）

``num_generations=8`` 的 8 份补全**不是** 8 次独立的 ``infer`` 调用，而是
**一次** ``TransformersEngine.infer`` 里的 8 条请求（ms-swift 在
``rlhf_trainers/rollout_mixin.py:1359`` 一次性把整批 ``infer_requests`` 交给引擎，
引擎在 ``infer_engine/transformers_engine.py:588-601`` 内循环、每次
``_infer_full`` → ``template.generate`` 可能是批量生成）。批量采样**每一行独立
消耗随机流**，所以"重播一次、采一批"本来就产生 8 个**不同**的样本。
真正会杀掉组内方差的是"**每份生成前各重播一次**"——本模块因此**每次
``infer`` 调用只重播一次**，且每次的种子不同。

**单一真相源（§1.4）：为什么"怎么播种"只写在这一处**

播种本身（``_reseed_before_rollout`` → ``torch.manual_seed``）与接线协议
（``rollout_seeding`` = "进入时装包装、退出时 fail-closed 校验真的触发过"）
都只在本模块实现一次。两条通道（OPD / GRPO）**只允许**用 ``rollout_seeding``
这一个入口把训练那次调用包起来，不得各自复制
"``rollout_seed_deterministic`` + ``assert_rollout_seed_applied``"的组合逻辑——
否则"播种口径"会分叉成两份，改一处忘另一处。

**GRPO 通道为什么也需要它**（真机定案，见工位报告 §①）

``T033``（9B·GRASPO·LoRA·ms-swift·4 卡）两跑的**生成内容本身**就不同
（``completions/mean_length`` 37.5 vs 39.0、``max_length`` 65 vs 77、reward 亦不同）
⇒ 生成侧未钉，而不是归约/内核类（归约类的签名是"首步同、终态漂"）。
GRPO 与 GKD 在 ``use_vllm=false`` 下**共用同一个** ``TransformersEngine``，
而 ``grpo_trainer.py:143`` 只在 trainer ``__init__`` 播一次种
（``device_specific=True`` ⇒ 有效种子 = ``seed + rank``）⇒ GRPO 通道与
OPD 通道的缺陷是同一个，修复也必须是同一处。

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
而不是"同一次运行内每步不同"。以**确定派生的**种子重播 ⇒ 两跑的 rollout 序列
逐字相同（本模块的测试用"两次独立进程式消耗"复现了这一点）。

**"每次调用同一个字面值"是错的，"每次调用同一个函数值"才对**：上一版把
"确定"读成了"相等"，于是两次调用、两个 rank 都被钉到同一条流上。本模块改成
"同一个**纯函数**、不同的**取值**"——确定性一点也不少，多样性一分也不少。

**与 ``beta`` 无关**（纪律 ``P-26``）：ms-swift GKD 的 ``beta`` 是**散度方向**，
graspo 的 ``beta`` 是 **KL 惩罚系数** —— 同名不同义。本模块一个 ``beta`` 都不碰。
"""

from __future__ import annotations

import contextlib
import inspect
import logging
import os
from collections.abc import Iterator
from typing import Any

logger = logging.getLogger(__name__)

#: ms-swift 里"真正从全局 RNG 取随机数"的那个 rollout 引擎方法。
#: 路径与类名逐字取自 ms-swift 4.5.3（``swift/infer_engine/transformers_engine.py:50/573``）。
_ENGINE_MODULE = "swift.infer_engine.transformers_engine"
_ENGINE_CLASS = "TransformersEngine"
_ENGINE_METHOD = "infer"

#: 派生种子时给进程 rank 预留的步长。
#:
#: 取值理由（不是"越大越好"，是"必然不撞"）：同一个 rank 在一次运行里的
#: rollout 调用次数 = 优化步数 / ``steps_per_generation``，实测量级是 10^2
#: （T033: 40 步 / 2 = 20 次）。10**6 比它大四个数量级，因此
#: ``seed + RANK_STRIDE*rank + call_index`` 在 rank ∈ [0, 世界大小) 内
#: **必然两两不等**（相邻 rank 的区间相隔 10**6，而 call_index 远小于它），
#: 也就不会出现"rank 1 的第 k 次调用"与"rank 0 的第 k+10**6 次调用"撞车。
#: 选 1000 而不是 10**6 则会在 1000 步以上发生跨 rank 撞车 —— 那是可预见的
#: 未来回归，不能接受。
_RANK_STRIDE = 10**6


def resolve_rollout_seed(config: Any) -> int:
    """从 graspo 配置取出本次运行的**基础**播种值（**唯一真相源** = ``training.seed``）。

    与 ``_config_mapping.py:479`` 送给 ms-swift 的 ``--seed`` **是同一个值**：
    ``training.seed``。这里不发明第二个种子、不新增配置项、不读环境变量
    （宪法 §1.4 / §7.1）。**它只是基础种子**——每次调用实际用的种子由
    :func:`derive_rollout_seed` 从它派生。

    Args:
        config: ``GraspoConfig``（或等价对象，带 ``training.seed``）。

    Returns:
        基础播种用的整数。

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


def resolve_rollout_rank() -> tuple[int, str]:
    """取本进程的 **rank**，返回 ``(rank, 来源说明)``。

    来源优先级（**显式优先**，§2.2）：``RANK`` 环境变量 → ``LOCAL_RANK`` 环境变量
    → ``torch.distributed.get_rank()`` → 单进程兜底 ``0``。

    为什么先读环境变量：ms-swift 的 rank 由 ``torchrun`` 通过 ``RANK``/``LOCAL_RANK``
    下发；在分布式进程组尚未初始化时读 ``torch.distributed`` 会抛错，而环境变量
    在任何时刻都可用。来源说明会被写进日志账本，让"这个 rank 是怎么来的"可事后核对。

    返回 "(0, ...)" 只在**确实**是单进程时成立——此时 rank 区分退化为无操作，
    与上游 ``device_specific=True`` 在 ``process_index=0`` 上的行为一致。
    """
    for key in ("RANK", "LOCAL_RANK"):
        raw = os.environ.get(key)
        if raw is None or raw.strip() == "":
            continue
        try:
            return int(raw), f"env:{key}"
        except ValueError:
            logger.warning(
                "graspo rollout seeding: %s=%r is not an int; ignoring it for the rank offset.",
                key,
                raw,
            )
    try:
        import torch.distributed as dist

        if dist.is_available() and dist.is_initialized():
            return int(dist.get_rank()), "torch.distributed"
    except ImportError:  # pragma: no cover - torch 总在 ms-swift 路径上
        # 无 torch ⇒ 显式落入单进程兜底（来源说明与下方逐字一致，口径不变）
        return 0, "single-process"
    return 0, "single-process"


def derive_rollout_seed(*, base_seed: int, rank: int, call_index: int) -> int:
    """把基础种子、rank、调用序号派生成**本次调用**的播种值（纯函数，可离线测试）。

    ``derived = base_seed + RANK_STRIDE * rank + call_index``

    三条性质（§6 可复现 + GRPO 组内多样性）：

    1. **确定性**：纯函数，只依赖 ``(config.seed, rank, call_index)`` 三个显式入参，
       不读时钟、不读 ``torch.initial_seed()``、不做随机化。
    2. **rank 区分**：不同 rank 的派生值落在互不相交的区间里（见 ``_RANK_STRIDE``），
       复现上游 ``set_seed(..., device_specific=True)`` 的意图——各进程生成不重复的补全。
    3. **调用序号区分**：同一 rank 内第 ``k`` 与第 ``k+1`` 次 rollout 从不同随机流起步，
       因此"相邻两次生成逐字相同"这种流复用不会发生。

    Args:
        base_seed: ``training.seed``（唯一真相源，见 :func:`resolve_rollout_seed`）。
        rank: 进程 rank（见 :func:`resolve_rollout_rank`），必须 ≥ 0。
        call_index: 本进程内第几次调用（从 0 起），必须 ≥ 0。

    Returns:
        派生后的非负种子。

    Raises:
        ValueError: ``rank``/``call_index`` 为负，或派生结果溢出到负数 ——
            ``torch.manual_seed`` 接受负数，但"负种子"只能来自整数溢出这类
            未经预期的输入，静默传下去等于把未定义语义当正常。
    """
    if rank < 0:
        raise ValueError(f"graspo rollout seeding: rank must be >= 0 (got {rank}).")
    if call_index < 0:
        raise ValueError(f"graspo rollout seeding: call_index must be >= 0 (got {call_index}).")
    derived = base_seed + _RANK_STRIDE * rank + call_index
    if derived < 0:
        raise ValueError(
            "graspo rollout seeding: derived seed overflowed to a negative value "
            f"(base_seed={base_seed}, rank={rank}, call_index={call_index}, derived={derived})."
        )
    return derived


def _reseed_before_rollout(seed: int) -> None:
    """把全局 torch RNG 钉到 ``seed``（每个 rollout 前调用一次）。

    ``torch`` 在此处**延迟导入**：本模块只在 ms-swift 引擎已存在（即真的要跑
    rollout）时才会走到这里，避免让"仅做链路接线/构造门面"的调用方被拖进重型导入。
    """
    import torch

    torch.manual_seed(seed)


def _first_step_probe_active() -> bool:
    """首步探针是否开启（判据**只有一处**：进程内绑定的 ``DeterminismSwitch``，§1.4）。

    默认关 ⇒ 生成侧一次 import / 一次计算都不做（零行为变化、零开销）。
    """
    try:
        from graspo.core.determinism import active_switch
    except ImportError:  # pragma: no cover - core 恒可用
        return False
    return bool(active_switch().probe_first_step)


def _capture_first_step_probe_generation_inputs(
    *, rank: int, call_index: int, call_seed: int, args: tuple[Any, ...], kwargs: dict[str, Any]
) -> None:
    """只读旁路：把本 rank **首次** rollout 的生成输入指纹交给首步探针（不落盘、不通信）。

    为什么在生成侧抓：``input_ids_sha256``（训练 micro-batch = prompt+completion）只能
    说明"整批输入变没变"，**分不出**是"prompt 变了（输入侧）"还是"prompt 没变、补全变了
    （采样/数值侧）"。补上"生成输入"的 sha 并列，E1 才能二选一：
    生成输入 sha 相同 + 训练 ``input_ids`` sha 不同 ⇒ 采样/数值侧。

    探针关闭（默认）⇒ **直接返回**，不 import 探针模块、不触摸请求对象。
    """
    if not _first_step_probe_active():
        return
    from graspo.flow.msswift.first_step_probe import capture_generation_inputs

    infer_requests = args[0] if args else kwargs.get("infer_requests")
    capture_generation_inputs(
        rank=rank, call_index=call_index, call_seed=call_seed, infer_requests=infer_requests
    )


@contextlib.contextmanager
def rollout_seed_deterministic(config: Any) -> Iterator[Any]:
    """在 ms-swift 的 rollout 生成边界上**作用域内**钉定全局 RNG。

    包装 ``swift.infer_engine.transformers_engine.TransformersEngine.infer``：
    每次调用**先** ``torch.manual_seed(derive_rollout_seed(...))``，再走原方法。
    于是"每次 on-policy rollout 生成之前全局 RNG 都处于确定起点"，而生成本身
    （采样分布、温度、``num_generations``）逐字不变。

    **链式种子（本方案的核心）**：第 ``k`` 次调用用
    ``training.seed + RANK_STRIDE * rank + k``。因此

    - 同一 rank 的同一次两跑 → 同一个 ``k`` → 同一个种子 → 逐位相同（可复现）；
    - 不同 rank → 种子相差 ``RANK_STRIDE`` 的整数倍 → 互不相同的流
      （复现 ``device_specific=True`` 的意图，保住 GRPO 组内方差）；
    - 同一 rank 的相邻两次调用 → 相差 1 → 不复用同一条流。

    Args:
        config: ``GraspoConfig``（基础种子取 ``training.seed``）。

    Yields:
        一个可变账本 ``dict``：``{"reseed_count": int, "required": bool,
        "seed": int, "rank": int, "rank_source": str, "call_seeds": list[int]}``。
        调用方在训练结束后**必须**检查 ``reseed_count > 0``：为 0 说明
        ``TransformersEngine.infer`` 一次都没被调用过 —— 那么这次运行的
        "已播种"是**假的**，必须 fail-closed（§2.3 边界校验即防呆）。

    Note:
        - 若 ms-swift 未安装 / 引擎类或方法不存在，本上下文**不做任何事**并把
          ``required`` 置 ``False``（此时连 ms-swift 都进不去，谈不上 rollout）；
          一旦类存在但方法缺失，抛 ``RuntimeError``（上游漂移必须显式暴露，
          不能静默失去播种）。
        - 包装是**幂等**的：重复进入不会叠加多层包装。
        - **每次 ``infer`` 只重播一次**：``num_generations`` 份补全是在**一次**
          调用内的批量采样，批量里每一行独立消耗随机流 ⇒ 组内样本天然不同。
          若改成"每份生成前各重播一次"，组内 8 份会逐字相同、advantage 全零
          （真机事故，见文件头与 ``tests/flow/msswift/test_rollout_seed.py`` 的
          负向对照）。
    """
    base_seed = resolve_rollout_seed(config)
    rank, rank_source = resolve_rollout_rank()
    ledger: dict[str, Any] = {
        "reseed_count": 0,
        "required": False,
        "seed": base_seed,
        "base_seed": base_seed,
        "rank": rank,
        "rank_source": rank_source,
        "call_seeds": [],
    }

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
        """包装后的 ``infer``：每次调用前按调用序号重播种子，再转交原方法。

        Args:
            self: ms-swift 推理引擎实例（原方法绑定对象）。
            *args: 原样转交原 ``infer``。
            **kwargs: 原样转交原 ``infer``（并作为首步探针的输入留证）。
        """
        # 先取"这是第几次调用"，再自增 —— 于是第一次是 0。
        call_index = ledger["reseed_count"]
        call_seed = derive_rollout_seed(base_seed=base_seed, rank=rank, call_index=call_index)
        ledger["reseed_count"] += 1
        ledger["call_seeds"].append(call_seed)
        _reseed_before_rollout(call_seed)
        _capture_first_step_probe_generation_inputs(
            rank=rank, call_index=call_index, call_seed=call_seed, args=args, kwargs=kwargs
        )
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
            "graspo rollout seeding: re-seeded the global torch RNG before each of %d on-policy "
            "rollout generation call(s); first=%s last=%s (base_seed=%s rank=%s rank_source=%s "
            "stride=%s).",
            ledger["reseed_count"],
            ledger["call_seeds"][0],
            ledger["call_seeds"][-1],
            base_seed,
            rank,
            rank_source,
            _RANK_STRIDE,
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


@contextlib.contextmanager
def rollout_seeding(config: Any) -> Iterator[dict[str, Any]]:
    """**唯一的 rollout 播种接线入口**（§1.4）：包住"跑训练的那一次调用"。

    OPD 与 GRPO 两条通道都只用 ``with rollout_seeding(config): <run training>``，
    不各自复制"装包装 + 退出校验"的组合逻辑。

    语义 = ``rollout_seed_deterministic``（进入时装作用域包装，退出即还原上游方法）
    **加上** ``assert_rollout_seed_applied``（退出时 fail-closed 校验播种真的触发过），
    并在退出前把账本**显式打进日志**（§2.2 显式即防呆）——"播种生效"这件事必须
    能在运行日志里被看到，而不是只能靠读代码相信。

    Args:
        config: ``GraspoConfig``（基础种子取 ``training.seed``，单一真相源）。

    Yields:
        账本 ``dict``（``reseed_count`` / ``required`` / ``seed`` / ``base_seed`` /
        ``rank`` / ``rank_source`` / ``call_seeds``）；需要时调用方可读。

    Raises:
        RuntimeError: 补丁装了但一次都没触发（见 ``assert_rollout_seed_applied``）。
    """
    with rollout_seed_deterministic(config) as ledger:
        yield ledger
    # 显式性（§2.2）：这一行是"本次运行确实在每次 rollout 前重播了、重播的是什么值"的
    # **日志凭证**。必须同时能看见：基础种子、rank 区分方式、重播次数、以及实际用过的
    # 首/末种子 —— 否则"播种生效"只能靠读代码相信，而"各 rank 是否播了同一个值"
    # （正是上一版的塌缩事故根因）在日志里看不出来。
    # 用 warning 级（与 ``graspo msswift RL argv``/``OPD argv`` 同口径）——
    # 训练进程的 INFO 未必落地，而这条事实必须事后可查（§2.2 / §6）。
    call_seeds = ledger.get("call_seeds") or []
    logger.warning(
        "graspo rollout seeding applied: backend=msswift patch_installed=%s "
        "reseed_count=%s base_seed=%s rank=%s rank_source=%s rank_stride=%s "
        "first_call_seed=%s last_call_seed=%s",
        ledger.get("required"),
        ledger.get("reseed_count"),
        ledger.get("base_seed"),
        ledger.get("rank"),
        ledger.get("rank_source"),
        _RANK_STRIDE,
        call_seeds[0] if call_seeds else None,
        call_seeds[-1] if call_seeds else None,
    )
    assert_rollout_seed_applied(ledger)
