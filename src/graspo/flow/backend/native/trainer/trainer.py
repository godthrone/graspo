"""Layer 2 — GraspoFlowTrainer：GRASPO 训练循环主类。

采用类改目录模式：统计、rollout、优化、checkpoint 分别驻留在独立文件中，
通过 mixin 组合到主类。外部使用者只 import 类名，完全不感知内部拆分。
"""

import json
import logging
import random
import time
from collections import deque
from pathlib import Path
from typing import Any

import torch.distributed as dist

from graspo.core.schema import GraspoConfig
from graspo.flow.data_io import load_jsonl
from graspo.flow.backend.native.logger.native_rollout_logger import NativeRolloutLogger
from graspo.flow.logging import setup_logging
from graspo.flow.backend.native.runtime import (
    GraspoFlowRuntime,
    GraspoFlowRuntimeBase,
    validate_native_runtime_config,
)
from graspo.flow.backend.native.trainer.checkpoint import CheckpointMixin
from graspo.flow.backend.native.trainer.helpers import (
    _backup_config,
    _set_model_init_seed,
    _set_random_seed,
    _timestamp,
)
from graspo.flow.backend.native.trainer.optimize import OptimizeMixin
from graspo.flow.backend.native.trainer.rollout import RolloutMixin
from graspo.ripple.buffer import ReplayBuffer
from graspo.ripple.monitoring.stats import (
    GraspoFlowEpochStats,
    GraspoFlowTrainStats,
    QueuedSample,
)
from graspo.ripple.monitoring.summary import round_timing_details
from graspo.ripple.reward.reward import create_reward

#: 默认进程组（WORLD）在 PyTorch/NCCL 日志里的名字，**仅用于诊断输出**。
#: 不通过私有 API（``distributed_c10d._get_default_group()``）取名——那会把日志耦合到
#: PyTorch 内部实现；这是与 NCCL 日志逐字一致的公开叫法（``PG ID 0 PG GUID 0(default_pg)``）。
_DEFAULT_PG_NAME = "default_pg"


class GraspoFlowTrainer(RolloutMixin, OptimizeMixin, CheckpointMixin):
    """GRASPO 训练循环，由 GraspoFlow 分布式运行时驱动。

    使用 mixin 组合：RolloutMixin（生成+评分）、OptimizeMixin（优化步骤）、
    CheckpointMixin（保存+恢复）。
    """

    def __init__(
        self,
        config: GraspoConfig,
        selection: Any | None = None,
        runtime: GraspoFlowRuntimeBase | None = None,
    ) -> None:
        self.config = config
        self.selection = selection
        self.runtime = runtime or GraspoFlowRuntime.from_config(config)
        self.reward = create_reward(config.reward)
        self.replay_buffer = ReplayBuffer()
        self._smoke_boundary = False  # 冒烟运行边界：首轮 optimize 后停止（见 train(smoke=)）
        self.stats = GraspoFlowTrainStats()
        self.backend_name = "native"
        self.global_step = 0
        self.sample_index = 0
        self.total_samples = 0
        self.started_at = time.monotonic()
        self.current_epoch_stats = GraspoFlowEpochStats()
        self.recent_groups: deque[dict[str, Any]] = deque(maxlen=50)
        self.pending_batch_attempts: list[dict[str, Any]] = []
        self.pending_batch_timings: list[dict[str, Any]] = []
        self.resume_info: dict[str, Any] | None = None
        self._last_checkpoint_time = 0.0  # 墙钟时间周期保存用（monotonic 秒）
        self.logger = self._build_rollout_logger()

    def _build_rollout_logger(self) -> NativeRolloutLogger:
        """构造 rollout 领域日志器（副作用：在 ``{out}/logs/<run_id>/`` 建目录）。

        **为什么必须是可重建的**（阻断 B 的真因，2026-09-20）：

        ``NativeRolloutLogger.__init__`` 会 ``mkdir`` 出 ``{out}/logs/<run_id>/``。
        ``train()`` 里 ``prepare_output_dir(out, overwrite=True)`` 在
        ``overwrite_output_dir: true`` 时执行 ``shutil.rmtree(out)``——**把刚建好的
        ``logs`` 子树一并删掉**，于是随后 ``run_start`` 事件写
        ``{out}/logs/<run_id>/events.jsonl`` 时 ``open(..., "a")`` 抛
        ``FileNotFoundError``（T028 实测：0 个 optimizer step，``exit_code=1``）。

        修法（阻断 B 选项 a）：**顺序对齐**——``train()`` 把
        ``prepare_output_dir`` 提到日志目录创建**之前**（见 ``train()`` 里的
        "顺序对齐"注释），并在它之后用本方法重建日志器。这样：
        - ``overwrite_output_dir: true``（清掉上次产物）语义**不变**：``rmtree``
          仍然删掉整棵输出目录（含上次的 ``logs``），只是本轮的 ``logs`` 在
          ``rmtree`` 之后才建，因此本轮日志正常落盘；
        - ``overwrite_output_dir: false`` 语义**不变**：非空目录仍抛
          ``FileExistsError``（fail-closed），本轮 ``logs`` 只会在空目录上建。
        """
        gf = self.config.native
        return NativeRolloutLogger(
            self.config.training.output_dir,
            readable_enabled=gf.readable_log_enabled,
            raw_enabled=gf.raw_log_enabled,
        )

    # ── 主训练循环 ────────────────────────────────────────────────────────────

    @property
    def adapter(self) -> Any:
        """当前运行时已 setup 的模型适配器（DP 分片等逻辑访问）。

        与 SFT 路径（``self.runtime._adapter``）一致；此属性让 mixin 无需各自
        探测 runtime 的私有字段。
        """
        return self.runtime._require_adapter()

    @property
    def parallel_degree(self) -> Any | None:
        """本进程并行度（``flow/backend/parallel_degree.py::ParallelDegree``）。

        ★ 单一真相源在后端（runtime），**不再**从模型适配器上读 ``dp_size``/
        ``dp_rank``（adapter 一名三义消歧，宪法 §1.4）。取不到 ⇒ ``None``。
        """
        degree = getattr(self.runtime, "parallel_degree", None)
        if degree is not None:
            return degree
        adapter = getattr(self.runtime, "_adapter", None)
        return getattr(adapter, "parallel", None) if adapter is not None else None

    def train(self, *, smoke: bool = False) -> None:
        """GRASPO 训练主入口。

        :param smoke: 冒烟运行边界（运行边界参数，不改变训练语义）——
            运行到首轮 optimize 后停止，**不修改任何配置值**。
        """
        self._smoke_boundary = bool(smoke)
        validate_native_runtime_config(self.config)
        self.runtime.validate()
        # ── 建模前播种（DP 副本初值一致性；与 SFT 路径同一类管道问题）────────────
        #
        # ``runtime.setup()`` 里会构造模型并随机初始化 LoRA ``lora_a``
        # （``kaiming_uniform_``，``lora/lora_linear.py:131``）。DP 只同步
        # **更新量**、不同步初值 ⇒ 各副本若从不同 RNG 状态建模，``lora_a`` 的初值差
        # 会被**永久保留**，训练完各副本仍不一致、产物不可合并。
        #
        # 此前建模发生在**任何播种之前** ⇒ 建模读的是 torch 进程默认种子（取自 OS
        # 熵、逐进程不同）。这里在建模**之前**用**与 rank 无关**的种子播种
        # （``_set_model_init_seed``，与 SFT 路径 `sft_trainer.py:154` 复用**同一个**
        # helper，⛔ 不新增第二份实现——"权重初始化"与"数据打散"是两种语义，
        # 不得共用一个旋钮）。
        #
        # 建模**之后**仍按原逻辑用 ``_set_random_seed(seed, rank=rank)`` 重新播种
        # （本文件 L198）⇒ dropout / 数据打散等逐 rank 随机性行为与改动前**逐位一致**
        # （鲁棒打散另用显式 ``random.Random(seed + epoch)``，本就不受此影响）。
        #
        # ⛔ 本处只改"建模之前的播种"：loss / advantage / 采样 rollout / 优化器实体 /
        #    DP 梯度同步机制一律零改动（见工位 report.md §4 红线证明）。
        _set_model_init_seed(int(self.config.training.seed))
        self.runtime.setup()
        # ── 输出目录身份先定下来（阻断 B 修复：顺序对齐）────────────────────
        #
        # ``__init__`` 构造 ``NativeRolloutLogger`` 时会 ``mkdir`` 出
        # ``{out}/logs/<run_id>/``。旧顺序把它放在 ``prepare_output_dir`` **之前**，
        # 于是 ``overwrite_output_dir: true`` 时 ``prepare_output_dir`` 的
        # ``shutil.rmtree(out)`` 会连本轮刚建好的 ``logs`` 子树一起删掉 ⇒ 随后
        # 所有写 ``logs/<run_id>/events.jsonl`` 的事件（``run_start``、``train_step``）
        # 全部 ``FileNotFoundError``（T028 实测：0 个 optimizer step，``exit_code=1``）。
        #
        # 现在把"清掉上次产物 / 建出空输出目录"提到**任何日志目录创建之前**：
        # - ``overwrite_output_dir: true``：语义不变——仍然清掉上次的全部产物
        #   （含上次的 ``logs``），只是本轮的 ``logs`` 在清完之后才建；
        # - ``overwrite_output_dir: false``：非空目录仍抛 ``FileExistsError``
        #   （fail-closed 防线，行为逐字不变）；
        # - ``resume_from_checkpoint`` 时仍按原逻辑跳过 overwrite（目录必须存在）。
        from graspo.flow.backend.native.trainer.helpers import prepare_output_dir_once

        output_dir = prepare_output_dir_once(
            self.config.training.output_dir,
            overwrite=(
                self.config.training.overwrite_output_dir
                if not self.config.training.resume_from_checkpoint
                else False  # resume 时跳过 overwrite 检查（目录必须存在）
            ),
            is_primary=self._is_primary(),
        )
        # 输出目录此刻为空且稳定 ⇒ 日志目录（``__init__`` 里建的已被 rmtree 掉，
        # 且这是新一次 launch）可以安全地建在同一棵树里。目录名仍是
        # ``{out}/logs/<run_id>/``，**布局不变**。
        self.logger = self._build_rollout_logger()
        # 初始化标准 Python logging 通道
        rank = self.runtime.rank
        setup_logging(self.config.training.output_dir, rank=rank)
        _set_random_seed(int(self.config.training.seed), rank=rank)
        _log = logging.getLogger("graspo.trainer")
        self._print_json(
            {
                "timestamp": _timestamp(),
                "event": "backend_selected",
                "backend": self.backend_name,
                "reason": self.selection.reason if self.selection is not None else "configured",
                "dependency_boundary": (
                    "PyTorch distributed TP/PP only; no NeMo/vLLM/Ray/DeepSpeed/FSDP/DDP/Accelerate"
                ),
                # §10.1：日志不含宿主机路径（config.yaml 备份为唯一真相源），只留文件名
                "model_path": Path(self.config.model.model_path).name,
                "train_path": Path(self.config.data.train_path).name,
                "completion_parser": self._completion_parser_name(),
                "tp_size": self.config.native.tp_size,
            }
        )
        _log.info(
            "Training started: backend=%s model=%s samples=%d",
            self.backend_name,
            self.config.model.model_path,
            self.total_samples,
        )

        samples = load_jsonl(self.config.data.train_path)
        self.total_samples = len(samples)
        # 多模态训练启动预检（防线）：数据含图时验证视觉链路完整、
        # visual LoRA 可训练，失败即拒绝启动，避免 v13 式静默丢图空跑。
        self._preflight_multimodal(samples)
        # ``output_dir`` 已在 ``train()`` 开头准备完毕（见该处的"顺序对齐"注释）：
        # 这里不再第二次调用 ``prepare_output_dir``——第二次调用时输出目录里已有
        # 本轮 ``logs/``，非空 ⇒ 会走进 rmtree 分支，正是阻断 B 的成因。
        self._resume_if_requested()
        # 初始化墙钟时间周期 checkpoint 计时器
        self._last_checkpoint_time = time.monotonic()
        save_period_min = int(self.config.training.save_checkpoint_time_period_minutes)
        if save_period_min > 0:
            _log.info(
                "RL: time-based checkpoint save enabled (every %d min)",
                save_period_min,
            )
        # 将当前配置备份到输出目录，确保可完整复现。resume 时也刷新——
        # 否则输出目录内的 config.yaml 停留在旧配置，误导复现（§1.4 单一真相源）
        _backup_config(self.config, output_dir)
        _log.info(
            "Run config: rollout_group_size=%d micro_batch_size=%d "
            "gradient_accumulation_micro_batches=%d max_epochs=%d max_new_tokens=%d",
            self.config.training.rollout_group_size,
            self.config.native.micro_batch_size,
            self.config.training.gradient_accumulation_micro_batches,
            self.config.training.max_epochs,
            self.config.training.max_new_tokens,
        )
        self._print_json(
            {
                "timestamp": _timestamp(),
                "event": "run_start",
                "backend": self.backend_name,
                "samples_total": self.total_samples,
                "resume": self.resume_info,
                # §1.4 单一真相源：config 全量以落盘备份 config.yaml 为准（§8.5），
                # 事件只引用文件名，不再内嵌人工子集（防子集漂移）
                "config_backup": "config.yaml",
            }
        )

        aborted = False
        try:
            start_epoch = int(self.current_epoch_stats.epoch)
            if (
                self.total_samples
                and int(self.current_epoch_stats.samples_seen) >= self.total_samples
            ):
                start_epoch += 1
                self.current_epoch_stats.epoch = start_epoch
                self.current_epoch_stats.samples_seen = 0
            for epoch in range(start_epoch, self.config.training.max_epochs):
                if epoch != start_epoch or self.current_epoch_stats.samples_seen == 0:
                    self.current_epoch_stats = GraspoFlowEpochStats(epoch=epoch)
                epoch_samples = list(samples)
                # DP 分片（★ β1b：整组不拆 + 按组数均衡；并行度取自 backend，§1.4）。
                # 默认模式的不变量与机核见 `flow/backend/native/assignment.py`；
                # 分配单位是"组"（此处 1 条 prompt = 1 组）⇒ 不存在半个组落在一个 rank。
                parallel = self.parallel_degree
                if parallel is not None and parallel.dp_size > 1:
                    from graspo.flow.backend.native.assignment import (
                        assign_whole_groups,
                        assert_balanced_assignment,
                    )

                    group_keys = list(range(len(epoch_samples)))
                    shard_of = assign_whole_groups(group_keys, parallel.dp_size)
                    assert_balanced_assignment(
                        group_keys, shard_of, n_shards=parallel.dp_size
                    )
                    epoch_samples = [
                        sample
                        for index, sample in enumerate(epoch_samples)
                        if shard_of[index] == parallel.dp_rank
                    ]
                random.Random(int(self.config.training.seed) + epoch).shuffle(epoch_samples)
                resume_sample_offset = (
                    int(self.current_epoch_stats.samples_seen) if epoch == start_epoch else 0
                )
                pending_samples = epoch_samples[resume_sample_offset:]
                queue_size = max(1, int(self.config.training.rollout_queue_batch_size))
                # ★ U-A：dp ≥ 2 时走"固定轮数 + 每轮补门"的对齐路径（见
                #   `_run_aligned_sample_queues` 的 docstring 与工位 plan.md §1）。
                if self._world_gate_alignment_enabled():
                    # ⚠ 本分支**不做** `if not pending_samples: continue`：本地无样本的
                    #   rank 也必须进循环（以空队列参与轮次判定与补门），否则 WORLD
                    #   集合逐 rank 错位 —— 那正是本单元要修的形态。
                    if self._run_aligned_sample_queues(
                        pending_samples, epoch=epoch, queue_size=queue_size
                    ):
                        # smoke 提前终止：先 flush replay buffer 再保存，
                        # 否则 _checkpoint_trainer_state 会因 buffer 非空而
                        # 拒绝保存（防线生效，但提前终止路径漏了 flush）。
                        # force 无条件调用：空的 dp_rank 也参与 WORLD 集结算子，
                        # 避免其在 teardown barrier 上等待仍在训练的 dp_rank。
                        self._maybe_optimize(epoch=epoch, force=True)
                        self._save_checkpoint(output_dir / "final", epoch=epoch)
                        return
                else:
                    # ── dp ≤ 1：原路径**逐字保留**（A-1 回归锚）─────────────────────
                    #   `_world_gate_alignment_enabled()` 在 dp_size ≤ 1（含单卡与
                    #   dp==1 的 PP 档）时恒 False ⇒ 本分支就是改动前的代码，零新增
                    #   集合、行为逐位不变（见工位 plan.md §1.5 与 A-1b 文本级对照）。
                    if not pending_samples:
                        continue
                    for start in range(0, len(pending_samples), queue_size):
                        sample_queue = pending_samples[start : start + queue_size]
                        if self._sample_queue(sample_queue, epoch=epoch):
                            # smoke 提前终止：先 flush replay buffer 再保存，
                            # 否则 _checkpoint_trainer_state 会因 buffer 非空而
                            # 拒绝保存（防线生效，但提前终止路径漏了 flush）。
                            # force 无条件调用：空的 dp_rank 也参与 WORLD 集结算子，
                            # 避免其在 teardown barrier 上等待仍在训练的 dp_rank。
                            self._maybe_optimize(epoch=epoch, force=True)
                            self._save_checkpoint(output_dir / "final", epoch=epoch)
                            return
                self._print_json(
                    {
                        "timestamp": _timestamp(),
                        "event": "epoch_summary",
                        "epoch": self.current_epoch_stats.epoch,
                        "elapsed_sec": round(time.monotonic() - self.started_at, 3),
                        "epoch_cumulative": self._epoch_summary(),
                        "run_cumulative": self._run_summary(),
                    }
                )
                _log.info(
                    "Epoch %d/%d finished: elapsed=%.1fs samples_seen=%d",
                    epoch + 1,
                    self.config.training.max_epochs,
                    time.monotonic() - self.started_at,
                    self.current_epoch_stats.samples_seen,
                )
                if self.config.training.save_checkpoint_every_epoch:
                    self._maybe_optimize(epoch=epoch, force=True)
                    self._save_checkpoint(output_dir / f"epoch_{epoch}", epoch=epoch)
                    self._last_checkpoint_time = time.monotonic()
            self._maybe_optimize(epoch=self.config.training.max_epochs - 1, force=True)
            self._save_checkpoint(output_dir / "final", epoch=self.config.training.max_epochs - 1)
            self._last_checkpoint_time = time.monotonic()
        except BaseException:
            # 异常退出路径：**不做任何集合对齐**，直接 abort 本 rank 的进程组后把首个
            # 异常原样抛出。为什么必须如此（2026-09-22 T035 实测，pp_size=2）：native PP
            # 下某一 stage 抛错时，另一 stage 通常正阻塞在 PP P2P / pp_group 集合上——
            # 退出路径上的 WORLD barrier 永远等不到对端，只能等满 600s
            # ``nccl_collective_timeout``，把**首个异常**掩盖成
            # ``c10::DistBackendError`` + SIGABRT（首错取证链断裂）。
            # 详见 :meth:`_abort_distributed` 与 :meth:`_teardown_after_train_loop`。
            aborted = True
            self._abort_distributed(aborted=aborted)
            raise
        finally:
            # 正常退出（含 ``return``，如 smoke 边界）：保留原语义，先对齐再 teardown。
            if not aborted:
                self._teardown_after_train_loop()

    def _teardown_after_train_loop(self) -> None:
        """训练循环**正常退出**路径的收口：先 WORLD 对齐，再销毁进程组。

        防呆（§2.1）原语义，逐字保留：不同 dp_rank 的 rollout 相互独立、跑速不同，
        最快/最慢 rank 需在销毁进程组前收敛——``self.runtime.close()`` 内部的
        ``destroy_parallel_state``（``flow/backend/native/parallel/state.py``）本身也带一道 WORLD
        barrier，两者同源同义（因此本方法在 pp_size==1 与既有 46 档上的行为不变）。
        """
        if dist.is_available() and dist.is_initialized():
            dist.barrier()
        self.runtime.close()

    def _abort_distributed(self, *, aborted: bool) -> None:
        """训练循环**异常退出**路径的收口：先记 WARNING，再非阻塞销毁进程组。

        **为什么不走 ``self.runtime.close()``**：它内部的 ``destroy_parallel_state``
        会再做一次 WORLD barrier（``state.py:273``）——在"一侧 stage 已失败、另一侧
        stage 仍阻塞在 pipeline 集合里"时同样等不到对端，会把首错再一次掩盖成 600s
        NCCL 超时。异常路径只做 NCCL abort 语义的 ``destroy_process_group()``：
        对端阻塞中的集合会立刻失败，两端一起快速退出，首个异常的完整 Python 栈得以
        打印，torchrun 随即回收其余 rank。

        **abort 前必须留下一条 WARNING（可观测性，2026-09-22 裁定 1）**：这条日志是
        "本 rank 是否进入了异常退出路径"的**直接判据**。实测教训（T035，GPU0/1 单档）：
        当时没有这条日志，只能靠"rank1 的 3 个 NCCL 通信组是否仍存活"**间接反推**它
        从未进入本方法——错误可见性本身就是可靠性的一部分。

        :param aborted: ``train()`` 的退出标志（调用点恒为 ``True``）；显式传入使日志
            自带该标志，便于事后按日志重建判定链。
        :raises: 不抛出——清理失败只记 WARNING（异常路径的二次失败绝不能替换首错）。
        """
        if not (dist.is_available() and dist.is_initialized()):
            return
        try:
            # ``self.runtime`` 是鸭子类型（单测用 fake）；诊断取数失败绝不能替换首错，
            # 故此处显式兜住取数异常并退化为 "?"（§2.2：不用 hasattr 探测业务接口）。
            rank: object = self.runtime.rank
        except Exception:  # noqa: BLE001 —— 诊断取数失败不得掩盖正在传播的首错
            rank = "?"
        logging.getLogger("graspo.trainer").warning(
            "exception-path teardown: aborted=%s rank=%s group=%s(WORLD); aborting the "
            "default process group WITHOUT a rendezvous (no WORLD barrier on the failure "
            "path) — the first error is re-raised as-is",
            bool(aborted),
            rank,
            _DEFAULT_PG_NAME,
        )
        try:
            dist.destroy_process_group()
        except Exception as exc:  # noqa: BLE001 —— 见 docstring：不得替换正在传播的首错
            logging.getLogger("graspo.trainer").warning(
                "exception-path teardown: destroy_process_group() failed and was ignored "
                "so the original error is preserved: %s",
                exc,
            )

    # ── 样本队列调度 ──────────────────────────────────────────────────────────

    def _sample_one(self, sample: Any, *, epoch: int) -> bool:
        """处理单个样本（仅用于测试）。"""
        return self._sample_queue([sample], epoch=epoch)

    def _collective_any_pending(self, n_pending: int) -> bool:
        """跨 rank 一致的“是否还有待重试样本”判定（**WORLD `all_reduce(MAX)` 标量**）。

        为什么必须集合化（2026-10-07 探针实测）：`should_retry` 是各 rank 本地判断
        （各 rank 数据分片不同 ⇒ 重试需求天然不同），而重试循环之后紧跟
        `default_pg(WORLD)` 的集合（`optimize.py:59` 的 `want`）⇒ **轮数不一致**
        ⇒ 先跑完的 rank 在上游 WORLD 集合上死等。实测（T041）：rank0 只 1 轮、
        rank1 跑 5 轮；两者进入同一个 `all_reduce(MAX, numel=1)` 错开 **3 分 23 秒**；
        修复后 **9 秒**（`native-restore/REPORT.md` §13.2）。

        为什么用 WORLD 而不是 `dp_group`：下游要等的那个集合就在 `default_pg` 上，
        必须让**同一通信域内的所有 rank** 对齐轮次。
        ⛔ 只读一个标量、只做 MAX，不改变任何判定语义。
        """
        import torch
        import torch.distributed as _dist

        if not (_dist.is_available() and _dist.is_initialized()):
            return n_pending > 0
        adapter = self.runtime._require_adapter()
        # ★ 守卫（2026-10-08 追加，指挥部裁决 2）：与 SFT 先例 `sft_trainer.py:255` 的
        #   `adapter.dp_size > 1` 同款判据。本战役授权范围 = T041/T042（dp=2/4）；
        #   dp_size == 1（T035 tp1/dp1/pp2、T036 tp1/dp1/pp4，world_size>1）时**不得**新增集合
        #   —— 那会落在 PP 线上，越出「不动 PP 线」的边界，且本机无法证伪。
        #   ★ dp ≥ 2 时本守卫**恒假** ⇒ T041/T042 行为逐字不变 ⇒ 在案消融（retryfix2/3）结论仍适用。
        tp_state = getattr(adapter, "tp_state", None)
        dp_size = int(getattr(tp_state, "dp_size", getattr(adapter, "dp_size", 1)))
        if dp_size <= 1:
            return n_pending > 0
        device = adapter.device
        flag = torch.tensor([1 if n_pending > 0 else 0], dtype=torch.int, device=device)
        try:
            _dist.all_reduce(flag, op=_dist.ReduceOp.MAX)
        except Exception as exc:  # noqa: BLE001 — 集合失败必须可见，绝不静默
            raise RuntimeError(f"retry-round collective failed: {type(exc).__name__}: {exc}") from exc
        return bool(flag.item())

    # ── ★ U-A：WORLD 门（`_maybe_optimize`）节奏对齐 ───────────────────────────

    def _world_gate_alignment_enabled(self) -> bool:
        """本 rank 是否走「WORLD 门节奏对齐」路径（U-A）。

        **为什么需要**：`_maybe_optimize` 在 `local_wants_train` 判定**之前**无条件发一次
        WORLD `all_reduce(MAX)`（`optimize.py:75-82`，无 `group=` ⇒ `default_pg`，与 K-会合
        的 `dp_group` **不是同一个 communicator**），而它是**每终结一条 prompt 调用一次**
        （`trainer.py` 的 `for state in finished:` → `_finalize_sample` → `rollout.py:434`）。
        各 rank 分到的 prompt 数可差 1（`assign_whole_groups` 只保证**组数**差 ≤ 1，
        `assignment.py:104-130`）⇒ 门次数不等 ⇒ 集合调用次数不等 ⇒ **静默自旋**（无超时兜底）。

        **判据 = 「进程组已初始化 ∧ dp_size > 1」**，与**分片判据同源**（`train()` 里只有
        `parallel.dp_size > 1` 才切分片）：不切分片 ⇒ 逐 rank prompt 数必然相同 ⇒ 无需对齐。
        `dp_size <= 1`（含单卡与 `dp==1` 的 PP 档）⇒ **恒 False** ⇒ 走原路径、**零新增集合**
        —— 与 `_collective_any_pending`（本类上方）的短路口径同族，⛔ 不新增 PP 线上的集合。
        """
        if not (dist.is_available() and dist.is_initialized()):
            return False
        degree = self.parallel_degree
        return degree is not None and int(degree.dp_size) > 1

    def _collective_any_queues_pending(self, has_more_local: bool) -> bool:
        """跨 rank 一致的「是否还有下一个样本队列」判定（**WORLD `all_reduce(MAX)` 标量**）。

        与 `_collective_any_pending`（重试轮次）**同款形制、同一通信域、同一短路口径**，
        只换了语义：那里问"还有待重试样本吗"，这里问"还有下一轮队列吗"。

        **为什么需要**：队列数 `ceil(len(分片)/Q)` 逐 rank 可不等（分片长度可差 1）⇒ 轮数不等
        ⇒ 循环体内每一个 WORLD 集合（`_collective_any_pending`、WORLD 门、以及循环之后
        `_teardown_after_train_loop` 的 barrier）**逐 rank 错位**。修法照抄既有 `go_on` 模式：
        每轮末尾做一次标量 MAX 一致判定，任一 rank 还有队列 ⇒ **所有** rank 都进下一轮
        （本地无队列的 rank 以**空队列**参与，见 `_run_aligned_sample_queues`）。

        ⛔ 只读一个标量、只做 MAX，不改变任何判定语义。
        """
        import torch
        import torch.distributed as _dist

        if not (_dist.is_available() and _dist.is_initialized()):
            return has_more_local
        degree = self.parallel_degree
        if degree is None or int(degree.dp_size) <= 1:
            return has_more_local
        device = self.runtime._require_adapter().device
        flag = torch.tensor([1 if has_more_local else 0], dtype=torch.int, device=device)
        try:
            _dist.all_reduce(flag, op=_dist.ReduceOp.MAX)
        except Exception as exc:  # noqa: BLE001 — 集合失败必须可见，绝不静默
            raise RuntimeError(
                f"queue-round collective failed: {type(exc).__name__}: {exc}"
            ) from exc
        return bool(flag.item())

    def _run_aligned_sample_queues(
        self, pending_samples: list[Any], *, epoch: int, queue_size: int
    ) -> bool:
        """dp≥2：把队列循环跑成**固定轮数**，并把每轮的 WORLD 门补齐到 `queue_size` 次（U-A）。

        **机制**（真值与逐行证据见工位 `plan.md` §0/§1）：门次数 = 本 rank 终结的 prompt 数；
        分片长度可差 1 ⇒ 门次数不等 ⇒ 集合调用次数不等 ⇒ 静默自旋。本方法（= 设计件
        §8-U-A 的**候选 (a)**）只做两件**节奏**上的事，⛔ 不动门本体（`optimize.py`）、
        ⛔ 不动门的调用点（`rollout.py:434`）：

        ① **轮数对齐**：轮数由 `_collective_any_queues_pending` 的 `go_on` 决定（= 全局 max
           队列数）；本地已无队列的 rank 以**空队列**参与（`_sample_queue([])` 只走那次轮次
           判定，不发门）。
        ② **每轮门数补齐到 `queue_size`**：`_sample_queue` 本轮实际发 `len(queue)` 次**真门**
           （= 本轮终结 prompt 数，见 `_sample_queue` 收尾的 fail-closed 校验），此处再补
           `queue_size - len(queue)` 次**空门**（`_maybe_optimize`：本地缓冲不达阈值时只是
           参与那次 MAX 归约，不训练）。

        ⇒ 每个 rank 每轮恰好 `queue_size` 次门、每 epoch 恰好 `轮数 × queue_size` 次门
        ⇒ **逐 rank 次数相等**；且每次门的布尔由**同一次** WORLD MAX 归约产生
        ⇒ `global_step` / `optimized_steps` **逐 rank 一致**（`optimize.py:134-135`）。

        **空门为什么安全**：空门就是同一个门，语义零变化；它与真门的差别只是"发它时本 rank
        缓冲有多少条"，而这正是**既有**语义（任一 rank 达阈值 ⇒ 所有 rank 一起进 `train_batch`，
        含本地缓冲为空的 rank，见 `optimize.py:60-92`）。等分片时（既有一切能跑通的 dp≥2 档）
        各 rank 每轮终结数相同 ⇒ 空门只与空门配对、之间不新增样本 ⇒ 布尔与最后一次真门相同
        ⇒ **空门恒为 no-op**；满轮（`len(queue)==queue_size`）时补 0 次 ⇒ 逐位不变。

        ⛔ **禁止**改成"按本 rank 应欠的步数发门"：那会让 `global_step`/`optimized_steps`
           逐 rank 分叉，污染 time-based checkpoint 门（`optimize.py:148-155`）、周期保存与
           smoke 边界。本方法发的是**同一个固定次数**，不是"各 rank 各自的欠数"。
        ⛔ **不改算法**：advantage / loss / clip / 决策 / 采样 / 阈值 / epoch / lr 全不动；
           本方法只决定**门发几次**。

        :returns: 是否触发 smoke 提前终止（与 `_sample_queue` 的返回值同义）。
        """
        rounds = 0
        go_on = True
        while go_on:
            rounds += 1
            sample_queue = pending_samples[(rounds - 1) * queue_size : rounds * queue_size]
            stop_requested = self._sample_queue(sample_queue, epoch=epoch)
            # ★ 补门：本轮真门数 = len(sample_queue)（每条 prompt 恰好终结一次，见
            #   `_sample_queue` 收尾的 fail-closed 校验）⇒ 补齐到 queue_size 次。
            for _ in range(queue_size - len(sample_queue)):
                self._maybe_optimize(epoch=epoch)
            if stop_requested:
                return True
            go_on = self._collective_any_queues_pending(
                len(pending_samples) > rounds * queue_size
            )
        return False

    def _sample_queue(self, samples: list[Any], *, epoch: int) -> bool:
        """处理一批样本：rollout → 评分 → 重试 → 最终化。"""
        active = [QueuedSample(sample=sample) for sample in samples]
        finished: list[QueuedSample] = []
        max_attempts = self.config.training.rollout_max_retries + 1
        # ★ 重试轮数跨 rank 一致化（2026-10-08，T041/T042 修法・候选 C；与候选 A **同批落源树**）：
        #   实测机制见 `_collective_any_pending` 的 docstring；修法 = 每轮末尾做一次
        #   WORLD `all_reduce(MAX)` 标量一致判定；任一 rank 还要重试 ⇒ 所有 rank 都进下一轮
        #   （本地无可重试样本的 rank 以空 `active` 参与，只走那次判定并被阻塞到对端到达）。
        #   ⛔ 不改 `should_retry` 语义、不改 `max_attempts`、不改采样域与阈值。
        #   ⚠ 已知残余：`retry_count` 仍在本 rank 累加 ⇒ 若某 rank 先撞 `max_attempts`，
        #     它在 `:381-382` 抛 RuntimeError 时其它 rank 正阻塞在轮次集合上 ⇒ **首错仍以
        #     "集合超时"形态出现，直到 `_abort_distributed` 的 `destroy_process_group()` 生效**
        #     （既有路径未变，非新增失败面；须在回执中登记，防被误读成"C 没生效"）。
        go_on = True
        _round = 0
        while active or go_on:
            _round += 1
            next_active: list[QueuedSample] = []
            if active:
                attempt_records = self._rollout_queue_attempt(active, epoch=epoch)
                for state, record in zip(active, attempt_records, strict=True):
                    if record.decision.should_retry:
                        state.attempts.append(record)
                        state.retry_count += 1
                        if state.retry_count >= max_attempts:
                            raise RuntimeError("GRASPO retry state exceeded configured max attempts")
                        next_active.append(state)
                    else:
                        state.attempts.append(record)
                        finished.append(state)
            go_on = self._collective_any_pending(len(next_active))
            # ★ 只读探针（诊断跑，与生成器版本的 round-collective 探针同风格 ⇒ 可对账）。
            #   取 rank 用**模块级** `dist`（= `torch.distributed`，本文件 `:15` 已导入）的
            #   `dist.get_rank()`（硬规矩 15：⛔ 不得用分维度 rank 字段冒充全局 rank）。
            if dist.is_available() and dist.is_initialized():
                print(
                    f"[DPSYNC] site=retry_round rank={dist.get_rank()} "
                    f"t={time.monotonic():.3f} round={_round} "
                    f"n_pending={len(next_active)} go_on={1 if go_on else 0}",
                    flush=True,
                )
            active = next_active

        # ★ fail-closed 不变式（U-A 依托，**放在任何门之前**）：本队列**每条** prompt
        #   恰好终结一次 ⇒ 本轮真门次数 == `len(samples)`。`_run_aligned_sample_queues`
        #   正是按这个数推算"要补几次空门"；若该不变式被破坏，补门次数逐 rank 就会不同
        #   ⇒ WORLD 集合错位 ⇒ 又回到静默自旋。故此处**显式失败**而不是继续发门：
        #   `finished` 在重试循环结束时已经定型（下面的终结循环只迭代它，不再增删），
        #   所以这个校验能在**发出第一个门之前**判红。
        #   ⛔ 纯本地判断、零集合；不改任何判定语义（dp ≤ 1 路径同样不受影响）。
        if len(finished) != len(samples):
            raise RuntimeError(
                "sample queue finished-count mismatch: "
                f"finished={len(finished)} queue={len(samples)} — WORLD 门节奏对齐按 "
                "len(queue) 推算补门次数，此处不等会使 WORLD 集合逐 rank 错位（静默自旋）"
            )
        stop_requested = False
        for state in finished:
            if self._finalize_sample(state, epoch=epoch):
                stop_requested = True
        return stop_requested

    # ── 日志输出辅助 ──────────────────────────────────────────────────────────

    def _preflight_multimodal(self, samples: list[Any]) -> None:
        """多模态训练启动预检（防线）。

        数据含图时验证 encode → attach → resolve 链路完整、visual LoRA
        可训练（fake 1-step 前向梯度非零），失败即拒绝启动训练——
        避免 v13 式的静默丢图空跑 19.5 小时。纯文本数据直接跳过。

        **两道防线，前一道不碰 GPU**：先按配置判定"视觉塔是否有可训载体"
        （``assert_lora_vision_targets_trainable``，纯逻辑，报错直接点名该改哪个键；
        判据按 ``tuner_type`` 分派——全参下不存在 LoRA 矩阵，该判据不适用），
        再走需要真机的前向预检。
        """
        from graspo.flow.backend.native.trainer.preflight import (
            assert_lora_vision_targets_trainable,
            run_multimodal_preflight,
        )

        model = self.runtime._require_adapter()
        model_config = getattr(getattr(model, "model", None), "config", None)
        image_token_id = getattr(model_config, "image_token_id", None)
        if image_token_id is not None and self._is_primary():
            _log = logging.getLogger("graspo.trainer")
            _log.info(
                "multimodal training detected (image_token_id=%s); running "
                "visual-link preflight before training starts",
                image_token_id,
            )
        # 配置期防线（§2.3）：视觉模型 + 语言-only 的 LoRA 目标 = 视觉塔永远冻结，
        # 运行期预检也会拦，但那时模型已加载完。这里提前到不触 GPU 的判定。
        # ★ 判据按 ``tuner_type`` 分派（全参下不存在 LoRA 矩阵 ⇒ 该判据不适用，
        #   同一风险点由下方 run_multimodal_preflight 的可训视觉参数/梯度判据覆盖）。
        assert_lora_vision_targets_trainable(
            lora_target_modules=self.config.lora.target_modules,
            lora_target_preset=self.config.lora.target_preset,
            image_token_id=image_token_id,
            model_name=str(self.config.model.model_path),
            tuner_type=self.config.effective_tuner_type,
        )
        run_multimodal_preflight(
            self.runtime,
            samples,
            data_dir=str(Path(self.config.data.train_path).parent),
            image_token_id=image_token_id,
            model_name=str(self.config.model.model_path),
            pp_size=int(self.config.native.pp_size),
        )

    def _print_json(self, payload: dict[str, Any]) -> None:
        """主 rank 输出结构化 JSON 事件（§13.3 易读/原始分离）。

        落盘到 ``logs/events.jsonl``（机器分析通道），同时打印到 stdout
        （部署端 nohup.out 兼容）；training.log 只保留人类可读文本。
        """
        if self._is_primary():
            self.logger.write_event(payload)
            print(json.dumps(payload, ensure_ascii=False), flush=True)

    def _is_primary(self) -> bool:
        """判断当前 rank 是否为主 rank（负责日志 I/O）。"""
        return self.runtime.is_primary()

    def _timestamp(self) -> str:
        """返回当前时区的 ISO 格式时间戳。"""
        return _timestamp()

    def _timing_event(
        self,
        *,
        phase: str,
        duration_sec: float,
        epoch: int,
        details: dict[str, Any],
        sample_index: int | None = None,
        attempt_number: int | None = None,
        retry_count: int | None = None,
    ) -> dict[str, Any]:
        """构建 timing 事件记录。"""
        return {
            "timestamp": _timestamp(),
            "elapsed_sec": round(time.monotonic() - self.started_at, 6),
            "phase": phase,
            "duration_sec": round(float(duration_sec), 6),
            "step": self.global_step,
            "epoch": epoch,
            "sample_index": sample_index,
            "attempt_number": attempt_number,
            "retry_count": retry_count,
            "rank": self.runtime.rank,
            "tp_rank": self.runtime.tp_rank,
            "details": round_timing_details(details),
        }
