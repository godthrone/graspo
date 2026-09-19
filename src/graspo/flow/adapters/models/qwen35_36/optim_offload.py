"""native 全参训练 — 优化器态 CPU offload（**默认关闭**，显式开启才生效）。

为什么需要（实测，不是算式）
---------------------------
``T016``（9B · SFT · 全参 · native · 1 卡）实测在 ``optimizer.step()`` 处 ①真
OOM：PyTorch 已分配 **78.23 GiB** / 进程占用 79.23 GiB / 卡容量 79.25 GiB。
显存墙在**优化器态**：

- 参数(bf16) 2 B/参数 + 梯度(bf16) 2 B/参数 = **35.14 GiB**（与实测逐位对上的
  ``after_backward=35.14GB``）；
- ``step()`` 首次调用时 ``torch.optim.AdamW`` 用 ``torch.zeros_like(param)`` 建
  两份矩估计 ⇒ bf16 下 4 B/参数 ≈ **35.14 GiB**——**这正好是爆点**。

⇒ 省显存必须在优化器态上做，不是在激活上做（8K 档与 4K 档是同一面墙）。

机制（与 ms-swift 侧 ``zero2_offload`` 同口径：优化器态常驻 CPU）
---------------------------------------------------------------
``CpuOffloadedAdamW`` **继承** ``torch.optim.AdamW``，把 ``param_groups`` 里的
参数换成 CPU 上的镜像。于是：

- AdamW 的数学**一处不重写**（``step`` 直接走 ``super()``）；
- ``state_dict`` / ``load_state_dict`` / ``param_groups`` 语义原样保留；
- ``torch.optim.lr_scheduler.LRScheduler`` 的
  ``isinstance(optimizer, Optimizer)`` 契约仍然成立（这正是"不用薄包装对象、
  必须真继承 Optimizer"的原因——薄包装会在 cosine/linear 调度器上直接 TypeError）；
- ``TransformerAdapter.load_checkpoint`` 的
  ``_normalize_optimizer_state_to_params`` 也自然正确：它按 **optimizer 自己的
  参数** 的 device 归一，而这里的参数就是 CPU 镜像 ⇒ 状态留在 CPU，不会被
  悄悄搬回 GPU（若用薄包装，状态会被强行搬到参数 device 上，offload 静默失效）；
- **resume 可用**：``load_checkpoint`` 在 ``_build_optimizer`` 之后把 checkpoint
  权重 load 进 GPU 参数，镜像那份是初始化快照 ⇒ 首次 ``step()`` 前强制对齐一次
  （``_align_mirrors_with_gpu_params``），否则第一次 AdamW 会用旧权重并把旧权重
  写回 GPU（静默训错）。

每次 ``step()`` 的搬运：GPU 梯度 →（pinned 缓冲）CPU 镜像 → ``super().step()``
→ 更新后的镜像 → GPU 参数。GPU 侧因此只剩 参数 + 梯度。

代价与前提（必须显式预授权，宪法 §3.3）
---------------------------------------
- **必然变慢**：每 step 约 35 GiB PCIe 流量（D2H 梯度 + H2D 参数）。"变慢"
  不构成失败，但必须由用户在配置里预先声明接受 ⇒ 由
  ``native.offload_optimizer_state``（默认 false）承载。
- **数值差异**：同一套 AdamW 公式，但在 CPU 上以参数 dtype 运行而不是 CUDA
  kernel ⇒ 舍入路径不同，loss 轨迹会有微小漂移（不是"错误"，但不是逐位一致）。
- **CPU 内存**：镜像 + 梯度缓冲 + 一阶/二阶矩 ≈ 8 B/参数（bf16），9.4B 参数
  约 **70 GiB** 常驻且其中一半是 pinned（不可换页）。开工前做一次显存/内存
  预算校验，不够就当场报错（§2.3），不要跑到一半被 OOM killer 杀。
"""

from __future__ import annotations

import logging
from typing import Any

import torch

logger = logging.getLogger("graspo.flow")

__all__ = [
    "CpuOffloadedAdamW",
    "build_cpu_offloaded_adamw",
    "estimate_cpu_offload_bytes",
    "available_cpu_memory_bytes",
]


def estimate_cpu_offload_bytes(params: list[torch.Tensor]) -> int:
    """纯算式：这套 offload 需要多少 CPU 常驻字节（不分配任何内存）。

    逐项：参数镜像 1 份 + 梯度 pinned 缓冲 1 份 + AdamW 一阶矩 + 二阶矩。
    ``torch.optim.AdamW`` 用 ``torch.zeros_like(param)`` 建矩估计，因此矩与
    参数同 dtype、同 size ⇒ 每项都是 ``numel × element_size``。
    """
    total = 0
    for param in params:
        per_copy = int(param.numel()) * int(param.element_size())
        total += per_copy * 4  # 镜像 + 梯度缓冲 + exp_avg + exp_avg_sq
    return total


def available_cpu_memory_bytes() -> int | None:
    """``/proc/meminfo`` 的 ``MemAvailable``（字节）。取不到时返回 ``None``。

    返回 ``None`` 表示"无法判定"，调用方按"不阻塞"处理——本函数是防呆的
    加强项，不是正确性依赖；拿不到读数时不应该把训练挡住。
    """
    try:
        with open("/proc/meminfo", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024
    except (OSError, ValueError, IndexError):
        return None
    return None


class CpuOffloadedAdamW(torch.optim.AdamW):
    """``torch.optim.AdamW`` + 参数/状态常驻 CPU、每步流式换入换出。

    只覆盖三件事，AdamW 本体（``step`` 的数学、``state_dict`` 的格式、
    ``param_groups`` 的语义）一律继承：

    1. ``__init__``：建 CPU 镜像参数，按镜像建 AdamW（状态因此天生在 CPU）；
    2. ``step``：梯度 D2H → ``super().step()`` → 参数 H2D；
    3. ``zero_grad``：同时清 GPU 侧梯度（模型侧张量），否则梯度会跨步累积。
    """

    def __init__(
        self,
        gpu_params: Any,
        *,
        pin_memory: bool = True,
        **adamw_kwargs: Any,
    ) -> None:
        params = list(gpu_params)
        if not params:
            raise ValueError(
                "CpuOffloadedAdamW needs at least one trainable parameter; got 0. "
                "An empty offloaded optimizer would silently train nothing."
            )
        devices = {str(param.device) for param in params}
        if len(devices) != 1 or not next(iter(devices)).startswith("cuda"):
            raise RuntimeError(
                "CpuOffloadedAdamW is only meaningful for CUDA parameters "
                f"(all on one device); got devices={sorted(devices)}. "
                "Without a GPU there is nothing to offload from."
            )
        if any(not param.is_floating_point() for param in params):
            raise TypeError(
                "CpuOffloadedAdamW requires floating-point parameters "
                "(AdamW is not defined for integer/bool tensors)."
            )
        use_pin = bool(pin_memory) and torch.cuda.is_available()
        mirrors = [
            torch.empty_like(param, device="cpu", pin_memory=use_pin) for param in params
        ]
        with torch.no_grad():
            for mirror, param in zip(mirrors, params):
                mirror.copy_(param.detach())
        # ``foreach=False``：显式关掉 torch 的批量实现，免得 step() 内部再建
        # 与"整模型同级"的临时张量列表——那正是本 offload 要消掉的东西，
        # 而且显式固定下来也让 CPU 侧的算子选择可复现（宪法 §6）。
        adamw_kwargs.setdefault("foreach", False)
        super().__init__(mirrors, **adamw_kwargs)
        self._gpu_params: list[torch.Tensor] = params
        self._mirrors: list[torch.Tensor] = mirrors
        self._grad_buffers: list[torch.Tensor] = [
            torch.empty_like(mirror, device="cpu", pin_memory=use_pin) for mirror in mirrors
        ]
        # 第一次 step 之前必须把镜像与 GPU 参数**重新对齐**一次：建优化器
        # （setup → _build_optimizer）之后，``load_checkpoint`` 还会把 checkpoint
        # 里的权重 load 进 GPU 参数（``full_param_state_dict``）。镜像那份是**初始化
        # 时刻**的快照，若不重同步，第一次 optimizer.step 就会拿旧权重算 AdamW，
        # 并把旧权重写回 GPU——resume 直接静默训错。只在首次做，不每步做
        # （每步多搬 17.6 GiB 只为对齐一个不会自己变的值，不值得）。
        self._mirrors_aligned = False
        logger.warning(
            "native optimizer-state offload ENABLED (native.offload_optimizer_state=true): "
            "%d trainable tensors, %d params, ~%.1f GiB CPU residency "
            "(mirrors + pinned grad buffers + AdamW moments), pinned=%s. "
            "Optimizer state stays on CPU and is streamed per step — expect a "
            "significant per-step slowdown (this is a pre-authorized retreat, §3.3).",
            len(params),
            sum(int(p.numel()) for p in params),
            estimate_cpu_offload_bytes(params) / (1024**3),
            use_pin,
        )

    @property
    def gpu_params(self) -> list[torch.Tensor]:
        """被 offload 的 GPU 侧参数（只读视图，供报告与测试用）。"""
        return list(self._gpu_params)

    def zero_grad(self, set_to_none: bool = True) -> None:  # noqa: FBT001,FBT002 - 与 torch 基类签名一致
        """清梯度：GPU 侧（模型持有）与 CPU 镜像侧都要清。

        否则 GPU 侧梯度会跨 step 累积（镜像侧即使清了也没用——下一步会把
        累积后的 GPU 梯度整份拷下来）。两侧的 ``set_to_none`` 语义保持一致。
        """
        for gpu_param in self._gpu_params:
            grad = gpu_param.grad
            if grad is None:
                continue
            if set_to_none:
                gpu_param.grad = None
            else:
                grad.detach_()
                grad.zero_()
        for mirror in self._mirrors:
            grad = mirror.grad
            if grad is None:
                continue
            if set_to_none:
                mirror.grad = None
            else:
                grad.detach_()
                grad.zero_()

    @torch.no_grad()
    def _align_mirrors_with_gpu_params(self) -> None:
        """把 CPU 镜像与 GPU 参数的取值对齐一次（首次 ``step()`` 前）。

        时间是关键：``_build_optimizer`` 之后，``load_checkpoint`` 仍会把 checkpoint
        权重 load 进 GPU 参数。镜像若是初始化时刻的快照，第一次 ``step()`` 就会用旧
        权重做 AdamW 并把旧权重写回 GPU（resume 静默训错）。
        """
        for mirror, gpu_param in zip(self._mirrors, self._gpu_params):
            mirror.copy_(gpu_param.detach())
        self._mirrors_aligned = True

    @torch.no_grad()
    def step(self, closure: Any = None) -> Any:
        """梯度 D2H → AdamW 更新（CPU）→ 参数 H2D。数学全部由基类完成。"""
        if not self._mirrors_aligned:
            self._align_mirrors_with_gpu_params()
        device = self._gpu_params[0].device
        for mirror, buffer, gpu_param in zip(
            self._mirrors, self._grad_buffers, self._gpu_params
        ):
            grad = gpu_param.grad
            if grad is None:
                mirror.grad = None
                continue
            if grad.shape != buffer.shape or grad.dtype != buffer.dtype:
                # 防呆（§2.3）：形状/dtype 变了说明模型布局与建优化器时不一致，
                # 静默 reshape 会让"搬错张量"变成数值事故。
                raise RuntimeError(
                    "Gradient does not match the offloaded parameter buffer: "
                    f"grad={tuple(grad.shape)}/{grad.dtype} "
                    f"param={tuple(buffer.shape)}/{buffer.dtype}. "
                    "The model layout changed after the optimizer was built."
                )
            buffer.copy_(grad.detach(), non_blocking=True)
            mirror.grad = buffer
        # 非阻塞 D2H 必须先同步，否则 CPU 侧 AdamW 会读到还没落地的字节。
        torch.cuda.synchronize(device)
        result = super().step(closure)
        for mirror, gpu_param in zip(self._mirrors, self._gpu_params):
            gpu_param.copy_(mirror, non_blocking=True)
        return result


def build_cpu_offloaded_adamw(optimizer: Any) -> CpuOffloadedAdamW:
    """把已经建好的 ``torch.optim.AdamW`` 换成 CPU-offload 变体。

    **超参一律从既有优化器的 ``param_groups`` 读**，不重新读配置——建优化器
    的收参与超参装配逻辑保持单一真相源（§1.4），这里只做"换存储位置"。
    """
    groups = list(getattr(optimizer, "param_groups", []) or [])
    if len(groups) != 1:
        raise RuntimeError(
            "build_cpu_offloaded_adamw expects exactly one param group "
            f"(the native adapter builds AdamW with a single group); got {len(groups)}. "
            "Refusing to guess how to merge per-group hyperparameters."
        )
    group = groups[0]
    if not isinstance(optimizer, torch.optim.AdamW):
        raise TypeError(
            "build_cpu_offloaded_adamw only supports torch.optim.AdamW "
            f"(the native adapter's optimizer); got {type(optimizer).__name__}."
        )
    params = [param for param in group["params"]]
    needed = estimate_cpu_offload_bytes(params)
    available = available_cpu_memory_bytes()
    if available is not None and needed > available:
        raise RuntimeError(
            "Refusing to enable native.offload_optimizer_state: it needs about "
            f"{needed / (1024**3):.1f} GiB of CPU residency (parameter mirrors + "
            "pinned gradient buffers + AdamW moments) but only "
            f"{available / (1024**3):.1f} GiB is available (MemAvailable). "
            "Failing at startup instead of getting OOM-killed mid-training (§2.3)."
        )
    return CpuOffloadedAdamW(
        params,
        lr=float(group["lr"]),
        betas=tuple(float(beta) for beta in group["betas"]),
        eps=float(group["eps"]),
        weight_decay=float(group["weight_decay"]),
        amsgrad=bool(group.get("amsgrad", False)),
        maximize=bool(group.get("maximize", False)),
    )
