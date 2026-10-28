"""LoRA 注入与 I/O 工具：微调参数应用、保存、加载、合并（设施层）。"""

from __future__ import annotations

import math
from collections.abc import Iterable
from typing import Any

import torch
import torch.distributed as dist
from torch import nn
from torch.nn import functional as F  # noqa: N812

from graspo.flow.backend.native.lora.lora_helpers import (
    VISUAL_HF_PREFIX,
    _lora_target_enabled,
)
from graspo.flow.backend.native.parallel.tensor_utils import _shard_bounds, _shard_tensor


def _replace_visual_lora_modules(
    visual: nn.Module,
    *,
    lora_targets: set[str],
    lora_r: int,
    lora_alpha: int,
    lora_dropout: float,
    device: torch.device,
    torch_dtype: torch.dtype,
) -> None:
    target_to_path = {
        "visual.merger.linear_fc1": "merger.linear_fc1",
        "visual.merger.linear_fc2": "merger.linear_fc2",
    }
    depth = len(getattr(visual, "blocks", []))
    for idx in range(depth):
        target_to_path.update(
            {
                f"visual.blocks.{idx}.attn.qkv": f"blocks.{idx}.attn.qkv",
                f"visual.blocks.{idx}.attn.proj": f"blocks.{idx}.attn.proj",
                f"visual.blocks.{idx}.mlp.linear_fc1": f"blocks.{idx}.mlp.linear_fc1",
                f"visual.blocks.{idx}.mlp.linear_fc2": f"blocks.{idx}.mlp.linear_fc2",
            }
        )
    for target_name, module_path in target_to_path.items():
        # ★ 判据用**真实 HF 模块名**（`model.visual.*`）——与 ms-swift/peft 侧同一匹配域，
        #   同一条 `lora.target_modules` 正则 ⇒ 同一集合（旧版拿 native 规范名
        #   `visual.*` 判，与正则的域不是同一个，故此处换域）。
        hf_module_path = f"{VISUAL_HF_PREFIX}{module_path}"
        if not _lora_target_enabled(lora_targets, hf_module_path):
            continue
        parent, attr = _module_parent_and_attr(visual, module_path)
        linear = getattr(parent, attr)
        if not isinstance(linear, nn.Linear):
            raise RuntimeError(f"visual LoRA target {target_name} is not an nn.Linear")
        replacement = LoRALinear(
            linear.weight.detach(),
            linear.bias.detach() if linear.bias is not None else None,
            lora_enabled=True,
            target_name=target_name,
            hf_module_path=hf_module_path,
            r=lora_r,
            alpha=lora_alpha,
            dropout=lora_dropout,
            device=device,
            dtype=torch_dtype,
        )
        setattr(parent, attr, replacement)


def _module_parent_and_attr(module: nn.Module, path: str) -> tuple[nn.Module, str]:
    parts = path.split(".")
    parent: nn.Module = module
    for part in parts[:-1]:
        parent = (
            parent[int(part)]
            if part.isdigit() and isinstance(parent, nn.ModuleList)
            else getattr(parent, part)
        )
    return parent, parts[-1]


class LoRALinear(nn.Module):
    def __init__(
        self,
        weight: torch.Tensor,
        bias: torch.Tensor | None,
        *,
        lora_enabled: bool,
        r: int,
        alpha: int,
        dropout: float,
        device: torch.device,
        dtype: torch.dtype,
        target_name: str | None = None,
        hf_module_path: str | None = None,
        base_weight_name: str | None = None,
        shard_kind: str = "none",
        init_full_in_features: int | None = None,
        row_start: int | None = None,
        row_stop: int | None = None,
        col_start: int | None = None,
        col_stop: int | None = None,
        row_indices: Iterable[int] | None = None,
        peft_exportable: bool = True,
    ) -> None:
        super().__init__()
        out_features, in_features = weight.shape
        self.weight = nn.Parameter(weight.to(device=device, dtype=dtype), requires_grad=False)
        self.bias = (
            nn.Parameter(bias.to(device=device, dtype=dtype), requires_grad=False)
            if bias is not None
            else None
        )
        self.lora_target_name = str(target_name or "unknown")
        self.hf_module_path = str(hf_module_path) if hf_module_path else None
        self.base_weight_name = (
            str(base_weight_name)
            if base_weight_name
            else (f"{self.hf_module_path}.weight" if self.hf_module_path else None)
        )
        self.lora_shard_kind = str(shard_kind or "none")
        self.lora_init_full_in_features = int(init_full_in_features or in_features)
        self.lora_row_start = row_start
        self.lora_row_stop = row_stop
        self.lora_col_start = col_start
        self.lora_col_stop = col_stop
        self.lora_row_indices = tuple(int(idx) for idx in row_indices) if row_indices else None
        self.peft_exportable = bool(peft_exportable)
        self.lora_enabled = bool(lora_enabled and r > 0)
        self.lora_r = int(r)
        self.lora_alpha = int(alpha)
        self.scaling = float(alpha) / float(r) if r > 0 else 1.0
        self.dropout = nn.Dropout(dropout)
        if self.lora_enabled:
            self.lora_a = nn.Parameter(torch.empty(r, in_features, device=device, dtype=dtype))
            self.lora_b = nn.Parameter(torch.zeros(out_features, r, device=device, dtype=dtype))
            _full_in = int(self.lora_init_full_in_features)
            if _full_in == in_features:
                # 单卡 / 非列切分：抽样量与旧实现**逐字节相同**（golden 锁）。
                nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))
            else:
                # ★ TP 列切分（shard_kind="in"）：先按**全形状**抽一次，再确定性切出本 rank
                #   的列块。各 rank 用同一 init seed ⇒ 抽到的是同一份全矩阵 ⇒ 拼接后与单卡
                #   整块逐位相同；且抽样量与单卡一致 ⇒ RNG 流不会从第一个 in 层起错位。
                #   ⛔ 不得改成"每 rank 各自抽样"（同 seed 会抽成重复块 ⇒ 逻辑矩阵秩退化，
                #      A=[X|X]；不同 seed 则不可复现）。
                _full = torch.empty(r, _full_in, device=device, dtype=dtype)
                nn.init.kaiming_uniform_(_full, a=math.sqrt(5))
                _start = int(col_start or 0)
                _stop = int(col_stop) if col_stop is not None else _full_in
                if _stop - _start != in_features:
                    raise RuntimeError(
                        f"LoRALinear 列切分区间 [{_start},{_stop}) 与本地宽度 {in_features} 不一致"
                    )
                self.lora_a.data.copy_(_full[:, _start:_stop])
                del _full
        else:
            self.register_parameter("lora_a", None)
            self.register_parameter("lora_b", None)

    @classmethod
    def from_hf(
        cls,
        weight: torch.Tensor,
        bias: torch.Tensor | None,
        *,
        shard: str,
        tp_rank: int,
        tp_size: int,
        lora_enabled: bool,
        r: int,
        alpha: int,
        dropout: float,
        device: torch.device,
        dtype: torch.dtype,
        target_name: str | None = None,
        hf_module_path: str | None = None,
        base_weight_name: str | None = None,
        peft_exportable: bool = True,
    ) -> LoRALinear:
        dim = 0 if shard == "out" else 1
        full_in_features = int(weight.shape[1])  # 切分前的输入宽（列切分的抽样域）
        row_start = row_stop = col_start = col_stop = None
        if shard == "out":
            row_start, row_stop = _shard_bounds(weight.shape[0], tp_rank=tp_rank, tp_size=tp_size)
        else:
            # 列切分（含 ``shard="in"``；``dim=1`` 的一切情形）：记录本 rank 的列块，
            # 供 ``lora_a`` 按全形状抽样后确定性切分（缺陷③）。
            col_start, col_stop = _shard_bounds(weight.shape[1], tp_rank=tp_rank, tp_size=tp_size)
        weight = _shard_tensor(weight, dim=dim, tp_rank=tp_rank, tp_size=tp_size)
        if bias is not None and shard == "out":
            bias = _shard_tensor(bias, dim=0, tp_rank=tp_rank, tp_size=tp_size)
        elif shard == "in":
            bias = None
        return cls(
            weight,
            bias,
            lora_enabled=lora_enabled,
            target_name=target_name,
            hf_module_path=hf_module_path,
            base_weight_name=base_weight_name,
            shard_kind=shard,
            init_full_in_features=full_in_features,
            row_start=row_start,
            row_stop=row_stop,
            col_start=col_start,
            col_stop=col_stop,
            peft_exportable=peft_exportable,
            r=r,
            alpha=alpha,
            dropout=dropout,
            device=device,
            dtype=dtype,
        )

    def lora_metadata(self, module_name: str) -> dict[str, Any]:
        if self.hf_module_path is None or self.base_weight_name is None:
            raise RuntimeError(f"Enabled LoRA module {module_name} is missing HF module metadata")
        return {
            "module_name": module_name,
            "lora_a_name": f"{module_name}.lora_a",
            "lora_b_name": f"{module_name}.lora_b",
            "target_name": self.lora_target_name,
            "hf_module_path": self.hf_module_path,
            "base_weight_name": self.base_weight_name,
            "shard_kind": self.lora_shard_kind,
            "row_start": self.lora_row_start,
            "row_stop": self.lora_row_stop,
            "col_start": self.lora_col_start,
            "col_stop": self.lora_col_stop,
            "row_indices": list(self.lora_row_indices)
            if self.lora_row_indices is not None
            else None,
            "peft_exportable": self.peft_exportable,
            "r": self.lora_r,
            "alpha": self.lora_alpha,
        }

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        output = F.linear(hidden_states, self.weight, self.bias)
        if self.lora_enabled:
            lora = F.linear(F.linear(self.dropout(hidden_states), self.lora_a), self.lora_b)
            output = output + lora * self.scaling
        return output


#: 复制类 LoRA 矩阵的**唯一判据**：``lora_shard_kind`` → 该 kind 下哪些 LoRA 矩阵在
#: 每个 TP rank 上持有**完整副本**（⇒ 其梯度必须跨 TP 同步/在范数里只计一次）。
#: 与 :meth:`LoRALinear.from_hf` 的切分语义互为补集：
#:
#: - ``"none"``：权重整块复制（如视觉塔，TP 不切它）⇒ a、b 都复制；
#: - ``"out"`` / ``"rows"``：权重按输出行切 ⇒ 只有 lora_a 复制，lora_b 随行切分；
#: - ``"in"``：权重按输入列切 ⇒ 只有 lora_b 复制，lora_a 随列切分。
#:
#: ⛔ 新增 shard_kind 必须同时登记在此：未登记 = fail-closed 报错，
#: 不静默跳过同步（缺陷① 的成因正是 ``"none"`` 缺失时被 if/elif 静默漏掉）。
TP_REPLICATED_LORA_MATRICES: dict[str, tuple[str, ...]] = {
    "none": ("a", "b"),
    "out": ("a",),
    "rows": ("a",),
    "in": ("b",),
}


def _tp_replicated_lora_params(module: LoRALinear) -> list[tuple[str, nn.Parameter]]:
    """返回该层**跨 TP 复制**的 ``("a"|"b", 参数)``，顺序稳定（先 a 后 b）。"""
    kind = str(getattr(module, "lora_shard_kind", "none") or "none")
    letters = TP_REPLICATED_LORA_MATRICES.get(kind)
    if letters is None:
        raise RuntimeError(
            f"LoRALinear shard_kind={kind!r} 未在 TP_REPLICATED_LORA_MATRICES 登记，"
            "无法判定哪些 LoRA 矩阵跨 TP 复制（fail-closed：不静默跳过同步）"
        )
    params: list[tuple[str, nn.Parameter]] = []
    for letter in letters:
        param = module.lora_a if letter == "a" else module.lora_b
        if param is not None:
            params.append((letter, param))
    return params


def _iter_lora_modules(model: nn.Module) -> Iterable[LoRALinear]:
    """同步与范数共用同一筛选口径（单一真相源，避免两处口径漂移）。"""
    for module in model.modules():
        if isinstance(module, LoRALinear) and module.lora_enabled and module.lora_a is not None:
            yield module


def _sync_nonsharded_lora_grads(model: nn.Module, tp_group: Any) -> None:
    """All-reduce (sum) the gradient of the non-sharded LoRA matrix across TP ranks.

    In TP-sharded layers, the non-sharded LoRA matrix receives different gradients
    per rank because the input/output gradient is partial.  All-reducing the
    gradient BEFORE the optimizer step ensures that every rank computes the same
    weight update, keeping both the weights AND the Adam optimizer state in sync.

    Uses SUM (not AVG) because TP ranks process the SAME data and compute PARTIAL
    gradients.  The full gradient = sum of partial gradients across ranks.
    This differs from DDP where each rank processes DIFFERENT data and AVG is
    the correct reduction.

    覆盖范围由 :data:`TP_REPLICATED_LORA_MATRICES` 单点定义（本函数不写死 kind 列表）：

    - ``"none"``（整块复制的层，如视觉塔 LoRA）：a、b 都求和——视觉塔输入在 rank 间
      复制，但反向是经 SP 序列切分回流的（``parallel/tensor_utils.py::_scatter_sp``
      的前向切片在反向是"只保留自己那段"），故每个 rank 只拿到**部分**梯度；
      缺此分支时两 rank 各走各的 Adam ⇒ 复制类参数静默分叉（缺陷①）。
    - ``"rows"`` / ``"out"``：lora_a 求和；
    - ``"in"``：lora_b 求和。

    **Collective safety**: every rank must call the same number of all_reduce
    operations on the same communicator.  When a rank has no gradient for a
    parameter, it still participates by all-reducing a zero tensor of the same
    shape.
    """
    if tp_group is None:
        return
    for _mod in _iter_lora_modules(model):
        for _letter, _param in _tp_replicated_lora_params(_mod):
            if _param.grad is not None:
                dist.all_reduce(_param.grad, op=dist.ReduceOp.SUM, group=tp_group)
            else:
                zero = torch.zeros_like(_param)
                dist.all_reduce(zero, op=dist.ReduceOp.SUM, group=tp_group)


def lora_grad_norm_sq_parts(
    model: nn.Module, trainable_params: Iterable[nn.Parameter]
) -> tuple[torch.Tensor, torch.Tensor]:
    """把可训参数的梯度平方和拆成 ``(复制桶, 切片桶)``——**无集合调用**（纯本地）。

    - 复制桶：:data:`TP_REPLICATED_LORA_MATRICES` 判定的 LoRA 矩阵。它们的梯度已被
      :func:`_sync_nonsharded_lora_grads` 归约成跨 rank 逐位相同 ⇒ 全模型范数里
      **只能计一次**（每 rank 都有一份完整副本）。
    - 切片桶：其余可训参数——LoRA 里被 TP 切开的矩阵，以及全参档的权重本体。
      每个 rank 只持有自己那份（互不重叠）⇒ 跨 TP **求和**才是全量。

    两个桶都按 fp32 累加（bf16 累加会引入与 rank 相关的量化误差，正是缺陷② 的放大器）。
    """
    params = [param for param in trainable_params if param is not None]
    device = params[0].device if params else torch.device("cpu")
    replicated_sq = torch.zeros((), device=device, dtype=torch.float32)
    sliced_sq = torch.zeros((), device=device, dtype=torch.float32)
    replicated_ids: set[int] = set()
    for module in _iter_lora_modules(model):
        for _letter, param in _tp_replicated_lora_params(module):
            replicated_ids.add(id(param))
    for param in params:
        if param.grad is None:
            continue
        _sq = param.grad.detach().float().pow(2).sum()
        if id(param) in replicated_ids:
            replicated_sq += _sq
        else:
            sliced_sq += _sq
    return replicated_sq, sliced_sq


def tp_invariant_grad_norm_sq(
    model: nn.Module, trainable_params: Iterable[nn.Parameter], tp_group: Any
) -> torch.Tensor:
    """跨 TP **不变量**的全模型梯度范数平方（fp32 标量，各 rank 逐位相同）。

    复制桶只计一次 + 切片桶跨 TP all_reduce(SUM) ⇒ 每个 rank 得到同一个全模型范数。
    ``clip`` 必须与梯度的域**同域**：用 rank 本地的偏范数算 clip 系数，会让同一个
    梯度在 rank 间被乘上不同标量（缺陷②），并逐 step 复利放大。

    ``tp_group is None``（单 TP rank）⇒ 不做集合，返回纯本地范数平方。
    """
    replicated_sq, sliced_sq = lora_grad_norm_sq_parts(model, trainable_params)
    if tp_group is not None:
        dist.all_reduce(sliced_sq, op=dist.ReduceOp.SUM, group=tp_group)
    return sliced_sq + replicated_sq


def clip_grads_tp_invariant(
    model: nn.Module,
    trainable_params: Iterable[nn.Parameter],
    max_grad_norm: float,
    tp_group: Any,
) -> torch.Tensor:
    """按跨 TP 不变量的全局范数裁剪梯度，返回**裁剪前**范数（fp32 标量张量）。

    阈值语义不变（仍是 ``max_grad_norm``），改的只是范数的**域**：从「本 rank 分片」
    换成「全模型」，与 1 卡语义对齐——同 ``training.py`` 里 E-84 对 PP 的做法，
    本函数把同一口径补齐到 TP。

    ⛔ 只在 ``tp_size > 1`` 时调用：tp=1 的调用点保持原 ``clip_grad_norm_`` 路径，
    保证单卡产物逐字节不变（golden）。
    """
    params = [param for param in trainable_params if param is not None]
    total_sq = tp_invariant_grad_norm_sq(model, params, tp_group)
    global_norm = math.sqrt(float(total_sq))
    if global_norm > 0.0:
        coef = min(1.0, float(max_grad_norm) / (global_norm + 1e-6))
        for param in params:
            if param.grad is not None:
                param.grad.mul_(coef)
    return total_sq.sqrt()


def _sync_dp_lora_grads(model: nn.Module, dp_group: Any) -> None:
    """All-reduce (AVG) all LoRA gradients across DP replicas.

    DP replicas process DIFFERENT data, so AVG is the correct reduction.
    Unlike TP sync (which only syncs non-sharded LoRA matrices with SUM),
    DP syncs ALL LoRA parameters because each DP rank computes a full
    gradient on its own data shard.

    Must be called BEFORE TP sync and BEFORE optimizer.step().
    Sync order: DP(AVG) → TP(SUM) → clip_grad → optimizer.step()

    **Collective safety**: every rank must call the same number of all_reduce
    operations on the same communicator.  When a rank has no gradient for a
    parameter (e.g. visual-encoder LoRA on a text-only batch), it still
    participates by all-reducing a zero tensor of the same shape.
    """
    if dp_group is None:
        return
    for _mod in model.modules():
        if not isinstance(_mod, LoRALinear) or not _mod.lora_enabled:
            continue
        for param in (_mod.lora_a, _mod.lora_b):
            if param is None:
                continue
            if param.grad is not None:
                dist.all_reduce(param.grad, op=dist.ReduceOp.AVG, group=dp_group)
            else:
                zero = torch.zeros_like(param)
                dist.all_reduce(zero, op=dist.ReduceOp.AVG, group=dp_group)
