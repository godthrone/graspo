# GRASPO 框架迁移实施计划

## 目标

删除 `backends/native_tp/`，将全部训练迁移到 `backends/graspoflow/`，实现调度与计算分离的 Flink-style 架构。不留技术负债，不向前兼容。

## 架构分层

```
Layer 0: 调度框架           ← 完全模型无关，Flink 原语（已有）
Layer 1: 通用 Transformer   ← 所有 decoder-only transformer 共用（从 qwen_ops.py + adapter.py 提取）
Layer 2: 训练编排           ← 模型无关，依赖 adapter 协议（新建 Runtime/Trainer）
Layer 3: 模型家族           ← 架构特定实现，按家族分目录
```

依赖方向：`trainer → runtime → adapter → transformer_adapter → transformer_op → Layer 0`

---

## 目标目录结构

```
graspoflow/
  __init__.py                 → 导出 GraspoFlowTrainer, GraspoFlowRuntime

  # ═══ Layer 0: 调度框架（完全模型无关，已有，不变） ═══
  operator.py                 → [已有] Microbatch, OpBuffer, ComputeOperator, OpMemoryProfile
  schedule.py                 → [已有] PipelineScheduler, GPipeScheduler, OneFOneBScheduler
  graph.py                    → [已有] PipelineGraph
  memory.py                   → [已有] 内存预算计算

  # ═══ Layer 1: 通用 Transformer 框架 ═══
  transformer_op.py           → [重构] 从 qwen_ops.py 提取通用 P2P/KV cache/memory_profile
  transformer_adapter.py      → [重构] 从 QwenNativeTPAdapter 提取 tokenizer/chat template/分布式/checkpoint

  # ═══ Layer 2: 训练编排（模型无关） ═══
  base_adapter.py             → [新建] BaseGraspoFlowAdapter 抽象协议
  runtime.py                  → [新建] GraspoFlowRuntime（委托给 adapter）
  trainer.py                  → [新建] GraspoFlowTrainer

  # ═══ 共享基础设施（从 native_tp 迁移） ═══
  parallel_state.py           → [移动] NativeTPState
  placement.py                → [移动] NativePlacementPlan
  tensor_utils.py             → [移动] ~2000 行工具函数
  checkpoint.py               → [移动] save_native_checkpoint
  lora_io.py                  → [移动] PEFT 导入导出
  lora.py                     → [新建] LoRALinear 通用类（从 models/qwen/lora.py 提取）
  multimodal.py               → [移动] 多模态编解码
  tool_parser.py              → [移动] Qwen XML tool call 解析
  logger.py                   → [移动] NativeRolloutLogger

  # ═══ 模型家族 ═══
  models/
    qwen3/                    # Qwen3: dense attention
      __init__.py
      model.py                # Qwen3DenseModel（从 modeling.py 移动）+ 新增 forward_stage()
      layers.py               # TensorParallelQwenDecoderLayer（从 layers.py 移动）
      ops.py                  # Qwen3EmbedStageOp, Qwen3DecoderStageOp, Qwen3HeadStageOp
      adapter.py              # Qwen3Adapter(TransformerAdapter)
    qwen35_36/                # Qwen3.5/3.6: hybrid attention + dense FFN
      __init__.py
      model.py                # Qwen35HybridTextModel（从 modeling_hybrid.py 移动）
      layers.py               # TensorParallelQwen35DecoderLayer（从 layers.py 移动）
      ops.py                  # Qwen35EmbedStageOp, Qwen35DecoderStageOp, Qwen35HeadStageOp
      adapter.py              # Qwen35Adapter(TransformerAdapter)

  # ═══ 待删除（重构后） ═══
  qwen_ops.py                 → 删除（逻辑已提取到 transformer_op.py + models/qwen*/ops.py）
  qwen_adapter.py             → 删除（逻辑已提取到 transformer_adapter.py + models/qwen*/adapter.py）
  rollout.py                  → 保留并增强（当前是 stub，需补全自回归 generate）
  optimize.py                 → 保留并增强（当前已工作，需与 adapter 集成）
```

### 删除清单

```
backends/native_tp/          整个目录删除
  ├── trainer.py              → graspoflow/trainer.py（重写为 GraspoFlowTrainer）
  ├── runtime.py              → graspoflow/runtime.py（重写为 GraspoFlowRuntime）
  ├── base_adapter.py         → graspoflow/base_adapter.py（重写为 BaseGraspoFlowAdapter）
  ├── qwen_tp_adapter.py      → 删除（已废弃的 re-export）
  ├── parallel_state.py       → graspoflow/parallel_state.py
  ├── placement.py            → graspoflow/placement.py
  ├── tensor_utils.py         → graspoflow/tensor_utils.py
  ├── checkpoint.py           → graspoflow/checkpoint.py
  ├── lora_io.py              → graspoflow/lora_io.py
  ├── multimodal.py           → graspoflow/multimodal.py
  ├── tool_parser.py          → graspoflow/tool_parser.py
  ├── logger.py               → graspoflow/logger.py
  └── models/qwen/            全部删除
      ├── adapter.py          → graspoflow/models/qwen3/adapter.py + qwen35_36/adapter.py
      ├── checkpoint.py       → 删除（空壳 mixin）
      ├── generator.py        → 删除（空壳 mixin）
      ├── encoding.py         → 删除（空壳 mixin）
      ├── config.py           → graspoflow/models/qwen3/ + qwen35_36/（共享）
      ├── modeling.py         → graspoflow/models/qwen3/model.py
      ├── modeling_hybrid.py  → graspoflow/models/qwen35_36/model.py
      ├── layers.py           → 拆分到各家族目录
      └── lora.py             → graspoflow/lora.py（通用 LoRALinear）+ 各家族目录（target 检测）

所有 YAML 配置:
  backend: native-tp → backend: graspoflow
  backend_config.native_tp → backend_config.graspoflow
  native_tp: → graspoflow:
```

---

## 模型家族分类

| 家族 | 目录 | 架构特征 | 当前支持 | 未来支持 |
|------|------|---------|---------|---------|
| `qwen3` | `models/qwen3/` | dense attention | Qwen3-8B | Qwen3-4B, Qwen3-14B 等 |
| `qwen35_36` | `models/qwen35_36/` | hybrid attention + dense FFN | Qwen3.5-9B, Qwen3.6-27B | Qwen3.5-4B, Qwen3.5-2B, Qwen3.6-35B-A3B 等 |
| `qwen35_36_moe` | `models/qwen35_36_moe/` | hybrid attention + MoE FFN | 无 | Qwen3.5-397B-A17B 等 |
| `deepseek_v3` | `models/deepseek_v3/` | MLA + MoE | 无 | DeepSeek-V3 等 |

**版本（尺寸）不是 class，是配置。** Qwen3.5-9B 和 Qwen3.6-27B 共享同一套 `Qwen35Adapter` 和 `Qwen35StageOp`，差异仅在于 `config.json`。

---

## 继承体系

### Operator 层

```
ComputeOperator (graspoflow/operator.py)              ← Layer 0: 纯调度抽象
  └── TransformerStageOp (graspoflow/transformer_op.py) ← Layer 1: P2P, KV cache, memory_profile
        ├── Qwen3EmbedStageOp (models/qwen3/ops.py)
        ├── Qwen3DecoderStageOp (models/qwen3/ops.py)
        ├── Qwen3HeadStageOp (models/qwen3/ops.py)
        ├── Qwen35EmbedStageOp (models/qwen35_36/ops.py)
        ├── Qwen35DecoderStageOp (models/qwen35_36/ops.py)
        └── Qwen35HeadStageOp (models/qwen35_36/ops.py)
```

### Adapter 层

```
BaseGraspoFlowAdapter (graspoflow/base_adapter.py)         ← Layer 2: 抽象协议
  └── TransformerAdapter (graspoflow/transformer_adapter.py) ← Layer 1: tokenizer, chat template, 分布式, checkpoint
        ├── Qwen3Adapter (models/qwen3/adapter.py)
        └── Qwen35Adapter (models/qwen35_36/adapter.py)
```

### 模型类

```
QwenFamilyBase (models/qwen3/model.py)  ← 共享基类
  ├── Qwen3DenseModel (models/qwen3/model.py)  ← dense attention
  └── Qwen35HybridTextModel (models/qwen35_36/model.py)  ← hybrid attention
```

---

## 各层职责

### Layer 0: 调度框架（已有，不变）

| 模块 | 职责 | 状态 |
|------|------|------|
| `operator.py` | `Microbatch`, `OpBuffer`, `ComputeOperator(ABC)`, `OpMemoryProfile` | 已有 |
| `schedule.py` | `PipelineScheduler(ABC)`, `GPipeScheduler`, `OneFOneBScheduler`, `AsyncOneFOneBScheduler` | 已有 |
| `graph.py` | `PipelineGraph` — 装配 operators + buffers | 已有 |
| `memory.py` | 内存预算计算 | 已有 |

### Layer 1: 通用 Transformer 框架

**`transformer_op.py` — `TransformerStageOp`**（从 `qwen_ops.py` 的 `QwenStageOp` 提取通用逻辑）

所有 decoder-only transformer 通用的 stage 逻辑：
- `_send_hidden()` / `_recv_hidden()` — 基于 `dist.send`/`dist.recv` 的 P2P 通信（使用 `tp_state.next_pp_rank` / `tp_state.prev_pp_rank`）
- `pp_rank`, `pp_size`, `device`, `placement` 属性
- `memory_profile` — 基于 `hidden_size` × `batch` × `seq_len` 的通用估算公式
- `trainable_parameters()` — 返回 `model.parameters()` 中 requires_grad 的参数
- `forward()` / `backward()` — 抽象方法，子类实现

**`transformer_adapter.py` — `TransformerAdapter`**（从 `QwenNativeTPAdapter` 提取通用逻辑）

所有 decoder-only transformer 通用的 adapter 逻辑：
- 分布式初始化：`NativeTPState.initialize(tp_size, pp_size)`
- Tokenizer 加载：`AutoTokenizer.from_pretrained(model_path)`
- Chat template 应用：`tokenizer.apply_chat_template(messages, ...)`
- Checkpoint 保存/恢复的通用格式（manifest + per-rank shards）
- 训练循环通用逻辑：`_shared_training_indices()`, `_aggregate_rank_metrics()`
- 内存事件：`_emit_rank_memory_event()`, `_cuda_memory_snapshot()`
- 生成辅助：`_generation_from_sequences()`, `_shared_generation_micro_batch_size()`, `_shared_rollout_prompt_chunk_size()`
- 工具方法：`_sync_timing()`, `_print_rank0()`, `_require_ready()`, `is_primary()`
- 抽象方法：`_load_model()`, `_build_ops()`, `generate_groups()`, `generate_sample_groups()`, `train_batch()`, `sequence_log_probs()`, `parse_completion()`

### Layer 2: 训练编排

**`base_adapter.py` — `BaseGraspoFlowAdapter(ABC)`**

抽象协议，定义 Trainer 看到的所有接口（与旧 `BaseNativeTPAdapter` 一致）：

```python
class BaseGraspoFlowAdapter(ABC):
    @abstractmethod
    def setup(self) -> None: ...
    @abstractmethod
    def close(self) -> None: ...
    @abstractmethod
    def is_primary(self) -> bool: ...
    @abstractmethod
    def generate_groups(self, ...) -> list[NativeGeneration]: ...
    @abstractmethod
    def generate_sample_groups(self, ...) -> list[NativeGeneration]: ...
    @abstractmethod
    def sequence_log_probs(self, ...) -> torch.Tensor: ...
    @abstractmethod
    def train_batch(self, ...) -> dict[str, Any]: ...
    @abstractmethod
    def save_checkpoint(self, ...) -> None: ...
    @abstractmethod
    def load_checkpoint(self, ...) -> dict[str, Any] | None: ...
    @abstractmethod
    def parse_completion(self, ...) -> ParsedCompletion: ...
    @abstractmethod
    def format_messages(self, ...) -> str: ...
```

**`runtime.py` — `GraspoFlowRuntime`**

与 `NativeTPRuntime` 结构相同，委托给 adapter：
- `__init__(config: GraspoConfig)` — 使用 `config.graspoflow`
- `setup()` — 动态加载 adapter 类（通过 `GRASPO_NATIVE_TP_ADAPTER` 环境变量或默认路径），调用 `adapter.setup()`
- 所有协议方法委托给 `self._adapter`
- `NativeGeneration` dataclass 从 `runtime.py` 复制到 `graspoflow/runtime.py`

**`trainer.py` — `GraspoFlowTrainer`**

与 `NativeTPGraspoTrainer` 逻辑相同（训练循环是 backend-agnostic 的）：
- rollout → reward → decision → logprob → buffer → optimize
- 使用 `GraspoFlowRuntime` 替代 `NativeTPRuntime`
- 统计类：`GraspoFlowTrainStats`（原 `NativeTrainStats`）、`GraspoFlowEpochStats`（原 `NativeEpochStats`）
- `backend_name = "graspoflow"`

### 模型家族

**`models/qwen3/` — Qwen3 家族（dense attention）**

| 文件 | 内容 | 来源 |
|------|------|------|
| `model.py` | `QwenFamilyBase`, `Qwen3DenseModel`, `TensorParallelQwenForCausalLM`, `load_native_qwen_config`, `build_native_qwen_model` | 从 `modeling.py` 移动 |
| `layers.py` | `TensorParallelQwenDecoderLayer`, `QwenRMSNorm` | 从 `layers.py` 移动 |
| `ops.py` | `Qwen3EmbedStageOp`, `Qwen3DecoderStageOp`, `Qwen3HeadStageOp` | 新建（参考 `qwen_ops.py` 但无 visual tower） |
| `adapter.py` | `Qwen3Adapter(TransformerAdapter)` | 新建（纯文本 rollout + 训练） |

**`models/qwen35_36/` — Qwen3.5/3.6 家族（hybrid attention）**

| 文件 | 内容 | 来源 |
|------|------|------|
| `model.py` | `Qwen35HybridTextModel`, `TensorParallelQwen35TextForCausalLM`, `_build_qwen35_visual_tower` | 从 `modeling_hybrid.py` 移动 |
| `layers.py` | `TensorParallelQwen35DecoderLayer`, `Qwen35RMSNorm`, `Qwen35RMSNormGated` | 从 `layers.py` 移动 |
| `ops.py` | `Qwen35EmbedStageOp`, `Qwen35DecoderStageOp`, `Qwen35HeadStageOp` | 从 `qwen_ops.py` 重构（保留 visual tower 逻辑） |
| `adapter.py` | `Qwen35Adapter(TransformerAdapter)` | 新建（多模态 rollout + 训练） |

**两个家族 Adapter 的核心差异：**

| 差异点 | Qwen3Adapter | Qwen35Adapter |
|--------|-------------|---------------|
| 模型类 | `Qwen3DenseModel` | `Qwen35HybridTextModel` |
| 模型加载 | 直接加载 dense model | 加载 hybrid model + visual tower |
| Config 读取 | `model_type: "qwen3"` | `model_type: "qwen3_5"` + `text_config` |
| 多模态 | 不支持 | 支持（image/video 编码） |
| Rollout 生成 | 纯文本 KV cache decode | 多模态 prefill + 文本 decode |
| Decoder Layers | `TensorParallelQwenDecoderLayer` | `TensorParallelQwen35DecoderLayer` |
| forward_stage | 需新增 | 已有 |

**`graspoflow/lora.py` — LoRA 通用模块**

- `LoRALinear` — 两个家族共用（从 `models/qwen/lora.py` 移动）
- `_lora_target_enabled` — 通用 target 检测逻辑
- `native_qwen_lora_available_targets` → 各家族目录（因为依赖 hf_config 结构）

---

## 分阶段实施

### Step 1: 复制共享基础设施 + 创建目录结构

**目标：** 将共享模块从 `native_tp/` 复制到 `graspoflow/`（不删除原文件），创建模型家族目录，更新 import 路径。

**操作：**

1. 创建目录：
   ```
   graspoflow/models/qwen3/
   graspoflow/models/qwen35_36/
   ```

2. 复制共享基础设施（保持内容不变，更新 import 路径以指向 graspoflow）：
   ```
   native_tp/parallel_state.py  → graspoflow/parallel_state.py
   native_tp/placement.py       → graspoflow/placement.py
   native_tp/tensor_utils.py    → graspoflow/tensor_utils.py
   native_tp/checkpoint.py      → graspoflow/checkpoint.py
   native_tp/lora_io.py         → graspoflow/lora_io.py
   native_tp/multimodal.py      → graspoflow/multimodal.py
   native_tp/tool_parser.py     → graspoflow/tool_parser.py
   native_tp/logger.py          → graspoflow/logger.py
   ```

3. 提取 LoRA 通用模块：
   ```
   native_tp/models/qwen/lora.py → graspoflow/lora.py（LoRALinear 通用类）
   ```

4. 复制模型代码到两个家族目录（此时不拆分，全量复制）：
   ```
   native_tp/models/qwen/modeling.py       → models/qwen3/model.py（全量）
   native_tp/models/qwen/modeling_hybrid.py → models/qwen35_36/model.py（全量）
   native_tp/models/qwen/layers.py         → models/qwen3/layers.py + models/qwen35_36/layers.py
   native_tp/models/qwen/config.py         → models/qwen3/config.py + models/qwen35_36/config.py
   ```

5. 更新 graspoflow 内部所有 import 路径，使其自洽（不依赖 native_tp）。

6. **验证：** `native-tp` 后端仍可正常运行（旧代码未删除）。

---

### Step 2: 重构 TransformerStageOp 和 TransformerAdapter

**目标：** 将 `qwen_ops.py` 的通用逻辑提取到 `transformer_op.py`，将 `QwenNativeTPAdapter` 的通用逻辑提取到 `transformer_adapter.py`。

#### 2a. `transformer_op.py` — `TransformerStageOp`

从 `qwen_ops.py` 的 `QwenStageOp` 提取通用逻辑，保留 Qwen 特有的部分在 `models/qwen35_36/ops.py`：

```python
class TransformerStageOp(ComputeOperator):
    """所有 decoder-only transformer 的通用 stage 基类。
    
    提供：
    - P2P 通信：_send_hidden / _recv_hidden
    - memory_profile 通用估算
    - 通用属性：pp_rank, pp_size, device, placement
    """
    def __init__(self, *, name, model, tp_state, tp_size):
        super().__init__(name=name, tp_size=tp_size)
        self.model = model
        self.tp_state = tp_state
    
    # P2P（从 QwenStageOp 移动）
    def _send_hidden(self, tensor): ...
    def _recv_hidden(self, batch, seq_len, hidden_size, dtype): ...
    
    # 通用属性
    @property
    def pp_rank(self) -> int: return self.tp_state.pp_rank
    @property
    def pp_size(self) -> int: return self.tp_state.pp_size
    @property
    def device(self) -> torch.device: return self.tp_state.device
    @property
    def placement(self): return self.model.placement
    
    # 内存（通用公式）
    @property
    def memory_profile(self) -> OpMemoryProfile:
        cfg = self.model.config
        hidden = int(cfg.hidden_size)
        return OpMemoryProfile(
            forward_activation_bytes=hidden * 2,
            backward_intermediate_bytes=hidden * 2,
            gradient_bytes=0,
        )
    
    def trainable_parameters(self):
        return [p for p in self.model.parameters() if p.requires_grad]
    
    # 抽象方法
    @abstractmethod
    def forward(self, mb): ...
    @abstractmethod
    def backward(self, mb): ...
```

#### 2b. `transformer_adapter.py` — `TransformerAdapter`

从 `QwenNativeTPAdapter`（3070 行）提取通用逻辑到 `TransformerAdapter`：

```python
class TransformerAdapter(BaseGraspoFlowAdapter):
    """所有 decoder-only transformer 的通用 adapter 基类。
    
    提供：
    - 分布式初始化（_setup_distributed）
    - Tokenizer/Processor 加载（_load_tokenizer）
    - Chat template 应用（_format_messages）
    - Checkpoint 保存/恢复格式（save_checkpoint, load_checkpoint）
    - 训练循环通用逻辑（_shared_training_indices, _aggregate_rank_metrics）
    - 内存事件（_emit_rank_memory_event）
    - 生成辅助（_generation_from_sequences, _shared_*_chunk_size）
    """
    def __init__(self, config): ...
    
    def _setup_distributed(self):
        state = NativeTPState.initialize(self.tp_size, self.pp_size)
        ...
    
    def _load_tokenizer(self, model_path): ...
    
    def setup(self):
        # 模板方法
        self._setup_distributed()
        self._load_tokenizer(...)
        self._load_model(...)          # 抽象
        self._build_ops(...)           # 抽象
        self._build_optimizer(...)     # 通用
    
    # 抽象方法
    @abstractmethod
    def _load_model(self): ...
    @abstractmethod
    def _build_ops(self): ...
    @abstractmethod
    def generate_groups(self, ...): ...
    @abstractmethod
    def generate_sample_groups(self, ...): ...
    @abstractmethod
    def train_batch(self, ...): ...
    @abstractmethod
    def sequence_log_probs(self, ...): ...
    @abstractmethod
    def parse_completion(self, ...): ...
    
    # 通用方法（从 QwenNativeTPAdapter 移动）
    def _format_messages(self, ...): ...
    def save_checkpoint(self, ...): ...
    def load_checkpoint(self, ...): ...
    def _shared_training_indices(self, ...): ...
    def _aggregate_rank_metrics(self, ...): ...
    def _generation_from_sequences(self, ...): ...
    def _shared_generation_micro_batch_size(self, ...): ...
    def _shared_rollout_prompt_chunk_size(self, ...): ...
    def _sync_timing(self): ...
    def _print_rank0(self, ...): ...
    def _emit_rank_memory_event(self, ...): ...
    def _require_ready(self): ...
    def is_primary(self): ...
    def close(self): ...  # destroy_native_tp()
```

#### 2c. `base_adapter.py` — `BaseGraspoFlowAdapter(ABC)`

纯抽象协议，定义所有方法签名（与旧 `BaseNativeTPAdapter` 一致）。

#### 2d. `runtime.py` — `GraspoFlowRuntime`

与 `NativeTPRuntime` 结构相同：
- 使用 `config.graspoflow`（而非 `config.native_tp`）
- `DEFAULT_GRASPOFLOW_ADAPTER = "graspo.backends.graspoflow.models.qwen35_36.adapter:Qwen35Adapter"`
- 环境变量 `GRASPO_NATIVE_TP_ADAPTER` 复用（不改名，保持兼容）
- 所有方法委托给 `self._adapter`
- `NativeGeneration` 复制到 `graspoflow/runtime.py`

#### 2e. `trainer.py` — `GraspoFlowTrainer`

与 `NativeTPGraspoTrainer` 几乎相同：
- 使用 `GraspoFlowRuntime` 替代 `NativeTPRuntime`
- `GraspoFlowTrainStats`（原 `NativeTrainStats`）
- `GraspoFlowEpochStats`（原 `NativeEpochStats`）
- `backend_name = "graspoflow"`

---

### Step 3: 实现模型家族

**目标：** 实现 `Qwen3Adapter` 和 `Qwen35Adapter`，以及对应的 StageOp 系列。

#### 3a. `models/qwen3/`

**`model.py`：**
- `QwenFamilyBase`（共享基类，从 `modeling.py` 移动）
- `Qwen3DenseModel`（从 `modeling.py` 移动）
- `TensorParallelQwenForCausalLM`（从 `modeling.py` 移动）
- `load_native_qwen_config`, `build_native_qwen_model`（从 `modeling.py` 移动）
- **新增** `Qwen3DenseModel.forward_stage()` 方法

**`layers.py`：**
- `TensorParallelQwenDecoderLayer`, `QwenRMSNorm`（从 `layers.py` 移动）
- `_checkpoint_decoder_layer_forward`（从 `layers.py` 移动）

**`ops.py`：**
```python
class Qwen3EmbedStageOp(TransformerStageOp):
    """Stage 0: embedding + first N dense layers"""
    def forward(self, mb):
        hidden = self.model.embed_tokens(mb.input_ids)
        position_ids = self.model.compute_position_ids(mb.input_ids, mb.attention_mask)
        for layer in self.model.layers:
            hidden = layer(hidden, position_ids, mb.attention_mask)
        if self.pp_size > 1:
            self._send_hidden(hidden.detach())
        mb.hidden_states = hidden.detach()
        mb._stage_output = hidden
        return mb
    def backward(self, mb): ...  # P2P recv grad → backward

class Qwen3DecoderStageOp(TransformerStageOp):
    """中间 stage: dense decoder layers"""
    ...

class Qwen3HeadStageOp(TransformerStageOp):
    """最终 stage: dense layers + norm + lm_head"""
    ...
```

**`adapter.py`：**
```python
class Qwen3Adapter(TransformerAdapter):
    model_family = "qwen3"
    
    def _load_model(self):
        # 加载 Qwen3DenseModel
        ...
    def _build_ops(self):
        # 构建 Qwen3StageOp 列表
        ...
    def generate_groups(self, ...):
        # 纯文本 rollout（TP-only 和 PP 两条路径）
        ...
    def generate_sample_groups(self, ...):
        raise NotImplementedError("Qwen3 does not support multimodal")
    def train_batch(self, ...):
        # TP-only 和 PP 训练
        ...
    def sequence_log_probs(self, ...):
        ...
    def parse_completion(self, ...):
        # Qwen tool call 解析
        ...
```

#### 3b. `models/qwen35_36/`

**`model.py`：**
- `Qwen35HybridTextModel`（从 `modeling_hybrid.py` 移动）
- `TensorParallelQwen35TextForCausalLM`（从 `modeling_hybrid.py` 移动）
- `_build_qwen35_visual_tower`（从 `modeling.py` 移动）

**`layers.py`：**
- `TensorParallelQwen35DecoderLayer`, `Qwen35RMSNorm`, `Qwen35RMSNormGated`（从 `layers.py` 移动）
- `_checkpoint_qwen35_decoder_layer_forward`, `_qwen35_cache_sequence_len`（从 `layers.py` 移动）

**`ops.py`：**（从 `qwen_ops.py` 重构，保留 visual tower 逻辑）
```python
class Qwen35EmbedStageOp(TransformerStageOp):
    """Stage 0: embedding + visual tower + first N hybrid layers"""
    def forward(self, mb):
        hidden = self.model.embed_inputs(mb.input_ids, multimodal_inputs=mb.multimodal_inputs)
        position_ids = self.model.compute_multimodal_position_ids(...)
        for layer in self.model.layers:
            hidden = layer(hidden, position_ids, mb.attention_mask)
        if self.pp_size > 1:
            self._send_hidden(hidden.detach())
        ...
    def backward(self, mb): ...

class Qwen35DecoderStageOp(TransformerStageOp):
    """中间 stage: hybrid decoder layers"""
    ...

class Qwen35HeadStageOp(TransformerStageOp):
    """最终 stage: hybrid layers + norm + lm_head"""
    ...
```

**`adapter.py`：**
```python
class Qwen35Adapter(TransformerAdapter):
    model_family = "qwen35_36"
    
    def _load_model(self):
        # 加载 Qwen35HybridTextModel + visual tower
        ...
    def _build_ops(self):
        # 构建 Qwen35StageOp 列表
        ...
    def generate_groups(self, ...):
        # 纯文本 rollout（TP-only 和 PP 两条路径）
        ...
    def generate_sample_groups(self, ...):
        # 多模态 rollout（TP-only 和 PP 两条路径）
        ...
    def train_batch(self, ...):
        # TP-only, PP simple, PP 1F1B 训练
        ...
    def sequence_log_probs(self, ...):
        ...
    def parse_completion(self, ...):
        # Qwen tool call 解析
        ...
```

#### 3c. `graspoflow/lora.py`

- `LoRALinear`（从 `models/qwen/lora.py` 移动，两个家族共用）
- `_lora_target_enabled`（通用 target 检测逻辑）
- 各家族目录的 `layers.py` 从 `graspoflow.lora` import `LoRALinear`
- `native_qwen_lora_available_targets` 分别放在各家族目录（依赖 hf_config 结构）

---

### Step 4: 补全 Rollout Pipeline

**目标：** 补全 `RolloutPipeline` 的自回归生成逻辑。

**参考实现：** 旧 adapter 的 `_pipeline_generate_sequences_with_cache()`（第 1737-1814 行）。

**核心逻辑：**
1. **Prefill 阶段：** 全量 prompt 走一遍 pipeline forward（embed → ... → head），得到 logits 和 KV cache
2. **Decode 阶段：** token-by-token，每次 head stage 采样一个 token，通过 `dist.broadcast` 同步到所有 stages
3. **PP 模式：** 每个 decode step 走全 pipeline（embed → decoder → head），head 采样后 broadcast
4. **TP-only 模式（pp=1）：** 直接调用 `model.forward()` 完成 prefill + decode

**Pipeline 负责：** 管理 micro-batch 流，push/drain 控制。

**HeadStageOp 负责：** 采样 + broadcast。

**关键差异 Qwen3 vs Qwen35：**
- Qwen3：纯文本，KV cache 结构 `(key, value)` 对
- Qwen35：多模态 prefill 阶段需要 visual feature embedding

---

### Step 5: 补全 Optimize Pipeline

**目标：** 将 `OptimizePipeline` 与 adapter 的 `train_batch()` 集成。

**参考实现：** 旧 adapter 的 `_pipeline_one_f_one_b_optimizer_step()`（第 2494-2649 行）。

**核心逻辑：**
1. 将 experiences 按 `pp_micro_batch_size` 分成 chunk_batches
2. 每个 chunk 执行 1F1B：fill（warmup 个 forward）→ steady（forward+backward 交替）→ drain（剩余 backward）
3. Loss 计算在 head stage（pp_rank == pp_size - 1）
4. 梯度通过 NCCL P2P 从下游传回上游
5. 所有 ranks 都执行 gradient clipping + optimizer.step()

**Adapter 负责：** 分 chunk、loss 计算、gradient clipping、optimizer.step()、metrics 聚合。

**Pipeline 负责：** 按 1F1B schedule 执行 forward/backward，P2P 传递 hidden states 和 gradients。

---

### Step 6: 集成、配置更新、删除旧代码

#### 6a. `selector.py`
```python
SUPPORTED_BACKENDS = {"graspoflow"}

def select_backend(config, requested=None):
    requested_backend = (requested or config.backend or "graspoflow").strip()
    if requested_backend != "graspoflow":
        raise ValueError(
            f"Unsupported backend '{requested_backend}'. "
            f"GRASPO only supports 'graspoflow'."
        )
    return BackendSelection(name="graspoflow", ...)

def create_trainer(config, selection):
    from graspo.backends.graspoflow import GraspoFlowTrainer
    return GraspoFlowTrainer(config, selection=selection)
```

#### 6b. `schema.py`
- `NativeTPConfig` 保留（重命名为 `GraspoFlowConfig` 或保留原名，因为它就是配置 struct）
- `GraspoConfig.native_tp` 字段保留但标记为 deprecated → 从 `backend_config.graspoflow` 读取
- 或者：`native_tp` → `graspoflow` 重命名，`native_tp` 字段接受但报 warning

#### 6c. 更新所有 YAML 配置
- `backend: native-tp` → `backend: graspoflow`
- `backend_config.native_tp` → `backend_config.graspoflow`
- `native_tp:` → `graspoflow:`

#### 6d. 删除 `backends/native_tp/` 整个目录

#### 6e. 更新 `graspoflow/__init__.py`
```python
from graspo.backends.graspoflow.trainer import GraspoFlowTrainer
from graspo.backends.graspoflow.runtime import GraspoFlowRuntime

__all__ = ["GraspoFlowTrainer", "GraspoFlowRuntime"]
```

#### 6f. 删除 `graspoflow/qwen_ops.py` 和 `graspoflow/qwen_adapter.py`（逻辑已迁移）

#### 6g. 更新 `pyproject.toml`、`CLAUDE.md` 和相关文档

---

### Step 7: 验证

**验证矩阵：**

| 模型 | 配置 | TP | PP | GPU | 验证项 |
|------|------|----|----|-----|--------|
| Qwen3-8B | `qwen3_8b_tp2.yaml` | 2 | 1 | 2 | TP-only: 生成+训练+checkpoint |
| Qwen3.5-9B | `qwen35_9b_tp4.yaml` | 4 | 1 | 4 | TP-only: 多模态生成+训练 |
| Qwen3.5-9B MM | `qwen35_9b_mm_tp2.yaml` | 2 | 1 | 2 | 多模态编码正确性 |
| Qwen3.6-27B | `qwen36_27b_pp8.yaml` | 1 | 8 | 8 | PP-only: 1F1B训练 |
| Qwen3.6-27B | `graspoflow_tp2_pp4_114.yaml` | 2 | 4 | 8 | TP+PP hybrid: 1F1B训练 |

**验证标准：**
- loss 收敛曲线与旧框架一致
- 显存使用与旧框架一致或更优
- checkpoint 可正常保存/恢复
- 多模态输入正常处理

**可用的 GPU 资源：**
- 114 服务器：GPU 4-7 可用（已验证 graspoflow 1F1B 训练）
- 228 服务器：GPU 4-7 可用（用于长训验证）

---

## 风险评估

### 风险 1: QwenNativeTPAdapter 的代码量巨大（3070 行）
提取通用逻辑时容易遗漏或引入 bug。

**缓解：** 采用对照式重构——原文件不动，新文件逐步建立，用 `diff` 验证关键方法一致。Step 2 完成后手动对比旧 adapter 和新 adapter 的同名方法。

### 风险 2: Qwen3DenseModel 缺少 forward_stage
旧代码中 TP-only 使用 `Qwen3DenseModel.forward()`，PP 模式使用 `Qwen35HybridTextModel.forward_stage()`。Qwen3DenseModel 没有 `forward_stage` 方法。

**缓解：** 在 Step 3 中给 `Qwen3DenseModel` 添加 `forward_stage()` 方法，签名与 `Qwen35HybridTextModel.forward_stage()` 一致。

### 风险 3: Rollout Pipeline 的 P2P 同步
旧代码中 `_pipeline_generate_sequences_with_cache` 的 P2P 通信涉及精密的 NCCL send/recv + broadcast 时序。

**缓解：** 逐行对比旧代码的 P2P 调用，保持相同顺序。先用 PP=8 的 Qwen3.6-27B 配置做集成测试。

### 风险 4: LoRA 模块的共享
`LoRALinear` 在两个家族中都需要，但 target 检测逻辑不同。

**缓解：** `LoRALinear` 放在 `graspoflow/lora.py`，`_lora_target_enabled` 和 `native_qwen_lora_available_targets` 放在各家族目录。

### 风险 5: 旧 TP-only 路径中的微优化
旧 adapter 中 TP-only 路径有一些硬件特定的优化（KV cache 管理、memory fragmentation 防护、`expandable_segments` 设置）。

**缓解：** 保留 `tensor_utils.py` 中的所有工具函数不变。Trainer 中的 `PYTORCH_CUDA_ALLOC_CONF` 设置移到 `GraspoFlowTrainer.train()` 中。

---

## 关键设计决策

| 决策 | 结论 |
|------|------|
| 是否加 Transformer 中间层 | ✅ 是，`TransformerStageOp` + `TransformerAdapter` |
| 家族命名 | `qwen3`, `qwen35_36`, `qwen35_36_moe`, `deepseek_v3` |
| 版本（尺寸）是否 class | ❌ 否，由 HF config.json 决定 |
| TP-only 是否走 pipeline | ✅ 是，pp=1 退化为单 operator |
| Decode loop 放哪里 | HeadStageOp.generate() — 模型专属逻辑 |
| Loss/Optimizer 放哪里 | Adapter.train_batch() — 不在 pipeline graph 内 |
| 旧配置兼容 | ❌ 不兼容，`native-tp` 直接报错 |
| 旧 checkpoint 兼容 | ❌ 不兼容 |
| LoRALinear 放哪里 | `graspoflow/lora.py` — 两个家族共用 |
| qwen_ops.py 处置 | 重构提取到 transformer_op.py + models/qwen35_36/ops.py，然后删除 |
| qwen_adapter.py 处置 | 重构提取到 transformer_adapter.py + models/qwen*/adapter.py，然后删除 |
| 迁移策略 | 先复制后删除：Step 1 复制 → Step 2-5 在新位置开发 → Step 6 删除旧代码 |

---

## 时间估算

| Step | 内容 | 预估 |
|------|------|------|
| Step 1 | 文件复制 + import 更新 | 0.5 天 |
| Step 2 | TransformerStageOp + TransformerAdapter + Runtime + Trainer | 1.5 天 |
| Step 3 | Qwen3Adapter + Qwen35Adapter + StageOps | 2 天 |
| Step 4 | Rollout Pipeline（自回归 + KV cache） | 2 天 |
| Step 5 | Optimize Pipeline（1F1B + 梯度同步） | 1 天 |
| Step 6 | 集成 + 删除旧代码 + 配置更新 | 0.5 天 |
| Step 7 | 验证 + 调试 | 2 天 |
| **总计** | | **9.5 天** |