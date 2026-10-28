# N3c：分片检查点合并（tp / pp / dp 的 LoRA 分片 → 唯一一份 PEFT adapter）

本文只记录**判据与实测数字**。实现见 `src/graspo/eval/sharded_merge.py`（合并器）、
`src/graspo/eval/merged_export.py`（通道选路）、`tests/scripts/test_sharded_merge.py`（34 例）。

## 1. 唯一依据：分片自带元数据

合并**不推断**切分与层归属，一律读分片 payload：

| 字段 | 用途 |
|---|---|
| `lora_tensor_metadata[].shard_kind` | `none`=两矩阵都复制；`out`=`lora_b` 行块；`rows`=`lora_b` 非连续行集；`in`=`lora_a` 列块 |
| `row_start/row_stop`、`row_indices`、`col_start/col_stop` | 切片落点，必须**恰好铺满**（0 空洞、0 重叠） |
| `hf_module_path` | ★ **模块全局身份**（见 §2）；导出 PEFT 键名以它为准 |
| `r` / `alpha` | ★ native 缩放 = `alpha / r`（实测 `r=32, alpha=64` ⇒ 2）；导出必须写**真 alpha** |
| `placement`（`local_layer_indices` / `include_embeddings` / `include_lm_head`） | 层归属；并用 `parallel/placement_plan` 复算对照 |

## 2. `pp>1` 的命名契约（容易踩）

`pp_rank>0` 的分片里 **`module_name` 是 stage 局部编号**，`hf_module_path` 才是全局：

```
module_name    = "layers.0.mlp.gate_proj"                        ← 本 stage 第 0 层
hf_module_path = "model.language_model.layers.17.mlp.gate_proj"   ← 模型第 17 层
placement.local_layer_indices = [17 … 31]
```

⇒ 两个 stage 会出现**同名不同层**。按下标名聚合会把 layer 0 与 layer 17 静默并成一个。
合并器因此：身份一律取 `hf_module_path`（缺失才退化为 `"<stage偏移>:<module_name>"`），
并校验 `stage偏移 + 局部层号 == 全局层号`，不符即 fail-closed。

> 生产路径不受这一条影响：`export/exporters/native_lora.py` 按 `hf_module_path` 分组、
> 张量名取自元数据显式字段 `lora_a_name`/`lora_b_name`；`lora/lora_io.py` 用**本 rank 自己的**
> `named_modules()` 匹配。

## 3. 判据一：复制张量必须**逐位相同**（默认，取代 1 ulp 过渡口径）

**实测（推翻"1 ulp ≈ 可忽略"的直觉）**：1 步 `tp=2` 真件上复制张量漂移
`max|Δ| = 4.0e-05`（亚 ulp ~ 个位 ulp），把合并产物重灌回**同一个 native 运行时**后，
前向差 `max|Δ| = 1.5`、`mean|Δ| = 0.0212`（信号 `mean|x| = 1.27` ⇒ 相对 **1.7%**，≈151 ulp）。

对照（同一 harness，两进程）：

| 口径 | `max|Δ|` | 逐位相同 |
|---|---|---|
| 底噪：同一模型重复前向 | **0.0** | 100% |
| 对照：先把复制张量对齐再走完整闭环 | **0.0** | **100%（闭环无损）** |
| 待判：真实未对齐分片 | **1.5** | 11.1% |

⇒ 底噪严格为 0，故 1.7% 全部由参数漂移造成；**"1 ulp 差"在 32 层网络中不是可忽略量**。
代码默认 `ReplicaPolicy.require_bitwise = True`；1 ulp 口径降级为 `--diagnose-only`
的**只读诊断**，该模式产物**不得**进能力矩阵、**不得**作为判据④依据。

## 4. 支持矩阵（当前代码实测）

| 拓扑 | 训练侧可产出真件 | 合并可放行 | 结论 |
|---|---|---|---|
| `tp=dp=pp=1` | ✅ | ✅ | 旧通道，产物**逐字节不变**（回归锁） |
| `tp=1 pp=2` | ✅ | ✅ | **判据④ 的「可行性」已在该形状上验证**（逐位口径）：310 模块/620 键、两 stage 归属核对 ok、复制张量 `max=0 ulp`；`peft` 挂上 620/620 模块（零静默丢键）；vLLM 起服 + 温度 0 推理有真实输出。⚠️ 该产物来自**冒烟 run**（1 epoch），**不构成任何档位的判定结论** |
| `tp>1`（含 `tp=2`/`tp=4`/`tp=2 dp=2`/`tp=2 pp=2`） | ✅ | ❌ fail-closed | 视觉塔 LoRA（`shard_kind="none"`）**无跨 rank 梯度同步**，从第 1 步起各 rank 就不同 ⇒ checkpoint 不编码唯一模型 |
| 全参分片（`native-full-param`） | ✅ | ❌ 显式拒绝 | 分片不带 LoRA 元数据，拼回需要第二套布局推断 |

## 5. 未做与已知限制

1. 跨运行时的前向 `max|Δ|` **不可作为判据**：同一输入、同 dtype 下 HF 基座与 native 分片模型
   本身就差 `max|Δ| = 1.125`（`mean 0.023`，仅 10% 元素相同）。模型级等价须用**同运行时配对**
   口径（§3 表即为该口径）。
2. `pp>1` 的**模型级**等价尚未测（需驱动 PP 流水线前向）；该档参数级已证（`max=0 ulp`）。
3. 全参分片的合并未实现（见 §4 理由，不是漏做）。
4. 复制张量跨 rank 逐位一致依赖训练侧修复：`shard_kind="none"` 的梯度同步缺失、
   TP 归约末位不一致。修复前 `tp>1` 档一律 fail-closed 是**正确行为**，不是能力缺口。

## 6. `tp>1` 训练侧：三个缺陷已修，残留一个不等价（2026-10-28，工作包 `task-fix-tp-sync`）

> 本节只登记**与合并直接相关**的训练侧结论与残留风险。实现见 `src/graspo/flow/backend/native/lora/lora_linear.py`。

### 6.1 已修（修复前 ⇒ 修复后，均以"两 rank 检查点逐位相同"为口径）

| 缺陷 | 修复前 | 修复后 | 证据位置 |
|---|---|---|---|
| ① `shard_kind="none"`（视觉塔）LoRA **无跨 rank 梯度同步** | 1 步后 `none lora_b` **0/110** 逐位，min cos=-0.0649 | 1 步 **110/110**；100 步抽样 10 点**全部 110/110** | `runs/prefix-smoke-tp2` vs `runs/fixed-smoke-tp2`、`runs/fixed-100step-tp2` |
| ② `in` 类 `lora_b` 归约后差若干 ulp 且逐 step 放大（根因：`clip_grad_norm_` 用**本 rank 分片范数**，两 rank 对同一梯度乘不同标量） | 1 步 **0/64**、max_ulp=26 | **64/64**、max_ulp=0.00；100 步 10 点全部 64/64 | 探针 `tp_pre/tp_post/clip_pre`（`runs/probe-smoke-tp2b`） |
| ③ TP 下 LoRA 初始化**每 rank 抽同一份分片** ⇒ 逻辑 `A=[X\|X]`（有效秩砍半）、与 tp=1 从 step 0 不同 | 两 rank 分片逐位相同；与 tp=1 整块 max\|Δ\|≈3.6e-01~4.1e-01 | 两 rank 分片**不同**；`concat(shards,dim=1)` 与 tp=1 整块 **逐位相同**（max\|Δ\|=0） | `runs/fixed3-smoke-tp1` vs `runs/fixed3-smoke-tp2` |

⇒ §4 表里 `tp>1` 档"checkpoint 不编码唯一模型"的**训练侧成因已消除**（复制张量与 Adam 状态跨 rank 逐位一致）。
合并侧是否放开 fail-closed **由合并器维护者另行判定**，本节不代替该判定。

### 6.2 残留：缺陷④ `tp>1` 前向/反向**结构不等价**（未修，不属上述工作包）

同一步、`dropout=0`、同一棵树、同一份数据，**前向已证等价**（`loss=0.041748` 两侧逐位、
`fwd_in/fwd_out` 28/28 逐位、`n_mask=30`、`hidden=(1,2414,4096)`），**梯度却差 8 倍**：

| 类 | tp=1 能量 | tp=2 每 rank | tp1/tp2 |
|---|---|---|---|
| `none lora_b`（视觉塔，复制） | 58.117 | 0.027366 | **2124×** |
| `in lora_b`（复制） | 1.8008 | 0.195888 | 9.2× |
| `out lora_b`（切分，两 rank 和） | 3.0601 | 0.614 | 5.0× |
| `rows lora_b`（切分，两 rank 和） | 1.5495 | 0.172 | 9.0× |
| 总 ‖g‖ | 8.03 | 1.0044 | 8.0× |

同时复制类范数两 rank **逐位相同**、`n_grad_none=0` ⇒ **归约本身正确**，差异在集合之前。
（运行：`runs/diag2-smoke-tp1` / `runs/diag2-smoke-tp2`。）

**读数的解读纪律（重要）**
1. 所有 `tp>1` 档的**"与 tp=1 训练等价"未证**。⛔ 不得用 tp=1 的读数当 tp>1 的期望值；⛔ 不得据本节 ① ~ ③ 推断等价。
2. `tp>1` 档只能按**该档自身**的口径判（本文件 §3 的判据即如此：同一档内跨 rank 配对比较）。
3. 视觉塔 LoRA **没有被冻结**：同 100 步同配置下 `none lora_b` 的更新幅值两侧同量级
   （step 10：tp1 `mean|w|`=6.334e-05 / tp2 6.093e-05；step 60：1.347e-04 / 1.314e-04，差 ≤4%）。
   Adam 对梯度整体尺度不变，故 8× 尺度差在权重幅值上不可见；但**裁剪系数**（tp1≈0.124、tp2≈0.996）不同，
   `max_grad_norm` 真正生效时会改变轨迹。
4. 连带观测（未定位）：`tp>1` 时日志 `optimizer_steps=2`（tp=1 为 1），且 tp=2 的 loss 序列等于 tp=1 的奇数位
   ⇒ 每次记录步似乎推进 **2 个 batch**。同样只登记、不定位。

### 6.3 归约语义 `SUM` 的证明力（措辞收紧）

TP 侧复制类梯度用 `all_reduce(SUM)`（与同函数内 `out/rows→lora_a`、`in→lora_b` 的既有约定一致；
结构依据：`parallel/tensor_utils.py` `_scatter_sp` 前向按序列切片、反向只回填本段 ⇒ 各 rank 只持部分梯度）。
**该约定未由"tp=2 归约后 ≡ tp=1"实测证明**——因为 6.2 的④使两者梯度本来就不同源。
可用的证据只有三条：结构证据、同函数既有约定、跨 rank 逐位一致（`tp_post` 64/64）。
