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
