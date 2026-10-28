# N3c 交付：分片检查点合并（tp / pp / dp）

**一句话**：native 后端的 `final/` 分片（`rank_<r>_tp_<t>_pp_<p>.pt`）现在能被**严格按分片
自带的元数据**拼回"整模型那一份" LoRA，并复用既有 PEFT→HF→vLLM 通道导出；拼不回去的
情形**全部 fail-closed** 并打印漂移清单。**但要如实说清：本次能拼对"切片"，拼不了"复制"
—— 训练侧有两个缺陷让 `tp>1` 的 checkpoint 里那些"复制张量"各 rank 不一样，见第 4 节。**

> ★ 本文件不含内网坐标（IP / 主机路径 / 作业号），原始证据在本地 `.local/` 下。

---

## 1. 做了什么

| 文件 | 作用 |
|---|---|
| `src/graspo/eval/sharded_merge.py`（新增） | 分片合并器本体：拓扑校验、切片拼接、复制一致性判定、回执 |
| `src/graspo/eval/merged_export.py` | 新增 `convert_native_checkpoint_to_peft`（按拓扑选路）＋ 抽出 `_write_peft_adapter` 供两通道共用；`--require-bitwise` 口径经 CLI 暴露 |
| `src/graspo/cli/eval_commands.py` | `graspo eval export` 改走选路入口；`--require-bitwise` 开关 |
| `tests/scripts/test_sharded_merge.py`（新增，33 例） | 三类切片还原 / 五类 fail-closed / 旧通道 golden / 通道选路 |

**旧通道不动**：`tp=dp=pp=1` 仍走原 `convert_native_rank_to_peft`，产物 **sha256 逐字节相同**
（用例 `test_golden_single_shard_matches_legacy_conversion` 锁死）。

### 合并规则（唯一依据 = 分片自带的 `lora_tensor_metadata`，不推断）

| `shard_kind` | `lora_a` | `lora_b` | 合并动作 |
|---|---|---|---|
| `none` | 完整（复制） | 完整（复制） | 取 canonical，各副本必须"同一个值" |
| `out`（权重按行切） | 完整（复制） | 行块 `[row_start,row_stop)` | 行块 scatter 回完整 |
| `rows`（按头跳行切） | 完整（复制） | `row_indices` 行集（**非连续**） | `index_copy` 回位 |
| `in`（权重按列切） | 列块 `[col_start,col_stop)` | 完整（复制） | 列块 scatter 回完整 |

层归属（PP 谁持有哪些层 / 谁持 embedding、lm_head）**不推断**：读分片 `placement`，
再用 `parallel/placement_plan.build_placement_plan`（该语义的唯一真相源）**复算对照**。

### 键名映射
**优先用元数据里的 `hf_module_path`**（落盘时记录的真实 HF 路径）。旧通道按
`"visual." → "…model.visual."` 拼，会得到 `…model.visual.visual.blocks…`（多一个
`visual`）⇒ peft 加载时**静默丢键**、视觉 LoRA 无声失效。新通道据实拼接，用例
`test_visual_tower_key_uses_hf_module_path_not_double_prefix` 把"旧错新对"同时钉住。

---

## 2. 支持矩阵（当前代码实测，不是设计意图）

| 拓扑 | 训练侧能产出真件？ | 合并器能拼？ | 实测结论 |
|---|---|---|---|
| `tp=1 dp=1 pp=1` | ✅ | ✅ | 旧通道，**逐字节不变**（回归锁） |
| `tp=2 dp=1 pp=1` | ✅ | 结构✅ / **放行❌** | 620 键里 174 个复制张量漂移超限 ⇒ fail-closed |
| `tp=4 dp=1 pp=1` | ✅ | 同上 | 同上（真件 4 分片） |
| `tp=2 dp=2 pp=1` | ✅ | 同上 | 同上（DP 副本**逐位相同**✅，卡在视觉塔） |
| **`tp=1 pp=2`（LoRA）** | ✅ | ✅ | ★ **真件端到端跑通**：2 分片 → 310 模块 / 620 键，两 stage `placement_cross_check=ok`、复制张量 `max=0 ulp`；`transformers` + `peft` 加载后 **620 个 LoRA 模块全部挂上**（= 620 键，无静默丢键），前向 `logits[1,12,248320]` 正常、无 NaN；并写出可发布的完整 HF 目录（18 GB） |
| `tp=2 pp=2`（LoRA） | ✅ | 结构✅ / **放行❌** | 真件 4 分片：层归属与本契约全过，最后卡在视觉塔复制漂移（`max\|Δ\|=4.029e-05`、78687/110592 元素超限）⇒ fail-closed（缺陷①） |
| `pp>1`（全参） | ✅ 有真件 | ❌ **显式拒绝** | 全参分片**不带** `lora_tensor_metadata`，拼回需第二套布局推断 → 天然违反"不许推断" |

---

## 3. 切片拼接：已证到**逐位**（真张量，不是合成数据）

用真件（1 步 `tp=2` checkpoint，真训练器产出）做**契约回环**：把合并产物按同一元数据
切片 → 与 **native 运行时里该 rank 实际持有的 LoRA 参数**逐位比对。

| 比对对象 | 结果 |
|---|---|
| `tp_rank=0`（canonical 副本） | **620 / 620 张量逐位相同** |
| `tp_rank=1` | **446 / 620 逐位相同**；其余 **174 个全部是"复制"类张量**（`in`→B、`out`→A、`rows`→A、`none`→A/B），差 **3.6e-7 ~ 4.0e-5**（亚 ulp ~ 个位 ulp） |

174 与合并回执里的 `n_not_bitwise=174` **完全一致** ⇒
① 三类切片（列块 / 行块 / 非连续行集）的还原**逐位正确**；
② 复制类张量确实各 rank 不同（第 4 节的缺陷），合并器只能忠于 canonical 那一份。

**为什么用参数级回环而不是前向 `max|Δ|`**：同一输入、同 dtype 下，**HF 基座（不挂任何
adapter）与 native 分片模型的前向本身**就差 `max|Δ|=1.125`（`mean|Δ|=0.023`，仅 10% 元素
相同）—— 两个运行时的实现差异（位置编码/注意力 kernel 等）构成的"底噪"比任何要判定的
效应都大。在这个底噪下报一个前向 `max|Δ|≤1e-3` 只会是自欺。故：
- 切片/复制的正确性 → 用**参数级逐位回环**证（上表）；
- 前向级等价 → 记为**未达成**（见第 6 节），并给出底噪实测值，供后续用"同运行时配对"
  的 harness 继续做。

### 3.1 ★ `pp>1` 分片的命名契约（真件实测，容易踩的坑）

`pp_rank=1` 的分片里 **`module_name` 用的是本 stage 内的局部编号**：

```
module_name      = "layers.0.mlp.gate_proj"                       ← 局部：本 stage 第 0 层
hf_module_path   = "model.language_model.layers.17.mlp.gate_proj"  ← 全局：模型第 17 层
placement.local_layer_indices = [17 … 31]
```

⇒ 两个 stage 都会出现 `layers.0.mlp.down_proj` 这种**同名不同层**。若按下标名聚合，
会把 layer 0 与 layer 17 静默并成一个（比崩溃更糟）。合并器因此：
**模块身份一律取 `hf_module_path`**；无它时退化为 `f"{stage偏移}:{module_name}"`，
⛔ 绝不用裸名当键；并校验 `stage偏移 + 局部层号 == hf_module_path 的全局层号`，不符即 fail-closed。

### 3.2 ★ (B) 同运行时配对 harness：模型级等价的**正判据**（底噪严格为 0）

跨运行时前向对比不可用（底噪 1.125，见 §6.2）。改用**配对设计**：两侧都跑在**同一个
native 运行时、同一位置编码、同一 dtype**，唯一变量是 LoRA 权重从哪来 ——

| 侧 | LoRA 来源 |
|---|---|
| A | `load_checkpoint(真分片)`：各 rank 的原分片 |
| B | 同一个模型，再用**产品加载器** `load_peft_adapter_into_native_model()` 把**合并产物**重新切片灌回去 |

⇒ B 与 A 的前向差 = "分片 → 合并 → 导出 PEFT → 加载回运行时"整条闭环的误差。

真件实测（1 步 `tp=2`，2 进程，固定 12 token 输入）：

| 口径 | `max\|Δ\|` | `mean\|Δ\|` | 逐位相同比例 | 判定 |
|---|---|---|---|---|
| **底噪 N1**：同一模型重复前向 | **0.0** | 0.0 | 100% | 确定性成立 |
| **对照组**：先把 420 个复制张量按 canonical 对齐，再走完整闭环 | **0.0** | 0.0 | **100%** | ★ **`BITWISE_EQUAL`** ⇒ 闭环无损、harness 灵敏 |
| **待判**：真实未对齐分片走同一闭环 | **1.5** | 0.0212 | 11.1% | 相对信号 1.7%、≈151 ulp，**显著** |

三条结论：
1. **闭环无损已被证明**（对照组逐位相同）：合并 + 导出 + 重灌回运行时**不引入任何误差**。
2. **底噪严格为 0** ⇒ 任何非零差异都只能是权重差异，判据可判定（区别于跨运行时的 1.125）。
3. ★ **1 ulp 容差并不等于"模型可忽略"**：复制张量上 4.0e-05 的漂移（亚 ulp~个位 ulp）
   经 32 层网络被放大到 hidden 的 **1.7%**、`max\|Δ\|=1.5`。⇒ **`--require-bitwise` 才是
   唯一可辩护的验收口径**；1 ulp 过渡口径只能用来"继续做诊断"，不能用来"放行产物"。

---

## 4. 复制张量为什么拼不了：两个训练侧缺陷（本次实测定位）

1 步 `tp=2` 真跑（真训练器）后逐类统计跨 tp rank：

| 类 | 声明 | 1 步后 |
|---|---|---|
| `out`/`rows` 的 `lora_a` | 复制（TP SUM 同步） | **逐位相同**（88/88、48/48） |
| `none` 的 `lora_a` | 复制（**无任何同步**） | 逐位相同（该步梯度为 0） |
| `in` 的 `lora_b` | 复制（TP SUM 同步） | 非逐位，但 `max|Δ|=4.77e-7`（亚 ulp，cos=1.0000） |
| `none` 的 `lora_b`（视觉塔） | 复制（**无任何同步**） | **0/110**，cos≈0，`max|Δ|=4.0e-5` ⇒ **1 步就各写各的** |

- **缺陷①（硬）**：`shard_kind="none"`（视觉塔）的 LoRA **完全没有跨 rank 梯度同步**——
  `lora_linear.py::_sync_nonsharded_lora_grads` 只有 `in`/`out`/`rows` 三个分支，**没有
  `none` 分支**；`dp=1` 时 DP 同步也不覆盖。⇒ 视觉塔 LoRA 在 `tp>1` 下每个 rank 独立训练，
  checkpoint 里**不存在**"唯一那个值"。
- **缺陷②（软/累积）**：SUM all-reduce 后两 rank 末位不同（亚 ulp），经 bf16 权重数百步
  复利放大（400 步真件上 `out/rows` 的 A 约 0.7%、`in` 的 B 约 24%、视觉塔 B 约 140%）。

**DP 副本是好的**：实测 DP 两个副本**逐位相同**（relFro 恰好 0），`_set_model_init_seed`
＋ DP AVG 同步有效；合并器对 DP 副本仍做逐位校验，不同则拒。

---

## 5. fail-closed 覆盖（33 例单测里的"必须拒"清单）

分片缺失（列名）· 切片空洞 · 切片重叠 · 切片范围与张量形状不符 · 复制张量超 1 ulp ·
`--require-bitwise` 下 1 ulp 也拒 · 未知 `shard_kind` · 元数据缺失 · 模块落在别的 PP stage ·
层归属重叠/空洞 · manifest 缺失或非法 · 拓扑与文件集合不符 · 分片内 `tp/dp/pp/rank` 与文件名
或 manifest 不符 · DP 副本不一致 · DP 副本元数据不一致 · 全参分片（显式拒绝并说明理由）。

## 6. 回归与红→绿

- 旧回归：`pytest tests/tools tests/scripts --ignore=tests/scripts/test_sharded_merge.py`
  ⇒ **3 failed / 424 passed**，红名与基线**逐字相同**（`test_entry_scripts_smoke_all_covered_and_broken_zero`、
  `test_t023_artifact_carries_final_formats`、`test_effective_set_matches_the_approved_list`）。
- 新套件：`pytest tests/scripts/test_sharded_merge.py` ⇒ **33 passed**。
- 红→绿（两段证据，均保留原文）：
  1. **能力缺失**（源码回退到 HEAD、测试保留）⇒ `ModuleNotFoundError: No module named
     'graspo.eval.sharded_merge'`（1 error）；
  2. **朴素实现**（不校验分片齐备 / 不拼切片 / 不查复制一致性）⇒ **9 failed, 21 passed**
     （切片还原、分片缺失、复制漂移、拓扑不符、CLI fail-closed 全部红）。
  恢复源码后 `sha256sum -c` 校验三文件一致，再跑 ⇒ 33 passed。

---

## 7. 未做与已知限制（**请连同第 4 节一起读**）

1. **部分达成：真件端到端导出**。
   - ✅ **`tp=1 pp=2` 真件已走通**：合并 → `peft` 加载（**620/620 模块全部挂上**）→
     前向正常 → 写出 18 GB 完整 HF 目录，可交付。
   - ❌ **`tp>1` 真件仍导不出**：视觉塔 LoRA 从第 1 步起各 rank 不同（缺陷①），
     按主席裁定**必须拒收**，⛔ 不做"挑一个 rank"的合并。
   - ❌ **vLLM 起服未做**：训练侧 rootfs 与 A800 宿主机**均未安装 vLLM**
     （`ModuleNotFoundError: No module named 'vllm'`），vLLM 只在 228 的镜像里；
     A800 与 228 之间无可写共享卷（`/data/remote` 在 A800 上不存在），
     19 GB 产物跨机搬运需要单独的窗口。⇒ **环境限制，非代码问题**，请安排。
2. **未达成：前向级 `max|Δ|` 数值等价**。底噪实测 `max|Δ|=1.125`（HF 基座 vs native，
   同输入同 dtype，无 adapter），比待判定的效应大两个数量级。要做成，需要"同运行时配对"
   的 harness（两侧都走同一份位置编码/注意力实现），本次未做。
3. **未做：`pp>1` 的模型级等价**。`pp>1 + LoRA` 早先"起不来"是**用错了树**（缺陷 N2 在
   `85dcab8` 已修）；换到修复后的树后 `tp=1 pp=2` 与 `tp=2 pp=2` **都能正常产出真件**
   （已实测 rc=0）。模型级等价仍缺"同运行时配对"harness（见第 2 条）。我用真实训练张量
   按 `placement_plan` 重切分只做了**结构测试**（单测
   `test_pp_stage_layers_concatenated` / `test_tp_and_pp_combined_restored_exactly`），
   ⛔ 不当作等价证明。
4. **未支持：全参（`native-full-param`）分片合并**——分片不带 LoRA 元数据，拼回需要第二套
   布局推断，与「唯一真相源」冲突，故显式拒绝（不是漏做）。
5. **`alpha` 口径**：native `scaling = alpha / r`（真件实测 `r=32、alpha=64` ⇒ 2）。旧通道
   在无元数据时 `alpha=r` 只是兜底；新通道**取元数据真值**。若旧 checkpoint 无 alpha 字段，
   仍退回 `alpha=r`（并在 `adapter_config._note` 里注明为推导值）。
6. **合并器未做张量级峰值内存实测**（轴 7）：实现上已用
   `torch.load(..., mmap=True)` 逐分片懒加载 + 逐模块拼接，但**本次没有测出真实峰值 RSS**，
   故不给结论。
7. **`docs/GRASPO训练框架功能测试报告.html` 未改动**：sha256 `b8ed080d…`、77,079 字节，
   与冻结值一致（改动前后各验一次）。
8. **影响面核查（两个导出 bug 是否波及已入表档）—— 可自证的三条判据**：
   1. 旧通道 `convert_native_rank_to_peft` 的**唯一可达路径**是
      `graspo eval export` → `convert_native_checkpoint_to_peft`（`src/graspo/eval/merged_export.py:974`、
      `src/graspo/cli/eval_commands.py:114`），输入必须是 `CheckpointKind.NATIVE_RANK` 分片。
   2. ms-swift 后端目录 `src/graspo/flow/backend/msswift/` 内对该通道的引用数 = **0**：
      `grep -rn "convert_native_rank_to_peft\|convert_native_checkpoint_to_peft\|NATIVE_RANK" \
        --include=*.py src/graspo/flow/backend/msswift/ | wc -l` ⇒ `0`。
      ms-swift 的产物由 ms-swift 自己写进 `output_dir/msswift`（标准 HF/PEFT），不经过分片通道。
   3. `docs/capability-matrix-20261025.data.json` 里已出结论的 **21** 档 =
      **20 个 `backend="ms-swift"` + 1 个 `native`（M011）**；M011 训练即崩
      （`ValueError: lora.target_modules is required`，实跑 0 步、未产出 checkpoint）⇒ 没走到导出。
      ⇒ **两个 bug 的影响面 = 0 档**（它们只作用于 native 分片通道，而 native 档至今无一带结论）。
9. `gate_content` 报 6 条阻断项，**全部来自 HEAD 既有文件**（我新增/修改的文件 0 违规）。
