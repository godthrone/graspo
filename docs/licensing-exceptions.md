# 依赖许可证例外登记（具名）

> **宪法依据**：§14.1「所有依赖包必须是 MIT、Apache 2.0 或等价的可商用宽松许可证。
> 禁止 GPL、AGPL 等传染性许可证。」
>
> **本文件是什么**：本项目**唯一**的依赖许可证**具名例外登记件**。凡运行时/构建时依赖中
> 出现非宽松（专有）许可证的包，必须在本表**逐包具名**：包名 / 版本 / 许可证表达式 /
> **形态** / 允许理由。泛述（"CUDA wheels 里是 NVIDIA 专有库"）**不满足**登记要求。
>
> **本文件不是什么**：不是许可证全文，也不是 `LICENSE`（本仓库自身 = MIT）的替代。
> 各包许可证原文随包分发（`.dist-info/METADATA` / 包内 `LICENSE`）。

## 0. 判定口径（三条闸门）

1. **可商用**：许可证允许商业使用（本项目的全部 NVIDIA 专有依赖均满足）。
2. **非传染**：无 copyleft/传染条款，不要求本项目开源派生作品（专有 EULA 满足）。
3. **未 vendor 进仓库**：依赖只以包管理器产物（wheel）存在，**其二进制/源码不进本仓库**。
   复核命令（结果 = **0 命中**）：
   ```
   git ls-files | grep -iE 'nvidia|/cuda|cuda-'          # → 空
   git ls-files | grep -iE '\.(so|whl)$'                 # → 未命中 NVIDIA/CUDA 二进制
   ```
   这些包只存在于 `.venv/`（`.gitignore` 已排除）。⇒ **不存在"被 vendor ⇒ 必须移除"的情形**。

许可证名来源：`.venv/lib/python*/site-packages/<pkg>.dist-info/METADATA` 的
`License-Expression:` / `License:` / `Classifier: License ::`。版本取自 `uv.lock`。

## 1. 需具名登记的例外（NVIDIA 专有，16 件）

> 全部为 `torch==2.11.0` 的**传递依赖**（`pyproject.toml` 无直接声明）；删除/手改会破坏
> `uv.lock` 解析与镜像构建（§14.2 禁止手改 `uv.lock`）。
> 形态 = 该 wheel 实际交付的内容物（"运行时二进制"指共享库/头文件，不含源代码）。

| # | 包 | 版本 | 许可证表达式 | 形态 | 允许理由 |
|--:|---|---|---|---|---|
| 1 | `cuda-bindings` | 13.3.1 | `LicenseRef-NVIDIA-SOFTWARE-LICENSE` | **Python 绑定包**（29 个 `.so` + `.py`/`.pxd`） | 可商用、非传染、仅运行时依赖、未 vendor 进仓库 |
| 2 | `cuda-toolkit` | 13.0.2 | （`METADATA` 无 License 字段）**元包** | **元包**（聚合 `cublas/cudart/…` extras，自身无代码） | 同上；许可证随其子包同 EULA |
| 3 | `nvidia-cublas` | 13.1.0.3 | `LicenseRef-NVIDIA-Proprietary` | 二进制 wheel（运行时 BLAS） | 可商用、非传染、仅运行时依赖、未 vendor 进仓库 |
| 4 | `nvidia-cuda-cupti` | 13.0.85 | `LicenseRef-NVIDIA-Proprietary`（+ `Classifier: Other/Proprietary`） | 二进制 wheel（运行时 profiling） | 同上 |
| 5 | `nvidia-cuda-nvrtc` | 13.0.88 | `LicenseRef-NVIDIA-Proprietary`（+ `Other/Proprietary`） | 二进制 wheel（运行时 JIT 编译） | 同上 |
| 6 | `nvidia-cuda-runtime` | 13.0.96 | `LicenseRef-NVIDIA-Proprietary` | 二进制 wheel（运行时 core） | 同上 |
| 7 | `nvidia-cudnn-cu13` | 9.19.0.56 | `LicenseRef-NVIDIA-Proprietary` | 二进制 wheel（运行时 cuDNN） | 同上 |
| 8 | `nvidia-cufft` | 12.0.0.61 | `LicenseRef-NVIDIA-Proprietary`（+ `Other/Proprietary`） | 二进制 wheel（运行时 FFT） | 同上 |
| 9 | `nvidia-cufile` | 1.15.1.6 | `LicenseRef-NVIDIA-Proprietary`（+ `Other/Proprietary`） | 二进制 wheel（运行时文件 IO 加速） | 同上 |
| 10 | `nvidia-curand` | 10.4.0.35 | `LicenseRef-NVIDIA-Proprietary`（+ `Other/Proprietary`） | 二进制 wheel（运行时 RNG） | 同上；版本号 `10.4.0.35` 撞私网段属 §15.1 已登记假阳性 |
| 11 | `nvidia-cusolver` | 12.0.4.66 | `LicenseRef-NVIDIA-Proprietary`（+ `Other/Proprietary`） | 二进制 wheel（运行时 solver） | 同上 |
| 12 | `nvidia-cusparse` | 12.6.3.3 | `LicenseRef-NVIDIA-Proprietary`（+ `Other/Proprietary`） | 二进制 wheel（运行时 sparse） | 同上 |
| 13 | `nvidia-cusparselt-cu13` | 0.8.0 | `NVIDIA Proprietary Software` | 二进制 wheel（运行时 sparse-LT） | 同上 |
| 14 | `nvidia-nccl-cu13` | 2.28.9 | `LicenseRef-NVIDIA-Proprietary` | 二进制 wheel（运行时集合通信） | 同上 |
| 15 | `nvidia-nvjitlink` | 13.0.88 | `LicenseRef-NVIDIA-Proprietary`（+ `Other/Proprietary`） | 二进制 wheel（运行时 JIT 链接） | 同上 |
| 16 | `nvidia-nvshmem-cu13` | 3.4.5 | `LicenseRef-NVIDIA-Proprietary` | 二进制 wheel（运行时 SHMEM） | 同上 |

## 2. 不需例外，但此处一并具名（2 件）

| # | 包 | 版本 | 许可证表达式 | 形态 | 结论 |
|--:|---|---|---|---|---|
| 17 | `cuda-pathfinder` | 1.5.6 | **`Apache-2.0`** | 纯 Python wheel（29 个 `.py`，无二进制） | 宽松许可，§14.1 白名单内，**不需例外** |
| 18 | `nvidia-nvtx` | 13.0.85 | **`License: Apache 2.0`**，但 `Classifier: License :: Other/Proprietary License` **冲突** | 二进制 wheel（头文件 + 共享库） | 以 `License:` 字段判为 Apache-2.0，**不需例外**；⚠ classifier 冲突宜后续向 NVIDIA 官方 LICENSE 原文裁定 |

## 3. 待构建环境核查后具名补登（5 件）

`docker/Dockerfile.msswift` 用 pip 额外安装下列 NVIDIA 相关包；它们**不在 `uv.lock`、
不在本地 `.venv`**，故本仓库侧无法读取其 `dist-info/METADATA`，**本轮无法取得许可证**，
登记为**待补登**（⛔ 本轮不跑构建）：

| # | 包 | 版本（Dockerfile 声明） | Dockerfile 行 | 状态 |
|--:|---|:--:|:--:|---|
| 19 | `nvidia-cudnn-frontend` | 1.25.0 | `:151` | **待构建环境核查后具名补登** |
| 20 | `nvidia-cutlass-dsl` | 4.5.2 | `:152` | 同上 |
| 21 | `nvidia-cutlass-dsl-libs-base` | 4.5.2 | `:223` | 同上 |
| 22 | `cuda-tile` | 1.3.0 | `:223` | 同上 |
| 23 | `nvidia-ml-py` | 13.595.45 | `:213` | 同上 |

补登方法（构建环境执行）：`pip show <pkg>` 或读
`site-packages/<pkg>.dist-info/METADATA` 的 `License-Expression:` / `License:`，
按本文件 §1 的列格式追加具名条目（若为宽松许可则落到 §2）。

## 4. 构建门禁条件（文档化；本轮不执行构建）

> ★ **镜像内所有包都已具名登记** = 下述条件全部满足，否则**不得发布镜像**：

1. 镜像内 `pip list` 的全部包名 ⊆（本文件 §1 ∪ §2 ∪ §3 补登后的具名清单）；
2. §3 的 5 件**已从"待补登"转为 §1/§2 具名条目**（含许可证表达式与形态）；
3. §0 第 3 条的 vendor 复核在待发布提交上重跑，结果仍为 **0 命中**；
4. 任何新增的非宽松依赖，必须在本文件新增具名条目后才可进入镜像。

## 5. 变更记录

- 2026-10-11 建件（宪法合规化 B4）：§14.1 具名例外登记；16 件专有 + 2 件宽松 +
  5 件待构建核查；修正 README"all runtime dependencies are permissively licensed"的
  字面不成立表述，并让两份 README 指向本文件。
