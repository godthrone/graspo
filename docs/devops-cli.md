# DevOps 稳定 CLI 入口（§10.2 制度化接口登记表）

本文档是 `scripts/` 下**已制度化**工具与它们的**稳定 CLI 入口**的登记处。宪法 §10.2
的判据：脚本一旦「被测试固化 / 被 README 或 docs 引用 / 被工作流反复使用」，即成为
制度化接口，必须升级为 CLI 子命令或删除。本表是 §10.2 处置的**单一登记点**。

## 1. 已升级为 `graspo` 稳定子命令（4 项）

实现唯一真相源仍是被登记脚本本身；`graspo <子命令>` 只是**同契约**的稳定入口，
参数**原样透传**（含 `--help`），因此不存在第二套参数表。

| 稳定子命令 | 实现（唯一真相源） | 参数契约 | 为何制度化（§10.2 命中判据） |
|---|---|---|---|
| `graspo gpu-lock-guard` | `scripts/gpu_lock_guard.py` | 见 `graspo gpu-lock-guard --help`（`--visible` / `--quiet` / `--assert-idle`） | 被 `run.sh:114`、`tests/conftest.py:53`（docstring）、`tests/e2e/*` 引用；工作流反复使用 |
| `graspo gpu-idle-assert` | `scripts/gpu_idle_assert.py` | 见 `graspo gpu-idle-assert --help`（`--visible` / `--quiet`） | 被 `tests/scripts/test_gpu_idle_assert.py` 专门测试固化 |
| `graspo check-env-versions` | `scripts/check_env_versions.py` | 见 `graspo check-env-versions --help`（`--observed-json` / `--local` / `--from-container` / `--python`） | 被 `tests/e2e/test_check_env_versions.py` 固化 + `pyproject.toml` + `docker/Dockerfile*` 构建门禁引用 |
| `graspo collect-results` | `scripts/collect_results.py` | 见 `graspo collect-results --help`（`--manifest` / `--run-root` / `--rerun-root` / `--repo-root` / `--out` / `--context-length` / `--date` / `--base-model-root`） | 被 `tests/e2e/test_collect_results.py`、`tests/core/test_result_judge_threshold.py`、`tests/compliance/test_oss_config_boundaries.py` 固化 + `docs/capability-matrix.md` 引用 |

登记实现在 `src/graspo/cli/devops.py`（`DEVOPS_ENTRIES` 是子命令名 → 脚本名的唯一映射点）。

## 2. 宿主 / 镜像引导层入口（脚本原址是稳定契约，不得迁址）

以上 4 个脚本**同时**服务于「尚未建 venv / 尚未安装 graspo」的引导时机（宿主
`run.sh` 的逐卡占用守卫、镜像构建期 version gate、矩阵 runner），此时只能走
`python3 scripts/<name>.py`。因此：

- **仓内路径 `scripts/<name>.py` 是稳定契约**，`run.sh:114`、`docker/Dockerfile:92`、
  `docker/Dockerfile.msswift:89,240` 的调用**不得改路径**；
- `graspo <子命令>` 只是「已安装环境」这一侧的同契约入口，两处调同一份实现。

## 3. 运维 / Docker 层入口（不登记为 python 子命令）

| 入口 | 处置 | 理由 |
|---|---|---|
| `scripts/eval_serve_vllm.sh` | **保留**，登记为**运维/Docker 层稳定入口**（§10.2「归入 docker/运维入口」） | 它是**容器生命周期**脚本（起/停 vLLM 服务），由 `src/graspo/cli/eval_commands.py` 组装命令、`tests/eval/test_cli_eval.py` 断言其名，属运维边界而非 python CLI 参数契约；`scripts/eval_cleanup.sh` 同层（只打印命令） |
| `scripts/eval_export_checkpoint.sh` | 同层保留 | 与上者同属评估链路的运维脚本 |

## 4. 已退役（§10.2「或删除」）

| 对象 | 处置 | 依据 |
|---|---|---|
| `scripts/a4_calibrate.py` | **⛔ 已作废 / 已退役**：移入 `.local/repo-moved-out/scripts/a4_calibrate.py` 后 `git rm` | **A4 容差体系已整体作废**（2026-10-17 冻结口径；`docs/a4-tolerance-calibration.md` 已同步标为历史件）⇒ 标定工具随之作废，该体系**不再参与任何通过 / 不通过判定**。现行判据以唯一真相源 `docs/capability-matrix.data.json`（人读件 `docs/capability-matrix.html`）为准。其契约测试 `tests/core/test_a4_calibrate.py` 同批退役（移入 `.local/repo-moved-out/tests/core/`），否则会按路径加载已不存在的被测件 |
| `scripts/measure_elam_seq_len.py` | **保留**（未制度化：无测试固化、无 README/docs 引用、非工作流常驻） | §10.2 仅约束制度化脚本；本件不属本批升级/退役清单 |
