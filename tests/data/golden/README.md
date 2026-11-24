# 冻结的历史实现（golden 基准）

这里的 `.py` **不是生产代码，也不是可维护的代码**：它们是某个历史版本实现的原样副本，
只用来当「修复前会算错」的**负向断言基准**。

- ⛔ 不要为了风格/类型/覆盖率去改它们，也不要把它们 `import` 进 `scripts/` 或 `src/`。
- ⛔ 不要用 `git show <sha>:<path>` 去历史里取同款实现：2026-11-24 做过一次全历史改写
  （把内网信息从所有历史版本里抹掉），**所有提交的 sha 都变了、旧 commit 对象已不在对象库**。
  新 clone 只有 HEAD 可达的对象，按旧 sha 取数必失败 —— 这正是这些文件存在的原因。
- 只在 `.gitignore` 与 `pyproject.toml`（ruff `extend-exclude`）里排除静态检查，不排除入库。

| 冻结副本 | 来源 | 字节 | sha256 | 谁在用 |
|---|---|---|---|---|
| `step_readings_pre_pp_probe_filter.py` | `scripts/step_readings.py` @ 旧 sha `312e095`（2026-11-12，全历史改写前的 sha，现已不可解析） | 8 493 | `b8447a0c9bfdb0fff3e9023a7511cb390a4bf84e937a2ab29fb324c35a65a353` | `tests/scripts/test_step_readings_pp_probe_filter.py` |
| `make_result_json_pre_native_step.py` | `scripts/make_result_json.py` @ 旧 sha `924f01a`（2026-10-27，全历史改写前的 sha，现已不可解析） | 114 771 | `28ce9931f704cde6f73237e3e3f692ecbe63ed480f43a786a977ee853281562b` | `tests/scripts/test_n3b_native_step_single_source.py` |

`step_readings.py` 那份**未脱敏、逐字节原文**（原文里没有任何内网标识）。

## 脱敏说明（`make_result_json_pre_native_step.py`）

该文件**不是逐字节原文**：它含 1 处内网标识，已脱敏后再入库。

- 位置：第 1259 行，`_crashed_tier_evidence` 的 docstring 里（**注释性文字，不是代码、不参与任何比较**）。
- 改法：`（M011 实证，228 实测 2026-10-25）` → `（M011 实证，远端机实测 2026-10-25）`，
  与当前 `scripts/make_result_json.py` 同一行的措辞一致。
- 影响：全文件 **1 行不同**（114 766 → 114 771 字节，+5）。`parse_steps` / `_scan_steps_text` /
  `GLOBAL_STEP_RE` 的**字节与行为一字未变** —— 已用备份原件与冻结副本对同一批输入
  （7 种 ms-swift 写法 + M007/M008/M016 三份真件）逐条比对 `(steps, why)`，全部相同。
