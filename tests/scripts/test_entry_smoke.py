#!/usr/bin/env python3
"""入口脚本**冒烟全覆盖**（B4/§(1) 常设检查）。

## 为什么需要它

B3 的冒烟抓出 `scripts/gpu_lock_guard.py` 自 P0 起加载错模块（`run.sh:114` 直接
`AttributeError` 拒起），而当时**三个闸门都没抓到**——说明闸门对"宿主侧入口/启动
脚本"有覆盖漏洞：单测只覆盖被 import 的函数，对拍台只覆盖 Ripple 数值，边界检查
只覆盖 import 图。本文件补这一层。

## 判据（逐脚本，可复跑）

- **Python 入口**：能 `--help` 且 `rc=0`；无 argparse 的入口退化为**语法/编译冒烟**；
- **Shell 入口**：`bash -n` 语法检查；可 `--help` 者跑一次并期望 `rc=0`；可
  `--dry-run` 者跑一次，接受"有据的 fail-closed"（环境根未注入时按设计拒起）；
- **输出**：`候选 N（py X / sh Y），坏了 B 个` + 逐个原因；`B != 0` 即测试失败。

## 发现规则与排除理由（★ 新的入口脚本不会被静默漏掉）

候选由 `git ls-files` **自动发现**：所有 `*.sh`、`scripts/*.py`、`docker/*.py`、
`tools/*.py`、`tests/e2e/{,rig_synth/}*.py`（去掉 pytest 测试件）。发现到的每个路径
必须在本文件的 `POLICY` 或 `EXCLUSIONS` 里有**具名**条目，否则测试**当场失败**
（fail-closed：新增入口必须显式登记冒烟方式，不得漏检）。

排除（`EXCLUSIONS`，附理由）：
- `tests/**/test_*.py`：pytest 测试件，由单测闸门覆盖，不是"启动脚本"；
- `scripts/errors.py`：纯模块（无 `__main__`、无 CLI），被单测 import；
- `tools/gate_common.py`：门禁库（无 `__main__`），由 6 个 gate 入口间接覆盖；
- `src/**`：库代码，由边界闸门 + 单测覆盖；
- `tests/e2e/rig_synth/test_*.py`：同上，pytest 测试件。

不作为"常设冒烟对象"但**已在 POLICY 里具名**的（附理由，均仍做 `bash -n` /
语法检查）：
- `docker/build.sh` / `docker/proxy-env.sh`：构建期脚本，跑一次会真去 build/需要
  BuildKit 环境；语法检查 + 构建门禁（`check_env_versions.py`）已覆盖其核心；
- `scripts/eval_serve_vllm.sh`：会起 vLLM 并占 GPU（红线：不占 GPU）；
- `tests/e2e/run_matrix.sh`：位置参数为 output_dir，跑一次会真跑矩阵；
- `tests/e2e/rig_synth/verify_build.sh`：需要注入宿主 rig 根，跑一次会 verify 真实产物。
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
TIMEOUT_SEC = 180

# ── 冒烟模式 ────────────────────────────────────────────────────────────────
HELP = "python-help"  # python <path> --help ⇒ rc==0
COMPILE = "python-compile"  # 只编译（无 argparse 的入口）⇒ rc==0
RUN0 = "python-run"  # 直接跑（忽略 --help 的入口）⇒ rc==0
SH_HELP = "shell-help"  # bash -n + bash <path> --help ⇒ rc==0
SH_SYNTAX = "shell-syntax"  # bash -n ⇒ rc==0
SH_FAILCLOSED = "shell-failclosed"  # bash -n + 跑一次，接受有据的 fail-closed


@dataclass(frozen=True)
class Spec:
    mode: str
    reason: str
    argv: tuple[str, ...] = ()
    #: None = 必须是"有据的 fail-closed"（即 rc ≠ 0）；否则必须落在该集合内。
    expect_rc: tuple[int, ...] | None = (0,)
    expect_in: str | None = None


#: 入口/启动脚本 → 冒烟方式（**每个候选都必须在此**）。
POLICY: dict[str, Spec] = {
    # ── 仓库根 / 宿主启动链 ────────────────────────────────────────────────
    "run.sh": Spec(SH_HELP, "训练启动主入口；支持 --help（宿主引导层稳定契约）"),
    # ── scripts/*.py（§10.2 制度化脚本，B3 已登记稳定子命令）────────────────
    "scripts/build_mini_elam_subset.py": Spec(HELP, "宿主侧数据工具；argparse CLI"),
    "scripts/check_env_versions.py": Spec(HELP, "镜像/构建期版本门禁入口"),
    "scripts/collect_results.py": Spec(HELP, "矩阵结果收集入口"),
    "scripts/convert_targets_format.py": Spec(
        COMPILE, "无 argparse 的位置参数 CLI（argv 直读）⇒ 语法/可加载冒烟"
    ),
    "scripts/env_snapshot.py": Spec(HELP, "环境快照记录入口"),
    "scripts/gpu_idle_assert.py": Spec(HELP, "F-10 空闲断言入口（ticket ⑤ 的 UUT）"),
    "scripts/gpu_lock_guard.py": Spec(HELP, "run.sh:114 锁卡守卫（B3 修过 P0 后损坏）"),
    "scripts/measure_elam_seq_len.py": Spec(HELP, "ELAM 序列长度测量入口"),
    # ── tools/matrix/ 文档工具链入口（功能矩阵 SSOT 的产出/校验链）────────
    "tools/matrix/render_matrix.py": Spec(HELP, "矩阵 HTML 的唯一渲染通道（模板+数据⇒HTML）"),
    "tools/matrix/verify_matrix.py": Spec(HELP, "矩阵执法：重渲染+逐字节比对+结构/内容/脱敏断言"),
    "tools/matrix/build_official.py": Spec(HELP, "正式机读件（脱敏）构建入口"),
    "tools/matrix/build_data.py": Spec(HELP, "内部数据构建入口（列定义+统一口径+逐档参数）"),
    "tools/matrix/extract_params.py": Spec(HELP, "逐档参数抽取入口"),
    "tools/matrix/extract_spec.py": Spec(HELP, "逐档规格抽取入口（含冲突/缺项登记）"),
    "tools/matrix/backfill_matrix.py": Spec(HELP, "跑次结果回填入口（增量/fail-closed）"),
    "tools/matrix/run_self_proof.py": Spec(HELP, "三组自证入口（空数据/假数据/手改 1 字符）"),
    "tools/matrix/scan_matrix_leaks.py": Spec(HELP, "矩阵脱敏扫描（§17.7 禁止项：宿主路径/内网/机器名/人名/账号）"),
    # ── shell 工具/运维入口 ────────────────────────────────────────────────
    "scripts/eval_cleanup.sh": Spec(SH_HELP, "评测清理入口；支持 -h/--help ⇒ rc=0"),
    "scripts/eval_export_checkpoint.sh": Spec(SH_HELP, "评测导出入口；-h 显式 exit 0"),
    "scripts/eval_serve_vllm.sh": Spec(
        SH_SYNTAX, "容器生命周期脚本，跑一次会起 vLLM 并占 GPU（红线：不占 GPU）"
    ),
    "docker/build.sh": Spec(SH_SYNTAX, "构建期脚本，跑一次会真 build 镜像（本轮 ⛔ 不跑构建）"),
    "docker/proxy-env.sh": Spec(SH_SYNTAX, "被 build.sh source 的代理注入片段，无独立 CLI"),
    "docker/gate_script.py": Spec(
        COMPILE, "构建期双向诚实门禁（Dockerfile RUN 调用），非镜像环境跑必 fail-closed"
    ),
    # ── tests/e2e 生成器/启动器（宿主侧实验链入口）──────────────────────────
    "tests/e2e/generate_matrix.py": Spec(HELP, "54 档矩阵生成器（v1 基线）"),
    "tests/e2e/generate_matrix_fsdp.py": Spec(HELP, "matrix54-fsdp 变体生成器"),
    "tests/e2e/generate_matrix_manifest_fsdp.py": Spec(HELP, "fsdp 清单生成器"),
    "tests/e2e/generate_matrix_manifest_source.py": Spec(HELP, "v2 清单生成器"),
    "tests/e2e/gpu_assignment.py": Spec(HELP, "卡位分配校验入口"),
    "tests/e2e/make_subsets.py": Spec(HELP, "subsets-v2 冻结件生成器（B1 并入 E-49/E-50）"),
    "tests/e2e/pp_rollout_smoke.py": Spec(HELP, "PP rollout 冒烟入口"),
    "tests/e2e/run_matrix54.sh": Spec(
        SH_FAILCLOSED,
        "矩阵主启动器；--dry-run 在环境根未注入时按 §15.1/§16 设计 fail-closed",
        argv=("T001", "--dry-run"),
        expect_rc=None,
        expect_in="GRASPO_ELAM_HOST_ROOT",
    ),
    "tests/e2e/run_matrix54_source.sh": Spec(
        SH_FAILCLOSED,
        "v2source 逐档 launcher；--dry-run 同样要求宿主根注入，缺失即拒起",
        argv=("T028", "--dry-run"),
        expect_rc=None,
        expect_in="GRASPO_ELAM_DATA_ROOT",
    ),
    "tests/e2e/run_matrix.sh": Spec(
        SH_SYNTAX, "旧矩阵 runner；位置参数 output_dir，跑一次会真跑矩阵"
    ),
    "tests/e2e/rig_synth/build_synth_seq.py": Spec(HELP, "合成长序列基线构建器"),
    "tests/e2e/rig_synth/emit_configs.py": Spec(HELP, "合成长序列配套配置生成器"),
    "tests/e2e/rig_synth/emit_manifests.py": Spec(HELP, "合成长序列清单生成器"),
    "tests/e2e/rig_synth/measure_synth_tokens.py": Spec(HELP, "合成长序列 token 测量入口"),
    "tests/e2e/rig_synth/verify_synth_artifacts.py": Spec(HELP, "合成长序列产物校验入口"),
    "tests/e2e/rig_synth/run_synth.sh": Spec(
        SH_FAILCLOSED,
        "合成长序列 runner；无 --variant 时按设计拒起（race-free 参数守卫）",
        argv=(),
        expect_rc=None,
        expect_in="必须给 --variant",
    ),
    "tests/e2e/rig_synth/verify_build.sh": Spec(
        SH_SYNTAX, "需要注入宿主 rig 根，跑一次会 verify 真实产物（本机无该根）"
    ),
    # ── tools/ 门禁入口 ────────────────────────────────────────────────────
    # 注：这 6 个 gate 的 `-h/--help` 曾因 run_gate 把 SystemExit(0) 吞成 rc=2 而报错退出，
    # 已被本轮的入口冒烟抓出并修（tools/gate_common.py）。
    "tools/gate_commit_identity.py": Spec(HELP, "§15.1 提交身份门禁入口"),
    "tools/gate_content.py": Spec(HELP, "§15.1 内容门禁入口"),
    "tools/gate_filetypes.py": Spec(HELP, "§15.1 文件类型门禁入口"),
    "tools/gate_pii.py": Spec(HELP, "§15.1 PII 门禁入口"),
    "tools/gate_secrets.py": Spec(HELP, "§15.1 凭据门禁入口"),
    "tools/gate_negative_demo.py": Spec(
        RUN0, "门禁负向演示（忽略参数直接跑；全部用例必须产生真实失败）"
    ),
    "tools/readme_mirror_check.py": Spec(HELP, "§17.1 README 双语镜像检查入口"),
}

#: 具名排除（附理由；自动发现到的、落在此表的路径不参与冒烟）。
EXCLUSIONS: dict[str, str] = {
    "scripts/errors.py": "纯模块（无 __main__/无 CLI），被单测 import，不是启动脚本",
    "tools/gate_common.py": "门禁库（无 __main__），由 6 个 gate 入口间接覆盖",
}


def _discover() -> list[str]:
    """`git ls-files` 自动发现入口/启动脚本候选（见模块 docstring 的发现规则）。"""
    tracked = subprocess.run(
        ["git", "ls-files"], cwd=REPO, capture_output=True, text=True, check=True
    ).stdout.splitlines()
    out: list[str] = []
    for path in tracked:
        if path.endswith(".sh"):
            out.append(path)
        elif path.endswith(".py") and (
            path.startswith("scripts/")
            or path.startswith("docker/")
            or path.startswith("tools/")
            or (
                path.startswith("tests/e2e/")
                and Path(path).name != "__init__.py"
                and not Path(path).name.startswith("test_")
            )
        ):
            out.append(path)
    return sorted(set(out))


def _env_for(spec: Spec) -> dict[str, str]:
    env = dict(os.environ)
    if spec.mode == SH_FAILCLOSED:
        for key in (
            "GRASPO_ELAM_HOST_ROOT",
            "GRASPO_ELAM_DATA_ROOT",
            "GRASPO_PREGEN_SUBSET_DIR",
            "GRASPO_ATTEMPT_ROOT",
            "RUN_ROOT",
            "NVIDIA_VISIBLE_DEVICES",
        ):
            env.pop(key, None)
    return env


def _run(path: str, spec: Spec) -> subprocess.CompletedProcess[str]:
    py = sys.executable
    if spec.mode == COMPILE:
        code = (
            "import pathlib,sys;"
            "compile(pathlib.Path(sys.argv[1]).read_text(encoding='utf-8'),sys.argv[1],'exec')"
        )
        argv = [py, "-c", code, path]
    elif spec.mode in (HELP, RUN0):
        argv = [py, path, *(("--help",) if spec.mode == HELP else spec.argv)]
    elif spec.mode == SH_SYNTAX:
        return subprocess.run(
            ["bash", "-n", path], cwd=REPO, capture_output=True, text=True, check=False
        )
    elif spec.mode in (SH_HELP, SH_FAILCLOSED):
        syntax = subprocess.run(
            ["bash", "-n", path], cwd=REPO, capture_output=True, text=True, check=False
        )
        if syntax.returncode != 0:
            return syntax
        if spec.mode == SH_HELP:
            argv = ["bash", path, "--help"]
        else:
            argv = ["bash", path, *spec.argv]
    else:  # pragma: no cover - 配置错误直接炸
        raise AssertionError(f"未知冒烟模式 {spec.mode!r}（{path}）")
    return subprocess.run(
        argv,
        cwd=REPO,
        capture_output=True,
        text=True,
        check=False,
        env=_env_for(spec),
        timeout=TIMEOUT_SEC,
    )


def _verdict(path: str, spec: Spec, proc: subprocess.CompletedProcess[str]) -> str | None:
    """返回 None = 通过；否则返回人类可读的失败原因。"""
    if spec.expect_rc is None:
        if proc.returncode == 0:
            return "rc=0（期望有据的 fail-closed，即 rc≠0）"
    elif proc.returncode not in spec.expect_rc:
        got = proc.returncode
        if got < 0:
            got = f"信号 {-got}"
        tail = (proc.stderr or proc.stdout or "").strip().splitlines()
        where = tail[-1] if tail else "(无输出)"
        return f"rc={got}（期望 {'/'.join(map(str, spec.expect_rc))}）；最后一行：{where[:200]}"
    if spec.expect_in and not re.search(spec.expect_in, proc.stderr + proc.stdout):
        return f"rc={proc.returncode} 但输出未含 fail-closed 证据 {spec.expect_in!r}"
    if spec.mode in (HELP, SH_HELP):
        blob = (proc.stdout + proc.stderr).lower()
        if not re.search(r"usage|用法", blob):
            return "rc=0 但输出里没有 'usage'/'用法'（可能没有真的走到 --help）"
    return None


def smoke() -> tuple[list[tuple[str, Spec, str | None, int]], list[str]]:
    """跑全部候选，返回 (逐项结果, 未登记候选)。"""
    discovered = _discover()
    uncovered = [p for p in discovered if p not in POLICY and p not in EXCLUSIONS]
    results = []
    for path in discovered:
        spec = POLICY.get(path)
        if spec is None:
            continue
        proc = _run(path, spec)
        results.append((path, spec, _verdict(path, spec, proc), proc.returncode))
    return results, uncovered


def render(results, uncovered) -> tuple[str, int]:
    lines = ["", "=" * 78, "入口脚本冒烟全覆盖（B4/§(1)）", "=" * 78]
    broken = [r for r in results if r[2] is not None]
    py = sum(1 for _, s, _, _ in results if s.mode.startswith("python"))
    sh = len(results) - py
    for path, spec, why, rc in results:
        mark = "OK  " if why is None else "BROKEN"
        lines.append(f"[{mark}] {path:<52} {spec.mode:<18} rc={rc}")
        if why is not None:
            lines.append(f"         ↳ {why}")
    lines.append("-" * 78)
    lines.append(
        f"候选 {len(results)} 个（python {py} / shell {sh}），"
        f"坏了 {len(broken)} 个，未具名登记 {len(uncovered)} 个"
    )
    for path in uncovered:
        lines.append(f"  ✗ 未登记候选：{path}（新增入口必须写进 POLICY 或 EXCLUSIONS）")
    lines.append("=" * 78)
    return "\n".join(lines), len(broken) + len(uncovered)


def main() -> int:
    results, uncovered = smoke()
    text, bad = render(results, uncovered)
    print(text)
    return 1 if bad else 0


def test_entry_scripts_smoke_all_covered_and_broken_zero() -> None:
    """★ 常设闸门：入口脚本全冒烟 + 坏了 0 个 + 无未登记候选。"""
    results, uncovered = smoke()
    text, bad = render(results, uncovered)
    print(text)
    assert not uncovered, f"发现未具名登记的入口脚本：{uncovered}"
    broken = [(p, why) for p, _, why, _ in results if why is not None]
    assert not broken, f"入口脚本冒烟有 {len(broken)} 个坏了：{broken}"


if __name__ == "__main__":
    raise SystemExit(main())
