"""54 档运行器 `run_matrix54.sh` 的端到端复核（**假 docker + 假 nvidia-smi + 假目录**，
不触 GPU、不起容器）。

红线：本模块只创建临时目录、只把假的 `docker` / `nvidia-smi` 可执行文件放进 `PATH`；
不调用真实 docker、不读真实模型/数据、不碰任何 GPU。

覆盖审查 🔴-1 的两条验收：
  ① 缺模型目录时 **fail-closed**（exit 4）且报错文本指明"模型未挂载 / 运行链路错误"；
  ② 模型目录存在时，传给 docker 的 argv 里出现正确的只读 `-v "$MODELS_ROOT:/models:ro"`。

**F-6 追加**：`blocked` 档在 `--dry-run` 下也必须 **rc≠0**（旧版可表达性检查排在
dry-run 分支之后 ⇒ 预检把不可跑的档报成 rc=0）。

**F-10 追加**：宿主侧目标卡空闲断言在真实代码路径上生效——目标卡被占时拒绝启动
（用假 `nvidia-smi` 构造"卡上有 299 MiB 他人占用"场景）。

**F-A 追加（本模块新增主线）**：挂载结构不变量。
旧命令 `-v "$ELAM_HOST:/work/elam-v5:ro"` + `-v "$RUN_DIR/subsets:/work/elam-v5/subsets:ro"`
把子挂载点放在只读绑定挂载之下，dockerd 无法创建嵌套 mountpoint ⇒ **rc=125，一档都起不来**
（228 实测确证）。现在：① 挂载目标彼此互不嵌套；② 该不变量在启动前被断言（fail-closed）；
③ 目的地由 `data.train_path` 反推（单一真相源）。

**F-B 追加**：媒体锚点。子集只读挂到 `dirname(train_path)`，图像根只读挂到
`dirname(dirname(train_path))/images`（= 锚点 `../images` 的落点）。宿主侧缺图像根 ⇒
拒绝启动；容器内还有一道媒体预检，逐个断言相对媒体路径可达。

**F-IPC 追加**：多卡共享内存。`DOCKER_ARGS` 必须带 `--ipc=host --shm-size=16g`，
否则容器 `/dev/shm` 只有 docker 默认 64 MiB ⇒ **任何 ≥2 卡档位启动即 ENOSPC**
（228 实测：4 rank 同抛 `unable to allocate shared memory(shm)`，`Train: 0%`）。
配一条**变异测试**证明该断言真能失败，以及一条"runner 与生成器模板逐字节一致"的不变量。
"""

import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_RUNNER = _REPO_ROOT / "tests" / "e2e" / "run_matrix54.sh"
_MANIFEST = _REPO_ROOT / "tests" / "e2e" / "matrix54_manifest.json"

#: 无 torch 的裸解释器也能跑锁卡守卫（守卫按文件路径独立加载，不走包 __init__）。
_PYBIN = sys.executable

_FAKE_DOCKER = """#!/bin/bash
# 假 docker：只记录 argv，不启动任何容器、不读模型/数据。
{ printf '%s\\n' "$@"; } > "$FAKE_DOCKER_ARGV"
exit 0
"""

#: 假 docker（**模拟 dockerd 的 mountpoint 规则**）：某个挂载目标位于另一个**只读**
#: 挂载之下时返回 125 —— 这正是 228 上实测的失败（`mkdirat ...: read-only file system`）。
#: 用它来证明"旧挂载方案会失败、新挂载方案被接受"这条判据真的能失败。
_FAKE_DOCKER_MOUNT_RULES = """#!/bin/bash
{ printf '%s\\n' "$@"; } > "$FAKE_DOCKER_ARGV"
args=("$@")
targets=()
modes=()
for ((i = 0; i < ${#args[@]}; i++)); do
  if [ "${args[$i]}" = "-v" ]; then
    spec="${args[$((i + 1))]}"
    IFS=':' read -r -a parts <<< "$spec"
    src="${parts[0]}"
    dst="${parts[1]}"
    mode="${parts[2]:-rw}"
    for k in "${!targets[@]}"; do
      case "$dst/" in
        "${targets[$k]}/"*)
          if [ "${modes[$k]}" = "ro" ]; then
            echo "docker: Error response from daemon: mkdirat $dst: read-only file system" >&2
            exit 125
          fi
          ;;
      esac
    done
    targets+=("$dst")
    modes+=("$mode")
    unset src
  fi
done
exit 0
"""

#: 假 nvidia-smi：**F-10 空闲断言的真实代码路径**要在这里跑通。
#: 只回答 0 号卡（run_matrix 的 T010 只用一张卡），实测空闲（4 MiB / 0%）。
#: `-L` 给出容器内可见卡；`--query-gpu=...` 只接受带 `-i` 的查询。
#: 空闲查询的字段串必须与 `cli/gpu_monitor.IDLE_FIELDS` 一致（含 index）。
#: 环境变量 ``FAKE_NVIDIA_SMI_BUSY_MIB`` 可让 0 号卡"被占"（构造 F-10 负向场景）。
_FAKE_NVIDIA_SMI = """#!/bin/bash
args="$*"
case "$args" in
  "-L") echo "GPU 0: NVIDIA A800-SXM4-80GB (UUID: GPU-fake000)"; exit 0;;
esac
case "$args" in
  *--query-gpu=index,memory.used,utilization.gpu*)
    echo "0, ${FAKE_NVIDIA_SMI_BUSY_MIB:-4}, 0"; exit 0;;
  *--query-gpu=index,uuid*) echo "0, GPU-fake000, 4, 81916, 81920, 0, 40, 100.0"; exit 0;;
esac
exit 0
"""


@pytest.fixture(autouse=True)
def _require_manifest():
    if not _MANIFEST.is_file():
        pytest.skip("matrix54_manifest.json 不存在（先跑 tests/e2e/generate_matrix.py）")


def _make_fakes(tmp_path: Path, *, with_images: bool = True) -> tuple[Path, Path, Path]:
    """造 假 docker + 假 nvidia-smi + 假 ELAM 数据根 + 假模型根（含 9B 目录）。

    ``with_images=False`` 用来构造 F-B 的负向场景（宿主图像根缺失）。
    返回 (fake_bin_dir, fake_elam_root, fake_models_root)。
    """
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    for name, body in (("docker", _FAKE_DOCKER), ("nvidia-smi", _FAKE_NVIDIA_SMI)):
        fake = bin_dir / name
        fake.write_text(body, encoding="utf-8")
        fake.chmod(0o755)

    elam_root = tmp_path / "elam"
    (elam_root / "data").mkdir(parents=True)
    (elam_root / "data" / "train.jsonl").write_text(
        "".join(f'{{"id": {i}}}\n' for i in range(1, 201)), encoding="utf-8"
    )
    if with_images:
        (elam_root / "images").mkdir(parents=True)
        (elam_root / "images" / "fake.jpg").write_bytes(b"not-a-real-jpeg")

    models_root = tmp_path / "models"
    (models_root / "Qwen3.5-9B").mkdir(parents=True)
    return bin_dir, elam_root, models_root


def _stage_mini_jsonl(elam_root: Path, lines: int = 200) -> Path:
    """在假宿主根下按**生产落点**放一份 mini 数据集（默认读取源）。

    落点：``<ELAM_HOST>/mini-dataset/mini-short-mm-train.jsonl``（带用途前缀、独立目录，
    与 ``data/train.jsonl`` 物理隔离）。专供"默认读取源就是 mini"的正/负向用例使用。
    """
    mini_dir = elam_root / "mini-dataset"
    mini_dir.mkdir(parents=True, exist_ok=True)
    mini = mini_dir / "mini-short-mm-train.jsonl"
    mini.write_text("".join(f'{{"id": {i}}}\n' for i in range(1, lines + 1)), encoding="utf-8")
    return mini


def _mount_specs(argv: list[str]) -> list[tuple[str, str, str]]:
    """从 docker argv 里取出 ``-v`` 三元组 ``(源, 目标, 模式)``。"""
    specs: list[tuple[str, str, str]] = []
    for index, token in enumerate(argv):
        if token != "-v":
            continue
        src, dst, mode = (argv[index + 1].split(":") + ["rw"])[:3]
        specs.append((src, dst, mode))
    return specs


def _nested_under_readonly(specs: list[tuple[str, str, str]]) -> list[tuple[str, str]]:
    """返回所有"位于只读挂载之下"的 (目标, 只读父目标) 组合。"""
    bad: list[tuple[str, str]] = []
    for _, dst, _ in specs:
        for _, parent, mode in specs:
            if mode == "ro" and (dst == parent or dst.startswith(parent.rstrip("/") + "/")):
                if dst != parent:
                    bad.append((dst, parent))
    return bad


def _run(
    tmp_path: Path,
    bin_dir: Path,
    *,
    elam_root: Path,
    models_root: Path | None,
    extra_env: dict[str, str] | None = None,
    tier: str = "T010",
    mode: str | None = None,
    runner: Path = _RUNNER,
    path_prepend: Path | None = None,
) -> subprocess.CompletedProcess[str]:
    path = f"{bin_dir}{os.pathsep}{os.environ.get('PATH', '')}"
    if path_prepend is not None:
        path = f"{path_prepend}{os.pathsep}{path}"
    env = {
        **os.environ,
        "PATH": path,
        "FAKE_DOCKER_ARGV": str(tmp_path / "docker_argv.txt"),
        "RUN_ROOT": str(tmp_path / "runs"),
        "GRASPO_ELAM_HOST_ROOT": str(elam_root),
        # 默认读取源是 mini 数据集（见 generate_matrix.py）。本 fixture 造的是"源数据"
        # `data/train.jsonl`（200 行），因此用 runner 的公开覆盖点显式指过去 ——
        # 这些用例断言的是挂载/守卫/入口，不是取数来源；取数来源由
        # `test_default_train_source_is_the_mini_dataset` 专门覆盖（不动默认落点）。
        "GRASPO_ELAM_MINI_JSONL": str(elam_root / "data" / "train.jsonl"),
        "PYTHON": _PYBIN,
        **(extra_env or {}),
    }
    env.pop("GRASPO_MODELS_HOST_ROOT", None)
    env.pop("GRASPO_9B_HOST_DIR", None)
    env.pop("GRASPO_27B_HOST_DIR", None)
    if models_root is not None:
        env["GRASPO_MODELS_HOST_ROOT"] = str(models_root)
    argv = ["bash", str(runner), tier]
    if mode is not None:
        argv.append(mode)
    return subprocess.run(
        argv,
        cwd=str(_REPO_ROOT),
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


def _blocked_tier() -> str:
    """从 manifest 里取一个 ``blocked`` 档（配置不可表达，``config`` 为 null）。"""
    manifest = json.loads(_MANIFEST.read_text(encoding="utf-8"))
    for tier in manifest["tiers"]:
        if tier["status"] == "blocked" and not tier.get("config"):
            return str(tier["tier_id"])
    pytest.skip("manifest 里没有 blocked 档")
    raise AssertionError("unreachable")


def test_missing_models_root_is_fail_closed_and_not_a_data_problem(tmp_path: Path):
    """① 未设置模型根 ⇒ 拒绝启动、exit 4、报错文本明确指向"模型未挂载"。"""
    bin_dir, elam_root, _ = _make_fakes(tmp_path)

    completed = _run(tmp_path, bin_dir, elam_root=elam_root, models_root=None)

    assert completed.returncode == 4, completed.stderr
    assert "模型未挂载" in completed.stderr
    assert "GRASPO_MODELS_HOST_ROOT" in completed.stderr
    # 没进 docker ⇒ 不会产生一个会被日志分类器误判成"数据问题"的容器日志。
    assert not (tmp_path / "docker_argv.txt").exists()


def test_missing_model_directory_is_fail_closed_with_runtime_link_message(tmp_path: Path):
    """① 模型根存在但缺该档子目录 ⇒ 同样 fail-closed，并声明这是运行链路错误。"""
    bin_dir, elam_root, models_root = _make_fakes(tmp_path)
    shutil.rmtree(models_root / "Qwen3.5-9B")

    completed = _run(tmp_path, bin_dir, elam_root=elam_root, models_root=models_root)

    assert completed.returncode == 4, completed.stderr
    assert "FATAL(runtime-link): 模型未挂载" in completed.stderr
    assert "Qwen3.5-9B" in completed.stderr
    assert "不是数据问题" in completed.stderr
    assert not (tmp_path / "docker_argv.txt").exists()


def test_existing_model_volume_is_mounted_read_only_in_docker_argv(tmp_path: Path):
    """② 模型目录存在 ⇒ argv 里出现正确的只读 `-v <models_root>:/models:ro`。"""
    bin_dir, elam_root, models_root = _make_fakes(tmp_path)

    completed = _run(tmp_path, bin_dir, elam_root=elam_root, models_root=models_root)

    assert completed.returncode == 0, completed.stderr
    argv = (tmp_path / "docker_argv.txt").read_text(encoding="utf-8").splitlines()
    expected = f"{models_root}:/models:ro"
    assert expected in argv, argv
    # 只读：宿主模型目录不会被容器改写。
    assert argv[argv.index(expected) - 1] == "-v"
    # ★F-B：数据集目录与**由 train_path 反推出来的图像根**各自独立挂载（互不嵌套）。
    assert f"{elam_root}/images:/work/elam-v5/images:ro" in argv, argv
    assert f"{tmp_path}/runs/T010/subsets:/work/elam-v5/subsets:ro" in argv, argv
    # 旧方案的"整棵数据根 + 其下嵌套子集"必须已经消失（那是 rc=125 的根因）。
    assert f"{elam_root}:/work/elam-v5:ro" not in argv, argv
    # 仍然没有 `--gpus`（锁卡只走单一路径）。
    assert "--gpus" not in argv


def test_per_model_host_dir_override_is_honoured(tmp_path: Path):
    """个别档模型不在同一根下时，`GRASPO_9B_HOST_DIR` 覆盖生效且同样 fail-closed。"""
    bin_dir, elam_root, models_root = _make_fakes(tmp_path)
    override = tmp_path / "elsewhere" / "Qwen3.5-9B"
    override.mkdir(parents=True)

    completed = _run(
        tmp_path,
        bin_dir,
        elam_root=elam_root,
        models_root=models_root,
        extra_env={"GRASPO_9B_HOST_DIR": str(override)},
    )

    assert completed.returncode == 0, completed.stderr
    argv = (tmp_path / "docker_argv.txt").read_text(encoding="utf-8").splitlines()
    # 挂载点仍是模型根（容器内 /models 的单一真相源），覆盖只影响存在性断言。
    assert f"{models_root}:/models:ro" in argv, argv

    # 覆盖值不存在 ⇒ fail-closed。
    empty = tmp_path / "empty"
    empty.mkdir()
    completed = _run(
        tmp_path,
        bin_dir,
        elam_root=elam_root,
        models_root=empty,
        extra_env={"GRASPO_9B_HOST_DIR": str(empty / "nope")},
    )
    assert completed.returncode == 4
    assert "FATAL(runtime-link): 模型未挂载" in completed.stderr


# ── F-6：blocked 档在 dry-run 下也 rc≠0 ─────────────────────────────────────


def test_blocked_tier_is_rejected_in_dry_run(tmp_path: Path):
    """★F-6 回归：``--dry-run`` 对 blocked 档必须 rc≠0（旧版 rc=0，预检误判可跑）。"""
    bin_dir, elam_root, models_root = _make_fakes(tmp_path)
    tier = _blocked_tier()

    completed = _run(
        tmp_path,
        bin_dir,
        elam_root=elam_root,
        models_root=models_root,
        tier=tier,
        mode="--dry-run",
    )

    assert completed.returncode != 0, completed.stdout
    assert completed.returncode == 2
    assert "不可表达（blocked）" in completed.stderr
    # 不可表达档绝不允许进入"可跑"的 dry-run 输出，也不得启动任何容器。
    assert "[dry-run]" not in completed.stdout
    assert not (tmp_path / "docker_argv.txt").exists()


def test_blocked_tier_is_rejected_even_without_mount_env(tmp_path: Path):
    """blocked 判定**先于**挂载环境检查：不给数据/模型根也照样按 blocked 拒绝（rc=2）。

    顺序即语义——若 blocked 判定排在环境检查之后，缺根环境下的 blocked 档会被
    报成 rc=3/4（"运行链路错误"），掩盖"这档根本不可表达"这个真因。
    """
    bin_dir, _, _ = _make_fakes(tmp_path)
    tier = _blocked_tier()

    completed = _run(
        tmp_path,
        bin_dir,
        elam_root=tmp_path / "elam",
        models_root=None,
        tier=tier,
    )

    assert completed.returncode == 2, completed.stderr
    assert "不可表达（blocked）" in completed.stderr


def test_ready_tier_still_passes_dry_run(tmp_path: Path):
    """不放松反向：可跑档（T010）的 dry-run 仍然 rc=0（F-6 没有把正常路径堵死）。"""
    bin_dir, elam_root, models_root = _make_fakes(tmp_path)

    completed = _run(
        tmp_path,
        bin_dir,
        elam_root=elam_root,
        models_root=models_root,
        tier="T010",
        mode="--dry-run",
    )

    assert completed.returncode == 0, completed.stderr
    assert "[dry-run] T010" in completed.stdout


# ── F-10：目标卡实测空闲断言在运行器里生效 ──────────────────────────────────


def test_occupied_target_gpu_is_refused_before_docker(tmp_path: Path):
    """★F-10 回归：目标卡被他人占用（299 MiB）⇒ 拒绝启动 rc=1，且**不进 docker**。

    这是本轮实战里救过场的一次断言（实测拦下含被占用 GPU3 的 4 卡目标集）。
    """
    bin_dir, elam_root, models_root = _make_fakes(tmp_path)

    completed = _run(
        tmp_path,
        bin_dir,
        elam_root=elam_root,
        models_root=models_root,
        extra_env={"FAKE_NVIDIA_SMI_BUSY_MIB": "299"},
    )

    assert completed.returncode == 1, completed.stdout
    assert "实测非空闲" in completed.stderr
    assert "GPU0" in completed.stderr
    assert not (tmp_path / "docker_argv.txt").exists()


def test_container_entry_runs_idle_assertion_before_training(tmp_path: Path):
    """容器内入口（entry.sh）必须先做空闲断言（``--idle-only``）再 torchrun。"""
    bin_dir, elam_root, models_root = _make_fakes(tmp_path)

    completed = _run(
        tmp_path,
        bin_dir,
        elam_root=elam_root,
        models_root=models_root,
    )

    assert completed.returncode == 0, completed.stderr
    entry = (tmp_path / "runs" / "T010" / "entry.sh").read_text(encoding="utf-8")
    assert "--idle-only" in entry
    assert entry.index("--idle-only") < entry.index("torchrun")


def test_idle_assert_can_be_skipped_explicitly_for_scaffold_checks(tmp_path: Path):
    """显式跳过开关存在但必须**显式**（默认不跳过；真实上机不得设置）。"""
    bin_dir, elam_root, models_root = _make_fakes(tmp_path)

    completed = _run(
        tmp_path,
        bin_dir,
        elam_root=elam_root,
        models_root=models_root,
        extra_env={"GRASPO_SKIP_IDLE_ASSERT": "1", "FAKE_NVIDIA_SMI_BUSY_MIB": "299"},
    )

    assert completed.returncode == 0, completed.stderr
    # 🟡-8：跳过防线必须**留痕**（宪法 §3.2 透明退路）——不能只在日志里"没有输出"。
    assert "WARNING" in completed.stderr
    assert "GRASPO_SKIP_IDLE_ASSERT" in completed.stderr
    audit = tmp_path / "runs" / "skipped_guards.log"
    assert audit.is_file(), "跳过防线必须写环境指纹/产物，供事后审计"
    audit_text = audit.read_text(encoding="utf-8")
    assert "skip_idle_assert=1" in audit_text
    assert "tier=T010" in audit_text


def test_idle_assert_skip_leaves_no_trace_when_not_requested(tmp_path: Path):
    """对照：不跳过时**不得**出现 WARNING，也不得造出审计日志。"""
    bin_dir, elam_root, models_root = _make_fakes(tmp_path)

    completed = _run(tmp_path, bin_dir, elam_root=elam_root, models_root=models_root)

    assert completed.returncode == 0, completed.stderr
    assert "WARNING" not in completed.stderr
    assert not (tmp_path / "runs" / "skipped_guards.log").exists()


# ── F-A：挂载结构不变量（旧方案的 rc=125 必须无法再出现）────────────────────


def _source_runner(script: str) -> subprocess.CompletedProcess[str]:
    """在**库模式**下 source runner（只加载断言函数，不执行主流程）。"""
    return subprocess.run(
        ["bash", "-c", f'source "{_RUNNER}"; {script}'],
        env={**os.environ, "GRASPO_RUNNER_LIB_ONLY": "1"},
        capture_output=True,
        text=True,
        check=False,
    )


def test_mount_invariant_rejects_a_target_nested_under_a_readonly_mount():
    """★F-A 负向：只读父挂载 + 其下子挂载 ⇒ 必须**被拒绝**并给出可操作报错。

    这是 228 实测的 rc=125 场景（`mkdirat .../subsets: read-only file system`）。
    旧 runner 会让 docker 返回一个语义模糊的 125；现在由启动前断言拦下（rc=6）。
    """
    completed = _source_runner(
        "assert_mount_targets_not_under_readonly 'H|/work/elam-v5|ro' 'R|/work/elam-v5/subsets|ro'"
    )

    assert completed.returncode == 6, completed
    assert "只读" in completed.stderr
    assert "/work/elam-v5/subsets" in completed.stderr
    assert "rc=125" in completed.stderr  # 报错必须点明 docker 侧的后果


def test_mount_invariant_accepts_the_sibling_layout():
    """★F-A 正向：互不嵌套（兄弟）布局必须放行。"""
    completed = _source_runner(
        "assert_mount_targets_not_under_readonly 'H|/work/elam-v5/images|ro' "
        "'R|/work/elam-v5/subsets|ro' 'R2|/out' 'C|/workspace/graspo|ro'"
    )

    assert completed.returncode == 0, completed.stderr


def test_fake_docker_with_real_mount_semantics_rejects_the_old_layout(tmp_path: Path):
    """★F-A 反向对照：假 docker 复刻 mountpoint 规则 ⇒ 旧挂载组合必须 125。

    这条证明上面的"新布局被接受"不是自说自话：同一个模拟器对旧命令报 125。
    """
    bin_dir, elam_root, models_root = _make_fakes(tmp_path)
    fake = bin_dir / "docker"
    fake.write_text(_FAKE_DOCKER_MOUNT_RULES, encoding="utf-8")
    fake.chmod(0o755)

    # 新 runner 的挂载组合 ⇒ 被接受。
    completed = _run(tmp_path, bin_dir, elam_root=elam_root, models_root=models_root)
    assert completed.returncode == 0, completed.stderr

    # 旧组合（父 ro + 其下子挂载）⇒ 该模拟器返回 125（= 228 实测行为）。
    old = subprocess.run(
        [
            str(fake),
            "--rm",
            "-v",
            f"{elam_root}:/work/elam-v5:ro",
            "-v",
            f"{tmp_path}/runs/T010/subsets:/work/elam-v5/subsets:ro",
            "image",
            "bash",
            "/entry.sh",
        ],
        capture_output=True,
        text=True,
        check=False,
        env={**os.environ, "FAKE_DOCKER_ARGV": str(tmp_path / "old_argv.txt")},
    )
    assert old.returncode == 125, old
    assert "read-only file system" in old.stderr


def test_default_train_source_is_the_mini_dataset(tmp_path: Path):
    """★ 默认读取源 = mini 数据集（生产落点 `<ELAM_HOST>/mini-dataset/mini-short-mm-train.jsonl`）。

    断言三件事：
      1. 默认（`--dry-run`）**不**要求源数据 `data/train.jsonl` 存在 —— 取数已切到 mini；
      2. 正式跑时真的从 mini 取数：子集内容 = mini 前 N 行（这里 200 行 ≠ 源数据内容）；
      3. 从而 `data.train_path`（生成位置）与挂载目的地**逐字不变**（仍是 <RUN_ROOT>/T010/subsets）。
    """
    bin_dir, elam_root, models_root = _make_fakes(tmp_path)
    # 移除源数据文件：若 runner 还在读它，就会在这里失败（防"看起来切了其实没切"）。
    (elam_root / "data" / "train.jsonl").unlink()
    mini = _stage_mini_jsonl(elam_root, lines=200)
    # 覆盖掉 _run 的 fixture 默认，还原生产默认行为（由 runner 自己拼 mini 路径）。
    no_override = {"GRASPO_ELAM_MINI_JSONL": ""}

    completed = _run(
        tmp_path,
        bin_dir,
        elam_root=elam_root,
        models_root=models_root,
        mode="--dry-run",
        extra_env=no_override,
    )
    assert completed.returncode == 0, completed.stderr
    assert "训练子集读取源不存在" not in completed.stderr

    completed = _run(
        tmp_path,
        bin_dir,
        elam_root=elam_root,
        models_root=models_root,
        extra_env=no_override,
    )
    assert completed.returncode == 0, completed.stderr
    subset = tmp_path / "runs" / "T010" / "subsets" / "T010.jsonl"
    assert subset.is_file()
    # 子集确实是 mini 的前 N 行（此档 SUBSET=100，见 manifest.data.subset_size），不掺源数据。
    assert subset.read_text(encoding="utf-8").splitlines() == (
        mini.read_text(encoding="utf-8").splitlines()
    )[:100]
    # 生成位置/挂载目的地不变（这正是"不改 data.train_path"的原因）。
    argv = (tmp_path / "docker_argv.txt").read_text(encoding="utf-8").splitlines()
    assert f"{tmp_path}/runs/T010/subsets:/work/elam-v5/subsets:ro" in argv, argv


def test_missing_mini_dataset_is_fail_closed_before_docker(tmp_path: Path):
    """★ 负向：默认读取源缺失 ⇒ 拒绝启动（rc=3）、给出两条可操作修法、且不进 docker。

    防呆目的：不能让"mini 没上传到 228"表现成训练期的数据问题（会被日志分类器误判）。
    """
    bin_dir, elam_root, models_root = _make_fakes(tmp_path)  # 不 staging mini 数据集

    completed = _run(
        tmp_path,
        bin_dir,
        elam_root=elam_root,
        models_root=models_root,
        extra_env={"GRASPO_ELAM_MINI_JSONL": ""},
    )

    assert completed.returncode == 3, completed.stderr
    assert "训练子集读取源不存在" in completed.stderr
    assert "mini-dataset/mini-short-mm-train.jsonl" in completed.stderr
    assert "GRASPO_ELAM_MINI_JSONL" in completed.stderr
    assert not (tmp_path / "docker_argv.txt").exists()


def test_runner_mount_destinations_follow_the_config_train_path(tmp_path: Path):
    """★F-B：挂载目的地由 `data.train_path` 反推（不是硬编码常量）。

    断言：子集目录 = dirname(train_path)，图像根 = dirname(dirname(train_path))/images。
    这样"子集放在哪"与"锚点是什么"就是同一条表达式，布局变化自动跟随。
    """
    bin_dir, elam_root, models_root = _make_fakes(tmp_path)

    completed = _run(tmp_path, bin_dir, elam_root=elam_root, models_root=models_root)
    assert completed.returncode == 0, completed.stderr

    argv = (tmp_path / "docker_argv.txt").read_text(encoding="utf-8").splitlines()
    specs = _mount_specs(argv)
    destinations = {dst for _, dst, _ in specs}
    assert destinations == {
        "/workspace/graspo",
        "/models",
        "/work/elam-v5/images",
        "/work/elam-v5/subsets",
        "/out",
        "/entry.sh",
    }, specs
    # 结构不变量：没有任何目的地落在只读挂载之下。
    assert _nested_under_readonly(specs) == []


def test_missing_image_root_is_fail_closed_before_docker(tmp_path: Path):
    """★F-B 负向：宿主图像根缺失 ⇒ 拒绝启动（rc=3）并声明这是运行链路错误。"""
    bin_dir, elam_root, models_root = _make_fakes(tmp_path, with_images=False)

    completed = _run(tmp_path, bin_dir, elam_root=elam_root, models_root=models_root)

    assert completed.returncode == 3, completed.stderr
    assert "媒体锚点不可达" in completed.stderr
    assert "运行链路错误" in completed.stderr
    # 不得进 docker：否则训练会抛 FileNotFoundError/PIL 错误，被分类器记成「数据问题」。
    assert not (tmp_path / "docker_argv.txt").exists()


def test_dry_run_also_rejects_missing_image_root(tmp_path: Path):
    """dry-run 是预检入口：图像根缺失时也必须 rc≠0（不能只在正式 run 里拦）。"""
    bin_dir, elam_root, models_root = _make_fakes(tmp_path, with_images=False)

    completed = _run(
        tmp_path,
        bin_dir,
        elam_root=elam_root,
        models_root=models_root,
        mode="--dry-run",
    )

    assert completed.returncode == 3, completed.stderr
    assert "媒体锚点不可达" in completed.stderr


def test_dry_run_reports_derived_anchor_paths(tmp_path: Path):
    """对照：图像根存在时 dry-run 仍 rc=0，并打印反推出来的锚点路径。"""
    bin_dir, elam_root, models_root = _make_fakes(tmp_path)

    completed = _run(
        tmp_path,
        bin_dir,
        elam_root=elam_root,
        models_root=models_root,
        mode="--dry-run",
    )

    assert completed.returncode == 0, completed.stderr
    assert "[dry-run] images=" in completed.stdout
    assert "/work/elam-v5/images" in completed.stdout


def test_media_preflight_script_fails_closed_when_the_anchor_is_unreachable(tmp_path: Path):
    """★F-B 负向（容器内那道防线）：媒体预检对锚点不可达必须 rc≠0。

    runner 会把预检脚本写进运行目录并经 `/out` 进容器；这里直接把生成的脚本
    对合成数据跑一遍（正/负两例），证明它是**真能失败**的断言而不是装饰。
    """
    bin_dir, elam_root, models_root = _make_fakes(tmp_path)
    completed = _run(tmp_path, bin_dir, elam_root=elam_root, models_root=models_root)
    assert completed.returncode == 0, completed.stderr
    preflight = tmp_path / "runs" / "T010" / "preflight_media.py"
    assert preflight.is_file(), "runner 必须把媒体预检脚本写进运行目录"

    # 正例：子集在 <root>/subsets/，图像在 <root>/images/（与锚点口径一致）。
    root = tmp_path / "case_ok"
    (root / "subsets").mkdir(parents=True)
    (root / "images").mkdir(parents=True)
    (root / "images" / "a.jpg").write_bytes(b"jpeg")
    (root / "subsets" / "T.jsonl").write_text(
        json.dumps(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": [{"type": "image", "image": "../images/a.jpg"}],
                    }
                ]
            }
        )
        + "\n",
        encoding="utf-8",
    )
    ok = subprocess.run(
        [sys.executable, str(preflight), str(root / "subsets" / "T.jsonl")],
        capture_output=True,
        text=True,
        check=False,
    )
    assert ok.returncode == 0, ok
    assert "media anchor OK" in ok.stdout

    # 负例 1：锚点错（子集被放到 <root>/data/v2/ ⇒ `../images` 落到 <root>/data/images）。
    deep = root / "data" / "v2"
    deep.mkdir(parents=True)
    (deep / "T.jsonl").write_text((root / "subsets" / "T.jsonl").read_text(), encoding="utf-8")
    bad_anchor = subprocess.run(
        [sys.executable, str(preflight), str(deep / "T.jsonl")],
        capture_output=True,
        text=True,
        check=False,
    )
    assert bad_anchor.returncode == 3, bad_anchor
    assert "媒体锚点不可达" in bad_anchor.stdout

    # 负例 2：图像缺失。
    (root / "images" / "a.jpg").unlink()
    missing = subprocess.run(
        [sys.executable, str(preflight), str(root / "subsets" / "T.jsonl")],
        capture_output=True,
        text=True,
        check=False,
    )
    assert missing.returncode == 3, missing
    assert "运行链路错误" in missing.stdout


def test_container_entry_runs_media_preflight_before_training(tmp_path: Path):
    """容器内入口顺序：媒体预检 → 空闲断言 → torchrun（预检必须最先）。"""
    bin_dir, elam_root, models_root = _make_fakes(tmp_path)

    completed = _run(tmp_path, bin_dir, elam_root=elam_root, models_root=models_root)
    assert completed.returncode == 0, completed.stderr

    entry = (tmp_path / "runs" / "T010" / "entry.sh").read_text(encoding="utf-8")
    assert "preflight_media.py" in entry
    assert entry.index("preflight_media.py") < entry.index("--idle-only")
    assert entry.index("--idle-only") < entry.index("torchrun")


def test_container_entry_makes_artifacts_host_readable(tmp_path: Path):
    """★产物必须可被**宿主侧** collect_results.py 读到：entry.sh 收尾要放开读权限。

    228 实测：容器内训练以 root 运行，ms-swift/HF 落盘的 ``adapter_model.safetensors``
    是 ``root:0600`` ⇒ 宿主侧普通用户打不开 ⇒ A2/A3 读不到证据（A3 一度被记成
    "checkpoint 无法重新加载"，把采集侧缺口记成了训练失败）。故收尾必须有 chmod。
    """
    bin_dir, elam_root, models_root = _make_fakes(tmp_path)

    completed = _run(tmp_path, bin_dir, elam_root=elam_root, models_root=models_root)
    assert completed.returncode == 0, completed.stderr

    entry = (tmp_path / "runs" / "T010" / "entry.sh").read_text(encoding="utf-8")
    assert "chmod -R a+rX /out" in entry
    # 必须在训练之后（此时产物才存在），且在退出码捕获之后（不影响 RC）。
    assert entry.index("torchrun") < entry.index("chmod -R a+rX /out")
    assert entry.index("RC=") < entry.index("chmod -R a+rX /out")


# ── IPC/shm：多卡启动的共享内存（阻断缺陷回归 + 变异测试）──────────────────
#
# 缺陷（228 实测确证）：`DOCKER_ARGS` 没有 `--ipc=host` / `--shm-size` ⇒ 容器
# `/dev/shm` 只有 docker 默认的 64 MiB ⇒ **任何 ≥2 卡档位启动即失败**：
#   4 个 rank 同抛 `RuntimeError: unable to allocate shared memory(shm) for file
#   </torch_491_*>: Resource temporarily unavailable (11)`（`Train: 0%`）。
# 分类：④运行链路缺陷（不是 OOM，也不是模型/数据问题）。

#: 变异目标——生成器模板里声明共享内存的那一行（删掉它必须让断言失败）。
_SHM_LINE = "    --ipc=host --shm-size=16g\n"


def _shm_args(argv: list[str]) -> list[str]:
    """从 docker argv 里取出共享内存配置，规整成 ``--flag=value`` 形态。

    同时接受 `--ipc=host`（带 `=``）与 `--ipc host`（分写）两种写法——
    断言要拦的是"缺配置"，不是"写法不同"。
    """
    found: list[str] = []
    for index, token in enumerate(argv):
        if token in ("--ipc", "--shm-size"):
            value = argv[index + 1] if index + 1 < len(argv) else ""
            found.append(f"{token}={value}")
        elif token.startswith("--ipc=") or token.startswith("--shm-size="):
            found.append(token)
    return found


def _assert_shm_configured(argv: list[str]) -> list[str]:
    """契约：传给 docker 的 argv 必须带共享内存配置。

    抽成独立函数，是为了让变异测试能**直接调用同一个断言**并证明它会失败——
    否则"把参数删掉、测试仍然通过"的空断言无法被排除（见下方负向测试）。
    """
    shm = _shm_args(argv)
    assert shm, (
        "docker argv 缺共享内存配置（--ipc=host 或 --shm-size=<n>）："
        "容器 /dev/shm 仅 docker 默认 64 MiB，≥2 卡的 torchrun 必然 ENOSPC。"
        f"argv={argv}"
    )
    return shm


def test_docker_argv_carries_shared_memory_config(tmp_path: Path):
    """★IPC 正向：docker argv 必须带共享内存配置，且它落在 image 之前（当作 run 选项）。"""
    bin_dir, elam_root, models_root = _make_fakes(tmp_path)

    completed = _run(tmp_path, bin_dir, elam_root=elam_root, models_root=models_root)

    assert completed.returncode == 0, completed.stderr
    argv = (tmp_path / "docker_argv.txt").read_text(encoding="utf-8").splitlines()
    shm = _assert_shm_configured(argv)
    # 与仓库既有约定逐字一致（run.sh / run_matrix.sh）：`--ipc=host --shm-size=16g`。
    assert "--ipc=host" in shm, shm
    assert "--shm-size=16g" in shm, shm
    # 必须是 `docker run` 的选项，不能沦为容器命令的一部分（否则 docker 直接报错）。
    assert argv.index("--ipc=host") < argv.index("bash")
    # 仍然没有 `--gpus`（锁卡只走单一路径）。
    assert "--gpus" not in argv


def _mutant_runner(tmp_path: Path, *, remove: str) -> Path:
    """造一个**删掉了共享内存行**的 runner 副本（只变异这一处，其余逐字相同）。

    变异体必须落在 `tests/e2e/` 下：runner 用 `dirname($0)/../..` 反推仓库根
    （见 `ROOT_DIR`），换到 `tmp_path` 会让 GPU 守卫脚本与挂载源指向错误位置。
    文件名带 `tmp_path.name` 保证唯一，调用方负责在 `finally` 里删除。
    """
    text = _RUNNER.read_text(encoding="utf-8")
    assert remove in text, f"变异目标不存在，测试前提失效：{remove!r}"
    mutant = _REPO_ROOT / "tests" / "e2e" / f"_mutant_run_matrix54_{tmp_path.name}.sh"
    mutant.write_text(text.replace(remove, "", 1), encoding="utf-8")
    mutant.chmod(0o755)
    return mutant


def test_shm_assertion_really_fails_when_the_parameter_is_removed(tmp_path: Path):
    """★IPC 负向（变异测试）：删掉 `--ipc=host --shm-size=16g` ⇒ 同一个断言必须失败。

    这条证明上面的正向断言**不是空断言**——参数一旦缺失，测试确实转红。
    """
    bin_dir, elam_root, models_root = _make_fakes(tmp_path)
    mutant = _mutant_runner(tmp_path, remove=_SHM_LINE)
    try:
        completed = _run(
            tmp_path, bin_dir, elam_root=elam_root, models_root=models_root, runner=mutant
        )
        # 变异体本身仍能跑完（删参数不会让脚本报错）——所以只有断言能拦住它。
        assert completed.returncode == 0, completed.stderr
        argv = (tmp_path / "docker_argv.txt").read_text(encoding="utf-8").splitlines()
        assert _shm_args(argv) == [], f"变异未生效，argv 仍带共享内存配置：{argv}"
        # 同一个断言在变异体上抛 AssertionError ⇒ 正向测试真的能失败。
        with pytest.raises(AssertionError, match="共享内存"):
            _assert_shm_configured(argv)
    finally:
        mutant.unlink(missing_ok=True)
    assert not mutant.exists(), "变异体必须清理干净（不得留在已跟踪目录里）"


def test_runner_matches_the_generator_template_byte_for_byte(tmp_path: Path):
    """★不变量：`run_matrix54.sh` 与 `generate_matrix.render_runner()` 必须逐字节一致。

    runner 是**生成物**（文件头写着"手改无效"）：有人直接改 runner 而忘了改模板，
    下次重新生成就会把修复回退掉——这正是本缺陷的复发路径。sha256 相同即证据。
    """
    spec = importlib.util.spec_from_file_location(
        "matrix54_generate_matrix", _REPO_ROOT / "tests" / "e2e" / "generate_matrix.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    rendered = module.render_runner()
    on_disk = _RUNNER.read_text(encoding="utf-8")
    assert on_disk == rendered, "run_matrix54.sh 与生成器模板不一致（手改过 runner？）"
    assert (
        hashlib.sha256(on_disk.encode("utf-8")).hexdigest()
        == hashlib.sha256(rendered.encode("utf-8")).hexdigest()
    )
    # 模板里也必须带共享内存配置（否则重新生成即复发）。
    assert _SHM_LINE in rendered


# ── ckpt 保留策略：三态语义 + 静默失败防线（本轮加固）────────────────────────
#
# 背景（先有反证，再有加固）：曾有人报"保留块里 heredoc 变量写成 `\$TIER` ⇒
# `OUTPUT_ROOT` 是字面量 `/out/$TIER` ⇒ 保留策略永不生效"。**该诊断被实测推翻**：
# `<<ENTRY` 不加引号 ⇒ 外层把 `\$TIER` 写成**字面量 `$TIER`**（交给内层展开），
# 产物逐字是 `OUTPUT_ROOT="/out/$TIER"`，由**内层**展开——不是"永不展开"。
#
# 但复核发现**真隐患**（本轮加固的正是它）：
#   ① 保留块原先靠"容器内 `TIER` 恰好未定义 ⇒ 空串 ⇒ `/out/`（递归 find 仍命中）"
#      才歪打正着生效 ⇒ 一旦有人给容器注入 `TIER`，保留策略会**静默失效且 0 阻断**。
#   ② `state=none`（产物目录不存在）原先**静默跳过、rc=0** —— 保留策略悄悄失效，
#      磁盘迟早爆。这正是本项目一直在打的"静默失败"。
#
# 加固：① 外层 `docker run -e "TIER=$TIER"` **显式**注入档号（§2 显式即防呆）；
#      ② 保留失败一律**硬失败**：`missing`/`none` 在训练成功（RC=0）时 `exit 4`；
#         ——唯一例外是训练本身失败（RC≠0）：此时保留原 rc，**不掩盖训练失败**。
#
# 三态语义（必须写在测试里，便于审查者对照）：
#   `ok`      —— 找到 ≥1 份**可判产物**（ms-swift 的 `checkpoint-*`，或 native 的**非空**
#                `final/`）并只留最新一份，删除动作成功。**唯一通过态**。
#                注：native 只落一份终态 ⇒ 保留动作"无事可做"，`删除数=0`（N2，2026-09-20）。
#   `missing` —— 产物目录存在，但**一份可判产物都没有**（既无 `checkpoint-*`，也无非空
#                `final/`）。A3 无证据可判
#                ⇒ 运行链路错误（save_steps > 实际优化步数？）⇒ **硬失败**。
#   `none`    —— 产物目录不存在。RC≠0 时合法（训练失败，保留原 rc）；
#                RC=0 时**硬失败**（产物通道/档号定位坏了）。
# ★ N2 只修了"native 的合法形态被误判成 missing"这一种**假失败**；
#   空 `final/` 与"什么都没有"**仍然** `missing`（三态语义未放松）。

#: 保留块的起点/终点标记。**注意**：`run_matrix54.sh` 是**未展开的模板源**（heredoc
#: 正文里内层变量写作 `\$var`）；真正进容器的 `entry.sh` 是外层 heredoc 展开后的形态。
#: 本测试要断言的是**后者**（容器实际执行的那份），故必须照做一次外层展开。
_RETENTION_START = "retain_single_checkpoint() {"
#: 终点 = runner 源码里那一行 `printf ... > /out/ckpt_retention.state`（**逐字取自源码**，
#: 不手写转义，避免 heredoc 的两层 `\$` / `\\n` 转义写错）。
_RETENTION_END_LINE_PREFIX = "printf '%s"
_RETENTION_END_LINE_SUFFIX = "ckpt_retention.state"


def _retention_end_marker(runner: Path | None = None) -> str:
    """从 runner 源码里取出终点行（含行尾换行），逐字、不手写转义。"""
    text = (runner or _RUNNER).read_text(encoding="utf-8")
    for line in text.splitlines(keepends=True):
        if line.startswith(_RETENTION_END_LINE_PREFIX) and _RETENTION_END_LINE_SUFFIX in line:
            return line
    raise AssertionError("runner 里找不到保留块的终点行（ckpt_retention.state）")


def _heredoc_expand(body: str) -> str:
    """照 `cat > entry.sh <<ENTRY`（**不加引号**）的规则展开 heredoc 正文。

    外层只做参数展开 / 命令替换：反斜杠-美元 ⇒ 字面量美元（交给内层）。
    """
    return body.replace(chr(92) + "$", "$").replace(chr(92) + "`", "`")


def _entry_final_exit_line(runner: Path | None = None) -> str:
    """从 runner 源码里取出 entry.sh 的**收口行**（`exit "$RC"`，含行尾换行）。

    ★ 这是"容器退出码 = torchrun 退出码"的唯一表达点（原缺陷：块跑完没人 exit ⇒ 退 0）。
    逐字取自源码并做一次 heredoc 外层展开（`\\$RC` ⇒ `$RC`），避免手写转义写错。
    """
    text = (runner or _RUNNER).read_text(encoding="utf-8")
    for line in text.splitlines(keepends=True):
        if line.strip().startswith("exit ") and "$RC" in line:
            return _heredoc_expand(line)
    raise AssertionError('runner 的 entry.sh 里找不到收口行 `exit "$RC"`')


def _retention_block_text(runner: Path | None = None) -> str:
    """抽出**容器实际执行**的 ckpt 保留块（不重写、不复制逻辑，逐字来自 runner）。"""
    text = (runner or _RUNNER).read_text(encoding="utf-8")
    start = text.index(_RETENTION_START)
    # 终点 = 保留块**最后**那行 state 落盘（块内还有一处 `missing` 分支的早写，
    # 故取最后一次出现；再往后就是"轻量证据一律不删"的哨兵注释）。
    marker = _retention_end_marker(runner)
    end = text.rindex(marker) + len(marker)
    assert end > start
    return _heredoc_expand(text[start:end])


def _run_retention(
    tmp_path: Path,
    *,
    tier: str | None,
    rc: str = "0",
    checkpoints: list[str] | None = None,
    extra_dirs: list[str] | None = None,
    make_output_root: bool = True,
) -> tuple[subprocess.CompletedProcess[str], Path]:
    """在临时目录里真跑保留块（把容器内 `/out` 映射到 tmp_path/out）。

    ``tier=None`` 复现"容器内 TIER 未定义"；``tier="T999"`` 复现"档号定位错误"。
    """
    out_root = tmp_path / "out"
    out_root.mkdir(parents=True, exist_ok=True)
    if make_output_root:
        run_root = out_root / "T046" / "T046-run"
        run_root.mkdir(parents=True, exist_ok=True)
        for name in checkpoints or []:
            one = run_root / name
            one.mkdir(parents=True, exist_ok=True)
            (one / "trainer_state.json").write_text('{"global_step": 1}\n', encoding="utf-8")
        for name in extra_dirs or []:
            (run_root / name).mkdir(parents=True, exist_ok=True)

    # 把块里的 `/out/` 换成真实临时路径（等价于容器内的 /out 挂载点）。
    block = _retention_block_text().replace("/out/", f"{out_root}/")
    script = tmp_path / "retention.sh"
    # ★ 必须与**产物 entry.sh 的真实 shell 选项**一致：只有 `set -o pipefail`，**无 `-u`**。
    #   （容器内 `bash /entry.sh` 不看宿主的 `set -u`；若这里加 `-u`，就会把
    #    "档号未定义"变成 shell 报错而不是我们想断言的三态行为——那会掩盖真行为。）
    script.write_text(
        "set -o pipefail\n"
        + f'RC="{rc}"\n'
        + (f'TIER="{tier}"\n' if tier is not None else "")
        + block
        # ★ 复刻 entry.sh 的**收口**：保留块自身不 exit（块内只写 `RC=4`），由 entry.sh
        #   末尾这一行把退出码交还给容器。本 harness 只抽了块，必须补上这一行，
        #   否则测的就不是容器真实行为（原缺陷正是"块跑完、谁都不 exit ⇒ 退 0"）。
        + _entry_final_exit_line(),
        encoding="utf-8",
    )
    env = {**os.environ}
    env.pop("TIER", None)
    if tier is not None:
        env["TIER"] = tier
    completed = subprocess.run(
        ["bash", str(script)], capture_output=True, text=True, check=False, env=env
    )
    return completed, out_root


def _checkpoint_dirs(out_root: Path) -> list[str]:
    return sorted(p.name for p in out_root.rglob("checkpoint-*") if p.is_dir())


def _state(out_root: Path) -> str:
    state_file = out_root / "ckpt_retention.state"
    return state_file.read_text(encoding="utf-8").strip() if state_file.is_file() else "<无文件>"


def test_docker_argv_injects_the_tier_into_the_container(tmp_path: Path):
    """★加固①：`docker run` 必须**显式**把档号注入容器（不再靠"恰好未定义"）。"""
    bin_dir, elam_root, models_root = _make_fakes(tmp_path)
    completed = _run(tmp_path, bin_dir, elam_root=elam_root, models_root=models_root)
    assert completed.returncode == 0, completed.stderr

    argv = (tmp_path / "docker_argv.txt").read_text(encoding="utf-8").splitlines()
    # 必须是 `-e TIER=<档号>` 这一对（与 NVIDIA_VISIBLE_DEVICES 同范式）。
    assert "TIER=T010" in argv, argv
    assert argv[argv.index("TIER=T010") - 1] == "-e", argv
    # 且必须落在 image 之前（是 docker run 的选项，不是容器命令的一部分）。
    assert argv.index("TIER=T010") < argv.index("bash")


def test_retention_keeps_only_the_newest_and_never_touches_final(tmp_path: Path):
    """★三态①`ok`：3 份 ckpt ⇒ 只留最新（步数最大），`final/` 不删。"""
    completed, out_root = _run_retention(
        tmp_path,
        tier="T046",
        checkpoints=["checkpoint-2", "checkpoint-4", "checkpoint-5"],
        extra_dirs=["final"],
    )
    assert completed.returncode == 0, completed.stderr
    assert _state(out_root) == "ok"
    assert _checkpoint_dirs(out_root) == ["checkpoint-5"]
    assert "KEEP" in completed.stdout
    assert (out_root / "T046" / "T046-run" / "final").is_dir(), "final/ 不得被删"


def test_retention_single_checkpoint_is_kept_without_error(tmp_path: Path):
    """★三态①`ok`（边界）：仅 1 份 ⇒ 原样保留、不报错。"""
    completed, out_root = _run_retention(tmp_path, tier="T046", checkpoints=["checkpoint-7"])
    assert completed.returncode == 0, completed.stderr
    assert _state(out_root) == "ok"
    assert _checkpoint_dirs(out_root) == ["checkpoint-7"]


def _assert_loud_failure(completed: subprocess.CompletedProcess[str], out_root: Path, state: str):
    """★加固②的共用断言：必须**硬失败**（rc≠0）且 state 落在预期态、报错可操作。"""
    assert completed.returncode != 0, (
        f"静默失败！state={_state(out_root)} 但 rc=0 —— 保留策略悄悄失效。"
        f"stdout={completed.stdout} stderr={completed.stderr}"
    )
    assert completed.returncode == 4, completed
    assert _state(out_root) == state, completed.stderr
    assert "FATAL" in completed.stderr, completed.stderr
    assert "运行链路错误" in completed.stderr, completed.stderr


def test_retention_missing_checkpoints_fails_loudly(tmp_path: Path):
    """★三态②`missing` + 加固②：目录在但**一份 ckpt 都没有** ⇒ 硬失败（原先 rc=0）。"""
    completed, out_root = _run_retention(tmp_path, tier="T046", checkpoints=[])
    _assert_loud_failure(completed, out_root, "missing")
    # 不猜不删：不该顺手造任何东西。
    assert _checkpoint_dirs(out_root) == []


def test_retention_absent_output_root_fails_loudly_when_training_succeeded(tmp_path: Path):
    """★三态③`none` + 加固②：训练成功却无产物目录 ⇒ 硬失败（原先 rc=0 静默跳过）。"""
    completed, out_root = _run_retention(tmp_path, tier="T046", rc="0", make_output_root=False)
    _assert_loud_failure(completed, out_root, "none")


def test_retention_absent_output_root_keeps_training_rc_when_training_failed(tmp_path: Path):
    """★三态③`none` 的**唯一合法情形**：训练本身失败（RC≠0）⇒ 记 none 但**保留原 rc**。

    理由：训练失败时产物目录本就不会存在；此时若报 rc=4 就会**掩盖训练失败的真因**
    （真实 rc 可能是 OOM=137 / 超时=124），属"退路改变结果"，宪法不允许。

    ★ 这里断言的是 **harness 拼出的"块 + entry.sh 收口"整体**的退出码（= 容器真实行为）：
    必须是 **137**（原样交还），不是 0（原缺陷：谁都不 exit ⇒ 被 printf 的成功盖成 0），
    也不是 4（那会用"保留策略失败"盖住"训练失败"）。
    """
    completed, out_root = _run_retention(tmp_path, tier="T046", rc="137", make_output_root=False)
    assert _state(out_root) == "none"
    assert "FATAL" in completed.stderr
    assert completed.returncode == 137, completed


def test_retention_missing_checkpoints_keeps_training_rc_when_training_failed(tmp_path: Path):
    """★三态②`missing` 的同一条优先级：训练失败（RC≠0）时交还原 rc，不被 4 盖住。"""
    completed, out_root = _run_retention(tmp_path, tier="T046", rc="124", checkpoints=[])
    assert _state(out_root) == "missing"
    assert "FATAL" in completed.stderr
    assert completed.returncode == 124, completed


def test_retention_block_itself_never_exits(tmp_path: Path):
    """★单点收口不变量：保留块内**只写 `RC=4`，不自己 exit**（否则两处 exit 语义分叉）。

    块必须把"要不要失败"表达成 `RC`，由 entry.sh 末尾唯一的 `exit "$RC"` 决定容器退出码；
    这样"训练失败 vs 保留失败"的优先级只有一个表达点（§1.4 单一真相源）。
    """
    block = _retention_block_text()
    assert "exit 4" not in block
    assert "then RC=4; fi" in block
    # 块内不得出现任何 `exit`（除了函数内 `return`）。
    exits = [line for line in block.splitlines() if line.strip().startswith("exit ")]
    assert exits == [], exits


def test_retention_wrong_tier_is_a_hard_failure_not_a_silent_skip(tmp_path: Path):
    """★★核心加固回归：注入**错误档号**（T999）⇒ 必须硬失败，**不得静默跳过**。

    这是本轮真正要防的坑：加固前若容器里 TIER 被设成 T999，
    `[ -d /out/T999 ]` 为假 ⇒ 原先只是打一行提示、**rc=0、一份 ckpt 都不删**。
    """
    completed, out_root = _run_retention(
        tmp_path,
        tier="T999",
        checkpoints=["checkpoint-2", "checkpoint-4", "checkpoint-5"],
    )
    # 档号错 ⇒ 定位到 /out/T999（不存在）⇒ 走 none 分支 ⇒ rc=0 时硬失败。
    _assert_loud_failure(completed, out_root, "none")
    # 且**确实没删任何 ckpt**（失败是"没做"，不能是"做错了"）。
    assert _checkpoint_dirs(out_root) == ["checkpoint-2", "checkpoint-4", "checkpoint-5"]


def test_retention_unset_tier_still_succeeds_only_for_the_wrong_reason(tmp_path: Path):
    """★加固回归：容器内 `TIER` **未定义**时的行为必须**可判定**，不再靠巧合。

    加固前：`TIER` 未定义 ⇒ `OUTPUT_ROOT="/out/"`（挂载根）⇒ 递归 `find` 命中 ⇒ 看似生效。
    那**不是**"定位正确"，只是"恰好落在根上"（多档共用挂载时会误伤别的档）。

    加固后：档号由 `docker run -e "TIER=$TIER"` 显式注入 ⇒ 运行时**不可能**未定义。
    本用例固定"未定义"这个非法输入，断言它**仍可判定**（要么按常规清理、要么硬失败），
    并据此说明：真正的防线是 `-e TIER` 的注入（见 `test_docker_argv_injects_...`）。
    """
    completed, out_root = _run_retention(
        tmp_path,
        tier=None,
        checkpoints=["checkpoint-2", "checkpoint-4", "checkpoint-5"],
    )
    # 未定义 ⇒ OUTPUT_ROOT 退化为 /out/ ⇒ 递归仍能找到 ⇒ 恰好清理成功（rc=0/state=ok）。
    # 关键不变量：**不再存在"rc=0 且什么都没做、也不留痕"的静默路径**。
    assert _state(out_root) in {"ok", "missing", "none"}, completed.stderr
    if completed.returncode == 0:
        assert _state(out_root) == "ok", "rc=0 必须是「确实做了保留」"
        assert _checkpoint_dirs(out_root) == ["checkpoint-5"]
    else:
        assert _state(out_root) in {"missing", "none"}, completed.stderr


# ── 跨 run 复跑：keeper 必须是**本轮最新**那份（2026-09-20 缺陷修复）──────────
#
# 缺陷（228 实测 T043 run2；本机已复现，见工位 task-retention-xrun/evidence/）：
# 同一档**第二次跑**时 `find /out/$TIER` **跨 v0-*/v1-* 两个 run 目录**找 ckpt，而旧判据
# **只比步数**且用 `-gt`（**严格大于**）⇒ 两份 `checkpoint-160` 步数相同时，keeper 完全由
# `find` 的 readdir 顺序决定（ext4 dir_index / overlayfs / tmpfs 各不相同）；顺序不利时
# **本轮最新**被当"中间段"删除（`last-checkpoint` 变断链、本轮 A2/A3 证据丢失），
# 而 `ckpt_retention.state` 仍报 `ok`、容器仍退 0 —— 又一次静默失败。
#
# 正确语义（单一真相源 = manifest `runtime.checkpoint_retention`，本包不得改它）：
#   `keep_per_tier: 1` + delete「**同一档内**除最新一份以外的 `checkpoint-*/`」
#   ⇒ **每档只留一份，且那一份必须是本轮最新**。故扫描范围（整档递归）是对的，
#     错的是"最新"的判据。
# 修法 = 三元全序（mtime → 步数 → 路径字典序），与 `find` 输出顺序**无关**。

_RUN1 = "v0-20260920-061152"
_RUN2 = "v1-20260920-155643"
#: 平手现场：两轮都到 160（T043 run2 的真实形态）。
_TIE_STEPS = (20, 40, 60, 80, 100, 120, 140, 160)
#: 更短的重跑：本轮只到 100（与 find 顺序无关的确定性形态）。
_SHORT_STEPS = (20, 40, 60, 80, 100)


def _build_runs(
    tmp_path: Path,
    runs: list[tuple[str, list[str]]],
    *,
    tier: str = "T046",
    stamps: dict[str, float] | None = None,
) -> Path:
    """造出**多个 run 目录**的产物树。

    mtime 默认按**列表顺序**递增；给了 ``stamps``（run 目录名 → 写入时刻）则以它为准
    —— 供"创建顺序与 run 新旧无关"的顺序无关性用例使用。
    """
    out_root = tmp_path / "out"
    tier_root = out_root / tier
    tier_root.mkdir(parents=True)
    for run_index, (run_name, ckpts) in enumerate(runs):
        run_dir = tier_root / run_name
        run_dir.mkdir()
        for ckpt in ckpts:
            one = run_dir / ckpt
            one.mkdir()
            (one / "trainer_state.json").write_text('{"global_step": 1}\n', encoding="utf-8")
            stamp = (stamps or {}).get(run_name, 1_700_000_000 + run_index * 3600)
            os.utime(one, (stamp, stamp))
    return out_root


def _run_block(
    tmp_path: Path, out_root: Path, *, block: str, tier: str = "T046", rc: str = "0"
) -> subprocess.CompletedProcess[str]:
    """在给定产物树上真跑**指定的**保留块（供突变测试注入自定义块）。"""
    script = tmp_path / "retention_variant.sh"
    script.write_text(
        "set -o pipefail\n"
        + f'RC="{rc}"\n'
        + f'TIER="{tier}"\n'
        + block.replace("/out/", f"{out_root}/")
        + _entry_final_exit_line(),
        encoding="utf-8",
    )
    env = {**os.environ}
    env.pop("TIER", None)
    env["TIER"] = tier
    return subprocess.run(
        ["bash", str(script)], capture_output=True, text=True, check=False, env=env
    )


def _checkpoint_rel_paths(out_root: Path, *, tier: str = "T046") -> list[str]:
    """档产物根（`<out>/<tier>`）下所有 checkpoint 目录，**相对档根**的路径（含 run 目录名）。"""
    tier_root = out_root / tier
    return sorted(
        str(p.relative_to(tier_root)) for p in tier_root.rglob("checkpoint-*") if p.is_dir()
    )


def _assert_only_the_current_runs_newest_survived(
    completed: subprocess.CompletedProcess[str], out_root: Path, *, current_run: str, expected: str
) -> None:
    """★核心不变量（修复点的断言；下方突变测试证明它**真能失败**）。

    ① rc=0 且 `state=ok`；② **本轮最新**那份必须还在；③ 同档只剩一份（`keep_per_tier`）。
    """
    assert completed.returncode == 0, completed.stderr
    assert _state(out_root) == "ok", completed.stderr
    survived = _checkpoint_rel_paths(out_root)
    assert f"{current_run}/{expected}" in survived, (
        f"跨 run 误删本轮最新 ckpt：{current_run}/{expected} 不在 {survived}"
    )
    assert len(survived) == 1, f"keep_per_tier=1 被破坏（同档多于一份）：{survived}"


def test_retention_single_run_dir_still_keeps_the_newest(tmp_path: Path):
    """情形①：**只有一个 run 目录** ⇒ 与加固前一致（留步数最大那份、不报错）。"""
    out_root = _build_runs(tmp_path, [(_RUN1, ["checkpoint-5", "checkpoint-2", "checkpoint-4"])])
    completed = _run_block(tmp_path, out_root, block=_retention_block_text())
    _assert_only_the_current_runs_newest_survived(
        completed, out_root, current_run=_RUN1, expected="checkpoint-5"
    )


def test_retention_second_run_keeps_the_current_runs_newest_on_a_step_tie(tmp_path: Path):
    """情形②（缺陷现场）：两轮都到 `checkpoint-160`（**步数平手**）⇒ 留**本轮**那份。"""
    out_root = _build_runs(
        tmp_path,
        [
            (_RUN1, ["checkpoint-160"]),
            (_RUN2, [f"checkpoint-{s}" for s in _TIE_STEPS]),
        ],
    )
    completed = _run_block(tmp_path, out_root, block=_retention_block_text())
    _assert_only_the_current_runs_newest_survived(
        completed, out_root, current_run=_RUN2, expected="checkpoint-160"
    )


def test_retention_shorter_rerun_still_keeps_the_current_runs_newest(tmp_path: Path):
    """情形②-确定性：本轮是**更短的重跑**（只到 100、上一轮有 160）⇒ 仍必须留本轮的 100。

    这一形态与 `find` 顺序**无关**：旧实现按步数取最大 ⇒ keeper 必是上一轮的 160、
    本轮 100 必被删。故它是"能真失败"的负向用例（见下方突变测试）。
    """
    out_root = _build_runs(
        tmp_path,
        [
            (_RUN1, ["checkpoint-160"]),
            (_RUN2, [f"checkpoint-{s}" for s in _SHORT_STEPS]),
        ],
    )
    completed = _run_block(tmp_path, out_root, block=_retention_block_text())
    _assert_only_the_current_runs_newest_survived(
        completed, out_root, current_run=_RUN2, expected="checkpoint-100"
    )


def test_retention_keeper_is_independent_of_find_order(tmp_path: Path):
    """情形②-顺序无关：同一布局、两种**创建顺序**（模拟不同文件系统的 readdir 顺序）
    ⇒ keeper 必须相同。旧实现正是"顺序决定 keeper"（本机 R1 vs R1c 实测）。"""
    tie_run2 = [f"checkpoint-{s}" for s in _TIE_STEPS]
    # ★ run 新旧由**写入时刻**表达，与"目录创建顺序"解耦（后者只影响 readdir 顺序）。
    stamps = {_RUN1: 1_700_000_000.0, _RUN2: 1_700_003_600.0}
    kept: set[str] = set()
    for index, runs in enumerate(
        [
            [(_RUN1, ["checkpoint-160"]), (_RUN2, tie_run2)],
            [(_RUN2, tie_run2), (_RUN1, ["checkpoint-160"])],
        ]
    ):
        case = tmp_path / f"order-{index}"
        case.mkdir()
        out_root = _build_runs(case, runs, stamps=stamps)
        completed = _run_block(case, out_root, block=_retention_block_text())
        _assert_only_the_current_runs_newest_survived(
            completed, out_root, current_run=_RUN2, expected="checkpoint-160"
        )
        kept |= set(_checkpoint_rel_paths(out_root))
    assert kept == {f"{_RUN2}/checkpoint-160"}, kept


def test_retention_no_checkpoint_in_any_run_is_still_a_loud_failure(tmp_path: Path):
    """情形③：多个 run 目录但**一份 ckpt 都没有** ⇒ 仍必须 FATAL + `missing` + rc=4。"""
    out_root = _build_runs(tmp_path, [(_RUN1, []), (_RUN2, [])])
    completed = _run_block(tmp_path, out_root, block=_retention_block_text())
    _assert_loud_failure(completed, out_root, "missing")


# ── N2（2026-09-20）：native 后端的产物形态是 `final/`，不是 `checkpoint-*` ──────
#
# 缺陷：保留块只 `find -name 'checkpoint-*'`，而 **native 只落 `final/`**
# （`collect_results.NATIVE_CHECKPOINT_DIRNAME = "final"`；
# `flow/backend/native/trainer/trainer.py` 的 `_save_checkpoint(output_dir / "final")`）
# ⇒ native 档产物根里没有 `checkpoint-*` ⇒ `state=missing`；而**上一轮加固**把
# `missing` + `RC=0` 变成 `exit 4` ⇒ **native 档"训练成功却 rc=4"（假失败，污染台账）**。
#
# 修法（最小）：把"是否已有 ≥1 份产物"的判定**按产物形态感知** —— 承认**非空**的
# `final/`（native 终态）为"已有产物"，保留动作无事可做 ⇒ `ok`。
# ★ 三态语义**不放松**：`missing` 的触发条件只少了"native 的合法形态"这一种误判；
#   空 `final/`（保存中断 ⇒ 产物没落盘）与"什么都没有"**仍然** `missing` + `exit 4`。

_NATIVE_TIER = "T028"


def _build_native_layout(tmp_path: Path, *, shape: str, tier: str = _NATIVE_TIER) -> Path:
    """造 native 后端的产物树：`<out>/<tier>/final/`（**不是** `checkpoint-*`）。

    ``shape``：``"files"`` 有非空 `final/`；``"empty"`` 空 `final/`；``"none"`` 什么都没有。
    """
    out_root = tmp_path / "out"
    tier_root = out_root / tier
    tier_root.mkdir(parents=True)
    if shape in {"files", "empty"}:
        final_dir = tier_root / "final"
        final_dir.mkdir()
        if shape == "files":
            (final_dir / "manifest.json").write_text("{}\n", encoding="utf-8")
            (final_dir / "trainer_state.json").write_text("{}\n", encoding="utf-8")
    return out_root


def _assert_native_final_accepted(
    completed: subprocess.CompletedProcess[str], out_root: Path, *, tier: str = _NATIVE_TIER
) -> None:
    """★核心断言：native 的 `final/` 必须被承认为"已有产物" ⇒ rc=0 / `state=ok`，
    且**一个文件都没删**（native 只落一份终态，本来就没有中间段可清理）。"""
    assert completed.returncode == 0, f"native final/ 被误判为失败：{completed.stderr}"
    assert _state(out_root) == "ok", completed.stderr
    assert "KEEP" in completed.stdout, completed.stdout
    assert "删除数=0" in completed.stdout, completed.stdout
    final_dir = out_root / tier / "final"
    assert final_dir.is_dir(), f"final/ 不见了：{sorted(p.name for p in out_root.rglob('*'))}"
    survivors = sorted(path.name for path in final_dir.iterdir())
    assert survivors == ["manifest.json", "trainer_state.json"], survivors


def test_retention_accepts_the_native_final_artifact_shape(tmp_path: Path):
    """★N2 修复：native 有非空 `final/` ⇒ `ok`（不假失败），且**什么都不删**。"""
    out_root = _build_native_layout(tmp_path, shape="files")
    completed = _run_block(tmp_path, out_root, block=_retention_block_text(), tier=_NATIVE_TIER)
    _assert_native_final_accepted(completed, out_root)


def test_retention_empty_native_final_is_still_a_loud_failure(tmp_path: Path):
    """★N2 不得放松①：**空** `final/`（保存中断 ⇒ 产物没落盘）⇒ 仍 `missing` + rc=4。"""
    out_root = _build_native_layout(tmp_path, shape="empty")
    completed = _run_block(tmp_path, out_root, block=_retention_block_text(), tier=_NATIVE_TIER)
    _assert_loud_failure(completed, out_root, "missing")


def test_retention_native_without_any_artifact_is_still_a_loud_failure(tmp_path: Path):
    """★N2 不得放松②：native 产物根里**什么都没有** ⇒ 仍 `missing` + rc=4。"""
    out_root = _build_native_layout(tmp_path, shape="none")
    completed = _run_block(tmp_path, out_root, block=_retention_block_text(), tier=_NATIVE_TIER)
    _assert_loud_failure(completed, out_root, "missing")


def test_retention_mswift_layout_is_unaffected_by_the_native_branch(tmp_path: Path):
    """★N2 不得放松③：ms-swift 布局（`checkpoint-*`）走的还是原路径、结论不变。"""
    out_root = _build_runs(tmp_path, [(_RUN1, ["checkpoint-2", "checkpoint-5"])])
    completed = _run_block(tmp_path, out_root, block=_retention_block_text())
    _assert_only_the_current_runs_newest_survived(
        completed, out_root, current_run=_RUN1, expected="checkpoint-5"
    )


#: 保留段的起点标记（说明注释 + 函数 + 三态收口都在它之后）。
_RETENTION_SECTION_START = "# ── ckpt 保留策略"
#: GPU 摘要段（说明注释 + `wait_for_gpu_summary` + 调用）的起止标记。
_GPU_SUMMARY_SECTION_START = "# ══ 收口前必须做："
_GPU_SUMMARY_SECTION_END = 'wait_for_gpu_summary "\\$SAMPLER"\n'
#: entry.sh 的 heredoc 起始行（**不加引号** ⇒ 外层会展开它）。
_ENTRY_HEREDOC_START_LINE = 'cat > "$RUN_DIR/entry.sh" <<ENTRY\n'


def _heredoc_region_text(start_marker: str, end_marker: str) -> str:
    """取 runner 模板源里 `start_marker` .. `end_marker`（含）之间的逐字文本。"""
    text = _RUNNER.read_text(encoding="utf-8")
    start = text.index(start_marker)
    end = text.index(end_marker, start) + len(end_marker)
    assert end > start
    return text[start:end]


def _retention_section_text() -> str:
    """保留段在 **runner 模板源**里的逐字文本（含函数前的说明注释）。

    注意：这不是进容器的产物，而是 `cat > entry.sh <<ENTRY` 的**heredoc 正文**——
    外层 `set -uo pipefail` 会对它做参数展开与命令替换。
    """
    return _heredoc_region_text(_RETENTION_SECTION_START, _retention_end_marker())


def _gpu_summary_section_text() -> str:
    """GPU 摘要段（同一 heredoc 正文里的另一段说明注释 + 函数 + 调用）。"""
    return _heredoc_region_text(_GPU_SUMMARY_SECTION_START, _GPU_SUMMARY_SECTION_END)


def _entry_heredoc_body_text() -> str:
    """`cat > entry.sh <<ENTRY ... ENTRY` 的**整段正文**（逐字、未展开）。"""
    text = _RUNNER.read_text(encoding="utf-8")
    start = text.index(_ENTRY_HEREDOC_START_LINE) + len(_ENTRY_HEREDOC_START_LINE)
    end = text.index("\nENTRY\n", start) + 1
    assert end > start
    return text[start:end]


def _assert_no_bare_backtick(section: str, *, where: str) -> None:
    """不加引号的 heredoc 正文里，裸反引号会被外层当**命令替换真的执行**。"""
    assert "`" not in section.replace("\\`", ""), (
        f"{where} 出现裸反引号：外层 heredoc 会把它当命令替换执行（转义形态 \\` 才安全）"
    )


def _assert_no_bare_dollar(section: str, *, where: str) -> None:
    """不加引号的 heredoc 正文里，裸美元符会被外层按**外层变量**展开（`set -u` 下报 unbound）。"""
    assert "$" not in section.replace("\\$", ""), (
        f"{where} 出现裸美元符：会被外层展开（转义形态 \\$var 才安全）"
    )


def test_retention_section_has_no_heredoc_expansion_hazards():
    """★哨兵（静态）：保留段是**不加引号的 heredoc 正文** ⇒ 段内不得出现裸反引号
    （外层会当**命令替换真的执行**）或裸美元符（外层会按外层变量展开，`set -u` 下还会报
    `unbound variable`）。转义形态 `\\$var` 是允许的（外层不展开、留给内层）。

    本包修 bug 时踩过这个坑：注释里写了裸反引号包着的 `find`，生成 entry.sh 时
    它把**整棵目录树**多行打印进产物 ⇒ 产物立刻语法错误。下方动态哨兵做端到端兜底。
    """
    section = _retention_section_text()
    _assert_no_bare_backtick(section, where="保留段")
    _assert_no_bare_dollar(section, where="保留段")


def test_gpu_summary_section_has_no_heredoc_expansion_hazards():
    """★哨兵（静态，2026-09-20 扩展）：**同一 heredoc 正文**里的 GPU 摘要段同样适用。

    该段原先有 15 处裸反引号（`record_gpu_memory`、`finally`、`<run>/gpu/...json` 等），
    每次生成 entry.sh 时外层**真的执行**它们，实测 stderr 逐字有
    `bash: $!: unbound variable` / `record_gpu_memory: command not found` / `printf: usage: ...`
    ⇒ 已按保留段同一口径处理（去裸反引号 + 保持 `\\$` 转义）。本哨兵防复发。
    """
    section = _gpu_summary_section_text()
    _assert_no_bare_backtick(section, where="GPU 摘要段")
    _assert_no_bare_dollar(section, where="GPU 摘要段")


def test_entry_heredoc_body_has_no_bare_backticks():
    """★哨兵（静态，整段）：**整段 entry.sh heredoc 正文**不得有裸反引号。

    与前两条的分工：那两条按"段"检查（含裸美元符，因为那两段只用内层变量）；
    本条的 `$` 无法整段禁（代码行**有意**留裸 `$CONTAINER_PY`/`$TRAIN_PATH`/... 给外层展开），
    故整段只查反引号——反引号在该正文里**没有任何合法用途**。
    """
    _assert_no_bare_backtick(_entry_heredoc_body_text(), where="entry.sh heredoc 正文")


def test_generated_entry_sh_parses(tmp_path: Path):
    """★哨兵（动态/端到端）：**真跑一次 runner**（外层 heredoc 真的展开）⇒ 产物
    `entry.sh` 必须 `bash -n` 通过。

    静态哨兵按段/整段查注释；本用例覆盖**真实展开结果**——任何被外层命令替换/变量展开
    污染的注释或代码都会在这里变成语法错误（而不是等到容器里才炸）。
    """
    script = tmp_path / "entry_parse_check.sh"
    script.write_text(_entry_sh_text(tmp_path), encoding="utf-8")
    parsed = subprocess.run(
        ["bash", "-n", str(script)], capture_output=True, text=True, check=False
    )
    assert parsed.returncode == 0, (
        f"entry.sh 语法错误（外层 heredoc 展开被污染？）：{parsed.stderr}"
    )


def _heredoc_continuation_count(body: str) -> int:
    """数正文里**真正被吃掉换行**的续行：反斜杠+换行，且反斜杠个数为**奇数**。

    （偶数个反斜杠 = 转义后的字面反斜杠，换行保留；这是 heredoc 展开的规则。）
    """
    count = 0
    for index, char in enumerate(body):
        if char != "\n":
            continue
        backslashes = 0
        probe = index - 1
        while probe >= 0 and body[probe] == "\\":
            backslashes += 1
            probe -= 1
        if backslashes % 2 == 1:
            count += 1
    return count


def test_generated_entry_sh_line_count_matches_the_heredoc_body(tmp_path: Path):
    """★哨兵（动态/行数）：生成物行数必须 = 正文行数 − 续行数。

    不加引号的 heredoc 会吃掉「反斜杠+换行」的续行，但**绝不会凭空多出换行**；
    一旦外层替某个裸反引号执行了命令、而它的输出含换行（本包踩过：裸 `find` 把整棵
    目录树按行打印进产物），生成物就会多出行来。`bash -n` 只抓得到"多出来的行恰好
    是语法错误"的情形；行数不变量连"多出来的行看起来人畜无害"也能抓。
    """
    body = _entry_heredoc_body_text()
    expected = body.count("\n") - _heredoc_continuation_count(body)
    actual = _entry_sh_text(tmp_path).count("\n")
    assert actual == expected, (
        f"entry.sh 行数 {actual} ≠ 正文行数−续行 {expected}：外层 heredoc 展开被污染？"
    )


# ── 负向（突变）测试：证明上面的硬失败断言**真能失败** ──────────────────────


def _mutant_retention_text(tmp_path: Path, *, old_silent_skip: bool) -> Path:
    """造一个"退回加固前行为"的 runner 副本（只变异保留块，其余逐字相同）。"""
    text = _RUNNER.read_text(encoding="utf-8")
    if old_silent_skip:
        # 退回加固②：把 `missing` / `none` 的硬失败拿掉（恢复成"静默跳过、rc=0"）。
        # 源码里是 4 空格缩进 + **字面量** `\n`（heredoc 正文，不是真换行）。
        text = text.replace('    if [ "\\$RC" = "0" ]; then exit 4; fi', "", 2)
    mutant = _REPO_ROOT / "tests" / "e2e" / f"_mutant_retention_{tmp_path.name}.sh"
    mutant.write_text(text, encoding="utf-8")
    mutant.chmod(0o755)
    return mutant


def test_retention_hard_failure_assertion_really_fails_on_the_old_silent_behaviour(
    tmp_path: Path,
):
    """★负向（突变）：把硬失败拿掉 ⇒ `missing`/`none` 必须被断言判红（证明不是空断言）。"""
    mutant = _mutant_retention_text(tmp_path, old_silent_skip=True)
    try:
        # 突变必须真的生效（两处 exit 4 都被删）——在**模板源**上核对。
        raw = mutant.read_text(encoding="utf-8")
        assert 'if [ "\\$RC" = "0" ]; then exit 4; fi' not in raw, "突变未生效"
        # 用变异体跑"目录不存在 + RC=0"：旧行为 ⇒ rc=0、state=none。
        out_root = tmp_path / "out"
        out_root.mkdir(parents=True)
        block = _retention_block_text(mutant).replace("/out/", f"{out_root}/")
        script = tmp_path / "mutant.sh"
        script.write_text('set -o pipefail\nRC="0"\nTIER="T046"\n' + block, encoding="utf-8")
        done = subprocess.run(["bash", str(script)], capture_output=True, text=True, check=False)
        assert done.returncode == 0, "突变未生效：仍然硬失败"
        assert _state(out_root) == "none"
        # 同一个断言在旧行为上必须抛 AssertionError ⇒ 正向断言真的能失败。
        with pytest.raises(AssertionError, match="静默失败"):
            _assert_loud_failure(done, out_root, "none")
    finally:
        mutant.unlink(missing_ok=True)
    assert not mutant.exists(), "变异体必须清理干净（不得留在已跟踪目录里）"


#: 修复前的 keeper 判据（**只比步数 + 严格大于**）——heredoc 展开后的逐字形态。
_OLD_STEP_ONLY_SELECTION = """\
        if [ -z "$newest_path" ] || [ "$step" -gt "$newest_step" ]; then
            newest_path="$path"; newest_step="$step"
        fi
"""


def _mutant_step_only_retention_block() -> str:
    """把选 keeper 的判据**退回修复前形态**（只比步数），其余逐字相同。"""
    block = _retention_block_text()
    start = block.index('        if [ -z "$newest_path" ]; then\n')
    end = block.index("    done\n", start)
    mutant = block[:start] + _OLD_STEP_ONLY_SELECTION + block[end:]
    assert mutant != block, "突变未生效"
    return mutant


def test_retention_current_run_assertion_really_fails_on_the_old_step_only_selection(
    tmp_path: Path,
):
    """★负向（突变）：判据退回"只比步数" ⇒ 上面的正向断言**必须判红**（证明不是空断言）。"""
    out_root = _build_runs(
        tmp_path,
        [
            (_RUN1, ["checkpoint-160"]),
            (_RUN2, [f"checkpoint-{s}" for s in _SHORT_STEPS]),
        ],
    )
    completed = _run_block(tmp_path, out_root, block=_mutant_step_only_retention_block())
    # 旧行为：上一轮的 160 胜出 ⇒ **本轮最新 100 被删**；而 rc=0 / state=ok（静默）。
    assert completed.returncode == 0, completed.stderr
    assert _state(out_root) == "ok"
    assert _checkpoint_rel_paths(out_root) == [f"{_RUN1}/checkpoint-160"], completed.stdout
    # 同一个断言在旧行为上必须抛 AssertionError ⇒ 正向断言真的能失败。
    with pytest.raises(AssertionError, match="误删"):
        _assert_only_the_current_runs_newest_survived(
            completed, out_root, current_run=_RUN2, expected="checkpoint-100"
        )


#: N2 native 识别段的起点标记（heredoc 展开后的逐字形态）。
_NATIVE_RECOGNITION_START = "        # ★ native 后端的产物形态"
#: `#all == 0` 分支的 FATAL 行（识别段之后紧接的就是它）。
_NATIVE_BRANCH_FATAL = '        echo "[ckpt-retention] FATAL: 未找到任何 checkpoint-* 目录'


def _mutant_without_native_recognition() -> str:
    """把 N2 的 native 识别段删掉（= 修复前行为），其余逐字相同。"""
    block = _retention_block_text()
    start = block.index(_NATIVE_RECOGNITION_START)
    end = block.index(_NATIVE_BRANCH_FATAL, start)
    mutant = block[:start] + block[end:]
    assert mutant != block, "突变未生效"
    assert _NATIVE_RECOGNITION_START not in mutant
    return mutant


def test_retention_native_ok_assertion_really_fails_without_the_recognition(
    tmp_path: Path,
):
    """★负向（突变）：删掉 native 识别段 ⇒ 上面的正向断言**必须判红**（证明不是空断言）。

    这正是 N2 的**假失败**现场：native 明明有 `final/`，旧代码却记 `missing` + rc=4。
    """
    out_root = _build_native_layout(tmp_path, shape="files")
    completed = _run_block(
        tmp_path, out_root, block=_mutant_without_native_recognition(), tier=_NATIVE_TIER
    )
    # 旧行为：有 final/ 也被判 missing + rc=4（**假失败**）。
    assert completed.returncode == 4, completed.stderr
    assert _state(out_root) == "missing"
    # 同一个断言在旧行为上必须抛 AssertionError ⇒ 正向断言真的能失败。
    with pytest.raises(AssertionError, match="native"):
        _assert_native_final_accepted(completed, out_root)


def test_retention_tier_injection_assertion_really_fails_when_it_is_removed(tmp_path: Path):
    """★负向（突变）：删掉 `-e "TIER=$TIER"` ⇒ 加固①的断言必须判红。"""
    text = _RUNNER.read_text(encoding="utf-8")
    needle = '    -e "TIER=$TIER"\n'
    assert needle in text, "突变目标不存在，测试前提失效"
    mutant = _REPO_ROOT / "tests" / "e2e" / f"_mutant_tier_{tmp_path.name}.sh"
    mutant.write_text(text.replace(needle, "", 1), encoding="utf-8")
    mutant.chmod(0o755)
    try:
        bin_dir, elam_root, models_root = _make_fakes(tmp_path)
        completed = _run(
            tmp_path, bin_dir, elam_root=elam_root, models_root=models_root, runner=mutant
        )
        assert completed.returncode == 0, completed.stderr
        argv = (tmp_path / "docker_argv.txt").read_text(encoding="utf-8").splitlines()
        assert "TIER=T010" not in argv, f"突变未生效：argv 仍带 TIER：{argv}"
        with pytest.raises(AssertionError):
            assert "TIER=T010" in argv, argv
    finally:
        mutant.unlink(missing_ok=True)
    assert not mutant.exists(), "变异体必须清理干净（不得留在已跟踪目录里）"


# ── W3 连带失效的修复：显存采样改为配置驱动（2026-09-19）──────────────────────
# W3 把 `record-gpu-memory` 的产物参数（`--output-dir` / `--tag` / `--interval-sec` …）
# 从 CLI 删除、改为 `--config` 的 `gpu_monitor` 段驱动，但矩阵 runner 的两处调用点
# 沿用了旧写法 ⇒ 开发机上跑 matrix 会 argparse 报错（rc=2）。
# 下面四条把"修复后必须成立"的性质钉住（并证明断言真的能失败）。

#: 档配置里必须存在的显存采样段（由 `generate_matrix.py::build_config` 写出）。
_MONITOR_FIELDS = ("output_dir", "tag", "interval_sec")


def _load_generate_matrix():
    """按路径加载 `generate_matrix.py`（它不依赖包导入，可独立加载）。"""
    spec = importlib.util.spec_from_file_location(
        "matrix54_generate_matrix_monitor", _REPO_ROOT / "tests" / "e2e" / "generate_matrix.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _entry_text(tmp_path: Path, *, tier: str = "T010") -> str:
    """跑一次 runner（假 docker 不执行 entry.sh），返回生成的 entry.sh 文本。"""
    bin_dir, elam_root, models_root = _make_fakes(tmp_path)
    completed = _run(tmp_path, bin_dir, elam_root=elam_root, models_root=models_root, tier=tier)
    assert completed.returncode == 0, completed.stderr
    return (tmp_path / "runs" / tier / "entry.sh").read_text(encoding="utf-8")


def test_tier_config_carries_the_gpu_monitor_section(tmp_path: Path):
    """★档配置必须自带 `gpu_monitor` 段，且 tag=档号、落点=取证链锚点（`/out/gpu`）。

    ★ 为什么落点**不能**是 `<training.output_dir>/gpu`（2026-09-20 真机确证）：
    矩阵档一律 `overwrite_output_dir: true`，训练进程启动时对**非空**的
    `<training.output_dir>` 执行 `shutil.rmtree`（`lora_io.prepare_output_dir`）；
    采样器先于 torchrun 起，若落点在该子树内，刚建好的 `gpu/` 会被整棵删掉 ⇒
    首个 `append_jsonl` 即 ENOENT（228 真跑首错签名：
    `gpu_monitor.py:621 append_jsonl ... '/out/T010/gpu/gpu_memory.jsonl'`）。

    本条把两条不变量**写成代码**：① 落点 == 取证链锚点（与消费端同源）；
    ② 落点**不在**训练输出目录之内（训练清洗伤不到证据）。
    这是**修正被固化的错误期望**，不是放宽断言——存在性/档号一致性守卫逐字未动。
    """
    generate_matrix = _load_generate_matrix()
    anchor = generate_matrix.GPU_MONITOR_CONTAINER_DIR
    assert anchor == "/out/gpu", f"取证链锚点被改：{anchor!r}"
    tiers = {tier["tier_id"]: tier for tier in generate_matrix.build_ledger()}
    for tier_id, tier in tiers.items():
        status, _ = generate_matrix.classify_expressibility(tier)
        if status in ("blocked", "not_applicable"):
            continue  # 这两类不产出可跑配置（既有语义，见 render_all_tiers）
        config = generate_matrix.build_config(tier)
        monitor = config.get("gpu_monitor")
        assert isinstance(monitor, dict), f"{tier_id} 缺 gpu_monitor 段"
        assert set(monitor) == set(_MONITOR_FIELDS), f"{tier_id} 字段集合变了：{monitor}"
        assert monitor["tag"] == tier_id, f"{tier_id} 的 tag 必须等于档号"
        assert monitor["output_dir"] == anchor, (
            f"{tier_id} 的落点必须等于取证链锚点 {anchor}（生产端与消费端同源）"
        )
        training_out = config["training"]["output_dir"]
        assert not monitor["output_dir"].startswith(f"{training_out}/"), (
            f"{tier_id} 的落点落在训练输出目录 {training_out} 之内 ⇒ 训练侧 "
            f"overwrite_output_dir 的 shutil.rmtree 会删掉采样产物（228 实测）"
        )
        assert monitor["interval_sec"] == 2.0, f"{tier_id} 的 interval_sec 必须是 2"


def test_rendered_config_yaml_contains_the_monitor_section(tmp_path: Path):
    """★渲染出的档配置 YAML 里真的写了 `gpu_monitor`（配置仍是产物的唯一描述）。"""
    generate_matrix = _load_generate_matrix()
    anchor = generate_matrix.GPU_MONITOR_CONTAINER_DIR
    tiers = {tier["tier_id"]: tier for tier in generate_matrix.build_ledger()}
    for tier_id in ("T001", "T010", "T046"):
        status, reason = generate_matrix.classify_expressibility(tiers[tier_id])
        text = generate_matrix.render_config_yaml(tiers[tier_id], status, reason)
        assert "\ngpu_monitor:\n" in text, f"{tier_id} 渲染结果缺 gpu_monitor 段"
        assert f"  tag: {tier_id}\n" in text
        assert f"  output_dir: {anchor}\n" in text, (
            f"{tier_id} 的采样落点必须渲染成取证链锚点 {anchor}"
        )


def _load_collector():
    """按路径加载 `scripts/collect_results.py`（模块级只 import 标准库，可独立加载）。"""
    spec = importlib.util.spec_from_file_location(
        "matrix54_collect_results_anchor", _REPO_ROOT / "scripts" / "collect_results.py"
    )
    module = importlib.util.module_from_spec(spec)
    # 必须先注册进 sys.modules：collect_results 里用了 @dataclass，dataclasses 解析类时
    # 会回查 sys.modules[cls.__module__]（generate_matrix 无 dataclass，故其加载器不需要）。
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_gpu_evidence_dir_is_one_anchor_across_producer_runner_and_collector(tmp_path: Path):
    """★★ 取证链单点真相源（§1.4）：显存摘要的「谁写 / 断言在哪 / 谁读」必须同一条路径。

    历史缺陷（2026-09-20 真机确证的两个第二真相源）：
      ① 生产端 `gpu_monitor.output_dir` 一度 = `<training.output_dir>/gpu` ⇒ 落进训练侧
         `overwrite_output_dir`（对非空输出目录 `shutil.rmtree`）的删除半径 ⇒ 采样器
         刚建好的 `gpu/` 被删、首个 `append_jsonl` 即 ENOENT；
      ② 断言端 entry.sh 收口段写死另一条 `/out/gpu` ⇒ 断言找不到摘要，且缺口登记
         自身因目录不存在而 ENOENT（228 实测 entry.sh 行 197/200）。

    本用例把四个使用点一次钉在同一个锚点上，任一漂移即失败。

    ★ 2026-09-21 契约变更（「方案 A：二分口径」）后的边界：本用例钉的是**宿主采样摘要的
      取证链锚点**（生产端写 → entry.sh 断言 → `_read_host_sample_peak_memory` 读），
      以及**§7 峰值列按后端各自落点取数**这条新契约；宿主采样摘要**不再是** §7 峰值列的
      出处（旧断言 `_read_peak_memory(run_dir) == 2.0` 编码的正是被废止的单一口径）。
    """
    generate_matrix = _load_generate_matrix()
    anchor = generate_matrix.GPU_MONITOR_CONTAINER_DIR

    # ① 锚点本体：RUN_DIR 直下（容器内 /out = 宿主 <RUN_ROOT>/<T###>）
    assert anchor == "/out/gpu", f"取证链锚点被改：{anchor!r}"

    # ② 生产端：档配置落点 == 锚点，且不在训练输出目录之内
    tier = next(t for t in generate_matrix.build_ledger() if t["tier_id"] == "T010")
    config = generate_matrix.build_config(tier)
    training_out = config["training"]["output_dir"]
    assert config["gpu_monitor"]["output_dir"] == anchor
    assert not anchor.startswith(f"{training_out}/"), "锚点落在训练输出的删除半径内"

    # ③ 挂载映射 + 断言端模板：容器 /out ⇔ 宿主 $RUN_DIR；前置校验把不变量写成代码
    runner_text = _RUNNER.read_text(encoding="utf-8")
    assert '"$RUN_DIR|/out|rw"' in runner_text, "挂载表不再把 /out 绑到 $RUN_DIR"
    assert f'!= "{anchor}"' in runner_text, "前置校验不再比对取证链锚点"
    assert '"$TRAINING_OUTPUT_DIR"/*' in runner_text, "反向不变量（落点不得落在训练输出目录内）缺失"

    # ④ 断言端（entry.sh 生成物）：唯一变量 + 同一锚点值；收口段不得再写死路径
    entry = _entry_text(tmp_path)
    assert f'GPU_EVIDENCE_DIR="{anchor}"\n' in entry, "entry.sh 未定义取证目录变量"
    assert 'mkdir -p "$GPU_EVIDENCE_DIR"\n' in entry, (
        "entry.sh 未在写盘前建出取证目录（缺口登记会 ENOENT）"
    )
    section = _gpu_summary_section_text()
    assert "/out/gpu" not in section, f"收口段又出现写死路径（第二真相源）：{section}"
    assert "\\$GPU_EVIDENCE_DIR/gpu_memory_summary.json" in section
    assert "\\$GPU_EVIDENCE_DIR/EVIDENCE_GAP_gpu_memory_summary" in section

    # ⑤ 消费端（行为实证，不依赖源码格式）：<run>/gpu/ 摘要 = 宿主侧那个锚点，仍必须读得到；
    #    §7 峰值列按新契约改由各后端自己的落点供给（见下方 ⑤′）。
    collector = _load_collector()
    run_dir = tmp_path / "runs" / "T010"
    (run_dir / "gpu").mkdir(parents=True)
    (run_dir / "gpu" / "gpu_memory_summary.json").write_text(
        json.dumps({"per_gpu": {"0": {"memory_used_mib_peak": 2048.0}}}), encoding="utf-8"
    )
    # ★ 契约变更（2026-09-21 指挥官裁定「方案 A：二分口径」）：§7 峰值列**不再**读宿主
    #   采样摘要，改为 `_read_peak_memory(run_dir, backend) -> (值, 口径标签)` **按后端选口径**。
    #   本条用例的**原意不变**（生产端 / 断言端 / 消费端同源：谁写、断言在哪、谁读必须对得上），
    #   只是「消费端那一条落点」随口径改到了各后端自己的产物上 ⇒ 断言随之改指新落点，
    #   并**保留旧约束**：宿主采样摘要仍必须是**可被宿主侧读到的**那一个锚点（它只是不再
    #   供 §7 峰值列使用），且**单靠它就填不出 §7 峰值列**（防「静默混口径」）。
    assert collector._read_host_sample_peak_memory(run_dir) == 2.0, (
        "collect_results 读的宿主采样摘要不再是 <run_dir>/gpu/gpu_memory_summary.json "
        "⇒ 生产端与宿主侧消费端不同源（取证链锚点漂移）"
    )
    assert collector._read_peak_memory(run_dir, "native") == (
        None,
        collector._judge.PEAK_MEMORY_CALIBER,
    ), "宿主采样摘要在场就填出了 native 档 §7 峰值列 ⇒ 又回落到被废止的单一口径"
    # 消费端（行为实证）：native 档 §7 峰值列的**合法落点** = rank0 的逐 rank 旁路，由
    # `_read_peak_memory` 按后端分派后 `rglob` 到；写进去必须读得到。
    rank_dir = run_dir / "outputs" / "T010" / "logs"
    rank_dir.mkdir(parents=True)
    (rank_dir / "rank_metrics.rank_00000.jsonl").write_text(
        json.dumps(
            {
                "event": "rank_memory",
                "phase": "sft_train_batch_after",
                "memory": {"max_allocated_mib": 2048.0},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    assert collector._read_peak_memory(run_dir, "native") == (
        2.0,
        collector._judge.PEAK_MEMORY_CALIBER,
    ), "native 档没有从 rank0 的 rank_metrics 旁路取到 §7 峰值 ⇒ 消费端与生产端不同源"
    # ms-swift 档走**它自己的**落点（自报 `memory(GiB)` = max_memory_reserved），
    # 与 native 不是同一条路径 —— 单值不可推导，必须按后端显式分派。
    (run_dir / "logging.jsonl").write_text(
        json.dumps({"loss": 0.5, "memory(GiB)": 24.45, "train_speed(s/it)": 1.0}) + "\n",
        encoding="utf-8",
    )
    assert collector._read_peak_memory(run_dir, "ms-swift") == (
        24.45,
        collector._judge.MSSWIFT_RESERVED_CALIBER,
    ), "ms-swift 档没有从自报 memory(GiB) 取到 §7 峰值 ⇒ 该档仍结构性填不出值"


def test_gpu_summary_gap_registration_wording_and_behaviour(tmp_path: Path):
    """★取证缺口登记：文案准确（宿主采样旁路，**非** allocator、**非** §7 峰值列口径），
    且**行为不变**——摘要缺失仍写出登记文件、在场仍写 present。

    旧文案把 `gpu_memory_summary.json` 称作「容器内 allocator 主口径摘要」，两处都错：
    ① 它是**宿主侧 nvidia-smi 采样**（容器内 allocator 是 `rank_metrics` 的
       `memory.max_allocated_mib`，另一个量）；② 它**不是** §7 峰值列口径，只喂旁路字段
       `host_sample_peak_gib`（§7 峰值列 = 后端二分口径，唯一映射见
       `result_judge.PEAK_MEMORY_CALIBER_BY_BACKEND`）。本用例把"文案改了、逻辑没改"
       同时钉住：既做文本断言，也**真跑**那段 shell 看两条分支的产物。
    """
    # 不加引号的 heredoc（`cat > entry.sh <<ENTRY`）展开：`\$`/`\`` ⇒ 字面量交给内层，
    # `\\` ⇒ 内层行续接符 `\`（这就是 `printf '%s\n' \` 在真 entry.sh 里的形态）。
    section = _heredoc_expand(_gpu_summary_section_text()).replace("\\\\", "\\")
    assert "printf '%s\\n' \\\n" in section, "内层行续接符形态变了（本用例的行为验证会失真）"
    # ① 文案：旧表述清零，新表述在位且指向唯一真相源。
    assert "allocator 主口径" not in section, "旧表述「allocator 主口径」残留"
    assert "主口径摘要" not in section, "旧表述「主口径摘要」残留"
    assert "宿主采样旁路摘要" in section, "新表述缺失（宿主采样旁路摘要）"
    assert "host_sample_peak_gib" in section, "未指明它喂的是哪个旁路字段"
    assert "PEAK_MEMORY_CALIBER_BY_BACKEND" in section, "未指明 §7 峰值列口径的唯一真相源"
    # ② 文件名逐字未变（既有取证链的一部分，消费端按名字读）。
    assert "$GPU_EVIDENCE_DIR/gpu_memory_summary.json" in section
    assert "$GPU_EVIDENCE_DIR/EVIDENCE_GAP_gpu_memory_summary" in section
    assert "$GPU_EVIDENCE_DIR/gpu_summary.state" in section

    def _run_section(evidence_dir: Path) -> None:
        evidence_dir.mkdir(parents=True, exist_ok=True)
        script = tmp_path / "gpu_section.sh"
        script.write_text(
            'set -uo pipefail\nGPU_EVIDENCE_DIR="$1"\nSAMPLER=""\n' + section + "\n",
            encoding="utf-8",
        )
        completed = subprocess.run(
            ["bash", str(script), str(evidence_dir)],
            capture_output=True,
            text=True,
            check=False,
        )
        assert completed.returncode == 0, completed.stderr

    # ③ 分支 A：摘要缺失 ⇒ 仍写登记文件，且登记内容是新文案（fail-closed 不静默）。
    missing_dir = tmp_path / "gpu_missing"
    _run_section(missing_dir)
    gap = missing_dir / "EVIDENCE_GAP_gpu_memory_summary"
    assert gap.read_text(encoding="utf-8").strip() == (
        f"MISSING: {missing_dir}/gpu_memory_summary.json（宿主采样旁路摘要缺失；"
        "不是容器内 allocator，也不是 §7 峰值列口径——该列口径见 "
        "result_judge.PEAK_MEMORY_CALIBER_BY_BACKEND）"
    )
    assert (missing_dir / "gpu_summary.state").read_text(encoding="utf-8") == "missing\n"

    # ④ 分支 B：摘要在场 ⇒ present，且**不**写缺口登记（逻辑与旧版逐字同效）。
    present_dir = tmp_path / "gpu_present"
    present_dir.mkdir(parents=True, exist_ok=True)
    (present_dir / "gpu_memory_summary.json").write_text('{"per_gpu": {}}', encoding="utf-8")
    script = tmp_path / "gpu_section_present.sh"
    script.write_text(
        'set -uo pipefail\nGPU_EVIDENCE_DIR="$1"\nSAMPLER=""\n' + section + "\n",
        encoding="utf-8",
    )
    completed = subprocess.run(
        ["bash", str(script), str(present_dir)], capture_output=True, text=True, check=False
    )
    assert completed.returncode == 0, completed.stderr
    assert (present_dir / "gpu_summary.state").read_text(encoding="utf-8") == "present\n"
    assert not (present_dir / "EVIDENCE_GAP_gpu_memory_summary").exists(), (
        "摘要在场却写了缺口登记 ⇒ 登记逻辑被文案改动带偏"
    )


def test_entry_sh_passes_the_tier_config_to_record_gpu_memory(tmp_path: Path):
    """★不变量：entry.sh 用 `--config <该档配置>` 调 record-gpu-memory，且不再有旧参数。"""
    entry = _entry_text(tmp_path)
    record_lines = [line for line in entry.splitlines() if "record-gpu-memory" in line]
    assert len(record_lines) == 2, record_lines  # --idle-only + 采样
    sampler = [line for line in record_lines if "--idle-only" not in line]
    assert len(sampler) == 1, sampler
    line = sampler[0]
    # 旧产物参数一个都不许回来（§10.1：CLI 与 config 字段零交集）
    for banned in ("--output-dir", "--tag", "--interval-sec", "--pid-filter", "--recent-limit"):
        assert banned not in line, f"采样行又出现产物参数 {banned}：{line}"
    # 必须走 --config，且指向该档配置的容器内路径
    assert (
        'record-gpu-memory --config "/workspace/graspo/samples/configs/matrix54/T010.yaml"' in line
    ), line


def test_monitor_preflight_rejects_a_tier_mismatch(tmp_path: Path):
    """★负向（突变）：把档配置的 tag 改成非档号 ⇒ 宿主侧前置校验必须拒绝启动。"""
    target = _REPO_ROOT / "samples" / "configs" / "matrix54" / "T010.yaml"
    original = target.read_text(encoding="utf-8")
    bin_dir, elam_root, models_root = _make_fakes(tmp_path)
    try:
        target.write_text(
            original.replace("  tag: T010\n", "  tag: NOT-A-TIER\n"), encoding="utf-8"
        )
        completed = _run(tmp_path, bin_dir, elam_root=elam_root, models_root=models_root)
        assert completed.returncode != 0, "tag 与档号不一致必须 fail-closed"
        assert "monitor-link" in (completed.stderr + completed.stdout), completed.stderr
        assert not (tmp_path / "docker_argv.txt").exists(), "拒绝必须发生在 docker run 之前"
    finally:
        target.write_text(original, encoding="utf-8")
    assert target.read_text(encoding="utf-8") == original, "档配置必须恢复原样"


def test_monitor_preflight_rejects_a_missing_section(tmp_path: Path):
    """★负向（突变）：档配置缺 `gpu_monitor` 段 ⇒ 必须 fail-closed，不静默继续。"""
    target = _REPO_ROOT / "samples" / "configs" / "matrix54" / "T010.yaml"
    original = target.read_text(encoding="utf-8")
    bin_dir, elam_root, models_root = _make_fakes(tmp_path)
    try:
        head, sep, _tail = original.partition("\ngpu_monitor:\n")
        assert sep, "突变目标不存在：T010.yaml 里没有 gpu_monitor 段"
        target.write_text(head + "\n", encoding="utf-8")
        completed = _run(tmp_path, bin_dir, elam_root=elam_root, models_root=models_root)
        assert completed.returncode != 0, "缺 gpu_monitor 段必须 fail-closed"
        assert "monitor-link" in (completed.stderr + completed.stdout), completed.stderr
    finally:
        target.write_text(original, encoding="utf-8")
    assert target.read_text(encoding="utf-8") == original, "档配置必须恢复原样"


def test_real_cli_rejects_the_removed_output_options(tmp_path: Path):
    """★负向（真代码）：W3 之前那条命令行在**真** CLI 上必须被 argparse 拒绝。

    本机 python3.13 缺 torch，用 shim 按文件路径加载 `cli.gpu_monitor` 的
    `build_gpu_monitor_parser`：解析器本身是真代码（不 import graspo 包）。
    """
    import argparse as _argparse

    spec = importlib.util.spec_from_file_location(
        "gpu_monitor_cli_for_test",
        _REPO_ROOT / "src" / "graspo" / "cli" / "gpu_monitor.py",
        submodule_search_locations=[],
    )
    if spec is None or spec.loader is None:
        pytest.skip("无法按路径加载 gpu_monitor.py")
    # 该模块 import 了 graspo.core.gpu_guard：先按文件路径把 core 侧也挂上命名空间包，
    # 避免拉起重依赖的 graspo/__init__.py（与 W3 的验证工装同口径）。
    sys.path.insert(0, str(_REPO_ROOT / "src"))
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    except (ImportError, OSError, AttributeError) as exc:  # pragma: no cover
        # 本机缺依赖时如实跳过，绝不假装通过（异常类型收窄到「加载失败」这一类）
        pytest.skip(f"无法加载 gpu_monitor.py（{type(exc).__name__}: {exc}）")

    parser = _argparse.ArgumentParser(prog="graspo")
    sub = parser.add_subparsers(dest="cmd")
    module.build_gpu_monitor_parser(sub)
    for banned in ("--output-dir", "--tag", "--interval-sec", "--pid-filter", "--recent-limit"):
        with pytest.raises(SystemExit) as excinfo:
            parser.parse_args(["record-gpu-memory", banned, "x"])
        assert excinfo.value.code == 2, banned
    # 修后的形态必须能解析
    parsed = parser.parse_args(
        ["record-gpu-memory", "--config", "samples/configs/matrix54/T010.yaml"]
    )
    assert parsed.config.endswith("T010.yaml")


# ══════════════════════════════════════════════════════════════════════════════
# exit_code 不得假报 0：**假容器**下真跑整条 entry.sh（不碰 GPU / 不起真容器）
# ══════════════════════════════════════════════════════════════════════════════
#
# 原缺陷（228 实测，T028/T031）：worker 真实 rc=1（stdout 里 `exitcode: 1`），宿主
# 记录的 `<RUN_ROOT>/<T###>/exit_code` 却是 **0**。根因两层：
#   ① 容器 `entry.sh` 捕获了 `RC=$?`（torchrun 退出码）却**从未 `exit "$RC"`**，
#      脚本最后一条是 `printf ... > /out/ckpt_retention.state`（成功 ⇒ 0）；
#   ② 宿主 `docker run ...; echo "$?" > "$RUN_DIR/exit_code"` 忠实记录了这个 0。
# 修法：entry.sh 末尾单点收口 `exit "$RC"`（保留块内只写 `RC=4`，不自己 exit）。
#
# 下面的 harness 让整条链路在**假容器**里真跑：把 entry.sh 里的 `/out` 换成临时目录、
# 把容器命令换成 PATH 上的替身（torchrun / python / nvidia-smi），worker 退出码由
# `GRASPO_FAKE_WORKER_RC` 注入。**判据 = 真的失败过**：注入 rc=1/137/124 ⇒ 宿主侧
# 记录的 exit_code 必须等于它，而不是 0。


def _entry_sh_text(tmp_path: Path, tier: str = "T010") -> str:
    """跑一次 runner 拿到**渲染后**的 entry.sh（假 docker 不执行它，只记录 argv）。"""
    bin_dir, elam_root, models_root = _make_fakes(tmp_path)
    completed = _run(tmp_path, bin_dir, elam_root=elam_root, models_root=models_root, tier=tier)
    assert completed.returncode == 0, completed.stderr
    return (tmp_path / "runs" / tier / "entry.sh").read_text(encoding="utf-8")


def _fake_container_stub_bodies(tmp_path: Path) -> dict[str, str]:
    """容器命令替身（torchrun / python / nvidia-smi），供两段 harness 共用。

    ``torchrun`` 的退出码由 ``GRASPO_FAKE_WORKER_RC`` 注入 ⇒ 可精确复现
    "worker rc=1 / 137 / 124" 三种真实失败；``nvidia-smi`` 交回**真**解析器
    （按文件路径加载 gpu_guard，不拉起重依赖的 graspo/__init__）。
    """
    del tmp_path  # 参数保留：替身内容当前与临时目录无关，便于将来按目录定制
    import importlib

    grader = importlib.import_module("graspo.core.gpu_guard")
    return {
        # torchrun 替身：记录收到的 argv（供"透传真的到没到 worker"取证），再交回注入的 rc。
        # 记录是被动旁路（不设 GRASPO_FAKE_WORKER_ARGV 时什么都不做）⇒ 既有用例零影响。
        "torchrun": (
            "#!/bin/bash\n"
            'if [ -n "${GRASPO_FAKE_WORKER_ARGV:-}" ]; then printf \'%s\\n\' "$@" > "$GRASPO_FAKE_WORKER_ARGV"; fi\n'
            'exit "${GRASPO_FAKE_WORKER_RC:-0}"\n'
        ),
        "python": "#!/bin/bash\nexit 0\n",
        "nvidia-smi": (
            "#!/bin/bash\n# 存量卡隔离断言用的解析器（真代码，按文件路径加载）。\n"
            f'exec "{_PYBIN}" "{grader.__file__}" "$@"\n'
        ),
    }


def _fake_container_harness(tmp_path: Path, *, worker_rc: int) -> tuple[dict[str, str], Path]:
    """造"假容器"：PATH 替身 + GRASPO_FAKE_WORKER_RC，用于**在宿主上真跑 entry.sh**。

    返回 (env, fake_out)。调用方需把 entry.sh 里的 `/out` 换成 ``fake_out``。
    """
    bin_dir = tmp_path / "fake_container_bin"
    bin_dir.mkdir(parents=True, exist_ok=True)
    for name, body in _fake_container_stub_bodies(tmp_path).items():
        path = bin_dir / name
        path.write_text(body, encoding="utf-8")
        path.chmod(0o755)
    fake_out = tmp_path / "container_out"
    fake_out.mkdir(parents=True, exist_ok=True)
    env = {
        **os.environ,
        "PATH": f"{bin_dir}{os.pathsep}{os.environ.get('PATH', '')}",
        "GRASPO_FAKE_WORKER_RC": str(worker_rc),
        "TIER": "T010",
        # 容器内入口用它调 python（本 harness 用替身，不需要真解释器可运行时）。
        "CONTAINER_PYTHON": str(bin_dir / "python"),
        # torchrun 替身的 argv 旁路（取证"worker 到底收到了什么"）。
        "GRASPO_FAKE_WORKER_ARGV": str(tmp_path / "worker_argv.txt"),
    }
    return env, fake_out


def _run_entry_sh_via_host(
    tmp_path: Path,
    *,
    worker_rc: int,
    tier: str = "T010",
    with_checkpoint: bool = False,
    with_output_root: bool = False,
) -> tuple[subprocess.CompletedProcess[str], Path]:
    """把渲染后的 entry.sh 在宿主上真跑一遍（/out 重定向到临时目录）。

    ``with_output_root`` / ``with_checkpoint`` 用来把"训练产物目录"造成三种真实形态：
    不存在（none）/ 存在但无 ckpt（missing）/ 存在且有 1 份 ckpt（ok）——只有造出
    合法产物，"训练成功"这条路才不会被保留策略的"链路错误"（RC=4）拦下。
    """
    env, fake_out = _fake_container_harness(tmp_path, worker_rc=worker_rc)
    if with_output_root or with_checkpoint:
        run_root = fake_out / tier / f"{tier}-run"
        run_root.mkdir(parents=True, exist_ok=True)
        if with_checkpoint:
            ckpt = run_root / "checkpoint-2"
            ckpt.mkdir(parents=True, exist_ok=True)
            (ckpt / "trainer_state.json").write_text('{"global_step": 2}\n', encoding="utf-8")
    body = _entry_sh_text(tmp_path, tier=tier).replace("/out", str(fake_out))
    script = tmp_path / "entry_host.sh"
    script.write_text(body, encoding="utf-8")
    completed = subprocess.run(
        ["bash", str(script)], capture_output=True, text=True, check=False, env=env
    )
    return completed, fake_out


def test_entry_sh_propagates_worker_rc_one(tmp_path: Path):
    """★★核心负向用例：容器内 worker rc=1 ⇒ entry.sh 退出码必须是 1（不得是 0）。

    这正是 228 实测的假通过：T028/T031 stdout 里 `exitcode: 1`，宿主 `exit_code=0`。
    """
    completed, fake_out = _run_entry_sh_via_host(tmp_path, worker_rc=1)
    assert completed.returncode == 1, (
        f"exit_code 假报！worker rc=1 但 entry.sh 退了 {completed.returncode}"
        f"（state={_state(fake_out)}）stdout={completed.stdout} stderr={completed.stderr}"
    )


def test_entry_sh_propagates_oom_and_timeout_rc(tmp_path: Path):
    """OOM(137) / 超时(124) 都必须原样交还——不得被保留策略的 4 或 printf 的 0 盖掉。"""
    for rc in (137, 124):
        case_dir = tmp_path / f"rc_{rc}"
        case_dir.mkdir()
        completed, _ = _run_entry_sh_via_host(case_dir, worker_rc=rc)
        assert completed.returncode == rc, completed.stderr


def test_entry_sh_propagates_success_rc_zero(tmp_path: Path):
    """对照组：worker 成功 **且产物合法** ⇒ 退出码 0（不是恒为 4，也不是恒为非 0）。"""
    completed, fake_out = _run_entry_sh_via_host(tmp_path, worker_rc=0, with_checkpoint=True)
    assert completed.returncode == 0, completed.stderr
    assert _state(fake_out) == "ok", completed.stderr


def test_entry_sh_success_without_artifacts_is_a_hard_failure(tmp_path: Path):
    """对照组的反向：worker 成功却**无产物目录** ⇒ 4（运行链路错误，不得静默成功）。

    这条同时证明"退出码不是恒为 0"：rc 由真实链路决定，不是写死的常量。
    """
    completed, _ = _run_entry_sh_via_host(tmp_path, worker_rc=0, with_output_root=False)
    assert completed.returncode == 4, completed.stderr


def test_entry_sh_rc_beats_retention_four(tmp_path: Path):
    """★优先级：训练失败（rc≠0）**不得**被保留策略的 4 覆盖（保住真因）。

    构造"训练失败 + 产物目录存在但无 ckpt"⇒ 保留块记 state=missing、RC 保持原值；
    收口交出原 rc（如 137），**不是** 4。
    """
    completed, fake_out = _run_entry_sh_via_host(
        tmp_path,
        worker_rc=137,
        with_output_root=True,  # 目录在、但一份 ckpt 都没有
    )
    # entry.sh 里 OUTPUT_ROOT = /out/T010；harness 已重定向到 fake_out/T010。
    assert _state(fake_out) == "missing", completed.stderr
    assert completed.returncode == 137, (
        f"训练失败的真因被保留策略的 4 盖掉了：rc={completed.returncode}"
    )


#: 假 docker：**执行**传给它的 entry.sh（而非只记录 argv），并模拟 dockerd 的
#: "-w 工作目录 + -v 绑定挂载"语义（把容器内 /workspace/graspo、/out 指到宿主真实路径）。
#: 它在 PATH 第一位，因此 entry.sh 里的 torchrun / python / nvidia-smi 会命中替身。
_FAKE_DOCKER_RUNS_ENTRY = """#!/bin/bash
# 假 docker：**执行**传给它的 entry.sh（而非只记录 argv），并模拟 dockerd 的
# "-w 工作目录 + -v 绑定挂载"语义（把容器内路径指到宿主真实路径）。
# 它在 PATH 第一位，因此 entry.sh 里的 torchrun / python / nvidia-smi 会命中替身。
write_dir="${FAKE_DOCKER_WRITE_DIR:?}"
mkdir -p "$write_dir"
args=("$@")
envspecs=()
declare -A binds=()
declare -A binds=()

workdir="$PWD"
i=0
while [ "$i" -lt "${#args[@]}" ]; do
  case "${args[$i]}" in
    -w) workdir="${args[$((i + 1))]}"; i=$((i + 2)) ;;
    -e) spec="${args[$((i + 1))]}"; i=$((i + 2))
        # 忠实模拟 docker 的两种 -e 形态：-e VAR=VAL（显式赋值）与 -e VAR
        # （从宿主环境透传；宿主未设 ⇒ 容器内即为未设置）。runner 的
        # GRASPO_DETERMINISM 守卫用的正是后者——写成 -e "VAR=$VAR" 会在宿主未设时
        # 注入空串，与"未设置"语义不同（§2.2：默认关必须真的是"没设"）。
        case "$spec" in
            *=*) envspecs+=("$spec") ;;
            *) if [ -n "${!spec:-}" ]; then envspecs+=("$spec=${!spec}"); fi ;;
        esac ;;
    -v) spec="${args[$((i + 1))]}"; i=$((i + 2))
        rest="${spec#*:}"; dst="${rest%%:*}"
        src="${spec%%:*}"; binds["$dst"]="$src" ;;
    bash) entry="${args[$((i + 1))]}"; break ;;
    *) i=$((i + 1)) ;;
  esac
done
[ -n "${entry:-}" ] || { echo "fake-docker: 找不到 entry.sh 参数" >&2; exit 127; }
host_entry="${binds[$entry]:-$entry}"
[ -f "$host_entry" ] || {
  echo "fake-docker: entry.sh 不存在: $entry -> $host_entry" >&2; exit 127; }
cp -- "$host_entry" "$write_dir/entry_ran.sh"
# 绑定目标翻译成宿主路径后交给真 bash 执行（等价于容器内 `bash /entry.sh`）。
# 绑定既是"挂载"（路径翻译）也是"环境可见"（-e 之外的容器路径变量），两者都演出来。
for k in "${!binds[@]}"; do envspecs+=("$k=${binds[$k]}"); done
# 容器命令替身只在**容器 PATH** 生效（宿主 PATH 顺序不受影响）。
if [ -n "${CONTAINER_BIN:-}" ]; then envspecs+=("PATH=${CONTAINER_BIN}:$PATH"); fi
host_workdir="${binds[$workdir]:-$workdir}"
cd "$host_workdir" || exit 1
exec env "${envspecs[@]}" bash "$host_entry"
"""


def _run_runner_with_fake_container(
    tmp_path: Path,
    *,
    worker_rc: int,
    tier: str = "T010",
    with_checkpoint: bool = False,
    extra_host_env: dict[str, str] | None = None,
) -> tuple[subprocess.CompletedProcess[str], Path]:
    """跑完整 runner：假 docker **真执行** entry.sh ⇒ 宿主写下 ``exit_code``。

    返回 (runner 结果, RUN_DIR)。这是"容器 worker rc ⇒ 宿主 exit_code"的**端到端**
    证据链：真实链路里 `exit_code` 就是这里被写下的。

    两个 bin 目录**必须分开**（踩过的坑）：
      · ``bin_dir``（PATH 最后）只放"会执行 entry.sh 的假 docker" + 真 nvidia-smi 垫片，
        供 runner 自身的锁卡守卫使用（它用 $PYTHON，不走 python）；
      · ``container_bin``（PATH 最前）放容器命令替身（torchrun/python/nvidia-smi），
        供 entry.sh 使用——**不能放进 bin_dir**，否则 `python` 替身会把
        `gpu_idle_assert.py` 的读数挡掉 ⇒ 宿主侧"目标卡空闲断言"直接把运行拒掉。
    """
    bin_dir, elam_root, models_root = _make_fakes(tmp_path)
    fake_docker = bin_dir / "docker"
    fake_docker.write_text(_FAKE_DOCKER_RUNS_ENTRY, encoding="utf-8")
    fake_docker.chmod(0o755)
    container_bin = tmp_path / "fake_container_bin"
    container_bin.mkdir(exist_ok=True)
    for name, body in _fake_container_stub_bodies(tmp_path).items():
        stub = container_bin / name
        stub.write_text(body, encoding="utf-8")
        stub.chmod(0o755)
    docker_write_dir = tmp_path / "docker_write"
    docker_write_dir.mkdir(exist_ok=True)
    if with_checkpoint:
        # 训练真产出：<RUN_DIR>/<T###>/<T###>-run/checkpoint-2（保留策略识别的形态）
        ckpt = tmp_path / "runs" / tier / f"{tier}-run" / "checkpoint-2"
        ckpt.mkdir(parents=True, exist_ok=True)
        (ckpt / "trainer_state.json").write_text('{"global_step": 2}\n', encoding="utf-8")
    # PATH 优先级：bin_dir（假 docker + 真 nvidia-smi 垫片）> container_bin（容器替身）。
    # 容器替身只允许在**容器内**生效 —— 由假 docker 在 exec 时把它们 prepend 进容器 PATH
    # （见 _FAKE_DOCKER_RUNS_ENTRY 的 CONTAINER_BIN 处理），宿主侧顺序不受影响。
    completed = _run(
        tmp_path,
        bin_dir,
        elam_root=elam_root,
        models_root=models_root,
        tier=tier,
        extra_env={
            "FAKE_DOCKER_WRITE_DIR": str(docker_write_dir),
            "CONTAINER_BIN": str(container_bin),
            "GRASPO_FAKE_WORKER_RC": str(worker_rc),
            "GRASPO_FAKE_WORKER_ARGV": str(docker_write_dir / "worker_argv.txt"),
            **(extra_host_env or {}),
        },
        runner=_RUNNER,
    )
    return completed, (tmp_path / "runs" / tier)


def test_host_exit_code_equals_worker_rc_end_to_end(tmp_path: Path):
    """★★★ 端到端负向用例（本题最关键）：worker rc=1 ⇒ 宿主 `exit_code` 文件内容 = 1。

    这是 228 实测假通过（`exit_code=0`，stdout `exitcode: 1`）的直接回归。
    """
    completed, run_dir = _run_runner_with_fake_container(tmp_path, worker_rc=1)
    exit_code_file = run_dir / "exit_code"
    assert exit_code_file.is_file(), f"runner 没写 exit_code：{run_dir} stderr={completed.stderr}"
    recorded = exit_code_file.read_text(encoding="utf-8").strip()
    assert recorded == "1", (
        f"exit_code 假报！worker rc=1，宿主记的是 {recorded!r}"
        f"（runner rc={completed.returncode}）stderr={completed.stderr}"
    )


def test_host_exit_code_equals_worker_rc_for_oom_end_to_end(tmp_path: Path):
    """端到端：OOM(137) 也必须是 137（不是 0、不是 4）。"""
    completed, run_dir = _run_runner_with_fake_container(tmp_path, worker_rc=137)
    recorded = (run_dir / "exit_code").read_text(encoding="utf-8").strip()
    assert recorded == "137", (
        f"OOM 真因被改写：worker rc=137，宿主记 {recorded!r}，stderr={completed.stderr}"
    )


# ── 卡集合覆盖点（运行期：换清单即换卡；runner 逻辑零改动）────────────────────


def _manifest_with_gpu_override(tmp_path: Path, overrides: dict[str, list[int]]) -> Path:
    """从仓内清单派生一份**变体清单**（只改指定档的 `gpus`），落在 tmp。"""
    manifest = json.loads(_MANIFEST.read_text(encoding="utf-8"))
    for tier in manifest["tiers"]:
        if tier["tier_id"] in overrides:
            tier["gpus"] = overrides[tier["tier_id"]]
    manifest["gpu_assignment"] = {
        "source": "override",
        "override_spec": ";".join(
            f"{tier_id}={','.join(str(gpu) for gpu in gpus)}"
            for tier_id, gpus in sorted(overrides.items())
        ),
        "fingerprint_sha256": _gpu_assignment_fingerprint(
            (tier["tier_id"], tier["gpus"]) for tier in manifest["tiers"]
        ),
    }
    path = tmp_path / "matrix54_manifest.gpu-alt.json"
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


#: 假 nvidia-smi（多卡版）：把 GPU0–5 全报成空闲，好让"覆盖到 GPU4/5"真的走完整条
#: F-10 空闲断言（默认假 nvidia-smi 只认 0 号卡，会把 GPU4 判成"没有读数"而拒绝）。
_FAKE_NVIDIA_SMI_SIX_IDLE = """#!/bin/bash
args="$*"
case "$args" in
  "-L") for i in 0 1 2 3 4 5; do echo "GPU $i: NVIDIA A800 (UUID: GPU-fake000$i)"; done; exit 0;;
esac
case "$args" in
  *--query-gpu=index,memory.used,utilization.gpu*)
    for i in 0 1 2 3 4 5; do echo "$i, 4, 0"; done; exit 0;;
  *--query-gpu=index,uuid*)
    for i in 0 1 2 3 4 5; do echo "$i, GPU-fake000$i, 4, 81916, 81920, 0, 40, 100.0"; done; exit 0;;
esac
exit 0
"""


def _use_six_idle_gpus(bin_dir: Path) -> None:
    fake = bin_dir / "nvidia-smi"
    fake.write_text(_FAKE_NVIDIA_SMI_SIX_IDLE, encoding="utf-8")
    fake.chmod(0o755)


def test_gpu_override_manifest_drives_visible_devices_without_runner_change(tmp_path: Path):
    """★覆盖生效：变体清单把 T001 挪到 GPU4 ⇒ dry-run `gpus=4`、两道守卫都收 4，且不进 docker。

    runner 逻辑**零改动**（它只从清单读 `tiers[].gpus`）——这正是"一处定义"的验收。
    """
    bin_dir, elam_root, models_root = _make_fakes(tmp_path)
    _use_six_idle_gpus(bin_dir)
    variant = _manifest_with_gpu_override(tmp_path, {"T001": [4]})

    completed = _run(
        tmp_path,
        bin_dir,
        elam_root=elam_root,
        models_root=models_root,
        tier="T001",
        mode="--dry-run",
        extra_env={"GRASPO_RUNNER_MANIFEST": str(variant)},
    )

    assert completed.returncode == 0, completed.stderr
    assert "[dry-run] T001 gpus=4 " in completed.stdout, completed.stdout
    # 运行产物里的生效卡集合行（既有守卫，无需新代码）：A4「同档两跑同卡位」的现场证据。
    assert "NVIDIA_VISIBLE_DEVICES=4 " in completed.stdout, completed.stdout
    # 覆盖后的卡真的过了 F-10 空闲断言（没走 GRASPO_SKIP_IDLE_ASSERT 的跳过路径）。
    assert "已跳过" not in completed.stderr, completed.stderr
    assert not (tmp_path / "docker_argv.txt").exists()

    # 2 卡档同理：GPU4,5（都在 NUMA1）也全程生效，卡数仍是 2（覆盖不改卡数）。
    two_card = _run(
        tmp_path,
        bin_dir,
        elam_root=elam_root,
        models_root=models_root,
        tier="T002",
        mode="--dry-run",
        extra_env={"GRASPO_RUNNER_MANIFEST": str(variant)},
    )
    # 变体只覆盖了 T001；T002 走默认 ⇒ 这一跑也证明"未点名档不受影响"。
    assert "[dry-run] T002 gpus=0,1 " in two_card.stdout, two_card.stdout

    both = _run(
        tmp_path,
        bin_dir,
        elam_root=elam_root,
        models_root=models_root,
        tier="T002",
        mode="--dry-run",
        extra_env={
            "GRASPO_RUNNER_MANIFEST": str(
                _manifest_with_gpu_override(tmp_path, {"T002": [4, 5]})
            )
        },
    )
    assert "[dry-run] T002 gpus=4,5 " in both.stdout, both.stdout
    assert "NVIDIA_VISIBLE_DEVICES=4,5 " in both.stdout, both.stdout
    assert "已跳过" not in both.stderr, both.stderr


def test_default_manifest_still_pins_one_card_tiers_to_gpu0(tmp_path: Path):
    """不覆盖时行为与旧版逐字相同：T001 仍在 GPU0（唯一的 backwards-compat 行为证据）。"""
    bin_dir, elam_root, models_root = _make_fakes(tmp_path)

    completed = _run(
        tmp_path,
        bin_dir,
        elam_root=elam_root,
        models_root=models_root,
        tier="T001",
        mode="--dry-run",
    )

    assert completed.returncode == 0, completed.stderr
    assert "[dry-run] T001 gpus=0 " in completed.stdout, completed.stdout


def test_gpu_override_runner_does_not_accept_a_tier_not_in_manifest(tmp_path: Path):
    """未知档号仍然 fail-closed（真因是"运行链路：清单里没这档"，不是静默按默认跑）。"""
    bin_dir, elam_root, models_root = _make_fakes(tmp_path)
    variant = _manifest_with_gpu_override(tmp_path, {"T001": [4]})

    completed = _run(
        tmp_path,
        bin_dir,
        elam_root=elam_root,
        models_root=models_root,
        tier="T099",
        mode="--dry-run",
        extra_env={"GRASPO_RUNNER_MANIFEST": str(variant)},
    )

    assert completed.returncode != 0
    assert not (tmp_path / "docker_argv.txt").exists()


def _gpu_assignment_fingerprint(pairs) -> str:
    """按文件路径独立加载共享模块（与仓内其它 e2e 测试同范式，不依赖 sys.path）。"""
    module_path = _REPO_ROOT / "tests" / "e2e" / "gpu_assignment.py"
    spec = importlib.util.spec_from_file_location("gpu_assignment_under_test", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.gpu_assignment_fingerprint(pairs)


# ── 产物属主（PG-13 / U5「产物跑完了测好了就删」）─────────────────────────────
#
# 缺陷（228 实测确证，两个独立现场）：镜像**没有 USER 指令** ⇒ 容器默认以 uid 0 运行
# ⇒ 容器写出的产物一律 ``root:root``，而宿主跑批账号对 root 文件**无 unlink 权限**
# ⇒ ``rm`` 报 Permission denied、``rmdir`` 失败、``sudo -n`` 要密码
# （task-f2-resync-verify §⑨：``<RUN_ROOT>/T010/`` 内 20 项 / 49,133,569 B ≈ 47 MiB
# 删不掉；task-e4-reclaim：r2/r3/r4 因树内 root:root 残留 522.7 MB，整树删不掉）。
# 共享机上残留只增不减，而 228 磁盘两次被顶到 100%。
#
# 修法（**防呆**，§2，不是"事后提权去删"）：让产物**从一开始**就归宿主跑批用户 ——
# ``docker run --user <宿主 uid>:<gid>``。非 root 进程写不了镜像里 root 的 ``/root``
# ⇒ 同时必须显式给一个可写的 HOME/缓存根与中性的 USER/LOGNAME。
#
# 本节的判据：① argv 里真有 ``--user <uid>:<gid>``；② dry-run 能看见生效用户（§2.2）；
# ③ uid/gid 与容器 HOME **只有一个来源**（§1.4）；④ 取不到/非法 ⇒ fail-closed（不静默回落）；
# ⑤ 显式回退必须留痕（§3.2 透明退路）；⑥ 既有守卫逐条未受影响。

#: 生成器（runner 模板的**唯一**来源）：单源断言在这里做。
_GENERATOR = _REPO_ROOT / "tests" / "e2e" / "generate_matrix.py"

#: 容器内 HOME/缓存根的字面量（与 generate_matrix.py::CONTAINER_HOME_DIR 同值）。
_CONTAINER_HOME = "/tmp/graspo-runner-home"

_IMAGE_NAME = "graspo-msswift:4.5.3"


def _docker_argv(tmp_path: Path) -> list[str]:
    """读出假 docker 记下的 argv（一次运行一份）。"""
    path = tmp_path / "docker_argv.txt"
    assert path.is_file(), "假 docker 没被调用 ⇒ 本次运行根本没进 docker"
    return path.read_text(encoding="utf-8").splitlines()


def test_docker_argv_runs_the_container_as_the_host_user(tmp_path: Path):
    """★主判据：``docker run`` 必须带 ``--user <宿主 uid>:<gid>``。

    产物属主 = 写它的进程的 uid ⇒ 只有让容器进程用**宿主跑批用户**的 uid/gid，
    产物才归宿主用户，U5「跑完即删」才真的执行得了（不需要任何提权）。
    """
    bin_dir, elam_root, models_root = _make_fakes(tmp_path)

    completed = _run(tmp_path, bin_dir, elam_root=elam_root, models_root=models_root)

    assert completed.returncode == 0, completed.stderr
    argv = _docker_argv(tmp_path)
    assert "--user" in argv, argv
    assert argv[argv.index("--user") + 1] == f"{os.getuid()}:{os.getgid()}", argv
    # 必须是 `docker run` 的选项（落在 image 之前），否则会沦为容器命令的一部分。
    assert argv.index("--user") < argv.index(_IMAGE_NAME), argv


def test_dry_run_reports_the_effective_container_user(tmp_path: Path):
    """★显式即防呆（§2.2）：预检就必须看得见"实际使用哪个用户"。

    产物属主是"跑完即删"能否执行的前提，而它由 ``--user`` 决定 ⇒ 不能只在真跑时才可见。
    """
    bin_dir, elam_root, models_root = _make_fakes(tmp_path)

    completed = _run(
        tmp_path, bin_dir, elam_root=elam_root, models_root=models_root, mode="--dry-run"
    )

    assert completed.returncode == 0, completed.stderr
    assert f"[dry-run] user={os.getuid()}:{os.getgid()} " in completed.stdout, completed.stdout
    assert f"home={_CONTAINER_HOME}" in completed.stdout, completed.stdout
    # dry-run 是预检：不落任何运行产物（不进 docker）。
    assert not (tmp_path / "docker_argv.txt").exists()


def test_effective_user_has_exactly_one_source(tmp_path: Path):
    """★单一真相源（§1.4）：uid/gid 只解析一次，DOCKER_ARGS 只引用它。"""
    source = _RUNNER.read_text(encoding="utf-8")

    # 解析点各一处：``${GRASPO_RUN_UID:-$(id -u)}`` / ``${GRASPO_RUN_GID:-$(id -g)}``。
    assert source.count('RUN_UID="${GRASPO_RUN_UID:-$(id -u)}"') == 1
    assert source.count('RUN_GID="${GRASPO_RUN_GID:-$(id -g)}"') == 1
    assert source.count("$(id -u)") == 1, source.count("$(id -u)")
    assert source.count("$(id -g)") == 1, source.count("$(id -g)")
    # 两条分支都只写进**同一个数组**，docker run 处只展开该数组（不重算 uid/gid）。
    assert source.count("RUN_USER_ARGS=(") == 2
    assert '--user "$RUN_UID:$RUN_GID"' in source
    assert '"${RUN_USER_ARGS[@]}"' in source
    assert source.count('"${RUN_USER_ARGS[@]}"') == 1

    # 容器 HOME 与中性用户名：字面量在生成器里各只声明一次，其余使用点全靠常量注入。
    generator = _GENERATOR.read_text(encoding="utf-8")
    assert generator.count(f'CONTAINER_HOME_DIR = "{_CONTAINER_HOME}"') == 1
    assert generator.count(_CONTAINER_HOME) == 1, generator.count(_CONTAINER_HOME)
    assert generator.count('CONTAINER_USER_NAME = "graspo"') == 1
    # 缓存根清单（含 HOME 本身）**只在生成器的 CONTAINER_CACHE_ROOTS 里定义一次**；
    # 生成物里的路径字面量全部由它渲染 ⇒ 计数必须恰好是「表长 × 渲染份数（runner + entry.sh）」。
    generate_matrix = _load_generate_matrix()
    roots = generate_matrix.container_cache_root_pairs()
    assert roots[0][0] == "HOME", roots
    assert ("MODELSCOPE_CACHE", f"{_CONTAINER_HOME}/.cache/modelscope/hub") in roots, roots
    expected_array = generate_matrix.render_cache_roots_array()
    # 两份数组（runner 顶层 与 entry.sh 内）逐字节相同、且等于由清单渲染的结果。
    assert source.count(f"CONTAINER_CACHE_ROOTS=(\n{expected_array}\n)") == 2, (
        "runner 与 entry.sh 的缓存根数组必须同源同形（否则就是两份真相）"
    )
    # 生成物里不得存在**不在**该数组内的缓存根字面量（防"散落一处就再漂移一处"）。
    # 合法出现点只有两处：① 两份数组块（每份 5 行）；② 043c776 既有的 dry-run `home=` 单点。
    assert source.count(f"home={_CONTAINER_HOME}") == 1, source.count(f"home={_CONTAINER_HOME}")
    assert source.count(_CONTAINER_HOME) == 2 * len(roots) + 1, source.count(_CONTAINER_HOME)
    assert (
        _CONTAINER_HOME
        not in source.replace(expected_array, "").replace(f"home={_CONTAINER_HOME}", "")
    ), "缓存根字面量散落在数组之外 ⇒ 清单不再是单一真相源"


def test_container_home_and_cache_dirs_are_explicitly_writable(tmp_path: Path):
    """非 root 进程写不了镜像里 root 的 /root ⇒ HOME/缓存根必须显式给且先建出来。

    ★ 回归判据（228 实测，本包修的缺陷）：镜像 ENV 把 ``MODELSCOPE_CACHE`` 写死成
    ``/mnt/workspace/.cache/modelscope/hub``，而 ``/mnt`` 在镜像里是 ``root:root 0755``
    ⇒ 容器改以宿主 uid 运行后建不出 ``/mnt/workspace``，ms-swift 的 ``dataset/loader.py:57``
    拿 ``get_cache_dir()`` 当 ``datasets`` 的 ``cache_dir`` ⇒ 训练**在数据集装载阶段**就
    ``PermissionError``（rank0 exit 1，宿主只看到"未找到任何 checkpoint-*"）。
    只补 HOME/XDG/TORCHINDUCTOR/TRITON 四个是**不完整**的修复。
    """
    bin_dir, elam_root, models_root = _make_fakes(tmp_path)

    completed = _run(tmp_path, bin_dir, elam_root=elam_root, models_root=models_root)

    assert completed.returncode == 0, completed.stderr
    argv = _docker_argv(tmp_path)
    assert f"HOME={_CONTAINER_HOME}" in argv, argv
    for key in ("XDG_CACHE_HOME", "TORCHINDUCTOR_CACHE_DIR", "TRITON_CACHE_DIR"):
        assert any(token.startswith(f"{key}={_CONTAINER_HOME}") for token in argv), argv
    # ★ 本包的正主：modelscope 缓存根必须被显式指到可写根，且**不得**再是镜像里的
    #   root-only 路径（否则又是"数据集装载阶段 PermissionError"）。
    assert f"MODELSCOPE_CACHE={_CONTAINER_HOME}/.cache/modelscope/hub" in argv, argv
    assert not any(token.startswith("MODELSCOPE_CACHE=/mnt/") for token in argv), argv
    # 镜像 /etc/passwd 里没有宿主这个 uid ⇒ `getpass.getuser()` 会落到 pwd.getpwuid 抛
    # KeyError（ms-swift/deepspeed/wandb 启动横幅都调）⇒ 必须给中性值兜住。
    assert "USER=graspo" in argv, argv
    assert "LOGNAME=graspo" in argv, argv

    entry = (tmp_path / "runs" / "T010" / "entry.sh").read_text(encoding="utf-8")
    # entry.sh 里：五个根逐个 `mkdir -p` + 预检可写性；必须在训练**之前**跑完，
    # 且失败时**指名道姓**报出是哪个变量/哪个路径（不是含糊的 PermissionError）。
    assert 'CACHE_DIR="${CACHE_KV#*=}"' in entry, entry
    assert 'mkdir -p "$CACHE_DIR"' in entry, entry
    assert '[ ! -w "$CACHE_DIR" ]' in entry, entry
    assert "FATAL(cache-root)" in entry, entry
    assert "exit 8" in entry, entry
    # 预检块必须**在 torchrun 之前**（不依赖任何库"自己会建"的隐性假设）。
    assert entry.index("CONTAINER_CACHE_ROOTS=(") < entry.index("torchrun")
    assert entry.index("FATAL(cache-root)") < entry.index("torchrun")


def test_dry_run_and_real_run_print_the_same_injected_cache_roots(tmp_path: Path):
    """★显式即防呆（§2.2）：实际注入的缓存根清单必须**打印出来**，dry-run 与真跑一致。

    镜像 ENV 把 ``MODELSCOPE_CACHE`` 写死到 root-only 的 ``/mnt/workspace/...`` 正是
    "训练在数据集装载阶段就死"的根因；运维必须能在预检阶段就看见它被指到了哪里，
    而不是等 rank0 抛 ``PermissionError`` 再回头查（那时宿主只看到"未找到任何 checkpoint-*"）。
    """
    # ① dry-run：清单可见，且此时**不进 docker**（预检不落产物）。
    bin_dir, elam_root, models_root = _make_fakes(tmp_path)
    dry = _run(
        tmp_path, bin_dir, elam_root=elam_root, models_root=models_root, mode="--dry-run"
    )
    assert dry.returncode == 0, dry.stderr
    assert "[dry-run] cache-roots" in dry.stdout, dry.stdout
    for line in (
        f"[cache-roots] HOME={_CONTAINER_HOME}",
        f"[cache-roots] MODELSCOPE_CACHE={_CONTAINER_HOME}/.cache/modelscope/hub",
    ):
        assert line in dry.stdout, dry.stdout
    assert not (tmp_path / "docker_argv.txt").exists()
    printed_dry = [ln for ln in dry.stdout.splitlines() if ln.startswith("[cache-roots] ")]

    # ② 真跑：打印的清单必须与 `-e` **实际注入**的逐项相同（同一份表、同一个函数）。
    real_dir = tmp_path / "real"
    real_dir.mkdir()
    bin_dir, elam_root, models_root = _make_fakes(real_dir)
    real = _run(real_dir, bin_dir, elam_root=elam_root, models_root=models_root)
    assert real.returncode == 0, real.stderr
    printed_real = [ln for ln in real.stdout.splitlines() if ln.startswith("[cache-roots] ")]
    assert printed_real == printed_dry, (printed_real, printed_dry)
    injected = sorted(tok for tok in _docker_argv(real_dir) if "=" in tok and "/" in tok)
    for ln in printed_real:
        assert ln.removeprefix("[cache-roots] ") in _docker_argv(real_dir), ln
    assert any(tok.startswith("MODELSCOPE_CACHE=") for tok in injected), injected


def test_cache_root_preflight_names_the_unwritable_root(tmp_path: Path):
    """★预检判据（§2.3）：某个缓存根不可写时，训练之前就**指名道姓**报出变量+路径并 fail-closed。

    用 0555 目录精确复现"root-only 路径"的形状（非 root 建不出子目录）——这正是
    ``/mnt/workspace`` 在镜像里对宿主 uid 的样子。判据不是"报了错"，而是
    **报出的是哪一个变量、哪一个路径**（含糊的 PermissionError 无法定位）。
    """
    if os.geteuid() == 0:
        pytest.skip("root 会绕过权限位，无法在宿主上复现 0555 的不可写形状")
    env, fake_out = _fake_container_harness(tmp_path, worker_rc=0)
    # 每测试独立的 HOME 根：让 HOME / XDG_CACHE_HOME 两个根**可写**，
    # 只把 modelscope 那一层造成不可写 ⇒ 失败必须**精确指向 MODELSCOPE_CACHE**。
    home = tmp_path / "home_root"
    (home / ".cache" / "modelscope").mkdir(parents=True)
    (home / ".cache" / "modelscope").chmod(0o555)  # 建不出 modelscope/hub
    body = _entry_sh_text(tmp_path).replace("/out", str(fake_out))
    body = body.replace(_CONTAINER_HOME, str(home))
    assert f"MODELSCOPE_CACHE={home}/.cache/modelscope/hub" in body, body
    script = tmp_path / "entry_readonly_cache_root.sh"
    script.write_text(body, encoding="utf-8")

    completed = subprocess.run(
        ["bash", str(script)], capture_output=True, text=True, check=False, env=env
    )

    assert completed.returncode == 8, (completed.returncode, completed.stderr)
    assert "FATAL(cache-root)" in completed.stderr, completed.stderr
    assert f"MODELSCOPE_CACHE={home}/.cache/modelscope/hub" in completed.stderr, completed.stderr
    # ★ fail-closed 的判据：**没进训练**（torchrun 一次都没被调用）。
    assert "torchrun" not in completed.stdout, completed.stdout


def test_container_home_is_not_inside_the_artifact_tree(tmp_path: Path):
    """★HOME 不得落在 `/out`（产物树）里：否则 U5「跑完即删」清理时会撞上一个非产物目录。"""
    bin_dir, elam_root, models_root = _make_fakes(tmp_path)

    completed = _run(tmp_path, bin_dir, elam_root=elam_root, models_root=models_root)

    assert completed.returncode == 0, completed.stderr
    argv = _docker_argv(tmp_path)
    home = next(token for token in argv if token.startswith("HOME="))
    assert home == f"HOME={_CONTAINER_HOME}", argv
    assert not home.startswith("HOME=/out"), home
    assert not _CONTAINER_HOME.startswith("/out")


def test_declared_user_environment_override_is_honoured(tmp_path: Path):
    """可显式指定 uid/gid（容器 uid 与宿主 uid 不一致的环境用）：覆盖值原样进 argv。"""
    bin_dir, elam_root, models_root = _make_fakes(tmp_path)

    completed = _run(
        tmp_path,
        bin_dir,
        elam_root=elam_root,
        models_root=models_root,
        extra_env={"GRASPO_RUN_UID": "1234", "GRASPO_RUN_GID": "5678"},
    )

    assert completed.returncode == 0, completed.stderr
    argv = _docker_argv(tmp_path)
    assert argv[argv.index("--user") + 1] == "1234:5678", argv


@pytest.mark.parametrize("bad_uid", ["abc", "-1", "12x", "1 2"])
def test_invalid_declared_user_is_fail_closed_before_docker(tmp_path: Path, bad_uid: str):
    """★边界校验（§2.3）：uid/gid 取不到或非法 ⇒ **拒绝启动**，绝不静默回落成 root。

    静默回落成 root 会复现本条要根治的缺陷（产物属 root:root、宿主删不掉）⇒ 只能 fail-closed。
    """
    bin_dir, elam_root, models_root = _make_fakes(tmp_path)

    completed = _run(
        tmp_path,
        bin_dir,
        elam_root=elam_root,
        models_root=models_root,
        extra_env={"GRASPO_RUN_UID": bad_uid},
    )

    assert completed.returncode == 7, completed.stderr
    assert "FATAL(runtime-link): 无法确定宿主 uid/gid" in completed.stderr, completed.stderr
    assert "fail-closed" in completed.stderr, completed.stderr
    # 拒绝得**够早**：根本没进 docker（不会留下一个属主错误的运行目录）。
    assert not (tmp_path / "docker_argv.txt").exists()


def test_run_as_root_override_is_explicit_and_audited(tmp_path: Path):
    """★透明退路（§3.2）：回退成 root 必须**显式声明 + 留痕**，不得静默。

    留痕两处：① stderr 的显式 WARNING；② ``$RUN_ROOT/skipped_guards.log``（事后可审计）。
    """
    bin_dir, elam_root, models_root = _make_fakes(tmp_path)

    completed = _run(
        tmp_path,
        bin_dir,
        elam_root=elam_root,
        models_root=models_root,
        extra_env={"GRASPO_RUN_AS_ROOT": "1"},
    )

    assert completed.returncode == 0, completed.stderr
    assert "WARNING(透明退路 §3.2)" in completed.stderr, completed.stderr
    assert "GRASPO_RUN_AS_ROOT=1" in completed.stderr, completed.stderr
    assert "删不掉" in completed.stderr, completed.stderr
    argv = _docker_argv(tmp_path)
    # 即便回退，也是**显式** root（--user 0:0），不是"恰好沿用镜像默认用户"的隐性巧合。
    assert argv[argv.index("--user") + 1] == "0:0", argv

    log = (tmp_path / "runs" / "skipped_guards.log").read_text(encoding="utf-8")
    assert "run_as_root=1" in log, log
    assert "tier=T010" in log, log


def test_root_override_leaves_no_trace_when_not_requested(tmp_path: Path):
    """默认路径不得写 skipped_guards.log（留痕只属于显式回退，不能被日常运行污染）。"""
    bin_dir, elam_root, models_root = _make_fakes(tmp_path)

    completed = _run(tmp_path, bin_dir, elam_root=elam_root, models_root=models_root)

    assert completed.returncode == 0, completed.stderr
    assert "GRASPO_RUN_AS_ROOT" not in completed.stderr, completed.stderr
    log = tmp_path / "runs" / "skipped_guards.log"
    assert (not log.exists()) or "run_as_root=1" not in log.read_text(encoding="utf-8")


def test_existing_guards_survive_the_user_change(tmp_path: Path):
    """★逐条确认：本次改动只新增 ``--user`` 与可写 HOME，既有守卫一个都没削弱。"""
    bin_dir, elam_root, models_root = _make_fakes(tmp_path)

    completed = _run(tmp_path, bin_dir, elam_root=elam_root, models_root=models_root)

    assert completed.returncode == 0, completed.stderr
    argv = _docker_argv(tmp_path)
    # ① 锁卡单一路径：只认 NVIDIA_VISIBLE_DEVICES，仍然不用 --gpus。
    assert "NVIDIA_VISIBLE_DEVICES=0" in argv, argv
    assert "--gpus" not in argv, argv
    assert "--runtime=nvidia" in argv, argv
    # ② 多卡共享内存（≥2 卡必需）。
    assert "--ipc=host" in argv and "--shm-size=16g" in argv, argv
    # ③ 档号显式注入容器（ckpt 保留块靠它定位）。
    assert "TIER=T010" in argv, argv
    # ④ 挂载结构不变量（不得有目标落在只读挂载之下）。
    assert _nested_under_readonly(_mount_specs(argv)) == []
    # ⑤ 退出码收口：entry.sh 里 `exit "$RC"` 仍是唯一表达点（且仍在探测之后）。
    entry = (tmp_path / "runs" / "T010" / "entry.sh").read_text(encoding="utf-8")
    assert 'exit "$RC"' in entry, entry[-400:]
    assert _entry_final_exit_line() == 'exit "$RC"\n'
    assert entry.index("torch_probe.py") < entry.index('exit "$RC"')
    assert entry.index("wait_for_gpu_summary") < entry.index('exit "$RC"')
    # ⑥ 显存采样的取证缺口登记与 ckpt 保留策略都还在。
    assert "EVIDENCE_GAP_gpu_memory_summary" in entry
    assert "retain_single_checkpoint" in entry
    assert "chmod -R a+rX /out" in entry
    # ⑦ 本包新增的缓存根预检**不取代**任何既有守卫：它只在更早处 fail-closed，
    #    ⑤⑥ 的收口点与保留策略逐字未动（见上），`--user` 的 rc=7 分支也仍在。
    assert 'exit "$RC"' in entry
    assert "exit 8" in entry
    assert entry.index("exit 8") < entry.index('exit "$RC"')
    source = _RUNNER.read_text(encoding="utf-8")
    assert "exit 7" in source, "GRASPO_RUN_UID 取不到时的 fail-closed（rc=7）不得被削弱"
    assert "GRASPO_RUN_AS_ROOT" in source, "留痕退路（透明退路 §3.2）不得被移除"
    assert "skipped_guards.log" in source
    assert "GRASPO_IMAGE" in source or "$IMAGE" in source



# ── 确定性开关的**环境变量守卫**透传（端到端：宿主 env ⇒ 容器内 worker argv）──────
#
# 判据（§2.2 显式即防呆）：默认关时 worker 的 argv **不带** --determinism-spec；
# 宿主设了 GRASPO_DETERMINISM 时，值**逐字**到达 worker 的命令行，且两侧都打日志。
# 这条链路跨了 docker 的 `-e` 语义、entry.sh 的运行时守卫两层，静态断言不足以覆盖。


def _worker_argv(tmp_path: Path) -> list[str]:
    text = (tmp_path / "docker_write" / "worker_argv.txt").read_text(encoding="utf-8")
    return text.splitlines()


def test_determinism_guard_default_off_passes_no_spec_end_to_end(tmp_path: Path):
    """★默认关端到端：容器内 worker 的 argv **不带** --determinism-spec。"""
    completed, fake_out = _run_entry_sh_via_host(tmp_path, worker_rc=0)
    argv = (tmp_path / "worker_argv.txt").read_text(encoding="utf-8").splitlines()
    assert "--determinism-spec" not in argv, f"默认关却透传了：{argv}"
    # worker 起始命令仍在（守卫没有把既有参数挤掉）。
    assert "graspo.cli.train_worker" in argv, argv
    assert "--config" in argv
    # 默认关必须显式留痕（不是"静默什么都没发生"）。
    assert "GRASPO_DETERMINISM 未设置" in completed.stderr
    assert (fake_out).is_dir()


def test_determinism_guard_on_passes_the_value_verbatim_end_to_end(tmp_path: Path, monkeypatch):
    """★开启端到端：宿主 env 值**逐字**到达容器内 worker 的命令行。"""
    spec = '{"enabled":true,"probe_first_step":true}'
    monkeypatch.setenv("GRASPO_DETERMINISM", spec)
    completed, _ = _run_entry_sh_via_host(tmp_path, worker_rc=0)
    argv = (tmp_path / "worker_argv.txt").read_text(encoding="utf-8").splitlines()
    assert "--determinism-spec" in argv, f"开了却没透传：{argv}"
    # 原值**逐字**到达（不解析、不改写、不加引号）。
    assert argv[argv.index("--determinism-spec") + 1] == spec
    assert "graspo.cli.train_worker" in argv and "--config" in argv
    # 生效的开关必须显式可见（§2.2）。
    assert spec in completed.stderr


def test_determinism_guard_empty_string_is_treated_as_off(tmp_path: Path, monkeypatch):
    """空串与"未设置"同解（`:-` 展开）——不得把空串当成一个合法的 spec 传下去。"""
    monkeypatch.setenv("GRASPO_DETERMINISM", "")
    completed, _ = _run_entry_sh_via_host(tmp_path, worker_rc=0)
    argv = (tmp_path / "worker_argv.txt").read_text(encoding="utf-8").splitlines()
    assert "--determinism-spec" not in argv, argv
    assert "GRASPO_DETERMINISM 未设置" in completed.stderr
