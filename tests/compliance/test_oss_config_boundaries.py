"""A 类合规 W3/W4 的回归与**负向**用例（宪法 §10.1 / §7.1 / §1.4）。

本文件锁定三类"改完还会退回去"的行为：

1. **§10.1 输出定位参数**：``graspo record-gpu-memory`` 的产物参数
   （``output_dir`` / ``tag`` / ``interval_sec`` / ``recent_limit`` / ``pid_filter``）
   只在 config 里；CLI 不得再有对应选项（推论 2：CLI 与 config 零交集）。
2. **§7.1 环境变量不得承载配置**：``GRASPO_RUN_ID``
   （已**整体删除**的通道，非"保留兼容"）与
   ``GRASPO_MODELS_HOST_ROOT`` / ``GRASPO_BASE_MODEL_FOR_COMPARE``
   （已删除通道）**都不得**影响产物位置或判据结论。
3. **§1.4 单一真相源**：``collect_results`` 的基座模型根只有一个来源
   （``--base-model-root``）。

每条负向用例的断言形态都是"设了环境变量后结果**不变**"——若哪天有人把环境变量
fallback 加回来，这些测试会立刻变红。
"""

from __future__ import annotations

import argparse
import ast
import importlib.util
from pathlib import Path

import pytest
from pydantic import ValidationError

from graspo.cli.app import _build_launch_env
from graspo.cli.gpu_monitor import build_gpu_monitor_parser
from graspo.core.schema import GraspoConfig, GpuMonitorConfig
from graspo.flow import logging as graspo_logging
from graspo.flow.adapters.models.qwen35_36.training_sft import (
    _REMOVED_NONFINITE_SKIP_ENV,
    _nonfinite_skip_preauthorized,
)

_REPO_ROOT = Path(__file__).resolve().parents[2]

#: 决定 ``record-gpu-memory`` 产物的字段（位置 + 内容）——一律不得出现在 CLI 上。
_PRODUCT_FIELDS = ("output_dir", "tag", "interval_sec", "recent_limit", "pid_filter")


# ── §10.1：CLI 与 config 零交集 ─────────────────────────────────────────────


def _parsed_record_gpu_memory(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    build_gpu_monitor_parser(parser.add_subparsers(dest="command"))
    return parser.parse_args(["record-gpu-memory", *(argv or [])])


@pytest.mark.parametrize(
    "banned_option",
    ["--output-dir", "--tag", "--interval-sec", "--pid-filter", "--recent-limit"],
)
def test_record_gpu_memory_rejects_output_affecting_options(banned_option: str) -> None:
    """产物相关参数已从 CLI 删除：再传即 argparse 报错（§10.1 输出约束）。"""
    with pytest.raises(SystemExit):
        _parsed_record_gpu_memory([banned_option, "x"])


def test_record_gpu_memory_cli_namespace_has_no_product_fields() -> None:
    """★负向：CLI 命名空间里不得出现任何决定产物的字段（零交集，§10.1 推论 2）。"""
    args = _parsed_record_gpu_memory()
    for field in _PRODUCT_FIELDS:
        assert not hasattr(args, field), f"{field} 不应由 CLI 提供（必须进 config）"
    assert hasattr(args, "config")


def test_gpu_monitor_config_is_the_single_source_for_products() -> None:
    """产物参数确实由 config 段承载（位置 + 内容）。"""
    config = GraspoConfig(
        gpu_monitor={
            "output_dir": "/out/gpu",
            "tag": "T016",
            "interval_sec": 2.0,
            "recent_limit": 30,
            "pid_filter": "python",
        }
    )
    monitor = config.gpu_monitor
    assert monitor.output_dir == "/out/gpu"
    assert monitor.tag == "T016"
    assert monitor.interval_sec == 2.0
    assert monitor.recent_limit == 30
    assert monitor.pid_filter == "python"


def test_gpu_monitor_defaults_do_not_change_existing_configs() -> None:
    """默认行为不变：既有配置（不含 gpu_monitor 键）行为逐位一致。"""
    monitor = GraspoConfig().gpu_monitor
    assert monitor.output_dir is None
    assert monitor.tag is None
    assert monitor.interval_sec == 1.0
    assert monitor.recent_limit == 120
    assert monitor.pid_filter is None


def test_gpu_monitor_empty_string_normalizes_to_none() -> None:
    """§2.2 序列化边界：YAML/TOML 的 ``""`` 在加载边界归一为 ``None``。"""
    monitor = GraspoConfig(gpu_monitor={"output_dir": "", "tag": "", "pid_filter": ""}).gpu_monitor
    assert monitor.output_dir is None
    assert monitor.tag is None
    assert monitor.pid_filter is None


@pytest.mark.parametrize(
    "bad",
    [{"interval_sec": 0}, {"interval_sec": -1.0}, {"recent_limit": -1}],
)
def test_gpu_monitor_rejects_illegal_numbers(bad: dict) -> None:
    """加载即校验：非法数值直接拒绝，不等跑到采样循环才发现（§2.3）。"""
    with pytest.raises(ValidationError):
        GpuMonitorConfig(**bad)


def test_gpu_monitor_rejects_unknown_field() -> None:
    """§7.2：拼错的字段名必须被 ``extra="forbid"`` 拒绝，不能静默忽略。"""
    with pytest.raises(ValidationError):
        GpuMonitorConfig(output_dirr="/out")


# ── §7.1：环境变量不得承载配置 ──────────────────────────────────────────────


@pytest.fixture
def _reset_run_id(monkeypatch: pytest.MonkeyPatch):
    """每个用 run_id 的用例都从"干净进程"开始，避免模块级缓存串味。"""
    monkeypatch.setattr(graspo_logging, "_run_id", None)
    yield
    monkeypatch.setattr(graspo_logging, "_run_id", None)


def test_run_id_env_var_cannot_move_log_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, _reset_run_id: None
) -> None:
    """★负向：config 绑定后，``GRASPO_RUN_ID`` 不得改变日志落盘目录（§10.1）。"""
    graspo_logging.set_run_id("20260101-000000")
    monkeypatch.setenv("GRASPO_RUN_ID", "ENV-MUST-BE-IGNORED")
    assert graspo_logging.run_log_dir(tmp_path) == tmp_path / "logs" / "20260101-000000"
    monkeypatch.setenv("GRASPO_RUN_ID", "ENV-OTHER-VALUE")
    assert graspo_logging.run_log_dir(tmp_path) == tmp_path / "logs" / "20260101-000000"


def test_run_id_env_var_is_removed_not_deprecated(
    monkeypatch: pytest.MonkeyPatch, _reset_run_id: None
) -> None:
    """★★负向：``GRASPO_RUN_ID`` 通道已**整体删除**，不再有任何兼容期（§7.1/§18.1）。

    它曾以"保留一个版本 + DeprecationWarning"过渡，并声明在 v0.26.0 删除；
    当前 tag 已到 v0.28.x ⇒ 过渡期已过，必须是不生效的死通道：即便设了它，
    未绑定时也只能拿到进程内时间戳，拿不到那个值。
    """
    monkeypatch.setenv("GRASPO_RUN_ID", "LEGACY-20250828-170000")
    assert graspo_logging.get_run_id() != "LEGACY-20250828-170000"


def test_removed_run_id_env_channel_has_no_reader() -> None:
    """★★负向：``GRASPO_RUN_ID`` 在真实代码里没有任何读取点（AST 判定）。

    用 AST 而不是子串匹配，避免把注释/文档字符串里的"历史说明"误当成读取点。
    """
    source = _REPO_ROOT / "src/graspo/flow/logging.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    readers = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Attribute) and node.attr in {"environ", "getenv"}
    ]
    assert readers == [], "flow/logging.py 不得再读进程环境（§7.1/§1.4）"


def test_set_run_id_rejects_empty_value(_reset_run_id: None) -> None:
    """防呆：空身份会静默退回环境变量/时间戳，必须当场拒绝。"""
    with pytest.raises(ValueError):
        graspo_logging.set_run_id("   ")


def test_run_log_id_is_rank_consistent() -> None:
    """★rank 一致性：同一 config 的 run_name ⇒ 同一日志目录名（不依赖本机时钟）。"""
    assert graspo_logging.run_log_id("graspo_20260919_143000") == "20260919-143000"
    assert graspo_logging.run_log_id("graspo_20260919_143000") == graspo_logging.run_log_id(
        "graspo_20260919_143000"
    )
    assert graspo_logging.run_log_id("explicit-run-name") == "explicit-run-name"


def test_launch_env_no_longer_injects_run_id() -> None:
    """★负向：``graspo launch`` 不再把 GRASPO_RUN_ID 注入 worker 环境。"""
    env = _build_launch_env(GraspoConfig())
    assert "GRASPO_RUN_ID" not in env


def test_allow_nonfinite_grad_skip_is_config_only(monkeypatch: pytest.MonkeyPatch) -> None:
    """★★负向：非有限梯度预授权只认 config——设了旧环境变量也**不会**被预授权。"""
    monkeypatch.setenv(_REMOVED_NONFINITE_SKIP_ENV, "1")
    assert _nonfinite_skip_preauthorized(False) is False
    assert _nonfinite_skip_preauthorized(True) is True


def test_removed_nonfinite_env_channel_has_no_reader() -> None:
    """★★负向：已删除的环境变量通道在真实代码里没有任何读取点（AST 判定）。

    用 AST 而不是子串匹配，避免把注释/文档字符串里的"历史说明"误当成读取点。
    """
    source = (
        _REPO_ROOT / "src/graspo/flow/adapters/models/qwen35_36/training_sft.py"
    )
    tree = ast.parse(source.read_text(encoding="utf-8"))
    readers = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Attribute) and node.attr in {"environ", "getenv"}
    ]
    assert readers == [], "training_sft.py 不得再读进程环境"


# ── §1.4：collect_results 的基座模型根只有一个来源 ──────────────────────────


def _load_collect_results():
    """按文件路径加载 ``scripts/collect_results.py``（它只依赖标准库）。

    必须在 ``exec_module`` **之前**登记到 ``sys.modules``：该模块内的
    ``@dataclass`` 在 exec 时依赖 ``cls.__module__`` 能在 ``sys.modules`` 里
    找到（与它自己的 ``_load_result_judge`` 同一原因，见该函数注释）。
    """
    import sys

    path = _REPO_ROOT / "scripts" / "collect_results.py"
    spec = importlib.util.spec_from_file_location("_collect_results_under_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_collect_results_base_model_root_ignores_env_var(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """★★负向（双真相源）：只设 ``GRASPO_MODELS_HOST_ROOT`` ⇒ 判据取不到根目录。

    旧写法 ``args.base_model_root or os.environ.get(...)`` 会让环境变量"恰好生效"；
    现在唯一来源是 CLI 参数。
    """
    collect = _load_collect_results()
    root = tmp_path / "host_models"
    (root / "Qwen3.5-9B").mkdir(parents=True)
    monkeypatch.setenv("GRASPO_MODELS_HOST_ROOT", str(root))

    tier = {"model_path": "/models/Qwen3.5-9B"}
    assert collect._base_model_dir(argparse.Namespace(base_model_root=None), tier) is None
    assert collect._base_model_dir(argparse.Namespace(base_model_root=str(root)), tier) == (
        root / "Qwen3.5-9B"
    )


def test_collect_results_cli_wins_when_env_var_also_set(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """★可核断言：CLI 与环境变量同时给值 ⇒ 只认 CLI，环境变量被彻底忽略。"""
    collect = _load_collect_results()
    cli_root = tmp_path / "cli_models"
    env_root = tmp_path / "env_models"
    (cli_root / "Qwen3.5-9B").mkdir(parents=True)
    (env_root / "Qwen3.5-9B").mkdir(parents=True)
    monkeypatch.setenv("GRASPO_MODELS_HOST_ROOT", str(env_root))

    tier = {"model_path": "/models/Qwen3.5-9B"}
    resolved = collect._base_model_dir(
        argparse.Namespace(base_model_root=str(cli_root)), tier
    )
    assert resolved == cli_root / "Qwen3.5-9B"
    assert resolved != env_root / "Qwen3.5-9B"


def _write_safetensors(path: Path, tensor: bytes) -> None:
    """写一个**可被 collector 解析**的最小 safetensors（纯 stdlib）。

    格式：8 字节小端头长 + JSON 头（``data_offsets`` 相对数据区起点）+ 原始数据。
    这是为了让"显式传入基座目录"的**正向**路径真的走到权重比较，而不是在
    "目录不存在/文件不可解析"处提前返回——否则负向断言会在旧代码上也通过（假绿）。
    """
    import json
    import struct

    header = json.dumps(
        {"w": {"dtype": "U8", "shape": [len(tensor)], "data_offsets": [0, len(tensor)]}}
    ).encode("utf-8")
    path.write_bytes(struct.pack("<Q", len(header)) + header + tensor)


def test_collect_results_full_weights_differ_ignores_env_var(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """★负向：``GRASPO_BASE_MODEL_FOR_COMPARE`` 不再是基座目录的来源。

    断言强度说明：基座目录里放了**可解析的 safetensors**，所以旧代码若仍读该
    环境变量，就会真的比出结论（同一权重 ⇒ ``False``）；修复后只认传入的
    ``base_model_dir``，``None`` ⇒ 取证缺口 ``None``。两者可区分（非假绿）。
    """
    collect = _load_collect_results()
    base = tmp_path / "base"
    base.mkdir()
    _write_safetensors(base / "model.safetensors", b"\x01" * 8)

    checkpoint = tmp_path / "ckpt"
    checkpoint.mkdir()
    _write_safetensors(checkpoint / "model.safetensors", b"\x01" * 8)

    monkeypatch.setenv("GRASPO_BASE_MODEL_FOR_COMPARE", str(base))

    # 环境变量不得让判据"恰好答出"结论：给不出基座目录 ⇒ 取证缺口。
    assert collect._full_weights_differ(checkpoint, None) is None
    # 显式传入同一基座 ⇒ 权重未变化（判据语义不变）。
    assert collect._full_weights_differ(checkpoint, base) is False


# ── 总闸：自定义 GRASPO_* 环境变量的读取点被逐个点名 ────────────────────────


def test_custom_graspo_env_var_reads_are_absent() -> None:
    """★★总闸：运行时代码里**不存在任何**自定义 ``GRASPO_*`` 环境变量读取点。

    过渡期已结束——``flow/logging.py`` 那处唯一的例外（曾发 DeprecationWarning
    并承诺 v0.26.0 删除）现已删除 ⇒ 允许清单为空。此测试把"零例外"变成可核
    事实——新增任何读取点都会在这里变红（§7.1/§1.4/§18.1）。

    标准基础设施变量（``NVIDIA_VISIBLE_DEVICES`` / ``CUDA_VISIBLE_DEVICES`` /
    ``NCCL_*`` …）不属自定义通道，§7.1 明确允许，故排除在点名之外。
    """
    targets = [
        "scripts/collect_results.py",
        "src/graspo/cli/app.py",
        "src/graspo/cli/gpu_monitor.py",
        "src/graspo/cli/train_worker.py",
        "src/graspo/flow/adapters/models/qwen35_36/training_sft.py",
        "src/graspo/flow/logging.py",
    ]
    #: §7.1 允许的标准基础设施变量前缀。
    allowed_prefixes = ("NVIDIA_", "CUDA_", "NCCL_", "PYTORCH_", "RANK", "WORLD_SIZE", "LOCAL_RANK")

    offenders: list[str] = []
    for relative in targets:
        tree = ast.parse((_REPO_ROOT / relative).read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            is_environ_get = (
                isinstance(func, ast.Attribute)
                and func.attr == "get"
                and isinstance(func.value, ast.Attribute)
                and func.value.attr == "environ"
            )
            if not is_environ_get or not node.args:
                continue
            label = ast.unparse(node.args[0])
            if label.strip("'\" ").startswith(allowed_prefixes):
                continue
            offenders.append(f"{relative} -> {label}")

    assert offenders == [], (
        f"仍有自定义 GRASPO_* 环境变量读取点：{offenders}；§7.1 只允许标准基础设施变量"
    )
