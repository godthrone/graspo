"""CLI 解析与启动计划（graspo.cli.app）的单元测试。"""

import json
import subprocess
from pathlib import Path

import pytest
import yaml

from graspo.cli.app import build_launch_plan, build_parser
from graspo.core.schema import GraspoConfig


def test_cli_main_commands_parse():
    parser = build_parser()

    commands = [
        ["launch", "--config", "samples/configs/config_example.yaml"],
        ["export", "--config", "samples/configs/config_example.yaml"],
        ["validate-reward", "--data", "samples/data/sample.jsonl"],
        [
            "evaluate-checkpoint",
            "--config",
            "samples/configs/config_example.yaml",
            "--data",
            "samples/data/sample.jsonl",
        ],
        ["analyze-profile", "outputs/some_run"],
    ]
    for command in commands:
        args = parser.parse_args(command)
        assert callable(args.func)


@pytest.mark.parametrize("command", ["train", "prepare-data", "analyze"])
def test_cli_removed_commands_are_not_public(command):
    parser = build_parser()

    with pytest.raises(SystemExit):
        parser.parse_args([command])


def test_config_example_loads():
    """样例配置可加载，且其中声明的训练超参被**原样带进来**。

    ★ 2026-09-28 定位到的真因：本断言原写死 ``max_epochs == 100``（= schema 的
    **默认值**），而 ``samples/configs/config_example.yaml`` 自 29a3db0 起显式声明
    ``max_epochs: 20`` ⇒ 断言过期、该用例长期红（只是被本文件的另外两条失败掩盖）。
    样例配置归 ``samples/configs/`` 维护（**本包不改它**，见任务书范围），因此这里不再
    把某个具体数字抄第二遍（§1.4 单一真相源），改为断言"加载器把样例里声明的值带过来"
    ——字段若被 schema 丢掉/改名，默认值与声明值不等就会在这里暴露。
    """
    example_path = Path("samples/configs/config_example.yaml")
    config = GraspoConfig.from_yaml(example_path)
    declared = yaml.safe_load(example_path.read_text(encoding="utf-8"))

    assert config.training.max_epochs == declared["training"]["max_epochs"]
    assert config.training.max_epochs > 0
    assert config.training.max_new_tokens == 2048
    assert config.native.tp_size == 2


def test_launch_plan_native_uses_torchrun(tmp_path):
    config_path = _write_launch_config(
        tmp_path,
        backend="native",
        nproc_per_node="null",
        tensor_parallel=2,
        pipeline_parallel=1,
    )

    plan = build_launch_plan(config_path)

    assert plan.backend == "native"
    assert plan.uses_torchrun
    assert plan.nproc_per_node == 2
    assert plan.command[:5] == [
        "python",
        "-m",
        "torch.distributed.run",
        "--nnodes=1",
        "--node_rank=0",
    ]
    assert "graspo.cli.train_worker" in plan.command


def test_launch_plan_native_world_size_one_uses_single_process(tmp_path):
    config_path = _write_launch_config(
        tmp_path,
        backend="native",
        nproc_per_node="null",
        tensor_parallel=1,
        pipeline_parallel=1,
    )

    plan = build_launch_plan(config_path)

    assert plan.backend == "native"
    assert not plan.uses_torchrun
    assert plan.nproc_per_node == 1
    assert plan.command[:3] == ["python", "-m", "graspo.cli.train_worker"]


def test_launch_plan_rejects_world_size_mismatch(tmp_path):
    config_path = _write_launch_config(
        tmp_path,
        backend="native",
        nproc_per_node=1,
        tensor_parallel=2,
        pipeline_parallel=1,
    )

    with pytest.raises(SystemExit, match="world size"):
        build_launch_plan(config_path)


def test_launch_plan_rejects_missing_paths(tmp_path):
    config_path = _write_launch_config(
        tmp_path,
        backend="native",
        model_path="<MODEL_PATH>",
        tensor_parallel=1,
        pipeline_parallel=1,
    )

    with pytest.raises(SystemExit, match="model.model_path"):
        build_launch_plan(config_path)


def test_readmes_document_single_yaml_entry_and_exports():
    """README 必须文档化"单 YAML 入口 + 导出开关"。

    ★ 2026-09-28：原期望里写死了 ``samples/configs/config_example.yaml``。README 已被
    重写（现指向 ``sft_example.yaml`` / ``rl_example.yaml`` / ``a800x8_...yaml`` /
    ``config_example_msswift.yaml``），该文件名不再出现在 README ⇒ 本用例红。
    README 归**别的单写者**（见任务书范围），因此这里不去规定它该点名哪一份，
    改为断言**不变量**："每份 README 至少点名一个 ``samples/configs/`` 下的**已存在**
    配置文件"——既保住原意（README 必须给出可用的单 YAML 入口），又不再随改名失效。
    """
    shipped = {f"samples/configs/{path.name}" for path in Path("samples/configs").glob("*.yaml")}
    assert shipped, "samples/configs/ 下应有可点名的配置文件"
    expected = [
        "uv run graspo launch --config",
        "lora.target_modules",
        "peft-adapter",
        "merged-hf",
        "training.max_new_tokens=2048",
    ]
    for path in (Path("README.md"), Path("README.zh-CN.md")):
        text = path.read_text(encoding="utf-8")
        for item in expected:
            assert item in text, f"{path} 缺少 {item!r}"
        named = sorted(entry for entry in shipped if entry in text)
        assert named, f"{path} 必须至少点名一份存在的样例配置（候选：{sorted(shipped)}）"
        forbidden = ["hf-reference", "prepare-data", "train --config", "prompt-only"]
        for item in forbidden:
            assert item not in text, f"{path} 不应再出现 {item!r}"


#: tracked markdown 的**允许目录前缀白名单**（宪法 §19.2 / §15.1）。
#:
#: **为什么不枚举文件、也不断言个数**（2026-09-19 由指挥官裁定改成本质不脆的形式）：
#: 旧版断言 ``tracked_markdown == {5 个具体文件名}``。那条断言**在当时的 HEAD 上就是红的**
#: ——实际的 tracked ``.md`` 比它多得多，只是因为 ``tests/cli/`` 在无 torch
#: 的机器上收集期就报错而没人看见。它同时犯两个错：
#:   ① **枚举式**：每新增一个合法文档就要改一次测试（脆弱）；
#:   ② **计数式**：把"有几个文档"当成不变量（文档数**不是**不变量，文档**位置**才是）。
#:
#: 现在断言的是**位置不变量**：``*.md`` 只允许出现在这几个目录前缀下。
#: 新增合法文档时**无需改本测试**；把 markdown 写进不该写的地方（如 ``src/``、
#: ``.local/`` 之外的临时目录、根目录随手放的笔记）才会失败。
#:
#: **两类前缀的语义**（见 `_is_allowed_markdown_path`）：
#:   * **以 ``/`` 结尾** = 目录前缀（该目录及其子目录）；
#:   * **不以 ``/`` 结尾** = **仓库根专有前缀**（只匹配仓库根下的文件）。
#: 后者让"仓库根例外"**无法被借用到子目录**（子目录下的同名开头文件仍判违规）。
_ALLOWED_TRACKED_MD_PREFIXES: tuple[str, ...] = (
    "README",  # 根目录双语 README（README.md / README.zh-CN.md）
    "docs/",  # 项目级架构文档（architecture / flow / ripple）
    "samples/configs/",  # 样例配置目录说明（samples/configs/README.md）
)
#: ★ 2026-09-30（R13）：本白名单**只保留「训练框架 + 样例」的文档位置**。
#: 旧白名单里的证据/登记册/运行装置/根目录事故记录四类位置，其**文件本身**已按宪法 §16.1
#: 移出仓库（迁往 ``.local/repo-moved-out/``）⇒ 对应前缀同步删除（死配置不留）。
#: 要新增一类位置，请在此显式加前缀并说明理由（单一真相源就是本元组）。


def _is_allowed_markdown_path(path: str) -> bool:
    """``path``（仓库相对、``/`` 分隔）是否落在白名单内。

    ``/`` 结尾的前缀 = 目录前缀；**不以 ``/`` 结尾的前缀 = 仓库根专有前缀**
    （只匹配仓库根下的文件）。后者把仓库根例外**限制在仓库根**：
    子目录下的同名开头文件不因该前缀而被放行。
    """
    for prefix in _ALLOWED_TRACKED_MD_PREFIXES:
        if prefix.endswith("/"):
            if path.startswith(prefix):
                return True
        elif "/" not in path and path.startswith(prefix):
            return True
    return False


def _tracked_markdown_paths() -> list[str] | None:
    """``git ls-files "*.md"`` 的归一化结果（``/`` 分隔、去空行）。

    ``None`` = **本检出没有 ``.git``**（目标容器把仓库 bind-mount 成 ``/work``，不含 ``.git``）
    ⇒ 这个"有没有把 markdown 提交进来"的检查**在该环境不适用**，由调用方 skip 并说明原因。
    **不返回空列表**：空列表会被读成"一个 tracked md 都没有"——那是假陈述（§2.2 空值语义）。
    """
    repo_root = Path(__file__).resolve().parents[2]
    if not (repo_root / ".git").exists():
        return None
    result = subprocess.run(
        ["git", "ls-files", "*.md"],
        check=True,
        capture_output=True,
        text=True,
    )
    return sorted(
        line.strip().replace("\\", "/") for line in result.stdout.splitlines() if line.strip()
    )


def test_only_whitelisted_locations_hold_tracked_markdown():
    """tracked ``.md`` 只能出现在白名单目录前缀下（**位置**不变量，不是计数不变量）。

    **失败信息必须可操作**（宪法 §2.3 边界校验即防呆）：直接列出违规文件，
    而不是甩一句"数量不等于 N"——后者让排查者只能自己去数。
    """
    paths = _tracked_markdown_paths()
    if paths is None:
        pytest.skip(
            "no .git in this checkout (the GPU container bind-mounts the repo without .git) "
            "⇒ 'tracked markdown' is not decidable here; "
            "this check runs on the dev machine, which has .git"
        )

    violations = [path for path in paths if not _is_allowed_markdown_path(path)]

    assert not violations, (
        "tracked markdown must live under one of the whitelisted prefixes "
        f"{list(_ALLOWED_TRACKED_MD_PREFIXES)}; offending file(s):\n  - "
        + "\n  - ".join(violations)
        + "\n(合法文档放对了位置就无需改本测试；要新增一类位置，请在 "
        "_ALLOWED_TRACKED_MD_PREFIXES 里显式加前缀并说明理由。)"
    )


def test_whitelist_actually_rejects_a_bad_location():
    """★ 负向：把 markdown 放进白名单外的位置必须被判违规（防"白名单写成空断言"）。

    同时锁住一条**既有的测试约定**：`tests/conftest.py` 的文档串写着"新增
    `tests/README.md` 会被这条检查拦下"。改名/改判据后那条约定**必须仍然成立**，
    否则 conftest 里的说明就成了过期事实（本用例把它变成可核断言）。
    """
    allowed = _ALLOWED_TRACKED_MD_PREFIXES
    for bad in ("src/graspo/notes.md", "tests/README.md", "outputs/report.md", "notes.md"):
        assert not bad.startswith(allowed), (
            f"{bad} must still be reported as a violation (see tests/conftest.py guidance)"
        )
    # 白名单本身必须非空且都是"位置前缀"（不以 .md 结尾 —— 那是枚举，不是前缀）
    assert allowed, "whitelist must not be empty"
    for prefix in allowed:
        assert not prefix.endswith(".md"), (
            f"whitelist entry {prefix!r} looks like a file, not a directory prefix"
        )


def test_export_config_fields_default_and_validate(tmp_path):
    """ExportConfig drives export from YAML, no CLI overrides."""
    from graspo.core.schema import ExportConfig

    cfg = ExportConfig()
    assert cfg.checkpoint_path == ""
    assert cfg.export_format == "peft-adapter"
    assert cfg.export_output == ""
    assert cfg.final_formats == []

    # Validate choices via a full config load
    config_path = _write_export_config(tmp_path, export_format="merged-hf")
    config = GraspoConfig.from_yaml(config_path)
    assert config.export.checkpoint_path == "outputs/test/final"
    assert config.export.export_format == "merged-hf"
    assert config.export.export_output == "outputs/test/merged"


def test_export_cli_only_accepts_config():
    """graspo export only accepts --config (config-driven CLI)."""
    parser = build_parser()
    args = parser.parse_args(["export", "--config", "samples/configs/config_example.yaml"])
    assert callable(args.func)


def test_e2e_config_roundtrip(tmp_path):
    """Minimal e2e: write a complete config, load it, verify all sections."""
    config_path = _write_launch_config(
        tmp_path,
        backend="native",
        tensor_parallel=2,
        pipeline_parallel=1,
    )
    config = GraspoConfig.from_yaml(config_path)
    assert config.backend == "native"
    assert config.training.seed == 42
    assert config.training.reject_unparseable_groups is True
    assert config.native.tp_size == 2
    assert config.launch.nnodes == 1


def _write_export_config(tmp_path: Path, *, export_format: str) -> Path:
    config_path = tmp_path / "export_config.yaml"
    config_path.write_text(
        f"""backend: native
model:
  model_path: models/test
export:
  checkpoint_path: outputs/test/final
  export_format: {export_format}
  export_output: outputs/test/merged
""",
        encoding="utf-8",
    )
    return config_path


def _write_launch_config(
    tmp_path: Path,
    *,
    backend: str,
    tensor_parallel: int,
    pipeline_parallel: int,
    nproc_per_node: int | str = "null",
    model_path: str | None = None,
) -> Path:
    data_path = tmp_path / "train.jsonl"
    data_path.write_text(
        '{"messages":[{"role":"user","content":"p"}],"targets":[{"output":{"content":{"x":1}}}]}\n',
        encoding="utf-8",
    )
    output_dir = tmp_path / "out"
    model_value = model_path or str(tmp_path / "model")
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        f"""
backend: {backend}
model:
  model_path: {json.dumps(model_value)}
data:
  train_path: {json.dumps(str(data_path))}
training:
  output_dir: {json.dumps(str(output_dir))}
native:
  tp_size: {tensor_parallel}
  pp_size: {pipeline_parallel}
launch:
  nproc_per_node: {nproc_per_node}
  nnodes: 1
  node_rank: 0
  master_addr: 127.0.0.1
  master_port: 29500
  python: python
""",
        encoding="utf-8",
    )
    return config_path


def test_launch_plan_smoke_appends_flag(tmp_path):
    """--smoke 时命令追加 --smoke，跑 1 步即停。"""
    config_path = _write_launch_config(
        tmp_path,
        backend="native",
        nproc_per_node="null",
        tensor_parallel=2,
        pipeline_parallel=1,
    )
    plan = build_launch_plan(config_path, smoke=True)
    assert plan.command[-1] == "--smoke"
    # 非 smoke 不加
    plan2 = build_launch_plan(config_path)
    assert plan2.command[-1] != "--smoke"


def test_launch_plan_smoke_keeps_config_untouched(tmp_path):
    """--smoke 不修改 config 文件（learning_rate 保留原值）。"""
    config_path = _write_launch_config(
        tmp_path,
        backend="native",
        nproc_per_node="null",
        tensor_parallel=1,
        pipeline_parallel=1,
    )
    config = GraspoConfig.from_yaml(config_path)
    original_lr = config.training.learning_rate
    build_launch_plan(config_path, smoke=True)
    config_after = GraspoConfig.from_yaml(config_path)
    assert config_after.training.learning_rate == original_lr
