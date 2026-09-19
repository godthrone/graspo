"""``GraspoConfig.eval`` 可选段的回归测试。

**背景（已在共享工作树上复现过的真实回归）**：``eval`` 段最初被声明为
``EvalConfig = Field(default_factory=EvalConfig)`` 且其校验器强制要求
``dataset_path`` / ``base_model_path`` / ``output_dir`` 非空。结果是**任何不含
``eval:`` 段的配置都加载失败**——包括仓库既有的三个样例配置，进而阻塞
``train_worker``（它第一件事就是 ``GraspoConfig.from_yaml``）等所有训练入口。

根因是两处叠加（宪法 §2.2：None 才是"未提供"）：
1. 缺省值不是 ``None``，而是"空 EvalConfig"，校验器照样跑；
2. ``from_dict`` 的 None 段归一循环把 ``eval`` 也归一成 ``{}``，同样触发校验。

修复后语义：**"未提供 eval 段" == "不做任何 eval 校验"**。

本文件的测试是**双向**的：既锁"不含 eval 段必须能加载"，也锁"显式写了 eval 段
但字段缺失必须报错"——只锁一侧会漏掉"校验被整体关掉"的退化。
"""

from __future__ import annotations

import glob
import importlib.machinery
import sys
from pathlib import Path
from types import ModuleType

import pytest


def _ensure_namespace(name: str, source_dir: Path) -> None:
    """确保 ``name`` 是一个**指向真实源码目录**的包命名空间；已存在则不动。

    为什么不能"随便造个 ModuleType 塞进 sys.modules"：
    ``sys.modules`` 是**全局**状态，测试文件在收集期留下的条目会污染**后继测试
    文件**的 import。这条路径上踩过一次真实缺陷——本文件曾把
    ``sys.modules["graspo.ripple"]`` 装成 ``__path__ = []`` 的假包，于是后继的
    ``test_schema_msswift.py`` 收集期报 ``No module named 'graspo.ripple.monitoring'``
    / ``graspo.ripple.buffer``——**错误信息把人引向与我们完全无关的方向**。

    防呆（宪法 §2）的做法不是"记得还原"，而是**让污染不可能发生**：

    - 只在 ``name`` 尚不存在时安装（绝不覆盖别人已装好的东西）；
    - ``__path__`` 一律指向**真实源码目录**，因此该包下的子模块仍可被任何
      后继测试正常发现——这正是原缺陷的反面；
    - 不凭空造叶子模块（如假的 ``graspo.ripple.reward.reward``），那会遮蔽真实现。
    """
    if name in sys.modules:
        return
    module = ModuleType(name)
    # 必须装一个**真的 ModuleSpec**：`importlib.util.find_spec("pkg.sub")` 会先看
    # 父包的 `__spec__`，父包 `__spec__ is None` 时直接抛 ValueError。只设
    # `__path__` 不够——那正是"看起来像包、实际不是"的半吊子状态。
    spec = importlib.machinery.ModuleSpec(name, loader=None, origin=None, is_package=True)
    spec.submodule_search_locations = [str(source_dir)]
    module.__spec__ = spec  # type: ignore[attr-defined]
    module.__loader__ = None  # type: ignore[attr-defined]
    module.__path__ = [str(source_dir)]  # type: ignore[attr-defined]
    module.__package__ = name
    module.__doc__ = (
        f"test-side namespace for {name!r}: points at the real source tree so that "
        "submodules stay importable for every later test file."
    )
    sys.modules[name] = module
    if "." in name:
        parent_name, attribute = name.rsplit(".", 1)
        setattr(sys.modules[parent_name], attribute, module)


def _prepare_import_namespaces(src_root: Path) -> None:
    """把相关**包**的 ``__path__`` 指向真实目录，仅此而已。

    ``graspo/__init__.py`` → ``graspo.ripple.algorithm`` → ``torch``，因此无 torch
    时任何 ``import graspo.*`` 都在收集期失败。把包的 ``__init__`` 跳过、但 ``__path__``
    指向真实目录之后：

    - ``graspo.core.schema`` 可正常导入（它只依赖 pydantic）；
    - 它延迟导入的 ``graspo.ripple.reward.reward`` 也走**真实现**（只依赖 pydantic）；
    - ``graspo.ripple.monitoring`` / ``graspo.ripple.buffer`` 等子模块对后继测试
      仍然可发现（原缺陷就是这里被 ``__path__ = []`` 挡死的）。
    """
    _ensure_namespace("graspo", src_root)
    _ensure_namespace("graspo.core", src_root / "core")
    _ensure_namespace("graspo.eval", src_root / "eval")
    _ensure_namespace("graspo.ripple", src_root / "ripple")
    # ``ripple/reward/__init__.py`` 会 re-export 重量级名字并触发循环导入链，
    # 因此 reward 子包也按命名空间处理（叶子 ``reward.reward`` 本身只依赖 pydantic）。
    _ensure_namespace("graspo.ripple.reward", src_root / "ripple" / "reward")


#: 只用真实 import 机制取配置类（不再手工装载 / 不再造门面对象）。
#: ``schema.py`` 内部的延迟导入（``graspo.ripple.reward.reward``）依赖正常的包解析,
#: 手工装载会逼我们伪造父包——那正是"污染全局 sys.modules"缺陷的来源。
_prepare_import_namespaces(Path(__file__).resolve().parents[2] / "src" / "graspo")
from graspo.core.schema import EvalConfig, GraspoConfig  # noqa: E402

__all__ = ["EvalConfig", "GraspoConfig"]

#: 仓库里的样例配置目录（用绝对路径定位，不依赖测试的 cwd）。
_REPO_ROOT = Path(__file__).resolve().parents[2]
_CONFIGS_DIR = _REPO_ROOT / "samples" / "configs"

#: 仓库既有、**不含** `eval:` 段的样例配置（回归的正是它们加载失败）。
_EXISTING_CONFIGS_WITHOUT_EVAL = (
    "sft_example.yaml",
    "config_example.yaml",
    "config_example_msswift.yaml",
)


def test_import_shim_does_not_pollute_global_sys_modules():
    """回归：本文件的导入垫片**不得**污染全局 ``sys.modules``。

    这是对一次真实缺陷的锁定。原实现把 ``sys.modules["graspo.ripple"]`` 装成
    ``__path__ = []`` 的**假包**，于是后继的 ``test_schema_msswift.py`` 在收集期报
    ``No module named 'graspo.ripple.monitoring'`` / ``graspo.ripple.buffer``——
    文件名与错误信息完全无关，**把排查者引向错误方向**。

    断言分两层：
    1. 本文件加载后，``sys.modules`` 里任何 ``graspo*`` 条目若要"看起来像个包"，
       其 ``__path__`` 必须非空 ⇒ 子模块仍可被后继测试发现；
    2. ``graspo.ripple`` 被垫片接管时，其 ``__path__`` 必须真的指向源码里的
       ``ripple/`` 目录 ⇒ 具体验证 ``monitoring`` / ``buffer`` 这些**别的测试文件
       会用的**子模块在磁盘上确实可达。
    """
    suspicious: dict[str, object] = {}
    for name, module in list(sys.modules.items()):
        if not (name == "graspo" or name.startswith("graspo.")):
            continue
        module_path = getattr(module, "__path__", None)
        if module_path is not None and len(module_path) == 0:
            suspicious[name] = module_path
    assert not suspicious, (
        "sys.modules contains graspo packages with an EMPTY __path__ "
        f"(this breaks submodule imports in later test files): {sorted(suspicious)}"
    )

    ripple = sys.modules.get("graspo.ripple")
    if ripple is None:
        pytest.skip("graspo.ripple is not loaded in this process; nothing to verify")
    ripple_paths = [Path(entry) for entry in getattr(ripple, "__path__", [])]
    assert ripple_paths, "graspo.ripple has no __path__ at all"

    source_root = Path(__file__).resolve().parents[2] / "src" / "graspo" / "ripple"
    assert any(path.resolve() == source_root for path in ripple_paths), (
        "graspo.ripple does not point at the real source tree "
        f"(got {ripple_paths}, expected {source_root}) — later test files that import "
        "graspo.ripple.* submodules would fail with a misleading error"
    )

    # 具体点名别的测试文件真正需要的子模块：它们必须在磁盘上**可解析**
    # （可以是包目录，也可以是 .py 模块），且不得被 __path__ = [] 挡死
    # ——原缺陷就是这样挡死这些子模块的。
    import importlib.util as _importlib_util

    for submodule in ("monitoring", "buffer", "parsing"):
        dotted = f"graspo.ripple.{submodule}"
        assert _importlib_util.find_spec(dotted) is not None, (
            f"{dotted} is not resolvable from sys.modules['graspo.ripple'].__path__; "
            "later test files would fail with a misleading ModuleNotFoundError"
        )


def _minimal_training_config(tmp_path: Path, extra: str = "") -> Path:
    path = tmp_path / "config.yaml"
    path.write_text(
        f"""backend: native
model:
  model_path: models/test
data:
  train_path: samples/data/sample.jsonl
training:
  output_dir: outputs/test
{extra}""",
        encoding="utf-8",
    )
    return path


def _eval_section(**overrides: str) -> str:
    """造一个合法的 `eval:` 段，用 overrides 覆盖单个字段。"""
    fields = {
        "dataset_path": "/data/test.jsonl",
        "base_model_path": "/models/base",
        # 产物路径必须绝对（fail-closed，修法见 task-eval-defects 工位 report）。
        "output_dir": "/abs/repo/.local/eval/runs/x",
        "gpus": '"0,1"',
        # role=base 时不需要 checkpoint_path；这是"不涉及训练产物"的最简合法段。
        "role": "base",
    }
    fields.update(overrides)
    body = "\n".join(f"  {key}: {value}" for key, value in fields.items())
    return f"eval:\n{body}\n"


@pytest.mark.parametrize("sample", _EXISTING_CONFIGS_WITHOUT_EVAL)
def test_existing_config_without_eval_section_still_loads(sample):
    """向后兼容是硬要求：既有配置原样可用，且 eval 段是明确的 None。"""
    config = GraspoConfig.from_yaml(_CONFIGS_DIR / sample)
    assert config.eval is None


def test_all_sample_configs_load():
    """全量扫描：仓库里任何样例配置都不得因为 eval 段而加载失败。"""
    files = sorted(
        glob.glob(str(_CONFIGS_DIR / "*.yaml")) + glob.glob(str(_CONFIGS_DIR / "matrix" / "*.yaml"))
    )
    assert files, "no sample configs found — the sweep would be vacuous"
    failures: list[str] = []
    for path in files:
        try:
            GraspoConfig.from_yaml(path)
        except BaseException as exc:  # noqa: BLE001 - 汇总所有失败，一次性报告
            failures.append(f"{path}: {exc}")
    assert not failures, "sample configs failed to load:\n" + "\n".join(failures)


def test_omitted_eval_section_is_none(tmp_path):
    config = GraspoConfig.from_yaml(_minimal_training_config(tmp_path))
    assert config.eval is None


def test_explicit_eval_null_is_treated_as_absent(tmp_path):
    """``eval: null`` 与"键不存在"等价——都是"未提供"（§2.2 唯一空值语义）。"""
    path = _minimal_training_config(tmp_path, "eval: null\n")
    assert GraspoConfig.from_yaml(path).eval is None


def test_explicit_eval_section_is_validated(tmp_path):
    """显式提供 eval 段时，校验照常执行——不能把校验整体关掉。"""
    path = _minimal_training_config(tmp_path, _eval_section())
    config = GraspoConfig.from_yaml(path)
    assert config.eval is not None
    assert config.eval.gpus == "0,1"
    assert config.eval.role == "base"
    assert config.eval.output_dir == "/abs/repo/.local/eval/runs/x"


def test_explicit_eval_section_missing_fields_is_rejected(tmp_path):
    """反方向：写了 eval 段但字段缺失必须失败（否则是"校验被静默关掉"）。"""
    path = _minimal_training_config(tmp_path, "eval:\n  dataset_path: /data/test.jsonl\n")
    with pytest.raises(SystemExit):
        GraspoConfig.from_yaml(path)


@pytest.mark.parametrize(
    ("field", "value", "reason"),
    [
        ("gpus", '"6"', "production card"),
        ("gpus", '"7"', "production card"),
        ("gpus", '"0,1,2,3,4"', "too many cards"),
        ("gpus", '""', "no explicit lock"),
        ("output_dir", "outputs/not-local", "output dir outside .local/"),
    ],
)
def test_validation_still_catches_dangerous_eval_sections(tmp_path, field, value, reason):
    """卡计划与产物目录的防呆在"显式提供"路径上必须照常生效。"""
    path = _minimal_training_config(tmp_path, _eval_section(**{field: value}))
    with pytest.raises(SystemExit):
        GraspoConfig.from_yaml(path)


def test_eval_train_limit_is_optional_and_recorded(tmp_path):
    """``train_limit`` 是可选切片长度；缺省 None（= 全量），显式值必须保留。"""
    default = EvalConfig.from_yaml(_minimal_training_config(tmp_path, _eval_section()))
    assert default.train_limit is None

    sliced = EvalConfig.from_yaml(
        _minimal_training_config(tmp_path, _eval_section(train_limit="100"))
    )
    assert sliced.train_limit == 100


def test_eval_config_standalone_from_yaml_unwraps_eval_section(tmp_path):
    """独立评测配置既支持平铺字段，也支持带 `eval:` 段（与训练配置共用文件）。"""
    flat = tmp_path / "flat.yaml"
    flat.write_text(
        """dataset_path: /data/test.jsonl
base_model_path: /models/base
output_dir: /abs/repo/.local/eval/runs/x
gpus: "0"
role: base
""",
        encoding="utf-8",
    )
    assert EvalConfig.from_yaml(flat).gpus == "0"

    example = EvalConfig.from_yaml(_CONFIGS_DIR / "eval_example.yaml")
    assert example.gpus == "0,1"
    assert example.role == "after"  # 仓库样例走的是训练后评测路径


def test_error_report_helper_survives_optional_section(tmp_path):
    """错误报告辅助函数不能因为段变成 Optional 而自己崩掉（曾在本修复中出现）。"""
    path = _minimal_training_config(tmp_path, "eval:\n  bogus_field: 1\n")
    with pytest.raises(SystemExit):
        GraspoConfig.from_yaml(path)
