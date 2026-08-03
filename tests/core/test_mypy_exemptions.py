"""mypy 豁免列表约束测试——防豁免扩散（防呆设计）。

pyproject.toml 的 [[tool.mypy.overrides]] ignore_errors 只允许覆盖
"类改目录"的 mixin 模块（mypy 对 mixin 组合的结构性类型盲区）。
本测试保证：
1. 豁免模块的文件必须存在（防失效豁免）
2. 豁免文件必须定义 mixin 风格类（类名以 Mixin 结尾或以 _ 开头）
3. 非 mixin 的组装点（adapter/model 等）不得豁免
"""

import ast
import tomllib
from pathlib import Path

# 组装点文件：mixin 组合发生在这些文件，豁免它们是掩盖问题而非解释盲区
_FORBIDDEN_EXEMPT_NAMES = {"adapter", "model", "transformer", "base", "runtime"}


def _mypy_ignored_modules() -> list[str]:
    with Path("pyproject.toml").open("rb") as f:
        pyproject = tomllib.load(f)
    overrides = pyproject["tool"]["mypy"].get("overrides", [])
    return [
        module
        for override in overrides
        if override.get("ignore_errors")
        for module in override.get("module", [])
    ]


def _file_defines_mixin_style_class(path: Path) -> bool:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            if node.name.endswith("Mixin") or node.name.startswith("_"):
                return True
    return False


def test_mypy_exemptions_exist_and_are_mixin_files():
    modules = _mypy_ignored_modules()
    assert modules, "mypy overrides with ignore_errors must exist"

    for module in modules:
        path = Path("src", *module.split(".")).with_suffix(".py")
        assert path.exists(), f"exempt module {module!r} does not exist at {path}"
        short = module.rsplit(".", 1)[-1]
        assert short not in _FORBIDDEN_EXEMPT_NAMES, (
            f"exempt module {module!r} is an assembly point ({short}.py), "
            "not a mixin — exemptions must not spread to non-mixin files"
        )
        assert _file_defines_mixin_style_class(path), (
            f"exempt module {module!r} does not define a mixin-style class "
            "(name ends with 'Mixin' or starts with '_')"
        )
