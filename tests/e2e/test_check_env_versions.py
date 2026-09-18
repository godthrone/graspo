"""版本门禁的负向测试：**故意错一个版本号必须报错**。

门禁若不能负向失败，它就不是门禁。这里用 `--observed-json` 离线模式验证
比较逻辑，不需要起容器（本轮红线禁止 docker run）。
另外验证**包集合自动派生**（不再手工列举，避免漏包）与**显式排除清单**。
"""

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "check_env_versions.py"

_spec = importlib.util.spec_from_file_location("check_env_versions", _SCRIPT)
assert _spec is not None and _spec.loader is not None
check_env_versions = importlib.util.module_from_spec(_spec)
sys.modules["check_env_versions"] = check_env_versions
_spec.loader.exec_module(check_env_versions)


def _gated() -> dict[str, str]:
    return check_env_versions.gated_packages()


def _run(observed: dict[str, str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(_SCRIPT), "--observed-json", json.dumps(observed)],
        capture_output=True,
        text=True,
        check=False,
    )


# ── 包集合自动派生（裁定 2）─────────────────────────────────────────────────


def test_gated_packages_are_auto_derived_and_include_pydantic():
    """包集合自动派生自 pyproject 全部 `==` pin——必须包含 pydantic（旧版漏过它）。"""
    gated = _gated()

    assert "pydantic" in gated
    assert "torch" in gated
    assert "transformers" in gated
    assert "ms-swift" in gated
    # 与 pyproject 实际 pin 对齐，而不是硬编码一份列表
    declared = check_env_versions.declared_pins()
    for name in gated:
        assert gated[name] == declared[name], name
    assert len(gated) >= 9


def test_dev_only_packages_are_excluded_explicitly():
    """dev 工具走**显式排除清单**（带理由），不是靠"没写进列表"。"""
    gated = _gated()

    for dev_only in ("pytest", "ruff", "mypy", "openpyxl"):
        assert dev_only not in gated, dev_only
        assert dev_only in check_env_versions.EXCLUDED_FROM_GATE, dev_only
        assert check_env_versions.EXCLUDED_FROM_GATE[dev_only].strip(), dev_only


# ── 正/负向 ─────────────────────────────────────────────────────────────────


def test_gate_passes_when_image_matches_declaration():
    declared = _gated()
    observed = {name: declared[name] for name in declared}
    # torch 镜像里带本地标签 +cu130，归一化后必须仍然通过。
    observed["torch"] = declared["torch"] + "+cu130"

    completed = _run(observed)

    assert completed.returncode == 0, completed.stderr
    assert "VERSION GATE OK" in completed.stdout


def test_gate_fails_when_pydantic_version_is_wrong():
    """裁定 2 的直接回归：pydantic 现在在门禁内。"""
    declared = _gated()
    observed = {name: declared[name] for name in declared}
    observed["pydantic"] = "9.9.9"

    completed = _run(observed)

    assert completed.returncode == 1
    assert "pydantic" in completed.stderr


def test_gate_fails_when_torch_version_is_wrong():
    declared = _gated()
    observed = {name: declared[name] for name in declared}
    observed["torch"] = "2.99.9"

    completed = _run(observed)

    assert completed.returncode == 1
    assert "VERSION GATE FAILED" in completed.stderr
    assert "torch" in completed.stderr
    assert "2.99.9" in completed.stderr


def test_gate_fails_when_transformers_or_ms_swift_is_wrong():
    declared = _gated()
    for package, wrong in (("transformers", "0.0.1"), ("ms-swift", "4.4.0")):
        observed = {name: declared[name] for name in declared}
        observed[package] = wrong

        completed = _run(observed)

        assert completed.returncode == 1, package
        assert package in completed.stderr


def test_gate_fails_when_package_missing_from_image():
    declared = _gated()
    observed = {name: declared[name] for name in declared}
    del observed["transformers"]

    completed = _run(observed)

    assert completed.returncode == 1
    assert "镜像内缺失" in completed.stderr


# ── 两个声明点不得分叉 ──────────────────────────────────────────────────────


def test_dockerfile_pins_match_pyproject_declarations():
    declared = check_env_versions.declared_pins()

    problems = check_env_versions.compare_declaration_sites(declared)

    assert problems == [], problems


def test_dockerfile_pin_parser_finds_all_pinned_packages():
    """两个 Dockerfile 的 pip pin 都要被自动解析到（用于跨声明点比对）。"""
    msswift_pins = check_env_versions.parse_dockerfile_pins(
        check_env_versions.PROJECT_ROOT / "docker" / "Dockerfile.msswift"
    )
    product_pins = check_env_versions.parse_dockerfile_pins(
        check_env_versions.PROJECT_ROOT / "docker" / "Dockerfile"
    )

    assert msswift_pins.get("torch") == "2.11.0"
    assert msswift_pins.get("transformers") == "5.12.1"
    assert msswift_pins.get("ms-swift") == "4.5.3"
    assert product_pins.get("torch") == "2.11.0"
    assert product_pins.get("pydantic") == "2.11.10"


def test_release_segment_normalizes_local_labels():
    assert check_env_versions.release_segment("2.11.0+cu130") == "2.11.0"
    assert check_env_versions.release_segment("5.12.1") == "5.12.1"
