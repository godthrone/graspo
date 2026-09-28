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


# ── 回归：门禁不得被**合法 shell 引号**打破（2026-09-28）────────────────────


def test_dockerfile_pins_tolerate_shell_quoting(tmp_path):
    """★负例回归：带引号的版本 token 必须解析出**干净的版本号**。

    真因（2026-09-28）：`docker/Dockerfile` 把 pip 安装改成 `sh -eu -c '…'` 后，闭合单引号
    紧贴最后一个版本号（`torchvision==0.26.0'`）；解析器按空白切 token 后把引号并进版本
    ⇒ 门禁误报"声明不一致"。后果不止单测：Dockerfile 内 `RUN … check_env_versions.py --local`
    会让**镜像构建直接失败**。本测试同时覆盖实验镜像里既有的整体引号形态
    （`docker/Dockerfile.msswift` 大量使用 `"vllm==0.23.0"`）。
    """
    dockerfile = tmp_path / "Dockerfile"
    dockerfile.write_text(
        "RUN sh -eu -c 'pip install --index-url https://example.invalid/simple/ \\\n"
        "    torch==2.11.0 torchvision==0.26.0'\n"                  # 尾随闭合引号（本轮真实形态）
        "RUN sh -c 'pip install \"ms-swift==4.5.3\" pillow==11.3.0'\n"   # 整体双引号
        "RUN sh -eu -c 'pip install numpy==1.26.4;'\n"                   # 引号前带命令分隔符
        "RUN sh -eu -c \"pip install 'pyyaml==6.0.3' safetensors==0.8.0\"\n",  # 整体单引号
        encoding="utf-8",
    )

    pins = check_env_versions.parse_dockerfile_pins(dockerfile)

    assert pins == {
        "torch": "2.11.0",
        "torchvision": "0.26.0",
        "ms-swift": "4.5.3",
        "pillow": "11.3.0",
        "numpy": "1.26.4",
        "pyyaml": "6.0.3",
        "safetensors": "0.8.0",
    }, pins


def test_quoted_pins_are_read_from_the_real_msswift_dockerfile():
    """正例：实验镜像里**带引号**的 pin 现在是**被读到**的（而不是被解析器静默漏掉）。"""
    msswift_pins = check_env_versions.parse_dockerfile_pins(
        check_env_versions.PROJECT_ROOT / "docker" / "Dockerfile.msswift"
    )

    for name, version in (("vllm", "0.23.0"), ("ray", "2.55.1"), ("deepspeed", "0.18.9")):
        assert msswift_pins.get(name) == version, (name, msswift_pins.get(name))


def test_experiment_image_extra_pins_are_allowed_by_named_rule(monkeypatch, tmp_path):
    """★判别力：门禁**没有被放松**——产品镜像的未声明 pin 仍必须报错。

    "无对应声明"只对产品镜像成立；实验镜像的额外 pin 面按 `EXTRA_PINS_ALLOWED` 的
    **具名理由**放行（此前是靠解析器读不懂引号**隐式**放行，见该常量注释）。
    """
    reason = check_env_versions.EXTRA_PINS_ALLOWED.get("Dockerfile.msswift")
    assert reason and reason.strip(), "实验镜像的额外 pin 面必须写明理由（§2.1 不许隐式排除）"
    # 理由必须**具体**（不能只写"实验镜像"四个字）：要说清"这些包本就不在 pyproject 声明范围"
    assert "不在 pyproject 声明范围" in reason, reason
    assert "vllm" in reason and "deepspeed" in reason, reason
    assert "Dockerfile" not in check_env_versions.EXTRA_PINS_ALLOWED, (
        "产品镜像不得进入放行名单：它的每个 pin 都必须能在 pyproject 找到声明"
    )

    declared = check_env_versions.declared_pins()
    rogue = tmp_path / "Dockerfile"
    rogue.write_text("RUN pip install --index-url https://example.invalid/simple/ typo-pkg==1.0\n")
    monkeypatch.setattr(check_env_versions, "DOCKERFILES", (rogue,))

    problems = check_env_versions.compare_declaration_sites(declared)

    assert problems == [
        "Dockerfile: pip pin typo-pkg==1.0 在 pyproject 中无对应声明"
    ], problems


def test_removing_the_experiment_allowance_exposes_extra_pins(monkeypatch):
    """★**有判据的豁免**：拿掉 `EXTRA_PINS_ALLOWED` ⇒ 门禁必须抓到实验镜像的额外 pin。

    这条把"允许"从**豁免**（可能只是"没人发现"）变成**有判据的豁免**：
    若哪天实验镜像不再有额外 pin 面（或有人把产品包名写错进实验镜像），该判据仍然成立；
    而**反向**——只要允许被拿掉，那批 pin 必须**立刻**现形（105 条，2026-09-28 实测）。
    """
    declared = check_env_versions.declared_pins()
    monkeypatch.setattr(check_env_versions, "EXTRA_PINS_ALLOWED", {})

    problems = check_env_versions.compare_declaration_sites(declared)

    assert problems, "拿掉允许后必须报出额外 pin，否则这个豁免没有判据（= 没人发现）"
    assert all(p.startswith("Dockerfile.msswift:") for p in problems), problems
    assert len(problems) >= 100, len(problems)
    joined = "\n".join(problems)
    for name in ("vllm", "ray", "deepspeed"):        # 具名抽查实验专属包
        assert f"pip pin {name}==" in joined, name


# ── F-5：门禁对**真镜像**必须通过（三项声明对齐实测值）────────────────────────

#: `graspo-msswift:4.5.3` 的实测血统（来源：task-r2-onmachine §0.4，
#: `python3 scripts/check_env_versions.py --from-container graspo-msswift:4.5.3`）。
#: **这是实测快照，不是"希望值"**——若谁改了声明或换了镜像，这里就会失败，
#: 逼他跑一次真门禁并更新快照。
_MEASURED_MSSWIFT_45_3: dict[str, str] = {
    "ms-swift": "4.5.3",
    "numpy": "1.26.4",
    "pillow": "11.3.0",
    "pydantic": "2.11.10",
    "pyyaml": "6.0.3",
    "safetensors": "0.8.0",
    "torch": "2.11.0+cu130",
    "torchvision": "0.26.0+cu130",
    "transformers": "5.12.1",
}


def test_gate_passes_against_measured_msswift_image_lineage():
    """★F-5 回归：把真镜像实测值喂给门禁，**必须 rc=0**。

    修前三项不一致（numpy 声明 2.4.5 / pillow 12.2.0 / safetensors 0.7.0）⇒ rc=1。
    """
    completed = _run(_MEASURED_MSSWIFT_45_3)

    assert completed.returncode == 0, completed.stderr
    assert "VERSION GATE OK" in completed.stdout


def test_gate_still_fails_against_pre_fix_lineage():
    """★不放松：门禁没有被"改成永远通过"——把三项调回**修前**声明仍必须 rc=1。"""
    pre_fix = dict(_MEASURED_MSSWIFT_45_3)
    pre_fix["numpy"] = "2.4.5"
    pre_fix["pillow"] = "12.2.0"
    pre_fix["safetensors"] = "0.7.0"

    completed = _run(pre_fix)

    assert completed.returncode == 1
    for name in ("numpy", "pillow", "safetensors"):  # type: ignore[assignment]
        assert name in completed.stderr


def test_f5_three_declarations_equal_measured_image_values():
    """声明值 == 实测值（逐项钉住，防止只改口径不改声明）。"""
    declared = _gated()

    for name in ("numpy", "pillow", "safetensors"):
        assert declared[name] == check_env_versions.release_segment(_MEASURED_MSSWIFT_45_3[name]), (
            name
        )


def test_measured_snapshot_covers_every_gated_package():
    """实测快照必须覆盖门禁集合——否则门禁会报"镜像内缺失"而掩盖真问题。"""
    gated = _gated()

    assert set(gated) == set(_MEASURED_MSSWIFT_45_3)


def test_release_segment_normalizes_local_labels():
    assert check_env_versions.release_segment("2.11.0+cu130") == "2.11.0"
    assert check_env_versions.release_segment("5.12.1") == "5.12.1"
