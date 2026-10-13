"""★ 产物级断言用例：对象 = **实际运行的 config**（2026-10-13 教训）。"""
import importlib.util
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    "check_run_config",
    Path(__file__).resolve().parents[2] / "tools" / "matrix" / "check_run_config.py",
)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
check = _mod.check

DEPLOY = "export:\n  final_formats: [\"merged-hf\"]\nlaunch:\n  cards: 2\n"


def test_run_config_missing_section_is_rejected():
    """★ 本次事故形态：部署树有、**运行 config 空** ⇒ 必须拒。"""
    run = "export:\n  final_formats: []\nlaunch:\n  cards: 2\n"
    assert check(run, DEPLOY) != []


def test_run_config_without_export_section_is_rejected():
    assert check("launch:\n  cards: 2\n", DEPLOY) != []


def test_run_config_matching_is_accepted():
    assert check(DEPLOY, DEPLOY) == []


def test_run_config_extra_format_is_rejected():
    run = "export:\n  final_formats: [\"merged-hf\", \"peft-adapter\"]\n"
    assert check(run, DEPLOY) != []
