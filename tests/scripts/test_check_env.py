"""★ 依赖校验脚本用例：正例（依赖齐）／负例（缺一项 ⇒ 报错、非零退出）。"""
import importlib.util
import sys
from pathlib import Path

sys.path.insert(0, str(Path(".").resolve()))
from tools.check_env import check  # noqa: E402


def test_check_env_reports_missing_without_silence():
    """★ 负例：模拟缺一项 ⇒ 必须出现在 missing 清单（⛔ 不静默）。"""
    missing, _ = check(required=(("definitely_absent_module_xyz", "测试用"),), image_only=())
    assert missing and "definitely_absent_module_xyz" in missing[0]


def test_check_env_image_only_separated():
    """★ 镜像专属依赖缺失应单独归类（⛔ 不计失败）。"""
    missing, img = check(required=(("json", "标准库"),), image_only=(("definitely_absent_imageonly_xyz", "镜像内"),))
    assert missing == []
    assert any("definitely_absent_imageonly_xyz" in x for x in img)


def test_check_env_main_returns_nonzero_on_missing(monkeypatch, capsys):
    import tools.check_env as m
    monkeypatch.setattr(m, "REQUIRED", (("definitely_absent_module_xyz", "测试用"),))
    monkeypatch.setattr(m, "IMAGE_ONLY", ())
    assert m.main() == 1
    assert "依赖缺失" in capsys.readouterr().out
