"""``tests/export/`` 的 conftest：把本目录加入 ``sys.path``（子目录用例可 ``import helpers``）。"""

from __future__ import annotations

import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import pytest  # noqa: E402


@pytest.fixture
def export_config():
    """按需构造 ``ExportConfig`` 的小工厂：``export_config(checkpoint, output, fmt=...)``。"""
    from helpers import make_export_config

    return make_export_config
