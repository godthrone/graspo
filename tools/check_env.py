#!/usr/bin/env python3
"""★ 依赖校验：矩阵/护栏工具链的纯 Python 依赖是否齐备（缺则明确报错，⛔ 不静默）。

口径见 ``docs/toolchain-pin.md``：
- **运行时 = 镜像**（训练/评测/导出一律在镜像内执行）；
- **宿主只跑纯 Python 工具**（``scripts/``、``tools/matrix/``）；
- 宿主**无 torch/pytest**（2026-10-14 实测）⇒ **需 torch/pytest 的校验（如
  ``tests/scripts/test_merged_export_classify.py`` 的转换类用例）一律在镜像内执行**；
  本地跑该文件时 ``importorskip`` 跳过（保持本地 gate 绿）。

用法：``python3 tools/check_env.py`` ⇒ 缺失 ⇒ 打印清单 ＋ **非零退出**。
"""
from __future__ import annotations

import importlib.util
import sys

#: ★ 矩阵/护栏工具链的纯 Python 依赖（宿主应齐备）。
REQUIRED: tuple[tuple[str, str], ...] = (
    ("pydantic", "run_guards / 配置解析"),
    ("msgspec", "配置 schema"),
    ("yaml", "yaml 配置读写"),
    ("pytest", "工具链自测（tests/scripts/）"),
)

#: ★ 只在镜像内需要的（宿主缺失属预期，⛔ 不计失败）。
IMAGE_ONLY: tuple[tuple[str, str], ...] = (
    ("torch", "native rank 转换（tests/scripts/test_merged_export_classify.py）"),
    ("safetensors", "adapter 读写"),
    ("vllm", "评测服务"),
)


def check(required=REQUIRED, image_only=IMAGE_ONLY) -> tuple[list[str], list[str]]:
    missing = [f"{n}（{why}）" for n, why in required if importlib.util.find_spec(n) is None]
    absent_img = [f"{n}（{why}）" for n, why in image_only if importlib.util.find_spec(n) is None]
    return missing, absent_img


def main() -> int:
    missing, absent_img = check()
    if absent_img:
        print("★ 以下依赖宿主缺失属预期（需 torch/pytest 的校验一律在镜像内执行）：")
        for item in absent_img:
            print(f"    - {item}")
    if missing:
        print("★ 依赖缺失（工具链无法运行）：")
        for item in missing:
            print(f"    - {item}")
        print("⇒ 请按 docs/toolchain-pin.md 安装；⛔ 不以『某次恰好装到的版本』为基线。")
        return 1
    print("★ 依赖校验通过：矩阵/护栏工具链的纯 Python 依赖齐备。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
