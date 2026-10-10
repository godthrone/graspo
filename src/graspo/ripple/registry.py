"""ripple 侧的开发模式回退注册表（★ X4：core 不硬编码 ripple 路径）。

- 生产真相源 = `pyproject.toml` 的 entry_points；
- 本表只是"包未 pip install"时的开发回退。
- **入口**：`graspo.ripple.__init__` 导入本模块即完成注册（幂等）。
"""

from __future__ import annotations

from graspo.core.discovery import register_dev_fallbacks

register_dev_fallbacks(
    "graspo.rewards",
    {
        "graspo": "graspo.ripple.reward.reward:GraspoReward",
    },
)

__all__: list[str] = []
