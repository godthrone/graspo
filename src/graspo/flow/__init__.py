"""GraspoFlow — 设施层：GPU / 分布式 / 网络 / 文件 IO 的执行载体。

分层职责：
- ripple/  算法层：什么值给多少分、怎么判格式、怎么判健康
- core/    通用契约：配置模型、chat template
- flow/    设施层：怎么跑起来、怎么上 GPU（本层）

子模块按功能域组织（★ P2 后与模块结构完全对应）：
- ``adapter/``  算法 ↔ 后端**对接契约**
- ``backend/``  **可切换实现**：``native/``（runtime / trainer / parallel / logger / lora / models）
                与 ``msswift/``
- ``guard/``    卡守卫**实现**（跨 backend + eval 共用）
- 根下 ``checkpoint*`` / ``data_io`` / ``logging`` / ``memory`` / ``progress_metrics``
  = 与 backend 无关的**共享设施**；``registry.py`` = entry_points 开发回退表
"""

from graspo.flow.backend.native.runtime import GraspoFlowRuntime
from graspo.flow.backend.native.trainer import GraspoFlowTrainer

# ★ X4：flow 的 entry_points 开发回退表**由本层自己登记**（core 不硬编码
# flow 路径）。导入即注册（幂等）。
from graspo.flow import registry as _registry  # noqa: E402,F401

__all__ = ["GraspoFlowTrainer", "GraspoFlowRuntime"]
