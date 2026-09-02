"""msswift 后端：将 Graspo ripple 算法注入 ms-swift GRPOTrainer。

通过继承 ``swift.rlhf_trainers.GRPOTrainer``，重写评分、优势计算、
损失计算和后处理四个方法，在不修改 ms-swift 源码的前提下注入
Graspo 的字符级标注、token 级 advantage 和 GRASPORippleLoss。

ms-swift 是可选依赖：``pip install graspo[msswift]``。
"""

from graspo.flow.msswift.trainer import MsSwiftTrainer, create_msswift_trainer

__all__ = ["MsSwiftTrainer", "create_msswift_trainer"]