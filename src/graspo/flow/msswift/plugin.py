"""ms-swift 扩展点注册入口（``--external_plugins`` / 进程内显式调用共用）。

把 graspo 的 GRPO 训练器挂到 ms-swift 的 ``TrainerFactory.TRAINER_MAPPING`` 上，
使 ``rlhf_type='grpo'`` 解析到 ``GraspoMsSwiftGRPOTrainer``（注入 ``ripple`` 算法核）。

**与旧写法的区别（T1 + 实测修正）**：① 旧版把条目注册在 ``"graspo_grpo"`` 名下，但
ms-swift 4.3.2/4.5.3 的 ``RLHFArguments.rlhf_type`` 是 ``Literal[...]``，**不接受**
该名字（实测：``swift_version=4.5.3``，字段定义
``rlhf_type: Literal['dpo','orpo','simpo','kto','cpo','rm','ppo','grpo','gkd']``）。
现在注册在 ``"grpo"`` 键上，由
``graspo.flow.msswift.trainer.registered_trainer_class`` 在作用域内安装/还原。
② 旧版的路径写成 ``module:Class``（冒号），但 ms-swift 的 ``TrainerFactory.get_cls``
用 ``rsplit('.', 1)`` 切分——冒号写法会把类名连冒号一起当属性名，抛
``AttributeError: module 'graspo.flow.msswift' has no attribute 'trainer:...'``（实测踩过）。
正确写法是**点分**路径。

**不做静默 catch**：ms-swift 未安装时 ``register()`` 抛 ``RuntimeError`` 并给出安装
指引（宪法 §13.1：不吞异常）。本模块只被 msswift 后端路径导入，因此"没装 ms-swift"
本身就是错误状态，不该被静默忽略。
"""

from __future__ import annotations


def register() -> None:
    """把 graspo 的 GRPO 训练器注册进 ms-swift 的 ``TrainerFactory``（幂等）。

    Raises:
        RuntimeError: ms-swift 未安装。
    """
    try:
        from swift.trainers import TrainerFactory
    except ImportError as exc:
        raise RuntimeError(
            "ms-swift is required to register the graspo GRPO trainer "
            "(pip install graspo[msswift]); native backend is unaffected."
        ) from exc

    # 点分路径（ms-swift 的 get_cls 用 rsplit('.', 1) 切分模块与类名）
    TrainerFactory.TRAINER_MAPPING["grpo"] = "graspo.flow.msswift.trainer.GraspoMsSwiftGRPOTrainer"


if __name__ == "__main__":  # pragma: no cover - 供 --external_plugins 直接加载
    register()
