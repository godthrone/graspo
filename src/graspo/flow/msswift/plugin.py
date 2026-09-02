"""ms-swift external_plugins 注册入口。

通过 ms-swift 的 TrainerFactory 机制注册 Graspo 自定义训练器，
使 ms-swift CLI 可通过 ``--external_plugins`` 加载本模块。
"""

try:
    from swift.trainers.trainer_factory import TrainerFactory

    TrainerFactory.TRAINER_MAPPING["graspo_grpo"] = (
        "graspo.flow.msswift.trainer:MsSwiftTrainer"
    )
except ImportError:
    pass  # ms-swift 未安装，静默跳过