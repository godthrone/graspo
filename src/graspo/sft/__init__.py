__all__ = ["ARDSFTTrainer"]


def __getattr__(name: str):
    if name == "ARDSFTTrainer":
        from graspo.sft.trainer import ARDSFTTrainer

        return ARDSFTTrainer
    raise AttributeError(name)
