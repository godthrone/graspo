"""训练 worker 进程入口：按 train_method 分派 SFT/RL 训练器（支持 --smoke 冒烟）。"""

import argparse

from graspo.core.schema import GraspoConfig


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="python -m graspo.cli.train_worker",
        description="Internal GRASPO training worker. Use `graspo launch --config ...`.",
    )
    parser.add_argument("--config", "-c", required=True)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help=(
            "Smoke boundary (infrastructure param): run until the first optimize"
            " step then stop. Never mutates the config object."
        ),
    )
    args = parser.parse_args()

    config = GraspoConfig.from_yaml(args.config)

    if config.train_method == "sft":
        from graspo.flow.backend_selection import select_backend
        from graspo.flow.trainer.sft_trainer import SFTTrainer
        from graspo.flow.runtime import GraspoFlowRuntime

        selection = select_backend(config)
        if selection.name == "native":
            runtime = GraspoFlowRuntime.from_config(config)
            SFTTrainer(config, runtime).train(smoke=args.smoke)
        else:
            # msswift 后端 SFT：TODO 阶段2实现
            raise NotImplementedError(
                f"SFT training is not yet supported for backend '{selection.name}'. "
                f"Use backend='native' for SFT."
            )
    else:
        from graspo.flow.backend_selection import create_trainer, select_backend

        selection = select_backend(config)
        create_trainer(config, selection).train(smoke=args.smoke)


if __name__ == "__main__":
    main()
