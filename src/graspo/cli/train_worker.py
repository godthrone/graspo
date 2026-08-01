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
        help="Smoke mode: run 1 training step then stop (set by `graspo launch --smoke`).",
    )
    args = parser.parse_args()

    config = GraspoConfig.from_yaml(args.config)
    if args.smoke:
        # 冒烟：跑 1 步即停。仅改内存中的 config，不写回文件。
        config.training.max_steps = 1

    if config.train_method == "sft":
        from graspo.backends.graspoflow.runtime import GraspoFlowRuntime
        from graspo.backends.graspoflow.trainer.sft_trainer import SFTTrainer

        runtime = GraspoFlowRuntime.from_config(config)
        SFTTrainer(config, runtime).train()
    else:
        from graspo.backends import create_trainer, select_backend

        selection = select_backend(config)
        create_trainer(config, selection).train()


if __name__ == "__main__":
    main()
