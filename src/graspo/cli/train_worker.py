import argparse

from graspo.backends import create_trainer, select_backend
from graspo.core.schema import GraspoConfig


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="python -m graspo.cli.train_worker",
        description="Internal GRASPO training worker. Use `graspo launch --config ...`.",
    )
    parser.add_argument("--config", "-c", required=True)
    args = parser.parse_args()

    config = GraspoConfig.from_yaml(args.config)
    selection = select_backend(config)
    create_trainer(config, selection).train()


if __name__ == "__main__":
    main()
