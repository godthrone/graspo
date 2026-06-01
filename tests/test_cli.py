from graspo.cli.app import build_parser


def test_cli_validate_reward():
    parser = build_parser()
    args = parser.parse_args(["validate-reward", "--data", "data/sample.jsonl", "--limit", "1"])

    assert args.func(args) == 0


def test_cli_main_commands_parse():
    parser = build_parser()

    commands = [
        ["prepare-data", "--input", "data/sample.jsonl", "--output", "out.jsonl"],
        ["analyze", "--rewards", "rewards.jsonl"],
        ["train", "--config", "configs/graspo.yaml"],
        ["train", "--config", "configs/graspo.yaml", "--backend", "megatron-native"],
        ["train", "--config", "configs/graspo.yaml", "--backend", "hf-reference"],
    ]
    for command in commands:
        args = parser.parse_args(command)
        assert callable(args.func)
