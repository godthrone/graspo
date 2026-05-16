from graspo.cli.app import build_parser


def test_cli_validate_reward():
    parser = build_parser()
    args = parser.parse_args(["validate-reward", "--data", "data/sample.jsonl", "--limit", "1"])

    assert args.func(args) == 0


def test_cli_new_commands_parse():
    parser = build_parser()

    commands = [
        ["anchor-generate", "--knowledge-ontology", "k.json", "--language-ontology", "l.json", "--output", "o.jsonl"],
        ["anchor-answer", "--model-path", "model", "--input", "i.jsonl", "--output", "o.jsonl"],
        ["anchor-filter", "--input", "i.jsonl", "--output", "o.jsonl"],
        ["anchor-split", "--input", "i.jsonl", "--train-output", "train.jsonl", "--eval-output", "eval.jsonl"],
        ["train-sft-ard", "--config", "configs/ard_sft_lora.yaml"],
        ["eval-forgetting", "--anchor-eval", "anchor.jsonl", "--completions", "completions.jsonl"],
    ]
    for command in commands:
        args = parser.parse_args(command)
        assert callable(args.func)
