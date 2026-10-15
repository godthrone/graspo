"""CLI 契约（T18 / §10.1 推论 2）：``graspo export`` **只接受** ``--config``。"""

from __future__ import annotations

import argparse

from graspo.cli.app import build_parser


def _export_parser() -> argparse.ArgumentParser:
    parser = build_parser()
    subparsers = next(
        action for action in parser._actions if isinstance(action, argparse._SubParsersAction)
    )
    return subparsers.choices["export"]


def _option_dests(parser: argparse.ArgumentParser) -> list[str]:
    return sorted(action.dest for action in parser._actions if action.dest != "help")


def test_export_cli_exposes_only_the_config_parameter():
    parser = _export_parser()
    assert _option_dests(parser) == ["config"]
    option_strings = {
        option for action in parser._actions for option in getattr(action, "option_strings", [])
    }
    assert "--config" in option_strings


def test_positive_control_an_extra_output_option_is_detected():
    """阳性对照：加一个 ``--allow-overwrite``（或任何输出定位参数）必须让上面的判据失败。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config")
    parser.add_argument("--allow-overwrite", action="store_true")
    assert _option_dests(parser) != ["config"]
