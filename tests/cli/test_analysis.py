"""analyze-profile 归因分析测试。

核心约束（用户裁定）：分析与训练数据内容无关——不假设字段名、
不假设数值字段、不做数值误差统计。测试覆盖：
1. not_correct 三类原因分类（tool_mismatch / content_all_wrong / format_shortfall）
2. 与数据内容无关：换一组完全不同的工具/字段名（非 rotate_arm），结果语义不变
3. 非 tool_call 数据（无 targets）不崩
4. 落盘文件存在且可解析
"""

import json
from pathlib import Path

import pytest

from graspo.cli.analysis import analyze_attribution, analyze_epochs


def _tool_call_completion(name: str, params: dict[str, str]) -> str:
    body = f"<function={name}>\n"
    for key, value in params.items():
        body += f"<parameter={key}>\n{value}\n</parameter>\n"
    return f"<tool_call>\n{body}</tool_call>"


def _record(
    *,
    sample_index: int,
    step: int,
    decision: str,
    completions: list[str],
    target_fn: str = "extend_arm",
    target_args: dict | None = None,
    attempt_number: int = 1,
) -> dict:
    return {
        "event": "graspo_group",
        "sample_index": sample_index,
        "step": step,
        "decision": decision,
        "attempt_number": attempt_number,
        "max_attempts": 6,
        "targets": [
            {
                "id": "primary",
                "output": {"tool_calls": [{"name": target_fn, "arguments": target_args or {}}]},
            }
        ],
        "completions": [{"idx": i, "completion": text} for i, text in enumerate(completions)],
    }


@pytest.fixture
def run_dir(tmp_path: Path) -> Path:
    rd = tmp_path / "run_v21"
    (rd / "logs").mkdir(parents=True)
    return rd


def _write_readable(run_dir: Path, records: list[dict]) -> None:
    path = run_dir / "logs" / "rollouts.readable.jsonl"
    with path.open("w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")


class TestNotCorrectCauses:
    def test_tool_mismatch(self, run_dir: Path) -> None:
        """工具名全错 → tool_mismatch。"""
        rec = _record(
            sample_index=0,
            step=1,
            decision="trainable_not_correct",
            completions=[
                _tool_call_completion(
                    "rotate_arm", {"action_type": "顺时针旋转", "angle_deg": "36.5"}
                )
                for _ in range(8)
            ],
            target_fn="extend_arm",
            target_args={"action_type": "收缩手臂", "distance_cm": 6.1},
        )
        _write_readable(run_dir, [rec])
        attr = analyze_attribution(run_dir)
        assert attr["available"] is True
        assert attr["not_correct_causes"] == {"tool_mismatch": 1}

    def test_content_all_wrong(self, run_dir: Path) -> None:
        """工具对但参数全错（数值/方向均不属于该层语义）→ content_all_wrong。"""
        rec = _record(
            sample_index=1,
            step=1,
            decision="trainable_not_correct",
            completions=[
                _tool_call_completion(
                    "extend_arm", {"action_type": "收缩手臂", "distance_cm": "30.0"}
                )
                for _ in range(8)
            ],
            target_fn="extend_arm",
            target_args={"action_type": "收缩手臂", "distance_cm": 6.1},
        )
        _write_readable(run_dir, [rec])
        attr = analyze_attribution(run_dir)
        assert attr["not_correct_causes"] == {"content_all_wrong": 1}

    def test_format_shortfall(self, run_dir: Path) -> None:
        """有参数全对的条但未达 perfect 阈值 → format_shortfall。"""
        # 8 条中 4 条参数完全匹配（含 target），4 条数值错
        ok = _tool_call_completion("extend_arm", {"action_type": "收缩手臂", "distance_cm": "6.1"})
        bad = _tool_call_completion(
            "extend_arm", {"action_type": "收缩手臂", "distance_cm": "30.0"}
        )
        rec = _record(
            sample_index=2,
            step=1,
            decision="trainable_not_correct",
            completions=[ok, ok, ok, ok, bad, bad, bad, bad],
            target_fn="extend_arm",
            target_args={"action_type": "收缩手臂", "distance_cm": 6.1},
        )
        _write_readable(run_dir, [rec])
        attr = analyze_attribution(run_dir)
        assert attr["not_correct_causes"] == {"format_shortfall": 1}

    def test_mixed_causes_aggregate(self, run_dir: Path) -> None:
        """多组混合：各原因计数独立聚合。"""
        recs = [
            _record(
                sample_index=0,
                step=1,
                decision="trainable_not_correct",
                completions=[
                    _tool_call_completion(
                        "rotate_arm", {"action_type": "顺时针旋转", "angle_deg": "1"}
                    )
                    for _ in range(8)
                ],
                target_fn="extend_arm",
                target_args={"action_type": "收缩手臂", "distance_cm": 6.1},
            ),
            _record(
                sample_index=1,
                step=1,
                decision="trainable_not_correct",
                completions=[
                    _tool_call_completion(
                        "extend_arm", {"action_type": "伸长手臂", "distance_cm": "30"}
                    )
                    for _ in range(8)
                ],
                target_fn="extend_arm",
                target_args={"action_type": "收缩手臂", "distance_cm": 6.1},
            ),
            _record(
                sample_index=2,
                step=1,
                decision="trainable_max_correct",
                completions=[
                    _tool_call_completion(
                        "extend_arm", {"action_type": "收缩手臂", "distance_cm": "6.1"}
                    )
                    for _ in range(8)
                ],
                target_fn="extend_arm",
                target_args={"action_type": "收缩手臂", "distance_cm": 6.1},
            ),
        ]
        _write_readable(run_dir, recs)
        attr = analyze_attribution(run_dir)
        assert attr["not_correct_causes"] == {"tool_mismatch": 1, "content_all_wrong": 1}
        cross = attr["decision_cross"]
        assert cross["trainable_max_correct"]["all_fn_ratio"] == 1.0
        assert cross["trainable_max_correct"]["all_param_ratio"] == 1.0
        # 工具错组：any_fn=0；参数错组：any_fn=1, any_param=0
        assert cross["trainable_not_correct"]["any_fn_ratio"] == 0.5
        assert cross["trainable_not_correct"]["any_param_ratio"] == 0.0


class TestDataIndependence:
    """与训练数据内容无关：换完全不同的工具/字段名，语义不变。"""

    def test_different_tool_names(self, run_dir: Path) -> None:
        """换一套工具（非 rotate_arm/extend_arm）分类语义一致。"""
        rec = _record(
            sample_index=0,
            step=1,
            decision="trainable_not_correct",
            completions=[_tool_call_completion("get_weather", {"city": "北京"}) for _ in range(8)],
            target_fn="get_time",
            target_args={"timezone": "UTC"},
        )
        _write_readable(run_dir, [rec])
        attr = analyze_attribution(run_dir)
        assert attr["not_correct_causes"] == {"tool_mismatch": 1}

    def test_string_only_targets(self, run_dir: Path) -> None:
        """纯字符串参数（无数值字段）正常工作。"""
        ok = _tool_call_completion("get_weather", {"city": "北京", "unit": "celsius"})
        rec = _record(
            sample_index=0,
            step=1,
            decision="trainable_max_correct",
            completions=[ok for _ in range(8)],
            target_fn="get_weather",
            target_args={"city": "北京", "unit": "celsius"},
        )
        _write_readable(run_dir, [rec])
        attr = analyze_attribution(run_dir)
        assert attr["decision_cross"]["trainable_max_correct"]["all_param_ratio"] == 1.0


class TestRobustness:
    def test_missing_readable(self, run_dir: Path) -> None:
        """readable 缺失 → available=False，不抛异常。"""
        attr = analyze_attribution(run_dir)
        assert attr["available"] is False

    def test_malformed_line_skipped(self, run_dir: Path) -> None:
        """损坏行跳过，不中断。"""
        path = run_dir / "logs" / "rollouts.readable.jsonl"
        path.write_text("not-json\n", encoding="utf-8")
        attr = analyze_attribution(run_dir)
        assert attr["available"] is True
        assert attr["records"] == 0

    def test_no_targets_record(self, run_dir: Path) -> None:
        """无 targets 的记录（非 tool_call 任务）跳过，不崩。"""
        rec = {
            "event": "graspo_group",
            "sample_index": 0,
            "step": 1,
            "decision": "trainable_not_correct",
            "targets": [],
            "completions": [{"idx": 0, "completion": "plain text"}],
        }
        _write_readable(run_dir, [rec])
        attr = analyze_attribution(run_dir)
        assert attr["available"] is True
        assert attr["not_correct_groups"] == 0

    def test_retry_records_excluded_from_trend(self, run_dir: Path) -> None:
        """retry 中间态不进 step_trend（terminal-only 口径）。"""
        recs = [
            _record(
                sample_index=0,
                step=2,
                decision="retry",
                completions=[
                    _tool_call_completion(
                        "extend_arm", {"action_type": "收缩手臂", "distance_cm": "6.1"}
                    )
                ]
                * 8,
                target_fn="extend_arm",
                target_args={"action_type": "收缩手臂", "distance_cm": 6.1},
                attempt_number=1,
            ),
            _record(
                sample_index=0,
                step=2,
                decision="trainable_max_correct",
                completions=[
                    _tool_call_completion(
                        "extend_arm", {"action_type": "收缩手臂", "distance_cm": "6.1"}
                    )
                ]
                * 8,
                target_fn="extend_arm",
                target_args={"action_type": "收缩手臂", "distance_cm": 6.1},
                attempt_number=2,
            ),
        ]
        _write_readable(run_dir, recs)
        attr = analyze_attribution(run_dir)
        # terminal 取 attempt_number 最大；trend 只有 1 组
        assert attr["terminal_groups"] == 1
        assert len(attr["step_trend"]) == 1
        assert attr["step_trend"][0]["groups"] == 1


def _epoch_summary(
    epoch: int,
    *,
    mc: int,
    nc: int,
    invalid: int,
    perfect: int,
    retry: int,
    total: int,
    reward: float,
    content: float,
    elapsed: int = 3600,
) -> dict:
    """构造一条 epoch_summary 事件（与训练侧写入结构一致）。"""
    return {
        "timestamp": "2026-08-07T00:00:00+00:00",
        "event": "epoch_summary",
        "epoch": epoch,
        "elapsed_sec": elapsed,
        "epoch_cumulative": {
            "epoch": epoch,
            "samples_seen": 322,
            "samples_total": 322,
            "progress": 1.0,
            "attempt_groups": 322,
            "completions": 2576,
            "decisions": {
                "rollout_attempts": {"total": total, "retry": retry, "terminal": 322},
                "terminal": {
                    "perfect_skip": perfect,
                    "trainable": mc + nc,
                    "invalid": invalid,
                    "invalid_no_preference_gap": 5,
                    "total": 322,
                },
                "trainable": {
                    "max_correct": mc,
                    "not_correct": nc,
                    "total": mc + nc,
                    "ratio": round(mc / (mc + nc), 4) if mc + nc else 0.0,
                },
            },
            "reward_mean": reward,
            "content_mean": content,
            "base_content_mean": round(content + 0.1, 4),
            "best_reward": 1.0047,
        },
        "run_cumulative": {"step": 22 + epoch * 23},
        "run_id": "test",
    }


class TestAnalyzeEpochs:
    def test_missing_events(self, run_dir: Path) -> None:
        """events.jsonl 缺失 → available=False。"""
        attr = analyze_epochs(run_dir)
        assert attr["available"] is False

    def test_no_epoch_summary(self, run_dir: Path) -> None:
        """只有 train_step 无 epoch_summary → available=False。"""
        path = run_dir / "logs" / "events.jsonl"
        path.write_text(json.dumps({"event": "train_step", "step": 1}) + "\n", encoding="utf-8")
        attr = analyze_epochs(run_dir)
        assert attr["available"] is False
        assert "no epoch_summary" in attr["reason"]

    def test_epochs_order_and_content(self, run_dir: Path) -> None:
        """多 epoch：顺序、字段、trends 正确。"""
        path = run_dir / "logs" / "events.jsonl"
        path.write_text(
            "\n".join(
                json.dumps(e, ensure_ascii=False)
                for e in [
                    _epoch_summary(
                        0,
                        mc=59,
                        nc=118,
                        invalid=127,
                        perfect=13,
                        retry=692,
                        total=1014,
                        reward=0.608,
                        content=0.704,
                    ),
                    _epoch_summary(
                        1,
                        mc=42,
                        nc=124,
                        invalid=129,
                        perfect=18,
                        retry=735,
                        total=1057,
                        reward=0.619,
                        content=0.679,
                    ),
                    _epoch_summary(
                        2,
                        mc=25,
                        nc=127,
                        invalid=115,
                        perfect=16,
                        retry=689,
                        total=1011,
                        reward=0.568,
                        content=0.635,
                    ),
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        result = analyze_epochs(run_dir)
        assert result["available"] is True
        assert [e["epoch"] for e in result["epochs"]] == [0, 1, 2]
        assert result["epochs"][0]["terminal"]["invalid"] == 127
        assert result["epochs"][0]["trainable"]["max_correct"] == 59
        assert result["epochs"][0]["samples_seen"] == 322
        assert result["epochs"][0]["elapsed_sec"] == 3600
        assert result["trends"]["mc_ratio"] == [0.3333, 0.253, 0.1645]
        assert result["trends"]["content_mean"] == [0.704, 0.679, 0.635]
        assert result["trends"]["invalid"] == [127, 129, 115]
        assert result["trends"]["max_correct"] == [59, 42, 25]
        assert result["trends"]["perfect_skip"] == [13, 18, 16]

    def test_malformed_line_skipped(self, run_dir: Path) -> None:
        """损坏行跳过，有效 epoch_summary 保留。"""
        path = run_dir / "logs" / "events.jsonl"
        path.write_text(
            "not-json\n"
            + json.dumps(
                _epoch_summary(
                    0, mc=1, nc=7, invalid=2, perfect=0, retry=10, total=20, reward=0.5, content=0.6
                )
            )
            + "\n",
            encoding="utf-8",
        )
        result = analyze_epochs(run_dir)
        assert result["available"] is True
        assert len(result["epochs"]) == 1
        assert result["epochs"][0]["epoch"] == 0
