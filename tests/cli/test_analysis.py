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

from graspo.cli.analysis import (
    analyze_attribution,
    analyze_epochs,
    analyze_errors,
    analyze_perf,
    analyze_steps,
)


def _tool_call_completion(name: str, params: dict[str, str]) -> str:
    body = f"<function={name}>\n"
    for key, value in params.items():
        body += f"<parameter={key}>\n{value}\n</parameter>\n"
    return f"<tool_call>\n{body}</function>\n</tool_call>"


def _record(
    *,
    sample_index: int,
    step: int,
    decision: str,
    completions: list[str],
    target_fn: str = "extend_arm",
    target_args: dict | None = None,
    attempt_number: int = 1,
    epoch: int = 0,
) -> dict:
    return {
        "event": "graspo_group",
        "sample_index": sample_index,
        "step": step,
        "epoch": epoch,
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


class TestErrorClassification:
    """completion 格式错误类型分类（与数据内容无关）。"""

    def test_ok_completion(self) -> None:
        """合法单调用 → ok。"""
        from graspo.cli.analysis import classify_completion_error

        text = _tool_call_completion(
            "extend_arm", {"action_type": "收缩手臂", "distance_cm": "6.1"}
        )
        assert classify_completion_error(text, None) == "ok"

    def test_multi_call(self) -> None:
        """多个 <tool_call> 标签 → multi_call（reward 层才报 too many tool calls）。"""
        from graspo.cli.analysis import classify_completion_error

        one = _tool_call_completion("extend_arm", {"action_type": "收缩手臂", "distance_cm": "6.1"})
        two = one + "\n" + one
        assert classify_completion_error(two, None) == "multi_call"

    def test_no_tool_call(self) -> None:
        """纯文本无工具调用 → no_tool_call。"""
        from graspo.cli.analysis import classify_completion_error

        assert classify_completion_error("好的，我来看看", None) == "no_tool_call"

    def test_malformed_xml(self) -> None:
        """内部结构坏（真实形态：think 残缺/嵌套参数）→ malformed_xml。"""
        from graspo.cli.analysis import classify_completion_error

        # 真实样本：<function=think</function> 残缺 + <!-- 注释
        text = (
            "<tool_call>\n<function=think</function>\n<!--\n\n</think>\n\n"
            "<tool_call>\n<function=rotate_arm>\n</function>\n</tool_call>"
        )
        assert classify_completion_error(text, None) == "malformed_xml"

    def test_missing_param(self) -> None:
        """缺必填参数 → missing_param（需 tools schema）。"""
        from graspo.cli.analysis import classify_completion_error

        tools = [
            {
                "type": "function",
                "function": {
                    "name": "extend_arm",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "action_type": {"type": "string"},
                            "distance_cm": {"type": "number"},
                        },
                        "required": ["action_type", "distance_cm"],
                    },
                },
            }
        ]
        # 只有 action_type，缺必填 distance_cm
        text = (
            "<tool_call>\n<function=extend_arm>\n<parameter=action_type>\n"
            "收缩手臂\n</parameter>\n</function>\n</tool_call>"
        )
        assert classify_completion_error(text, tools) == "missing_param"

    def test_error_types_by_epoch(self, run_dir: Path) -> None:
        """按 epoch 聚合错误类型计数正确。"""
        ok = _tool_call_completion("extend_arm", {"action_type": "收缩手臂", "distance_cm": "6.1"})
        bad = (  # 多调用（第二个 tool_call 残缺）
            _tool_call_completion("extend_arm", {"action_type": "收缩手臂", "distance_cm": "6.1"})
            + "\n<tool_call>\n<function=rotate_arm>\n</function>\n</tool_call>"
        )
        rec = {
            "event": "graspo_group",
            "sample_index": 0,
            "step": 1,
            "epoch": 1,
            "decision": "trainable_not_correct",
            "targets": [{"output": {"tool_calls": [{"name": "extend_arm", "arguments": {}}]}}],
            "completions": [{"idx": i, "completion": ok if i < 6 else bad} for i in range(8)],
        }
        _write_readable(run_dir, [rec])
        attr = analyze_attribution(run_dir)
        errors = attr["error_types_by_epoch"]
        assert errors["1"]["ok"] == 6
        assert errors["1"]["other_parse"] == 2


# ── 四表重构：step 进度 / 错误原因 / 性能 ────────────────────────────────────


def _write_events(run_dir: Path, events: list[dict]) -> None:
    """写入 events.jsonl（train_step / epoch_summary 事件的测试构造）。"""
    path = run_dir / "logs" / "events.jsonl"
    with path.open("w", encoding="utf-8") as fh:
        for ev in events:
            fh.write(json.dumps(ev, ensure_ascii=False) + "\n")


def _train_step(
    step: int,
    *,
    epoch: int,
    samples_seen: int,
    ratio: float = 0.5,
    mc: int = 4,
    nc: int = 4,
    invalid: int = 0,
    retry: int = 0,
    reward: float = 0.6,
    content: float = 0.7,
    loss: float = -0.5,
    alarms: list[str] | None = None,
    total_sec: float = 100.0,
) -> dict:
    """构造一条 train_step 事件（与训练侧写入结构一致）。"""
    return {
        "event": "train_step",
        "step": step,
        "epoch": epoch,
        "run_cumulative": {"step": step},
        "epoch_cumulative": {
            "epoch": epoch,
            "samples_seen": samples_seen,
            "samples_total": 322,
        },
        "batch": {
            "decisions": {
                "rollout_attempts": {"total": retry + 8, "retry": retry, "terminal": 8},
                "terminal": {
                    "perfect_skip": 0,
                    "trainable": mc + nc,
                    "invalid": invalid,
                    "invalid_no_preference_gap": 0,
                    "total": 8,
                },
                "trainable": {
                    "max_correct": mc,
                    "not_correct": nc,
                    "total": mc + nc,
                    "ratio": ratio,
                },
            },
            "reward": {"mean": reward},
            "content": {"mean": content},
        },
        "optimize": {"loss_mean": loss},
        "health": {"ok": not alarms, "reasons": alarms or []},
        "timing": {
            "total_observed_sec": total_sec,
            "rollout_total_sec": 40.0,
            "rollout_queue_sec": 30.0,
            "prefill_sec": 10.0,
            "decode_sec": 20.0,
            "decode_tokens": 400,
            "optimize_sec": 30.0,
        },
    }


class TestAnalyzeSteps:
    def test_sample_range_differential_within_epoch(self, run_dir: Path) -> None:
        """同 epoch 内样本区间 = samples_seen 差分。"""
        _write_events(
            run_dir,
            [
                _train_step(1, epoch=0, samples_seen=8),
                _train_step(2, epoch=0, samples_seen=16),
            ],
        )
        res = analyze_steps(run_dir)
        assert res["available"] is True
        assert res["steps"][0]["samples_start"] == 0
        assert res["steps"][0]["samples_end"] == 8
        assert res["steps"][1]["samples_start"] == 8
        assert res["steps"][1]["samples_end"] == 16

    def test_sample_range_resets_at_epoch_boundary(self, run_dir: Path) -> None:
        """epoch 切换时样本区间从 0 重新累计（步数≠进度，区间是锚点）。"""
        _write_events(
            run_dir,
            [
                _train_step(24, epoch=0, samples_seen=322),
                _train_step(25, epoch=1, samples_seen=8),
                _train_step(26, epoch=1, samples_seen=16),
            ],
        )
        res = analyze_steps(run_dir)
        assert res["steps"][0]["samples_start"] == 0
        assert res["steps"][0]["samples_end"] == 322
        assert res["steps"][1]["samples_start"] == 0
        assert res["steps"][1]["samples_end"] == 8
        assert res["steps"][2]["samples_start"] == 8

    def test_metrics_and_alarms(self, run_dir: Path) -> None:
        """决策计数/质量/告警分类计数正确透传。"""
        _write_events(
            run_dir,
            [
                _train_step(
                    1,
                    epoch=0,
                    samples_seen=8,
                    mc=1,
                    nc=7,
                    ratio=0.125,
                    retry=6,
                    invalid=1,
                    reward=0.3,
                    loss=-0.16,
                    alarms=["batch_high_retry_rate", "batch_high_retry_rate"],
                    total_sec=187.9,
                )
            ],
        )
        row = analyze_steps(run_dir)["steps"][0]
        assert row["mc"] == 1 and row["nc"] == 7
        assert row["mc_ratio"] == 0.125
        assert row["invalid"] == 1
        assert row["reward_mean"] == 0.3
        assert row["loss_mean"] == -0.16
        assert row["total_sec"] == 187.9
        # retry_rate = retry/(retry+terminal_total) = 6/14
        assert row["retry_rate"] == round(6 / 14, 4)
        # 告警去重计数（同一原因出现两次记 2）
        assert row["alarms"] == {"batch_high_retry_rate": 2}

    def test_missing_events(self, run_dir: Path) -> None:
        """events.jsonl 缺失 → available=False。"""
        res = analyze_steps(run_dir)
        assert res["available"] is False


class TestAnalyzeErrors:
    def test_l2_tool_mismatch(self, run_dir: Path) -> None:
        """格式对但工具名错 → tool_mismatch（L2 细分，旧表给不出的信号）。"""
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
        res = analyze_errors(run_dir)
        assert res["available"] is True
        assert res["by_step"][0]["category"] == "tool_mismatch"
        assert res["by_step"][0]["count"] == 8
        assert res["by_step"][0]["samples"] == [0]

    def test_l2_param_name_mismatch(self, run_dir: Path) -> None:
        """工具对但参数名集合不等 → param_name_mismatch（缺参数名）。"""
        rec = _record(
            sample_index=0,
            step=1,
            decision="trainable_not_correct",
            completions=[
                _tool_call_completion("extend_arm", {"action_type": "收缩手臂"}) for _ in range(8)
            ],
            target_fn="extend_arm",
            target_args={"action_type": "收缩手臂", "distance_cm": 6.1},
        )
        _write_readable(run_dir, [rec])
        res = analyze_errors(run_dir)
        assert res["by_step"][0]["category"] == "param_name_mismatch"

    def test_l2_param_value_mismatch(self, run_dir: Path) -> None:
        """工具对、参数名对、值文本不等 → param_value_mismatch（字符串比较，不做数值误差）。"""
        rec = _record(
            sample_index=0,
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
        res = analyze_errors(run_dir)
        assert res["by_step"][0]["category"] == "param_value_mismatch"

    def test_l1_multi_call(self, run_dir: Path) -> None:
        """双 tool_call → multi_call（L1，修复验证核心指标）。"""
        rec = _record(
            sample_index=0,
            step=1,
            decision="trainable_not_correct",
            completions=[
                _tool_call_completion("extend_arm", {"action_type": "收缩手臂"})
                + _tool_call_completion("rotate_arm", {"angle_deg": "10"})
                for _ in range(8)
            ],
            target_fn="extend_arm",
            target_args={"action_type": "收缩手臂", "distance_cm": 6.1},
        )
        _write_readable(run_dir, [rec])
        res = analyze_errors(run_dir)
        assert res["by_step"][0]["category"] == "multi_call"

    def test_ok_and_other(self, run_dir: Path) -> None:
        """全对 → ok；无 targets 记录 → other。"""
        ok = _tool_call_completion("extend_arm", {"action_type": "收缩手臂", "distance_cm": "6.1"})
        recs = [
            _record(
                sample_index=0,
                step=1,
                decision="trainable_max_correct",
                completions=[ok for _ in range(8)],
                target_fn="extend_arm",
                target_args={"action_type": "收缩手臂", "distance_cm": 6.1},
            ),
            {
                "event": "graspo_group",
                "sample_index": 1,
                "step": 1,
                "decision": "trainable_not_correct",
                "targets": [],
                "completions": [{"idx": 0, "completion": "plain text"}],
            },
        ]
        _write_readable(run_dir, recs)
        res = analyze_errors(run_dir)
        cats = {r["category"]: r for r in res["by_step"]}
        assert cats["ok"]["count"] == 8
        assert cats["other"]["count"] == 1

    def test_dual_granularity_and_terminal_only(self, run_dir: Path) -> None:
        """step/epoch 双粒度；retry 中间态不统计（终态口径）。"""
        recs = [
            _record(
                sample_index=0,
                step=2,
                decision="retry",
                completions=[
                    _tool_call_completion("rotate_arm", {"action_type": "顺时针旋转"})
                    for _ in range(8)
                ],
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
                    for _ in range(8)
                ],
                target_fn="extend_arm",
                target_args={"action_type": "收缩手臂", "distance_cm": 6.1},
                attempt_number=2,
            ),
        ]
        _write_readable(run_dir, recs)
        res = analyze_errors(run_dir)
        step_cats = {r["category"]: r["count"] for r in res["by_step"]}
        assert step_cats == {"ok": 8}  # retry 中间态被跳过
        assert res["by_epoch"][0]["category"] == "ok"
        assert res["by_epoch"][0]["count"] == 8


class TestAnalyzePerf:
    def test_step_rows_and_derived_metrics(self, run_dir: Path) -> None:
        """step 行：queue_pct / throughput 从 timing 推导正确。"""
        _write_events(
            run_dir,
            [
                _train_step(1, epoch=0, samples_seen=8),
                _train_step(2, epoch=0, samples_seen=16),
            ],
        )
        res = analyze_perf(run_dir)
        assert res["available"] is True
        row = res["by_step"][0]
        # queue_pct = 30/40 = 75%；throughput = 400/20 = 20 tok/s
        assert row["queue_pct"] == 75.0
        assert row["throughput_tok_s"] == 20.0
        assert row["groups"] == 8
        assert row["retry_rate"] == 0.0
        assert row["total_sec"] == 100.0

    def test_epoch_aggregation(self, run_dir: Path) -> None:
        """epoch 聚合：耗时求和、比率重算（非均值）。"""
        _write_events(
            run_dir,
            [
                _train_step(1, epoch=0, samples_seen=8, retry=4, total_sec=100.0),
                _train_step(2, epoch=0, samples_seen=16, retry=4, total_sec=100.0),
            ],
        )
        res = analyze_perf(run_dir)
        row = res["by_epoch"][0]
        assert row["bucket"] == 0
        assert row["total_sec"] == 200.0
        assert row["rollout_sec"] == 80.0
        # 聚合 queue_pct = (30+30)/(40+40) = 75%
        assert row["queue_pct"] == 75.0
        # 聚合 throughput = (400+400)/(20+20) = 20
        assert row["throughput_tok_s"] == 20.0
        # retry_rate = 8/(8+16) = 0.3333
        assert row["retry_rate"] == round(8 / 24, 4)

    def test_missing_events(self, run_dir: Path) -> None:
        """events.jsonl 缺失 → available=False。"""
        res = analyze_perf(run_dir)
        assert res["available"] is False


class TestAnalyzeEpochsExtended:
    def test_loss_and_alarms_aggregated_from_train_steps(self, run_dir: Path) -> None:
        """epoch 表补齐 loss_mean（聚合）与 alarms（分类计数）与样本区间。"""
        _write_events(
            run_dir,
            [
                _train_step(
                    1, epoch=0, samples_seen=8, loss=-0.5, alarms=["batch_high_retry_rate"]
                ),
                _train_step(
                    2, epoch=0, samples_seen=16, loss=-0.7, alarms=["batch_high_retry_rate"]
                ),
                _epoch_summary(
                    0,
                    mc=4,
                    nc=4,
                    invalid=2,
                    perfect=0,
                    retry=10,
                    total=18,
                    reward=0.6,
                    content=0.7,
                ),
            ],
        )
        res = analyze_epochs(run_dir)
        assert res["available"] is True
        e = res["epochs"][0]
        assert e["loss_mean"] == round(-1.2 / 2, 6)
        assert e["alarms"] == {"batch_high_retry_rate": 2}
        assert e["samples_start"] == 0
        assert e["samples_end"] == 322
