"""``graspo.eval.vllm_client`` 与 ``graspo.eval.schema`` 的单测。

核心是**温度防呆回归**：温度必须锁在 0，且不存在任何调用方可覆盖的通道。
"""

from __future__ import annotations

import base64
import time

import pytest

from graspo.eval.schema import EVAL_ARTIFACT_SCHEMA_VERSION, EvalDecoding, EvalSummary, make_run_id
from graspo.eval.vllm_client import (
    EVAL_MAX_TOKENS,
    EVAL_TEMPERATURE,
    EVAL_TOP_P,
    ClientConfig,
    TemperatureLockError,
    VllmEvalClient,
    VllmEvalError,
    assert_request_body_locked,
    build_messages,
    build_request_body,
)


def _sample(image_path: str = "/tmp/does-not-matter.jpg"):
    """构造一条评测样本。``image_path`` 必须是真实存在的文件（缺图会明确报错）。"""
    from graspo.eval.dataset import EvalSample

    return EvalSample(
        index=0,
        sample_key="test.jsonl#0",
        messages=[
            {"role": "system", "content": [{"type": "text", "text": "sys"}]},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_path},
                    {"type": "text", "text": "pick"},
                ],
            },
            {"role": "assistant", "content": "plain string stays a string"},
        ],
        tools=[{"type": "function", "function": {"name": "rotate_arm"}}],
        targets=[
            {"output": {"tool_calls": [{"name": "rotate_arm", "arguments": {"action_type": "l"}}]}}
        ],
        image_paths=[image_path],
    )


def test_temperature_constant_is_zero():
    assert EVAL_TEMPERATURE == 0.0
    assert EVAL_TOP_P == 0.9
    assert EVAL_MAX_TOKENS == 128


def test_missing_image_fails_loudly_instead_of_silently_skipping():
    """缺图必须明确失败：静默跳过会让分母悄悄变小，与历史数字不可比。"""
    from graspo.eval.dataset import DatasetError
    from graspo.eval.vllm_client import build_request_body

    with pytest.raises(DatasetError, match="image missing"):
        build_request_body(_sample("/tmp/definitely-absent.jpg"), "served")


def test_messages_use_the_resolved_paths_not_the_raw_relative_reference(tmp_path):
    """回归：数据里的 image 是相对引用（../images/x.jpg），直接拿它读文件会失败。

    图像路径的唯一真相源是 ``sample.image_paths``（dataset.py 已按数据文件父目录
    解析）。这个 bug 是靠对真实 v5 数据的端到端干跑发现的。
    """
    from graspo.eval.dataset import EvalSample

    image = tmp_path / "images" / "resolved.jpg"
    image.parent.mkdir(parents=True, exist_ok=True)
    image.write_bytes(b"\xff\xd8\xffRESOLVED")
    sample = EvalSample(
        index=0,
        sample_key="k",
        # message 里是相对引用；image_paths 是解析后的绝对路径
        messages=[
            {"role": "user", "content": [{"type": "image", "image": "../images/resolved.jpg"}]},
        ],
        tools=None,
        targets=[],
        image_paths=[str(image)],
    )
    body = build_request_body(sample, "served")
    url = body["messages"][0]["content"][0]["image_url"]["url"]
    assert url.startswith("data:image/jpeg;base64,")
    assert base64.b64decode(url.split(",", 1)[1]) == b"\xff\xd8\xffRESOLVED"


def test_inconsistent_image_count_is_rejected(tmp_path):
    """messages 里的图片数与解析出的路径数不一致时，报错而不是猜。"""
    from graspo.eval.dataset import DatasetError, EvalSample
    from graspo.eval.vllm_client import build_messages

    sample = EvalSample(
        index=0,
        sample_key="k",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": "a.jpg"},
                    {"type": "image", "image": "b.jpg"},
                ],
            }
        ],
        tools=None,
        targets=[],
        image_paths=[str(tmp_path / "a.jpg")],
    )
    with pytest.raises(DatasetError, match="inconsistent"):
        build_messages(sample)


def test_build_request_body_has_no_temperature_parameter():
    """温度不是参数——签名里没有它，调用方无从覆盖（§2.2 显式即防呆）。"""
    import inspect

    signature = inspect.signature(build_request_body)
    assert "temperature" not in signature.parameters
    assert set(signature.parameters) == {"sample", "served_model_name"}


def test_build_request_body_pins_temperature_zero(tmp_path):
    image = tmp_path / "pic.jpg"
    image.write_bytes(b"\xff\xd8\xff")
    body = build_request_body(_sample(str(image)), "served")
    assert body["temperature"] == 0.0
    assert body["model"] == "served"
    assert body["max_tokens"] == EVAL_MAX_TOKENS
    assert body["chat_template_kwargs"] == {"enable_thinking": False}
    assert body["tools"] == [{"type": "function", "function": {"name": "rotate_arm"}}]
    assert "top_p" in body


def test_build_request_body_omits_tools_when_absent(tmp_path):
    image = tmp_path / "pic.jpg"
    image.write_bytes(b"\xff\xd8\xff")
    sample = _sample(str(image)).model_copy(update={"tools": None})
    body = build_request_body(sample, "served")
    assert "tools" not in body


def test_temperature_lock_catches_nonzero_values():
    with pytest.raises(TemperatureLockError):
        assert_request_body_locked({"temperature": 0.1})
    with pytest.raises(TemperatureLockError):
        assert_request_body_locked({"Temperature": 1.0})
    with pytest.raises(TemperatureLockError):
        assert_request_body_locked({"nested_temperature": 0.7})


def test_temperature_lock_allows_zero_and_unrelated_keys():
    assert_request_body_locked({"temperature": 0.0, "max_tokens": 128, "top_p": 0.9})
    assert_request_body_locked({"model": "x", "messages": []})


# ── 嵌套绕过（审查者发现：只看顶层会被 extra_body.temperature 绕过）──────────


def test_temperature_lock_catches_nested_temperature():
    """回归：``extra_body.temperature`` 这类嵌套结构必须被拦住。

    vLLM 的 OpenAI 兼容接口通过 ``extra_body`` 透传采样参数；只扫顶层等于给
    嵌套路径开后门——而本函数的存在意义就是防绕过。
    """
    with pytest.raises(TemperatureLockError, match=r"extra_body\.temperature"):
        assert_request_body_locked({"extra_body": {"temperature": 0.7}})

    with pytest.raises(TemperatureLockError):
        assert_request_body_locked({"a": {"b": {"c": {"temperature": 1.0}}}})


def test_temperature_lock_catches_deeply_nested_variants():
    with pytest.raises(TemperatureLockError):
        assert_request_body_locked({"sampling_params": {"Temperature": 0.5}})
    with pytest.raises(TemperatureLockError):
        assert_request_body_locked({"sampling_params": {"top_k_temp": 0.3}})
    with pytest.raises(TemperatureLockError):
        assert_request_body_locked({"x": {"Temperature": 2}})


def test_temperature_lock_scans_lists_and_tuples():
    with pytest.raises(TemperatureLockError, match=r"\[0\]"):
        assert_request_body_locked({"samples": [{"temperature": 0.9}]})
    with pytest.raises(TemperatureLockError):
        assert_request_body_locked({"samples": ({"temperature": 0.9},)})
    with pytest.raises(TemperatureLockError):
        assert_request_body_locked({"outer": {"inner": [{"temperature": 0.4}]}})


def test_temperature_lock_reports_full_path():
    try:
        assert_request_body_locked({"extra_body": {"nested": [{"temperature": 0.8}]}})
    except TemperatureLockError as exc:
        message = str(exc)
        assert "extra_body.nested[0].temperature" in message
    else:  # pragma: no cover - 上面必须抛
        pytest.fail("nested temperature was not caught")


def test_temperature_lock_allows_nested_zero():
    assert_request_body_locked({"extra_body": {"temperature": 0.0}, "sampling": [{"temp": 0}]})
    # 非数值的同名键无法当采样温度用，放行（误报比漏报更打断合法请求）
    assert_request_body_locked({"extra_body": {"temperature": "auto"}})
    assert_request_body_locked({"extra_body": {"temperature": None}})


def test_temperature_lock_rejects_absurdly_deep_nesting():
    """超过深度上限时**拒绝扫描**而不是跳过——跳过等于给深层结构留后门。"""
    from graspo.eval.vllm_client import _MAX_SCAN_DEPTH

    body: dict = {"level": {}}
    cursor = body["level"]
    for _ in range(_MAX_SCAN_DEPTH + 5):
        cursor["level"] = {}
        cursor = cursor["level"]
    with pytest.raises(TemperatureLockError, match="nests deeper than"):
        assert_request_body_locked(body)


def test_build_messages_keeps_string_content_and_converts_list_content(tmp_path):
    """与 v3 一致：list content 逐项转换，str content 原样保留。"""
    image = tmp_path / "pic.jpg"
    image.write_bytes(b"\xff\xd8\xff")
    from graspo.eval.dataset import EvalSample

    sample = EvalSample(
        index=0,
        sample_key="k",
        messages=[
            {"role": "user", "content": [{"type": "image", "image": str(image)}]},
            {"role": "assistant", "content": "text"},
        ],
        tools=None,
        targets=[],
        image_paths=[str(image)],
    )
    messages = build_messages(sample)
    assert messages[1] == {"role": "assistant", "content": "text"}
    part = messages[0]["content"][0]
    assert part["type"] == "image_url"
    # build_messages 只放占位；真正的编码在 build_request_body（每样本一次）
    assert part["image_url"]["url"] == ""

    body = build_request_body(sample, "served")
    url = body["messages"][0]["content"][0]["image_url"]["url"]
    assert url.startswith("data:image/jpeg;base64,")


def test_decoding_model_rejects_nonzero_temperature():
    """产物契约层的第二道防线：非 0 温度连产物都构造不出来。"""
    with pytest.raises(Exception, match="temperature must be exactly 0.0"):
        EvalDecoding(temperature=0.1, top_p=0.9, max_tokens=128, enable_thinking=False)
    ok = EvalDecoding(temperature=0.0, top_p=0.9, max_tokens=128, enable_thinking=False)
    assert ok.temperature == 0.0


def test_summary_carries_overlap_fields_as_none_when_not_computed():
    """None 表示"未计算"，不是 0——两者语义不同（§2.2）。"""
    summary = EvalSummary(
        sample_count_total=1,
        sample_count_valid=1,
        sample_count_error=0,
        correct=1,
        incorrect=0,
        accuracy=1.0,
        accuracy_percent=100.0,
    )
    assert summary.accuracy_percent_excluding_overlap is None
    assert summary.overlap_excluded_count is None


def test_artifact_schema_version_and_run_id_shape():
    assert EVAL_ARTIFACT_SCHEMA_VERSION == "graspo-eval-report-1"
    run_id = make_run_id("base")
    assert run_id.startswith("base-")
    assert len(run_id.split("-")) == 3


# ── 就绪轮询的**总等待预算**（2026-09-28 挂死修复）────────────────────────────


def test_wait_until_ready_is_bounded_by_the_total_deadline(monkeypatch):
    """★ 判别力实证：预算用尽即返回，**不会**跑满 60 轮 × （3 重试 × 120s 超时）。

    缺陷背景（本机实测）：旧实现只按 ``attempts`` 收口，而每次探测的 ``_post`` 自带
    ``max_retries × timeout_sec`` 的重试上界 ⇒ 真实上界 ≈ 6 小时，与 docstring 宣称的
    "最多等 10 分钟"不符。实测后果：没有真 vLLM 服务时整套单测在 40% 处**挂死**。
    本用例让 ``_post`` 永远抛错、预算设为 0.05s，断言它很快返回且探测次数远小于 60。
    """
    client = VllmEvalClient(ClientConfig(base_url="http://127.0.0.1:1", served_model_name="s"))
    probe_calls: list[int] = []

    def _always_fail(body: dict) -> dict:
        probe_calls.append(1)
        raise VllmEvalError("connection refused (mocked)")

    monkeypatch.setattr(client, "_post", _always_fail)
    started = time.monotonic()
    with pytest.raises(VllmEvalError, match="budget"):
        client.wait_until_ready(interval_sec=1.0, deadline_sec=0.05)
    elapsed = time.monotonic() - started

    assert elapsed < 5.0, f"预算 0.05s 却等了 {elapsed:.2f}s —— 上限没生效"
    assert len(probe_calls) < 60, f"探测了 {len(probe_calls)} 次 —— 次数没按预算收口"


def test_wait_until_ready_returns_as_soon_as_the_service_answers(monkeypatch):
    """反向：服务就绪即返回，不许被预算拖住（预算不是"至少等这么久"）。"""
    client = VllmEvalClient(ClientConfig(base_url="http://127.0.0.1:1", served_model_name="s"))
    monkeypatch.setattr(client, "_post", lambda body: {"choices": [{"message": {"content": "hi"}}]})

    started = time.monotonic()
    client.wait_until_ready(interval_sec=1.0, deadline_sec=600.0)

    assert time.monotonic() - started < 5.0
