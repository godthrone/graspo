"""vLLM OpenAI 兼容接口的评测客户端——温度锁 0。

**职责**：构造与历史口径一致的 Chat Completions 请求体、并发发请求、
解析响应为 ``Prediction``，并逐样本产出明细记录。

**本文件不负责**：起 vLLM 服务（`orchestrator` / 调用方）、聚合统计
（`evaluate.py`）、判定对错（`criteria.py`）、锁卡（`guard.py`）。

**温度为什么必须锁 0，以及锁在哪**

用户已拍板：推理温度 = 0。原因有实测依据——同一份权重、温度 0.1 复测出现过
51.1% vs 52.7% 的 ±1.6pp 抖动，与达标余量同阶；不锁 0 就无法区分"模型变好了"
和"这次采样运气好"。

实现上做了**三重**保障，而不是靠调用方自觉：

1. 模块级常量 ``EVAL_TEMPERATURE = 0.0``，全项目唯一温度真相源。
2. :func:`build_request_body` **不接受**温度参数——没有可覆盖的开关（宪法 §2.2
   显式即防呆：不留"反正有默认值"的隐式通道）。
3. :class:`VllmEvalClient` 在发送前对请求体调用 :func:`assert_request_body_locked`
   ，**递归遍历 dict / list / tuple 的全部层级**，对任意含 ``temperature`` /
   ``temp`` / ``temp_`` 的键名（大小写不敏感）检查取值，发现非 0 数值即抛错终止。
   这一层防的是"以后有人绕过 builder 直接塞 body"，包括
   ``extra_body.temperature`` 这类**嵌套**路径——只看顶层会留下绕过口子。

**top_p 与温度 0 的关系**：温度为 0（贪心解码）时 top_p 不参与采样，取值无效果。
本模块显式发送 ``top_p: 0.9``——与 v3 请求体保持逐字段可比，便于事后对照
（v3 见 ``run_vllm_eval.py:76``）。

**与 v3 的一致性**：``chat_template_kwargs.enable_thinking=false``、
``max_tokens=128``、``tools`` 仅在非空时发送、取 ``tool_calls[0]``
——均逐项对齐 ``run_vllm_eval.py:71-112``。
"""

from __future__ import annotations

import base64
import json
import time
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any
from urllib import error as urllib_error
from urllib import request as urllib_request

from graspo.eval.criteria import extract_ground_truth, extract_prediction, is_all_right
from graspo.eval.dataset import DatasetError, EvalSample, read_image_bytes

#: 全项目唯一的评测温度真相源。改动它等于改变口径，必须走用户决策。
EVAL_TEMPERATURE = 0.0

#: 与 v3 对齐的解码参数（``run_vllm_eval.py:74-77``）。
EVAL_TOP_P = 0.9
EVAL_MAX_TOKENS = 128
EVAL_ENABLE_THINKING = False

#: 请求体里任何与温度相关的键名（大小写不敏感）都必须在禁区名单内。
#: 扫描是**递归**的（见 :func:`assert_request_body_locked`）——只看顶层会被
#: ``extra_body.temperature`` 这类嵌套结构绕过。
_TEMPERATURE_KEY_PATTERN = ("temperature", "temp", "temp_")

#: 递归扫描的最大深度。真实 OpenAI/vLLM 请求体的嵌套层级是个位数；
#: 设上限是为了让**畸形或恶意构造的深层结构快速失败**，而不是在深层递归上耗时。
_MAX_SCAN_DEPTH = 32


class VllmEvalError(RuntimeError):
    """评测链路中与 vLLM 交互相关的失败。"""


class TemperatureLockError(VllmEvalError):
    """温度防呆被触发：请求体里出现了非 0 温度。"""


def assert_request_body_locked(body: dict[str, Any]) -> None:
    """防呆：请求体里**任意层级**都不允许出现非 0 温度。

    递归遍历 dict / list / tuple 的全部层级。只看顶层是不够的——vLLM 的
    ``extra_body``、以及任何未来新增的嵌套采样参数，都能把温度藏进子字典里绕过
    顶层检查。既然本函数的存在意义就是"防绕过"，就不能留这个口子。

    判定规则：键名（大小写不敏感）含 ``temperature`` / ``temp`` / ``temp_`` 任一片段，
    且值是非 0 的数值（``int``/``float``；``bool`` 不算数值，但也不该出现在温度位）
    ⇒ 抛错。**非数值**的同名键（如 ``"temperature": "auto"``）不报错——它无法被用作
    采样温度，且直接放行比误报更少打断合法请求。

    Args:
        body: 即将发送的 JSON 请求体。

    Raises:
        TemperatureLockError: 任意层级出现非 0 温度字段（错误信息带完整路径）。
        TemperatureLockError: 嵌套深度超过 :data:`_MAX_SCAN_DEPTH`（拒绝扫描，
            而不是跳过——跳过等于给深层结构开后门）。
    """
    for path, value in _iter_key_values(body):
        if len(path) > _MAX_SCAN_DEPTH:
            raise TemperatureLockError(
                f"request body nests deeper than {_MAX_SCAN_DEPTH} levels at "
                f"{_render_path(path[:8])}…; refusing to scan — a structure deeper than "
                "any real request cannot be verified, and skipping it would leave a "
                "temperature back door"
            )
        key = path[-1]
        if not _is_temperature_key(key):
            continue
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            continue
        if float(value) != 0.0:
            raise TemperatureLockError(
                f"request body carries {_render_path(path)}={value!r}; evaluation temperature "
                f"is locked to {EVAL_TEMPERATURE} — non-zero sampling produced "
                "±1.6pp jitter at 0.1"
            )


def _iter_key_values(
    value: Any,
    path: tuple[str, ...] = (),
) -> Iterator[tuple[tuple[str, ...], Any]]:
    """深度优先产出 ``(键路径, 值)``；只下钻 dict / list / tuple。

    产出的是**每一个**键值对（含中间层级），因此调用方可以用统一的
    "路径 + 值"规则做判定，不必为顶层与嵌套写两套逻辑。
    """
    if isinstance(value, dict):
        for key, item in value.items():
            child_path = (*path, str(key))
            yield child_path, item
            yield from _iter_key_values(item, child_path)
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            child_path = (*path, f"[{index}]")
            yield child_path, item
            yield from _iter_key_values(item, child_path)


def _is_temperature_key(key: str) -> bool:
    """键名是否与温度相关（大小写不敏感的子串匹配）。"""
    lowered = key.lower()
    return any(token in lowered for token in _TEMPERATURE_KEY_PATTERN)


def _render_path(path: tuple[str, ...]) -> str:
    """把键路径渲染成 ``extra_body.temperature`` 形式（便于定位问题）。"""
    rendered = ""
    for part in path:
        if part.startswith("["):
            rendered += part
        else:
            rendered += ("." if rendered else "") + part
    return rendered


def build_messages(sample: EvalSample) -> list[dict[str, Any]]:
    """把样本 messages 还原成请求格式，图像就地 base64 内联。

    与 v3 ``run_vllm_eval.py:28-43`` 逐项一致：``content`` 为 list 的消息逐项
    转换；``image`` 项转成 ``image_url`` + data URL；``text`` 项转成 ``text``；
    ``content`` 为 str 的消息原样保留。

    **图像路径的唯一真相源是 ``sample.image_paths``**（已由 ``dataset.py`` 按数据文件
    父目录解析为绝对路径），不是 message 里的相对引用。直接拿 ``item["image"]``
    去读文件会失败——数据里的值是 ``../images/x.jpg``，只有相对数据目录才有意义。
    两者个数不一致说明样本被改动过，直接报错而不是猜。
    """
    messages: list[dict[str, Any]] = []
    for message in sample.messages:
        content = message.get("content")
        if not isinstance(content, list):
            messages.append({"role": message["role"], "content": content})
            continue
        parts: list[dict[str, Any]] = []
        for item in content:
            if item.get("type") == "image":
                # data URL 占位：真正的编码在 build_request_body 里一次性完成。
                parts.append({"type": "image_url", "image_url": {"url": ""}})
            else:
                parts.append({"type": "text", "text": item.get("text", "")})
        messages.append({"role": message["role"], "content": parts})

    expected = sum(
        1
        for message in messages
        if isinstance(message.get("content"), list)
        for item in message["content"]
        if item.get("type") == "image_url"
    )
    if expected != len(sample.image_paths):
        raise DatasetError(
            f"sample {sample.sample_key}: {expected} image reference(s) in messages but "
            f"{len(sample.image_paths)} resolved path(s) — the sample is inconsistent"
        )
    return messages


def build_request_body(sample: EvalSample, served_model_name: str) -> dict[str, Any]:
    """构造请求体。**没有温度参数**——温度在实现上不可被覆盖。

    图像在这里读取并编码为 data URL（每样本一次，4/6 张图分别编码）。
    读取失败会抛 ``DatasetError``，由 :meth:`VllmEvalClient.evaluate_sample` 记成
    无效样本——不会静默跳过（静默跳过会让分母悄悄变小，与历史数字不可比）。
    """
    messages = build_messages(sample)
    data_urls: list[str] = []
    for image_path in sample.image_paths:
        suffix, payload = read_image_bytes(image_path)
        encoded = base64.b64encode(payload).decode("ascii")
        data_urls.append(f"data:{_mime_for(suffix)};base64,{encoded}")
    cursor = 0
    for message in messages:
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for item in content:
            if item.get("type") == "image_url":
                item["image_url"]["url"] = data_urls[cursor]
                cursor += 1

    body: dict[str, Any] = {
        "model": served_model_name,
        "messages": messages,
        "max_tokens": EVAL_MAX_TOKENS,
        "temperature": EVAL_TEMPERATURE,
        "top_p": EVAL_TOP_P,
        "chat_template_kwargs": {"enable_thinking": EVAL_ENABLE_THINKING},
    }
    if sample.tools:
        body["tools"] = sample.tools
    return body


def _mime_for(suffix: str) -> str:
    """扩展名 → MIME。jpg/jpeg → image/jpeg，其余按 ``image/<suffix>``。"""
    return "image/jpeg" if suffix in {"jpg", "jpeg"} else f"image/{suffix}"


@dataclass(slots=True)
class SampleRecord:
    """逐样本评测明细——报告里可重算聚合结论的最小充分信息。"""

    sample_index: int
    sample_key: str
    gt_tool_name: str
    gt_action_type: str
    pred_tool_name: str
    pred_action_type: str
    all_right: bool
    completion_text: str
    parse_error: str | None
    error: str | None
    elapsed_sec: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "sample_index": self.sample_index,
            "sample_key": self.sample_key,
            "gt_name": self.gt_tool_name,
            "gt_action": self.gt_action_type,
            "pred_name": self.pred_tool_name,
            "pred_action": self.pred_action_type,
            "all_right": self.all_right,
            "completion": self.completion_text,
            "parse_error": self.parse_error,
            "error": self.error,
            "elapsed_sec": round(self.elapsed_sec, 4),
        }


@dataclass(slots=True)
class ClientConfig:
    """vLLM 客户端配置。``base_url`` 指向 OpenAI 兼容端点。"""

    base_url: str
    served_model_name: str
    timeout_sec: float = 120.0
    max_workers: int = 8
    max_retries: int = 3


class VllmEvalClient:
    """并发调用 vLLM Chat Completions 的评测客户端。

    并发用线程池（标准库，无新依赖）。请求之间无共享状态，逐样本记录在返回后
    按原始下标排序，因此结果顺序与串行执行一致（可复现）。
    """

    def __init__(self, config: ClientConfig) -> None:
        self._config = config
        self._endpoint = config.base_url.rstrip("/")
        if not self._endpoint.endswith("/chat/completions"):
            self._endpoint = f"{self._endpoint}/v1/chat/completions"

    def wait_until_ready(self, *, attempts: int | None = None, interval_sec: float = 10.0) -> None:
        """轮询直到服务可用。默认最多等 10 分钟（60 次 × 10s，与 v3 一致）。

        Raises:
            VllmEvalError: 超时。
        """
        limit = attempts if attempts is not None else 60
        probe = {
            "model": self._config.served_model_name,
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 4,
            "temperature": EVAL_TEMPERATURE,
        }
        assert_request_body_locked(probe)
        last_error = ""
        for attempt in range(1, limit + 1):
            try:
                payload = self._post(probe)
                if payload.get("choices"):
                    return
                last_error = f"response without choices: {payload!r}"
            except VllmEvalError as exc:
                last_error = str(exc)
            if attempt < limit:
                time.sleep(interval_sec)
        raise VllmEvalError(
            f"vLLM not ready after {limit} attempts ({last_error}); endpoint={self._endpoint}"
        )

    def evaluate_samples(self, samples: list[EvalSample]) -> list[SampleRecord]:
        """并发评测全部样本，按 ``sample.index`` 升序返回明细。"""
        if not samples:
            return []
        with ThreadPoolExecutor(max_workers=max(1, self._config.max_workers)) as pool:
            records = list(pool.map(self.evaluate_sample, samples))
        return sorted(records, key=lambda record: record.sample_index)

    def evaluate_sample(self, sample: EvalSample) -> SampleRecord:
        """评测单样本。请求失败时返回带 ``error`` 的记录（计为无效样本，不中断整轮）。"""
        ground_truth = extract_ground_truth(sample.targets)
        started = time.monotonic()
        try:
            body = build_request_body(sample, self._config.served_model_name)
            assert_request_body_locked(body)
            payload = self._post(body)
        except (VllmEvalError, DatasetError) as exc:
            return SampleRecord(
                sample_index=sample.index,
                sample_key=sample.sample_key,
                gt_tool_name=ground_truth.tool_name,
                gt_action_type=ground_truth.action_type,
                pred_tool_name="",
                pred_action_type="",
                all_right=False,
                completion_text="",
                parse_error=None,
                error=str(exc),
                elapsed_sec=time.monotonic() - started,
            )

        choice = (payload.get("choices") or [{}])[0]
        message = choice.get("message") or {}
        prediction = extract_prediction(message)
        completion_text = str(message.get("content") or "")
        if prediction.tool_name:
            completion_text = json.dumps(
                {
                    "name": prediction.tool_name,
                    "arguments": {"action_type": prediction.action_type},
                },
                ensure_ascii=False,
            )
        return SampleRecord(
            sample_index=sample.index,
            sample_key=sample.sample_key,
            gt_tool_name=ground_truth.tool_name,
            gt_action_type=ground_truth.action_type,
            pred_tool_name=prediction.tool_name,
            pred_action_type=prediction.action_type,
            all_right=is_all_right(prediction, ground_truth),
            completion_text=completion_text,
            parse_error=prediction.parse_error,
            error=None,
            elapsed_sec=time.monotonic() - started,
        )

    def _post(self, body: dict[str, Any]) -> dict[str, Any]:
        """POST 一次，带同效退路（重试）。网络类失败重试不改变结果（宪法 §3.1）。"""
        data = json.dumps(body).encode("utf-8")
        last_error = ""
        for attempt in range(1, self._config.max_retries + 1):
            request = urllib_request.Request(
                self._endpoint,
                data=data,
                headers={"Content-Type": "application/json"},
            )
            try:
                with urllib_request.urlopen(request, timeout=self._config.timeout_sec) as response:
                    return json.loads(response.read().decode("utf-8"))
            except urllib_error.HTTPError as exc:
                detail = exc.read().decode("utf-8", errors="replace")[:400]
                last_error = f"HTTP {exc.code}: {detail}"
            except (urllib_error.URLError, TimeoutError, json.JSONDecodeError) as exc:
                last_error = f"{type(exc).__name__}: {exc}"
            if attempt < self._config.max_retries:
                time.sleep(min(2.0 * attempt, 10.0))
        raise VllmEvalError(
            f"request failed after {self._config.max_retries} attempts: {last_error}"
        )
