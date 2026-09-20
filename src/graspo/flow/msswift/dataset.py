"""graspo / ARD JSONL → ms-swift 数据集（SFT / GRPO / CPT / OPD 四种形态）。

**职责边界**

- **本模块**：读 graspo 训练 JSONL（原生 graspo 样本，或 ARD v3 ``anchor_bank.jsonl``
  记录），经 E1 已签收的 ``GraspoToMsSwiftAdapter`` 归一到 graspo 样本契约，再产出
  ms-swift 能直接消费的行；写盘到本次运行的隔离目录。
- **不负责**：ms-swift 参数映射（``_config_mapping.py``）、训练循环与算法注入
  （``trainer.py`` / ``sft_trainer.py``）、奖励打分（``reward.py``）。

**两种形态（同一份输入数据，两种用法——对应决策 D4「SFT 与 RL 共享基座」）**

- SFT：``{"messages": [system?, user, assistant=<target text>]}``。
  target 文本由 ``ripple`` 的 ``build_sft_target_text`` 生成——**与 native SFT 用的是
  同一个函数**（宪法 §1.3：算法只实现一次）。
- GRPO：``{"messages": [system?, user], "targets": <JSON 字符串>}``。
  提示词交给 ms-swift 采样，``targets`` 作为额外列透传给奖励函数（ms-swift 会把
  数据集列原样喂给 reward，见 ``GRPO 奖励适配器`` ``reward.py``）。

**另两种形态（CPT / OPD，ms-swift 独有能力，见能力矩阵 §4）**

- CPT（继续预训练）：``{"messages": [{"role": "assistant", "content": <纯文本>}]}``
  ——ms-swift 官方预训练数据格式（``docs/.../Custom-dataset.md`` §Pre-training）。
- OPD（on-policy 蒸馏，走 GKD）：``{"messages": [system?, user]}``——纯提示词，
  **不带** ``targets`` 奖励列；监督信号来自教师模型的现场 logits。与 GRPO 的差别
  只有这一处，提示词适配逻辑复用同一次修复（§1.4）。

**输入格式自适应**：ARD v3 记录的 ``targets[i].output.content`` 是 **str**，graspo 原生
样本是 **dict**。本模块显式判别二者（``content`` 的类型），不做"猜字段"的隐式约定；
非法记录在边界处被拒（§2.3）。

**ms-swift 4.5.3 的消息契约（F-3 实测确证，2026-09-18）**

graspo 样本的 ``messages`` **不能**原样交给 ms-swift——ELAM V5 形态会同时踩中两条
上游契约，而失败在 ms-swift 里被包成一句与真因无关的
``ValueError: Failed to retrieve the dataset``（``swift/dataset/utils.py:108``，
``LazyLLMDataset.__getitem__`` 把每条样本的异常吞掉、重试耗尽后抛通用错误）：

1. **相对图像路径**：ELAM 落盘写作 ``{"type": "image", "image": "../images/x.jpg"}``，
   而 ms-swift 的 ``Template._preprocess_inputs`` → ``_load_image`` 直接用
   ``open(path)``——**按进程 cwd 解析**。cwd 一旦不是数据文件所在目录就是
   ``FileNotFoundError``（一手证据：``task-lora-multicard/evidence/run2gpu_full_first_fail.log:529``
   逐行 ``FileNotFoundError: '../images/...'``，抛出点与 F-3 完全一致）。
   修法：复用 native 侧已有的单一真相源
   ``ripple/multimodal/rows.py::resolve_messages_media_paths``，锚点取
   ``Path(data.train_path).parent``（与 native 的
   ``flow/trainer/sft_trainer.py:82`` / ``flow/trainer/trainer.py:305`` 逐字相同）。
2. **``content=None`` 破坏消息配对**：ELAM 多轮样本的中间 assistant 轮是
   ``{"role": "assistant", "content": null, "tool_calls": [...]}``（实测：6378 行中
   1902 行 5 条消息、462 行 8 条消息）。ms-swift 的 ``Template._swift_encode``
   要求剥掉 system 后 ``messages`` 是严格的 ``(query, response)`` 交替对，
   且 ``content is not None``；``content=null`` 会让该轮被
   ``StdTemplateInputs.normalize_openai_tool_calls`` 整条丢掉，配对随即错位。
   修法：:func:`build_ms_swift_messages` 把历史压缩成「system + 末轮 user」，
   历史里的 assistant 工具调用与工具返回**不丢**——以文本渲染进末轮 user，
   图像块原样保留在末轮 user 的内容列表里。

**为什么是"压平成单轮"而不是"修补多轮"**（§6.1 简单优先 + §18.1 不留负债）

ms-swift 的 swift 后端为多轮工具对话设计了一条复杂链路（``_preprocess_tool_call`` /
``_preprocess_standalone_tools`` / ``agent_template._format_tool_responses``），但
**ELAM 历史轮的 assistant 是 ``content=null``**——上游自己的
``normalize_openai_tool_calls`` 就会把这一轮丢掉，graspo 无法在不改 ms-swift 源码的
前提下把它救回来（§1.2 边界：不 patch 上游）。已经**实测跑通**的对照形态
（``task-lora-multicard``：2 卡/4 卡各 1 step、loss/grad_norm finite）也是
「system + 单轮 user（2 张图） + assistant」的扁平形态。因此本模块显式产出扁平形态，
并把历史**渲染成文本**（而不是静默丢弃）——这是"已知且可复核"的取舍，不是隐藏降级。
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

from graspo.flow.msswift._config_mapping import Stage
from graspo.flow.msswift.reward import GRASPO_TARGETS_COLUMN, GRASPO_TOOLS_COLUMN

logger = logging.getLogger(__name__)

#: ms-swift SFT / GRPO 行统一的对话列名（ms-swift AutoPreprocessor 的标准键）。
MESSAGES_KEY = "messages"

#: ms-swift 侧的**数据集列**：chat template 参数（如 ``enable_thinking``）。
#: T1 复验（4.5.3）：它不是命令行参数，而是 per-sample 列
#: （``rl_core/data.py::OnPolicySample.to_template_dict`` 只消费这一列）。
CHAT_TEMPLATE_KWARGS_KEY = "chat_template_kwargs"


def _first_target_output(record: dict[str, Any]) -> dict[str, Any] | None:
    targets = record.get("targets")
    if not isinstance(targets, list) or not targets:
        return None
    first = targets[0]
    if not isinstance(first, dict):
        return None
    output = first.get("output")
    return output if isinstance(output, dict) else None


def is_ard_record(record: dict[str, Any]) -> bool:
    """ARD v3 记录判据：``targets[0].output.content`` 是 **str**（graspo 原生为 dict）。"""
    output = _first_target_output(record)
    return output is not None and isinstance(output.get("content"), str)


def load_graspo_samples(path: str | Path) -> list[Any]:
    """读取训练 JSONL 并归一到 graspo ``Sample`` 契约（ARD 记录先过适配器）。

    Raises:
        ArdContractError: ARD 记录不符合共享基座契约。
        ValueError: JSONL 解析失败或记录格式非法（带行号定位）。
    """
    from graspo.core.schema import Sample
    from graspo.flow.msswift.adapter import GraspoToMsSwiftAdapter
    from graspo.ripple.data import sample_from_record

    adapter = GraspoToMsSwiftAdapter()
    samples: list[Sample] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at {path}:{line_no}: {exc}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"{path}:{line_no}: record must be a JSON object")
            try:
                normalized = adapter.convert_sample(record) if is_ard_record(record) else record
                samples.append(sample_from_record(normalized))
            except Exception as exc:
                raise ValueError(f"{path}:{line_no}: {exc}") from exc
    if not samples:
        raise ValueError(f"no training samples found in {path}")
    return samples


def _chat_template_column(config: Any) -> dict[str, Any]:
    """``model.chat_template_kwargs`` → ms-swift 数据集列（空则不加列）。

    ms-swift 把 chat template 参数当**数据集列**消费（per-sample），因此这里逐行带上，
    语义与 graspo 的全局配置等价（同一份值应用到每一行）。
    """
    kwargs = dict(getattr(config.model, "chat_template_kwargs", None) or {})
    return {CHAT_TEMPLATE_KWARGS_KEY: kwargs} if kwargs else {}


#: 历史轮渲染里工具返回的展示上限（字符）。超长截断只影响"给模型看的历史文本"，
#: 不影响任何被判定的目标内容（targets 不经过本函数）。
_HISTORY_TOOL_RESPONSE_LIMIT = 500


def resolve_sample_media_paths(samples: list[Any], data_dir: str | Path | None) -> int:
    """把每个样本的**相对图像/视频路径就地解析为绝对路径**；返回改写的路径数。

    **锚点与判定规则**与 native 侧逐字一致（``ripple/multimodal/rows.py::
    resolve_messages_media_paths`` 的前缀白名单 + ``image``/``path``/``url`` 三个键；
    native 传的锚点见 ``flow/trainer/sft_trainer.py:82``：``Path(train_path).parent``）。

    **唯一有意偏离**：拼路径只用 ``os.path.join``，**不调任何会规范化 ``..`` 的函数**。
    实测（本机 Python 3.13 与目标容器 Python 3.12 均复现）：``os.path.abspath``、
    ``Path.resolve``、``Path.absolute`` 都会把 ``..`` 折叠掉——数据写
    ``/subsets/../images/x.jpg`` 时它们产出 ``/images/x.jpg``，锚点当场失效、图像全找不到
    （本修复第一版踩的坑）。ms-swift 的 ``_check_path`` 用 ``os.path.exists``（由操作系统
    解析 ``..``），因此"绝对路径 + 原样保留 ``..``"才是与消费方一致的口径。

    Args:
        samples: :func:`load_graspo_samples` 产出的 ``Sample`` 列表（就地修改）。
        data_dir: 相对路径的锚点（本模块传训练 JSONL 的父目录）；``None`` 时不动任何路径。

    Returns:
        被改写的路径条数（供调用方/测试断言——0 表示没有任何相对路径需要改写）。
    """
    if data_dir is None:
        return 0
    base = _absolute_without_normalizing(data_dir)
    rewritten = 0
    for sample in samples:
        messages = getattr(sample, "messages", None)
        if not isinstance(messages, list):
            continue
        for message in messages:
            if not isinstance(message, dict):
                continue
            content = message.get("content")
            if not isinstance(content, list):
                continue
            for block in content:
                if not isinstance(block, dict):
                    continue
                kind = str(block.get("type") or "").lower()
                if kind not in {"image", "image_url", "video", "video_url"}:
                    continue
                for key in ("image", "path", "url", "video"):
                    value = block.get(key)
                    if isinstance(value, str) and _is_relative_media_path(value):
                        block[key] = os.path.join(base, value)
                        rewritten += 1
    return rewritten


def _absolute_without_normalizing(path: str | Path) -> str:
    """把路径变绝对，但**不**折叠 ``..``、**不**解析符号链接（§2.2 显式即防呆）。

    为什么不用现成函数：``os.path.abspath`` / ``Path.resolve`` / ``Path.absolute``
    在本项目实测环境下都会把 ``..`` 规范化掉（见 :func:`resolve_sample_media_paths`）。
    """
    text = os.fspath(path)
    return text if os.path.isabs(text) else os.path.join(os.getcwd(), text)


def _is_relative_media_path(value: str) -> bool:
    """相对路径判据（与 ``resolve_messages_media_paths`` 的前缀白名单逐字一致）。"""
    return not value.startswith(("http://", "https://", "/", "data:"))


def _block_text(block: Any) -> str:
    """把 ms-swift/OpenAI 形态的 content 块渲染成纯文本（图像块 → ``<image>``）。"""
    if isinstance(block, str):
        return block
    if not isinstance(block, dict):
        return ""
    kind = str(block.get("type") or "").lower()
    if kind == "text":
        return str(block.get("text") or "")
    if kind in {"image", "image_url"}:
        return "<image>"
    if kind in {"video", "video_url"}:
        return "<video>"
    if kind == "tool_call":
        return json.dumps(
            {
                "name": block.get("name"),
                "arguments": block.get("arguments"),
            },
            ensure_ascii=False,
        )
    return ""


def _content_text(content: Any) -> str:
    """消息 ``content`` → 纯文本（``str`` 原样；list 逐块渲染；``None`` → 空串）。"""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(_block_text(block) for block in content)
    return str(content)


def _final_user_message(messages: list[Any], index: int) -> dict[str, Any]:
    """取末轮 user 消息的**浅拷贝**（内容列表深拷贝，避免改动调用方的样本）。"""
    message = messages[index]
    if not isinstance(message, dict):
        raise ValueError(f"messages[{index}] must be a JSON object, got {type(message).__name__}")
    content = message.get("content")
    if content is None:
        raise ValueError(f"messages[{index}] (user) has no content; cannot build a prompt")
    if isinstance(content, list):
        content = [dict(block) if isinstance(block, dict) else block for block in content]
    return {"role": "user", "content": content}


def _render_history(messages: list[Any], last_user_index: int) -> str:
    """把末轮 user **之前**的历史渲染成一段纯文本（信息不丢，只换形态）。

    渲染规则（逐条显式，不猜字段类型）：

    - ``assistant``：先放 ``content`` 文本（工具调用常为 ``None`` → 空），再放
      ``tool_calls`` 里的 ``name(arguments)``；两者都空则整条跳过。
    - ``tool``：放工具返回内容（超 :data:`_HISTORY_TOOL_RESPONSE_LIMIT` 截断）。
    - 其余角色（system/user 等）：放其 ``content`` 文本，跳过空串。

    图像块渲染成 ``<image>`` 占位（历史轮的图**不重复输入**——ms-swift 会把
    每个 ``<image>`` 占位换成一张真实图像，重复输入会让显存随轮数线性膨胀）。
    """
    lines: list[str] = []
    for index, message in enumerate(messages[:last_user_index]):
        if not isinstance(message, dict):
            continue
        role = str(message.get("role") or "").lower()
        if role == "system":
            continue  # system 单独作为 system 消息保留，不重复进历史
        body = _content_text(message.get("content")).strip()
        tool_calls = message.get("tool_calls")
        if isinstance(tool_calls, list) and tool_calls:
            rendered = []
            for call in tool_calls:
                if not isinstance(call, dict):
                    continue
                function = call.get("function") if isinstance(call.get("function"), dict) else call
                arguments = json.dumps(function.get("arguments"), ensure_ascii=False)
                rendered.append(f"{function.get('name')}({arguments})")
            if rendered:
                body = f"{body} {' '.join(rendered)}".strip()
        if role == "tool":
            body = body[:_HISTORY_TOOL_RESPONSE_LIMIT]
        if not body:
            continue
        lines.append(f"[{role}] {body}")
    return "\n".join(lines)


def build_ms_swift_messages(messages: list[Any]) -> list[dict[str, Any]]:
    """graspo ``Sample.messages`` → ms-swift 可编码的消息序列（system + 末轮 user）。

    这是本模块对 ms-swift 4.5.3 消息契约的**唯一**适配点（模块 docstring 的两条实测
    约束在此落实）。产出的序列满足：

    - ``system``（若原样本首条是 system）：``content`` 为 **str**（不是块列表）。
    - ``user``：**最后一条** user 消息，图像块原样保留（顺序不变）；
      其文本 = 历史渲染 + 原文本（历史在前、当前指令在后）。
    - 长度恒为 1 或 2，且末条是 user——SFT 侧追一条 assistant 后即成为
      ms-swift 要求的 ``(user, assistant)`` 交替对；GRPO 侧直接就是提示词。

    Args:
        messages: 样本的 ``messages``（``core.schema`` 已保证非空、且末条不是 assistant）。

    Returns:
        ``[{"role": "system", "content": str}, {"role": "user", "content": str | list}]``
        或 ``[{"role": "user", ...}]``。

    Raises:
        ValueError: 消息列表里没有任何 user 轮（无法构造提示词——拒绝静默产出行）。
    """
    last_user_index = next(
        (
            index
            for index in range(len(messages) - 1, -1, -1)
            if isinstance(messages[index], dict)
            and str(messages[index].get("role") or "").lower() == "user"
        ),
        None,
    )
    if last_user_index is None:
        raise ValueError("messages contains no 'user' turn; cannot build an ms-swift prompt")

    normalized: list[dict[str, Any]] = []
    first = messages[0]
    if (
        isinstance(first, dict)
        and str(first.get("role") or "").lower() == "system"
        and first.get("content") is not None
    ):
        normalized.append({"role": "system", "content": _content_text(first.get("content"))})

    final_user = _final_user_message(messages, last_user_index)
    history = _render_history(messages, last_user_index)
    if history:
        _prepend_history_text(final_user, history)
    normalized.append(final_user)
    return normalized


def _prepend_history_text(final_user: dict[str, Any], history: str) -> None:
    """把历史文本插到末轮 user 的**第一个图像块之前**（就地修改，§2.2 显式）。"""
    content = final_user["content"]
    if isinstance(content, str):
        final_user["content"] = f"{history}\n{content}" if content else history
        return
    if not isinstance(content, list):
        final_user["content"] = history
        return
    insert_at = len(content)
    for index, block in enumerate(content):
        if isinstance(block, dict) and str(block.get("type") or "").lower() in {
            "image",
            "image_url",
        }:
            insert_at = index
            break
    content.insert(insert_at, {"type": "text", "text": history})


def build_sft_rows(samples: list[Any], *, config: Any = None) -> list[dict[str, Any]]:
    """graspo 样本 → ms-swift SFT 行（messages 末条为 assistant 目标文本）。

    消息序列先过 :func:`build_ms_swift_messages`（system + 末轮 user，图像块保留），
    再追一条 ``{"role": "assistant", "content": <纯文本>}``。assistant 侧**必须**是
    ``str``：ms-swift 的 ``Qwen3_5Template`` 对 assistant 内容有

    - ``Template._swift_encode`` 的 ``assert response_role in {'assistant'}`` 配对约束；
    - 以及"assistant ``content`` 若是裸 JSON 字符串，``datasets`` 的 Arrow 会把它推断成
      ``Json`` 并解码为 dict"这一实测坑（``task-lora-multicard`` 复现踩坑 ③）。

    ``build_sft_target_text`` 产出的正是 Qwen 原生 XML 纯文本，两条都满足。
    """
    from graspo.ripple.parsing.xml import build_sft_target_text

    extra_columns = _chat_template_column(config) if config is not None else {}
    rows: list[dict[str, Any]] = []
    for index, sample in enumerate(samples):
        if not sample.targets:
            raise ValueError(f"samples[{index}] has no targets; cannot build an SFT row")
        target_text = build_sft_target_text(sample.targets[0].get("output") or {})
        if not target_text.strip():
            raise ValueError(
                f"samples[{index}] produced an empty SFT target text "
                "(check targets[0].output.content / tool_calls)"
            )
        messages = build_ms_swift_messages(sample.messages)
        rows.append(
            {
                MESSAGES_KEY: [*messages, {"role": "assistant", "content": target_text}],
                **extra_columns,
            }
        )
    return rows


def build_grpo_rows(samples: list[Any], *, config: Any = None) -> list[dict[str, Any]]:
    """graspo 样本 → ms-swift GRPO 行（提示词 + ``targets`` 奖励列）。

    GRPO 只喂提示词：completion 由 ms-swift 现场采样（这正是 RL 与 SFT 的区别）。
    提示词同样过 :func:`build_ms_swift_messages`——``content=null`` 与相对图像路径
    在两处是一模一样的失败形态，修一次、两处生效（§1.4 单一真相源）。

    ``targets`` 以 JSON 字符串承载，避免 ms-swift 数据集层对嵌套结构做非预期展开。
    """
    extra_columns = _chat_template_column(config) if config is not None else {}
    rows: list[dict[str, Any]] = []
    for index, sample in enumerate(samples):
        if not sample.messages:
            raise ValueError(f"samples[{index}] has no messages; cannot build a GRPO prompt")
        if not sample.targets:
            raise ValueError(
                f"samples[{index}] has no targets; the graspo reward needs them "
                "(refusing to build a GRPO row that would train on a constant reward)"
            )
        rows.append(
            {
                MESSAGES_KEY: build_ms_swift_messages(sample.messages),
                GRASPO_TARGETS_COLUMN: json.dumps(sample.targets, ensure_ascii=False),
                # 工具调用样本必须带上 ``tools`` schema：奖励适配器在 tool-call
                # 路径上把它交给严格 XML 解析器做 required 参数校验（native 侧
                # 走 ``Sample.tools``，同源）。没有工具的样本不带该列，奖励
                # 适配器按 ``None`` 透明降级。
                **(
                    {GRASPO_TOOLS_COLUMN: json.dumps(sample.tools, ensure_ascii=False)}
                    if getattr(sample, "tools", None)
                    else {}
                ),
                **extra_columns,
            }
        )
    return rows


def _flatten_blocks(messages: list[Any]) -> list[dict[str, Any]]:
    """把样本的全部消息内容摊平成 ms-swift 的 content 块列表（顺序不变）。

    文本块原样保留为 ``{"type": "text", "text": ...}``；媒体块（``image`` /
    ``image_url`` / ``video`` / ``video_url`` / ``audio`` / ``audio_url``）**原样透传**——
    ms-swift 的 ``StdTemplateInputs.remove_messages_media`` 会扫描**每条**消息的
    content 列表、把它们抽成 ``images``/``videos``/``audios`` 数据集列，并在原位留下
    ``<image>`` / ``<video>`` / ``<audio>`` 占位。因此"多模态通路可用"不需要我们
    自己拼媒体列：把块放回 content 里交给上游即可（§1.2 不重复实现上游职责）。
    """
    blocks: list[dict[str, Any]] = []
    for message in messages:
        if not isinstance(message, dict):
            continue
        content = message.get("content")
        if isinstance(content, str):
            if content:
                blocks.append({"type": "text", "text": content})
        elif isinstance(content, list):
            for block in content:
                if isinstance(block, dict):
                    blocks.append(dict(block))
                elif isinstance(block, str):
                    blocks.append({"type": "text", "text": block})
        # content 为 None（ELAM 的工具调用轮）或其它类型：由 tool_calls 分支处理
        tool_calls = message.get("tool_calls")
        if isinstance(tool_calls, list) and tool_calls:
            rendered = _render_history([message], 1)
            if rendered:
                blocks.append({"type": "text", "text": rendered})
    return blocks


def build_cpt_rows(samples: list[Any], *, config: Any = None) -> list[dict[str, Any]]:
    """graspo 样本 → ms-swift **CPT（预训练）** 行。

    行形态取自 ms-swift 4.5.3 **官方预训练数据格式**
    （``docs/.../Customization/Custom-dataset.md`` §Pre-training）：

    .. code-block:: json

        {"messages": [{"role": "assistant", "content": "Pre-trained text goes here"}]}

    多模态样本额外把媒体块留在 content 里 ⇒ ms-swift 抽出 ``images`` / ``videos``
    列（"所有训练都支持多模态"这一范围约束因此在 CPT 通道上同样成立）。

    **★ 数据口径（不得误读）**：本函数把一条**指令形态的样本**（ELAM V5 / ARD 形态）
    摊平成**一条纯文本续训行**，目的是让 CPT 通道**能跑通一步**（判据 A1–A6），
    **不是**"CPT 语料已就绪"。真正的预训练语料是本期未决事项，见工作包 report §⑤
    （用户 2026-09-18 拍板：CPT 只判跑通、语料问题"回头再说"，**不得为此造语料**）。

    文本 = 全部消息内容（含 ``<image>`` 等占位）+ 首条 target 的答案文本。
    答案文本用 :func:`graspo.ripple.parsing.xml.build_sft_target_text` 生成——
    与 SFT 通道**同一个函数**（§1.4 单一真相源），不另写一套渲染。

    Raises:
        ValueError: 某条样本摊平后没有任何文本（拒绝产出空行去"训练"，§2.3）。
    """
    from graspo.ripple.parsing.xml import build_sft_target_text

    extra_columns = _chat_template_column(config) if config is not None else {}
    rows: list[dict[str, Any]] = []
    for index, sample in enumerate(samples):
        blocks = _flatten_blocks(sample.messages)
        if sample.targets:
            target_text = build_sft_target_text(sample.targets[0].get("output") or {})
            if target_text.strip():
                blocks.append({"type": "text", "text": target_text})
        if not any(
            block.get("type") == "text" and str(block.get("text") or "").strip() for block in blocks
        ):
            raise ValueError(
                f"samples[{index}] produced no plain text; refusing to write an empty "
                "CPT row (an all-empty corpus would train on padding only)"
            )
        rows.append({MESSAGES_KEY: [{"role": "assistant", "content": blocks}], **extra_columns})
    return rows


def build_opd_rows(samples: list[Any], *, config: Any = None) -> list[dict[str, Any]]:
    """graspo 样本 → ms-swift **OPD（on-policy 蒸馏）** 行 = **纯提示词**。

    OPD 与 GRPO 同属"completion 由学生现场采样"的一类，因此数据集只喂提示词；
    与 :func:`build_grpo_rows` 的唯一差别是**不带 ``targets`` 奖励列**——GKD 的
    监督信号来自教师模型的现场 logits，不是 graspo 奖励函数。带一个没人消费的
    ``targets`` 列等于制造"看起来生效、实际被忽略"的假配置（§1.4 / §7.2）。

    提示词同样过 :func:`build_ms_swift_messages`——相对媒体路径、``content=null``
    配对这两条 4.5.3 实测约束在此**复用同一次修复**（§1.4）。
    """
    extra_columns = _chat_template_column(config) if config is not None else {}
    rows: list[dict[str, Any]] = []
    for index, sample in enumerate(samples):
        if not sample.messages:
            raise ValueError(f"samples[{index}] has no messages; cannot build an OPD prompt")
        rows.append({MESSAGES_KEY: build_ms_swift_messages(sample.messages), **extra_columns})
    return rows


def write_rows(path: str | Path, rows: list[dict[str, Any]]) -> int:
    """把行写成 JSONL；返回写入条数。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return len(rows)


def prepare_ms_swift_dataset(
    config: Any,
    *,
    stage: Stage,
    work_dir: str | Path,
) -> str:
    """``config.data.train_path`` → ms-swift 数据集文件路径（本次运行的隔离目录内）。

    流程（四步，逐项显式）：读样本 → **把相对媒体路径解析为绝对路径** → 按 stage 建行
    → 写盘。第二步是 F-3 的修复落点：ms-swift 按 cwd 打开图像路径，只在写盘前解析
    成绝对路径才能与容器 cwd 解耦（模块 docstring「ms-swift 4.5.3 的消息契约」第 1 条）。

    Args:
        config: ``GraspoConfig`` 实例（只用 ``data.train_path``）。
        stage: ``"sft"`` / ``"rlhf"`` / ``"cpt"`` / ``"opd"``——决定行形态
            （CPT = 纯文本续训行；OPD = 纯提示词行；SFT = 提示词 + 目标答案；
            RLHF = 提示词 + 奖励用 ``targets`` 列）。
        work_dir: 本次运行的隔离目录（数据集是运行产物，不写回源码树，§8.5）。

    Returns:
        写好的 JSONL 路径。

    Raises:
        SystemExit: 输入文件不存在 / 无样本（边界校验，失败即退出，不静默用空数据训练）。
    """
    source = Path(str(config.data.train_path or ""))
    if not source.is_file():
        raise SystemExit(f"data.train_path does not exist: {source}")

    samples = load_graspo_samples(source)
    # 锚点与 native 后端逐字一致：相对媒体路径相对**数据文件所在目录**解析
    # （``flow/trainer/sft_trainer.py:82`` / ``flow/trainer/trainer.py:305``）。
    rewritten = resolve_sample_media_paths(samples, source.parent)
    logger.info(
        "msswift dataset: resolved %d relative media path(s) against %s", rewritten, source.parent
    )
    builders = {
        "sft": build_sft_rows,
        "rlhf": build_grpo_rows,
        "cpt": build_cpt_rows,
        "opd": build_opd_rows,
    }
    builder = builders.get(stage)
    if builder is None:
        raise ValueError(f"stage must be one of {sorted(builders)}, got {stage!r}")
    rows = builder(samples, config=config)
    target = Path(work_dir) / f"ms_swift_{stage}.jsonl"
    write_rows(target, rows)
    return str(target)
