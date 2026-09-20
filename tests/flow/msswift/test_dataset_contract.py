"""L1：``msswift`` 数据集行的 **ms-swift 消息契约**（F-3 修复的锁定测试）。

**覆盖什么**

``flow/msswift/dataset.py`` 把 graspo 样本转成 ms-swift 数据集行。ms-swift 4.5.3 对它
能编码的消息序列有两条硬约束，ELAM V5 形态会同时踩中（实测，2026-09-18）：

1. 相对图像路径按**进程 cwd** 打开（``Template._preprocess_inputs`` → ``_load_image``
   → ``open(path)``）⇒ 必须写成绝对路径；
2. 剥掉 system 后必须是严格的 ``(query, response)`` 交替对、且 ``content is not None``
   （``Template._swift_encode:1393-1398``）⇒ 中间 assistant 轮的 ``content=null``
   （工具调用）与 ``role=tool`` 轮都必须被规整掉。

两条违约在 ms-swift 里都被包成一句与真因无关的
``ValueError: Failed to retrieve the dataset``（``swift/dataset/utils.py:108``），
因此这些用例断言的是**行本身的形态**（含图像参数与绝对路径），而不是训练是否跑起来。

**断言的是纯计算**：不加载模型、不启动训练、不 import torch/transformers。
本机（开发机无 torch）即可跑——``graspo.core.schema`` 会经 ``graspo.ripple`` 拉
``torch``，因此本文件在 torch 不可用时按 ``tests/conftest.py`` 的同一条思路给
``graspo.ripple.reward.reward`` 装一个最小替身（只补 ``REWARD_REGISTRY``，
其余照旧）；容器内 torch 存在时该替身不生效，行为与既有测试一致。
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
import types
from pathlib import Path

import pytest

pytestmark = pytest.mark.filterwarnings("ignore::UserWarning")


def _install_no_torch_shim() -> None:
    """仅在本机无 torch 时安装：让 ``graspo.flow.msswift.dataset`` 可被导入。

    本机导入被三个重型 ``__init__.py`` 挡住（``graspo.ripple`` → ``algorithm`` →
    ``import torch``；``graspo.flow`` → ``runtime`` → ``ripple.buffer`` → 同上）。
    因此把它们注册成**命名空间包**（``__path__`` 指向真实目录、不执行 ``__init__``），
    并只补 ``graspo.ripple.reward.reward`` 一个模块（``core.schema`` 的 reward 校验
    要用它的 ``REWARD_REGISTRY``）。容器内 torch 可用时**完全不动**。
    """
    if importlib.util.find_spec("torch") is not None:
        return
    src = Path(__file__).resolve().parents[3] / "src"
    for name, relative in (
        ("graspo", "graspo"),
        ("graspo.core", "graspo/core"),
        ("graspo.flow", "graspo/flow"),
        ("graspo.flow.msswift", "graspo/flow/msswift"),
        ("graspo.ripple", "graspo/ripple"),
        ("graspo.ripple.reward", "graspo/ripple/reward"),
        ("graspo.ripple.multimodal", "graspo/ripple/multimodal"),
        ("graspo.ripple.parsing", "graspo/ripple/parsing"),
    ):
        if name in sys.modules:
            continue
        module = types.ModuleType(name)
        module.__path__ = [str(src / relative)]
        module.__package__ = name
        sys.modules[name] = module
    if "graspo.ripple.reward.reward" in sys.modules:
        return
    module = types.ModuleType("graspo.ripple.reward.reward")

    class GraspoReward:  # pragma: no cover - 仅占位，测试不跑奖励
        """奖励类替身（本文件不涉及奖励语义）。"""

    module.GraspoReward = GraspoReward
    module.REWARD_REGISTRY = {"graspo": GraspoReward}
    sys.modules["graspo.ripple.reward.reward"] = module


_install_no_torch_shim()

from graspo.flow.msswift.dataset import (  # noqa: E402
    build_grpo_rows,
    build_ms_swift_messages,
    build_sft_rows,
    resolve_sample_media_paths,
    write_rows,
)


class _Sample:
    """``core.schema.Sample`` 的最小替身（本文件只读 ``messages``/``targets``）。"""

    def __init__(
        self,
        messages: list[dict],
        targets: list[dict] | None = None,
        tools: list[dict] | None = None,
    ) -> None:
        self.messages = messages
        self.targets = targets if targets is not None else [_TOOL_CALL_TARGET]
        self.tools = tools


#: 与 ELAM V5 一致的 tool_calls 目标（``targets[0].output.tool_calls``）。
_TOOL_CALL_TARGET = {
    "id": "primary",
    "output": {"tool_calls": [{"name": "extend_arm", "arguments": {"action_type": "伸长手臂"}}]},
}


def _elam_single_turn(*, image_path: str = "../images/a.jpg", n_images: int = 2) -> _Sample:
    """ELAM 形态的单轮样本：system + user（图像块 + 文本）。"""
    content: list[dict] = [
        {"type": "image", "image": image_path.replace("a.jpg", f"l{i}.jpg")}
        for i in range(n_images)
    ]
    content.append({"type": "text", "text": "点击提交按钮"})
    return _Sample(
        [
            {"role": "system", "content": [{"type": "text", "text": "你是机器人"}]},
            {"role": "user", "content": content},
        ]
    )


def _elam_multi_turn() -> _Sample:
    """ELAM 形态的多轮样本：中间 assistant 的 ``content`` 是 **null**（工具调用）。

    实测分布（ELAM V5 ``data/train.jsonl``）：4014 行 2 条消息 / 1902 行 5 条 /
    462 行 8 条；多轮样本的中间 assistant 一律 ``content=null`` + ``tool_calls``。
    """
    return _Sample(
        [
            {"role": "system", "content": [{"type": "text", "text": "你是机器人"}]},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": "../images/step0_l.jpg"},
                    {"type": "image", "image": "../images/step0_r.jpg"},
                    {"type": "text", "text": "桌面场景"},
                ],
            },
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "extend_arm",
                            "arguments": {"action_type": "伸长手臂"},
                        },
                    }
                ],
            },
            {"role": "tool", "content": "OK", "tool_call_id": "call_1"},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": "../images/step1_l.jpg"},
                    {"type": "image", "image": "../images/step1_r.jpg"},
                    {"type": "text", "text": "继续"},
                ],
            },
        ]
    )


def _image_paths(row: dict) -> list[str]:
    return [
        block["image"]
        for message in row["messages"]
        if isinstance(message.get("content"), list)
        for block in message["content"]
        if isinstance(block, dict) and block.get("type") == "image"
    ]


def _images_from_messages(messages: list[dict]) -> list[str]:
    return [
        block["image"]
        for message in messages
        if isinstance(message.get("content"), list)
        for block in message["content"]
        if isinstance(block, dict) and block.get("type") == "image"
    ]


# ── 1. 相对图像路径必须被解析成绝对路径（F-3 真因 1）─────────────────────────


def test_a_relative_image_paths_are_resolved_against_data_dir(tmp_path: Path):
    """改写锚点 = 数据文件父目录；``../images`` 因此指向数据根的 ``images/``。

    **不能**调任何会规范化 ``..`` 的函数——实测（本机 Python 3.13 与目标容器
    Python 3.12 均复现）：``os.path.abspath`` / ``Path.resolve`` / ``Path.absolute``
    都把 ``..`` 折叠掉，``/subsets/../images/x.jpg`` 被折叠成 ``/images/x.jpg``，
    锚点当场失效（本修复第一版踩的坑）。因此断言"绝对 + 原样保留 ``..``"。
    """
    data_dir = tmp_path / "elam" / "data"
    data_dir.mkdir(parents=True)
    sample = _elam_single_turn()

    rewritten = resolve_sample_media_paths([sample], data_dir)

    assert rewritten == 2
    paths = _images_from_messages(sample.messages)
    expected = [
        os.path.join(str(data_dir), "../images/l0.jpg"),
        os.path.join(str(data_dir), "../images/l1.jpg"),
    ]
    assert paths == expected
    assert all(Path(path).is_absolute() for path in paths)
    # ``..`` 必须原样保留（交给操作系统解析），且语义上仍落到数据根的 images/
    assert ".." in paths[0]
    assert Path(paths[0]).resolve() == (data_dir.parent / "images" / "l0.jpg").resolve()


def test_b_absolute_and_remote_paths_are_left_untouched(tmp_path: Path):
    """已经绝对的路径与 http(s)/data: URL 不改写（与 native 侧规则逐字一致）。"""
    data_dir = tmp_path
    sample = _Sample(
        [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": "/already/abs.jpg"},
                    {"type": "image", "image": "https://example.com/x.jpg"},
                ],
            }
        ]
    )
    assert resolve_sample_media_paths([sample], data_dir) == 0
    assert _images_from_messages(sample.messages) == [
        "/already/abs.jpg",
        "https://example.com/x.jpg",
    ]


def test_c_no_data_dir_means_no_rewrite():
    """``data_dir=None`` 时不改任何路径（显式空值语义，§2.2）。"""
    sample = _elam_single_turn()
    assert resolve_sample_media_paths([sample], None) == 0
    assert all(not Path(p).is_absolute() for p in _images_from_messages(sample.messages))


# ── 2. SFT 行必须满足 ms-swift 配对契约、且带上图像 ───────────────────────────


def test_sft_row_keeps_images_and_matches_swift_pair_contract(tmp_path: Path):
    """修复后：``[system, user, assistant]``，图像块保留，assistant 是纯文本 str。"""
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True)
    sample = _elam_single_turn()
    resolve_sample_media_paths([sample], data_dir)

    row = build_sft_rows([sample])[0]

    assert [message["role"] for message in row["messages"]] == ["system", "user", "assistant"]
    images = _image_paths(row)
    assert len(images) == 2, "图像参数必须真的出现在产出行里（不是只有 <image> 文本）"
    assert all(Path(path).is_absolute() for path in images)
    # 内容顺序约定：两张图在前、文本在后（占位符顺序 = 图像顺序）
    assert [block["type"] for block in row["messages"][1]["content"]] == [
        "image",
        "image",
        "text",
    ]
    assistant = row["messages"][-1]
    assert isinstance(assistant["content"], str) and assistant["content"].strip()
    assert "<function=extend_arm>" in assistant["content"]


def test_sft_multi_turn_row_has_no_null_content_and_valid_pairs():
    """多轮样本：中间 assistant 的 ``content=None`` 与 ``tool`` 轮不得泄漏进行里。"""
    row = build_sft_rows([_elam_multi_turn()])[0]
    messages = row["messages"]

    assert all(message.get("content") is not None for message in messages)
    assert "tool" not in [message["role"] for message in messages]
    # ms-swift Template._swift_encode:1393-1398 的配对断言（剥 system 后）
    body = [message for message in messages if message["role"] != "system"]
    for query, response in zip(body[::2], body[1::2], strict=True):
        assert query["role"] in {"user", "tool"}
        assert response["role"] == "assistant"
    assert body[-1]["role"] == "assistant"


def test_sft_multi_turn_keeps_only_final_turn_images():
    """只保留末轮图像（历史轮的图不重复输入——否则显存随轮数线性膨胀）。"""
    row = build_sft_rows([_elam_multi_turn()])[0]
    images = _image_paths(row)
    assert len(images) == 2
    assert all("step1" in path for path in images)


def test_sft_multi_turn_history_is_rendered_not_dropped():
    """历史不静默丢弃：工具调用与工具返回以文本形式进入末轮 user。"""
    row = build_sft_rows([_elam_multi_turn()])[0]
    user_content = row["messages"][1]["content"]
    text = "".join(block["text"] for block in user_content if block.get("type") == "text")
    assert "extend_arm" in text, "历史里的工具调用必须可见"
    assert "伸长手臂" in text, "工具调用参数必须可见"
    assert "OK" in text, "工具返回必须可见"
    assert "继续" in text, "当前轮指令必须保留"
    # 历史文本插在图像块之前，当前指令文本留在图像之后：
    # 占位符顺序仍与图像顺序一致（图像块位置不变）
    assert [block["type"] for block in user_content] == ["text", "image", "image", "text"]


def test_sft_assistant_content_is_plain_string_not_block_list():
    """回归护栏：assistant 侧的 content 必须是 str。

    若写成 ``[{"type": "text", ...}]``，``datasets`` 的 Arrow 推断路径与 ms-swift
    ``Template._swift_encode`` 对 assistant 的处理都会偏离 Qwen 原生 chat template
    （``task-lora-multicard`` 复现踩坑 ③）。
    """
    row = build_sft_rows([_elam_single_turn()])[0]
    for message in row["messages"]:
        if message["role"] == "assistant":
            assert isinstance(message["content"], str)


# ── 3. GRPO 行同样走同一套规整（修一次、两处生效）────────────────────────────


def test_grpo_row_is_flat_and_carries_targets_column(tmp_path: Path):
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True)
    sample = _elam_multi_turn()
    resolve_sample_media_paths([sample], data_dir)

    row = build_grpo_rows([sample])[0]

    assert [message["role"] for message in row["messages"]] == ["system", "user"]
    assert len(_image_paths(row)) == 2
    assert json.loads(row["targets"]) == sample.targets


# ── 4. 边界：没有 user 轮 / 空目标必须拒绝（§2.3 边界校验）────────────────────


def test_messages_without_user_turn_are_rejected():
    with pytest.raises(ValueError, match="no 'user' turn"):
        build_ms_swift_messages([{"role": "system", "content": "只有系统提示词"}])


def test_sft_row_without_targets_is_rejected():
    sample = _Sample([{"role": "user", "content": "问题"}], targets=[])
    with pytest.raises(ValueError, match="has no targets"):
        build_sft_rows([sample])


def test_grpo_row_without_targets_is_rejected():
    sample = _Sample([{"role": "user", "content": "问题"}], targets=[])
    with pytest.raises(ValueError, match="has no targets"):
        build_grpo_rows([sample])


# ── 5. 行落盘形态（ms-swift 读的就是这个 JSONL）──────────────────────────────


def test_written_rows_are_jsonl_with_absolute_images(tmp_path: Path):
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True)
    sample = _elam_single_turn()
    resolve_sample_media_paths([sample], data_dir)
    rows = build_sft_rows([sample])

    path = tmp_path / "out" / "ms_swift_sft.jsonl"
    assert write_rows(path, rows) == 1

    written = json.loads(path.read_text(encoding="utf-8").strip())
    images = _image_paths(written)
    assert len(images) == 2
    assert all(Path(image).is_absolute() for image in images)
