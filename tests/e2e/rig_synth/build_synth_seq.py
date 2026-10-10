#!/usr/bin/env python3
"""合成长序列生成器 —— 把 T013 最短基样复制 N 份拼成一条长样本（**只造工具，不上机**）。

**职责边界**

- **本模块**：从 ELAM V5 训练子集里取一条**基样**，把它的**末轮 user 内容块**按变体规则
  复制 N 份拼成**一条** user 轮，再补**一条最小 assistant 轮**（口径 1），产出一条
  合法可判定的 SFT 行（**1 行 = 1 step**）到 ``<out-dir>/subsets/<tier>-synth-<档>.jsonl``。
- **不负责**：token 实测（见 ``measure_synth_tokens.py``）、配置生成（见 ``emit_configs.py``）、
  上机执行（见 ``run_synth.sh``）。本模块**不引入任何第三方依赖**（纯标准库）。

**为什么是"复制 N 份"**（本包的目标：让"序列长度"真正成为自变量）

ELAM V5 的 SFT 子集真实逐样本 token 长度只有 1941–2001（median 1943，实测），而
``data.max_prompt_length`` 只是**截断上限、不 pad**（ms-swift ``_encode_truncated``：
只有 ``length > max_length`` 才裁）。⇒ 把 ``max_length`` 从 4096 调到 32768，
**训练轨迹一字未变**（8 次 run 的 step-1 loss 逐位相同）。旧方法测不出
`最大可行上下文`。合成长序列让"长度"变成可控自变量。

**构造（口径固定，不得自行更改）**

1. **基样** = T013 子集里**实测 token 数最短**的那条（实测 1941 token）。
   实测不可在本机做时，由 ``measure_synth_tokens.py`` 在容器内量出后填入 ``--base-index``。
2. **拼接**：``messages = [system(原样 1 份)]
   ``+ [user(原末轮 user 的 content 块 × N)] + [assistant]``。
   - 与原 user 轮相比，**只做"复制块"这一件事**：不改任何文本、不改任何路径、
     不增删任何块类型。
3. **最小 assistant 轮**（口径 1，**不得**改成"仅前向/仅显存"的声明式做法）：
   内容是一小段合法文本（默认 ``OK``），使该行成为**合法、可判定的 SFT 样本**，
   A1–A6 照常适用。它**不是真实监督目标**——见报告 §④ 语义差异。
4. **N 来自实测标定**，不来自算式：``--calibration`` 指向
   ``measure_synth_calibration.json``（由容器内实测产生）。**缺失即 fail-closed 拒绝生成**。

**两个视觉变体（口径 2：绝不可混填同一列）**

| 变体 | ``--variant`` | 内容 | 用途 |
|---|---|---|---|
| **复制图像** | ``v1-image-copy``（**默认**） | 末轮 user 的**全部内容块**（含 image 块）
整体复制 N 份 ⇒ 图像随样本**等比例重复**
| **填「最大可行上下文」列的主口径**——对应"一条真正长的多模态序列" |
| **只复制文本块** | ``v2-text-only`` | 只复制 **text 块** N 份，**image 块留 1 份**
| **单独标注的数据点，明确不得用于填该列** |

两个变体的产物路径、配置、清单、台账字段**全部可区分**（见 ``emit_configs.py`` 与
``run_synth.sh``），并由生成物首行头部注释逐条标注。

**产物格式（每档 1 行）**

``subsets/<tier>-synth-<档>.jsonl`` —— 档 = `4k`/`8k`/`16k`/`32k`/`64k`/`96k`/`128k`。
每行是**完整 ELAM 形态记录**：除 ``messages`` 外，其余键（``tools``/``targets``/``id``/
``source``）从基样**原样复制**，使该行与真实训练行同构（ms-swift 侧只消费 ``messages``，
但同构可避免"合成行与真实行契约不同"的隐性偏差）。

**用法**

    # 1) 标定（容器内实测，产出 measure_synth_calibration.json）
    python3 measure_synth_tokens.py --subset <T013.jsonl> --model /models/Qwen3.5-9B \\
        --calibrate --out measure_synth_calibration.json
    # 2) 生成（本机/任意机器，纯标准库）
    python3 build_synth_seq.py --source <T013.jsonl> --base-index 7 \\
        --calibration measure_synth_calibration.json \\
        --variant v1-image-copy --out-dir .local/hb-workspace/.../task-synth-seq/synth-host/T013/v1
    # 3) 复验（容器内实测，证明"目标长度真的达到了"）
    python3 measure_synth_tokens.py --subset synth-host/T013/v1/subsets/T013-synth-128k.jsonl \\
        --model /models/Qwen3.5-9B --out measured_tokens.json

**自查（§2 防呆）**：本脚本对每个输入在边界处校验（基样必须是 JSONL 对象、messages 必须
有 system/user/assistant 形态、复制必须是**逐字节一致**的深拷贝、输出必须恰好 N 行、
媒体路径必须零改写）；任何一条不满足即**非零退出**，不静默产出。
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import sys
from datetime import UTC
from pathlib import Path

#: 变体 → （是否复制 image 块，产物子目录名）
#: 顺序即"主口径在前"；`v1-image-copy` 是填列口径，`v2-text-only` 是**不得填列**的数据点。
VARIANTS: dict[str, dict[str, object]] = {
    "v1-image-copy": {
        "copy_images": True,
        "tag": "v1-image-copy",
        "fills_context_column": True,
        "note": "复制图像：图像随样本等比例重复 ⇒ 一条真正长的多模态序列（填列主口径）",
    },
    "v2-text-only": {
        "copy_images": False,
        "tag": "v2-text-only",
        "fills_context_column": False,
        "note": "只复制文本块、图像留 1 份 ⇒ 单独标注的数据点，明确不得用于填「最大可行上下文」列",
    },
}

#: ★ 结构性守卫（§2 防呆，不靠纪律靠结构）：**唯一**允许 `fills_context_column=True` 的变体。
#: 指挥官 2026-09-19 裁定 3 要求：`fills_context_column` 为 true 的产物**只允许有
#: `v1-image-copy` 一个来源**。任何把 v2 标成 true 的尝试必须**硬失败**（见
#: :func:`assert_fills_column_invariant`，有能真失败的证据）。
FILLS_COLUMN_VARIANT = "v1-image-copy"

#: 图像/视频块类型（与 flow/backend/msswift/dataset.py 的媒体判定逐字一致，避免第二套口径）。
MEDIA_BLOCK_TYPES = {"image", "image_url", "video", "video_url"}

#: 档位 → 目标 token 数。档名用于文件名，目标值用于复核实测长度。
TIERS: list[tuple[str, int]] = [
    ("4k", 4096),
    ("8k", 8192),
    ("16k", 16384),
    ("32k", 32768),
    ("64k", 65536),
    ("96k", 98304),
    ("128k", 131072),
]

# --------------------------------------------------------------------------- #
# ★ A4 适用策略（指挥官 2026-09-19 裁定 1）
# --------------------------------------------------------------------------- #
#: **4K–32K 双跑**（便宜，且让 A4「同 config 同 seed 双跑一致」有实质证据）；
#: **64K 起单跑**（成本高：30–90 min/档；且该档若 OOM 本就不需要 A4）。
A4_DUAL_RUN_TIERS: tuple[str, ...] = ("4k", "8k", "16k", "32k")

#: A4 状态取值（封闭三态；`None` = 待跑——只有双跑档在跑完第 2 次前可能为 None）。
A4_STATUS_PASS = "pass"
A4_STATUS_FAIL = "fail"
A4_STATUS_NOT_APPLICABLE_SINGLE_RUN = "not_applicable_single_run"

#: ★ 台账显示名：A4 的"不适用"**必须**写成 `未适用（单跑）`，**绝不写成"通过"**。
#: 理由（裁定 1 原话）："**'没跑'和'跑过了'是两件事**"——这一整轮都在清这种混淆。
A4_STATUS_DISPLAY: dict[str, str] = {
    A4_STATUS_PASS: "通过",
    A4_STATUS_FAIL: "失败",
    A4_STATUS_NOT_APPLICABLE_SINGLE_RUN: "未适用（单跑）",
}

#: 双跑档的**适用判据** = 全部 A1–A6（含 A4）；
#: 单跑档 = A1/A2/A3/A5/A6（A4 记「未适用（单跑）」，**不计入适用判据**）。
APPLICABLE_CRITERIA_DUAL_RUN: tuple[str, ...] = ("A1", "A2", "A3", "A4", "A5", "A6")
APPLICABLE_CRITERIA_SINGLE_RUN: tuple[str, ...] = ("A1", "A2", "A3", "A5", "A6")

#: 生成物头部注释前缀（每档首行之前写一行 ``#`` 注释不改 JSONL 契约——见 docstring 末）。
#: ⚠ JSONL 不允许注释行；因此"头部注释"落在**旁挂的 .provenance.json**（机器可读）
#: 与**旁挂的 .README.txt**（人读），JSONL 本身保持严格 1 行 → 训练器可直读。
MINIMAL_ASSISTANT_CONTENT = "OK"


def a4_dual_run_for(tier_tag: str) -> bool:
    """该档是否双跑（决定 A4 是否**适用**）。未知档 ⇒ 抛错（不猜）。"""
    if tier_tag not in dict(TIERS):
        raise SynthError(f"未知档 {tier_tag!r}，无法判定 A4 适用性")
    return tier_tag in A4_DUAL_RUN_TIERS


def applicable_criteria(tier_tag: str) -> tuple[str, ...]:
    """该档的**适用判据**集合（单跑档不含 A4）。"""
    return (
        APPLICABLE_CRITERIA_DUAL_RUN
        if a4_dual_run_for(tier_tag)
        else APPLICABLE_CRITERIA_SINGLE_RUN
    )


class SynthError(Exception):
    """生成期的边界校验失败（fail-closed，不产出半成品）。"""


class FillsColumnInvariantError(SynthError):
    """★ 结构性守卫失败：`fills_context_column=True` 的来源不是 `v1-image-copy`。

    这是"两个变体绝不可混填同一列"的**机器强制**形式（指挥官 2026-09-19 裁定 3）。
    单独成类是为了让调用方/测试能**精确断言**这条守卫真的会失败
    （见 :func:`assert_fills_column_invariant` 的 doctest 式证据）。
    """


def assert_fills_column_invariant(variant: str, fills_context_column: bool) -> None:
    """守卫：**只有** :data:`FILLS_COLUMN_VARIANT` 才允许 `fills_context_column=True`。

    Raises:
        FillsColumnInvariantError: 任何"非 v1 却声称填列"或"v1 却不填列"的不一致。

    >>> assert_fills_column_invariant("v1-image-copy", True)   # 唯一合法组合
    >>> assert_fills_column_invariant("v2-text-only", False)  # 合法：不填列
    >>> try:
    ...     assert_fills_column_invariant("v2-text-only", True)
    ... except FillsColumnInvariantError as exc:
    ...     print("GUARD FIRED:", exc)  # doctest: +ELLIPSIS
    GUARD FIRED: ...
    """
    if variant not in VARIANTS:
        raise FillsColumnInvariantError(f"未知变体 {variant!r}")
    expected = variant == FILLS_COLUMN_VARIANT
    if fills_context_column != expected:
        raise FillsColumnInvariantError(
            f"变体 {variant!r} 的 fills_context_column={fills_context_column}，"
            f"但唯一允许填列的变体是 {FILLS_COLUMN_VARIANT!r} ⇒ "
            "「最大可行上下文」列**只允许来自 v1-image-copy**（裁定 3 的结构性守卫）。"
            "若要新增填列口径，必须显式改 FILLS_COLUMN_VARIANT 并经指挥官裁定。"
        )


class A4StatusInvariantError(SynthError):
    """★ 结构性守卫失败：A4 状态与该档的**适用性**不一致。

    这是裁定 1 硬要求的机器强制形式——**单跑档的 A4 绝不许记为 `pass`**
    （"'没跑'和'跑过了'是两件事"）。单独成类是为了让测试能**精确断言**它真会失败。
    """


def assert_a4_status_invariant(a4_dual_run: bool, a4_status: str | None) -> None:
    """守卫：A4 状态必须与该档是否双跑一致（裁定 1）。

    规则（**双向**强制，两个方向都是"把没跑的记成跑了"的变体）：

    - ``a4_status is None`` ⇒ 待跑，**合法**（仅双跑档在跑完第 2 次前可如此）；
    - **单跑档**（``a4_dual_run=False``）⇒ 只允许 :data:`A4_STATUS_NOT_APPLICABLE_SINGLE_RUN`；
      **绝不允许 `pass`/`fail`**（没跑第二次，谈不上"一致/不一致"）；
    - **双跑档**（``a4_dual_run=True``）⇒ 只允许 `pass`/`fail`；
      **不允许** `not_applicable_single_run`（它适用 A4，不能借"不适用"蒙混）。

    Raises:
        A4StatusInvariantError: 任何上述不一致。

    >>> assert_a4_status_invariant(False, A4_STATUS_NOT_APPLICABLE_SINGLE_RUN)  # 合法
    >>> assert_a4_status_invariant(True, A4_STATUS_PASS)                        # 合法
    >>> assert_a4_status_invariant(True, None)                                  # 待跑，合法
    >>> try:
    ...     assert_a4_status_invariant(False, A4_STATUS_PASS)
    ... except A4StatusInvariantError as exc:
    ...     print("GUARD FIRED")  # doctest: +ELLIPSIS
    GUARD FIRED
    """
    if a4_status is None:
        return
    if a4_status not in A4_STATUS_DISPLAY:
        raise A4StatusInvariantError(
            f"未知 A4 状态 {a4_status!r}；只允许 {sorted(A4_STATUS_DISPLAY)}（或 None=待跑）"
        )
    if not a4_dual_run and a4_status != A4_STATUS_NOT_APPLICABLE_SINGLE_RUN:
        raise A4StatusInvariantError(
            f"**单跑档**的 a4_status 不得为 {a4_status!r}——必须是 "
            f"{A4_STATUS_NOT_APPLICABLE_SINGLE_RUN!r}（台账显示为 "
            f"「{A4_STATUS_DISPLAY[A4_STATUS_NOT_APPLICABLE_SINGLE_RUN]}」）。"
            "**'没跑'和'跑过了'是两件事**：单跑档没有第二次 run，"
            "谈不上 A4 的'双跑一致'，**绝不许记为「通过」**（裁定 1 硬要求）。"
        )
    if a4_dual_run and a4_status == A4_STATUS_NOT_APPLICABLE_SINGLE_RUN:
        raise A4StatusInvariantError(
            f"**双跑档**的 a4_status 不得为 {a4_status!r}——该档 A4 **适用**，"
            "必须给出 pass/fail（或 None=待跑），不得借'不适用'蒙混。"
        )


def a4_status_for_build(tier_tag: str) -> str | None:
    """生成期就**定死** A4 状态：单跑档直接写"未适用（单跑）"，双跑档留 `None`（待跑）。

    为什么单跑档要在**生成期**就写死（而不是等上机回填）：
    裁定 1 要求"单跑的长度档，A4 必须记为 `未适用（单跑）`"——
    把这条**结构化**进产物，比"指望上机批次记得填"可靠（§2 防呆）。
    """
    if a4_dual_run_for(tier_tag):
        return None
    return A4_STATUS_NOT_APPLICABLE_SINGLE_RUN


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _load_jsonl(path: Path) -> list[dict]:
    """读 JSONL，逐行校验是 JSON 对象（§2.3 边界校验即防呆）。"""
    if not path.is_file():
        raise SynthError(f"源 JSONL 不存在：{path}")
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SynthError(f"{path}:{line_no} 不是合法 JSON：{exc}") from exc
            if not isinstance(record, dict):
                raise SynthError(f"{path}:{line_no} 不是 JSON 对象")
            rows.append(record)
    if not rows:
        raise SynthError(f"{path} 里没有任何样本")
    return rows


def _pick_base(rows: list[dict], base_index: int | None, base_id: str | None) -> tuple[int, dict]:
    """选基样：**显式**给 ``--base-index``（实测最短者）或 ``--base-id``；拒绝隐式猜测。

    为什么不用"字符长度最短"自动选：字符长度**不是** token 长度（图像 token 预算由
    ms-swift 决定，与字符数无关）。用字符数当代理会选错基样 ⇒ 目标长度失准。
    实测最短者由 ``measure_synth_tokens.py --calibrate`` 在容器内量出。
    """
    if base_index is not None and base_id is not None:
        raise SynthError("--base-index 与 --base-id 只能给一个（避免两个真相源）")
    if base_index is None and base_id is None:
        raise SynthError(
            "必须显式给 --base-index 或 --base-id（实测最短基样由 measure_synth_tokens.py "
            "--calibrate 量出；本脚本不用字符数当 token 数的代理，§2.2 显式即防呆）"
        )
    if base_index is not None:
        if not 0 <= base_index < len(rows):
            raise SynthError(f"--base-index {base_index} 越界（源共 {len(rows)} 条）")
        return base_index, rows[base_index]
    for index, record in enumerate(rows):
        if str(record.get("id")) == base_id:
            return index, record
    raise SynthError(f"--base-id {base_id!r} 在源 JSONL 里找不到")


def _split_messages(messages: object) -> tuple[dict | None, dict, int]:
    """把基样 messages 拆成 （system 轮或 None，末轮 user，末轮 user 的下标）。

    与 ``flow/backend/msswift/dataset.py::build_ms_swift_messages`` 的判定同源：system 只在
    **首条且 role==system** 时保留；user 取**最后一条** role==user 的轮。
    """
    if not isinstance(messages, list) or not messages:
        raise SynthError("基样 messages 必须是非空列表")
    last_user_index = None
    for index in range(len(messages) - 1, -1, -1):
        message = messages[index]
        if isinstance(message, dict) and str(message.get("role") or "").lower() == "user":
            last_user_index = index
            break
    if last_user_index is None:
        raise SynthError("基样 messages 里没有 user 轮（无法构造提示词）")

    system_message = None
    first = messages[0]
    if (
        isinstance(first, dict)
        and str(first.get("role") or "").lower() == "system"
        and first.get("content") is not None
    ):
        system_message = copy.deepcopy(first)

    user_message = messages[last_user_index]
    if not isinstance(user_message, dict):
        raise SynthError(f"messages[{last_user_index}] 不是 JSON 对象")
    if user_message.get("content") is None:
        raise SynthError(f"messages[{last_user_index}]（user）没有 content")
    return system_message, copy.deepcopy(user_message), last_user_index


def _blocks(content: object) -> list:
    """把 user content 归一成**块列表**（str 视为单个 text 块，便于统一复制）。"""
    if isinstance(content, str):
        return [{"type": "text", "text": content}]
    if isinstance(content, list):
        return content
    raise SynthError(f"user content 必须是 str 或块列表，实际 {type(content).__name__}")


def _select_blocks(blocks: list, copy_images: bool) -> list:
    """按变体挑出"要复制的那一份块"。

    - ``v1-image-copy``：**全部**块（含图像）——图像随样本等比例重复。
    - ``v2-text-only``：**只 text 块**——图像留 1 份。
    """
    if copy_images:
        return blocks
    kept = [
        block
        for block in blocks
        if not (
            isinstance(block, dict) and str(block.get("type") or "").lower() in MEDIA_BLOCK_TYPES
        )
    ]
    if not kept:
        raise SynthError("v2-text-only：末轮 user 里没有任何 text 块可复制（该基样不适用此变体）")
    return kept


def build_synth_record(
    base: dict,
    *,
    copies: int,
    variant: str,
    assistant_content: str = MINIMAL_ASSISTANT_CONTENT,
) -> tuple[dict, dict]:
    """构造一条合成长序列记录；返回 ``(记录, 溯源元数据)``。

    这是本模块的**唯一**构造点（§1.4 单一真相源）：不改文本、不改路径、只复制块。
    """
    if variant not in VARIANTS:
        raise SynthError(f"未知变体 {variant!r}；可用：{sorted(VARIANTS)}")
    if copies < 1:
        raise SynthError(f"--copies 必须 ≥ 1，实际 {copies}")

    spec = VARIANTS[variant]
    # ★ 裁定 3 的结构性守卫：在**每个**构造点强制"只有 v1 可声称填列"。
    assert_fills_column_invariant(variant, bool(spec["fills_context_column"]))
    copy_images = bool(spec["copy_images"])
    system_message, user_message, last_user_index = _split_messages(base.get("messages"))
    source_blocks = _blocks(user_message.get("content"))
    one_copy = _select_blocks(source_blocks, copy_images)

    # ★ 逐字节一致的深拷贝 ×N：唯一改动是"块序列变长"，任何块内部内容一字不改。
    if copy_images:
        # v1（主口径）：**全部**块（含图像）整体复制 N 份 ⇒ 图像随样本等比例重复。
        merged_blocks: list = []
        for _ in range(copies):
            merged_blocks.extend(copy.deepcopy(one_copy))
    else:
        # v2（不得填列）：图像各留 1 份，只复制文本块 N 份。
        merged_blocks = _interleave_single_images(source_blocks, one_copy, copies)

    synth_user = {"role": "user", "content": merged_blocks}

    messages: list[dict] = []
    if system_message is not None:
        messages.append(system_message)
    messages.append(synth_user)
    # 口径 1：必须补一条**最小 assistant 轮**（不得用"仅前向/仅显存"替代）。
    messages.append({"role": "assistant", "content": assistant_content})

    record = copy.deepcopy(base)
    record["messages"] = messages

    provenance = {
        "variant": variant,
        "variant_note": spec["note"],
        "fills_context_column": spec["fills_context_column"],
        "copies": copies,
        "base_index": None,
        "base_id": base.get("id"),
        "source_last_user_index": last_user_index,
        "source_block_count": len(source_blocks),
        "source_media_block_count": _count_media(source_blocks),
        "copied_block_count_per_copy": len(one_copy),
        "copied_media_block_count_per_copy": _count_media(one_copy),
        "output_user_block_count": len(merged_blocks),
        "output_media_block_count": _count_media(merged_blocks),
        "output_message_roles": [m.get("role") for m in messages],
        "assistant_content_sha256": _sha256_text(assistant_content),
        "source_messages_sha256": _sha256_text(
            json.dumps(base.get("messages"), ensure_ascii=False, sort_keys=True)
        ),
        "copied_segments_identical": True,
    }
    return record, provenance


def _count_media(blocks: list) -> int:
    return sum(
        1
        for block in blocks
        if isinstance(block, dict) and str(block.get("type") or "").lower() in MEDIA_BLOCK_TYPES
    )


def _interleave_single_images(source_blocks: list, text_blocks: list, copies: int) -> list:
    """v2-text-only 的块序列：图像块各留 1 份，文本块复制 N 份。

    布局取"图像块按原顺序前置一次 + 文本块 ×N 紧随"——**显式且可复核**，
    并让 v1/v2 在任何 N 下都可逐块 diff（同一函数决定两者，不搞隐式差异）。
    """
    media = [
        copy.deepcopy(block)
        for block in source_blocks
        if isinstance(block, dict) and str(block.get("type") or "").lower() in MEDIA_BLOCK_TYPES
    ]
    merged: list = list(media)
    for _ in range(copies):
        merged.extend(copy.deepcopy(text_blocks))
    return merged


def _assert_copied_segments_identical(record: dict, provenance: dict) -> None:
    """★ 核心自查（§2.3 边界校验即防呆）：复制出来的 N 份必须**逐字节相同**。

    这是"图文内容一字不改"这句话的可验证形式。任一份不同即说明块被改写/漂移 ⇒ 非零退出。
    """
    if provenance["variant"] != "v1-image-copy":
        return
    synth_user = record["messages"][-2]
    content = synth_user.get("content")
    if not isinstance(content, list) or not content:
        raise SynthError("自查失败：合成 user 的 content 不是非空块列表")
    per_copy = provenance["copied_block_count_per_copy"]
    copies = provenance["copies"]
    if len(content) != per_copy * copies:
        raise SynthError(f"自查失败：合成块数 {len(content)} ≠ 每份 {per_copy} × N {copies}")
    first = json.dumps(content[:per_copy], ensure_ascii=False, sort_keys=True)
    for index in range(1, copies):
        segment = json.dumps(
            content[index * per_copy : (index + 1) * per_copy], ensure_ascii=False, sort_keys=True
        )
        if segment != first:
            raise SynthError(f"自查失败：第 {index + 1} 份复制内容与第 1 份不一致（内容被改写）")


def write_one_tier(
    *,
    base: dict,
    base_index: int,
    copies: int,
    variant: str,
    tier_tag: str,
    target_tokens: int,
    out_dir: Path,
    tier_id: str,
    assistant_content: str = MINIMAL_ASSISTANT_CONTENT,
) -> dict:
    """落一个档：``<out-dir>/subsets/<tier>-synth-<档>.jsonl``（**严格 1 行**）。"""
    record, provenance = build_synth_record(
        base, copies=copies, variant=variant, assistant_content=assistant_content
    )
    _assert_copied_segments_identical(record, provenance)
    provenance["base_index"] = base_index
    provenance["tier_tag"] = tier_tag
    provenance["target_tokens"] = target_tokens
    provenance["tier_id"] = tier_id
    # ★ A4 适用性（裁定 1）：生成期就定死"是否双跑"与"该档适用判据"，
    #   单跑档的 a4_status 直接写 `not_applicable_single_run`（**绝不是 pass**）。
    dual_run = a4_dual_run_for(tier_tag)
    a4_status = a4_status_for_build(tier_tag)
    assert_a4_status_invariant(dual_run, a4_status)
    provenance["a4_dual_run"] = dual_run
    provenance["a4_status"] = a4_status
    provenance["a4_status_display"] = A4_STATUS_DISPLAY[a4_status] if a4_status else "待跑"
    provenance["applicable_criteria"] = list(applicable_criteria(tier_tag))

    subsets_dir = out_dir / "subsets"
    subsets_dir.mkdir(parents=True, exist_ok=True)
    # ★ 训练器直读的文件名：run_matrix54.sh 的 path 一致性检查要求
    #   `data.train_path == dirname(train_path)/<tier_id>.jsonl`。故"被跑的那一份"必须是
    #   `<tier_id>.jsonl`（单一文件名，逐档轮换——见 run_synth.sh 的 stage 阶段）。
    out_path = subsets_dir / f"{tier_id}.jsonl"
    record["id"] = f"{tier_id}-synth-{variant}-{tier_tag}"
    line = json.dumps(record, ensure_ascii=False)
    if "\n" in line:
        raise SynthError("自查失败：单行内容里出现换行")
    out_path.write_text(line + "\n", encoding="utf-8")

    # ★ 逐档留档：因为"被跑的那一份"文件名恒定（上一条），**必须**另存一份带档名的副本，
    #   否则后一档会覆盖前一档、复核阶段无法逐档实测（本包最关键的判据会失效）。
    by_tier_dir = out_dir / "by-tier"
    by_tier_dir.mkdir(parents=True, exist_ok=True)
    by_tier_path = by_tier_dir / f"{tier_id}-synth-{tier_tag}.jsonl"
    by_tier_path.write_text(line + "\n", encoding="utf-8")

    for path in (out_path, by_tier_path):
        lines = path.read_text(encoding="utf-8").splitlines()
        if len(lines) != 1:
            raise SynthError(f"自查失败：{path} 应为 1 行，实际 {len(lines)} 行")

    provenance["path"] = str(out_path)
    provenance["by_tier_path"] = str(by_tier_path)
    provenance["bytes"] = by_tier_path.stat().st_size
    # ★ 实测回填字段（指挥官 2026-09-19 裁定 1 硬要求）：留出字段，由**执行上机的那一批**
    #   回写。`None` = 待上机；回填时必须是
    #   `{"tokens": <int>, "measured": true, "source": "<量法/命令>", "measured_at": "<ISO>"}`
    #   —— `verify_synth_artifacts.py` 的 G4/G5 会强制"非空即必须自证是实测"。
    provenance["tokens_measured"] = None
    return provenance


def _render_readme(provenance: dict, source: Path, base: dict, variant: str) -> str:
    """生成旁挂 README（人读；JSONL 本身不能有注释行）。"""
    spec = VARIANTS[variant]
    return f"""# 合成长序列档 —— {provenance["tier_id"]} / {variant} / 目标\
 {provenance["target_tokens"]} token

★ 本文件是**人读的旁挂说明**；同目录的 `{provenance["tier_id"]}.jsonl` 是被训练的**数据本体**。
  JSONL 里**不能**写注释行（会破坏 1 行 = 1 step 的契约），所以语义差异写在这里 + provenance.json。

## 变体语义（口径 2：两个变体绝不可混填同一列）

- 变体 = `{variant}`
- {spec["note"]}
- **是否用于填「最大可行上下文」列 = {provenance["fills_context_column"]}**
{
        "  ⚠ **本变体不得用于填该列**，只能作为单独标注的数据点。"
        if not provenance["fills_context_column"]
        else "  ← 本变体是填该列的**主口径**。"
    }

## 构造（逐字口径）

- 基样来源：`{source}` 第 **{provenance["base_index"]}** 条（id = `{provenance["base_id"]}`），
  实测 token 长度 = **{
        provenance.get("base_tokens_measured", "未传入（须由 measure_synth_tokens.py 实测）")
    }**
- 复制份数 N = **{provenance["copies"]}**（由 `measure_synth_calibration.json` 的**实测**\
标定给出，非算式）
- 拼接口径：`messages = [system(1 份)] + [user(原末轮 user 内容块 × N)] + [assistant(最小占位)]`
- 源 user 块数 = {provenance["source_block_count"]}（其中媒体块 \
{provenance["source_media_block_count"]}）
- 合成后 user 块数 = {provenance["output_user_block_count"]}（其中媒体块 \
{provenance["output_media_block_count"]}）
- 最小 assistant 轮内容 sha256 = `{provenance["assistant_content_sha256"]}`

## ★★ 必须随本档一起登记的语义差异

1. **注意力跨拼接边界**：N 份内容在**同一条序列内互相可见**；真实数据里它们分属不同样本，
   彼此**不可能互见**。⇒ 本档测的是"长序列下框架/显存能不能撑住"，**不是**"模型在真实长
   上下文上的能力"。**不得**据此声称任何"长上下文能力"结论。
2. **合成行没有真实的 assistant 内容**：只有最小占位轮（口径 1 要求它存在，使该行成为合法
   可判定的 SFT 样本）。⇒ 该档的 loss 数值**不具训练语义**，只用于判定"能跑通、不 OOM"。
3. **视觉塔代价随 N 线性放大**（v1 变体）：N 份图像 ⇒ 视觉塔前向与图像 token 都按 N 放大。
   v2 变体只放大文本部分，**不可**与本档混填同一列。
4. **与原始分布的关系**：本档**不是**原始数据分布。原始 T013 子集实测真实长度
   **median 1943 / max 2001 / min 1941**，必须同时登记。

## 每档标注文本（供跟踪文档 / 回填说明；受保护矩阵不能加列）

> `最大可行上下文 = <值>（实测，由合成长序列测得：基样=T013 子集最短样本 1941 token，\
复制 N 份拼接 + 最小 assistant 轮；注意力跨拼接边界；原始数据真实长度 median 1943 / max 2001）`
"""


def backfill_a4_status(
    *,
    out_dir: Path,
    tier_id: str,
    tier_tag: str,
    a4_status: str,
    source_note: str,
    recorded_at: str | None = None,
) -> dict:
    """把 A4 状态回填进某一档的 provenance（裁定 1）。

    ★ **不变式在写入前强制**（§2 防呆）：单跑档传 `pass` 会**硬失败**
    （:func:`assert_a4_status_invariant`）——从结构上挡住"把没跑的记成跑了"。

    双跑档由上机批次判定 pass/fail 后调用；单跑档**不需要**调用
    （生成期已写死 `not_applicable_single_run`）。
    """
    from datetime import datetime

    provenance_path = out_dir / "subsets" / f"{tier_id}-synth-{tier_tag}.provenance.json"
    if not provenance_path.is_file():
        raise SynthError(f"provenance 不存在，无法回填：{provenance_path}")
    doc = json.loads(provenance_path.read_text(encoding="utf-8"))

    # ★ 以 provenance 里**已定死**的 a4_dual_run 为准（单一真相源），不信调用方传的值。
    dual_run = bool(doc.get("a4_dual_run"))
    assert_a4_status_invariant(dual_run, a4_status)

    doc["a4_status"] = a4_status
    doc["a4_status_display"] = A4_STATUS_DISPLAY[a4_status]
    doc["a4_evidence"] = {
        "a4_dual_run": dual_run,
        "source": source_note,
        "recorded_at": recorded_at or datetime.now(UTC).isoformat(),
    }
    provenance_path.write_text(json.dumps(doc, ensure_ascii=False, indent=1), encoding="utf-8")
    print(
        f"[backfill-a4] {tier_tag}: a4_status={a4_status} "
        f"({A4_STATUS_DISPLAY[a4_status]}) dual_run={dual_run} -> {provenance_path}"
    )
    return doc


def backfill_measured_tokens(
    *,
    out_dir: Path,
    measured_path: Path,
    tier_id: str,
    tier_tag: str,
    source_note: str,
    measured_at: str | None = None,
) -> dict:
    """把**实测** token 数回填进某一档的 provenance（裁定 1 的硬要求）。

    "不得让实测读数永久缺失"——上机量出后由执行上机的那一批调用本函数回写。
    回填值必须自证是实测（``measured: true`` + ``source``），否则
    ``verify_synth_artifacts.py`` 的 G5 会报错（挡算式冒充实测）。
    """
    from datetime import datetime

    provenance_path = out_dir / "subsets" / f"{tier_id}-synth-{tier_tag}.provenance.json"
    if not provenance_path.is_file():
        raise SynthError(f"provenance 不存在，无法回填：{provenance_path}")
    doc = json.loads(provenance_path.read_text(encoding="utf-8"))
    measured = json.loads(measured_path.read_text(encoding="utf-8"))
    if not isinstance(measured, dict) or measured.get("measured") is not True:
        raise SynthError(
            f"{measured_path} 的 measured 不为 true ⇒ 拒绝回填（只接受实测产物，"
            "不得用算式冒充实测）"
        )
    tokens = None
    for entry in measured.get("per_sample") or []:
        if isinstance(entry, dict) and entry.get("tokens"):
            tokens = int(entry["tokens"])
            break
    if tokens is None:
        summary = measured.get("summary") or {}
        if summary.get("min"):
            tokens = int(summary["min"])
    if tokens is None:
        raise SynthError(f"{measured_path} 里找不到任何 token 读数")

    doc["tokens_measured"] = {
        "tokens": tokens,
        "measured": True,
        "source": source_note,
        "measured_at": measured_at or datetime.now(UTC).isoformat(),
        "measurement_file": str(measured_path),
    }
    provenance_path.write_text(json.dumps(doc, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[backfill] {tier_tag}: tokens_measured={tokens} -> {provenance_path}")
    return doc


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="合成长序列生成器（纯标准库；把 T013 最短基样复制 N 份拼成长样本）"
    )
    parser.add_argument("--source", required=False, type=Path, help="T013 训练子集 JSONL")
    parser.add_argument(
        "--base-index",
        type=int,
        default=None,
        help="基样下标（实测 token 最短者；由 measure_synth_tokens.py --calibrate 给出）",
    )
    parser.add_argument("--base-id", default=None, help="基样 id（与 --base-index 二选一）")
    parser.add_argument(
        "--calibration",
        required=False,
        type=Path,
        help="实测标定 JSON（measure_synth_tokens.py --calibrate 产出）",
    )
    parser.add_argument("--variant", default="v1-image-copy", choices=sorted(VARIANTS))
    parser.add_argument("--out-dir", required=True, type=Path, help="产物根目录（其下建 subsets/）")
    parser.add_argument(
        "--tier-id", default="T013", help="档号（run_matrix54.sh 要求文件名 = <tier-id>.jsonl）"
    )
    parser.add_argument(
        "--targets",
        default=",".join(tag for tag, _ in TIERS),
        help="要生成的档（逗号分隔：4k,8k,...）",
    )
    parser.add_argument(
        "--assistant-content",
        default=MINIMAL_ASSISTANT_CONTENT,
        help="最小 assistant 轮内容（口径 1；默认 OK）",
    )
    parser.add_argument(
        "--backfill-tier", default=None, help="回填模式：档名（如 128k），配合 --backfill-measured"
    )
    parser.add_argument(
        "--backfill-measured",
        type=Path,
        default=None,
        help="回填模式：measure_synth_tokens.py 的实测产物 JSON",
    )
    parser.add_argument(
        "--backfill-source",
        default=None,
        help="回填模式：量法/命令说明（写入 tokens_measured.source）",
    )
    parser.add_argument(
        "--backfill-a4-tier",
        default=None,
        help="A4 回填模式：档名（如 4k）；单跑档必须传 not_applicable_single_run",
    )
    parser.add_argument(
        "--backfill-a4-status",
        default=None,
        help="A4 回填模式：pass / fail / not_applicable_single_run",
    )
    args = parser.parse_args(argv)

    try:
        # ★ A4 状态回填模式（裁定 1）：单跑档传 pass 会**硬失败**（不变式强制）。
        if args.backfill_a4_tier or args.backfill_a4_status:
            if not (args.backfill_a4_tier and args.backfill_a4_status):
                raise SynthError("A4 回填模式需要同时给 --backfill-a4-tier 与 --backfill-a4-status")
            backfill_a4_status(
                out_dir=args.out_dir,
                tier_id=args.tier_id,
                tier_tag=args.backfill_a4_tier,
                a4_status=args.backfill_a4_status,
                source_note=args.backfill_source or "run_synth.sh（同 config 同 seed 双跑比对）",
            )
            return 0

        # ★ 回填模式（裁定 1）：把上机量出的 token 数写进 provenance，不让实测永久缺失。
        if args.backfill_tier or args.backfill_measured:
            if not (args.backfill_tier and args.backfill_measured):
                raise SynthError("回填模式需要同时给 --backfill-tier 与 --backfill-measured")
            backfill_measured_tokens(
                out_dir=args.out_dir,
                measured_path=args.backfill_measured,
                tier_id=args.tier_id,
                tier_tag=args.backfill_tier,
                source_note=args.backfill_source
                or "measure_synth_tokens.py（ms-swift 真 collator，batch=1，无截断）",
            )
            return 0

        if not args.source:
            parser.error("生成模式需要 --source")
        if not args.calibration:
            parser.error("生成模式需要 --calibration")
        rows = _load_jsonl(args.source)
        base_index, base = _pick_base(rows, args.base_index, args.base_id)
        calibration = _load_calibration(args.calibration)
        wanted = [tag.strip() for tag in args.targets.split(",") if tag.strip()]
        known = dict(TIERS)
        unknown = [tag for tag in wanted if tag not in known]
        if unknown:
            raise SynthError(f"未知档位 {unknown}；可用 {sorted(known)}")

        args.out_dir.mkdir(parents=True, exist_ok=True)
        written: list[dict] = []
        for tag in wanted:
            target = known[tag]
            # 该档 `max_prompt_length` = **同一份档位表**（`emit_configs.TIERS`，逐档已
            # 断言同源）⇒ 这里用的上限就是配置会写进 YAML 的那个数（§1.4 单一真相源）。
            max_length = known[tag]
            copies, predicted = _copies_for(calibration, target, max_length=max_length)
            provenance = write_one_tier(
                base=base,
                base_index=base_index,
                copies=copies,
                variant=args.variant,
                tier_tag=tag,
                target_tokens=target,
                out_dir=args.out_dir,
                tier_id=args.tier_id,
                assistant_content=args.assistant_content,
            )
            # ★ 上限合规证据（§2 防呆：把"没超上限"写成产物里的可核事实，而不是靠复查）。
            provenance["max_prompt_length"] = max_length
            provenance["predicted_tokens_at_copies"] = predicted
            provenance["predicted_within_max_length"] = predicted <= max_length
            provenance["copies_selection_rule"] = (
                "实测标定内取满足 预测长度 ≤ max_prompt_length 的最大 N（floor 语义）"
            )
            provenance["base_tokens_measured"] = calibration.get("base_tokens_measured")
            provenance["points_measured"] = calibration.get("points")
            provenance["calibration_source"] = str(args.calibration)
            # 逐档 provenance（各自独立文件，避免后档覆盖前档）。
            (args.out_dir / "subsets" / f"{args.tier_id}-synth-{tag}.provenance.json").write_text(
                json.dumps(provenance, ensure_ascii=False, indent=1), encoding="utf-8"
            )
            (args.out_dir / "subsets" / f"{args.tier_id}-synth-{tag}.README.txt").write_text(
                _render_readme(provenance, args.source, base, args.variant), encoding="utf-8"
            )
            written.append(provenance)
            print(
                f"[ok] {tag}: target={target} copies={copies} "
                f"bytes={provenance['bytes']} media_blocks={provenance['output_media_block_count']}"
            )

        summary = {
            "source": str(args.source),
            "base_index": base_index,
            "base_id": base.get("id"),
            "variant": args.variant,
            "fills_context_column": VARIANTS[args.variant]["fills_context_column"],
            "tier_id": args.tier_id,
            "out_dir": str(args.out_dir),
            "calibration_source": str(args.calibration),
            "base_tokens_measured": calibration.get("base_tokens_measured"),
            "points_measured": calibration.get("points"),
            "tiers": written,
        }
        (args.out_dir / "synth_summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8"
        )
        print(f"[ok] wrote {len(written)} 档 -> {args.out_dir}")
        print(
            "⚠ 下一步（必须）：用 measure_synth_tokens.py 在容器内**实测**每档 token 数，"
            "证明目标长度真的达到；未实测前不得填「最大可行上下文」列。"
        )
        return 0
    except SynthError as exc:
        print(f"FATAL: {exc}", file=sys.stderr)
        return 2


def _load_calibration(path: Path) -> dict:
    """读实测标定；**必须是实测产物**（带 measured=true 与逐点读数）。"""
    if not path.is_file():
        raise SynthError(
            f"标定文件不存在：{path}\n"
            "  ⇒ 先跑 measure_synth_tokens.py --calibrate"
            "（在容器内用 ms-swift 真 collator 实测）。\n"
            "  ⚠ 本生成器**不接受算式估算**：N 必须由实测标定给出（本包最关键的判据）。"
        )
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SynthError(f"标定文件不是合法 JSON：{path}: {exc}") from exc
    if not isinstance(doc, dict):
        raise SynthError("标定文件必须是 JSON 对象")
    if not doc.get("measured"):
        raise SynthError(
            f"标定文件 {path} 的 measured 不为 true ⇒ 拒绝生成。"
            "本生成器只接受**实测**标定（宪法 §2.2 显式即防呆 / 本节判据）。"
        )
    points = doc.get("points")
    if not isinstance(points, list) or not points:
        raise SynthError(f"标定文件 {path} 缺 points 实测点")
    return doc


def _interpolated_copies(low: dict, high: dict, target_tokens: int) -> float:
    """两个实测点之间（或外推）解出"预测命中 ``target_tokens``"的**实数** N。"""
    slope = (float(high["tokens"]) - float(low["tokens"])) / (high["copies"] - low["copies"])
    if slope <= 0:
        raise SynthError(
            f"实测点 {low} → {high} 的斜率 {slope:.3f} ≤ 0（长度没有随 N 增长？）⇒ "
            "标定不可用，请检查测量方法（是否为真 collator、是否 batch=1）"
        )
    return low["copies"] + (target_tokens - float(low["tokens"])) / slope


def _predict_tokens(usable: list[dict], copies: int) -> float:
    """由**同一份实测标定**预测 N 份时的长度（分段线性插值；端点外用最后两点外推）。

    与 :func:`_interpolated_copies` 互为逆运算，**只用实测点**，不含任何拍脑袋算式。
    """
    if copies <= usable[0]["copies"]:
        return float(usable[0]["tokens"])
    if copies >= usable[-1]["copies"]:
        low, high = usable[-2], usable[-1]
    else:
        low = max((p for p in usable if p["copies"] <= copies), key=lambda p: p["copies"])
        high = min((p for p in usable if p["copies"] >= copies), key=lambda p: p["copies"])
        if low["copies"] == high["copies"]:
            return float(low["tokens"])
    slope = (float(high["tokens"]) - float(low["tokens"])) / (high["copies"] - low["copies"])
    return float(low["tokens"]) + slope * (copies - low["copies"])


def _copies_for(
    calibration: dict,
    target_tokens: int,
    *,
    max_length: int,
    safety_margin_tokens: int = 0,
) -> tuple[int, float]:
    """由**实测点**定 N：**满足「预测长度 ≤ 上限」的最大 N**（floor 语义），返回 ``(N, 预测长度)``。

    ★★ 为什么必须是 floor 而不是 ceil（本函数的**唯一**存在理由，别退回去）

    该档配置的 ``data.max_prompt_length`` 就是本参数 ``max_length``（``emit_configs.TIERS``
    逐档同源），而 ms-swift 把它当 **truncation 上限且策略 delete**：样本**实际长度 > 上限
    时整条被丢弃**（实测报文 ``ValueError: Failed to retrieve the dataset``，训练 0 步）。
    旧实现用 ``math.ceil`` 向上取整 ⇒ **保证预测长度 > 目标 = 上限** ⇒ **7 档全部超限、
    全部 0 步**（2026-09-19 task-r3-bulk 实测：4k 5503>4096 … 128k 132727>131072）。
    一个"必然跑不出步数"的档不是能力上限，是**假否定**。

    所以本函数产出的 N 必须**同时**满足：

    1. **上限合规（硬约束）**：``预测长度 + safety_margin_tokens ≤ max_length``。
       预测长度来自**同一份实测标定**（分段线性插值/外推），不是算式代替实测。
    2. **贴近目标（软约束）**：在合规的 N 里取**最大**的（目标长度是长度自变量，
       差太远就不是"该档的长序列"了）。

    两条**无法同时满足**时（例如最小实测点本身就超上限、或标定斜率 ≤ 0）
    ⇒ :class:`SynthError` **fail-closed**：报错并**列出可选 N 及其预测长度**，
    **绝不**静默产出一个必 0 步的档（§2.3 边界校验即防呆 / §3.4 退路与防线的区别）。

    ``safety_margin_tokens``：给"标定值 vs 训练器真实行长"之间已知的系统差留余量
    （缺省 0 = 标定即真实行长，与实测标定口径一致）。标定器若与训练器口径不同
    （例如 SFT 行尾比标定口径多 28 token），**调用方**必须显式把该差传进来——
    宁可 N 小一档，不可超限。

    ⚠ 这是"用实测点插值/外推"，**不是**用算式代替实测——生成的产物随后必须再经
    ``measure_synth_tokens.py`` **实测复核**（报告 §② 的判据）。
    """
    if not isinstance(max_length, int) or isinstance(max_length, bool) or max_length <= 0:
        raise SynthError(
            f"max_length={max_length!r} 非法：必须是 ≥1 的整数（该档 max_prompt_length）。"
            "没有上限就无法保证'不超限'，拒绝生成（fail-closed）。"
        )
    points = calibration["points"]
    usable = [
        point
        for point in points
        if isinstance(point, dict)
        and isinstance(point.get("copies"), int)
        and isinstance(point.get("tokens"), (int, float))
    ]
    if not usable:
        raise SynthError("标定文件的 points 里没有任何可用的 (copies, tokens) 实测点")
    usable.sort(key=lambda point: point["copies"])

    upper = max_length - max(int(safety_margin_tokens), 0)
    best_point = None
    for point in usable:
        if float(point["tokens"]) <= upper:
            best_point = point

    if best_point is not None and len(usable) >= 2 and usable[-1]["copies"] > best_point["copies"]:
        low = best_point
        high = next(p for p in usable if p["copies"] > best_point["copies"])
    elif best_point is not None and len(usable) >= 2:
        # 最大实测点本身就 ≤ 上限：用**最后两点外推**（与标定口径一致），而不是
        # 直接取最大实测点——否则 128k 这类"上限高于最大标定点"的档会被无谓取小。
        low, high = usable[-2], usable[-1]
    else:
        low = high = None

    if low is not None and high is not None:
        want = _interpolated_copies(low, high, upper)  # 预测恰好触顶的实数 N
        candidate = max(int(math.floor(want)), 1)
        floor_n = max(int(best_point["copies"]) if best_point is not None else 1, 1)
        if candidate > floor_n:
            predicted = _predict_tokens(usable, candidate)
            if predicted <= upper:
                return candidate, predicted
            # 步进回退（单调性兜底；实测共线时上面的 floor 已够）
            for probe in range(candidate - 1, floor_n - 1, -1):
                predicted = _predict_tokens(usable, probe)
                if predicted <= upper:
                    return probe, predicted

    if best_point is not None:
        return max(int(best_point["copies"]), 1), float(best_point["tokens"])

    options = "、".join(f"N={int(p['copies'])}→{int(p['tokens'])}" for p in usable)
    smallest = usable[0]
    raise SynthError(
        f"无可行 N：**最小实测点**就超上限（N={int(smallest['copies'])} 预测 "
        f"{int(smallest['tokens'])} token > 上限 {max_length}"
        + (f"（已扣 safety_margin={safety_margin_tokens}）" if safety_margin_tokens else "")
        + f"）。可选实测点：{options}。"
        "⇒ 不改上限、不静默产出一个必 0 步的档；请调整目标长度/上限，或补测更小的 N。"
    )


if __name__ == "__main__":
    raise SystemExit(main())
