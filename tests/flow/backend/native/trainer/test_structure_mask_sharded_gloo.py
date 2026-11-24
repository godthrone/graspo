"""★ P3 的 ``k`` 必须取**整组**（跨卡）结构字符数最小值，不是**本卡切片**的最小值。

## 缺陷（本文件钉住的形态）

``native`` 在 ``dp>1`` 时每张卡只拿到本卡的 ``G//m`` 条 completion
（``flow/backend/native/trainer/rollout.py``）。``k`` 若只在本卡切片上取 ``min``，
必有 ``k_shard >= k_group`` ⇒ 「本卡都写对、别的卡写短了」的结构位置也被置 0
（**过度 mask**）⇒ 同一算法在 dp=1 与 dp=4 下行为不同、读数不可比。

## 本文件钉什么

1. **真 gloo 双 rank 判别**（``test_sharded_structure_min_is_the_whole_group_min``）：
   组 = 卡0 的 4 条结构字符数 ``[22,22,22,22]`` + 卡1 的 ``[22,22,22,15]``。
   经**生产路径**（``gather_rollout_values`` 会合 → ``compute_group_advantages``）后
   ``k = 15``，**卡 0 上只 mask 前 15 个结构字符**。
2. **阳性对照**（同一子进程内的 ``control`` 分支）：不给 ``group_min_structure_chars``
   ⇒ 算法层退回**本卡切片 min**（= 修复前的行为）⇒ 卡 0 mask 前 22 个结构字符。
   两条读数不同 ⇒ 断言 1 **能失败**，不是哑断言。
3. **dp=1 逐位不变**（``test_explicit_whole_group_k_is_a_no_op``）：整组就在手里时，
   显式传 ``k``（= 本组 min）与不传**逐位相同** ⇒ 未分卡路径传不传都无差别。
4. **流程接线**（``test_finalize_wires_the_gathered_k_into_the_algorithm``）：源码级
   断言 `k` 从会合结果流到 ``compute_group_advantages``（⛔ 防止"算出来了但没接上"）。

⛔ 期望值不用被测实现算：结构字符数由**模板常量**独立给出（公共前缀
``<tool_call>\\n<function=`` = 22 字符；短的那条独立算出 15），masked 字符由
「标签是 S 且 advantage 恰为 0」反推。
"""

from __future__ import annotations

import inspect
import json
import os
import socket
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

torch = pytest.importorskip("torch", reason="分卡真进程组测试需要 torch（CPU/gloo 即可）")

if not torch.distributed.is_gloo_available():  # pragma: no cover - 环境相关
    pytest.skip("torch.distributed gloo 不可用", allow_module_level=True)

_REPO = Path(__file__).resolve().parents[5]
_SRC = _REPO / "src"

#: 组内公共前缀的字符数：``<tool_call>``(11) + ``\\n``(1) + ``<function=``(10)。
#: ★ 独立算出（模板常量），不用实现里的 S 字符统计。
COMMON_PREFIX_CHARS = len("<tool_call>") + 1 + len("<function=")

#: 写对了公共前缀、工具名是乱码的那条：结构字符数 = 公共前缀长度。
STATIC_TEXT = "<tool_call>\n<function=!!!"

#: 短的那一条：模型只写到 ``<tool_call>\\n<fun`` 就停了（**非** max_len 硬截断）。
#: 结构字符数 = 文本长度 − 1（末尾 EOS 把最后一个字符判成 E：它没写完）。独立数出，
#: 不用实现算。
SHORT_TEXT = "<tool_call>\n<fun"
SHORT_STRUCTURE_CHARS = len(SHORT_TEXT) - 1

#: 组内被测目标的 GT（工具调用，两个参数）。
_TARGETS = [
    {
        "output": {
            "tool_calls": [
                {
                    "name": "robot_atomic_control",
                    "arguments": {"action_type": "逆时针旋转", "angle_deg": 16.9},
                }
            ]
        }
    }
]


class _FakeTokenizer:
    """每字符一个 token，offset ``(i, i+1)``（与既有 advantage 测试同一模式）。"""

    def __call__(self, text: str, return_offsets_mapping: bool = False, **kwargs):  # noqa: D102
        class _FakeEncoding:
            def __init__(self, input_ids: list[int], offset_mapping: list[tuple[int, int]]):
                self.input_ids = input_ids
                self.offset_mapping = offset_mapping

            def get(self, key, default=None):  # noqa: D102
                return getattr(self, key, default)

        return _FakeEncoding(
            [ord(char) % 1000 for char in text], [(i, i + 1) for i in range(len(text))]
        )


_CHILD = textwrap.dedent(
    '''
    """gloo 双子进程子脚本：用**真**会合 + 真算法层，把逐条读数写 JSONL。"""
    import json
    import os
    import sys
    from pathlib import Path
    from types import SimpleNamespace

    import torch
    import torch.distributed as dist

    SRC = Path(sys.argv[1])
    OUT = Path(sys.argv[2])
    RANK = int(sys.argv[3])
    WORLD = int(sys.argv[4])
    RENDEZVOUS = sys.argv[5]

    torch.set_num_threads(1)
    sys.path.insert(0, str(SRC))

    from graspo.flow.backend.native.trainer.rollout_shard import gather_rollout_values
    from graspo.ripple.annotation.advantages import (
        apply_tail_eos,
        compute_group_advantages,
        group_structure_char_counts,
    )
    from graspo.ripple.annotation.char_tag import CharTag
    from graspo.ripple.annotation.labeler import AnnotationInput, annotate
    from graspo.ripple.annotation.tokenize import char_to_token_labels

    STATIC = "<tool_call>\\n<function=!!!"          # 结构字符数 22（写对了公共前缀）
    SHORT = "<tool_call>\\n<fun"                    # 结构字符数 15（只写到一半）
    TARGETS = [
        {
            "output": {
                "tool_calls": [
                    {
                        "name": "robot_atomic_control",
                        "arguments": {"action_type": "逆时针旋转", "angle_deg": 16.9},
                    }
                ]
            }
        }
    ]

    class FakeTokenizer:
        """每字符一个 token，offset ``(i, i+1)``（与既有 advantage 测试同一模式）。"""

        def encode(self, text, add_special_tokens=False):
            return [ord(char) % 1000 for char in text]

        def __call__(self, text, return_offsets_mapping=False, **kwargs):
            offsets = [(i, i + 1) for i in range(len(text))]

            class FakeEncoding:
                def __init__(self, input_ids, offset_mapping):
                    self.input_ids = input_ids
                    self.offset_mapping = offset_mapping

                def get(self, key, default=None):
                    return getattr(self, key, default)

            return FakeEncoding(self.encode(text), offsets)

    def annotate_one(text):
        return annotate(AnnotationInput(text, TARGETS, None, "tool_call"))

    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", RENDEZVOUS)
    dist.init_process_group(backend="gloo", rank=RANK, world_size=WORLD)

    class _Adapter:
        device = torch.device("cpu")
        tp_state = SimpleNamespace(
            dp_size=WORLD, dp_group=dist.group.WORLD, device=torch.device("cpu")
        )

    class _Runtime:
        def _require_adapter(self):
            return _Adapter()

    # ── 整组 8 条：卡 0 = [22,22,22,22]，卡 1 = [22,22,22,15] ────────────────
    whole_texts = [STATIC] * 7 + [SHORT]
    RPR = len(whole_texts) // WORLD   # 本卡切片条数 = G // m = 4
    lo, hi = RANK * RPR, (RANK + 1) * RPR
    local_texts = whole_texts[lo:hi]
    local_anns = [annotate_one(text) for text in local_texts]
    tokenizer = FakeTokenizer()

    # ★ 逐条结构字符数由**算法层的单一真相源**给出（末尾 EOS 口径与 advantage 同一个）。
    local_counts = group_structure_char_counts(
        completions=local_texts, annotations=local_anns, format_type="tool_call"
    )

    # ── 生产路径：按既有 gather 通道会合（载荷 = 整数标量；⛔ 不扩契约）────────
    gathered = gather_rollout_values(
        _Runtime(),
        [torch.tensor(local_counts, dtype=torch.float64)],
        cutoff=1,
        rollouts_per_rank=RPR,
        what="结构字符数（P3 的 k）",
    )
    whole_counts = [int(value) for value in gathered[0].tolist()]
    k = int(gathered[0].min().item())
    # ★ 对照（= 修复前的行为）：k 只在本卡切片上取 min（本卡看不到别的卡的条）。
    k_local_slice = min(local_counts)

    def masked_positions(text, annotation, group_min):
        ragged = compute_group_advantages(
            completions=[text],
            annotations=[annotation],
            targets=TARGETS,
            tokenizer=tokenizer,
            format_type="tool_call",
            group_min_structure_chars=group_min,
        )
        tags = apply_tail_eos(text, annotation.tags, "tool_call")
        labels = char_to_token_labels(text, tags, annotation.fields, tokenizer)
        advantage = ragged[0]
        assert len(labels) == len(advantage), "token 数与 advantage 长度必须对齐"
        masked, unmasked = [], []
        for index, (label, value) in enumerate(zip(labels, advantage, strict=True)):
            if label.tag is not CharTag.STRUCTURE:
                continue
            if value == 0.0:
                masked.append(index)
            else:
                assert value == 1.0, f"S token 只能是 0（全对）或 +1.0，实测 {value}"
                unmasked.append(index)
        return masked, unmasked

    rows = []
    for text, annotation, count in zip(local_texts, local_anns, local_counts, strict=True):
        # ★ 生产路径：整组 k；对照（修复前）：本卡切片 min。
        masked_prod, unmasked_prod = masked_positions(text, annotation, k)
        masked_ctrl, unmasked_ctrl = masked_positions(text, annotation, k_local_slice)
        rows.append(
            {
                "count": int(count),
                "masked_production": masked_prod,
                "unmasked_production": unmasked_prod,
                "masked_control": masked_ctrl,
                "unmasked_control": unmasked_ctrl,
            }
        )

    with OUT.open("a", encoding="utf-8") as fh:
        fh.write(
            json.dumps(
                {
                    "rank": RANK,
                    "rpr": RPR,
                    "local_counts": [int(value) for value in local_counts],
                    "whole_counts": whole_counts,
                    "k": k,
                    "rows": rows,
                }
            )
            + "\\n"
        )
    dist.barrier()
    dist.destroy_process_group()
    '''
)


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _launch(tmp_path: Path, *, world: int = 2) -> list[dict]:
    child = tmp_path / "gloo_child_structure_min.py"
    child.write_text(_CHILD, encoding="utf-8")
    out = tmp_path / "structure_min.jsonl"
    port = str(_free_port())
    env = dict(os.environ)
    env["PYTHONPATH"] = str(_SRC)
    env["OMP_NUM_THREADS"] = "1"
    procs = [
        subprocess.Popen(
            [
                sys.executable,
                str(child),
                str(_SRC),
                str(out),
                str(rank),
                str(world),
                port,
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=env,
        )
        for rank in range(world)
    ]
    errs: list[str] = []
    for proc in procs:
        _, stderr = proc.communicate(timeout=300)
        errs.append(stderr)
    if any(proc.returncode != 0 for proc in procs):
        detail = "\n".join(
            f"----- rank{index} (rc={proc.returncode}) -----\n{errs[index]}"
            for index, proc in enumerate(procs)
        )
        raise AssertionError(detail)
    return [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]


# ══════════════════════════════════════════════════════════════════════════════
# 1. 真 gloo 双 rank：k = 整组 min ⇒ 卡 0 只 mask 前 15 个结构字符
# ══════════════════════════════════════════════════════════════════════════════


def test_sharded_structure_min_is_the_whole_group_min(tmp_path: Path) -> None:
    """★ 验收标准 1：``k`` 取**整组** min（15），不是本卡切片 min（卡 0 = 22）。"""
    rows = _launch(tmp_path)
    assert len(rows) == 2
    by_rank = {row["rank"]: row for row in rows}

    # 构造前提（独立算出的模板常量）
    assert by_rank[0]["local_counts"] == [COMMON_PREFIX_CHARS] * 4, by_rank[0]["local_counts"]
    assert by_rank[1]["local_counts"] == [COMMON_PREFIX_CHARS] * 3 + [SHORT_STRUCTURE_CHARS]
    assert COMMON_PREFIX_CHARS == 22 and SHORT_STRUCTURE_CHARS == 15, (
        "构造前提漂移：本文件按「22 / 15」写死期望值"
    )
    for rank in (0, 1):
        assert by_rank[rank]["k"] == SHORT_STRUCTURE_CHARS, (
            f"rank{rank} 的 k 应为整组 min {SHORT_STRUCTURE_CHARS}"
            f"（本卡切片 min = {min(by_rank[rank]['local_counts'])}），"
            f"会合结果 whole_counts={by_rank[rank]['whole_counts']}"
        )
        assert by_rank[rank]["whole_counts"] == [COMMON_PREFIX_CHARS] * 7 + [
            SHORT_STRUCTURE_CHARS
        ], "会合后必须看到**整组** 8 条的结构字符数"

    expected_masked = list(range(SHORT_STRUCTURE_CHARS))
    for rank in (0, 1):
        for index, row in enumerate(by_rank[rank]["rows"]):
            assert row["masked_production"] == expected_masked, (
                f"rank{rank} 第 {index} 条：生产路径只应 mask 前 {SHORT_STRUCTURE_CHARS} 个"
                f"结构字符，实测 {row['masked_production']}"
            )
    # 卡 0 的 22 字符结构里，后 7 个必须仍是 +1.0（不是「组内全对」）
    for row in by_rank[0]["rows"]:
        assert row["unmasked_production"] == list(
            range(SHORT_STRUCTURE_CHARS, COMMON_PREFIX_CHARS)
        ), row["unmasked_production"]


def test_local_slice_min_would_over_mask_positive_control(tmp_path: Path) -> None:
    """★ 阳性对照（**能失败**）：去掉跨卡 ``k`` ⇒ 卡 0 退回本卡 min 22 ⇒ **多 mask 7 个**。

    这条证明上一条不是恒真式：同一批输入，只有「k 取整组 min」这一处差别，
    卡 0 的 masked 集合就从 ``range(22)`` 变成 ``range(15)``。
    """
    rows = _launch(tmp_path)
    by_rank = {row["rank"]: row for row in rows}
    # 卡 0：本卡 min = 22 ⇒ 22 个结构字符全部 mask（修复前的过度 mask 形态）
    for row in by_rank[0]["rows"]:
        assert row["masked_control"] == list(range(COMMON_PREFIX_CHARS)), row["masked_control"]
        assert row["unmasked_control"] == []
        assert row["masked_control"] != row["masked_production"], (
            "修复前后读数必须不同，否则本文件的判别断言是哑的"
        )
    # 卡 1：本卡就含那条短的 ⇒ 本卡 min 也是 15（修复前后同值）
    for row in by_rank[1]["rows"]:
        assert row["masked_control"] == list(range(SHORT_STRUCTURE_CHARS)), row["masked_control"]


# ══════════════════════════════════════════════════════════════════════════════
# 2. dp=1 逐位不变：整组就在手里时，显式传 k 与不传完全等价
# ══════════════════════════════════════════════════════════════════════════════


def test_explicit_whole_group_k_is_a_no_op() -> None:
    """★ 整组 8 条（= dp=1 的输入）下，``group_min_structure_chars=k`` 与 ``None`` 逐位相同。"""
    from graspo.ripple.annotation.advantages import (
        compute_group_advantages,
        group_structure_char_counts,
    )
    from graspo.ripple.annotation.labeler import AnnotationInput, annotate

    texts = [STATIC_TEXT] * 7 + [SHORT_TEXT]
    annotations = [annotate(AnnotationInput(text, _TARGETS, None, "tool_call")) for text in texts]
    counts = group_structure_char_counts(
        completions=texts, annotations=annotations, format_type="tool_call"
    )
    assert counts == [COMMON_PREFIX_CHARS] * 7 + [SHORT_STRUCTURE_CHARS], counts

    kwargs = dict(
        completions=texts,
        annotations=annotations,
        targets=_TARGETS,
        tokenizer=_FakeTokenizer(),
        format_type="tool_call",
    )
    implicit = compute_group_advantages(**kwargs)
    explicit = compute_group_advantages(**kwargs, group_min_structure_chars=min(counts))
    assert implicit == explicit, "整组在场时显式 k 必须与本地 min 完全等价（dp=1 逐位不变）"


# ══════════════════════════════════════════════════════════════════════════════
# 3. 流程层的两个新辅助：口径必须与算法层同一个
# ══════════════════════════════════════════════════════════════════════════════


def test_flow_helper_counts_share_the_algorithm_layer_and_truncation_flag() -> None:
    """★ 流程层 ``_prompt_structure_char_counts`` 必须与算法层同口径（含截断标志）。

    ★ 第二段是**能失败**的对照：同一条文本在 ``max_new_tokens`` 下算"被硬截断" ⇒ 不标
    末尾 E ⇒ 结构字符数 15 → 16。若流程层自己重写一遍计数（或忘了传截断标志），
    会合出的 ``k`` 就与算法层实际用的计数错位。
    """
    from types import SimpleNamespace

    from graspo.flow.backend.native.trainer.rollout import RolloutMixin
    from graspo.ripple.annotation.advantages import group_structure_char_counts
    from graspo.ripple.annotation.labeler import AnnotationInput, annotate

    texts = [STATIC_TEXT, STATIC_TEXT, STATIC_TEXT, SHORT_TEXT]
    annotations = [annotate(AnnotationInput(text, _TARGETS, None, "tool_call")) for text in texts]
    state = SimpleNamespace(sample=SimpleNamespace(expects_tool_calls=True))
    generation = SimpleNamespace(completions=texts)
    expected = group_structure_char_counts(
        completions=texts, annotations=annotations, format_type="tool_call"
    )

    def _trainer(max_new_tokens: int):
        trainer = RolloutMixin.__new__(RolloutMixin)  # 只借方法，不需要完整初始化
        trainer.runtime = SimpleNamespace(
            _require_adapter=lambda: SimpleNamespace(tokenizer=_FakeTokenizer())
        )
        trainer.config = SimpleNamespace(training=SimpleNamespace(max_new_tokens=max_new_tokens))
        return trainer

    big = _trainer(max_new_tokens=4096)
    assert big._prompt_structure_char_counts(state, generation, annotations) == expected
    assert expected == [COMMON_PREFIX_CHARS] * 3 + [SHORT_STRUCTURE_CHARS]
    assert big._truncated_by_max(generation) == [False] * 4

    # ★ 混合：只有 STATIC 那三条算硬截断（它本来就有 E，计数不受末尾 EOS 影响）
    mixed = _trainer(max_new_tokens=len(SHORT_TEXT) + 1)
    assert mixed._truncated_by_max(generation) == [True, True, True, False]
    assert mixed._prompt_structure_char_counts(state, generation, annotations) == expected

    # ★ 能失败对照：短的那条也被判硬截断 ⇒ 不标末尾 E ⇒ 结构字符数 15 → 16。
    #   若流程层自己重写一遍计数（或忘了传截断标志），会合出的 k 就与算法层实际计数错位。
    tight = _trainer(max_new_tokens=len(SHORT_TEXT))
    assert tight._truncated_by_max(generation) == [True] * 4
    assert tight._prompt_structure_char_counts(state, generation, annotations) == [
        COMMON_PREFIX_CHARS
    ] * 3 + [SHORT_STRUCTURE_CHARS + 1], "被 max_len 硬截断 ⇒ 末尾 E 不标 ⇒ 结构字符数 +1"


# ══════════════════════════════════════════════════════════════════════════════
# 3. 流程接线：会合出的 k 必须真的流到算法层
# ══════════════════════════════════════════════════════════════════════════════


def test_finalize_wires_the_gathered_k_into_the_algorithm() -> None:
    """★ 源码级：``k`` 由会合结果出、经旁挂表传给 ``compute_group_advantages``。"""
    import importlib

    src = inspect.getsource(importlib.import_module("graspo.flow.backend.native.trainer.rollout"))
    assert "group_min_structure_chars=group_min_structure_chars" in src, (
        "最终化必须把整组 k 传给算法层（否则算出来也没接上）"
    )
    assert 'shard.get("group_min_structure_chars")' in src, (
        "k 必须从分卡会合的旁挂表取（`None` ⇒ 未分卡/dp=1 走本地 min）"
    )
    assert src.index("whole_structure_counts") < src.index(
        "group_min_structure_chars=group_min_structure_chars"
    ), "k 的会合必须发生在使用之前"
    assert "group_structure_char_counts" in src, (
        "逐条结构字符数必须走 ripple 的单一真相源（⛔ 不在本层自己数 S 字符）"
    )
