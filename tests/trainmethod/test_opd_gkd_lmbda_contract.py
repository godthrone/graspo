"""L1 契约测试：OPD(GKD) 的 **on-policy 路由** —— 纯提示词行必须恒走学生现场采样。

**为什么需要这份测试（真实缺陷，2026-09-19 T046 冒烟）**

ms-swift 4.5.3 的 GKD 每个 batch 先丢一次硬币决定数据来源
（``rlhf_trainers/gkd_trainer.py:293-298``）::

    if self._get_random_num() <= self.lmbda:
        self._data_source = DataSource.STUDENT   # 学生现场采样（on-policy）
    else:
        self._data_source = DataSource.DATASET   # 数据集里既有回答（off-policy）

``_get_random_num``（同文件 ``:435-448``）是 ``random.Random(seed + global_step).random()``，
上游默认 ``lmbda = 0.5``（``arguments/rlhf_args.py:262``）。graspo 的 OPD 数据是
**纯提示词**（``flow/msswift/dataset.py::build_opd_rows``）——DATASET 分支按定义要
消费"数据集里的回答"，纯提示词行没有回答 ⇒ 全部 label 为 ``-100`` ⇒
``extract_active`` 取到 0 行 ⇒ ``jsd_loss``（``gkd_loss.py:121-122``）直接返回
``s_logits.new_zeros(())``（**detached**，没有 ``grad_fn``）⇒ ``gkd_trainer.py:113``
再乘 0 ⇒ 第 0 步 ``accelerator.backward(loss)`` 抛
``RuntimeError: element 0 of tensors does not require grad and does not have a grad_fn``。

seed=42 时 ``random.Random(42).random() == 0.6394267984578837 > 0.5`` ⇒ **第 0 步必然**
落在 DATASET 分支（确定性，不是 50% 概率）。真机证据：T046 冒烟 stdout 的 args dump
里 ``lmbda=0.5``（我们没传，落的是上游默认）＋ 上述 traceback。

因此本文件的断言是"**可核的**"：把"prompt-only 行必须被路由到能算 loss 的分支"
写成一条能真失败的检查（见 ``test_contract_check_rejects_the_pre_fix_argv``）。

**不依赖 torch**：被测的是 ``_config_mapping``（纯计算）与 ``dataset.build_opd_rows``
（纯数据变换），本目录的 ``conftest.py`` 负责无 torch 时的命名空间垫片（§1.3）。
"""

from __future__ import annotations

import random
from pathlib import Path
from types import SimpleNamespace

import pytest

from graspo.core.schema import GraspoConfig
from graspo.flow.msswift._config_mapping import (
    DEFAULT_OPD_LMBDA,
    graspo_to_ms_swift_argv,
    validate_combinations,
)
from graspo.flow.msswift.dataset import build_opd_rows

STUDENT = "/models/Qwen3.5-9B"
TEACHER = "/models/Qwen3.8-27B"

#: ms-swift 4.5.3 的 GKD λ 默认值（``arguments/rlhf_args.py:262``）。
#: 我们**不传** ``--lmbda`` 时实际生效的就是它——这正是缺陷的入口。
UPSTREAM_DEFAULT_LMBDA = 0.5

#: T046 冒烟配方里学生采样的随机种子（``task-r3-lenramp/rig/configs/T046-smoke1.yaml``）。
SMOKE_SEED = 42

#: 上游 4.5.3 参考源码根（``.local/refs/`` 是 gitignored 的本地参考；
#: 不存在时相关用例 skip——不做"假装核对过上游"的假断言）。
_REFS_ROOT = (
    Path(__file__).resolve().parents[2] / ".local" / "refs" / "ms-swift-4.5.3" / "src" / "swift"
)


def _opd_config(**distill: object) -> GraspoConfig:
    """最小合法 OPD + msswift 配置（教师与学生都为具体路径）。"""
    return GraspoConfig.model_validate(
        {
            "train_method": "opd",
            "backend": "msswift",
            "model": {"model_path": STUDENT},
            "data": {"train_path": "unused.jsonl"},
            "training": {"seed": SMOKE_SEED, "max_new_tokens": 256},
            "distill": {"teacher_model_path": TEACHER, **distill},
        }
    )


def _argv(config: GraspoConfig) -> list[str]:
    return graspo_to_ms_swift_argv(config, stage="opd", dataset_path="d.jsonl", output_dir="o")


def _value_of(argv: list[str], flag: str) -> str | None:
    if flag not in argv:
        return None
    return argv[argv.index(flag) + 1]


def _argv_lmbda(argv: list[str]) -> float:
    """argv 里真正生效的 λ；缺失 = 上游默认值（**这一条就是缺陷的入场券**）。"""
    raw = _value_of(argv, "--lmbda")
    return UPSTREAM_DEFAULT_LMBDA if raw is None else float(raw)


def _data_source_at_step(argv: list[str], *, seed: int, step: int) -> str:
    """复刻上游 ``gkd_trainer._rollout_samples`` 的分流判据（逐字同式）。

    上游：``seed = int(args.seed) + int(state.global_step)``，
    ``random.Random(seed).random() <= self.lmbda`` ⇒ ``STUDENT``，否则 ``DATASET``。
    """
    return "STUDENT" if random.Random(seed + step).random() <= _argv_lmbda(argv) else "DATASET"


def assert_prompt_only_routes_on_policy(argv: list[str], *, seed: int, steps: int = 10) -> None:
    """★ 核心断言：纯提示词的 OPD 行 ⇒ 每一步都必须是 ``STUDENT`` 分支。

    失败信息直接给出机制与上游出处，避免下一个人重新侦查（宪法 §2.1 契约即防呆）。
    """
    offenders = [
        step
        for step in range(steps)
        if _data_source_at_step(argv, seed=seed, step=step) != "STUDENT"
    ]
    assert not offenders, (
        f"OPD rows are prompt-only, so every GKD step must use DataSource.STUDENT "
        f"(on-policy student sampling). Got DATASET at step(s) {offenders} with "
        f"lmbda={_argv_lmbda(argv)!r} (seed={seed}). The DATASET branch distills on the "
        "dataset's existing responses (rlhf_trainers/gkd_trainer.py:293-298); with "
        "prompt-only rows it has zero supervised tokens, so gkd_loss returns a detached "
        "zero (rlhf_trainers/gkd_loss.py:121-122) and step 0 dies with "
        "`RuntimeError: element 0 of tensors does not require grad and does not have a grad_fn`."
    )


def _prompt_only_rows(config: GraspoConfig) -> list[dict]:
    sample = SimpleNamespace(
        messages=[
            {"role": "system", "content": "You are a careful assistant."},
            {"role": "user", "content": "这张图里的读数是多少？"},
        ]
    )
    return build_opd_rows([sample], config=config)


# ── ① argv 契约：λ 被显式钉在 1.0 ────────────────────────────────────────────


def test_opd_argv_pins_lmbda_to_one():
    """★ 未提供 ``distill.lmbda`` 时，argv 必须**显式**写 ``--lmbda 1.0``。

    不能靠上游默认值：上游默认 0.5 在 seed=42/step=0 就命中 DATASET 分支。
    """
    argv = _argv(_opd_config())

    assert _value_of(argv, "--lmbda") is not None, "--lmbda must be passed explicitly"
    assert _value_of(argv, "--lmbda") == str(DEFAULT_OPD_LMBDA)
    assert _argv_lmbda(argv) == DEFAULT_OPD_LMBDA == 1.0


def test_opd_argv_keeps_the_other_gkd_mandatory_items():
    """GKD 必需项同时在场：``--rlhf_type gkd`` + 本地教师 + 采样长度上限。"""
    argv = _argv(_opd_config(offload_teacher_model=True))

    assert _value_of(argv, "--rlhf_type") == "gkd"
    assert _value_of(argv, "--teacher_model") == TEACHER
    assert _value_of(argv, "--offload_teacher_model") == "true"
    assert _value_of(argv, "--max_completion_length") == "256"
    assert _value_of(argv, "--seed") == str(SMOKE_SEED)


def test_opd_argv_lmbda_is_parsable_as_float():
    """``_flag`` 的标量分支产出的字面量必须能被 ms-swift 的 ``float`` 字段解析。"""
    argv = _argv(_opd_config())

    assert isinstance(float(_value_of(argv, "--lmbda") or ""), float)


# ── ② 数据形态契约：行是纯提示词（⇒ off-policy 分支不可能有监督）────────────


def test_opd_rows_are_prompt_only_without_supervision():
    """★ 数据形态：OPD 行只有 ``system``/``user``，没有 assistant 轮。

    这与上游 DATASET（off-policy）分支的前提**直接矛盾**——所以 λ < 1 在本通道
    不可能成立。同一断言的"不带 targets 列"版本已由 ``test_cpt_opd_channels.py::
    test_opd_rows_are_prompt_only`` 覆盖，此处补的是"**没有 assistant 轮**"这一面
    （断图机制的充分条件）。
    """
    rows = _prompt_only_rows(_opd_config())

    assert len(rows) == 1
    roles = [message["role"] for message in rows[0]["messages"]]
    assert roles and roles[-1] == "user"
    assert "assistant" not in roles, "prompt-only is the OPD contract; no supervision in the row"
    assert "targets" not in rows[0]


# ── ③ 路由契约（正向 + 反向）────────────────────────────────────────────────


def test_prompt_only_rows_are_routed_to_the_student_branch():
    """正向：真实 argv 下，seed=42 的每一步都落在能算 loss 的 STUDENT 分支。"""
    argv = _argv(_opd_config())

    assert_prompt_only_routes_on_policy(argv, seed=SMOKE_SEED, steps=10)


def test_contract_check_rejects_the_pre_fix_argv():
    """★ 负向用例（**证明上面那条检查真的会失败**）：把 ``--lmbda`` 从 argv 里摘掉
    （= 修复前的行为，实际生效值落回上游默认 0.5）后，同一个检查必须报错。

    这一条不是"再断言一次 1.0"——它拿的是**修复前真实产出的 argv 形态**，
    并断言第 0 步会落到 DATASET（= 真机上抛 detached loss 的那一步）。
    """
    argv = _argv(_opd_config())
    index = argv.index("--lmbda")
    pre_fix_argv = argv[:index] + argv[index + 2 :]

    assert "--lmbda" not in pre_fix_argv
    assert _argv_lmbda(pre_fix_argv) == UPSTREAM_DEFAULT_LMBDA
    assert _data_source_at_step(pre_fix_argv, seed=SMOKE_SEED, step=0) == "DATASET"
    with pytest.raises(AssertionError, match="must use DataSource.STUDENT"):
        assert_prompt_only_routes_on_policy(pre_fix_argv, seed=SMOKE_SEED, steps=10)


# ── ④ 启动前 fail-closed：显式 λ < 1 被拒（而不是上机等一次必死的运行）─────


def test_validate_rejects_explicit_offpolicy_lmbda():
    """显式 ``lmbda < 1`` ⇒ 启动前报错，消息带机制与上游出处。"""
    with pytest.raises(ValueError) as excinfo:
        validate_combinations(_opd_config(lmbda=0.5))

    message = str(excinfo.value)
    assert "distill.lmbda=0.5" in message
    assert "prompt-only" in message
    assert "does not require grad" in message


def test_validate_accepts_the_default_and_explicit_one():
    """``None``（映射层补 1.0）与显式 ``1.0`` 都必须通过校验。"""
    validate_combinations(_opd_config())
    validate_combinations(_opd_config(lmbda=1.0))


def test_validate_leaves_other_train_methods_alone():
    """λ 是 GKD 的旋钮：非 OPD 阶段不受本校验影响（不误伤 GRPO/SFT）。"""
    config = GraspoConfig.model_validate(
        {
            "train_method": "graspo",
            "backend": "msswift",
            "model": {"model_path": STUDENT},
            "data": {"train_path": "unused.jsonl"},
            "distill": {"teacher_model_path": TEACHER, "lmbda": 0.5},
        }
    )

    validate_combinations(config)


# ── ⑤ 上游行号锚点仍在（有参考源码时才核对，不假装核对）─────────────────────


def test_cited_upstream_lines_still_match():
    """核对本文件与 ``_config_mapping`` 引用的上游出处确实还是那几行。"""
    gkd_trainer = _REFS_ROOT / "rlhf_trainers" / "gkd_trainer.py"
    gkd_loss = _REFS_ROOT / "rlhf_trainers" / "gkd_loss.py"
    rlhf_args = _REFS_ROOT / "arguments" / "rlhf_args.py"
    if not gkd_trainer.is_file():
        pytest.skip(f"ms-swift 4.5.3 reference source not available at {_REFS_ROOT}")

    trainer_lines = gkd_trainer.read_text(encoding="utf-8").splitlines()
    loss_lines = gkd_loss.read_text(encoding="utf-8").splitlines()

    assert "if self._get_random_num() <= self.lmbda:" in "\n".join(trainer_lines[290:300])
    assert "self._data_source = DataSource.DATASET" in "\n".join(trainer_lines[290:300])
    assert "def _get_random_num" in "\n".join(trainer_lines[430:450])
    # 零有效 token ⇒ detached 零张量（断图的最后一步）
    assert "if N == 0:" in "\n".join(loss_lines[115:125])
    assert "return s_logits.new_zeros(())" in "\n".join(loss_lines[115:125])
    assert "lmbda: float = 0.5" in rlhf_args.read_text(encoding="utf-8")
