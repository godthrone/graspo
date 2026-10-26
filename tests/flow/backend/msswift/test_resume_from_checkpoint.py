"""msswift 续训开关（``training.resume_from_checkpoint``）的接线与 fail-closed 锁。

**背景（实测缺陷，2026-10-22 A800 M003 副本）**：配置里给了
``training.resume_from_checkpoint=/out/M003r/checkpoint-40``，ms-swift 的 ``args.json``
却是 ``resume_from_checkpoint = None``，日志 ``'global_step/max_steps': '1/100'``
⇒ **从第 1 步重跑，续训被静默丢弃**。根因是映射层一条 argv 都没产出它。

本文件锁四件事（各自对应一段失败历史）：

1. **不设续训 ⇒ argv 逐字节不变**（防回归：新增映射行不许改动既有参数向量）；
2. **设了续训 ⇒ argv 真的带上该路径**（修的就是这一条）；
3. **路径不成立 ⇒ 启动前报错**（不存在／不是目录／没有 ``trainer_state.json``）——
   静默忽略是这条 bug 的原始形态；
4. **非法组合 ⇒ 报错**（``overwrite_output_dir: true``、``lora.adapter_path``）。

**单测锁不到的**：续训起始步 = ckpt ``global_step``+1、LR/优化器状态接续、断点前后
loss 连续——这三条要真跑一次 GPU 训练。它们由真机验证脚本覆盖（见工位
``task-resume-wiring/verify/`` 的证据与结论）；本文件在 argparse 层锁定"参数确实被
产出、且只在该产出时产出"，使真机侧只剩"ms-swift 是否消费它"这一件事——后者已在本
模块注释里按 ms-swift 4.5.3 源码核实（``swift/pipelines/train/sft.py:237-263``
``trainer.train(resume_checkpoint)``）。
"""

from __future__ import annotations

import inspect
import json
from pathlib import Path

import pytest

from graspo.core.schema import GraspoConfig
from graspo.flow.backend.msswift._config_mapping import graspo_to_ms_swift_argv
from graspo.flow.backend.msswift.resume import TRAINER_STATE_FILE, validate_resume_checkpoint

#: M003 形状的 SFT 配置（9B / LoRA / 4 卡 / save_steps=20 / 8192 上下文）。
#: 续训场景就是在这张形状上实测出来的，故回归金值也用它。
_M003_LIKE: dict = {
    "train_method": "sft",
    "backend": "msswift",
    "tuner_type": "lora",
    "model": {
        "model_path": "/models/Qwen3.5-9B",
        "torch_dtype": "bfloat16",
        "gradient_checkpointing": True,
    },
    "data": {"train_path": "unused.jsonl", "max_prompt_length": 8192},
    "training": {
        "output_dir": "/out/M003r",
        "run_name": "M003r",
        "seed": 42,
        "max_epochs": 4,
        "learning_rate": 2.0e-05,
        "save_steps": 20,
        "save_checkpoint_every_epoch": False,
    },
    "launch": {"nproc_per_node": 4},
    "lora": {
        "r": 32,
        "alpha": 64,
        "dropout": 0.05,
        "target_modules": r"^(?:model\.)?language_model\..*\.(q_proj|k_proj)$",
    },
    "msswift": {
        "nproc_per_node": 4,
        "nnodes": 1,
        "per_device_train_batch_size": 1,
        "sequence_parallel_size": 1,
        "use_vllm": False,
    },
}

_DATASET = "/out/M003r/msswift/train.jsonl"

#: ``training.resume_from_checkpoint`` 接线**之前**（commit eb95c2f 的映射层）对
#: :data:`_M003_LIKE` 产出的**完整参数向量**——逐字节金值。新增映射行若在不设续训时
#: 多产出/少产出/改了顺序任何一个字节，本测试立刻红。
_GOLDEN_ARGV: list[str] = [
    "--model",
    "/models/Qwen3.5-9B",
    "--torch_dtype",
    "bfloat16",
    "--gradient_checkpointing",
    "true",
    "--dataset",
    _DATASET,
    "--max_length",
    "8192",
    "--tuner_type",
    "lora",
    "--lora_rank",
    "32",
    "--lora_alpha",
    "64",
    "--lora_dropout",
    "0.05",
    "--target_regex",
    r"^(?:model\.)?language_model\..*\.(q_proj|k_proj)$",
    "--output_dir",
    "/out/M003r",
    "--run_name",
    "M003r",
    "--seed",
    "42",
    "--num_train_epochs",
    "4",
    "--learning_rate",
    "2e-05",
    "--weight_decay",
    "0.01",
    "--max_grad_norm",
    "1.0",
    "--gradient_accumulation_steps",
    "4",
    "--save_strategy",
    "steps",
    "--save_steps",
    "20",
    "--sequence_parallel_size",
    "1",
    "--packing",
    "false",
    "--padding_free",
    "false",
    "--use_liger_kernel",
    "false",
    "--use_logits_to_keep",
    "false",
    "--per_device_train_batch_size",
    "1",
]


def _config(**overrides: object) -> GraspoConfig:
    """``_M003_LIKE`` + 段级覆盖（``training=...`` / ``lora=...``）。"""
    payload = json.loads(json.dumps(_M003_LIKE))
    for section, values in overrides.items():
        if values is None:
            payload.pop(section, None)
        elif isinstance(values, dict):
            payload.setdefault(section, {}).update(values)
        else:
            payload[section] = values
    return GraspoConfig.model_validate(payload)


def _argv(config: GraspoConfig) -> list[str]:
    return graspo_to_ms_swift_argv(
        config, stage="sft", dataset_path=_DATASET, output_dir="/out/M003r"
    )


def _checkpoint_dir(tmp_path: Path, *, with_state: bool = True) -> Path:
    """造一个 checkpoint-40 形状的目录（有/没有 ``trainer_state.json``）。"""
    ckpt = tmp_path / "checkpoint-40"
    ckpt.mkdir()
    (ckpt / "adapter_model.safetensors").write_bytes(b"")
    (ckpt / "optimizer.pt").write_bytes(b"")
    (ckpt / "scheduler.pt").write_bytes(b"")
    if with_state:
        (ckpt / TRAINER_STATE_FILE).write_text(
            json.dumps({"global_step": 40, "epoch": 0.8}), encoding="utf-8"
        )
    return ckpt


# ── 1. 防回归：不设续训 ⇒ argv 逐字节不变 ─────────────────────────────────


def test_argv_is_byte_identical_when_resume_is_not_configured():
    """不设 ``resume_from_checkpoint`` ⇒ argv 与接线前**逐字节相同**（52 个元素）。"""
    argv = _argv(_config())
    assert argv == _GOLDEN_ARGV
    assert len(argv) == len(_GOLDEN_ARGV) == 52
    assert "--resume_from_checkpoint" not in argv


# ── 2. 接线：设了续训 ⇒ argv 带上该参数与正确路径 ─────────────────────────


def test_resume_checkpoint_is_mapped_to_swift_argv(tmp_path):
    """``training.resume_from_checkpoint`` ⇒ ``--resume_from_checkpoint <path>``。"""
    ckpt = _checkpoint_dir(tmp_path)
    argv = _argv(_config(training={"resume_from_checkpoint": str(ckpt)}))

    assert argv.count("--resume_from_checkpoint") == 1
    index = argv.index("--resume_from_checkpoint")
    assert argv[index + 1] == str(ckpt)
    # 唯一差异就是这一对元素——其余参数与金值一致（不多不少）。
    assert argv[:index] + argv[index + 2 :] == _GOLDEN_ARGV


def test_resume_checkpoint_value_is_not_gated_by_stage(tmp_path):
    """四个通道（sft/rlhf/cpt/opd）都从同一处映射出该参数（§1.4 单一真相源）。"""
    ckpt = _checkpoint_dir(tmp_path)
    config = _config(training={"resume_from_checkpoint": str(ckpt)})
    for stage in ("sft", "rlhf", "cpt", "opd"):
        argv = graspo_to_ms_swift_argv(
            config, stage=stage, dataset_path=_DATASET, output_dir="/out/M003r"
        )
        assert "--resume_from_checkpoint" in argv, stage
        assert argv[argv.index("--resume_from_checkpoint") + 1] == str(ckpt), stage


# ── 3. fail-closed：路径不成立 ⇒ 报错（⛔ 不许静默忽略 / 退回从头训练）──────


def test_missing_resume_path_fails_closed(tmp_path):
    """路径不存在 ⇒ ``FileNotFoundError``（改前：静默通过，跑到第 1 步才发现）。"""
    missing = tmp_path / "nope" / "checkpoint-40"
    config = _config(training={"resume_from_checkpoint": str(missing)})

    with pytest.raises(FileNotFoundError, match="resume_from_checkpoint does not exist"):
        validate_resume_checkpoint(config)


def test_resume_path_that_is_a_file_fails_closed(tmp_path):
    """路径存在但不是目录 ⇒ ``NotADirectoryError``。"""
    file_path = tmp_path / TRAINER_STATE_FILE
    file_path.write_text("{}", encoding="utf-8")
    config = _config(training={"resume_from_checkpoint": str(file_path)})

    with pytest.raises(NotADirectoryError, match="is not a directory"):
        validate_resume_checkpoint(config)


def test_resume_dir_without_trainer_state_fails_closed(tmp_path):
    """目录里没有 ``trainer_state.json`` ⇒ ``RuntimeError``（无可恢复的步数/优化器状态）。"""
    ckpt = _checkpoint_dir(tmp_path, with_state=False)
    config = _config(training={"resume_from_checkpoint": str(ckpt)})

    with pytest.raises(RuntimeError, match=TRAINER_STATE_FILE):
        validate_resume_checkpoint(config)


# ── 4. fail-closed：非法组合 ⇒ 报错 ───────────────────────────────────────


def test_resume_conflicts_with_overwrite_output_dir(tmp_path):
    """与 ``overwrite_output_dir: true`` 并存 ⇒ 报错（那会删掉要续训的目录）。"""
    ckpt = _checkpoint_dir(tmp_path)
    config = _config(training={"resume_from_checkpoint": str(ckpt), "overwrite_output_dir": True})

    with pytest.raises(ValueError, match="overwrite_output_dir"):
        validate_resume_checkpoint(config)


def test_resume_conflicts_with_lora_adapter_path(tmp_path):
    """与 ``lora.adapter_path`` 并存 ⇒ 报错（与 native 侧同一条规则、同一文案）。"""
    ckpt = _checkpoint_dir(tmp_path)
    config = _config(
        training={"resume_from_checkpoint": str(ckpt)},
        lora={"adapter_path": str(tmp_path / "adapter")},
    )

    with pytest.raises(ValueError, match="cannot both be set"):
        validate_resume_checkpoint(config)


# ── 白路径与接线契约 ──────────────────────────────────────────────────────


def test_recoverable_checkpoint_passes_and_is_not_rewritten(tmp_path):
    """合格 checkpoint ⇒ 返回路径本身、且**不动**目录内容（只读校验）。"""
    ckpt = _checkpoint_dir(tmp_path)
    before = sorted(p.name for p in ckpt.iterdir())

    config = _config(training={"resume_from_checkpoint": str(ckpt)})
    assert validate_resume_checkpoint(config) == str(ckpt)
    assert sorted(p.name for p in ckpt.iterdir()) == before


def test_no_resume_request_is_a_noop():
    """不设续训 ⇒ 守卫什么都不做（返回 ``None``、不碰文件系统）。"""
    assert validate_resume_checkpoint(_config()) is None


@pytest.mark.parametrize("module", ["sft_trainer", "trainer", "cpt_trainer", "opd_trainer"])
def test_every_msswift_trainer_invokes_the_resume_gate(module):
    """四条训练通道都在 ``validate_combinations`` 之后调用续训守卫（接线不漏）。

    这条是**防漂移**契约：守卫只有一处实现（``resume.py``），四个调用点各一行；
    若将来新增通道忘了接，这里立刻红（否则会重演"配置能写、后端不生效"）。
    """
    from graspo.flow.backend.msswift import (
        cpt_trainer,
        opd_trainer,
        sft_trainer,
        trainer,
    )

    source = inspect.getsource(
        {
            "sft_trainer": sft_trainer,
            "trainer": trainer,
            "cpt_trainer": cpt_trainer,
            "opd_trainer": opd_trainer,
        }[module]
    )
    assert "validate_resume_checkpoint(self.config)" in source
    assert source.index("validate_combinations(self.config)") < source.index(
        "validate_resume_checkpoint(self.config)"
    )
