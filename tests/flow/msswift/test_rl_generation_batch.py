"""GRPO rollout 批次与 ``num_generations`` 的相容性（阻断 2 的回归测试）。

**背景（实测 T031，2026-09-19）**：清单档 ``per_device_train_batch_size=1``、
``rollout_group_size=8``（graspo 默认），旧实现硬写 ``--steps_per_generation 1``
且从不派生 ``--generation_batch_size``，ms-swift 4.5.3 于是在**参数解析期**抛

    ValueError: generation_batch_size (1) must be evenly divisible by
    num_generations (8). Valid values: [].

本文件锁三件事：

1. :func:`graspo_to_ms_swift_argv` 下发的 ``(generation_batch_size,
   steps_per_generation)`` **能过**上游 ``_init_generation_batch_params`` 的检查
   （把上游算法逐行搬来当判据，不另立一套）；
2. 旧组合（gbs 未派生 ⇒ 上游取 1、spg=1）在同一段上游算法上**必炸**——
   即这条判据真的会失败，不是永真断言；
3. 不相容的组合在**启动前**就被 :func:`validate_combinations` fail-closed 拦住
   （不是等 ms-swift 起进程）。

上游算法见下方 ``_upstream_init_generation_batch_params``：ms-swift 4.5.3
``swift/rlhf_trainers/args_mixin.py:203-232``
（``RolloutTrainerArgumentsMixin._init_generation_batch_params``）的等价重写，
只去掉 torch/dataclass 依赖；逐句标注对应行号，便于上游升级时比对。
"""

from __future__ import annotations

import math

import pytest

from graspo.core.schema import GraspoConfig
from graspo.flow.msswift._config_mapping import (
    graspo_to_ms_swift_argv,
    validate_combinations,
)


def _value_of(argv: list[str], flag: str) -> str | None:
    return argv[argv.index(flag) + 1] if flag in argv else None


def _rlhf_argv(config: GraspoConfig) -> list[str]:
    return graspo_to_ms_swift_argv(config, stage="rlhf", dataset_path="d.jsonl", output_dir="out")


def _config(**msswift: object) -> GraspoConfig:
    return GraspoConfig.model_validate(
        {
            "backend": "msswift",
            "train_method": "graspo",
            "model": {"model_path": "models/Qwen3.5-9B"},
            "data": {"train_path": "unused.jsonl", "max_prompt_length": 2048},
            "msswift": dict(msswift),
        }
    )


class _UpstreamArgs:
    """喂给上游算法的最小参数载体（只有该算法读的字段）。"""

    def __init__(
        self,
        *,
        per_device_train_batch_size: int = 1,
        world_size: int = 1,
        gradient_accumulation_steps: int = 1,
        num_generations: int = 8,
        generation_batch_size: int | None = None,
        steps_per_generation: int | None = None,
    ) -> None:
        self.per_device_train_batch_size = per_device_train_batch_size
        self.world_size = world_size
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.num_generations = num_generations
        self.generation_batch_size = generation_batch_size
        self.steps_per_generation = steps_per_generation


def _upstream_init_generation_batch_params(args: _UpstreamArgs) -> None:
    """ms-swift 4.5.3 ``args_mixin.py:203-232`` 的等价重写（去掉 dataclass 外壳）。

    行号对应 ``.local/refs/ms-swift-4.5.3/src/swift/rlhf_trainers/args_mixin.py``。
    """
    num_generations = getattr(args, "num_generations", 1)  # :204
    num_processes = args.world_size  # :205
    global_batch_size = args.per_device_train_batch_size * num_processes  # :206

    if args.generation_batch_size is None and args.steps_per_generation is None:  # :208
        args.steps_per_generation = args.gradient_accumulation_steps  # :209
        args.generation_batch_size = global_batch_size * args.steps_per_generation  # :210
    elif args.generation_batch_size is not None and args.steps_per_generation is None:  # :211
        if args.generation_batch_size % global_batch_size != 0:  # :212
            raise ValueError(
                f"generation_batch_size ({args.generation_batch_size}) must be divisible by "
                f"the global batch size ({global_batch_size})."
            )
        args.steps_per_generation = args.generation_batch_size // global_batch_size  # :215
    elif args.generation_batch_size is None and args.steps_per_generation is not None:  # :216
        args.generation_batch_size = global_batch_size * args.steps_per_generation  # :217
    else:  # :219
        expected = global_batch_size * args.steps_per_generation  # :220
        if args.generation_batch_size != expected:  # :221
            raise ValueError(
                f"generation_batch_size ({args.generation_batch_size}) must equal "
                f"per_device_train_batch_size * world_size * steps_per_generation = {expected}."
            )

    if args.steps_per_generation <= 0:  # :224
        raise ValueError(f"steps_per_generation must be > 0, got {args.steps_per_generation}.")

    if num_generations > 1:  # :227
        if args.generation_batch_size % num_generations != 0:  # :229
            possible_values = [  # :230
                n
                for n in range(2, args.generation_batch_size + 1)
                if args.generation_batch_size % n == 0
            ]
            raise ValueError(  # :233
                f"generation_batch_size ({args.generation_batch_size}) must be evenly "
                f"divisible by num_generations ({num_generations}). "
                f"Valid values: {possible_values}."
            )


def _feed_graspo_argv_to_upstream(config: GraspoConfig) -> _UpstreamArgs:
    """把 graspo **实际下发**的 argv 喂进上游算法（模拟 ms-swift 参数解析）。"""
    argv = _rlhf_argv(config)
    world_size = max(1, int(config.msswift.nproc_per_node or 1)) * max(
        1, int(config.msswift.nnodes or 1)
    )
    args = _UpstreamArgs(
        per_device_train_batch_size=max(1, int(config.msswift.per_device_train_batch_size or 1)),
        world_size=world_size,
        gradient_accumulation_steps=int(config.training.gradient_accumulation_micro_batches),
        num_generations=int(_value_of(argv, "--num_generations") or 0),
        generation_batch_size=int(_value_of(argv, "--generation_batch_size") or 0),
        steps_per_generation=int(_value_of(argv, "--steps_per_generation") or 0),
    )
    _upstream_init_generation_batch_params(args)
    return args


class TestUpstreamAlgorithmDifferential:
    """graspo 下发的对必须过上游算法（同一段判据，不做两套）。"""

    @pytest.mark.parametrize("rollout_group_size", [1, 2, 4, 8])
    @pytest.mark.parametrize("per_device_batch,world_size", [(1, 1), (1, 2), (2, 2), (4, 1)])
    def test_emitted_pair_passes_upstream_check(
        self, rollout_group_size: int, per_device_batch: int, world_size: int
    ) -> None:
        config = GraspoConfig.model_validate(
            {
                "backend": "msswift",
                "train_method": "graspo",
                "training": {"rollout_group_size": rollout_group_size},
                "msswift": {
                    "per_device_train_batch_size": per_device_batch,
                    "nproc_per_node": world_size,
                },
            }
        )
        argv = _rlhf_argv(config)
        args = _feed_graspo_argv_to_upstream(config)  # 不抛 = 上游相容
        assert args.generation_batch_size % max(1, args.num_generations) == 0
        # 与上游"两个都提供"分支的自洽式逐字一致
        global_batch_size = per_device_batch * world_size
        assert args.generation_batch_size == global_batch_size * args.steps_per_generation
        assert _value_of(argv, "--generation_batch_size") == str(args.generation_batch_size)
        assert _value_of(argv, "--steps_per_generation") == str(args.steps_per_generation)

    def test_t031_shape_regression_before_and_after(self) -> None:
        """T031 形状（pdb=1, world=1, G=8）：修复后 gbs=8/spg=8；旧的 None/1 必炸。"""
        config = _config(per_device_train_batch_size=1, nproc_per_node=1)
        args = _feed_graspo_argv_to_upstream(config)
        assert (args.generation_batch_size, args.steps_per_generation) == (8, 8)

        legacy = _UpstreamArgs(  # 修复前：gbs 从不派生 + spg 硬写 1
            per_device_train_batch_size=1,
            world_size=1,
            gradient_accumulation_steps=1,
            num_generations=8,
            generation_batch_size=None,
            steps_per_generation=1,
        )
        with pytest.raises(ValueError, match=r"must be evenly divisible by num_generations \(8\)"):
            _upstream_init_generation_batch_params(legacy)
        # 与 228 上 T031 实测报错同一条：gbs=1、ng=8、Valid values: []
        assert legacy.generation_batch_size == 1
        assert [
            n
            for n in range(2, legacy.generation_batch_size + 1)
            if legacy.generation_batch_size % n == 0
        ] == []

    def test_steps_per_generation_covers_one_full_group(self) -> None:
        """gbs 必须"至少覆盖一个完整 GRPO 组"，否则组内 advantage 会被切碎。"""
        for pdb, world in ((1, 1), (2, 1), (1, 2), (2, 2), (4, 2)):
            config = GraspoConfig.model_validate(
                {
                    "backend": "msswift",
                    "train_method": "graspo",
                    "training": {"rollout_group_size": 8},
                    "msswift": {"per_device_train_batch_size": pdb, "nproc_per_node": world},
                }
            )
            argv = _rlhf_argv(config)
            gbs = int(_value_of(argv, "--generation_batch_size") or 0)
            ng = int(_value_of(argv, "--num_generations") or 0)
            assert gbs >= ng, f"pdb={pdb} world={world}: gbs={gbs} < ng={ng}"
            assert gbs % ng == 0
            assert int(_value_of(argv, "--steps_per_generation") or 0) == max(
                1, math.ceil(ng / (pdb * world))
            )

    def test_old_implementation_pair_is_rejected_by_same_upstream_code(self) -> None:
        """★ 负向用例：旧 argv（无 --generation_batch_size，--steps_per_generation 1）。"""
        argv = ["--num_generations", "8", "--steps_per_generation", "1"]
        args = _UpstreamArgs(
            per_device_train_batch_size=1,
            world_size=1,
            gradient_accumulation_steps=1,
            num_generations=int(_value_of(argv, "--num_generations") or 0),
            generation_batch_size=int(_value_of(argv, "--generation_batch_size") or 0)
            if _value_of(argv, "--generation_batch_size")
            else None,
            steps_per_generation=int(_value_of(argv, "--steps_per_generation") or 0),
        )
        with pytest.raises(ValueError, match="must be evenly divisible by num_generations"):
            _upstream_init_generation_batch_params(args)


class TestValidateCombinationsFailClosed:
    """启动前 fail-closed：不相容的组合不该等到 ms-swift 参数解析期。"""

    def test_compatible_config_passes(self) -> None:
        for pdb, world in ((1, 1), (1, 2), (2, 2), (4, 1), (8, 1)):
            validate_combinations(_config(per_device_train_batch_size=pdb, nproc_per_node=world))

    def test_incompatible_config_raises_at_validation_not_at_parsing(self) -> None:
        """pdb=3, world=1, G=8 ⇒ 派生的 gbs=9 不含 G 的整倍 ⇒ 启动前报错。

        该组合**加卡也修不好**（3 卡 ⇒ global=3，gbs=9 仍 %8≠0），与显存无关。
        """
        config = _config(per_device_train_batch_size=3, nproc_per_node=1)
        with pytest.raises(ValueError, match="batch/group mismatch"):
            validate_combinations(config)
        # 同一组合喂给上游算法也必炸——证明确实是上游约束，不是 graspo 自造判据
        args = _UpstreamArgs(
            per_device_train_batch_size=3,
            world_size=1,
            gradient_accumulation_steps=1,
            num_generations=8,
            generation_batch_size=9,
            steps_per_generation=3,
        )
        with pytest.raises(ValueError, match="must be evenly divisible"):
            _upstream_init_generation_batch_params(args)

    def test_error_message_is_actionable(self) -> None:
        config = _config(per_device_train_batch_size=3, nproc_per_node=1)
        with pytest.raises(ValueError) as excinfo:
            validate_combinations(config)
        message = str(excinfo.value)
        for needle in (
            "rollout_group_size=8",
            "generation_batch_size=9",
            "adding cards cannot fix it",
            "msswift.per_device_train_batch_size",
        ):
            assert needle in message, needle

    def test_non_graspo_method_is_not_judged(self) -> None:
        """SFT/CPT/OPD 没有 GRPO 组，不该被这条判据误伤。"""
        sft = GraspoConfig.model_validate(
            {
                "backend": "msswift",
                "train_method": "sft",
                "training": {"rollout_group_size": 8},
                "msswift": {"per_device_train_batch_size": 4, "nproc_per_node": 1},
            }
        )
        validate_combinations(sft)
