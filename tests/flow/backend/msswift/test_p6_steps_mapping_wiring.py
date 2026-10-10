"""★ P6 首项验收：**配置层**证明 `steps_per_generation` 新映射已接线（无需 GPU）。

判据（指挥部 2026-10-11 钉死）：
用 **T044/T045 的真实 yaml** 走一遍推导 ⇒ `steps_per_generation` 由 **4/2 变为 8**，
且**每进程本地行数是 G 的整数倍**（`local_batch_rows % G == 0`，即每进程恰 k 个**整组**）。
拿不到这条 ⇒ 不许说接线完成。
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest
import yaml

from graspo.core.schema import GraspoConfig
from graspo.flow.backend.msswift._config_mapping import (
    graspo_to_ms_swift_argv,
    validate_combinations,
)
from graspo.flow.backend.msswift.group_sharding import (
    groups_per_rank_for,
    local_batch_rows,
    steps_per_generation_for,
)

REPO = Path(__file__).resolve().parents[4]
CASES = {
    "T044": REPO / "samples/configs/matrix54-v2/T044.yaml",
    "T045": REPO / "samples/configs/matrix54-v2/T045.yaml",
}
#: (pdb, world, G) → 旧式 `ceil(G/(pdb×world))`（即"切组"的旧推导）
OLD_EXPECTED = {"T044": 4, "T045": 2}
NEW_EXPECTED = {"T044": 8, "T045": 8}


def _config_from_yaml(path: Path) -> GraspoConfig:
    return GraspoConfig.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))


def _argv_value(argv: list[str], flag: str) -> str | None:
    return argv[argv.index(f"--{flag}") + 1] if f"--{flag}" in argv else None


@pytest.mark.parametrize("case", sorted(CASES))
def test_t044_t045_steps_changed_from_old_to_new(case: str):
    """★★ P6 首项判据：旧 4/2 → 新 8（且每进程 = G 的整数倍 = 1 个整组）。"""
    config = _config_from_yaml(CASES[case])
    section = config.msswift
    pdb = max(1, int(section.per_device_train_batch_size or 1))
    world = max(1, int(section.nproc_per_node or 1)) * max(1, int(section.nnodes or 1))
    g = int(config.training.rollout_group_size)

    old_steps = max(1, math.ceil(g / (pdb * world)))
    new_steps = steps_per_generation_for(pdb, g)
    assert old_steps == OLD_EXPECTED[case], f"{case}: 旧式应给 {OLD_EXPECTED[case]}，得到 {old_steps}"
    assert new_steps == NEW_EXPECTED[case], f"{case}: 新式应给 {NEW_EXPECTED[case]}，得到 {new_steps}"
    assert old_steps * pdb < g, f"{case}: 旧式每进程 {old_steps * pdb} < G={g} ⇒ 会切组（反证）"

    # ★ 每进程本地行数是 G 的整数倍（V1）
    rows = local_batch_rows(pdb, g, new_steps)
    assert rows % g == 0, f"{case}: 每进程 {rows} 行不是 G={g} 的整数倍 ⇒ 仍会切组"
    assert groups_per_rank_for(pdb, g, new_steps) == rows // g


@pytest.mark.parametrize("case", sorted(CASES))
def test_emitted_argv_uses_the_new_mapping(case: str):
    """★ 端到端接线证据：真正下发的 argv 里就是新值（不是只有模块函数对）。"""
    config = _config_from_yaml(CASES[case])
    argv = graspo_to_ms_swift_argv(config, stage="rlhf", dataset_path="ds", output_dir="out")
    assert _argv_value(argv, "steps_per_generation") == str(NEW_EXPECTED[case])
    pdb = max(1, int(config.msswift.per_device_train_batch_size or 1))
    world = max(1, int(config.msswift.nproc_per_node or 1))
    assert _argv_value(argv, "generation_batch_size") == str(
        pdb * world * NEW_EXPECTED[case]
    )
    assert _argv_value(argv, "num_generations") == str(int(config.training.rollout_group_size))


def test_validate_accepts_t044_t045_under_new_mapping():
    """校验侧与发送侧同源：T044/T045 在新映射下必须通过（不抛）。"""
    for case, path in CASES.items():
        validate_combinations(_config_from_yaml(path))


@pytest.mark.parametrize("pdb", [1, 2, 3, 4, 8])
@pytest.mark.parametrize("g", [1, 2, 4, 8, 16])
def test_new_mapping_always_gives_whole_groups(pdb: int, g: int):
    """新映射的构造性保证：任意 (per_device, G) 下每进程行数恒为 G 的整数倍。"""
    steps = steps_per_generation_for(pdb, g)
    rows = local_batch_rows(pdb, g, steps)
    assert rows % g == 0
    assert steps == g // math.gcd(pdb, g)


def test_old_mapping_could_violate_the_invariant():
    """★ 反证：旧式在 T044 形状下**违反** V1（这是本阶段要修的根因）。"""
    pdb, world, g = 1, 2, 8
    old_steps = max(1, math.ceil(g / (pdb * world)))
    assert (pdb * old_steps) % g != 0
