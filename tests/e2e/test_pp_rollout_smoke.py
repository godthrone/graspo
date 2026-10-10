"""``tests/e2e/pp_rollout_smoke.py``（PP rollout 真 NCCL 冒烟）的测试外壳。

分两层：

* **本机可跑**（无 GPU、无模型）：配置收窄逻辑、torchrun 命令构造、**墙钟上限**；
* **需要 2 卡 + 真模型**（``@pytest.mark.gpu`` + ``skipif``）：把冒烟脚本整体跑一遍，
  退出码必须是 0。这条是"PP+rollout 覆盖从 0 到 1"的自动入口——在容器里
  （``GRASPO_SMOKE_MODEL_PATH`` 指向真模型 + 2 张可见卡）会自动执行；
  在本机/CI 无卡环境自动跳过。
"""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from types import ModuleType

import pytest

pytest.importorskip("torch", reason="PP rollout 冒烟走 native runtime（依赖 torch）")

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SMOKE_SCRIPT = _REPO_ROOT / "tests" / "e2e" / "pp_rollout_smoke.py"
_T035_YAML = _REPO_ROOT / "samples" / "configs" / "matrix54" / "T035.yaml"

#: 真模型路径（容器内默认挂载点）；缺省时"真跑"用例自动跳过。
_MODEL_PATH = Path(os.environ.get("GRASPO_SMOKE_MODEL_PATH", "/models/Qwen3.5-9B"))


def _load_smoke_module() -> ModuleType:
    """按文件路径加载冒烟脚本（``tests/`` 不是 package，与仓内既有做法一致）。"""
    spec = importlib.util.spec_from_file_location("_pp_rollout_smoke_under_test", _SMOKE_SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.skipif(not _T035_YAML.exists(), reason="仓库内没有 T035 档位 YAML")
def test_config_is_shrunk_to_the_agreed_smoke_shape(tmp_path: Path) -> None:
    """冒烟配置必须**恰好**是约定形状：pp=2、文本-only、极小规模、显式预授权。"""
    module = _load_smoke_module()
    cfg = module.build_smoke_config(
        config_path=_T035_YAML,
        model_path=None,
        out_dir=tmp_path,
        max_prompt_length=256,
        max_new_tokens=2,
        rollout_group_size=1,
        p2p_timeout_sec=120,
        no_progress_sec=120,
    )
    assert cfg.native.pp_size == 2
    assert cfg.native.tp_size == 1 and cfg.native.dp_size == 1
    # 关键：显式预授权（否则 P6 的 c1 闸门会在启动期拒绝这一个组合）
    assert cfg.native.allow_unverified_pp_rollout is True
    assert cfg.training.rollout_group_size == 1
    assert cfg.training.max_new_tokens == 2
    assert cfg.data.max_prompt_length == 256
    assert str(cfg.training.output_dir) == str(tmp_path)
    assert cfg.native.pp_rollout_p2p_timeout_sec == 120
    assert cfg.native.pp_rollout_no_progress_sec == 120


@pytest.mark.skipif(not _T035_YAML.exists(), reason="仓库内没有 T035 档位 YAML")
def test_smoke_config_passes_the_startup_gate_because_it_opts_in(tmp_path: Path) -> None:
    module = _load_smoke_module()
    from graspo.flow.backend.native.runtime import validate_native_runtime_config

    cfg = module.build_smoke_config(
        config_path=_T035_YAML,
        model_path=None,
        out_dir=tmp_path,
        max_prompt_length=256,
        max_new_tokens=2,
        rollout_group_size=1,
        p2p_timeout_sec=120,
        no_progress_sec=120,
    )
    validate_native_runtime_config(cfg)  # 不抛：闸门被显式预授权打开


def test_dry_run_prints_a_bounded_torchrun_command() -> None:
    proc = subprocess.run(
        [sys.executable, str(_SMOKE_SCRIPT), "--dry-run", "--timeout-sec", "300"],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
    command = proc.stdout.strip()
    assert "torch.distributed.run" in command
    assert "--nproc_per_node 2" in command
    assert "--master_port" in command
    assert "--worker" in command
    assert "--max-new-tokens 2" in command
    assert "--rollout-group-size 1" in command
    assert "--max-prompt-length 256" in command


def test_prompt_scale_knobs_build_the_t035_like_shape() -> None:
    """规模旋钮：``prompt_count=8`` × ``rollout_group_size=8`` = T035 的 batch=64；
    ``prompt_repeat`` 把 prompt 拉长（P6 的挂死**只在大规模出现**，所以必须有旋钮）。"""
    module = _load_smoke_module()
    batches, tools = module.build_smoke_prompts(prompt_count=8, prompt_repeat=340)
    assert len(batches) == 8 and len(tools) == 8
    assert all(len(batch) == 1 for batch in batches)
    longest = max(len(batch[0]["content"]) for batch in batches)
    assert longest == len(module.BASE_PROMPT) * 340
    assert longest > 2000, "340 次重复应给出千字符级 prompt（≈2000 tokens 量级）"

    with pytest.raises(ValueError, match="prompt_count"):
        module.build_smoke_prompts(prompt_count=0, prompt_repeat=1)
    with pytest.raises(ValueError, match="prompt_repeat"):
        module.build_smoke_prompts(prompt_count=1, prompt_repeat=0)


def test_dry_run_exposes_the_scale_knobs() -> None:
    proc = subprocess.run(
        [
            sys.executable,
            str(_SMOKE_SCRIPT),
            "--dry-run",
            "--prompt-count",
            "8",
            "--prompt-repeat",
            "340",
            "--max-prompt-length",
            "4096",
        ],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
    command = proc.stdout.strip()
    assert "--prompt-count 8" in command
    assert "--prompt-repeat 340" in command
    assert "--max-prompt-length 4096" in command


def test_wall_clock_cap_kills_the_child_and_reports_its_own_exit_code() -> None:
    """**有界性**：父进程墙钟命中 ⇒ 精确杀进程组 + 返回具名退出码（不是干等）。"""
    module = _load_smoke_module()
    started = time.monotonic()
    code = module.run_with_wall_clock(
        [sys.executable, "-c", "import time; time.sleep(120)"],
        timeout_sec=3,
        label="test-smoke",
    )
    elapsed = time.monotonic() - started
    assert code == module.WALL_CLOCK_EXIT_CODE == 21
    assert elapsed < 60, f"墙钟兜底必须及时（实际 {elapsed:.1f}s）"


@pytest.mark.gpu
@pytest.mark.skipif(
    not _MODEL_PATH.exists(),
    reason="无真模型（容器内默认 /models/Qwen3.5-9B；可用 GRASPO_SMOKE_MODEL_PATH 覆盖）",
)
def test_pp2_rollout_smoke_runs_end_to_end(tmp_path: Path) -> None:
    """真 NCCL · pp=2 · 文本-only 的端到端冒烟（需要 2 张可见 GPU + 真模型）。"""
    import torch

    if torch.cuda.device_count() < 2:
        pytest.skip("PP rollout 冒烟需要 ≥2 张可见 GPU")
    if not _T035_YAML.exists():
        pytest.skip("仓库内没有 T035 档位 YAML")
    proc = subprocess.run(
        [
            sys.executable,
            str(_SMOKE_SCRIPT),
            "--config",
            str(_T035_YAML),
            "--model",
            str(_MODEL_PATH),
            "--out",
            str(tmp_path / "smoke"),
            "--timeout-sec",
            "600",
        ],
        capture_output=True,
        text=True,
        timeout=660,
        check=False,
    )
    assert proc.returncode == 0, (
        f"PP rollout 冒烟失败：rc={proc.returncode}\n"
        f"stdout 尾部：\n{proc.stdout[-4000:]}\nstderr 尾部：\n{proc.stderr[-4000:]}"
    )
    assert "[pp-rollout-smoke] rank=0 OK" in proc.stdout
    # b1（缺陷 P6）：prefill 必须走"只算末位 logits"这条路径（探针里带位置数）
    assert "logits_positions=1" in proc.stdout, proc.stdout[-4000:]
    # 两端产出必须一致（token 经 pp_group 广播）+ 输出健全
    assert "RANK_CONSISTENT=1" in proc.stdout, proc.stdout[-4000:]
    assert "completions=['OK']" in proc.stdout, proc.stdout[-4000:]
    # b1 的显存读数必须落盘（每次验证都带证据，便于与"整条序列 logits"对比）
    record_path = tmp_path / "smoke" / "smoke.rank0.json"
    assert record_path.exists(), sorted(p.name for p in (tmp_path / "smoke").iterdir())
    record = json.loads(record_path.read_text(encoding="utf-8"))
    assert record["peak_allocated_gib"] is not None
    assert record["seed"] == 0
