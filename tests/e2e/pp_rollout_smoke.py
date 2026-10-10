#!/usr/bin/env python
"""PP rollout 冒烟 —— 真 NCCL · ``pp_size=2`` · 文本-only · **有界墙钟**。

**它存在的理由**（缺陷 P6）：``native + pp_size>1 + 有 rollout`` 这个组合在 54 档
矩阵里只有 T035/T036 命中，且**从未有一档端到端通过**；唯一的自动覆盖是纯 CPU
mock（``tests/flow/backend/native/parallel/test_pipeline_comm.py``）。真机复现一次失败要
**3–18 分钟**（首 prefill 15 s + 无界挂死），而本冒烟把同一段 PP 生成路径压到
**20–60 秒**（``prompt_count=1``、``rollout_group_size=1``、``max_new_tokens=2``、
``max_prompt_length=256``），并且**自带墙钟上限**——绝不再出现"挂死白耗卡"。

用法（在**容器内**、**需要 2 张可见 GPU**）：

    # 干跑：只打印将要执行的 torchrun 命令（不需要 GPU，可用来核对参数）
    python tests/e2e/pp_rollout_smoke.py --dry-run

    # 真跑（默认读 samples/configs/matrix54/T035.yaml 的模型路径与数据口径）
    python tests/e2e/pp_rollout_smoke.py \
        --config samples/configs/matrix54/T035.yaml \
        --model /models/Qwen3.5-9B \
        --out /tmp/pp-rollout-smoke \
        --timeout-sec 300

退出码（唯一真相源，本文件常量）：

* ``0``   —— 冒烟通过（两端都返回且序列一致）；
* ``21``  —— **墙钟上限命中**（被本脚本自己杀掉）；这正是"无界挂死"的有界形态；
* 其它    —— torchrun/子进程的退出码原样透传（例如具名 ``PipelineP2PTimeoutError``
  经 ``graspo.flow.backend.native.trainer`` 的异常路径退出）。

**这不是档位、不是档位生成器的一部分**：它只服务 PP rollout 的可重复验证，
不写 `docs/`、不改 `samples/configs/matrix54/*.yaml`。
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

#: 墙钟上限命中时的退出码（唯一真相源；与 ``rendezvous_watchdog.EXIT_CODE_NO_PROGRESS``
#: 刻意区分：那个是"会合点无进展"，这个是"父进程的墙钟兜底"）。
WALL_CLOCK_EXIT_CODE = 21

#: 本脚本默认的 torchrun 每机进程数 = 2（= ``pp_size``），与冒烟目标一致。
_LAUNCHER_NPROC = 2

#: 冒烟提示词基句（``--prompt-repeat`` 会把它重复 R 次，用来把 prompt 拉到指定规模）。
BASE_PROMPT = "Reply with exactly: OK. "

_REPO_ROOT = Path(__file__).resolve().parents[2]


def build_smoke_prompts(
    *, prompt_count: int, prompt_repeat: int
) -> tuple[list[list[dict[str, str]]], list[None]]:
    """构造 N 条相同（可选加长）的文本提示词 + 对应 tool 批次（**纯逻辑，可单测**）。

    ``prompt_repeat`` 是"规模旋钮"：T035 的实测 prompt 长度是 2382 tokens、batch=64
    （``rollout_group_size=8`` × 8 prompts）；而 P6 的挂死**只在大规模出现**
    （最小冒烟 17 tokens × 1 未复现）。所以这个旋钮是"从 20 秒到 T035 规模"之间的
    连续可调复现手段；缺省 1 = 只跑最小冒烟。
    """
    if prompt_count < 1:
        raise ValueError(f"prompt_count must be >= 1, got {prompt_count}")
    if prompt_repeat < 1:
        raise ValueError(f"prompt_repeat must be >= 1, got {prompt_repeat}")
    prompt = [{"role": "user", "content": BASE_PROMPT * int(prompt_repeat)}]
    return [list(prompt) for _ in range(int(prompt_count))], [None] * int(prompt_count)


def build_smoke_config(
    *,
    config_path: str | Path,
    model_path: str | None,
    out_dir: str | Path,
    max_prompt_length: int,
    max_new_tokens: int,
    rollout_group_size: int,
    p2p_timeout_sec: int,
    no_progress_sec: int,
) -> Any:
    """把档位 YAML 收窄成"文本-only · pp=2 · 极小规模"的冒烟配置（**纯逻辑**）。

    所有覆盖都在**内存里**做：既不写回 YAML、也不新增档位。三项不可省的覆盖：

    * ``native.pp_size=2``（本冒烟要复现的正是 PP 路径）；
    * ``native.allow_unverified_pp_rollout=True``（**显式预授权**：这正是该开关的
      用途——"我知道它未验证、仍要跑"，而不是 SKIP/静默跳过）；
    * 极小规模 + 有界超时（``max_new_tokens=2`` / ``max_prompt_length=256`` /
      ``pp_rollout_p2p_timeout_sec`` / ``pp_rollout_no_progress_sec``）。
    """
    import yaml

    from graspo.core.schema import GraspoConfig

    raw: dict[str, Any] = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    native = dict(raw.get("native") or {})
    native.update(
        {
            "tp_size": 1,
            "dp_size": 1,
            "pp_size": _LAUNCHER_NPROC,
            "allow_unverified_pp_rollout": True,
            "pp_rollout_p2p_timeout_sec": int(p2p_timeout_sec),
            "pp_rollout_no_progress_sec": int(no_progress_sec),
        }
    )
    raw["native"] = native
    training = dict(raw.get("training") or {})
    training.update(
        {
            "output_dir": str(out_dir),
            "rollout_group_size": int(rollout_group_size),
            "max_new_tokens": int(max_new_tokens),
            "max_epochs": 1,
            "save_checkpoint_every_epoch": False,
        }
    )
    raw["training"] = training
    data = dict(raw.get("data") or {})
    data["max_prompt_length"] = int(max_prompt_length)
    raw["data"] = data
    if model_path:
        model = dict(raw.get("model") or {})
        model["model_path"] = str(model_path)
        raw["model"] = model
    return GraspoConfig.model_validate(raw)


def build_torchrun_command(args: argparse.Namespace, *, self_path: Path) -> list[str]:
    """构造 torchrun 命令（``--standalone`` + 显式 ``--master_port``，避免撞端口）。"""
    return [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nproc_per_node",
        str(_LAUNCHER_NPROC),
        "--master_port",
        str(_free_port()),
        str(self_path),
        "--worker",
        "--config",
        str(args.config),
        "--out",
        str(args.out),
        "--max-prompt-length",
        str(args.max_prompt_length),
        "--max-new-tokens",
        str(args.max_new_tokens),
        "--seed",
        str(args.seed),
        "--rollout-group-size",
        str(args.rollout_group_size),
        "--prompt-count",
        str(args.prompt_count),
        "--prompt-repeat",
        str(args.prompt_repeat),
        "--p2p-timeout-sec",
        str(args.p2p_timeout_sec),
        "--no-progress-sec",
        str(args.no_progress_sec),
        *(["--model", str(args.model)] if args.model else []),
    ]


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _run_worker(args: argparse.Namespace) -> int:
    """rank 侧：建 native runtime → 跑一次极小 PP rollout → 落结果。"""
    import torch.distributed as dist

    from graspo.flow.logging import run_log_id, set_run_id
    from graspo.flow.backend.native.runtime import GraspoFlowRuntime

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    config = build_smoke_config(
        config_path=args.config,
        model_path=args.model,
        out_dir=out_dir,
        max_prompt_length=args.max_prompt_length,
        max_new_tokens=args.max_new_tokens,
        rollout_group_size=args.rollout_group_size,
        p2p_timeout_sec=args.p2p_timeout_sec,
        no_progress_sec=args.no_progress_sec,
    )
    # 与真实入口（`cli/train_worker.main`）同源：run id 由**配置**推导 ⇒ 所有 rank
    # 落到同一个 `logs/<run_id>/`。缺这一步时各 rank 会各用进程本地时间戳，
    # 探针日志与看门狗转储就被劈成两个目录（首次实测踩到，证据收集很不方便）。
    set_run_id(run_log_id(config.training.run_name))
    # 采样可复现（§11 测试）：`_next_token_from_logits` 在 temperature>0 时走
    # `torch.multinomial` ⇒ 不设种子则每次 token 不同，无法做跨 run 对比。
    import torch as _torch

    _torch.manual_seed(int(args.seed))
    if _torch.cuda.is_available():
        _torch.cuda.manual_seed_all(int(args.seed))
    runtime = GraspoFlowRuntime.from_config(config)
    runtime.validate()
    runtime.setup()

    rank = int(os.environ.get("RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    message_batches, tool_batches = build_smoke_prompts(
        prompt_count=int(args.prompt_count), prompt_repeat=int(args.prompt_repeat)
    )
    started = time.monotonic()
    generations = runtime.generate_groups(
        message_batches=message_batches,
        tool_batches=tool_batches,
        rollout_group_size=int(args.rollout_group_size),
        max_new_tokens=int(args.max_new_tokens),
        max_prompt_length=int(args.max_prompt_length),
        temperature=1.0,
        top_p=1.0,
    )
    elapsed = time.monotonic() - started
    if len(generations) != len(message_batches):
        raise RuntimeError(
            f"PP rollout smoke expected {len(message_batches)} generation groups, "
            f"got {len(generations)}"
        )
    generation = generations[0]
    sequences = generation.sequences
    sequence_token_ids = sequences.tolist() if hasattr(sequences, "tolist") else list(sequences)
    # b1 的显存读数：峰值分配（GiB）。修前 T035 形状在这里要 60–93 GiB 并直接 OOM；
    # 修后（只算末位）应回到"模型 + 激活"的正常量级。落盘是为了让每次验证都带证据。
    peak_allocated_gib = None
    try:
        import torch

        if torch.cuda.is_available():
            # 本冒烟是 2 进程 / 2 卡，local_rank == rank；显式指定设备避免读到别的卡。
            peak_allocated_gib = round(
                float(torch.cuda.max_memory_allocated(torch.device(f"cuda:{rank}"))) / 1024**3,
                2,
            )
    except (ImportError, RuntimeError, ValueError) as exc:
        print(f"[pp-rollout-smoke] rank={rank} peak-memory read failed: {exc}", flush=True)
    record = {
        "rank": rank,
        "world_size": world,
        "prompt_count": int(args.prompt_count),
        "prompt_repeat": int(args.prompt_repeat),
        "rollout_group_size": int(args.rollout_group_size),
        "elapsed_sec": round(elapsed, 2),
        "sequence_token_ids": sequence_token_ids,
        "completions": list(generation.completions),
        "prompt_len": int(generation.prompt_len),
        "peak_allocated_gib": peak_allocated_gib,
        "seed": int(args.seed),
    }
    (out_dir / f"smoke.rank{rank}.json").write_text(
        json.dumps(record, ensure_ascii=False), encoding="utf-8"
    )
    print(
        f"[pp-rollout-smoke] rank={rank} OK elapsed={elapsed:.2f}s "
        f"completions={record['completions']!r} peak_allocated_gib={peak_allocated_gib}",
        flush=True,
    )
    # 两端都必须产出同一条序列（token 经 pp_group 广播 ⇒ 逐位一致）
    if dist.is_available() and dist.is_initialized():
        dist.barrier()
    if rank == 0 and world > 1:
        others = [
            json.loads((out_dir / f"smoke.rank{peer}.json").read_text(encoding="utf-8"))
            for peer in range(1, world)
        ]
        same = all(other["sequence_token_ids"] == record["sequence_token_ids"] for other in others)
        print(
            f"[pp-rollout-smoke] RANK_CONSISTENT={int(same)} "
            f"(compared {world} ranks; tokens={len(sequence_token_ids)})",
            flush=True,
        )
        if not same:
            raise RuntimeError(
                "PP rollout smoke: ranks disagree on the generated sequence "
                f"(rank0={record['sequence_token_ids'][:8]}..., "
                f"others={[o['sequence_token_ids'][:8] for o in others]})"
            )
    runtime.close()
    return 0


def run_with_wall_clock(
    command: list[str], *, timeout_sec: int, label: str = "pp-rollout-smoke"
) -> int:
    """跑 ``command``，**带墙钟上限**；命中上限就精确回收整个进程组。

    这是"别再让它挂死白耗卡"的落点：父进程的墙钟与容器内的会合点看门狗/有界等待
    **互为独立防线**（前者不依赖被测代码是否还活着）。

    :return: 子进程退出码；命中墙钟上限返回 :data:`WALL_CLOCK_EXIT_CODE`。
    """
    print(f"[{label}] wall-clock cap = {timeout_sec}s", flush=True)
    started = time.monotonic()
    # start_new_session=True ⇒ 子进程独占一个进程组，超时用 killpg 精确回收
    # （不使用 pkill/f -f 这类按命令行匹配的模糊杀法）。
    process = subprocess.Popen(command, start_new_session=True)
    try:
        return int(process.wait(timeout=timeout_sec))
    except subprocess.TimeoutExpired:
        print(
            f"[{label}] WALL-CLOCK CAP HIT after {time.monotonic() - started:.0f}s "
            f"⇒ killing process group {process.pid} (exit {WALL_CLOCK_EXIT_CODE}); "
            "this is the bounded form of the pre-P6 unbounded hang",
            file=sys.stderr,
            flush=True,
        )
        try:
            os.killpg(os.getpgid(process.pid), signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            process.kill()
        process.wait(timeout=60)
        return WALL_CLOCK_EXIT_CODE


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default=str(_REPO_ROOT / "samples" / "configs" / "matrix54" / "T035.yaml"),
        help="源档位 YAML（只读；覆盖在内存里做，不写回）",
    )
    parser.add_argument(
        "--model", default=None, help="覆盖 model.model_path（如 /models/Qwen3.5-9B）"
    )
    parser.add_argument("--out", default="/tmp/pp-rollout-smoke", help="产物目录")
    parser.add_argument("--max-prompt-length", type=int, default=256)
    parser.add_argument("--max-new-tokens", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0, help="采样种子（可复现对比用）")
    parser.add_argument(
        "--prompt-count",
        type=int,
        default=1,
        help="prompt 条数（T035 规模 = 8；batch = prompt_count × rollout_group_size）",
    )
    parser.add_argument(
        "--prompt-repeat",
        type=int,
        default=1,
        help="把基句重复 R 次以拉长 prompt（T035 规模约需 R≈340 ⇒ 约 2000 tokens）",
    )
    parser.add_argument("--rollout-group-size", type=int, default=1)
    parser.add_argument(
        "--p2p-timeout-sec",
        type=int,
        default=120,
        help="PP 会合点有界等待（秒）；冒烟取比档位默认 600 更紧的值以便快速判定",
    )
    parser.add_argument(
        "--no-progress-sec",
        type=int,
        default=120,
        help="PP rollout 无进展看门狗阈值（秒）",
    )
    parser.add_argument(
        "--timeout-sec",
        type=int,
        default=300,
        help="**父进程墙钟上限**（秒）；命中即杀掉整个进程组并返回 21",
    )
    parser.add_argument("--dry-run", action="store_true", help="只打印将执行的命令")
    parser.add_argument(
        "--worker", action="store_true", help="内部：作为 torchrun 的 rank 进程运行"
    )
    args = parser.parse_args(argv)

    if args.worker:
        return _run_worker(args)

    command = build_torchrun_command(args, self_path=Path(__file__).resolve())
    if args.dry_run:
        print(" ".join(command))
        return 0

    return run_with_wall_clock(command, timeout_sec=int(args.timeout_sec))


if __name__ == "__main__":
    raise SystemExit(main())
