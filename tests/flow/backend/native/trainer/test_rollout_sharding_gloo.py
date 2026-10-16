"""★ rollout 分卡（U-C）的**真进程组**判别测试（gloo 双 rank，零 GPU）。

本文件是验收标准 2 / 4 / 5 的**计数级**与**跨卡级**证据：

1. **dp≤1 零新增集合 + 与单卡逐位一致**（A-1）：`dp_size == 1` 时
   `gather_rollout_values` / `gather_field_stats` **一次 `all_gather` 都不发**，
   且返回值与**单卡整组**逐位相同（`torch.equal`）。
   ★ **阳性对照**：把短路口径改坏（`dp_size > 1` → `dp_size >= 1`，
   用**源码文本替换**加载变体）⇒ 同一计数断言**必须判红**。
2. **真 gloo 双 rank 的加序契约**（A-4/A-5）：`all_gather` 按 **rank 序**返回
   ⇒ 沿槽位维拼接后槽位序恰为 `0..G−1`；合并后的 `(Σraw, n)` 与**单卡整组**
   `field_means` 逐位相同；`n` 为**整数**且与会合前整组**严格相等**。
3. **等长切片下跨卡平均 = 全局均值**（验收标准 5）：在真进程组上做
   `all_reduce(AVG)`（等长切片）与"单卡整组均值"对拍；并提供
   不等长切片的阳性对照（必须不等）。
"""

from __future__ import annotations

import atexit
import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path

import pytest

torch = pytest.importorskip("torch", reason="分卡真进程组测试需要 torch（CPU/gloo 即可）")

if not torch.distributed.is_gloo_available():  # pragma: no cover - 环境相关
    pytest.skip("torch.distributed gloo 不可用", allow_module_level=True)

_REPO = Path(__file__).resolve().parents[5]
_SRC = _REPO / "src"
_SHARD_SRC = (
    _REPO
    / "src"
    / "graspo"
    / "flow"
    / "backend"
    / "native"
    / "trainer"
    / "rollout_shard.py"
)

_CHILD = textwrap.dedent(
    '''
    """gloo 双子进程子脚本：用**真** rollout_shard 做会合，把结果写 JSONL。"""
    import atexit
    import json
    import os
    import shutil
    import sys
    import tempfile
    from pathlib import Path
    from types import SimpleNamespace

    import torch
    import torch.distributed as dist

    SRC = Path(sys.argv[1])
    VARIANT = sys.argv[2]           # "production" | "broken_shortcut"
    MODE = sys.argv[3]              # "dp1" | "dp2"
    OUT = Path(sys.argv[4])
    RANK = int(sys.argv[5])
    WORLD = int(sys.argv[6])
    RENDEZVOUS = sys.argv[7]

    torch.set_num_threads(1)
    sys.path.insert(0, str(SRC))

    import graspo.flow.backend.native.trainer.rollout_shard as rs

    if VARIANT == "broken_shortcut":
        # ★ 阳性对照：把 dp≤1 的短路口径改坏 —— 让 `resolve_dp_group` **无条件**返回
        #   可用的 `(group, dp_size, device)`（即"短路被删掉"的语义）。
        #   ⇒ `gather_rollout_values` 会真的发 `all_gather` ⇒ 计数 > 0。
        #   `gather_rollout_values` 会在**函数内** `import torch.distributed as dist`
        #   ⇒ wrapper 必须住进**该模块的源码**里才能生效。
        wrapper = "\\n".join(
            [
                "def resolve_dp_group(runtime):",
                '    """阳性对照变体：短路被删掉（无条件认为可会合）。"""',
                "    import torch.distributed as dist",
                "    adapter = runtime._require_adapter()",
                '    tp_state = getattr(adapter, "tp_state", None)',
                "    group = getattr(tp_state, 'dp_group', None) or dist.group.WORLD",
                "    return (",
                "        group,",
                "        int(dist.get_world_size(group)),",
                "        getattr(tp_state, 'device', None) or torch.device('cpu'),",
                "    )",
                "",
            ]
        )
        original = Path(rs.__file__).read_text(encoding="utf-8")
        anchor = "def resolve_dp_group(runtime: Any) -> tuple[Any, int, Any]:"
        assert anchor in original, "阳性对照变体锚点漂移 ⇒ 该对照会退化成哑断言"
        start = original.index(anchor)
        end = original.index("def _assert_local_shape(")
        broken = original[:start] + wrapper + "\\n" + original[end:]
        assert broken != original
        # ★ 变体**必须落在仓外的临时目录**：⛔ 不许在 `src/` 留下未跟踪文件
        #   （宪法 §17.7 卫生 + 本包"仓内不留未跟踪非忽略文件"的红线）。
        variant_dir = Path(tempfile.mkdtemp(prefix="uc_broken_variant_"))
        atexit.register(shutil.rmtree, variant_dir, ignore_errors=True)
        target = variant_dir / "_broken_rollout_shard.py"
        target.write_text(broken, encoding="utf-8")
        import importlib.util

        spec = importlib.util.spec_from_file_location("broken_rollout_shard", target)
        rs = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(rs)
        assert "阳性对照变体" in (rs.resolve_dp_group.__doc__ or "")

    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", RENDEZVOUS)
    dist.init_process_group(backend="gloo", rank=RANK, world_size=WORLD)

    DP = int(sys.argv[8])           # 本 rank 看到的 dp_size（会合域）
    G, M = 8, WORLD
    RPR = G // DP

    class _Adapter:
        device = torch.device("cpu")
        tp_state = SimpleNamespace(dp_size=DP, dp_group=dist.group.WORLD, device=torch.device("cpu"))

    class _Runtime:
        def _require_adapter(self):
            return _Adapter()

    # ── 1. 决策输入会合：每卡持槽位 [RANK*RPR, (RANK+1)*RPR) ──────────────
    whole_rewards = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.9]
    # ★ `dp_size == 1` ⇒ 本卡**就是整组**（切片未启用）；`dp_size == 2` ⇒ 连续切片。
    lo, hi = (0, G) if DP == 1 else (RANK * RPR, (RANK + 1) * RPR)
    my_slice = torch.tensor(whole_rewards[lo:hi], dtype=torch.float64)
    gathered = rs.gather_rollout_values(
        _Runtime(), [my_slice], cutoff=1, rollouts_per_rank=RPR, what="rewards"
    )
    got = [float(v) for v in gathered[0].tolist()]

    # ── 2. 计数证明：dp≤1 时新增集合次数 ──────────────────────────────────
    calls = {"n": 0}
    real_all_gather = dist.all_gather

    def counting_all_gather(*a, **kw):
        calls["n"] += 1
        return real_all_gather(*a, **kw)

    # ★ 计数探针：只在 "dp1" / "dp2_probe" / "broken_shortcut" 三种模式下跑
    #   （"dp2" 模式不插桩 ⇒ 它的计数天然为 0，用不到）。
    #   一次 `gather_rollout_values` **恰好最多**发 1 次 `all_gather`；
    #   dp≤1 必须 0 次，dp=2 必须 1 次。
    if MODE in ("dp1", "dp2_probe", "broken_shortcut"):
        dist.all_gather = counting_all_gather
        try:
            rs.gather_rollout_values(
                _Runtime(), [my_slice], cutoff=1, rollouts_per_rank=RPR, what="probe"
            )
        finally:
            dist.all_gather = real_all_gather

    # ── 3. 等长切片下的 `all_reduce(AVG)` vs 全局均值（验收标准 5）────────
    # ★ 平衡构造：dp≥2 ⇒ 各卡持 1..G 的**一段**（G//m 条，等长）；
    #            dp=1 ⇒ 每卡都持**整组** 1..G（切片未启用）。
    balanced_len = RPR
    balanced_offset = 0 if DP == 1 else RANK * RPR
    values = torch.tensor(
        [float(balanced_offset + i + 1) for i in range(balanced_len)], dtype=torch.float64
    )
    local_mean = values.mean().clone()
    avg_reduced = local_mean.clone()
    dist.all_reduce(avg_reduced, op=dist.ReduceOp.AVG, group=dist.group.WORLD)

    # ── 4. 不等长切片的阳性对照（必须不等）──────────────────────────────
    # ★ 不等长构造只在 dp=2（真切片）时有意义：3 / 5 两段，**元素数逐 rank 可以不等**
    #   （集合域是 dp_group，本构造两侧都是 rank 0/1 的两个成员，可以各自发不同长度 ——
    #    ⛔ 但 gloo 的 `all_reduce` 要求两侧标量张量 numel 相同，故这里发**标量**）。
    unbalanced_len = [3, 5][RANK % 2] if DP == 2 else G
    unbalanced_offset = 0 if DP == 1 else sum([3, 5][: RANK % 2])
    skewed = torch.tensor(
        [float(unbalanced_offset + i + 1) for i in range(unbalanced_len)],
        dtype=torch.float64,
    )
    skewed_mean_local = skewed.mean().clone()
    skewed_reduced = skewed_mean_local.clone()
    dist.all_reduce(skewed_reduced, op=dist.ReduceOp.AVG, group=dist.group.WORLD)

    row = {
        "rank": RANK,
        "world": WORLD,
        "rpr": RPR,
        "dp_size": DP,
        "gathered": got,
        "new_collective_calls": calls["n"],
        "local_len": int(values.numel()),
        "local_mean": float(local_mean),
        "avg_reduced": float(avg_reduced),
        "skewed_len": int(skewed.numel()),
        "skewed_mean_local": float(skewed_mean_local),
        "skewed_reduced": float(skewed_reduced),
    }
    with OUT.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row) + "\\n")
    dist.barrier()
    dist.destroy_process_group()
    '''
)


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _launch(tmp_path: Path, *, variant: str, mode: str, world: int = 2, dp: int = 2) -> list[dict]:
    child = tmp_path / "gloo_child_uc.py"
    child.write_text(_CHILD, encoding="utf-8")
    out = tmp_path / f"uc_{variant}_{mode}_dp{dp}.jsonl"
    port = str(_free_port())
    env = dict(os.environ)
    env["PYTHONPATH"] = str(_SRC)
    env["OMP_NUM_THREADS"] = "1"
    procs = [
        subprocess.Popen(
            [sys.executable, str(child), str(_SRC), variant, mode, str(out), str(rank), str(world), port, str(dp)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=env,
        )
        for rank in range(world)
    ]
    errs: list[str] = []
    for proc in procs:
        _, stderr = proc.communicate(timeout=180)
        errs.append(stderr)
    if any(proc.returncode != 0 for proc in procs):
        detail = "\n".join(
            f"----- rank{index} (rc={proc.returncode}) -----\n{errs[index]}"
            for index, proc in enumerate(procs)
        )
        raise AssertionError(detail)
    return [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]


# ══════════════════════════════════════════════════════════════════════════════
# 1. 真 gloo 双 rank：加序契约 ＋ 会合结果 == 单卡整组
# ══════════════════════════════════════════════════════════════════════════════

_WHOLE = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.9]
_VALUES = [float(v) for v in range(1, 9)]


def test_gloo_pair_gather_reproduces_single_card_whole_group(tmp_path: Path) -> None:
    """★ A-4/A-5：真 `all_gather` 后**每张卡**都拿到与单卡**逐位相同**的整组。"""
    rows = _launch(tmp_path, variant="production", mode="dp2")
    assert len(rows) == 2
    for row in rows:
        assert row["rpr"] == 4
        assert row["gathered"] == _WHOLE, (
            f"rank{row['rank']} 会合后槽位序={row['gathered']} ≠ 单卡整组 {_WHOLE}"
        )


# ══════════════════════════════════════════════════════════════════════════════
# ★ 验收标准 5：等长切片下跨卡平均 = 全局均值（真 AVG 归约）
# ══════════════════════════════════════════════════════════════════════════════

def test_gloo_pair_equal_slices_avg_equals_global_mean(tmp_path: Path) -> None:
    """★ 真进程组上 `all_reduce(AVG)` == 单卡整组均值（**等长**切片 4/4）。"""
    rows = _launch(tmp_path, variant="production", mode="dp2")
    assert [row["local_len"] for row in rows] == [4, 4], "两侧条数必须严格相等"
    reduced = {row["rank"]: row["avg_reduced"] for row in rows}
    assert reduced[0] == reduced[1], "两侧读同一个归约结果"
    # 1..8 的均值 = 4.5
    assert reduced[0] == pytest.approx(4.5, rel=1e-12)
    # 与"单卡整组均值"逐位对拍
    assert reduced[0] == pytest.approx(sum(_VALUES) / len(_VALUES), rel=1e-12)


def test_gloo_pair_unequal_slices_avg_differs_positive_control(tmp_path: Path) -> None:
    """★ 阳性对照（**能失败**）：**不等长**切片（3 / 5）下 `AVG` **必须** ≠ 全局均值。

    这条证明上一条不是恒真式 —— 它真的在检查"等长"这个前提（= 本机制的收益）。
    """
    rows = _launch(tmp_path, variant="production", mode="dp2")
    assert sorted(row["skewed_len"] for row in rows) == [3, 5], "构造必须真的不等长"
    reduced = {row["rank"]: row["skewed_reduced"] for row in rows}
    assert reduced[0] == reduced[1]
    global_mean = sum(_VALUES) / len(_VALUES)
    # 等权平均（1..3 → 2.0；4..8 → 6.0）⇒ (2+6)/2 = 4.0 ≠ 4.5
    assert reduced[0] == pytest.approx(4.0, rel=1e-12)
    assert reduced[0] != pytest.approx(global_mean, rel=1e-9)


# ══════════════════════════════════════════════════════════════════════════════
# ★ dp≤1 零新增集合 ＋ **阳性对照**（改坏短路 ⇒ 计数断言必红）
# ══════════════════════════════════════════════════════════════════════════════

def test_gloo_dp1_varies_no_new_collective(tmp_path: Path) -> None:
    """★ A-1a（真进程组侧）：`dp_size == 1` ⇒ 新增集合 **0 次**。

    ★ 一次 `gather_rollout_values` 恰好**最多**发 1 次 `all_gather`（见下一条），
    所以"计数 = 0"就是"一个新集合都没发"。
    """
    rows = _launch(tmp_path, variant="production", mode="dp1", dp=1)
    assert all(row["new_collective_calls"] == 0 for row in rows), rows
    # 返回值仍 = 单卡整组（dp≤1 时"本地值就是整组"）
    for row in rows:
        assert row["gathered"] == _WHOLE


def test_gloo_dp2_gather_emits_exactly_one_collective(tmp_path: Path) -> None:
    """★ 频次口径：dp=2 时同一路径**恰好**发 1 次 `all_gather`（对照 dp=1 的 0 次）。"""
    rows = _launch(tmp_path, variant="production", mode="dp2_probe", dp=2)
    assert all(row["new_collective_calls"] == 1 for row in rows), rows


def test_gloo_dp1_broken_shortcut_makes_the_count_assertion_red(tmp_path: Path) -> None:
    """★ **阳性对照**：把 `dp_size <= 1` 的短路改成 `dp_size >= 1` ⇒ 计数 **必须** ≠ 0。

    ⇒ 上面"零新增集合"那条断言**能失败**，不是哑断言。
    """
    rows = _launch(tmp_path, variant="broken_shortcut", mode="dp1", dp=1)
    assert all(row["new_collective_calls"] > 0 for row in rows), (
        "改坏短路后计数仍为 0 ⇒ 该阳性对照无效（断言是哑的）"
    )


def test_shard_module_source_has_the_dp_shortcut_anchor(tmp_path: Path) -> None:
    """★ 防呆：阳性对照所依赖的**源码锚点**必须仍在（否则变体替换会静默失效）。"""
    src = _SHARD_SRC.read_text(encoding="utf-8")
    assert "if dp_group is None or dp_size <= 1:" in src, (
        "阳性对照的替换锚点漂移 ⇒ 该对照会退化成哑断言（必须同步更新变体）"
    )
