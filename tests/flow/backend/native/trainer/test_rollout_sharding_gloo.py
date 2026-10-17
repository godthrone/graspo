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
4. ★ **U2 字段并集**（真 gloo，dp∈{1,2,4} × 三支）：`gather_field_names` 让每卡得到
   **同一个 sorted 全局并集** ⇒ `gather_field_stats` 的合并结果与 `μ_f` 与 **dp=1 单卡路径
   逐值相同**。覆盖 `same_set` / `distinct_same_cardinality`（支 C，修复前静默错 μ）/
   `distinct_cardinality`（支 A，修复前 `all_gather` 形状失配）；两条阳性对照证明断言能失败。
5. ★ **U2 载荷边界机核**：录下**真正发给集合**的载荷元素，断言它**恰是本卡切片字段名**且
   每个元素都是**标识符形态**（正则 `^[A-Za-z_][A-Za-z0-9_.-]*$` ＋ 长度上限 ＋ 无空白/换行/
   控制字符）⇒ 守住 `rollout_shard.py` 模块 docstring「载荷里没有文本、没有标注」那条边界。
   ★ **能失败对照**：`text_leak` 模式把一段真正的文本追加进载荷 ⇒ 形态断言**必红**。
"""

from __future__ import annotations

import atexit
import json
import os
import re
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


# ══════════════════════════════════════════════════════════════════════════════
# ★ U2 字段并集（验收判据 1/3）：真 gloo 上「分卡路径 == dp=1 单卡路径」逐值
# ══════════════════════════════════════════════════════════════════════════════

_FIELD_UNION_CHILD = textwrap.dedent(
    '''
    """gloo 子进程：用**真** rollout_shard 做 μ_f 字段通道会合，把结果写 JSONL。"""
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
    VARIANT = sys.argv[2]           # "production" | "local_union"
    SCENARIO = sys.argv[3]          # same_set | distinct_same_cardinality | distinct_cardinality
    OUT = Path(sys.argv[4])
    RANK = int(sys.argv[5])
    WORLD = int(sys.argv[6])
    RENDEZVOUS = sys.argv[7]
    M = int(sys.argv[8])            # 组内切片数（decomposition size）
    DP = int(sys.argv[9])           # 本 rank 看到的 dp_size（会合域）
    PAYLOAD_MODE = sys.argv[10]     # "clean" | "text_leak"（载荷边界阳性对照）

    torch.set_num_threads(1)
    sys.path.insert(0, str(SRC))

    import graspo.flow.backend.native.trainer.rollout_shard as rs
    from graspo.ripple.annotation.advantages import field_means, merge_field_stats

    if VARIANT == "local_union":
        # ★ 阳性对照：把 `gather_field_names` 的**全局会合**撤掉 ⇒ 退回
        #   "本卡切片并集"（= 修复前 `rollout.py` 的行为）。
        source = Path(rs.__file__).read_text(encoding="utf-8")
        start = source.index("def gather_field_names(")
        end = source.index("def gather_field_stats(")
        replacement = "\\n".join(
            [
                "def gather_field_names(runtime, local_field_names, *, what='μ_f 字段名'):",
                '    """阳性对照变体：撤掉全局会合 ⇒ 本卡切片并集（修复前行为）。"""',
                "    return sorted({str(name) for name in local_field_names})",
                "",
                "",
            ]
        )
        broken = source[:start] + replacement + source[end:]
        assert broken != source, "阳性对照变体必须真的改动了源码"
        variant_dir = Path(tempfile.mkdtemp(prefix="u2_broken_field_union_"))
        atexit.register(shutil.rmtree, variant_dir, ignore_errors=True)
        target = variant_dir / "_broken_rollout_shard.py"
        target.write_text(broken, encoding="utf-8")
        import importlib.util

        spec = importlib.util.spec_from_file_location("broken_rollout_shard_u2", target)
        assert spec is not None and spec.loader is not None
        rs = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(rs)
        assert "阳性对照变体" in (rs.gather_field_names.__doc__ or "")

    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", RENDEZVOUS)
    dist.init_process_group(backend="gloo", rank=RANK, world_size=WORLD)

    class _Adapter:
        def __init__(self, dp_size):
            self.tp_state = SimpleNamespace(
                dp_size=dp_size, dp_group=dist.group.WORLD, device=torch.device("cpu")
            )

    class _Runtime:
        def __init__(self, dp_size):
            self._adapter = _Adapter(dp_size)

        def _require_adapter(self):
            return self._adapter

    CUTOFF = 2
    NAMES = ["a", "b", "c", "d"]

    def _cell(scenario, r, i):
        if scenario == "same_set":
            return {"a": (float((r + 1) * (r + 2) * (i + 1)), (r + 1) + i)}
        if scenario == "distinct_same_cardinality":
            return {NAMES[r]: (float((r + 1) * (r + 2) * (i + 1)), (r + 1) + i)}
        if scenario == "distinct_cardinality":
            return {
                NAMES[k]: (float((k + 1) * (k + 2) * (i + 1)), (k + 1) + i)
                for k in range(r + 1)
            }
        raise AssertionError("unknown scenario " + repr(scenario))

    CONTRIB = [[_cell(SCENARIO, r, i) for i in range(CUTOFF)] for r in range(M)]
    whole_stats = [merge_field_stats([CONTRIB[r][i] for r in range(M)]) for i in range(CUTOFF)]
    if DP == 1:
        local_stats = [dict(s) for s in whole_stats]      # 单卡 = 整组（切片未启用）
    else:
        local_stats = [dict(s) for s in CONTRIB[RANK]]    # 本卡切片

    local_names = rs.union_field_names(local_stats)

    # ★ U2 载荷边界：把**真正发给集合**的载荷元素录下来，供父进程做"标识符形态"机核
    #   （守住模块 docstring「载荷里没有文本、没有标注」那条边界）。
    _payload_log = {}
    _real_all_gather_object = dist.all_gather_object

    def _spy_all_gather_object(object_list, obj, group=None):
        if PAYLOAD_MODE == "text_leak":
            # ★ 阳性对照：模拟"将来有人往这个会合塞真正的文本/标注"。
            obj = list(obj) + ["sorry i cannot help with that request, please contact support"]
        _payload_log["sent"] = [str(element) for element in obj]
        return _real_all_gather_object(object_list, obj, group=group)

    dist.all_gather_object = _spy_all_gather_object

    global_names = rs.gather_field_names(
        _Runtime(DP), local_names, what="μ_f 的 (Σraw, n) 字段名"
    )
    sharded = rs.gather_field_stats(
        _Runtime(DP),
        local_stats,
        cutoff=CUTOFF,
        field_names=global_names,
        what="μ_f 的 (Σraw, n)",
    )
    # ★ dp=1 单卡路径：同一个函数 + dp_size=1 的 runtime（短路 ⇒ 原样返回整组统计）
    dp1 = rs.gather_field_stats(
        _Runtime(1),
        whole_stats,
        cutoff=CUTOFF,
        field_names=global_names,
        what="μ_f 的 (Σraw, n)",
    )

    def _enc(stats):
        return [{f: [float(raw), int(n)] for f, (raw, n) in s.items()} for s in stats]

    row = {
        "rank": RANK,
        "world": WORLD,
        "dp_size": DP,
        "m": M,
        "scenario": SCENARIO,
        "local_names": list(local_names),
        "global_names": list(global_names),
        "payload_elements": list(_payload_log.get("sent", [])),
        "sharded_stats": _enc(sharded),
        "dp1_stats": _enc(dp1),
        "sharded_mu": [field_means(s) for s in sharded],
        "dp1_mu": [field_means(s) for s in dp1],
    }
    with OUT.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row) + "\\n")
    dist.barrier()
    dist.destroy_process_group()
    '''
)

_FIELD_SCENARIOS = ["same_set", "distinct_same_cardinality", "distinct_cardinality"]

# ★ U2 载荷边界机核（守住 `rollout_shard.py` 模块 docstring「载荷里没有文本、没有标注」）：
#   进入 `gather_field_names` 会合的载荷元素**必须**是**标识符形态**的短字段名——
#   ⛔ 不是 completion 文本、⛔ 不是标注内容。正则/长度上限/无空白三条一起判。
_FIELD_PAYLOAD_MAX_CHARS = 64
_FIELD_PAYLOAD_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_.-]*$")
_TEXT_LIKE_PAYLOAD = "sorry i cannot help with that request, please contact support"


def _assert_identifier_payload(elements: list[str]) -> None:
    """★ U2 载荷边界机核：会合载荷元素必须是**标识符形态**（不是文本/标注）。

    三条判据（任一不满足 ⇒ `AssertionError`，⛔ 不静默放过）：
    1. 载荷非空、每个元素是 `str`；
    2. 匹配标识符正则 ``^[A-Za-z_][A-Za-z0-9_.-]*$``；
    3. 长度 ≤ `_FIELD_PAYLOAD_MAX_CHARS` 且**不含空白/换行/控制字符**（文本/标注特征）。
    """
    assert elements, "会合载荷不能为空（空载荷 ⇒ 求不出全局并集）"
    for element in elements:
        assert isinstance(element, str), f"载荷元素必须是 str，实际 {type(element)!r}"
        assert _FIELD_PAYLOAD_RE.match(element), (
            f"载荷元素不是标识符形态（疑似把文本/标注塞进了这个会合）：{element!r}"
        )
        assert len(element) <= _FIELD_PAYLOAD_MAX_CHARS, (
            f"载荷元素超长：len={len(element)} > {_FIELD_PAYLOAD_MAX_CHARS}：{element!r}"
        )
        assert not any(ch.isspace() or ord(ch) < 32 for ch in element), (
            f"载荷元素含空白/换行/控制字符（疑似文本/标注）：{element!r}"
        )


def _launch_field_union(
    tmp_path: Path,
    *,
    variant: str,
    scenario: str,
    m: int,
    dp: int,
    expect_launch_failure: bool = False,
    payload_mode: str = "clean",
) -> list[dict]:
    """跑一次字段并集子进程（dp=1 ⇒ world=1；dp≥2 ⇒ world=dp）。"""
    world = 1 if dp == 1 else dp
    assert dp == 1 or m >= world, "切片数必须 >= 会合成员数"
    child = tmp_path / "gloo_child_field_union_u2.py"
    child.write_text(_FIELD_UNION_CHILD, encoding="utf-8")
    out = tmp_path / f"fu_{variant}_{scenario}_m{m}_dp{dp}_{payload_mode}.jsonl"
    port = str(_free_port())
    env = dict(os.environ)
    env["PYTHONPATH"] = str(_SRC)
    env["OMP_NUM_THREADS"] = "1"
    procs = [
        subprocess.Popen(
            [
                sys.executable, str(child), str(_SRC), variant, scenario, str(out),
                str(rank), str(world), port, str(m), str(dp), payload_mode,
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
        _, stderr = proc.communicate(timeout=180)
        errs.append(stderr)
    failed = any(proc.returncode != 0 for proc in procs)
    if expect_launch_failure:
        assert failed, "阳性对照必须发起失败（撤掉全局会合 ⇒ 各卡载荷宽度不一致）"
        return []
    if failed:
        detail = "\n".join(
            f"----- rank{index} (rc={proc.returncode}) -----\n{errs[index]}"
            for index, proc in enumerate(procs)
        )
        raise AssertionError(detail)
    return [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]


@pytest.mark.parametrize("scenario", _FIELD_SCENARIOS)
@pytest.mark.parametrize(("m", "dp"), [(2, 2), (4, 4)])
def test_field_union_sharded_path_equals_dp1_single_card(
    tmp_path: Path, scenario: str, m: int, dp: int
) -> None:
    """★ U2 验收 1：dp=m 分卡路径与 dp=1 单卡路径**逐值相同**（真 gloo 实会合）。

    三支都跑：

    - `same_set`（集合相同，对照：修复不改数值）；
    - `distinct_same_cardinality`（**支 C**：同基数不同名 —— 修复前静默错 μ，且两卡 μ 不一致）；
    - `distinct_cardinality`（**支 A**：基数不同 —— 修复前 `all_gather` 形状失配）。
    """
    single = _launch_field_union(tmp_path, variant="production", scenario=scenario, m=m, dp=1)
    assert len(single) == 1
    # dp=1 是真正的单卡路径：`sharded` 与 `dp1` 引用都应是"整组统计的原样返回"。
    assert single[0]["sharded_stats"] == single[0]["dp1_stats"]

    rows = _launch_field_union(tmp_path, variant="production", scenario=scenario, m=m, dp=dp)
    assert len(rows) == dp

    # ① 每 rank 的全局并集 = **同一个** sorted 去重并集
    names = rows[0]["global_names"]
    assert names == sorted(names) and len(set(names)) == len(names), f"必须是 sorted 去重：{names}"
    assert all(row["global_names"] == names for row in rows), (
        f"各 rank 的全局并集必须一致：{[row['global_names'] for row in rows]}"
    )
    # ② 它就是**每张卡局部并集的并**（每卡局部集 ⊆ 全局并集，且无多余）
    assert names == sorted({n for row in rows for n in row["local_names"]}), (
        f"全局并集 {names} ≠ 各卡局部并集之并"
    )
    # ③ 分卡结果与 dp=1 单卡路径**逐值相同**（同加序 ⇒ 精确相等，不用容差）
    for row in rows:
        assert row["sharded_stats"] == single[0]["dp1_stats"], (
            f"scenario={scenario} rank{row['rank']} 分卡统计 ≠ dp=1 单卡路径："
            f"{row['sharded_stats']} vs {single[0]['dp1_stats']}"
        )
        assert row["sharded_mu"] == single[0]["dp1_mu"], (
            f"scenario={scenario} rank{row['rank']} 分卡 μ ≠ dp=1 单卡路径："
            f"{row['sharded_mu']} vs {single[0]['dp1_mu']}"
        )
    # ④ ★ 载荷边界：真实发出去的会合载荷元素必须是**标识符形态**，且**恰是本卡切片字段名**
    #    （守住模块 docstring「载荷里没有文本、没有标注」那条边界）。
    for row in rows:
        _assert_identifier_payload(row["payload_elements"])
        assert row["payload_elements"] == row["local_names"], (
            f"会合载荷必须恰是本卡切片字段名（⛔ 无附加内容）：{row['payload_elements']}"
        )


# ══════════════════════════════════════════════════════════════════════════════
# ★ 阳性对照（验收判据 2）：撤掉全局会合 ⇒ 上面两条断言必红
# ══════════════════════════════════════════════════════════════════════════════


def test_field_union_positive_control_same_cardinality_is_silently_wrong(tmp_path: Path) -> None:
    """★ 阳性对照（**支 C**，真 gloo）：撤掉全局会合 ⇒ 本卡局部并集 ⇒ 两卡 μ 不一致且 ≠ dp=1。

    ⇒ `test_field_union_sharded_path_equals_dp1_single_card` 的 ①②③ 断言**能失败**，
    不是哑断言。
    """
    scenario = "distinct_same_cardinality"
    broken = _launch_field_union(tmp_path, variant="local_union", scenario=scenario, m=2, dp=2)
    single = _launch_field_union(tmp_path, variant="production", scenario=scenario, m=2, dp=1)
    assert len(broken) == 2
    mu = {row["rank"]: row["sharded_mu"] for row in broken}
    assert mu[0] != mu[1], f"阳性对照必须让两卡 μ 不一致（修复前静默错），实际 {mu}"
    ref = single[0]["dp1_mu"]
    assert mu[0] != ref and mu[1] != ref, f"阳性对照必须偏离 dp=1 单卡路径：{mu} vs {ref}"
    # 并集断言也必红：撤掉会合后各卡的"全局并集"就是本卡局部集
    assert broken[0]["global_names"] != broken[1]["global_names"]


def test_field_union_positive_control_distinct_cardinality_launch_fails(tmp_path: Path) -> None:
    """★ 阳性对照（**支 A**，真 gloo）：撤掉全局会合 ⇒ 载荷宽度 1 vs 2 ⇒ `all_gather` 形状失配。

    gloo 对形状失配是 **fail-fast**（`EnforceNotMet` / connection reset），不会挂起
    （本机实测，见工位 evidence）。
    """
    _launch_field_union(
        tmp_path,
        variant="local_union",
        scenario="distinct_cardinality",
        m=2,
        dp=2,
        expect_launch_failure=True,
    )


def test_field_union_helper_source_anchor_is_present() -> None:
    """★ 防呆：上面两条阳性对照依赖的源码锚点必须仍在（否则对照会退化成哑断言）。"""
    src = _SHARD_SRC.read_text(encoding="utf-8")
    assert "def gather_field_names(" in src, "阳性对照变体的起始锚点漂移"
    assert src.index("def gather_field_names(") < src.index("def gather_field_stats("), (
        "阳性对照变体的结束锚点漂移"
    )
    assert "if dp_group is None or dp_size <= 1:" in src, "dp≤1 短路锚点漂移"


# ══════════════════════════════════════════════════════════════════════════════
# ★ U2 载荷边界（审查加强建议）：塞入"像文本"的元素 ⇒ 形态断言**必红**
# ══════════════════════════════════════════════════════════════════════════════


def test_payload_shape_assertion_goes_red_on_text_like_element(tmp_path: Path) -> None:
    """★ **能失败对照**（载荷边界，真 gloo）：往这个会合里塞一个"像文本"的元素 ⇒ 断言必红。

    `text_leak` 模式在子进程里对 `dist.all_gather_object` 装 spy，并在**真正转发前**把一段
    真正的文本追加进载荷（= 模拟"将来有人往这个会合里塞文本/标注"）⇒ 父进程的
    `_assert_identifier_payload` **必须**判红。⛔ 若不红 ⇒ 该机核是哑断言（本用例会失败）。
    """
    rows = _launch_field_union(
        tmp_path,
        variant="production",
        scenario="distinct_same_cardinality",
        m=2,
        dp=2,
        payload_mode="text_leak",
    )
    assert len(rows) == 2
    leaked = [
        element
        for row in rows
        for element in row["payload_elements"]
        if element == _TEXT_LIKE_PAYLOAD
    ]
    assert leaked, (
        "text_leak 变体必须真的把文本塞进会合载荷，实际载荷："
        f"{[row['payload_elements'] for row in rows]}"
    )
    with pytest.raises(AssertionError, match="标识符形态"):
        for row in rows:
            _assert_identifier_payload(row["payload_elements"])
