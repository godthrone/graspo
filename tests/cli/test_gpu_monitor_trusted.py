"""可信显存采样的负向测试——"可见 2 卡但机上 8 卡"。

历史事故：采样命令不带 ``-i``，峰值混入未点名（历史上是生产 GPU6/7）的卡。本测试构造最坏场景
（nvidia-smi 无视 ``-i``、返回全部 8 卡），断言采样结果只含可见的 2 卡。
纯逻辑测试，不触 GPU（runner 注入）。

**F-2 追加（容器内设备重编号）**：runtime 把宿主 GPU2 映射成容器内 index 0 后，
容器内按宿主卡号查 ``nvidia-smi -i 2`` 报 ``exit status 6``。修后容器内改按
**实测可见卡的本地序号**采样；"只采可见卡、绝不发不带 ``-i`` 的全卡查询"
这条铁律在两条通道上都由测试锁死。

**F-10 → B10 再推导（2026-10-12）**：目标卡**起跑前可用**断言——判据由"``memory.used``
> 64 MiB 或 util>5% 即拒绝"改为"**剩余显存 ≥ 本档头寸** ∧ **util == 0**"，且默认**连续
采样**（本模块用 autouse fixture 注入 2 帧 × 0 s，避免单测真等 24 s；真语义不变）。
"""

import pytest

from graspo.cli.gpu_monitor import (
    IDLE_FIELDS,
    assert_target_gpus_idle,
    assert_visible_gpus_idle,
    build_gpu_query_command,
    parse_idle_query,
    parse_visible_device_indices,
    probe_gpu_inventory,
    query_gpu_rows,
    resolve_sample_gpus,
)
from graspo.core.gpu_guard import GpuInventory, GpuLockError

@pytest.fixture(autouse=True)
def _fast_idle_sampling(monkeypatch):
    """默认 3 帧 × 10 s 会让单测真等 20 s ⇒ 统一注入 2 帧 × 0 s（判据语义不变）。"""
    monkeypatch.setenv("GRASPO_GPU_IDLE_SAMPLES", "2")
    monkeypatch.setenv("GRASPO_GPU_IDLE_INTERVAL_S", "0")


@pytest.fixture(autouse=True)
def _no_container_declarations(monkeypatch):
    """三判据的 ③ 面在单测里注入"无容器声明"（⛔ 不让单测去连真 docker）。"""
    import graspo.cli.gpu_monitor as monitor

    monkeypatch.setattr(monitor, "read_container_declarations", lambda: {})


#: 机上 8 卡（含 0–3；B10 起全卡均在允许集合内）的 nvidia-smi 输出：index,uuid,used,free,total,util,temp,power
_EIGHT_CARD_OUTPUT = "".join(
    f"{idx}, GPU-{idx:03d}, {100 * idx}, {81920 - 100 * idx}, 81920, {idx * 5}, 40, 100.0\n"
    for idx in range(8)
)


class _NoisyRunner:
    """最坏情况模拟器：无视 argv，总是返回全部 8 卡。"""

    def __init__(self) -> None:
        self.commands: list[list[str]] = []

    def __call__(self, command: list[str]) -> str:
        self.commands.append(command)
        return _EIGHT_CARD_OUTPUT


def test_query_only_returns_visible_two_cards():
    """可见 2 卡（4,6）但机上 8 卡 → 采样结果只含 4 和 6。"""
    runner = _NoisyRunner()

    rows = query_gpu_rows(["4", "6"], runner=runner)

    assert [row["gpu_index"] for row in rows] == [4, 6]
    assert len(rows) == 2
    sampled = {int(row["gpu_index"]) for row in rows}
    # ★ 未点名的卡（含同属允许集合的 7）不得出现在样本中——"只采可见卡"这条铁律
    assert 7 not in sampled, "未点名的允许集合内卡 7 不得出现在样本中"
    assert 0 not in sampled, "允许集合外的卡 0 不得出现在样本中"


def test_query_command_always_carries_dash_i():
    """查询命令必须带 ``-i <目标卡>``——不带 -i 的全卡查询在结构上不可能发生。"""
    command = build_gpu_query_command(["0", "3"])

    assert "-i" in command
    assert command[command.index("-i") + 1] == "0,3"
    assert "all" not in command


def test_query_command_rejects_empty_targets():
    """没有目标卡就不构造命令（不给"全卡查询"留缺口）。"""
    with pytest.raises(ValueError):
        build_gpu_query_command([])


def test_resolve_sample_gpus_uses_visible_set():
    """默认采样目标 = 可见卡；显式取值必须是可见集子集。"""
    visible = {"NVIDIA_VISIBLE_DEVICES": "4,6"}

    assert resolve_sample_gpus(None, visible=visible["NVIDIA_VISIBLE_DEVICES"]) == ["4", "6"]
    assert resolve_sample_gpus("6", visible=visible["NVIDIA_VISIBLE_DEVICES"]) == ["6"]


@pytest.mark.parametrize(
    "gpus,visible",
    [
        (None, None),  # 未锁卡
        (None, "all"),  # 全卡
        (None, "4,8"),  # 含允许集合外的卡（≥8 不存在）
        ("4,7", "4,6"),  # 显式越出可见集
    ],
)
def test_resolve_sample_gpus_is_fail_closed(gpus, visible):
    """任一边界不满足 → 拒绝采样，绝不回退到全卡。"""
    with pytest.raises(GpuLockError):
        resolve_sample_gpus(gpus, visible=visible)


# ── F-2：容器内重编号（宿主 GPU2 → 容器 index 0）─────────────────────────────


def test_parse_visible_device_indices_reads_local_indices():
    """``nvidia-smi -L`` 的行首序号就是容器内本地序号。"""
    text = (
        "GPU 0: NVIDIA A800-SXM4-80GB (UUID: GPU-aaa)\n"
        "GPU 1: NVIDIA A800-SXM4-80GB (UUID: GPU-bbb)\n"
    )

    assert parse_visible_device_indices(text) == (0, 1)


def test_parse_visible_device_indices_rejects_unparseable_output():
    """行结构不认识 → 抛错（宁可停，也不猜一组卡号去查）。"""
    with pytest.raises(RuntimeError):
        parse_visible_device_indices("not a gpu list\n")


def test_probe_gpu_inventory_uses_local_indices():
    """探针返回实测可见卡（本地序号）——注入 runner，不调真 nvidia-smi。"""

    def runner(command: list[str]) -> str:
        assert command == ["nvidia-smi", "-L"]
        return "GPU 0: NVIDIA A800-SXM4-80GB (UUID: GPU-aaa)\n"

    inventory = probe_gpu_inventory(runner=runner)

    assert inventory.count == 1
    assert inventory.indices == (0,)


def test_resolve_sample_gpus_remapped_container_uses_local_indices():
    """★F-2 核心回归：``void``（runtime 收窄 + 重编号）→ 采样用容器内本地序号 0。

    旧实现按宿主卡号（如 2）查 ``nvidia-smi -i 2``，容器内没有这张卡 ⇒
    ``exit status 6``。修后目标卡是本地序号（探测一次，共享给断言与查询）。
    """
    inventory = GpuInventory(source="nvidia-smi -L", count=1, indices=(0,))

    assert resolve_sample_gpus(None, visible="void", inventory_probe=lambda: inventory) == ["0"]
    assert resolve_sample_gpus("0", visible="void", inventory_probe=lambda: inventory) == ["0"]


def test_resolve_sample_gpus_remapped_container_rejects_out_of_range_target():
    """显式采样卡必须落在实测可见卡之内——越界（如按宿主卡号给 2）即拒绝。"""
    inventory = GpuInventory(source="nvidia-smi -L", count=1, indices=(0,))

    with pytest.raises(GpuLockError) as exc:
        resolve_sample_gpus("2", visible="void", inventory_probe=lambda: inventory)

    assert "越出实测可见卡" in str(exc.value)


def test_resolve_sample_gpus_remapped_container_is_fail_closed_on_probe_failure():
    """探不到可见卡 → 拒绝采样（不猜、不退回全卡查询）。"""

    def broken() -> GpuInventory:
        raise RuntimeError("nvidia-smi 不可用")

    with pytest.raises(RuntimeError):
        resolve_sample_gpus(None, visible="void", inventory_probe=broken)


def test_query_gpu_rows_raises_when_nvidia_smi_missing(monkeypatch):
    """查询命令失败 → ``RuntimeError``（fail-closed，不返回空样本冒充成功）。"""
    import graspo.cli.gpu_monitor as monitor

    def boom(*_args, **_kwargs):
        raise FileNotFoundError("nvidia-smi")

    monkeypatch.setattr(monitor.subprocess, "run", boom)

    with pytest.raises(RuntimeError):
        query_gpu_rows(["0"])


# ── F-10：目标卡实测空闲断言 ────────────────────────────────────────────────


class _IdleRunner:
    """按目标卡返回实测读数：``busy[卡] = (memory.used, utilization.gpu)``，total 固定 81920。

    ★ B10：读数形状 = ``index, memory.used, memory.total, utilization.gpu``（4 字段）。
    """

    TOTAL = 81920.0

    def __init__(
        self,
        busy: dict[int, tuple[float, float]] | None = None,
        apps: str = "",
    ) -> None:
        self.busy = busy or {}
        self.apps = apps
        self.commands: list[list[str]] = []

    def __call__(self, command: list[str]) -> str:
        self.commands.append(command)
        joined = " ".join(command)
        # 三判据（2026-10-25）后同一注入点要应答三类查询：
        if "--query-compute-apps" in joined:
            return self.apps       # 默认空 = 卡上没有 compute 进程
        if "--query-gpu=index,uuid" in joined:
            selected = command[command.index("-i") + 1].split(",")
            return "".join(f"{int(idx)}, GPU-fake{int(idx)}\n" for idx in selected)
        selected = command[command.index("-i") + 1].split(",")
        return "".join(
            f"{int(idx)}, {self.busy.get(int(idx), (4.0, 0.0))[0]}, "
            f"{self.TOTAL:g}, {self.busy.get(int(idx), (4.0, 0.0))[1]}\n"
            for idx in selected
        )


def test_assert_target_gpus_idle_passes_and_only_queries_target_cards():
    """全空闲 → 通过；命令必须带 ``-i`` 且只含目标卡（铁律的机器化检查）。"""
    runner = _IdleRunner()

    assert_target_gpus_idle([0, 1], runner=runner)

    command = runner.commands[0]
    assert command[command.index("-i") + 1] == "0,1"
    assert f"--query-gpu={','.join(IDLE_FIELDS)}" in command
    assert "-L" not in command


def test_assert_target_gpus_idle_rejects_occupied_card():
    """★B10 再推导：目标卡剩余显存跌破头寸（used 40 GiB ⇒ 剩余 41 GiB < 48 GiB）→ 拒绝。"""
    runner = _IdleRunner(busy={3: (40 * 1024, 0.0)})

    with pytest.raises(GpuLockError) as exc:
        assert_target_gpus_idle([0, 1, 2, 3], runner=runner)

    assert "GPU3" in str(exc.value)
    assert "头寸" in str(exc.value)


def test_assert_target_gpus_idle_rejects_busy_utilization():
    """util != 0 → 拒绝（显存没涨也不行；B10 起 util 必须恰为 0）。"""
    runner = _IdleRunner(busy={1: (4.0, 77.0)})

    with pytest.raises(GpuLockError):
        assert_target_gpus_idle([0, 1], runner=runner)


def test_assert_target_gpus_idle_rejects_card_with_compute_process():
    """★三判据 ②（2026-10-25）：卡上有 compute 进程 ⇒ 拒绝，即使显存/util/剩余都正常。"""
    runner = _IdleRunner(apps="GPU-fake1, 3912182, 13326, python3\n")

    with pytest.raises(GpuLockError) as exc:
        assert_target_gpus_idle([0, 1], runner=runner)

    assert "GPU1" in str(exc.value)
    assert "3912182" in str(exc.value)


def test_assert_target_gpus_idle_rejects_resident_memory_even_if_util_zero():
    """★三判据 ①（2026-10-25）：邻居常驻占 16 GiB、util=0 ⇒ 已用显存超上限 ⇒ 拒绝。"""
    runner = _IdleRunner(busy={1: (16 * 1024, 0.0)})

    with pytest.raises(GpuLockError) as exc:
        assert_target_gpus_idle([1], runner=runner)

    assert "GPU1" in str(exc.value)
    assert "空闲上限" in str(exc.value)


def test_assert_target_gpus_idle_is_fail_closed_on_incomplete_readings():
    """少一张卡的读数 = 有一张没验过 → 拒绝（不当"空闲"放行）。"""

    def runner(command: list[str]) -> str:
        return "0, 4, 0\n"  # 只回 0 号卡，缺 1 号卡

    with pytest.raises(RuntimeError):
        assert_target_gpus_idle([0, 1], runner=runner)


def test_assert_visible_gpus_idle_host_side_uses_host_indices():
    """宿主侧显式卡号通道：逐张查这些宿主卡。"""
    runner = _IdleRunner()

    assert_visible_gpus_idle(visible="4,6", runner=runner)

    assert runner.commands[0][runner.commands[0].index("-i") + 1] == "4,6"


def test_assert_visible_gpus_idle_container_side_uses_local_indices_once():
    """容器侧 ``void``：用实测本地序号断言，且**只探测一次**（与断言共用）。"""
    calls: list[str] = []
    runner = _IdleRunner()

    def probe() -> GpuInventory:
        calls.append("probe")
        return GpuInventory(source="nvidia-smi -L", count=1, indices=(0,))

    assert_visible_gpus_idle(visible="void", inventory_probe=probe, runner=runner)

    assert calls == ["probe"]
    assert runner.commands[0][runner.commands[0].index("-i") + 1] == "0"


def test_assert_visible_gpus_idle_rejects_illegal_visible_set():
    """``all`` 不是 runtime 哨兵 → 仍按宿主卡号通道拒绝（不放松）。"""
    with pytest.raises(GpuLockError):
        assert_visible_gpus_idle(visible="all", runner=_IdleRunner())


# ── F-10 真机回归：查询必须带 index，解析必须吃下 nvidia-smi 的真实输出 ──────


def test_idle_query_asks_for_index_field():
    """★真机回归：空闲查询**必须一起查 ``index``**。

    只查 ``memory.used,utilization.gpu`` 时 nvidia-smi 只返回 ``'0, 0'``（2 字段），
    解析器拿不到卡号 ⇒ 真机上断言直接失败（目标 GPU 服务器实测：``无法解析 ... '0, 0'``），
    等于 F-10 在真机路径上不可用。
    """
    runner = _IdleRunner()

    assert_visible_gpus_idle(visible="4", runner=runner)

    query = next(item for item in runner.commands[0] if item.startswith("--query-gpu="))
    assert IDLE_FIELDS[0] == "index"
    assert query == f"--query-gpu={','.join(IDLE_FIELDS)}"


@pytest.mark.parametrize(
    "raw,expected",
    [
        # 守卫自用：--format=csv,noheader,nounits（★B10：4 字段，返回**剩余**显存）
        ("0, 4096, 81920, 0\n1, 0, 81920, 0\n", [(0, 77824.0, 0.0), (1, 81920.0, 0.0)]),
        # --format=csv（带表头 + 单位；目标 GPU 服务器实测原文）
        (
            "index, memory.used [MiB], memory.total [MiB], utilization.gpu [%]\n"
            "0, 0 MiB, 81920 MiB, 0 %\n1, 0 MiB, 81920 MiB, 0 %\n",
            [(0, 81920.0, 0.0), (1, 81920.0, 0.0)],
        ),
        # --format=csv,noheader（带单位、无表头）
        ("0, 0 MiB, 81920 MiB, 0 %\n1, 0 MiB, 81920 MiB, 0 %\n",
         [(0, 81920.0, 0.0), (1, 81920.0, 0.0)]),
        # 额外列（温度）不影响目标列
        (
            "index, memory.used [MiB], memory.total [MiB], utilization.gpu [%], temperature.gpu\n"
            "0, 7 MiB, 81920 MiB, 1 %, 40\n",
            [(0, 81913.0, 1.0)],
        ),
        # 非标准空白分隔
        ("0 1024 81920 12\n", [(0, 80896.0, 12.0)]),
        # 标准 CSV 引号包裹的千位逗号
        (
            'index, memory.used [MiB], memory.total [MiB], utilization.gpu [%]\n'
            '0, "1,234.5 MiB", 81920 MiB, 4.5 %\n',
            [(0, 80685.5, 4.5)],
        ),
    ],
)
def test_parse_idle_query_accepts_real_nvidia_smi_formats(raw, expected):
    """★真机回归：csv（带表头/不带）、带单位/不带、逗号/空格、千位逗号、多余列都吃下。"""
    assert parse_idle_query(raw) == expected


@pytest.mark.parametrize(
    "raw",
    [
        "0, 0\n",  # ★旧 bug 现场：只查 2 字段时 nvidia-smi 的真实输出
        "0, 0, 0\n",  # ★B10：3 字段（缺 memory.total）⇒ 必须拒绝，否则算不出剩余
        "0, 0 MiB, 0 %\n",  # 同上（带单位、无表头）
        "0, 0 %\n",  # 带表头却缺 memory.used 列 ⇒ 不猜位置
        "0, abc MiB, 0 %\n",  # 数值字段坏掉
        "0, , 0\n",  # 空字段
        "0, 0 MiB, nan %\n",  # nan 不是 nvidia-smi 读数写法
        "0, 0 MiB\n",  # 字段数不足
    ],
)
def test_parse_idle_query_rejects_bad_rows(raw):
    """★负向：坏行**必须抛错**（绝不能"解析失败却当空闲通过"）。"""
    with pytest.raises(RuntimeError):
        parse_idle_query(raw)


def test_parse_idle_query_returns_nothing_for_empty_output():
    """空输出 ⇒ 空列表（由调用方按 fail-closed 处理，**不是**"通过"）。"""
    assert parse_idle_query("") == []


def test_assert_target_gpus_idle_is_fail_closed_on_empty_readings():
    """★负向：一条读数都没有 ⇒ 抛错（fail-closed），绝不当"空闲"放行。

    这是最危险的事故模式：断言读不到数却放行 ⇒ 去抢生产/他人的卡。
    """
    with pytest.raises(RuntimeError, match="未返回任何读数"):
        assert_target_gpus_idle([0, 1], runner=lambda command: "")


def test_idle_only_cli_is_fail_closed_when_readings_are_missing(monkeypatch):
    """★负向（CLI 层）：``record-gpu-memory --idle-only`` 读不到数 ⇒ **rc≠0**。

    走的是 ``graspo.cli.gpu_monitor`` 的命令处理函数（不启真进程、不碰 GPU）。
    """
    import graspo.cli.gpu_monitor as monitor

    monkeypatch.setattr(monitor, "query_idle_rows", lambda *_a, **_k: [])

    rc = monitor.cmd_record_gpu_memory(_idle_only_args())

    assert rc != 0


def test_idle_only_cli_rejects_occupied_card(monkeypatch):
    """★负向（CLI 层）：目标卡被占 ⇒ ``--idle-only`` rc≠0。"""
    import graspo.cli.gpu_monitor as monitor

    monkeypatch.setenv("NVIDIA_VISIBLE_DEVICES", "0")
    monkeypatch.setattr(monitor, "query_idle_rows", lambda *_a, **_k: [(0, 6029.0, 0.0)])

    assert monitor.cmd_record_gpu_memory(_idle_only_args()) != 0


def _idle_only_args() -> object:
    """构造 ``--idle-only`` 的最小 argparse 命名空间（不依赖真 CLI 装配）。"""
    import argparse

    return argparse.Namespace(
        idle_only=True,
        assert_idle=False,
        output_dir=None,
        gpus=None,
        interval_sec=1.0,
        tag="",
        pid_filter="",
        duration_sec=None,
        recent_limit=120,
    )
