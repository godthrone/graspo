"""GPU 显存/利用率采样记录（运维监控工具，**只采可见卡**）。

v0.23.0 起从 ``scripts/record_gpu_memory.py`` 提升为 CLI 命令
（``graspo record-gpu-memory``）：被测试固化的运维工具按宪法 §10.2
必须制度化，不能留在 scripts/（scripts/ 已删除，防止内网路径/凭据
随脚本入库）。

职责：按固定间隔采样 ``nvidia-smi`` 的 GPU 显存/利用率/温度/功耗与
进程占用，追加写入 JSONL，结束时输出摘要 JSON。纯诊断工具，不参与
训练产物。

**配置驱动（v0.25.0，宪法 §10.1）：** 本命令的落盘输出（产物位置与内容）
全部由 ``--config`` 的 ``gpu_monitor`` 段决定——``output_dir`` / ``tag`` /
``interval_sec`` / ``recent_limit`` / ``pid_filter`` 五个字段。
旧版把它们做成 CLI 参数（``--output-dir`` / ``--tag`` / ``--interval-sec`` /
``--pid-filter`` / ``--recent-limit``），其中 ``--output-dir`` 是**输出定位
参数**（§10.1 明令禁止），``--tag`` 直接写进每行记录内容——同一 config 会
因 CLI 参数不同而产出不同，config 与产物的对应关系被打破。现已删除这些
参数（§18.1 不留负债）：新写法只有 ``--config``（输入定位）+ ``--gpus``
（运行环境）+ ``--duration-sec``（运行边界）+ ``--assert-idle``/``--idle-only``
（只 print 不落盘）。

**可信采样（防呆，§2）——历史事故的修复：**

历史采样用不带 ``-i`` 的 ``nvidia-smi --query-gpu``，峰值混入生产
GPU6/7 与他人并发作业，显存数据不可作依据。本工具现在三重防护：

1. 采样目标由 ``NVIDIA_VISIBLE_DEVICES`` 唯一决定（``--gpus`` 只能是
   它的子集，越界即拒绝）——守卫逻辑在 ``graspo.core.gpu_guard``；
2. 查询命令**永远**带 ``-i <目标卡>``，不提供"全卡查询"入口；
3. 返回行再做一次白名单过滤——即使 ``nvidia-smi`` 忽略 ``-i``，
   也不会有可见集之外的行进入样本。

**容器内重编号（F-2 修复）：** nvidia-container-runtime 按设备收窄可见集时，
会把容器内 ``NVIDIA_VISIBLE_DEVICES`` 覆写成哨兵 ``void``，并把**宿主卡重编号**
（宿主 GPU2 → 容器内 index 0）。此时若仍按宿主卡号查 ``nvidia-smi -i 2``，
容器内没有这张卡，命令以 ``exit status 6`` 失败。修法：**容器内的采样目标改为
容器命名空间里的本地序号**——由 :func:`probe_gpu_inventory` 实测可见卡
（``nvidia-smi -L``）再经 ``gpu_guard.select_sample_targets_for_inventory``
解析。宿主侧采样路径**不变**（宿主侧 ``NVIDIA_VISIBLE_DEVICES`` 就是真实卡号，
仍是"只采目标卡 + 命令必带 ``-i``"）。

**本次新增**：:func:`probe_gpu_inventory`（F-1/F-2 所需的"实测可见卡"）与
:func:`assert_target_gpus_idle`（F-10 目标卡实测空闲断言：>64 MiB 或 util>5%
即拒绝，宁等不抢、不 kill 他人进程）。
"""

import argparse
import csv
import json
import os
import re
import signal
import subprocess
import time
from collections import defaultdict, deque
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from graspo.core.gpu_guard import (
    GpuInventory,
    assert_gpu_idle,
    assert_gpu_lock,
    is_runtime_managed,
    resolve_device_source,
    select_sample_targets,
    select_sample_targets_for_inventory,
)

GPU_FIELDS = (
    "index",
    "uuid",
    "memory.used",
    "memory.free",
    "memory.total",
    "utilization.gpu",
    "temperature.gpu",
    "power.draw",
)
PROCESS_FIELDS = ("gpu_uuid", "pid", "process_name", "used_memory")

#: 目标卡空闲断言的查询字段：**index 必须一起查**。
#: 这不是装饰——nvidia-smi 的 ``--query-gpu`` 只返回被请求的列，若只查
#: ``memory.used,utilization.gpu``，输出就是 ``0, 0``（2 字段），解析器拿不到卡号，
#: 也失去了"这一行到底是哪张卡"的校验锚点。曾因漏掉 index 导致真机上
#: ``无法解析 ...（字段数 2）：'0, 0'``，使 F-10 的空闲断言在真机路径上不可用。
IDLE_FIELDS = ("index", "memory.used", "utilization.gpu")

#: 单个数值字段的严格形状（剥单位后必须整串就是一个数）。
#: 刻意**不**放行 ``nan`` / ``inf`` / ``1e5``——它们不是 nvidia-smi 的读数写法。
_NUMERIC_FIELD = re.compile(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)")


def parse_float(value: str) -> float:
    cleaned = value.strip()
    if cleaned in {"", "[N/A]", "N/A", "Not Supported"}:
        return 0.0
    return float(cleaned)


def math_floor(value: float) -> int:
    return int(value // 1)


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    idx = min(len(ordered) - 1, max(0, math_floor(q * (len(ordered) - 1))))
    return ordered[idx]


def parse_gpu_query(text: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in text.splitlines():
        if not line.strip():
            continue
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != len(GPU_FIELDS):
            raise ValueError(f"Unexpected nvidia-smi GPU row: {line}")
        rows.append(
            {
                "gpu_index": int(parts[0]),
                "gpu_uuid": parts[1],
                "memory_used_mib": parse_float(parts[2]),
                "memory_free_mib": parse_float(parts[3]),
                "memory_total_mib": parse_float(parts[4]),
                "utilization_gpu_pct": parse_float(parts[5]),
                "temperature_gpu_c": parse_float(parts[6]),
                "power_draw_w": parse_float(parts[7]),
            }
        )
    return rows


def _numeric(text: str) -> float | None:
    """把 ``nvidia-smi`` 的**单个数值字段**解析成 float；不能解析时返回 ``None``。

    真实 nvidia-smi 的 CSV 字段是**带单位**的（`--format=csv` / `--format=csv,noheader`
    下为 ``0 MiB`` / ``0 %``，`nounits` 下才是裸数字），带表头时第一行还是
    中文/英文表头（``index``、``memory.used [MiB]``…）。因此这里统一剥掉单位后缀
    与千位逗号，再用一个严格正则确认"整串就是一个数"，避免 ``float()`` 接受
    ``nan`` / ``inf`` / ``1e5`` 这类不可能是 nvidia-smi 读数的写法。
    """
    cleaned = (
        text.strip()
        .replace("\u2212", "-")
        .replace("MiB", "")
        .replace("MB", "")
        .replace("mW", "")
        .replace("W", "")
        .replace("%", "")
        .replace(",", "")
        .strip()
    )
    if _NUMERIC_FIELD.fullmatch(cleaned) is None:
        return None
    return float(cleaned)


def _csv_fields(line: str) -> list[str]:
    """按 CSV 切行：优先逗号；整行无逗号时退回按空白切（兼容非标准输出）。

    支持双引号包裹的字段（标准 CSV）——nvidia-smi 若把数字按本地化格式写成
    ``"1,234.5 MiB"``，引号内的逗号不会被误切。
    """
    if '"' in line:
        try:
            parsed = next(csv.reader([line], skipinitialspace=True))
        except csv.Error:
            parsed = None
        if parsed:
            return [part.strip() for part in parsed]
    if "," in line:
        return [part.strip() for part in line.split(",")]
    return line.split()


def _is_header_row(fields: list[str]) -> bool:
    """是否为 ``--format=csv`` 的表头行（如 ``index, memory.used [MiB], utilization.gpu [%]``）。

    表头判定用"首个字段不是数值"——比匹配具体字面量更稳（nvidia-smi 的表头
    在不同版本/字段集合下会变，且带单位方括号）。
    """
    return bool(fields) and _numeric(fields[0]) is None


def _header_field_names(fields: list[str]) -> list[str]:
    """把表头字段归一为列名（``memory.used [MiB]`` → ``memory.used``）。"""
    names: list[str] = []
    for field in fields:
        name = re.sub(r"\s*\[[^\]]*\]\s*$", "", field.strip().lower())
        name = name.split(" ", 1)[0]
        if name:
            names.append(name)
    return names


def _idle_value(
    fields: list[str],
    index: int,
    name: str,
    *,
    header: list[str] | None,
) -> float | None:
    """取某列的数值：**有表头就按列名定位，无表头按位置**。

    按列名定位是防"列顺序/字段集合变化 ⇒ 静默错读"的正解——例如别人用 4 字段
    查询（多一个 ``memory.total``）时，按位置读会把 ``81920 MiB`` 当成利用率，
    报出荒谬的"利用率 81920%"；按列名读则永远取到真正的那一列。
    """
    if header is not None:
        if name not in header:
            # 表头里没有这一列 ⇒ 输出与查询不符，不猜位置（宁可 fail-closed）。
            return None
        pos = header.index(name)
        if pos < len(fields):
            return _numeric(fields[pos])
        return None
    if index < len(fields):
        return _numeric(fields[index])
    return None


def parse_idle_query(text: str) -> list[tuple[int, float, float]]:
    """解析空闲断言查询行 → ``(index, memory_used_mib, utilization_pct)``。

    **必须同时覆盖真实输出**（目标 GPU 服务器实测原文，见工单 F-10 修复；
    机器与环境记录见 `infra` skill 与 `.local/`）：

    1. ``--format=csv,noheader,nounits``（守卫自用）::

           0, 0, 0

    2. ``--format=csv``（带表头 + 单位）::

           index, memory.used [MiB], utilization.gpu [%]
           0, 0 MiB, 0 %

    解析规则（防呆，§2.3）：

    - 跳过表头行（首个字段不是数值）与空行；
    - 数值字段剥单位（``MiB`` / ``MB`` / ``W`` / ``%``）与千位逗号（含标准 CSV 引号
      包裹的 ``"1,234.5 MiB"``），接受整数与小数；
    - **有表头时按列名定位**（``index`` / ``memory.used`` / ``utilization.gpu``）；
      无表头时按位置，字段数 ≥3（多余字段忽略）。带表头却没有目标列名 ⇒ 拒绝——
      不猜位置（防"改了查询列顺序 ⇒ 静默错读"，例如把 ``memory.total`` 当利用率）；
    - **任何一行非法 ⇒ 抛 :class:`RuntimeError`（fail-closed）**：空闲断言拿不到
      可信读数时必须**拒绝**，绝不能"解析失败却当空闲通过"（那会去抢别人的卡）。
    """
    rows: list[tuple[int, float, float]] = []
    header: list[str] | None = None
    for line in text.splitlines():
        if not line.strip():
            continue
        fields = _csv_fields(line)
        if _is_header_row(fields):
            header = _header_field_names(fields)
            continue
        index = _numeric(fields[0]) if fields else None
        memory_used = _idle_value(fields, 1, "memory.used", header=header)
        utilization = _idle_value(fields, 2, "utilization.gpu", header=header)
        if index is None or memory_used is None or utilization is None:
            raise RuntimeError(
                f"无法解析 nvidia-smi 空闲查询行（字段数 {len(fields)}）：{line!r}"
            )
        rows.append((int(index), memory_used, utilization))
    return rows



def parse_process_query(text: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in text.splitlines():
        if not line.strip():
            continue
        parts = [part.strip() for part in line.split(",", maxsplit=len(PROCESS_FIELDS) - 1)]
        if len(parts) != len(PROCESS_FIELDS):
            continue
        rows.append(
            {
                "gpu_uuid": parts[0],
                "pid": int(parts[1]),
                "process_name": parts[2],
                "used_memory_mib": parse_float(parts[3]),
            }
        )
    return rows


def build_gpu_query_command(gpu_indices: Sequence[str]) -> list[str]:
    """构造**只查指定卡**的 nvidia-smi 命令——``-i`` 是不可省略的防线。

    单一入口：采样永远经由本函数构造命令，因此"忘了带 -i"在结构上不可能发生
    （宪法 §2 防呆：不靠调用方自觉）。
    """
    if not gpu_indices:
        raise ValueError("build_gpu_query_command requires at least one GPU index")
    return [
        "nvidia-smi",
        f"--query-gpu={','.join(GPU_FIELDS)}",
        "--format=csv,noheader,nounits",
        "-i",
        ",".join(str(item) for item in gpu_indices),
    ]


def parse_visible_device_indices(text: str) -> tuple[int, ...]:
    """从 ``nvidia-smi -L`` 输出解析**容器内本地序号**（探测可见卡的唯一入口）。

    ``-L`` 每个可见卡一行（``GPU 0: NVIDIA ... (UUID: GPU-xxx)``）。行首序号即
    容器命名空间里的本地 index——runtime 重编号后，宿主卡号在这里是看不到的
    （这正是 F-2 的根因：不能拿宿主卡号去查容器内的卡）。

    :raises RuntimeError: 行结构与 ``GPU <n>:`` 不符——宁可 fail-closed，
        也不要猜出一组卡号去查。
    """
    indices: list[int] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if not stripped.startswith("GPU ") or ":" not in stripped:
            raise RuntimeError(f"无法解析 nvidia-smi -L 行：{line!r}")
        head = stripped[len("GPU ") :].split(":", 1)[0].strip()
        if not head.isdigit():
            raise RuntimeError(f"无法解析 nvidia-smi -L 卡号：{line!r}")
        indices.append(int(head))
    return tuple(indices)


def query_gpu_rows(
    gpu_indices: list[str],
    *,
    runner: Callable[[list[str]], str] | None = None,
) -> list[dict[str, Any]]:
    """查询给定卡的显存行；结果再按白名单过滤一次。

    :param gpu_indices: 目标卡号（字符串，来自可信采样目标解析）。
    :param runner: 命令执行器（接收 argv，返回 stdout）；默认走 ``subprocess``。
        注入点让负向测试能构造"可见 2 卡但机上 8 卡"的场景。
    :raises RuntimeError: ``nvidia-smi`` 不在或执行失败。
    """
    command = build_gpu_query_command(gpu_indices)
    stdout = _run_query(command, runner)
    requested = {int(item) for item in gpu_indices}
    # 白名单过滤：即使 nvidia-smi 忽略 -i 返回了全卡，也不让可见集之外的行进入样本。
    return [row for row in parse_gpu_query(stdout) if int(row["gpu_index"]) in requested]


def _run_query(
    command: list[str],
    runner: Callable[[list[str]], str] | None,
) -> str:
    """执行只读查询命令；失败即 ``RuntimeError``（调用方 fail-closed）。"""
    if runner is not None:
        return runner(command)
    try:
        completed = subprocess.run(command, check=True, capture_output=True, text=True)
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        raise RuntimeError(f"查询命令失败：{' '.join(command)}（{exc}）") from None
    return completed.stdout


def query_process_rows(pid_filters: list[str]) -> list[dict[str, Any]]:
    command = [
        "nvidia-smi",
        f"--query-compute-apps={','.join(PROCESS_FIELDS)}",
        "--format=csv,noheader,nounits",
    ]
    result = subprocess.run(command, check=False, capture_output=True, text=True)
    if result.returncode != 0:
        return []
    rows = parse_process_query(result.stdout)
    if not pid_filters:
        return rows
    filtered = []
    for row in rows:
        haystack = f"{row.get('pid', '')} {row.get('process_name', '')}".lower()
        if any(item in haystack for item in pid_filters):
            filtered.append(row)
    return filtered


def summarize_samples(
    samples: list[dict[str, Any]], recent_samples: list[dict[str, Any]]
) -> dict[str, Any]:
    by_gpu: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for sample in samples:
        by_gpu[int(sample["gpu_index"])].append(sample)
    per_gpu = {}
    for gpu_index, rows in by_gpu.items():
        used = [float(row["memory_used_mib"]) for row in rows]
        util = [float(row["utilization_gpu_pct"]) for row in rows]
        per_gpu[str(gpu_index)] = {
            "samples": len(rows),
            "memory_used_mib_peak": max(used),
            "memory_used_mib_mean": sum(used) / len(used),
            "memory_used_mib_p95": percentile(used, 0.95),
            "utilization_gpu_pct_mean": sum(util) / len(util),
            "last": rows[-1],
        }
    peak_values = [float(str(item["memory_used_mib_peak"])) for item in per_gpu.values()]
    return {
        "per_gpu": per_gpu,
        "max_peak_memory_gap_mib": max(peak_values) - min(peak_values)
        if len(peak_values) >= 2
        else 0.0,
        "recent_samples": recent_samples,
    }


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def utc_timestamp() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def probe_gpu_inventory(
    runner: Callable[[list[str]], str] | None = None,
) -> GpuInventory:
    """实测容器内**可见卡**（F-1/F-2 的设施层入口）。

    用 ``nvidia-smi -L``（结构稳定：每个可见卡一行）数卡并取容器内本地序号。
    ``-L`` 不接受 ``-i``，但它天然只列**可见**卡——这正是我们要的口径
    （不要 ``--query-gpu`` 的全卡枚举语义）。带 ``-i`` 的查询仍由
    :func:`build_gpu_query_command` 唯一构造（采样路径）。

    :param runner: 命令执行器（接收 argv、返回 stdout）；默认走 ``subprocess``。
    :raises RuntimeError: ``nvidia-smi`` 不在或执行失败——调用方必须 fail-closed。
    """
    command = ["nvidia-smi", "-L"]
    if runner is None:
        try:
            completed = subprocess.run(command, check=False, capture_output=True, text=True)
        except FileNotFoundError as exc:
            raise RuntimeError(f"nvidia-smi 不可用：{exc}") from None
        if completed.returncode != 0:
            raise RuntimeError(
                f"nvidia-smi -L 失败（rc={completed.returncode}）：{completed.stderr.strip()[:200]}"
            )
        stdout = completed.stdout
    else:
        stdout = runner(command)
    indices = parse_visible_device_indices(stdout)
    return GpuInventory(
        source="nvidia-smi -L",
        count=len(indices),
        indices=indices,
    )


def query_idle_rows(
    gpu_indices: Sequence[str],
    *,
    runner: Callable[[list[str]], str] | None = None,
) -> list[tuple[int, float, float]]:
    """只查给定卡的 ``memory.used`` / ``utilization.gpu``（命令必带 ``-i``）。"""
    command = [
        "nvidia-smi",
        f"--query-gpu={','.join(IDLE_FIELDS)}",
        "--format=csv,noheader,nounits",
        "-i",
        ",".join(str(item) for item in gpu_indices),
    ]
    return parse_idle_query(_run_query(command, runner))


def assert_target_gpus_idle(
    gpu_indices: Sequence[int | str],
    *,
    runner: Callable[[list[str]], str] | None = None,
) -> None:
    """断言目标卡**实测空闲**（F-10）：>64 MiB 或 util>5% 即拒绝启动。

    逐卡只查该卡（命令必带 ``-i``），任一卡非空闲即抛 :class:`GpuLockError`。
    **宁等不抢**：不 kill 他人进程、不与他人混跑。判定边界与消息在
    ``gpu_guard.assert_gpu_idle``（纯逻辑层，§1.3）。

    :param gpu_indices: 目标卡号（宿主侧为宿主卡号，容器内为本地序号）。
    :param runner: 命令执行器（注入点，便于负向测试构造"卡被占"场景）。
    :raises GpuLockError: 任一目标卡非空闲。
    :raises RuntimeError: 查询命令失败——拿不到读数时 fail-closed。
    """
    targets = [str(item) for item in gpu_indices]
    if not targets:
        raise RuntimeError("assert_target_gpus_idle 需要至少一个目标卡号")
    rows = query_idle_rows(targets, runner=runner)
    if not rows:
        raise RuntimeError(f"nvidia-smi 未返回任何读数（目标卡 {targets}）——拒绝启动")
    # 读数必须覆盖每一个目标卡：少一张就等于有一张没验过（fail-closed）。
    reported = {index for index, _, _ in rows}
    requested = {int(item) for item in targets}
    missing = sorted(requested - reported)
    if missing:
        raise RuntimeError(f"nvidia-smi 未返回这些目标卡的读数：{missing}——拒绝启动")
    for index, used_mib, util_pct in rows:
        assert_gpu_idle(index, used_mib, util_pct)


def assert_visible_gpus_idle(
    *,
    visible: str | None = None,
    inventory_probe: Callable[[], GpuInventory] | None = None,
    runner: Callable[[list[str]], str] | None = None,
) -> None:
    """按当前设备来源断言目标卡空闲（宿主侧卡号 / 容器内实测可见卡都支持）。

    - ``NVIDIA_VISIBLE_DEVICES`` 是显式卡号 → 逐张断言这些宿主卡；
    - 是 runtime 哨兵（``void``）→ 先实测可见卡拿到容器内本地序号，再逐张断言
      （重编号后按宿主卡号查必然失败，见 F-2）。

    :raises GpuLockError: 设备边界不满足，或任一目标卡非空闲。
    :raises RuntimeError: 实测可见卡探测失败/读数不完整。
    """
    raw = os.environ.get("NVIDIA_VISIBLE_DEVICES") if visible is None else visible
    if not is_runtime_managed(raw):
        # 显式宿主卡号通道：非法取值由 assert_gpu_lock 抛错（原样上抛）。
        targets: Sequence[int | str] = assert_gpu_lock(raw)
    else:
        # runtime 收窄 + 重编号：只能用容器内本地序号（F-2）。只探测一次，
        # 断言与后续查询共用同一份实测结果（避免两次探测之间卡集变化）。
        inventory = (inventory_probe or probe_gpu_inventory)()
        source = resolve_device_source(raw, inventory)
        targets = source.inventory.indices if source.inventory is not None else ()
    assert_target_gpus_idle(targets, runner=runner)


def resolve_sample_gpus(
    gpus: str | None,
    *,
    visible: str | None = None,
    inventory_probe: Callable[[], GpuInventory] | None = None,
) -> list[str]:
    """确定采样目标卡号：只允许落在**可见集**之内。

    两条通道（F-2）：

    - 宿主侧（``NVIDIA_VISIBLE_DEVICES`` 是显式卡号）→ 目标 = 这些宿主卡号
      （``--gpus`` 必须是其子集）；
    - 容器内（runtime 把取值覆写成 ``void``，且宿主卡重编号为本地 0..n-1）→
      目标 = 容器内**实测可见卡的本地序号**（``--gpus`` 必须是其子集）。
      旧版在这里按宿主卡号查，容器内查不到 ⇒ ``exit status 6``。

    fail-closed：可见集未设置 / ``all`` / 含生产卡 6,7（宿主侧）/ 超过 4 卡 →
    抛 ``GpuLockError``，不返回任何"全卡"退路（宪法 §2.3 边界校验）。
    """
    if visible is None:
        visible = os.environ.get("NVIDIA_VISIBLE_DEVICES")
    if not is_runtime_managed(visible):
        return [str(device) for device in select_sample_targets(gpus, visible)]

    if inventory_probe is None:
        inventory_probe = probe_gpu_inventory
    return [
        str(device)
        for device in select_sample_targets_for_inventory(gpus, inventory_probe())
    ]


def record_gpu_memory(
    *,
    gpus: str | None,
    interval_sec: float,
    output_dir: str,
    tag: str | None,
    pid_filter: str | None,
    duration_sec: float | None,
    recent_limit: int,
    inventory_probe: Callable[[], GpuInventory] | None = None,
) -> int:
    """采样循环：按固定间隔记录 GPU 状态到 JSONL，结束时写摘要。

    ``gpus=None`` 表示"全部可见卡"；显式取值必须是可见集的子集。容器内 runtime
    哨兵场景走实测可见卡通道（F-2），``inventory_probe`` 是它的注入点。

    ``output_dir`` / ``tag`` / ``interval_sec`` / ``recent_limit`` / ``pid_filter``
    全部来自 **config**（``gpu_monitor`` 段，§10.1），本函数不再有对应的 CLI 参数。
    ``tag=None`` 与 ``pid_filter=None`` 表示"未提供"，落盘时写空串（保持 JSONL
    字段形状不变——读方按字符串处理，不引入 null）。
    """
    tag = tag or ""
    pid_filter = pid_filter or ""
    gpu_indices = resolve_sample_gpus(gpus, inventory_probe=inventory_probe)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    memory_path = out_dir / "gpu_memory.jsonl"
    process_path = out_dir / "gpu_processes.jsonl"
    summary_path = out_dir / "gpu_memory_summary.json"
    memory_path.touch()
    process_path.touch()
    stop = {"requested": False}

    def _stop(_signum: int, _frame: object) -> None:
        stop["requested"] = True

    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)

    pid_filters = [item.strip().lower() for item in pid_filter.split(",") if item.strip()]
    samples: list[dict[str, Any]] = []
    recent_samples: deque[dict[str, Any]] = deque(maxlen=recent_limit)
    started = time.monotonic()

    try:
        while not stop["requested"]:
            timestamp = utc_timestamp()
            gpu_rows = query_gpu_rows(gpu_indices)
            # 进程行按可见卡的 UUID 过滤：--query-compute-apps 不支持 -i，
            # 生产进程会出现在原始输出里，必须在进入样本前剔除。
            visible_uuids = {str(row["gpu_uuid"]) for row in gpu_rows}
            process_rows = [
                row
                for row in query_process_rows(pid_filters)
                if str(row.get("gpu_uuid", "")) in visible_uuids
            ]
            for row in gpu_rows:
                row["timestamp"] = timestamp
                row["tag"] = tag
                samples.append(row)
                recent_samples.append(row)
                append_jsonl(memory_path, row)
            for row in process_rows:
                row["timestamp"] = timestamp
                row["tag"] = tag
                append_jsonl(process_path, row)
            if duration_sec is not None and time.monotonic() - started >= duration_sec:
                break
            time.sleep(interval_sec)
    finally:
        summary = summarize_samples(samples, list(recent_samples))
        summary.update(
            {
                "tag": tag,
                "gpus": gpu_indices,
                "interval_sec": interval_sec,
                "started_monotonic": started,
                "finished_at": utc_timestamp(),
                "sample_count": len(samples),
            }
        )
        summary_path.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    return 0


def build_gpu_monitor_parser(subparsers: Any) -> None:
    """注册 ``record-gpu-memory`` 子命令（由 cli.app.build_parser 调用）。

    **配置驱动（宪法 §10.1）**：本命令落盘 `gpu_memory.jsonl` /
    `gpu_processes.jsonl` / `gpu_memory_summary.json` —— 一切落盘输出都由
    ``--config`` 里的 ``gpu_monitor`` 段决定。因此**没有** ``--output-dir`` /
    ``--tag`` / ``--interval-sec`` / ``--pid-filter`` / ``--recent-limit`` 这些
    参与决定产物的参数（推论 1/2：CLI 参数与 config 字段零交集）。

    留下的 CLI 参数按 §10.1 三原则归类：

    * ``--config``：输入定位参数（唯一明确列出的参数名）；
    * ``--gpus``：运行环境参数（在哪张卡采样，不影响落盘位置/内容）；
    * ``--duration-sec``：运行边界参数（冒烟试跑，不产生完整产物）；
    * ``--assert-idle`` / ``--idle-only``：运行边界参数，只 print 不落盘
      （``--idle-only`` 不需要 ``--config``）。
    """
    gpu = subparsers.add_parser(
        "record-gpu-memory",
        help="Record nvidia-smi GPU memory/utilization to JSONL (visible GPUs only).",
    )
    gpu.add_argument(
        "--config",
        "-c",
        default=None,
        help=(
            "YAML config carrying the `gpu_monitor` section: output_dir, tag, "
            "interval_sec, recent_limit, pid_filter. Required unless --idle-only "
            "(which writes nothing and prints only)."
        ),
    )
    gpu.add_argument(
        "--gpus",
        default=None,
        help=(
            "Comma-separated GPU indices; must be a subset of NVIDIA_VISIBLE_DEVICES. "
            "Default (None) = all visible GPUs. 'all' is rejected (fail-closed)."
        ),
    )
    gpu.add_argument(
        "--duration-sec", type=float, default=None, help="Optional duration for smoke/dry runs."
    )
    gpu.add_argument(
        "--assert-idle",
        action="store_true",
        help=(
            "F-10: before sampling, assert every target GPU is idle "
            "(memory.used <= 64 MiB and utilization.gpu <= 5%%); refuse if busy. "
            "Never kills other processes."
        ),
    )
    gpu.add_argument(
        "--idle-only",
        action="store_true",
        help="F-10: run only the idle assertion (no sampling loop) and exit.",
    )
    gpu.set_defaults(func=cmd_record_gpu_memory)


def cmd_record_gpu_memory(args: argparse.Namespace) -> int:
    """``record-gpu-memory`` 命令入口。锁卡守卫/空闲断言拒绝时打印原因并返回 1。"""
    import sys

    from graspo.core.gpu_guard import GpuLockError

    try:
        if getattr(args, "assert_idle", False) or getattr(args, "idle_only", False):
            assert_visible_gpus_idle()
        if getattr(args, "idle_only", False):
            return 0
        if not args.config:
            print(
                "record-gpu-memory: --config is required（除非用 --idle-only，"
                "它只 print 不落盘）",
                file=sys.stderr,
            )
            return 2
        # 延迟导入，与 cli.tools 同口径；同时避免本模块在"只做空闲断言"时
        # 也要拉起配置栈（§6.1 简单优先）。
        from graspo.core.schema import GraspoConfig

        config = GraspoConfig.from_yaml(args.config)
        monitor = config.gpu_monitor
        if not monitor.output_dir:
            print(
                "record-gpu-memory: config 缺少 gpu_monitor.output_dir"
                "（落盘位置必须由 config 决定，§10.1；若只想看数请用 --idle-only）",
                file=sys.stderr,
            )
            return 2
        return record_gpu_memory(
            gpus=args.gpus,
            interval_sec=monitor.interval_sec,
            output_dir=monitor.output_dir,
            tag=monitor.tag,
            pid_filter=monitor.pid_filter,
            duration_sec=args.duration_sec,
            recent_limit=monitor.recent_limit,
        )
    except GpuLockError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    except RuntimeError as exc:
        # 拿不到可信读数（探针失败/命令失败）时 fail-closed，不当"空闲"放行。
        print(f"GPU 空闲断言/采样前置失败（fail-closed）：{exc}", file=sys.stderr)
        return 1
