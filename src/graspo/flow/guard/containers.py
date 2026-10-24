"""flow.guard.containers — **容器声明的卡占用**读取（取卡判据 ③ 的设施侧）。

为什么需要它（228 实测缺陷，2026-10-25）：取卡闸门此前只看 ``docker ps/inspect``
得到的"我方占卡台账"（``run_guards cards-busy``），而**判定"这张卡能不能用"的
断言**（``gpu_idle_assert``）压根不读容器面 ⇒ 一张卡**显存 0 MiB、util 0**、但已被
他人容器 ``--gpus``/``NVIDIA_VISIBLE_DEVICES`` **声明预留**（228 上 GPU7 就是
``pf228-transform-worker-1`` 预留）时，断言会把它判成"可用"⇒ 我们起跑就撞车。

本模块只做**读**：``docker ps`` 取名 → ``docker inspect`` 取
``HostConfig.DeviceRequests.DeviceIDs`` 与 ``Config.Env`` 里的
``NVIDIA_VISIBLE_DEVICES``。⛔ 不 stop/rm/kill 任何容器、⛔ 不启容器。

判据（单一真相源）在 ``graspo.core.gpu_guard``；本模块只负责把 docker 的原始输出
归一成"卡号 → 容器名列表"。被 ``graspo.cli.gpu_monitor`` 与 ``scripts/gpu_idle_assert.py``
共用。
"""

from __future__ import annotations

import json
import subprocess
from collections.abc import Callable, Iterable, Sequence

#: ``NVIDIA_VISIBLE_DEVICES`` 里不代表"某张具体卡"的取值——不构成卡声明。
_NON_CARD_VALUES = frozenset({"", "all", "void", "none"})


def parse_docker_inspect_stream(text: str) -> list[dict]:
    """解析 ``docker inspect`` 的**多 JSON 对象流**（非单个数组）。

    ``docker inspect <c1> <c2>`` 输出并排的多个 JSON 对象；单容器时是数组。
    用 ``JSONDecoder.raw_decode`` 逐个解码，两种形态都吃下。
    """
    decoder = json.JSONDecoder()
    objects: list[dict] = []
    index = 0
    size = len(text or "")
    while index < size:
        while index < size and text[index] in " \t\r\n,":
            index += 1
        if index >= size:
            break
        try:
            obj, index = decoder.raw_decode(text, index)
        except ValueError:
            break
        if isinstance(obj, list):
            objects.extend(item for item in obj if isinstance(item, dict))
        elif isinstance(obj, dict):
            objects.append(obj)
    return objects


def cards_declared_by_containers(items: Iterable[dict]) -> dict[str, list[str]]:
    """从 ``docker inspect`` 的对象列表算出 ``{卡号: [容器名, ...]}``。

    两类声明都要采（228 实测两种都在用）：

    - ``HostConfig.DeviceRequests[].DeviceIDs``：``docker run --gpus '"device=6,7"'``；
    - ``Config.Env`` 里的 ``NVIDIA_VISIBLE_DEVICES=1,3,6,7``：``--runtime=nvidia -e``
      ⇒ 此时 ``DeviceRequests`` 为 ``null``，只看前者会**恒为空**。
    """
    declared: dict[str, list[str]] = {}
    for item in items:
        name = str(item.get("Name") or "").lstrip("/") or "<未命名容器>"
        cards: list[str] = []
        requests = (item.get("HostConfig") or {}).get("DeviceRequests") or []
        for request in requests:
            for device in request.get("DeviceIDs") or []:
                cards.extend(part.strip() for part in str(device).split(","))
        for entry in (item.get("Config") or {}).get("Env") or []:
            if not str(entry).startswith("NVIDIA_VISIBLE_DEVICES="):
                continue
            value = str(entry).split("=", 1)[1]
            cards.extend(part.strip() for part in value.split(","))
        for card in cards:
            if card.lower() in _NON_CARD_VALUES:
                continue
            bucket = declared.setdefault(card, [])
            if name not in bucket:
                bucket.append(name)
    return declared


def read_container_declarations(
    *,
    runner: Callable[[Sequence[str]], str] | None = None,
) -> dict[str, list[str]]:
    """只读读取当前**全部容器**的卡声明，返回 ``{卡号: [容器名, ...]}``。

    :param runner: 命令执行器（接收 argv、返回 stdout）；默认走 ``subprocess``。
    :raises RuntimeError: docker 不可用 / 命令失败 ⇒ **fail-closed**：
        「读不到容器声明」绝不当成「没有容器声明」——那正是 GPU7 事故的形态。
    """
    execute = runner if runner is not None else _run_command
    try:
        names_text = execute(["docker", "ps", "--format", "{{.Names}}"])
    except RuntimeError:
        raise
    names = [line.strip() for line in names_text.splitlines() if line.strip()]
    if not names:
        # docker 命令跑通了、确实一个容器都没有 ⇒ 没有任何卡声明（不是"读不到"）。
        return {}
    inspect_text = execute(["docker", "inspect", *names])
    return cards_declared_by_containers(parse_docker_inspect_stream(inspect_text))


def _run_command(argv: Sequence[str]) -> str:
    """执行只读命令；失败即 ``RuntimeError``（调用方 fail-closed）。"""
    try:
        completed = subprocess.run(
            list(argv), check=True, capture_output=True, text=True
        )
    except (subprocess.CalledProcessError, FileNotFoundError, OSError) as exc:
        raise RuntimeError(
            f"无法读取容器声明的卡占用：{' '.join(argv)} 失败（{exc}）"
            "——拿不到 docker 读数就拒绝，绝不把「读不到」当成「没人占」"
        ) from None
    return completed.stdout


__all__ = [
    "cards_declared_by_containers",
    "parse_docker_inspect_stream",
    "read_container_declarations",
]
