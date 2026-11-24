"""`scripts/gpu_idle_assert.py`（宿主侧**起跑前可用**断言 CLI）的用例。

**为什么单独测这个脚本**：它是"防抢别人卡 / 防撞别人服务"的前置装置，入口层的
fail-closed 语义（读不到数 / 卡不满足任一判据 ⇒ rc≠0）必须在 CLI 边界上被钉住——
底层函数正确但 CLI 吞掉异常返回 0，等于防线不存在。

★★ 判据换代（2026-10-25「取卡闸门补漏」工作包）：由**两条**升为**三条并取**：

- ① 显存：``memory.used`` ≤ ``--max-used-mib``（默认 2 GiB）**且** 剩余 ≥ 本档头寸
  （训练档 ``--position-mib`` 默认 48 GiB；服务档 ``--position-ratio 0.85``）；
- ② 进程：``--query-compute-apps`` 里该卡**没有任何** compute 进程；
- ③ 容器：``docker ps/inspect`` 的 ``DeviceRequests`` / ``NVIDIA_VISIBLE_DEVICES``
  里**不得**出现该卡；
- ④ 采样：``--frames`` 帧内**帧帧**满足 ①②③。

为什么必须补（远端机实测缺陷；旧实现对下列场景 **rc=0** 放行 = 本文件红→绿的证据）：

- 旧判据只看 util，而远端机上 GPU0/1/4/5 被他人的**常驻推理服务**占着 13–16 GiB、
  util 采样**全是 0** ⇒ 旧实现对 ``--visible 0,1,2,4,5`` **全部放行**；
- ``cards-busy`` 只有 docker 面 ⇒ **永远漏宿主进程**（GPU2 上的宿主推理进程）；
- GPU7 显存 0 MiB、util 0 %，但被他人容器预留 ⇒ 只有读容器面才判得出来；
- 服务档（vLLM 需 ``0.85 × 总显存`` ≈ 69 632 MiB）用 48 GiB 头寸会被放行 ⇒ 起服 OOM。

红线：只用**假 PATH 里的 nvidia-smi / docker**，不调用真 nvidia-smi、不碰任何 GPU、
不启停任何容器。
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "gpu_idle_assert.py"

#: 假 nvidia-smi：按查询类型分派，读数由 ``FAKE_SMI_ROW<卡号>`` 环境变量控制。
#:
#: 三条查询都要应答（真机也是三条）：
#: - ``--query-gpu=index,memory.used,memory.total,utilization.gpu -i <cards>``：显存行；
#: - ``--query-gpu=index,uuid ... -i <cards>``：uuid 映射（compute-apps 只给 uuid）；
#: - ``--query-compute-apps=...``：全机 compute 进程（该子命令不支持 ``-i``）。
_FAKE_SMI = r"""#!/bin/bash
args="$*"
case "$args" in
  *"--query-compute-apps"*) printf '%b' "${FAKE_SMI_APPS:-}"; exit 0;;
  *"--query-gpu=index,uuid"*)
     sel="0,1,2,3,4,5,6,7"
     case "$args" in *" -i "*) sel="${args##* -i }";; esac
     IFS=',' read -ra ids <<< "$sel"
     for id in "${ids[@]}"; do printf '%s, GPU-fake%s\n' "$id" "$id"; done
     exit 0;;
esac
if [ -n "${FAKE_SMI_EMPTY:-}" ]; then exit 0; fi
sel="none"
case "$args" in *" -i "*) sel="${args##* -i }";; esac
IFS=',' read -ra ids <<< "$sel"
for id in "${ids[@]}"; do
  case "$id" in
    0) printf '%s\n' "${FAKE_SMI_ROW0:-0, 4, 81920, 0}";;
    1) printf '%s\n' "${FAKE_SMI_ROW1:-1, 4, 81920, 0}";;
    2) printf '%s\n' "${FAKE_SMI_ROW2:-2, 4, 81920, 0}";;
    3) printf '%s\n' "${FAKE_SMI_ROW3:-3, 4, 81920, 0}";;
    4) printf '%s\n' "${FAKE_SMI_ROW4:-4, 4, 81920, 0}";;
    5) printf '%s\n' "${FAKE_SMI_ROW5:-5, 4, 81920, 0}";;
    6) printf '%s\n' "${FAKE_SMI_ROW6:-6, 4, 81920, 0}";;
    7) printf '%s\n' "${FAKE_SMI_ROW7:-7, 4, 81920, 0}";;
    *) printf '%s\n' "$id, 4, 81920, 0";;
  esac
done
exit 0
"""

#: 假 docker：``ps`` 只回名字、``inspect`` 回 ``FAKE_DOCKER_INSPECT``（多对象流）。
_FAKE_DOCKER = r"""#!/bin/bash
args="$*"
if [ -n "${FAKE_DOCKER_FAIL:-}" ]; then
  echo "Cannot connect to the Docker daemon" >&2
  exit 1
fi
case "$args" in
  *"inspect"*) printf '%b' "${FAKE_DOCKER_INSPECT:-[]}"; exit 0;;
esac
printf '%b' "${FAKE_DOCKER_NAMES:-}"
exit 0
"""


def _write_fakes(bin_dir: Path) -> None:
    bin_dir.mkdir(exist_ok=True)
    for name, body in (("nvidia-smi", _FAKE_SMI), ("docker", _FAKE_DOCKER)):
        path = bin_dir / name
        path.write_text(body, encoding="utf-8")
        path.chmod(0o755)


def _run(
    tmp_path: Path,
    argv: list[str],
    env_extra: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    bin_dir = tmp_path / "bin"
    _write_fakes(bin_dir)
    env = {
        **os.environ,
        "PATH": f"{bin_dir}{os.pathsep}{os.environ.get('PATH', '')}",
    }
    for inherited in (
        "NVIDIA_VISIBLE_DEVICES",
        "GRASPO_GPU_POSITION_MIB",
        "GRASPO_GPU_IDLE_SAMPLES",
        "GRASPO_GPU_IDLE_INTERVAL_S",
    ):
        env.pop(inherited, None)
    env.update(env_extra or {})
    # 连续采样是真语义，但测试不该真等 10 s ⇒ 默认采 2 帧、间隔 0（显式给了就尊重）。
    # ★ 用 ``--samples``/``--interval-s``（旧版就有的别名）：这样"旧实现 vs 新实现"
    #   的红→绿对比可以用**同一组参数**跑，差异只来自判据而不来自参数解析。
    extra = (
        []
        if "--frames" in argv or "--samples" in argv
        else ["--samples", "2", "--interval-s", "0"]
    )
    return subprocess.run(
        [sys.executable, str(_SCRIPT), *argv, *extra],
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )


#: 远端机实测读数（2026-10-25 00:47–00:48 三帧一致），用于"真机形态"回归：
#: GPU0 2404 / GPU1 16303 / GPU2 13335(+宿主进程 13.3 GiB) / GPU3 0 /
#: GPU4 14041 / GPU5 4253 / GPU6 68419(VLLM) / GPU7 0(被容器声明)。
_REMOTE_ROWS = {
    "FAKE_SMI_ROW0": "0, 2404, 81920, 0",
    "FAKE_SMI_ROW1": "1, 16303, 81920, 0",
    "FAKE_SMI_ROW2": "2, 13335, 81920, 0",
    "FAKE_SMI_ROW3": "3, 0, 81920, 0",
    "FAKE_SMI_ROW4": "4, 14041, 81920, 0",
    "FAKE_SMI_ROW5": "5, 4253, 81920, 0",
    "FAKE_SMI_ROW6": "6, 68419, 81920, 0",
    "FAKE_SMI_ROW7": "7, 0, 81920, 0",
}
_REMOTE_APPS = "GPU-fake2, 3912182, 13326, /srv/example/VLN/LightNav-0/.venv/bin/python\n"
_REMOTE_CONTAINERS = json.dumps(
    {
        "Name": "/pf-remote-transform-worker-1",
        "State": {"Status": "running"},
        "HostConfig": {"DeviceRequests": None},
        "Config": {"Env": ["NVIDIA_VISIBLE_DEVICES=7"]},
    }
)


# ── 场景①（绿）：全空闲 ⇒ rc=0 ─────────────────────────────────────────────
def test_scenario1_all_idle_passes(tmp_path: Path):
    """①全空闲：已用 4 MiB ≤ 2 GiB、剩余 81 916 ≥ 48 GiB、无进程、无容器声明 ⇒ rc=0。"""
    completed = _run(tmp_path, ["--visible", "3"])

    assert completed.returncode == 0, completed.stderr
    assert "GPU3 可用" in completed.stdout


def test_scenario1_real_remote_idle_card_passes_when_nothing_is_declared(tmp_path: Path):
    """★真机形态：GPU3 在远端机上确实全空（0 MiB / util 0 / 无进程 / 无容器）⇒ rc=0。"""
    completed = _run(tmp_path, ["--visible", "3"], dict(_REMOTE_ROWS))

    assert completed.returncode == 0, completed.stderr


# ── 场景②（红）：有 compute 进程 ⇒ 非 0 并点名 pid ─────────────────────────
def test_scenario2_compute_process_rejects_and_names_pid(tmp_path: Path):
    """②卡上有 compute 进程（宿主进程占卡）⇒ rc≠0 且点名 pid。

    ★ 旧实现漏判：显存/剩余都够、util=0 ⇒ 旧实现 **rc=0** 放行（远端机 GPU2 的形态）。
    """
    completed = _run(
        tmp_path,
        ["--visible", "2"],
        {"FAKE_SMI_ROW2": "2, 0, 81920, 0", "FAKE_SMI_APPS": _REMOTE_APPS},
    )

    assert completed.returncode == 1, completed.stdout
    assert "3912182" in completed.stderr
    assert "compute 进程" in completed.stderr


# ── 场景③（红）：容器声明该卡 ⇒ 非 0 并点名容器 ────────────────────────────
def test_scenario3_container_declaration_rejects_and_names_container(tmp_path: Path):
    """③卡被容器声明预留（显存 0 MiB、util 0、无进程）⇒ rc≠0 且点名容器。

    ★ 旧实现漏判：旧实现根本不读 docker 面 ⇒ 旧实现 **rc=0**（远端机 GPU7 的形态）。
    """
    completed = _run(
        tmp_path,
        ["--visible", "7"],
        {
            "FAKE_SMI_ROW7": "7, 0, 81920, 0",
            "FAKE_DOCKER_NAMES": "pf-remote-transform-worker-1\n",
            "FAKE_DOCKER_INSPECT": _REMOTE_CONTAINERS,
        },
    )

    assert completed.returncode == 1, completed.stdout
    assert "pf-remote-transform-worker-1" in completed.stderr
    assert "容器" in completed.stderr


def test_container_declaration_is_read_from_device_requests_too(tmp_path: Path):
    """③两种声明形态都要采：``HostConfig.DeviceRequests.DeviceIDs``（--gpus 写法）。"""
    completed = _run(
        tmp_path,
        ["--visible", "6"],
        {
            "FAKE_SMI_ROW6": "6, 0, 81920, 0",
            "FAKE_DOCKER_NAMES": "our-vllm\n",
            "FAKE_DOCKER_INSPECT": json.dumps(
                {
                    "Name": "/our-vllm",
                    "State": {"Status": "running"},
                    "HostConfig": {"DeviceRequests": [{"DeviceIDs": ["6,7"]}]},
                }
            ),
        },
    )

    assert completed.returncode == 1, completed.stdout
    assert "our-vllm" in completed.stderr


def test_container_check_is_fail_closed_when_docker_is_unreadable(tmp_path: Path):
    """★「读不到容器声明」≠「没有容器声明」⇒ docker 失败必须 rc≠0（fail-closed）。"""
    completed = _run(tmp_path, ["--visible", "3"], {"FAKE_DOCKER_FAIL": "1"})

    assert completed.returncode == 1, completed.stdout
    assert "容器声明" in completed.stderr


# ── 场景④（红）：显存 used 高但 util=0 ⇒ 非 0 ──────────────────────────────
@pytest.mark.parametrize(
    "card,row",
    [
        (1, "1, 16303, 81920, 0"),  # 远端机实测：他人常驻服务占 16 GiB，util 0
        (0, "0, 2404, 81920, 0"),  # 远端机实测：2.4 GiB ⇒ 也超过 2 GiB 默认上限
        (4, "4, 14041, 81920, 0"),
    ],
)
def test_scenario4_resident_service_rejects_even_with_util_zero(tmp_path: Path, card, row):
    """④已用显存超上限但 util=0 ⇒ rc≠0（"卡上有人占着"不能因为 util=0 就放行）。

    ★ 旧实现漏判：剩余 65 GiB ≥ 48 GiB 头寸 ∧ util == 0 ⇒ 旧实现 **rc=0**。
    """
    completed = _run(tmp_path, ["--visible", str(card)], {f"FAKE_SMI_ROW{card}": row})

    assert completed.returncode == 1, completed.stdout
    assert "空闲上限" in completed.stderr


def test_max_used_mib_is_an_escape_hatch_when_deliberately_shared(tmp_path: Path):
    """★显式放宽 ``--max-used-mib`` 时才允许共用卡（默认值不可被环境悄悄放宽）。"""
    completed = _run(
        tmp_path,
        ["--visible", "1", "--max-used-mib", "20480"],
        {"FAKE_SMI_ROW1": "1, 16303, 81920, 0"},
    )

    assert completed.returncode == 0, completed.stderr


# ── 场景⑤（红）：服务档头寸不足 ⇒ 非 0 ─────────────────────────────────────
#: 场景⑤的读数：剩余 61 440 MiB —— 够 48 GiB（49 152）但**不够**服务档 0.85 × 81 920
#: = 69 632 MiB。这正是"48 GiB 默认头寸对服务档远远不够"的形态（旧实现在 48 GiB
#: 位置下会 rc=0 放行，见 `evidence/red-green.md` 的 ⑤ 行）。
_SERVICE_TIER_ROW = "0, 20480, 81920, 0"


def test_scenario5_service_tier_position_ratio_rejects_48gib_position(tmp_path: Path):
    """⑤服务档真实口径 ``--position-ratio 0.85`` ⇒ 同一张卡 rc≠0 且点明头寸不足。

    ★ 旧实现漏判：旧实现既没有 ``--position-ratio``（rc=2 参数错误），也没有
    "服务档头寸"这个概念 ⇒ 它只能用 48 GiB 表达，于是上一条用例那样 **rc=0** 放行。
    """
    completed = _run(
        tmp_path,
        ["--visible", "0", "--position-ratio", "0.85", "--max-used-mib", "24576"],
        {"FAKE_SMI_ROW0": _SERVICE_TIER_ROW},
    )

    assert completed.returncode == 1, completed.stdout
    assert "头寸" in completed.stderr
    assert "69 632" in completed.stderr or "69632" in completed.stderr


def test_position_and_ratio_are_mutually_exclusive(tmp_path: Path):
    """⑤ 头寸只能有一个来源：``--position-mib`` 与 ``--position-ratio`` 同时给 ⇒ rc=2。"""
    completed = _run(
        tmp_path, ["--visible", "3", "--position-mib", "49152", "--position-ratio", "0.85"]
    )

    assert completed.returncode == 2, completed.stdout


# ── ④ 采样稳定：帧帧满足才算可用 ───────────────────────────────────────────
def test_frames_are_all_sampled(tmp_path: Path):
    """④ 采样稳定：``--frames 3`` ⇒ 必须真的查 3 轮（每帧都有 ``-i`` 查询）。"""
    bin_dir = tmp_path / "bin"
    _write_fakes(bin_dir)
    log = tmp_path / "calls.log"
    wrapper = bin_dir / "nvidia-smi"
    body = wrapper.read_text(encoding="utf-8").replace(
        'args="$*"', f'args="$*"\necho "$args" >> "{log}"'
    )
    wrapper.write_text(body, encoding="utf-8")
    wrapper.chmod(0o755)
    env = {**os.environ, "PATH": f"{bin_dir}{os.pathsep}{os.environ.get('PATH', '')}"}
    env.pop("NVIDIA_VISIBLE_DEVICES", None)
    completed = subprocess.run(
        [sys.executable, str(_SCRIPT), "--visible", "3", "--frames", "3", "--interval", "0"],
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )

    assert completed.returncode == 0, completed.stderr
    queries = [line for line in log.read_text(encoding="utf-8").splitlines() if "-i" in line]
    assert len(queries) >= 3 * 2, queries  # 3 帧 × (显存行 + uuid 映射)


# ── 兼容性与既有防线回归 ───────────────────────────────────────────────────
def test_idle_assert_accepts_cards_that_used_to_be_outside_the_allowlist(tmp_path: Path):
    """旧的集合外卡（0–3）现在**默认允许**——只要三判据都满足就通过。"""
    completed = _run(tmp_path, ["--visible", "0,3"])

    assert completed.returncode == 0, completed.stderr


def test_idle_assert_rejects_when_tier_position_env_raises_the_bar(tmp_path: Path):
    """★env 覆盖生效：剩余 81 916 MiB 在头寸 70 GiB 下够用、90 GiB 下拒绝。"""
    ok = _run(tmp_path, ["--visible", "4"], {"GRASPO_GPU_POSITION_MIB": str(70 * 1024)})
    assert ok.returncode == 0, ok.stderr

    bad = _run(tmp_path, ["--visible", "4"], {"GRASPO_GPU_POSITION_MIB": str(90 * 1024)})
    assert bad.returncode == 1, bad.stdout
    assert "头寸" in bad.stderr


@pytest.mark.parametrize(
    "row0,expect_in_stderr",
    [
        # total 81920、头寸默认 48 GiB：used 40000 ⇒ 剩余 41920 < 49152 ⇒ 拒绝
        ("4, 40000, 81920, 0", "头寸"),
        # util 高但显存没涨（别人在做推理）
        ("4, 4, 81920, 90", "利用率"),
    ],
)
def test_idle_assert_rejects_occupied_target_card(tmp_path: Path, row0, expect_in_stderr):
    """★负向：目标卡不满足显存/util 面判据 ⇒ rc=1 且消息点明是哪一条不满足。"""
    completed = _run(tmp_path, ["--visible", "4,5"], {"FAKE_SMI_ROW4": row0})

    assert completed.returncode == 1, completed.stdout
    assert expect_in_stderr in completed.stderr


def test_idle_assert_is_fail_closed_when_readings_are_empty(tmp_path: Path):
    """★负向（最关键）：**读不到任何读数 ⇒ rc≠0**。

    若这里 fail-open，"读不到"就会被当成"可用"，断言反过来变成抢卡许可证。
    """
    completed = _run(tmp_path, ["--visible", "4"], {"FAKE_SMI_EMPTY": "1"})

    assert completed.returncode != 0
    assert "fail-closed" in completed.stderr or "拒绝" in completed.stderr


def test_idle_assert_rejects_single_field_reading(tmp_path: Path):
    """★负向：解析不了的行（字段数不足）⇒ rc≠0，绝不静默通过。"""
    completed = _run(tmp_path, ["--visible", "4"], {"FAKE_SMI_ROW4": "4, 0"})

    assert completed.returncode != 0
    assert "解析" in completed.stderr


def test_idle_assert_rejects_unset_visible_devices(tmp_path: Path):
    """未锁卡（env 未设置）⇒ 拒绝（不放松既有防呆）。"""
    completed = _run(tmp_path, [], {})

    assert completed.returncode != 0
    assert "未设置" in completed.stderr


def test_json_output_lists_every_card_with_reasons_processes_and_containers(tmp_path: Path):
    """★``--json``：逐卡 available/card/reasons[]/processes[]/containers[]，供编排脚本消费。"""
    completed = _run(
        tmp_path,
        ["--visible", "2,7,3", "--json"],
        {
            **_REMOTE_ROWS,
            "FAKE_SMI_APPS": _REMOTE_APPS,
            "FAKE_DOCKER_NAMES": "pf-remote-transform-worker-1\n",
            "FAKE_DOCKER_INSPECT": _REMOTE_CONTAINERS,
        },
    )

    assert completed.returncode == 1, completed.stdout
    payload = json.loads(completed.stdout)
    assert payload["ok"] is False
    by_card = {card["card"]: card for card in payload["cards"]}
    assert by_card[3]["available"] is True
    assert by_card[2]["available"] is False
    assert any("3912182" in reason for reason in by_card[2]["reasons"])
    assert by_card[2]["processes"][0]["pid"] == 3912182
    assert by_card[7]["available"] is False
    assert by_card[7]["containers"] == ["pf-remote-transform-worker-1"]
    assert [card["card"] for card in payload["available"]] == [3]
    assert {card["card"] for card in payload["unavailable"]} == {2, 7}


def test_json_reports_available_cards_when_everything_is_free(tmp_path: Path):
    """★``--json`` 正例：全空闲 ⇒ ok=true、rc=0。"""
    completed = _run(tmp_path, ["--visible", "3", "--json"])

    assert completed.returncode == 0, completed.stderr
    payload = json.loads(completed.stdout)
    assert payload["ok"] is True and payload["unavailable"] == []
    assert payload["cards"][0]["max_used_mib"] == 2048.0
