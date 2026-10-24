"""`scripts/run_guards.py` 的用例（§7 形态谓词 / §8 新鲜度 / §10 Mamba 并发）。

每族各含**正例 + 负例**；只创建临时目录，不碰 GPU / 容器 / 真实产物。
"""

import os
import sys
import time
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "scripts"))
import run_guards as g  # noqa: E402


# ---------- §7 形态谓词 ----------
def test_form_accepts_full_hf(tmp_path: Path):
    (tmp_path / "config.json").write_text("{}")
    (tmp_path / "model-00001-of-00004.safetensors").write_text("x")
    form, ok, msg = g.detect_form_and_validate(str(tmp_path))
    assert ok and form == "full", msg


def test_form_accepts_peft_adapter_without_config_json(tmp_path: Path):
    """★ 关键回归：adapter 目录**没有** config.json（T008 曾被误拒）⇒ 必须通过。"""
    (tmp_path / "adapter_config.json").write_text("{}")
    (tmp_path / "adapter_model.safetensors").write_text("x")
    form, ok, msg = g.detect_form_and_validate(str(tmp_path))
    assert ok and form == "adapter", msg


def test_form_accepts_native_rank_shards(tmp_path: Path):
    (tmp_path / "rank_00000_tp_00_pp_00.pt").write_text("x")
    form, ok, msg = g.detect_form_and_validate(str(tmp_path))
    assert ok and form == "native_rank", msg


def test_form_rejects_missing_directory(tmp_path: Path):
    ok, form = g.detect_form_and_validate(str(tmp_path / "nope"))[1], None
    assert ok is False


def test_form_rejects_unknown_layout(tmp_path: Path):
    (tmp_path / "readme.txt").write_text("nothing useful")
    form, ok, msg = g.detect_form_and_validate(str(tmp_path))
    assert ok is False and form == "unknown", msg


# ---------- §8 产物新鲜度 ----------
def test_fresh_pre_passes_when_absent(tmp_path: Path):
    ok, msg = g.check_fresh(str(tmp_path / "merged"), "pre")
    assert ok, msg


def test_fresh_pre_rejects_stale_directory(tmp_path: Path):
    d = tmp_path / "merged"
    d.mkdir()
    (d / "model-00001-of-00018.safetensors").write_text("stale")
    ok, msg = g.check_fresh(str(d), "pre")
    assert ok is False and "陈旧" in msg, msg


def test_fresh_post_passes_for_new_dir_with_weights(tmp_path: Path):
    d = tmp_path / "merged"
    d.mkdir()
    (d / "model.safetensors").write_text("x")
    ok, msg = g.check_fresh(str(d), "post", since_epoch=time.time() - 60)
    assert ok, msg


def test_fresh_post_rejects_dir_without_weights(tmp_path: Path):
    d = tmp_path / "merged"
    d.mkdir()
    (d / "config.json").write_text("{}")
    ok, msg = g.check_fresh(str(d), "post")
    assert ok is False and "无权重" in msg, msg


def test_fresh_post_rejects_older_than_batch(tmp_path: Path):
    d = tmp_path / "merged"
    d.mkdir()
    (d / "model.safetensors").write_text("x")
    ok, msg = g.check_fresh(str(d), "post", since_epoch=time.time() + 3600)
    assert ok is False and "疑似陈旧" in msg, msg


# ---------- §10 Mamba 并发 ----------
def test_mamba_accepts_within_blocks():
    ok, msg = g.check_mamba(192, 232)
    assert ok, msg


def test_mamba_accepts_equal_bound():
    ok, msg = g.check_mamba(232, 232)
    assert ok, msg


def test_mamba_rejects_default_over_blocks():
    """★ 关键回归：缺省 1024 > 232 blocks（T024 实测事故）。"""
    ok, msg = g.check_mamba(1024, 232)
    assert ok is False and "Engine core initialization failed" in msg, msg


# ---------- §17 结构性兜底：我方占卡台账（含 loading 态） ----------
import json as _json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "scripts"))
import run_guards as rg  # noqa: E402


def _inspect(name, status, gpus):
    return {"Name": "/" + name, "State": {"Status": status},
            "HostConfig": {"DeviceRequests": [{"DeviceIDs": [gpus]}] if gpus else []}}


def test_busy_cards_covers_running_and_loading():
    """★ 回归（T002/§17）：loading 中的容器同样禁用其卡（用户可传 names 过滤我方）。"""
    data = _json.dumps([_inspect("t044-train", "running", "6,7"),
                        _inspect("t002-vllm-after", "created", "6")])
    busy, notes = rg.busy_cards_from_docker(names=["t044-train", "t002-vllm-after"], docker_json=data)
    assert {"6", "7"} <= busy, (busy, notes)
    assert any("created" in n for n in notes), notes


def test_busy_cards_ignores_containers_without_gpus():
    data = _json.dumps([_inspect("neighbour", "running", None)])
    busy, _ = rg.busy_cards_from_docker(names=["neighbour"], docker_json=data)
    assert busy == set()


def test_exclude_cards_held_by_us_hard_excludes():
    """★ 择卡硬排除：loading 态的卡不得被选中。"""
    assert rg.exclude_cards_held_by_us(["1", "3", "6", "7"], {"6", "7"}) == ["1", "3"]


def test_parse_docker_inspect_stream_multiple_objects():
    """★ 回归：`docker inspect c1 c2` 是**多对象流**（非数组）⇒ 两个都要识别。"""
    text = ('[{"Name":"/a","State":{"Status":"running"},"HostConfig":{"DeviceRequests":[{"DeviceIDs":["0"]}]}}]'
            '[{"Name":"/b","State":{"Status":"created"},"HostConfig":{"DeviceRequests":[{"DeviceIDs":["6,7"]}]}}]')
    objs = rg.parse_docker_inspect_stream(text)
    assert len(objs) == 2, objs
    busy, notes = rg.busy_cards_from_docker(names=["a", "b"], docker_json=_json.dumps(objs))
    assert {"0", "6", "7"} <= busy, (busy, notes)


def test_busy_cards_cli_end_to_end():
    ps = "a\nb\n"
    ins = ('[{"Name":"/a","State":{"Status":"running"},"HostConfig":{"DeviceRequests":[{"DeviceIDs":["1,3"]}]}}]'
           '[{"Name":"/b","State":{"Status":"created"},"HostConfig":{"DeviceRequests":[{"DeviceIDs":["6"]}]}}]')
    busy, csv = rg.busy_cards_cli(ps, ins)
    assert csv == "1,3,6", (busy, csv)


def test_busy_cards_from_nvidia_visible_devices_env():
    """★ 真机实测形态：容器用 `-e NVIDIA_VISIBLE_DEVICES=1,3,6,7` 传卡 ⇒ DeviceRequests 为 null。"""
    item = {"Name": "/t045", "State": {"Status": "running"},
            "HostConfig": {"DeviceRequests": None},
            "Config": {"Env": ["NVIDIA_VISIBLE_DEVICES=1,3,6,7", "PATH=/usr/bin"]}}
    busy, notes = rg.busy_cards_from_docker(names=["t045"], docker_json=_json.dumps([item]))
    assert busy == {"1", "3", "6", "7"}, (busy, notes)


# ---------- ★ 三判据并取（2026-10-25）：compute 进程快照 = cards-busy 的等价输入 ----------
def test_busy_cards_from_compute_apps_catches_host_processes():
    """★ 缺陷回归：**宿主进程占卡** docker 面看不见 ⇒ 必须由 compute-apps 快照判 busy。

    228 实测：GPU2 上有他人在宿主直接起的推理进程（13.3 GiB、util 0 %）而无任何容器。
    """
    apps = "GPU-cbefcbab-7651-516c-da18-797296a24902, 3912182, 13326, python\n"
    umap = (
        "0, GPU-6b18a0f1-01f5-c1e3-3233-f5b31a7eb108\n"
        "2, GPU-cbefcbab-7651-516c-da18-797296a24902\n"
    )
    busy, notes = rg.busy_cards_from_compute_apps(apps, umap)
    assert busy == {"2"}, (busy, notes)
    assert any("3912182" in n and "GPU2" in n for n in notes), notes


def test_busy_cards_cli_unions_docker_and_compute_apps():
    """★ 两类输入是**等价输入**，取并集：docker 的卡 + 宿主进程的卡都要判 busy。"""
    ps = "our-vllm\n"
    ins = (
        '[{"Name":"/our-vllm","State":{"Status":"running"},'
        '"HostConfig":{"DeviceRequests":[{"DeviceIDs":["7"]}]}}]'
    )
    apps = "GPU-fake2, 3912182, 13326, python\n"
    umap = "2, GPU-fake2\n"
    busy, csv = rg.busy_cards_cli(ps, ins, apps, umap)
    assert csv == "2,7", (busy, csv)


def test_busy_cards_from_compute_apps_is_fail_closed_on_unmapped_uuid():
    """★ fail-closed：进程快照里有 uuid 不在映射里 ⇒ 必须报错（⛔ 不当成"没有进程"）。"""
    apps = "GPU-unknown-uuid, 3912182, 13326, python\n"
    with pytest.raises(ValueError):
        rg.busy_cards_from_compute_apps(apps, "2, GPU-fake2\n")


def test_busy_cards_from_compute_apps_requires_the_uuid_map():
    """★ fail-closed：只给进程快照、不给 uuid 映射 ⇒ 报错（无法归卡）。"""
    with pytest.raises(ValueError):
        rg.busy_cards_from_compute_apps("GPU-fake2, 3912182, 13326, python\n", "")


def test_busy_cards_cli_without_compute_apps_keeps_old_behavior():
    """兼容：不给 ``--compute-apps-file`` 时行为与旧版逐字一致。"""
    ps = "a\n"
    ins = (
        '[{"Name":"/a","State":{"Status":"running"},'
        '"HostConfig":{"DeviceRequests":[{"DeviceIDs":["1,3"]}]}}]'
    )
    busy, csv = rg.busy_cards_cli(ps, ins)
    assert csv == "1,3", (busy, csv)
