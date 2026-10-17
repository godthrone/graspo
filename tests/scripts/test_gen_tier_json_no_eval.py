"""★ 回归（T018，2026-10-14）：`gen_tier_json.py` 的"无评测分支"。

不可评档（如产物为 PP 分片、拼接未实现）既不能评出读数，也不能被当成
"达标性失败"。故：三件为空时必须给非空原因，且原因**不得含"效果"字样**
（与矩阵侧 `backfill_matrix.py::not_evaluable_ok` 同判据）。
"""
import json
import os
import subprocess
import sys
from pathlib import Path

GEN = "scripts/gen_tier_json.py"
BASE = [sys.executable, GEN, "--tier", "T999", "--mode", "full", "--parallel",
        "TP=1 PP=1 DP=1", "--model", "9B", "--label", "SFT", "--nproc", "1",
        "--steps", "1", "--recon", "1", "--backend", "native", "--host", "example-host",
        "--gpu", "0", "--exit-code", "1", "--non-graspo",
        "--evidence", "e.txt", "--command", "x", "--out", "/tmp/_ne_test.json"]


def _run(extra, env_extra=None):
    env = {k: v for k, v in os.environ.items() if k != "GRASPO_HOST_LABEL"}
    env.update(env_extra or {})
    return subprocess.run(BASE + extra, capture_output=True, text=True,
                          cwd=Path(__file__).resolve().parents[2], env=env)


def test_no_eval_reason_accepted():
    """★ 正例：三件为空 + 原因非空 ⇒ 放行，且三件落 None。"""
    r = _run(["--no-eval", "--not-evaluable-reason", "不可评：PP 分片拼接未实现"])
    assert r.returncode == 0, r.stderr[-400:]
    d = json.load(open("/tmp/_ne_test.json", encoding="utf-8"))
    assert d["after_acc"] is None and d["delta_pp"] is None
    assert d["not_evaluable_reason"]


def test_no_eval_empty_reason_rejected():
    """★ 负例：三件为空但原因为空 ⇒ FAIL-CLOSED。"""
    r = _run(["--no-eval", "--not-evaluable-reason", "  "])
    assert r.returncode != 0
    assert "not-evaluable-reason" in (r.stderr + r.stdout)


def test_no_eval_reason_with_effect_word_rejected():
    """★ 负例：原因含"效果"字样 ⇒ FAIL-CLOSED（不可评 ≠ 达标性失败）。"""
    r = _run(["--no-eval", "--not-evaluable-reason", "效果未达标"])
    assert r.returncode != 0
    assert "效果" in (r.stderr + r.stdout)


def test_parallel_must_match_nproc():
    """★ 回归（T018 真错位）：`params.parallel` 抄了别档模板 ⇒ TP×PP×DP 与卡数不符 ⇒ FAIL-CLOSED。"""
    r = _run(["--no-eval", "--not-evaluable-reason", "不可评：测试用",
              "--parallel", "TP=1 PP=1 DP=1", "--nproc", "4"])
    assert r.returncode != 0, "★ 应 fail-closed 却放行"
    assert "parallel" in (r.stderr + r.stdout)


_CFG_27B_LORA = "/tmp/_id_cfg.yaml"


def _write_cfg(text):
    open(_CFG_27B_LORA, "w", encoding="utf-8").write(text)


def test_identity_rejects_copied_template():
    """★ 回归（T042 真错位）：27B×LoRA 档被抄成 9B×full ⇒ FAIL-CLOSED。"""
    _write_cfg("Qwen3.8-27B\ntuner_type: lora\nbackend: native\ntp_size: 1\npp_size: 1\ndp_size: 4\n")
    r = _run(["--no-eval", "--not-evaluable-reason", "不可评：测试用",
              "--identity-from-config", _CFG_27B_LORA,
              "--model", "9B", "--mode", "full", "--backend", "native",
              "--parallel", "TP=1 PP=1 DP=4", "--nproc", "4"])
    assert r.returncode != 0, "★ 应 fail-closed 却放行"
    out = r.stderr + r.stdout
    assert "身份字段与该档配置不一致" in out and "9B" in out


def test_identity_accepts_matching_model_scale():
    """★ 正例：身份与配置同源 ⇒ 放行；★ 同时覆盖"model 写错规模必拒"的反面。"""
    _write_cfg("Qwen3.8-27B\ntuner_type: lora\nbackend: native\ntp_size: 1\npp_size: 1\ndp_size: 4\n")
    r = _run(["--no-eval", "--not-evaluable-reason", "不可评：测试用",
              "--identity-from-config", _CFG_27B_LORA,
              "--model", "27B", "--mode", "LoRA", "--backend", "native",
              "--parallel", "TP=1 PP=1 DP=4", "--nproc", "4"])
    assert r.returncode == 0, r.stderr[-400:]


# ---------- 去内部化（2026-10-17）：`--host` 无内部默认值 ----------
# 背景：`--host` 的 argparse 默认值原先写死「机器号 + 真内网主机名」——运行环境
# 信息按宪法 §16.2 只许进 `.local/`。现改为 `--host` / `GRASPO_HOST_LABEL` 二者
# 之一，都没有即 FAIL-CLOSED。★ 下面三条是**能失败对照**：把"缺失"钉成必红。
_HOST_ENV = "GRASPO_HOST_LABEL"


def _base_without_host() -> list[str]:
    """BASE 去掉 `--host <value>` 两枚 token（用于验证缺失时的 fail-closed）。"""
    out, it = [], iter(BASE)
    for tok in it:
        if tok == "--host":
            next(it)  # 丢弃它的值
            continue
        out.append(tok)
    return out


def _run_bare(extra, env_extra=None):
    env = {k: v for k, v in os.environ.items() if k != _HOST_ENV}
    env.update(env_extra or {})
    return subprocess.run(_base_without_host() + extra, capture_output=True, text=True,
                          cwd=Path(__file__).resolve().parents[2], env=env)


def test_host_missing_and_env_missing_fails_closed():
    """★ 负例：`--host` 未给且 `GRASPO_HOST_LABEL` 未设 ⇒ 必须明确报错退出。

    ⛔ 不得静默回落到任何内部默认值（旧行为：机器号 + 真内网主机名）。
    """
    r = _run_bare(["--no-eval", "--not-evaluable-reason", "不可评：测试用"])
    assert r.returncode != 0, "★ 应 fail-closed 却放行"
    out = r.stderr + r.stdout
    assert "FAIL-CLOSED" in out and _HOST_ENV in out, out[-400:]


def test_host_from_env_is_accepted_and_recorded():
    """★ 正例：`--host` 缺省时由 `GRASPO_HOST_LABEL` 供给 ⇒ 放行且落盘同值。"""
    r = _run_bare(["--no-eval", "--not-evaluable-reason", "不可评：测试用"],
                  env_extra={_HOST_ENV: "example-host"})
    assert r.returncode == 0, r.stderr[-400:]
    d = json.load(open("/tmp/_ne_test.json", encoding="utf-8"))
    assert d["provenance"]["run_environment"]["host"] == "example-host"


def test_explicit_host_wins_over_env():
    """★ 优先级：显式 `--host` > env（CLI 语义不变，env 只是缺失时的兜底源）。"""
    r = _run(["--no-eval", "--not-evaluable-reason", "不可评：测试用"],
             env_extra={_HOST_ENV: "env-host"})
    assert r.returncode == 0, r.stderr[-400:]
    d = json.load(open("/tmp/_ne_test.json", encoding="utf-8"))
    assert d["provenance"]["run_environment"]["host"] == "example-host"
