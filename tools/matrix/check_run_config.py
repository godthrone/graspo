"""★ 产物级断言：**实际被执行的 config**（`runs/<T>/<T>/config.yaml`）必须与部署树 yaml 一致。

★ 教训（2026-10-13）：断言在"生成器写出的 yaml"上通过，而**被执行的那份 config** 里
字段是空的 ⇒ 导出未触发、`ec=1`。**校验对象必须 = 实际执行对象**。
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path


def read_final_formats(text: str) -> list[str] | None:
    """从 config/yaml 文本里取 `export.final_formats`；返回 None = 该段缺失。"""
    m = re.search(r"^export:\s*$([\s\S]*?)(?=^\S|\Z)", text, re.M)
    if not m:
        return None
    m2 = re.search(r"^\s+final_formats:\s*\[(.*?)\]\s*$", m.group(1), re.M)
    if not m2:
        return None
    return [x.strip().strip('"\'') for x in m2.group(1).split(",") if x.strip()]


def check(run_config_text: str, deploy_text: str) -> list[str]:
    """返回问题清单（空 = 通过）。"""
    want = read_final_formats(deploy_text) or []
    got = read_final_formats(run_config_text)
    if got is None:
        return ["运行 config 缺 `export.final_formats` 段"]
    if got != want:
        return [f"运行 config 的 final_formats={got} 与部署树={want} 不一致"]
    return []


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-config", required=True, help="runs/<T>/<T>/config.yaml")
    ap.add_argument("--deploy-yaml", required=True, help="部署树 samples/configs/.../<T>.yaml")
    a = ap.parse_args()
    problems = check(Path(a.run_config).read_text(encoding="utf-8"),
                     Path(a.deploy_yaml).read_text(encoding="utf-8"))
    if problems:
        print("FAIL-CLOSED：运行 config 与部署树不一致 ⇒ ⛔ 不得起跑：" + "；".join(problems))
        return 1
    print("OK：运行 config 的 export.final_formats 与部署树一致")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
