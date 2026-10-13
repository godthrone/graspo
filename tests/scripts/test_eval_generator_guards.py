"""★ 回归（T022 事故）：评测脚本生成器与矩阵生成器的两处护栏。

背景：评测脚本模板原先在 eval 命令之后**立即** `docker stop`，没有断言评测真的产出了报告。
T022 首跑即因此失败 —— 服务在健康检查通过后随即被优雅关闭，在途请求 `RemoteDisconnected`，
after 报告从未写出。同时矩阵生成器缺少"起跑前打印 config 路径 ＋ sha256"的留痕。
"""
import re
from pathlib import Path


def test_eval_generator_stops_only_after_report_exists():
    """★ ⓐ 渲染脚本里每个 stop 都必须晚于对应 eval，且带报告存在性断言。"""
    src = Path("scripts/gen_eval.py").read_text(encoding="utf-8")
    assert "报告缺失" in src, "生成器必须含缺报告不 stop 的分支"
    assert "eval_report.json" in src
    # 每个 stop 之前必须紧邻一次报告存在性判断
    for phase in ("after", "base"):
        m = re.search(r'if \[ -f "\$EV/[^"]*reports/%s/eval_report\.json" \]; then\s*\n\s*docker stop' % phase, src)
        assert m, f"{phase}: stop 必须被报告存在性断言守护"
    # stop 不得出现在同行的 eval 之后（旧缺陷形态）
    assert not re.search(r'say "eval (after|base) rc=\$\?"; docker stop', src), "stop 不得紧跟 eval 同行"


def test_eval_generator_keeps_service_on_abnormal_exit():
    """★ 中途退出不得拆服务（否则失败原因不可见、也无法重试）。"""
    src = Path("scripts/gen_eval.py").read_text(encoding="utf-8")
    assert re.search(r"trap .*INT TERM", src), "生成器必须含 INT/TERM 的 trap"


def test_matrix_generator_prints_config_sha():
    """★ ⓑ 生成的 entry.sh 必须含 [config-sha] 行（路径 ＋ 64 位 sha）。"""
    src = Path("tests/e2e/generate_matrix.py").read_text(encoding="utf-8")
    assert "[config-sha]" in src
    assert re.search(r'\[config-sha\][^\n]*sha256sum', src)
    assert "TIER_CONFIG_PATH" in src


def test_gen_tier_json_asserts_structured_params():
    """★ 回归：params.mode/parallel 是结构化字段，生成时必须断言（T022 曾误填叙述长文本）。"""
    src = Path("scripts/gen_tier_json.py").read_text(encoding="utf-8")
    assert "MODE_ALLOWED" in src or "mode" in src
    assert "TP=" in src, "parallel 必须按 TP=.. PP=.. DP=.. 断言"
