"""R1 / R3 契约与判别力测试（**零 GPU**：纯 CPU + 真 backward + gloo 双 rank 子进程）。

**背景（2026-09-23 复核 T018 逐 rank 原始读数后定）**：native PP 的 fail-closed 判据此前
**只看末 stage 的 loss 是否有限**，完全不看梯度是否有限 —— T018（9B·SFT·全参·native·4 卡
``pp=4``）实测：

- ``step3``：rank2 ``grad_norm_mean=nan`` 而 ``optimizer_steps=1``、``skipped_nonfinite=0``
  ⇒ **NaN 梯度被照常 ``optimizer.step()`` 写进权重**（``trainable_norm_after=nan``）；
- ``step4``：rank3 梯度 NaN 而它的 ``loss_mean=9.8125`` **仍然有限** ⇒ 仍照常 step；
- ``step5``：直到末 stage 的 **loss** 变 NaN 才硬失败，而报错文案写"本步梯度含非有限值"；
- 且 raise 早于 metrics 构造 ⇒ **崩溃步没有任何 rank_metrics 行**（取证只能靠推理）。

本模块锁死修复后的四条契约（都不需要 GPU）：

1. **纯判据**：``grad_gate_verdict`` 的规则表（新增"梯度非有限"与"某 rank 零梯度"两条硬失败）；
2. **探针**：``grad_finiteness_report`` 能在真 backward 产生的 NaN 梯度上定位到**张量名**与
   ``max|g|``（"注入 NaN 梯度"的零 GPU 判别力用例）；
3. **跨 rank 归约**：gloo 双 rank 下"rank1 注入 NaN ⇒ 两侧一致判失败"（真进程组 e2e）；
4. **落盘侧**：两个新 phase 已登记、训练模块**不重复字面量**（§1.4），
   以及 collector 生产端 ``global_skipped_nonfinite_sum`` 的**生产→消费回路**。
"""

from __future__ import annotations

import importlib.util
import json
import os
import socket
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

torch = pytest.importorskip("torch", reason="R1/R3 契约测试需要 torch（CPU 即可）")

from graspo.core.result_judge import DIAGNOSTIC_PHASES, STEP_METRICS_PHASES  # noqa: E402
from graspo.flow.adapters.models.common.grad_probe import (  # noqa: E402
    FAIL_CLOSED_PHASE,
    GRAD_FAIL_GRAD_NONFINITE,
    GRAD_FAIL_GRAD_UNPOPULATED,
    GRAD_FAIL_LOSS_NONFINITE,
    PP_NUMERIC_PROBE_PHASE,
    grad_fail_reason_text,
    grad_finiteness_report,
    grad_gate_verdict,
    reduced_grad_flags,
    step_index_one_based,
    tensor_is_finite,
)

_REPO = Path(__file__).resolve().parents[3]
_TRAINING_MODULES = (
    _REPO / "src" / "graspo" / "flow" / "adapters" / "models" / "qwen35_36" / "training_sft.py",
    _REPO / "src" / "graspo" / "flow" / "adapters" / "models" / "qwen35_36" / "training.py",
)
_COLLECTOR = _REPO / "scripts" / "collect_results.py"


class _ParamBag(torch.nn.Module):
    """最小参数容器：直接给 ``.grad`` 赋值，模拟"backward 已完成"的任意梯度状态。"""

    def __init__(self, specs: dict[str, tuple[torch.Tensor, torch.Tensor | None]]) -> None:
        super().__init__()
        for name, (weight, grad) in specs.items():
            param = torch.nn.Parameter(weight.clone())
            if grad is not None:
                param.grad = grad.clone()
            self.register_parameter(name.replace(".", "_"), param)
            # 用点号名字复刻真实 ``named_parameters()``（如 ``layers.0.attn.q_proj.weight``）：
            # register_parameter 不接受点号，故用子模块承载。
            del self._parameters[name.replace(".", "_")]
            parent = self
            parts = name.split(".")
            for part in parts[:-1]:
                if not hasattr(parent, part):
                    parent.add_module(part, torch.nn.Module())
                parent = getattr(parent, part)
            parent.register_parameter(parts[-1], param)


# ── 1) tensor_is_finite / 探针 ───────────────────────────────────────────────


class TestTensorIsFinite:
    def test_healthy_tensor_is_finite(self) -> None:
        assert tensor_is_finite(torch.zeros(8)) is True

    def test_nan_is_not_finite(self) -> None:
        value = torch.zeros(8)
        value[5] = float("nan")
        assert tensor_is_finite(value) is False

    def test_inf_is_not_finite(self) -> None:
        value = torch.zeros(4)
        value[0] = float("inf")
        assert tensor_is_finite(value) is False

    def test_empty_tensor_is_finite(self) -> None:
        assert tensor_is_finite(torch.zeros(0)) is True

    def test_chunked_scan_catches_nonfinite_in_later_chunk(self) -> None:
        """分块不是"只看第一块"：非有限值落在**最后一块**也必须被抓到。"""
        value = torch.zeros(10)
        value[-1] = float("nan")
        assert tensor_is_finite(value, chunk_elements=2) is False

    def test_bf16_large_values_are_finite(self) -> None:
        """极大但有限的 bf16（T017/T035 的 1e13 量级）**不得**被误判为非有限。"""
        assert tensor_is_finite(torch.full((4,), 1e13, dtype=torch.bfloat16)) is True


class TestGradFinitenessReport:
    def _bag(self) -> _ParamBag:
        healthy = torch.ones(4)
        nan_grad = torch.ones(4)
        nan_grad[2] = float("nan")
        return _ParamBag(
            {
                "layers.0.attn.q_proj.weight": (torch.zeros(4), healthy),
                "layers.1.mlp.gate.weight": (torch.zeros(4), nan_grad),
                "norm.weight": (torch.zeros(4), None),  # 未收到梯度
            }
        )

    def test_counts_and_first_nonfinite_name(self) -> None:
        report = grad_finiteness_report(self._bag().named_parameters(), with_max_abs=True)
        assert report["trainable_tensor_count"] == 3
        assert report["grad_populated_count"] == 2  # norm 未收到梯度
        assert report["grad_missing_count"] == 1
        assert report["grad_nonfinite_count"] == 1
        names = [item["name"] for item in report["first_nonfinite_grad_names"]]
        assert names == ["layers.1.mlp.gate.weight"]
        assert report["grad_argmax_tensor_name"] == "layers.1.mlp.gate.weight"

    def test_nan_makes_max_abs_nan(self) -> None:
        """``max|g|`` 必须把 NaN 原样透出（诊断要看到"荒唐值"，不得过滤）。"""
        report = grad_finiteness_report(self._bag().named_parameters(), with_max_abs=True)
        assert report["grad_max_abs"] != report["grad_max_abs"]  # NaN != NaN

    def test_max_abs_skipped_by_default(self) -> None:
        """默认不做第二遍 ``max|·|`` 归约（每步都要 cheap）⇒ 该字段为 None。"""
        report = grad_finiteness_report(self._bag().named_parameters())
        assert report["grad_max_abs"] is None
        assert report["grad_argmax_tensor_name"] is None

    def test_all_missing_grads_reports_zero_populated(self) -> None:
        empty = _ParamBag({"a.weight": (torch.zeros(2), None)})
        report = grad_finiteness_report(empty.named_parameters())
        assert report["grad_populated_count"] == 0
        assert report["trainable_tensor_count"] == 1
        assert report["grad_nonfinite_count"] == 0

    def test_frozen_params_are_not_counted(self) -> None:
        bag = self._bag()
        for param in bag.parameters():
            param.requires_grad = False
        report = grad_finiteness_report(bag.named_parameters())
        assert report["trainable_tensor_count"] == 0
        assert report["grad_populated_count"] == 0

    def test_probe_does_not_modify_grads(self) -> None:
        bag = self._bag()
        before = [None if p.grad is None else p.grad.clone() for p in bag.parameters()]
        grad_finiteness_report(bag.named_parameters(), with_max_abs=True)
        for original, param in zip(before, bag.parameters(), strict=True):
            if original is None:
                assert param.grad is None
            else:
                # NaN 梯度：``torch.equal`` 对 NaN 恒 False，用 equal_nan 比较
                assert torch.allclose(original, param.grad, equal_nan=True)


# ── 2) 纯判据（规则表）────────────────────────────────────────────────────


class TestGradGateVerdict:
    @pytest.mark.parametrize(
        ("loss_all_finite", "nonfinite_any", "unpopulated_any", "failed", "reason"),
        [
            # 新增核心判据：任一 rank 梯度非有限 ⇒ 硬失败（无论 loss 是否有限）
            (True, True, False, True, GRAD_FAIL_GRAD_NONFINITE),
            (False, True, False, True, GRAD_FAIL_GRAD_NONFINITE),
            # 某 rank 零梯度 + loss 全有限 ⇒ 硬失败
            (True, False, True, True, GRAD_FAIL_GRAD_UNPOPULATED),
            # 既有判据保留：loss 非有限 ⇒ 硬失败（此时"零梯度"由 loss 归因，不重复报）
            (False, False, True, True, GRAD_FAIL_LOSS_NONFINITE),
            (False, False, False, True, GRAD_FAIL_LOSS_NONFINITE),
            # 健康步不得误拦
            (True, False, False, False, ""),
        ],
    )
    def test_truth_table(
        self,
        loss_all_finite: bool,
        nonfinite_any: bool,
        unpopulated_any: bool,
        failed: bool,
        reason: str,
    ) -> None:
        assert grad_gate_verdict(
            loss_all_finite=loss_all_finite,
            grad_nonfinite_any=nonfinite_any,
            grad_unpopulated_any=unpopulated_any,
        ) == (failed, reason)

    def test_every_reason_has_human_text(self) -> None:
        for reason in (
            GRAD_FAIL_GRAD_NONFINITE,
            GRAD_FAIL_GRAD_UNPOPULATED,
            GRAD_FAIL_LOSS_NONFINITE,
        ):
            text = grad_fail_reason_text(reason)
            assert text and not text.startswith("未知")
        assert grad_fail_reason_text("不存在的码").startswith("未知")

    def test_loss_reason_text_does_not_blame_gradients(self) -> None:
        """文案纠偏：loss 非有限不得说成"梯度含非有限值"（旧文案的指错方向）。"""
        text = grad_fail_reason_text(GRAD_FAIL_LOSS_NONFINITE)
        assert "loss" in text and "梯度含非有限值" not in text


# ── 3) 真 backward 注入 NaN 梯度（零 GPU 判别力用例）─────────────────────────


class TestNaNInjectionWithRealBackward:
    def test_nan_gradient_is_detected_and_fails_closed(self) -> None:
        model = torch.nn.Linear(4, 2)
        out = model(torch.ones(1, 4))
        # 注入式构造：乘 NaN ⇒ backward 把 NaN 写进**真** grad
        (out * float("nan")).sum().backward()
        report = grad_finiteness_report(model.named_parameters(), with_max_abs=True)
        assert report["grad_nonfinite_count"] == 2  # weight + bias
        assert report["grad_populated_count"] == 2
        failed, reason = grad_gate_verdict(
            loss_all_finite=True,  # ★ 关键：loss 侧"看起来正常"，旧判据会放行
            grad_nonfinite_any=bool(report["grad_nonfinite_count"]),
            grad_unpopulated_any=False,
        )
        assert (failed, reason) == (True, GRAD_FAIL_GRAD_NONFINITE)

    def test_healthy_backward_passes(self) -> None:
        model = torch.nn.Linear(4, 2)
        model(torch.ones(1, 4)).sum().backward()
        report = grad_finiteness_report(model.named_parameters(), with_max_abs=True)
        assert report["grad_nonfinite_count"] == 0
        assert grad_gate_verdict(
            loss_all_finite=True,
            grad_nonfinite_any=False,
            grad_unpopulated_any=False,
        ) == (False, "")


# ── 4) 跨 rank 归约 + 真进程组（gloo）e2e ───────────────────────────────────


class TestReducedGradFlagsWithoutDistributed:
    def test_falls_back_to_local_when_not_initialized(self) -> None:
        healthy = {"grad_nonfinite_count": 0, "grad_populated_count": 3}
        assert reduced_grad_flags(local_report=healthy, distributed=None) == (False, False)

        nan_report = {"grad_nonfinite_count": 1, "grad_populated_count": 3}
        assert reduced_grad_flags(local_report=nan_report, distributed=None) == (True, False)

        empty_report = {"grad_nonfinite_count": 0, "grad_populated_count": 0}
        assert reduced_grad_flags(local_report=empty_report, distributed=None) == (False, True)

    def test_uninitialized_real_dist_is_noop(self) -> None:
        import torch.distributed as dist

        if dist.is_initialized():  # pragma: no cover - 正常单测进程不会初始化
            pytest.skip("进程组已被初始化，本用例前提不成立")
        report = {"grad_nonfinite_count": 0, "grad_populated_count": 2}
        assert reduced_grad_flags(local_report=report, distributed=dist) == (False, False)


_GLOO_CHILD = textwrap.dedent(
    """
    import json, sys
    import torch, torch.distributed as dist

    from graspo.flow.adapters.models.common.grad_probe import (
        grad_finiteness_report, grad_gate_verdict, reduced_grad_flags,
    )

    rank, world_size, port, mode, out_path = (
        int(sys.argv[1]), int(sys.argv[2]), sys.argv[3], sys.argv[4], sys.argv[5]
    )
    dist.init_process_group(
        "gloo", rank=rank, world_size=world_size,
        init_method=f"tcp://127.0.0.1:{port}",
    )
    model = torch.nn.Linear(4, 2)
    model(torch.ones(1, 4)).sum().backward()
    if mode == "rank1_nan" and rank == 1:
        for param in model.parameters():
            param.grad[0] = float("nan")
    report = grad_finiteness_report(model.named_parameters())
    nonfinite_any, unpopulated_any = reduced_grad_flags(
        local_report=report, distributed=dist, device=torch.device("cpu")
    )
    failed, reason = grad_gate_verdict(
        loss_all_finite=True,
        grad_nonfinite_any=nonfinite_any,
        grad_unpopulated_any=unpopulated_any,
    )
    dist.destroy_process_group()
    with open(out_path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps({"rank": rank, "failed": failed, "reason": reason}) + "\\n")
    """
)


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _run_gloo_pair(tmp_path: Path, mode: str) -> list[dict]:
    """起两个真进程（gloo/CPU）跑同一段判据，返回两侧结果。"""
    script = tmp_path / "gloo_child.py"
    script.write_text(_GLOO_CHILD, encoding="utf-8")
    out_path = tmp_path / f"out_{mode}.jsonl"
    port = _free_port()
    env = dict(os.environ)
    env["PYTHONPATH"] = str(_REPO / "src")
    procs = [
        subprocess.Popen(  # noqa: S603 - 固定 argv，无 shell
            [sys.executable, str(script), str(rank), "2", str(port), mode, str(out_path)],
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        for rank in (0, 1)
    ]
    try:
        for proc in procs:
            _, stderr = proc.communicate(timeout=120)
            assert proc.returncode == 0, stderr
    finally:
        for proc in procs:
            if proc.poll() is None:  # pragma: no cover - 兜底，避免留下子进程
                proc.kill()
    rows = [json.loads(line) for line in out_path.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 2, rows
    return sorted(rows, key=lambda row: row["rank"])


def test_gloo_pair_rank1_nan_fails_on_both_ranks(tmp_path: Path) -> None:
    """★ 跨 rank 判据 e2e（零 GPU）：rank1 注入 NaN 梯度 ⇒ **两侧一致**判失败。

    这正是 T018 的形状：loss 侧全有限（本轮不注入 loss NaN），只有 rank2 的梯度是 NaN；
    旧判据（只看 loss）会在这两个 rank 上**同时放行**并照常 step。
    """
    import torch.distributed as dist

    if not dist.is_gloo_available():  # pragma: no cover - 环境性
        pytest.skip("torch.distributed gloo 不可用")
    rows = _run_gloo_pair(tmp_path, "rank1_nan")
    assert [row["failed"] for row in rows] == [True, True], rows
    assert {row["reason"] for row in rows} == {GRAD_FAIL_GRAD_NONFINITE}, rows


def test_gloo_pair_healthy_passes_on_both_ranks(tmp_path: Path) -> None:
    """对照：健康梯度在两个 rank 上都**不得**被误拦。"""
    import torch.distributed as dist

    if not dist.is_gloo_available():  # pragma: no cover - 环境性
        pytest.skip("torch.distributed gloo 不可用")
    rows = _run_gloo_pair(tmp_path, "healthy")
    assert [row["failed"] for row in rows] == [False, False], rows
    assert {row["reason"] for row in rows} == {""}, rows


# ── 5) phase 登记 + 单一真相源 + collector 生产/消费回路 ─────────────────────


class TestPhaseRegistryAndSingleSource:
    def test_new_phases_are_registered_diagnostic_not_step_metrics(self) -> None:
        assert PP_NUMERIC_PROBE_PHASE in DIAGNOSTIC_PHASES
        assert FAIL_CLOSED_PHASE in DIAGNOSTIC_PHASES
        # 两者都**不是**"成功步逐步指标"：失败步/首步探针不得混进逐步序列
        assert PP_NUMERIC_PROBE_PHASE not in STEP_METRICS_PHASES
        assert FAIL_CLOSED_PHASE not in STEP_METRICS_PHASES

    @pytest.mark.parametrize("module_path", _TRAINING_MODULES, ids=lambda p: p.name)
    def test_training_modules_do_not_duplicate_phase_literals(self, module_path: Path) -> None:
        """§1.4：phase 名只能在 ``grad_probe`` 定义一次，训练模块必须 import 常量。"""
        source = module_path.read_text(encoding="utf-8")
        assert f'"{PP_NUMERIC_PROBE_PHASE}"' not in source
        assert f'"{FAIL_CLOSED_PHASE}"' not in source
        assert "PP_NUMERIC_PROBE_PHASE" in source
        assert "FAIL_CLOSED_PHASE" in source


# ── 6) P16：批序号口径统一（1-based）+ 首步零新增集合通信（裁定 5(B)）──────────


class TestStepIndexConvention:
    """★ P16（2026-09-23 裁定）：rank_metrics 行的 `step` 统一为 **1-based**。

    口径与 `global_step` 一致（故用 1-based）。

    背景：最初 `pp_numeric_probe` 硬编码 `step=0`、`fail_closed` 在 SFT 侧 1-based / RL 侧 0-based
    ⇒ 下游按 `step` 对行会错位。只读核查确认**无消费方**读该键做对行/join/去重，故统一到 1-based。
    """

    @pytest.mark.parametrize(("call_index", "expected"), [(0, 1), (1, 2), (2, 3), (99, 100)])
    def test_conversion_is_one_based(self, call_index: int, expected: int) -> None:
        assert step_index_one_based(call_index) == expected

    @pytest.mark.parametrize("module_path", _TRAINING_MODULES, ids=lambda p: p.name)
    def test_training_modules_route_step_through_the_helper(self, module_path: Path) -> None:
        """落盘必须经 `step_index_one_based(...)`，且不得留 0-based 字面量（`"step": 0`）。"""
        source = module_path.read_text(encoding="utf-8")
        assert "step_index_one_based(" in source
        assert '"step": 0,' not in source, "不得再出现硬编码 0-based 的 step 字面量"
        assert '"step": self._train_batch_call_index,' not in source, (
            "不得直接落 0-based 的 _train_batch_call_index（必须经 step_index_one_based）"
        )


class TestFirstStepProbeAddsNoCollective:
    """★ 裁定 5(B)：健康首步**不得**因 R3 探针而新增集合通信（观察者效应）。

    被测对象不得被观测动作扰动：NaN 缺陷的候选机制之一正是 PP 边界的流/时序纪律，
    在首步插一次集合通信可能正好扰动要观测的对象。代价：健康首步探针行 `rank_grad_norms=[]`。
    """

    def test_first_batch_not_in_the_gather_condition(self) -> None:
        source = (_REPO / "src/graspo/flow/adapters/models/qwen35_36/training_sft.py").read_text(
            encoding="utf-8"
        )
        assert "is_first_train_batch" in source, "首步探针本身仍必须在（只增行）"
        # 审计 gather 的条件里**不得**再出现首步（否则健康首步会多出 2 次 all_gather_object）
        gather_cond_start = source.index("rank_grad_norms: list[float] = []")
        gather_cond_end = source.index("local_grad_norm = self._trainable_grad_norm()")
        condition = source[gather_cond_start:gather_cond_end]
        assert "is_first_train_batch" not in condition, (
            "健康首步不得并入审计 gather 条件（裁定 5(B)）"
        )
        assert "not step_ok" in condition, "本步要拦时仍必须 gather（失败路径的诊断表）"

    def test_fail_closed_path_still_gathers(self) -> None:
        """失败路径的逐 rank 诊断表必须保留（与首步解耦）。"""
        source = (_REPO / "src/graspo/flow/adapters/models/qwen35_36/training_sft.py").read_text(
            encoding="utf-8"
        )
        assert "rank_grad_reports = [item for item in gathered_reports" in source


# ── 7) 机制探针 #4：按层/按种类的直方图 + 首/末局部层命中（纯本地，无集合通信）──


class TestMechanismHistogram:
    """★ 方案 #4（2026-09-23 裁定）：在"PP 边界先坏"与"某层先坏"之间裁决所需的字段。

    判读约定（与 `grad_probe` 里的字段注释逐字对应）：
    · PP 下每个 rank 的**局部层号从 0 起** ⇒ "0 层"就是离上游 P2P 边界最近的那层；
    · `nonfinite_at_first_local_layer` 先亮 ⇒ 支持"边界先坏"；
    · 非有限集中在**特定 kind**（如 `token_mixer.*`）且不随离边界远近分布 ⇒ 支持"某层自身数值问题"；
    · `nonfinite_nonlayer_names`（lm_head/norm/embed）先坏 ⇒ 指向 loss 侧通道。
    """

    def _bag(self, specs: dict[str, str]) -> _ParamBag:
        """specs: 参数名 → 'ok' | 'nan'。"""

        def grad(kind: str):
            if kind == "nan":
                value = torch.ones(4)
                value[0] = float("nan")
                return value
            return torch.ones(4)

        return _ParamBag({name: (torch.zeros(4), grad(kind)) for name, kind in specs.items()})

    def test_histogram_locates_first_and_last_local_layer(self) -> None:
        bag = self._bag(
            {
                "layers.0.token_mixer.conv1d_weight": "nan",
                "layers.1.mlp.up_proj.weight": "ok",
                "layers.2.token_mixer.conv1d_weight": "nan",
                "layers.3.mlp.down_proj.weight": "ok",
            }
        )
        report = grad_finiteness_report(bag.named_parameters())
        assert report["nonfinite_by_layer"] == {"0": 1, "2": 1}
        assert report["nonfinite_by_kind"] == {"token_mixer.conv1d_weight": 2}
        assert report["nonfinite_local_layer_span"] == [0, 2]
        assert report["local_layer_span"] == [0, 3]
        assert report["nonfinite_at_first_local_layer"] is True
        assert report["nonfinite_at_last_local_layer"] is False

    def test_last_layer_only_sets_the_last_flag(self) -> None:
        bag = self._bag(
            {"layers.0.mlp.up_proj.weight": "ok", "layers.5.mlp.down_proj.weight": "nan"}
        )
        report = grad_finiteness_report(bag.named_parameters())
        assert report["nonfinite_by_layer"] == {"5": 1}
        assert report["nonfinite_at_first_local_layer"] is False
        assert report["nonfinite_at_last_local_layer"] is True

    def test_nonlayer_tensors_are_reported_separately(self) -> None:
        """非层参数（末 stage 的 lm_head/norm ⇒ loss 侧通道）必须单独列出，不能混进层直方图。"""
        bag = self._bag({"layers.0.mlp.up_proj.weight": "ok", "lm_head.weight": "nan"})
        report = grad_finiteness_report(bag.named_parameters())
        assert report["nonfinite_by_layer"] == {}
        assert report["nonfinite_nonlayer_names"] == ["lm_head.weight"]
        assert report["nonfinite_at_first_local_layer"] is False
        assert report["nonfinite_at_last_local_layer"] is False
        assert report["local_layer_span"] == [0, 0]

    def test_healthy_step_has_empty_histogram_but_keeps_the_layer_span(self) -> None:
        bag = self._bag({"layers.0.a.weight": "ok", "layers.7.b.weight": "ok"})
        report = grad_finiteness_report(bag.named_parameters())
        assert report["nonfinite_by_layer"] == {}
        assert report["nonfinite_by_kind"] == {}
        assert report["nonfinite_local_layer_span"] is None
        assert report["local_layer_span"] == [0, 7]
        assert report["nonfinite_at_first_local_layer"] is False

    def test_name_cap_is_enlarged_for_the_mechanism_probe(self) -> None:
        """★ 机制探针需要一个比 5 大得多的名单上限（pp=4 实测 75/134 个张量非有限）。"""
        from graspo.flow.adapters.models.common import grad_probe

        assert grad_probe.MAX_NONFINITE_GRAD_NAMES >= 64

        bag = self._bag({f"layers.{i}.w.weight": "nan" for i in range(20)})
        report = grad_finiteness_report(bag.named_parameters())
        assert len(report["first_nonfinite_grad_names"]) == 20  # 不再被 5 截断
        assert report["nonfinite_by_layer"] == {str(i): 1 for i in range(20)}


# ── 8) 机制探针 #1/#2：PP 交换张量读数 + 方向信号（纯本地，零集合通信）──────────────


class TestExchangeTensorReport:
    """★ 方案 #1（2026-09-23 裁定）：裁定"边界携带 vs 本地生成"所需的最小读数。"""

    def test_healthy_tensor_fields(self) -> None:
        from graspo.flow.adapters.models.common.grad_probe import (
            PP_EXCHANGE_BWD_RECEIVED,
            exchange_tensor_report,
        )

        tensor = torch.ones(2, 3)
        report = exchange_tensor_report("grad_output", tensor, direction=PP_EXCHANGE_BWD_RECEIVED)
        assert report["direction"] == PP_EXCHANGE_BWD_RECEIVED
        assert report["name"] == "grad_output"
        assert report["shape"] == [2, 3]
        assert report["numel"] == 6
        assert report["isfinite"] is True
        assert report["max_abs"] == 1.0
        assert report["first_nonfinite_index"] is None
        assert "float32" in report["dtype"]

    def test_nan_tensor_reports_nan_max_and_first_index(self) -> None:
        from graspo.flow.adapters.models.common.grad_probe import exchange_tensor_report

        tensor = torch.ones(2, 3)
        tensor.view(-1)[4] = float("nan")
        report = exchange_tensor_report("stage_input", tensor, direction="bwd_sent_grad")
        assert report["isfinite"] is False
        assert report["max_abs"] != report["max_abs"]  # NaN 原样透出
        assert report["first_nonfinite_index"] == 4

    def test_inf_is_not_finite_but_max_abs_is_inf(self) -> None:
        from graspo.flow.adapters.models.common.grad_probe import exchange_tensor_report

        tensor = torch.zeros(4)
        tensor[1] = float("inf")
        report = exchange_tensor_report("x", tensor, direction="fwd_sent_hidden")
        assert report["isfinite"] is False
        assert report["max_abs"] == float("inf")
        assert report["first_nonfinite_index"] == 1

    def test_probe_does_not_modify_the_tensor(self) -> None:
        from graspo.flow.adapters.models.common.grad_probe import exchange_tensor_report

        tensor = torch.ones(2, 3)
        tensor.view(-1)[0] = float("nan")
        before = tensor.clone()
        exchange_tensor_report("x", tensor, direction="fwd_received_hidden")
        assert torch.allclose(before, tensor, equal_nan=True)

    def test_none_tensors_are_skipped_and_cap_applies(self) -> None:
        from graspo.flow.adapters.models.common.grad_probe import pp_exchange_readings

        items = [("d", "none", None), ("d", "a", torch.ones(1)), ("d", "b", torch.ones(1))]
        readings = pp_exchange_readings(items)
        assert [r["name"] for r in readings] == ["a", "b"]
        capped = pp_exchange_readings(
            [("d", f"t{i}", torch.ones(1)) for i in range(10)], max_items=3
        )
        assert [r["name"] for r in capped] == ["t0", "t1", "t2"]

    def test_readings_are_json_safe(self) -> None:
        """读数要能直接进 rank_metrics（jsonl）——不得含 tensor 对象。"""
        import json

        from graspo.flow.adapters.models.common.grad_probe import pp_exchange_readings

        readings = pp_exchange_readings([("d", "x", torch.ones(2, 2))])
        assert json.loads(json.dumps(readings))[0]["shape"] == [2, 2]

    def test_probe_helpers_never_call_distributed_collectives(self) -> None:
        """★ 硬约束（裁定要求）：**绝不新增集合通信** —— 两个函数源码里不得出现 `dist.`。"""
        import inspect

        from graspo.flow.adapters.models.common import grad_probe

        for func in (grad_probe.exchange_tensor_report, grad_probe.pp_exchange_readings):
            source = inspect.getsource(func)
            assert "dist." not in source, func.__name__
            for forbidden in ("all_reduce", "all_gather", "broadcast", "barrier"):
                assert forbidden not in source, (func.__name__, forbidden)


# ── 9) 方案 #1c/#1b：交换张量"内容摘要"（发/收逐 bit 对照）+ 发送后复读 ─────────────


class TestTensorDigest:
    """★ #1c（2026-09-23 裁定）：发端与收端**各自本地**算同一套摘要 ⇒ 分析者离线对照，
    零新增通信即可判定"通信是否保真"。"""

    def test_identical_content_yields_identical_digest(self) -> None:
        from graspo.flow.adapters.models.common.grad_probe import tensor_digest

        a = torch.arange(64, dtype=torch.float32).reshape(8, 8)
        b = a.clone()
        assert tensor_digest(a, with_sha256=True) == tensor_digest(b, with_sha256=True)

    def test_single_element_change_is_detected_bitwise(self) -> None:
        from graspo.flow.adapters.models.common.grad_probe import tensor_digest

        a = torch.arange(64, dtype=torch.float32)
        b = a.clone()
        b[7] = b[7] + 1e-3
        da, db = tensor_digest(a, with_sha256=True), tensor_digest(b, with_sha256=True)
        assert da["digest_sha256"] != db["digest_sha256"], "逐 bit 级差异必须被 sha256 抓到"
        assert da["digest_sum"] != db["digest_sum"]
        assert da["digest_head"] == db["digest_head"]  # 头一致、只有第 8 个元素变

    def test_tail_and_head_are_sampled(self) -> None:
        from graspo.flow.adapters.models.common.grad_probe import tensor_digest

        digest = tensor_digest(torch.arange(10, dtype=torch.float32), sample=3)
        assert digest["digest_head"] == [0.0, 1.0, 2.0]
        assert digest["digest_tail"] == [7.0, 8.0, 9.0]

    def test_sha256_is_opt_in(self) -> None:
        from graspo.flow.adapters.models.common.grad_probe import tensor_digest

        assert "digest_sha256" not in tensor_digest(torch.ones(4))
        assert tensor_digest(torch.ones(4), with_sha256=True)["digest_sha256"] is not None

    def test_nan_dominates_the_checksum(self) -> None:
        from graspo.flow.adapters.models.common.grad_probe import tensor_digest

        tensor = torch.ones(8)
        tensor[3] = float("nan")
        assert tensor_digest(tensor)["digest_sum"] != tensor_digest(tensor)["digest_sum"]

    def test_digest_is_read_only(self) -> None:
        from graspo.flow.adapters.models.common.grad_probe import tensor_digest

        tensor = torch.arange(8, dtype=torch.float32)
        before = tensor.clone()
        tensor_digest(tensor, with_sha256=True)
        assert torch.equal(before, tensor)

    def test_reading_carries_the_digest_fields(self) -> None:
        from graspo.flow.adapters.models.common.grad_probe import pp_exchange_readings

        reads = pp_exchange_readings([("d", "x", torch.ones(4))], with_sha256=True)
        for key in ("digest_sum", "digest_head", "digest_tail", "digest_sha256"):
            assert key in reads[0], key

    def test_digest_helper_never_calls_distributed_collectives(self) -> None:
        """裁定约束：**零集合通信**（除保真测试自身）——摘要在两端各自本地算，不需要任何通信。"""
        import inspect

        from graspo.flow.adapters.models.common import grad_probe

        source = inspect.getsource(grad_probe.tensor_digest)
        assert "dist." not in source
        for forbidden in ("all_reduce", "all_gather", "broadcast", "barrier", "isend", "irecv"):
            assert forbidden not in source, forbidden


# ── 10) #1a：PipelineComm 真 send/recv 处取样 + 诊断专用一次性同步（默认关）──────────────


class TestPipelineCommDiagHook:
    """★ #1a：取数点在**真 buffer** 处，且**默认关**（hook=None ⇒ 零行为变化）。"""

    def _comm(self):
        from graspo.flow.parallel.pipeline_comm import PipelineComm

        return PipelineComm.__new__(PipelineComm)  # 不建进程组：只测诊断钩子逻辑

    def test_default_off_does_nothing(self) -> None:
        comm = self._comm()
        comm.diag_hook = None
        comm.diag_sync_once = True
        comm._diag_synced = False
        comm.diag_sample("fwd_send", torch.ones(2))  # 不得抛、不得做任何事
        assert comm._diag_synced is False

    def test_hook_receives_the_real_tensor(self) -> None:
        comm = self._comm()
        seen: list[tuple[str, object]] = []
        comm.diag_hook = lambda direction, tensor: seen.append((direction, tensor))
        comm.diag_sync_once = False
        tensor = torch.arange(4, dtype=torch.float32)
        comm.diag_sample("bwd_send", tensor)
        assert len(seen) == 1
        assert seen[0][0] == "bwd_send"
        assert seen[0][1] is tensor, "必须是同一个真张量对象（不是拷贝）"

    def test_none_tensor_is_skipped(self) -> None:
        comm = self._comm()
        calls: list[str] = []
        comm.diag_hook = lambda direction, tensor: calls.append(direction)
        comm.diag_sync_once = False
        comm.diag_sample("fwd_recv", None)
        assert calls == []

    def test_sync_once_per_step_and_rearmed_by_diag_begin_step(self, monkeypatch) -> None:
        """★ 诊断专用**一次性**同步：本步首个取样同步一次；`diag_begin_step()` 后重新武装。"""
        import graspo.flow.parallel.pipeline_comm as pc

        comm = self._comm()
        comm.diag_hook = lambda direction, tensor: None
        comm.diag_sync_once = True
        comm.diag_begin_step()
        syncs: list[int] = []
        monkeypatch.setattr(pc.torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(pc.torch.cuda, "synchronize", lambda: syncs.append(1))
        comm.diag_sample("fwd_send", torch.ones(2))
        comm.diag_sample("fwd_recv", torch.ones(2))
        comm.diag_sample("bwd_send", torch.ones(2))
        assert len(syncs) == 1, "同一诊断步内至多同步一次"
        comm.diag_begin_step()
        comm.diag_sample("fwd_send", torch.ones(2))
        assert len(syncs) == 2, "新的一步要重新武装"

    def test_recv_handle_carries_the_buffer_and_samples_after_wait(self) -> None:
        """recv 侧：句柄带上真 buffer，`wait()` 完成后才取样（数据就绪）。"""
        from graspo.flow.parallel.pipeline_comm import _RecvHandle

        class _FakeWork:
            def wait(self, timeout=None):  # noqa: ANN001, ANN204
                return True

        class _FakeComm:
            def __init__(self) -> None:
                self.sampled: list[tuple[str, object]] = []

            def diag_sample(self, direction, tensor):  # noqa: ANN001, ANN202
                self.sampled.append((direction, tensor))

            def _forget(self, work):  # noqa: ANN001, ANN202
                return None

        comm = _FakeComm()
        tensor = torch.ones(3)
        handle = _RecvHandle(
            _FakeWork(), None, None, comm, info={"direction": "bwd_recv"}, tensor=tensor
        )
        assert handle._tensor is tensor
        handle.wait()
        assert comm.sampled == [("bwd_recv", tensor)]


class TestSamplingSitesAreTheRealBuffers:
    """★ 源码钉子：取样点必须在 `contiguous()` 之后 / recv 完成之后（不是训练层的事后引用）。"""

    _COMM = _REPO / "src" / "graspo" / "flow" / "parallel" / "pipeline_comm.py"

    def _source(self) -> str:
        return self._COMM.read_text(encoding="utf-8")

    def test_send_samples_the_contiguous_buffer_before_isend(self) -> None:
        text = self._source()
        contiguous = text.index("tensor = tensor.contiguous()")
        sample = text.index('self.diag_sample(f"{direction}_send", tensor)')
        isend = text.index("work = dist.isend(tensor")
        assert contiguous < sample < isend, "必须在 contiguous 之后、isend 之前取样"

    def test_recv_samples_after_wait(self) -> None:
        text = self._source()
        assert text.count('self._comm.diag_sample(str(self._info.get("direction") or "recv")') == 2
        # 两处都必须在 wait 之后
        first = text.index("torch.cuda.current_stream().wait_event(self._ev)")
        second = text.index("self._work.wait()", first)
        assert first < text.index("diag_sample", first)
        assert second < text.index("diag_sample", second)

    def test_handle_carries_tensor_field(self) -> None:
        text = self._source()
        assert '"_tensor"' in text.split("__slots__")[1][:200]


class TestDirtyBlockSide:
    """★ 方案 #2：脏块位置 / 层号逆序（口径：**层号**，不是观测时序；见 grad_probe 注释）。"""

    def _bag(self, specs: dict[str, str]) -> _ParamBag:
        def grad(kind: str):
            value = torch.ones(4)
            if kind == "nan":
                value[0] = float("nan")
            return value

        return _ParamBag({name: (torch.zeros(4), grad(kind)) for name, kind in specs.items()})

    def test_adjacent_to_first_local_layer(self) -> None:
        bag = self._bag({"layers.0.a.weight": "nan", "layers.1.a.weight": "ok"})
        report = grad_finiteness_report(bag.named_parameters())
        assert report["dirty_block_side"] == "adjacent_to_first_local_layer"
        assert report["nonfinite_layers_descending"] == [0]

    def test_adjacent_to_last_local_layer(self) -> None:
        bag = self._bag(
            {"layers.0.a.weight": "ok", "layers.1.a.weight": "ok", "layers.2.a.weight": "nan"}
        )
        report = grad_finiteness_report(bag.named_parameters())
        assert report["dirty_block_side"] == "adjacent_to_last_local_layer"
        assert report["nonfinite_layers_descending"] == [2]

    def test_spans_both_ends_and_middle(self) -> None:
        both = self._bag(
            {
                "layers.0.a.weight": "nan",
                "layers.1.a.weight": "ok",
                "layers.2.a.weight": "nan",
            }
        )
        assert (
            grad_finiteness_report(both.named_parameters())["dirty_block_side"] == "spans_both_ends"
        )
        middle = self._bag(
            {
                "layers.0.a.weight": "ok",
                "layers.1.a.weight": "nan",
                "layers.2.a.weight": "ok",
            }
        )
        assert grad_finiteness_report(middle.named_parameters())["dirty_block_side"] == "middle"

    def test_all_layers_dirty_and_none(self) -> None:
        all_dirty = self._bag({"layers.0.a.weight": "nan", "layers.1.a.weight": "nan"})
        assert (
            grad_finiteness_report(all_dirty.named_parameters())["dirty_block_side"]
            == "all_layers_dirty"
        )
        clean = self._bag({"layers.0.a.weight": "ok"})
        report = grad_finiteness_report(clean.named_parameters())
        assert report["dirty_block_side"] is None
        assert report["nonfinite_layers_descending"] == []


class TestExchangeProbeWiringIsPpOnlyAndDiagnosticOnly:
    """★ 接线钉子（裁定要求："只接在 PP 路径 + 诊断步、零行为变化"）。用**源码断言**实现
    （不 import 训练 mixin，避免拉起 transformers）。"""

    _TRAINING_SFT = (
        _REPO / "src" / "graspo" / "flow" / "adapters" / "models" / "qwen35_36" / "training_sft.py"
    )

    def _source(self) -> str:
        return self._TRAINING_SFT.read_text(encoding="utf-8")

    def test_recording_happens_only_inside_the_pp_method(self) -> None:
        text = self._source()
        start = text.index("def _pipeline_train_batch_sft(")
        end = text.index("def _pipeline_forward_for_sft(")
        pp_body = text[start:end]
        # #1a 后的接线：PP 方法内装 comm 钩子 + 每步重新武装一次性同步；训练层不再自己取样
        assert pp_body.count("comm.diag_hook = _on_comm_sample") == 1
        assert pp_body.count("comm.diag_sync_once = True") == 1
        assert pp_body.count("comm.diag_begin_step()") == 1
        assert "def _on_comm_sample(" in pp_body
        assert pp_body.count("record_exchange(") == 0
        non_pp_start = text.index("def train_batch_sft(")
        non_pp_body = text[non_pp_start:start]
        assert "record_exchange(" not in non_pp_body, "非 PP 路径不得接线"

    def test_readings_land_only_in_the_two_diagnostic_rows(self) -> None:
        text = self._source()
        assert text.count('"pp_exchange"') == 2, text.count('"pp_exchange"')
        probe_block = text[text.index("PP_NUMERIC_PROBE_PHASE: {") :]
        assert '"pp_exchange"' in probe_block[:800]
        fail_block = text[text.index("fail_closed_metrics = {") :]
        assert '"pp_exchange"' in fail_block[:1400]

    def test_post_wait_recheck_is_wired_only_for_sent_tensors(self) -> None:
        """★ #1b：发送侧 3 处（fwd_sent 1 + bwd_sent 2）要求"发送完成后复读"，接收侧不复读。"""
        text = self._source()
        # 只有两个"发送"方向会登记复读
        assert "if direction in (PP_EXCHANGE_FWD_SENT, PP_EXCHANGE_BWD_SENT):" in text, (
            "复读登记必须只覆盖发送方向"
        )
        assert text.count("pending_recheck.append(") == 1
        # 复读必须发生在 wait_all(send_works) **之后**
        wait_index = text.index("wait_all(send_works)")
        loop_index = text.index("for direction, name, tensor in pending_recheck:")
        assert wait_index < loop_index

    def test_post_wait_readings_land_in_the_two_diagnostic_rows(self) -> None:
        text = self._source()
        assert text.count('"pp_exchange_post_wait"') == 2, text.count('"pp_exchange_post_wait"')

    def test_bitwise_digest_is_gated_to_the_first_batch(self) -> None:
        """逐 bit sha256 开销大 ⇒ 只在首个 batch 开（其余步只有廉价摘要）。"""
        text = self._source()
        assert "probe_sha256 = int(self._train_batch_call_index) == 0" in text
        assert "with_sha256=probe_sha256" in text

    def test_retention_caveat_is_documented(self) -> None:
        """复读要保留引用 ⇒ 对"释放后被复用"这一支**保守**；此口径必须写在源码里（§2.2）。"""
        assert "保守" in self._source()

    def test_no_collectives_or_tensor_mutation_in_the_recording_helper(self) -> None:
        text = self._source()
        start = text.index("def _on_comm_sample(")
        end = text.index("# 梯度累积：zero_grad 只调一次")  # helper 之后紧接着的就是训练逻辑
        helper = text[start:end]
        for forbidden in ("all_reduce", "all_gather", ".backward(", ".copy_(", "zero_grad"):
            assert forbidden not in helper, forbidden
        assert "pp_exchange_readings(" in helper


def _load_collector():
    spec = importlib.util.spec_from_file_location("_r1_collector", _COLLECTOR)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["_r1_collector"] = module
    spec.loader.exec_module(module)
    return module


class TestSkippedNonfiniteProducerAndConsumer:
    def test_collector_reads_the_producer_key(self, tmp_path: Path) -> None:
        """生产→消费回路：``global_skipped_nonfinite_sum`` 必须被采集层读到。

        此前该键在 ``scripts/collect_results.py`` 已被消费、``src/`` 却**没有生产者**
        （既有债）⇒ 这里锁死"生产者写出的键形状 = 采集层读的键形状"。
        """
        collector = _load_collector()
        (tmp_path / "rank_metrics.rank_00000.jsonl").write_text(
            json.dumps(
                {
                    "event": "rank_memory",
                    "phase": "pipeline_sft_train_batch_after",
                    "rank": 0,
                    "metrics": {
                        "global_skipped_nonfinite_sum": 3,
                        "skipped_nonfinite": 0,
                    },
                }
            )
            + "\n",
            encoding="utf-8",
        )
        maximum, per_rank, detail = collector.extract_skipped_nonfinite_all_ranks(tmp_path)
        assert maximum == 3, detail
        assert per_rank == {"rank_metrics.rank_00000.jsonl": 3}, detail

    def test_single_process_branch_emits_the_key(self) -> None:
        """``_aggregate_rank_metrics`` 的单进程分支也必须产出该键（只增不改地补齐）。"""
        from graspo.flow.adapters.transformer_adapter import TransformerAdapter

        stub = _AdapterStub(world_size=1)
        metrics = TransformerAdapter._aggregate_rank_metrics(stub, {"skipped_nonfinite": 2})
        assert metrics["global_skipped_nonfinite_sum"] == 2
        # 既有键不得被改动（§18 只增不改）
        assert metrics["global_optimizer_steps_sum"] == 0

    def test_dist_branch_sums_over_all_ranks(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """dist 分支口径 = **逐 rank 求和**（任一 rank 跳过都计入）。"""
        import graspo.flow.adapters.transformer_adapter as adapter_module

        per_rank = [
            {"rank": 0, "skipped_nonfinite": 0, "optimizer_steps": 1},
            {"rank": 1, "skipped_nonfinite": 2, "optimizer_steps": 0},
            {"rank": 2, "skipped_nonfinite": 1, "optimizer_steps": 0},
            {"rank": 3, "skipped_nonfinite": 0, "optimizer_steps": 1},
        ]

        class _FakeDist:
            @staticmethod
            def is_available() -> bool:
                return True

            @staticmethod
            def is_initialized() -> bool:
                return True

            @staticmethod
            def all_gather_object(out: list, local: dict) -> None:
                out[:] = list(per_rank)

        monkeypatch.setattr(adapter_module, "dist", _FakeDist)
        from graspo.flow.adapters.transformer_adapter import TransformerAdapter

        stub = _AdapterStub(world_size=4)
        metrics = TransformerAdapter._aggregate_rank_metrics(stub, {"skipped_nonfinite": 0})
        assert metrics["global_skipped_nonfinite_sum"] == 3
        assert metrics["global_optimizer_steps_sum"] == 2


class _AdapterStub:
    """``_aggregate_rank_metrics`` 需要的最小鸭子类型（不构造真适配器：不碰 GPU/权重）。"""

    def __init__(self, *, world_size: int) -> None:
        self.rank = 0
        self.tp_rank = 0
        self.pp_rank = 0
        self.pp_size = 1
        self.tp_size = 1
        self.dp_size = 1
        self.world_size = world_size

    @staticmethod
    def _loss_bearing_ranks(ranks: list[dict]) -> tuple[list[dict], str]:
        from graspo.flow.adapters.transformer_adapter import TransformerAdapter

        return TransformerAdapter._loss_bearing_ranks(ranks)
