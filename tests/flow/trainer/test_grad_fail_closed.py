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
