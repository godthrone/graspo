"""rollout 的 ``inference_mode`` → ``no_grad`` 上下文修正：接线事实 + 能真失败的负向用例。

**为什么需要它**（D46 探针 2026-10-29，真实 OPD 全链路逐字读数）：训练 forward 里视觉塔
``patch_embed`` 的 conv3d 报 ``Inference tensors cannot be saved for backward``，现场是
``weight.is_inference=True`` / ``input.is_inference=False`` —— 被污染的是**参数对象**
（rollout 的 ``inference_mode`` 罩住了 FSDP2 的 all-gather 物化，物化出的参数张量带
inference 标记并被缓存）。本补丁只把那段执行体的上下文换成 ``no_grad``。

**本文件回答五个问题**

1. **真的换了上下文**：全参 + 多模态 ⇒ 引擎方法内 ``torch.is_inference_mode_enabled()``
   为 ``False``、新建张量 ``is_inference()`` 为 ``False``（这就是崩溃的充分条件）。
2. **作用面最小**：LoRA 档**不安装**（类属性对象身份不变）；文本模型安装但**逐字直通**
   （上游 ``inference_mode`` 语义与内存收益不变）；``use_vllm=true`` 不安装。
3. **fail-closed**：上游符号缺失或已改成 ``no_grad`` ⇒ 不安装并记 ``degraded_reason``；
   **安装后被外层 inference_mode 罩回去 ⇒ 抛错点名**（⛔ 不许"看起来修了"）。
4. **只换上下文、不改数值**：同 seed 下，同一段计算在 ``inference_mode`` 与 ``no_grad``
   里产出的张量**逐位相同**（真实链路的逐位对照在 A800 上另做，见工位报告）。
5. **GRASPO 分支真的接上了**（⑤）：``trainer.run_graspo_rlhf`` 走同一个入口；
   不经它的等价路径会抛 ``Inference tensors cannot be saved for backward``（红），
   经它的同一个流水线体通过（绿）；LoRA / ``use_vllm=true`` 不进入该作用域；
   作用域结束（含异常路径）``_infer`` 恢复原样。

**为什么不用真 ms-swift**：被测的是 graspo 自己的接线。测试用一个**同形替身类**注册进
``sys.modules`` 顶替 ``swift.infer_engine.transformers_engine``（与
``test_rollout_seed.py`` 同一手法）。
"""

from __future__ import annotations

import sys
import types
from collections.abc import Callable
from types import SimpleNamespace
from typing import Any

import pytest

from graspo.flow.backend.msswift import _rollout_no_grad as rg

try:  # 无 torch 的裸环境整文件跳过
    import torch
except ImportError:  # pragma: no cover - 环境相关
    torch = None

pytestmark = pytest.mark.skipif(torch is None, reason="this test needs a real torch")


# ---------------------------------------------------------------------------
# 替身引擎（与上游同形：``_infer`` 带 ``@torch.inference_mode()``）
# ---------------------------------------------------------------------------


def _make_engine_stub(*, multimodal: bool = True, decorator: str = "inference_mode") -> type:
    """造一个同形替身类。

    Args:
        multimodal: 引擎实例 ``model_meta.is_multimodal`` 的取值（上游真实判据）。
        decorator: ``_infer`` 上的装饰器（``inference_mode`` / ``no_grad``）。

    返回的 ``_infer`` 会把**执行期事实**记进类属性 ``_graspo_test_calls``：
    ``{"inference_mode": bool, "grad_enabled": bool, "tensor_is_inference": bool}``。
    """

    def _body(self):
        probe = torch.empty(4, device="cpu")
        type(self)._graspo_test_calls.append(
            {
                "inference_mode": torch.is_inference_mode_enabled(),
                "grad_enabled": torch.is_grad_enabled(),
                "tensor_is_inference": probe.is_inference(),
            }
        )
        return probe

    if decorator == "inference_mode":
        decorated = torch.inference_mode()(_body)
    elif decorator == "no_grad":
        decorated = torch.no_grad()(_body)
    else:
        decorated = _body

    class TransformersEngine:  # noqa: N801 - 名称必须与上游一致
        _graspo_test_calls: list[dict] = []

        def __init__(self) -> None:
            self.model_meta = SimpleNamespace(is_multimodal=multimodal)

        _infer = decorated

    TransformersEngine._graspo_test_calls = []
    return TransformersEngine


def _install_engine_stub(monkeypatch: pytest.MonkeyPatch, cls: type | None) -> None:
    """把替身注册成 ``swift.infer_engine.transformers_engine``（``cls=None`` 造缺失）。"""
    swift_pkg = types.ModuleType("swift")
    swift_pkg.__path__ = []  # type: ignore[attr-defined]
    infer_pkg = types.ModuleType("swift.infer_engine")
    infer_pkg.__path__ = []  # type: ignore[attr-defined]
    engine_mod = types.ModuleType(rg._ENGINE_MODULE)
    if cls is not None:
        engine_mod.TransformersEngine = cls  # type: ignore[attr-defined]
    for name, mod in (
        ("swift", swift_pkg),
        ("swift.infer_engine", infer_pkg),
        (rg._ENGINE_MODULE, engine_mod),
    ):
        monkeypatch.setitem(sys.modules, name, mod)


def _config(*, tuner_type: str = "full", use_vllm: bool = False) -> SimpleNamespace:
    return SimpleNamespace(
        effective_tuner_type=tuner_type,
        msswift=SimpleNamespace(use_vllm=use_vllm),
    )


# ---------------------------------------------------------------------------
# ① 真的换了上下文（这是崩溃的充分条件被消除的直接证据）
# ---------------------------------------------------------------------------


def test_full_multimodal_rollout_runs_under_no_grad(monkeypatch: pytest.MonkeyPatch) -> None:
    cls = _make_engine_stub(multimodal=True)
    _install_engine_stub(monkeypatch, cls)

    with rg.rollout_no_grad(_config()) as ledger:
        assert ledger["installed"] is True
        probe = cls()._infer()

    seen = cls._graspo_test_calls[-1]
    assert seen == {"inference_mode": False, "grad_enabled": False, "tensor_is_inference": False}
    # 参数/激活在 no_grad 下是**普通张量** ⇒ conv3d 反向不再需要保存 inference 张量。
    assert probe.is_inference() is False
    assert ledger["rewritten_calls"] == 1
    assert ledger["passthrough_calls"] == 0
    assert ledger["installed_at"].endswith("._infer")


def test_unpatched_upstream_reproduces_the_defect(monkeypatch: pytest.MonkeyPatch) -> None:
    """负向对照：不装补丁时同一替身产出的张量**就是** inference 张量（红）。"""
    cls = _make_engine_stub(multimodal=True)
    _install_engine_stub(monkeypatch, cls)

    probe = cls()._infer()

    assert cls._graspo_test_calls[-1]["tensor_is_inference"] is True
    assert probe.is_inference() is True
    assert cls._graspo_test_calls[-1]["inference_mode"] is True


# ---------------------------------------------------------------------------
# ② 作用面最小：LoRA 不装、文本直通、vLLM 不装
# ---------------------------------------------------------------------------


def test_lora_tier_does_not_install_the_patch(monkeypatch: pytest.MonkeyPatch) -> None:
    cls = _make_engine_stub(multimodal=True)
    _install_engine_stub(monkeypatch, cls)
    original = cls.__dict__["_infer"]

    with rg.rollout_no_grad(_config(tuner_type="lora")) as ledger:
        assert ledger["installed"] is False
        assert ledger["required"] is False
        assert "tuner_type" in ledger["degraded_reason"]
        # 类属性对象身份不变：LoRA 档的行为与修前逐字相同。
        assert cls.__dict__["_infer"] is original
        cls()._infer()

    assert cls._graspo_test_calls[-1]["inference_mode"] is True  # 上游语义未被改动


def test_text_only_model_passes_through_upstream_inference_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cls = _make_engine_stub(multimodal=False)
    _install_engine_stub(monkeypatch, cls)

    with rg.rollout_no_grad(_config()) as ledger:
        assert ledger["installed"] is True
        cls()._infer()

    assert ledger["rewritten_calls"] == 0
    assert ledger["passthrough_calls"] == 1
    # 直通 = 走上游原方法 ⇒ inference_mode 的内存收益与语义都保持原样。
    assert cls._graspo_test_calls[-1]["inference_mode"] is True


def test_use_vllm_rollout_does_not_install(monkeypatch: pytest.MonkeyPatch) -> None:
    cls = _make_engine_stub()
    _install_engine_stub(monkeypatch, cls)
    original = cls.__dict__["_infer"]

    with rg.rollout_no_grad(_config(use_vllm=True)) as ledger:
        assert ledger["installed"] is False
        assert "use_vllm" in ledger["degraded_reason"]
        assert cls.__dict__["_infer"] is original


# ---------------------------------------------------------------------------
# ③ fail-closed：上游变了 / 符号缺失 / 装了却没生效
# ---------------------------------------------------------------------------


def test_upstream_already_no_grad_degrades_harmlessly(monkeypatch: pytest.MonkeyPatch) -> None:
    cls = _make_engine_stub(decorator="no_grad")
    _install_engine_stub(monkeypatch, cls)
    original = cls.__dict__["_infer"]

    with rg.rollout_no_grad(_config()) as ledger:
        assert ledger["installed"] is False
        assert "no_grad" in ledger["degraded_reason"]
        assert cls.__dict__["_infer"] is original


def test_missing_engine_method_degrades(monkeypatch: pytest.MonkeyPatch) -> None:
    class TransformersEngine:  # noqa: N801 - 同形替身，故意缺 ``_infer``
        pass

    _install_engine_stub(monkeypatch, TransformersEngine)

    with rg.rollout_no_grad(_config()) as ledger:
        assert ledger["installed"] is False
        assert "_infer" in ledger["degraded_reason"]


def test_missing_ms_swift_degrades(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_engine_stub(monkeypatch, None)

    with rg.rollout_no_grad(_config()) as ledger:
        assert ledger["installed"] is False
        assert "ms-swift" in ledger["degraded_reason"]


def test_installed_but_still_inference_mode_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    """安装了却仍被外层 ``inference_mode`` 罩住 ⇒ 必须报错点名，⛔ 不静默。"""
    cls = _make_engine_stub(multimodal=True)
    _install_engine_stub(monkeypatch, cls)

    with rg.rollout_no_grad(_config()):
        with torch.inference_mode():
            with pytest.raises(RuntimeError, match="had no effect"):
                cls()._infer()

    assert "is_inference_mode_enabled" in rg.ROLLOUT_NO_GRAD_NOT_EFFECTIVE


def test_patch_is_idempotent_and_restored_on_exit(monkeypatch: pytest.MonkeyPatch) -> None:
    cls = _make_engine_stub()
    _install_engine_stub(monkeypatch, cls)
    original = cls.__dict__["_infer"]

    with rg.rollout_no_grad(_config()):
        patched = cls.__dict__["_infer"]
        assert patched is not original
        assert patched.__name__ == "_infer"
        with rg.rollout_no_grad(_config()) as inner_ledger:
            # 幂等：复用既有包装，不叠第二层。
            assert inner_ledger["installed"] is True
            assert cls.__dict__["_infer"] is patched

    assert cls.__dict__["_infer"] is original


def test_restores_original_even_when_body_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    class BoomError(RuntimeError):
        pass

    def _explode(self):
        raise BoomError("rollout failed")

    class TransformersEngine:  # noqa: N801 - 同形替身
        _infer = torch.inference_mode()(_explode)
        model_meta = SimpleNamespace(is_multimodal=True)

    _install_engine_stub(monkeypatch, TransformersEngine)
    original = TransformersEngine.__dict__["_infer"]

    with pytest.raises(BoomError):
        with rg.rollout_no_grad(_config()):
            TransformersEngine()._infer()

    assert TransformersEngine.__dict__["_infer"] is original


# ---------------------------------------------------------------------------
# ④ 只换上下文、不改数值：同 seed 下两种上下文产出逐位相同
# ---------------------------------------------------------------------------


def test_no_grad_output_is_bitwise_identical_to_inference_mode() -> None:
    """同一段真实计算（conv3d + 采样）在两种上下文里逐位相同。

    这条是本补丁"不动数值"的**充分**证据：``no_grad`` 与 ``inference_mode`` 都关
    autograd 记录，区别只在 inference 标记，而标记不影响任何算子结果。
    """

    def compute() -> tuple[torch.Tensor, torch.Tensor]:
        torch.manual_seed(1234)
        conv = torch.nn.Conv3d(3, 8, kernel_size=(2, 2, 2), stride=(2, 2, 2))
        weight_before = conv.weight.detach().clone()
        x = torch.randn(4, 3, 4, 4, 4)
        out = conv(x)
        sampled = torch.multinomial(torch.softmax(out.flatten(), dim=0), 8, replacement=True)
        return out, sampled, weight_before, conv.weight.detach().clone()

    with torch.inference_mode():
        out_a, tok_a, w0_a, w1_a = compute()
    with torch.no_grad():
        out_b, tok_b, w0_b, w1_b = compute()

    assert torch.equal(out_a, out_b)
    assert torch.equal(tok_a, tok_b)
    assert torch.equal(w0_a, w0_b) and torch.equal(w1_a, w1_b)
    # 两种上下文都不记录梯度，但只有 inference_mode 会让输出带 inference 标记。
    assert out_a.is_inference() is True
    assert out_b.is_inference() is False


# ---------------------------------------------------------------------------
# ⑤ GRASPO 分支的接线（``trainer.run_graspo_rlhf``）：红→绿、不误伤、退出还原
# ---------------------------------------------------------------------------


def _install_swift_with_pipeline(
    monkeypatch: pytest.MonkeyPatch, engine_cls: Any, main: Callable[[], object]
) -> None:
    """注册 ms-swift 替身：引擎模块（同 ``_install_engine_stub``）+ ``SwiftRLHF`` 入口。

    ``SwiftRLHF.main`` 直接调用传入的流水线体——被测的是**作用域接在哪一层**，
    不是 ms-swift 自己的参数解析（那部分由 ``test_msswift_wiring.py`` 覆盖）。
    """
    _install_engine_stub(monkeypatch, engine_cls)

    trainers_mod = types.ModuleType("swift.trainers")
    trainers_mod.TrainerFactory = SimpleNamespace(TRAINER_MAPPING={})  # type: ignore[attr-defined]
    pipelines_mod = types.ModuleType("swift.pipelines")
    pipelines_mod.__path__ = []  # type: ignore[attr-defined]
    train_mod = types.ModuleType("swift.pipelines.train")
    train_mod.__path__ = []  # type: ignore[attr-defined]
    rlhf_mod = types.ModuleType("swift.pipelines.train.rlhf")

    class SwiftRLHF:  # noqa: N801 - 名称必须与上游一致
        def __init__(self, args: object = None, *, graspo_config: object = None) -> None:
            self.args = SimpleNamespace(rlhf_type="grpo")
            self.graspo_config = graspo_config

        def main(self) -> object:
            return main()

    rlhf_mod.SwiftRLHF = SwiftRLHF  # type: ignore[attr-defined]
    for name, mod in (
        ("swift.trainers", trainers_mod),
        ("swift.pipelines", pipelines_mod),
        ("swift.pipelines.train", train_mod),
        ("swift.pipelines.train.rlhf", rlhf_mod),
    ):
        monkeypatch.setitem(sys.modules, name, mod)

    from graspo.flow.backend.msswift import trainer as trainer_module

    # 流水线子类按模块级缓存复用；换替身时必须清空，否则拿到上一个用的类。
    monkeypatch.setattr(trainer_module, "_PIPELINE_CLASS_CACHE", {})


def _graspo_config(*, tuner_type: str = "full", use_vllm: bool = False) -> object:
    """真 ``GraspoConfig``（作用面判据读的就是它的 ``effective_tuner_type``/``use_vllm``）。"""
    from graspo.core.schema import GraspoConfig

    return GraspoConfig.model_validate(
        {
            "backend": "msswift",
            "train_method": "graspo",
            "tuner_type": tuner_type,
            "msswift": {"use_vllm": use_vllm},
        }
    )


def _make_defect_engine(
    *, multimodal: bool = True, train_forward: bool = True
) -> tuple[Any, Callable[[], str]]:
    """同形替身引擎 + "先 rollout、再训练 forward"的流水线体（复刻 M028 崩溃机理）。

    ``_infer`` 在 ``inference_mode`` 里物化并缓存的那个张量，就是全参档 FSDP2 缓存的
    视觉塔参数对象的等价物：带 inference 标记。训练 forward 在 autograd 下复用它
    （``x @ weight``）⇒ 反向为算 ``x`` 的梯度需要保存 ``weight`` ⇒ 抛
    ``RuntimeError: Inference tensors cannot be saved for backward``（与现场同一条）。
    换成 ``no_grad`` 后同一段计算产出的 ``weight`` 是普通张量，反向正常。

    Returns:
        ``(TransformersEngine, main)``。``main`` 把执行期事实记在类属性
        ``_patch_active`` / ``_graspo_test_calls`` 上供断言。
    """

    def _infer_body(self):
        weight = torch.randn(4, 4)
        type(self)._cached_param = weight
        type(self)._graspo_test_calls.append(
            {
                "inference_mode": torch.is_inference_mode_enabled(),
                "tensor_is_inference": weight.is_inference(),
            }
        )
        return weight

    class TransformersEngine:  # noqa: N801 - 名称必须与上游一致
        _cached_param: Any = None
        _patch_active = False
        _graspo_test_calls: list[dict] = []

        def __init__(self) -> None:
            self.model_meta = SimpleNamespace(is_multimodal=multimodal)

        def infer(self, *args: object, **kwargs: object) -> object:
            """上游 ``infer`` 的替身：真实实现做完请求校验后调 ``_infer``，这里直接转交。

            保留这一层是因为 ``rollout_seeding`` 的接线点在 ``infer`` 上（与上游一致）。
            """
            return self._infer()

        _infer = torch.inference_mode()(_infer_body)

    original_infer = TransformersEngine.__dict__["_infer"]

    def main() -> str:
        # 类属性对象身份是否被换过 = 本补丁是否真的装在这个替身上（作用域内）。
        TransformersEngine._patch_active = (
            TransformersEngine.__dict__["_infer"] is not original_infer
        )
        engine = TransformersEngine()
        engine.infer()  # rollout（上游入口：``infer`` → ``_infer``）
        if not train_forward:
            return "rollout only"
        weight = TransformersEngine._cached_param
        x = torch.randn(4, 4, requires_grad=True)
        (x @ weight).sum().backward()  # 训练 forward + backward（崩溃点）
        return "forward+backward ok"

    return TransformersEngine, main


def test_graspo_branch_without_the_wiring_reproduces_the_crash(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """**红**：不经 ``run_graspo_rlhf`` 的作用域（接线前的等价路径）⇒ 同一崩溃。"""
    engine_cls, main = _make_defect_engine()
    _install_swift_with_pipeline(monkeypatch, engine_cls, main)

    from graspo.flow.backend.msswift import trainer as trainer_module

    with pytest.raises(RuntimeError, match="Inference tensors cannot be saved for backward"):
        trainer_module._graspo_rlhf_pipeline_class()([], graspo_config=_graspo_config()).main()

    assert engine_cls._patch_active is False
    assert engine_cls._graspo_test_calls[-1] == {
        "inference_mode": True,
        "tensor_is_inference": True,
    }


def test_graspo_branch_wiring_makes_the_same_pipeline_body_pass(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """**绿**：经 ``run_graspo_rlhf``（接线后）⇒ 同一个流水线体跑通。"""
    engine_cls, main = _make_defect_engine()
    _install_swift_with_pipeline(monkeypatch, engine_cls, main)

    from graspo.flow.backend.msswift.trainer import run_graspo_rlhf

    result = run_graspo_rlhf([], graspo_config=_graspo_config())

    assert result == "forward+backward ok"
    assert engine_cls._patch_active is True
    # rollout 里物化出的"参数"是普通张量 ⇒ 训练 forward 复用它能进反向。
    assert engine_cls._graspo_test_calls[-1] == {
        "inference_mode": False,
        "tensor_is_inference": False,
    }
    assert engine_cls._cached_param.is_inference() is False


@pytest.mark.parametrize(
    ("tuner_type", "use_vllm"),
    [("lora", False), ("full", True)],
    ids=["lora", "use_vllm"],
)
def test_graspo_wiring_leaves_matching_out_of_scope_tiers_untouched(
    monkeypatch: pytest.MonkeyPatch, tuner_type: str, use_vllm: bool
) -> None:
    """**不误伤**：LoRA 档与 ``use_vllm=true`` 都不进入该作用域（类属性身份不变）。"""
    engine_cls, main = _make_defect_engine(train_forward=False)
    _install_swift_with_pipeline(monkeypatch, engine_cls, main)

    from graspo.flow.backend.msswift.trainer import run_graspo_rlhf

    run_graspo_rlhf([], graspo_config=_graspo_config(tuner_type=tuner_type, use_vllm=use_vllm))

    assert engine_cls._patch_active is False
    # 上游 inference_mode 原样保留 ⇒ 这两档的行为与接线前逐字相同。
    assert engine_cls._graspo_test_calls[-1]["inference_mode"] is True


def test_graspo_wiring_installs_but_passes_text_only_rollouts_through(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """**不误伤**：全参 + 文本模型 ⇒ 装了补丁但该次 rollout 仍走上游 ``inference_mode``。"""
    engine_cls, main = _make_defect_engine(multimodal=False, train_forward=False)
    _install_swift_with_pipeline(monkeypatch, engine_cls, main)

    from graspo.flow.backend.msswift.trainer import run_graspo_rlhf

    run_graspo_rlhf([], graspo_config=_graspo_config())

    assert engine_cls._patch_active is True
    assert engine_cls._graspo_test_calls[-1]["inference_mode"] is True


def test_graspo_wiring_restores_infer_on_normal_and_exception_exit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """**退出还原**：作用域结束后（含流水线体抛异常的路径）``_infer`` 恢复原属性。"""
    engine_cls, main = _make_defect_engine(train_forward=False)
    original_infer = engine_cls.__dict__["_infer"]
    _install_swift_with_pipeline(monkeypatch, engine_cls, main)

    from graspo.flow.backend.msswift.trainer import run_graspo_rlhf

    assert run_graspo_rlhf([], graspo_config=_graspo_config()) == "rollout only"
    assert engine_cls.__dict__["_infer"] is original_infer

    class BoomError(RuntimeError):
        pass

    def _explode() -> None:
        main()
        raise BoomError("training loop failed")

    _install_swift_with_pipeline(monkeypatch, engine_cls, _explode)

    with pytest.raises(BoomError):
        run_graspo_rlhf([], graspo_config=_graspo_config())

    assert engine_cls.__dict__["_infer"] is original_infer
