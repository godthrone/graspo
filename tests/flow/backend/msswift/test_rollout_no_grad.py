"""rollout 的 ``inference_mode`` → ``no_grad`` 上下文修正：接线事实 + 能真失败的负向用例。

**为什么需要它**（D46 探针 2026-10-29，真实 OPD 全链路逐字读数）：训练 forward 里视觉塔
``patch_embed`` 的 conv3d 报 ``Inference tensors cannot be saved for backward``，现场是
``weight.is_inference=True`` / ``input.is_inference=False`` —— 被污染的是**参数对象**
（rollout 的 ``inference_mode`` 罩住了 FSDP2 的 all-gather 物化，物化出的参数张量带
inference 标记并被缓存）。本补丁只把那段执行体的上下文换成 ``no_grad``。

**本文件回答四个问题**

1. **真的换了上下文**：全参 + 多模态 ⇒ 引擎方法内 ``torch.is_inference_mode_enabled()``
   为 ``False``、新建张量 ``is_inference()`` 为 ``False``（这就是崩溃的充分条件）。
2. **作用面最小**：LoRA 档**不安装**（类属性对象身份不变）；文本模型安装但**逐字直通**
   （上游 ``inference_mode`` 语义与内存收益不变）；``use_vllm=true`` 不安装。
3. **fail-closed**：上游符号缺失或已改成 ``no_grad`` ⇒ 不安装并记 ``degraded_reason``；
   **安装后被外层 inference_mode 罩回去 ⇒ 抛错点名**（⛔ 不许"看起来修了"）。
4. **只换上下文、不改数值**：同 seed 下，同一段计算在 ``inference_mode`` 与 ``no_grad``
   里产出的张量**逐位相同**（真实链路的逐位对照在 A800 上另做，见工位报告）。

**为什么不用真 ms-swift**：被测的是 graspo 自己的接线。测试用一个**同形替身类**注册进
``sys.modules`` 顶替 ``swift.infer_engine.transformers_engine``（与
``test_rollout_seed.py`` 同一手法）。
"""

from __future__ import annotations

import sys
import types
from types import SimpleNamespace

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
