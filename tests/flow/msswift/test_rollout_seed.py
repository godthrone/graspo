"""OPD rollout 播种：接线事实 + **能真失败的负向用例**（无需 ms-swift / GPU）。

**本文件回答两个问题**

1. **接线**：graspo 的 OPD 路径是否真的在**每次 rollout 生成之前**把全局 torch RNG
   钉到了 ``training.seed``（``_rollout_seed.rollout_seed_deterministic``），
   且只在 ms-swift 的引擎边界上**作用域内**生效、退出即还原。
2. **负向用例（能真失败）**：在**没有**这个补丁的等价条件下，同 config 同 seed 的
   两跑会得到**不同**的首步监督信号；加了补丁就一致。

**为什么能不用 ms-swift 也测出这件事**

``rollout_seed_deterministic`` 找的是
``swift.infer_engine.transformers_engine.TransformersEngine.infer``。测试用一个
**同形替身类**（真 ``torch`` + 真全局 RNG）注册进 ``sys.modules`` 顶替该模块——
被测的是**graspo 自己的接线**，不是 ms-swift。

**"首步监督信号"在这里的等价物**

真实通道里它是"现场采样的补全 token + 教师对其打分"。本测试取其中**可复现性会被
影响的那一半**：现场采样。做法是让替身引擎照真实语义从**全局** torch RNG 采样
（``torch.multinomial``），于是"生成时刻全局 RNG 起点是否确定"直接决定两跑是否
逐字相同。这**不是**在测 ms-swift 的采样算法——那部分（温度/top_p/num_generations）
本补丁一个字都不改。

**跑法**（需要一个真 torch；本机用 conda ``lerobot`` env）：

    ~/miniconda3/envs/lerobot/bin/python -m pytest \
      tests/flow/msswift/test_rollout_seed.py -q -p no:cacheprovider
"""

from __future__ import annotations

import sys
import types
from types import SimpleNamespace

import pytest

from graspo.flow.msswift import _rollout_seed as rs

try:  # 本机（裸 python3）无 torch ⇒ 整文件跳过；容器内/lerobot env 正常跑。
    import torch
except ImportError:  # pragma: no cover - 环境相关
    torch = None

pytestmark = pytest.mark.skipif(torch is None, reason="this test needs a real torch (global RNG)")

#: 负向用例的判据：一次"首步"里采多少个 token。用 64 保证"未播种 ⇒ 必然不同"
#: 的概率约为 1 - 2**-64（不是"希望它不同"，是几乎必然不同）。
_STEP_TOKENS = 64
_VOCAB = 64


# ---------------------------------------------------------------------------
# 替身：与 TransformersEngine.infer 同形的"真从全局 RNG 采样"的引擎
# ---------------------------------------------------------------------------


def _make_engine_stub(record: list[list[int]] | None = None) -> type:
    """造一个同形替身类：``infer`` 从**全局** torch RNG 采样并把 token 记下来。

    真实 ``TransformersEngine.infer``（``swift/infer_engine/transformers_engine.py:573``）
    的签名与本替身一致（``infer_requests`` 位置参数 + ``request_config`` 等），
    本补丁只关心"方法在、能被包装"。

    Args:
        record: 记录器。为 ``None`` 时读类属性 ``_graspo_test_record``
            （供"一个进程内造一次类、跑一次"的用法切换记录器）。
    """

    class TransformersEngine:  # noqa: N801 - 名称必须与上游一致（被包装的判据）
        _graspo_test_record: list[list[int]] = [] if record is None else record

        def infer(self, infer_requests, request_config=None, metrics=None, *, use_tqdm=None, adapter_request=None):
            probs = torch.ones(_VOCAB, dtype=torch.float32) / _VOCAB
            tokens = torch.multinomial(probs, _STEP_TOKENS, replacement=True).tolist()
            type(self)._graspo_test_record.append(tokens)
            return [SimpleNamespace(tokens=tokens)]

    if record is not None:
        TransformersEngine._graspo_test_record = record
    return TransformersEngine


def _install_engine_stub(monkeypatch: pytest.MonkeyPatch, cls: type) -> None:
    """把替身注册成 ``swift.infer_engine.transformers_engine``（顶替真 ms-swift）。

    同时补齐父包链，避免真实 ms-swift 存在时被误导入。
    """
    swift_pkg = types.ModuleType("swift")
    swift_pkg.__path__ = []  # type: ignore[attr-defined]
    infer_pkg = types.ModuleType("swift.infer_engine")
    infer_pkg.__path__ = []  # type: ignore[attr-defined]
    engine_mod = types.ModuleType(rs._ENGINE_MODULE)
    engine_mod.TransformersEngine = cls  # type: ignore[attr-defined]
    for name, mod in (
        ("swift", swift_pkg),
        ("swift.infer_engine", infer_pkg),
        (rs._ENGINE_MODULE, engine_mod),
    ):
        monkeypatch.setitem(sys.modules, name, mod)


def _config(seed: int = 42) -> SimpleNamespace:
    return SimpleNamespace(training=SimpleNamespace(seed=seed))


# ---------------------------------------------------------------------------
# ① 接线事实
# ---------------------------------------------------------------------------


def test_resolve_rollout_seed_reads_training_seed() -> None:
    assert rs.resolve_rollout_seed(_config(42)) == 42
    assert rs.resolve_rollout_seed(_config(7)) == 7


def test_resolve_rollout_seed_rejects_missing_or_non_int() -> None:
    """播种值不确定时必须 fail-closed，不能假装可复现。"""
    with pytest.raises(ValueError, match="training.seed is required"):
        rs.resolve_rollout_seed(SimpleNamespace(training=SimpleNamespace(seed=None)))
    with pytest.raises(ValueError, match="training.seed is required"):
        rs.resolve_rollout_seed(SimpleNamespace(training=SimpleNamespace()))
    # bool 是 int 的子类，必须显式拒绝（True 当 seed 语义上说不通）。
    with pytest.raises(ValueError, match="training.seed is required"):
        rs.resolve_rollout_seed(SimpleNamespace(training=SimpleNamespace(seed=True)))


def test_patch_reseeds_before_every_engine_infer(monkeypatch: pytest.MonkeyPatch) -> None:
    """每次 ``infer`` 之前都重播 ⇒ 每个 rollout 都从确定起点采样。"""
    cls = _make_engine_stub()
    _install_engine_stub(monkeypatch, cls)
    original_infer = cls.infer

    with rs.rollout_seed_deterministic(_config(42)) as ledger:
        assert ledger["required"] is True
        assert cls.infer is not original_infer, "补丁没有装上去（接线失败）"
        engine = cls()
        first = engine.infer([], None)[0].tokens
        second = engine.infer([], None)[0].tokens

    assert ledger["reseed_count"] == 2, "每个 rollout 都应各触发一次播种"
    # 每次 infer 前都重播同一个 seed ⇒ 两次采样逐字相同。
    assert first == second, "播种没有落在每次 infer 之前"


def test_patch_is_idempotent_and_restored(monkeypatch: pytest.MonkeyPatch) -> None:
    """嵌套进入不叠包装；退出即还原（§2.2 显式即防呆）。"""
    cls = _make_engine_stub()
    _install_engine_stub(monkeypatch, cls)
    original_infer = cls.infer

    with rs.rollout_seed_deterministic(_config(42)):
        wrapped = cls.infer
        with rs.rollout_seed_deterministic(_config(42)):
            assert cls.infer is wrapped, "嵌套进入叠加了第二层包装"
        assert cls.infer is wrapped, "内层退出把外层的包装也还原了"

    assert cls.infer is original_infer, "作用域退出后没有还原上游方法"


def test_missing_engine_method_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    """上游漂移（类在、方法不在）必须显式报错，不能静默失去播种。"""

    class DriftedEngine:  # noqa: N801
        pass

    _install_engine_stub(monkeypatch, DriftedEngine)
    with pytest.raises(RuntimeError, match="upstream drift"):
        with rs.rollout_seed_deterministic(_config(42)):
            pass


def test_missing_ms_swift_is_a_noop(monkeypatch: pytest.MonkeyPatch) -> None:
    """ms-swift 不在 ⇒ 连 rollout 都进不去，补丁不装、不报错。"""
    monkeypatch.setitem(sys.modules, rs._ENGINE_MODULE, None)
    with rs.rollout_seed_deterministic(_config(42)) as ledger:
        assert ledger["required"] is False
        assert ledger["reseed_count"] == 0
    rs.assert_rollout_seed_applied(ledger)  # required=False ⇒ 不抛


def test_assert_rollout_seed_applied_fails_closed_when_never_triggered(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """装了补丁却一次没触发 ⇒ 这次运行的可复现性没被保证，必须失败。"""
    cls = _make_engine_stub()
    _install_engine_stub(monkeypatch, cls)
    with rs.rollout_seed_deterministic(_config(42)) as ledger:
        pass  # 一次 infer 都不调
    with pytest.raises(RuntimeError, match="never called"):
        rs.assert_rollout_seed_applied(ledger)


# ---------------------------------------------------------------------------
# ② 负向用例：修复前两跑不同；修复后一致
# ---------------------------------------------------------------------------


def _one_run(cls: type, *, pre_consume: int) -> list[list[int]]:
    """在一个"进程"里跑一次 rollout，返回这次采到的 token 序列（列表套一层）。

    ``pre_consume`` = 进程内此前一切消耗全局随机数的东西（dataloader worker 播种、
    数据集/模型构造…）。真实两跑在这里**可以不同**，而 ms-swift 只在 trainer
    ``__init__`` 播一次种，所以这个差值会直接传到生成时刻的 RNG 起点。
    """
    record: list[list[int]] = []
    cls._graspo_test_record = record  # type: ignore[attr-defined]
    engine = cls()
    if pre_consume:
        torch.rand(pre_consume)
    engine.infer([], None)
    return record


def test_negative_case_without_patch_two_runs_diverge(monkeypatch: pytest.MonkeyPatch) -> None:
    """★ **修复前**的等价条件：rollout 前不重播 ⇒ 两跑首步监督信号不同。"""
    # 注意：这里**不进入** rollout_seed_deterministic —— 这正是"修复前"。
    cls = _make_engine_stub()
    run_a = _one_run(cls, pre_consume=0)
    run_b = _one_run(cls, pre_consume=1)  # 只多消耗 1 个随机数

    assert run_a != run_b, (
        "负向用例没有真失败：未播种时两跑竟然一致 —— 说明这个替身没有真的从全局 RNG 采样，"
        "测试本身失效（必须修测试，不能把失效的测试当通过）"
    )


def test_negative_case_with_patch_two_runs_are_identical(monkeypatch: pytest.MonkeyPatch) -> None:
    """★ **修复后**：同 config 同 seed、rollout 前消耗量不同 ⇒ 首步监督信号逐字相同。"""
    cls = _make_engine_stub()
    _install_engine_stub(monkeypatch, cls)

    with rs.rollout_seed_deterministic(_config(42)) as ledger:
        run_a = _one_run(cls, pre_consume=0)
        run_b = _one_run(cls, pre_consume=1)

    assert ledger["reseed_count"] == 2, "两跑各一次 rollout ⇒ 应触发两次播种"
    assert run_a == run_b, "修复后两跑仍然不同 —— 播种没有覆盖到每次 rollout"


# ---------------------------------------------------------------------------
# ③ 端到端接线：走真的 MsSwiftOpdTrainer.train()，只把 ms-swift 换成替身
# ---------------------------------------------------------------------------


def _opd_config(tmp_path) -> object:
    """造一个 ``train_method="opd"`` + ``backend="msswift"`` 的最小真配置。"""
    from graspo.core.schema import GraspoConfig

    sample = tmp_path / "ard.jsonl"
    sample.write_text(
        '{"id":"s1","source":"unit","data_source":"ard_text","schema_version":"3.0.0",'
        '"messages":[{"role":"user","content":"抬起手臂"}],'
        '"targets":[{"id":null,"output":{"content":"lift_arm({})","reasoning":null}}]}\n',
        encoding="utf-8",
    )
    return GraspoConfig.model_validate(
        {
            "train_method": "opd",
            "backend": "msswift",
            "model": {"model_path": str(tmp_path / "Qwen3.5-9B")},
            "data": {"train_path": str(sample)},
            "training": {
                "output_dir": str(tmp_path / "out"),
                "run_name": "unit",
                "overwrite_output_dir": True,
                "seed": 1234,
            },
            "distill": {"teacher_model_path": str(tmp_path / "Qwen3.8-27B")},
        }
    )


def _stub_ms_swift_opd_entry(train_calls: list[list[str]]) -> type:
    """替身 ms-swift 入口：``main()`` 只**模拟一次 on-policy rollout** 就返回。

    刻意在"训练内"调用一次 ``TransformersEngine.infer`` —— 这正是真实 GKD 通道里
    会触发的那个方法。于是本用例能断言"graspo 的 OPD 路径在训练期间确实播种过"，
    而不只是"模块自己能工作"。
    """

    class _StubSwiftRLHF:
        def __init__(self, args=None):
            self.args = args

        def main(self):
            train_calls.append(["main"])
            from swift.infer_engine import TransformersEngine

            TransformersEngine().infer([], None)
            return {"ok": True}

    return _StubSwiftRLHF


def test_opd_channel_seeds_rollout_end_to_end(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    """★ 通道级判据：真 ``MsSwiftOpdTrainer.train()`` 期间，rollout 前被播种过。"""
    record: list[list[int]] = []
    cls = _make_engine_stub(record)

    train_calls: list[list[str]] = []
    stub = _stub_ms_swift_opd_entry(train_calls)

    stub_rlhf_module = types.ModuleType("swift.pipelines.train.rlhf")
    stub_rlhf_module.SwiftRLHF = stub  # type: ignore[attr-defined]
    stub_train_module = types.ModuleType("swift.pipelines.train")
    stub_train_module.rlhf = stub_rlhf_module  # type: ignore[attr-defined]
    engines_mod = types.ModuleType(rs._ENGINE_MODULE)
    engines_mod.TransformersEngine = cls  # type: ignore[attr-defined]
    infer_pkg = types.ModuleType("swift.infer_engine")
    infer_pkg.__path__ = []  # type: ignore[attr-defined]
    infer_pkg.TransformersEngine = cls  # type: ignore[attr-defined]
    infer_pkg.transformers_engine = engines_mod  # type: ignore[attr-defined]

    def _fake_rlhf_main(argv):
        # graspo 走 ``from swift.pipelines import rlhf_main``；替身在此实例化并跑 main()。
        train_calls.append(list(argv))
        stub(argv).main()

    pipelines_mod = types.ModuleType("swift.pipelines")
    pipelines_mod.rlhf_main = _fake_rlhf_main  # type: ignore[attr-defined]

    for name, mod in (
        ("swift", types.ModuleType("swift")),
        ("swift.infer_engine", infer_pkg),
        (rs._ENGINE_MODULE, engines_mod),
        ("swift.pipelines", pipelines_mod),
        ("swift.pipelines.train", stub_train_module),
        ("swift.pipelines.train.rlhf", stub_rlhf_module),
    ):
        monkeypatch.setitem(sys.modules, name, mod)

    from graspo.flow.msswift.opd_trainer import MsSwiftOpdTrainer

    config = _opd_config(tmp_path)

    # 不真跑 ms-swift 的 argv 解析：本用例只验证"种子接线"这一段。
    monkeypatch.setattr(
        "graspo.flow.msswift._config_mapping.graspo_to_ms_swift_argv",
        lambda *a, **k: ["--rlhf_type", "gkd"],
    )
    # 数据集准备会写盘并依赖 ms-swift 的 dataset 模块；此处把它替换成"已备好"。
    monkeypatch.setattr(
        "graspo.flow.msswift.dataset.prepare_ms_swift_dataset",
        lambda *a, **k: str(tmp_path / "opd.jsonl"),
    )

    MsSwiftOpdTrainer(config, None).train(smoke=True)

    assert train_calls and train_calls[0][0] == "--rlhf_type", "替身入口没有被调用 —— 接线没通"
    assert record, "训练期间没有发生任何 on-policy rollout（测试没覆盖到真实路径）"
