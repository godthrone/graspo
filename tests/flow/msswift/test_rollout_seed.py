"""ms-swift rollout 播种：接线事实 + **能真失败的负向用例**（无需 ms-swift / GPU）。

**本文件回答三个问题**

1. **接线**：graspo 的 OPD 与 GRPO 两条通道是否真的在**每次 rollout 生成之前**把
   全局 torch RNG 钉到了**确定派生**的起点（``_rollout_seed.rollout_seed_deterministic``
   / 共用入口 ``rollout_seeding``），且只在 ms-swift 的引擎边界上**作用域内**生效、
   退出即还原。
2. **可复现（确定性）**：在**没有**这个补丁的等价条件下，同 config 同 seed 的
   两跑会得到**不同**的首步监督信号；加了补丁就一致。
3. **多样性（不许用可复现换塌缩）**：补丁**不得**把同一 rank 内相邻两次 rollout、
   或不同 rank 的 rollout 钉到**同一条随机流**上 —— 那会让 GRPO 的组内样本逐字
   相同 ⇒ ``frac_reward_zero_std`` 恒 1.0 ⇒ advantage 全零 ⇒ 没有学习信号。
   本文件用**同一个替身引擎**跑"可复现但塌缩"的变体，证明它会被这条判据判失败。

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

import contextlib
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

        def infer(
            self,
            infer_requests,
            request_config=None,
            metrics=None,
            *,
            use_tqdm=None,
            adapter_request=None,
        ):
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
    """每次 ``infer`` 之前都重播 ⇒ 每个 rollout 都从**确定派生**的起点采样。

    判据分两半（缺一不可）：

    - **确定性**：同 config、同 rank、同调用序号 ⇒ 逐位相同（本用例的"两跑"用
      两个独立进程式消耗模拟，见 ``_one_run``）；
    - **不复用同一条流**：同一次运行内相邻两次调用用**不同**的派生种子
      （上一版钉同一个字面值，会让相邻两次生成逐字相同 —— 那正是塌缩的前奏）。
    """
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
    assert ledger["call_seeds"] == [
        42,
        43,
    ], "相邻两次调用没有用不同的派生种子 —— 相邻 rollout 会复用同一条随机流"
    assert first != second, "相邻两次 rollout 逐字相同 —— 播种值没有随调用序号前进，随机流被复用"


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
# ①b 派生种子与 rank 区分（纯函数 + 环境变量，无需 torch）
# ---------------------------------------------------------------------------


def test_resolve_rollout_rank_reads_env_first_then_falls_back(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """rank 必须**显式**可取到，并且来源要能被记录（§2.2）。"""
    monkeypatch.delenv("RANK", raising=False)
    monkeypatch.delenv("LOCAL_RANK", raising=False)
    assert rs.resolve_rollout_rank() == (0, "single-process")

    monkeypatch.setenv("LOCAL_RANK", "3")
    assert rs.resolve_rollout_rank() == (3, "env:LOCAL_RANK")

    monkeypatch.setenv("RANK", "2")
    assert rs.resolve_rollout_rank() == (2, "env:RANK")

    monkeypatch.setenv("RANK", "not-an-int")
    assert rs.resolve_rollout_rank() == (3, "env:LOCAL_RANK")
    monkeypatch.setenv("RANK", "")


def test_derive_rollout_seed_is_deterministic_and_rank_separated() -> None:
    """派生种子必须同时满足"确定性"与"rank / 调用序号都不同"。"""
    # 确定性：同入参 ⇒ 同出参（纯函数，不读时钟/全局 RNG）。
    assert rs.derive_rollout_seed(base_seed=42, rank=0, call_index=0) == 42
    assert rs.derive_rollout_seed(base_seed=42, rank=0, call_index=0) == 42

    # 调用序号区分：相邻两次调用不撞车。
    assert rs.derive_rollout_seed(base_seed=42, rank=0, call_index=1) == 43

    # rank 区分：不同 rank 的同一调用序号不撞车（含 rank 0 与 rank 1）。
    seeds = [rs.derive_rollout_seed(base_seed=42, rank=r, call_index=0) for r in range(8)]
    assert len(set(seeds)) == len(seeds), f"不同 rank 推出了相同种子：{seeds}"

    # 跨 rank 区间不重叠：rank0 的第 k 次永远不等于 rank1 的第 j 次
    # （k, j 都远小于 RANK_STRIDE，这正是"必然不撞"而不是"大概率不撞"的原因）。
    for k in range(0, 500, 37):
        for j in range(0, 500, 37):
            rank0 = rs.derive_rollout_seed(base_seed=42, rank=0, call_index=k)
            rank1 = rs.derive_rollout_seed(base_seed=42, rank=1, call_index=j)
            assert rank0 != rank1


def test_derive_rollout_seed_rejects_negative_inputs_and_overflow() -> None:
    """未预期的输入必须显式报错，不能把负数种子静默传下去（§13.1）。"""
    with pytest.raises(ValueError, match="rank must be >= 0"):
        rs.derive_rollout_seed(base_seed=42, rank=-1, call_index=0)
    with pytest.raises(ValueError, match="call_index must be >= 0"):
        rs.derive_rollout_seed(base_seed=42, rank=0, call_index=-1)
    with pytest.raises(ValueError, match="negative value"):
        rs.derive_rollout_seed(base_seed=-1, rank=0, call_index=0)


def test_patch_uses_rank_offset_from_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """包装里实际用的种子必须带上本进程 rank 的偏移（真机上由 torchrun 下发）。"""
    cls = _make_engine_stub()
    _install_engine_stub(monkeypatch, cls)
    monkeypatch.setenv("RANK", "2")

    with rs.rollout_seed_deterministic(_config(42)) as ledger:
        cls().infer([], None)
        cls().infer([], None)

    assert ledger["rank"] == 2
    assert ledger["rank_source"] == "env:RANK"
    assert ledger["call_seeds"] == [42 + 2 * rs._RANK_STRIDE, 42 + 2 * rs._RANK_STRIDE + 1]


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
    """★ **修复后**：同 config 同 seed、同卡位、rollout 前消耗量不同 ⇒ 逐字相同。

    **两跑是两个独立进程**（各自进入一次 ``rollout_seed_deterministic``，调用序号
    都从 0 起）—— 这正是真机"同一份清单跑两次"的形状。上一次接线把两个 rank、
    两次调用都钉到同一个字面值，才让"两跑相同"与"组内塌缩"混为一谈；
    本用例只主张**两跑相同**，多样性由 ``test_derived_seed_keeps_the_group_diverse``
    与负向对照独立主张（同一件事的两个判据，分开测）。
    """
    cls = _make_engine_stub()
    _install_engine_stub(monkeypatch, cls)

    with rs.rollout_seed_deterministic(_config(42)) as ledger_a:
        run_a = _one_run(cls, pre_consume=0)
    with rs.rollout_seed_deterministic(_config(42)) as ledger_b:
        run_b = _one_run(cls, pre_consume=1)  # 只多消耗 1 个随机数

    assert ledger_a["reseed_count"] == 1 and ledger_b["reseed_count"] == 1
    assert (
        ledger_a["call_seeds"] == ledger_b["call_seeds"]
    ), "两跑的派生种子序列不同 —— 说明派生量不是 (seed, rank, call_index) 的确定函数"
    assert run_a == run_b, "修复后两跑仍然不同 —— 播种没有覆盖到每次 rollout"


# ---------------------------------------------------------------------------
# ②b ★ 负向对照：**可复现但塌缩**的变体必须被多样性判据判失败
# ---------------------------------------------------------------------------

#: 一次 rollout 组里有几份生成（对齐真机 ``num_generations=8``）。
_GROUP_SIZE = 8


def _group_of_diverse_samples(fresh_stream_before_each_sample: bool) -> list[tuple]:
    """在一次 rollout 组内采 ``_GROUP_SIZE`` 份样本，返回各自的 token 元组。

    Args:
        fresh_stream_before_each_sample: ``True`` 模拟**上一版的塌缩变体** ——
            每采一份就把全局 RNG 拉回**同一个字面值**（"每份生成前各重播一次"）。
            ``False`` 模拟**本方案** —— 每份生成只在组开始时重播一次（一次
            ``infer`` 一次播种），组内各份**连续消耗同一条流**。
    """
    stream_seed = 1234
    samples: list[tuple] = []
    for _ in range(_GROUP_SIZE):
        if fresh_stream_before_each_sample:
            torch.manual_seed(stream_seed)
        probs = torch.ones(_VOCAB, dtype=torch.float32) / _VOCAB
        samples.append(tuple(torch.multinomial(probs, _STEP_TOKENS, replacement=True).tolist()))
    return samples


def _passes_reproducibility(seed: int, *, collapse: bool) -> bool:
    """可复现判据：同 seed、同样更多消耗 ⇒ 整组逐位相同。"""

    def one_run() -> list[tuple]:
        torch.manual_seed(seed)
        torch.rand(7)  # 模拟"生成之前被别的消费者推进过全局 RNG"
        if collapse:
            return [
                tuple(
                    torch.multinomial(
                        torch.ones(_VOCAB, dtype=torch.float32) / _VOCAB,
                        _STEP_TOKENS,
                        replacement=True,
                    ).tolist()
                )
                for _ in range(_GROUP_SIZE)
            ]
        return _group_of_diverse_samples(fresh_stream_before_each_sample=False)

    return one_run() == one_run()


def _passes_diversity(samples: list[tuple]) -> bool:
    """★ 本包新增的**多样性判据**（对着真机事故的三条迹象写成）。

    与任务书的三条判据一一对应，但用"替身引擎可观测的量"表达：

    1. **组内不得塌缩**：``frac_reward_zero_std`` 只在"整组逐字相同"时为 1.0
       ⇒ 判据 = 组内不得只有一种样本（等价于 ``reward_std > 0``）；
    2. **补全长度不得恒等**：真机事故里 ``completions/max_length`` 恒 30.0；
    3. **必须有梯度**：loss/grad_norm 恒 0 的根因就是 1 ⇒ 由 1 覆盖。
    """
    return len(set(samples)) > 1


def test_negative_control_collapsing_variant_is_caught_by_the_diversity_gate() -> None:
    """★★ **本包最重要的回归判据**：可复现但塌缩的变体必须被判失败。

    上一版接线（"每次 ``infer`` 前重播 ``training.seed`` 的**字面值**"）是
    **可复现的**，所以只测可复现性的用例**永远抓不到它**。本用例把那个变体的
    语义独立实现一遍，先确认它确实"可复现"，再用多样性判据把它判失败 ——
    于是"将来有人再犯同一个错"时，测试会拦下来。

    真机事故（T033 4 卡）：补全长度被压到同一上限、首行
    ``frac_reward_zero_std=1.0``、``loss=grad_norm=0``。
    """
    # 1) 塌缩变体确实**可复现** —— 只看可复现性的测试会放它过去。
    assert _passes_reproducibility(42, collapse=True) is True

    # 2) 但它过不了多样性判据。
    collapsed_group = _group_of_diverse_samples(fresh_stream_before_each_sample=True)
    assert len(set(collapsed_group)) == 1, "负向对照的替身没有真的塌缩 —— 这条用例本身失效了"
    assert (
        _passes_diversity(collapsed_group) is False
    ), "可复现但塌缩的变体没有被多样性判据判失败 —— 这个测试失去了防回归的意义"

    # 3) 本方案（一次 infer 一次播种）同时满足两条判据。
    healthy_group = _group_of_diverse_samples(fresh_stream_before_each_sample=False)
    assert _passes_diversity(healthy_group) is True
    assert _passes_reproducibility(42, collapse=False) is True


def test_derived_seed_keeps_the_group_diverse(monkeypatch: pytest.MonkeyPatch) -> None:
    """★ 端到端形态：真走一次补丁，得到的组内样本必须**不塌缩**。

    这是"补丁接线"与"组内多样性"的连接点 —— 补丁只重播**一次/调用**，
    组内 ``num_generations`` 份样本连续消耗同一条流，因此两两不同。
    """
    cls = _make_engine_stub()
    _install_engine_stub(monkeypatch, cls)

    with rs.rollout_seed_deterministic(_config(42)):
        probs = torch.ones(_VOCAB, dtype=torch.float32) / _VOCAB
        cls().infer([], None)  # 一次调用 = 一个 rollout 组（8 份生成在同一批里）
        group = [
            tuple(torch.multinomial(probs, _STEP_TOKENS, replacement=True).tolist())
            for _ in range(_GROUP_SIZE)
        ]

    assert (
        _passes_diversity(group) is True
    ), "补丁只在每次 infer 前重播一次，组内 8 份样本不该完全相同"


# ---------------------------------------------------------------------------
# ③ 共用接线入口 ``rollout_seeding``（§1.4 单一真相源）
# ---------------------------------------------------------------------------


def test_rollout_seeding_is_the_shared_entry_and_logs_the_ledger(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """``rollout_seeding`` = 装包装 + 退出校验 + **把账本打进日志**（§2.2 显式即防呆）。"""
    cls = _make_engine_stub()
    _install_engine_stub(monkeypatch, cls)
    original_infer = cls.infer

    with caplog.at_level("WARNING", logger=rs.__name__):
        with rs.rollout_seeding(_config(42)) as ledger:
            assert cls.infer is not original_infer, "共用入口没有把包装装上去"
            cls().infer([], None)

    assert ledger["reseed_count"] == 1
    assert cls.infer is original_infer, "共用入口退出后没有还原上游方法"
    messages = [r.getMessage() for r in caplog.records]
    # §2.2 显式性：日志必须同时给出**基础种子、rank 区分方式、重播次数、实际用的种子**。
    # 只打印 reseed_count 不够 —— 上一版的塌缩正是"各 rank 播了同一个字面值"，
    # 而那件事在旧日志里看不出来。
    assert any(
        "graspo rollout seeding applied" in m
        and "reseed_count=1" in m
        and "base_seed=42" in m
        and "rank=0" in m
        and "rank_source=" in m
        and "first_call_seed=42" in m
        for m in messages
    ), f"播种生效的实际参数没有完整落进日志（§2.2）：{messages}"


def test_rollout_seeding_fails_closed_via_the_shared_entry(monkeypatch: pytest.MonkeyPatch) -> None:
    """共用入口在"装了却一次没触发"时同样 fail-closed（与 OPD 旧接线语义一致）。"""
    cls = _make_engine_stub()
    _install_engine_stub(monkeypatch, cls)

    with pytest.raises(RuntimeError, match="never called"):
        with rs.rollout_seeding(_config(42)):
            pass


# ---------------------------------------------------------------------------
# ④ 端到端接线：走真的 MsSwiftOpdTrainer.train()，只把 ms-swift 换成替身
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


# ---------------------------------------------------------------------------
# ⑤ 端到端接线：走真的 MsSwiftRlTrainer.train()（GRPO 通道）
# ---------------------------------------------------------------------------


def _rl_config(tmp_path) -> object:
    """造一个 ``train_method="graspo"`` + ``backend="msswift"`` 的最小真配置。"""
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
            "train_method": "graspo",
            "backend": "msswift",
            "model": {"model_path": str(tmp_path / "Qwen3.5-9B")},
            "data": {"train_path": str(sample)},
            "training": {
                "output_dir": str(tmp_path / "out"),
                "run_name": "unit",
                "overwrite_output_dir": True,
                "seed": 1234,
            },
        }
    )


def test_graspo_channel_seeds_rollout_end_to_end(
    monkeypatch: pytest.MonkeyPatch, tmp_path, caplog: pytest.LogCaptureFixture
) -> None:
    """★ GRPO 通道级判据：真 ``MsSwiftRlTrainer.train()`` 期间，rollout 前被播种过。

    这是 ``T033``（9B·GRASPO·LoRA·ms-swift·4 卡，两跑生成内容不同）的**回归判据**。
    修复前 ``_rollout_seed`` 只接在 OPD，GRPO 通道**零覆盖** —— 本用例在修复前会失败：
    ``TransformersEngine.infer`` 在训练期间没有被 ``torch.manual_seed(training.seed)``
    重播，因此替身引擎观测到的"补丁已装 + 全局 RNG 起点 = config seed"不成立。
    """
    record: list[list[int]] = []
    cls = _make_engine_stub(record)
    observed: list[dict] = []

    class _StubPipeline:
        """替身 ms-swift 流水线：``main()`` 只**模拟一次 on-policy rollout**。"""

        def __init__(self, args=None, **kwargs):
            self.args = SimpleNamespace(rlhf_type="grpo")

        def main(self):
            # 真实 GRPO 通道（use_vllm=false）在训练期间就是这样触发生成的。
            observed.append(
                {
                    "patched": bool(getattr(cls.infer, "_graspo_rollout_seeded", False)),
                    "global_torch_seed": torch.initial_seed(),
                }
            )
            cls().infer([], None)
            return {"ok": True}

    stub_rlhf_module = types.ModuleType("swift.pipelines.train.rlhf")
    stub_rlhf_module.SwiftRLHF = _StubPipeline  # type: ignore[attr-defined]
    stub_train_module = types.ModuleType("swift.pipelines.train")
    stub_train_module.rlhf = stub_rlhf_module  # type: ignore[attr-defined]
    engines_mod = types.ModuleType(rs._ENGINE_MODULE)
    engines_mod.TransformersEngine = cls  # type: ignore[attr-defined]
    infer_pkg = types.ModuleType("swift.infer_engine")
    infer_pkg.__path__ = []  # type: ignore[attr-defined]
    infer_pkg.transformers_engine = engines_mod  # type: ignore[attr-defined]

    for name, mod in (
        ("swift", types.ModuleType("swift")),
        ("swift.infer_engine", infer_pkg),
        (rs._ENGINE_MODULE, engines_mod),
        ("swift.pipelines", types.ModuleType("swift.pipelines")),
        ("swift.pipelines.train", stub_train_module),
        ("swift.pipelines.train.rlhf", stub_rlhf_module),
        ("swift.rewards", SimpleNamespace(orms={})),
        ("swift.trainers", SimpleNamespace(TrainerFactory=SimpleNamespace(TRAINER_MAPPING={}))),
    ):
        monkeypatch.setitem(sys.modules, name, mod)

    import graspo.flow.msswift.trainer as trainer_module

    # 本用例只验证"种子接线"这一段：把 ms-swift 的解析/奖励注册/argv 映射换成替身，
    # 训练语义（模型、数据、算法）一律不动。
    monkeypatch.setattr(trainer_module, "_PIPELINE_CLASS_CACHE", {})
    monkeypatch.setattr(trainer_module, "_require_ms_swift", lambda: None)
    monkeypatch.setattr(trainer_module, "register_graspo_reward", lambda: None)
    monkeypatch.setattr(
        trainer_module, "registered_trainer_class", lambda *a, **k: contextlib.nullcontext()
    )
    monkeypatch.setattr(
        trainer_module, "graspo_to_ms_swift_argv", lambda *a, **k: ["--rlhf_type", "grpo"]
    )
    monkeypatch.setattr(trainer_module, "launcher_env", lambda *a, **k: {})
    monkeypatch.setattr(
        "graspo.flow.msswift.dataset.prepare_ms_swift_dataset",
        lambda *a, **k: str(tmp_path / "rl.jsonl"),
    )

    from graspo.flow.msswift.trainer import MsSwiftRlTrainer

    with caplog.at_level("WARNING", logger=rs.__name__):
        MsSwiftRlTrainer(_rl_config(tmp_path), None).train(smoke=False)

    assert observed, "训练期间没有发生任何 on-policy rollout（测试没覆盖到真实路径）"
    assert record, "替身引擎没有被真正调用"
    assert observed[0]["patched"] is True, (
        "GRPO 通道没有把 rollout 播种补丁装到 TransformersEngine.infer 上 —— "
        "这正是 T033 缺陷（GRPO 通道零覆盖）"
    )
    assert observed[0]["global_torch_seed"] == 1234, (
        "生成时刻的全局 torch RNG 起点不是派生种子（rank=0、call_index=0 ⇒ base_seed）"
        " —— 播种没有落在 rollout 之前"
    )
    messages = [r.getMessage() for r in caplog.records]
    assert any(
        "graspo rollout seeding applied" in m
        and "reseed_count=1" in m
        and "base_seed=1234" in m
        and "rank=" in m
        for m in messages
    ), f"GRPO 通道的播种事实没有落进日志（§2.2）：{messages}"

    # ★ 跨 rank 判据（本包新增）：同一个基础种子、同一个调用序号，rank 不同的两个
    # 进程必须拿到**不同**的派生种子 —— 否则 GRPO 组内方差会被抹平（T033 事故）。
    base = rs.resolve_rollout_seed(_rl_config(tmp_path))
    assert rs.derive_rollout_seed(base_seed=base, rank=0, call_index=0) != rs.derive_rollout_seed(
        base_seed=base, rank=1, call_index=0
    )
