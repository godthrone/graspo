"""L3 集成测试：msswift 后端的**接入链路**（方案 §3 M3 / 决策 D6；无需 GPU/模型）。

用替身模块（stub）拦截 ms-swift 的真实训练入口，断言 graspo 侧"送进去的东西"
是对的——这是"链路接通"与"链路没接通"之间最直接的判据，且不需要加载模型。

覆盖：

1. **SFT**（G4）：``train_method: sft`` + ``backend: msswift`` 时，graspo 会
   ①把 ``data.train_path`` 转成 ms-swift messages 数据集（末条 assistant 为目标文本）；
   ②用 ``swift.pipelines.sft_main``（ms-swift 的 Python API 入口）跑训练，
   argv 里带上模型/数据/输出目录与 ``msswift`` 段的透传参数。
2. **RL**（G3）：``graspo.backends`` 的 msswift 工厂返回含 ``.train(smoke)`` 的门面，
   且门面走 ``swift.pipelines.train.rlhf.SwiftRLHF``（Python API）而不是 CLI。
3. **ratio 恒 1 bug**（G3 的核心）：同一 loss 函数，传同一个张量当新旧 logprob 时
   ratio ≡ 1（旧 bug 的等价物）；传当前策略前向 + 旧基线时 ratio ≠ 1。
"""

from __future__ import annotations

import importlib.util
import json
import sys
import types
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import pytest

from graspo.core.schema import GraspoConfig

_ARD_SAMPLE = {
    "id": "sample-1",
    "source": "unit-test",
    "data_source": "ard_text",
    "schema_version": "3.0.0",
    "messages": [{"role": "user", "content": "列出 2 的平方"}],
    "targets": [{"id": None, "output": {"content": '{"answer": "4"}', "reasoning": None}}],
}


def _write_ard_sample(path: Path) -> str:
    path.write_text(json.dumps(_ARD_SAMPLE, ensure_ascii=False) + "\n", encoding="utf-8")
    return str(path)


def _config(tmp_path: Path, *, train_method: str) -> GraspoConfig:
    return GraspoConfig.model_validate(
        {
            "train_method": train_method,
            "backend": "msswift",
            "model": {"model_path": "models/Qwen3.5-9B"},
            "data": {"train_path": _write_ard_sample(tmp_path / "ard.jsonl")},
            "training": {
                "output_dir": str(tmp_path / "out"),
                "run_name": "unit",
                "overwrite_output_dir": True,
            },
            "msswift": {"sequence_parallel_size": 2, "deepspeed": "zero2"},
        }
    )


@pytest.fixture()
def stub_ms_swift(monkeypatch):
    """用替身拦截 ms-swift：记录入口被调用的参数，不真的跑训练。"""
    calls: dict[str, list] = {"sft_main": [], "rlhf_pipeline": []}

    swift_module = types.ModuleType("swift")
    trainers_module = types.ModuleType("swift.trainers")

    class _StubTrainer:  # noqa: D401 - 只需要可被 importorskip 风格解析
        """ms-swift SFT 训练器替身。"""

    trainers_module.Trainer = _StubTrainer
    trainers_module.TrainerFactory = SimpleNamespace(TRAINER_MAPPING={"grpo": "unused"})

    pipelines_module = types.ModuleType("swift.pipelines")
    pipelines_module.sft_main = lambda argv: calls["sft_main"].append(list(argv))

    monkeypatch.setitem(sys.modules, "swift", swift_module)
    monkeypatch.setitem(sys.modules, "swift.trainers", trainers_module)
    monkeypatch.setitem(sys.modules, "swift.pipelines", pipelines_module)
    return calls


def _sft_trainer(config: GraspoConfig):
    from graspo.flow.msswift.sft_trainer import create_msswift_sft_trainer

    return create_msswift_sft_trainer(config, None)


def test_sft_wiring_converts_dataset_and_calls_ms_swift_entrypoint(
    tmp_path, stub_ms_swift, monkeypatch
):
    """G4 链路：ARD 样例 → ms-swift messages 数据集 → ``sft_main(argv)``。"""
    config = _config(tmp_path, train_method="sft")
    trainer = _sft_trainer(config)

    trainer.train(smoke=True)

    assert len(stub_ms_swift["sft_main"]) == 1, "ms-swift SFT entrypoint must be called once"
    argv = stub_ms_swift["sft_main"][0]

    def value_of(flag: str) -> str | None:
        return argv[argv.index(flag) + 1] if flag in argv else None

    assert value_of("--model") == "models/Qwen3.5-9B"
    assert value_of("--output_dir") == str(tmp_path / "out")
    assert value_of("--sequence_parallel_size") == "2"
    assert value_of("--deepspeed") == "zero2"
    assert value_of("--max_steps") == "1", "smoke must bound the run to one step"

    dataset_path = Path(value_of("--dataset") or "")
    assert dataset_path.is_file(), "the converted ms-swift dataset must exist on disk"
    rows = [json.loads(line) for line in dataset_path.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 1
    messages = rows[0]["messages"]
    assert messages[0] == {"role": "user", "content": "列出 2 的平方"}
    assert messages[-1]["role"] == "assistant"
    # graspo 的 SFT target 文本 = fenced JSON（与 native 用同一个 build_sft_target_text）
    assert "4" in messages[-1]["content"]


def test_sft_dataset_output_lives_under_the_run_output_dir(tmp_path, stub_ms_swift):
    config = _config(tmp_path, train_method="sft")

    _sft_trainer(config).train(smoke=True)

    dataset_path = Path(
        stub_ms_swift["sft_main"][0][stub_ms_swift["sft_main"][0].index("--dataset") + 1]
    )
    assert str(dataset_path).startswith(str(tmp_path / "out"))


def test_sft_entrypoint_is_not_a_subprocess_and_not_the_swift_cli(tmp_path, stub_ms_swift):
    """D6：主路线是 Python API 作库——不 spawn 子进程、不调 ``swift`` CLI。"""
    config = _config(tmp_path, train_method="sft")
    trainer = _sft_trainer(config)

    import subprocess

    def _forbidden(*args, **kwargs):  # pragma: no cover - 只在被误用时触发
        raise AssertionError("msswift SFT must not spawn subprocesses (decision D6)")

    original_run = subprocess.run
    subprocess.run = _forbidden
    try:
        trainer.train(smoke=True)
    finally:
        subprocess.run = original_run

    assert len(stub_ms_swift["sft_main"]) == 1


def test_ms_sft_available_reflects_swift_importability(stub_ms_swift):
    from graspo.flow.msswift.sft_trainer import ms_sft_available

    assert ms_sft_available() is True


def test_rl_factory_returns_trainer_with_train_contract():
    """``graspo.backends``（RL）的 msswift 工厂契约：``factory(config, sel) -> 含 train``。"""
    from graspo.core.discovery import resolve_backend_builder

    builder = resolve_backend_builder("msswift", train_method="graspo")
    trainer = builder(GraspoConfig.model_validate({"backend": "msswift"}), None)

    assert hasattr(trainer, "train"), f"{type(trainer).__name__} must expose train(smoke)"


def test_rl_path_uses_ms_swift_python_api_not_cli(tmp_path, monkeypatch):
    """G3 链路：RL 门面走 ``SwiftRLHF``（Python API）而不是 ``swift rlhf`` CLI。"""
    recorded: dict[str, object] = {}

    class _StubPipeline:
        def __init__(self, args=None, **kwargs):
            recorded["args"] = args
            recorded["kwargs"] = kwargs
            self.args = SimpleNamespace(rlhf_type="grpo")

        def main(self):
            recorded["main_called"] = True
            return {"ok": True}

    stub_rlhf_module = types.ModuleType("swift.pipelines.train.rlhf")
    stub_rlhf_module.SwiftRLHF = _StubPipeline
    stub_train_module = types.ModuleType("swift.pipelines.train")
    stub_train_module.rlhf = stub_rlhf_module

    monkeypatch.setitem(sys.modules, "swift", types.ModuleType("swift"))
    monkeypatch.setitem(
        sys.modules,
        "swift.trainers",
        SimpleNamespace(TrainerFactory=SimpleNamespace(TRAINER_MAPPING={"grpo": "unused"})),
    )
    monkeypatch.setitem(sys.modules, "swift.pipelines", types.ModuleType("swift.pipelines"))
    monkeypatch.setitem(sys.modules, "swift.pipelines.train", stub_train_module)
    monkeypatch.setitem(sys.modules, "swift.pipelines.train.rlhf", stub_rlhf_module)
    monkeypatch.setitem(sys.modules, "swift.rewards", SimpleNamespace(orms={}))

    import graspo.flow.msswift.trainer as trainer_module

    monkeypatch.setattr(trainer_module, "_PIPELINE_CLASS_CACHE", {})

    from graspo.flow.msswift.trainer import MsSwiftRlTrainer

    config = _config(tmp_path, train_method="graspo")
    # 显式禁用 ms-swift 的 grpo 训练器解析依赖（替身不做真训练）
    monkeypatch.setattr(
        "graspo.flow.msswift.trainer.registered_trainer_class",
        lambda *a, **k: __import__("contextlib").nullcontext(),
    )

    MsSwiftRlTrainer(config, None).train(smoke=True)

    # argv 必须原样交给 ms-swift 的流水线（由它自己的 parse_args 解析）
    argv = recorded["args"]
    assert isinstance(argv, list)
    assert "--reward_funcs" in argv and "graspo" in argv
    assert "--max_steps" in argv
    assert recorded["main_called"] is True


# ── ratio 恒 1 bug 的行为级判据（G3 核心）────────────────────────────────────


def _stub_trainer_for_metrics():
    """构造只带 ``_record_ratio_metrics`` 所需状态的最小对象。"""
    from graspo.flow.msswift.trainer import GraspoMsSwiftGRPOTrainer

    return SimpleNamespace(
        _metrics={"train": defaultdict(list), "eval": defaultdict(list)},
        model=SimpleNamespace(training=True),
        _graspo_injection_failures=0,
        # 组决策台账（E3 引入）：`_record_ratio_metrics` 会读它产出
        # graspo/groups_seen / skipped_groups 等指标。
        _graspo_group_decisions=[],
        _graspo_token_advantages={},
        # 未绑定函数（SimpleNamespace 不做描述符绑定），调用时要显式传 self。
        _record=GraspoMsSwiftGRPOTrainer._record_ratio_metrics,
    )


def test_ratio_is_exactly_one_when_new_and_old_logprobs_are_the_same_tensor():
    """旧 bug 的等价物：``log_probs`` 与 ``old_log_probs`` 是同一个张量 → ratio ≡ 1。"""
    torch = pytest.importorskip("torch")

    stub = _stub_trainer_for_metrics()
    mask = torch.ones(2, 4)
    logps = torch.tensor([[-0.5, -1.0, -2.0, -0.3], [-1.5, -0.7, -0.9, -1.1]])
    advantages = torch.ones(2, 4)

    stub._record(stub, logps, logps, mask, advantages, torch.tensor(0.0))

    metrics = stub._metrics["train"]
    assert metrics["graspo/ratio_mean"][-1] == pytest.approx(1.0)
    assert metrics["graspo/ratio_abs_dev_max"][-1] == pytest.approx(0.0)
    assert metrics["graspo/ratio_is_exactly_one"][-1] == 1.0


def test_ratio_is_not_one_when_the_policy_moved():
    """修复后的判据：当前策略 logprob ≠ 旧基线 → ratio ≠ 1（PPO 裁剪重新有意义）。"""
    torch = pytest.importorskip("torch")

    stub = _stub_trainer_for_metrics()
    mask = torch.ones(2, 4)
    old = torch.tensor([[-0.5, -1.0, -2.0, -0.3], [-1.5, -0.7, -0.9, -1.1]])
    new = old + torch.tensor([[0.05, -0.03, 0.2, 0.0], [-0.1, 0.02, 0.0, 0.15]])
    advantages = torch.ones(2, 4)

    stub._record(stub, new, old, mask, advantages, torch.tensor(0.0))

    metrics = stub._metrics["train"]
    assert metrics["graspo/ratio_is_exactly_one"][-1] == 0.0
    assert metrics["graspo/ratio_abs_dev_max"][-1] > 1e-3
    assert metrics["graspo/ratio_mean"][-1] != pytest.approx(1.0, abs=1e-6)


def test_completion_text_is_captured_before_messages_are_tokenized():
    """回归：不能再把 token id 列表 ``str()`` 当 completion 文本用。

    实测踩过：ms-swift 的 ``_prepare_batch_inputs`` 会把 assistant 消息的 content
    **原地替换成 token ids**，而 graspo 的 ``_compute_advantages`` 在那之后运行——
    若那时才去读 messages，拿到的是 ``[1, 2, 3]``，字符级标注会与 token 完全错位。
    正确做法是在 ``_score_completions`` 阶段就把文本抓下来缓存。
    """
    pytest.importorskip("torch")
    from graspo.flow.msswift.trainer import GraspoMsSwiftGRPOTrainer, _decode_completion

    stub = SimpleNamespace(_graspo_completion_text={"req-1": "hello world"})
    sample = SimpleNamespace(
        request_id="req-1",
        prompt_id="p",
        messages=[{"role": "assistant", "content": [11, 22, 33]}],
    )

    assert GraspoMsSwiftGRPOTrainer._completion_text(stub, sample) == "hello world"

    # 没有缓存时，按 ms-swift 自己的口径解码（三种形态）
    class _Tok:
        def decode(self, ids):
            return f"decoded:{ids}"

    no_cache = SimpleNamespace(_graspo_completion_text={}, _resolve_tokenizer=lambda: _Tok())
    assert GraspoMsSwiftGRPOTrainer._completion_text(no_cache, sample) == "decoded:[11, 22, 33]"
    assert (
        _decode_completion(
            SimpleNamespace(messages=[{"role": "assistant", "content": {"token_ids": [7, 8]}}]),
            _Tok(),
        )
        == "decoded:[7, 8]"
    )
    assert (
        _decode_completion(
            SimpleNamespace(messages=[{"role": "assistant", "content": "plain"}]), None
        )
        == "plain"
    )


def test_graspo_loss_uses_current_logprobs_so_ratio_can_differ():
    """算法核层面：ratio 由 ``(log_probs - old_log_probs)`` 决定，不再被调用方锁死为 1。"""
    torch = pytest.importorskip("torch")

    from graspo.ripple.algorithm import GraspoAlgorithmCore
    from graspo.core.schema import RewardConfig

    core = GraspoAlgorithmCore(reward_config=RewardConfig(), policy_ratio_clip_eps=0.2)
    mask = torch.ones(1, 4)
    old = torch.tensor([[-1.0, -1.0, -1.0, -1.0]])
    advantages = torch.ones(1, 4)

    loss_same = core.compute_loss(old, old, advantages, mask)
    moved = old + 0.5
    loss_moved = core.compute_loss(moved, old, advantages, mask)

    assert loss_same.item() == pytest.approx(-1.0)
    assert loss_moved.item() != pytest.approx(loss_same.item())


# ── G3 有效性：classify_group 组决策注入（E2b/E2a 暴露的功能性缺口）────────────


class _FakeTokenizer:
    """最小 tokenizer：`return_offsets_mapping=True` 时返回逐字符 offset。"""

    def __call__(self, text, return_offsets_mapping=False, add_special_tokens=True):
        tokens = list(text)
        payload = {"input_ids": list(range(len(tokens)))}
        if return_offsets_mapping:
            payload["offset_mapping"] = [(index, index + 1) for index in range(len(tokens))]
        return payload

    def decode(self, ids):
        return "".join(str(i) for i in ids)


def _decision_stub(config=None, *, num_generations=2):
    """构造一个只带组决策链路所需状态的最小 trainer 替身。"""
    import torch

    from graspo.flow.msswift import trainer as trainer_module
    from graspo.flow.msswift.trainer import GraspoMsSwiftGRPOTrainer
    from graspo.ripple.algorithm import GraspoAlgorithmCore

    _config = config or GraspoConfig.model_validate({"backend": "msswift"})
    # 与 trainer 里同一处装配方式（trainer 在 __init__ 内延迟导入该核）
    core = GraspoAlgorithmCore(
        reward_config=_config.reward,
        policy_ratio_clip_eps=_config.training.policy_ratio_clip_eps,
    )
    stub = SimpleNamespace(
        _graspo_config=_config,
        _graspo=core,
        num_generations=num_generations,
        reward_weights=torch.ones(1, dtype=torch.float32),
        _graspo_annotations={},
        _graspo_completion_text={},
        _graspo_token_advantages={},
        _graspo_group_decisions=[],
        _graspo_injection_failures=0,
        _resolve_tokenizer=lambda: _FakeTokenizer(),
        # SimpleNamespace 不做描述符绑定，逐个显式绑成可调用属性（被测方法内部会调它们）。
        _completion_text=lambda sample: trainer_module.GraspoMsSwiftGRPOTrainer._completion_text(
            stub, sample
        ),
        _warn_injection=lambda reason: trainer_module.GraspoMsSwiftGRPOTrainer._warn_injection(
            stub, reason
        ),
        _compute=trainer_module.GraspoMsSwiftGRPOTrainer._compute_graspo_token_advantages,
    )
    return stub, torch


#: graspo 归一化目标（D4 数据契约）：GT 值放在 ``output.content`` 下。
_ARD_TARGETS = [{"id": None, "output": {"content": {"answer": "4"}, "reasoning": None}}]


def _sample(request_id: str, completion: str, *, prompt_id: str = "p1", targets=None):
    return SimpleNamespace(
        request_id=request_id,
        prompt_id=prompt_id,
        is_truncated=False,
        messages=[{"role": "assistant", "content": completion}],
        extra={"targets": json.dumps(targets if targets is not None else _ARD_TARGETS)},
    )


def _annotate(stub, sample, completion):
    from graspo.ripple.annotation.labeler import AnnotationInput, annotate

    return annotate(
        AnnotationInput(
            completion_text=completion,
            targets=_ARD_TARGETS,
            tokenizer=_FakeTokenizer(),
            format_type="json",
            check_json_markdown=True,
            check_think=False,
        )
    )


def test_unparseable_group_is_skipped_instead_of_silently_zero_advantages():
    """G3 缺口的核心判据：不可解析的组必须被**判定出来并跳过**，且原因可读。

    E2a 的实测现象：真实 ARD 样例上 8B 模型输出散文 → 标注全是 TEXT → 每个 token
    的 advantage 都是 0 → loss 0 → 无梯度，而指标只显示 ``advantage_abs_mean: 0``，
    看不出"是数据不可解析"还是"链路坏了"。注入 ``classify_group`` 后，
    这种组必须落成 ``decision=invalid``（或 retry），并进 ``skipped_groups``。
    """
    from graspo.flow.msswift.trainer import GraspoMsSwiftGRPOTrainer

    stub, torch = _decision_stub()
    completions = ["这是一段散文，没有 JSON 围栏。", "另一段散文。"]
    samples = [_sample("r1", completions[0]), _sample("r2", completions[1])]
    for sample, text in zip(samples, completions):
        stub._graspo_annotations[sample.request_id] = _annotate(stub, sample, text)
        stub._graspo_completion_text[sample.request_id] = text

    rewards = torch.zeros(2, 1)
    stub._compute(stub, samples, rewards)

    assert stub._graspo_token_advantages == {}, "不可解析的组不得产出 advantage"
    assert len(stub._graspo_group_decisions) == 1
    keys, group_rewards, content_scores, decision = stub._graspo_group_decisions[0]
    assert content_scores == [0.0, 0.0], "纯散文没有 S/V/E 标注 → content_score 必须为 0"
    assert decision.should_train is False
    assert decision.decision.value in {"retry", "invalid"}, decision.decision.value


def test_parseable_trainable_group_produces_nonzero_token_advantages():
    """可解析且组内有差异的组：必须产出**非零** token 级 advantage 并被判定可训练。"""
    from graspo.flow.msswift.trainer import GraspoMsSwiftGRPOTrainer

    stub, torch = _decision_stub()
    good = '```json\n{"answer": "4"}\n```'
    bad = '```json\n{"answer": "5"}\n```'
    samples = [_sample("r1", good), _sample("r2", bad)]
    for sample, text in zip(samples, [good, bad]):
        stub._graspo_annotations[sample.request_id] = _annotate(stub, sample, text)
        stub._graspo_completion_text[sample.request_id] = text

    # 组内差异：一条满分（>= perfect_skip_reward_threshold），一条 0 分
    rewards = torch.tensor([[1.0], [0.0]])
    stub._compute(stub, samples, rewards)

    assert set(stub._graspo_token_advantages) == {"r1", "r2"}, "可训练的组必须注入 advantage"
    flat = [
        abs(value) for per_token in stub._graspo_token_advantages.values() for value in per_token
    ]
    assert flat and any(value != 0.0 for value in flat), "必须存在非零 token 级 advantage"
    _, _, _, decision = stub._graspo_group_decisions[0]
    assert decision.should_train is True
    assert decision.decision.value == "trainable_max_correct"


def test_group_decisions_are_reported_as_metrics():
    """组决策必须可观测（``graspo/*`` 指标）：跳过组数与被跳过组的原因可读。"""
    from graspo.flow.msswift.trainer import GraspoMsSwiftGRPOTrainer

    stub, torch = _decision_stub()
    text = "散文，没有结构。"
    samples = [_sample("r1", text), _sample("r2", text)]
    for sample in samples:
        stub._graspo_annotations[sample.request_id] = _annotate(stub, sample, text)
        stub._graspo_completion_text[sample.request_id] = text
    stub._compute(stub, samples, torch.zeros(2, 1))

    # 补上指标记录所需的其余状态
    stub._metrics = {"train": defaultdict(list), "eval": defaultdict(list)}
    stub.model = SimpleNamespace(training=True)
    mask = torch.ones(1, 4)
    logps = torch.tensor([[-0.5, -1.0, -2.0, -0.3]])
    GraspoMsSwiftGRPOTrainer._record_ratio_metrics(
        stub, logps, logps, mask, torch.zeros(1, 4), torch.tensor(0.0)
    )

    metrics = stub._metrics["train"]
    assert metrics["graspo/groups_seen"][-1] == 1.0
    assert metrics["graspo/groups_trainable"][-1] == 0.0
    assert metrics["graspo/skipped_groups"][-1] == 1.0
    assert metrics["graspo/unparseable_groups"][-1] == 1.0
    assert metrics["graspo/token_advantage_abs_mean"][-1] == 0.0
    assert metrics["graspo/advantage_abs_mean"][-1] == 0.0
    # 被跳过的原因必须能从台账读出（decision 取值本身即原因）
    assert {entry[3].decision.value for entry in stub._graspo_group_decisions} <= {
        "retry",
        "invalid",
        "invalid_no_preference_gap",
    }


def test_group_rewards_are_read_from_rewards_per_func_by_request_id():
    """分类用的 reward 必须来自 ms-swift 的奖励表（按 request_id 对齐，不按行号猜）。"""
    from graspo.flow.msswift.trainer import _group_rewards

    torch = pytest.importorskip("torch")
    # 传入顺序与 request_id 顺序**故意不同**：必须按 request_id 对齐而不是按行号。
    # ms-swift 的奖励表行序 = request_id 升序（`_compute_rewards_per_func` 的 gather 序），
    # 所以行 0 是 r1、行 1 是 r2。
    samples = [_sample("r2", "b"), _sample("r1", "a")]
    table = torch.tensor([[0.75], [0.25]])  # row0 = r1 = 0.75, row1 = r2 = 0.25

    rewards = _group_rewards(samples, table, torch.ones(1))

    assert rewards == {"r1": 0.75, "r2": 0.25}


def test_group_rewards_degrade_transparently_when_the_table_is_absent():
    from graspo.flow.msswift.trainer import _group_rewards

    assert _group_rewards([_sample("r1", "a")], None, None) == {}
