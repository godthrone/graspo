"""lr 策略测试：① lr 不随卡数变化 ② lr 表按方法来（每条带阳性对照）。

对应主席裁定（2026-10-15）：
  · 「针对不同的训练方式，给它不同的步长。比如：SFT 的步长要长一点，强化学习的步长要短一点。」
  · 「这种东西，既然我们取消了，那就直接让这个机制消失就完了，干嘛后再除一次？」
    ⇒「按卡数缩放学习率」这个机制已**整体删除**（⛔ 不是设成 none、⛔ 不留别名/兼容分支）。

两个测试面：
  ① `src/graspo/core/schema.py` —— 训练配置层：lr 是**常量**，dp=1/2/4 不改变它。
  ② `tools/matrix/extract_spec.py` 的 `LR_POLICY` —— 配置生成层：一个方法一个值。

★ 阳性对照（positive control）：每条断言都配一条"若机制复辟 / 表写错，本断言会当场失败"
的反向证明，避免"断言恒真"的假绿。

★ 为什么不直接写被删除的字段名：本包的静态判据要求 `git grep` 该机制的两个标识符
**均为 0 命中**（含注释、字符串与测试）。故结构类断言改用**名字无关**的谓词
（"不存在名字含 scal 的字段""不存在接收 dp_size 的成员"），照样能证伪，且不把标识符
重新引回源码。机制消失的**静态**证明见 `report.md §③`。
"""

from __future__ import annotations

import importlib.util
import inspect
import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

from graspo.core.schema import GraspoConfig, GraspoFlowConfig, TrainingConfig

REPO = Path(__file__).resolve().parents[1]
_MATRIX_TOOLS = REPO / "tools" / "matrix"


def _load_extract_spec():
    """按文件路径加载 `tools/matrix/extract_spec.py`（不污染 sys.path 顺序以外的状态）。

    该模块以 `import extract_params as ep` 引用同目录模块，故需先把 `tools/matrix`
    放进 `sys.path`（与仓内其它按路径加载共享模块的 e2e 测试同范式）。
    """
    if str(_MATRIX_TOOLS) not in sys.path:
        sys.path.insert(0, str(_MATRIX_TOOLS))
    name = "matrix_extract_spec_lr_under_test"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, _MATRIX_TOOLS / "extract_spec.py")
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# ── ① lr 不随卡数变化 ────────────────────────────────────────────────────────


@pytest.mark.parametrize("tuner_type", ["lora", None])
def test_learning_rate_is_invariant_to_dp_size(tuner_type):
    """dp=1/2/4 三个配置对象的 `training.learning_rate` **逐位相等**。"""
    cfg_kwargs = {} if tuner_type is None else {"tuner_type": tuner_type}
    cfgs = [
        GraspoConfig.from_dict(
            {
                "train_method": "sft",
                "backend": "native",
                **cfg_kwargs,
                "training": {"learning_rate": 2.0e-5},
                "native": {"dp_size": dp, "tp_size": 1, "pp_size": 1},
            }
        )
        for dp in (1, 2, 4)
    ]

    assert [c.training.learning_rate for c in cfgs] == [2.0e-5, 2.0e-5, 2.0e-5]

    # ★ 阳性对照：卡数**真的被配置吃进去了**。否则"恒定"可能只是 dp 没生效，
    #   断言就会变成恒真的假绿。同时证明这里确实构造了三个不同的 dp 语境。
    assert [c.native.dp_size for c in cfgs] == [1, 2, 4]
    # ★ 阳性对照之二：若旧机制复辟（lr × dp_size），这三次取值必然互不相同
    #   —— 该列表"三个值互异"能成立，说明上面的"三者相等"是有信息量的。
    scaled_under_the_removed_mechanism = [2.0e-5 * dp for dp in (1, 2, 4)]
    assert len(set(scaled_under_the_removed_mechanism)) == 3


def test_learning_rate_survives_config_round_trip_unchanged():
    """`learning_rate` 只由配置值决定：改 dp_size 不产生任何二次修正。"""
    base = {"train_method": "graspo", "backend": "native", "training": {"learning_rate": 5.0e-6}}
    one_card = GraspoConfig.from_dict({**base, "native": {"dp_size": 1, "tp_size": 1, "pp_size": 1}})
    four_cards = GraspoConfig.from_dict({**base, "native": {"dp_size": 4, "tp_size": 1, "pp_size": 1}})
    assert one_card.training.learning_rate == four_cards.training.learning_rate == 5.0e-6
    # ★ 阳性对照：两个对象确实不同（dp 生效），不是同一个对象被比较了两次。
    assert one_card.native.dp_size == 1 and four_cards.native.dp_size == 4
    assert one_card is not four_cards


# ── ①′ 机制真的消失了（结构类断言，名字无关）────────────────────────────────


def test_no_card_count_scaling_field_remains_on_training_config():
    """`TrainingConfig` 上**不存在**任何名字含 `scal` 的字段（缩放开关已整体删除）。"""
    scaling_fields = [n for n in TrainingConfig.model_fields if "scal" in n.lower()]
    assert scaling_fields == []
    # ★ 阳性对照：扫描谓词本身是有效的（能量到 `learning_rate`），不是"永远返回空"。
    assert "learning_rate" in TrainingConfig.model_fields
    assert [n for n in TrainingConfig.model_fields if "rate" in n.lower()]


def test_no_training_config_member_takes_dp_size():
    """`TrainingConfig` 上**不存在**接收 `dp_size` 的公开成员（缩放计算已整体删除）。"""
    receivers = [
        name
        for name in dir(TrainingConfig)
        if not name.startswith("_")
        and callable(getattr(TrainingConfig, name, None))
        and "dp_size" in inspect.signature(getattr(TrainingConfig, name)).parameters
    ]
    assert receivers == []
    # ★ 阳性对照：扫描谓词本身有效（`native` 配置段里确实有 `dp_size` 字段，
    #   而它在**训练配置**上不存在 ⇒ 上面的空列表是"删掉了"，不是"扫描坏了"）。
    assert "dp_size" in GraspoFlowConfig.model_fields


def test_max_steps_guard_is_still_intact():
    """★ 本轮**只**否掉"按卡数改步长"，`max_steps` 那条防呆必须原样保留。"""
    with pytest.raises(ValidationError, match="max_steps has been removed"):
        TrainingConfig(max_steps=100)
    # ★ 阳性对照：不传该字段时构造正常（证明报错来自防呆，不是模型坏掉见谁都报错）。
    assert TrainingConfig(max_epochs=2).max_epochs == 2


# ── ② lr 表按方法来（配置生成层）─────────────────────────────────────────────


def test_lr_policy_table_is_one_value_per_method():
    """一个方法一个值，且 SFT ≠ RL（同一模式）。"""
    lr_for = _load_extract_spec().lr_for

    assert lr_for("sft", "lora") == "2.0e-05"
    assert lr_for("sft", "full") == "5.0e-06"
    assert lr_for("graspo", "lora") == "5.0e-06"
    assert lr_for("graspo", "full") == "5.0e-06"
    assert lr_for("cpt", "lora") == "5.0e-06"
    assert lr_for("cpt", "full") == "5.0e-06"
    assert lr_for("opd", "lora") == "2.0e-05"
    assert lr_for("opd", "full") == "5.0e-06"

    # ★ 核心不变量：SFT 的步长要长一点，RL 的要短一点。
    assert lr_for("sft", "lora") != lr_for("graspo", "lora")
    assert float(lr_for("sft", "lora")) > float(lr_for("graspo", "lora"))
    # ★ 如实登记偏差：全参档 SFT == RL（5e-06），⛔ 不硬凑成 10×。
    assert lr_for("sft", "full") == lr_for("graspo", "full")
    # ★ 阳性对照：表确实按**方法**分叉，而不是所有方法一个值（那会让上面全通过）。
    assert len({lr_for(m, "lora") for m in ("sft", "graspo", "cpt", "opd")}) == 2


def test_lr_policy_is_independent_of_card_count():
    """表函数**不接收**卡数：同一方法在同一模式下的取值与 dp/tp/pp 无关。"""
    params = set(inspect.signature(_load_extract_spec().lr_for).parameters)
    assert params == {"train_method", "tuner_type"}
    # ★ 阳性对照：调用签名里若混进任何卡数维度，这条断言会失败。
    assert not (params & {"dp_size", "tp_size", "pp_size", "cards", "world_size"})


def test_lr_policy_fails_closed_on_unknown_combination():
    """未知组合当场报错，⛔ 不静默回退到某个默认值。"""
    spec = _load_extract_spec()
    with pytest.raises(KeyError):
        spec.lr_for("dpo", "lora")
    with pytest.raises(KeyError):
        spec.lr_for("sft", "qlora")
    # ★ 阳性对照：合法组合照常返回值（证明不是表整体不可用）。
    assert spec.lr_for("sft", "lora") == "2.0e-05"
    # `tuner_type=None` 归一为 lora（与 schema.resolve_tuner_type 同语义）。
    assert spec.lr_for("sft", None) == spec.lr_for("sft", "lora")


def test_ruled_out_caliber_is_recorded():
    """「已裁定不做」的口径必须**机读可查**（避免后人再提）。"""
    spec = _load_extract_spec()
    blob = " ".join(spec.RULED_OUT)
    assert "不补 epoch" in blob
    assert "按全局步数定长" in blob
    assert "按卡数补偿训练量" in blob
    # ★ 阳性对照：口径里同时留了原「RL 比 SFT 低一个数量级」的逐字出处与偏差说明。
    assert spec.LEGACY_CALIBER["text"] == "RL 推荐比 SFT 低一个数量级"
    assert spec.LEGACY_CALIBER["sources"]
    assert "4×" in spec.LEGACY_CALIBER["deviation"]
