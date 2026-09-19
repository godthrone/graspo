"""`_nested_get` 的防呆语义回归（lock-in）与鉴别力对照。

**背景（缺陷①的独立复核结论）**：工作包转述称 ``tests/core/test_schema.py``
的 ``_nested_get`` 返回**值**、``:223`` 用真值判定 ⇒ ``false/null/0/""`` 会被
误判为"模板缺字段"、约 90 项假警报、该测试在 HEAD 上"本就该失败"。

**实测结论：转述与 HEAD 代码不符，①不成立。**

- ``_nested_get``（HEAD ``:203-209``）返回 **bool**（``return False``/``return True``）；
- 缺字段判定用的是**真正的成员判定** ``part not in node``，不是真值判定；
- 该函数自引入提交 ``9b88ea8`` 起就是这个形式；
- 实跑 ``tests/core/test_schema.py`` ⇒ **66 passed**（含模板覆盖那条），``missing=0/167``。

本文件是**回归锁**：把"假值 ≠ 缺失"这一正确语义钉住。若将来有人把
``part not in node`` 改成真值判定，下面的用例会真失败。
"""

from __future__ import annotations

from typing import Any


def _nested_get(mapping: dict, path: str) -> bool:
    """HEAD ``tests/core/test_schema.py::_nested_get`` 的逐字副本。

    故意复制在这里，而不是 import 测试模块——这样本文件的断言不会因为
    pytest 递归收集测试模块而出问题，同时"钉住的语义"在文件内自足。
    """
    node: Any = mapping
    for part in path.split("."):
        if not isinstance(node, dict) or part not in node:
            return False
        node = node[part]
    return True


def _buggy_truthiness_get(mapping: dict, path: str) -> bool:
    """**错误实现**的对照件：按真值判定 ⇒ 把假值误判为缺失。

    仅用于证明下面"假值"用例有鉴别力（会真的抓住这种退化），**不是**生产代码。
    """
    node: Any = mapping
    for part in path.split("."):
        if not isinstance(node, dict) or not node.get(part):
            return False
        node = node[part]
    return True


FALSY_VALUES = [False, None, 0, "", [], {}, 0.0]


class TestNestedGetTreatsFalsyAsPresent:
    def test_present_falsy_values_are_present(self) -> None:
        """★ 核心不变式：字段**存在**但值为假 ⇒ 必须判为"存在"，不是"缺失"。"""
        mapping = {f"k{i}": value for i, value in enumerate(FALSY_VALUES)}
        mapping["nested"] = {"inner": False}
        for key in list(mapping):
            assert _nested_get(mapping, key) is True, f"{key} 存在（值可能为假）⇒ 应判存在"
        assert _nested_get(mapping, "nested.inner") is True

    def test_absent_keys_are_missing(self) -> None:
        mapping = {"a": {"b": None}}
        assert _nested_get(mapping, "a.zz") is False
        assert _nested_get(mapping, "zz") is False
        # 空 dict 本身是"存在"，其下级的键才是"缺失"
        assert _nested_get({"a": {}}, "a") is True
        assert _nested_get({"a": {}}, "a.b") is False

    def test_non_dict_intermediate_is_missing_not_crash(self) -> None:
        """类型边界：中间节点不是 dict ⇒ 判缺失，不得抛异常。"""
        assert _nested_get({"a": 5}, "a.b") is False
        assert _nested_get({"a": None}, "a.b") is False
        assert _nested_get({"a": []}, "a.b") is False

    def test_return_type_is_bool_not_value(self) -> None:
        """★ 钉住签名：返回布尔，不返回字段值（转述的错误来源）。"""
        mapping = {"a": {"b": 0}}
        result = _nested_get(mapping, "a.b")
        assert result is True
        assert isinstance(result, bool)
        assert result is not 0  # `is` 比较：True is not 0，防止"返回值"式退化

    def test_buggy_truthiness_variant_would_report_false_alarms(self) -> None:
        """★ 鉴别力对照：真值判定实现在假值上**确实**产生假警报。

        这条证明"假值被误判"是**可被本文件的用例抓住**的退化——即上面的
        不变式断言不是空转：换成真值实现后，同样的假值会由 True 变 False。
        """
        mapping = {"a": {"flag": False, "none": None, "zero": 0, "empty": ""}}
        for key in ("a.flag", "a.none", "a.zero", "a.empty"):
            assert _nested_get(mapping, key) is True, "正确实现：假值=存在"
            assert _buggy_truthiness_get(mapping, key) is False, "错误实现：假值=缺失"
        # 两者在"真值存在"与"缺失"上必须一致，差异只出现在假值上。
        assert _nested_get({"a": 1}, "a") is _buggy_truthiness_get({"a": 1}, "a") is True
        assert _nested_get({}, "a") is _buggy_truthiness_get({}, "a") is False


class TestSchemaTemplateCoverageStillHolds:
    """把"模板 0 缺字段"这一当前事实钉住（不放松判据：仍要求全覆盖）。"""

    def test_config_example_reports_no_missing_fields(self) -> None:
        import importlib.util
        from pathlib import Path

        import yaml

        from graspo.core.schema import GraspoConfig

        # tests/ 不是 package，按文件路径加载同目录的生产测试模块，
        # 直接复用它的 `_field_paths` / `_nested_get`（单一真相源，不抄第二份）。
        spec = importlib.util.spec_from_file_location(
            "_schema_guard_under_test", Path(__file__).with_name("test_schema.py")
        )
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        example = yaml.safe_load(
            Path("samples/configs/config_example.yaml").read_text(encoding="utf-8")
        )
        paths = sorted(module._field_paths(GraspoConfig))
        assert len(paths) == 167, f"schema 字段路径数变了（{len(paths)}），请复核本断言"
        missing = [path for path in paths if not module._nested_get(example, path)]
        assert missing == [], f"config_example.yaml 缺 schema 字段: {missing}"
