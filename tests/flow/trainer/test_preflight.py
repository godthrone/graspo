"""预检纯逻辑单测（不触 GPU）。

覆盖:
- assert_data_vision_compatible: 数据含图 + 模型无视觉 → raise；纯文本 → 通过
- run_multimodal_preflight 的纯文本短路路径（无 GPU 依赖部分）
"""

import pytest

from graspo.flow.trainer.preflight import (
    assert_data_vision_compatible,
    run_multimodal_preflight,
)


class _Sample:
    def __init__(self, media: list[dict] | None = None) -> None:
        self.media = media or []


class TestDataVisionCompatible:
    def test_media_data_without_vision_model_raises(self) -> None:
        samples = [_Sample(media=[{"type": "image"}])]
        with pytest.raises(RuntimeError, match="does not support vision"):
            assert_data_vision_compatible(
                samples, model_supports_vision=False, model_name="Qwen3-8B"
            )

    def test_media_data_with_vision_model_passes(self) -> None:
        samples = [_Sample(media=[{"type": "image"}])]
        assert_data_vision_compatible(samples, model_supports_vision=True, model_name="Qwen3.5-9B")

    def test_text_data_without_vision_model_passes(self) -> None:
        samples = [_Sample()]
        assert_data_vision_compatible(samples, model_supports_vision=False, model_name="Qwen3-8B")

    def test_empty_samples_passes(self) -> None:
        assert_data_vision_compatible([], model_supports_vision=False, model_name="Qwen3-8B")


class _FakeRuntime:
    """最小 runtime 鸭子类型：预检短路路径（无 vision）不触碰 adapter。"""

    def _require_adapter(self):  # noqa: ANN001
        raise AssertionError("should not be reached without vision")


class TestPreflightShortCircuit:
    def test_text_only_model_skips_facility_checks(self) -> None:
        # image_token_id=None（纯文本模型）→ 不触 GPU 设施
        run_multimodal_preflight(
            _FakeRuntime(),
            [_Sample()],
            data_dir="/tmp",
            image_token_id=None,
            model_name="Qwen3-8B",
        )

    def test_text_only_data_skips_facility_checks(self) -> None:
        # 模型有视觉但数据纯文本 → 不触 GPU 设施
        run_multimodal_preflight(
            _FakeRuntime(),
            [_Sample()],
            data_dir="/tmp",
            image_token_id=151655,
            model_name="Qwen3.5-9B",
        )


class TestPpMultimodalGuard:
    def test_media_data_with_pp_gt_1_raises(self) -> None:
        """多模态 + PP>1 是未实现路径——启动即拒绝（不跑到 generate 才炸）。"""
        with pytest.raises(RuntimeError, match="pp_size=2 is not supported"):
            run_multimodal_preflight(
                _FakeRuntime(),
                [_Sample(media=[{"type": "image"}])],
                data_dir="/tmp",
                image_token_id=151655,
                model_name="Qwen3.5-9B",
                pp_size=2,
            )

    def test_media_data_with_pp_1_passes_guard(self) -> None:
        """PP=1 时防线不拦截（继续走设施检查，由后续链路抛错）。"""
        # _FakeRuntime 无 adapter → 抛其固有断言，而不是 pp 防线错误
        with pytest.raises(AssertionError, match="should not be reached"):
            run_multimodal_preflight(
                _FakeRuntime(),
                [_Sample(media=[{"type": "image"}])],
                data_dir="/tmp",
                image_token_id=151655,
                model_name="Qwen3.5-9B",
                pp_size=1,
            )
