"""Tests for ``graspo.backends.graspoflow.trainer.checkpoint`` — CheckpointMixin."""


def test_checkpoint_module_is_importable():
    """Verify the checkpoint mixin module is importable."""
    from graspo.backends.graspoflow.trainer.checkpoint import CheckpointMixin  # noqa: F401
