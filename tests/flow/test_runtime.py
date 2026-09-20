"""Tests for graspo.flow.runtime._AVAILABLE_ADAPTERS.

The graspo package has import-time dependencies on torch, trainer, and other
modules not available in a pure test environment.  We use importlib.util +
sys.modules mocking to load runtime.py directly, bypassing the broken
package __init__.py chain.

This test validates the current behaviour of _AVAILABLE_ADAPTERS — whether
it is a hard-coded tuple of strings (pre-refactoring) or derived from
_discover(...).keys() (post-refactoring).  The assertions are designed to
pass in both states.
"""

from __future__ import annotations

import importlib.util
import inspect
import sys
import types

import pytest


# ---------------------------------------------------------------------------
# Helper: load graspo.flow.runtime directly
# ---------------------------------------------------------------------------


def _load_runtime_module() -> types.ModuleType:
    """Load ``graspo.flow.runtime``, bypassing the broken package init chain.

    Returns the loaded module object.  Cached in ``sys.modules`` so repeated
    calls return the same instance.
    """
    _RUNTIME_KEY = "graspo.flow.runtime"

    if _RUNTIME_KEY in sys.modules:
        return sys.modules[_RUNTIME_KEY]

    # -- torch (not available in this env) ---------------------------------
    if "torch" not in sys.modules:
        _torch = types.ModuleType("torch")
        _torch.__version__ = "2.12.1"
        _torch.Tensor = type("Tensor", (), {})
        sys.modules["torch"] = _torch
        sys.modules["torch.distributed"] = types.ModuleType("torch.distributed")

    # -- graspo.ripple.group_decision (missing on disk) ---------------------------
    if "graspo.ripple.group_decision" not in sys.modules:
        _parity = types.ModuleType("graspo.ripple.group_decision")
        _parity.has_reward_variance = lambda *a: False  # noqa: ARG005
        sys.modules["graspo.ripple.group_decision"] = _parity

    # -- parent packages (need __path__ for sub-package resolution) --------
    for _name in [
        "graspo",
        "graspo.core",
        "graspo.flow",
        "graspo.ripple",
        "graspo.ripple.parsing",
    ]:
        if _name not in sys.modules:
            _pkg = types.ModuleType(_name)
            _pkg.__path__ = []
            sys.modules[_name] = _pkg

    # -- dependency modules that runtime.py imports ------------------------
    if "graspo.core.schema" not in sys.modules:
        _m = types.ModuleType("graspo.core.schema")
        _m.GraspoConfig = type("GraspoConfig", (), {})
        _m.Sample = type("Sample", (), {})
        sys.modules["graspo.core.schema"] = _m

    if "graspo.ripple.buffer" not in sys.modules:
        _m = types.ModuleType("graspo.ripple.buffer")
        _m.Experience = type("Experience", (), {})
        sys.modules["graspo.ripple.buffer"] = _m

    if "graspo.ripple.parsing.completion" not in sys.modules:
        _m = types.ModuleType("graspo.ripple.parsing.completion")
        _m.ParsedCompletion = type("ParsedCompletion", (), {})
        sys.modules["graspo.ripple.parsing.completion"] = _m

    # -- graspo.core.discovery (mock _discover to return known adapters) ---
    if "graspo.core.discovery" not in sys.modules:
        _m = types.ModuleType("graspo.core.discovery")

        def _discover(group: str) -> dict[str, object]:
            if group == "graspo.adapters":
                return {"qwen3": lambda: None, "qwen35_36": lambda: None}
            return {}

        _m._discover = _discover
        sys.modules["graspo.core.discovery"] = _m

    # -- load the real runtime.py ------------------------------------------
    _spec = importlib.util.spec_from_file_location(
        _RUNTIME_KEY,
        "src/graspo/flow/runtime.py",
    )
    _module = importlib.util.module_from_spec(_spec)
    sys.modules[_RUNTIME_KEY] = _module
    _spec.loader.exec_module(_module)
    return _module


# Load once at module level so all tests share the same instance.
_runtime = _load_runtime_module()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

# Snapshot sys.modules before the mock injection so we can restore afterwards.
_SYS_MODULES_SNAPSHOT = dict(sys.modules)


def _cleanup_mocked_modules():
    """Remove any modules that were injected by _load_runtime_module."""
    for key in list(sys.modules):
        if key not in _SYS_MODULES_SNAPSHOT:
            del sys.modules[key]
    # Restore originals that were replaced
    sys.modules.update(_SYS_MODULES_SNAPSHOT)


class TestAvailableAdapters:
    """Tests for ``_AVAILABLE_ADAPTERS`` in ``graspo.flow.runtime``."""

    def test_contains_qwen3(self) -> None:
        """``_AVAILABLE_ADAPTERS`` must include the 'qwen3' adapter."""
        assert "qwen3" in _runtime._AVAILABLE_ADAPTERS, (
            f"Expected 'qwen3' in _AVAILABLE_ADAPTERS, got {_runtime._AVAILABLE_ADAPTERS}"
        )

    def test_contains_qwen35_36(self) -> None:
        """``_AVAILABLE_ADAPTERS`` must include the 'qwen35_36' adapter."""
        assert "qwen35_36" in _runtime._AVAILABLE_ADAPTERS, (
            f"Expected 'qwen35_36' in _AVAILABLE_ADAPTERS, got {_runtime._AVAILABLE_ADAPTERS}"
        )

    def test_is_tuple(self) -> None:
        """``_AVAILABLE_ADAPTERS`` must be a ``tuple`` instance."""
        assert isinstance(_runtime._AVAILABLE_ADAPTERS, tuple), (
            f"Expected tuple, got {type(_runtime._AVAILABLE_ADAPTERS)}"
        )

    def test_error_messages_reference_available_adapters(self) -> None:
        """The ``setup()`` error messages must reference ``_AVAILABLE_ADAPTERS``.

        We verify by inspecting the source code of the ``setup`` method:
        both error branches (missing ':' separator and failed import) must
        use ``_AVAILABLE_ADAPTERS`` in their f-string messages.
        """
        setup_source = inspect.getsource(_runtime.GraspoFlowRuntime.setup)

        # The error messages join _AVAILABLE_ADAPTERS with ', '.
        assert "_AVAILABLE_ADAPTERS" in setup_source, (
            "setup() must reference _AVAILABLE_ADAPTERS in its error messages"
        )

        # The joined string should contain the adapter names.
        adapters_csv = ", ".join(_runtime._AVAILABLE_ADAPTERS)
        assert "qwen3" in adapters_csv
        assert "qwen35_36" in adapters_csv
