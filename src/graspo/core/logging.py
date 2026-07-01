"""Standard Python ``logging`` channel for GRASPO.

Every project must provide at least one
standard ``logging`` channel using the four canonical levels
(DEBUG / INFO / WARNING / ERROR).  This module satisfies that requirement
while the existing ``NativeRolloutLogger`` continues to serve as the
domain-specific structured-JSONL extension.

Usage::

    from graspo.core.logging import setup_logging
    setup_logging(output_dir, rank=rank)
    # After setup, standard logging calls work:
    import logging
    logging.getLogger("graspo.trainer").info("Epoch %d finished", epoch)

``setup_logging`` is idempotent — repeated calls with the same *output_dir*
are harmless.
"""

import logging
import sys
from pathlib import Path

_SETUP_DONE: set[str] = set()
"""Track which output directories already have file handlers attached."""


def setup_logging(output_dir: str | Path, *, rank: int = 0) -> None:
    """Configure the ``graspo`` root logger.

    Parameters
    ----------
    output_dir:
        Training output directory.  A ``FileHandler`` writing to
        ``{output_dir}/logs/training.log`` is attached for DEBUG-level
        post-hoc analysis.
    rank:
        Distributed rank.  The file handler is only attached on rank 0;
        console output follows the same policy.
    """
    root = logging.getLogger("graspo")
    root.setLevel(logging.DEBUG)

    log_path = str(Path(output_dir).resolve())

    # ── Console handler (every rank, INFO+) ──────────────────────────
    if not _has_handler(root, logging.StreamHandler):
        console = logging.StreamHandler(sys.stdout)
        console.setLevel(logging.INFO)
        console.setFormatter(logging.Formatter("%(message)s"))
        root.addHandler(console)

    # ── File handler (rank 0 only, DEBUG+) ───────────────────────────
    if rank == 0 and log_path not in _SETUP_DONE:
        file_dir = Path(output_dir) / "logs"
        file_dir.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(file_dir / "training.log", encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(
            logging.Formatter(
                "%(asctime)s [%(levelname)-5s] %(name)s: %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
        root.addHandler(file_handler)
        _SETUP_DONE.add(log_path)


def _has_handler(logger: logging.Logger, handler_type: type) -> bool:
    return any(isinstance(h, handler_type) for h in logger.handlers)
