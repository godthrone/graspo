"""Standard Python ``logging`` channel for GRASPO + shared log-file helpers.

Every project must provide at least one standard ``logging`` channel using the
four canonical levels (DEBUG / INFO / WARNING / ERROR).  This module satisfies
that requirement while the existing ``NativeRolloutLogger`` continues to serve
as the domain-specific structured-JSONL extension.

Log-file layout
---------------
The log folder identity comes from **config**, not from the process environment
(§7.1/§10.1).  ``cli/train_worker.main`` binds ``config.training.run_name`` via
:func:`set_run_id`, and because every rank loads the same YAML, all ranks share
one ``run_id``.  All logs for that launch are written under
``{output_dir}/logs/<run_id>/{name}``, giving a fresh, self-contained folder per
launch (no more mixing old and new logs across restarts).  The python-logging
files (``training.log`` /
``error.log``) rotate at 10 MB via ``RotatingFileHandler`` (separate numbered
files); the domain JSONL files rotate via :func:`append_jsonl_segment`.

Usage::

    from graspo.flow.logging import setup_logging
    setup_logging(output_dir, rank=rank)
    # After setup, standard logging calls work:
    import logging
    logging.getLogger("graspo.trainer").info("Epoch %d finished", epoch)
"""

import datetime
import json
import logging
import logging.handlers
import sys
from pathlib import Path
from typing import Any

MAX_LOG_BYTES = 10 * 1024 * 1024  # 10 MB
"""Per-log-file size cap before rotating to a new numbered file."""

_SETUP_DONE: set[str] = set()
"""Track which run-log directories already have file handlers attached."""

_run_id: str | None = None


def set_run_id(run_id: str) -> None:
    """Bind this process's log identity to a **config-derived** value (§1.4).

    Called once by the worker entry point (``cli/train_worker.main``) with
    ``config.training.run_name`` — that field is the config's own, git-tracked
    identifier for the run and is therefore identical on every rank (each rank
    loads the same YAML).  Binding it here means the log location has exactly
    one source: the config.

    Raises:
        ValueError: empty value (防呆 §2.3 —— 空身份会静默退回环境变量/时间戳，
            让"配置不生效"表现得和成功一样).
    """
    global _run_id
    text = str(run_id or "").strip()
    if not text:
        raise ValueError("set_run_id() 需要非空的 run id（不得为空字符串）")
    _run_id = text


def get_run_id() -> str:
    """Return the launch-scoped run id shared by all ranks in this process.

    **Resolution order (single authority, §1.4):**

    1. :func:`set_run_id` — the config-derived value bound by the worker entry
       point (``training.run_name``).  This is the normal path.
    2. Process-local timestamp — standalone/unit-test use.

    §10.1: the log folder is part of the run's on-disk output, so its identity
    must come from config, not from the process environment.  The transitional
    ``GRASPO_RUN_ID`` environment variable that used to be honoured here was
    removed outright — a second source for an artifact path is a dual source of
    truth (§1.4), and the deprecation window it was kept for has closed (§18.1).
    """
    global _run_id
    if _run_id is not None:
        return _run_id
    _run_id = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    return _run_id


def run_log_id(run_name: str) -> str:
    """Derive the **rank-consistent** log-folder id from a config run name (§1.4).

    Why not use ``run_name`` verbatim: ``cli/train_worker.main`` runs once per rank,
    and each rank loads the YAML in its own process.  ``schema.training.run_name``
    keeps the auto-generated form ``graspo_<YYYYmmdd_HHMMSS>`` (mirroring the
    pre-existing ``output_dir`` derivation), so the folder name must be the
    **timestamp part**, which is identical on every rank because every rank reads
    the same config file.  Using the raw ``run_name`` would also work for
    consistency but would drop the ``YYYYMMDD-HHMMSS`` folder shape that existing
    tooling greps for.

    ``run_name`` is never empty at this point (``TrainingConfig._validate_output_dir``
    derives it from ``output_dir`` when unset), so the last-resort fallback below is
    unreachable in practice and only guards a hand-constructed config object.
    """
    text = str(run_name or "").strip()
    if text.startswith("graspo_"):
        stamp = text[len("graspo_") :]
        if len(stamp) == 15 and stamp[8] == "_":
            return f"{stamp[:8]}-{stamp[9:]}"
    return text or datetime.datetime.now().strftime("%Y%m%d-%H%M%S")


def run_log_dir(output_dir: str | Path) -> Path:
    """Return ``{output_dir}/logs/<run_id>/`` (created): one folder per launch."""
    directory = Path(output_dir) / "logs" / get_run_id()
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def rotating_append(path: Path, line: str) -> Path:
    """Append ``line``, rotating to a new numbered file past ``MAX_LOG_BYTES``.

    ``events.jsonl`` becomes ``events.jsonl.1``, ``.2`` ... once a segment
    exceeds 10 MB.  Returns the path that was written.
    """
    seg = 0
    active = path
    nums: list[int] = []
    for candidate in path.parent.glob(f"{path.name}.*"):
        suffix = candidate.name.rsplit(".", 1)[1]
        try:
            nums.append(int(suffix))
        except ValueError:
            continue
    if nums:
        seg = max(nums)
        active = path.with_name(f"{path.name}.{seg}")
    if active.exists() and active.stat().st_size >= MAX_LOG_BYTES:
        seg += 1
        active = path.with_name(f"{path.name}.{seg}")
    with active.open("a", encoding="utf-8") as handle:
        handle.write(line)
    return active


def append_jsonl_segment(path: Path, payload: dict[str, Any]) -> Path:
    """Append one JSON line (rotating at 10 MB).  Returns the path written."""
    return rotating_append(path, json.dumps(payload, ensure_ascii=False) + "\n")


def setup_logging(output_dir: str | Path, *, rank: int = 0) -> None:
    """Configure the ``graspo`` root logger (per-launch run-log folder).

    Parameters
    ----------
    output_dir:
        Training output directory.  File handlers are attached to
        ``{output_dir}/logs/<run_id>/training.log`` (DEBUG+) and
        ``{output_dir}/logs/<run_id>/error.log`` (ERROR+) on rank 0.  Each
        restart uses a new ``<run_id>`` folder; files rotate at 10 MB.
    rank:
        Distributed rank.  The file handler is only attached on rank 0;
        console output follows the same policy.
    """
    root = logging.getLogger("graspo")
    root.setLevel(logging.DEBUG)

    if not _has_handler(root, logging.StreamHandler):
        console = logging.StreamHandler(sys.stdout)
        console.setLevel(logging.INFO)
        console.setFormatter(logging.Formatter("%(message)s"))
        root.addHandler(console)

    if rank == 0:
        log_dir = run_log_dir(output_dir)
        if str(log_dir) not in _SETUP_DONE:
            file_handler = logging.handlers.RotatingFileHandler(
                log_dir / "training.log",
                maxBytes=MAX_LOG_BYTES,
                backupCount=5,
                encoding="utf-8",
            )
            file_handler.setLevel(logging.DEBUG)
            file_handler.setFormatter(
                logging.Formatter(
                    "%(asctime)s [%(levelname)-5s] %(name)s: %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S",
                )
            )
            root.addHandler(file_handler)
            error_handler = logging.handlers.RotatingFileHandler(
                log_dir / "error.log",
                maxBytes=MAX_LOG_BYTES,
                backupCount=5,
                encoding="utf-8",
            )
            error_handler.setLevel(logging.ERROR)
            error_handler.setFormatter(
                logging.Formatter(
                    "%(asctime)s [%(levelname)-5s] %(name)s: %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S",
                )
            )
            root.addHandler(error_handler)
            _SETUP_DONE.add(str(log_dir))


def _has_handler(logger: logging.Logger, handler_type: type) -> bool:
    return any(isinstance(h, handler_type) for h in logger.handlers)
