"""Dashboard file logger.

Creates ripe_autotrain_<PID>.log in the configured log directory (default /tmp).
Import `log_error` and `log_debug` anywhere in the dashboard to write timestamped
entries with caller context.

Usage:
    from ripe_autotrain.dashboard_log import log_error, log_debug
    log_error("Submit failed", exc=e, gpu_id=3)
"""

from __future__ import annotations

import inspect
import logging
import os
import traceback
from pathlib import Path

_logger: logging.Logger | None = None


def init(log_dir: Path | str = "/tmp") -> None:
    """Call once at dashboard startup. Subsequent calls are no-ops."""
    global _logger
    if _logger is not None:
        return
    log_path = Path(log_dir) / f"ripe_autotrain_{os.getpid()}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    handler = logging.FileHandler(log_path, encoding="utf-8")
    handler.setFormatter(logging.Formatter(
        "%(asctime)s  %(levelname)-7s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    ))
    _logger = logging.getLogger("ripe_autotrain")
    _logger.setLevel(logging.DEBUG)
    _logger.addHandler(handler)
    _logger.propagate = False
    _logger.info(f"Log started — pid={os.getpid()}  file={log_path}")


def _caller() -> str:
    """Return 'module:lineno' of the caller two frames up."""
    frame = inspect.stack()[2]
    return f"{Path(frame.filename).name}:{frame.lineno}"


def log_error(msg: str, exc: BaseException | None = None, **ctx) -> None:
    """Log an error with optional exception traceback and key=value context."""
    if _logger is None:
        init()
    parts = [msg]
    if ctx:
        parts.append("  " + "  ".join(f"{k}={v!r}" for k, v in ctx.items()))
    _logger.error("%s  [%s]", "  ".join(parts), _caller())  # type: ignore[union-attr]
    if exc is not None:
        _logger.error("  traceback:\n%s", "".join(traceback.format_exception(exc)))


def log_debug(msg: str, **ctx) -> None:
    """Log a debug message with key=value context."""
    if _logger is None:
        init()
    parts = [msg]
    if ctx:
        parts.append("  " + "  ".join(f"{k}={v!r}" for k, v in ctx.items()))
    _logger.debug("%s  [%s]", "  ".join(parts), _caller())  # type: ignore[union-attr]
