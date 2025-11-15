import builtins
import os
import time
from typing import Any

_start_time = time.perf_counter()
_LOG_LEVEL_ENV = "LMDB_LOG_LEVEL"


def _read_log_level() -> int:
    raw = os.getenv(_LOG_LEVEL_ENV, "1").strip()
    try:
        return max(0, int(raw))
    except ValueError:
        normalized = raw.lower()
        if normalized in {"debug", "trace"}:
            return 3
        if normalized in {"info", "warning", "warn"}:
            return 1
        return 1


_LOG_LEVEL = _read_log_level()


def _elapsed() -> float:
    return time.perf_counter() - _start_time


def timestamp_prefix() -> str:
    return f"+[{_elapsed():7.2f}]"


def reset_timestamp() -> None:
    global _start_time
    _start_time = time.perf_counter()


def log(*objects: Any, sep: str = " ", end: str = "\n", file=None, flush: bool = False, prefix: bool = True) -> None:
    message = sep.join(str(obj) for obj in objects)
    if prefix:
        message = f"{timestamp_prefix()} {message}"
    builtins.print(message, end=end, file=file, flush=flush)


def verbose_enabled(level: int) -> bool:
    return _LOG_LEVEL >= level


def log_verbose(level: int, *objects: Any, **kwargs: Any) -> None:
    """Emit a log line only when the configured verbosity is high enough."""
    if verbose_enabled(level):
        log(*objects, **kwargs)
