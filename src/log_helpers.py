import builtins
import time
from typing import Any

_start_time = time.perf_counter()


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
