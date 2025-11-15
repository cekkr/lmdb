from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Any, Optional, Tuple

import sys

try:  # pragma: no cover - psutil is optional at runtime.
    import psutil  # type: ignore
except Exception:  # pragma: no cover - we gracefully degrade without psutil.
    psutil = None  # type: ignore

try:  # pragma: no cover - resource is POSIX-only.
    import resource  # type: ignore
except Exception:  # pragma: no cover - Windows fallback.
    resource = None  # type: ignore


LoadAverage = Tuple[float, float, float]


@dataclass(frozen=True)
class ResourceSample:
    """Captures the absolute process/system metrics at a point in time."""

    taken_at: float
    wall_time: float
    rss_mb: float | None
    memory_percent: float | None
    cpu_user: float | None
    cpu_system: float | None
    cpu_total: float | None
    thread_count: int | None
    load_avg: LoadAverage | None
    io_read_bytes: float | None
    io_write_bytes: float | None
    io_read_ops: float | None
    io_write_ops: float | None


@dataclass(frozen=True)
class ResourceDelta:
    """Summarizes how the metrics changed between two samples."""

    duration_sec: float
    cpu_percent: float | None
    cpu_user_delta: float | None
    cpu_system_delta: float | None
    rss_after_mb: float | None
    rss_delta_mb: float | None
    memory_percent: float | None
    thread_count: int | None
    load_avg: LoadAverage | None
    io_read_mb: float | None
    io_write_mb: float | None
    io_read_rate_mb_s: float | None
    io_write_rate_mb_s: float | None
    io_read_ops: float | None
    io_write_ops: float | None


class ResourceMonitor:
    """Lightweight process/system telemetry collector used for profiling."""

    _KB = 1024
    _MB = 1024 * 1024

    def __init__(self) -> None:
        self._cpu_count = psutil.cpu_count(logical=True) if psutil else os.cpu_count()
        if not self._cpu_count or self._cpu_count <= 0:
            self._cpu_count = 1
        self._process: Optional[Any] = None
        if psutil:
            try:
                self._process = psutil.Process(os.getpid())
            except Exception:  # pragma: no cover - psutil failure fallback.
                self._process = None

    def snapshot(self) -> ResourceSample:
        now = time.perf_counter()
        wall = time.time()
        load_avg = self._load_average()
        if self._process:
            try:
                with self._process.oneshot():
                    mem_info = self._process.memory_info()
                    rss_mb = mem_info.rss / self._MB
                    memory_percent = self._process.memory_percent()
                    cpu_times = self._process.cpu_times()
                    thread_count = self._process.num_threads()
                    io_read_bytes: float | None = None
                    io_write_bytes: float | None = None
                    io_read_ops: float | None = None
                    io_write_ops: float | None = None
                    try:
                        io_counters = self._process.io_counters()
                        read_bytes_raw = getattr(io_counters, "read_bytes", None)
                        if read_bytes_raw is not None:
                            io_read_bytes = float(read_bytes_raw)
                        write_bytes_raw = getattr(io_counters, "write_bytes", None)
                        if write_bytes_raw is not None:
                            io_write_bytes = float(write_bytes_raw)
                        read_count_raw = getattr(io_counters, "read_count", None)
                        if read_count_raw is not None:
                            io_read_ops = float(read_count_raw)
                        write_count_raw = getattr(io_counters, "write_count", None)
                        if write_count_raw is not None:
                            io_write_ops = float(write_count_raw)
                    except Exception:
                        pass
                return ResourceSample(
                    taken_at=now,
                    wall_time=wall,
                    rss_mb=float(rss_mb),
                    memory_percent=float(memory_percent),
                    cpu_user=float(cpu_times.user),
                    cpu_system=float(cpu_times.system),
                    cpu_total=float(cpu_times.user + cpu_times.system),
                    thread_count=int(thread_count),
                    load_avg=load_avg,
                    io_read_bytes=io_read_bytes,
                    io_write_bytes=io_write_bytes,
                    io_read_ops=io_read_ops,
                    io_write_ops=io_write_ops,
                )
            except Exception:
                # Fall back to the POSIX resource module below.
                pass
        rss_mb: float | None = None
        cpu_user: float | None = None
        cpu_system: float | None = None
        io_read_ops: float | None = None
        io_write_ops: float | None = None
        if resource:
            usage = resource.getrusage(resource.RUSAGE_SELF)
            rss_raw = usage.ru_maxrss
            if sys.platform == "darwin":  # type: ignore[name-defined]
                rss_mb = float(rss_raw / self._MB)
            else:
                rss_mb = float(rss_raw / self._KB)
            cpu_user = float(usage.ru_utime)
            cpu_system = float(usage.ru_stime)
            io_read_ops = float(usage.ru_inblock)
            io_write_ops = float(usage.ru_oublock)
        return ResourceSample(
            taken_at=now,
            wall_time=wall,
            rss_mb=rss_mb,
            memory_percent=None,
            cpu_user=cpu_user,
            cpu_system=cpu_system,
            cpu_total=(cpu_user + cpu_system) if cpu_user is not None and cpu_system is not None else None,
            thread_count=None,
            load_avg=load_avg,
            io_read_bytes=None,
            io_write_bytes=None,
            io_read_ops=io_read_ops,
            io_write_ops=io_write_ops,
        )

    def delta(self, before: ResourceSample, after: ResourceSample) -> ResourceDelta:
        duration = max(0.0, after.taken_at - before.taken_at)
        cpu_percent: float | None = None
        cpu_user_delta: float | None = None
        cpu_system_delta: float | None = None
        if (
            duration > 0
            and before.cpu_total is not None
            and after.cpu_total is not None
        ):
            cpu_user_delta = (
                after.cpu_user - before.cpu_user
                if after.cpu_user is not None and before.cpu_user is not None
                else None
            )
            cpu_system_delta = (
                after.cpu_system - before.cpu_system
                if after.cpu_system is not None and before.cpu_system is not None
                else None
            )
            total_delta = after.cpu_total - before.cpu_total
            cpu_percent = max(0.0, (total_delta / duration) * 100.0 / float(self._cpu_count))
        rss_delta: float | None = None
        if before.rss_mb is not None and after.rss_mb is not None:
            rss_delta = after.rss_mb - before.rss_mb
        io_read_mb: float | None = None
        io_write_mb: float | None = None
        io_read_rate: float | None = None
        io_write_rate: float | None = None
        io_read_ops: float | None = None
        io_write_ops: float | None = None
        if before.io_read_bytes is not None and after.io_read_bytes is not None:
            read_bytes = max(0.0, after.io_read_bytes - before.io_read_bytes)
            io_read_mb = read_bytes / self._MB
            if duration > 0:
                io_read_rate = io_read_mb / duration
        if before.io_write_bytes is not None and after.io_write_bytes is not None:
            write_bytes = max(0.0, after.io_write_bytes - before.io_write_bytes)
            io_write_mb = write_bytes / self._MB
            if duration > 0:
                io_write_rate = io_write_mb / duration
        if before.io_read_ops is not None and after.io_read_ops is not None:
            io_read_ops = max(0.0, after.io_read_ops - before.io_read_ops)
        if before.io_write_ops is not None and after.io_write_ops is not None:
            io_write_ops = max(0.0, after.io_write_ops - before.io_write_ops)
        return ResourceDelta(
            duration_sec=duration,
            cpu_percent=cpu_percent,
            cpu_user_delta=cpu_user_delta,
            cpu_system_delta=cpu_system_delta,
            rss_after_mb=after.rss_mb,
            rss_delta_mb=rss_delta,
            memory_percent=after.memory_percent,
            thread_count=after.thread_count,
            load_avg=after.load_avg,
            io_read_mb=io_read_mb,
            io_write_mb=io_write_mb,
            io_read_rate_mb_s=io_read_rate,
            io_write_rate_mb_s=io_write_rate,
            io_read_ops=io_read_ops,
            io_write_ops=io_write_ops,
        )

    def describe(self, delta: ResourceDelta) -> str:
        """Return a short, human-friendly summary string."""
        parts: list[str] = []
        if delta.cpu_percent is not None:
            parts.append(f"cpu={delta.cpu_percent:.1f}%/{self._cpu_count}c")
        if delta.rss_after_mb is not None:
            if delta.rss_delta_mb is not None:
                parts.append(f"rss={delta.rss_after_mb:.1f}MB(Δ{delta.rss_delta_mb:+.1f})")
            else:
                parts.append(f"rss={delta.rss_after_mb:.1f}MB")
        if delta.memory_percent is not None:
            parts.append(f"mem%={delta.memory_percent:.1f}")
        if delta.io_read_mb is not None or delta.io_write_mb is not None:
            read_str = (
                f"{delta.io_read_mb:.2f}MB"
                if delta.io_read_mb is not None
                else f"{delta.io_read_ops:.0f}ops"
                if delta.io_read_ops is not None
                else "n/a"
            )
            write_str = (
                f"{delta.io_write_mb:.2f}MB"
                if delta.io_write_mb is not None
                else f"{delta.io_write_ops:.0f}ops"
                if delta.io_write_ops is not None
                else "n/a"
            )
            parts.append(f"io[R/W]={read_str}/{write_str}")
        elif delta.io_read_ops is not None or delta.io_write_ops is not None:
            read_ops = f"{delta.io_read_ops:.0f}" if delta.io_read_ops is not None else "n/a"
            write_ops = f"{delta.io_write_ops:.0f}" if delta.io_write_ops is not None else "n/a"
            parts.append(f"io_ops[R/W]={read_ops}/{write_ops}")
        if delta.thread_count is not None:
            parts.append(f"threads={delta.thread_count}")
        if delta.load_avg is not None:
            parts.append("load=" + ",".join(f"{value:.2f}" for value in delta.load_avg))
        return " ".join(parts)

    @staticmethod
    def to_event(delta: ResourceDelta) -> dict[str, float | int | str]:
        """Serialize the delta into a JSON-friendly payload."""
        payload: dict[str, float | int | str] = {
            "duration_sec": round(delta.duration_sec, 3),
        }
        if delta.cpu_percent is not None:
            payload["cpu_percent"] = round(delta.cpu_percent, 3)
        if delta.cpu_user_delta is not None:
            payload["cpu_user_delta"] = round(delta.cpu_user_delta, 4)
        if delta.cpu_system_delta is not None:
            payload["cpu_system_delta"] = round(delta.cpu_system_delta, 4)
        if delta.rss_after_mb is not None:
            payload["rss_after_mb"] = round(delta.rss_after_mb, 3)
        if delta.rss_delta_mb is not None:
            payload["rss_delta_mb"] = round(delta.rss_delta_mb, 3)
        if delta.memory_percent is not None:
            payload["memory_percent"] = round(delta.memory_percent, 3)
        if delta.thread_count is not None:
            payload["thread_count"] = int(delta.thread_count)
        if delta.load_avg is not None:
            payload["load_avg"] = tuple(round(value, 3) for value in delta.load_avg)
        if delta.io_read_mb is not None:
            payload["io_read_mb"] = round(delta.io_read_mb, 4)
        if delta.io_write_mb is not None:
            payload["io_write_mb"] = round(delta.io_write_mb, 4)
        if delta.io_read_rate_mb_s is not None:
            payload["io_read_rate_mb_s"] = round(delta.io_read_rate_mb_s, 4)
        if delta.io_write_rate_mb_s is not None:
            payload["io_write_rate_mb_s"] = round(delta.io_write_rate_mb_s, 4)
        if delta.io_read_ops is not None:
            payload["io_read_ops"] = round(delta.io_read_ops, 3)
        if delta.io_write_ops is not None:
            payload["io_write_ops"] = round(delta.io_write_ops, 3)
        return payload

    @staticmethod
    def _load_average() -> LoadAverage | None:
        try:
            load = os.getloadavg()
        except (AttributeError, OSError):
            return None
        return float(load[0]), float(load[1]), float(load[2])
