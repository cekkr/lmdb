from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class RawContextProjection:
    """Minimal projection returned by cheetah context scans."""

    context_hash: str
    order_size: int
    token_ids: Tuple[int, ...]
    ranked: Tuple[tuple[int, int], ...]
    cheetah_key: int
    vector_signature: bytes


@dataclass(frozen=True)
class RawCountsProjection:
    """Follower counts stored inside cheetah namespaces."""

    context_hash: str
    order: int
    totals: int
    followers: Tuple[tuple[int, int], ...]


@dataclass(frozen=True)
class RawProbabilityProjection:
    """Quantized probability rows mirrored from MKNS."""

    context_hash: str
    order: int
    followers: Tuple[tuple[int, int, int | None], ...]


@dataclass(frozen=True)
class RawContinuationProjection:
    """Continuation metadata mirrored into cheetah."""

    token_id: int
    num_contexts: int


@dataclass(frozen=True)
class NamespaceSummary:
    """Aggregate statistics for a namespace prefix."""

    prefix: bytes
    terminal_count: int
    total_payload_bytes: int
    min_payload_bytes: int
    max_payload_bytes: int
    min_key: int | None
    max_key: int | None
    max_depth: int
    self_terminal: bool
    branches: Tuple[tuple[bytes, int], ...]


@dataclass(frozen=True)
class CheetahSystemStats:
    """Snapshot of the cheetah-db resource monitor."""

    logical_cores: int
    gomaxprocs: int
    goroutines: int
    mem_alloc_bytes: int
    mem_sys_bytes: int
    process_cpu_pct: float | None
    system_cpu_pct: float | None
    process_cpu_supported: bool
    system_cpu_supported: bool
    io_supported: bool
    io_read_bytes_per_sec: float | None
    io_write_bytes_per_sec: float | None
    timestamp: str | None
    recommended_workers: Tuple[tuple[int, int], ...]


__all__ = [
    "RawContextProjection",
    "RawCountsProjection",
    "RawProbabilityProjection",
    "RawContinuationProjection",
    "NamespaceSummary",
    "CheetahSystemStats",
]
