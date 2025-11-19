from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

CHEETAH_DEFAULT_REDUCE_PAGE_SIZE = 1024
CHEETAH_PAIR_SCAN_MIN_LIMIT = 256
CHEETAH_PAIR_SCAN_MAX_LIMIT = 4096


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
class PredictionValueResult:
    """Single value returned from a cheetah prediction table query."""

    value: bytes
    probability: float


@dataclass(frozen=True)
class PredictionQueryResult:
    """Result metadata returned by `PREDICT_QUERY`."""

    table: str
    backend: str
    count: int
    entries: Tuple[PredictionValueResult, ...]


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
    payload_cache_enabled: bool
    payload_cache_entries: int
    payload_cache_max_entries: int
    payload_cache_bytes: int
    payload_cache_max_bytes: int
    payload_cache_hits: int
    payload_cache_misses: int
    payload_cache_evictions: int
    payload_cache_hit_pct: float | None
    payload_cache_advisory_bypass_bytes: int | None

    def derive_reduce_page_limit(
        self,
        *,
        default_limit: int = CHEETAH_DEFAULT_REDUCE_PAGE_SIZE,
        min_limit: int = CHEETAH_PAIR_SCAN_MIN_LIMIT,
        max_limit: int = CHEETAH_PAIR_SCAN_MAX_LIMIT,
        target_pending: int = CHEETAH_PAIR_SCAN_MAX_LIMIT,
    ) -> int | None:
        """Return a batch size tuned to the current worker hints."""
        if not self.recommended_workers:
            return None
        hints = {pending: workers for pending, workers in self.recommended_workers}
        if target_pending not in hints:
            # Fall back to the largest pending bucket advertised by the server.
            candidate = max(hints.keys(), default=0)
            if candidate == 0:
                return None
            target_pending = candidate
        workers = hints.get(target_pending, 0)
        gomax = max(self.gomaxprocs, 1)
        if workers <= 0 or gomax <= 0:
            return None
        ratio = min(1.0, max(0.0, workers / float(gomax)))
        adaptive_max = min(max_limit, max(default_limit, gomax * 256))
        bounded_min = max(min_limit, default_limit // 4)
        window = adaptive_max - bounded_min
        limit = bounded_min + int(window * ratio)
        if limit < bounded_min:
            limit = bounded_min
        if limit > adaptive_max:
            limit = adaptive_max
        # Align to payload pages to avoid thrashing on odd sizes.
        if limit > 0:
            limit = max(bounded_min, (limit // 64) * 64)
        return limit


__all__ = [
    "RawContextProjection",
    "RawCountsProjection",
    "RawProbabilityProjection",
    "RawContinuationProjection",
    "PredictionValueResult",
    "PredictionQueryResult",
    "NamespaceSummary",
    "CheetahSystemStats",
    "CHEETAH_DEFAULT_REDUCE_PAGE_SIZE",
    "CHEETAH_PAIR_SCAN_MIN_LIMIT",
    "CHEETAH_PAIR_SCAN_MAX_LIMIT",
]
