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


__all__ = [
    "RawContextProjection",
    "RawCountsProjection",
    "RawProbabilityProjection",
    "RawContinuationProjection",
]
