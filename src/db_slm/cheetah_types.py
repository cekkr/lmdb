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


__all__ = ["RawContextProjection", "RawCountsProjection"]
