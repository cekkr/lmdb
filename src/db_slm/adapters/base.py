from __future__ import annotations

from typing import Iterable, Protocol, Sequence, Tuple

from ..cheetah_types import (
    RawContinuationProjection,
    RawContextProjection,
    RawCountsProjection,
    RawProbabilityProjection,
)


class HotPathAdapter(Protocol):
    """Interface for optional low-latency mirrors such as cheetah-db."""

    def publish_context(self, context_hash: str, order_size: int, token_ids: Sequence[int]) -> None:
        """Persist context metadata so trie-style traversals avoid SQL lookups."""

    def publish_topk(self, order: int, context_hash: str, ranked: Sequence[tuple[int, int]]) -> None:
        """Store the ranked (token_id, q_logprob) list for a context."""

    def fetch_topk(self, order: int, context_hash: str, limit: int) -> list[tuple[int, int]] | None:
        """Return cached ranked results or None when unavailable."""

    def publish_counts(self, order: int, context_hash: str, followers: Sequence[tuple[int, int]]) -> None:
        """Mirror raw follower counts for a context hash."""

    def publish_probabilities(
        self,
        order: int,
        context_hash: str,
        entries: Sequence[tuple[int, int, int | None]],
    ) -> None:
        """Mirror quantized probability rows per context."""

    def publish_continuations(self, entries: Sequence[tuple[int, int]]) -> None:
        """Mirror continuation metadata (token id -> num contexts)."""

    def fetch_context_tokens(self, context_hash: str) -> Sequence[int] | None:
        """Return the token ids representing the context, if mirrored."""

    def write_metadata(self, key: str, value: str) -> None:
        """Persist metadata (e.g., context dimensions, decode profiles) inside cheetah."""

    def read_metadata(self, key: str) -> str | None:
        """Fetch metadata stored inside cheetah namespaces."""

    def scan_namespace(
        self,
        namespace: str,
        *,
        prefix: bytes = b"",
        limit: int = 0,
    ) -> Iterable[Tuple[bytes, int]]:
        """Iterate namespace-prefixed keys (e.g., contexts, cached slices) in byte order."""

    def iter_counts(self, order: int) -> Iterable[RawCountsProjection]:
        """Iterate mirrored follower counts for an order."""

    def iter_probabilities(self, order: int) -> Iterable[RawProbabilityProjection]:
        """Iterate mirrored probability/backoff entries for an order."""

    def iter_continuations(self) -> Iterable[RawContinuationProjection]:
        """Iterate mirrored continuation metadata."""

    def context_relativism(
        self,
        context_tree,
        *,
        limit: int = 32,
        depth: int | None = None,
    ) -> Iterable[RawContextProjection]:
        """Return probabilistic projections for a nested context description."""

    def topk_hit_ratio(self) -> float:
        """Return the observed cheetah Top-K cache hit ratio (0-1)."""


class NullHotPathAdapter:
    """Default adapter that keeps the SQLite-only behavior."""

    def publish_context(self, context_hash: str, order_size: int, token_ids: Sequence[int]) -> None:
        return None

    def publish_topk(self, order: int, context_hash: str, ranked: Sequence[tuple[int, int]]) -> None:
        return None

    def fetch_topk(self, order: int, context_hash: str, limit: int) -> list[tuple[int, int]] | None:
        return None

    def publish_counts(self, order: int, context_hash: str, followers: Sequence[tuple[int, int]]) -> None:
        return None

    def publish_probabilities(
        self,
        order: int,
        context_hash: str,
        entries: Sequence[tuple[int, int, int | None]],
    ) -> None:
        return None

    def publish_continuations(self, entries: Sequence[tuple[int, int]]) -> None:
        return None

    def fetch_context_tokens(self, context_hash: str) -> Sequence[int] | None:
        return None

    def write_metadata(self, key: str, value: str) -> None:
        return None

    def read_metadata(self, key: str) -> str | None:
        return None

    def scan_namespace(
        self,
        namespace: str,
        *,
        prefix: bytes = b"",
        limit: int = 0,
    ) -> Iterable[Tuple[bytes, int]]:
        return []

    def iter_counts(self, order: int) -> Iterable[RawCountsProjection]:
        return []

    def iter_probabilities(self, order: int) -> Iterable[RawProbabilityProjection]:
        return []

    def iter_continuations(self) -> Iterable[RawContinuationProjection]:
        return []

    def context_relativism(
        self,
        context_tree,
        *,
        limit: int = 32,
        depth: int | None = None,
    ) -> Iterable[RawContextProjection]:
        return []

    def topk_hit_ratio(self) -> float:
        return 0.0


__all__ = ["HotPathAdapter", "NullHotPathAdapter"]
