from __future__ import annotations

from typing import Iterable, Protocol, Sequence, Tuple

from ..cheetah_types import (
    CheetahSystemStats,
    NamespaceSummary,
    PredictionQueryResult,
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

    def describe(self) -> str:
        """Return a human-readable description of the adapter for logging."""

    def namespace_summary(
        self,
        prefix: bytes,
        *,
        depth: int = 1,
        branch_limit: int = 32,
    ) -> NamespaceSummary | None:
        """Return aggregate stats for a namespace prefix (PAIR_SUMMARY)."""

    def system_stats(self) -> CheetahSystemStats | None:
        """Return the latest SYSTEM_STATS snapshot, when available."""

    def predict_query(
        self,
        *,
        key: bytes | str | None = None,
        keys: Sequence[bytes | str] | None = None,
        context_matrix: Sequence[Sequence[float]] | None = None,
        windows: Sequence[Sequence[float]] | None = None,
        key_windows: Sequence[tuple[bytes | str, Sequence[Sequence[float]]]] | None = None,
        merge_mode: str | None = None,
        table: str | None = None,
    ) -> PredictionQueryResult | None:
        """Query cheetah prediction tables (PREDICT_QUERY)."""

    def predict_set(
        self,
        *,
        key: bytes | str,
        value: bytes | str,
        probability: float = 0.5,
        table: str | None = None,
        weights: Sequence[dict[str, object]] | None = None,
    ) -> bool:
        """Seed or update a prediction entry (PREDICT_SET)."""

    def predict_train(
        self,
        *,
        key: bytes | str,
        target: bytes | str,
        context_matrix: Sequence[Sequence[float]] | None,
        learning_rate: float = 0.01,
        table: str | None = None,
        negatives: Sequence[bytes | str] | None = None,
    ) -> bool:
        """Adjust prediction weights using the provided context matrix (PREDICT_TRAIN)."""


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

    def describe(self) -> str:
        return "hot-path:disabled"

    def namespace_summary(
        self,
        prefix: bytes,
        *,
        depth: int = 1,
        branch_limit: int = 32,
    ) -> NamespaceSummary | None:
        return None

    def system_stats(self) -> CheetahSystemStats | None:
        return None

    def predict_query(
        self,
        *,
        key: bytes | str | None = None,
        keys: Sequence[bytes | str] | None = None,
        context_matrix: Sequence[Sequence[float]] | None = None,
        windows: Sequence[Sequence[float]] | None = None,
        key_windows: Sequence[tuple[bytes | str, Sequence[Sequence[float]]]] | None = None,
        merge_mode: str | None = None,
        table: str | None = None,
    ) -> PredictionQueryResult | None:
        return None

    def predict_set(
        self,
        *,
        key: bytes | str,
        value: bytes | str,
        probability: float = 0.5,
        table: str | None = None,
        weights: Sequence[dict[str, object]] | None = None,
    ) -> bool:
        return False

    def predict_train(
        self,
        *,
        key: bytes | str,
        target: bytes | str,
        context_matrix: Sequence[Sequence[float]] | None,
        learning_rate: float = 0.01,
        table: str | None = None,
        negatives: Sequence[bytes | str] | None = None,
    ) -> bool:
        return False


__all__ = ["HotPathAdapter", "NullHotPathAdapter"]
