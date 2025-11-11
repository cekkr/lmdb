from __future__ import annotations

from typing import Iterable, Protocol, Sequence, Tuple


class HotPathAdapter(Protocol):
    """Interface for optional low-latency mirrors such as cheetah-db."""

    def publish_context(self, context_hash: str, order_size: int, token_ids: Sequence[int]) -> None:
        """Persist context metadata so trie-style traversals avoid SQL lookups."""

    def publish_topk(self, order: int, context_hash: str, ranked: Sequence[tuple[int, int]]) -> None:
        """Store the ranked (token_id, q_logprob) list for a context."""

    def fetch_topk(self, order: int, context_hash: str, limit: int) -> list[tuple[int, int]] | None:
        """Return cached ranked results or None when unavailable."""

    def scan_namespace(
        self,
        namespace: str,
        *,
        prefix: bytes = b"",
        limit: int = 0,
    ) -> Iterable[Tuple[bytes, int]]:
        """Iterate namespace-prefixed keys (e.g., contexts, cached slices) in byte order."""


class NullHotPathAdapter:
    """Default adapter that keeps the SQLite-only behavior."""

    def publish_context(self, context_hash: str, order_size: int, token_ids: Sequence[int]) -> None:
        return None

    def publish_topk(self, order: int, context_hash: str, ranked: Sequence[tuple[int, int]]) -> None:
        return None

    def fetch_topk(self, order: int, context_hash: str, limit: int) -> list[tuple[int, int]] | None:
        return None

    def scan_namespace(
        self,
        namespace: str,
        *,
        prefix: bytes = b"",
        limit: int = 0,
    ) -> Iterable[Tuple[bytes, int]]:
        return []


__all__ = ["HotPathAdapter", "NullHotPathAdapter"]
