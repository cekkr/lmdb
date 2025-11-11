from __future__ import annotations

from typing import Protocol, Sequence


class HotPathAdapter(Protocol):
    """Interface for optional low-latency mirrors such as cheetah-mldb."""

    def publish_context(self, context_hash: str, order_size: int, token_ids: Sequence[int]) -> None:
        """Persist context metadata so trie-style traversals avoid SQL lookups."""

    def publish_topk(self, order: int, context_hash: str, ranked: Sequence[tuple[int, int]]) -> None:
        """Store the ranked (token_id, q_logprob) list for a context."""

    def fetch_topk(self, order: int, context_hash: str, limit: int) -> list[tuple[int, int]] | None:
        """Return cached ranked results or None when unavailable."""


class NullHotPathAdapter:
    """Default adapter that keeps the SQLite-only behavior."""

    def publish_context(self, context_hash: str, order_size: int, token_ids: Sequence[int]) -> None:
        return None

    def publish_topk(self, order: int, context_hash: str, ranked: Sequence[tuple[int, int]]) -> None:
        return None

    def fetch_topk(self, order: int, context_hash: str, limit: int) -> list[tuple[int, int]] | None:
        return None


__all__ = ["HotPathAdapter", "NullHotPathAdapter"]
