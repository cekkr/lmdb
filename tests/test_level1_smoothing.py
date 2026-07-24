from __future__ import annotations

import unittest
from types import SimpleNamespace

from db_slm.level1 import MKNSmoother


class _HotPath:
    def __init__(self) -> None:
        self.projections = [
            SimpleNamespace(context_hash="same", followers=((1, 2),)),
            SimpleNamespace(context_hash="stale", followers=((2, 1),)),
        ]
        self.published: list[tuple[int, str, tuple[tuple[int, int], ...]]] = []
        self.flushes = 0

    def iter_counts(self, order: int):
        return list(self.projections)

    def publish_counts(self, order: int, context_hash: str, followers):
        self.published.append((order, context_hash, tuple(followers)))

    def flush_pending(self) -> None:
        self.flushes += 1


class _Store:
    def __init__(self) -> None:
        self.hot_path = _HotPath()

    def iter_counts(self, order: int):
        return [
            {"context_hash": "same", "next_token_id": 1, "count": 2},
            {"context_hash": "stale", "next_token_id": 2, "count": 4},
            {"context_hash": "new", "next_token_id": 3, "count": 1},
        ]


class SmoothingCountSourceTests(unittest.TestCase):
    def test_sqlite_ingest_counts_override_stale_hot_path_and_mirror_deltas(self) -> None:
        smoother = MKNSmoother.__new__(MKNSmoother)
        smoother.store = _Store()

        contexts, sourced_from_sqlite = smoother._collect_context_followers(3)
        smoother._mirror_counts(3, contexts)

        self.assertTrue(sourced_from_sqlite)
        self.assertEqual(contexts["stale"], [(2, 4)])
        self.assertEqual(
            smoother.store.hot_path.published,
            [
                (3, "stale", ((2, 4),)),
                (3, "new", ((3, 1),)),
            ],
        )
        self.assertEqual(smoother.store.hot_path.flushes, 1)


if __name__ == "__main__":
    unittest.main()
