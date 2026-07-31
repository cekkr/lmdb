from __future__ import annotations

import unittest
from types import SimpleNamespace

from db_slm.level1 import MKNSmoother, NGramStore


class _HotPath:
    def __init__(self) -> None:
        self.projections = [
            SimpleNamespace(context_hash="same", followers=((1, 2),)),
            SimpleNamespace(context_hash="stale", followers=((2, 1),)),
        ]
        self.published: list[tuple[int, str, tuple[tuple[int, int], ...]]] = []
        self.published_batches: list[
            tuple[int, tuple[tuple[str, tuple[tuple[int, int], ...]], ...]]
        ] = []
        self.flushes = 0

    def iter_counts(self, order: int):
        return list(self.projections)

    def publish_counts(self, order: int, context_hash: str, followers):
        self.published.append((order, context_hash, tuple(followers)))

    def publish_counts_batch(self, order: int, entries):
        self.published_batches.append(
            (
                order,
                tuple(
                    (context_hash, tuple(followers))
                    for context_hash, followers in entries
                ),
            )
        )

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
            smoother.store.hot_path.published_batches,
            [
                (
                    3,
                    (
                        ("stale", ((2, 4),)),
                        ("new", ((3, 1),)),
                    ),
                ),
            ],
        )
        self.assertEqual(smoother.store.hot_path.published, [])
        self.assertEqual(smoother.store.hot_path.flushes, 1)


class _BatchMirrorHotPath:
    def __init__(self) -> None:
        self.context_batches = []
        self.topk_batches = []
        self.probability_batches = []

    def publish_contexts(self, entries) -> None:
        self.context_batches.append(list(entries))

    def publish_context(self, context_hash, order_size, token_ids) -> None:
        raise AssertionError("ingest should preserve the grouped context batch")

    def publish_topk_batch(self, order, entries) -> None:
        self.topk_batches.append((order, list(entries)))

    def publish_topk(self, order, context_hash, ranked) -> None:
        raise AssertionError("Top-K synchronization should use the batch surface")

    def publish_probabilities(self, order, context_hash, entries) -> None:
        raise AssertionError("probability synchronization should use the batch surface")

    def publish_probabilities_batch(self, order, entries) -> None:
        self.probability_batches.append((order, list(entries)))


class _IngestDatabase:
    def __init__(self) -> None:
        self.statements = 0

    def hash_tokens(self, tokens) -> str:
        values = tuple(tokens)
        return "__root__" if not values else "".join(f"{value:08x}" for value in values)

    def execute(self, statement, params=()) -> None:
        self.statements += 1


class _Vocabulary:
    def __init__(self) -> None:
        self.frequencies = []

    def increment_frequency(self, token_id: int) -> None:
        self.frequencies.append(token_id)


class Level1BatchPublicationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.hot_path = _BatchMirrorHotPath()
        self.store = NGramStore.__new__(NGramStore)
        self.store.db = _IngestDatabase()
        self.store.vocab = _Vocabulary()
        self.store.order = 2
        self.store.hot_path = self.hot_path

    def test_ingest_publishes_distinct_contexts_as_one_batch(self) -> None:
        self.store.ingest([1, 2, 3])

        self.assertEqual(len(self.hot_path.context_batches), 1)
        self.assertEqual(
            self.hot_path.context_batches[0],
            [
                ("__root__", 0, ()),
                ("00000001", 1, (1,)),
                ("00000002", 1, (2,)),
            ],
        )

    def test_probability_materialization_uses_grouped_batch_methods(self) -> None:
        topk_rows = [
            ("a", 0, 10, 200),
            ("a", 1, 11, 180),
            ("b", 0, 12, 170),
        ]
        probability_rows = [
            ("a", 10, 200, None),
            ("a", 11, 180, 400),
            ("b", 12, 170, None),
        ]

        self.store._sync_topk(2, topk_rows)
        self.store._sync_probabilities(2, probability_rows)

        self.assertEqual(
            self.hot_path.topk_batches,
            [(2, [("a", [(10, 200), (11, 180)]), ("b", [(12, 170)])])],
        )
        self.assertEqual(
            self.hot_path.probability_batches,
            [
                (
                    2,
                    [
                        ("a", [(10, 200, None), (11, 180, 400)]),
                        ("b", [(12, 170, None)]),
                    ],
                )
            ],
        )


if __name__ == "__main__":
    unittest.main()
