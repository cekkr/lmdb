from __future__ import annotations

import unittest

from db_slm.adapters.cheetah import CheetahHotPathAdapter, CheetahSerializer


class FakeCheetahClient:
    def __init__(self) -> None:
        self.storage: dict[int, bytes] = {}
        self.pairs: dict[bytes, int] = {}
        self._next_key = 1

    def insert(self, payload: bytes) -> int:
        key = self._next_key
        self._next_key += 1
        self.storage[key] = payload
        return key

    def edit(self, key: int, payload: bytes) -> bool:
        self.storage[key] = payload
        return True

    def read(self, key: int) -> bytes | None:
        return self.storage.get(key)

    def pair_set(self, value: bytes, key: int) -> bool:
        self.pairs[value] = key
        return True

    def pair_get(self, value: bytes) -> int | None:
        return self.pairs.get(value)


class CheetahSerializerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.serializer = CheetahSerializer()

    def test_context_payload_encodes_tokens(self) -> None:
        payload = self.serializer.encode_context(3, [10, 20, 30])
        self.assertEqual(payload[0], CheetahSerializer.CONTEXT_VERSION)
        self.assertEqual(payload[1], 3)
        self.assertEqual(payload[2], 3)
        self.assertEqual(len(payload), 3 + 3 * 4)

    def test_topk_payload_round_trip(self) -> None:
        ranked = [(5, 120), (99, 80)]
        payload = self.serializer.encode_topk(2, ranked)
        record = self.serializer.decode_topk(payload)
        self.assertIsNotNone(record)
        assert record is not None
        self.assertEqual(record.order, 2)
        self.assertEqual(record.ranked, ranked)


class CheetahHotPathAdapterTests(unittest.TestCase):
    def setUp(self) -> None:
        self.client = FakeCheetahClient()
        self.adapter = CheetahHotPathAdapter(self.client)

    def test_publish_context_is_idempotent(self) -> None:
        context_hash = "deadbeefcafebabe"
        self.adapter.publish_context(context_hash, 2, [11, 12])
        first_pairs = dict(self.client.pairs)
        self.adapter.publish_context(context_hash, 2, [11, 12])
        self.assertEqual(first_pairs, self.client.pairs)

    def test_publish_and_fetch_topk(self) -> None:
        context_hash = "facefeedbabe0001"
        ranked = [(7, 200), (8, 180), (9, 160)]
        self.adapter.publish_context(context_hash, 2, [1, 2])
        self.adapter.publish_topk(3, context_hash, ranked)
        cached = self.adapter.fetch_topk(3, context_hash, 2)
        self.assertEqual(cached, ranked[:2])


if __name__ == "__main__":
    unittest.main()
