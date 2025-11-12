from __future__ import annotations

import unittest

from db_slm.adapters.cheetah import CheetahClient, CheetahHotPathAdapter, CheetahSerializer


class FakeCheetahClient:
    def __init__(self) -> None:
        self.storage: dict[int, bytes] = {}
        self.pairs: dict[bytes, int] = {}
        self._next_key = 1

    def insert(self, payload: bytes) -> tuple[int, str | None]:
        key = self._next_key
        self._next_key += 1
        self.storage[key] = payload
        return key, None

    def edit(self, key: int, payload: bytes) -> bool:
        self.storage[key] = payload
        return True

    def read(self, key: int) -> bytes | None:
        return self.storage.get(key)

    def pair_set(self, value: bytes, key: int) -> tuple[bool, str | None]:
        self.pairs[value] = key
        return True, None

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

    def test_probability_payload_round_trip(self) -> None:
        entries = [(7, 255, None), (8, 120, 400)]
        payload = self.serializer.encode_probabilities(3, entries)
        record = self.serializer.decode_probabilities(payload)
        self.assertIsNotNone(record)
        assert record is not None
        self.assertEqual(record.order, 3)
        self.assertEqual(record.entries, tuple(entries))

    def test_continuation_payload_round_trip(self) -> None:
        payload = self.serializer.encode_continuation(42, 1024)
        record = self.serializer.decode_continuation(payload)
        self.assertIsNotNone(record)
        assert record is not None
        self.assertEqual(record.token_id, 42)
        self.assertEqual(record.num_contexts, 1024)


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


class CheetahClientParsingTests(unittest.TestCase):
    def test_parse_pair_reduce_response_with_payload(self) -> None:
        response = "SUCCESS,reducer=counts,count=1,items=636e743a:42:SGVsbG8="
        parsed, cursor = CheetahClient._parse_pair_reduce_response(response)
        self.assertEqual(len(parsed), 1)
        self.assertIsNone(cursor)
        value, key, payload = parsed[0]
        self.assertEqual(value, bytes.fromhex("636e743a"))
        self.assertEqual(key, 42)
        self.assertEqual(payload, b"Hello")

    def test_parse_pair_scan_response_with_cursor(self) -> None:
        response = "SUCCESS,count=1,next_cursor=x616263,items=74657374:10"
        parsed, cursor = CheetahClient._parse_pair_scan_response(response)
        self.assertEqual(cursor, b"abc")
        self.assertEqual(parsed, [(b"test", 10)])


if __name__ == "__main__":
    unittest.main()
