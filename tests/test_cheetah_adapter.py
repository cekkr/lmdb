from __future__ import annotations

import base64
import random
import socket
import unittest

from db_slm.adapters.cheetah import CheetahClient, CheetahHotPathAdapter, CheetahSerializer
from db_slm.pipeline import TaggedResponseFormatter, _strip_prompt_scaffolding


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

    def test_topk_payload_is_fixed_length(self) -> None:
        expected = 3 + self.serializer.MAX_TOPK * 5
        ranked_short = [(5, 200)]
        ranked_full = [(i, 255) for i in range(self.serializer.MAX_TOPK)]
        payload_short = self.serializer.encode_topk(2, ranked_short)
        payload_empty = self.serializer.encode_topk(2, [])
        payload_full = self.serializer.encode_topk(2, ranked_full)
        self.assertEqual(len(payload_short), expected)
        self.assertEqual(len(payload_empty), expected)
        self.assertEqual(len(payload_full), expected)

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

    def test_iter_counts_decodes_reducer_storage_transport(self) -> None:
        payload = CheetahSerializer().encode_counts(1, [(7, 3), (8, 2)])

        def pair_reduce(
            mode: str,
            prefix: bytes,
            *,
            limit: int,
            cursor: bytes | None,
        ) -> tuple[list[tuple[bytes, int, bytes | None]], bytes | None]:
            self.assertEqual((mode, prefix), ("counts", b"cnt:1:"))
            return [(b"cnt:1:__root__", 42, base64.b64encode(payload))], None

        self.client.pair_reduce = pair_reduce  # type: ignore[attr-defined]
        self.adapter._recommended_reduce_page_size = lambda: 256  # type: ignore[method-assign]
        projections = self.adapter.iter_counts(1)
        self.assertEqual(len(projections), 1)
        self.assertEqual(projections[0].context_hash, "__root__")
        self.assertEqual(projections[0].totals, 5)
        self.assertEqual(projections[0].followers, ((7, 3), (8, 2)))


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

    def test_decode_reduced_payload_removes_insert_transport_layer(self) -> None:
        original = b"\x01\x02fixed-size-counts"
        stored = base64.b64encode(original)
        self.assertEqual(CheetahClient.decode_reduced_payload(stored), original)
        self.assertIsNone(CheetahClient.decode_reduced_payload(b"not base64!"))

    def test_parse_pair_scan_response_with_cursor(self) -> None:
        response = "SUCCESS,count=1,next_cursor=x616263,items=74657374:10"
        parsed, cursor = CheetahClient._parse_pair_scan_response(response)
        self.assertEqual(cursor, b"abc")
        self.assertEqual(parsed, [(b"test", 10)])

    def test_readline_recovers_after_socket_timeouts(self) -> None:
        client = CheetahClient("127.0.0.1", 0, timeout=0.01)

        class FlakySocket:
            def __init__(self) -> None:
                self._timeouts = 3
                self._payload = b"SUCCESS\n"
                self._offset = 0

            def recv(self, size: int) -> bytes:
                if self._timeouts > 0:
                    self._timeouts -= 1
                    raise socket.timeout()
                if self._offset >= len(self._payload):
                    return b""
                chunk = self._payload[self._offset : self._offset + size]
                self._offset += size
                return chunk

            def close(self) -> None:  # pragma: no cover - not triggered in this test
                pass

        client._sock = FlakySocket()  # type: ignore[assignment]
        client._readline_idle_grace = 0.5  # make the test deterministic
        self.assertEqual(client._readline(), "SUCCESS")

    def test_pair_reduce_uses_canonical_job_status_then_fetch(self) -> None:
        client = CheetahClient("127.0.0.1", 0)
        responses = iter(
            [
                "SUCCESS,job=reduce_1,kind=reduce,command=PAIR_REDUCE,state=queued,total=0,reducer=counts",
                "SUCCESS,job=reduce_1,kind=reduce,state=completed,progress=100.00,completed=1,total=1,reducer=counts",
                "SUCCESS,job=reduce_1,reducer=counts,count=1,items=636e743a:42:SGVsbG8=",
            ]
        )
        commands: list[str] = []

        def scripted(command: str) -> str:
            commands.append(command)
            return next(responses)

        client._command = scripted  # type: ignore[method-assign]
        result = client.pair_reduce("counts", b"cnt:", limit=256)
        self.assertIsNotNone(result)
        assert result is not None
        self.assertEqual(result[0], [(b"cnt:", 42, b"Hello")])
        self.assertTrue(commands[0].startswith("JOB submit command="))
        self.assertEqual(commands[1:], ["JOB status id=reduce_1", "JOB fetch id=reduce_1"])
        self.assertTrue(client._job_api)

    def test_pair_reduce_falls_back_to_legacy_alias_without_job_api(self) -> None:
        client = CheetahClient("127.0.0.1", 0)
        responses = iter(
            [
                "ERROR,unknown_command",
                "SUCCESS,reducer=counts,job=reduce_1,state=queued",
                "SUCCESS,reducer=counts,count=1,items=636e743a:42:SGVsbG8=",
            ]
        )
        commands: list[str] = []

        def scripted(command: str) -> str:
            commands.append(command)
            return next(responses)

        client._command = scripted  # type: ignore[method-assign]
        result = client.pair_reduce("counts", b"cnt:", limit=256)
        self.assertIsNotNone(result)
        self.assertTrue(commands[0].startswith("JOB submit command="))
        self.assertTrue(commands[1].startswith("PAIR_REDUCE_ASYNC counts "))
        self.assertEqual(commands[2], "PAIR_REDUCE_FETCH reduce_1")
        self.assertFalse(client._job_api)


class TaggedResponseFormatterTests(unittest.TestCase):
    def test_framed_prompt_is_not_nested_inside_another_user_tag(self) -> None:
        formatter = TaggedResponseFormatter()
        prompt = "|USER|: How can curiosity help with an ethical dilemma?\n|RESPONSE|: "
        rendered = formatter.wrap(
            prompt,
            "It encourages gathering evidence before choosing.",
            rng=random.Random(7),
        )
        self.assertEqual(rendered.count("|USER|:"), 1)
        self.assertEqual(rendered.count("|RESPONSE|:"), 1)
        self.assertNotIn("|USER|: |USER|:", rendered)
        self.assertNotIn("|, :", rendered)

    def test_strip_prompt_scaffolding_keeps_only_content(self) -> None:
        prompt = "|INSTRUCTION|: Be concise.\n|USER|: Explain curiosity.\n|RESPONSE|: "
        self.assertEqual(
            _strip_prompt_scaffolding(prompt),
            "Be concise. Explain curiosity.",
        )


if __name__ == "__main__":
    unittest.main()
