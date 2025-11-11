from __future__ import annotations

import base64
import logging
import socket
import struct
import threading
from collections import OrderedDict
from dataclasses import dataclass
from typing import Iterable, Sequence

from ..cheetah_types import RawContextProjection, RawCountsProjection
from ..cheetah_vectors import AbsoluteVectorOrder
from ..hashing import hash_tokens

from ..settings import DBSLMSettings
from .base import HotPathAdapter, NullHotPathAdapter

logger = logging.getLogger(__name__)


class CheetahError(RuntimeError):
    """Raised when the cheetah-db bridge encounters a fatal error."""


class CheetahClient:
    """Minimal TCP client for cheetah-db's newline-delimited protocol."""

    def __init__(
        self,
        host: str,
        port: int,
        *,
        database: str = "default",
        timeout: float = 1.0,
    ) -> None:
        self.host = host
        self.port = port
        self.database = database
        self.timeout = timeout
        self._sock: socket.socket | None = None
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ #
    # Public helpers
    # ------------------------------------------------------------------ #
    def healthy(self) -> bool:
        return self._sock is not None

    def connect(self) -> bool:
        with self._lock:
            return self._ensure_connection()

    def close(self) -> None:
        with self._lock:
            self._close_socket()

    def insert(self, payload: bytes) -> int | None:
        encoded = self._encode_value(payload)
        response = self._command(f"INSERT:{len(encoded)} {encoded}")
        return self._parse_key_response(response)

    def edit(self, key: int, payload: bytes) -> bool:
        encoded = self._encode_value(payload)
        response = self._command(f"EDIT {key} {encoded}")
        return response is not None and response.startswith("SUCCESS")

    def read(self, key: int) -> bytes | None:
        response = self._command(f"READ {key}")
        if not response or not response.startswith("SUCCESS"):
            return None
        parts = response.split(",")
        for part in parts:
            if part.startswith("value="):
                raw = part.split("=", 1)[1]
                return self._decode_value(raw)
        return None

    def pair_set(self, value: bytes, key: int) -> bool:
        response = self._command(f"PAIR_SET x{value.hex()} {key}")
        return response is not None and response.startswith("SUCCESS")

    def pair_get(self, value: bytes) -> int | None:
        response = self._command(f"PAIR_GET x{value.hex()}")
        return self._parse_key_response(response)

    def pair_scan(self, prefix: bytes = b"", limit: int = 0) -> list[tuple[bytes, int]] | None:
        arg = "*" if not prefix else f"x{prefix.hex()}"
        command = f"PAIR_SCAN {arg}"
        if limit > 0:
            command = f"{command} {limit}"
        response = self._command(command)
        if not response or not response.startswith("SUCCESS"):
            return None
        return self._parse_pair_scan_response(response)

    def pair_reduce(self, mode: str, prefix: bytes = b"", limit: int = 0) -> list[tuple[bytes, int]] | None:
        arg = "*" if not prefix else f"x{prefix.hex()}"
        command = f"PAIR_REDUCE {mode} {arg}"
        if limit != 0:
            command = f"{command} {limit}"
        response = self._command(command)
        if not response or not response.startswith("SUCCESS"):
            return None
        return self._parse_pair_scan_response(response)

    # ------------------------------------------------------------------ #
    # Low-level protocol management
    # ------------------------------------------------------------------ #
    def _encode_value(self, payload: bytes) -> str:
        return base64.b64encode(payload).decode("ascii")

    def _decode_value(self, encoded: str) -> bytes:
        return base64.b64decode(encoded.encode("ascii"))

    def _command(self, text: str) -> str | None:
        line = (text.strip() + "\n").encode("utf-8")
        with self._lock:
            if not self._ensure_connection():
                return None
            assert self._sock is not None
            try:
                self._sock.sendall(line)
                return self._readline()
            except OSError as exc:
                logger.debug("cheetah command failed (%s), reconnecting...", exc)
                self._close_socket()
                if not self._ensure_connection():
                    return None
                self._sock.sendall(line)
                return self._readline()

    def _ensure_connection(self) -> bool:
        if self._sock:
            return True
        try:
            sock = socket.create_connection((self.host, self.port), self.timeout)
            sock.settimeout(self.timeout)
            self._sock = sock
        except OSError as exc:
            logger.debug("Unable to reach cheetah-db at %s:%s (%s)", self.host, self.port, exc)
            self._sock = None
            return False
        if self.database and self.database != "default":
            response = self._command_unlocked(f"DATABASE {self.database}")
            if not response or not response.startswith("SUCCESS"):
                logger.debug("Failed to switch cheetah database: %s", response)
                self._close_socket()
                return False
        return True

    def _command_unlocked(self, text: str) -> str | None:
        line = (text.strip() + "\n").encode("utf-8")
        if not self._sock:
            return None
        try:
            self._sock.sendall(line)
            return self._readline()
        except OSError as exc:
            logger.debug("cheetah command failed (%s) before lock acquisition", exc)
            self._close_socket()
            return None

    def _readline(self) -> str | None:
        if not self._sock:
            return None
        chunks: list[bytes] = []
        while True:
            try:
                data = self._sock.recv(1)
            except socket.timeout:
                return None
            if not data:
                self._close_socket()
                return None
            if data == b"\n":
                break
            if data != b"\r":
                chunks.append(data)
        return b"".join(chunks).decode("utf-8", "replace")

    def _close_socket(self) -> None:
        if self._sock:
            try:
                self._sock.close()
            except OSError:
                pass
            self._sock = None

    @staticmethod
    def _parse_key_response(response: str | None) -> int | None:
        if not response or not response.startswith("SUCCESS"):
            return None
        parts = response.split(",")
        for part in parts:
            if part.startswith("key="):
                try:
                    return int(part.split("=", 1)[1])
                except ValueError:
                    return None
        return None

    @staticmethod
    def _parse_pair_scan_response(response: str) -> list[tuple[bytes, int]]:
        if not response.startswith("SUCCESS"):
            return []
        payload_start = response.find("items=")
        if payload_start == -1:
            return []
        payload = response[payload_start + len("items=") :]
        entries: list[tuple[bytes, int]] = []
        for item in payload.split(";"):
            if not item:
                continue
            try:
                value_hex, key_text = item.rsplit(":", 1)
                value = bytes.fromhex(value_hex)
                key = int(key_text)
            except (ValueError, TypeError):
                continue
            entries.append((value, key))
        return entries


@dataclass(frozen=True)
class TopKPayload:
    order: int
    ranked: list[tuple[int, int]]


@dataclass(frozen=True)
class ContextPayload:
    order_size: int
    token_ids: tuple[int, ...]


@dataclass(frozen=True)
class CountsPayload:
    order: int
    followers: tuple[tuple[int, int], ...]


class CheetahSerializer:
    """Binary codec for the cheetah hot-path payloads."""

    CONTEXT_VERSION = 1
    TOPK_VERSION = 1
    COUNTS_VERSION = 1
    MAX_TOPK = 32

    def encode_context(self, order_size: int, token_ids: Sequence[int]) -> bytes:
        if order_size > 255:
            raise CheetahError("order_size exceeds single-byte limit")
        if len(token_ids) > 254:
            raise CheetahError("token sequence too long for cheetah payload")
        buf = bytearray()
        buf.append(self.CONTEXT_VERSION)
        buf.append(order_size)
        buf.append(len(token_ids))
        for token_id in token_ids:
            buf.extend(struct.pack(">I", int(token_id)))
        return bytes(buf)

    def decode_context(self, payload: bytes) -> ContextPayload | None:
        if not payload or payload[0] != self.CONTEXT_VERSION or len(payload) < 3:
            return None
        order_size = payload[1]
        count = payload[2]
        expected = 3 + count * 4
        if len(payload) < expected:
            return None
        token_ids: list[int] = []
        offset = 3
        for _ in range(count):
            token_ids.append(struct.unpack(">I", payload[offset : offset + 4])[0])
            offset += 4
        return ContextPayload(order_size=order_size, token_ids=tuple(token_ids))

    def encode_counts(self, order: int, followers: Sequence[tuple[int, int]]) -> bytes:
        if order > 255:
            raise CheetahError("order exceeds single-byte limit for counts payload")
        follower_count = min(len(followers), 65535)
        buf = bytearray()
        buf.append(self.COUNTS_VERSION)
        buf.append(order & 0xFF)
        buf.extend(struct.pack(">H", follower_count))
        for token_id, count in followers[:follower_count]:
            buf.extend(struct.pack(">I", int(token_id)))
            buf.extend(struct.pack(">I", max(int(count), 0)))
        return bytes(buf)

    def decode_counts(self, payload: bytes) -> CountsPayload | None:
        if not payload or payload[0] != self.COUNTS_VERSION or len(payload) < 4:
            return None
        order = payload[1]
        follower_count = struct.unpack(">H", payload[2:4])[0]
        expected = 4 + follower_count * 8
        if len(payload) < expected:
            return None
        followers: list[tuple[int, int]] = []
        offset = 4
        for _ in range(follower_count):
            token_id = struct.unpack(">I", payload[offset : offset + 4])[0]
            offset += 4
            count = struct.unpack(">I", payload[offset : offset + 4])[0]
            offset += 4
            followers.append((token_id, count))
        return CountsPayload(order=order, followers=tuple(followers))

    def encode_topk(self, order: int, ranked: Sequence[tuple[int, int]]) -> bytes:
        if order > 255:
            raise CheetahError("order exceeds single-byte limit")
        buf = bytearray()
        buf.append(self.TOPK_VERSION)
        buf.append(order)
        clamped = list(ranked[: self.MAX_TOPK])
        buf.append(len(clamped))
        for token_id, q in clamped:
            buf.extend(struct.pack(">I", int(token_id)))
            buf.append(int(q) & 0xFF)
        return bytes(buf)

    def decode_topk(self, payload: bytes) -> TopKPayload | None:
        if not payload or payload[0] != self.TOPK_VERSION or len(payload) < 3:
            return None
        order = payload[1]
        count = payload[2]
        expected = 3 + count * 5
        if len(payload) < expected:
            return None
        ranked: list[tuple[int, int]] = []
        offset = 3
        for _ in range(count):
            token_id = struct.unpack(">I", payload[offset : offset + 4])[0]
            offset += 4
            q = payload[offset]
            offset += 1
            ranked.append((token_id, q))
        return TopKPayload(order=order, ranked=ranked)


class CheetahHotPathAdapter(HotPathAdapter):
    """Mirrors context metadata + Top-K slices into cheetah-db for low-latency reads."""

    def __init__(
        self,
        client: CheetahClient,
        *,
        cache_size: int = 50000,
        serializer: CheetahSerializer | None = None,
    ) -> None:
        self._client = client
        self._serializer = serializer or CheetahSerializer()
        self._vector_order = AbsoluteVectorOrder()
        self._key_cache: "OrderedDict[tuple[str, str], int]" = OrderedDict()
        self._cache_size = max(cache_size, 1024)
        self._enabled = True
        self._topk_total = 0
        self._topk_hits = 0

    # ------------------------------------------------------------------ #
    # HotPathAdapter API
    # ------------------------------------------------------------------ #
    def publish_context(self, context_hash: str, order_size: int, token_ids: Sequence[int]) -> None:
        if not self._enabled:
            return
        namespace = "ctx"
        if self._lookup_key(namespace, context_hash=context_hash) is not None:
            return
        payload = self._serializer.encode_context(order_size, token_ids)
        try:
            key = self._insert(namespace, context_hash, payload)
            self._register_vector_alias(key, token_ids)
        except CheetahError as exc:
            self._disable(exc)

    def publish_topk(self, order: int, context_hash: str, ranked: Sequence[tuple[int, int]]) -> None:
        if not self._enabled or not ranked:
            return
        namespace = f"topk:{order}"
        payload = self._serializer.encode_topk(order, ranked)
        try:
            key = self._lookup_key(namespace, context_hash=context_hash)
            if key is None:
                self._insert(namespace, context_hash, payload)
            else:
                if not self._client.edit(key, payload):
                    raise CheetahError("failed to edit cheetah value")
        except CheetahError as exc:
            self._disable(exc)

    def publish_counts(self, order: int, context_hash: str, followers: Sequence[tuple[int, int]]) -> None:
        if not self._enabled or not followers:
            return
        namespace = f"cnt:{order}"
        payload = self._serializer.encode_counts(order, followers)
        try:
            key = self._lookup_key(namespace, context_hash=context_hash)
            if key is None:
                self._insert(namespace, context_hash, payload)
            else:
                if not self._client.edit(key, payload):
                    raise CheetahError("failed to edit cheetah counts payload")
        except CheetahError as exc:
            self._disable(exc)

    def fetch_topk(self, order: int, context_hash: str, limit: int) -> list[tuple[int, int]] | None:
        if not self._enabled:
            return None
        self._topk_total += 1
        namespace = f"topk:{order}"
        key = self._lookup_key(namespace, context_hash=context_hash)
        if key is None:
            return None
        payload = self._client.read(key)
        if not payload:
            return None
        record = self._serializer.decode_topk(payload)
        if not record or record.order != order:
            return None
        self._topk_hits += 1
        return record.ranked[:limit]

    def fetch_context_tokens(self, context_hash: str) -> Sequence[int] | None:
        if not self._enabled:
            return None
        namespace = "ctx"
        key = self._lookup_key(namespace, context_hash=context_hash)
        if key is None:
            return None
        payload = self._client.read(key)
        if not payload:
            return None
        record = self._serializer.decode_context(payload)
        if not record:
            return None
        return list(record.token_ids)

    def write_metadata(self, key: str, value: str) -> None:
        if not self._enabled:
            return
        namespace = "meta"
        raw_value = key.encode("utf-8")
        payload = value.encode("utf-8")
        try:
            existing = self._lookup_key(namespace, raw_value=raw_value)
            if existing is None:
                new_key = self._client.insert(payload)
                if new_key is None:
                    raise CheetahError("failed to insert metadata payload")
                self._register_pair(namespace, new_key, raw_value=raw_value)
            else:
                if not self._client.edit(existing, payload):
                    raise CheetahError("failed to edit metadata payload")
        except CheetahError as exc:
            self._disable(exc)

    def read_metadata(self, key: str) -> str | None:
        if not self._enabled:
            return None
        namespace = "meta"
        raw_value = key.encode("utf-8")
        entry_key = self._lookup_key(namespace, raw_value=raw_value)
        if entry_key is None:
            return None
        payload = self._client.read(entry_key)
        if not payload:
            return None
        return payload.decode("utf-8", "replace")

    def scan_namespace(
        self,
        namespace: str,
        *,
        prefix: bytes = b"",
        limit: int = 0,
    ) -> Iterable[tuple[bytes, int]]:
        if not self._enabled:
            return []
        namespace_bytes = namespace.encode("utf-8") + b":"
        scoped_prefix = namespace_bytes + prefix
        results = self._client.pair_scan(scoped_prefix, limit=limit)
        if results is None:
            self._disable(CheetahError("pair_scan failed"))
            return []
        trimmed: list[tuple[bytes, int]] = []
        for raw_value, key in results:
            if not raw_value.startswith(namespace_bytes):
                continue
            trimmed.append((raw_value[len(namespace_bytes) :], key))
        return trimmed

    def iter_counts(self, order: int) -> list[RawCountsProjection]:
        if not self._enabled:
            return []
        namespace = f"cnt:{order}"
        projections: list[RawCountsProjection] = []
        namespace_bytes = namespace.encode("utf-8") + b":"
        results = self._client.pair_reduce("counts", namespace_bytes, limit=-1)
        if results is None:
            self._disable(CheetahError("pair_reduce counts failed"))
            return []
        for raw_value, key in results:
            if not raw_value.startswith(namespace_bytes):
                continue
            trimmed = raw_value[len(namespace_bytes) :]
            payload = self._client.read(key)
            if not payload:
                continue
            record = self._serializer.decode_counts(payload)
            if not record or record.order != order:
                continue
            if trimmed == b"__root__":
                context_hash = "__root__"
            else:
                context_hash = trimmed.hex()
            totals = sum(count for _, count in record.followers)
            projections.append(
                RawCountsProjection(
                    context_hash=context_hash,
                    order=order,
                    totals=totals,
                    followers=record.followers,
                )
            )
        return projections

    def context_relativism(
        self,
        context_tree,
        *,
        limit: int = 32,
        depth: int | None = None,
    ) -> list[RawContextProjection]:
        if not self._enabled:
            return []
        try:
            vector_prefix = self._vector_order.encode_tree(context_tree, depth_limit=depth)
        except (TypeError, ValueError) as exc:
            logger.debug("failed to encode context tree for relativism query: %s", exc)
            return []
        matches = self.scan_namespace("ctxv", prefix=vector_prefix, limit=limit)
        projections: list[RawContextProjection] = []
        for vector_bytes, key in matches:
            payload = self._client.read(key)
            if not payload:
                continue
            context_record = self._serializer.decode_context(payload)
            if not context_record:
                continue
            tokens = tuple(context_record.token_ids)
            context_hash = hash_tokens(tokens)
            ranked = self.fetch_topk(
                context_record.order_size,
                context_hash,
                self._serializer.MAX_TOPK,
            ) or []
            projections.append(
                RawContextProjection(
                    context_hash=context_hash,
                    order_size=context_record.order_size,
                    token_ids=tokens,
                    ranked=tuple(ranked),
                    cheetah_key=key,
                    vector_signature=bytes(vector_bytes),
                )
            )
        return projections

    def topk_hit_ratio(self) -> float:
        if self._topk_total <= 0:
            return 0.0
        return min(1.0, max(0.0, self._topk_hits / float(self._topk_total)))

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #
    def _insert(self, namespace: str, context_hash: str, payload: bytes) -> int:
        key = self._client.insert(payload)
        if key is None:
            raise CheetahError("failed to insert cheetah payload")
        self._register_pair(namespace, key, context_hash=context_hash)
        return key

    def _register_pair(
        self,
        namespace: str,
        key: int,
        *,
        context_hash: str | None = None,
        raw_value: bytes | None = None,
    ) -> None:
        value = self._pair_value(namespace, context_hash=context_hash, raw_value=raw_value)
        if not self._client.pair_set(value, key):
            raise CheetahError("failed to register cheetah pair mapping")
        self._set_cache(namespace, key, context_hash=context_hash, raw_value=raw_value)

    def _register_vector_alias(self, key: int, token_ids: Sequence[int]) -> None:
        if not token_ids:
            return
        try:
            vector_bytes = self._vector_order.encode_tokens(token_ids)
        except (TypeError, ValueError) as exc:
            logger.debug("skipping vector alias for context: %s", exc)
            return
        self._register_pair("ctxv", key, raw_value=vector_bytes)

    def _lookup_key(
        self,
        namespace: str,
        *,
        context_hash: str | None = None,
        raw_value: bytes | None = None,
    ) -> int | None:
        cache_key = self._cache_key(namespace, context_hash=context_hash, raw_value=raw_value)
        if cache_key and cache_key in self._key_cache:
            key = self._key_cache[cache_key]
            self._key_cache.move_to_end(cache_key)
            return key
        value = self._pair_value(namespace, context_hash=context_hash, raw_value=raw_value)
        if value is None:
            return None
        key = self._client.pair_get(value)
        if key is not None:
            self._set_cache(namespace, key, context_hash=context_hash, raw_value=raw_value)
        return key

    def _set_cache(
        self,
        namespace: str,
        key: int,
        *,
        context_hash: str | None = None,
        raw_value: bytes | None = None,
    ) -> None:
        cache_key = self._cache_key(namespace, context_hash=context_hash, raw_value=raw_value)
        if cache_key is None:
            return
        self._key_cache[cache_key] = key
        self._key_cache.move_to_end(cache_key)
        if len(self._key_cache) > self._cache_size:
            self._key_cache.popitem(last=False)

    def _cache_key(
        self,
        namespace: str,
        *,
        context_hash: str | None = None,
        raw_value: bytes | None = None,
    ) -> tuple[str, str] | None:
        identifier = self._normalize_identifier(context_hash=context_hash, raw_value=raw_value)
        if identifier is None:
            return None
        return (namespace, identifier)

    def _pair_value(
        self,
        namespace: str,
        *,
        context_hash: str | None = None,
        raw_value: bytes | None = None,
    ) -> bytes | None:
        namespace_bytes = namespace.encode("utf-8") + b":"
        if raw_value is not None:
            return namespace_bytes + raw_value
        if context_hash is None:
            return None
        if context_hash == "__root__":
            context_bytes = b"__root__"
        else:
            try:
                context_bytes = bytes.fromhex(context_hash)
            except ValueError:
                context_bytes = context_hash.encode("utf-8")
        return namespace_bytes + context_bytes

    def _normalize_identifier(
        self,
        *,
        context_hash: str | None = None,
        raw_value: bytes | None = None,
    ) -> str | None:
        if raw_value is not None:
            return f"bytes:{raw_value.hex()}"
        if context_hash is not None:
            return f"hash:{context_hash}"
        return None

    def _disable(self, exc: Exception) -> None:
        if self._enabled:
            logger.warning("Disabling cheetah hot-path adapter: %s", exc)
            self._enabled = False


def build_cheetah_adapter(
    settings: DBSLMSettings,
    *,
    client: CheetahClient | None = None,
) -> HotPathAdapter:
    backend_active = settings.backend == "cheetah-db" or settings.cheetah_mirror
    if not backend_active:
        return NullHotPathAdapter()
    client = client or CheetahClient(
        settings.cheetah_host,
        settings.cheetah_port,
        database=settings.cheetah_database,
        timeout=settings.cheetah_timeout_seconds,
    )
    adapter = CheetahHotPathAdapter(client)
    if not client.connect():
        logger.warning("cheetah hot-path backend unreachable; falling back to SQLite-only mode")
        return NullHotPathAdapter()
    return adapter


__all__ = [
    "CheetahClient",
    "CheetahError",
    "CheetahHotPathAdapter",
    "CheetahSerializer",
    "CountsPayload",
    "ContextPayload",
    "TopKPayload",
    "build_cheetah_adapter",
]
