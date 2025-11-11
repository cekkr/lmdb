from __future__ import annotations

import base64
import logging
import socket
import struct
import threading
from collections import OrderedDict
from dataclasses import dataclass
from typing import Sequence

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


@dataclass(frozen=True)
class TopKPayload:
    order: int
    ranked: list[tuple[int, int]]


class CheetahSerializer:
    """Binary codec for the cheetah hot-path payloads."""

    CONTEXT_VERSION = 1
    TOPK_VERSION = 1
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
        self._key_cache: "OrderedDict[tuple[str, str], int]" = OrderedDict()
        self._cache_size = max(cache_size, 1024)
        self._enabled = True

    # ------------------------------------------------------------------ #
    # HotPathAdapter API
    # ------------------------------------------------------------------ #
    def publish_context(self, context_hash: str, order_size: int, token_ids: Sequence[int]) -> None:
        if not self._enabled:
            return
        namespace = "ctx"
        if self._lookup_key(namespace, context_hash) is not None:
            return
        payload = self._serializer.encode_context(order_size, token_ids)
        try:
            self._insert(namespace, context_hash, payload)
        except CheetahError as exc:
            self._disable(exc)

    def publish_topk(self, order: int, context_hash: str, ranked: Sequence[tuple[int, int]]) -> None:
        if not self._enabled or not ranked:
            return
        namespace = f"topk:{order}"
        payload = self._serializer.encode_topk(order, ranked)
        try:
            key = self._lookup_key(namespace, context_hash)
            if key is None:
                self._insert(namespace, context_hash, payload)
            else:
                if not self._client.edit(key, payload):
                    raise CheetahError("failed to edit cheetah value")
        except CheetahError as exc:
            self._disable(exc)

    def fetch_topk(self, order: int, context_hash: str, limit: int) -> list[tuple[int, int]] | None:
        if not self._enabled:
            return None
        namespace = f"topk:{order}"
        key = self._lookup_key(namespace, context_hash)
        if key is None:
            return None
        payload = self._client.read(key)
        if not payload:
            return None
        record = self._serializer.decode_topk(payload)
        if not record or record.order != order:
            return None
        return record.ranked[:limit]

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #
    def _insert(self, namespace: str, context_hash: str, payload: bytes) -> None:
        key = self._client.insert(payload)
        if key is None:
            raise CheetahError("failed to insert cheetah payload")
        value = self._pair_value(namespace, context_hash)
        if not self._client.pair_set(value, key):
            raise CheetahError("failed to register cheetah pair mapping")
        self._set_cache(namespace, context_hash, key)

    def _lookup_key(self, namespace: str, context_hash: str) -> int | None:
        cache_key = (namespace, context_hash)
        if cache_key in self._key_cache:
            key = self._key_cache[cache_key]
            self._key_cache.move_to_end(cache_key)
            return key
        value = self._pair_value(namespace, context_hash)
        key = self._client.pair_get(value)
        if key is not None:
            self._set_cache(namespace, context_hash, key)
        return key

    def _set_cache(self, namespace: str, context_hash: str, key: int) -> None:
        cache_key = (namespace, context_hash)
        self._key_cache[cache_key] = key
        self._key_cache.move_to_end(cache_key)
        if len(self._key_cache) > self._cache_size:
            self._key_cache.popitem(last=False)

    def _pair_value(self, namespace: str, context_hash: str) -> bytes:
        if context_hash == "__root__":
            context_bytes = b"__root__"
        else:
            try:
                context_bytes = bytes.fromhex(context_hash)
            except ValueError:
                context_bytes = context_hash.encode("utf-8")
        return namespace.encode("utf-8") + b":" + context_bytes

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
    "TopKPayload",
    "build_cheetah_adapter",
]
