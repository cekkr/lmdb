from __future__ import annotations

import numbers
import struct
from collections.abc import Sequence
from typing import Any, Iterable


class AbsoluteVectorOrder:
    """Canonicalizes nested token evidence into byte prefixes for cheetah."""

    VERSION = 1
    TYPE_LIST = 0xA0
    TYPE_INT = 0xA1

    def encode_tree(
        self,
        tree: Any,
        *,
        depth_limit: int | None = None,
    ) -> bytes:
        """Return a canonical byte string for an arbitrary nested structure."""
        buf = bytearray()
        buf.append(self.VERSION)
        self._encode_node(tree, buf, depth=0, depth_limit=depth_limit)
        return bytes(buf)

    def encode_tokens(self, token_ids: Sequence[int], *, depth_limit: int | None = None) -> bytes:
        """Encode a flat token sequence as a canonical vector."""
        canonical = [[int(token)] for token in token_ids]
        return self.encode_tree(canonical, depth_limit=depth_limit)

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #
    def _encode_node(
        self,
        node: Any,
        buf: bytearray,
        *,
        depth: int,
        depth_limit: int | None,
    ) -> None:
        if isinstance(node, numbers.Integral):
            buf.append(self.TYPE_INT)
            buf.extend(struct.pack(">I", int(node)))
            return
        if not self._is_sequence(node):
            raise TypeError(f"Unsupported node type for AbsoluteVectorOrder: {type(node)!r}")
        if depth_limit is not None and depth >= depth_limit:
            buf.append(self.TYPE_LIST)
            buf.append(0)
            return
        child_payloads: list[bytes] = []
        for child in node:
            child_buf = bytearray()
            self._encode_node(child, child_buf, depth=depth + 1, depth_limit=depth_limit)
            child_payloads.append(bytes(child_buf))
        child_payloads.sort()
        if len(child_payloads) > 255:
            raise ValueError("AbsoluteVectorOrder only supports ≤255 children per node")
        buf.append(self.TYPE_LIST)
        buf.append(len(child_payloads))
        for payload in child_payloads:
            if len(payload) > 65535:
                raise ValueError("AbsoluteVectorOrder child payload exceeds 64KB")
            buf.extend(struct.pack(">H", len(payload)))
            buf.extend(payload)

    def _is_sequence(self, value: Any) -> bool:
        if isinstance(value, (str, bytes, bytearray)):
            return False
        return isinstance(value, Sequence)


__all__ = ["AbsoluteVectorOrder"]
