from __future__ import annotations

import hashlib
from typing import Sequence


def hash_tokens(token_ids: Sequence[int]) -> str:
    """Return the canonical context hash for a sequence of tokens."""
    if not token_ids:
        return "__root__"
    raw = ",".join(str(tok) for tok in token_ids).encode("utf-8")
    digest = hashlib.blake2b(raw, digest_size=8).hexdigest()
    return digest


__all__ = ["hash_tokens"]
