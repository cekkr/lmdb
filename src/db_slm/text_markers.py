from __future__ import annotations

import re
from typing import Tuple

END_OF_RESPONSE_TOKEN = "|END|"
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")

__all__ = [
    "END_OF_RESPONSE_TOKEN",
    "append_end_marker",
    "strip_end_marker",
    "extract_complete_sentence",
]


def append_end_marker(text: str) -> str:
    """Ensure the |END| tag is appended to the provided text."""
    stripped = text.rstrip()
    if not stripped:
        return END_OF_RESPONSE_TOKEN
    if stripped.endswith(END_OF_RESPONSE_TOKEN):
        return stripped
    return f"{stripped} {END_OF_RESPONSE_TOKEN}"


def strip_end_marker(text: str) -> Tuple[str, bool]:
    """
    Remove the first |END| tag (if present) and return the trimmed text plus a flag.
    """
    marker_index = text.find(END_OF_RESPONSE_TOKEN)
    if marker_index == -1:
        return text.strip(), False
    before = text[:marker_index].rstrip()
    return before, True


def extract_complete_sentence(text: str) -> str:
    """
    Return the first full sentence (respecting |END| markers when available).
    """
    cleaned, _ = strip_end_marker(text)
    cleaned = cleaned.strip()
    if not cleaned:
        return ""
    sentences = [segment.strip() for segment in _SENTENCE_SPLIT_RE.split(cleaned) if segment.strip()]
    if sentences:
        return sentences[0]
    return cleaned
