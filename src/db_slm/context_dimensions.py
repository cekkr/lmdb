from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable, Sequence


@dataclass(frozen=True)
class ContextDimension:
    """Defines a contiguous span of tokens that should be grouped."""

    start: int
    end: int

    def __post_init__(self) -> None:
        if self.start < 1:
            raise ValueError("context dimension start must be >= 1")
        if self.end < self.start:
            raise ValueError("context dimension end must be >= start")

    @property
    def span(self) -> int:
        return self.end - self.start + 1

    @property
    def weight(self) -> float:
        span = float(self.span)
        return 1.0 / span if span > 0 else 1.0


DEFAULT_CONTEXT_DIMENSIONS: tuple[ContextDimension, ...] = (
    ContextDimension(1, 2),
    ContextDimension(3, 5),
)


class ContextDimensionTracker:
    """Tracks grouped token frequencies to extend frequency penalties across spans."""

    def __init__(
        self,
        dimensions: Sequence[ContextDimension],
        seed_tokens: Sequence[int] | None = None,
    ) -> None:
        self.dimensions = list(dimensions)
        self.history: list[int] = []
        self.max_window = max((dim.end for dim in self.dimensions), default=0)
        self.counts = [defaultdict(int) for _ in self.dimensions]
        if seed_tokens:
            for token_id in seed_tokens:
                self.record(token_id)

    def penalty_for(self, candidate: int, presence_penalty: float, frequency_penalty: float) -> float:
        if not self.dimensions:
            return 0.0
        penalty = 0.0
        history = self.history
        for idx, dim in enumerate(self.dimensions):
            counts = self.counts[idx]
            max_seen = 0
            for size in range(dim.start, dim.end + 1):
                if size == 1:
                    seq = (candidate,)
                else:
                    context_len = size - 1
                    if len(history) < context_len:
                        continue
                    seq = tuple(history[-context_len:] + [candidate])
                seen = counts.get(seq, 0)
                if seen > max_seen:
                    max_seen = seen
            if max_seen:
                weight = dim.weight
                penalty += presence_penalty * weight
                penalty += max_seen * frequency_penalty * weight
        return penalty

    def record(self, token_id: int) -> None:
        self.history.append(token_id)
        if self.max_window and len(self.history) > self.max_window:
            self.history = self.history[-self.max_window :]
        if not self.dimensions:
            return
        for idx, dim in enumerate(self.dimensions):
            counts = self.counts[idx]
            for size in range(dim.start, dim.end + 1):
                if len(self.history) < size:
                    continue
                seq = tuple(self.history[-size:])
                counts[seq] += 1


def parse_context_dimensions_arg(
    raw_value: str | None,
    *,
    default: Sequence[ContextDimension] | None = DEFAULT_CONTEXT_DIMENSIONS,
) -> list[ContextDimension] | None:
    """Parse CLI-style dimension strings like '1-2,3-5'."""
    if raw_value is None:
        return list(default) if default is not None else None
    text = raw_value.strip()
    if not text:
        return []
    lowered = text.lower()
    if lowered in {"off", "none", "disable"}:
        return []
    parts = [part.strip() for part in text.split(",") if part.strip()]
    if not parts:
        return []
    dimensions: list[ContextDimension] = []
    for part in parts:
        if "-" in part:
            tokens = part.split("-", 1)
        elif ":" in part:
            tokens = part.split(":", 1)
        else:
            tokens = [part, part]
        try:
            start = int(tokens[0])
            end = int(tokens[1])
        except (ValueError, IndexError) as exc:
            raise ValueError(f"Invalid context dimension '{part}'") from exc
        dimensions.append(ContextDimension(start, end))
    return dimensions


def serialize_context_dimensions(dimensions: Sequence[ContextDimension]) -> str:
    payload = [[dim.start, dim.end] for dim in dimensions]
    return json.dumps(payload)


def deserialize_context_dimensions(raw_value: str | None) -> list[ContextDimension]:
    if not raw_value:
        return []
    try:
        payload = json.loads(raw_value)
    except json.JSONDecodeError as exc:
        raise ValueError("Invalid context dimension metadata payload") from exc
    dimensions: list[ContextDimension] = []
    for entry in payload:
        if not isinstance(entry, (list, tuple)) or len(entry) != 2:
            raise ValueError("Malformed context dimension entry")
        start, end = entry
        dimensions.append(ContextDimension(int(start), int(end)))
    return dimensions


def format_context_dimensions(dimensions: Sequence[ContextDimension]) -> str:
    if not dimensions:
        return "off"
    formatted: list[str] = []
    for dim in dimensions:
        if dim.start == dim.end:
            formatted.append(str(dim.start))
        else:
            formatted.append(f"{dim.start}-{dim.end}")
    return ",".join(formatted)


def ensure_context_dimensions(dimensions: Sequence[ContextDimension] | None) -> list[ContextDimension]:
    if dimensions is None:
        return list(DEFAULT_CONTEXT_DIMENSIONS)
    return list(dimensions)
