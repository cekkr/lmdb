from __future__ import annotations

import hashlib
import json
import math
import random
import re
import shlex
from dataclasses import dataclass, field
from typing import List, Sequence

from .adapters.base import HotPathAdapter
from .context_dimensions import ContextDimension
from .db import DatabaseEnvironment
from .sentence_parts import ExternalEmbedder
from log_helpers import log, log_verbose


@dataclass
class ContextWindowSnippet:
    """Holds the text extracted for a particular context dimension."""

    dimension_index: int
    text: str


@dataclass
class ContextDimensionPrototype:
    """Tracks the running embedding average for a dimension."""

    dimension: ContextDimension
    stride: int
    vector: List[float] = field(default_factory=list)
    count: int = 0

    def update(self, new_vector: Sequence[float]) -> None:
        if not new_vector:
            return
        if not self.vector:
            self.vector = list(new_vector)
            self.count = 1
            return
        if len(new_vector) > len(self.vector):
            self.vector.extend([0.0] * (len(new_vector) - len(self.vector)))
        elif len(new_vector) < len(self.vector):
            new_vector = list(new_vector) + [0.0] * (len(self.vector) - len(new_vector))
        total = float(self.count)
        for idx, value in enumerate(new_vector):
            prior = self.vector[idx]
            self.vector[idx] = (prior * total + value) / (total + 1.0)
        self.count += 1

    def as_dict(self) -> dict[str, object]:
        return {
            "start": self.dimension.start,
            "end": self.dimension.end,
            "window": self.dimension.span,
            "stride": self.stride,
            "count": self.count,
            "vector": [round(value, 6) for value in self.vector],
        }


_CAMEL_CASE_PATTERN = re.compile(r"(?<!^)(?=[A-Z][a-z])")


class ContextWindowExtractor:
    """Generates representative word windows for each configured dimension."""

    def __init__(self, dimensions: Sequence[ContextDimension], stride_ratio: float = 0.5) -> None:
        self.dimensions = list(dimensions)
        self.stride_ratio = max(0.1, min(stride_ratio, 1.0))

    def sample(
        self,
        text: str,
        *,
        windows_per_dimension: int,
        rng: random.Random,
    ) -> list[ContextWindowSnippet]:
        if not text.strip():
            return []
        tokens = self._tokens(text)
        if not tokens:
            return []
        snippets: list[ContextWindowSnippet] = []
        for index, dim in enumerate(self.dimensions):
            window = max(2, dim.span)
            stride = max(1, int(round(window * self.stride_ratio)))
            candidates: list[list[str]] = []
            for start in range(0, len(tokens), stride):
                chunk = tokens[start : start + window]
                if len(chunk) < 2:
                    continue
                candidates.append(chunk)
                if len(candidates) >= windows_per_dimension * 3:
                    break
            if not candidates:
                continue
            if len(candidates) > windows_per_dimension:
                rng.shuffle(candidates)
                selected = candidates[:windows_per_dimension]
            else:
                selected = candidates
            for chunk in selected:
                snippets.append(ContextWindowSnippet(index, " ".join(chunk)))
        return snippets

    def _tokens(self, text: str) -> list[str]:
        raw_tokens = self._lex(text)
        condensed = self._merge_titles(raw_tokens)
        expanded: list[str] = []
        for token in condensed:
            expanded.extend(self._split_compound(token))
        return [token for token in expanded if token]

    def _lex(self, text: str) -> list[str]:
        lexer = shlex.shlex(text, posix=True)
        lexer.whitespace_split = True
        lexer.commenters = ""
        tokens: list[str] = []
        for piece in lexer:
            normalized = piece.strip()
            if not normalized:
                continue
            if normalized.startswith("|"):
                # Skip metadata markers such as |SEGMENT|/|CTX|
                continue
            if normalized.endswith(":") and ":" not in normalized[:-1]:
                continue
            tokens.append(normalized)
        return tokens

    def _split_compound(self, token: str) -> list[str]:
        lowered = token.strip()
        if not lowered:
            return []
        for delimiter in ("::", "->", "=>"):
            if delimiter in lowered:
                parts = [part for part in lowered.split(delimiter) if part]
                if parts:
                    return parts
        if any(sep in lowered for sep in ("_", "/", "+", "-", "#")):
            parts = re.split(r"[_/\+\-#]+", lowered)
            return [part for part in parts if part]
        camel = _CAMEL_CASE_PATTERN.split(lowered)
        if len(camel) > 1:
            return [part for part in camel if part]
        return [lowered]

    def _merge_titles(self, tokens: Sequence[str]) -> list[str]:
        merged: list[str] = []
        idx = 0
        while idx < len(tokens):
            token = tokens[idx]
            if self._is_title_token(token):
                chain = [token]
                offset = idx + 1
                while offset < len(tokens) and self._is_title_token(tokens[offset]):
                    chain.append(tokens[offset])
                    offset += 1
                if len(chain) > 1:
                    merged.append(" ".join(chain))
                else:
                    merged.append(token)
                idx = offset
                continue
            merged.append(token)
            idx += 1
        return merged

    @staticmethod
    def _is_title_token(token: str) -> bool:
        if not token:
            return False
        if token[0].islower():
            return False
        if len(token) == 1:
            return False
        if token.endswith("."):
            return False
        letters = [ch for ch in token if ch.isalpha()]
        if not letters:
            return False
        return True


class ContextWindowEmbeddingManager:
    """Learns and applies embedding prototypes for multi-scale context windows."""

    _METADATA_KEY = "context_dimension_embeddings"

    def __init__(
        self,
        dimensions: Sequence[ContextDimension],
        *,
        embedder: ExternalEmbedder | None,
        db: DatabaseEnvironment,
        hot_path: HotPathAdapter,
        stride_ratio: float = 0.5,
        max_train_windows: int = 24,
        max_infer_windows: int = 6,
    ) -> None:
        self.dimensions = list(dimensions)
        self.embedder = embedder or ExternalEmbedder("all-MiniLM-L6-v2")
        self.db = db
        self.hot_path = hot_path
        self.extractor = ContextWindowExtractor(self.dimensions, stride_ratio=stride_ratio)
        self.max_train_windows = max(1, max_train_windows)
        self.max_infer_windows = max(1, max_infer_windows)
        self._prototypes: list[ContextDimensionPrototype] = []
        self._rng = random.Random(0xCEEDA7)
        self._dirty = False
        self._load_metadata()

    def enabled(self) -> bool:
        return bool(self.dimensions)

    def observe_corpus(self, text: str) -> None:
        if not self.enabled():
            return
        snippets = self.extractor.sample(
            text,
            windows_per_dimension=self.max_train_windows,
            rng=self._rng,
        )
        if not snippets:
            return
        vectors = self.embedder.embed([snippet.text for snippet in snippets])
        for snippet, vector in zip(snippets, vectors):
            proto = self._ensure_prototype(snippet.dimension_index)
            proto.update(vector)
            self._dirty = True
        if self._dirty:
            log_verbose(
                4,
                f"[context-dim] Learned {len(self._prototypes)} embedding prototype(s) (dirty={self._dirty})",
            )

    def weights_for_text(self, text: str) -> list[float]:
        if not self.enabled() or not text.strip():
            return [1.0 for _ in self.dimensions]
        seed = self._seed_for_text(text)
        infer_rng = random.Random(seed)
        snippets = self.extractor.sample(
            text,
            windows_per_dimension=self.max_infer_windows,
            rng=infer_rng,
        )
        if not snippets:
            return [1.0 for _ in self.dimensions]
        vectors = self.embedder.embed([snippet.text for snippet in snippets])
        similarity_by_dim: dict[int, list[float]] = {}
        for snippet, vector in zip(snippets, vectors):
            proto = self._prototype_for(snippet.dimension_index)
            if proto is None or not proto.vector or not vector:
                continue
            similarity = self._cosine_similarity(proto.vector, vector)
            similarity_by_dim.setdefault(snippet.dimension_index, []).append(similarity)
        weights: list[float] = []
        for idx, _dim in enumerate(self.dimensions):
            proto = self._prototype_for(idx)
            if proto is None or proto.count == 0:
                weights.append(1.0)
                continue
            candidates = similarity_by_dim.get(idx)
            if not candidates:
                weights.append(1.0)
                continue
            best = max(candidates)
            weights.append(self._weight_from_similarity(best, proto))
        return weights

    def flush(self) -> None:
        if not self._dirty:
            return
        payload = {
            "model": self.embedder.model_name,
            "stride_ratio": self.extractor.stride_ratio,
            "max_train_windows": self.max_train_windows,
            "dimensions": [proto.as_dict() for proto in self._prototypes if proto.count > 0],
        }
        serialized = json.dumps(payload, separators=(",", ":"))
        self.db.set_metadata(self._METADATA_KEY, serialized)
        writer = getattr(self.hot_path, "write_metadata", None)
        if callable(writer):
            writer(self._METADATA_KEY, serialized)
        self._dirty = False

    def describe(self) -> str | None:
        if not self._prototypes:
            return None
        descriptors = []
        for proto in self._prototypes:
            if proto.count <= 0:
                continue
            descriptors.append(f"{proto.dimension.span}w/{proto.count} samples")
        if not descriptors:
            return None
        return "; ".join(descriptors)

    def _ensure_prototype(self, index: int) -> ContextDimensionPrototype:
        while len(self._prototypes) <= index:
            dimension = self.dimensions[len(self._prototypes)]
            stride = max(1, int(round(dimension.span * self.extractor.stride_ratio)))
            self._prototypes.append(ContextDimensionPrototype(dimension, stride))
        return self._prototypes[index]

    def _prototype_for(self, index: int) -> ContextDimensionPrototype | None:
        if 0 <= index < len(self._prototypes):
            proto = self._prototypes[index]
            if proto.count > 0 and proto.vector:
                return proto
        return None

    def _weight_from_similarity(self, similarity: float, proto: ContextDimensionPrototype) -> float:
        # Similarity ranges [-1, 1]; convert into a smooth penalty multiplier.
        intensity = min(0.35, math.log1p(proto.count) / 15.0)
        scale = 1.0 - similarity * (0.2 + intensity)
        return max(0.5, min(1.5, scale))

    def _cosine_similarity(self, base: Sequence[float], other: Sequence[float]) -> float:
        length = min(len(base), len(other))
        if length == 0:
            return 0.0
        dot = 0.0
        norm_base = 0.0
        norm_other = 0.0
        for idx in range(length):
            a = base[idx]
            b = other[idx]
            dot += a * b
            norm_base += a * a
            norm_other += b * b
        if norm_base <= 0 or norm_other <= 0:
            return 0.0
        return dot / math.sqrt(norm_base * norm_other)

    def _load_metadata(self) -> None:
        reader = getattr(self.hot_path, "read_metadata", None)
        raw_value = None
        if callable(reader):
            raw_value = reader(self._METADATA_KEY)
        if not raw_value:
            raw_value = self.db.get_metadata(self._METADATA_KEY)
        if not raw_value:
            return
        try:
            payload = json.loads(raw_value)
        except json.JSONDecodeError:
            log("[context-dim] Warning: invalid context embedding metadata; ignoring.")
            return
        raw_dimensions = payload.get("dimensions") or []
        if not isinstance(raw_dimensions, list):
            return
        prototypes: list[ContextDimensionPrototype] = []
        for dim_data in raw_dimensions:
            if not isinstance(dim_data, dict):
                continue
            try:
                start = int(dim_data["start"])
                end = int(dim_data["end"])
                stride = int(dim_data.get("stride", max(1, (end - start + 1) // 2)))
                count = int(dim_data.get("count", 0))
                vector = [float(value) for value in dim_data.get("vector", [])]
            except (KeyError, ValueError, TypeError):
                continue
            dimension = ContextDimension(start, end)
            proto = ContextDimensionPrototype(dimension, stride, list(vector), max(0, count))
            prototypes.append(proto)
        if prototypes:
            ordered: list[ContextDimensionPrototype] = []
            for dim in self.dimensions:
                matched_proto = None
                for proto in list(prototypes):
                    if proto.dimension.start == dim.start and proto.dimension.end == dim.end:
                        matched_proto = proto
                        prototypes.remove(proto)
                        break
                if matched_proto:
                    matched_proto.dimension = dim
                    ordered.append(matched_proto)
                else:
                    stride = max(1, int(round(dim.span * self.extractor.stride_ratio)))
                    ordered.append(ContextDimensionPrototype(dim, stride))
            self._prototypes = ordered
            log_verbose(
                3,
                f"[context-dim] Loaded {len(ordered)} context window prototypes from metadata.",
            )

    @staticmethod
    def _seed_for_text(text: str) -> int:
        payload = text.encode("utf-8", "ignore")
        digest = hashlib.blake2b(payload, digest_size=8).digest()
        return int.from_bytes(digest, "big")


__all__ = [
    "ContextWindowEmbeddingManager",
]
