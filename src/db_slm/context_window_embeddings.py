from __future__ import annotations

import hashlib
import json
import math
import random
import re
import shlex
from collections import Counter
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
    tag_marker: str | None = None


@dataclass
class TaggedToken:
    """Tracks the originating prompt-tag for each lexical token."""

    text: str
    tag: str | None = None


@dataclass
class ContextDimensionPrototype:
    """Tracks the running embedding average for a dimension."""

    dimension: ContextDimension
    stride: int
    vector: List[float] = field(default_factory=list)
    count: int = 0
    tag_sum: float = 0.0
    tag_sq_sum: float = 0.0
    tag_total: int = 0

    def update(self, new_vector: Sequence[float], tag_index: int | None = None) -> None:
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
        if tag_index is not None:
            tag_value = float(tag_index)
            self.tag_sum += tag_value
            self.tag_sq_sum += tag_value * tag_value
            self.tag_total += 1

    def as_dict(self) -> dict[str, object]:
        return {
            "start": self.dimension.start,
            "end": self.dimension.end,
            "window": self.dimension.span,
            "stride": self.stride,
            "count": self.count,
            "vector": [round(value, 6) for value in self.vector],
            "tag_sum": round(self.tag_sum, 6),
            "tag_sq_sum": round(self.tag_sq_sum, 6),
            "tag_total": self.tag_total,
        }

    def tag_weight(self, tag_index: int | None) -> float:
        if tag_index is None or self.tag_total <= 0:
            return 1.0
        mean = self.tag_sum / float(self.tag_total)
        variance = max(0.0, (self.tag_sq_sum / float(self.tag_total)) - mean * mean)
        deviation = abs(tag_index - mean)
        baseline = 1.0 + deviation / (1.0 + variance)
        return max(0.75, min(1.5, baseline))


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
            candidates: list[list[TaggedToken]] = []
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
                snippet_text, tag_marker = self._build_snippet(index, chunk)
                snippets.append(ContextWindowSnippet(index, snippet_text, tag_marker))
        return snippets

    def _build_snippet(
        self,
        dimension_index: int,
        chunk: Sequence[TaggedToken],
    ) -> tuple[str, str | None]:
        words = [token.text for token in chunk if token.text]
        tag_marker = self._dimension_tag_marker(dimension_index, chunk)
        parts: list[str] = []
        if tag_marker:
            parts.append(tag_marker)
        if words:
            parts.append(" ".join(words))
        snippet_text = " ".join(part for part in parts if part).strip()
        return snippet_text or " ".join(words), tag_marker

    def _dimension_tag_marker(
        self,
        dimension_index: int,
        chunk: Sequence[TaggedToken],
    ) -> str | None:
        counts = Counter(token.tag for token in chunk if token.tag)
        if not counts:
            return None
        tag, _ = counts.most_common(1)[0]
        if not tag:
            return None
        canonical = self._normalize_tag_token(tag) or tag.strip()
        if not canonical:
            return None
        return f"|CTX_DIM_{dimension_index}:{canonical}|"

    def _tokens(self, text: str) -> list[TaggedToken]:
        raw_tokens = self._lex(text)
        condensed = self._merge_titles(raw_tokens)
        expanded: list[TaggedToken] = []
        for token in condensed:
            expanded.extend(self._split_compound_token(token))
        return [token for token in expanded if token.text]

    def _lex(self, text: str) -> list[TaggedToken]:
        lexer = shlex.shlex(text, posix=True)
        lexer.whitespace_split = True
        lexer.commenters = ""
        tokens: list[TaggedToken] = []
        current_tag: str | None = None

        def consume(piece: str) -> None:
            nonlocal current_tag
            normalized = piece.strip()
            if not normalized:
                return
            tag_token = self._normalize_tag_token(normalized)
            if tag_token:
                current_tag = tag_token
                tokens.append(TaggedToken(tag_token, current_tag))
                return
            tokens.append(TaggedToken(normalized, current_tag))

        try:
            for piece in lexer:
                consume(piece)
        except ValueError as exc:
            log_verbose(
                2,
                f"[context-dim] Falling back to whitespace tokenization due to malformed lexeme: {exc}",
            )
            for piece in re.findall(r"\S+", text):
                consume(piece)
        return tokens

    def _split_compound_token(self, token: TaggedToken) -> list[TaggedToken]:
        lowered = token.text.strip()
        if not lowered:
            return []
        if self._normalize_tag_token(lowered):
            return [TaggedToken(self._normalize_tag_token(lowered) or lowered, token.tag)]
        for delimiter in ("::", "->", "=>"):
            if delimiter in lowered:
                parts = [part for part in lowered.split(delimiter) if part]
                if parts:
                    return [TaggedToken(part, token.tag) for part in parts]
        if any(sep in lowered for sep in ("_", "/", "+", "-", "#")):
            parts = re.split(r"[_/\+\-#]+", lowered)
            return [TaggedToken(part, token.tag) for part in parts if part]
        camel = _CAMEL_CASE_PATTERN.split(lowered)
        if len(camel) > 1:
            return [TaggedToken(part, token.tag) for part in camel if part]
        return [TaggedToken(lowered, token.tag)]

    def _merge_titles(self, tokens: Sequence[TaggedToken]) -> list[TaggedToken]:
        merged: list[TaggedToken] = []
        idx = 0
        while idx < len(tokens):
            token = tokens[idx]
            if self._is_title_token(token.text):
                chain = [token]
                offset = idx + 1
                while offset < len(tokens) and self._is_title_token(tokens[offset].text):
                    chain.append(tokens[offset])
                    offset += 1
                if len(chain) > 1:
                    joined = " ".join(part.text for part in chain)
                    merged.append(TaggedToken(joined, chain[0].tag))
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

    @staticmethod
    def _normalize_tag_token(token: str) -> str | None:
        candidate = token.strip()
        if not candidate:
            return None
        trailing_colon = candidate.endswith(":")
        if trailing_colon:
            candidate = candidate[:-1]
        if len(candidate) < 3 or not candidate.startswith("|") or not candidate.endswith("|"):
            return None
        return f"{candidate}:"


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
        self._tag_enumerator: dict[str, int] = {}
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
            tag_index = self._tag_index_from_marker(snippet.tag_marker)
            proto.update(vector, tag_index)
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
        similarity_by_dim: dict[int, list[tuple[float, float]]] = {}
        for snippet, vector in zip(snippets, vectors):
            proto = self._prototype_for(snippet.dimension_index)
            if proto is None or not proto.vector or not vector:
                continue
            similarity = self._cosine_similarity(proto.vector, vector)
            tag_index = self._tag_index_from_marker(snippet.tag_marker)
            tag_weight = proto.tag_weight(tag_index)
            similarity_by_dim.setdefault(snippet.dimension_index, []).append((similarity, tag_weight))
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
            best_similarity, tag_weight = max(candidates, key=lambda item: item[0])
            base_weight = self._weight_from_similarity(best_similarity, proto)
            combined = base_weight * tag_weight
            weights.append(max(0.25, min(2.0, combined)))
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

    def set_tag_enumerator(self, enumerator: dict[str, int]) -> None:
        cleaned: dict[str, int] = {}
        for key, value in enumerator.items():
            canonical = key.strip()
            if not canonical:
                continue
            cleaned[canonical] = int(value)
        self._tag_enumerator = cleaned

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

    def _tag_index_from_marker(self, marker: str | None) -> int | None:
        if not marker or not self._tag_enumerator:
            return None
        _, _, suffix = marker.partition(":")
        if not suffix:
            return None
        canonical = suffix.strip()
        if canonical.endswith("|"):
            canonical = canonical[:-1]
        return self._tag_enumerator.get(canonical)

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
                tag_sum = float(dim_data.get("tag_sum", 0.0))
                tag_sq_sum = float(dim_data.get("tag_sq_sum", 0.0))
                tag_total = int(dim_data.get("tag_total", 0))
            except (KeyError, ValueError, TypeError):
                continue
            dimension = ContextDimension(start, end)
            proto = ContextDimensionPrototype(
                dimension,
                stride,
                list(vector),
                max(0, count),
                tag_sum=max(0.0, tag_sum),
                tag_sq_sum=max(0.0, tag_sq_sum),
                tag_total=max(0, tag_total),
            )
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
