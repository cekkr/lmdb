from __future__ import annotations

import random
import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

from .db import DatabaseEnvironment


TOKEN_PATTERN = re.compile(r"\w+|[^\w\s]", re.UNICODE)


@dataclass(frozen=True)
class NGramPrediction:
    token: str
    probability: float
    count: float


class NGramModel:
    """Aria-backed N-gram probability lookup with adaptive smoothing and caching."""

    def __init__(self, db: DatabaseEnvironment, order: int = 3, smoothing_alpha: float = 0.25) -> None:
        if order < 2:
            raise ValueError("N-gram order must be >= 2")
        if smoothing_alpha <= 0:
            raise ValueError("smoothing_alpha must be > 0")
        self.db = db
        self.order = order
        self.smoothing_alpha = smoothing_alpha
        self._prediction_cache: Dict[Tuple[str, int], List[NGramPrediction]] = {}
        self._cache_index: Dict[str, set[int]] = defaultdict(set)

    # ------------------------------------------------------------------ #
    # Data management
    # ------------------------------------------------------------------ #
    def observe(self, tokens: Sequence[str], weight: float | None = None) -> None:
        """
        Persist an N-gram probability row. `tokens` contains the full n-gram,
        so the final element becomes the predicted token.
        """
        if len(tokens) != self.order:
            raise ValueError(f"Expected {self.order} tokens, got {len(tokens)}")
        context_tokens = tokens[:-1]
        next_token = tokens[-1]
        increment = float(weight if weight is not None else 1.0)
        if increment <= 0:
            return
        context_hash = self._context_hash(context_tokens)
        self.db.execute(
            """
            INSERT INTO tbl_l1_context_registry(context_hash, order_size, total_count, hot_rank)
            VALUES (?, ?, ?, 0.0)
            ON CONFLICT(context_hash) DO UPDATE
            SET total_count = total_count + excluded.total_count,
                last_seen_at = CURRENT_TIMESTAMP
            """,
            (context_hash, self.order - 1, increment),
        )
        self.db.execute(
            """
            INSERT INTO tbl_l1_ngram_counts(context_hash, next_token, observed_count)
            VALUES (?, ?, ?)
            ON CONFLICT(context_hash, next_token) DO UPDATE
            SET observed_count = observed_count + excluded.observed_count,
                last_seen_at = CURRENT_TIMESTAMP
            """,
            (context_hash, next_token, increment),
        )
        self._invalidate_cache(context_hash)

    def seed_defaults(self) -> None:
        """Populate a tiny fallback distribution used by demos/tests."""
        defaults = [
            (("__default__",), "Furthermore", 0.2),
            (("__default__",), "Additionally", 0.2),
            (("__default__",), "In", 0.2),
            (("__default__",), "Overall", 0.2),
            (("__default__",), "This", 0.2),
        ]
        for context_tokens, token, prob in defaults:
            context_hash = context_tokens[0]
            self.db.execute(
                """
                INSERT INTO tbl_l1_context_registry(context_hash, order_size, total_count, hot_rank)
                VALUES (?, ?, ?, 0.0)
                ON CONFLICT(context_hash) DO UPDATE
                SET total_count = total_count + excluded.total_count
                """,
                (context_hash, self.order - 1, prob),
            )
            self.db.execute(
                """
                INSERT INTO tbl_l1_ngram_counts(context_hash, next_token, observed_count)
                VALUES (?, ?, ?)
                ON CONFLICT(context_hash, next_token) DO UPDATE
                SET observed_count = observed_count + excluded.observed_count
                """,
                (context_hash, token, prob),
            )
        self._invalidate_cache("__default__")

    # ------------------------------------------------------------------ #
    # Generation helpers
    # ------------------------------------------------------------------ #
    def predict_next(self, context_tokens: Sequence[str], limit: int = 5) -> List[NGramPrediction]:
        context_hash = self._context_hash(context_tokens)
        cache_key = (context_hash, limit)
        cached = self._prediction_cache.get(cache_key)
        if cached is not None:
            return cached

        fallback_hash = context_hash
        rows = self._fetch_predictions(context_hash, limit)
        if not rows and context_hash != "__default__":
            rows = self._fetch_predictions("__default__", limit)
            fallback_hash = "__default__"
        self._prediction_cache[cache_key] = rows
        self._cache_index[context_hash].add(limit)
        if fallback_hash != context_hash:
            self._cache_index[fallback_hash].add(limit)
        return rows

    def stitch_tokens(self, seed_tokens: Sequence[str], target_length: int = 12) -> str:
        """
        Generate connective tissue tokens that smooth the gap between concepts.
        Uses greedy sampling with a temperature-free probability pick.
        """
        context = list(seed_tokens)[-(self.order - 1) :]
        generated: list[str] = []
        for _ in range(target_length):
            predictions = self.predict_next(context, limit=3)
            if not predictions:
                break
            next_token = self._sample(predictions)
            generated.append(next_token)
            context.append(next_token)
            context = context[-(self.order - 1) :]
        return self.detokenize(generated)

    @staticmethod
    def tokenize(text: str) -> list[str]:
        return [match.group(0).lower() for match in TOKEN_PATTERN.finditer(text)]

    @staticmethod
    def detokenize(tokens: Iterable[str]) -> str:
        buffer: list[str] = []
        for token in tokens:
            if not buffer:
                buffer.append(token)
                continue
            if token.isalnum():
                buffer.append(f" {token}")
            else:
                buffer.append(token)
        return "".join(buffer)

    @staticmethod
    def _sample(predictions: Sequence[NGramPrediction]) -> str:
        if not predictions:
            return ""
        total = sum(p.probability for p in predictions)
        cutoff = random.random() * total
        running = 0.0
        for prediction in predictions:
            running += prediction.probability
            if running >= cutoff:
                return prediction.token
        return predictions[-1].token

    def _context_hash(self, tokens: Sequence[str]) -> str:
        context_window = list(tokens)[-(self.order - 1) :]
        return self.db.hash_tokens(context_window) if context_window else "__default__"

    def _fetch_predictions(self, context_hash: str, limit: int) -> List[NGramPrediction]:
        meta_rows = self.db.query(
            """
            SELECT total_count
            FROM tbl_l1_context_registry
            WHERE context_hash = ?
            """,
            (context_hash,),
        )
        if not meta_rows:
            return []
        total_count = meta_rows[0]["total_count"] or 0.0
        if total_count <= 0:
            return []

        vocab_rows = self.db.query(
            """
            SELECT COUNT(*) AS variant_count
            FROM tbl_l1_ngram_counts
            WHERE context_hash = ?
            """,
            (context_hash,),
        )
        variant_count = max(vocab_rows[0]["variant_count"], 1)

        rows = self.db.query(
            """
            SELECT next_token, observed_count
            FROM tbl_l1_ngram_counts
            WHERE context_hash = ?
            ORDER BY observed_count DESC, last_seen_at DESC
            LIMIT ?
            """,
            (context_hash, limit),
        )
        denom = total_count + self.smoothing_alpha * variant_count
        predictions: list[NGramPrediction] = []
        if denom <= 0:
            return predictions
        for row in rows:
            count = row["observed_count"]
            prob = (count + self.smoothing_alpha) / denom
            predictions.append(NGramPrediction(row["next_token"], prob, count))
        return predictions

    def _invalidate_cache(self, context_hash: str) -> None:
        limits = self._cache_index.pop(context_hash, set())
        for limit in limits:
            self._prediction_cache.pop((context_hash, limit), None)
