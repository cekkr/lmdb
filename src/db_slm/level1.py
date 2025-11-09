from __future__ import annotations

import random
import re
from dataclasses import dataclass
from typing import Iterable, List, Sequence

from .db import DatabaseEnvironment


TOKEN_PATTERN = re.compile(r"\w+|[^\w\s]", re.UNICODE)


@dataclass(frozen=True)
class NGramPrediction:
    token: str
    probability: float


class NGramModel:
    """Aria-backed N-gram probability lookup."""

    def __init__(self, db: DatabaseEnvironment, order: int = 3) -> None:
        if order < 2:
            raise ValueError("N-gram order must be >= 2")
        self.db = db
        self.order = order

    # ------------------------------------------------------------------ #
    # Data management
    # ------------------------------------------------------------------ #
    def observe(self, tokens: Sequence[str], probability: float | None = None) -> None:
        """
        Persist an N-gram probability row. `tokens` contains the full n-gram,
        so the final element becomes the predicted token.
        """
        if len(tokens) != self.order:
            raise ValueError(f"Expected {self.order} tokens, got {len(tokens)}")
        context_tokens = tokens[:-1]
        next_token = tokens[-1]
        prob = probability if probability is not None else 1.0
        context_hash = self.db.hash_tokens(context_tokens)
        self.db.execute(
            """
            INSERT INTO tbl_l1_ngram_probs(context_hash, next_token, probability)
            VALUES (?, ?, ?)
            """,
            (context_hash, next_token, prob),
        )

    def seed_defaults(self) -> None:
        """Populate a tiny fallback distribution used by demos/tests."""
        defaults = [
            (("__default__",), "Furthermore", 0.2),
            (("__default__",), "Additionally", 0.2),
            (("__default__",), "In", 0.2),
            (("__default__",), "Overall", 0.2),
            (("__default__",), "This", 0.2),
        ]
        rows = []
        for context_tokens, token, prob in defaults:
            context_hash = context_tokens[0]
            rows.append((context_hash, token, prob))
        self.db.executemany(
            """
            INSERT INTO tbl_l1_ngram_probs(context_hash, next_token, probability)
            VALUES (?, ?, ?)
            """,
            rows,
        )

    # ------------------------------------------------------------------ #
    # Generation helpers
    # ------------------------------------------------------------------ #
    def predict_next(self, context_tokens: Sequence[str], limit: int = 5) -> List[NGramPrediction]:
        context_window = list(context_tokens)[-(self.order - 1) :]
        context_hash = self.db.hash_tokens(context_window)
        rows = self.db.query(
            """
            SELECT next_token, probability
            FROM tbl_l1_ngram_probs
            WHERE context_hash = ?
            ORDER BY probability DESC
            LIMIT ?
            """,
            (context_hash, limit),
        )
        if not rows:
            rows = self.db.query(
                """
                SELECT next_token, probability
                FROM tbl_l1_ngram_probs
                WHERE context_hash = ?
                ORDER BY probability DESC
                LIMIT ?
                """,
                ("__default__", limit),
            )
        return [NGramPrediction(row["next_token"], row["probability"]) for row in rows]

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
