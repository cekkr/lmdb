from __future__ import annotations

import math
import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

from .db import DatabaseEnvironment

TOKEN_PATTERN = re.compile(r"\w+|[^\w\s]", re.UNICODE)


@dataclass(frozen=True)
class TokenCandidate:
    token_id: int
    token_text: str
    probability: float
    q_logprob: int


class Vocabulary:
    """SQLite-backed vocabulary with transparent caching."""

    def __init__(self, db: DatabaseEnvironment) -> None:
        self.db = db
        self._text_to_id: Dict[str, int] = {}
        self._id_to_text: Dict[int, str] = {}
        self._ensure_special_tokens()

    def _ensure_special_tokens(self) -> None:
        for token in ("<PAD>", "<BOS>", "<EOS>"):
            self.get_or_create(token, is_control=True)

    def get_or_create(self, token: str, is_control: bool = False) -> int:
        cached = self._text_to_id.get(token)
        if cached is not None:
            return cached
        rows = self.db.query(
            "SELECT token_id FROM tbl_l1_vocabulary WHERE token_text = ?",
            (token,),
        )
        if rows:
            token_id = rows[0]["token_id"]
        else:
            token_id = self.db.insert_with_id(
                """
                INSERT INTO tbl_l1_vocabulary(token_text, is_control, freq_global)
                VALUES (?, ?, 0)
                """,
                (token, 1 if is_control else 0),
            )
        self._text_to_id[token] = token_id
        self._id_to_text[token_id] = token
        return token_id

    def token_id(self, token: str) -> int:
        return self.get_or_create(token)

    def token_text(self, token_id: int) -> str:
        cached = self._id_to_text.get(token_id)
        if cached is not None:
            return cached
        rows = self.db.query(
            "SELECT token_text FROM tbl_l1_vocabulary WHERE token_id = ?",
            (token_id,),
        )
        if not rows:
            return f"<unk:{token_id}>"
        token_text = rows[0]["token_text"]
        self._id_to_text[token_id] = token_text
        self._text_to_id[token_text] = token_id
        return token_text

    def increment_frequency(self, token_id: int, delta: int = 1) -> None:
        self.db.execute(
            """
            UPDATE tbl_l1_vocabulary
            SET freq_global = freq_global + ?
            WHERE token_id = ?
            """,
            (delta, token_id),
        )


class Tokenizer:
    """Simple regex tokenizer wrapped around the vocabulary."""

    def __init__(self, vocab: Vocabulary) -> None:
        self.vocab = vocab

    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        tokens = [match.group(0).lower() for match in TOKEN_PATTERN.finditer(text)]
        if add_special_tokens:
            tokens = ["<BOS>", *tokens, "<EOS>"]
        return [self.vocab.token_id(token) for token in tokens]

    def decode(self, token_ids: Iterable[int]) -> str:
        pieces: List[str] = []
        for token_id in token_ids:
            token = self.vocab.token_text(token_id)
            if token in {"<BOS>", "<EOS>", "<PAD>"}:
                continue
            if not pieces:
                pieces.append(token)
            elif token.isalnum():
                pieces.append(f" {token}")
            else:
                pieces.append(token)
        return "".join(pieces).strip()

    def tokens_to_text(self, token_ids: Iterable[int]) -> List[str]:
        return [self.vocab.token_text(token_id) for token_id in token_ids]


class LogProbQuantizer:
    def __init__(self, db: DatabaseEnvironment) -> None:
        meta = db.query("SELECT Lmin, Lmax FROM tbl_quant_meta WHERE name='default'")[0]
        self.Lmin = meta["Lmin"]
        self.Lmax = meta["Lmax"]
        self._lookup = {
            row["q"]: (row["prob"], row["log10"]) for row in db.query("SELECT q, prob, log10 FROM tbl_q_to_mass")
        }

    def quantize(self, probability: float) -> int:
        probability = max(probability, 10 ** self.Lmin)
        log10_value = math.log10(probability)
        ratio = (log10_value - self.Lmin) / (self.Lmax - self.Lmin)
        ratio = min(max(ratio, 0.0), 1.0)
        return int(round(ratio * 255))

    def dequantize_prob(self, q: int) -> float:
        return self._lookup.get(q, (10 ** self.Lmin, self.Lmin))[0]

    def dequantize_log10(self, q: int) -> float:
        return self._lookup.get(q, (10 ** self.Lmin, self.Lmin))[1]


class NGramStore:
    """Handles ingestion and retrieval of Level 1 statistics."""

    def __init__(self, db: DatabaseEnvironment, vocab: Vocabulary, order: int, quantizer: LogProbQuantizer) -> None:
        self.db = db
        self.vocab = vocab
        self.order = order
        self.quantizer = quantizer

    # ------------------------------------------------------------------ #
    # Ingestion
    # ------------------------------------------------------------------ #
    def ingest(self, token_ids: Sequence[int]) -> None:
        if len(token_ids) < 2:
            return
        for token_id in token_ids:
            self.vocab.increment_frequency(token_id)
        for n in range(1, self.order + 1):
            if len(token_ids) < n:
                break
            table = self._counts_table(n)
            for idx in range(len(token_ids) - n + 1):
                ngram = token_ids[idx : idx + n]
                context = ngram[:-1]
                next_token = ngram[-1]
                context_hash = self._hash(context)
                self._upsert_context(context_hash, context)
                self.db.execute(
                    f"""
                    INSERT INTO {table}(context_hash, next_token_id, count)
                    VALUES (?, ?, 1)
                    ON CONFLICT(context_hash, next_token_id) DO UPDATE
                    SET count = count + 1,
                        updated_at = CURRENT_TIMESTAMP
                    """,
                    (context_hash, next_token),
                )
                self.db.execute(
                    """
                    UPDATE tbl_l1_context_registry
                    SET total_count = total_count + 1,
                        updated_at = CURRENT_TIMESTAMP
                    WHERE context_hash = ?
                    """,
                    (context_hash,),
                )

    def _counts_table(self, order: int) -> str:
        return f"tbl_l1_ng_counts_{order}"

    def _prob_table(self, order: int) -> str:
        return f"tbl_l1_ng_probs_{order}"

    def _topk_table(self, order: int) -> str:
        return f"tbl_l1_ng_topk_{order}"

    def _hash(self, tokens: Sequence[int]) -> str:
        return self.db.hash_tokens(tokens)

    def _upsert_context(self, context_hash: str, token_ids: Sequence[int]) -> None:
        parent = self.db.hash_tokens(token_ids[1:]) if token_ids else "__root__"
        token_blob = ",".join(str(tok) for tok in token_ids)
        self.db.execute(
            """
            INSERT INTO tbl_l1_context_registry(context_hash, order_size, token_ids, parent_hash, total_count, hot_rank)
            VALUES (?, ?, ?, ?, 0, 0.0)
            ON CONFLICT(context_hash) DO UPDATE SET
                order_size = excluded.order_size,
                token_ids = excluded.token_ids,
                parent_hash = excluded.parent_hash
            """,
            (context_hash, len(token_ids), token_blob, parent),
        )

    # ------------------------------------------------------------------ #
    # Retrieval
    # ------------------------------------------------------------------ #
    def get_topk(self, context_ids: Sequence[int], order: int, k: int) -> List[TokenCandidate]:
        context_hash = self._hash(context_ids[-(order - 1) :]) if order > 1 else "__root__"
        table = self._topk_table(order)
        rows = self.db.query(
            f"""
            SELECT next_token_id, q_logprob
            FROM {table}
            WHERE context_hash = ?
            ORDER BY k_rank ASC
            LIMIT ?
            """,
            (context_hash, k),
        )
        if not rows:
            fallback_table = self._prob_table(order)
            rows = self.db.query(
                f"""
                SELECT next_token_id, q_logprob
                FROM {fallback_table}
                WHERE context_hash = ?
                ORDER BY q_logprob DESC
                LIMIT ?
                """,
                (context_hash, k),
            )
        return [self._row_to_candidate(row) for row in rows]

    def _row_to_candidate(self, row) -> TokenCandidate:
        q = row["q_logprob"]
        prob = self.quantizer.dequantize_prob(q)
        token_id = row["next_token_id"]
        return TokenCandidate(token_id, self.vocab.token_text(token_id), prob, q)

    def fetch_context_tokens(self, context_hash: str) -> List[int]:
        if context_hash == "__root__":
            return []
        rows = self.db.query(
            """
            SELECT token_ids
            FROM tbl_l1_context_registry
            WHERE context_hash = ?
            """,
            (context_hash,),
        )
        if not rows:
            return []
        token_blob = rows[0]["token_ids"]
        if not token_blob:
            return []
        return [int(tok) for tok in token_blob.split(",") if tok]

    # ------------------------------------------------------------------ #
    # Materialization helpers
    # ------------------------------------------------------------------ #
    def iter_counts(self, order: int):
        table = self._counts_table(order)
        return self.db.query(
            f"""
            SELECT context_hash, next_token_id, count
            FROM {table}
            """
        )

    def clear_probabilities(self, order: int) -> None:
        self.db.execute(f"DELETE FROM {self._prob_table(order)}")
        self.db.execute(f"DELETE FROM {self._topk_table(order)}")

    def store_probabilities(
        self,
        order: int,
        rows: List[Tuple[str, int, int, int | None]],
        topk_rows: List[Tuple[str, int, int, int]],
    ) -> None:
        prob_table = self._prob_table(order)
        topk_table = self._topk_table(order)
        self.db.executemany(
            f"INSERT OR REPLACE INTO {prob_table}(context_hash, next_token_id, q_logprob, backoff_alpha) VALUES (?, ?, ?, ?)",
            rows,
        )
        self.db.executemany(
            f"INSERT OR REPLACE INTO {topk_table}(context_hash, k_rank, next_token_id, q_logprob) VALUES (?, ?, ?, ?)",
            topk_rows,
        )


class MKNSmoother:
    def __init__(self, db: DatabaseEnvironment, store: NGramStore, quantizer: LogProbQuantizer, topk: int = 20) -> None:
        self.db = db
        self.store = store
        self.quantizer = quantizer
        self.topk = topk

    def rebuild_all(self) -> None:
        self._rebuild_continuations()
        for order in range(1, self.store.order + 1):
            self.rebuild_order(order)

    def rebuild_order(self, order: int) -> None:
        rows = self.store.iter_counts(order)
        if not rows:
            return
        contexts: Dict[str, List[Tuple[int, int]]] = defaultdict(list)
        counts_of_counts: Dict[int, int] = defaultdict(int)
        for row in rows:
            context_hash = row["context_hash"]
            count = row["count"]
            contexts[context_hash].append((row["next_token_id"], count))
            bucket = 4 if count >= 4 else count
            counts_of_counts[bucket] += 1
        D1, D2, D3p = self._discounts(counts_of_counts)
        self._persist_counts_of_counts(order, counts_of_counts)
        self._persist_params(order, D1, D2, D3p, len(contexts))

        lower_lookup = self._lower_lookup(order)
        continuation_probs = self._continuation_lookup()

        prob_rows: list[tuple[str, int, int, int | None]] = []
        topk_rows: list[tuple[str, int, int, int]] = []
        for context_hash, followers in contexts.items():
            totals = sum(count for _, count in followers)
            if totals == 0:
                continue
            bucket_counts = self._bucket_counts(followers)
            alpha = self._backoff_alpha(bucket_counts, totals, D1, D2, D3p)
            ranked = []
            parent_hash = self._parent_hash(context_hash)
            for token_id, count in followers:
                discount = D1 if count == 1 else D2 if count == 2 else D3p
                base = max(count - discount, 0.0) / totals
                default_lower = continuation_probs.get(token_id, 1e-9)
                if order == 1:
                    lower = default_lower
                else:
                    lower = lower_lookup.get((parent_hash, token_id), default_lower)
                prob = base + alpha * lower
                q = self.quantizer.quantize(prob)
                prob_rows.append((context_hash, token_id, q, self._quantize_alpha(alpha)))
                ranked.append((token_id, q))
            ranked.sort(key=lambda item: item[1], reverse=True)
            for rank, (token_id, q) in enumerate(ranked[: self.topk], start=1):
                topk_rows.append((context_hash, rank, token_id, q))
        self.store.clear_probabilities(order)
        if prob_rows:
            self.store.store_probabilities(order, prob_rows, topk_rows)

    def _lower_lookup(self, order: int) -> Dict[Tuple[str, int], float]:
        if order == 1:
            return {}
        lower_table = self.store._prob_table(order - 1)
        rows = self.db.query(
            f"SELECT context_hash, next_token_id, q_logprob FROM {lower_table}"
        )
        lookup: Dict[Tuple[str, int], float] = {}
        for row in rows:
            lookup[(row["context_hash"], row["next_token_id"])] = self.quantizer.dequantize_prob(
                row["q_logprob"]
            )
        return lookup

    def _continuation_lookup(self) -> Dict[int, float]:
        rows = self.db.query("SELECT token_id, num_contexts FROM tbl_l1_continuations")
        totals = sum(row["num_contexts"] for row in rows) or 1
        return {row["token_id"]: row["num_contexts"] / totals for row in rows}

    def _parent_hash(self, context_hash: str) -> str:
        tokens = self.store.fetch_context_tokens(context_hash)
        if not tokens:
            return "__root__"
        return self.db.hash_tokens(tokens[1:])

    @staticmethod
    def _bucket_counts(followers: List[Tuple[int, int]]) -> Dict[int, int]:
        buckets: Dict[int, int] = {1: 0, 2: 0, 3: 0}
        for _, count in followers:
            if count == 1:
                buckets[1] += 1
            elif count == 2:
                buckets[2] += 1
            else:
                buckets[3] += 1
        return buckets

    @staticmethod
    def _backoff_alpha(buckets: Dict[int, int], total: int, D1: float, D2: float, D3p: float) -> float:
        numerator = D1 * buckets[1] + D2 * buckets[2] + D3p * buckets[3]
        return numerator / total if total else 0.0

    def _discounts(self, counts_of_counts: Dict[int, int]) -> Tuple[float, float, float]:
        N1 = counts_of_counts.get(1, 0)
        N2 = counts_of_counts.get(2, 0)
        N3 = counts_of_counts.get(3, 0)
        N4 = counts_of_counts.get(4, 0)
        D1 = 1 - (2 * N2) / (N1 + 2 * N2) if (N1 + 2 * N2) else 0.5
        D2 = 2 - (3 * N3) / (N2 + 2 * N3) if (N2 + 2 * N3) else 1.0
        D3p = 3 - (4 * N4) / (N3 + 2 * N4) if (N3 + 2 * N4) else 1.5
        return D1, D2, D3p

    def _persist_counts_of_counts(self, order: int, buckets: Dict[int, int]) -> None:
        payload = []
        for c_value, num in buckets.items():
            payload.append((order, c_value, num))
        self.db.execute("DELETE FROM tbl_l1_counts_of_counts WHERE n_order = ?", (order,))
        if payload:
            self.db.executemany(
                "INSERT INTO tbl_l1_counts_of_counts(n_order, c_value, num_ngrams) VALUES (?, ?, ?)",
                payload,
            )

    def _persist_params(self, order: int, D1: float, D2: float, D3p: float, total_contexts: int) -> None:
        total_types = sum(row["num_ngrams"] for row in self.db.query(
            "SELECT num_ngrams FROM tbl_l1_counts_of_counts WHERE n_order = ?",
            (order,),
        ))
        self.db.execute(
            """
            INSERT INTO tbl_l1_mkn_params(n_order, D1, D2, D3p, total_contexts, total_types, built_at)
            VALUES (?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT(n_order) DO UPDATE SET
                D1 = excluded.D1,
                D2 = excluded.D2,
                D3p = excluded.D3p,
                total_contexts = excluded.total_contexts,
                total_types = excluded.total_types,
                built_at = CURRENT_TIMESTAMP
            """,
            (order, D1, D2, D3p, total_contexts, total_types),
        )

    def _quantize_alpha(self, alpha: float) -> int:
        alpha = min(max(alpha, 0.0), 1.0)
        return int(round(alpha * 1000))

    def _rebuild_continuations(self) -> None:
        rows = self.db.query(
            """
            SELECT next_token_id, COUNT(DISTINCT context_hash) AS num_ctx
            FROM tbl_l1_ng_counts_2
            GROUP BY next_token_id
            """
        )
        if not rows:
            return
        self.db.execute("DELETE FROM tbl_l1_continuations")
        payload = [(row["next_token_id"], row["num_ctx"]) for row in rows]
        self.db.executemany(
            "INSERT INTO tbl_l1_continuations(token_id, num_contexts) VALUES (?, ?)",
            payload,
        )
