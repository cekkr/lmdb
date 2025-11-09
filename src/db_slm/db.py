from __future__ import annotations

import contextlib
import hashlib
import sqlite3
from pathlib import Path
from typing import Generator, Iterable, Sequence


class DatabaseEnvironment:
    """SQLite environment that mirrors the relational layout from the DB‑SLM spec."""

    def __init__(self, path: str | Path = ":memory:", max_order: int = 3) -> None:
        self.path = str(path)
        self.max_order = max(2, max_order)
        self._conn = sqlite3.connect(self.path, isolation_level=None, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL;")
        self._conn.execute("PRAGMA synchronous=NORMAL;")
        self._bootstrap_schema()

    # ------------------------------------------------------------------ #
    # Schema management
    # ------------------------------------------------------------------ #
    def _bootstrap_schema(self) -> None:
        cur = self._conn.cursor()
        cur.executescript(
            """
            CREATE TABLE IF NOT EXISTS tbl_metadata (
                key TEXT PRIMARY KEY,
                value TEXT
            );

            CREATE TABLE IF NOT EXISTS tbl_l1_vocabulary (
                token_id      INTEGER PRIMARY KEY AUTOINCREMENT,
                token_text    TEXT UNIQUE NOT NULL,
                is_control    INTEGER NOT NULL DEFAULT 0,
                freq_global   INTEGER NOT NULL DEFAULT 0,
                byte_value    INTEGER,
                engine_hint   TEXT NOT NULL DEFAULT 'Aria'
            );
            CREATE TABLE IF NOT EXISTS tbl_token_normalization (
                raw TEXT PRIMARY KEY,
                norm TEXT NOT NULL,
                reason TEXT
            );

            CREATE TABLE IF NOT EXISTS tbl_l1_context_registry (
                context_hash  TEXT PRIMARY KEY,
                order_size    INTEGER NOT NULL,
                token_ids     TEXT NOT NULL,
                parent_hash   TEXT,
                total_count   INTEGER NOT NULL DEFAULT 0,
                hot_rank      REAL NOT NULL DEFAULT 0.0,
                updated_at    TEXT DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS tbl_l1_counts_of_counts (
                n_order     INTEGER NOT NULL,
                c_value     INTEGER NOT NULL,
                num_ngrams  INTEGER NOT NULL,
                PRIMARY KEY (n_order, c_value)
            );

            CREATE TABLE IF NOT EXISTS tbl_l1_continuations (
                token_id     INTEGER PRIMARY KEY,
                num_contexts INTEGER NOT NULL,
                last_rebuild TEXT DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS tbl_l1_mkn_params (
                n_order        INTEGER PRIMARY KEY,
                D1             REAL NOT NULL,
                D2             REAL NOT NULL,
                D3p            REAL NOT NULL,
                total_contexts INTEGER NOT NULL,
                total_types    INTEGER NOT NULL,
                built_at       TEXT DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS tbl_quant_meta (
                name TEXT PRIMARY KEY,
                Lmin REAL NOT NULL,
                Lmax REAL NOT NULL
            );

            CREATE TABLE IF NOT EXISTS tbl_q_to_mass (
                q TINYINT PRIMARY KEY,
                prob REAL NOT NULL,
                log10 REAL NOT NULL
            );

            CREATE TABLE IF NOT EXISTS tbl_temp_lut (
                temp REAL NOT NULL,
                q_in  TINYINT NOT NULL,
                q_out TINYINT NOT NULL,
                PRIMARY KEY (temp, q_in)
            );

            CREATE TABLE IF NOT EXISTS tbl_l1_session_cache (
                conversation_id TEXT NOT NULL,
                token_id        INTEGER NOT NULL,
                recency_weight  REAL NOT NULL,
                PRIMARY KEY (conversation_id, token_id)
            );

            CREATE TABLE IF NOT EXISTS tbl_decode_hparams (
                profile   TEXT PRIMARY KEY,
                lambda_cache REAL NOT NULL DEFAULT 0.15,
                temp      REAL NOT NULL DEFAULT 1.0,
                topk      INTEGER NOT NULL DEFAULT 20,
                topp      REAL NOT NULL DEFAULT 0.9
            );

            CREATE TABLE IF NOT EXISTS tbl_decode_bans (
                profile   TEXT NOT NULL,
                token_id  INTEGER NOT NULL,
                PRIMARY KEY (profile, token_id)
            );

            CREATE TABLE IF NOT EXISTS tbl_l2_conversations (
                id TEXT PRIMARY KEY,
                user_id TEXT,
                agent_name TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS tbl_l2_messages (
                id TEXT PRIMARY KEY,
                conversation_id TEXT NOT NULL,
                sender TEXT NOT NULL CHECK(sender IN ('user','assistant')),
                content TEXT NOT NULL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            );
            CREATE INDEX IF NOT EXISTS idx_l2_messages_conv
                ON tbl_l2_messages(conversation_id, created_at);

            CREATE TABLE IF NOT EXISTS tbl_l2_correction_log (
                correction_id TEXT PRIMARY KEY,
                conversation_id TEXT NOT NULL,
                error_message_id TEXT NOT NULL,
                correction_message_id TEXT NOT NULL,
                error_context TEXT,
                corrected_fact_json TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS tbl_l2_token_bias (
                conversation_id TEXT,
                pattern TEXT,
                token_id INTEGER NOT NULL,
                q_bias INTEGER NOT NULL,
                expires_at TEXT,
                PRIMARY KEY (conversation_id, pattern, token_id)
            );

            CREATE TABLE IF NOT EXISTS tbl_l2_window_cache (
                conversation_id TEXT PRIMARY KEY,
                window_text TEXT NOT NULL,
                updated_at TEXT DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS tbl_l3_concept_repo (
                concept_id INTEGER PRIMARY KEY AUTOINCREMENT,
                concept_name TEXT UNIQUE NOT NULL,
                metadata_schema TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS tbl_l3_verbal_templates (
                template_id INTEGER PRIMARY KEY AUTOINCREMENT,
                concept_id INTEGER NOT NULL,
                template_string TEXT NOT NULL,
                language_code TEXT NOT NULL DEFAULT 'en',
                FOREIGN KEY(concept_id) REFERENCES tbl_l3_concept_repo(concept_id)
            );

            CREATE TABLE IF NOT EXISTS tbl_l3_concept_probs (
                context_hash TEXT NOT NULL,
                concept_id INTEGER NOT NULL,
                q_logprob INTEGER NOT NULL,
                PRIMARY KEY (context_hash, concept_id)
            );

            CREATE TABLE IF NOT EXISTS tbl_l3_concept_signals (
                signal_id TEXT PRIMARY KEY,
                conversation_id TEXT NOT NULL,
                concept_id INTEGER NOT NULL,
                score REAL NOT NULL,
                expires_at TEXT,
                consume_once INTEGER NOT NULL DEFAULT 1,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            );
            CREATE INDEX IF NOT EXISTS idx_l3_signals_conv
                ON tbl_l3_concept_signals(conversation_id, score DESC);
            """
        )
        cur.close()
        for order in range(1, self.max_order + 1):
            self.ensure_order_tables(order)
        self._ensure_metadata_defaults()

    def ensure_order_tables(self, order: int) -> None:
        ctx_table = f"tbl_l1_ng_counts_{order}"
        prob_table = f"tbl_l1_ng_probs_{order}"
        topk_table = f"tbl_l1_ng_topk_{order}"
        cur = self._conn.cursor()
        cur.executescript(
            f"""
            CREATE TABLE IF NOT EXISTS {ctx_table} (
                context_hash TEXT NOT NULL,
                next_token_id INTEGER NOT NULL,
                count INTEGER NOT NULL,
                updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (context_hash, next_token_id)
            );
            CREATE INDEX IF NOT EXISTS idx_counts_{order}_ctx
                ON {ctx_table}(context_hash, count DESC);

            CREATE TABLE IF NOT EXISTS {prob_table} (
                context_hash TEXT NOT NULL,
                next_token_id INTEGER NOT NULL,
                q_logprob INTEGER NOT NULL,
                backoff_alpha INTEGER,
                PRIMARY KEY (context_hash, next_token_id)
            );
            CREATE INDEX IF NOT EXISTS idx_probs_{order}_ctx
                ON {prob_table}(context_hash, q_logprob DESC);

            CREATE TABLE IF NOT EXISTS {topk_table} (
                context_hash TEXT NOT NULL,
                k_rank INTEGER NOT NULL,
                next_token_id INTEGER NOT NULL,
                q_logprob INTEGER NOT NULL,
                PRIMARY KEY (context_hash, k_rank)
            );
            """
        )
        cur.close()

    def _ensure_metadata_defaults(self) -> None:
        rows = self.query("SELECT 1 FROM tbl_quant_meta WHERE name = 'default' LIMIT 1")
        if not rows:
            self.execute(
                "INSERT INTO tbl_quant_meta(name, Lmin, Lmax) VALUES (?,?,?)",
                ("default", -12.0, 0.0),
            )
        # Build q lookup table once.
        if not self.query("SELECT 1 FROM tbl_q_to_mass LIMIT 1"):
            self._populate_q_table()

    def _populate_q_table(self) -> None:
        meta = self.query("SELECT Lmin, Lmax FROM tbl_quant_meta WHERE name='default'")[0]
        Lmin, Lmax = meta["Lmin"], meta["Lmax"]
        rows = []
        for q in range(256):
            ratio = q / 255
            log10_val = Lmin + ratio * (Lmax - Lmin)
            prob = 10 ** log10_val
            rows.append((q, prob, log10_val))
        self.executemany("INSERT OR REPLACE INTO tbl_q_to_mass(q, prob, log10) VALUES (?, ?, ?)", rows)

    # ------------------------------------------------------------------ #
    # Basic query helpers
    # ------------------------------------------------------------------ #
    def close(self) -> None:
        self._conn.close()

    def execute(self, sql: str, params: Sequence | None = None) -> None:
        self._conn.execute(sql, params or [])
        self._conn.commit()

    def executemany(self, sql: str, params_seq: Iterable[Sequence]) -> None:
        self._conn.executemany(sql, params_seq)
        self._conn.commit()

    def query(self, sql: str, params: Sequence | None = None):
        cur = self._conn.execute(sql, params or [])
        rows = cur.fetchall()
        cur.close()
        return rows

    def insert_with_id(self, sql: str, params: Sequence | None = None) -> int:
        cur = self._conn.execute(sql, params or [])
        self._conn.commit()
        return cur.lastrowid

    @contextlib.contextmanager
    def transaction(self) -> Generator[sqlite3.Connection, None, None]:
        try:
            self._conn.execute("BEGIN IMMEDIATE")
            yield self._conn
            self._conn.commit()
        except Exception:
            self._conn.rollback()
            raise

    # ------------------------------------------------------------------ #
    # Hash helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def hash_tokens(token_ids: Sequence[int]) -> str:
        if not token_ids:
            return "__root__"
        raw = ",".join(str(tok) for tok in token_ids).encode("utf-8")
        digest = hashlib.blake2b(raw, digest_size=8).hexdigest()
        return digest
