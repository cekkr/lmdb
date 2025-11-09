from __future__ import annotations

import hashlib
import sqlite3
from pathlib import Path
from typing import Iterable, Sequence


class DatabaseEnvironment:
    """
    Bootstraps the relational layout that mirrors the three-level DB-SLM blueprint.

    MariaDB engine choices (Aria/MyRocks/InnoDB) are simulated by storing an `engine_hint`
    column so that the logical split is preserved even though SQLite acts as the backing store.
    """

    def __init__(self, path: str | Path = ":memory:") -> None:
        self.path = str(path)
        self._conn = sqlite3.connect(self.path)
        self._conn.row_factory = sqlite3.Row
        self._bootstrap_schema()

    # --------------------------------------------------------------------- #
    # Schema
    # --------------------------------------------------------------------- #
    def _bootstrap_schema(self) -> None:
        cur = self._conn.cursor()
        cur.executescript(
            """
            CREATE TABLE IF NOT EXISTS tbl_l1_ngram_probs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                context_hash TEXT NOT NULL,
                next_token TEXT NOT NULL,
                probability REAL NOT NULL,
                engine_hint TEXT NOT NULL DEFAULT 'Aria'
            );
            CREATE INDEX IF NOT EXISTS idx_l1_context ON tbl_l1_ngram_probs(context_hash);

            CREATE TABLE IF NOT EXISTS tbl_l2_conversations (
                id TEXT PRIMARY KEY,
                user_id TEXT,
                agent_name TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                engine_hint TEXT NOT NULL DEFAULT 'InnoDB'
            );

            CREATE TABLE IF NOT EXISTS tbl_l2_messages (
                id TEXT PRIMARY KEY,
                conversation_id TEXT NOT NULL,
                sender TEXT NOT NULL CHECK(sender IN ('user', 'assistant')),
                content TEXT NOT NULL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                engine_hint TEXT NOT NULL DEFAULT 'MyRocks'
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
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                engine_hint TEXT NOT NULL DEFAULT 'InnoDB'
            );

            CREATE TABLE IF NOT EXISTS tbl_l3_concept_repo (
                concept_id INTEGER PRIMARY KEY AUTOINCREMENT,
                concept_name TEXT UNIQUE NOT NULL,
                metadata_schema TEXT NOT NULL,
                engine_hint TEXT NOT NULL DEFAULT 'InnoDB'
            );

            CREATE TABLE IF NOT EXISTS tbl_l3_verbal_templates (
                template_id INTEGER PRIMARY KEY AUTOINCREMENT,
                concept_id INTEGER NOT NULL,
                template_string TEXT NOT NULL,
                language_code TEXT NOT NULL DEFAULT 'en',
                engine_hint TEXT NOT NULL DEFAULT 'InnoDB',
                FOREIGN KEY(concept_id) REFERENCES tbl_l3_concept_repo(concept_id)
            );
            CREATE INDEX IF NOT EXISTS idx_l3_templates
                ON tbl_l3_verbal_templates(concept_id, language_code);

            CREATE TABLE IF NOT EXISTS tbl_l3_concept_probs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                context_hash TEXT NOT NULL,
                next_concept_id INTEGER NOT NULL,
                quantized_prob REAL NOT NULL,
                engine_hint TEXT NOT NULL DEFAULT 'Aria',
                FOREIGN KEY(next_concept_id) REFERENCES tbl_l3_concept_repo(concept_id)
            );
            CREATE INDEX IF NOT EXISTS idx_l3_context
                ON tbl_l3_concept_probs(context_hash);
            """
        )
        self._conn.commit()

    # --------------------------------------------------------------------- #
    # Helpers
    # --------------------------------------------------------------------- #
    def close(self) -> None:
        self._conn.close()

    def execute(self, sql: str, params: Sequence | None = None) -> None:
        self._conn.execute(sql, params or [])
        self._conn.commit()

    def insert_with_id(self, sql: str, params: Sequence | None = None) -> int:
        cursor = self._conn.execute(sql, params or [])
        self._conn.commit()
        return cursor.lastrowid

    def executemany(self, sql: str, params: Iterable[Sequence]) -> None:
        self._conn.executemany(sql, params)
        self._conn.commit()

    def query(self, sql: str, params: Sequence | None = None):
        cur = self._conn.execute(sql, params or [])
        rows = cur.fetchall()
        cur.close()
        return rows

    @staticmethod
    def hash_tokens(tokens: Sequence[str]) -> str:
        normalized = "|".join(token.lower() for token in tokens)
        digest = hashlib.sha1(normalized.encode("utf-8")).hexdigest()
        return digest or "__empty__"
