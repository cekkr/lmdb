from __future__ import annotations

import time
from pathlib import Path
from typing import Iterable, List, Sequence, TYPE_CHECKING

from .settings import DBSLMSettings

if TYPE_CHECKING:  # pragma: no cover - type narrowing only
    from .pipeline import DBSLMEngine


class ColdStorageFlusher:
    """Ships low-frequency SQLite contexts to MariaDB when memory grows too much."""

    def __init__(
        self,
        engine: "DBSLMEngine",
        settings: DBSLMSettings,
        *,
        threshold_mb: int | None = None,
        cold_threshold: int | None = None,
        batch_size: int | None = None,
        cooldown_seconds: int = 180,
    ) -> None:
        self.engine = engine
        self.settings = settings
        self.threshold_mb = threshold_mb or settings.sqlite_flush_threshold_mb
        self.cold_threshold = cold_threshold or settings.sqlite_flush_cold_threshold
        self.batch_size = batch_size or settings.sqlite_flush_batch_size
        self.cooldown = max(cooldown_seconds, 30)
        self._last_flush = 0.0
        self._mysql_warned = False

    def maybe_flush(self) -> bool:
        if not self._eligible():
            return False
        size_mb = self._sqlite_size_mb()
        if size_mb < self.threshold_mb:
            return False
        if time.monotonic() - self._last_flush < self.cooldown:
            return False
        contexts = self._select_cold_contexts()
        if not contexts:
            return False
        if not self._ship_to_mysql(contexts):
            return False
        self._purge_sqlite(contexts)
        self._last_flush = time.monotonic()
        print(
            f"[flush] Migrated {len(contexts)} cold context(s) to MariaDB; "
            f"SQLite file now ~{self._sqlite_size_mb():.1f}MB."
        )
        return True

    def _eligible(self) -> bool:
        path = self.engine.db.path
        if path == ":memory:":
            return False
        if self.settings.backend != "sqlite":
            return False
        return True

    def _sqlite_size_mb(self) -> float:
        try:
            path = Path(self.engine.db.path)
            return path.stat().st_size / (1024 * 1024)
        except (FileNotFoundError, OSError):
            return 0.0

    def _select_cold_contexts(self) -> List[str]:
        rows = self.engine.db.query(
            """
            SELECT context_hash
            FROM tbl_l1_context_registry
            WHERE total_count <= ?
            ORDER BY hot_rank ASC, total_count ASC, updated_at ASC
            LIMIT ?
            """,
            (self.cold_threshold, self.batch_size),
        )
        return [row["context_hash"] for row in rows if row["context_hash"] not in {None, "__root__"}]

    def _ship_to_mysql(self, contexts: Sequence[str]) -> bool:
        try:
            import mysql.connector  # type: ignore
        except ImportError:
            if not self._mysql_warned:
                print("[flush] mysql-connector-python missing; cannot flush cold contexts to MariaDB.")
                self._mysql_warned = True
            return False
        conn = mysql.connector.connect(
            host=self.settings.mariadb_host,
            port=self.settings.mariadb_port,
            user=self.settings.mariadb_user,
            password=self.settings.mariadb_password,
            database=self.settings.mariadb_database,
        )
        cursor = conn.cursor()
        self._ensure_tables(cursor)
        placeholders = ",".join(["?"] * len(contexts))
        context_rows = self.engine.db.query(
            f"""
            SELECT context_hash, order_size, token_ids, parent_hash, total_count, hot_rank, updated_at
            FROM tbl_l1_context_registry
            WHERE context_hash IN ({placeholders})
            """,
            contexts,
        )
        self._insert_contexts(cursor, context_rows)
        for order in range(1, self.engine.store.order + 1):
            counts = self._fetch_rows(
                f"tbl_l1_ng_counts_{order}",
                ("context_hash", "next_token_id", "count", "updated_at"),
                contexts,
            )
            probs = self._fetch_rows(
                f"tbl_l1_ng_probs_{order}",
                ("context_hash", "next_token_id", "q_logprob", "backoff_alpha"),
                contexts,
            )
            topk = self._fetch_rows(
                f"tbl_l1_ng_topk_{order}",
                ("context_hash", "k_rank", "next_token_id", "q_logprob"),
                contexts,
            )
            self._insert_counts(cursor, order, counts)
            self._insert_probs(cursor, order, probs)
            self._insert_topk(cursor, order, topk)
        conn.commit()
        cursor.close()
        conn.close()
        return True

    def _fetch_rows(
        self, table: str, columns: Sequence[str], contexts: Sequence[str]
    ) -> List[Sequence]:
        if not contexts:
            return []
        placeholders = ",".join(["?"] * len(contexts))
        cols = ", ".join(columns)
        return self.engine.db.query(
            f"""
            SELECT {cols}
            FROM {table}
            WHERE context_hash IN ({placeholders})
            """,
            contexts,
        )

    def _insert_contexts(self, cursor, rows: Iterable[Sequence]) -> None:
        if not rows:
            return
        cursor.executemany(
            """
            INSERT INTO tbl_l1_context_registry
                (context_hash, order_size, token_ids, parent_hash, total_count, hot_rank, updated_at)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            ON DUPLICATE KEY UPDATE
                order_size=VALUES(order_size),
                token_ids=VALUES(token_ids),
                parent_hash=VALUES(parent_hash),
                total_count=VALUES(total_count),
                hot_rank=VALUES(hot_rank),
                updated_at=VALUES(updated_at)
            """,
            [tuple(row) for row in rows],
        )

    def _insert_counts(self, cursor, order: int, rows: Iterable[Sequence]) -> None:
        if not rows:
            return
        cursor.executemany(
            f"""
            INSERT INTO tbl_l1_ng_counts_{order}
                (context_hash, next_token_id, count, updated_at)
            VALUES (%s, %s, %s, %s)
            ON DUPLICATE KEY UPDATE
                count=VALUES(count),
                updated_at=VALUES(updated_at)
            """,
            [tuple(row) for row in rows],
        )

    def _insert_probs(self, cursor, order: int, rows: Iterable[Sequence]) -> None:
        if not rows:
            return
        cursor.executemany(
            f"""
            INSERT INTO tbl_l1_ng_probs_{order}
                (context_hash, next_token_id, q_logprob, backoff_alpha)
            VALUES (%s, %s, %s, %s)
            ON DUPLICATE KEY UPDATE
                q_logprob=VALUES(q_logprob),
                backoff_alpha=VALUES(backoff_alpha)
            """,
            [tuple(row) for row in rows],
        )

    def _insert_topk(self, cursor, order: int, rows: Iterable[Sequence]) -> None:
        if not rows:
            return
        cursor.executemany(
            f"""
            INSERT INTO tbl_l1_ng_topk_{order}
                (context_hash, k_rank, next_token_id, q_logprob)
            VALUES (%s, %s, %s, %s)
            ON DUPLICATE KEY UPDATE
                next_token_id=VALUES(next_token_id),
                q_logprob=VALUES(q_logprob)
            """,
            [tuple(row) for row in rows],
        )

    def _ensure_tables(self, cursor) -> None:
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS tbl_l1_context_registry (
                context_hash VARCHAR(32) PRIMARY KEY,
                order_size INT NOT NULL,
                token_ids TEXT NOT NULL,
                parent_hash VARCHAR(32),
                total_count BIGINT NOT NULL,
                hot_rank DOUBLE NOT NULL,
                updated_at DATETIME NULL
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
            """
        )
        for order in range(1, self.engine.store.order + 1):
            cursor.execute(
                f"""
                CREATE TABLE IF NOT EXISTS tbl_l1_ng_counts_{order} (
                    context_hash VARCHAR(32) NOT NULL,
                    next_token_id BIGINT NOT NULL,
                    count BIGINT NOT NULL,
                    updated_at DATETIME NULL,
                    PRIMARY KEY (context_hash, next_token_id)
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
                """
            )
            cursor.execute(
                f"""
                CREATE TABLE IF NOT EXISTS tbl_l1_ng_probs_{order} (
                    context_hash VARCHAR(32) NOT NULL,
                    next_token_id BIGINT NOT NULL,
                    q_logprob INT NOT NULL,
                    backoff_alpha INT,
                    PRIMARY KEY (context_hash, next_token_id)
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
                """
            )
            cursor.execute(
                f"""
                CREATE TABLE IF NOT EXISTS tbl_l1_ng_topk_{order} (
                    context_hash VARCHAR(32) NOT NULL,
                    k_rank INT NOT NULL,
                    next_token_id BIGINT NOT NULL,
                    q_logprob INT NOT NULL,
                    PRIMARY KEY (context_hash, k_rank)
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
                """
            )

    def _purge_sqlite(self, contexts: Sequence[str]) -> None:
        if not contexts:
            return
        placeholders = ",".join(["?"] * len(contexts))
        for order in range(1, self.engine.store.order + 1):
            for table in (
                f"tbl_l1_ng_counts_{order}",
                f"tbl_l1_ng_probs_{order}",
                f"tbl_l1_ng_topk_{order}",
            ):
                self.engine.db.execute(
                    f"DELETE FROM {table} WHERE context_hash IN ({placeholders})",
                    contexts,
                )
        self.engine.db.execute(
            f"DELETE FROM tbl_l1_context_registry WHERE context_hash IN ({placeholders})",
            contexts,
        )
