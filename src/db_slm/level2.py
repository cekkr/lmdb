from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Sequence

from .db import DatabaseEnvironment


@dataclass(frozen=True)
class Message:
    id: str
    conversation_id: str
    sender: str
    content: str


@dataclass(frozen=True)
class Correction:
    correction_id: str
    conversation_id: str
    payload: dict


class ConversationMemory:
    """Level 2 episodic store with cached rolling windows."""

    def __init__(self, db: DatabaseEnvironment, window: int = 8) -> None:
        self.db = db
        self.window = window

    def start_conversation(self, user_id: str, agent_name: str) -> str:
        conversation_id = str(uuid.uuid4())
        self.db.execute(
            """
            INSERT INTO tbl_l2_conversations(id, user_id, agent_name)
            VALUES (?, ?, ?)
            """,
            (conversation_id, user_id, agent_name),
        )
        return conversation_id

    def log_message(self, conversation_id: str, sender: str, content: str) -> str:
        message_id = str(uuid.uuid4())
        self.db.execute(
            """
            INSERT INTO tbl_l2_messages(id, conversation_id, sender, content)
            VALUES (?, ?, ?, ?)
            """,
            (message_id, conversation_id, sender, content),
        )
        self._refresh_window_cache(conversation_id)
        return message_id

    def _refresh_window_cache(self, conversation_id: str) -> None:
        window_text = self.context_window(conversation_id, use_cache=False)
        self.db.execute(
            """
            INSERT INTO tbl_l2_window_cache(conversation_id, window_text)
            VALUES (?, ?)
            ON CONFLICT(conversation_id) DO UPDATE SET
                window_text = excluded.window_text,
                updated_at = CURRENT_TIMESTAMP
            """,
            (conversation_id, window_text),
        )

    def context_window(self, conversation_id: str, use_cache: bool = True) -> str:
        if use_cache:
            rows = self.db.query(
                "SELECT window_text FROM tbl_l2_window_cache WHERE conversation_id = ?",
                (conversation_id,),
            )
            if rows:
                return rows[0]["window_text"]
        rows = self.db.query(
            """
            SELECT sender, content
            FROM (
                SELECT sender, content, created_at
                FROM tbl_l2_messages
                WHERE conversation_id = ?
                ORDER BY created_at DESC
                LIMIT ?
            )
            ORDER BY created_at
            """,
            (conversation_id, self.window),
        )
        lines = [f"{row['sender']}: {row['content']}" for row in rows]
        return "\n".join(lines)

    def recent_messages(self, conversation_id: str, limit: int = 10) -> List[Message]:
        rows = self.db.query(
            """
            SELECT id, conversation_id, sender, content
            FROM tbl_l2_messages
            WHERE conversation_id = ?
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (conversation_id, limit),
        )
        return [Message(row["id"], row["conversation_id"], row["sender"], row["content"]) for row in rows]

    def record_correction(
        self,
        conversation_id: str,
        error_message_id: str,
        correction_message_id: str,
        error_context: str,
        corrected_fact: dict,
    ) -> str:
        correction_id = str(uuid.uuid4())
        self.db.execute(
            """
            INSERT INTO tbl_l2_correction_log(
                correction_id,
                conversation_id,
                error_message_id,
                correction_message_id,
                error_context,
                corrected_fact_json
            )
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                correction_id,
                conversation_id,
                error_message_id,
                correction_message_id,
                error_context,
                json.dumps(corrected_fact),
            ),
        )
        return correction_id

    def lookup_corrections(self, conversation_id: str, limit: int = 5) -> List[Correction]:
        rows = self.db.query(
            """
            SELECT correction_id, conversation_id, corrected_fact_json
            FROM tbl_l2_correction_log
            WHERE conversation_id = ?
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (conversation_id, limit),
        )
        corrections: List[Correction] = []
        for row in rows:
            payload = json.loads(row["corrected_fact_json"]) if row["corrected_fact_json"] else {}
            corrections.append(Correction(row["correction_id"], row["conversation_id"], payload))
        return corrections

    def conversation_stats(self, conversation_id: str) -> Dict[str, int | str | None]:
        rows = self.db.query(
            """
            SELECT
                COUNT(*) AS message_count,
                SUM(CASE WHEN sender='user' THEN 1 ELSE 0 END) AS user_turns,
                SUM(CASE WHEN sender='assistant' THEN 1 ELSE 0 END) AS assistant_turns,
                MIN(created_at) AS started_at,
                MAX(created_at) AS updated_at
            FROM tbl_l2_messages
            WHERE conversation_id = ?
            """,
            (conversation_id,),
        )
        if not rows:
            return {
                "message_count": 0,
                "user_turns": 0,
                "assistant_turns": 0,
                "started_at": None,
                "updated_at": None,
            }
        row = rows[0]
        return {
            "message_count": row["message_count"] or 0,
            "user_turns": row["user_turns"] or 0,
            "assistant_turns": row["assistant_turns"] or 0,
            "started_at": row["started_at"],
            "updated_at": row["updated_at"],
        }


class SessionCache:
    """Pointer-sentinel style cache that adds recency bias per conversation."""

    def __init__(self, db: DatabaseEnvironment, decay: float = 0.8) -> None:
        self.db = db
        self.decay = decay
        self._ensure_profile()

    def _ensure_profile(self) -> None:
        if not self.db.query("SELECT 1 FROM tbl_decode_hparams WHERE profile='default' LIMIT 1"):
            self.db.execute(
                "INSERT INTO tbl_decode_hparams(profile, lambda_cache, temp, topk, topp) VALUES ('default', 0.15, 1.0, 20, 0.9)"
            )

    def update(self, conversation_id: str, token_ids: Sequence[int]) -> None:
        rows = self.db.query(
            "SELECT token_id, recency_weight FROM tbl_l1_session_cache WHERE conversation_id = ?",
            (conversation_id,),
        )
        weights = {row["token_id"]: row["recency_weight"] * self.decay for row in rows}
        for token_id in token_ids:
            weights[token_id] = weights.get(token_id, 0.0) + 1.0
        self.db.execute("DELETE FROM tbl_l1_session_cache WHERE conversation_id = ?", (conversation_id,))
        payload = [(conversation_id, token_id, weight) for token_id, weight in weights.items() if weight > 1e-6]
        if payload:
            self.db.executemany(
                "INSERT INTO tbl_l1_session_cache(conversation_id, token_id, recency_weight) VALUES (?, ?, ?)",
                payload,
            )

    def distribution(self, conversation_id: str) -> Dict[int, float]:
        rows = self.db.query(
            "SELECT token_id, recency_weight FROM tbl_l1_session_cache WHERE conversation_id = ?",
            (conversation_id,),
        )
        total = sum(row["recency_weight"] for row in rows) or 1.0
        return {row["token_id"]: row["recency_weight"] / total for row in rows}

    def mixture_lambda(self, profile: str = "default") -> float:
        rows = self.db.query(
            "SELECT lambda_cache FROM tbl_decode_hparams WHERE profile = ?",
            (profile,),
        )
        return rows[0]["lambda_cache"] if rows else 0.15

    def decode_profile(self, profile: str = "default") -> Dict[str, float | int]:
        rows = self.db.query(
            "SELECT lambda_cache, temp, topk, topp FROM tbl_decode_hparams WHERE profile = ?",
            (profile,),
        )
        if rows:
            return dict(rows[0])
        return {"lambda_cache": 0.15, "temp": 1.0, "topk": 20, "topp": 0.9}


class BiasEngine:
    def __init__(self, db: DatabaseEnvironment) -> None:
        self.db = db

    def upsert_bias(
        self,
        conversation_id: str,
        pattern: str,
        token_id: int,
        q_bias: int,
        ttl_seconds: int | None = None,
    ) -> None:
        expires_at = None
        if ttl_seconds:
            expires_at = (datetime.now(timezone.utc) + timedelta(seconds=ttl_seconds)).isoformat(timespec="seconds")
        self.db.execute(
            """
            INSERT INTO tbl_l2_token_bias(conversation_id, pattern, token_id, q_bias, expires_at)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(conversation_id, pattern, token_id) DO UPDATE SET
                q_bias = excluded.q_bias,
                expires_at = excluded.expires_at
            """,
            (conversation_id, pattern or "", token_id, q_bias, expires_at),
        )

    def lookup(self, conversation_id: str, context_snippet: str) -> Dict[int, int]:
        rows = self.db.query(
            """
            SELECT pattern, token_id, q_bias, expires_at
            FROM tbl_l2_token_bias
            WHERE conversation_id = ? OR conversation_id IS NULL
            """,
            (conversation_id,),
        )
        active: Dict[int, int] = {}
        now = datetime.now(timezone.utc)
        for row in rows:
            expires_at = row["expires_at"]
            if expires_at:
                try:
                    expires_dt = datetime.fromisoformat(expires_at)
                except ValueError:
                    continue
                if expires_dt.replace(tzinfo=timezone.utc) < now:
                    continue
            pattern = row["pattern"] or ""
            if pattern and pattern.lower() not in context_snippet.lower():
                continue
            token_id = row["token_id"]
            active[token_id] = active.get(token_id, 0) + row["q_bias"]
        return active
