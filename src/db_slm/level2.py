from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from typing import Dict, Iterable, List

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
    """
    Implements the Level 2 memory layer: episodic logging plus correctional RAG.
    """

    def __init__(self, db: DatabaseEnvironment) -> None:
        self.db = db

    # ------------------------------------------------------------------ #
    # Conversations & messages
    # ------------------------------------------------------------------ #
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
        return message_id

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
        return [
            Message(row["id"], row["conversation_id"], row["sender"], row["content"])
            for row in rows
        ]

    # ------------------------------------------------------------------ #
    # Corrections
    # ------------------------------------------------------------------ #
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

    def lookup_corrections(
        self, conversation_id: str, context_snippet: str, limit: int = 5, match_any: bool = False
    ) -> List[Correction]:
        snippet = context_snippet or ""
        placeholder = "" if match_any else snippet
        rows = self.db.query(
            """
            SELECT correction_id, conversation_id, corrected_fact_json
            FROM tbl_l2_correction_log
            WHERE conversation_id = ?
              AND (
                    error_context IS NULL
                    OR error_context = ''
                    OR ? = ''
                    OR ? LIKE '%' || error_context || '%'
                  )
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (conversation_id, placeholder, placeholder, limit),
        )
        corrections: list[Correction] = []
        for row in rows:
            payload = json.loads(row["corrected_fact_json"]) if row["corrected_fact_json"] else {}
            corrections.append(Correction(row["correction_id"], row["conversation_id"], payload))
        return corrections

    # ------------------------------------------------------------------ #
    # Convenience helpers
    # ------------------------------------------------------------------ #
    def context_window(self, conversation_id: str, limit: int = 6) -> str:
        """
        Return the last `limit` messages in chronological order as a single string.
        """
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
            (conversation_id, limit),
        )
        chunks: list[str] = []
        for row in rows:
            prefix = "User" if row["sender"] == "user" else "Assistant"
            chunks.append(f"{prefix}: {row['content']}")
        return "\n".join(chunks)

    def conversation_stats(self, conversation_id: str) -> Dict[str, int | str | None]:
        """
        Aggregate lightweight metrics leveraged by higher-level engines.
        """
        rows = self.db.query(
            """
            SELECT
                COUNT(*) AS message_count,
                SUM(CASE WHEN sender = 'user' THEN 1 ELSE 0 END) AS user_turns,
                SUM(CASE WHEN sender = 'assistant' THEN 1 ELSE 0 END) AS assistant_turns,
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

    def correction_digest(self, conversation_id: str, limit: int = 5) -> List[Correction]:
        """
        Retrieve the newest corrections irrespective of context filters. Used to
        bias concept prediction toward known adjustments.
        """
        return self.lookup_corrections(conversation_id, "", limit=limit, match_any=True)
