from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Sequence, TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from .adapters.base import HotPathAdapter

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

    def __init__(
        self,
        db: DatabaseEnvironment,
        window: int = 8,
        *,
        hot_path: "HotPathAdapter | None" = None,
        correction_cache: int = 5,
    ) -> None:
        self.db = db
        self.window = window
        self.hot_path = hot_path
        self._correction_cache = max(1, correction_cache)

    def start_conversation(self, user_id: str, agent_name: str) -> str:
        conversation_id = str(uuid.uuid4())
        self.db.execute(
            """
            INSERT INTO tbl_l2_conversations(id, user_id, agent_name)
            VALUES (?, ?, ?)
            """,
            (conversation_id, user_id, agent_name),
        )
        self._write_metadata(self._stats_metadata_key(conversation_id), self._empty_stats())
        self._write_metadata(self._correction_metadata_key(conversation_id), [])
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
        self._mirror_stats(conversation_id)
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
        self._mirror_corrections(conversation_id)
        return correction_id

    def lookup_corrections(self, conversation_id: str, limit: int = 5) -> List[Correction]:
        cache_key = self._correction_metadata_key(conversation_id)
        cached = self._read_metadata(cache_key)
        corrections = self._deserialize_corrections(conversation_id, cached)
        if corrections is not None:
            return corrections[:limit]
        fetch_limit = max(limit, self._correction_cache)
        corrections = self._query_corrections(conversation_id, fetch_limit)
        self._write_metadata(
            cache_key,
            [
                {"correction_id": item.correction_id, "payload": item.payload}
                for item in corrections[: self._correction_cache]
            ],
        )
        return corrections[:limit]

    def conversation_stats(self, conversation_id: str) -> Dict[str, int | str | None]:
        cache_key = self._stats_metadata_key(conversation_id)
        cached = self._deserialize_stats(self._read_metadata(cache_key))
        if cached is not None:
            return cached
        stats = self._compute_stats(conversation_id)
        self._write_metadata(cache_key, stats)
        return stats

    # ------------------------------------------------------------------ #
    # Metadata helpers
    # ------------------------------------------------------------------ #
    def _stats_metadata_key(self, conversation_id: str) -> str:
        return f"l2:stats:{conversation_id}"

    def _correction_metadata_key(self, conversation_id: str) -> str:
        return f"l2:corr:{conversation_id}"

    def _write_metadata(self, key: str, value: Any) -> None:
        if not self.hot_path:
            return
        writer = getattr(self.hot_path, "write_metadata", None)
        if not writer:
            return
        writer(key, json.dumps(value, separators=(",", ":")))

    def _read_metadata(self, key: str) -> Any | None:
        if not self.hot_path:
            return None
        reader = getattr(self.hot_path, "read_metadata", None)
        if not reader:
            return None
        raw = reader(key)
        if not raw:
            return None
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return None

    def _empty_stats(self) -> Dict[str, int | str | None]:
        return {
            "message_count": 0,
            "user_turns": 0,
            "assistant_turns": 0,
            "started_at": None,
            "updated_at": None,
        }

    def _mirror_stats(self, conversation_id: str) -> None:
        stats = self._compute_stats(conversation_id)
        self._write_metadata(self._stats_metadata_key(conversation_id), stats)

    def _mirror_corrections(self, conversation_id: str) -> None:
        corrections = self._query_corrections(conversation_id, self._correction_cache)
        self._write_metadata(
            self._correction_metadata_key(conversation_id),
            [
                {"correction_id": item.correction_id, "payload": item.payload}
                for item in corrections
            ],
        )

    def _compute_stats(self, conversation_id: str) -> Dict[str, int | str | None]:
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
            return self._empty_stats()
        row = rows[0]
        return {
            "message_count": row["message_count"] or 0,
            "user_turns": row["user_turns"] or 0,
            "assistant_turns": row["assistant_turns"] or 0,
            "started_at": row["started_at"],
            "updated_at": row["updated_at"],
        }

    def _query_corrections(self, conversation_id: str, limit: int) -> List[Correction]:
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

    def _deserialize_stats(self, payload: Any) -> Dict[str, int | str | None] | None:
        if not isinstance(payload, dict):
            return None
        required = {"message_count", "user_turns", "assistant_turns", "started_at", "updated_at"}
        if not required.issubset(payload):
            return None
        return {
            "message_count": payload.get("message_count", 0),
            "user_turns": payload.get("user_turns", 0),
            "assistant_turns": payload.get("assistant_turns", 0),
            "started_at": payload.get("started_at"),
            "updated_at": payload.get("updated_at"),
        }

    def _deserialize_corrections(self, conversation_id: str, payload: Any) -> List[Correction] | None:
        if not isinstance(payload, list):
            return None
        parsed: List[Correction] = []
        for item in payload:
            if not isinstance(item, dict):
                continue
            correction_id = item.get("correction_id")
            if not correction_id:
                continue
            parsed.append(Correction(str(correction_id), conversation_id, dict(item.get("payload") or {})))
        return parsed


class SessionCache:
    """Pointer-sentinel style cache that adds recency bias per conversation."""

    def __init__(
        self,
        db: DatabaseEnvironment,
        decay: float = 0.8,
        hot_path: "HotPathAdapter | None" = None,
    ) -> None:
        self.db = db
        self.decay = decay
        self.hot_path = hot_path
        self._ensure_profile()

    def _ensure_profile(self) -> None:
        profile = self._read_profile_metadata("default")
        if profile:
            return
        profile = self._fetch_profile_from_sqlite("default")
        if profile is None:
            profile = {"lambda_cache": 0.15, "temp": 1.0, "topk": 20, "topp": 0.9}
            self.db.execute(
                "INSERT INTO tbl_decode_hparams(profile, lambda_cache, temp, topk, topp) VALUES ('default', 0.15, 1.0, 20, 0.9)"
            )
        self._write_profile_metadata("default", profile)

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
        params = self.decode_profile(profile)
        return float(params.get("lambda_cache", 0.15))

    def decode_profile(self, profile: str = "default") -> Dict[str, float | int]:
        cached = self._read_profile_metadata(profile)
        if cached:
            return cached
        record = self._fetch_profile_from_sqlite(profile)
        if record:
            self._write_profile_metadata(profile, record)
            return record
        return {"lambda_cache": 0.15, "temp": 1.0, "topk": 20, "topp": 0.9}

    def _fetch_profile_from_sqlite(self, profile: str) -> Dict[str, float | int] | None:
        rows = self.db.query(
            "SELECT lambda_cache, temp, topk, topp FROM tbl_decode_hparams WHERE profile = ?",
            (profile,),
        )
        if rows:
            return dict(rows[0])
        return None

    def _profile_metadata_key(self, profile: str) -> str:
        return f"decode:{profile}"

    def _read_profile_metadata(self, profile: str) -> Dict[str, float | int] | None:
        if not self.hot_path:
            return None
        reader = getattr(self.hot_path, "read_metadata", None)
        if not reader:
            return None
        raw = reader(self._profile_metadata_key(profile))
        if not raw:
            return None
        try:
            data = json.loads(raw)
            if isinstance(data, dict):
                return data  # type: ignore[return-value]
        except json.JSONDecodeError:
            return None
        return None

    def _write_profile_metadata(self, profile: str, payload: Dict[str, float | int]) -> None:
        if not self.hot_path:
            return
        writer = getattr(self.hot_path, "write_metadata", None)
        if not writer:
            return
        writer(
            self._profile_metadata_key(profile),
            json.dumps(payload, separators=(",", ":")),
        )


class BiasEngine:
    def __init__(self, db: DatabaseEnvironment, *, hot_path: "HotPathAdapter | None" = None, cache_limit: int = 128) -> None:
        self.db = db
        self.hot_path = hot_path
        self._bias_cache_limit = max(1, cache_limit)

    def upsert_bias(
        self,
        conversation_id: str | None,
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
        self._mirror_bias(conversation_id)

    def lookup(self, conversation_id: str, context_snippet: str) -> Dict[int, int]:
        entries = self._load_bias_entries(None) + self._load_bias_entries(conversation_id)
        active: Dict[int, int] = {}
        if not entries:
            return active
        now = datetime.now(timezone.utc)
        snippet = context_snippet.lower()
        for entry in entries:
            expires_at = entry.get("expires_at")
            if expires_at:
                try:
                    expires_dt = datetime.fromisoformat(expires_at)
                except ValueError:
                    continue
                if expires_dt.replace(tzinfo=timezone.utc) < now:
                    continue
            pattern = entry.get("pattern") or ""
            if pattern and pattern.lower() not in snippet:
                continue
            token_id = int(entry.get("token_id", 0))
            q_bias = int(entry.get("q_bias", 0))
            active[token_id] = active.get(token_id, 0) + q_bias
        return active

    # ------------------------------------------------------------------ #
    # Bias metadata helpers
    # ------------------------------------------------------------------ #
    def _bias_metadata_key(self, conversation_id: str | None) -> str:
        suffix = conversation_id or "__global__"
        return f"l2:bias:{suffix}"

    def _mirror_bias(self, conversation_id: str | None) -> None:
        entries = self._query_bias_entries(conversation_id, self._bias_cache_limit)
        self._write_bias_metadata(
            self._bias_metadata_key(conversation_id),
            entries,
        )

    def _load_bias_entries(self, conversation_id: str | None) -> List[dict[str, Any]]:
        key = self._bias_metadata_key(conversation_id)
        cached = self._read_bias_metadata(key)
        if cached is not None:
            return cached
        entries = self._query_bias_entries(conversation_id, self._bias_cache_limit)
        self._write_bias_metadata(key, entries)
        return entries

    def _query_bias_entries(self, conversation_id: str | None, limit: int) -> List[dict[str, Any]]:
        params: tuple[Any, ...]
        if conversation_id is None:
            sql = """
                SELECT conversation_id, pattern, token_id, q_bias, expires_at
                FROM tbl_l2_token_bias
                WHERE conversation_id IS NULL
                ORDER BY updated_at DESC
                LIMIT ?
            """
            params = (limit,)
        else:
            sql = """
                SELECT conversation_id, pattern, token_id, q_bias, expires_at
                FROM tbl_l2_token_bias
                WHERE conversation_id = ?
                ORDER BY updated_at DESC
                LIMIT ?
            """
            params = (conversation_id, limit)
        rows = self.db.query(sql, params)
        entries: List[dict[str, Any]] = []
        for row in rows:
            entries.append(
                {
                    "conversation_id": row["conversation_id"],
                    "pattern": row["pattern"],
                    "token_id": row["token_id"],
                    "q_bias": row["q_bias"],
                    "expires_at": row["expires_at"],
                }
            )
        return entries

    def _write_bias_metadata(self, key: str, entries: List[dict[str, Any]]) -> None:
        if not self.hot_path:
            return
        writer = getattr(self.hot_path, "write_metadata", None)
        if not writer:
            return
        writer(key, json.dumps(entries[: self._bias_cache_limit], separators=(",", ":")))

    def _read_bias_metadata(self, key: str) -> List[dict[str, Any]] | None:
        if not self.hot_path:
            return None
        reader = getattr(self.hot_path, "read_metadata", None)
        if not reader:
            return None
        raw = reader(key)
        if not raw:
            return None
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            return None
        if not isinstance(data, list):
            return None
        entries: List[dict[str, Any]] = []
        for item in data:
            if isinstance(item, dict):
                entries.append(item)
        return entries
