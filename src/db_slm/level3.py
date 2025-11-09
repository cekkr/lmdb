from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Callable, Dict, Optional, Sequence

from .db import DatabaseEnvironment
from .level1 import LogProbQuantizer
from .level2 import ConversationMemory


@dataclass(frozen=True)
class ConceptDefinition:
    concept_id: int
    name: str
    metadata_schema: dict


@dataclass(frozen=True)
class ConceptPrediction:
    concept_id: int
    probability: float


@dataclass(frozen=True)
class ConceptExecution:
    concept_name: str
    text: str
    payload: dict
    probability: float


PayloadProvider = Callable[[str, ConceptDefinition, Sequence[int], ConversationMemory], dict]


class ConceptRepository:
    def __init__(self, db: DatabaseEnvironment) -> None:
        self.db = db

    def register(self, concept_name: str, metadata_schema: dict) -> int:
        existing = self.fetch_by_name(concept_name)
        if existing:
            return existing.concept_id
        concept_id = self.db.insert_with_id(
            """
            INSERT INTO tbl_l3_concept_repo(concept_name, metadata_schema)
            VALUES (?, ?)
            """,
            (concept_name, json.dumps(metadata_schema)),
        )
        return concept_id

    def fetch_by_name(self, concept_name: str) -> Optional[ConceptDefinition]:
        rows = self.db.query(
            "SELECT concept_id, concept_name, metadata_schema FROM tbl_l3_concept_repo WHERE concept_name = ?",
            (concept_name,),
        )
        if not rows:
            return None
        row = rows[0]
        return ConceptDefinition(row["concept_id"], row["concept_name"], json.loads(row["metadata_schema"]))

    def fetch_by_id(self, concept_id: int) -> Optional[ConceptDefinition]:
        rows = self.db.query(
            "SELECT concept_id, concept_name, metadata_schema FROM tbl_l3_concept_repo WHERE concept_id = ?",
            (concept_id,),
        )
        if not rows:
            return None
        row = rows[0]
        return ConceptDefinition(row["concept_id"], row["concept_name"], json.loads(row["metadata_schema"]))


class Verbalizer:
    def __init__(self, db: DatabaseEnvironment) -> None:
        self.db = db

    def register_template(self, concept_id: int, template_string: str, language_code: str = "en") -> int:
        return self.db.insert_with_id(
            """
            INSERT INTO tbl_l3_verbal_templates(concept_id, template_string, language_code)
            VALUES (?, ?, ?)
            """,
            (concept_id, template_string, language_code),
        )

    def render(self, concept_id: int, payload: dict, language_code: str = "en") -> str:
        rows = self.db.query(
            """
            SELECT template_string
            FROM tbl_l3_verbal_templates
            WHERE concept_id = ? AND language_code = ?
            ORDER BY template_id ASC
            LIMIT 1
            """,
            (concept_id, language_code),
        )
        if not rows:
            raise RuntimeError(f"Missing template for concept_id={concept_id}")
        template = rows[0]["template_string"]
        return template.format(**payload)


class ConceptPredictor:
    def __init__(self, db: DatabaseEnvironment, quantizer: LogProbQuantizer) -> None:
        self.db = db
        self.quantizer = quantizer

    def record_probability(self, context_hash: str, concept_id: int, probability: float) -> None:
        q = self.quantizer.quantize(probability)
        self.db.execute(
            """
            INSERT INTO tbl_l3_concept_probs(context_hash, concept_id, q_logprob)
            VALUES (?, ?, ?)
            ON CONFLICT(context_hash, concept_id) DO UPDATE SET
                q_logprob = excluded.q_logprob
            """,
            (context_hash, concept_id, q),
        )

    def predict(self, context_hash: str, fallback_hash: str = "__default__") -> Optional[ConceptPrediction]:
        row = self._fetch_row(context_hash)
        if row is None and fallback_hash:
            row = self._fetch_row(fallback_hash)
        if row is None:
            return None
        prob = self.quantizer.dequantize_prob(row["q_logprob"])
        return ConceptPrediction(row["concept_id"], prob)

    def _fetch_row(self, context_hash: str):
        rows = self.db.query(
            """
            SELECT concept_id, q_logprob
            FROM tbl_l3_concept_probs
            WHERE context_hash = ?
            ORDER BY q_logprob DESC
            LIMIT 1
            """,
            (context_hash,),
        )
        return rows[0] if rows else None


class ConceptEngine:
    def __init__(
        self,
        db: DatabaseEnvironment,
        memory: ConversationMemory,
        quantizer: LogProbQuantizer,
    ) -> None:
        self.db = db
        self.memory = memory
        self.repo = ConceptRepository(db)
        self.verbalizer = Verbalizer(db)
        self.predictor = ConceptPredictor(db, quantizer)
        self.payload_providers: Dict[str, PayloadProvider] = {}

    def register_payload_provider(self, concept_name: str, provider: PayloadProvider) -> None:
        self.payload_providers[concept_name] = provider

    def push_signal(
        self,
        conversation_id: str,
        concept_name: str,
        score: float = 1.0,
        ttl_seconds: int | None = 300,
        consume_once: bool = True,
    ) -> str:
        concept = self.repo.fetch_by_name(concept_name)
        if not concept:
            raise ValueError(f"Concept '{concept_name}' is not registered")
        expires_at = None
        if ttl_seconds:
            expires_at = (datetime.utcnow() + timedelta(seconds=ttl_seconds)).isoformat(timespec="seconds")
        signal_id = str(uuid.uuid4())
        self.db.execute(
            """
            INSERT INTO tbl_l3_concept_signals(signal_id, conversation_id, concept_id, score, expires_at, consume_once)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (signal_id, conversation_id, concept.concept_id, score, expires_at, 1 if consume_once else 0),
        )
        return signal_id

    def generate(
        self,
        conversation_id: str,
        context_tokens: Sequence[int],
        language_code: str = "en",
    ) -> Optional[ConceptExecution]:
        context_hash = self.db.hash_tokens(context_tokens[-4:]) if context_tokens else "__default__"
        prediction = self._next_signal(conversation_id)
        if prediction is None:
            prediction = self.predictor.predict(context_hash)
        if prediction is None:
            return None
        concept = self.repo.fetch_by_id(prediction.concept_id)
        if concept is None:
            return None
        provider = self.payload_providers.get(concept.name, self._default_payload)
        payload = provider(conversation_id, concept, context_tokens, self.memory)
        text = self.verbalizer.render(concept.concept_id, payload, language_code)
        return ConceptExecution(concept.name, text, payload, prediction.probability)

    def _next_signal(self, conversation_id: str) -> Optional[ConceptPrediction]:
        rows = self.db.query(
            """
            SELECT signal_id, concept_id, score, consume_once
            FROM tbl_l3_concept_signals
            WHERE conversation_id = ?
              AND (expires_at IS NULL OR expires_at > CURRENT_TIMESTAMP)
            ORDER BY score DESC, created_at DESC
            LIMIT 1
            """,
            (conversation_id,),
        )
        if not rows:
            return None
        row = rows[0]
        if row["consume_once"]:
            self.db.execute("DELETE FROM tbl_l3_concept_signals WHERE signal_id = ?", (row["signal_id"],))
        prob = min(max(row["score"], 0.0), 1.0)
        return ConceptPrediction(row["concept_id"], prob)

    @staticmethod
    def _default_payload(
        conversation_id: str,
        concept: ConceptDefinition,
        context_tokens: Sequence[int],
        memory: ConversationMemory,
    ) -> dict:
        context_text = memory.context_window(conversation_id)
        stats = memory.conversation_stats(conversation_id)
        corrections = [corr.payload for corr in memory.lookup_corrections(conversation_id, limit=3)]
        return {
            "context": context_text,
            "stats": stats,
            "corrections": corrections,
            "token_count": len(context_tokens),
        }
