from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Callable, Dict, Optional, Sequence

from .db import DatabaseEnvironment
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


PayloadProvider = Callable[[str, ConceptDefinition, Sequence[str], ConversationMemory], dict]


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
            """
            SELECT concept_id, concept_name, metadata_schema
            FROM tbl_l3_concept_repo
            WHERE concept_name = ?
            """,
            (concept_name,),
        )
        if not rows:
            return None
        row = rows[0]
        return ConceptDefinition(
            row["concept_id"], row["concept_name"], json.loads(row["metadata_schema"])
        )

    def fetch_by_id(self, concept_id: int) -> Optional[ConceptDefinition]:
        rows = self.db.query(
            """
            SELECT concept_id, concept_name, metadata_schema
            FROM tbl_l3_concept_repo
            WHERE concept_id = ?
            """,
            (concept_id,),
        )
        if not rows:
            return None
        row = rows[0]
        return ConceptDefinition(
            row["concept_id"], row["concept_name"], json.loads(row["metadata_schema"])
        )


class Verbalizer:
    def __init__(self, db: DatabaseEnvironment) -> None:
        self.db = db

    def register_template(self, concept_id: int, template_string: str, language_code: str = "en") -> int:
        template_id = self.db.insert_with_id(
            """
            INSERT INTO tbl_l3_verbal_templates(concept_id, template_string, language_code)
            VALUES (?, ?, ?)
            """,
            (concept_id, template_string, language_code),
        )
        return template_id

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
            raise RuntimeError(f"No template registered for concept_id={concept_id}")
        template = rows[0]["template_string"]
        try:
            return template.format(**payload)
        except KeyError as exc:  # pragma: no cover - defensive logging path
            missing = exc.args[0]
            raise KeyError(f"Template requires missing key '{missing}' for concept {concept_id}") from exc


class ConceptPredictor:
    def __init__(self, db: DatabaseEnvironment) -> None:
        self.db = db

    def record_probability(self, context_hash: str, concept_id: int, quantized_prob: float) -> None:
        self.db.execute(
            """
            INSERT INTO tbl_l3_concept_probs(context_hash, next_concept_id, quantized_prob)
            VALUES (?, ?, ?)
            """,
            (context_hash, concept_id, quantized_prob),
        )

    def predict(self, context_hash: str, fallback_hash: str = "__default__") -> Optional[ConceptPrediction]:
        rows = self.db.query(
            """
            SELECT next_concept_id, quantized_prob
            FROM tbl_l3_concept_probs
            WHERE context_hash = ?
            ORDER BY quantized_prob DESC
            LIMIT 1
            """,
            (context_hash,),
        )
        if not rows and fallback_hash:
            rows = self.db.query(
                """
                SELECT next_concept_id, quantized_prob
                FROM tbl_l3_concept_probs
                WHERE context_hash = ?
                ORDER BY quantized_prob DESC
                LIMIT 1
                """,
                (fallback_hash,),
            )
        if not rows:
            return None
        row = rows[0]
        return ConceptPrediction(row["next_concept_id"], row["quantized_prob"])


class ConceptEngine:
    """
    High-level façade that ties together prediction, repository lookups, and verbalization.
    """

    def __init__(self, db: DatabaseEnvironment, memory: ConversationMemory) -> None:
        self.repo = ConceptRepository(db)
        self.verbalizer = Verbalizer(db)
        self.predictor = ConceptPredictor(db)
        self.memory = memory
        self.payload_providers: Dict[str, PayloadProvider] = {}

    def register_payload_provider(self, concept_name: str, provider: PayloadProvider) -> None:
        self.payload_providers[concept_name] = provider

    def generate(
        self, conversation_id: str, context_tokens: Sequence[str], language_code: str = "en"
    ) -> Optional[ConceptExecution]:
        context_hash = self.repo.db.hash_tokens(context_tokens)
        prediction = self.predictor.predict(context_hash)
        if not prediction:
            return None
        concept = self.repo.fetch_by_id(prediction.concept_id)
        if not concept:
            return None
        provider = self.payload_providers.get(concept.name, self._default_payload)
        payload = provider(conversation_id, concept, context_tokens, self.memory)
        rendered = self.verbalizer.render(concept.concept_id, payload, language_code)
        return ConceptExecution(concept.name, rendered, payload, prediction.probability)

    @staticmethod
    def _default_payload(
        conversation_id: str,
        concept: ConceptDefinition,
        context_tokens: Sequence[str],
        memory: ConversationMemory,
    ) -> dict:
        """
        Default payload includes the rolling textual context and exposes it under the
        generic key `context`. Concepts can override via custom payload providers.
        """
        context_text = memory.context_window(conversation_id)
        return {"context": context_text, "tokens": " ".join(context_tokens)}
