from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

from .db import DatabaseEnvironment
from .level1 import NGramModel
from .level2 import ConversationMemory
from .level3 import ConceptDefinition, ConceptEngine, ConceptExecution


class DBSLMEngine:
    """
    High-level façade that wires Level 1/2/3 together so experiments can start quickly.
    """

    def __init__(self, db_path: str | Path = ":memory:", ngram_order: int = 3) -> None:
        self.db = DatabaseEnvironment(db_path)
        self.memory = ConversationMemory(self.db)
        self.level1 = NGramModel(self.db, order=ngram_order)
        self.concepts = ConceptEngine(self.db, self.memory)
        self.concepts.register_payload_provider("CorrectionReplay", self._correction_payload)
        self._ensure_seed_data()

    # ------------------------------------------------------------------ #
    # Bootstrapping
    # ------------------------------------------------------------------ #
    def _ensure_seed_data(self) -> None:
        has_ngrams = self.db.query("SELECT 1 FROM tbl_l1_ngram_counts LIMIT 1")
        if not has_ngrams:
            self.level1.seed_defaults()
        self._ensure_concept_defaults()

    def _ensure_concept_defaults(self) -> None:
        summary = self.concepts.repo.fetch_by_name("ContextSummary")
        if summary is None:
            summary_id = self.concepts.repo.register(
                "ContextSummary",
                {"context": "text", "tokens": "text"},
            )
        else:
            summary_id = summary.concept_id
        self._ensure_template(summary_id, "Based on our recent exchange: {context}")
        self._ensure_probability("__default__", summary_id, 0.9)

        correction = self.concepts.repo.fetch_by_name("CorrectionReplay")
        if correction is None:
            correction_id = self.concepts.repo.register(
                "CorrectionReplay",
                {"corrections_text": "text"},
            )
        else:
            correction_id = correction.concept_id
        self._ensure_template(
            correction_id, "Incorporating your latest corrections: {corrections_text}"
        )
        self._ensure_probability("__default__", correction_id, 0.1)

    def _ensure_template(self, concept_id: int, template: str, language: str = "en") -> None:
        rows = self.db.query(
            """
            SELECT 1
            FROM tbl_l3_verbal_templates
            WHERE concept_id = ? AND language_code = ?
            LIMIT 1
            """,
            (concept_id, language),
        )
        if not rows:
            self.concepts.verbalizer.register_template(concept_id, template, language)

    def _ensure_probability(self, context_hash: str, concept_id: int, probability: float) -> None:
        rows = self.db.query(
            """
            SELECT 1
            FROM tbl_l3_concept_probs
            WHERE context_hash = ? AND next_concept_id = ?
            LIMIT 1
            """,
            (context_hash, concept_id),
        )
        if not rows:
            self.concepts.predictor.record_probability(context_hash, concept_id, probability)

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def start_conversation(self, user_id: str, agent_name: str = "db-slm") -> str:
        return self.memory.start_conversation(user_id, agent_name)

    def respond(self, conversation_id: str, user_message: str) -> str:
        self.memory.log_message(conversation_id, "user", user_message)
        context_text = self.memory.context_window(conversation_id)
        context_tokens = self.level1.tokenize(context_text)

        segments: list[str] = []
        concept_exec = self._run_concept_layer(conversation_id, context_tokens)
        if concept_exec:
            segments.append(concept_exec.text)
            context_tokens.extend(self.level1.tokenize(concept_exec.text))

        stitching = self.level1.stitch_tokens(context_tokens, target_length=18)
        if stitching:
            segments.append(stitching)

        response = " ".join(segment.strip() for segment in segments if segment.strip()).strip()
        self.memory.log_message(conversation_id, "assistant", response)
        return response

    def _run_concept_layer(
        self, conversation_id: str, context_tokens: list[str]
    ) -> Optional[ConceptExecution]:
        return self.concepts.generate(conversation_id, context_tokens)

    def record_correction(
        self,
        conversation_id: str,
        error_message_id: str,
        correction_message_id: str,
        error_context: str,
        corrected_fact: dict,
    ) -> str:
        correction_id = self.memory.record_correction(
            conversation_id,
            error_message_id,
            correction_message_id,
            error_context,
            corrected_fact,
        )
        # Pushing a signal ensures the correction surfaces in the next reply.
        self.concepts.push_signal(conversation_id, "CorrectionReplay", score=2.0, ttl_seconds=900)
        return correction_id

    def train_from_text(self, corpus: str) -> None:
        """
        Simple ETL helper that turns a raw corpus string into N-gram rows.
        """
        tokens = self.level1.tokenize(corpus)
        if len(tokens) < self.level1.order:
            return
        window = self.level1.order
        for idx in range(len(tokens) - window + 1):
            gram = tokens[idx : idx + window]
            self.level1.observe(gram)

    # ------------------------------------------------------------------ #
    # Payload providers
    # ------------------------------------------------------------------ #
    def _correction_payload(
        self,
        conversation_id: str,
        concept: ConceptDefinition,
        context_tokens: Sequence[str],
        memory: ConversationMemory,
    ) -> dict:
        corrections = memory.correction_digest(conversation_id, limit=5)
        if corrections:
            lines: list[str] = []
            for idx, correction in enumerate(corrections, start=1):
                facts = ", ".join(f"{key}={value}" for key, value in correction.payload.items())
                lines.append(f"{idx}. {facts or 'unstructured correction'}")
            corrections_text = " ".join(lines)
        else:
            corrections_text = f"No explicit corrections recorded for {concept.name.lower()} yet."
        return {
            "corrections_text": corrections_text,
            "context": memory.context_window(conversation_id),
            "tokens": " ".join(context_tokens),
        }
