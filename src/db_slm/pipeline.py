from __future__ import annotations

from pathlib import Path
from typing import Optional

from .db import DatabaseEnvironment
from .level1 import NGramModel
from .level2 import ConversationMemory
from .level3 import ConceptEngine, ConceptExecution


class DBSLMEngine:
    """
    High-level façade that wires Level 1/2/3 together so experiments can start quickly.
    """

    def __init__(self, db_path: str | Path = ":memory:", ngram_order: int = 3) -> None:
        self.db = DatabaseEnvironment(db_path)
        self.memory = ConversationMemory(self.db)
        self.level1 = NGramModel(self.db, order=ngram_order)
        self.concepts = ConceptEngine(self.db, self.memory)
        self._ensure_seed_data()

    # ------------------------------------------------------------------ #
    # Bootstrapping
    # ------------------------------------------------------------------ #
    def _ensure_seed_data(self) -> None:
        has_ngrams = self.db.query("SELECT 1 FROM tbl_l1_ngram_probs LIMIT 1")
        if not has_ngrams:
            self.level1.seed_defaults()

        has_concepts = self.db.query("SELECT 1 FROM tbl_l3_concept_repo LIMIT 1")
        if has_concepts:
            return
        concept_id = self.concepts.repo.register(
            "ContextSummary",
            {"context": "text", "tokens": "text"},
        )
        self.concepts.verbalizer.register_template(
            concept_id,
            "Based on our recent exchange: {context}",
        )
        self.concepts.predictor.record_probability("__default__", concept_id, 0.9)

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
        return self.memory.record_correction(
            conversation_id,
            error_message_id,
            correction_message_id,
            error_context,
            corrected_fact,
        )

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
            self.level1.observe(gram, probability=1.0)
