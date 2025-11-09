from __future__ import annotations

import re
from pathlib import Path
from typing import List, Optional, Set, Tuple

from .db import DatabaseEnvironment
from .metrics import keyword_summary, lexical_overlap
from .decoder import Decoder, DecoderConfig
from .level1 import LogProbQuantizer, MKNSmoother, NGramStore, Tokenizer, Vocabulary
from .level2 import BiasEngine, ConversationMemory, SessionCache
from .level3 import ConceptDefinition, ConceptEngine, ConceptExecution


class DBSLMEngine:
    """Facilitates training + inference using the three-level DB-SLM stack."""

    def __init__(self, db_path: str | Path = ":memory:", ngram_order: int = 3) -> None:
        self.db = DatabaseEnvironment(db_path, max_order=ngram_order)
        self.vocab = Vocabulary(self.db)
        self.tokenizer = Tokenizer(self.vocab)
        self.quantizer = LogProbQuantizer(self.db)
        self.store = NGramStore(self.db, self.vocab, ngram_order, self.quantizer)
        self.smoother = MKNSmoother(self.db, self.store, self.quantizer)
        self.memory = ConversationMemory(self.db)
        self.cache = SessionCache(self.db)
        self.bias = BiasEngine(self.db)
        self.decoder = Decoder(self.db, self.store, self.tokenizer, self.quantizer, self.cache, self.bias)
        self.concepts = ConceptEngine(self.db, self.memory, self.quantizer)
        self.level1 = self.store  # backwards compatibility for callers expecting this attr
        self._ensure_seed_data()
        self._low_resource_helper = LowResourceHelper(self)

    # ------------------------------------------------------------------ #
    # Training utilities
    # ------------------------------------------------------------------ #
    def train_from_text(self, corpus: str) -> int:
        token_ids = self.tokenizer.encode(corpus)
        if len(token_ids) < 2:
            return 0
        self.store.ingest(token_ids)
        self.smoother.rebuild_all()
        return len(token_ids)

    # ------------------------------------------------------------------ #
    # Conversation helpers
    # ------------------------------------------------------------------ #
    def start_conversation(self, user_id: str, agent_name: str = "db-slm") -> str:
        conversation_id = self.memory.start_conversation(user_id, agent_name)
        self._low_resource_helper.maybe_seed_history(conversation_id)
        return conversation_id

    def respond(self, conversation_id: str, user_message: str, decoder_cfg: DecoderConfig | None = None) -> str:
        self.memory.log_message(conversation_id, "user", user_message)
        user_ids = self.tokenizer.encode(user_message, add_special_tokens=False)
        if user_ids:
            self.cache.update(conversation_id, user_ids)
        history_text = self.memory.context_window(conversation_id)
        history_ids = self.tokenizer.encode(history_text, add_special_tokens=False)
        if not history_ids:
            history_ids = [self.vocab.token_id("<BOS>")]

        segments: List[str] = []
        concept_exec = self._run_concept_layer(conversation_id, history_ids)
        bias_context = history_text
        if concept_exec:
            segments.append(concept_exec.text)
            concept_ids = self.tokenizer.encode(concept_exec.text, add_special_tokens=False)
            history_ids.extend(concept_ids)
            if concept_ids:
                self.cache.update(conversation_id, concept_ids)
            bias_context = f"{history_text}\n{concept_exec.text}".strip()

        rolling_context = history_ids[-(self.store.order - 1) :] if self.store.order > 1 else history_ids
        rolling = list(rolling_context)
        decoded_ids = self.decoder.decode(conversation_id, rolling, decoder_cfg, bias_context)
        decoded_text = self.tokenizer.decode(decoded_ids)
        if decoded_text:
            segments.append(decoded_text)
        response = " ".join(segment.strip() for segment in segments if segment.strip()).strip()
        response = self._low_resource_helper.maybe_paraphrase(user_message, response)
        self.memory.log_message(conversation_id, "assistant", response)
        return response

    def _run_concept_layer(
        self, conversation_id: str, context_tokens: List[int]
    ) -> Optional[ConceptExecution]:
        return self.concepts.generate(conversation_id, context_tokens)

    # ------------------------------------------------------------------ #
    # Corrections & bias
    # ------------------------------------------------------------------ #
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
        self.concepts.push_signal(conversation_id, "CorrectionReplay", score=2.0, ttl_seconds=900)
        for key, value in corrected_fact.items():
            token_id = self.vocab.token_id(str(value))
            self.bias.upsert_bias(conversation_id, key, token_id, q_bias=40, ttl_seconds=3600)
        return correction_id

    # ------------------------------------------------------------------ #
    # Bootstrapping
    # ------------------------------------------------------------------ #
    def _ensure_seed_data(self) -> None:
        has_counts = self.db.query("SELECT 1 FROM tbl_l1_ng_counts_1 LIMIT 1")
        if not has_counts:
            sample = "Furthermore, this database-native LM is online."
            self.train_from_text(sample)
        self._ensure_concepts()

    def _ensure_concepts(self) -> None:
        summary = self.concepts.repo.fetch_by_name("ContextSummary")
        if summary is None:
            summary_id = self.concepts.repo.register(
                "ContextSummary",
                {"context": "text", "stats": "json"},
            )
        else:
            summary_id = summary.concept_id
        self._ensure_template(summary_id, "|CONTEXT|: {context}")
        self.concepts.predictor.record_probability("__default__", summary_id, 0.7)

        correction = self.concepts.repo.fetch_by_name("CorrectionReplay")
        if correction is None:
            correction_id = self.concepts.repo.register(
                "CorrectionReplay",
                {"corrections": "list"},
            )
        else:
            correction_id = correction.concept_id
        self._ensure_template(
            correction_id,
            "Incorporating your corrections: {corrections}",
        )
        self.concepts.predictor.record_probability("__default__", correction_id, 0.3)
        self.concepts.register_payload_provider("CorrectionReplay", self._correction_payload)

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

    # ------------------------------------------------------------------ #
    # Payload providers
    # ------------------------------------------------------------------ #
    def _correction_payload(
        self,
        conversation_id: str,
        concept: ConceptDefinition,
        context_tokens: List[int],
        memory: ConversationMemory,
    ) -> dict:
        corrections = memory.lookup_corrections(conversation_id, limit=5)
        formatted = []
        for idx, correction in enumerate(corrections, start=1):
            summary = ", ".join(f"{k}={v}" for k, v in correction.payload.items()) or "(unstructured)"
            formatted.append(f"{idx}. {summary}")
        return {
            "corrections": " ".join(formatted) if formatted else "None on file",
            "context": memory.context_window(conversation_id),
            "stats": memory.conversation_stats(conversation_id),
        }


class LowResourceHelper:
    """Adds light scaffolding for tiny training runs (seed history + paraphrasing)."""

    _SEED_DIALOG: Tuple[Tuple[str, str], ...] = (
        (
            "|SEED_PROMPT|: Track earlier notes even when training data is tiny?",
            "|SEED_RESPONSE|: Maintain a lightweight journal per exchange so short validation runs still have summaries.",
        ),
        (
            "|SEED_PROMPT|: Summarize what the database-focused LM proves.",
            "|SEED_RESPONSE|: SQL tables alone can handle vocabulary stats, caches, concepts, and decoding without tensors.",
        ),
    )

    def __init__(self, engine: DBSLMEngine) -> None:
        self.engine = engine
        self.enabled = self._detect_low_resource()
        self._seeded: Set[str] = set()
        self._paraphraser = SimpleParaphraser()

    def _detect_low_resource(self) -> bool:
        total_windows = self.engine.db.scalar(
            "SELECT SUM(total_count) FROM tbl_l1_context_registry", default=0
        ) or 0
        vocab_size = self.engine.db.scalar("SELECT COUNT(*) FROM tbl_l1_vocabulary", default=0) or 0
        return total_windows < 10_000 or vocab_size < 750

    def maybe_seed_history(self, conversation_id: str) -> None:
        if not self.enabled or conversation_id in self._seeded:
            return
        for user_msg, assistant_msg in self._SEED_DIALOG:
            self.engine.memory.log_message(conversation_id, "user", user_msg)
            self.engine.memory.log_message(conversation_id, "assistant", assistant_msg)
        self._seeded.add(conversation_id)

    def maybe_paraphrase(self, prompt: str, response: str) -> str:
        if not self.enabled or not response.strip():
            return response
        similarity = lexical_overlap(prompt, response)
        if similarity < 0.65:
            return response
        return self._paraphraser.rephrase(prompt, response)


class SimpleParaphraser:
    """String-level paraphraser to avoid verbatim echoes."""

    _RE_WORD = re.compile(r"\b\w+\b")
    _SYNONYMS = {
        "remember": "recall",
        "remind": "refresh",
        "discussed": "covered",
        "talked": "spoke",
        "small": "compact",
        "tiny": "minimal",
        "repeat": "restate",
        "echo": "mirror",
        "explain": "clarify",
        "summary": "synopsis",
    }

    def rephrase(self, prompt: str, response: str) -> str:
        swapped = self._swap_terms(response)
        if swapped.strip().lower() == response.strip().lower():
            swapped = f"\n|RESTATE|: {response}"
        keywords = keyword_summary(prompt, limit=3)
        suffix = ""
        if keywords:
            suffix = f"\n|KEYWORD|: {', '.join(keywords)}."
        return f"\n|RESPONSE|: {swapped.strip()}{suffix}"

    def _swap_terms(self, text: str) -> str:
        def _replace(match: re.Match[str]) -> str:
            token = match.group(0)
            lookup = token.lower()
            replacement = self._SYNONYMS.get(lookup)
            if replacement is None:
                return token
            return self._match_case(token, replacement)

        return self._RE_WORD.sub(_replace, text)

    @staticmethod
    def _match_case(source: str, replacement: str) -> str:
        if source.isupper():
            return replacement.upper()
        if source[0].isupper():
            return replacement.capitalize()
        return replacement
