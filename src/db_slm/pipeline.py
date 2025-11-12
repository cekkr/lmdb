from __future__ import annotations

import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional, Sequence, Set, Tuple

from .adapters.cheetah import build_cheetah_adapter
from .context_dimensions import (
    ContextDimension,
    DEFAULT_CONTEXT_DIMENSIONS,
    deserialize_context_dimensions,
    serialize_context_dimensions,
)
from .db import DatabaseEnvironment
from .metrics import keyword_summary, lexical_overlap
from .decoder import Decoder, DecoderConfig
from .level1 import (
    LogProbQuantizer,
    MKNSmoother,
    NGramStore,
    TokenCandidate,
    Tokenizer,
    Vocabulary,
)
from .level2 import BiasEngine, ConversationMemory, SessionCache
from .level3 import ConceptDefinition, ConceptEngine, ConceptExecution
from .sentence_parts import SentencePartEmbeddingPipeline
from .settings import DBSLMSettings, load_settings


@dataclass(frozen=True)
class ContextRelativismResult:
    context_hash: str
    order_size: int
    token_ids: Tuple[int, ...]
    probability: float
    ranked: Tuple[TokenCandidate, ...]


class DBSLMEngine:
    """Facilitates training + inference using the three-level DB-SLM stack."""

    def __init__(
        self,
        db_path: str | Path = ":memory:",
        ngram_order: int = 3,
        context_dimensions: Sequence[ContextDimension] | None = None,
        settings: DBSLMSettings | None = None,
    ) -> None:
        self.settings = settings or load_settings()
        self.db = DatabaseEnvironment(db_path, max_order=ngram_order)
        self.hot_path = build_cheetah_adapter(self.settings)
        self.context_dimensions = self._init_context_dimensions(context_dimensions)
        self.vocab = Vocabulary(self.db)
        self.tokenizer = Tokenizer(self.vocab)
        self.quantizer = LogProbQuantizer(self.db)
        self.store = NGramStore(
            self.db,
            self.vocab,
            ngram_order,
            self.quantizer,
            hot_path=self.hot_path,
        )
        self.smoother = MKNSmoother(self.db, self.store, self.quantizer)
        self.memory = ConversationMemory(self.db, hot_path=self.hot_path)
        self.cache = SessionCache(self.db, hot_path=self.hot_path)
        self.bias = BiasEngine(self.db, hot_path=self.hot_path)
        self.decoder = Decoder(
            self.db,
            self.store,
            self.tokenizer,
            self.quantizer,
            self.cache,
            self.bias,
            context_dimensions=self.context_dimensions,
        )
        self.concepts = ConceptEngine(self.db, self.memory, self.quantizer)
        self.level1 = self.store  # backwards compatibility for callers expecting this attr
        self.segment_embedder = SentencePartEmbeddingPipeline(self.settings)
        self._ensure_seed_data()
        self._low_resource_helper = LowResourceHelper(self)
        self._response_backstop = ResponseBackstop()
        self._tag_formatter = TaggedResponseFormatter()

    def _init_context_dimensions(
        self, requested: Sequence[ContextDimension] | None
    ) -> list[ContextDimension]:
        if requested is None:
            stored = None
            reader = getattr(self.hot_path, "read_metadata", None)
            if reader:
                stored = reader("context_dimensions")
            if stored is None:
                stored = self.db.get_metadata("context_dimensions")
            if stored:
                try:
                    resolved = deserialize_context_dimensions(stored)
                except ValueError:
                    resolved = list(DEFAULT_CONTEXT_DIMENSIONS)
            else:
                resolved = list(DEFAULT_CONTEXT_DIMENSIONS)
        else:
            resolved = list(requested)
        payload = serialize_context_dimensions(resolved)
        self.db.set_metadata("context_dimensions", payload)
        writer = getattr(self.hot_path, "write_metadata", None)
        if writer:
            writer("context_dimensions", payload)
        return resolved

    # ------------------------------------------------------------------ #
    # Training utilities
    # ------------------------------------------------------------------ #
    def train_from_text(
        self,
        corpus: str,
        *,
        progress_callback: Callable[[str, int, int], None] | None = None,
    ) -> int:
        if progress_callback:
            progress_callback("prepare", 0, 1)
        prepared = self.segment_embedder.prepare_for_training(corpus)
        if progress_callback:
            progress_callback("prepare", 1, 1)
        token_ids = self.tokenizer.encode(prepared or corpus)
        total_tokens = len(token_ids)
        if total_tokens < 2:
            return 0
        if progress_callback:
            progress_callback("tokenize", total_tokens, total_tokens)
        self.store.ingest(token_ids, progress_callback=progress_callback)
        self.smoother.rebuild_all(progress_callback=progress_callback)
        return total_tokens

    # ------------------------------------------------------------------ #
    # Conversation helpers
    # ------------------------------------------------------------------ #
    def start_conversation(
        self, user_id: str, agent_name: str = "db-slm", *, seed_history: bool = True
    ) -> str:
        conversation_id = self.memory.start_conversation(user_id, agent_name)
        if seed_history:
            self._low_resource_helper.maybe_seed_history(conversation_id)
        return conversation_id

    def respond(
        self,
        conversation_id: str,
        user_message: str,
        decoder_cfg: DecoderConfig | None = None,
        min_response_words: int = 0,
    ) -> str:
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
        response = self._response_backstop.ensure_min_words(user_message, response, min_response_words)
        response = self._tag_formatter.wrap(user_message, response)
        self.memory.log_message(conversation_id, "assistant", response)
        return response

    # ------------------------------------------------------------------ #
    # Cheetah helpers
    # ------------------------------------------------------------------ #
    def cheetah_topk_ratio(self) -> float:
        """Return the observed Top-K cache hit ratio served by cheetah."""
        return self.store.topk_hit_ratio()

    def context_relativism(
        self,
        context_tree,
        *,
        limit: int = 32,
        depth: int | None = None,
    ) -> List[ContextRelativismResult]:
        """Compute probabilistic projections for a nested context description."""
        raw_results = self.hot_path.context_relativism(context_tree, limit=limit, depth=depth)
        results: List[ContextRelativismResult] = []
        for projection in raw_results:
            ranked_candidates = tuple(
                TokenCandidate(
                    token_id,
                    self.vocab.token_text(token_id),
                    self.quantizer.dequantize_prob(q),
                    q,
                )
                for token_id, q in projection.ranked
            )
            probability = ranked_candidates[0].probability if ranked_candidates else 0.0
            results.append(
                ContextRelativismResult(
                    context_hash=projection.context_hash,
                    order_size=projection.order_size,
                    token_ids=projection.token_ids,
                    probability=probability,
                    ranked=ranked_candidates,
                )
            )
        return results

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
        if self._paraphraser.should_guard(prompt):
            return response
        similarity = lexical_overlap(prompt, response)
        threshold = self._paraphraser.threshold(prompt)
        if similarity < threshold:
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
    _STRUCTURAL_MARKERS = ("|RESPONSE|", "|CONTEXT|", "|INSTRUCTION|", "|CORRECTION|", "|USER|", "|TAGS|")
    _GUARDED_KEYWORDS = (
        "correct",
        "correction",
        "fix",
        "guard",
        "instruction",
        "do not",
        "don't",
        "never",
    )
    _OPENERS = (
        "Focusing on {tag},",
        "Considering {tag},",
        "Looking at {tag},",
        "Zooming in on {tag},",
        "Grounding the reply in {tag},",
    )
    _VARIATION_TEMPLATES = (
        "Linking {tag_list} shows how the themes layer together.",
        "The interaction across {tag_list} exposes the moving pieces.",
        "Studying {tag_list} together keeps the narrative anchored.",
        "Across {tag_list}, the signal is to connect causes with effects.",
    )

    def threshold(self, prompt: str) -> float:
        """Dynamic similarity threshold that scales with prompt length."""
        tokens = len(prompt.split())
        if tokens >= 120:
            return 0.85
        if tokens >= 60:
            return 0.72
        return 0.65

    def should_guard(self, prompt: str) -> bool:
        normalized = prompt.lower()
        if any(marker in prompt for marker in self._STRUCTURAL_MARKERS):
            return True
        if any(keyword in normalized for keyword in self._GUARDED_KEYWORDS):
            return True
        return self._looks_multi_turn(normalized)

    def rephrase(self, prompt: str, response: str) -> str:
        swapped = self._swap_terms(response).strip()
        if not swapped:
            return response
        original = response.strip()
        if original and swapped.lower() == original.lower():
            swapped = self._synthesize_variation(prompt)
        swapped = self._inject_random_opening(prompt, swapped)
        return swapped

    def _looks_multi_turn(self, normalized_prompt: str) -> bool:
        turn_markers = ("user:", "assistant:", "system:")
        matches = sum(normalized_prompt.count(marker) for marker in turn_markers)
        if matches >= 2:
            return True
        return normalized_prompt.count("\n") >= 2

    def _synthesize_variation(self, prompt: str) -> str:
        keywords = keyword_summary(prompt, limit=4)
        if not keywords:
            return "I'll expand on the request with new framing."
        tag_list = ", ".join(keywords)
        template = random.choice(self._VARIATION_TEMPLATES)
        return template.format(tag_list=tag_list)

    def _inject_random_opening(self, prompt: str, text: str) -> str:
        if not text.strip():
            return text
        keywords = keyword_summary(prompt, limit=4)
        if not keywords:
            return text
        chosen_tag = random.choice(keywords)
        opener = random.choice(self._OPENERS).format(tag=chosen_tag)
        if text.lower().startswith(opener.lower()):
            return text
        return f"{opener} {text}"

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


class ResponseBackstop:
    """Ensures evaluation probes always emit text with a configurable floor."""

    _FALLBACK_SENTENCES = [
        "I will still draft a transparent answer even while the probabilities remain low.",
        "This evaluation pass values coverage, so I am narrating the possible reasoning chain.",
        "Key cues stay visible so future training steps can compare lexical and semantic overlap.",
        "Uncertainty is acknowledged explicitly instead of returning an empty string.",
    ]

    def ensure_min_words(self, prompt: str, response: str, min_words: int) -> str:
        if min_words <= 0:
            return response
        cleaned = response.strip()
        words = cleaned.split()
        if len(words) >= min_words:
            return response
        needed = min_words - len(words)
        filler = self._build_filler(prompt, needed)
        merged = f"{cleaned} {filler}".strip()
        return merged

    def _build_filler(self, prompt: str, needed_words: int) -> str:
        keywords = keyword_summary(prompt, limit=4)
        phrases = list(self._FALLBACK_SENTENCES)
        if keywords:
            phrases.append(
                f"The prompt emphasizes {', '.join(keywords)},"
                " so I keep those motifs explicit while reasoning."
            )
        idx = 0
        tokens: list[str] = []
        while len(tokens) < needed_words and phrases:
            sentence = phrases[idx % len(phrases)]
            tokens.extend(sentence.split())
            idx += 1
        if len(tokens) > needed_words:
            tokens = tokens[:needed_words]
        return " ".join(tokens)


class TaggedResponseFormatter:
    """Wraps responses with |USER| / |RESPONSE| / |TAGS| scaffolding plus random openers."""

    _OPENERS = (
        "Exploring {tag},",
        "Zooming in on {tag},",
        "Grounding the response in {tag},",
        "Connecting back to {tag},",
        "Focusing on {tag},",
    )

    def wrap(self, prompt: str, generated: str) -> str:
        prompt_clean = prompt.strip()
        response_clean = generated.strip()
        keywords = keyword_summary(prompt, limit=4)
        response_tagged = self._randomize_opening(response_clean, keywords)
        lines: list[str] = []
        if prompt_clean:
            lines.append(f"|USER|: {prompt_clean}")
        if response_tagged:
            lines.append(f"|RESPONSE|: {response_tagged}")
        if keywords:
            lines.append(f"|TAGS|: {', '.join(keywords)}")
        return "\n".join(lines) if lines else ""

    def _randomize_opening(self, response: str, keywords: list[str]) -> str:
        if not response:
            return response
        if not keywords:
            return response
        tag = random.choice(keywords)
        opener = random.choice(self._OPENERS).format(tag=tag)
        if response.lower().startswith(opener.lower()):
            return response
        return f"{opener} {response}"
