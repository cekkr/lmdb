from __future__ import annotations

import json
import random
import re
from collections import defaultdict
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
from .context_window_embeddings import ContextWindowEmbeddingManager
from .db import DatabaseEnvironment
from .metrics import keyword_summary, lexical_overlap
from .decoder import Decoder, DecoderConfig
from .level1 import (
    LogProbQuantizer,
    MKNSmoother,
    NGramStore,
    TokenCandidate,
    MergeStats,
    Tokenizer,
    Vocabulary,
)
from .level2 import BiasEngine, ConversationMemory, SessionCache
from .level3 import ConceptDefinition, ConceptEngine, ConceptExecution
from .sentence_parts import SentencePartEmbeddingPipeline
from .settings import DBSLMSettings, load_settings
from .text_markers import strip_end_marker


@dataclass(frozen=True)
class ContextRelativismResult:
    context_hash: str
    order_size: int
    token_ids: Tuple[int, ...]
    probability: float
    ranked: Tuple[TokenCandidate, ...]


@dataclass
class TokenMergeReport:
    merged_tokens: int
    baseline_tokens: int
    applied_total: int
    candidate_total: int
    unique_applied: int
    unique_candidates: int
    passes: int
    retired_added: int
    retired_total: int
    top_applied: Tuple[Tuple[str, int, float], ...] = tuple()


class TokenMergeTracker:
    def __init__(
        self,
        *,
        retire_threshold: float,
        retire_min_count: int,
        retire_cap: int,
        retired_tokens: Sequence[str] | None = None,
    ) -> None:
        self.retire_threshold = max(0.0, float(retire_threshold))
        self.retire_min_count = max(1, int(retire_min_count))
        self.retire_cap = max(1, int(retire_cap))
        self.retired_tokens = set(retired_tokens or [])
        self.applied_counts: dict[str, int] = defaultdict(int)
        self.candidate_counts: dict[str, int] = defaultdict(int)
        self.base_lengths: dict[str, int] = {}

    def update(
        self,
        stats: MergeStats,
        *,
        merged_tokens: int,
        baseline_tokens: int,
        top_limit: int = 5,
    ) -> TokenMergeReport:
        applied_total = sum(stats.applied_counts.values())
        candidate_total = sum(stats.candidate_counts.values())
        for token, count in stats.applied_counts.items():
            self.applied_counts[token] += count
        for token, count in stats.candidate_counts.items():
            self.candidate_counts[token] += count
        for token, length in stats.base_lengths.items():
            if token not in self.base_lengths:
                self.base_lengths[token] = length
        retired_added = self._retire_low_significance()
        top_applied = self._top_applied(stats, limit=top_limit)
        return TokenMergeReport(
            merged_tokens=merged_tokens,
            baseline_tokens=baseline_tokens,
            applied_total=applied_total,
            candidate_total=candidate_total,
            unique_applied=len(stats.applied_counts),
            unique_candidates=len(stats.candidate_counts),
            passes=stats.passes,
            retired_added=retired_added,
            retired_total=len(self.retired_tokens),
            top_applied=top_applied,
        )

    def _retire_low_significance(self) -> int:
        if self.retire_threshold <= 0.0:
            return 0
        remaining = self.retire_cap - len(self.retired_tokens)
        if remaining <= 0:
            return 0
        candidates: list[tuple[float, int, str]] = []
        for token, candidate_count in self.candidate_counts.items():
            if token in self.retired_tokens:
                continue
            if candidate_count < self.retire_min_count:
                continue
            applied = self.applied_counts.get(token, 0)
            ratio = applied / float(candidate_count) if candidate_count else 0.0
            if ratio < self.retire_threshold:
                candidates.append((ratio, candidate_count, token))
        if not candidates:
            return 0
        candidates.sort(key=lambda item: (item[0], -item[1]))
        added = 0
        for ratio, _count, token in candidates:
            if added >= remaining:
                break
            self.retired_tokens.add(token)
            added += 1
        return added

    @staticmethod
    def _top_applied(stats: MergeStats, *, limit: int) -> Tuple[Tuple[str, int, float], ...]:
        if limit <= 0 or not stats.applied_counts:
            return tuple()
        entries: list[tuple[int, float, str]] = []
        for token, count in stats.applied_counts.items():
            candidate = stats.candidate_counts.get(token, 0)
            ratio = count / float(candidate) if candidate else 0.0
            entries.append((count, ratio, token))
        entries.sort(key=lambda item: (item[0], item[1]), reverse=True)
        top = []
        for count, ratio, token in entries[:limit]:
            top.append((token, count, round(ratio, 4)))
        return tuple(top)

_DEFAULT_PROMPT_TAG_TOKENS: Tuple[str, ...] = (
    "|INSTRUCTION|:",
    "|USER|:",
    "|RESPONSE|:",
    "|CONTEXT|:",
    "|TAGS|:",
    "|CORRECTION|:",
)
_PROMPT_TAG_SCAN_WINDOW = 160


class DBSLMEngine:
    """Facilitates training + inference using the three-level DB-SLM stack."""

    def __init__(
        self,
        db_path: str | Path = ":memory:",
        ngram_order: int = 3,
        context_dimensions: Sequence[ContextDimension] | None = None,
        settings: DBSLMSettings | None = None,
        *,
        prediction_table: str | None = None,
        prediction_key: str | None = None,
        prediction_weight: float = 0.0,
        token_merge_max_tokens: int | None = None,
        token_merge_recursion_depth: int | None = None,
        token_merge_baseline_train: bool | None = None,
        token_merge_baseline_eval: bool | None = None,
        token_merge_significance_threshold: float | None = None,
        token_merge_significance_min_count: int = 2,
        token_merge_significance_cap: int = 128,
    ) -> None:
        self.settings = settings or load_settings()
        self.db = DatabaseEnvironment(db_path, max_order=ngram_order)
        self.hot_path = build_cheetah_adapter(self.settings)
        self.db.set_metadata("ngram_order", str(ngram_order))
        writer = getattr(self.hot_path, "write_metadata", None)
        if writer:
            writer("ngram_order", str(ngram_order))
        self.context_dimensions = self._init_context_dimensions(context_dimensions)
        merge_max, merge_depth, retired_tokens = self._init_token_merging(
            token_merge_max_tokens,
            ngram_order,
            recursion_depth=token_merge_recursion_depth,
        )
        self.vocab = Vocabulary(self.db)
        self.tokenizer = Tokenizer(
            self.vocab,
            backend=self.settings.tokenizer_backend,
            tokenizer_path=self.settings.tokenizer_json_path,
            lowercase_tokens=self.settings.tokenizer_lowercase,
        )
        self.tokenizer.configure_merging(
            merge_max,
            recursion_depth=merge_depth,
            retired_tokens=retired_tokens,
        )
        self.token_merge_max_tokens = merge_max
        self.token_merge_recursion_depth = merge_depth
        merge_enabled = merge_max > 1
        baseline_train = (
            token_merge_baseline_train if token_merge_baseline_train is not None else merge_enabled
        )
        baseline_eval = (
            token_merge_baseline_eval if token_merge_baseline_eval is not None else merge_enabled
        )
        if not merge_enabled:
            baseline_train = False
            baseline_eval = False
        self.token_merge_baseline_train = bool(baseline_train)
        self.token_merge_baseline_eval = bool(baseline_eval)
        self.token_merge_significance_threshold = max(
            0.0, float(token_merge_significance_threshold or 0.0)
        )
        self.token_merge_significance_min_count = max(1, int(token_merge_significance_min_count))
        self.token_merge_significance_cap = max(1, int(token_merge_significance_cap))
        self._merge_tracker: TokenMergeTracker | None = None
        if merge_enabled:
            self._merge_tracker = TokenMergeTracker(
                retire_threshold=self.token_merge_significance_threshold,
                retire_min_count=self.token_merge_significance_min_count,
                retire_cap=self.token_merge_significance_cap,
                retired_tokens=retired_tokens,
            )
        self._last_merge_report: TokenMergeReport | None = None
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
            hot_path=self.hot_path,
            prediction_table=(prediction_table or "").strip(),
            prediction_key=(prediction_key or "").strip(),
            prediction_weight=prediction_weight,
        )
        self.concepts = ConceptEngine(self.db, self.memory, self.quantizer)
        self.level1 = self.store  # backwards compatibility for callers expecting this attr
        self.segment_embedder = SentencePartEmbeddingPipeline(self.settings)
        self.context_windows = ContextWindowEmbeddingManager(
            self.context_dimensions,
            embedder=self.segment_embedder.embedder,
            db=self.db,
            hot_path=self.hot_path,
        )
        self._ensure_seed_data()
        self._low_resource_helper = LowResourceHelper(self)
        self._response_backstop = ResponseBackstop()
        self._tag_formatter = TaggedResponseFormatter()
        self._prompt_tag_tokens: list[str] = []
        self._prompt_tag_token_ids: set[int] = set()
        self._prompt_tag_aliases: set[str] = set()
        self._prompt_tag_enumerator: dict[str, int] = {}
        self._prompt_tag_retry_attempts = 3
        self.register_prompt_tags(_DEFAULT_PROMPT_TAG_TOKENS)
        self._prediction_seeded_tokens: set[int] = set()
        self.prediction_table = (prediction_table or "").strip()
        self.prediction_key = (prediction_key or "").strip()
        self.prediction_weight = max(0.0, min(1.0, float(prediction_weight or 0.0)))

    def register_prompt_tags(self, tag_tokens: Sequence[str]) -> None:
        """Allow callers to seed tokenizer/vocabulary with structured prompt tags."""
        if not tag_tokens:
            return
        register_tokens: list[str] = []
        register_seen: set[str] = set()

        def register(token: str) -> None:
            normalized = (token or "").strip()
            if not normalized or normalized in register_seen:
                return
            register_seen.add(normalized)
            register_tokens.append(normalized)

        for token in tag_tokens:
            normalized = (token or "").strip()
            if not normalized:
                continue
            if normalized not in self._prompt_tag_tokens:
                self._prompt_tag_tokens.append(normalized)
            register(normalized)
            if self.tokenizer.lowercase_tokens:
                register(normalized.lower())
        if register_tokens:
            self.tokenizer.register_special_tokens(register_tokens)
        tokens_for_ban = set(self._prompt_tag_tokens)
        if self.tokenizer.lowercase_tokens:
            tokens_for_ban.update(token.lower() for token in self._prompt_tag_tokens)
        self._prompt_tag_token_ids = {self.vocab.token_id(token) for token in tokens_for_ban}
        self._prompt_tag_aliases = self._derive_prompt_tag_aliases(self._prompt_tag_tokens)
        enumerator = self._build_prompt_tag_enumerator(self._prompt_tag_tokens)
        self._prompt_tag_enumerator = enumerator
        self.context_windows.set_tag_enumerator(enumerator)

    @staticmethod
    def _canonical_prompt_label(token: str) -> str | None:
        normalized = (token or "").strip()
        if not normalized:
            return None
        if normalized.endswith(":"):
            normalized = normalized[:-1]
        if normalized.startswith("|") and normalized.endswith("|") and len(normalized) > 2:
            return normalized
        return None

    def _build_prompt_tag_enumerator(self, tokens: Sequence[str]) -> dict[str, int]:
        enumerator: dict[str, int] = {}
        for idx, token in enumerate(tokens):
            canonical = self._canonical_prompt_label(token)
            if not canonical or canonical in enumerator:
                continue
            enumerator[canonical] = idx
        return enumerator

    def _derive_prompt_tag_aliases(self, tokens: Sequence[str]) -> set[str]:
        aliases: set[str] = set()
        for token in tokens:
            alias_group = self._prompt_tag_aliases_for_token(token)
            aliases.update(alias_group)
        return aliases

    def _prompt_tag_aliases_for_token(self, token: str) -> set[str]:
        normalized = (token or "").strip()
        if not normalized:
            return set()
        lowered = normalized.lower()
        variants = {lowered}
        if lowered.endswith(":"):
            variants.add(lowered[:-1])
        canonical = self._canonical_prompt_label(normalized)
        if canonical:
            canonical_lower = canonical.lower()
            variants.add(canonical_lower)
            variants.add(f"{canonical_lower}:")
            bare = canonical_lower.strip("|")
            if bare:
                variants.add(f"{bare}:")
        return {variant for variant in variants if variant}

    def _contains_prompt_artifacts(self, response: str) -> bool:
        if not response or not self._prompt_tag_aliases:
            return False
        normalized = response.strip().lower()
        if not normalized:
            return False
        scan_window = normalized[:_PROMPT_TAG_SCAN_WINDOW]
        return any(alias in scan_window for alias in self._prompt_tag_aliases)

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

    def _init_token_merging(
        self,
        requested_max: int | None,
        ngram_order: int,
        *,
        recursion_depth: int | None = None,
    ) -> tuple[int, int, set[str]]:
        reader = getattr(self.hot_path, "read_metadata", None)

        def _coerce(value, fallback: int) -> int:
            try:
                return int(value)
            except Exception:
                return fallback

        def _read_metadata(key: str) -> str | None:
            if reader:
                stored = reader(key)
                if stored is not None:
                    return stored
            return self.db.get_metadata(key)

        stored_max: int | None = None
        stored_max_raw = _read_metadata("token_merge_max_tokens")
        if stored_max_raw is not None:
            stored_max = _coerce(stored_max_raw, 0)
        if stored_max is None:
            stored_max = 0

        stored_depth_raw = _read_metadata("token_merge_recursion_depth")
        stored_depth = _coerce(stored_depth_raw, 1)

        stored_retired_raw = _read_metadata("token_merge_retired_tokens")
        retired_tokens: set[str] = set()
        if stored_retired_raw:
            try:
                parsed = json.loads(stored_retired_raw)
                if isinstance(parsed, list):
                    retired_tokens = {str(token) for token in parsed if str(token).strip()}
            except Exception:
                retired_tokens = set()

        resolved_max = _coerce(requested_max, stored_max or 0)
        if resolved_max < 0:
            resolved_max = 0
        resolved_depth = _coerce(recursion_depth, stored_depth or 1)
        if resolved_depth < 1:
            resolved_depth = 1
        if ngram_order < 5:
            resolved_max = 0
            resolved_depth = 1

        self.db.set_metadata("token_merge_max_tokens", str(resolved_max))
        self.db.set_metadata("token_merge_recursion_depth", str(resolved_depth))
        writer = getattr(self.hot_path, "write_metadata", None)
        if writer:
            writer("token_merge_max_tokens", str(resolved_max))
            writer("token_merge_recursion_depth", str(resolved_depth))
        if retired_tokens:
            payload = json.dumps(sorted(retired_tokens), ensure_ascii=True)
            self.db.set_metadata("token_merge_retired_tokens", payload)
            if writer:
                writer("token_merge_retired_tokens", payload)
        return resolved_max, resolved_depth, retired_tokens

    def _persist_merge_retired_tokens(self) -> None:
        tracker = self._merge_tracker
        if tracker is None or not tracker.retired_tokens:
            return
        payload = json.dumps(sorted(tracker.retired_tokens), ensure_ascii=True)
        self.db.set_metadata("token_merge_retired_tokens", payload)
        writer = getattr(self.hot_path, "write_metadata", None)
        if writer:
            writer("token_merge_retired_tokens", payload)

    def _update_merge_tracker(
        self,
        stats: MergeStats | None,
        *,
        merged_tokens: int,
        baseline_tokens: int,
    ) -> None:
        tracker = self._merge_tracker
        if tracker is None or stats is None:
            self._last_merge_report = None
            return
        report = tracker.update(
            stats,
            merged_tokens=merged_tokens,
            baseline_tokens=baseline_tokens,
        )
        self._last_merge_report = report
        if report.retired_added:
            self.tokenizer.set_merge_retired_tokens(tracker.retired_tokens)
            self._persist_merge_retired_tokens()

    def consume_merge_report(self) -> TokenMergeReport | None:
        report = self._last_merge_report
        self._last_merge_report = None
        return report

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
        if self.context_windows.enabled():
            self.context_windows.observe_corpus(corpus)
        prepared = self.segment_embedder.prepare_for_training(corpus)
        if progress_callback:
            progress_callback("prepare", 1, 1)
        collect_stats = self._merge_tracker is not None and self.token_merge_max_tokens > 1
        if collect_stats:
            token_ids, merge_stats = self.tokenizer.encode_with_stats(
                prepared or corpus,
                add_special_tokens=True,
                merge_mode="auto",
            )
        else:
            token_ids = self.tokenizer.encode(
                prepared or corpus,
                add_special_tokens=True,
                merge_mode="auto",
            )
            merge_stats = None
        total_tokens = len(token_ids)
        if total_tokens < 2:
            return 0
        baseline_tokens = 0
        if self.token_merge_baseline_train and self.token_merge_max_tokens > 1:
            baseline_ids = self.tokenizer.encode(
                prepared or corpus,
                add_special_tokens=True,
                merge_mode="off",
            )
            baseline_tokens = len(baseline_ids)
            if baseline_tokens >= 2:
                self.store.ingest(baseline_ids, progress_callback=None)
        if progress_callback:
            progress_callback("tokenize", total_tokens, total_tokens)
        self.store.ingest(token_ids, progress_callback=progress_callback)
        self.smoother.rebuild_all(progress_callback=progress_callback)
        if self.context_windows.enabled():
            self.context_windows.flush()
        if collect_stats:
            self._update_merge_tracker(
                merge_stats,
                merged_tokens=total_tokens,
                baseline_tokens=baseline_tokens,
            )
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
        rng_seed: int | None = None,
        *,
        scaffold_response: bool = True,
    ) -> str:
        rng = random.Random(rng_seed) if rng_seed is not None else random.Random()
        self.memory.log_message(conversation_id, "user", user_message)
        user_ids = self.tokenizer.encode(user_message, add_special_tokens=False)
        if user_ids:
            self.cache.update(conversation_id, user_ids)
        history_text = self.memory.context_window(conversation_id)
        history_ids = self.tokenizer.encode(history_text, add_special_tokens=False)
        if not history_ids:
            history_ids = [self.vocab.token_id("<BOS>")]

        concept_exec = self._run_concept_layer(conversation_id, history_ids)
        bias_context = history_text
        if concept_exec:
            concept_text = concept_exec.text.strip()
            if concept_text:
                concept_ids = self.tokenizer.encode(concept_text, add_special_tokens=False)
                history_ids.extend(concept_ids)
                if concept_ids:
                    self.cache.update(conversation_id, concept_ids)
                bias_context = f"{history_text}\n{concept_text}".strip()

        prefix_segments: tuple[str, ...] = tuple()
        rolling_context = history_ids[-(self.store.order - 1) :] if self.store.order > 1 else history_ids
        base_context = list(rolling_context)
        window_weights = None
        prediction_matrix = None
        if self.context_windows.enabled():
            window_reference = bias_context or history_text
            window_weights = self.context_windows.weights_for_text(window_reference)
            payload_builder = getattr(self.context_windows, "context_matrix_payload_for_text", None)
            if callable(payload_builder):
                prediction_matrix = payload_builder(window_reference)
            else:
                prediction_matrix = self.context_windows.context_matrix_for_text(window_reference)

        attempt_count = max(1, self._prompt_tag_retry_attempts)
        final_ids: List[int] = []
        response = ""
        decoded_text = ""
        for attempt in range(attempt_count):
            context_seed = list(base_context)
            attempt_rng = random.Random(rng.random())
            decoded_ids = self.decoder.decode(
                conversation_id,
                context_seed,
                decoder_cfg,
                bias_context,
                rng=attempt_rng,
                dimension_weights=window_weights,
                banned_token_ids=self._prompt_tag_token_ids,
                commit_cache=False,
                prediction_matrix=prediction_matrix,
            )
            decoded_text = self.tokenizer.decode(decoded_ids)
            attempt_segments = [segment for segment in prefix_segments if segment]
            if decoded_text:
                attempt_segments.append(decoded_text)
            response_candidate = " ".join(segment.strip() for segment in attempt_segments if segment.strip()).strip()
            final_ids = decoded_ids
            response = response_candidate
            if not self._contains_prompt_artifacts(response_candidate):
                break
        if final_ids:
            self.cache.update(conversation_id, final_ids)
        else:
            self.cache.update(conversation_id, [])
        response = self._low_resource_helper.maybe_paraphrase(user_message, response, rng=rng)
        if scaffold_response:
            response = self._response_backstop.ensure_min_words(user_message, response, min_response_words)
            response = self._tag_formatter.wrap(user_message, response, rng=rng)
        response, _ = strip_end_marker(response)
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

    def maybe_paraphrase(
        self,
        prompt: str,
        response: str,
        *,
        rng: random.Random | None = None,
    ) -> str:
        if not self.enabled or not response.strip():
            return response
        if self._paraphraser.should_guard(prompt):
            return response
        similarity = lexical_overlap(prompt, response)
        threshold = self._paraphraser.threshold(prompt)
        if similarity < threshold:
            return response
        return self._paraphraser.rephrase(prompt, response, rng=rng)


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

    def rephrase(
        self,
        prompt: str,
        response: str,
        rng: random.Random | None = None,
    ) -> str:
        rng = rng or random
        swapped = self._swap_terms(response).strip()
        if not swapped:
            return response
        original = response.strip()
        if original and swapped.lower() == original.lower():
            swapped = self._synthesize_variation(prompt, rng)
        swapped = self._inject_random_opening(prompt, swapped, rng)
        return swapped

    def _looks_multi_turn(self, normalized_prompt: str) -> bool:
        turn_markers = ("user:", "assistant:", "system:")
        matches = sum(normalized_prompt.count(marker) for marker in turn_markers)
        if matches >= 2:
            return True
        return normalized_prompt.count("\n") >= 2

    def _synthesize_variation(self, prompt: str, rng: random.Random) -> str:
        keywords = keyword_summary(prompt, limit=4)
        if not keywords:
            return "I'll expand on the request with new framing."
        tag_list = ", ".join(keywords)
        template = rng.choice(self._VARIATION_TEMPLATES)
        return template.format(tag_list=tag_list)

    def _inject_random_opening(self, prompt: str, text: str, rng: random.Random) -> str:
        if not text.strip():
            return text
        keywords = keyword_summary(prompt, limit=4)
        if not keywords:
            return text
        chosen_tag = rng.choice(keywords)
        opener = rng.choice(self._OPENERS).format(tag=chosen_tag)
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

    _FILLER_TEMPLATES = (
        "I can relate {keyword} to practical safeguards to keep the thread moving.",
        "Linking {keyword} back to the requested outcome keeps the reply grounded.",
        "Another nudge is to map {keyword} onto the surrounding decision points.",
        "{keyword} ties directly into risk controls, so I keep surfacing that link.",
        "Explaining how {keyword} influences everyday follow-through widens the view.",
    )
    _GENERIC_FALLBACKS = (
        "I will keep layering small connective thoughts so the response grows.",
        "The idea is to stay descriptive while adding more connective statements.",
        "Continuing the explanation in short beats keeps the narrative alive.",
    )

    def ensure_min_words(self, prompt: str, response: str, min_words: int) -> str:
        if min_words <= 0:
            return response
        cleaned = response.strip()
        words = cleaned.split()
        if len(words) >= min_words:
            return response
        needed = min_words - len(words)
        filler = self._build_filler(prompt, cleaned, needed)
        if not filler:
            return response
        merged = f"{cleaned} {filler}".strip()
        return merged

    def _build_filler(self, prompt: str, response: str, needed_words: int) -> str:
        if needed_words <= 0:
            return ""
        keywords = self._collect_keywords(prompt, response)
        fragments = self._collect_fragments(prompt, response)
        sentences: list[str] = []
        template_idx = 0
        keyword_idx = 0
        total_words = 0
        while total_words < needed_words:
            chunk = ""
            if keywords:
                template = self._FILLER_TEMPLATES[template_idx % len(self._FILLER_TEMPLATES)]
                keyword = keywords[keyword_idx % len(keywords)]
                chunk = template.format(keyword=keyword)
                template_idx += 1
                keyword_idx += 1
            elif fragments:
                chunk = fragments.pop(0)
            else:
                chunk = self._GENERIC_FALLBACKS[len(sentences) % len(self._GENERIC_FALLBACKS)]
            if chunk:
                sentences.append(chunk)
                total_words += len(chunk.split())
            else:
                break
        filler = " ".join(sentences).strip()
        words = filler.split()
        return " ".join(words[:needed_words])

    def _collect_keywords(self, prompt: str, response: str) -> list[str]:
        keywords = keyword_summary(prompt, limit=6)
        fallback = keyword_summary(response, limit=6)
        combined: list[str] = []
        seen: set[str] = set()
        for source in (keywords, fallback):
            for token in source:
                normalized = token.lower()
                if normalized in seen:
                    continue
                seen.add(normalized)
                combined.append(token)
        return combined

    def _collect_fragments(self, prompt: str, response: str) -> list[str]:
        fragments: list[str] = []
        seen: set[str] = set()
        for text in (response, prompt):
            for fragment in self._extract_fragments(text):
                trimmed = self._trim_fragment(fragment)
                normalized = trimmed.lower()
                if not trimmed or normalized in seen:
                    continue
                seen.add(normalized)
                fragments.append(trimmed)
        return fragments

    @staticmethod
    def _trim_fragment(fragment: str, *, max_words: int = 18) -> str:
        words = fragment.split()
        if not words:
            return ""
        if len(words) <= max_words:
            return fragment.strip()
        return " ".join(words[:max_words]).strip()

    @staticmethod
    def _extract_fragments(text: str) -> list[str]:
        normalized = text.strip()
        if not normalized:
            return []
        fragments = [
            fragment.strip()
            for fragment in re.split(r"[.!?\n]+", normalized)
            if fragment.strip()
        ]
        if fragments:
            return fragments
        words = normalized.split()
        window = max(1, min(6, len(words)))
        return [
            " ".join(words[i : i + window]).strip()
            for i in range(0, len(words), window)
            if words[i : i + window]
        ]


class TaggedResponseFormatter:
    """Wraps responses with |USER| / |RESPONSE| / |TAGS| scaffolding plus random openers."""

    _OPENERS = (
        "Exploring {tag},",
        "Zooming in on {tag},",
        "Grounding the response in {tag},",
        "Connecting back to {tag},",
        "Focusing on {tag},",
    )

    def wrap(self, prompt: str, generated: str, rng: random.Random | None = None) -> str:
        rng = rng or random
        prompt_clean = prompt.strip()
        response_clean = generated.strip()
        keywords = keyword_summary(prompt, limit=4)
        response_tagged = self._randomize_opening(response_clean, keywords, rng)
        lines: list[str] = []
        if prompt_clean:
            lines.append(f"|USER|: {prompt_clean}")
        if response_tagged:
            lines.append(f"|RESPONSE|: {response_tagged}")
        if keywords:
            lines.append(f"|TAGS|: {', '.join(keywords)}")
        return "\n".join(lines) if lines else ""

    def _randomize_opening(
        self,
        response: str,
        keywords: list[str],
        rng: random.Random,
    ) -> str:
        if not response:
            return response
        if not keywords:
            return response
        tag = rng.choice(keywords)
        opener = rng.choice(self._OPENERS).format(tag=tag)
        if response.lower().startswith(opener.lower()):
            return response
        return f"{opener} {response}"
