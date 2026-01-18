from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Optional
import random

from .db import DatabaseEnvironment
from .context_dimensions import ContextDimension, ContextDimensionTracker
from .level1 import LogProbQuantizer, NGramStore, TokenCandidate, Tokenizer
from .level2 import BiasEngine, SessionCache
from .scoring import ScoreObserver, ScoreResult, ScoreSnapshot, TokenScoringPipeline
from .adapters.base import HotPathAdapter


@dataclass
class DecoderConfig:
    max_tokens: int = 32
    profile: str = "default"
    presence_penalty: float = 0.15
    frequency_penalty: float = 0.05


class Decoder:
    """Sampling utilities that combine Level 1 stats, cache mixtures, and biases."""

    def __init__(
        self,
        db: DatabaseEnvironment,
        store: NGramStore,
        tokenizer: Tokenizer,
        quantizer: LogProbQuantizer,
        cache: SessionCache,
        bias: BiasEngine,
        *,
        context_dimensions: Sequence[ContextDimension] | None = None,
        hot_path: HotPathAdapter | None = None,
        prediction_table: str | None = None,
        prediction_key: str | None = None,
        prediction_weight: float = 0.0,
    ) -> None:
        self.db = db
        self.store = store
        self.tokenizer = tokenizer
        self.quantizer = quantizer
        self.cache = cache
        self.bias = bias
        self.context_dimensions = list(context_dimensions or [])
        self.hot_path = hot_path
        self.prediction_table = (prediction_table or "").strip()
        self.prediction_key = (prediction_key or "").strip()
        self.prediction_weight = max(0.0, min(1.0, float(prediction_weight or 0.0)))
        self.scoring = TokenScoringPipeline(self.quantizer, self.tokenizer, self.cache, self.bias)

    def decode(
        self,
        conversation_id: str,
        context_ids: List[int],
        config: DecoderConfig | None = None,
        context_snippet: str = "",
        *,
        rng: random.Random | None = None,
        dimension_weights: Sequence[float] | None = None,
        banned_token_ids: Sequence[int] | None = None,
        commit_cache: bool = True,
        prediction_matrix: Sequence[Sequence[float]] | dict[str, object] | None = None,
        score_observer: ScoreObserver | None = None,
    ) -> List[int]:
        config = config or DecoderConfig()
        rng = rng or random
        generated: List[int] = []
        profile = self.cache.decode_profile(config.profile)
        banned = set(self._load_bans(config.profile))
        if banned_token_ids:
            banned.update(int(token_id) for token_id in banned_token_ids)
        dimension_tracker: ContextDimensionTracker | None = None
        if self.context_dimensions:
            dimension_tracker = ContextDimensionTracker(
                self.context_dimensions,
                list(context_ids),
                dimension_weights=dimension_weights,
            )
        prediction_bias = self._prediction_distribution(prediction_matrix)
        for step_index in range(config.max_tokens):
            context_snapshot = tuple(context_ids)
            order = min(self.store.order, len(context_ids) + 1)
            candidates = self._resolve_candidates(context_ids, order, profile["topk"])
            if not candidates:
                candidates = self._relativistic_fallback(context_ids, order, profile["topk"])
                if not candidates:
                    break
            score_result = self._score_candidates(
                conversation_id,
                candidates,
                generated,
                profile,
                banned,
                config,
                context_snippet,
                dimension_tracker,
                prediction_bias,
                collect_trace=score_observer is not None,
            )
            adjusted = score_result.distribution
            if not adjusted:
                break
            next_token = self._sample(adjusted, profile["topp"], rng)
            if next_token is None:
                break
            if score_observer and score_result.trace is not None:
                score_observer(
                    ScoreSnapshot(
                        step_index=step_index,
                        context_ids=context_snapshot,
                        scores=score_result.trace,
                        chosen_token_id=next_token,
                    )
                )
            generated.append(next_token)
            context_ids.append(next_token)
            context_ids[:] = context_ids[-(self.store.order - 1) :]
            if dimension_tracker:
                dimension_tracker.record(next_token)
            if self.tokenizer.vocab.token_text(next_token) == "<EOS>":
                break
        if commit_cache:
            self.cache.update(conversation_id, generated)
        return generated

    def _score_candidates(
        self,
        conversation_id: str,
        candidates: Sequence[TokenCandidate],
        generated: List[int],
        profile: Dict[str, float | int],
        banned: set[int],
        config: DecoderConfig,
        context_snippet: str,
        dimension_tracker: ContextDimensionTracker | None,
        prediction_bias: Optional[Dict[int, float]],
        *,
        collect_trace: bool = False,
    ) -> ScoreResult:
        temperature = float(profile.get("temp", 1.0))
        lambda_cache = float(profile.get("lambda_cache", 0.15))
        return self.scoring.score(
            conversation_id,
            candidates,
            generated,
            banned=banned,
            context_snippet=context_snippet,
            temperature=temperature,
            lambda_cache=lambda_cache,
            presence_penalty=config.presence_penalty,
            frequency_penalty=config.frequency_penalty,
            dimension_tracker=dimension_tracker,
            prediction_bias=prediction_bias,
            prediction_weight=self.prediction_weight,
            collect_trace=collect_trace,
        )

    def _resolve_candidates(
        self,
        context_ids: Sequence[int],
        order: int,
        k: int,
    ) -> List[TokenCandidate]:
        """
        Step down through shorter contexts when the full-order lookup has no hits.
        """
        for current_order in range(order, 0, -1):
            candidates = self.store.get_topk(context_ids, current_order, k)
            if candidates:
                return candidates
        return []

    def _sample(
        self,
        probs: Dict[int, float],
        top_p: float,
        rng: random.Random,
    ) -> int | None:
        if not probs:
            return None
        sorted_items = sorted(probs.items(), key=lambda item: item[1], reverse=True)
        cumulative = 0.0
        cutoff_items: List[tuple[int, float]] = []
        threshold = max(min(top_p, 0.999), 0.05)
        for token_id, prob in sorted_items:
            cutoff_items.append((token_id, prob))
            cumulative += prob
            if cumulative >= threshold:
                break
        total = sum(prob for _, prob in cutoff_items)
        if total <= 0:
            return None
        choice = rng.random() * total
        running = 0.0
        for token_id, prob in cutoff_items:
            running += prob
            if running >= choice:
                return token_id
        return cutoff_items[-1][0]

    def _prediction_distribution(
        self,
        prediction_matrix: Sequence[Sequence[float]] | dict[str, object] | None,
    ) -> Optional[Dict[int, float]]:
        if (
            prediction_matrix is None
            or not prediction_matrix
            or not self.hot_path
            or not self.prediction_key
            or not self.prediction_table
            or self.prediction_weight <= 0.0
        ):
            return None
        result = self.hot_path.predict_query(
            key=self.prediction_key,
            context_matrix=prediction_matrix,
            table=self.prediction_table,
        )
        if result is None or not result.entries:
            return None
        distribution: Dict[int, float] = {}
        total = 0.0
        for entry in result.entries:
            token_id = self._decode_prediction_token(entry.value)
            if token_id is None:
                continue
            prob = max(0.0, float(entry.probability))
            distribution[token_id] = prob
            total += prob
        # Avoid collapsing decoding when only a single prediction survives.
        if len(distribution) < 2:
            return None
        if total <= 0.0 or not distribution:
            return None
        for token_id in list(distribution.keys()):
            distribution[token_id] = distribution[token_id] / total
        return distribution

    @staticmethod
    def _decode_prediction_token(raw_value: bytes) -> Optional[int]:
        if not raw_value:
            return None
        if len(raw_value) == 4:
            return int.from_bytes(raw_value, "big", signed=False)
        return None

    def _load_bans(self, profile: str) -> set[int]:
        rows = self.db.query(
            "SELECT token_id FROM tbl_decode_bans WHERE profile = ?",
            (profile,),
        )
        return {row["token_id"] for row in rows}

    def _relativistic_fallback(self, context_ids: Sequence[int], order: int, k: int) -> List[TokenCandidate]:
        if order <= 1:
            return []
        window = context_ids[-(order - 1) :]
        if not window:
            return []
        structure = [[token_id] for token_id in window]
        projections = self.store.hot_path.context_relativism(structure, limit=1, depth=None)
        if not projections:
            return []
        ranked = projections[0].ranked[:k]
        if not ranked:
            return []
        candidates: List[TokenCandidate] = []
        for token_id, q in ranked:
            candidates.append(
                TokenCandidate(
                    token_id=token_id,
                    token_text=self.tokenizer.vocab.token_text(token_id),
                    probability=self.quantizer.dequantize_prob(q),
                    q_logprob=q,
                )
            )
        return candidates
