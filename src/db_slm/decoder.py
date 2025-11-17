from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence
import random

from .db import DatabaseEnvironment
from .context_dimensions import ContextDimension, ContextDimensionTracker
from .level1 import LogProbQuantizer, NGramStore, TokenCandidate, Tokenizer
from .level2 import BiasEngine, SessionCache


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
    ) -> None:
        self.db = db
        self.store = store
        self.tokenizer = tokenizer
        self.quantizer = quantizer
        self.cache = cache
        self.bias = bias
        self.context_dimensions = list(context_dimensions or [])

    def decode(
        self,
        conversation_id: str,
        context_ids: List[int],
        config: DecoderConfig | None = None,
        context_snippet: str = "",
        *,
        rng: random.Random | None = None,
        dimension_weights: Sequence[float] | None = None,
    ) -> List[int]:
        config = config or DecoderConfig()
        rng = rng or random
        generated: List[int] = []
        profile = self.cache.decode_profile(config.profile)
        banned = self._load_bans(config.profile)
        dimension_tracker: ContextDimensionTracker | None = None
        if self.context_dimensions:
            dimension_tracker = ContextDimensionTracker(
                self.context_dimensions,
                list(context_ids),
                dimension_weights=dimension_weights,
            )
        for _ in range(config.max_tokens):
            order = min(self.store.order, len(context_ids) + 1)
            candidates = self._resolve_candidates(context_ids, order, profile["topk"])
            if not candidates:
                candidates = self._relativistic_fallback(context_ids, order, profile["topk"])
                if not candidates:
                    break
            adjusted = self._adjust_candidates(
                conversation_id,
                candidates,
                generated,
                profile,
                banned,
                config,
                context_snippet,
                dimension_tracker,
            )
            if not adjusted:
                break
            next_token = self._sample(adjusted, profile["topp"], rng)
            if next_token is None:
                break
            generated.append(next_token)
            context_ids.append(next_token)
            context_ids[:] = context_ids[-(self.store.order - 1) :]
            if dimension_tracker:
                dimension_tracker.record(next_token)
            if self.tokenizer.vocab.token_text(next_token) == "<EOS>":
                break
        self.cache.update(conversation_id, generated)
        return generated

    def _adjust_candidates(
        self,
        conversation_id: str,
        candidates: Sequence[TokenCandidate],
        generated: List[int],
        profile: Dict[str, float | int],
        banned: set[int],
        config: DecoderConfig,
        context_snippet: str,
        dimension_tracker: ContextDimensionTracker | None,
    ) -> Dict[int, float]:
        base: Dict[int, float] = {}
        bias_map = self.bias.lookup(conversation_id, context_snippet)
        cache_dist = self.cache.distribution(conversation_id)
        lambda_cache = float(profile.get("lambda_cache", 0.15))
        lambda_cache = max(0.0, min(lambda_cache, 0.95))
        penalties: Dict[int, int] = {}
        for token_id in generated:
            penalties[token_id] = penalties.get(token_id, 0) + 1
        for candidate in candidates:
            token_id = candidate.token_id
            if token_id in banned:
                continue
            log10_val = self.quantizer.dequantize_log10(candidate.q_logprob)
            temperature = float(profile.get("temp", 1.0))
            if temperature != 1.0 and temperature > 0:
                log10_val /= temperature
            bias_delta = bias_map.get(token_id, 0)
            if bias_delta:
                log10_val += self._log_delta(bias_delta)
            penalty = 0.0
            if token_id in penalties:
                penalty += config.presence_penalty
                penalty += penalties[token_id] * config.frequency_penalty
            if dimension_tracker:
                penalty += dimension_tracker.penalty_for(
                    token_id,
                    config.presence_penalty,
                    config.frequency_penalty,
                )
            if penalty:
                log10_val -= penalty
            prob = 10 ** log10_val
            base[token_id] = prob
        if not base:
            return {}
        if cache_dist and lambda_cache > 0:
            for token_id in set(base) | set(cache_dist):
                cache_prob = cache_dist.get(token_id, 0.0)
                base_prob = base.get(token_id, 0.0)
                base[token_id] = (1 - lambda_cache) * base_prob + lambda_cache * cache_prob
        total = sum(base.values())
        if total <= 0:
            return {}
        for token_id in list(base.keys()):
            base[token_id] = base[token_id] / total
        return base

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

    def _load_bans(self, profile: str) -> set[int]:
        rows = self.db.query(
            "SELECT token_id FROM tbl_decode_bans WHERE profile = ?",
            (profile,),
        )
        return {row["token_id"] for row in rows}

    def _log_delta(self, q_bias: int) -> float:
        span = self.quantizer.Lmax - self.quantizer.Lmin
        return (q_bias / 255.0) * span

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
