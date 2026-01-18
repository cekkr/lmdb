from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Sequence

from .context_dimensions import ContextDimensionTracker
from .level1 import LogProbQuantizer, TokenCandidate, Tokenizer
from .level2 import BiasEngine, SessionCache


@dataclass
class CandidateScore:
    token_id: int
    token_text: str
    base_log10: float
    temperature: float
    bias_delta: float
    presence_penalty: float
    frequency_penalty: float
    dimension_penalty: float
    base_prob: float
    cache_prob: float
    prediction_prob: float
    final_prob: float


@dataclass
class ScoreResult:
    distribution: Dict[int, float]
    trace: List[CandidateScore] | None = None


@dataclass
class ScoreSnapshot:
    step_index: int
    context_ids: Sequence[int]
    scores: Sequence[CandidateScore]
    chosen_token_id: int | None


ScoreObserver = Callable[[ScoreSnapshot], None]


class TokenScoringPipeline:
    """Score candidate tokens with optional trace data for debugging."""

    def __init__(
        self,
        quantizer: LogProbQuantizer,
        tokenizer: Tokenizer,
        cache: SessionCache,
        bias: BiasEngine,
    ) -> None:
        self.quantizer = quantizer
        self.tokenizer = tokenizer
        self.cache = cache
        self.bias = bias

    def score(
        self,
        conversation_id: str,
        candidates: Sequence[TokenCandidate],
        generated: Sequence[int],
        *,
        banned: set[int],
        context_snippet: str,
        temperature: float,
        lambda_cache: float,
        presence_penalty: float,
        frequency_penalty: float,
        dimension_tracker: ContextDimensionTracker | None,
        prediction_bias: Dict[int, float] | None,
        prediction_weight: float,
        collect_trace: bool = False,
    ) -> ScoreResult:
        bias_map = self.bias.lookup(conversation_id, context_snippet)
        cache_dist = self.cache.distribution(conversation_id)
        lambda_cache = max(0.0, min(float(lambda_cache), 0.95))
        prediction_weight = max(0.0, min(float(prediction_weight), 1.0))
        penalties: Dict[int, int] = {}
        for token_id in generated:
            penalties[token_id] = penalties.get(token_id, 0) + 1

        trace_map: Dict[int, CandidateScore] | None = {} if collect_trace else None
        base: Dict[int, float] = {}
        for candidate in candidates:
            token_id = candidate.token_id
            if token_id in banned:
                continue
            log10_val = self.quantizer.dequantize_log10(candidate.q_logprob)
            if temperature != 1.0 and temperature > 0:
                log10_val /= temperature
            bias_delta = bias_map.get(token_id, 0)
            if bias_delta:
                log10_val += self._log_delta(bias_delta)
            presence_hit = 0.0
            frequency_hit = 0.0
            if token_id in penalties:
                presence_hit = presence_penalty
                frequency_hit = penalties[token_id] * frequency_penalty
            dimension_hit = 0.0
            if dimension_tracker:
                dimension_hit = dimension_tracker.penalty_for(
                    token_id,
                    presence_penalty,
                    frequency_penalty,
                )
            total_penalty = presence_hit + frequency_hit + dimension_hit
            if total_penalty:
                log10_val -= total_penalty
            prob = 10 ** log10_val
            base[token_id] = prob
            if trace_map is not None:
                trace_map[token_id] = CandidateScore(
                    token_id=token_id,
                    token_text=self.tokenizer.vocab.token_text(token_id),
                    base_log10=log10_val,
                    temperature=temperature,
                    bias_delta=bias_delta,
                    presence_penalty=presence_hit,
                    frequency_penalty=frequency_hit,
                    dimension_penalty=dimension_hit,
                    base_prob=prob,
                    cache_prob=0.0,
                    prediction_prob=0.0,
                    final_prob=0.0,
                )

        if not base:
            return ScoreResult({}, self._sorted_trace(trace_map))

        if cache_dist and lambda_cache > 0:
            for token_id in set(base) | set(cache_dist):
                base_prob = base.get(token_id, 0.0)
                cache_prob = cache_dist.get(token_id, 0.0)
                base[token_id] = (1.0 - lambda_cache) * base_prob + lambda_cache * cache_prob
                if trace_map is not None:
                    entry = trace_map.get(token_id)
                    if entry is None:
                        entry = CandidateScore(
                            token_id=token_id,
                            token_text=self.tokenizer.vocab.token_text(token_id),
                            base_log10=0.0,
                            temperature=temperature,
                            bias_delta=0.0,
                            presence_penalty=0.0,
                            frequency_penalty=0.0,
                            dimension_penalty=0.0,
                            base_prob=base_prob,
                            cache_prob=cache_prob,
                            prediction_prob=0.0,
                            final_prob=0.0,
                        )
                        trace_map[token_id] = entry
                    else:
                        entry.cache_prob = cache_prob

        distribution = self._normalize(base)
        if not distribution:
            return ScoreResult({}, self._sorted_trace(trace_map))

        if prediction_bias and prediction_weight > 0.0:
            blended = dict(distribution)
            for token_id, prob in prediction_bias.items():
                existing = blended.get(token_id, 0.0)
                blended[token_id] = (1.0 - prediction_weight) * existing + prediction_weight * prob
                if trace_map is not None:
                    entry = trace_map.get(token_id)
                    if entry is None:
                        entry = CandidateScore(
                            token_id=token_id,
                            token_text=self.tokenizer.vocab.token_text(token_id),
                            base_log10=0.0,
                            temperature=temperature,
                            bias_delta=0.0,
                            presence_penalty=0.0,
                            frequency_penalty=0.0,
                            dimension_penalty=0.0,
                            base_prob=existing,
                            cache_prob=0.0,
                            prediction_prob=prob,
                            final_prob=0.0,
                        )
                        trace_map[token_id] = entry
                    else:
                        entry.prediction_prob = prob
            distribution = self._normalize(blended)

        if trace_map is not None:
            for token_id, entry in trace_map.items():
                entry.final_prob = distribution.get(token_id, 0.0)

        return ScoreResult(distribution, self._sorted_trace(trace_map))

    def _normalize(self, distribution: Dict[int, float]) -> Dict[int, float]:
        total = sum(distribution.values())
        if total <= 0:
            return {}
        for token_id in list(distribution.keys()):
            distribution[token_id] = distribution[token_id] / total
        return distribution

    @staticmethod
    def _sorted_trace(trace_map: Dict[int, CandidateScore] | None) -> List[CandidateScore] | None:
        if trace_map is None:
            return None
        return sorted(trace_map.values(), key=lambda item: item.final_prob, reverse=True)

    def _log_delta(self, q_bias: int) -> float:
        span = self.quantizer.Lmax - self.quantizer.Lmin
        return (q_bias / 255.0) * span

