from __future__ import annotations

import unittest

from db_slm.level1 import TokenCandidate
from db_slm.scoring import TokenScoringPipeline


class _FakeQuantizer:
    Lmin = -8.0
    Lmax = 0.0

    @staticmethod
    def dequantize_log10(_q_logprob: int) -> float:
        return -1.0


class _FakeVocabulary:
    @staticmethod
    def token_text(token_id: int) -> str:
        return f"token-{token_id}"


class _FakeTokenizer:
    vocab = _FakeVocabulary()


class _FakeCache:
    def __init__(self, distribution: dict[int, float]) -> None:
        self._distribution = distribution

    def distribution(self, _conversation_id: str) -> dict[int, float]:
        return dict(self._distribution)


class _FakeBias:
    @staticmethod
    def lookup(_conversation_id: str, _context_snippet: str) -> dict[int, int]:
        return {}


class TokenScoringPipelineTests(unittest.TestCase):
    def _score(
        self,
        *,
        cache_distribution: dict[int, float],
        prediction_bias: dict[int, float] | None,
        prediction_weight: float,
        extra_bias: dict[int, int] | None = None,
    ):
        pipeline = TokenScoringPipeline(
            _FakeQuantizer(),  # type: ignore[arg-type]
            _FakeTokenizer(),  # type: ignore[arg-type]
            _FakeCache(cache_distribution),  # type: ignore[arg-type]
            _FakeBias(),  # type: ignore[arg-type]
        )
        return pipeline.score(
            "conversation",
            [TokenCandidate(token_id=1, token_text="allowed", probability=0.1, q_logprob=128)],
            [],
            banned={2},
            context_snippet="",
            temperature=1.0,
            lambda_cache=0.5,
            presence_penalty=0.0,
            frequency_penalty=0.0,
            dimension_tracker=None,
            prediction_bias=prediction_bias,
            prediction_weight=prediction_weight,
            collect_trace=True,
            extra_bias=extra_bias,
        )

    def test_banned_token_cannot_reenter_through_session_cache(self) -> None:
        result = self._score(
            cache_distribution={1: 0.1, 2: 0.9},
            prediction_bias=None,
            prediction_weight=0.0,
        )

        self.assertEqual(set(result.distribution), {1})
        self.assertNotIn(2, {entry.token_id for entry in result.trace or []})

    def test_banned_token_cannot_reenter_through_prediction_bias(self) -> None:
        result = self._score(
            cache_distribution={},
            prediction_bias={1: 0.1, 2: 0.9},
            prediction_weight=0.5,
        )

        self.assertEqual(set(result.distribution), {1})
        self.assertNotIn(2, {entry.token_id for entry in result.trace or []})

    def test_banned_token_cannot_reenter_through_graph_bias(self) -> None:
        result = self._score(
            cache_distribution={},
            prediction_bias=None,
            prediction_weight=0.0,
            extra_bias={1: 40, 2: 200},
        )

        self.assertEqual(set(result.distribution), {1})
        self.assertNotIn(2, {entry.token_id for entry in result.trace or []})

    def test_graph_bias_raises_the_score_of_an_allowed_token(self) -> None:
        baseline = self._score(
            cache_distribution={},
            prediction_bias=None,
            prediction_weight=0.0,
        )
        biased = self._score(
            cache_distribution={},
            prediction_bias=None,
            prediction_weight=0.0,
            extra_bias={1: 64},
        )

        baseline_entry = next(entry for entry in baseline.trace or [] if entry.token_id == 1)
        biased_entry = next(entry for entry in biased.trace or [] if entry.token_id == 1)
        self.assertEqual(baseline_entry.bias_delta, 0)
        self.assertEqual(biased_entry.bias_delta, 64)
        self.assertGreater(biased_entry.base_log10, baseline_entry.base_log10)


if __name__ == "__main__":
    unittest.main()
