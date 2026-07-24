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


if __name__ == "__main__":
    unittest.main()
