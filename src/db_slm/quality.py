from __future__ import annotations

import math
import threading
from dataclasses import dataclass
from typing import Any, Sequence

from .metrics import lexical_overlap
from .sentence_parts import ExternalEmbedder
from .system import AdaptiveLoadController

__all__ = ["SentenceQualityScorer"]


def _cosine_similarity(vec_a: Sequence[float], vec_b: Sequence[float]) -> float:
    if not vec_a or not vec_b:
        return 0.0
    length = min(len(vec_a), len(vec_b))
    if length == 0:
        return 0.0
    dot = 0.0
    norm_a = 0.0
    norm_b = 0.0
    for idx in range(length):
        va = float(vec_a[idx])
        vb = float(vec_b[idx])
        dot += va * vb
        norm_a += va * va
        norm_b += vb * vb
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return max(-1.0, min(1.0, dot / math.sqrt(norm_a * norm_b)))


class _LanguageToolProxy:
    """Lazily loads language_tool_python when CPU headroom permits."""

    def __init__(self, language: str, load_guard: AdaptiveLoadController) -> None:
        self.language = language
        self._guard = load_guard
        self._tool = None
        self._lock = threading.Lock()
        self._warned = False

    def _ensure_tool(self):
        if self._tool is not None:
            return self._tool
        with self._lock:
            if self._tool is not None or not self._guard.allow_heavy_task():
                return self._tool
            try:
                import language_tool_python  # type: ignore

                self._tool = language_tool_python.LanguageTool(self.language)
                print(f"[quality] Loaded language_tool_python for grammar checks ({self.language}).")
            except Exception as exc:  # pragma: no cover - optional dependency
                if not self._warned:
                    print(f"[quality] Grammar checks unavailable ({exc}).")
                    self._warned = True
                self._tool = None
        return self._tool

    def issues(self, text: str) -> list[Any]:
        tool = self._ensure_tool()
        if tool is None or not text.strip():
            return []
        try:
            return tool.check(text)
        except Exception as exc:  # pragma: no cover - runtime guard
            if not self._warned:
                print(f"[quality] Grammar check failed ({exc}). Disabling tool.")
                self._warned = True
            self._tool = None
            return []


class _CoLAClassifier:
    """Wrapper around a HuggingFace acceptability classifier."""

    def __init__(self, model_name: str, load_guard: AdaptiveLoadController) -> None:
        self.model_name = model_name
        self._guard = load_guard
        self._model = None
        self._tokenizer = None
        self._lock = threading.Lock()
        self._warned = False

    def _ensure_model(self):
        if self._model is not None and self._tokenizer is not None:
            return self._tokenizer, self._model
        with self._lock:
            if (self._model is not None and self._tokenizer is not None) or not self._guard.allow_heavy_task():
                return self._tokenizer, self._model
            try:
                from transformers import AutoModelForSequenceClassification, AutoTokenizer  # type: ignore

                self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                self._model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
                print(f"[quality] Loaded CoLA classifier ({self.model_name}).")
            except Exception as exc:  # pragma: no cover - optional dependency
                if not self._warned:
                    print(f"[quality] Semantic acceptability model unavailable ({exc}).")
                    self._warned = True
                self._model = None
                self._tokenizer = None
        return self._tokenizer, self._model

    def score(self, text: str) -> float | None:
        if not text.strip():
            return None
        tokenizer, model = self._ensure_model()
        if tokenizer is None or model is None:
            return None
        try:
            import torch  # type: ignore

            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
            with torch.no_grad():
                logits = model(**inputs).logits
            probs = torch.nn.functional.softmax(logits, dim=1)
            acceptable = float(probs[0][1].item())
            return acceptable
        except Exception as exc:  # pragma: no cover - runtime guard
            if not self._warned:
                print(f"[quality] Semantic acceptability scoring failed ({exc}).")
                self._warned = True
            self._model = None
            self._tokenizer = None
            return None


@dataclass
class SentenceQualityScorer:
    """Bundles grammar, semantic acceptability, and embedding-based similarity checks."""

    embedder_model: str = "all-MiniLM-L6-v2"
    grammar_language: str = "en-US"
    embedder: ExternalEmbedder | None = None
    load_guard: AdaptiveLoadController | None = None

    def __post_init__(self) -> None:
        guard = self.load_guard or AdaptiveLoadController()
        self.load_guard = guard
        self.embedder = self.embedder or ExternalEmbedder(self.embedder_model)
        self._grammar = _LanguageToolProxy(self.grammar_language, guard)
        self._cola = _CoLAClassifier("textattack/roberta-base-CoLA", guard)

    def grammar_metrics(self, text: str) -> dict[str, float | int]:
        issues = self._grammar.issues(text)
        if not issues:
            return {"grammar_errors": 0, "grammar_score": 1.0}
        tokens = max(1, len(text.split()))
        score = max(0.0, 1.0 - (len(issues) / tokens))
        return {"grammar_errors": len(issues), "grammar_score": round(score, 4)}

    def acceptability_metrics(self, text: str) -> dict[str, float | None]:
        score = self._cola.score(text)
        if score is None:
            return {"cola_acceptability": None}
        return {"cola_acceptability": round(score, 4)}

    def semantic_comparison(
        self,
        reference: str,
        candidate: str,
        lexical_similarity: float | None = None,
    ) -> dict[str, float]:
        if not reference.strip() or not candidate.strip():
            return {
                "semantic_similarity": 0.0,
                "semantic_distance": 1.0,
                "semantic_novelty": 0.0,
                "length_ratio": 1.0,
                "length_gap": 0.0,
            }
        vectors = self.embedder.embed([reference, candidate])
        ref_vector = vectors[0] if len(vectors) > 0 else []
        cand_vector = vectors[1] if len(vectors) > 1 else []
        similarity = (_cosine_similarity(ref_vector, cand_vector) + 1.0) / 2.0
        lexical_sim = lexical_similarity if lexical_similarity is not None else lexical_overlap(reference, candidate)
        lexical_novelty = max(0.0, 1.0 - lexical_sim)
        semantic_distance = 1.0 - similarity
        semantic_novelty = similarity * lexical_novelty
        ref_words = max(1, len(reference.split()))
        cand_words = max(1, len(candidate.split()))
        length_ratio = cand_words / ref_words
        return {
            "semantic_similarity": round(similarity, 4),
            "semantic_distance": round(semantic_distance, 4),
            "semantic_novelty": round(semantic_novelty, 4),
            "length_ratio": round(length_ratio, 3),
            "length_gap": round(abs(1.0 - length_ratio), 3),
        }

    def combined_quality(
        self,
        candidate: str,
        reference: str,
        lexical_similarity: float,
    ) -> dict[str, float | int | None]:
        metrics: dict[str, float | int | None] = {}
        metrics.update(self.grammar_metrics(candidate))
        metrics.update(self.acceptability_metrics(candidate))
        metrics.update(self.semantic_comparison(reference, candidate, lexical_similarity))
        grammar = metrics.get("grammar_score") or 0.0
        cola = metrics.get("cola_acceptability") or 0.0
        semantic = metrics.get("semantic_similarity") or 0.0
        lexical_novelty = max(0.0, 1.0 - lexical_similarity)
        metrics["quality_score"] = round(
            0.45 * semantic + 0.35 * cola + 0.2 * grammar + 0.15 * lexical_novelty,
            4,
        )
        return metrics
