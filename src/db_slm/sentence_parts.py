from __future__ import annotations

import os
import re
import statistics
import threading
from collections import Counter
from typing import Iterable, List, Sequence

from .metrics import keyword_summary
from .settings import DBSLMSettings

_DEFAULT_SPLITS = [".", "!", "?", ";", ":", ",", "\n"]
_EMOTION_RE = re.compile(r"emotion\s*:\s*([A-Za-z0-9 _-]+)", re.IGNORECASE)
_EMOTION_WORDS = {
    "joy",
    "trust",
    "fear",
    "surprise",
    "sadness",
    "disgust",
    "anger",
    "anticipation",
    "calm",
    "hope",
    "gratitude",
    "curiosity",
    "empathy",
    "doubt",
    "stress",
}


class RealtimeTokenizerProfiler:
    """Collects punctuation stats to drive adaptive sentence splits."""

    def __init__(self, report_interval_chars: int = 50_000) -> None:
        self._punct_counts: Counter[str] = Counter()
        self._segment_lengths: list[int] = []
        self._chars_seen = 0
        self._last_report = 0
        self._report_interval = report_interval_chars
        self._lock = threading.Lock()

    def observe(self, text: str) -> None:
        if not text:
            return
        with self._lock:
            self._chars_seen += len(text)
            for ch in text:
                if ch in _DEFAULT_SPLITS:
                    self._punct_counts[ch] += 1
            segments = [chunk.strip() for chunk in re.split(r"[.!?\n]", text) if chunk.strip()]
            for segment in segments:
                self._segment_lengths.append(len(segment))
            # cap memory usage
            if len(self._segment_lengths) > 10_000:
                self._segment_lengths = self._segment_lengths[-5_000:]

    def preferred_splits(self) -> Sequence[str]:
        if not self._punct_counts:
            return list(_DEFAULT_SPLITS)
        return [token for token, _ in self._punct_counts.most_common(6)]

    def target_segment_length(self) -> int:
        if not self._segment_lengths:
            return 220
        trimmed = sorted(self._segment_lengths)
        median_idx = int(len(trimmed) * 0.6)
        median_idx = min(max(median_idx, 0), len(trimmed) - 1)
        target = trimmed[median_idx]
        return max(80, min(int(target), 420))

    def snapshot(self) -> dict[str, object] | None:
        with self._lock:
            if self._chars_seen - self._last_report < self._report_interval:
                return None
            self._last_report = self._chars_seen
            return {
                "chars_seen": self._chars_seen,
                "top_splits": list(self.preferred_splits()),
                "target_segment_len": self.target_segment_length(),
            }


class SentenceSegmenter:
    """Splits long documents into punctuation-aware chunks."""

    def segment(self, text: str, split_chars: Sequence[str], max_length: int) -> List[str]:
        if not text.strip():
            return []
        delimiters = set(split_chars) | {"\n"}
        segments: list[str] = []
        buffer: list[str] = []
        for ch in text:
            buffer.append(ch)
            if ch in delimiters or len(buffer) >= max_length:
                segment = "".join(buffer).strip()
                if segment:
                    segments.append(segment)
                buffer = []
        if buffer:
            segment = "".join(buffer).strip()
            if segment:
                segments.append(segment)
        return segments


class ExternalEmbedder:
    """Wrapper around sentence-transformers with graceful degradation."""

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self._model = None
        self._device = "cpu"
        self._warned = False
        self._load_lock = threading.Lock()
        self._load_model()

    def _load_model(self) -> None:
        with self._load_lock:
            try:
                from sentence_transformers import SentenceTransformer  # type: ignore
            except ImportError:
                if not self._warned:
                    print(
                        "[embedding] sentence-transformers not installed; "
                        "install it to enable external embedding guidance."
                    )
                    self._warned = True
                return
            try:
                self._device = self._select_device()
                self._model = SentenceTransformer(self.model_name, device=self._device)
                print(
                    f"[embedding] Loaded {self.model_name} on {self._device} "
                    "for sentence segmentation guidance."
                )
            except Exception as exc:  # pragma: no cover - optional dependency
                if not self._warned:
                    print(f"[embedding] Failed to load {self.model_name}: {exc}")
                    self._warned = True
                self._model = None

    def _select_device(self) -> str:
        try:
            import torch  # type: ignore

            if torch.cuda.is_available():
                return "cuda"
            if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
                return "mps"
        except Exception:  # pragma: no cover - optional detection
            pass
        return "cpu"

    def embed(self, segments: Sequence[str]) -> List[List[float]]:
        if not segments:
            return []
        if self._model is None:
            return [self._hashed_vector(text) for text in segments]
        try:
            vectors = self._model.encode(
                list(segments),
                convert_to_numpy=True,
                batch_size=min(32, len(segments)),
                show_progress_bar=False,
            )
            return [vector.tolist() for vector in vectors]
        except Exception as exc:  # pragma: no cover - runtime guard
            if not self._warned:
                print(f"[embedding] Encoding failed ({exc}); falling back to hashed vectors.")
                self._warned = True
            self._model = None
            return [self._hashed_vector(text) for text in segments]

    def signature(self, vector: Sequence[float] | None) -> str:
        if not vector:
            return "|EMB|:0.000:0.000:0.000"
        bucket = max(1, len(vector) // 3)
        slices = [vector[i : i + bucket] for i in range(0, min(len(vector), bucket * 3), bucket)]
        means: list[str] = []
        for chunk in slices:
            if not chunk:
                continue
            mean = statistics.fmean(chunk)
            means.append(f"{mean:.3f}")
        while len(means) < 3:
            means.append("0.000")
        return "|EMB|:" + ":".join(means[:3])

    def energy(self, vector: Sequence[float] | None) -> float:
        if not vector:
            return 0.0
        return sum(abs(value) for value in vector) / max(len(vector), 1)

    def _hashed_vector(self, text: str) -> List[float]:
        digest = abs(hash(text)) % 10_000
        return [digest / 10_000.0, (digest % 997) / 997.0, (digest % 577) / 577.0]


class SentencePartEmbeddingPipeline:
    """
    Performs punctuation-aware segmentation, optional embedding lookups,
    and injects emotional keywords ahead of Level 1 tokenization.
    """

    def __init__(self, settings: DBSLMSettings) -> None:
        model = settings.embedder_model or os.environ.get("DBSLM_EMBEDDER_MODEL", "all-MiniLM-L6-v2")
        self.profiler = RealtimeTokenizerProfiler()
        self.segmenter = SentenceSegmenter()
        self.embedder = ExternalEmbedder(model)

    def prepare_for_training(self, text: str) -> str:
        payload = text.strip()
        if not payload:
            return text
        self.profiler.observe(payload)
        splits = self.profiler.preferred_splits()
        target = self.profiler.target_segment_length()
        segments = self.segmenter.segment(payload, splits, target)
        if not segments:
            segments = [payload]
        vectors = self.embedder.embed(segments)
        lines: list[str] = []
        header = self._emotion_header(payload)
        if header:
            lines.append(header)
        for idx, segment in enumerate(segments):
            vector = vectors[idx] if idx < len(vectors) else []
            signature = self.embedder.signature(vector)
            segment_line = f"|SEGMENT|#{idx + 1} {signature} {segment.strip()}"
            lines.append(segment_line)
            emo_keywords = self._emotional_keywords(segment, vector)
            if emo_keywords:
                lines.append("|EMO_KEY| " + " ".join(emo_keywords))
        lines.append("|RAW|")
        lines.append(payload)

        snapshot = self.profiler.snapshot()
        if snapshot:
            print(
                "[tokenizer] realtime splits={splits} target_len={target} chars={chars}".format(
                    splits=",".join(snapshot["top_splits"]),
                    target=snapshot["target_segment_len"],
                    chars=snapshot["chars_seen"],
                )
            )
        return "\n".join(lines)

    def _emotion_header(self, text: str) -> str | None:
        matches = _EMOTION_RE.findall(text)
        if not matches:
            return None
        tokens = []
        for label in dict.fromkeys(match.strip().lower() for match in matches if match.strip()):
            normalized = label.replace(" ", "_")
            tokens.append(f"|EMOTION|:{normalized}")
        if not tokens:
            return None
        return " ".join(tokens)

    def _emotional_keywords(self, segment: str, vector: Sequence[float]) -> List[str]:
        candidates = keyword_summary(segment, limit=5)
        hits = [word for word in candidates if word in _EMOTION_WORDS]
        if hits:
            return hits
        energy = self.embedder.energy(vector)
        if energy <= 0.12:
            limit = 1
        elif energy <= 0.25:
            limit = 2
        else:
            limit = 3
        return candidates[:limit]

