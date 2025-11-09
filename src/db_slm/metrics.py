from __future__ import annotations

from collections import Counter
import re
from typing import Iterable, List

__all__ = ["lexical_overlap", "rouge_l_score", "keyword_summary"]

_WORD_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)
_PUNCT_STRIP = '.,!?()[]{}"\'`“”’'
_STOPWORDS = {
    "the",
    "a",
    "an",
    "and",
    "or",
    "but",
    "so",
    "of",
    "in",
    "on",
    "for",
    "to",
    "with",
    "that",
    "this",
    "these",
    "those",
    "it",
    "its",
    "be",
    "is",
    "are",
    "was",
    "were",
    "as",
    "at",
    "by",
    "about",
    "from",
    "into",
    "your",
    "my",
    "our",
    "we",
    "you",
    "me",
}


def tokenize(text: str) -> List[str]:
    """Lightweight tokenizer for metric computations."""
    if not text:
        return []
    tokens: list[str] = []
    for raw in _WORD_RE.findall(text):
        cleaned = raw.strip(_PUNCT_STRIP).lower()
        if cleaned:
            tokens.append(cleaned)
    return tokens


def lexical_overlap(reference: str, candidate: str) -> float:
    """Compute lexical overlap ratio between reference and candidate tokens."""
    ref_tokens = set(tokenize(reference))
    if not ref_tokens:
        return 0.0
    cand_tokens = set(tokenize(candidate))
    if not cand_tokens:
        return 0.0
    return len(ref_tokens & cand_tokens) / len(ref_tokens)


def rouge_l_score(reference: str, candidate: str) -> float:
    """ROUGE-L F1 approximation using token LCS."""
    ref_tokens = tokenize(reference)
    cand_tokens = tokenize(candidate)
    if not ref_tokens or not cand_tokens:
        return 0.0
    lcs = _lcs_length(ref_tokens, cand_tokens)
    if lcs == 0:
        return 0.0
    recall = lcs / len(ref_tokens)
    precision = lcs / len(cand_tokens)
    if recall == 0.0 or precision == 0.0:
        return 0.0
    beta2 = 1.0
    return (1 + beta2) * precision * recall / (recall + beta2 * precision)


def _lcs_length(seq_a: Iterable[str], seq_b: Iterable[str]) -> int:
    a = list(seq_a)
    b = list(seq_b)
    if not a or not b:
        return 0
    dp = [0] * (len(b) + 1)
    for token_a in a:
        prev = 0
        for idx, token_b in enumerate(b, start=1):
            tmp = dp[idx]
            if token_a == token_b:
                dp[idx] = prev + 1
            else:
                dp[idx] = max(dp[idx], dp[idx - 1])
            prev = tmp
    return dp[-1]


def keyword_summary(text: str, limit: int = 5) -> List[str]:
    """Return up to `limit` key terms ranked by frequency (stop words removed)."""
    tokens = [tok for tok in tokenize(text) if tok not in _STOPWORDS]
    if not tokens:
        return []
    counts = Counter(tokens)
    return [word for word, _ in counts.most_common(limit)]
