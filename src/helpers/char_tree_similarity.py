"""
char_tree_similarity.py

A small library that exposes a `similarity_score` function.

The algorithm works in two layers:

1. Character-level CharTree:
   - For each input string we build a prefix tree of all substrings
     up to a maximum length (default: 32 characters).
   - While building the tree we also count how many times every substring
     occurs. This is a kind of n-gram statistics without pre-splitting
     into words.
   - From these substring counts we derive a first similarity score
     by comparing the weighted multiset of substrings of the two strings.

2. Token / "word" level:
   - We split the (upper-cased) strings on whitespace into tokens.
   - We compute a Levenshtein-like distance on the token sequence,
     where substituting two tokens has a cost proportional to their
     character-level edit distance.
   - This gives a second score that captures sentence structure and
     word order, so that e.g. "BEEN IN BERLIN" and "BEEN AT BERLIN"
     stay very similar.

The final similarity is a weighted mix of the two scores.

By default both input strings are converted to UPPERCASE before
any processing, as requested.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Union, Any


# ---------------------------------------------------------------------------
# Char tree data structures
# ---------------------------------------------------------------------------

@dataclass
class CharTreeNode:
    """Node of a character prefix tree.

    Attributes
    ----------
    children:
        Mapping from next character to child node.
    count_end:
        How many times the substring that ends at this node appears
        in the text.
    """
    children: Dict[str, "CharTreeNode"] = field(default_factory=dict)
    count_end: int = 0


@dataclass
class CharTree:
    """Character prefix tree plus substring statistics."""
    root: CharTreeNode
    substring_counts: Dict[str, int]
    max_length: int = 32

    @classmethod
    def from_text(
        cls,
        text: str,
        max_length: int = 32,
        min_substring_len: int = 1,
    ) -> "CharTree":
        """Build a CharTree from `text`.

        Parameters
        ----------
        text:
            Input string (you probably want to uppercase it beforehand).
        max_length:
            Maximum length of substrings to consider.
        min_substring_len:
            Minimum substring length to store in `substring_counts`.
        """
        root = CharTreeNode()
        counts: Dict[str, int] = defaultdict(int)
        n = len(text)

        for i in range(n):
            node = root
            limit = min(max_length, n - i)
            for length in range(1, limit + 1):
                ch = text[i + length - 1]
                child = node.children.get(ch)
                if child is None:
                    child = CharTreeNode()
                    node.children[ch] = child
                node = child
                node.count_end += 1

                if length >= min_substring_len:
                    substring = text[i : i + length]
                    counts[substring] += 1

        return cls(root=root, substring_counts=dict(counts), max_length=max_length)


# ---------------------------------------------------------------------------
# Substring multiset similarity (CharTree-based)
# ---------------------------------------------------------------------------

def substring_multiset_similarity(
    counts1: Dict[str, int],
    counts2: Dict[str, int],
    min_len: int = 2,
) -> float:
    """Compare two substring frequency maps as weighted multisets.

    Similarity is in [0, 1].

    We weight each substring by its length so longer common patterns
    contribute more than short ones, and we ignore substrings shorter
    than `min_len` (usually 2) to avoid noise from single characters.
    """
    if not counts1 or not counts2:
        return 0.0

    keys = set(k for k in counts1 if len(k) >= min_len)
    keys |= set(k for k in counts2 if len(k) >= min_len)

    if not keys:
        return 0.0

    num = 0.0
    den = 0.0

    for k in keys:
        c1 = counts1.get(k, 0)
        c2 = counts2.get(k, 0)
        if c1 == 0 and c2 == 0:
            continue
        weight = float(len(k))
        num += min(c1, c2) * weight
        den += max(c1, c2) * weight

    if den == 0.0:
        return 0.0

    return num / den


# ---------------------------------------------------------------------------
# Token-level Levenshtein (on "words")
# ---------------------------------------------------------------------------

def levenshtein_char(a: str, b: str) -> int:
    """Standard Levenshtein distance on characters."""
    if a == b:
        return 0
    la, lb = len(a), len(b)
    if la == 0:
        return lb
    if lb == 0:
        return la

    # 1D dynamic programming
    prev_row = list(range(lb + 1))
    for i, ca in enumerate(a, start=1):
        cur_row = [i] + [0] * lb
        for j, cb in enumerate(b, start=1):
            cost = 0 if ca == cb else 1
            cur_row[j] = min(
                prev_row[j] + 1,      # deletion
                cur_row[j - 1] + 1,   # insertion
                prev_row[j - 1] + cost,  # substitution
            )
        prev_row = cur_row
    return prev_row[lb]


def token_sub_cost(t1: str, t2: str) -> float:
    """Cost to substitute token t1 with t2, in [0, 1]."""
    if t1 == t2:
        return 0.0
    dist = levenshtein_char(t1, t2)
    maxlen = max(len(t1), len(t2))
    if maxlen == 0:
        return 0.0
    return dist / maxlen


def levenshtein_tokens(tokens1: List[str], tokens2: List[str]) -> float:
    """Levenshtein-like distance on sequences of tokens.

    Insertion/deletion cost 1.0, substitution cost in [0,1] obtained
    from normalised character-level distance between tokens.
    """
    n, m = len(tokens1), len(tokens2)
    if n == 0:
        return float(m)
    if m == 0:
        return float(n)

    dp = [[0.0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        dp[i][0] = float(i)
    for j in range(1, m + 1):
        dp[0][j] = float(j)

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            sub_cost = token_sub_cost(tokens1[i - 1], tokens2[j - 1])
            dp[i][j] = min(
                dp[i - 1][j] + 1.0,        # deletion
                dp[i][j - 1] + 1.0,        # insertion
                dp[i - 1][j - 1] + sub_cost,  # substitution
            )

    return dp[n][m]


def token_sequence_similarity(tokens1: List[str], tokens2: List[str]) -> float:
    """Convert token-distance into a similarity in [0, 1]."""
    if not tokens1 and not tokens2:
        return 1.0

    dist = levenshtein_tokens(tokens1, tokens2)
    max_len = max(len(tokens1), len(tokens2))
    if max_len == 0:
        return 1.0

    norm = dist / max_len  # normalised distance
    sim = 1.0 - norm
    return max(0.0, sim)


# ---------------------------------------------------------------------------
# Extract "important" recurring patterns (mostly for inspection/debug)
# ---------------------------------------------------------------------------

def extract_significant_patterns(
    counts: Dict[str, int],
    min_len: int = 3,
    max_patterns: int = 50,
) -> List[Tuple[str, int]]:
    """Return the most recurrent substrings above the average frequency.

    This does not directly influence the similarity score, but it gives
    you a compact "tokenization" of the most important character-series
    found in a string.

    Returns a list of (substring, count) sorted by (count * length)
    descending, truncated to `max_patterns` elements.
    """
    candidates = [(s, c) for s, c in counts.items() if len(s) >= min_len]
    if not candidates:
        return []

    avg = sum(c for _, c in candidates) / len(candidates)
    significant = [(s, c) for s, c in candidates if c > avg]

    significant.sort(
        key=lambda sc: (sc[1] * len(sc[0]), sc[1], len(sc[0])),
        reverse=True,
    )

    if max_patterns:
        significant = significant[: max_patterns]

    return significant


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def similarity_score(
    text1: str,
    text2: str,
    *,
    max_substring_len: int = 32,
    uppercase: bool = True,
    substring_weight: float = 0.3,
    return_details: bool = False,
) -> Union[float, Tuple[float, Dict[str, Any]]]:
    """Compute a similarity score between two strings.

    Parameters
    ----------
    text1, text2:
        Input strings. By default both are converted to UPPERCASE so
        that the comparison is case-insensitive.
    max_substring_len:
        Maximum length of substrings to consider when building the
        CharTree (default: 32).
    uppercase:
        If True (default) the two strings are uppercased before any
        processing.
    substring_weight:
        Weight in [0,1] for the CharTree-based substring similarity.
        The token / word-level similarity gets weight (1 - substring_weight).
        A smaller value means the algorithm trusts more the word-level
        Levenshtein distance. Default: 0.3.
    return_details:
        If True, return a tuple (score, details_dict) where
        `details_dict` exposes internal components of the score.

    Returns
    -------
    float
        Similarity score in [0, 1]; 1 means identical, 0 means
        completely different.
    or (float, dict)
        If `return_details=True`.
    """
    if uppercase:
        text1 = text1.upper()
        text2 = text2.upper()

    # --- CharTree / substring-based similarity ---
    tree1 = CharTree.from_text(text1, max_length=max_substring_len)
    tree2 = CharTree.from_text(text2, max_length=max_substring_len)

    substring_score = substring_multiset_similarity(
        tree1.substring_counts,
        tree2.substring_counts,
        min_len=2,
    )

    # --- Token / word-level similarity ---
    tokens1 = text1.split()
    tokens2 = text2.split()
    token_score = token_sequence_similarity(tokens1, tokens2)

    # --- Combine the two scores ---
    if substring_score == 0.0 and token_score != 0.0:
        overall = token_score
    elif token_score == 0.0 and substring_score != 0.0:
        overall = substring_score
    else:
        w = float(substring_weight)
        w = min(max(w, 0.0), 1.0)
        overall = w * substring_score + (1.0 - w) * token_score

    if not return_details:
        return overall

    details: Dict[str, Any] = {
        "substring_score": substring_score,
        "token_score": token_score,
        "patterns1": extract_significant_patterns(tree1.substring_counts),
        "patterns2": extract_significant_patterns(tree2.substring_counts),
    }
    return overall, details


# Convenience alias, if you prefer a more "verb-like" name.
compare_strings = similarity_score


if __name__ == "__main__":
    # Small manual test
    s1 = "Been in Berlin"
    s2 = "Been at Berlin"
    score, info = similarity_score(s1, s2, return_details=True)
    print("Score:", score)
    print("Substring score:", info["substring_score"])
    print("Token score:", info["token_score"])
