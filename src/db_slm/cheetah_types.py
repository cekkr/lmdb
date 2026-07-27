from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple

CHEETAH_DEFAULT_REDUCE_PAGE_SIZE = 1024
CHEETAH_PAIR_SCAN_MIN_LIMIT = 256
CHEETAH_PAIR_SCAN_MAX_LIMIT = 4096

# Server-side bounds mirrored from cheetah-db/src/graph_recall.go and
# cheetah-db/src/graph.go. Requests above these values are silently clamped by
# the server; clamping here keeps the command line honest about what was asked.
GRAPH_RECALL_MAX_SEEDS = 32
GRAPH_RECALL_MAX_HOPS = 6
GRAPH_RECALL_MAX_BRANCH = 1024
GRAPH_RECALL_MAX_BUDGET = 262144
GRAPH_RECALL_MAX_REFERENCES = 256
GRAPH_NODE_MAX_REFERENCES = 64
GRAPH_NODE_MAX_REFERENCE_CHARS = 4096
GRAPH_NODE_MAX_REFERENCE_TOTAL_CHARS = 65536


@dataclass(frozen=True)
class RawContextProjection:
    """Minimal projection returned by cheetah context scans."""

    context_hash: str
    order_size: int
    token_ids: Tuple[int, ...]
    ranked: Tuple[tuple[int, int], ...]
    cheetah_key: int
    vector_signature: bytes


@dataclass(frozen=True)
class RawCountsProjection:
    """Follower counts stored inside cheetah namespaces."""

    context_hash: str
    order: int
    totals: int
    followers: Tuple[tuple[int, int], ...]


@dataclass(frozen=True)
class RawProbabilityProjection:
    """Quantized probability rows mirrored from MKNS."""

    context_hash: str
    order: int
    followers: Tuple[tuple[int, int, int | None], ...]


@dataclass(frozen=True)
class RawContinuationProjection:
    """Continuation metadata mirrored into cheetah."""

    token_id: int
    num_contexts: int


@dataclass(frozen=True)
class PredictionValueResult:
    """Single value returned from a cheetah prediction table query."""

    value: bytes
    probability: float


@dataclass(frozen=True)
class PredictionQueryResult:
    """Result metadata returned by `PREDICT_QUERY`."""

    table: str
    backend: str
    count: int
    entries: Tuple[PredictionValueResult, ...]


@dataclass(frozen=True)
class GraphReferenceSentence:
    """A complete sentence stored as readable provenance on a graph node."""

    reference_id: str
    text: str
    source: str = ""
    ordinal: int = 0

    def as_payload(self) -> dict[str, object]:
        payload: dict[str, object] = {"text": self.text}
        if self.reference_id:
            payload["id"] = self.reference_id
        if self.source:
            payload["source"] = self.source
        if self.ordinal:
            payload["ordinal"] = int(self.ordinal)
        return payload


@dataclass(frozen=True)
class GraphRecallEdge:
    """One edge of the ``via`` evidence path returned by ``GRAPH_RECALL``."""

    from_id: str
    to_id: str
    edge_type: str
    weight: float
    confidence: float
    modality: str
    source: str = ""


@dataclass(frozen=True)
class GraphRecallSource:
    """A seed that reached an association, with its own activation and hops."""

    seed: str
    activation: float
    hops: int


@dataclass(frozen=True)
class GraphRecallSeedMatch:
    """A node a free-text seed term resolved to."""

    node_id: str
    score: float
    match: str


@dataclass(frozen=True)
class GraphRecallSeed:
    """One requested seed term and the nodes it resolved to."""

    term: str
    matches: Tuple[GraphRecallSeedMatch, ...] = tuple()


@dataclass(frozen=True)
class GraphAssociation:
    """A node reached by activation spreading, with the evidence that reached it."""

    node_id: str
    score: float
    novelty: float
    distance: int
    source_count: int
    bridge: bool = False
    labels: Tuple[str, ...] = tuple()
    references: Tuple[GraphReferenceSentence, ...] = tuple()
    sources: Tuple[GraphRecallSource, ...] = tuple()
    via: Tuple[GraphRecallEdge, ...] = tuple()


@dataclass(frozen=True)
class GraphRecallResult:
    """Full ``GRAPH_RECALL`` answer: header counters plus the decoded payload."""

    seeds: int
    resolved: int
    visited: int
    expanded: int
    hydrated: int
    reference_count: int
    count: int
    bridges: int
    truncated: bool
    precision: float
    seed_resolutions: Tuple[GraphRecallSeed, ...] = tuple()
    unresolved: Tuple[str, ...] = tuple()
    associations: Tuple[GraphAssociation, ...] = tuple()


@dataclass(frozen=True)
class GraphSimilarMatch:
    """A node that behaves like the queried node (shared neighbours or words)."""

    node_id: str
    score: float
    context: float = 0.0
    lexical: float = 0.0
    shared_count: int = 0
    shared: Tuple[str, ...] = tuple()
    labels: Tuple[str, ...] = tuple()


@dataclass(frozen=True)
class GraphSimilarResult:
    """Full ``GRAPH_SIMILAR`` answer."""

    node_id: str
    count: int
    truncated: bool
    matches: Tuple[GraphSimilarMatch, ...] = tuple()


@dataclass(frozen=True)
class GraphNodeRecord:
    """A node record read back through ``GRAPH_NODE_GET``."""

    node_id: str
    labels: Tuple[str, ...] = tuple()
    props: dict[str, object] = field(default_factory=dict)
    references: Tuple[GraphReferenceSentence, ...] = tuple()


@dataclass(frozen=True)
class GraphEdgeBatchResult:
    """Counters reported by ``GRAPH_EDGE_SET_BATCH``."""

    requested: int
    applied: int
    created: int
    updated: int
    failed: int


@dataclass(frozen=True)
class GraphTermIndexStats:
    """Answer of ``GRAPH_TERM_INDEX action=stats|rebuild|drop``."""

    action: str
    enabled: bool = False
    entries: int = 0
    nodes: int = 0
    terms: int = 0
    removed: int = 0
    next_cursor: str = ""


@dataclass(frozen=True)
class NamespaceSummary:
    """Aggregate statistics for a namespace prefix."""

    prefix: bytes
    terminal_count: int
    total_payload_bytes: int
    min_payload_bytes: int
    max_payload_bytes: int
    min_key: int | None
    max_key: int | None
    max_depth: int
    self_terminal: bool
    branches: Tuple[tuple[bytes, int], ...]


@dataclass(frozen=True)
class CheetahSystemStats:
    """Snapshot of the cheetah-db resource monitor."""

    logical_cores: int
    gomaxprocs: int
    goroutines: int
    mem_alloc_bytes: int
    mem_sys_bytes: int
    process_cpu_pct: float | None
    system_cpu_pct: float | None
    process_cpu_supported: bool
    system_cpu_supported: bool
    io_supported: bool
    io_read_bytes_per_sec: float | None
    io_write_bytes_per_sec: float | None
    timestamp: str | None
    recommended_workers: Tuple[tuple[int, int], ...]
    payload_cache_enabled: bool
    payload_cache_entries: int
    payload_cache_max_entries: int
    payload_cache_bytes: int
    payload_cache_max_bytes: int
    payload_cache_hits: int
    payload_cache_misses: int
    payload_cache_evictions: int
    payload_cache_hit_pct: float | None
    payload_cache_advisory_bypass_bytes: int | None

    def derive_reduce_page_limit(
        self,
        *,
        default_limit: int = CHEETAH_DEFAULT_REDUCE_PAGE_SIZE,
        min_limit: int = CHEETAH_PAIR_SCAN_MIN_LIMIT,
        max_limit: int = CHEETAH_PAIR_SCAN_MAX_LIMIT,
        target_pending: int = CHEETAH_PAIR_SCAN_MAX_LIMIT,
    ) -> int | None:
        """Return a batch size tuned to the current worker hints."""
        if not self.recommended_workers:
            return None
        hints = {pending: workers for pending, workers in self.recommended_workers}
        if target_pending not in hints:
            # Fall back to the largest pending bucket advertised by the server.
            candidate = max(hints.keys(), default=0)
            if candidate == 0:
                return None
            target_pending = candidate
        workers = hints.get(target_pending, 0)
        gomax = max(self.gomaxprocs, 1)
        if workers <= 0 or gomax <= 0:
            return None
        ratio = min(1.0, max(0.0, workers / float(gomax)))
        adaptive_max = min(max_limit, max(default_limit, gomax * 256))
        bounded_min = max(min_limit, default_limit // 4)
        window = adaptive_max - bounded_min
        limit = bounded_min + int(window * ratio)
        if limit < bounded_min:
            limit = bounded_min
        if limit > adaptive_max:
            limit = adaptive_max
        # Align to payload pages to avoid thrashing on odd sizes.
        if limit > 0:
            limit = max(bounded_min, (limit // 64) * 64)
        return limit


__all__ = [
    "RawContextProjection",
    "RawCountsProjection",
    "RawProbabilityProjection",
    "RawContinuationProjection",
    "PredictionValueResult",
    "PredictionQueryResult",
    "GraphReferenceSentence",
    "GraphRecallEdge",
    "GraphRecallSource",
    "GraphRecallSeedMatch",
    "GraphRecallSeed",
    "GraphAssociation",
    "GraphRecallResult",
    "GraphSimilarMatch",
    "GraphSimilarResult",
    "GraphNodeRecord",
    "GraphEdgeBatchResult",
    "GraphTermIndexStats",
    "NamespaceSummary",
    "CheetahSystemStats",
    "CHEETAH_DEFAULT_REDUCE_PAGE_SIZE",
    "CHEETAH_PAIR_SCAN_MIN_LIMIT",
    "CHEETAH_PAIR_SCAN_MAX_LIMIT",
    "GRAPH_RECALL_MAX_SEEDS",
    "GRAPH_RECALL_MAX_HOPS",
    "GRAPH_RECALL_MAX_BRANCH",
    "GRAPH_RECALL_MAX_BUDGET",
    "GRAPH_RECALL_MAX_REFERENCES",
    "GRAPH_NODE_MAX_REFERENCES",
    "GRAPH_NODE_MAX_REFERENCE_CHARS",
    "GRAPH_NODE_MAX_REFERENCE_TOTAL_CHARS",
]
