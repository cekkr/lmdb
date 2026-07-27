"""Graph context memory: DB-SLM semantics on top of the Cheetah graph store.

Cheetah owns the graph itself (`GRAPH_NODE_SET`, `GRAPH_EDGE_SET_BATCH`,
`GRAPH_RECALL`, `GRAPH_SIMILAR`, `GRAPH_TERM_INDEX`) and every bound that keeps a
walk finite. This module owns the DB-SLM conventions layered on top of it: which
ids the trainer mints, which relations a prompt/response pair produces, which
seeds a turn recalls with, and how a recall answer becomes decoder context.

Two invariants make it safe to switch on:

* the graph is a **side channel**, exactly like the dependency layers. Nothing it
  produces enters `CorpusChunk.train_text`, and its recalled sentences stay
  internal context for biasing/embedding — they are never appended to a visible
  response.
* every write and read is **bounded**. Per-record term/edge caps, a per-run node
  budget, and the recall limit/branch/budget arguments keep both the command
  stream and the returned payload finite even on a hub-shaped graph.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

from .cheetah_types import (
    GRAPH_NODE_MAX_REFERENCE_CHARS,
    GRAPH_NODE_MAX_REFERENCES,
    GRAPH_RECALL_MAX_SEEDS,
    GraphAssociation,
    GraphRecallResult,
)

CONTEXT_NODE_PREFIX = "ctx"
TERM_NODE_PREFIX = "term"
CONTEXT_NODE_LABEL = "dbslm_context"
TERM_NODE_LABEL = "dbslm_term"

EDGE_EVOKES = "evokes"
EDGE_PRECEDES = "precedes"
EDGE_DEPENDS_PREFIX = "dep_"

# Generic English function words. This is deliberately *not* a dataset word list:
# it only removes closed-class tokens that carry no association value as graph
# seeds. Whenever a dependency layer is available its lemmas are preferred, and
# this list is a fallback for raw inference prompts.
_FUNCTION_WORDS = frozenset(
    """
    a about above after again against all am an and any are aren as at be because been
    before being below between both but by can cannot could couldn did didn do does
    doesn doing don down during each few for from further had hadn has hasn have haven
    having he her here hers herself him himself his how i if in into is isn it its
    itself just ll me mightn more most must mustn my myself needn no nor not now of off
    on once only or other others our ours ourselves out over own re s same shan she
    should shouldn so some such t than that the their theirs them themselves then there
    these they this those through to too under until up ve very was wasn we were weren
    what when where which while who whom why will with won would wouldn y you your
    yours yourself yourselves
    """.split()
)

_TERM_PATTERN = re.compile(r"[A-Za-z][A-Za-z0-9'\-]*")
_STRUCTURAL_TAG_PATTERN = re.compile(r"\|[^|]*\|")
_SLUG_STRIP_PATTERN = re.compile(r"[^a-z0-9]+")


def slugify(value: str) -> str:
    """Lowercase, accent-free, underscore-joined id fragment.

    Ids are the contract across turns and runs: the same surface form must always
    produce the same id, and the result must be a single protocol token because
    Cheetah splits `GRAPH_*` arguments on whitespace.
    """
    text = unicodedata.normalize("NFKD", str(value or ""))
    text = "".join(char for char in text if not unicodedata.combining(char))
    slug = _SLUG_STRIP_PATTERN.sub("_", text.lower()).strip("_")
    return slug


def context_node_id(field_name: str, value: str) -> str:
    """`ctx:<field>:<value>` — a dataset context value such as `ctx:emotion:joy`."""
    field_slug = slugify(field_name)
    value_slug = slugify(value)
    if not field_slug or not value_slug:
        return ""
    return f"{CONTEXT_NODE_PREFIX}:{field_slug}:{value_slug}"


def term_node_id(term: str) -> str:
    """`term:<lemma>` — one content word of the corpus."""
    term_slug = slugify(term)
    if not term_slug:
        return ""
    return f"{TERM_NODE_PREFIX}:{term_slug}"


def term_text_from_node_id(node_id: str) -> str:
    """Return the surface term behind a `term:` node id, or "" for other nodes."""
    if not node_id.startswith(f"{TERM_NODE_PREFIX}:"):
        return ""
    return node_id.split(":", 1)[1].replace("_", " ").strip()


def strip_structural_tags(text: str) -> str:
    """Remove `|TAG|` markers so they can never become graph ids or seeds."""
    return _STRUCTURAL_TAG_PATTERN.sub(" ", text or "")


def content_terms(text: str, *, limit: int = 0) -> list[str]:
    """Extract deduplicated content words from raw text, in order of appearance."""
    cleaned = strip_structural_tags(text)
    terms: list[str] = []
    seen: set[str] = set()
    for match in _TERM_PATTERN.finditer(cleaned):
        word = match.group(0).lower().strip("'-")
        if len(word) < 3 or word in _FUNCTION_WORDS or word in seen:
            continue
        seen.add(word)
        terms.append(word)
        if limit and len(terms) >= limit:
            break
    return terms


def _truncate_to_sentence(text: str, limit: int) -> str:
    """Keep whole sentences within `limit`, or "" when none fits.

    Mirrors `graphBoundEpisodeReference` in cheetah-db's `graph_recall.go`: it
    refuses to store half a sentence rather than storing a fragment.
    """
    if len(text) <= limit:
        return text
    prefix = text[:limit]
    cut = max(prefix.rfind("."), prefix.rfind("!"), prefix.rfind("?"))
    if cut < limit // 2:
        return ""
    return prefix[: cut + 1].strip()


def dependency_terms(layer: Any, *, limit: int = 0) -> list[str]:
    """Prefer dependency lemmas over surface words when a layer is available."""
    if layer is None:
        return []
    groups = getattr(layer, "strong_token_groups", None) or {}
    terms: list[str] = []
    seen: set[str] = set()
    # Iterate buckets in a stable order so the same record always mints the same
    # ids regardless of dict ordering in the parser backend.
    for bucket in sorted(groups):
        for lemma in groups[bucket]:
            word = str(lemma or "").strip().lower()
            if len(word) < 3 or word in _FUNCTION_WORDS or word in seen:
                continue
            seen.add(word)
            terms.append(word)
            if limit and len(terms) >= limit:
                return terms
    return terms


@dataclass
class GraphIngestStats:
    """Bounded, reportable outcome of one graph observation pass."""

    records: int = 0
    nodes_written: int = 0
    edges_requested: int = 0
    edges_applied: int = 0
    edges_created: int = 0
    edges_updated: int = 0
    edges_failed: int = 0
    references_attached: int = 0
    skipped_records: int = 0

    def merge(self, other: "GraphIngestStats") -> None:
        self.records += other.records
        self.nodes_written += other.nodes_written
        self.edges_requested += other.edges_requested
        self.edges_applied += other.edges_applied
        self.edges_created += other.edges_created
        self.edges_updated += other.edges_updated
        self.edges_failed += other.edges_failed
        self.references_attached += other.references_attached
        self.skipped_records += other.skipped_records

    def describe(self) -> str:
        return (
            f"records={self.records}, nodes={self.nodes_written}, "
            f"edges={self.edges_applied}/{self.edges_requested} "
            f"(created={self.edges_created}, updated={self.edges_updated}, "
            f"failed={self.edges_failed}), references={self.references_attached}, "
            f"skipped={self.skipped_records}"
        )


@dataclass(frozen=True)
class GraphContextSignal:
    """Internal decoder context recalled from the graph for a single turn.

    `context_text` is bias/embedding input only. Like the Level 3 `ContextSummary`
    it MUST NOT be prepended to a generated response.
    """

    seeds: tuple[str, ...] = tuple()
    associations: tuple[GraphAssociation, ...] = tuple()
    term_weights: Mapping[str, float] = field(default_factory=dict)
    context_text: str = ""
    truncated: bool = False
    unresolved: tuple[str, ...] = tuple()

    def __bool__(self) -> bool:
        return bool(self.associations or self.term_weights or self.context_text)

    def describe(self) -> str:
        return (
            f"seeds={len(self.seeds)}, associations={len(self.associations)}, "
            f"terms={len(self.term_weights)}, truncated={int(self.truncated)}"
        )


class GraphContextMemory:
    """Writes DB-SLM corpus structure into the graph and recalls it at decode time."""

    def __init__(
        self,
        hot_path: Any,
        *,
        enabled: bool = True,
        recall_hops: int = 2,
        recall_precision: float = 0.2,
        recall_limit: int = 8,
        recall_min_sources: int = 1,
        recall_branch_limit: int = 64,
        recall_budget: int = 2048,
        recall_references: bool = True,
        reference_limit: int = 8,
        reference_chars: int = 480,
        max_seeds: int = 8,
        max_terms_per_side: int = 6,
        max_dependency_arcs: int = 12,
        max_nodes_per_run: int = 20000,
    ) -> None:
        self.hot_path = hot_path
        self._requested = bool(enabled)
        self.recall_hops = max(1, int(recall_hops))
        self.recall_precision = min(1.0, max(0.0, float(recall_precision)))
        self.recall_limit = max(1, int(recall_limit))
        self.recall_min_sources = max(1, int(recall_min_sources))
        self.recall_branch_limit = max(1, int(recall_branch_limit))
        self.recall_budget = max(1, int(recall_budget))
        self.recall_references = bool(recall_references)
        self.reference_limit = max(1, int(reference_limit))
        self.reference_chars = max(32, min(int(reference_chars), GRAPH_NODE_MAX_REFERENCE_CHARS))
        self.max_seeds = max(1, min(int(max_seeds), GRAPH_RECALL_MAX_SEEDS))
        self.max_terms_per_side = max(1, int(max_terms_per_side))
        self.max_dependency_arcs = max(0, int(max_dependency_arcs))
        self.max_nodes_per_run = max(1, int(max_nodes_per_run))
        # Node reference lists are replaced, not merged, by GRAPH_NODE_SET. The
        # first touch of a node in this process reads the stored list back so a
        # second training run extends the provenance instead of erasing it.
        self._node_references: dict[str, list[dict[str, object]]] = {}
        self._node_labels: dict[str, tuple[str, ...]] = {}
        self._reference_ids: dict[str, set[str]] = {}

    # ------------------------------------------------------------------ #
    # Availability
    # ------------------------------------------------------------------ #
    def available(self) -> bool:
        """True when graph memory is requested and the adapter can serve it."""
        if not self._requested or self.hot_path is None:
            return False
        recall = getattr(self.hot_path, "graph_recall", None)
        node_set = getattr(self.hot_path, "graph_node_set", None)
        return callable(recall) and callable(node_set)

    def disable(self) -> None:
        self._requested = False

    # ------------------------------------------------------------------ #
    # Training (write path)
    # ------------------------------------------------------------------ #
    def observe_records(
        self,
        records: Sequence[Any] | None,
        *,
        source_label: str = "",
        max_records: int = 0,
    ) -> GraphIngestStats:
        """Write one graph batch per record: context/term nodes plus their relations.

        `records` are duck-typed evaluation records (`prompt`, `response`,
        `context_tokens`, `prompt_dependencies`, `response_dependencies`); the
        module deliberately does not import the evaluation layer.
        """
        stats = GraphIngestStats()
        if not records or not self.available():
            return stats
        limit = max_records if max_records and max_records > 0 else len(records)
        for record in list(records)[:limit]:
            record_stats = self._observe_record(record, source_label=source_label)
            stats.merge(record_stats)
        return stats

    def _observe_record(self, record: Any, *, source_label: str) -> GraphIngestStats:
        stats = GraphIngestStats()
        prompt = strip_structural_tags(getattr(record, "prompt", "") or "").strip()
        response = (getattr(record, "response", "") or "").strip()
        if not response:
            stats.skipped_records = 1
            return stats
        stats.records = 1

        context_nodes = self._context_nodes(getattr(record, "context_tokens", None) or {})
        prompt_terms = self._record_terms(
            prompt, getattr(record, "prompt_dependencies", None)
        )
        response_terms = self._record_terms(
            response, getattr(record, "response_dependencies", None)
        )
        if not context_nodes and not (prompt_terms and response_terms):
            stats.skipped_records = 1
            stats.records = 0
            return stats

        reference = self._reference_payload(response, source_label=source_label)
        # Provenance rides on the nodes a turn is actually *about*: its context
        # values, and otherwise its strongest response terms.
        reference_targets = list(context_nodes) or [
            term_node_id(term) for term in response_terms[:2]
        ]

        touched: dict[str, tuple[str, ...]] = {}
        for node_id in context_nodes:
            touched[node_id] = (CONTEXT_NODE_LABEL,)
        for term in prompt_terms + response_terms:
            node_id = term_node_id(term)
            if node_id:
                touched.setdefault(node_id, (TERM_NODE_LABEL,))

        for node_id, labels in touched.items():
            attach = reference if node_id in reference_targets else None
            written, attached = self._write_node(node_id, labels, attach)
            stats.nodes_written += int(written)
            stats.references_attached += int(attached)

        items = self._edge_items(
            context_nodes,
            prompt_terms,
            response_terms,
            getattr(record, "response_dependencies", None),
            source_label=source_label,
        )
        if items:
            stats.edges_requested = len(items)
            batch = self.hot_path.graph_edge_set_batch(items, continue_on_error=True)
            if batch is not None:
                stats.edges_applied = batch.applied
                stats.edges_created = batch.created
                stats.edges_updated = batch.updated
                stats.edges_failed = batch.failed
        return stats

    def _context_nodes(self, context_tokens: Mapping[str, str]) -> list[str]:
        nodes: list[str] = []
        for key, value in list(context_tokens.items()):
            node_id = context_node_id(key, value)
            if node_id and node_id not in nodes:
                nodes.append(node_id)
        return nodes

    def _record_terms(self, text: str, layer: Any) -> list[str]:
        terms = dependency_terms(layer, limit=self.max_terms_per_side)
        if len(terms) < self.max_terms_per_side:
            seen = set(terms)
            for word in content_terms(text, limit=self.max_terms_per_side * 3):
                if word in seen:
                    continue
                seen.add(word)
                terms.append(word)
                if len(terms) >= self.max_terms_per_side:
                    break
        return terms[: self.max_terms_per_side]

    def _reference_payload(self, response: str, *, source_label: str) -> dict[str, object] | None:
        """Bound a response to complete sentences, mirroring the server's rule.

        A node reference is readable provenance, so a mid-word fragment is worse
        than nothing: over budget, keep whole sentences and drop the reference
        entirely when not even the first sentence fits.
        """
        text = " ".join(response.split())
        if not text:
            return None
        if len(text) > self.reference_chars:
            text = _truncate_to_sentence(text, self.reference_chars)
            if not text:
                return None
        payload: dict[str, object] = {"text": text}
        if source_label:
            payload["source"] = source_label
        return payload

    def _write_node(
        self,
        node_id: str,
        labels: Sequence[str],
        reference: dict[str, object] | None,
    ) -> tuple[bool, bool]:
        """Upsert a node, merging references locally against the stored list."""
        if not node_id:
            return False, False
        known = node_id in self._node_references
        if not known:
            if len(self._node_references) >= self.max_nodes_per_run:
                # Budget exhausted: keep writing relations, stop growing the
                # per-process reference mirror.
                return False, False
            stored = self._load_stored_references(node_id)
            self._node_references[node_id] = stored
            self._reference_ids[node_id] = {
                str(entry.get("text", "")) for entry in stored
            }
            self._node_labels[node_id] = tuple(labels)
        references = self._node_references[node_id]
        attached = False
        if reference is not None:
            text = str(reference.get("text", ""))
            if text and text not in self._reference_ids[node_id]:
                if len(references) < GRAPH_NODE_MAX_REFERENCES:
                    references.append(reference)
                    self._reference_ids[node_id].add(text)
                    attached = True
        if known and not attached:
            # The node is already registered and this record adds nothing new to
            # it. Rewriting the identical list would only cost a round trip.
            return False, False
        written = self.hot_path.graph_node_set(
            node_id,
            labels=labels,
            references=references or None,
        )
        return bool(written), attached and bool(written)

    def _load_stored_references(self, node_id: str) -> list[dict[str, object]]:
        getter = getattr(self.hot_path, "graph_node_get", None)
        if not callable(getter):
            return []
        record = getter(node_id)
        if record is None:
            return []
        return [reference.as_payload() for reference in record.references]

    def _edge_items(
        self,
        context_nodes: Sequence[str],
        prompt_terms: Sequence[str],
        response_terms: Sequence[str],
        response_layer: Any,
        *,
        source_label: str,
    ) -> list[dict[str, object]]:
        props: dict[str, object] = {}
        if source_label:
            props["dataset"] = source_label
        items: list[dict[str, object]] = []
        seen: set[tuple[str, str, str]] = set()

        def add(from_id: str, to_id: str, edge_type: str, weight: float) -> None:
            if not from_id or not to_id or from_id == to_id:
                return
            key = (from_id, edge_type, to_id)
            if key in seen:
                return
            seen.add(key)
            item: dict[str, object] = {
                "from": from_id,
                "to": to_id,
                "type": edge_type,
                "weight": round(max(0.05, min(1.0, weight)), 4),
            }
            if props:
                item["props"] = dict(props)
            items.append(item)

        response_nodes = [term_node_id(term) for term in response_terms]
        for context_node in context_nodes:
            for position, node_id in enumerate(response_nodes):
                add(context_node, node_id, EDGE_EVOKES, 1.0 - 0.1 * position)
        for prompt_position, prompt_term in enumerate(prompt_terms):
            from_id = term_node_id(prompt_term)
            for response_position, to_id in enumerate(response_nodes):
                add(
                    from_id,
                    to_id,
                    EDGE_PRECEDES,
                    0.9 - 0.05 * (prompt_position + response_position),
                )
        for arc in self._dependency_arcs(response_layer):
            head, child, label = arc
            add(
                term_node_id(head),
                term_node_id(child),
                f"{EDGE_DEPENDS_PREFIX}{slugify(label) or 'dep'}",
                0.8,
            )
        return items

    def _dependency_arcs(self, layer: Any) -> list[tuple[str, str, str]]:
        if layer is None or self.max_dependency_arcs <= 0:
            return []
        arcs: list[tuple[str, str, str]] = []
        for arc in getattr(layer, "arcs", ()) or ():
            head = str(getattr(arc, "head", "") or "").strip().lower()
            child = str(getattr(arc, "lemma", "") or getattr(arc, "token", "") or "").strip().lower()
            label = str(getattr(arc, "dep", "") or "").strip()
            if not head or not child or head == "root":
                continue
            if head in _FUNCTION_WORDS or child in _FUNCTION_WORDS:
                continue
            if len(head) < 3 or len(child) < 3:
                continue
            arcs.append((head, child, label))
            if len(arcs) >= self.max_dependency_arcs:
                break
        return arcs

    # ------------------------------------------------------------------ #
    # Inference (read path)
    # ------------------------------------------------------------------ #
    def build_seeds(
        self,
        text: str,
        *,
        context_tokens: Mapping[str, str] | None = None,
        extra_seeds: Iterable[str] | None = None,
    ) -> list[str]:
        """Compose recall seeds: declared context values first, then content words."""
        seeds: list[str] = []
        seen: set[str] = set()

        def push(candidate: str) -> None:
            value = (candidate or "").strip()
            if not value or value in seen:
                return
            seen.add(value)
            seeds.append(value)

        for node_id in self._context_nodes(context_tokens or {}):
            push(node_id)
        for seed in extra_seeds or ():
            push(str(seed))
        for word in content_terms(text, limit=self.max_seeds * 2):
            if len(seeds) >= self.max_seeds:
                break
            push(word)
        return seeds[: self.max_seeds]

    def recall(
        self,
        text: str,
        *,
        context_tokens: Mapping[str, str] | None = None,
        extra_seeds: Iterable[str] | None = None,
        min_sources: int | None = None,
    ) -> GraphContextSignal | None:
        """Recall the neighbourhood a turn touches, as internal decoder context."""
        if not self.available():
            return None
        seeds = self.build_seeds(text, context_tokens=context_tokens, extra_seeds=extra_seeds)
        if not seeds:
            return None
        result = self.hot_path.graph_recall(
            seeds,
            precision=self.recall_precision,
            hops=self.recall_hops,
            min_sources=min_sources if min_sources is not None else self.recall_min_sources,
            direction="both",
            references=self.recall_references,
            reference_limit=self.reference_limit if self.recall_references else None,
            # A seed node is excluded from the answer by default, so the sentences
            # recorded for the very context of this turn would never come back.
            # Ask for them whenever references are being hydrated; their terms are
            # then dropped from the bias below, because the prompt is already in
            # the session cache and re-biasing it only echoes the input.
            include_seeds=self.recall_references,
            limit=self.recall_limit,
            branch_limit=self.recall_branch_limit,
            budget=self.recall_budget,
        )
        if result is None:
            return None
        return self.signal_from_result(result, seeds=seeds)

    def signal_from_result(
        self,
        result: GraphRecallResult,
        *,
        seeds: Sequence[str],
    ) -> GraphContextSignal:
        """Project a recall answer onto decoder inputs (terms + internal context).

        Seed nodes contribute their reference sentences but not their terms: they
        are the prompt itself, already represented in the Level 2 session cache.
        """
        resolved_seed_ids = {
            match.node_id
            for resolution in result.seed_resolutions
            for match in resolution.matches
        }
        term_weights: dict[str, float] = {}
        sentences: list[str] = []
        seen_sentences: set[str] = set()
        for association in result.associations:
            term = term_text_from_node_id(association.node_id)
            if term and association.node_id not in resolved_seed_ids:
                weight = max(0.0, min(1.0, float(association.score)))
                if weight > 0.0:
                    term_weights[term] = max(term_weights.get(term, 0.0), weight)
            for reference in association.references:
                text = " ".join(reference.text.split())
                if not text or text in seen_sentences:
                    continue
                seen_sentences.add(text)
                sentences.append(text)
                if len(sentences) >= self.reference_limit:
                    break
            if len(sentences) >= self.reference_limit:
                break
        return GraphContextSignal(
            seeds=tuple(seeds),
            associations=result.associations,
            term_weights=term_weights,
            context_text=" ".join(sentences).strip(),
            truncated=result.truncated,
            unresolved=result.unresolved,
        )


__all__ = [
    "CONTEXT_NODE_LABEL",
    "CONTEXT_NODE_PREFIX",
    "EDGE_DEPENDS_PREFIX",
    "EDGE_EVOKES",
    "EDGE_PRECEDES",
    "TERM_NODE_LABEL",
    "TERM_NODE_PREFIX",
    "GraphContextMemory",
    "GraphContextSignal",
    "GraphIngestStats",
    "content_terms",
    "context_node_id",
    "dependency_terms",
    "slugify",
    "strip_structural_tags",
    "term_node_id",
    "term_text_from_node_id",
]
