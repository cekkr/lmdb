from __future__ import annotations

import datetime as dt
import hashlib
import json
import math
import os
import random
import re
import textwrap
import time
import uuid
from collections import Counter, defaultdict, deque
from dataclasses import dataclass, field, replace
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Any, Sequence

from .decoder import DecoderConfig
from .inference_shared import issue_prompt
from .metrics import lexical_overlap, rouge_l_score
from .pipeline import DBSLMEngine
from .quality import SentenceQualityScorer
from .text_markers import extract_complete_sentence
from helpers.char_tree_similarity import similarity_score
from helpers.resource_monitor import ResourceMonitor

if TYPE_CHECKING:
    from helpers.resource_monitor import ResourceDelta

from log_helpers import log


@dataclass(frozen=True)
class DependencyArc:
    token: str
    lemma: str
    head: str
    dep: str
    pos: str


@dataclass(frozen=True)
class DependencyLayer:
    backend: str
    arcs: tuple[DependencyArc, ...]
    strong_token_groups: dict[str, tuple[str, ...]]
    token_count: int


@dataclass(frozen=True)
class EvaluationRecord:
    prompt: str
    response: str
    context_tokens: dict[str, str] = field(default_factory=dict)
    prompt_dependencies: DependencyLayer | None = None
    response_dependencies: DependencyLayer | None = None


@dataclass(frozen=True)
class EvaluationSampleResult:
    index: int
    label: str
    prompt: str
    reference: str
    generated: str
    context_tokens: dict[str, str]
    metrics: dict[str, float | int | None]
    flagged: bool = False
    variant: int = 1


_STRONG_DEP_GROUPS = {
    "subjects": {"nsubj", "nsubjpass", "csubj", "csubjpass", "expl"},
    "objects": {"dobj", "obj", "pobj", "iobj", "dative", "attr"},
    "actions": {"ROOT", "ccomp", "xcomp", "advcl", "acl", "relcl"},
    "modifiers": {"amod", "advmod", "acomp", "appos"},
    "quantifiers": {"nummod", "quantmod"},
    "auxiliaries": {"aux", "auxpass", "cop"},
}
_DEPENDENCY_PIPELINE_BACKEND: str | None = None
_DEPENDENCY_PIPELINE: Any | None = None
_DEPENDENCY_PIPELINE_ATTEMPTED = False
_DEPENDENCY_DISABLED_NOTICE_EMITTED = False
_DEPENDENCY_FAILURE_NOTICE_EMITTED = False


def _get_dependency_pipeline() -> tuple[str | None, Any | None]:
    global _DEPENDENCY_PIPELINE_ATTEMPTED, _DEPENDENCY_PIPELINE_BACKEND, _DEPENDENCY_PIPELINE
    if _DEPENDENCY_PIPELINE is not None or _DEPENDENCY_PIPELINE_ATTEMPTED:
        return _DEPENDENCY_PIPELINE_BACKEND, _DEPENDENCY_PIPELINE
    backend, pipeline = _load_dependency_pipeline()
    _DEPENDENCY_PIPELINE_ATTEMPTED = True
    _DEPENDENCY_PIPELINE_BACKEND = backend
    _DEPENDENCY_PIPELINE = pipeline
    if backend and pipeline:
        log(f"[deps] Enabled dependency parsing backend: {backend}.")
    else:
        _emit_dependency_disabled_notice()
    return backend, pipeline


def _load_dependency_pipeline() -> tuple[str | None, Any | None]:
    loaders = (_load_spacy_pipeline, _load_stanza_pipeline)
    for loader in loaders:
        backend, pipeline = loader()
        if backend and pipeline:
            return backend, pipeline
    return None, None


def _load_spacy_pipeline() -> tuple[str | None, Any | None]:
    model_name = os.environ.get("DBSLM_SPACY_MODEL", "en_core_web_sm")
    try:
        import spacy  # type: ignore
    except Exception as exc:  # pragma: no cover - optional dependency
        log(f"[deps] spaCy unavailable: {exc}")
        return None, None
    try:
        pipeline = spacy.load(model_name, disable=["ner", "textcat"])
        return "spacy", pipeline
    except Exception as exc:  # pragma: no cover - dynamic model loading
        log(f"[deps] spaCy model '{model_name}' could not be loaded: {exc}")
        return None, None


def _load_stanza_pipeline() -> tuple[str | None, Any | None]:
    try:
        import stanza  # type: ignore
    except Exception as exc:  # pragma: no cover - optional dependency
        log(f"[deps] Stanza unavailable: {exc}")
        return None, None
    lang = os.environ.get("DBSLM_DEP_LANG", "en")
    processors = os.environ.get("DBSLM_STANZA_PROCESSORS", "tokenize,pos,lemma,depparse")
    try:
        pipeline = stanza.Pipeline(
            lang=lang,
            processors=processors,
            tokenize_no_ssplit=True,
            use_gpu=False,
            verbose=False,
        )
        return "stanza", pipeline
    except Exception as exc:  # pragma: no cover - download/runtime errors
        log(f"[deps] Stanza pipeline error ({lang}): {exc}")
        return None, None


def _emit_dependency_disabled_notice() -> None:
    global _DEPENDENCY_DISABLED_NOTICE_EMITTED
    if _DEPENDENCY_DISABLED_NOTICE_EMITTED:
        return
    _DEPENDENCY_DISABLED_NOTICE_EMITTED = True
    log(
        "[deps] Dependency parsing layer disabled. Install spaCy with an English model "
        "or Stanza to enable strong token grouping."
    )


@lru_cache(maxsize=512)
def build_dependency_layer(text: str) -> DependencyLayer | None:
    normalized = " ".join(text.split())
    if not normalized:
        return None
    backend, pipeline = _get_dependency_pipeline()
    if not backend or pipeline is None:
        return None
    try:
        if backend == "spacy":
            return _dependency_layer_from_spacy(pipeline, normalized)
        if backend == "stanza":
            return _dependency_layer_from_stanza(pipeline, normalized)
    except Exception as exc:
        _emit_dependency_failure_notice(exc, backend)
    return None


def _dependency_layer_from_spacy(pipeline: Any, text: str) -> DependencyLayer:
    doc = pipeline(text)
    arcs: list[DependencyArc] = []
    grouped: defaultdict[str, set[str]] = defaultdict(set)
    for token in doc:
        if getattr(token, "is_space", False):
            continue
        head_text = token.head.text if token.head is not token else "ROOT"
        arcs.append(
            DependencyArc(
                token=token.text,
                lemma=token.lemma_,
                head=head_text,
                dep=token.dep_,
                pos=token.pos_,
            )
        )
        _categorize_token(grouped, token.dep_, token.pos_, token.lemma_)
    return DependencyLayer(
        backend="spacy",
        arcs=tuple(arcs),
        strong_token_groups=_finalize_groups(grouped),
        token_count=len(arcs),
    )


def _dependency_layer_from_stanza(pipeline: Any, text: str) -> DependencyLayer:
    doc = pipeline(text)
    arcs: list[DependencyArc] = []
    grouped: defaultdict[str, set[str]] = defaultdict(set)
    for sentence in getattr(doc, "sentences", []):
        words = getattr(sentence, "words", [])
        for word in words:
            head = "ROOT"
            head_index = getattr(word, "head", 0)
            if isinstance(head_index, int) and 0 < head_index <= len(words):
                head = words[head_index - 1].text
            lemma = word.lemma if getattr(word, "lemma", None) else word.text
            dep = word.deprel or ""
            pos = word.upos or ""
            arcs.append(
                DependencyArc(
                    token=word.text,
                    lemma=lemma,
                    head=head,
                    dep=dep,
                    pos=pos,
                )
            )
            _categorize_token(grouped, dep, pos, lemma)
    return DependencyLayer(
        backend="stanza",
        arcs=tuple(arcs),
        strong_token_groups=_finalize_groups(grouped),
        token_count=len(arcs),
    )


def _categorize_token(
    grouped: defaultdict[str, set[str]],
    dep_label: str | None,
    pos_tag: str | None,
    lemma: str | None,
) -> None:
    lemma_norm = (lemma or "").strip().lower()
    if not lemma_norm:
        return
    dep_lower = (dep_label or "").lower()
    for bucket, labels in _STRONG_DEP_GROUPS.items():
        if dep_lower in labels:
            grouped[bucket].add(lemma_norm)
            return
    pos_upper = (pos_tag or "").upper()
    if pos_upper in {"VERB", "AUX"}:
        grouped["actions"].add(lemma_norm)
    elif pos_upper in {"NOUN", "PROPN", "PRON"}:
        grouped.setdefault("entities", set()).add(lemma_norm)
    elif pos_upper in {"ADV", "ADJ"}:
        grouped.setdefault("modifiers", set()).add(lemma_norm)


def _finalize_groups(grouped: defaultdict[str, set[str]]) -> dict[str, tuple[str, ...]]:
    finalized: dict[str, tuple[str, ...]] = {}
    for bucket, tokens in grouped.items():
        if not tokens:
            continue
        finalized[bucket] = tuple(sorted(tokens))
    return finalized


def _flatten_strong_tokens(layer: DependencyLayer | None) -> set[str]:
    if not layer or not layer.strong_token_groups:
        return set()
    tokens: set[str] = set()
    for bucket_tokens in layer.strong_token_groups.values():
        tokens.update(bucket_tokens)
    return tokens


def _dependency_arc_overlap(
    reference_layer: DependencyLayer | None,
    candidate_layer: DependencyLayer | None,
) -> float | None:
    if not reference_layer or not candidate_layer:
        return None
    ref_arcs = {
        (arc.lemma.lower(), arc.dep.lower(), arc.head.lower())
        for arc in reference_layer.arcs
        if arc.lemma and arc.dep and arc.head
    }
    cand_arcs = {
        (arc.lemma.lower(), arc.dep.lower(), arc.head.lower())
        for arc in candidate_layer.arcs
        if arc.lemma and arc.dep and arc.head
    }
    if not ref_arcs:
        return 1.0 if not cand_arcs else 0.0
    intersection = ref_arcs & cand_arcs
    return len(intersection) / max(1, len(ref_arcs))


def _emit_dependency_failure_notice(exc: Exception, backend: str | None) -> None:
    global _DEPENDENCY_FAILURE_NOTICE_EMITTED
    if _DEPENDENCY_FAILURE_NOTICE_EMITTED:
        return
    _DEPENDENCY_FAILURE_NOTICE_EMITTED = True
    label = backend or "unknown"
    log(f"[deps] Dependency parsing failed via {label}: {exc}")


def _dependency_alignment_metrics(
    reference_layer: DependencyLayer | None,
    candidate_text: str,
) -> dict[str, float | None]:
    metrics: dict[str, float | None] = {
        "strong_token_overlap": None,
        "dependency_arc_overlap": None,
    }
    if not candidate_text.strip():
        return metrics
    candidate_layer = build_dependency_layer(candidate_text)
    if not reference_layer or not candidate_layer:
        return metrics
    ref_tokens = _flatten_strong_tokens(reference_layer)
    cand_tokens = _flatten_strong_tokens(candidate_layer)
    if ref_tokens:
        overlap = len(ref_tokens & cand_tokens) / max(1, len(ref_tokens))
        metrics["strong_token_overlap"] = round(overlap, 4)
    else:
        metrics["strong_token_overlap"] = 1.0 if not cand_tokens else 0.0
    arc_overlap = _dependency_arc_overlap(reference_layer, candidate_layer)
    if arc_overlap is not None:
        metrics["dependency_arc_overlap"] = round(arc_overlap, 4)
    return metrics


class VariantSeedPlanner:
    """Derives deterministic-yet-unique seeds for evaluation variants."""

    def __init__(self, base_seed: int | None = None) -> None:
        if base_seed is None:
            base_seed = random.SystemRandom().randrange(1, 2**63 - 1)
        self.base_seed = base_seed
        self._counter = 0
        self._queue_rng = random.Random(base_seed ^ 0x9E3779B97F4A7C15)

    @property
    def queue_rng(self) -> random.Random:
        return self._queue_rng

    def seed_for(self, sample_index: int, variant: int, attempt: int) -> int:
        self._counter += 1
        digest = hash((self.base_seed, sample_index, variant, attempt, self._counter))
        seed = abs(digest) % (2**31 - 1)
        if seed == 0:
            seed = 1
        return seed


_MAX_BATCH_ATTEMPTS = 2
_MAX_FUTURE_BATCH_RETRIES = 3
_CHAR_REPEAT_ALERT = 0.92
_RECENT_GENERATION_LIMIT = 64
_flagged_retry_budget: dict[str, int] = {}


def _boost_decoder_penalties(
    config: DecoderConfig | None,
    *,
    presence_delta: float = 0.05,
    frequency_delta: float = 0.1,
) -> DecoderConfig:
    base = config if config is not None else DecoderConfig()
    presence = min(1.5, base.presence_penalty + presence_delta)
    frequency = min(1.5, base.frequency_penalty + frequency_delta)
    return replace(base, presence_penalty=presence, frequency_penalty=frequency)


def _record_signature(record: EvaluationRecord) -> str:
    hasher = hashlib.sha1()
    hasher.update(record.prompt.encode("utf-8", errors="ignore"))
    hasher.update(b"\x1f")
    hasher.update(record.response.encode("utf-8", errors="ignore"))
    for key in sorted(record.context_tokens):
        hasher.update(b"\x1f")
        hasher.update(key.encode("utf-8", errors="ignore"))
        hasher.update(b"=")
        hasher.update(record.context_tokens[key].encode("utf-8", errors="ignore"))
    return hasher.hexdigest()


def _format_context_tokens(context: dict[str, str]) -> str:
    if not context:
        return "none"
    parts = [f"{key}={value}" for key, value in context.items()]
    return ", ".join(parts)


def _consume_future_retry_budget(record_key: str) -> bool:
    """Return True if the sample should be skipped because the budget is exhausted."""
    remaining = _flagged_retry_budget.get(record_key)
    if remaining is None:
        return False
    if remaining <= 0:
        return True
    _flagged_retry_budget[record_key] = remaining - 1
    return False


def _schedule_future_retries(record_key: str) -> None:
    _flagged_retry_budget.setdefault(record_key, _MAX_FUTURE_BATCH_RETRIES)


def _clear_future_retries(record_key: str) -> None:
    _flagged_retry_budget.pop(record_key, None)


def preview(text: str, width: int = 1000) -> str:
    collapsed = " ".join(text.split())
    if not collapsed:
        return ""
    return textwrap.shorten(collapsed, width=width, placeholder="…\n")


_STRUCTURE_WORD_RE = re.compile(r"[A-Za-z0-9']+")
_STRUCTURE_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")
_STRUCTURE_PUNCTUATION = set(".,;:!?-")


def _structure_stats(text: str) -> dict[str, float]:
    lowered = text.lower()
    tokens = _STRUCTURE_WORD_RE.findall(lowered)
    token_count = len(tokens)
    top_share = 0.0
    unique_bigram_ratio = 1.0
    token_group_share = 0.0
    if token_count:
        counts = Counter(tokens)
        top_share = sum(count for _, count in counts.most_common(3)) / token_count
        if token_count > 1:
            bigrams = list(zip(tokens, tokens[1:]))
            unique_bigram_ratio = len(set(bigrams)) / max(1, len(bigrams))
            if bigrams:
                bigram_counts = Counter(bigrams)
                token_group_share = max(
                    token_group_share,
                    max(bigram_counts.values()) / max(1, len(bigrams)),
                )
        if token_count > 2:
            trigrams = list(zip(tokens, tokens[1:], tokens[2:]))
            if trigrams:
                trigram_counts = Counter(trigrams)
                token_group_share = max(
                    token_group_share,
                    max(trigram_counts.values()) / max(1, len(trigrams)),
                )
    sentences = [segment.strip() for segment in _STRUCTURE_SENTENCE_SPLIT.split(text) if segment.strip()]
    if not sentences and text.strip():
        sentences = [text.strip()]
    openers: list[str] = []
    for sentence in sentences:
        match = _STRUCTURE_WORD_RE.search(sentence)
        if match:
            openers.append(match.group(0).lower())
    opener_diversity = len(set(openers)) / len(openers) if openers else 1.0
    punctuation_chars = [char for char in text if char in _STRUCTURE_PUNCTUATION]
    punctuation_density = len(punctuation_chars) / max(1, token_count or len(text))
    punctuation_target = 0.065
    punctuation_tolerance = 0.04
    punctuation_balance = 1.0
    if punctuation_tolerance > 0:
        delta = abs(punctuation_density - punctuation_target)
        punctuation_balance = max(0.0, 1.0 - (delta / punctuation_tolerance))
    return {
        "token_count": float(token_count),
        "top_share": float(top_share),
        "unique_bigram_ratio": float(unique_bigram_ratio),
        "token_group_share": float(token_group_share),
        "opener_diversity": float(opener_diversity),
        "punctuation_balance": float(punctuation_balance),
    }


def _structure_metrics(reference: str, candidate: str) -> dict[str, float]:
    reference_stats = _structure_stats(reference or "")
    candidate_stats = _structure_stats(candidate or "")
    opener_penalty = 1.0 - candidate_stats["opener_diversity"]
    relative_top_share = max(0.0, candidate_stats["top_share"] - reference_stats["top_share"])
    base_penalty = 0.65 * candidate_stats["top_share"] + 0.35 * (opener_penalty + relative_top_share)
    length_factor = 1.0
    if candidate_stats["token_count"] <= 25.0:
        length_factor = max(0.25, candidate_stats["token_count"] / 25.0)
    common_token_penalty = max(0.0, min(1.0, base_penalty * length_factor))
    structure_variety = (
        0.6 * candidate_stats["unique_bigram_ratio"]
        + 0.25 * candidate_stats["opener_diversity"]
        + 0.15 * candidate_stats["punctuation_balance"]
    )
    structure_variety = max(0.0, min(1.0, structure_variety))
    return {
        "top_token_share": round(candidate_stats["top_share"], 4),
        "unique_bigram_ratio": round(candidate_stats["unique_bigram_ratio"], 4),
        "token_group_share": round(candidate_stats["token_group_share"], 4),
        "sentence_opener_diversity": round(candidate_stats["opener_diversity"], 4),
        "punctuation_balance": round(candidate_stats["punctuation_balance"], 4),
        "common_token_penalty": round(common_token_penalty, 4),
        "structure_variety": round(structure_variety, 4),
    }


class ResponseEvaluator:
    """Aggregates lightweight eval metrics for streaming probes."""

    def __init__(self, engine: DBSLMEngine) -> None:
        self.engine = engine
        embedder_model = getattr(getattr(engine, "settings", None), "embedder_model", "all-MiniLM-L6-v2")
        shared_embedder = getattr(getattr(engine, "segment_embedder", None), "embedder", None)
        self.quality = SentenceQualityScorer(embedder_model=embedder_model, embedder=shared_embedder)
        self._recent_generations: deque[str] = deque(maxlen=_RECENT_GENERATION_LIMIT)

    def evaluate(self, record: EvaluationRecord, candidate: str) -> dict[str, float | int | None]:
        prompt = record.prompt
        reference = record.response
        lexical = lexical_overlap(reference, candidate)
        rouge = rouge_l_score(reference, candidate)
        ppl_generated = self._perplexity(prompt, candidate)
        ppl_reference = self._perplexity(prompt, reference)
        metrics: dict[str, float | int | None] = {
            "lexical": lexical,
            "rougeL": rouge,
            "ppl_generated": ppl_generated,
            "ppl_reference": ppl_reference,
        }
        metrics["lexical_novelty"] = max(0.0, 1.0 - lexical)
        structure_metrics = _structure_metrics(reference, candidate)
        metrics.update(structure_metrics)
        metrics.update(_dependency_alignment_metrics(record.response_dependencies, candidate))
        metrics.update(
            self.quality.combined_quality(
                candidate,
                reference,
                lexical,
                structure_metrics=structure_metrics,
            )
        )
        metrics.update(self._repetition_metrics(candidate))
        self._remember_generation(candidate)
        return metrics

    def _repetition_metrics(self, candidate: str) -> dict[str, float]:
        normalized = candidate.strip()
        if not normalized or not self._recent_generations:
            return {"char_repeat_max": 0.0, "char_repeat_avg": 0.0}
        scores: list[float] = []
        for previous in self._recent_generations:
            try:
                score = similarity_score(normalized, previous, substring_weight=0.35)
            except Exception:
                continue
            scores.append(score)
        if not scores:
            return {"char_repeat_max": 0.0, "char_repeat_avg": 0.0}
        max_score = max(scores)
        avg_score = sum(scores) / len(scores)
        return {
            "char_repeat_max": round(max_score, 4),
            "char_repeat_avg": round(avg_score, 4),
        }

    def _remember_generation(self, candidate: str) -> None:
        normalized = candidate.strip()
        if not normalized:
            return
        self._recent_generations.append(normalized)

    def _perplexity(self, prompt: str, target: str) -> float:
        tokens = self.engine.tokenizer.encode(target, add_special_tokens=False)
        if not tokens:
            return float("inf")
        history = self.engine.tokenizer.encode(prompt, add_special_tokens=False)
        if not history:
            history = [self.engine.vocab.token_id("<BOS>")]
        log_sum = 0.0
        for token_id in tokens:
            log_prob = self.engine.store.token_log_probability(history, token_id)
            log_sum += log_prob
            history.append(token_id)
        avg_log_prob = log_sum / len(tokens)
        try:
            return math.exp(-avg_log_prob)
        except OverflowError:
            return float("inf")


def _mean_metric(
    samples: Sequence[EvaluationSampleResult],
    metric_name: str,
) -> float | None:
    values: list[float] = []
    for sample in samples:
        value = sample.metrics.get(metric_name)
        if value is None:
            continue
        if isinstance(value, (int, float)) and math.isfinite(value):
            values.append(float(value))
    if not values:
        return None
    return sum(values) / len(values)


def summarize_samples(samples: Sequence[EvaluationSampleResult]) -> dict[str, float | None]:
    if not samples:
        return {}
    summary_keys = {
        "lexical_mean": "lexical",
        "lexical_novelty_mean": "lexical_novelty",
        "rougeL_mean": "rougeL",
        "ppl_generated_mean": "ppl_generated",
        "ppl_reference_mean": "ppl_reference",
        "grammar_errors_mean": "grammar_errors",
        "grammar_score_mean": "grammar_score",
        "cola_acceptability_mean": "cola_acceptability",
        "semantic_similarity_mean": "semantic_similarity",
        "semantic_novelty_mean": "semantic_novelty",
        "length_ratio_mean": "length_ratio",
        "quality_score_mean": "quality_score",
        "structure_variety_mean": "structure_variety",
        "common_token_penalty_mean": "common_token_penalty",
        "top_token_share_mean": "top_token_share",
        "token_group_share_mean": "token_group_share",
        "sentence_opener_diversity_mean": "sentence_opener_diversity",
        "punctuation_balance_mean": "punctuation_balance",
        "unique_bigram_ratio_mean": "unique_bigram_ratio",
        "char_repeat_max_mean": "char_repeat_max",
        "char_repeat_avg_mean": "char_repeat_avg",
        "strong_token_overlap_mean": "strong_token_overlap",
        "dependency_arc_overlap_mean": "dependency_arc_overlap",
    }
    return {summary_name: _mean_metric(samples, metric_name) for summary_name, metric_name in summary_keys.items()}


def run_inference_records(
    engine: DBSLMEngine,
    evaluator: ResponseEvaluator,
    records: Sequence[EvaluationRecord],
    *,
    label: str,
    user_id: str = "trainer",
    agent_name: str = "db-slm",
    logger: EvalLogWriter | None = None,
    quality_gate: "QualityGate | None" = None,
    decoder_cfg: DecoderConfig | None = None,
    variants_per_prompt: int = 1,
    seed_planner: VariantSeedPlanner | None = None,
) -> list[EvaluationSampleResult]:
    if not records:
        return []
    resource_monitor = ResourceMonitor()
    try:
        telemetry_before = resource_monitor.snapshot()
    except Exception:
        telemetry_before = None
    eval_start = time.perf_counter()
    variant_runs = max(1, variants_per_prompt)
    total_runs = len(records) * variant_runs
    log(
        f"[eval] Running {total_runs} inference probe(s) from {label} to gauge training quality."
    )
    results: list[EvaluationSampleResult] = []
    pending: list[dict[str, Any]] = []
    queue_rng = seed_planner.queue_rng if seed_planner else random
    for idx, record in enumerate(records, start=1):
        for variant in range(1, variant_runs + 1):
            record_key = f"{_record_signature(record)}@{variant}"
            tag = f"#{idx}.{variant}" if variant_runs > 1 else f"#{idx}"
            if _consume_future_retry_budget(record_key):
                log(
                    f"[eval] {tag}: skipping flagged sample; retry budget exhausted "
                    f"(prompt='{preview(record.prompt, 120)}')."
                )
                continue
            pending.append(
                {
                    "index": idx,
                    "record": record,
                    "record_key": record_key,
                    "attempts": 0,
                    "variant": variant,
                    "decoder_cfg_override": None,
                }
            )
    while pending:
        entry = pending.pop(0)
        record = entry["record"]
        idx = entry["index"]
        variant = entry.get("variant", 1)
        record_key = entry["record_key"]
        entry["attempts"] += 1
        attempts = entry["attempts"]
        tag = f"#{idx}.{variant}" if variant_runs > 1 else f"#{idx}"
        generated = ""
        metrics: dict[str, float | int | None] = {}
        flagged = False
        flag_reasons: list[str] = []
        active_decoder_cfg = entry.get("decoder_cfg_override") or decoder_cfg
        _, generated = issue_prompt(
            engine,
            record.prompt,
            user_id=user_id,
            agent_name=agent_name,
            seed_history=False,
            min_response_words=0,
            decoder_cfg=active_decoder_cfg,
            rng_seed=seed_planner.seed_for(idx, variant, attempts) if seed_planner else None,
            scaffold_response=False,
        )
        metrics = evaluator.evaluate(record, generated)
        repeat_similarity = metrics.get("char_repeat_max")
        repeat_similarity_val = (
            float(repeat_similarity)
            if isinstance(repeat_similarity, (int, float)) and math.isfinite(repeat_similarity)
            else 0.0
        )
        force_repeat_flag = repeat_similarity_val >= _CHAR_REPEAT_ALERT
        flagged_for_repetition = False
        if quality_gate:
            flagged, flag_reasons = quality_gate.process(record, generated, metrics)
            if flagged:
                joined = "; ".join(flag_reasons) if flag_reasons else "low-quality sample"
                log(f"[eval] {tag}: flagged for retraining ({joined}).")
        if force_repeat_flag:
            flagged_for_repetition = True
            if not flagged:
                flagged = True
            reason = f"repeat_similarity>={_CHAR_REPEAT_ALERT:.2f}"
            if reason not in flag_reasons:
                flag_reasons.append(reason)
        elif flagged:
            flagged_for_repetition = any(
                str(reason).startswith("repeat_similarity>=") for reason in flag_reasons
            )
        if flagged and attempts < _MAX_BATCH_ATTEMPTS:
            joined = "; ".join(flag_reasons) if flag_reasons else "low-quality sample"
            if flagged_for_repetition:
                boosted_cfg = _boost_decoder_penalties(
                    entry.get("decoder_cfg_override") or decoder_cfg,
                )
                entry["decoder_cfg_override"] = boosted_cfg
                log(
                    f"[eval] {tag}: boosting decoder penalties to presence={boosted_cfg.presence_penalty:.2f} "
                    f"/ frequency={boosted_cfg.frequency_penalty:.2f} after repeat similarity "
                    f"{repeat_similarity_val:.2f}."
                )
            log(
                f"[eval] {tag}: re-queueing flagged sample "
                f"(attempt {attempts + 1}/{_MAX_BATCH_ATTEMPTS}) due to {joined}."
            )
            insert_at = queue_rng.randint(0, len(pending))
            pending.insert(insert_at, entry)
            continue
        if flagged:
            _schedule_future_retries(record_key)
        else:
            _clear_future_retries(record_key)
        context_label = _format_context_tokens(record.context_tokens)
        log(
            "[eval] {tag}: context={context} lexical={lex:.2f} rougeL={rouge:.2f} "
            "ppl(gen)={ppl_gen:.1f} ppl(ref)={ppl_ref:.1f} sim={sim:.2f} len_ratio={ratio:.2f} "
            "prompt='{prompt}' response='{response}'".format(
                tag=tag,
                context=context_label,
                lex=metrics["lexical"],
                rouge=metrics["rougeL"],
                ppl_gen=metrics["ppl_generated"],
                ppl_ref=metrics["ppl_reference"],
                sim=metrics.get("semantic_similarity") or 0.0,
                ratio=metrics.get("length_ratio") or 0.0,
                prompt=preview(record.prompt),
                response=preview(generated),
            )
        )
        results.append(
            EvaluationSampleResult(
                index=idx,
                label=label,
                prompt=record.prompt,
                reference=record.response,
                generated=generated,
                context_tokens=dict(record.context_tokens),
                metrics=metrics,
                flagged=flagged,
                variant=variant,
            )
        )
    summary = summarize_samples(results)
    if summary:
        metric_order = [
            ("lexical_mean", "lex"),
            ("rougeL_mean", "rougeL"),
            ("ppl_generated_mean", "ppl(gen)"),
            ("ppl_reference_mean", "ppl(ref)"),
            ("semantic_similarity_mean", "sim"),
            ("length_ratio_mean", "len_ratio"),
            ("quality_score_mean", "quality"),
        ]
        parts: list[str] = []
        for key, label_name in metric_order:
            value = summary.get(key)
            if value is None:
                continue
            fmt = "{value:.2f}"
            if "ppl" in label_name:
                fmt = "{value:.1f}"
            parts.append(f"{label_name}={fmt.format(value=value)}")
        if parts:
            joined = ", ".join(parts)
            log(f"[eval] {label} averages -> {joined}")
    if logger:
        logger.log_eval_batch(label, results, summary)
    duration = time.perf_counter() - eval_start
    try:
        telemetry_after = resource_monitor.snapshot()
    except Exception:
        telemetry_after = None
    telemetry_delta: ResourceDelta | None = None
    if telemetry_before and telemetry_after:
        try:
            telemetry_delta = resource_monitor.delta(telemetry_before, telemetry_after)
        except Exception:
            telemetry_delta = None
    if telemetry_delta:
        log(f"[eval] {label} resources -> {resource_monitor.describe(telemetry_delta)}")
    rss_before = telemetry_before.rss_mb if telemetry_before else None
    rss_after = telemetry_after.rss_mb if telemetry_after else None
    rss_delta = (
        rss_after - rss_before if rss_after is not None and rss_before is not None else None
    )
    if logger:
        logger.log_profile(
            f"{label} eval",
            tokens=len(results),
            duration=duration,
            rss_before=rss_before,
            rss_after=rss_after,
            rss_delta=rss_delta,
            resources=ResourceMonitor.to_event(telemetry_delta) if telemetry_delta else None,
        )
    return results


class EvalLogWriter:
    """Persists evaluation + profiling metrics as structured JSON."""

    def __init__(self, path: Path, run_metadata: dict[str, Any]) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.run_id = run_metadata.get("run_id") or f"train-{uuid.uuid4().hex}"
        self.started_at = dt.datetime.utcnow()
        self.metadata = run_metadata
        self.events: list[dict[str, Any]] = []
        self._closed = False

    def log_eval_batch(
        self,
        label: str,
        samples: Sequence[EvaluationSampleResult],
        summary: dict[str, float | None] | None,
    ) -> None:
        if not samples:
            return
        event = {
            "ts": self._timestamp(),
            "type": "evaluation",
            "label": label,
            "sample_count": len(samples),
            "summary": self._sanitize_summary(summary or {}),
            "samples": [
                {
                    "index": sample.index,
                    "variant": sample.variant,
                    "context": sample.context_tokens,
                    "prompt": preview(sample.prompt, width=240),
                    "reference": preview(sample.reference, width=240),
                    "generated": preview(sample.generated, width=240),
                    "metrics": self._sanitize_metrics(sample.metrics),
                    "flagged": sample.flagged,
                }
                for sample in samples
            ],
        }
        cycle_reference = self._cycle_reference(samples)
        if cycle_reference:
            event["cycle_reference"] = cycle_reference
        self.events.append(event)

    def log_profile(
        self,
        label: str,
        *,
        tokens: int,
        duration: float,
        rss_before: float | None,
        rss_after: float | None,
        rss_delta: float | None,
        resources: dict[str, Any] | None = None,
    ) -> None:
        event: dict[str, Any] = {
            "ts": self._timestamp(),
            "type": "profile",
            "label": label,
            "tokens": tokens,
            "duration_sec": round(duration, 3),
        }
        if rss_before is not None:
            event["rss_before_mb"] = round(rss_before, 3)
        if rss_after is not None:
            event["rss_after_mb"] = round(rss_after, 3)
        if rss_delta is not None:
            event["rss_delta_mb"] = round(rss_delta, 3)
        if resources:
            event["resources"] = resources
        self.events.append(event)

    def finalize(
        self,
        *,
        totals: dict[str, Any],
        status: str = "success",
    ) -> None:
        if self._closed:
            return
        payload = {
            "run_id": self.run_id,
            "status": status,
            "started_at": self.started_at.isoformat(timespec="seconds") + "Z",
            "completed_at": self._timestamp(),
            "metadata": self.metadata,
            "totals": totals,
            "events": self.events,
        }
        with self.path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
        self._closed = True

    @staticmethod
    def _clean_number(value: Any) -> float | None:
        if not isinstance(value, (int, float)):
            return None
        if not math.isfinite(value):
            return None
        return round(float(value), 4)

    def _sanitize_metrics(self, metrics: dict[str, Any]) -> dict[str, float | None]:
        return {key: self._clean_number(value) for key, value in metrics.items()}

    def _sanitize_summary(self, summary: dict[str, Any]) -> dict[str, float | None]:
        return {key: self._clean_number(value) for key, value in summary.items()}

    @staticmethod
    def _timestamp() -> str:
        return dt.datetime.utcnow().isoformat(timespec="seconds") + "Z"

    def _cycle_reference(self, samples: Sequence[EvaluationSampleResult]) -> dict[str, Any] | None:
        for sample in samples:
            reference_sentence = extract_complete_sentence(sample.reference)
            generated_sentence = extract_complete_sentence(sample.generated)
            if not reference_sentence and not generated_sentence:
                continue
            return {
                "index": sample.index,
                "variant": sample.variant,
                "prompt": sample.prompt,
                "reference_sentence": reference_sentence,
                "generated_sentence": generated_sentence,
                "flagged": sample.flagged,
            }
        return None


class QualityGate:
    """Collects low-quality generations for targeted retraining."""

    def __init__(
        self,
        output_path: str | Path,
        *,
        grammar_threshold: int = 3,
        cola_floor: float = 0.45,
        similarity_floor: float = 0.55,
        structure_floor: float = 0.35,
        common_token_ceiling: float = 0.55,
        repeat_similarity_ceiling: float = _CHAR_REPEAT_ALERT,
    ) -> None:
        self.path = Path(output_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.grammar_threshold = grammar_threshold
        self.cola_floor = cola_floor
        self.similarity_floor = similarity_floor
        self.structure_floor = structure_floor
        self.common_token_ceiling = common_token_ceiling
        self.repeat_similarity_ceiling = repeat_similarity_ceiling

    def process(
        self,
        record: EvaluationRecord,
        candidate: str,
        metrics: dict[str, Any],
    ) -> tuple[bool, list[str]]:
        flagged, reasons = self._should_flag(metrics)
        if not flagged:
            return False, []
        event = {
            "ts": dt.datetime.utcnow().isoformat(timespec="seconds") + "Z",
            "prompt": record.prompt,
            "reference": record.response,
            "generated": candidate,
            "context": record.context_tokens,
            "metrics": metrics,
            "reasons": reasons,
        }
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event) + "\n")
        return True, reasons

    def _should_flag(self, metrics: dict[str, Any]) -> tuple[bool, list[str]]:
        reasons: list[str] = []
        grammar_errors = metrics.get("grammar_errors")
        if isinstance(grammar_errors, (int, float)) and grammar_errors >= self.grammar_threshold:
            reasons.append(f"grammar_errors>={self.grammar_threshold}")
        cola = metrics.get("cola_acceptability")
        if isinstance(cola, (int, float)) and cola < self.cola_floor:
            reasons.append(f"cola<{self.cola_floor}")
        similarity = metrics.get("semantic_similarity")
        if isinstance(similarity, (int, float)) and similarity < self.similarity_floor:
            reasons.append(f"semantic_similarity<{self.similarity_floor}")
        length_ratio = metrics.get("length_ratio")
        if isinstance(length_ratio, (int, float)) and (length_ratio < 0.6 or length_ratio > 1.4):
            reasons.append("length_mismatch")
        structure_variety = metrics.get("structure_variety")
        if isinstance(structure_variety, (int, float)) and structure_variety < self.structure_floor:
            reasons.append(f"structure_variety<{self.structure_floor}")
        token_penalty = metrics.get("common_token_penalty")
        if isinstance(token_penalty, (int, float)) and token_penalty > self.common_token_ceiling:
            reasons.append(f"common_token_penalty>{self.common_token_ceiling}")
        repeat_similarity = metrics.get("char_repeat_max")
        if (
            isinstance(repeat_similarity, (int, float))
            and repeat_similarity >= self.repeat_similarity_ceiling
        ):
            reasons.append(f"repeat_similarity>={self.repeat_similarity_ceiling:.2f}")
        return (len(reasons) > 0, reasons)
