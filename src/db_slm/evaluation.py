from __future__ import annotations

import datetime as dt
import hashlib
import json
import math
import random
import textwrap
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from .inference_shared import issue_prompt
from .metrics import lexical_overlap, rouge_l_score
from .pipeline import DBSLMEngine
from .quality import SentenceQualityScorer


@dataclass(frozen=True)
class EvaluationRecord:
    prompt: str
    response: str
    emotion: str = "unknown"


@dataclass(frozen=True)
class EvaluationSampleResult:
    index: int
    label: str
    prompt: str
    reference: str
    generated: str
    emotion: str
    metrics: dict[str, float | int | None]
    flagged: bool = False


_MAX_BATCH_ATTEMPTS = 2
_MAX_FUTURE_BATCH_RETRIES = 3
_flagged_retry_budget: dict[str, int] = {}


def _record_signature(record: EvaluationRecord) -> str:
    hasher = hashlib.sha1()
    hasher.update(record.prompt.encode("utf-8", errors="ignore"))
    hasher.update(b"\x1f")
    hasher.update(record.response.encode("utf-8", errors="ignore"))
    hasher.update(b"\x1f")
    hasher.update(record.emotion.encode("utf-8", errors="ignore"))
    return hasher.hexdigest()


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


class ResponseEvaluator:
    """Aggregates lightweight eval metrics for streaming probes."""

    def __init__(self, engine: DBSLMEngine) -> None:
        self.engine = engine
        embedder_model = getattr(getattr(engine, "settings", None), "embedder_model", "all-MiniLM-L6-v2")
        shared_embedder = getattr(getattr(engine, "segment_embedder", None), "embedder", None)
        self.quality = SentenceQualityScorer(embedder_model=embedder_model, embedder=shared_embedder)

    def evaluate(self, prompt: str, reference: str, candidate: str) -> dict[str, float | int | None]:
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
        metrics.update(self.quality.combined_quality(candidate, reference, lexical))
        return metrics

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
) -> list[EvaluationSampleResult]:
    if not records:
        return []
    print(
        f"[eval] Running {len(records)} inference probe(s) from {label} to gauge training quality."
    )
    results: list[EvaluationSampleResult] = []
    pending: list[dict[str, Any]] = []
    for idx, record in enumerate(records, start=1):
        record_key = _record_signature(record)
        if _consume_future_retry_budget(record_key):
            print(
                f"[eval] #{idx}: skipping flagged sample; retry budget exhausted "
                f"(prompt='{preview(record.prompt, 120)}')."
            )
            continue
        pending.append(
            {
                "index": idx,
                "record": record,
                "record_key": record_key,
                "attempts": 0,
            }
        )
    while pending:
        entry = pending.pop(0)
        record = entry["record"]
        idx = entry["index"]
        record_key = entry["record_key"]
        entry["attempts"] += 1
        attempts = entry["attempts"]
        generated = ""
        metrics: dict[str, float | int | None] = {}
        ref_words = max(1, len(record.response.split()))
        min_words = max(20, min(512, int(ref_words * 0.85)))
        flagged = False
        flag_reasons: list[str] = []
        _, generated = issue_prompt(
            engine,
            record.prompt,
            user_id=user_id,
            agent_name=agent_name,
            seed_history=False,
            min_response_words=min_words,
        )
        metrics = evaluator.evaluate(record.prompt, record.response, generated)
        if quality_gate:
            flagged, flag_reasons = quality_gate.process(record, generated, metrics)
            if flagged:
                joined = "; ".join(flag_reasons) if flag_reasons else "low-quality sample"
                print(f"[eval] #{idx}: flagged for retraining ({joined}).")
        if flagged and attempts < _MAX_BATCH_ATTEMPTS:
            joined = "; ".join(flag_reasons) if flag_reasons else "low-quality sample"
            print(
                f"[eval] #{idx}: re-queueing flagged sample "
                f"(attempt {attempts + 1}/{_MAX_BATCH_ATTEMPTS}) due to {joined}."
            )
            insert_at = random.randint(0, len(pending))
            pending.insert(insert_at, entry)
            continue
        if flagged:
            _schedule_future_retries(record_key)
        else:
            _clear_future_retries(record_key)
        print(
            "[eval] #{idx}: emotion={emotion} lexical={lex:.2f} rougeL={rouge:.2f} "
            "ppl(gen)={ppl_gen:.1f} ppl(ref)={ppl_ref:.1f} sim={sim:.2f} len_ratio={ratio:.2f} "
            "prompt='{prompt}' response='{response}'".format(
                idx=idx,
                emotion=record.emotion,
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
                emotion=record.emotion,
                metrics=metrics,
                flagged=flagged,
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
            print(f"[eval] {label} averages -> {joined}")
    if logger:
        logger.log_eval_batch(label, results, summary)
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
                    "emotion": sample.emotion,
                    "prompt": preview(sample.prompt, width=240),
                    "reference": preview(sample.reference, width=240),
                    "generated": preview(sample.generated, width=240),
                    "metrics": self._sanitize_metrics(sample.metrics),
                    "flagged": sample.flagged,
                }
                for sample in samples
            ],
        }
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


class QualityGate:
    """Collects low-quality generations for targeted retraining."""

    def __init__(
        self,
        output_path: str | Path,
        *,
        grammar_threshold: int = 3,
        cola_floor: float = 0.45,
        similarity_floor: float = 0.55,
    ) -> None:
        self.path = Path(output_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.grammar_threshold = grammar_threshold
        self.cola_floor = cola_floor
        self.similarity_floor = similarity_floor

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
            "emotion": record.emotion,
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
        return (len(reasons) > 0, reasons)
