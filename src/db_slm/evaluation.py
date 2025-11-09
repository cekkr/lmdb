from __future__ import annotations

import datetime as dt
import json
import math
import textwrap
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from .inference_shared import issue_prompt
from .metrics import lexical_overlap, rouge_l_score
from .pipeline import DBSLMEngine


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
    metrics: dict[str, float]


def preview(text: str, width: int = 1000) -> str:
    collapsed = " ".join(text.split())
    if not collapsed:
        return ""
    return textwrap.shorten(collapsed, width=width, placeholder="…\n")


class ResponseEvaluator:
    """Aggregates lightweight eval metrics for streaming probes."""

    def __init__(self, engine: DBSLMEngine) -> None:
        self.engine = engine

    def evaluate(self, prompt: str, reference: str, candidate: str) -> dict[str, float]:
        lexical = lexical_overlap(reference, candidate)
        rouge = rouge_l_score(reference, candidate)
        ppl_generated = self._perplexity(prompt, candidate)
        ppl_reference = self._perplexity(prompt, reference)
        return {
            "lexical": lexical,
            "rougeL": rouge,
            "ppl_generated": ppl_generated,
            "ppl_reference": ppl_reference,
        }

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
    return {
        "lexical_mean": _mean_metric(samples, "lexical"),
        "rougeL_mean": _mean_metric(samples, "rougeL"),
        "ppl_generated_mean": _mean_metric(samples, "ppl_generated"),
        "ppl_reference_mean": _mean_metric(samples, "ppl_reference"),
    }


def run_inference_records(
    engine: DBSLMEngine,
    evaluator: ResponseEvaluator,
    records: Sequence[EvaluationRecord],
    *,
    label: str,
    user_id: str = "trainer",
    agent_name: str = "db-slm",
    logger: EvalLogWriter | None = None,
) -> list[EvaluationSampleResult]:
    if not records:
        return []
    print(
        f"[eval] Running {len(records)} inference probe(s) from {label} to gauge training quality."
    )
    results: list[EvaluationSampleResult] = []
    for idx, record in enumerate(records, start=1):
        _, generated = issue_prompt(
            engine,
            record.prompt,
            user_id=user_id,
            agent_name=agent_name,
            seed_history=False,
            min_response_words=20,
        )
        metrics = evaluator.evaluate(record.prompt, record.response, generated)
        print(
            "[eval] #{idx}: emotion={emotion} lexical={lex:.2f} rougeL={rouge:.2f} "
            "ppl(gen)={ppl_gen:.1f} ppl(ref)={ppl_ref:.1f} prompt='{prompt}' response='{response}'".format(
                idx=idx,
                emotion=record.emotion,
                lex=metrics["lexical"],
                rouge=metrics["rougeL"],
                ppl_gen=metrics["ppl_generated"],
                ppl_ref=metrics["ppl_reference"],
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
            )
        )
    summary = summarize_samples(results)
    if summary:
        metric_order = [
            ("lexical_mean", "lex"),
            ("rougeL_mean", "rougeL"),
            ("ppl_generated_mean", "ppl(gen)"),
            ("ppl_reference_mean", "ppl(ref)"),
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
