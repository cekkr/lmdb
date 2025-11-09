from __future__ import annotations

import math
import textwrap
from dataclasses import dataclass
from typing import Sequence

from .inference_shared import issue_prompt
from .metrics import lexical_overlap, rouge_l_score
from .pipeline import DBSLMEngine


@dataclass(frozen=True)
class EvaluationRecord:
    prompt: str
    response: str
    emotion: str = "unknown"


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


def run_inference_records(
    engine: DBSLMEngine,
    evaluator: ResponseEvaluator,
    records: Sequence[EvaluationRecord],
    *,
    label: str,
    user_id: str = "trainer",
    agent_name: str = "db-slm",
) -> None:
    if not records:
        return
    print(
        f"[eval] Running {len(records)} inference probe(s) from {label} to gauge training quality."
    )
    for idx, record in enumerate(records, start=1):
        _, generated = issue_prompt(engine, record.prompt, user_id=user_id, agent_name=agent_name)
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
