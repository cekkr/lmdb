from __future__ import annotations

import argparse
import itertools
import json
import math
import random
import sys
import textwrap
import time
from pathlib import Path
from typing import Callable, Iterable, List, Sequence, Tuple

try:  # resource is Unix-only but gives precise RSS readings when available.
    import resource  # type: ignore
except ImportError:  # pragma: no cover - Windows fallback
    resource = None  # type: ignore

from db_slm import DBSLMEngine
from db_slm.metrics import lexical_overlap, rouge_l_score
from db_slm.settings import load_settings


def build_parser(default_db_path: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Ingest raw text corpora into the DB-SLM SQLite backing store."
    )
    parser.add_argument(
        "inputs",
        nargs="*",
        help="Text files or directories containing *.txt files to ingest.",
    )
    parser.add_argument(
        "--db",
        default=default_db_path,
        help="Path to the SQLite database file (default: %(default)s).",
    )
    parser.add_argument(
        "--ngram-order",
        type=int,
        default=3,
        help="N-gram order to enforce while training (default: %(default)s).",
    )
    parser.add_argument(
        "--encoding",
        default="utf-8",
        help="File encoding used while reading corpora (default: %(default)s).",
    )
    parser.add_argument(
        "--stdin",
        action="store_true",
        help="Read an additional corpus from STDIN.",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="When a directory is provided, recursively ingest *.txt files.",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Delete the target database (if it exists) before training.",
    )
    parser.add_argument(
        "--eval-interval",
        type=int,
        default=0,
        help="When > 0, run inference probes every N tokens ingested (default: disabled).",
    )
    parser.add_argument(
        "--eval-samples",
        type=int,
        default=3,
        help="Number of held-out prompts to test during each inference probe (default: %(default)s).",
    )
    parser.add_argument(
        "--eval-dataset",
        help="Path to an NDJSON dataset containing 'prompt' and 'response' fields. Defaults to DBSLM_DATASET_PATH.",
    )
    parser.add_argument(
        "--eval-pool-size",
        type=int,
        default=200,
        help="Maximum number of held-out rows to keep in memory for inference probes (default: %(default)s).",
    )
    parser.add_argument(
        "--json-chunk-size",
        type=int,
        default=500,
        help="Number of JSON rows to concatenate per training chunk when ingesting *.json/NDJSON corpora (default: %(default)s).",
    )
    parser.add_argument(
        "--max-json-lines",
        type=int,
        default=0,
        help="Optional cap on the number of JSON/NDJSON lines to ingest per file (default: 0 = unlimited).",
    )
    parser.add_argument(
        "--profile-ingest",
        action="store_true",
        help="Measure ingest latency + RSS per corpus to size streaming runs.",
    )
    return parser


def resolve_db_path(raw: str, reset: bool) -> Tuple[str, Path | None]:
    if raw == ":memory:":
        if reset:
            raise ValueError("--reset cannot be combined with the in-memory database")
        return raw, None
    path = Path(raw).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    if reset and path.exists():
        path.unlink()
    return str(path), path


def collect_files(entries: Sequence[str], recursive: bool) -> List[Path]:
    files: list[Path] = []
    for entry in entries:
        path = Path(entry).expanduser()
        if path.is_file():
            files.append(path)
            continue
        if path.is_dir():
            pattern = "**/*.txt" if recursive else "*.txt"
            for candidate in sorted(path.glob(pattern)):
                if candidate.is_file():
                    files.append(candidate)
            continue
        raise FileNotFoundError(f"No such file or directory: {path}")
    return files


def iter_corpora(
    paths: Iterable[Path],
    encoding: str,
    json_chunk_size: int,
    max_json_lines: int,
) -> Iterable[Tuple[str, str]]:
    for path in paths:
        suffix = path.suffix.lower()
        if suffix in {".json", ".ndjson"}:
            yield from iter_json_chunks(path, json_chunk_size, max_json_lines)
            continue
        text = path.read_text(encoding=encoding)
        yield str(path), text


def iter_json_chunks(path: Path, chunk_size: int, max_lines: int) -> Iterable[Tuple[str, str]]:
    chunk_size = max(1, chunk_size)
    buffer: list[str] = []
    chunk_index = 0
    consumed = 0
    limit = max(0, max_lines)
    with path.open("r", encoding="utf-8") as handle:
        for line_no, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                print(f"[train] JSON ingest warning ({path} line {line_no}): {exc}")
                continue
            prompt = payload.get("prompt", "")
            response = payload.get("response", "")
            emotion = payload.get("emotion", "")
            if not response:
                continue
            segment = "\n".join(
                filter(
                    None,
                    [
                        f"Prompt: {prompt.strip()}" if prompt else None,
                        f"Emotion: {emotion}" if emotion else None,
                        f"Response: {response.strip()}",
                    ],
                )
            )

            print("Trained line #", line_no, "\t (Prompt: ", prompt,")")

            buffer.append(segment)
            consumed += 1
            if len(buffer) >= chunk_size:
                chunk_index += 1
                yield (f"{path}#chunk{chunk_index}", "\n\n".join(buffer))
                buffer.clear()
            if limit and consumed >= limit:
                break
    if buffer:
        chunk_index += 1
        yield (f"{path}#chunk{chunk_index}", "\n\n".join(buffer))
    suffix = f" (capped at {limit} line(s))" if limit and consumed >= limit else ""
    print(f"[train] Prepared {chunk_index} chunk(s) from {path} using chunk size {chunk_size}{suffix}.")


def load_eval_dataset(path: Path, max_records: int | None = None) -> List[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Evaluation dataset not found: {path}")
    records: list[dict[str, str]] = []
    limit = max_records if max_records is not None and max_records > 0 else None
    with path.open("r", encoding="utf-8") as handle:
        for line_no, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                print(f"[eval] Skipping line {line_no}: {exc}")
                continue
            prompt = payload.get("prompt")
            response = payload.get("response")
            if not prompt or not response:
                continue
            record = {
                "prompt": prompt,
                "response": response,
                "emotion": payload.get("emotion", "unknown"),
            }
            records.append(record)
            if limit is not None and len(records) >= limit:
                break
    if limit is not None:
        random.shuffle(records)
    return records


def preview(text: str, width: int = 500) -> str:
    collapsed = " ".join(text.split())
    if not collapsed:
        return ""
    return textwrap.shorten(collapsed, width=width, placeholder="…")


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


class InferenceMonitor:
    def __init__(
        self,
        engine: DBSLMEngine,
        dataset: List[dict[str, str]],
        interval_tokens: int,
        samples_per_cycle: int,
        evaluator: ResponseEvaluator,
    ) -> None:
        self.engine = engine
        self.dataset = dataset
        self.interval = interval_tokens
        self.samples = max(1, samples_per_cycle)
        self.next_threshold = interval_tokens
        self.evaluator = evaluator

    def enabled(self) -> bool:
        return self.interval > 0 and bool(self.dataset)

    def maybe_run(self, total_tokens: int) -> None:
        if not self.enabled() or total_tokens < self.next_threshold:
            return
        while total_tokens >= self.next_threshold:
            self._run_cycle(self.next_threshold)
            self.next_threshold += self.interval

    def _run_cycle(self, threshold: int) -> None:
        sample_size = min(len(self.dataset), self.samples)
        selections = random.sample(self.dataset, sample_size)
        print(
            f"[eval] Running {sample_size} inference probe(s) at {threshold} ingested tokens to gauge training quality."
        )
        for idx, record in enumerate(selections, start=1):
            conversation_id = self.engine.start_conversation(user_id="trainer", agent_name="db-slm")
            generated = self.engine.respond(conversation_id, record["prompt"])
            metrics = self.evaluator.evaluate(record["prompt"], record["response"], generated)
            print(
                "[eval] #{idx}: emotion={emotion} lexical={lex:.2f} rougeL={rouge:.2f} "
                "ppl(gen)={ppl_gen:.1f} ppl(ref)={ppl_ref:.1f} prompt='{prompt}' response='{response}'".format(
                    idx=idx,
                    emotion=record["emotion"],
                    lex=metrics["lexical"],
                    rouge=metrics["rougeL"],
                    ppl_gen=metrics["ppl_generated"],
                    ppl_ref=metrics["ppl_reference"],
                    prompt=preview(record["prompt"]),
                    response=preview(generated),
                )
            )


class IngestProfiler:
    """Optional profiler that logs ingest latency and current RSS."""

    def __init__(self, enabled: bool) -> None:
        self.enabled = enabled

    def measure(self, label: str, fn: Callable[[], int]) -> int:
        if not self.enabled:
            return fn()
        rss_before = self._rss_mb()
        start = time.perf_counter()
        tokens = fn()
        duration = time.perf_counter() - start
        rss_after = self._rss_mb()
        rss_delta = rss_after - rss_before if rss_before is not None and rss_after is not None else None
        suffix = ""
        if rss_after is not None:
            delta_str = f"{rss_delta:+.1f}MB" if rss_delta is not None else "n/a"
            suffix = f" rss={rss_after:.1f}MB (Δ{delta_str})"
        print(f"[profile] {label}: {tokens} tokens in {duration:.2f}s{suffix}")
        return tokens

    @staticmethod
    def _rss_mb() -> float | None:
        if resource is None:  # pragma: no cover - platform without resource
            return None
        usage = resource.getrusage(resource.RUSAGE_SELF)
        rss = usage.ru_maxrss
        # On macOS the value is in bytes, elsewhere it is kilobytes.
        if sys.platform == "darwin":
            rss_mb = rss / (1024 * 1024)
        else:
            rss_mb = rss / 1024
        return float(rss_mb)


def main() -> None:
    settings = load_settings()
    parser = build_parser(settings.sqlite_dsn())
    args = parser.parse_args()

    if not args.inputs and not args.stdin:
        parser.error("Provide at least one input path or enable --stdin")

    db_path_str, db_path = resolve_db_path(args.db, args.reset)
    engine = DBSLMEngine(db_path_str, ngram_order=args.ngram_order)

    file_inputs = collect_files(args.inputs, args.recursive)
    corpora_iter: Iterable[Tuple[str, str]] = iter_corpora(
        file_inputs, args.encoding, args.json_chunk_size, args.max_json_lines
    )
    if args.stdin:
        stdin_payload = sys.stdin.read()
        if stdin_payload.strip():
            corpora_iter = itertools.chain(corpora_iter, [("<stdin>", stdin_payload)])

    # We defer the "empty" validation until after attempting to iterate so JSON inputs
    # can stream chunks without pre-loading them into memory.

    eval_dataset_path = args.eval_dataset or settings.dataset_path
    evaluator = ResponseEvaluator(engine)
    eval_records: list[dict[str, str]] = []
    monitor = InferenceMonitor(engine, eval_records, args.eval_interval, args.eval_samples, evaluator)
    if args.eval_interval > 0:
        dataset_path = Path(eval_dataset_path).expanduser()
        try:
            eval_records = load_eval_dataset(dataset_path, args.eval_pool_size)
            monitor = InferenceMonitor(
                engine, eval_records, args.eval_interval, args.eval_samples, evaluator
            )
            print(f"[eval] Loaded {len(eval_records)} held-out sample(s) from {dataset_path}.")
        except FileNotFoundError as exc:
            print(f"[eval] Warning: {exc}. Disabling evaluation probes.")
            monitor = InferenceMonitor(engine, [], 0, args.eval_samples, evaluator)

    profiler = IngestProfiler(args.profile_ingest)
    total_tokens = 0
    total_windows = 0
    processed_corpora = 0
    print(f"[train] Starting ingest into {db_path_str} with order={engine.store.order}.")
    for label, corpus in corpora_iter:
        processed_corpora += 1
        print(f"[train] Processing {label} ({len(corpus)} bytes)...")
        token_count = profiler.measure(label, lambda text=corpus: engine.train_from_text(text))
        if token_count == 0:
            print(f"[train] Skipping {label} (corpus too small for order={engine.store.order})")
            continue
        window = max(0, token_count - engine.store.order + 1)
        total_tokens += token_count
        total_windows += window
        print(f"[train] Ingested {label}: {token_count} tokens -> {window} n-grams")
        monitor.maybe_run(total_tokens)

    if processed_corpora == 0:
        parser.error("No readable corpora found in the provided inputs")

    if total_windows == 0:
        parser.error(
            "No usable training windows were produced. Provide larger corpora or reduce --ngram-order."
        )

    location = db_path if db_path is not None else db_path_str
    print(
        f"[train] Completed ingest: {total_tokens} tokens / {total_windows} n-grams stored in {location}"
    )
    engine.db.close()


if __name__ == "__main__":
    main()
