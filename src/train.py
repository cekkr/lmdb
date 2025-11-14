from __future__ import annotations

import argparse
import itertools
import json
import os
import random
import sys
import time
from datetime import datetime
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, List, Sequence, Tuple

try:  # resource is Unix-only but gives precise RSS readings when available.
    import resource  # type: ignore
except ImportError:  # pragma: no cover - Windows fallback
    resource = None  # type: ignore

from db_slm import DBSLMEngine
from db_slm.adapters.base import NullHotPathAdapter
from db_slm.adapters.cheetah import CheetahClient
from db_slm.context_dimensions import (
    DEFAULT_CONTEXT_DIMENSIONS,
    format_context_dimensions,
    parse_context_dimensions_arg,
)
from db_slm.decoder import DecoderConfig
from db_slm.evaluation import (
    EvalLogWriter,
    DependencyLayer,
    EvaluationRecord,
    QualityGate,
    ResponseEvaluator,
    VariantSeedPlanner,
    build_dependency_layer,
    run_inference_records,
)
from db_slm.settings import DBSLMSettings, load_settings

from log_helpers import log


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
        "--context-dimensions",
        help=(
            "Comma-separated token span ranges (e.g. '1-2,3-5') or progressive lengths like '4,8,4' "
            "used to group context penalties. Use 'off' to disable the additional grouping penalties."
        ),
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
        "--chunk-eval-percent",
        type=float,
        default=0.0,
        help="Percentage of each JSON chunk to reserve for immediate evaluation probes (default: %(default)s).",
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
        "--seed",
        type=int,
        help="Seed the Python RNG for reproducible chunking, hold-outs, and dataset shuffles (default: system entropy).",
    )
    parser.add_argument(
        "--eval-seed",
        type=int,
        help="Base seed for evaluation-time sampling. When omitted, a fresh random seed is generated per run.",
    )
    parser.add_argument(
        "--eval-variants",
        type=int,
        help="Number of decoded variants to produce per evaluation prompt (default: 2 when context dimensions are enabled, otherwise 1).",
    )
    parser.add_argument(
        "--profile-ingest",
        action="store_true",
        help="Measure ingest latency + RSS per corpus to size streaming runs.",
    )
    parser.add_argument(
        "--metrics-export",
        help=(
            "Path to a JSON file that will store the evaluation/perplexity timeline."
            " Defaults to var/eval_logs/train-<timestamp>.json. Use '-' to disable."
        ),
    )
    parser.add_argument(
        "--decoder-presence-penalty",
        type=float,
        default=None,
        help=(
            "Override DecoderConfig.presence_penalty during evaluation probes/hold-outs "
            "(default: decoder profile value)."
        ),
    )
    parser.add_argument(
        "--decoder-frequency-penalty",
        type=float,
        default=None,
        help=(
            "Override DecoderConfig.frequency_penalty during evaluation probes/hold-outs "
            "(default: decoder profile value)."
        ),
    )
    parser.add_argument(
        "--backonsqlite",
        action="store_true",
        help=(
            "Permit a SQLite-only fallback when DBSLM_BACKEND=cheetah-db but the cheetah server is unavailable. "
            "Default behavior aborts instead of silently downgrading."
        ),
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


_CHEETAH_PURGE_PREFIXES: tuple[bytes, ...] = (
    b"ctx:",
    b"ctxv:",
    b"topk:",
    b"cnt:",
    b"prob:",
    b"cont:",
    b"meta:",
)


def _purge_cheetah_namespace(
    client: CheetahClient,
    prefix: bytes,
    *,
    page_size: int = 1024,
) -> int:
    """Remove all pair entries (and backing values) for the given namespace prefix."""
    namespace_label = prefix.decode("utf-8", "ignore").rstrip(":") or prefix.hex()
    removed = 0
    scan_warned = False
    delete_warned = False
    pair_warned = False
    cursor: bytes | None = None
    started = time.monotonic()
    progress_interval = max(page_size * 5, 5000)
    while True:
        result = client.pair_scan(prefix=prefix, limit=page_size, cursor=cursor)
        if result is None:
            if not scan_warned:
                log(f"[train] Warning: cheetah reset aborted while scanning '{namespace_label}'.")
                scan_warned = True
            break
        entries, cursor = result
        if not entries:
            break
        for raw_value, key in entries:
            deleted, response = client.delete(key)
            if not deleted:
                if not delete_warned:
                    log(
                        f"[train] Warning: failed to delete cheetah key {key} in '{namespace_label}': {response}"
                    )
                    delete_warned = True
                continue
            removed += 1
            success, response = client.pair_del(raw_value)
            if not success and not pair_warned:
                identifier = raw_value.hex()
                log(
                    f"[train] Warning: unable to drop cheetah pair '{namespace_label}' entry {identifier}: {response}"
                )
                pair_warned = True
            if removed % progress_interval == 0:
                elapsed = time.monotonic() - started
                log(
                    "[train] cheetah reset: removed {count} '{label}' mappings so far "
                    "(~{rate:.1f}/s)".format(
                        count=removed,
                        label=namespace_label,
                        rate=removed / elapsed if elapsed > 0 else 0.0,
                    )
                )
    return removed


def reset_cheetah_store(settings: DBSLMSettings) -> None:
    """Clear the cached cheetah-db namespaces used by the trainer when --reset is supplied."""
    backend_active = settings.backend == "cheetah-db" or settings.cheetah_mirror
    if not backend_active:
        log("[train] --reset requested: cheetah hot-path disabled, skipping cheetah-db cleanup.")
        return
    client = CheetahClient(
        settings.cheetah_host,
        settings.cheetah_port,
        database=settings.cheetah_database,
        timeout=settings.cheetah_timeout_seconds,
    )
    if not client.connect():
        log(
            "[train] Warning: --reset requested but cheetah-db is unreachable "
            f"(targets: {client.describe_targets()}; errors: {client.describe_failures()})."
        )
        return
    try:
        total_removed = 0
        for prefix in _CHEETAH_PURGE_PREFIXES:
            removed = _purge_cheetah_namespace(client, prefix)
            total_removed += removed
            if removed:
                label = prefix.decode("utf-8", "ignore").rstrip(":") or prefix.hex()
                log(f"[train] cheetah reset: cleared {removed} '{label}' mapping(s).")
        if total_removed == 0:
            log("[train] cheetah reset: no cached namespaces required clearing.")
        else:
            log(f"[train] cheetah reset: removed {total_removed} cached mapping(s) total.")
    finally:
        client.close()


def resolve_metrics_export_path(raw: str | None) -> Path | None:
    if raw is None or not raw.strip():
        base = Path("var/eval_logs")
        base.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
        return base / f"train-{timestamp}.json"
    cleaned = raw.strip()
    if cleaned == "-":
        return None
    path = Path(cleaned).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


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


@dataclass
class CorpusChunk:
    label: str
    train_text: str
    eval_records: list[EvaluationRecord]
    total_rows: int
    train_rows: int


class TrainingProgressPrinter:
    """Provides throttle-controlled training progress logs."""

    def __init__(self, label: str, train_rows: int) -> None:
        self.label = label
        self.train_rows = max(1, train_rows)
        self._last_emit = 0.0

    def __call__(self, stage: str, completed: int, total: int) -> None:
        if total <= 0:
            return
        now = time.perf_counter()
        if completed != total and (now - self._last_emit) < 0.75:
            return
        self._last_emit = now
        pct = (completed / total) * 100.0
        approx_row = min(self.train_rows, max(1, int(round(self.train_rows * (completed / total)))))
        stage_label = self._format_stage(stage)
        log(
            f"[train] {self.label}: {stage_label} {pct:5.1f}% "
            f"({completed}/{total}) ~line {approx_row}/{self.train_rows}"
        )

    def _format_stage(self, stage: str) -> str:
        if stage.startswith("order_"):
            return f"{stage.replace('_', '-')} windows"
        if stage.startswith("smooth_"):
            suffix = stage.split("_", 1)[1]
            if suffix.isdigit():
                return f"smoothing order-{suffix}"
            return "smoothing"
        labels = {
            "prepare": "preparing segments",
            "tokenize": "tokenizing corpus",
            "vocab": "updating vocabulary",
            "smooth_continuations": "continuation stats",
        }
        return labels.get(stage, stage)


def _serialize_dependency_layer(layer: DependencyLayer | None) -> dict[str, Any] | None:
    if layer is None:
        return None
    dependencies = [
        {
            "token": arc.token,
            "lemma": arc.lemma,
            "head": arc.head,
            "dep": arc.dep,
            "pos": arc.pos,
        }
        for arc in layer.arcs
    ]
    return {
        "backend": layer.backend,
        "token_count": layer.token_count,
        "strong_tokens": layer.strong_token_groups,
        "dependencies": dependencies,
    }


def _flatten_dependency_tokens(layer: DependencyLayer | None) -> list[str]:
    if layer is None or not layer.strong_token_groups:
        return []
    ordered: list[str] = []
    seen: set[str] = set()
    for bucket in sorted(layer.strong_token_groups):
        for token in layer.strong_token_groups[bucket]:
            if token in seen:
                continue
            ordered.append(token)
            seen.add(token)
    return ordered


def _dependency_layer_annotation(
    prompt_layer: DependencyLayer | None,
    response_layer: DependencyLayer | None,
) -> str | None:
    prompt_payload = _serialize_dependency_layer(prompt_layer)
    response_payload = _serialize_dependency_layer(response_layer)
    if not prompt_payload and not response_payload:
        return None
    payload = {
        "prompt": prompt_payload,
        "response": response_payload,
        "strong_reference": {
            "prompt": _flatten_dependency_tokens(prompt_layer),
            "response": _flatten_dependency_tokens(response_layer),
        },
    }
    return json.dumps(payload, ensure_ascii=False)


def iter_corpora(
    paths: Iterable[Path],
    encoding: str,
    json_chunk_size: int,
    max_json_lines: int,
    chunk_eval_percent: float,
) -> Iterable[CorpusChunk]:
    holdout_fraction = max(0.0, min(chunk_eval_percent, 100.0)) / 100.0
    for path in paths:
        suffix = path.suffix.lower()
        if suffix in {".json", ".ndjson"}:
            yield from iter_json_chunks(path, json_chunk_size, max_json_lines, holdout_fraction)
            continue
        text = path.read_text(encoding=encoding)
        row_count = max(1, len([line for line in text.splitlines() if line.strip()]))
        yield CorpusChunk(str(path), text, [], total_rows=row_count, train_rows=row_count)


def iter_json_chunks(
    path: Path,
    chunk_size: int,
    max_lines: int,
    holdout_fraction: float,
) -> Iterable[CorpusChunk]:
    chunk_size = max(1, chunk_size)
    entries: list[tuple[str, EvaluationRecord]] = []
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
                log(f"[train] JSON ingest warning ({path} line {line_no}): {exc}")
                continue
            prompt = payload.get("prompt", "")
            response = payload.get("response", "")
            emotion = payload.get("emotion", "unknown")
            if not response:
                continue
            prompt_layer = build_dependency_layer(prompt or "")
            response_layer = build_dependency_layer(response)
            segment_lines = list(
                filter(
                    None,
                    [
                        f"Prompt: {prompt.strip()}" if prompt else None,
                        f"Emotion: {emotion}" if emotion else None,
                        f"Response: {response.strip()}",
                    ],
                )
            )
            annotation = _dependency_layer_annotation(prompt_layer, response_layer)
            if annotation:
                segment_lines.append(f"DependencyLayer: {annotation}")
            segment = "\n".join(segment_lines)
            record = EvaluationRecord(
                prompt=prompt or "",
                response=response,
                emotion=emotion,
                prompt_dependencies=prompt_layer,
                response_dependencies=response_layer,
            )

            log(f"[train] Staged line #{line_no} (Prompt: {prompt})")

            entries.append((segment, record))
            consumed += 1
            if len(entries) >= chunk_size:
                chunk_index += 1
                yield _build_chunk(f"{path}#chunk{chunk_index}", entries, holdout_fraction)
                entries = []
            if limit and consumed >= limit:
                break
    if entries:
        chunk_index += 1
        yield _build_chunk(f"{path}#chunk{chunk_index}", entries, holdout_fraction)
    suffix = f" (capped at {limit} line(s))" if limit and consumed >= limit else ""
    log(f"[train] Prepared {chunk_index} chunk(s) from {path} using chunk size {chunk_size}{suffix}.")


def _build_chunk(
    label: str, entries: list[tuple[str, EvaluationRecord]], holdout_fraction: float
) -> CorpusChunk:
    holdout_indexes = _sample_holdouts(len(entries), holdout_fraction)
    train_segments = [
        segment for idx, (segment, _) in enumerate(entries) if idx not in holdout_indexes
    ]
    holdout_records = [
        record for idx, (_, record) in enumerate(entries) if idx in holdout_indexes
    ]
    train_text = "\n\n".join(train_segments)
    total_rows = len(entries) or 1
    train_rows = len(train_segments) or 1
    return CorpusChunk(label, train_text, holdout_records, total_rows=total_rows, train_rows=train_rows)


def _sample_holdouts(total_entries: int, fraction: float) -> set[int]:
    if total_entries <= 1 or fraction <= 0.0:
        return set()
    holdout_count = max(1, int(round(total_entries * fraction)))
    if holdout_count >= total_entries:
        holdout_count = total_entries - 1
    if holdout_count <= 0:
        return set()
    return set(random.sample(range(total_entries), holdout_count))


def load_eval_dataset(path: Path, max_records: int | None = None) -> List[EvaluationRecord]:
    if not path.exists():
        raise FileNotFoundError(f"Evaluation dataset not found: {path}")
    records: list[EvaluationRecord] = []
    limit = max_records if max_records is not None and max_records > 0 else None
    with path.open("r", encoding="utf-8") as handle:
        for line_no, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                log(f"[eval] Skipping line {line_no}: {exc}")
                continue
            prompt = payload.get("prompt")
            response = payload.get("response")
            if not prompt or not response:
                continue
            prompt_layer = build_dependency_layer(prompt)
            response_layer = build_dependency_layer(response)
            records.append(
                EvaluationRecord(
                    prompt=prompt,
                    response=response,
                    emotion=payload.get("emotion", "unknown"),
                    prompt_dependencies=prompt_layer,
                    response_dependencies=response_layer,
                )
            )
            if limit is not None and len(records) >= limit:
                break
    if limit is not None:
        random.shuffle(records)
    return records


class InferenceMonitor:
    _MIN_SAMPLES = 2

    def __init__(
        self,
        engine: DBSLMEngine,
        dataset: List[EvaluationRecord],
        interval_tokens: int,
        samples_per_cycle: int,
        evaluator: ResponseEvaluator,
        logger: EvalLogWriter | None = None,
        quality_gate: QualityGate | None = None,
        max_dataset_size: int | None = None,
        decoder_cfg: DecoderConfig | None = None,
        variants_per_prompt: int = 1,
        seed_planner: VariantSeedPlanner | None = None,
    ) -> None:
        self.engine = engine
        self.dataset = dataset
        self.interval = interval_tokens
        self.min_samples = self._MIN_SAMPLES
        self.samples = max(self.min_samples, samples_per_cycle)
        self.next_threshold = interval_tokens
        self.evaluator = evaluator
        self.logger = logger
        self.quality_gate = quality_gate
        self.max_dataset_size = max_dataset_size if max_dataset_size and max_dataset_size > 0 else None
        self.decoder_cfg = decoder_cfg
        self.variants_per_prompt = max(1, variants_per_prompt)
        self.seed_planner = seed_planner

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
        run_inference_records(
            self.engine,
            self.evaluator,
            selections,
            label=f"{threshold} ingested tokens",
            user_id="trainer",
            agent_name="db-slm",
            logger=self.logger,
            quality_gate=self.quality_gate,
            decoder_cfg=self.decoder_cfg,
            variants_per_prompt=self.variants_per_prompt,
            seed_planner=self.seed_planner,
        )

    def refresh_dataset(self, new_records: Sequence[EvaluationRecord]) -> None:
        if not new_records:
            return
        if self.max_dataset_size is None or self.max_dataset_size <= 0:
            self.dataset.extend(new_records)
            return
        for record in new_records:
            if len(self.dataset) < self.max_dataset_size:
                self.dataset.append(record)
                continue
            replace_idx = random.randrange(self.max_dataset_size)
            self.dataset[replace_idx] = record

    def borrow_records(self, count: int) -> list[EvaluationRecord]:
        if count <= 0 or not self.dataset:
            return []
        sample_size = min(count, len(self.dataset))
        return random.sample(self.dataset, sample_size)


class IngestProfiler:
    """Optional profiler that logs ingest latency and current RSS."""

    def __init__(self, enabled: bool, logger: EvalLogWriter | None = None) -> None:
        self.enabled = enabled
        self.logger = logger

    def measure(self, label: str, fn: Callable[[], int]) -> int:
        if not self.enabled:
            return fn()
        rss_before = self._rss_mb()
        start = time.perf_counter()
        tokens = fn()
        duration = time.perf_counter() - start
        rss_after = self._rss_mb()
        rss_delta = (
            rss_after - rss_before if rss_before is not None and rss_after is not None else None
        )
        suffix = ""
        if rss_after is not None:
            delta_str = f"{rss_delta:+.1f}MB" if rss_delta is not None else "n/a"
            suffix = f" rss={rss_after:.1f}MB (Δ{delta_str})"
        log(f"[profile] {label}: {tokens} tokens in {duration:.2f}s{suffix}")
        if self.logger:
            self.logger.log_profile(
                label,
                tokens=tokens,
                duration=duration,
                rss_before=rss_before,
                rss_after=rss_after,
                rss_delta=rss_delta,
            )
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

    for field_name, cli_name in (
        ("decoder_presence_penalty", "--decoder-presence-penalty"),
        ("decoder_frequency_penalty", "--decoder-frequency-penalty"),
    ):
        value = getattr(args, field_name, None)
        if value is not None and value < 0:
            parser.error(f"{cli_name} must be >= 0 (got {value})")

    if not args.inputs and not args.stdin:
        parser.error("Provide at least one input path or enable --stdin")

    if args.seed is not None:
        random.seed(args.seed)
        log(f"[seed] Trainer RNG initialized with seed={args.seed}")

    if args.eval_seed is not None and args.eval_seed < 0:
        parser.error("--eval-seed must be >= 0 when provided.")

    try:
        context_dimensions = parse_context_dimensions_arg(
            args.context_dimensions,
            default=DEFAULT_CONTEXT_DIMENSIONS,
        )
    except ValueError as exc:
        parser.error(str(exc))

    db_path_str, db_path = resolve_db_path(args.db, args.reset)
    if args.reset:
        reset_cheetah_store(settings)
    metrics_path = resolve_metrics_export_path(args.metrics_export)
    metrics_writer: EvalLogWriter | None = None
    run_metadata: dict[str, Any] | None = None
    if metrics_path is not None:
        arg_snapshot = vars(args).copy()
        run_metadata = {
            "args": arg_snapshot,
            "db_path": db_path_str,
            "pid": os.getpid(),
            "metrics_file": str(metrics_path),
        }
        metrics_writer = EvalLogWriter(metrics_path, run_metadata)
        log(f"[metrics] Exporting evaluation timeline to {metrics_writer.path}")
    quality_gate: QualityGate | None = None
    if getattr(settings, "quality_queue_path", None):
        quality_gate = QualityGate(settings.quality_queue_path)
    engine = DBSLMEngine(
        db_path_str,
        ngram_order=args.ngram_order,
        context_dimensions=context_dimensions,
        settings=settings,
    )
    cheetah_primary = settings.backend == "cheetah-db" and not settings.cheetah_mirror
    if cheetah_primary and isinstance(engine.hot_path, NullHotPathAdapter):
        if not args.backonsqlite:
            engine.db.close()
            parser.error(
                "cheetah-db backend unreachable. Start cheetah-db/cheetah-server or rerun with --backonsqlite "
                "to allow the SQLite fallback."
            )
        log("[train] Warning: cheetah-db backend unreachable; proceeding on SQLite because --backonsqlite was set.")
    describe_adapter = getattr(engine.hot_path, "describe", None)
    if callable(describe_adapter):
        log(f"[train] Hot-path adapter active -> {describe_adapter()}")
    dims_label = format_context_dimensions(engine.context_dimensions)
    log(f"[train] Context dimensions: {dims_label}")
    if run_metadata is not None:
        run_metadata["context_dimensions"] = dims_label
    if args.eval_variants is not None:
        if args.eval_variants < 1:
            parser.error("--eval-variants must be >= 1")
        eval_variants = args.eval_variants
    else:
        eval_variants = 2 if engine.context_dimensions else 1
    seed_planner = VariantSeedPlanner(args.eval_seed)
    log(f"[seed] Evaluation RNG base seed={seed_planner.base_seed}")
    decoder_cfg_override: DecoderConfig | None = None
    decoder_overrides: dict[str, float] = {}
    if args.decoder_presence_penalty is not None:
        decoder_overrides["presence_penalty"] = float(args.decoder_presence_penalty)
    if args.decoder_frequency_penalty is not None:
        decoder_overrides["frequency_penalty"] = float(args.decoder_frequency_penalty)
    if decoder_overrides:
        decoder_cfg_override = DecoderConfig(**decoder_overrides)

    file_inputs = collect_files(args.inputs, args.recursive)
    corpora_iter: Iterable[CorpusChunk] = iter_corpora(
        file_inputs,
        args.encoding,
        args.json_chunk_size,
        args.max_json_lines,
        args.chunk_eval_percent,
    )
    if args.stdin:
        stdin_payload = sys.stdin.read()
        if stdin_payload.strip():
            stdin_rows = max(1, len([line for line in stdin_payload.splitlines() if line.strip()]))
            corpora_iter = itertools.chain(
                corpora_iter,
                [CorpusChunk("<stdin>", stdin_payload, [], total_rows=stdin_rows, train_rows=stdin_rows)],
            )

    # We defer the "empty" validation until after attempting to iterate so JSON inputs
    # can stream chunks without pre-loading them into memory.

    eval_dataset_path = args.eval_dataset or settings.dataset_path
    evaluator = ResponseEvaluator(engine)
    eval_records: list[EvaluationRecord] = []
    monitor = InferenceMonitor(
        engine,
        eval_records,
        args.eval_interval,
        args.eval_samples,
        evaluator,
        logger=metrics_writer,
        quality_gate=quality_gate,
        max_dataset_size=args.eval_pool_size,
        decoder_cfg=decoder_cfg_override,
        variants_per_prompt=eval_variants,
        seed_planner=seed_planner,
    )
    if args.eval_interval > 0:
        dataset_path = Path(eval_dataset_path).expanduser()
        try:
            eval_records = load_eval_dataset(dataset_path, args.eval_pool_size)
            monitor = InferenceMonitor(
                engine,
                eval_records,
                args.eval_interval,
                args.eval_samples,
                evaluator,
                logger=metrics_writer,
                quality_gate=quality_gate,
                max_dataset_size=args.eval_pool_size,
                decoder_cfg=decoder_cfg_override,
                variants_per_prompt=eval_variants,
                seed_planner=seed_planner,
            )
            log(f"[eval] Loaded {len(eval_records)} held-out sample(s) from {dataset_path}.")
        except FileNotFoundError as exc:
            log(f"[eval] Warning: {exc}. Disabling evaluation probes.")
            monitor = InferenceMonitor(
                engine,
                [],
                0,
                args.eval_samples,
                evaluator,
                logger=metrics_writer,
                quality_gate=quality_gate,
                max_dataset_size=args.eval_pool_size,
                decoder_cfg=decoder_cfg_override,
                variants_per_prompt=eval_variants,
                seed_planner=seed_planner,
            )

    profiler = IngestProfiler(args.profile_ingest, metrics_writer)
    total_tokens = 0
    total_windows = 0
    processed_corpora = 0
    success = False
    try:
        log(f"[train] Starting ingest into {db_path_str} with order={engine.store.order}.")
        for chunk in corpora_iter:
            label = chunk.label
            corpus = chunk.train_text
            processed_corpora += 1
            log(
                f"[train] Processing {label} ({len(corpus)} bytes, "
                f"{chunk.train_rows}/{chunk.total_rows} training rows)..."
            )
            reporter = TrainingProgressPrinter(label, chunk.train_rows)
            token_count = profiler.measure(
                label,
                lambda text=corpus, rep=reporter: engine.train_from_text(text, progress_callback=rep),
            )
            if token_count == 0:
                log(f"[train] Skipping {label} (corpus too small for order={engine.store.order})")
                continue
            window = max(0, token_count - engine.store.order + 1)
            total_tokens += token_count
            total_windows += window
            log(f"[train] Ingested {label}: {token_count} tokens -> {window} n-grams")
            monitor.maybe_run(total_tokens)
            if chunk.eval_records:
                eval_batch = list(chunk.eval_records)
                min_batch = getattr(monitor, "min_samples", 2)
                if len(eval_batch) < min_batch:
                    needed = min_batch - len(eval_batch)
                    eval_batch.extend(monitor.borrow_records(needed))
                if len(eval_batch) < min_batch and eval_batch:
                    seed_records = list(eval_batch)
                    idx = 0
                    while len(eval_batch) < min_batch:
                        eval_batch.append(seed_records[idx % len(seed_records)])
                        idx += 1
                run_inference_records(
                    engine,
                    evaluator,
                    eval_batch,
                    label=f"{label} hold-out",
                    user_id="trainer",
                    agent_name="db-slm",
                    logger=metrics_writer,
                    quality_gate=quality_gate,
                    decoder_cfg=decoder_cfg_override,
                    variants_per_prompt=eval_variants,
                    seed_planner=seed_planner,
                )
                monitor.refresh_dataset(chunk.eval_records)
        if processed_corpora == 0:
            parser.error("No readable corpora found in the provided inputs")

        if total_windows == 0:
            parser.error(
                "No usable training windows were produced. Provide larger corpora or reduce --ngram-order."
            )

        location = db_path if db_path is not None else db_path_str
        log(
            f"[train] Completed ingest: {total_tokens} tokens / {total_windows} n-grams stored in {location}"
        )
        ratio = engine.cheetah_topk_ratio()
        if ratio:
            log(f"[train] cheetah Top-K hit ratio ~ {ratio:.2%}")
        success = True
    finally:
        engine.db.close()
        if metrics_writer:
            metrics_writer.finalize(
                totals={
                    "tokens": total_tokens,
                    "windows": total_windows,
                    "processed_corpora": processed_corpora,
                    "db_path": db_path_str if db_path is None else str(db_path),
                },
                status="success" if success else "aborted",
            )


if __name__ == "__main__":
    main()
