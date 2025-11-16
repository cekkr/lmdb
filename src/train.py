from __future__ import annotations

import argparse
import concurrent.futures
import itertools
import json
import multiprocessing
import os
import random
import shutil
import sys
import time
from collections import deque
from datetime import datetime
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Iterable, List, Sequence, Tuple

from db_slm import DBSLMEngine
from db_slm.adapters.base import NullHotPathAdapter
from db_slm.adapters.cheetah import CheetahClient
from db_slm.context_dimensions import (
    DEFAULT_CONTEXT_DIMENSIONS,
    format_context_dimensions,
    parse_context_dimensions_arg,
)
from db_slm.dataset_config import load_dataset_config
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
from db_slm.text_markers import append_end_marker

from helpers.resource_monitor import ResourceMonitor
from helpers.cheetah_cli import (
    collect_namespace_summary_lines,
    collect_system_stats_lines,
)
if TYPE_CHECKING:
    from helpers.resource_monitor import ResourceDelta, ResourceSample
from log_helpers import log, log_verbose


def build_parser(default_db_path: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Ingest raw text corpora into the DB-SLM SQLite backing store."
    )
    parser.add_argument(
        "inputs",
        nargs="*",
        help="Text/JSON files or directories to ingest. Directories pull in *.txt files while JSON/NDJSON inputs honor dataset configs.",
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
        "--dataset-config",
        help=(
            "Path to a dataset config JSON applied to *.json/NDJSON corpora. "
            "Defaults to <dataset>.config.json or DBSLM_DATASET_CONFIG_PATH."
        ),
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
        "--eval-dataset-config",
        help=(
            "Override dataset metadata for --eval-dataset. "
            "Defaults to <dataset>.config.json or DBSLM_DATASET_CONFIG_PATH."
        ),
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
        "--prep-workers",
        type=int,
        default=0,
        help=(
            "Number of worker processes used to stage corpora (dependency parsing + chunking). "
            "Defaults to cpu_count()-1 when <= 0."
        ),
    )
    parser.add_argument(
        "--prep-prefetch",
        type=int,
        default=4,
        help="Maximum number of in-flight corpus staging jobs when --prep-workers > 1 (default: %(default)s).",
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
    parser.add_argument(
        "--cheetah-summary",
        action="append",
        default=[],
        metavar="PREFIX",
        help=(
            "Namespace prefix to summarize via cheetah's PAIR_SUMMARY command (e.g., 'ctx:', 'prob:2'). "
            "Repeat to capture multiple namespaces. Prefixes starting with 'x' are interpreted as hex."
        ),
    )
    parser.add_argument(
        "--cheetah-summary-depth",
        type=int,
        default=1,
        help="Relative depth for cheetah namespace summaries (default: %(default)s, use -1 for unlimited).",
    )
    parser.add_argument(
        "--cheetah-summary-branches",
        type=int,
        default=32,
        help="Maximum number of branch digests returned per cheetah summary (default: %(default)s).",
    )
    parser.add_argument(
        "--cheetah-system-stats",
        action="store_true",
        help="Log cheetah SYSTEM_STATS before training (CPU/memory hints plus recommended workers).",
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
_CHEETAH_PURGE_MIN_PAGE_SIZE = 64
_CHEETAH_PURGE_MAX_SCAN_FAILURES = 3
_CHEETAH_PURGE_RETRY_DELAY = 0.05
_CHEETAH_FAST_PURGE_PAGE_SIZE = 4096


def _retry_cheetah_delete(
    client: CheetahClient,
    key: int,
    *,
    attempts: int = 3,
) -> tuple[bool, str | None]:
    """Retry cheetah DELETE operations to absorb transient timeouts."""
    last_response: str | None = None
    for attempt in range(attempts):
        success, response = client.delete(key)
        last_response = response
        if success:
            return True, response
        if response:
            normalized = response.lower()
            if "already_deleted" in normalized or "not_found" in normalized:
                return True, response
        if attempt < attempts - 1:
            time.sleep(_CHEETAH_PURGE_RETRY_DELAY * (attempt + 1))
    return False, last_response


def _retry_cheetah_pair_del(
    client: CheetahClient,
    raw_value: bytes,
    *,
    attempts: int = 3,
) -> tuple[bool, str | None]:
    """Retry cheetah PAIR_DEL operations with transient error handling."""
    last_response: str | None = None
    for attempt in range(attempts):
        success, response = client.pair_del(raw_value)
        last_response = response
        if success:
            return True, response
        if response and "not_found" in response.lower():
            return True, response
        if attempt < attempts - 1:
            time.sleep(_CHEETAH_PURGE_RETRY_DELAY * (attempt + 1))
    return False, last_response


def _try_fast_cheetah_purge(
    client: CheetahClient,
    prefix: bytes,
) -> tuple[int | None, bool, str | None]:
    """Attempt to purge a namespace via cheetah's PAIR_PURGE command.

    Returns a tuple of (removed_count, disable_fast_path, error_message).
    """
    removed, response = client.pair_purge(prefix, limit=_CHEETAH_FAST_PURGE_PAGE_SIZE)
    if removed is not None:
        return removed, False, None
    message = response or "no response from server"
    if response and "unknown_command" in response.lower():
        return None, True, message
    return None, False, message


def _purge_cheetah_namespace(
    client: CheetahClient,
    prefix: bytes,
    *,
    page_size: int = 1024,
) -> int:
    """Remove all pair entries (and backing values) for the given namespace prefix."""
    namespace_label = prefix.decode("utf-8", "ignore").rstrip(":") or prefix.hex()
    log_verbose(3, f"[train:v3] Starting cheetah purge for '{namespace_label}' (page_size={page_size}).")
    removed = 0
    scan_warned = False
    delete_warned = False
    pair_warned = False
    cursor: bytes | None = None
    started = time.monotonic()
    progress_interval = max(page_size * 5, 5000)
    current_page_size = max(page_size, _CHEETAH_PURGE_MIN_PAGE_SIZE)
    scan_failures = 0
    shrink_logged = False
    while True:
        result = client.pair_scan(prefix=prefix, limit=current_page_size, cursor=cursor)
        if result is None:
            scan_failures += 1
            if current_page_size > _CHEETAH_PURGE_MIN_PAGE_SIZE:
                current_page_size = max(_CHEETAH_PURGE_MIN_PAGE_SIZE, current_page_size // 2)
                if not shrink_logged:
                    log(
                        f"[train] cheetah reset: reducing scan page size for "
                        f"'{namespace_label}' to {current_page_size} after a timeout."
                    )
                    shrink_logged = True
                continue
            if scan_failures < _CHEETAH_PURGE_MAX_SCAN_FAILURES:
                time.sleep(_CHEETAH_PURGE_RETRY_DELAY)
                continue
            if not scan_warned:
                log(f"[train] Warning: cheetah reset aborted while scanning '{namespace_label}'.")
                scan_warned = True
            break
        scan_failures = 0
        shrink_logged = False
        entries, cursor = result
        if not entries:
            break
        for raw_value, key in entries:
            deleted, response = _retry_cheetah_delete(client, key)
            if not deleted:
                if not delete_warned:
                    log(
                        f"[train] Warning: failed to delete cheetah key {key} in '{namespace_label}': {response}"
                    )
                    delete_warned = True
                continue
            removed += 1
            success, response = _retry_cheetah_pair_del(client, raw_value)
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
        idle_grace=max(settings.cheetah_timeout_seconds * 180.0, 60.0),
    )
    if not client.connect():
        log(
            "[train] Warning: --reset requested but cheetah-db is unreachable "
            f"(targets: {client.describe_targets()}; errors: {client.describe_failures()})."
        )
        return
    try:
        db_name = settings.cheetah_database or "default"
        reset_success, reset_response = client.reset_database(db_name)
        if reset_success:
            log(f"[train] cheetah reset: RESET_DB cleared database '{db_name}'.")
            return
        if reset_response and "unknown_command" in reset_response.lower():
            log(
                "[train] cheetah reset: RESET_DB unsupported on the connected server; "
                "falling back to namespace purge."
            )
        else:
            detail = reset_response or "no response from server"
            log(f"[train] Warning: RESET_DB '{db_name}' failed ({detail}); falling back to namespace purge.")
        total_removed = 0
        fast_disabled = False
        fast_notice_logged = False
        for prefix in _CHEETAH_PURGE_PREFIXES:
            label = prefix.decode("utf-8", "ignore").rstrip(":") or prefix.hex()
            removed = 0
            used_fast = False
            if not fast_disabled:
                fast_removed, disable_fast, fast_message = _try_fast_cheetah_purge(client, prefix)
                if fast_removed is not None:
                    removed = fast_removed
                    used_fast = True
                else:
                    if disable_fast:
                        fast_disabled = True
                        if not fast_notice_logged:
                            detail = f" ({fast_message})" if fast_message else ""
                            log(
                                "[train] cheetah reset: PAIR_PURGE unsupported on the connected server; "
                                f"falling back to incremental deletes{detail}."
                            )
                            fast_notice_logged = True
                    elif fast_message:
                        log(f"[train] Warning: fast cheetah purge for '{label}' failed: {fast_message}")
            if not used_fast:
                removed = _purge_cheetah_namespace(client, prefix)
            total_removed += removed
            if removed:
                log(f"[train] cheetah reset: cleared {removed} '{label}' mapping(s).")
        if total_removed == 0:
            log("[train] cheetah reset: no cached namespaces required clearing.")
        else:
            log(f"[train] cheetah reset: removed {total_removed} cached mapping(s) total.")
    finally:
        client.close()


def _emit_cheetah_reports(engine: DBSLMEngine, args: argparse.Namespace) -> None:
    if getattr(args, "cheetah_system_stats", False):
        for line in collect_system_stats_lines(engine.hot_path):
            log(f"[train] {line}")
    prefixes = getattr(args, "cheetah_summary", []) or []
    if prefixes:
        depth = getattr(args, "cheetah_summary_depth", 1)
        branch_limit = getattr(args, "cheetah_summary_branches", 32)
        for line in collect_namespace_summary_lines(
            engine.hot_path,
            prefixes,
            depth=depth,
            branch_limit=branch_limit,
        ):
            log(f"[train] {line}")


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
            log_verbose(3, f"[train:v3] Queued input file {path}")
            continue
        if path.is_dir():
            pattern = "**/*.txt" if recursive else "*.txt"
            for candidate in sorted(path.glob(pattern)):
                if candidate.is_file():
                    files.append(candidate)
                    log_verbose(3, f"[train:v3] Discovered input file {candidate}")
            continue
        raise FileNotFoundError(f"No such file or directory: {path}")
    return files


def _resolve_worker_count(requested: int | None) -> int:
    if requested is None or requested == 1:
        return 1
    if requested is not None and requested > 1:
        return requested
    cpu_total = os.cpu_count() or 1
    # Leave one core idle by default to avoid starving the OS / cheetah worker.
    return max(1, cpu_total - 1)


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
    dataset_config_path: str | None,
) -> Iterable[CorpusChunk]:
    holdout_fraction = max(0.0, min(chunk_eval_percent, 100.0)) / 100.0
    for path in paths:
        suffix = path.suffix.lower()
        if suffix in {".json", ".ndjson"}:
            yield from iter_json_chunks(
                path,
                json_chunk_size,
                max_json_lines,
                holdout_fraction,
                dataset_config_path,
            )
            continue
        text = path.read_text(encoding=encoding)
        row_count = max(1, len([line for line in text.splitlines() if line.strip()]))
        yield CorpusChunk(str(path), text, [], total_rows=row_count, train_rows=row_count)


def _prepare_corpus_chunks(
    path_str: str,
    encoding: str,
    json_chunk_size: int,
    max_json_lines: int,
    chunk_eval_percent: float,
    dataset_config_path: str | None,
) -> list[CorpusChunk]:
    path = Path(path_str)
    return list(
        iter_corpora(
            [path],
            encoding,
            json_chunk_size,
            max_json_lines,
            chunk_eval_percent,
            dataset_config_path,
        )
    )


def parallel_corpus_stream(
    paths: Sequence[Path],
    encoding: str,
    json_chunk_size: int,
    max_json_lines: int,
    chunk_eval_percent: float,
    workers: int,
    prefetch: int = 4,
    dataset_config_path: str | None = None,
) -> Iterable[CorpusChunk]:
    if workers <= 1 or len(paths) <= 1:
        yield from iter_corpora(
            paths,
            encoding,
            json_chunk_size,
            max_json_lines,
            chunk_eval_percent,
            dataset_config_path,
        )
        return
    max_inflight = max(1, prefetch)
    mp_ctx = multiprocessing.get_context("spawn")

    def _generator() -> Iterable[CorpusChunk]:
        pending: deque[tuple[str, concurrent.futures.Future[list[CorpusChunk]]]] = deque()
        path_iter = iter(paths)
        with concurrent.futures.ProcessPoolExecutor(max_workers=workers, mp_context=mp_ctx) as pool:
            while len(pending) < max_inflight:
                try:
                    path_obj = next(path_iter)
                except StopIteration:
                    break
                future = pool.submit(
                    _prepare_corpus_chunks,
                    str(path_obj),
                    encoding,
                    json_chunk_size,
                    max_json_lines,
                    chunk_eval_percent,
                    dataset_config_path,
                )
                pending.append((str(path_obj), future))
            while pending:
                path_label, future = pending.popleft()
                try:
                    for chunk in future.result():
                        yield chunk
                except Exception as exc:
                    raise RuntimeError(f"Failed to stage corpus {path_label}: {exc}") from exc
                while len(pending) < max_inflight:
                    try:
                        next_path = next(path_iter)
                    except StopIteration:
                        break
                    pending.append(
                        (
                            str(next_path),
                            pool.submit(
                                _prepare_corpus_chunks,
                                str(next_path),
                                encoding,
                                json_chunk_size,
                                max_json_lines,
                                chunk_eval_percent,
                                dataset_config_path,
                            ),
                        )
                    )

    yield from _generator()


def iter_json_chunks(
    path: Path,
    chunk_size: int,
    max_lines: int,
    holdout_fraction: float,
    dataset_config_override: str | None,
) -> Iterable[CorpusChunk]:
    dataset_cfg = load_dataset_config(path, override=dataset_config_override)
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
            prompt = dataset_cfg.extract_prompt(payload)
            response = dataset_cfg.extract_response(payload)
            context_values = list(dataset_cfg.iter_context_values(payload))
            if not response:
                continue
            prompt_layer = build_dependency_layer(prompt or "")
            response_layer = build_dependency_layer(response)
            segment_lines: list[str] = []
            if prompt:
                segment_lines.append(f"{dataset_cfg.prompt.label}: {prompt.strip()}")
            for field, ctx_value in context_values:
                if field.label:
                    segment_lines.append(f"{field.label}: {ctx_value}")
                normalized = field.normalized_token(ctx_value)
                segment_lines.append(f"|CTX|:{field.token}:{normalized}")
            response_line = f"{dataset_cfg.response.label}: {response.strip()}"
            response_line = append_end_marker(response_line)
            segment_lines.append(response_line)
            annotation = _dependency_layer_annotation(prompt_layer, response_layer)
            if annotation:
                segment_lines.append(f"DependencyLayer: {annotation}")
            segment = "\n".join(segment_lines)
            record = EvaluationRecord(
                prompt=prompt or "",
                response=response,
                context_tokens=dataset_cfg.context_map(payload),
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


def load_eval_dataset(
    path: Path,
    max_records: int | None = None,
    *,
    config_override: str | None = None,
) -> List[EvaluationRecord]:
    if not path.exists():
        raise FileNotFoundError(f"Evaluation dataset not found: {path}")
    dataset_cfg = load_dataset_config(path, override=config_override)
    records: list[EvaluationRecord] = []
    limit = max_records if max_records is not None and max_records > 0 else None
    log_verbose(
        3,
        f"[eval:v3] Loading evaluation dataset from {path} (limit={limit or 'all'} records).",
    )
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
            prompt = dataset_cfg.extract_prompt(payload)
            response = dataset_cfg.extract_response(payload)
            if not prompt or not response:
                continue
            prompt_layer = build_dependency_layer(prompt)
            response_layer = build_dependency_layer(response)
            records.append(
                EvaluationRecord(
                    prompt=prompt,
                    response=response,
                    context_tokens=dataset_cfg.context_map(payload),
                    prompt_dependencies=prompt_layer,
                    response_dependencies=response_layer,
                )
            )
            if limit is not None and len(records) >= limit:
                break
    if limit is not None:
        random.shuffle(records)
    log_verbose(3, f"[eval:v3] Loaded {len(records)} evaluation record(s) from {path}.")
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
        log_verbose(
            3,
            "[eval:v3] Inference monitor configured "
            f"(interval={self.interval}, samples_per_cycle={self.samples}, "
            f"variants_per_prompt={self.variants_per_prompt}, dataset_size={len(self.dataset)}, "
            f"max_dataset_size={self.max_dataset_size or 'unbounded'}).",
        )

    def enabled(self) -> bool:
        return self.interval > 0 and bool(self.dataset)

    def maybe_run(self, total_tokens: int) -> None:
        if not self.enabled() or total_tokens < self.next_threshold:
            return
        while total_tokens >= self.next_threshold:
            log_verbose(
                3,
                f"[eval:v3] Triggering evaluation cycle at threshold={self.next_threshold} "
                f"with dataset_size={len(self.dataset)} (total_tokens={total_tokens}).",
            )
            self._run_cycle(self.next_threshold)
            self.next_threshold += self.interval

    def _run_cycle(self, threshold: int) -> None:
        sample_size = min(len(self.dataset), self.samples)
        log_verbose(
            3,
            f"[eval:v3] Sampling {sample_size} record(s) for evaluation at {threshold} tokens "
            f"(available={len(self.dataset)}).",
        )
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
    """Optional profiler that logs ingest latency together with resource telemetry."""

    def __init__(self, enabled: bool, logger: EvalLogWriter | None = None) -> None:
        self.enabled = enabled
        self.logger = logger
        self.monitor = ResourceMonitor() if enabled else None

    def measure(self, label: str, fn: Callable[[], int]) -> int:
        if not self.enabled:
            return fn()
        before = self._snapshot()
        start = time.perf_counter()
        tokens = fn()
        duration = time.perf_counter() - start
        after = self._snapshot()
        delta = self._delta(before, after)
        suffix = ""
        if delta and self.monitor:
            suffix = f" {self.monitor.describe(delta)}"
        log(f"[profile] {label}: {tokens} tokens in {duration:.2f}s{suffix}")
        if self.logger:
            rss_before = before.rss_mb if before else None
            rss_after = after.rss_mb if after else None
            rss_delta = (
                rss_after - rss_before if rss_after is not None and rss_before is not None else None
            )
            resources = ResourceMonitor.to_event(delta) if delta else None
            self.logger.log_profile(
                label,
                tokens=tokens,
                duration=duration,
                rss_before=rss_before,
                rss_after=rss_after,
                rss_delta=rss_delta,
                resources=resources,
            )
        return tokens

    def _snapshot(self) -> ResourceSample | None:
        if not self.monitor:
            return None
        try:
            return self.monitor.snapshot()
        except Exception:
            return None

    def _delta(self, before: ResourceSample | None, after: ResourceSample | None) -> ResourceDelta | None:
        if not self.monitor or before is None or after is None:
            return None
        try:
            return self.monitor.delta(before, after)
        except Exception:
            return None


def _collect_requirement_errors() -> list[str]:
    errors: list[str] = []
    try:
        import language_tool_python  # type: ignore
    except Exception as exc:
        errors.append(
            "language_tool_python is not installed or cannot be imported "
            f"(pip install language-tool-python): {exc}"
        )
        return errors
    java_path = shutil.which("java")
    if not java_path:
        errors.append("Java runtime (java) is required by language_tool_python but was not found on PATH.")
        return errors
    try:
        tool = language_tool_python.LanguageTool("en-US")
    except Exception as exc:  # pragma: no cover - optional dependency probe
        errors.append(
            "language_tool_python failed to initialize (verify Java installation and JAVA_HOME): "
            f"{exc}"
        )
        return errors
    try:
        tool.close()
    except Exception:
        pass
    return errors


def main() -> None:
    settings = load_settings()
    parser = build_parser(settings.sqlite_dsn())
    args = parser.parse_args()
    log_verbose(3, f"[train:v3] Parsed CLI arguments: {vars(args)}")

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
    log_verbose(
        3,
        f"[train:v3] Context dims argument '{args.context_dimensions}' resolved to "
        f"{format_context_dimensions(context_dimensions)}.",
    )
    requirement_errors = _collect_requirement_errors()
    if requirement_errors:
        formatted = "\n".join(f"- {message}" for message in requirement_errors)
        parser.error(f"Missing training requirements:\n{formatted}")

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
    _emit_cheetah_reports(engine, args)
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
    log_verbose(3, f"[train:v3] Prepared {len(file_inputs)} file input(s) for ingestion.")
    prep_workers = _resolve_worker_count(args.prep_workers)
    prep_prefetch = max(1, args.prep_prefetch)
    if prep_workers > 1:
        log(
            f"[train] Staging corpora with {prep_workers} worker process(es) "
            f"(prefetch={prep_prefetch}, chunk_size={args.json_chunk_size})."
        )
    corpora_iter: Iterable[CorpusChunk] = parallel_corpus_stream(
        file_inputs,
        args.encoding,
        args.json_chunk_size,
        args.max_json_lines,
        args.chunk_eval_percent,
        workers=prep_workers,
        prefetch=prep_prefetch,
        dataset_config_path=args.dataset_config,
    )
    if args.stdin:
        stdin_payload = sys.stdin.read()
        if stdin_payload.strip():
            stdin_rows = max(1, len([line for line in stdin_payload.splitlines() if line.strip()]))
            corpora_iter = itertools.chain(
                corpora_iter,
                [CorpusChunk("<stdin>", stdin_payload, [], total_rows=stdin_rows, train_rows=stdin_rows)],
            )
            log_verbose(3, f"[train:v3] STDIN payload appended ({stdin_rows} rows).")

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
            eval_records = load_eval_dataset(
                dataset_path,
                args.eval_pool_size,
                config_override=args.eval_dataset_config or args.dataset_config,
            )
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
            log_verbose(
                3,
                f"[eval:v3] Evaluation pool seeded from {dataset_path} "
                f"(pool_size={len(eval_records)}, max_pool={args.eval_pool_size or 'unbounded'}).",
            )
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
                log_verbose(
                    3,
                    f"[eval:v3] Running hold-out evaluation for {label} with "
                    f"{len(eval_batch)} record(s) (borrowed_from_pool={len(eval_batch) - len(chunk.eval_records)}).",
                )
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
