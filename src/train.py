from __future__ import annotations

import argparse
import concurrent.futures
import itertools
import json
import math
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
from db_slm.adapters.cheetah import CheetahClient, CheetahFatalError
from db_slm.context_dimensions import (
    DEFAULT_CONTEXT_DIMENSIONS,
    format_context_dimensions,
    parse_context_dimensions_arg,
)
from db_slm.dataset_config import DatasetConfig, load_dataset_config
from db_slm.decoder import DecoderConfig
from db_slm.evaluation import (
    EvalLogWriter,
    DependencyLayer,
    EvaluationRecord,
    EvaluationSampleResult,
    QualityGate,
    ResponseEvaluator,
    VariantSeedPlanner,
    build_dependency_layer,
    run_inference_records,
    ContextProbabilityProbe,
)
from db_slm.settings import DBSLMSettings, load_settings
from db_slm.text_markers import append_end_marker
from db_slm.prompt_tags import ensure_response_prompt_tag

from helpers.resource_monitor import ResourceMonitor
from helpers.cheetah_cli import (
    collect_namespace_summary_lines,
    collect_system_stats_lines,
    format_prediction_query,
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
        "--merge-max-tokens",
        type=int,
        default=5,
        help=(
            "Maximum number of consecutive tokens to merge into a composite token when --ngram-order >= 5. "
            "Set to 0 to disable (default: %(default)s)."
        ),
    )
    parser.add_argument(
        "--merge-recursion-depth",
        type=int,
        default=None,
        help=(
            "Number of recursive merge passes to attempt per tokenization step. "
            "Defaults to 2 when merging is enabled."
        ),
    )
    parser.add_argument(
        "--merge-train-baseline",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Train the unmerged token sequence alongside merged tokens. "
            "Defaults to enabled when merging is active."
        ),
    )
    parser.add_argument(
        "--merge-eval-baseline",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Compute evaluation perplexity with merging disabled for comparison. "
            "Defaults to enabled when merging is active."
        ),
    )
    parser.add_argument(
        "--merge-significance-threshold",
        type=float,
        default=0.0,
        help=(
            "Mark merge tokens as retired when applied/candidate ratio falls below this threshold "
            "(default: %(default)s)."
        ),
    )
    parser.add_argument(
        "--merge-significance-min-count",
        type=int,
        default=2,
        help=(
            "Minimum candidate occurrences before evaluating merge significance "
            "(default: %(default)s)."
        ),
    )
    parser.add_argument(
        "--merge-significance-cap",
        type=int,
        default=128,
        help="Maximum retired merge tokens stored in metadata (default: %(default)s).",
    )
    parser.add_argument(
        "--context-dimensions",
        help=(
            "Comma-separated token span ranges (e.g. '1-2,3-5') or progressive lengths like "
            "'8,12,16,22,32' or '16,24,32,48,64' used to group context penalties and build MiniLM "
            "context-window embeddings for the prediction context matrix. This is independent of "
            "--cheetah-* probes/training. Use presets 'default', 'deep', or 'shallow', or set "
            "'off' to disable."
        ),
    )
    parser.add_argument(
        "--context-window-train-windows",
        type=int,
        default=0,
        help=(
            "Cap on adaptive windows per dimension sampled during training for context embeddings "
            "(default: engine preset)."
        ),
    )
    parser.add_argument(
        "--context-window-infer-windows",
        type=int,
        default=0,
        help=(
            "Max windows per dimension sampled during inference/evaluation for context embeddings "
            "(default: engine preset)."
        ),
    )
    parser.add_argument(
        "--context-window-stride-ratio",
        type=float,
        default=0.0,
        help=(
            "Stride ratio for context window sampling (0.1-1.0, default: engine preset)."
        ),
    )
    parser.add_argument(
        "--context-window-depth",
        type=int,
        default=None,
        help=(
            "Extra hidden-layer depth tiers to apply when building context matrices "
            "(default: engine preset). Use 0 for legacy depth; negative values reduce depth."
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
    parser.add_argument(
        "--cheetah-context-probe",
        action="append",
        default=[],
        metavar="TEXT",
        help=(
            "Text snippet used to derive context-window embeddings (based on --context-dimensions) and "
            "issue a PREDICT_QUERY against the cheetah prediction table before ingest. "
            "Repeat to probe multiple snippets."
        ),
    )
    parser.add_argument(
        "--cheetah-predict-table",
        default="context_matrices",
        help="Prediction-table name used by --cheetah-context-probe (default: %(default)s).",
    )
    parser.add_argument(
        "--cheetah-predict-key",
        default="meta:context_dimension_embeddings",
        help="Prediction-table key used by --cheetah-context-probe (default: %(default)s).",
    )
    parser.add_argument(
        "--disable-cheetah-token-train",
        action="store_true",
        help="Skip training cheetah prediction tables during ingest (default: enabled when cheetah hot-path is active).",
    )
    parser.add_argument(
        "--cheetah-token-table",
        default="token_predictions",
        help="Prediction table used for token-level training/inference (default: %(default)s).",
    )
    parser.add_argument(
        "--cheetah-token-key",
        default="meta:token_predictions",
        help="Prediction key used for token-level training/inference (default: %(default)s).",
    )
    parser.add_argument(
        "--cheetah-token-max-tokens",
        type=int,
        default=24,
        help="Maximum number of response tokens mirrored into cheetah prediction training per sample (default: %(default)s).",
    )
    parser.add_argument(
        "--cheetah-token-learning-rate",
        type=float,
        default=0.05,
        help="Learning rate applied to cheetah PREDICT_TRAIN updates (default: %(default)s).",
    )
    parser.add_argument(
        "--cheetah-token-value-cap",
        type=int,
        default=8192,
        help="Maximum number of unique token entries to seed inside the cheetah prediction table during a run (default: %(default)s).",
    )
    parser.add_argument(
        "--cheetah-token-progress-interval",
        type=float,
        default=60.0,
        help="Seconds between cheetah prediction training progress logs (default: %(default)s).",
    )
    parser.add_argument(
        "--cheetah-token-weight",
        type=float,
        default=0.25,
        help="Blending weight applied to cheetah prediction outputs during decoding (default: %(default)s).",
    )
    parser.add_argument(
        "--disable-cheetah-adversarial-train",
        action="store_true",
        help="Skip adversarial prediction updates triggered by low-quality evaluation outputs (default: enabled).",
    )
    parser.add_argument(
        "--cheetah-adversarial-threshold",
        type=float,
        default=0.45,
        help="Quality-score floor that triggers adversarial prediction updates (default: %(default)s).",
    )
    parser.add_argument(
        "--cheetah-adversarial-max-negatives",
        type=int,
        default=6,
        help="Maximum negative tokens pushed per adversarial sample (default: %(default)s).",
    )
    parser.add_argument(
        "--cheetah-adversarial-learning-rate",
        type=float,
        default=None,
        help="Override learning rate for adversarial prediction updates (default: 60% of --cheetah-token-learning-rate).",
    )
    parser.add_argument(
        "--cheetah-eval-predict",
        action="store_true",
        help=(
            "When set, evaluation probes issue a cheetah PREDICT_QUERY using dependency/context text "
            "and log the resulting probabilities."
        ),
    )
    parser.add_argument(
        "--cheetah-eval-predict-table",
        default="context_matrices",
        help="Prediction-table name used during evaluation prediction probes (default: %(default)s).",
    )
    parser.add_argument(
        "--cheetah-eval-predict-key",
        default="meta:context_dimension_embeddings",
        help="Prediction-table key used during evaluation prediction probes (default: %(default)s).",
    )
    parser.add_argument(
        "--cheetah-eval-predict-source",
        choices=("dependency", "prompt", "response", "reference", "generated", "context"),
        default="dependency",
        help=(
            "Text source used to derive context matrices for evaluation prediction probes "
            "(default: %(default)s)."
        ),
    )
    parser.add_argument(
        "--cheetah-eval-predict-limit",
        type=int,
        default=3,
        help="Maximum prediction entries to log during evaluation prediction probes (default: %(default)s).",
    )
    return parser


def _wipe_sqlite_artifacts(path: Path) -> None:
    """Remove the SQLite database plus WAL/SHM companions when --reset is supplied."""
    artifacts = [path, Path(f"{path}-wal"), Path(f"{path}-shm")]
    for artifact in artifacts:
        try:
            artifact.unlink()
        except FileNotFoundError:
            continue
        except IsADirectoryError:
            continue


def resolve_db_path(raw: str, reset: bool) -> Tuple[str, Path | None]:
    if raw == ":memory:":
        if reset:
            raise ValueError("--reset cannot be combined with the in-memory database")
        return raw, None
    path = Path(raw).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    if reset:
        _wipe_sqlite_artifacts(path)
    return str(path), path


def _configure_context_windows(
    engine: DBSLMEngine,
    args: argparse.Namespace,
) -> dict[str, float | int] | None:
    manager = getattr(engine, "context_windows", None)
    if not manager or not manager.enabled():
        return None
    changed = False
    train_windows = int(getattr(args, "context_window_train_windows", 0) or 0)
    if train_windows > 0:
        manager.max_train_windows = max(1, train_windows)
        changed = True
    infer_windows = int(getattr(args, "context_window_infer_windows", 0) or 0)
    if infer_windows > 0:
        manager.max_infer_windows = max(1, infer_windows)
        changed = True
    stride_ratio = float(getattr(args, "context_window_stride_ratio", 0.0) or 0.0)
    if stride_ratio > 0:
        manager.extractor.stride_ratio = max(0.1, min(stride_ratio, 1.0))
        changed = True
    depth_bonus = getattr(args, "context_window_depth", None)
    if depth_bonus is not None:
        manager.depth_bonus = int(depth_bonus)
        changed = True
    if not changed:
        return None
    return {
        "stride_ratio": manager.extractor.stride_ratio,
        "max_train_windows": manager.max_train_windows,
        "max_infer_windows": manager.max_infer_windows,
        "depth_bonus": manager.depth_bonus,
    }


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
        idle_grace=settings.cheetah_idle_grace_seconds,
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


def _probe_context_predictions(engine: DBSLMEngine, args: argparse.Namespace) -> None:
    samples = [sample for sample in getattr(args, "cheetah_context_probe", []) if sample]
    if not samples:
        return
    if not engine.context_windows.enabled():
        log("[train] cheetah context probes skipped: context windows disabled.")
        return
    matrix_builder = getattr(engine.context_windows, "context_matrix_for_text", None)
    if not callable(matrix_builder):
        log("[train] cheetah context probes unavailable: context window manager missing context_matrix_for_text().")
        return
    predict_query = getattr(engine.hot_path, "predict_query", None)
    if not callable(predict_query):
        log("[train] cheetah context probes unavailable: adapter missing predict_query().")
        return
    table = (getattr(args, "cheetah_predict_table", None) or "context_matrices").strip()
    key = getattr(args, "cheetah_predict_key", None) or "meta:context_dimension_embeddings"
    log(
        f"[train] Running {len(samples)} cheetah context probe(s) via PREDICT_QUERY "
        f"(table={table}, key={key})."
    )
    for index, sample in enumerate(samples, 1):
        snippet = (sample or "").strip()
        label = snippet.splitlines()[0] if snippet else ""
        if len(label) > 48:
            label = f"{label[:45]}..."
        matrix = matrix_builder(sample)
        if not matrix:
            log(
                f"[train] cheetah context probe #{index}: unable to derive context matrix "
                f"from sample '{label or '<empty>'}'."
            )
            continue
        log(
            f"[train] cheetah context probe #{index}: extracted {len(matrix)} window vector(s) "
            f"from sample '{label or '<empty>'}'."
        )
        try:
            result = predict_query(
                key=key,
                context_matrix=matrix,
                table=table,
            )
        except Exception as exc:
            log(f"[train] cheetah context probe #{index}: prediction query failed ({exc}).")
            continue
        for line in format_prediction_query(result, label=f"probe#{index}:{label}"):
            log(f"[train] {line}")


def _build_eval_prediction_probe(
    engine: DBSLMEngine,
    args: argparse.Namespace,
) -> tuple[ContextProbabilityProbe | None, str | None]:
    enabled = getattr(args, "cheetah_eval_predict", False)
    if not enabled:
        return None, None
    table = (getattr(args, "cheetah_eval_predict_table", None) or "context_matrices").strip()
    key = getattr(args, "cheetah_eval_predict_key", None) or "meta:context_dimension_embeddings"
    source = (getattr(args, "cheetah_eval_predict_source", "dependency") or "dependency").strip().lower()
    limit = getattr(args, "cheetah_eval_predict_limit", 3)
    probe = ContextProbabilityProbe(
        engine,
        table=table,
        key=key,
        max_entries=max(1, int(limit)),
        log_prefix="[eval]",
    )
    if not probe.available():
        log("[train] cheetah eval prediction probes unavailable (adapter disabled or context windows off).")
        return None, None
    log(
        "[train] Evaluation prediction probes enabled via PREDICT_QUERY "
        f"(table={table}, key={key}, source={source})."
    )
    return probe, source or "dependency"


def _prediction_dependency_summary(layer: DependencyLayer | None, limit: int = 32) -> str:
    if not layer:
        return ""
    parts: list[str] = []
    for arc in layer.arcs[:limit]:
        lemma = (arc.lemma or arc.token or "").strip()
        head = (arc.head or "ROOT").strip()
        dep = (arc.dep or "").strip()
        if not lemma:
            continue
        parts.append(f"{lemma}->{head}/{dep}")
    if layer.strong_token_groups:
        grouped = []
        for bucket, tokens in layer.strong_token_groups.items():
            bucket_tokens = " ".join(tokens[:4])
            if bucket_tokens:
                grouped.append(f"{bucket}:{bucket_tokens}")
        if grouped:
            parts.append(" ".join(grouped))
    return " ".join(parts)


def _prediction_context_text(record: EvaluationRecord) -> str:
    segments: list[str] = []
    prompt = (record.prompt or "").strip()
    if prompt:
        segments.append(prompt)
    if record.context_tokens:
        ctx_line = " ".join(f"{key}:{value}" for key, value in record.context_tokens.items())
        if ctx_line:
            segments.append(ctx_line)
    prompt_summary = _prediction_dependency_summary(record.prompt_dependencies)
    if prompt_summary:
        segments.append(prompt_summary)
    response_summary = _prediction_dependency_summary(getattr(record, "response_dependencies", None))
    if response_summary:
        segments.append(response_summary)
    return "\n".join(segment for segment in segments if segment).strip()


def _encode_prediction_token(token_id: int) -> bytes:
    return token_id.to_bytes(4, "big", signed=False)


def _train_prediction_tables(
    engine: DBSLMEngine,
    records: Sequence[EvaluationRecord] | None,
    args: argparse.Namespace,
) -> None:
    if getattr(args, "disable_cheetah_token_train", False):
        log("[train] cheetah prediction training skipped: disabled via --disable-cheetah-token-train.")
        return
    if not records:
        log("[train] cheetah prediction training skipped: no evaluation records available.")
        return
    predict_train = getattr(engine.hot_path, "predict_train", None)
    predict_set = getattr(engine.hot_path, "predict_set", None)
    if (
        not callable(predict_train)
        or not callable(predict_set)
        or isinstance(engine.hot_path, NullHotPathAdapter)
    ):
        log("[train] cheetah prediction training unavailable: hot-path adapter disabled.")
        return
    if not engine.context_windows.enabled():
        log("[train] cheetah prediction training skipped: context windows disabled.")
        return
    record_list = list(records)
    table = (getattr(args, "cheetah_token_table", None) or "token_predictions").strip()
    key_label = (getattr(args, "cheetah_token_key", None) or "meta:token_predictions").strip()
    if not table or not key_label:
        log("[train] cheetah prediction training skipped: missing table/key configuration.")
        return
    key_bytes = key_label.encode("utf-8")
    max_tokens = max(1, int(getattr(args, "cheetah_token_max_tokens", 24)))
    learning_rate = float(getattr(args, "cheetah_token_learning_rate", 0.05))
    if learning_rate <= 0:
        learning_rate = 0.01
    value_cap = max(1, int(getattr(args, "cheetah_token_value_cap", 8192)))
    total_records = len(record_list)
    log(
        "[train] cheetah prediction training: staging "
        f"{total_records} record(s) for table '{table}' "
        f"(key='{key_label}', lr={learning_rate:.4f}, max_tokens={max_tokens}, value_cap={value_cap})."
    )
    seeded_tokens: set[int] = getattr(engine, "_prediction_seeded_tokens", set())
    updates = 0
    progress_interval = float(getattr(args, "cheetah_token_progress_interval", 60))
    if progress_interval <= 0:
        progress_interval = 60.0
    last_progress = time.monotonic()
    for index, record in enumerate(record_list, 1):
        context_text = _prediction_context_text(record)
        if not context_text:
            continue
        matrix = engine.context_windows.context_matrix_for_text(context_text)
        if not matrix:
            continue
        token_ids = engine.tokenizer.encode(record.response or "", add_special_tokens=False)
        if not token_ids:
            continue
        limit = min(max_tokens, len(token_ids))
        # De-duplicate within a single response to avoid over-weighting repeats of the same token.
        unique_tokens = list(dict.fromkeys(token_ids[:limit]))
        for token_id in unique_tokens:
            value_bytes = _encode_prediction_token(token_id)
            if token_id not in seeded_tokens:
                if len(seeded_tokens) >= value_cap:
                    continue
                if not predict_set(
                    key=key_bytes,
                    value=value_bytes,
                    probability=0.05,
                    table=table,
                ):
                    continue
                seeded_tokens.add(token_id)
            success = predict_train(
                key=key_bytes,
                target=value_bytes,
                context_matrix=matrix,
                learning_rate=learning_rate,
                table=table,
            )
            if success:
                updates += 1
        if index % 250 == 0 or (time.monotonic() - last_progress) >= progress_interval:
            log(
                "[train] cheetah prediction training progress: "
                f"{index}/{total_records} record(s) processed, "
                f"{updates} update(s) applied, {len(seeded_tokens)} token(s) seeded."
            )
            last_progress = time.monotonic()
    if updates:
        setattr(engine, "_prediction_seeded_tokens", seeded_tokens)
        log(
            f"[train] cheetah prediction training: applied {updates} update(s) to table '{table}'."
        )
    else:
        log(f"[train] cheetah prediction training: no eligible updates for table '{table}'.")


def _safe_metric(metrics: dict[str, Any], key: str) -> float | None:
    value = metrics.get(key)
    if isinstance(value, (int, float)) and math.isfinite(value):
        return float(value)
    return None


def _sample_to_record(sample: EvaluationSampleResult) -> EvaluationRecord:
    return EvaluationRecord(
        prompt=sample.prompt,
        response=sample.reference,
        context_tokens=sample.context_tokens,
        prompt_dependencies=getattr(sample, "prompt_dependencies", None),
        response_dependencies=getattr(sample, "response_dependencies", None),
        response_label="|RESPONSE|",
        prompt_tags=(),
    )


def _adversarial_context_text(sample: EvaluationSampleResult) -> str:
    record = _sample_to_record(sample)
    context_text = _prediction_context_text(record)
    reference = (sample.reference or "").strip()
    if len(reference) > 240:
        reference = f"{reference[:237]}..."
    parts = [context_text, reference]
    return "\n".join(part for part in parts if part).strip()


def _context_energy(matrix: Sequence[Sequence[float]] | None) -> float:
    if not matrix:
        return 0.0
    energies: list[float] = []
    for row in matrix:
        if not row:
            continue
        total = sum(abs(component) for component in row)
        energies.append(total / len(row))
    return max(energies) if energies else 0.0


class AdversarialTrainer:
    """Runs adversarial prediction updates against cheetah when evaluation samples look weak."""

    def __init__(self, engine: DBSLMEngine, args: argparse.Namespace) -> None:
        self.engine = engine
        self.enabled = not getattr(args, "disable_cheetah_adversarial_train", False)
        self._predict_train = getattr(engine.hot_path, "predict_train", None)
        self._predict_set = getattr(engine.hot_path, "predict_set", None)
        self._matrix_builder = getattr(engine.context_windows, "context_matrix_for_text", None)
        self.table = (getattr(args, "cheetah_token_table", None) or "token_predictions").strip()
        key_label = (getattr(args, "cheetah_token_key", None) or "meta:token_predictions").strip()
        self.key_bytes = key_label.encode("utf-8")
        base_lr = float(getattr(args, "cheetah_token_learning_rate", 0.05) or 0.05)
        override_lr = getattr(args, "cheetah_adversarial_learning_rate", None)
        self.learning_rate = base_lr * 0.6 if override_lr is None or override_lr <= 0 else float(override_lr)
        self.learning_rate = max(0.001, self.learning_rate)
        self.quality_floor = float(getattr(args, "cheetah_adversarial_threshold", 0.45))
        self.max_negatives = max(1, int(getattr(args, "cheetah_adversarial_max_negatives", 6)))
        self.max_tokens = max(1, int(getattr(args, "cheetah_token_max_tokens", 24)))
        self.value_cap = max(1, int(getattr(args, "cheetah_token_value_cap", 8192)))
        self.seeded_tokens: set[int] = getattr(engine, "_prediction_seeded_tokens", set())
        self._predictable = (
            self.enabled
            and not isinstance(engine.hot_path, NullHotPathAdapter)
            and callable(self._predict_train)
            and engine.context_windows.enabled()
            and callable(self._matrix_builder)
        )
        if not self.enabled:
            log("[adv] cheetah adversarial corrections disabled via flag.")
            return
        if not self._predictable:
            log("[adv] cheetah adversarial corrections unavailable: hot-path adapter or context windows inactive.")
            self.enabled = False
            return
        if not self.table or not self.key_bytes:
            log("[adv] cheetah adversarial corrections skipped: missing token table/key configuration.")
            self.enabled = False
            return
        log(
            "[adv] Adversarial cheetah updates enabled "
            f"(table={self.table}, key={key_label}, lr={self.learning_rate:.4f}, "
            f"quality_floor={self.quality_floor:.2f}, max_negatives={self.max_negatives})."
        )

    def apply(self, samples: Sequence[EvaluationSampleResult], *, label: str) -> None:
        if not self.enabled or not samples:
            return
        updates = 0
        touched = 0
        max_energy = 0.0
        for sample in samples:
            if not self._should_update(sample):
                continue
            context_text = _adversarial_context_text(sample)
            if not context_text:
                continue
            matrix = self._matrix_builder(context_text) if callable(self._matrix_builder) else None
            if not matrix:
                continue
            max_energy = max(max_energy, _context_energy(matrix))
            positives = self._ensure_seeded(self._token_ids(sample.reference, self.max_tokens))
            negatives = self._ensure_seeded(self._token_ids(sample.generated, self.max_negatives))
            if not positives:
                continue
            touched += 1
            negatives = [token_id for token_id in negatives if token_id not in positives]
            encoded_negatives = [_encode_prediction_token(token_id) for token_id in negatives[: self.max_negatives]]
            for token_id in positives:
                success = self._predict_train(
                    key=self.key_bytes,
                    target=_encode_prediction_token(token_id),
                    context_matrix=matrix,
                    learning_rate=self.learning_rate,
                    table=self.table,
                    negatives=encoded_negatives,
                )
                if success:
                    updates += 1
        if updates:
            setattr(self.engine, "_prediction_seeded_tokens", self.seeded_tokens)
            log(
                "[adv] {label}: applied {updates} adversarial update(s) across {touched} sample(s) "
                "(lr={lr:.4f}, max_negatives={neg}, quality_floor={floor:.2f}, max_ctx_energy={energy:.3f}).".format(
                    label=label,
                    updates=updates,
                    touched=touched,
                    lr=self.learning_rate,
                    neg=self.max_negatives,
                    floor=self.quality_floor,
                    energy=max_energy,
                )
            )

    def _should_update(self, sample: EvaluationSampleResult) -> bool:
        metrics = sample.metrics or {}
        quality = _safe_metric(metrics, "quality_score")
        semantic = _safe_metric(metrics, "semantic_similarity")
        lexical = _safe_metric(metrics, "lexical")
        if sample.flagged:
            return True
        if quality is not None and quality < self.quality_floor:
            return True
        if semantic is not None and semantic < 0.5 and (lexical is None or lexical < 0.5):
            return True
        return False

    def _token_ids(self, text: str, limit: int) -> list[int]:
        tokens = self.engine.tokenizer.encode(text or "", add_special_tokens=False)
        if not tokens:
            return []
        unique: list[int] = []
        seen: set[int] = set()
        for token_id in tokens[:limit]:
            if token_id in seen:
                continue
            unique.append(token_id)
            seen.add(token_id)
        return unique

    def _ensure_seeded(self, token_ids: Sequence[int]) -> list[int]:
        ensured: list[int] = []
        for token_id in token_ids:
            if token_id in self.seeded_tokens:
                ensured.append(token_id)
                continue
            if len(self.seeded_tokens) >= self.value_cap:
                continue
            if callable(self._predict_set):
                ok = self._predict_set(
                    key=self.key_bytes,
                    value=_encode_prediction_token(token_id),
                    probability=0.05,
                    table=self.table,
                )
                if not ok:
                    continue
            self.seeded_tokens.add(token_id)
            ensured.append(token_id)
        return ensured


def _build_adversarial_trainer(engine: DBSLMEngine, args: argparse.Namespace) -> AdversarialTrainer | None:
    trainer = AdversarialTrainer(engine, args)
    if not trainer.enabled:
        return None
    return trainer


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


_RESUME_STATE_PATH = Path("var/train_resume.json")
_RESUME_STATE_VERSION = 1


def _utc_timestamp() -> str:
    return datetime.utcnow().isoformat(timespec="seconds") + "Z"


def _load_resume_state(path: Path = _RESUME_STATE_PATH) -> dict[str, Any] | None:
    try:
        raw = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return None
    except Exception as exc:
        log(f"[resume] Warning: unable to read {path}: {exc}")
        return None
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        log(f"[resume] Warning: invalid resume state {path}: {exc}")
        return None
    if not isinstance(payload, dict):
        log(f"[resume] Warning: malformed resume state {path}")
        return None
    return payload


def _write_resume_state(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(".tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    tmp_path.replace(path)


def _persist_resume_state(state: dict[str, Any]) -> None:
    state["updated_at"] = _utc_timestamp()
    _write_resume_state(_RESUME_STATE_PATH, state)


def _fingerprint_input(path: Path) -> dict[str, int]:
    stat = path.stat()
    return {"size": int(stat.st_size), "mtime": int(stat.st_mtime)}


def _collect_input_fingerprints(paths: Sequence[Path]) -> dict[str, dict[str, int]]:
    return {str(path): _fingerprint_input(path) for path in paths}


def _validate_resume_inputs(
    paths: Sequence[Path],
    fingerprints: dict[str, dict[str, int]] | None,
) -> list[str]:
    if not fingerprints:
        return []
    errors: list[str] = []
    for path in paths:
        key = str(path)
        if not path.exists():
            errors.append(f"missing: {path}")
            continue
        expected = fingerprints.get(key)
        if not expected:
            errors.append(f"untracked: {path}")
            continue
        actual = _fingerprint_input(path)
        if actual["size"] != expected.get("size") or actual["mtime"] != expected.get("mtime"):
            errors.append(f"changed: {path}")
    return errors


def _merge_resume_args(
    parser: argparse.ArgumentParser, saved_args: dict[str, Any] | None
) -> argparse.Namespace:
    defaults = parser.parse_args([])
    merged = vars(defaults)
    if saved_args:
        merged.update(saved_args)
    return argparse.Namespace(**merged)


def _build_resume_state(
    args: argparse.Namespace,
    file_inputs: Sequence[Path],
    metrics_path: Path | None,
    *,
    prior_state: dict[str, Any] | None = None,
    resumed: bool = False,
) -> dict[str, Any]:
    now = _utc_timestamp()
    state = dict(prior_state) if prior_state else {}
    state["version"] = _RESUME_STATE_VERSION
    state["status"] = "running"
    state["started_at"] = state.get("started_at") or now
    if resumed:
        state["resumed_at"] = now
        state["resume_count"] = int(state.get("resume_count", 0)) + 1
    state["args"] = vars(args)
    state["expanded_inputs"] = [str(path) for path in file_inputs]
    state["input_fingerprints"] = _collect_input_fingerprints(file_inputs)
    state["metrics_file"] = str(metrics_path) if metrics_path else None
    state["db_path"] = args.db
    state.setdefault("completed_chunks", [])
    state.setdefault("current_chunk", None)
    return state


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


_JSON_SUFFIXES = {".json", ".ndjson"}


def collect_prompt_tag_tokens(paths: Sequence[Path], dataset_config_override: str | None) -> list[str]:
    tokens: list[str] = []
    seen: set[str] = set()
    for token in DatasetConfig.default().prompt_tag_tokens():
        if token and token not in seen:
            tokens.append(token)
            seen.add(token)
    for path in paths:
        if path.suffix.lower() not in _JSON_SUFFIXES:
            continue
        cfg = load_dataset_config(path, override=dataset_config_override)
        for token in cfg.prompt_tag_tokens():
            if token and token not in seen:
                tokens.append(token)
                seen.add(token)
    return tokens


def resolve_eval_dataset_path(
    args: argparse.Namespace,
    settings: DBSLMSettings,
    file_inputs: Sequence[Path],
) -> str | None:
    """Infer the dataset used to seed evaluation probes."""
    default_path = args.eval_dataset or settings.dataset_path
    if args.eval_interval <= 0:
        return default_path
    if args.eval_dataset:
        return args.eval_dataset
    json_inputs = [path for path in file_inputs if path.suffix.lower() in _JSON_SUFFIXES]
    if json_inputs:
        chosen = str(json_inputs[0])
        notice = (
            "[eval] No --eval-dataset provided; defaulting evaluation pool to training input "
            f"{chosen}."
        )
        if len(json_inputs) > 1:
            notice += f" Additional JSON inputs detected: {len(json_inputs) - 1}."
        log(notice)
        return chosen
    if default_path:
        log_verbose(
            3,
            "[eval:v3] No JSON inputs supplied; falling back to DBSLM_DATASET_PATH "
            f"{default_path} for evaluation.",
        )
    else:
        log("[eval] Warning: evaluation enabled but no dataset could be inferred from inputs.")
    return default_path


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
    prediction_records: list[EvaluationRecord] | None = None


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
        text = append_end_marker(path.read_text(encoding=encoding))
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


def _iter_json_records(path: Path) -> Iterable[tuple[int, str]]:
    """Yield raw JSON snippets (and their starting line numbers) from a file.

    Supports traditional NDJSON (one object per line) and JSON arrays where
    objects may be pretty-printed across multiple lines.
    """
    with path.open("r", encoding="utf-8") as handle:
        first_non_ws_char: str | None = None
        array_mode = False
        buffer: list[str] = []
        buffer_start_line: int | None = None
        brace_depth = 0
        in_string = False
        escape = False
        for line_no, raw_line in enumerate(handle, start=1):
            line = raw_line
            if first_non_ws_char is None:
                idx = 0
                while idx < len(raw_line) and (raw_line[idx].isspace() or raw_line[idx] == "\ufeff"):
                    idx += 1
                if idx >= len(raw_line):
                    continue
                first_non_ws_char = raw_line[idx]
                if first_non_ws_char == "[":
                    array_mode = True
                    line = raw_line[idx + 1 :]
                else:
                    line = raw_line
            if not array_mode:
                stripped = raw_line.strip()
                if stripped:
                    yield line_no, stripped
                continue
            idx = 0
            while idx < len(line):
                ch = line[idx]
                if buffer_start_line is None:
                    if ch.isspace() or ch == ",":
                        idx += 1
                        continue
                    if ch == "]":
                        idx += 1
                        continue
                    if ch == "{":
                        buffer_start_line = line_no
                        brace_depth = 1
                        in_string = False
                        escape = False
                        buffer = ["{"]
                        idx += 1
                        continue
                    idx += 1
                    continue
                buffer.append(ch)
                if escape:
                    escape = False
                elif ch == "\\" and in_string:
                    escape = True
                elif ch == '"' and not escape:
                    in_string = not in_string
                elif not in_string:
                    if ch == "{":
                        brace_depth += 1
                    elif ch == "}":
                        brace_depth -= 1
                        if brace_depth == 0 and buffer_start_line is not None:
                            json_text = "".join(buffer)
                            yield buffer_start_line, json_text
                            buffer = []
                            buffer_start_line = None
                idx += 1


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
    for line_no, json_text in _iter_json_records(path):
        if limit and consumed >= limit:
            break
        try:
            payload = json.loads(json_text)
        except json.JSONDecodeError as exc:
            log(f"[train] JSON ingest warning ({path} line {line_no}): {exc}")
            continue
        prompt_value = dataset_cfg.extract_prompt(payload)
        response = dataset_cfg.extract_response(payload)
        context_values = list(dataset_cfg.iter_context_values(payload))
        framed_prompt = dataset_cfg.compose_prompt(
            payload, raw_prompt=prompt_value, context_values=context_values
        )
        preface_contexts, trailing_contexts = dataset_cfg.partition_context_values(context_values)
        if not response:
            continue
        prompt_layer = build_dependency_layer(framed_prompt or prompt_value or "")
        response_layer = build_dependency_layer(response)
        segment_lines: list[str] = []
        prompt_tags = dataset_cfg.prompt_tag_labels()
        canonical_tag_prefix = lambda field: f"{field.canonical_tag}:" if field.canonical_tag else None
        for field, ctx_value in preface_contexts:
            if field.label:
                segment_lines.append(f"{field.label}: {ctx_value}")
            normalized = field.normalized_token(ctx_value)
            prefix = canonical_tag_prefix(field)
            if prefix:
                segment_lines.append(f"{prefix}{field.token}:{normalized}")
        if prompt_value:
            segment_lines.append(f"{dataset_cfg.prompt.label}: {prompt_value.strip()}")
        for field, ctx_value in trailing_contexts:
            if field.label:
                segment_lines.append(f"{field.label}: {ctx_value}")
            normalized = field.normalized_token(ctx_value)
            prefix = canonical_tag_prefix(field)
            if prefix:
                segment_lines.append(f"{prefix}{field.token}:{normalized}")
        response_line = f"{dataset_cfg.response.label}: {response.strip()}"
        response_line = append_end_marker(response_line)
        segment_lines.append(response_line)
        annotation = _dependency_layer_annotation(prompt_layer, response_layer)
        if annotation:
            segment_lines.append(f"DependencyLayer: {annotation}")
        segment = "\n".join(segment_lines)
        evaluation_prompt = ensure_response_prompt_tag(
            framed_prompt or prompt_value or "",
            dataset_cfg.response.label,
        )
        record = EvaluationRecord(
            prompt=evaluation_prompt,
            response=response,
            context_tokens=dataset_cfg.context_map(payload),
            prompt_dependencies=prompt_layer,
            response_dependencies=response_layer,
            response_label=dataset_cfg.response.label,
            prompt_tags=prompt_tags,
        )

        log_prompt = evaluation_prompt
        if len(log_prompt) > 160:
            log_prompt = f"{log_prompt[:157]}..."
        log(f"[train] Staged line #{line_no} (Prompt: {log_prompt})")

        entries.append((segment, record))
        consumed += 1
        if len(entries) >= chunk_size:
            chunk_index += 1
            yield _build_chunk(f"{path}#chunk{chunk_index}", entries, holdout_fraction)
            entries = []
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
    prediction_records = [record for _, record in entries]
    return CorpusChunk(
        label,
        train_text,
        holdout_records,
        total_rows=total_rows,
        train_rows=train_rows,
        prediction_records=prediction_records,
    )


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
    for line_no, json_text in _iter_json_records(path):
        try:
            payload = json.loads(json_text)
        except json.JSONDecodeError as exc:
            log(f"[eval] Skipping line {line_no}: {exc}")
            continue
        prompt_value = dataset_cfg.extract_prompt(payload)
        response = dataset_cfg.extract_response(payload)
        if not prompt_value or not response:
            continue
        context_values = list(dataset_cfg.iter_context_values(payload))
        framed_prompt = dataset_cfg.compose_prompt(
            payload, raw_prompt=prompt_value, context_values=context_values
        )
        prompt_layer = build_dependency_layer(framed_prompt or prompt_value)
        response_layer = build_dependency_layer(response)
        evaluation_prompt = ensure_response_prompt_tag(
            framed_prompt or prompt_value,
            dataset_cfg.response.label,
        )
        records.append(
            EvaluationRecord(
                prompt=evaluation_prompt,
                response=response,
                context_tokens=dataset_cfg.context_map(payload),
                prompt_dependencies=prompt_layer,
                response_dependencies=response_layer,
                response_label=dataset_cfg.response.label,
                prompt_tags=dataset_cfg.prompt_tag_labels(),
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
        prediction_probe: ContextProbabilityProbe | None = None,
        prediction_source: str | None = None,
        adversarial_trainer: "AdversarialTrainer | None" = None,
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
        self.prediction_probe = prediction_probe
        self.prediction_source = (prediction_source or "").strip().lower() if prediction_probe else None
        self.adversarial_trainer = adversarial_trainer
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
        results = run_inference_records(
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
            prediction_probe=self.prediction_probe,
            prediction_source=self.prediction_source,
        )
        if self.adversarial_trainer:
            self.adversarial_trainer.apply(results, label=f"{threshold} token eval")

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
    resume_state: dict[str, Any] | None = None
    resume_completed: set[str] = set()
    resume_completed_list: list[str] = []
    resume_origin_metrics: str | None = None
    resume_mode = False

    if not args.inputs and not args.stdin:
        if len(sys.argv) <= 1:
            resume_state = _load_resume_state()
            if not resume_state:
                parser.error("Provide at least one input path or enable --stdin (no resume state found).")
            if resume_state.get("status") == "success":
                parser.error("No interrupted run to resume; the last run completed successfully.")
            resume_origin_metrics = resume_state.get("metrics_file")
            saved_args = resume_state.get("args")
            if not isinstance(saved_args, dict):
                parser.error("Resume state is missing saved CLI arguments.")
            args = _merge_resume_args(parser, saved_args)
            if getattr(args, "stdin", False):
                parser.error("Cannot resume --stdin runs without explicit inputs.")
            expanded_inputs = resume_state.get("expanded_inputs")
            if not expanded_inputs:
                parser.error("Resume state is missing the expanded input list.")
            args.inputs = list(expanded_inputs)
            resume_completed_list = list(resume_state.get("completed_chunks") or [])
            resume_completed = set(resume_completed_list)
            current_chunk = resume_state.get("current_chunk")
            if current_chunk:
                resume_completed.discard(current_chunk)
                if current_chunk in resume_completed_list:
                    resume_completed_list.remove(current_chunk)
                resume_state["current_chunk"] = None
            resume_state["completed_chunks"] = resume_completed_list
            resume_mode = True
        else:
            parser.error("Provide at least one input path or enable --stdin")

    if resume_mode and getattr(args, "reset", False):
        log("[resume] Ignoring --reset from the interrupted run to preserve progress.")
        args.reset = False

    log_verbose(3, f"[train:v3] Parsed CLI arguments: {vars(args)}")

    for field_name, cli_name in (
        ("decoder_presence_penalty", "--decoder-presence-penalty"),
        ("decoder_frequency_penalty", "--decoder-frequency-penalty"),
    ):
        value = getattr(args, field_name, None)
        if value is not None and value < 0:
            parser.error(f"{cli_name} must be >= 0 (got {value})")
    if args.merge_max_tokens is not None and args.merge_max_tokens < 0:
        parser.error("--merge-max-tokens must be >= 0")
    if args.merge_recursion_depth is not None and args.merge_recursion_depth < 1:
        parser.error("--merge-recursion-depth must be >= 1")
    if args.merge_significance_threshold is not None and args.merge_significance_threshold < 0:
        parser.error("--merge-significance-threshold must be >= 0")
    if args.merge_significance_min_count is not None and args.merge_significance_min_count < 1:
        parser.error("--merge-significance-min-count must be >= 1")
    if args.merge_significance_cap is not None and args.merge_significance_cap < 1:
        parser.error("--merge-significance-cap must be >= 1")

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

    merge_max_tokens = max(0, int(args.merge_max_tokens or 0))
    if args.ngram_order < 5:
        merge_max_tokens = 0
    if args.merge_recursion_depth is not None:
        merge_recursion_depth = int(args.merge_recursion_depth)
    else:
        merge_recursion_depth = 2 if merge_max_tokens > 1 else 1
    if merge_max_tokens <= 1:
        merge_recursion_depth = 1
    if args.merge_train_baseline is None:
        merge_train_baseline = merge_max_tokens > 1
    else:
        merge_train_baseline = bool(args.merge_train_baseline)
    if args.merge_eval_baseline is None:
        merge_eval_baseline = merge_max_tokens > 1
    else:
        merge_eval_baseline = bool(args.merge_eval_baseline)
    if merge_max_tokens <= 1:
        merge_train_baseline = False
        merge_eval_baseline = False
    merge_significance_threshold = max(0.0, float(args.merge_significance_threshold or 0.0))
    merge_significance_min_count = max(1, int(args.merge_significance_min_count or 1))
    merge_significance_cap = max(1, int(args.merge_significance_cap or 1))
    args.merge_recursion_depth = merge_recursion_depth
    args.merge_train_baseline = merge_train_baseline
    args.merge_eval_baseline = merge_eval_baseline
    args.merge_significance_threshold = merge_significance_threshold
    args.merge_significance_min_count = merge_significance_min_count
    args.merge_significance_cap = merge_significance_cap
    log_verbose(
        3,
        "[train:v3] Token merge configuration: "
        f"max_tokens={merge_max_tokens}, recursion={merge_recursion_depth}, "
        f"baseline_train={merge_train_baseline}, baseline_eval={merge_eval_baseline}, "
        f"dynamic_threshold=avg-count, ngram_order={args.ngram_order}.",
    )
    if merge_significance_threshold > 0:
        log_verbose(
            3,
            "[train:v3] Token merge significance: "
            f"threshold={merge_significance_threshold}, "
            f"min_count={merge_significance_min_count}, "
            f"cap={merge_significance_cap}.",
        )

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
        prediction_table=args.cheetah_token_table,
        prediction_key=args.cheetah_token_key,
        prediction_weight=args.cheetah_token_weight,
        token_merge_max_tokens=merge_max_tokens,
        token_merge_recursion_depth=merge_recursion_depth,
        token_merge_baseline_train=merge_train_baseline,
        token_merge_baseline_eval=merge_eval_baseline,
        token_merge_significance_threshold=merge_significance_threshold,
        token_merge_significance_min_count=merge_significance_min_count,
        token_merge_significance_cap=merge_significance_cap,
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
    context_window_config = _configure_context_windows(engine, args)
    if context_window_config:
        log(
            "[train] Context window config: "
            "stride={stride:.2f}, train_windows={train}, infer_windows={infer}, depth_bonus={depth}.".format(
                stride=context_window_config["stride_ratio"],
                train=context_window_config["max_train_windows"],
                infer=context_window_config["max_infer_windows"],
                depth=context_window_config["depth_bonus"],
            )
        )
    if getattr(engine, "token_merge_max_tokens", 0) > 1:
        log(
            "[train] Token merging enabled "
            f"(max_span={engine.token_merge_max_tokens}, "
            f"recursion={engine.token_merge_recursion_depth}, "
            f"baseline_train={engine.token_merge_baseline_train}, "
            f"baseline_eval={engine.token_merge_baseline_eval}, "
            "threshold=ceil(avg span frequency))."
        )
        if engine.token_merge_significance_threshold > 0:
            log(
                "[train] Token merge significance threshold "
                f"{engine.token_merge_significance_threshold:.3f} "
                f"(min_count={engine.token_merge_significance_min_count}, "
                f"cap={engine.token_merge_significance_cap})."
            )
    context_window_label = engine.context_windows.describe()
    if context_window_label:
        log(f"[train] Context window embeddings: {context_window_label}")
    if run_metadata is not None:
        run_metadata["context_dimensions"] = dims_label
        if context_window_config:
            run_metadata["context_window_config"] = context_window_config
        if context_window_label:
            run_metadata["context_window_embeddings"] = context_window_label
    _emit_cheetah_reports(engine, args)
    _probe_context_predictions(engine, args)
    eval_prediction_probe, eval_prediction_source = _build_eval_prediction_probe(engine, args)
    adversarial_trainer = _build_adversarial_trainer(engine, args)
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
    file_inputs = [path.resolve() for path in file_inputs]
    if resume_mode and resume_state:
        resume_errors = _validate_resume_inputs(
            file_inputs,
            resume_state.get("input_fingerprints"),
        )
        if resume_errors:
            formatted = "\n".join(f"- {message}" for message in resume_errors)
            parser.error(f"Resume inputs have changed:\n{formatted}")
    log_verbose(3, f"[train:v3] Prepared {len(file_inputs)} file input(s) for ingestion.")
    prompt_tag_tokens = collect_prompt_tag_tokens(file_inputs, args.dataset_config)
    resume_state = _build_resume_state(
        args,
        file_inputs,
        metrics_path,
        prior_state=resume_state,
        resumed=resume_mode,
    )
    resume_completed_list = list(resume_state.get("completed_chunks") or [])
    resume_completed = set(resume_completed_list)
    _persist_resume_state(resume_state)
    if resume_mode:
        log(f"[resume] Resuming training with {len(resume_completed)} completed chunk(s) skipped.")
        if run_metadata is not None:
            run_metadata["resume_from"] = {
                "started_at": resume_state.get("started_at"),
                "metrics_file": resume_origin_metrics,
                "completed_chunks": len(resume_completed),
            }
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
                [
                    CorpusChunk(
                        "<stdin>",
                        append_end_marker(stdin_payload),
                        [],
                        total_rows=stdin_rows,
                        train_rows=stdin_rows,
                    )
                ],
            )
            log_verbose(3, f"[train:v3] STDIN payload appended ({stdin_rows} rows).")

    # We defer the "empty" validation until after attempting to iterate so JSON inputs
    # can stream chunks without pre-loading them into memory.

    eval_dataset_path = resolve_eval_dataset_path(args, settings, file_inputs)
    if eval_dataset_path:
        eval_tag_tokens = collect_prompt_tag_tokens(
            [Path(eval_dataset_path).expanduser()],
            args.eval_dataset_config or args.dataset_config,
        )
        for token in eval_tag_tokens:
            if token not in prompt_tag_tokens:
                prompt_tag_tokens.append(token)
    if prompt_tag_tokens:
        unique_tokens = list(dict.fromkeys(prompt_tag_tokens))
        engine.register_prompt_tags(unique_tokens)
        log(f"[train] Registered prompt tags -> {', '.join(unique_tokens)}")
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
        prediction_probe=eval_prediction_probe,
        prediction_source=eval_prediction_source,
        adversarial_trainer=adversarial_trainer,
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
                prediction_probe=eval_prediction_probe,
                prediction_source=eval_prediction_source,
                adversarial_trainer=adversarial_trainer,
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
                prediction_probe=eval_prediction_probe,
                prediction_source=eval_prediction_source,
                adversarial_trainer=adversarial_trainer,
            )

    profiler = IngestProfiler(args.profile_ingest, metrics_writer)
    total_tokens = 0
    total_windows = 0
    processed_corpora = 0
    skipped_corpora = 0
    success = False
    try:
        log(f"[train] Starting ingest into {db_path_str} with order={engine.store.order}.")
        for chunk in corpora_iter:
            label = chunk.label
            corpus = chunk.train_text
            if resume_completed and label in resume_completed:
                skipped_corpora += 1
                continue
            if resume_state:
                resume_state["current_chunk"] = label
                _persist_resume_state(resume_state)
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
                if resume_state:
                    if label not in resume_completed:
                        resume_completed.add(label)
                        resume_completed_list.append(label)
                        resume_state["completed_chunks"] = resume_completed_list
                    resume_state["current_chunk"] = None
                    _persist_resume_state(resume_state)
                continue
            window = max(0, token_count - engine.store.order + 1)
            total_tokens += token_count
            total_windows += window
            log(f"[train] Ingested {label}: {token_count} tokens -> {window} n-grams")
            merge_reporter = getattr(engine, "consume_merge_report", None)
            if callable(merge_reporter):
                report = merge_reporter()
                if report:
                    baseline_note = ""
                    if report.baseline_tokens:
                        baseline_note = f", baseline_tokens={report.baseline_tokens}"
                    log(
                        f"[merge] {label}: merged_tokens={report.merged_tokens}{baseline_note}, "
                        f"applied={report.applied_total}, candidates={report.candidate_total}, "
                        f"passes={report.passes}, retired+{report.retired_added} "
                        f"(total={report.retired_total})."
                    )
                    if report.top_applied:
                        previews = []
                        for token_text, count, ratio in report.top_applied:
                            preview = token_text
                            if len(preview) > 60:
                                preview = preview[:57] + "..."
                            previews.append(f"{preview}*{count}@{ratio:.2f}")
                        log_verbose(
                            3,
                            f"[merge] {label}: top merges -> {', '.join(previews)}",
                        )
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
                eval_results = run_inference_records(
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
                    prediction_probe=eval_prediction_probe,
                    prediction_source=eval_prediction_source,
                )
                if adversarial_trainer:
                    adversarial_trainer.apply(eval_results, label=f"{label} hold-out")
                monitor.refresh_dataset(chunk.eval_records)
            if chunk.prediction_records:
                _train_prediction_tables(engine, chunk.prediction_records, args)
            if resume_state:
                if label not in resume_completed:
                    resume_completed.add(label)
                    resume_completed_list.append(label)
                    resume_state["completed_chunks"] = resume_completed_list
                resume_state["current_chunk"] = None
                _persist_resume_state(resume_state)
        if processed_corpora == 0:
            if skipped_corpora > 0:
                log("[resume] No remaining corpora to ingest; training already up to date.")
                success = True
                return
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
        if resume_state:
            resume_state["status"] = "success" if success else "aborted"
            resume_state["current_chunk"] = None
            resume_state["totals"] = {
                "tokens": total_tokens,
                "windows": total_windows,
                "processed_corpora": processed_corpora,
                "skipped_corpora": skipped_corpora,
            }
            _persist_resume_state(resume_state)


if __name__ == "__main__":
    try:
        main()
    except CheetahFatalError as exc:
        log(
            "[train] Fatal cheetah failure: "
            f"{exc}. Consult cheetah-db/AI_REFERENCE.md and ensure the server is healthy."
        )
        sys.exit(1)
