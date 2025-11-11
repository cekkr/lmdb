#!/usr/bin/env python3.14
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

from db_slm.settings import load_settings


def _count_lines(path: Path) -> int:
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        return sum(1 for _ in handle)


def _timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")


def _parse_metrics(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        try:
            data = json.load(handle)
        except json.JSONDecodeError:
            return {}
    started = data.get("started_at")
    completed = data.get("completed_at")
    tokens = data.get("totals", {}).get("tokens")
    throughput = None
    if started and completed and tokens:
        try:
            start_ts = datetime.fromisoformat(started.replace("Z", "+00:00"))
            end_ts = datetime.fromisoformat(completed.replace("Z", "+00:00"))
            elapsed = max((end_ts - start_ts).total_seconds(), 0.001)
            throughput = float(tokens) / elapsed
        except ValueError:
            throughput = None
    return {
        "tokens": tokens,
        "started_at": started,
        "completed_at": completed,
        "tokens_per_second": throughput,
        "metrics_path": str(path),
    }


def build_command(
    settings,
    queue_path: Path,
    metrics_path: Path,
    args: argparse.Namespace,
) -> list[str]:
    cmd = [
        args.python,
        "src/train.py",
        str(queue_path),
        "--db",
        settings.sqlite_dsn(),
        "--ngram-order",
        str(args.ngram_order),
        "--json-chunk-size",
        str(args.chunk_size),
        "--eval-interval",
        "0",
        "--chunk-eval-percent",
        "0",
        "--profile-ingest",
        "--metrics-export",
        str(metrics_path),
    ]
    return cmd


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="Run the queue-drain preset when the retrain queue grows too large.")
    parser.add_argument(
        "--threshold",
        type=int,
        default=150,
        help="Run the drain when queue length exceeds this value (default: %(default)s).",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=200,
        help="Chunk size to reuse from the queue-drain preset (default: %(default)s).",
    )
    parser.add_argument(
        "--ngram-order",
        type=int,
        default=5,
        help="N-gram order for the retrain drain (default: %(default)s).",
    )
    parser.add_argument(
        "--python",
        default=os.environ.get("PYTHON_BIN", "python3.14"),
        help="Python executable to invoke (default: %(default)s).",
    )
    parser.add_argument(
        "--queue",
        help="Override the queue path; defaults to DBSLM_QUALITY_QUEUE_PATH.",
    )
    parser.add_argument(
        "--metrics",
        help="Optional metrics-export path. When omitted a timestamped file under var/eval_logs/ is used.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the planned command instead of executing it.",
    )
    args = parser.parse_args(argv)

    settings = load_settings()
    queue_path = Path(args.queue or settings.quality_queue_path)
    if not queue_path.exists():
        print(f"[drain] Queue file {queue_path} does not exist; nothing to do.", file=sys.stderr)
        return 0
    pending = _count_lines(queue_path)
    print(f"[drain] {pending} pending entries in {queue_path}")
    if pending <= args.threshold:
        print(f"[drain] Threshold {args.threshold} not reached; skipping drain.")
        return 0

    metrics_path = Path(args.metrics or f"var/eval_logs/train-queue-drain-{_timestamp()}.json")
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = build_command(settings, queue_path, metrics_path, args)
    env = os.environ.copy()
    env.setdefault("DBSLM_BACKEND", "cheetah-db")
    env.setdefault("DBSLM_CHEETAH_MIRROR", "0")
    if args.dry_run:
        print("[drain] dry-run command:", " ".join(cmd))
        return 0
    subprocess.run(cmd, check=True, env=env)
    stats = _parse_metrics(metrics_path)
    if stats:
        tokens = stats.get("tokens")
        rate = stats.get("tokens_per_second")
        print(
            f"[drain] Completed drain ({tokens} tokens) -> {metrics_path} "
            f"(~{rate:.2f} tokens/s)" if rate else f"[drain] Completed drain -> {metrics_path}"
        )
    else:
        print(f"[drain] Completed drain; metrics not available at {metrics_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
