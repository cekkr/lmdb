#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import signal
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

try:
    import resource  # type: ignore
except ImportError:  # pragma: no cover - Windows compatibility
    resource = None  # type: ignore


SCRIPT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = SCRIPT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from db_slm.adapters.cheetah import CheetahClient
from db_slm.settings import DBSLMSettings, load_settings
from log_helpers import log


DEFAULT_SCENARIOS: list[dict[str, Any]] = [
    {
        "name": "baseline_profiled",
        "description": "Legacy capped ingest with profiling plus coarse evaluation probes.",
        "db": "var/smoke-train-baseline.sqlite3",
        "cheetah_database": "smoke_baseline_profiled",
        "train_args": [
            "{dataset}",
            "--db",
            "{db}",
            "--reset",
            "--json-chunk-size",
            "120",
            "--max-json-lines",
            "400",
            "--eval-interval",
            "1500",
            "--eval-samples",
            "2",
            "--eval-pool-size",
            "40",
            "--profile-ingest",
        ],
        "run_args": [
            "--db",
            "{db}",
            "--prompt",
            "Summarize how the DB-SLM handles short validation runs.",
            "--user",
            "smoke-test",
            "--agent",
            "db-slm",
        ],
    },
    {
        "name": "penalty_sweep_holdout",
        "description": "Chunked ingest with chunk hold-outs and decoder penalty overrides.",
        "db": "var/smoke-train-penalty.sqlite3",
        "cheetah_database": "smoke_penalty_holdout",
        "train_args": [
            "{dataset}",
            "--db",
            "{db}",
            "--reset",
            "--json-chunk-size",
            "90",
            "--max-json-lines",
            "240",
            "--chunk-eval-percent",
            "0.15",
            "--eval-interval",
            "800",
            "--eval-samples",
            "3",
            "--eval-pool-size",
            "60",
            "--decoder-presence-penalty",
            "0.65",
            "--decoder-frequency-penalty",
            "0.35",
            "--context-dimensions",
            "1-2,2-4",
        ],
        "run_args": [
            "--db",
            "{db}",
            "--prompt",
            "Give three bullet points about how the penalty sweep behaved.",
            "--user",
            "smoke-penalty",
            "--agent",
            "db-slm",
        ],
    },
]


SMOKE_TRACKER_PATH = Path("var/smoke_train/active_runs.json")


class RunTracker:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._entries: dict[int, dict[str, Any]] = {}
        self._lock = threading.Lock()
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return
        for key, value in payload.items():
            if not isinstance(value, dict):
                continue
            try:
                pid = int(key)
            except ValueError:
                continue
            self._entries[pid] = value

    def _write(self) -> None:
        tmp_path = self.path.with_suffix(".tmp")
        tmp_path.write_text(json.dumps({str(pid): entry for pid, entry in self._entries.items()}, indent=2), encoding="utf-8")
        tmp_path.replace(self.path)

    def register(self, pid: int, label: str) -> None:
        with self._lock:
            self._entries[pid] = {"label": label, "started_at": time.time()}
            self._write()

    def deregister(self, pid: int) -> None:
        with self._lock:
            if pid in self._entries:
                self._entries.pop(pid, None)
                self._write()

    def cleanup_orphans(self) -> None:
        touched = False
        for pid, info in list(self._entries.items()):
            if pid == os.getpid():
                touched = True
                self._entries.pop(pid, None)
                continue
            if not self._is_running(pid):
                touched = True
                self._entries.pop(pid, None)
                continue
            label = info.get("label", "unknown")
            log(f"[smoke-train] Terminating lingering process {pid} ({label}).")
            try:
                os.kill(pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            except OSError:
                pass
            touched = True
            self._entries.pop(pid, None)
        if touched:
            self._write()

    @staticmethod
    def _is_running(pid: int) -> bool:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return False
        except OSError:
            return True
        return True


class TelemetryMonitor(threading.Thread):
    def __init__(
        self,
        settings: DBSLMSettings,
        interval: float = 15.0,
        queue_drain: "QueueDrainAutomation | None" = None,
    ) -> None:
        super().__init__(daemon=True)
        self.settings = settings
        self.interval = interval
        self._stop_event = threading.Event()
        self._queue_drain = queue_drain
        self._client = CheetahClient(
            settings.cheetah_host,
            settings.cheetah_port,
            database=settings.cheetah_database,
            timeout=settings.cheetah_timeout_seconds,
            idle_grace=settings.cheetah_idle_grace_seconds,
        )
        self._queue_path = Path(settings.quality_queue_path)

    def run(self) -> None:
        while not self._stop_event.wait(self.interval):
            snapshot = self._snapshot()
            self._log(snapshot)

    def stop(self) -> None:
        self._stop_event.set()
        self._client.close()
        self.join(timeout=0.5)

    def _snapshot(self) -> dict[str, Any]:
        latency, healthy = self._probe_latency()
        queue_depth = self._queue_depth()
        cpu_seconds, rss_mb = self._process_stats()
        return {
            "latency_ms": latency,
            "healthy": healthy,
            "queue_depth": queue_depth,
            "cpu_seconds": cpu_seconds,
            "rss_mb": rss_mb,
        }

    def _probe_latency(self) -> tuple[float | None, bool]:
        start = time.monotonic()
        try:
            result = self._client.pair_scan(limit=1)
        except Exception:
            return None, False
        elapsed = (time.monotonic() - start) * 1000.0
        return elapsed, result is not None

    def _queue_depth(self) -> int:
        if not self._queue_path.exists():
            return 0
        try:
            with self._queue_path.open("r", encoding="utf-8", errors="ignore") as handle:
                return sum(1 for _ in handle)
        except OSError:
            return 0

    def _process_stats(self) -> tuple[float | None, float | None]:
        if resource is None:
            return None, None
        usage = resource.getrusage(resource.RUSAGE_SELF)
        cpu_seconds = usage.ru_utime + usage.ru_stime
        rss = usage.ru_maxrss
        if sys.platform == "darwin":
            rss_mb = rss / (1024 * 1024)
        else:
            rss_mb = rss / 1024
        return cpu_seconds, rss_mb

    def _log(self, snapshot: dict[str, Any]) -> None:
        latency = snapshot["latency_ms"]
        healthy = snapshot["healthy"]
        queue_depth = snapshot["queue_depth"]
        cpu_seconds = snapshot["cpu_seconds"]
        rss_mb = snapshot["rss_mb"]
        log(
            "[telemetry]",
            f"cheetah_latency_ms={latency:.1f}" if latency is not None else "cheetah_latency_ms=unavailable",
            f"cheetah_healthy={int(healthy)}",
            f"queue_depth={queue_depth}",
            f"cpu_seconds={cpu_seconds:.2f}" if cpu_seconds is not None else "cpu_seconds=unknown",
            f"rss_mb={rss_mb:.1f}" if rss_mb is not None else "rss_mb=unknown",
        )
        if self._queue_drain:
            self._queue_drain.maybe_trigger(queue_depth)


class QueueDrainAutomation:
    """Automatically kicks off queue drains and mirrors their metrics into studies."""

    def __init__(
        self,
        *,
        python_bin: str,
        script_path: Path,
        metrics_dir: Path,
        benchmarks_path: Path,
        threshold: int,
        cooldown_seconds: float,
        chunk_size: int = 200,
        max_json_lines: int = 500,
        queue_cap: int = 200,
    ) -> None:
        self.python_bin = python_bin
        self.script_path = script_path
        self.metrics_dir = metrics_dir
        self.metrics_dir.mkdir(parents=True, exist_ok=True)
        self.benchmarks_path = benchmarks_path
        self.benchmarks_path.parent.mkdir(parents=True, exist_ok=True)
        self.threshold = max(1, threshold)
        self.cooldown_seconds = max(10.0, cooldown_seconds)
        self.chunk_size = max(1, chunk_size)
        self.max_json_lines = max(1, max_json_lines)
        self.queue_cap = max(1, queue_cap)
        self._lock = threading.Lock()
        self._active = False
        self._last_trigger = 0.0

    def maybe_trigger(self, queue_depth: int) -> None:
        if queue_depth < self.threshold:
            return
        with self._lock:
            now = time.time()
            if self._active:
                return
            if (now - self._last_trigger) < self.cooldown_seconds:
                return
            self._active = True
        thread = threading.Thread(target=self._run_once, args=(queue_depth,), daemon=True)
        thread.start()

    def _run_once(self, queue_depth: int) -> None:
        metrics_path = self.metrics_dir / f"train-queue-auto-{time.strftime('%Y%m%d-%H%M%S')}.json"
        command = [
            self.python_bin,
            str(self.script_path),
            "--threshold",
            str(self.threshold),
            "--chunk-size",
            str(self.chunk_size),
            "--max-json-lines",
            str(self.max_json_lines),
            "--queue-cap",
            str(self.queue_cap),
            "--metrics",
            str(metrics_path),
        ]
        env = os.environ.copy()
        src_path = str(SRC_ROOT)
        existing_pythonpath = env.get("PYTHONPATH")
        if existing_pythonpath and src_path not in existing_pythonpath.split(os.pathsep):
            env["PYTHONPATH"] = os.pathsep.join([src_path, existing_pythonpath])
        elif not existing_pythonpath:
            env["PYTHONPATH"] = src_path
        log(
            "[queue-drain]",
            f"trigger depth={queue_depth}",
            f"command={' '.join(shlex.quote(part) for part in command)}",
        )
        try:
            result = subprocess.run(
                command,
                cwd=str(SCRIPT_ROOT),
                env=env,
                capture_output=True,
                text=True,
                check=False,
            )
        except Exception as exc:  # pragma: no cover - defensive logging
            log("[queue-drain]", f"failed to start drain: {exc}")
            self._mark_finished()
            return
        if result.returncode != 0:
            log(
                "[queue-drain]",
                f"drain command failed (exit={result.returncode})",
                f"stdout={result.stdout.strip() or '<empty>'}",
                f"stderr={result.stderr.strip() or '<empty>'}",
            )
            self._mark_finished()
            return
        self._record_benchmark(queue_depth, metrics_path, command)
        self._mark_finished()

    def _mark_finished(self) -> None:
        with self._lock:
            self._active = False
            self._last_trigger = time.time()

    def _record_benchmark(self, queue_depth: int, metrics_path: Path, command: list[str]) -> None:
        if not metrics_path.exists():
            log("[queue-drain]", f"metrics not found: {metrics_path}")
            return
        try:
            payload = json.loads(metrics_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            log("[queue-drain]", f"invalid metrics JSON: {metrics_path}")
            return
        summary = self._summarize_metrics(payload)
        formatted_command = " ".join(shlex.quote(part) for part in command)
        heading = f"## {datetime.now(timezone.utc).strftime('%Y-%m-%d')} - Queue Drain (auto smoke harness)"
        lines = [
            "",
            heading,
            f"- Trigger: queue depth {queue_depth} exceeded {self.threshold}; command `{formatted_command}`.",
            f"- Metrics file: `{metrics_path.as_posix()}`; tokens={summary['tokens']} windows={summary['windows']} throughput={summary['throughput']}.",
            f"- Status: {summary['status']} (started {summary['started_at']} / completed {summary['completed_at']}).",
            "- Notes: appended automatically by `scripts/smoke_train.py` when the quality queue overflowed.",
            "",
        ]
        with self.benchmarks_path.open("a", encoding="utf-8") as handle:
            handle.write("\n".join(lines))
        log("[queue-drain]", f"recorded metrics -> {self.benchmarks_path}")

    def _summarize_metrics(self, payload: dict[str, Any]) -> dict[str, Any]:
        totals = payload.get("totals", {})
        tokens = totals.get("tokens", "unknown")
        windows = totals.get("windows", "unknown")
        started = payload.get("started_at")
        completed = payload.get("completed_at")
        throughput = "unknown"
        if started and completed and isinstance(tokens, (int, float)):
            try:
                start_ts = datetime.fromisoformat(str(started).replace("Z", "+00:00"))
                end_ts = datetime.fromisoformat(str(completed).replace("Z", "+00:00"))
                elapsed = max((end_ts - start_ts).total_seconds(), 0.001)
                throughput = f"{float(tokens) / elapsed:.2f} tokens/s"
            except Exception:
                throughput = "unknown"
        status = payload.get("status", "unknown")
        return {
            "tokens": tokens,
            "windows": windows,
            "throughput": throughput,
            "status": status,
            "started_at": started or "unknown",
            "completed_at": completed or "unknown",
        }

def timestamp() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


PROGRESS_RE = re.compile(
    r"\[train\]\s+(?P<label>[^:]+):\s+(?P<stage>.+?)\s+(?P<pct>\d+(?:\.\d+)?)%\s+\((?P<completed>\d+)/(?P<total>\d+)\)"
)
INGEST_RE = re.compile(
    r"Ingested\s+(?P<label>.+?):\s+(?P<tokens>\d+)\s+tokens\s+->\s+(?P<windows>\d+)\s+n-grams", re.IGNORECASE
)
PROFILE_RE = re.compile(
    r"\[profile\]\s+(?P<label>.+?):\s+(?P<tokens>\d+)\s+tokens\s+in\s+(?P<duration>[0-9.]+)s", re.IGNORECASE
)
COMPLETE_RE = re.compile(
    r"Completed ingest:\s+(?P<tokens>\d+)\s+tokens\s*/\s*(?P<windows>\d+)\s+n-grams", re.IGNORECASE
)


def replace_tokens(value: str, mapping: dict[str, str]) -> str:
    result = value
    for key, token in mapping.items():
        result = result.replace(f"{{{key}}}", token)
    return result


def render_args(args: Iterable[str], mapping: dict[str, str]) -> list[str]:
    return [replace_tokens(arg, mapping) for arg in args]


class BenchmarkRecorder:
    """Maintains a machine-readable description of active smoke-train runs."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.data: dict[str, Any] = {"updated_at": None, "scenarios": {}}
        self._last_write = 0.0
        if self.path.exists():
            try:
                self.data = json.loads(self.path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                self.data = {"updated_at": None, "scenarios": {}}

    def ensure_entry(self, scenario: "Scenario") -> None:
        scenarios = self.data.setdefault("scenarios", {})
        scenarios.setdefault(
            scenario.name,
            {
                "status": "pending",
                "db_path": scenario.db_path,
                "cheetah_database": scenario.cheetah_database,
                "description": scenario.description,
                "history": [],
            },
        )
        self._write(force=True)

    def update(self, scenario_name: str, **fields: Any) -> None:
        entry = self.data.setdefault("scenarios", {}).setdefault(scenario_name, {})
        entry.update(fields)
        entry["updated_at"] = timestamp()
        self.data["updated_at"] = entry["updated_at"]
        self._write()

    def capture_log(self, scenario_name: str, line: str) -> None:
        entry = self.data.setdefault("scenarios", {}).setdefault(scenario_name, {})
        entry["last_log"] = line
        log_tail = entry.setdefault("log_tail", [])
        log_tail.append(line)
        entry["log_tail"] = log_tail[-25:]
        live = entry.setdefault("live_metrics", {})
        self._parse_progress(line, live)
        self._parse_ingest(line, live)
        self._parse_profile(line, live)
        self._parse_completion(line, live)
        self.update(scenario_name, live_metrics=live, last_log=line, log_tail=entry["log_tail"])

    def finalize_metrics(self, scenario_name: str, metrics_path: Path) -> None:
        if not metrics_path.exists():
            return
        try:
            payload = json.loads(metrics_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return
        totals = payload.get("totals", {})
        summary = {}
        for event in payload.get("events", []):
            if event.get("type") != "evaluation":
                continue
            summary = event.get("summary") or {}
        record = {
            "ts": timestamp(),
            "totals": totals,
            "summary": summary,
            "metrics_file": str(metrics_path),
            "run_id": payload.get("run_id"),
            "status": payload.get("status"),
        }
        entry = self.data.setdefault("scenarios", {}).setdefault(scenario_name, {})
        history = entry.setdefault("history", [])
        already_recorded = any(
            item.get("run_id") == record["run_id"] and item.get("metrics_file") == record["metrics_file"]
            for item in history
        )
        if not already_recorded:
            history.append(record)
            entry["history"] = history[-20:]
        entry["latest_metrics"] = record
        self.update(scenario_name, latest_metrics=entry["latest_metrics"], history=entry["history"])

    def flush(self) -> None:
        self._write(force=True)

    def _write(self, *, force: bool = False) -> None:
        now = time.time()
        if not force and (now - self._last_write) < 0.3:
            return
        tmp_path = self.path.with_suffix(".tmp")
        tmp_path.write_text(json.dumps(self.data, indent=2), encoding="utf-8")
        tmp_path.replace(self.path)
        self._last_write = now

    @staticmethod
    def _parse_progress(line: str, live: dict[str, Any]) -> None:
        match = PROGRESS_RE.search(line)
        if not match:
            return
        live.setdefault("progress", {}).update(
            {
                "percent": float(match.group("pct")),
                "completed": int(match.group("completed")),
                "total": int(match.group("total")),
                "stage": match.group("stage"),
                "label": match.group("label"),
            }
        )

    @staticmethod
    def _parse_ingest(line: str, live: dict[str, Any]) -> None:
        match = INGEST_RE.search(line)
        if not match:
            return
        live["latest_ingest"] = {
            "tokens": int(match.group("tokens")),
            "windows": int(match.group("windows")),
            "label": match.group("label"),
        }

    @staticmethod
    def _parse_profile(line: str, live: dict[str, Any]) -> None:
        match = PROFILE_RE.search(line)
        if not match:
            return
        live["last_profile"] = {
            "label": match.group("label").strip(),
            "tokens": int(match.group("tokens")),
            "duration_sec": float(match.group("duration")),
        }

    @staticmethod
    def _parse_completion(line: str, live: dict[str, Any]) -> None:
        match = COMPLETE_RE.search(line)
        if not match:
            return
        live["totals"] = {
            "tokens": int(match.group("tokens")),
            "windows": int(match.group("windows")),
        }


@dataclass
class Scenario:
    name: str
    description: str
    db_path: str
    cheetah_database: str
    train_args: list[str]
    run_args: list[str]
    dataset: str
    metrics_path: Path


def load_scenarios(
    *,
    dataset_default: str,
    metrics_dir: Path,
    matrix_path: Path | None,
) -> list[Scenario]:
    raw = DEFAULT_SCENARIOS
    if matrix_path:
        payload = json.loads(matrix_path.read_text(encoding="utf-8"))
        raw = payload.get("scenarios", payload)
    scenarios: list[Scenario] = []
    for entry in raw:
        name = entry["name"]
        description = entry.get("description", "")
        db_path = entry.get("db") or entry.get("db_path") or f"var/{name}.sqlite3"
        cheetah_database = entry.get("cheetah_database", f"smoke_{name}")
        dataset = entry.get("dataset", dataset_default)
        train_args = entry.get("train_args") or []
        run_args = entry.get("run_args") or []
        metrics_path = metrics_dir / f"{name}.json"
        scenarios.append(
            Scenario(
                name=name,
                description=description,
                db_path=db_path,
                cheetah_database=cheetah_database,
                train_args=list(train_args),
                run_args=list(run_args),
                dataset=dataset,
                metrics_path=metrics_path,
            )
        )
    return scenarios


def build_command(python: str, script: str, args: list[str]) -> list[str]:
    return [python, script, *args]


def command_to_text(cmd: list[str]) -> str:
    return shlex.join(cmd)


def run_subprocess(
    cmd: list[str],
    *,
    env: dict[str, str],
    scenario: Scenario,
    recorder: BenchmarkRecorder,
    status_label: str,
    deadline: float | None = None,
    tracker: "RunTracker" | None = None,
) -> int:
    recorder.update(scenario.name, status=status_label)
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=env,
    )
    if tracker:
        tracker.register(process.pid, f"{scenario.name}:{status_label}")
    assert process.stdout is not None
    timed_out = False
    try:
        for line in process.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            recorder.capture_log(scenario.name, line.rstrip())
            if deadline and time.monotonic() > deadline:
                log(
                    f"[smoke-train] {scenario.name} {status_label} exceeded wall time limit; terminating."
                )
                process.send_signal(signal.SIGTERM)
                timed_out = True
                break
    except KeyboardInterrupt:
        log("[smoke-train] Interrupt received, terminating current command...")
        process.send_signal(signal.SIGINT)
        try:
            process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            process.kill()
        return 130
    finally:
        if tracker:
            tracker.deregister(process.pid)
    if timed_out:
        try:
            process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
        return 124
    return process.wait()


def scenario_mapping(scenario: Scenario) -> dict[str, str]:
    return {
        "db": scenario.db_path,
        "dataset": scenario.dataset,
        "metrics": str(scenario.metrics_path),
        "scenario": scenario.name,
        "cheetah_db": scenario.cheetah_database,
    }


def ensure_metrics_arg(args: list[str], metrics_path: Path) -> list[str]:
    if any(arg == "--metrics-export" for arg in args):
        return args
    return [*args, "--metrics-export", str(metrics_path)]


def select_scenarios(
    scenarios: list[Scenario],
    *,
    only: set[str] | None,
    resume_from: str | None,
) -> list[Scenario]:
    ordered = []
    resume = resume_from is None
    for scenario in scenarios:
        if not resume:
            if scenario.name == resume_from:
                resume = True
            else:
                continue
        if only and scenario.name not in only:
            continue
        ordered.append(scenario)
    return ordered


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run sequential smoke-train scenarios with live benchmarks."
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python interpreter to use (default: current).",
    )
    parser.add_argument(
        "--dataset",
        default="datasets/emotion_data.json",
        help="Default dataset path to inject into scenarios.",
    )
    parser.add_argument(
        "--metrics-dir",
        default="var/smoke_train/metrics",
        help="Directory to store per-scenario metrics exports.",
    )
    parser.add_argument(
        "--benchmarks",
        default="var/smoke_train/benchmarks.json",
        help="Path to the aggregated benchmark JSON.",
    )
    parser.add_argument(
        "--wall-time-limit",
        type=int,
        default=1800,
        help="Wall-clock limit per command in seconds (default 1800s).",
    )
    parser.add_argument(
        "--telemetry-interval",
        type=float,
        default=15.0,
        help="Seconds between cheetah telemetry snapshots (default %(default)s).",
    )
    parser.add_argument(
        "--queue-drain-threshold",
        type=int,
        default=175,
        help="Queue depth that triggers the automatic drain helper (default: %(default)s).",
    )
    parser.add_argument(
        "--queue-drain-cooldown",
        type=float,
        default=900.0,
        help="Minimum seconds between automated drains (default: %(default)s).",
    )
    parser.add_argument(
        "--queue-drain-script",
        default="scripts/drain_queue.py",
        help="Path to the drain helper script (default: %(default)s).",
    )
    parser.add_argument(
        "--queue-drain-metrics-dir",
        default="var/smoke_train/drains",
        help="Directory for automated queue-drain metrics exports.",
    )
    parser.add_argument(
        "--queue-drain-benchmarks",
        default="studies/BENCHMARKS.md",
        help="Markdown file that receives auto queue-drain entries (default: %(default)s).",
    )
    parser.add_argument(
        "--disable-auto-queue-drain",
        action="store_true",
        help="Disable automatic queue draining even when depth exceeds the threshold.",
    )
    parser.add_argument(
        "--matrix",
        type=Path,
        help="Optional JSON file describing the smoke-train scenario matrix.",
    )
    parser.add_argument(
        "--scenarios",
        help="Comma-separated list of scenario names to run (default: all).",
    )
    parser.add_argument(
        "--resume-from",
        help="Skip scenarios until the provided name is reached.",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop after the first failure or interruption.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the commands without executing them.",
    )
    return parser.parse_args()


def main() -> int:
    settings = load_settings()
    args = parse_args()
    metrics_dir = Path(args.metrics_dir)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    benchmark_path = Path(args.benchmarks)
    recorder = BenchmarkRecorder(benchmark_path)
    tracker = RunTracker(SMOKE_TRACKER_PATH)
    tracker.cleanup_orphans()
    scenarios = load_scenarios(
        dataset_default=args.dataset,
        metrics_dir=metrics_dir,
        matrix_path=args.matrix,
    )
    requested = set(args.scenarios.split(",")) if args.scenarios else None
    queue = select_scenarios(scenarios, only=requested, resume_from=args.resume_from)
    if not queue:
        log("[smoke-train] No scenarios selected.")
        return 0
    monitor: TelemetryMonitor | None = None
    drain_automation: QueueDrainAutomation | None = None
    if not args.disable_auto_queue_drain and not args.dry_run:
        drain_automation = QueueDrainAutomation(
            python_bin=args.python,
            script_path=Path(args.queue_drain_script),
            metrics_dir=Path(args.queue_drain_metrics_dir),
            benchmarks_path=Path(args.queue_drain_benchmarks),
            threshold=args.queue_drain_threshold,
            cooldown_seconds=args.queue_drain_cooldown,
        )
    if not args.dry_run:
        monitor = TelemetryMonitor(
            settings,
            interval=max(1.0, args.telemetry_interval),
            queue_drain=drain_automation,
        )
        monitor.start()
    result_code = 0
    try:
        for scenario in queue:
            recorder.ensure_entry(scenario)
            mapping = scenario_mapping(scenario)
            train_args = ensure_metrics_arg(render_args(scenario.train_args, mapping), scenario.metrics_path)
            run_args = render_args(scenario.run_args, mapping)
            train_cmd = build_command(args.python, "src/train.py", train_args)
            run_cmd = build_command(args.python, "src/run.py", run_args)
            env = os.environ.copy()
            env["DBSLM_SQLITE_PATH"] = scenario.db_path
            env["DBSLM_CHEETAH_DATABASE"] = scenario.cheetah_database
            log(f"[smoke-train] Scenario {scenario.name}: {scenario.description}")
            log(f"[smoke-train] Train command: {command_to_text(train_cmd)}")
            log(f"[smoke-train] Run command:   {command_to_text(run_cmd)}")
            if args.dry_run:
                recorder.update(
                    scenario.name,
                    status="dry-run",
                    train_command=command_to_text(train_cmd),
                    run_command=command_to_text(run_cmd),
                )
                continue
            start_ts = timestamp()
            recorder.update(
                scenario.name,
                status="running:train",
                train_command=command_to_text(train_cmd),
                run_command=command_to_text(run_cmd),
                started_at=start_ts,
            )
            train_deadline = time.monotonic() + args.wall_time_limit
            code = run_subprocess(
                train_cmd,
                env=env,
                scenario=scenario,
                recorder=recorder,
                status_label="running:train",
                deadline=train_deadline,
                tracker=tracker,
            )
            if code != 0:
                recorder.update(scenario.name, status="failed:train", train_exit=code)
                result_code = code
                if args.fail_fast:
                    break
                continue
            recorder.finalize_metrics(scenario.name, scenario.metrics_path)
            recorder.update(scenario.name, status="running:run")
            run_deadline = time.monotonic() + args.wall_time_limit
            code = run_subprocess(
                run_cmd,
                env=env,
                scenario=scenario,
                recorder=recorder,
                status_label="running:run",
                deadline=run_deadline,
                tracker=tracker,
            )
            if code != 0:
                recorder.update(scenario.name, status="failed:run", run_exit=code)
                result_code = code
                if args.fail_fast:
                    break
                continue
            recorder.update(
                scenario.name,
                status="completed",
                completed_at=timestamp(),
            )
            recorder.finalize_metrics(scenario.name, scenario.metrics_path)
    finally:
        if monitor:
            monitor.stop()
    recorder.flush()
    return result_code


if __name__ == "__main__":
    raise SystemExit(main())
