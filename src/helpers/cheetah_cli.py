from __future__ import annotations

from typing import Sequence

from db_slm.adapters.base import HotPathAdapter
from db_slm.cheetah_types import (
    CheetahSystemStats,
    NamespaceSummary,
    PredictionQueryResult,
)


def parse_summary_prefix(expr: str) -> bytes:
    trimmed = (expr or "").strip()
    if not trimmed:
        raise ValueError("prefix cannot be empty")
    if trimmed.startswith("x"):
        try:
            return bytes.fromhex(trimmed[1:])
        except ValueError as exc:
            raise ValueError(f"invalid hex prefix '{expr}': {exc}") from exc
    return trimmed.encode("utf-8")


def describe_bytes(value: bytes) -> str:
    if not value:
        return "<root>"
    try:
        decoded = value.decode("utf-8")
    except UnicodeDecodeError:
        return f"x{value.hex()}"
    if decoded.isprintable():
        return decoded
    return f"x{value.hex()}"


def format_namespace_summary(summary: NamespaceSummary, *, label: str | None = None) -> list[str]:
    prefix_label = label or describe_bytes(summary.prefix)
    key_range = ""
    if summary.min_key is not None or summary.max_key is not None:
        key_range = f", keys={summary.min_key or 0}..{summary.max_key or 0}"
    lines = [
        (
            f"cheetah summary {prefix_label}: count={summary.terminal_count}, "
            f"payload_bytes={summary.total_payload_bytes} "
            f"(min={summary.min_payload_bytes}, max={summary.max_payload_bytes}), "
            f"max_depth={summary.max_depth}, self_terminal={'yes' if summary.self_terminal else 'no'}"
            f"{key_range}"
        )
    ]
    if summary.branches:
        branch_parts = []
        prefix = summary.prefix
        for branch_path, count in summary.branches:
            display = describe_bytes(prefix + branch_path)
            branch_parts.append(f"{display}:{count}")
        lines.append(f"  branches: {', '.join(branch_parts)}")
    return lines


def format_system_stats(stats: CheetahSystemStats) -> list[str]:
    cpu_proc = "NA" if stats.process_cpu_pct is None else f"{stats.process_cpu_pct:.2f}%"
    cpu_sys = "NA" if stats.system_cpu_pct is None else f"{stats.system_cpu_pct:.2f}%"
    line = (
        "cheetah system stats: "
        f"cores={stats.logical_cores}, gomaxprocs={stats.gomaxprocs}, goroutines={stats.goroutines}, "
        f"mem_alloc={stats.mem_alloc_bytes}, mem_sys={stats.mem_sys_bytes}, "
        f"cpu(process/system)={cpu_proc}/{cpu_sys}"
    )
    lines = [line]
    if stats.io_supported:
        read_rate = (
            "NA" if stats.io_read_bytes_per_sec is None else f"{stats.io_read_bytes_per_sec:.2f} B/s"
        )
        write_rate = (
            "NA" if stats.io_write_bytes_per_sec is None else f"{stats.io_write_bytes_per_sec:.2f} B/s"
        )
        lines.append(f"  io: read={read_rate}, write={write_rate}")
    if stats.recommended_workers:
        hints = ", ".join(f"{pending}->{workers}" for pending, workers in stats.recommended_workers)
        lines.append(f"  recommended_workers: {hints}")
    if stats.payload_cache_enabled:
        hit_pct = (
            "NA" if stats.payload_cache_hit_pct is None else f"{stats.payload_cache_hit_pct:.2f}%"
        )
        lines.append(
            "  payload_cache: "
            f"entries={stats.payload_cache_entries}/{stats.payload_cache_max_entries}, "
            f"bytes={stats.payload_cache_bytes}/{stats.payload_cache_max_bytes}, "
            f"hits={stats.payload_cache_hits}, misses={stats.payload_cache_misses}, "
            f"evictions={stats.payload_cache_evictions}, hit_ratio={hit_pct}"
        )
        if stats.payload_cache_advisory_bypass_bytes:
            lines.append(
                f"    advisory_bypass_bytes>{stats.payload_cache_advisory_bypass_bytes}"
            )
    else:
        lines.append("  payload_cache: disabled")
    auto_limit = stats.derive_reduce_page_limit()
    if auto_limit is not None:
        lines.append(f"  reducer_page_hint: limit~{auto_limit}")
    return lines


def collect_namespace_summary_lines(
    hot_path: HotPathAdapter,
    prefixes: Sequence[str],
    *,
    depth: int,
    branch_limit: int,
) -> list[str]:
    if not prefixes:
        return []
    summary_fn = getattr(hot_path, "namespace_summary", None)
    if not callable(summary_fn):
        return ["cheetah summary unavailable: adapter missing namespace_summary()"]
    lines: list[str] = []
    for raw in prefixes:
        try:
            prefix_bytes = parse_summary_prefix(raw)
        except ValueError as exc:
            lines.append(f"cheetah summary skipped for '{raw}': {exc}")
            continue
        summary = summary_fn(prefix_bytes, depth=depth, branch_limit=branch_limit)
        if summary is None:
            lines.append(f"cheetah summary '{raw}' unavailable (empty namespace or adapter disabled)")
            continue
        lines.extend(format_namespace_summary(summary, label=raw))
    return lines


def collect_system_stats_lines(hot_path: HotPathAdapter) -> list[str]:
    stats_fn = getattr(hot_path, "system_stats", None)
    if not callable(stats_fn):
        return ["cheetah system stats unavailable: adapter missing system_stats()"]
    stats = stats_fn()
    if stats is None:
        return ["cheetah system stats unavailable (adapter disabled or command unsupported)"]
    return format_system_stats(stats)


def format_prediction_query(
    result: PredictionQueryResult | None,
    *,
    label: str | None = None,
    max_entries: int = 5,
) -> list[str]:
    if result is None:
        return ["cheetah prediction query unavailable (adapter disabled or query failed)"]
    prefix = label or f"{result.table}"
    lines = [
        (
            f"cheetah predict {prefix}: table={result.table}, backend={result.backend}, "
            f"count={result.count}, showing<= {min(max_entries, len(result.entries))}"
        )
    ]
    if not result.entries:
        lines.append("  (no prediction entries returned)")
        return lines
    limit = max(1, max_entries)
    for idx, entry in enumerate(result.entries[:limit], 1):
        lines.append(f"  [{idx}] {describe_bytes(entry.value)} -> {entry.probability:.4f}")
    remaining = len(result.entries) - limit
    if remaining > 0:
        lines.append(f"  ... {remaining} more entr{'y' if remaining == 1 else 'ies'} truncated")
    return lines
