from __future__ import annotations

import base64
import binascii
import json
import logging
import os
import struct
import threading
import time
from collections import OrderedDict, deque
from dataclasses import dataclass
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Callable, Iterable, Sequence, NoReturn

from ..cheetah_types import (
    CHEETAH_DEFAULT_REDUCE_PAGE_SIZE,
    CHEETAH_PAIR_SCAN_MAX_LIMIT,
    CHEETAH_PAIR_SCAN_MIN_LIMIT,
    GRAPH_RECALL_MAX_BRANCH,
    GRAPH_RECALL_MAX_BUDGET,
    GRAPH_RECALL_MAX_HOPS,
    GRAPH_RECALL_MAX_REFERENCES,
    GRAPH_RECALL_MAX_SEEDS,
    CheetahSystemStats,
    GraphAssociation,
    GraphEdgeBatchResult,
    GraphNodeRecord,
    GraphRecallEdge,
    GraphRecallResult,
    GraphRecallSeed,
    GraphRecallSeedMatch,
    GraphRecallSource,
    GraphReferenceSentence,
    GraphSimilarMatch,
    GraphSimilarResult,
    GraphTermIndexStats,
    NamespaceSummary,
    PredictionQueryResult,
    PredictionValueResult,
    RawContinuationProjection,
    RawContextProjection,
    RawCountsProjection,
    RawProbabilityProjection,
)
from ..cheetah_vectors import AbsoluteVectorOrder
from ..hashing import hash_tokens

from ..settings import DBSLMSettings
from .base import HotPathAdapter, NullHotPathAdapter
from .cheetah_binder import (
    BinderCheetahClient,
    CheetahError,
    ThreadLocalClientPool,
    admin as binder_admin,
    graph as binder_graph,
    jobs as binder_jobs,
    kv as binder_kv,
    predict as binder_predict,
)

logger = logging.getLogger(__name__)

DEFAULT_REDUCE_PAGE_SIZE = CHEETAH_DEFAULT_REDUCE_PAGE_SIZE
PAIR_PUT_BATCH_MAX_ITEMS = binder_kv.PAIR_PUT_BATCH_MAX_ITEMS
PAIR_SCAN_MAX_LIMIT = CHEETAH_PAIR_SCAN_MAX_LIMIT
PAIR_SCAN_MIN_LIMIT = CHEETAH_PAIR_SCAN_MIN_LIMIT
READLINE_IDLE_GRACE_SECONDS = 30.0
_REDUCE_LIMIT_CACHE_TTL_SECONDS = 30.0
_CONTEXT_MATRIX_TABLE = "context_matrices"


# Destination resolution (`0.0.0.0` is a listen address, not a target; a WSL
# client may have to reach the Windows host) lives in the binder's `hosts`
# module, which `BinderCheetahClient` consults on connect.


# --------------------------------------------------------------------------- #
# Graph protocol encoding helpers
#
# Cheetah splits `GRAPH_*` arguments on whitespace, so ids, labels and types are
# single tokens and anything with a space (props, references, item lists, free
# text seeds) travels base64-encoded. Slugging and encoding belong here, never in
# the caller's string concatenation.
# --------------------------------------------------------------------------- #
def _graph_token(value: str | None) -> str:
    """Return a whitespace/comma-free protocol token, or "" when unusable."""
    text = (value or "").strip()
    if not text:
        return ""
    if any(char.isspace() for char in text) or "," in text:
        return ""
    return text


def _graph_encode_json(payload: object) -> str:
    raw = json.dumps(payload, separators=(",", ":"), ensure_ascii=False)
    return base64.b64encode(raw.encode("utf-8")).decode("ascii")


def _graph_encode_seeds(seeds: Sequence[str]) -> str:
    """Encode seed terms, switching to `base64:` when any term is not a token."""
    cleaned: list[str] = []
    seen: set[str] = set()
    for seed in seeds or ():
        term = " ".join(str(seed or "").split()).replace(",", " ").strip()
        if not term or term in seen:
            continue
        seen.add(term)
        cleaned.append(term)
        if len(cleaned) >= GRAPH_RECALL_MAX_SEEDS:
            break
    if not cleaned:
        return ""
    joined = ",".join(cleaned)
    if any(char.isspace() for char in joined):
        return "base64:" + base64.b64encode(joined.encode("utf-8")).decode("ascii")
    return joined


def _graph_format_precision(value: float | str) -> str:
    """`precision` accepts a number or a word from the modality scale."""
    if isinstance(value, str):
        return value.strip().lower()
    return f"{float(value):.4f}"


def _graph_float(value: object) -> float:
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 0.0


def _graph_references_from_payload(entries: object) -> tuple[GraphReferenceSentence, ...]:
    if not isinstance(entries, list):
        return tuple()
    references: list[GraphReferenceSentence] = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        text = str(entry.get("text") or "").strip()
        if not text:
            continue
        references.append(
            GraphReferenceSentence(
                reference_id=str(entry.get("id") or ""),
                text=text,
                source=str(entry.get("source") or ""),
                ordinal=int(_graph_float(entry.get("ordinal"))),
            )
        )
    return tuple(references)


def _graph_node_record_from_payload(payload: dict, *, fallback_id: str) -> GraphNodeRecord:
    props = payload.get("props")
    return GraphNodeRecord(
        node_id=str(payload.get("id") or fallback_id),
        labels=tuple(str(label) for label in payload.get("labels") or ()),
        props=dict(props) if isinstance(props, dict) else {},
        references=_graph_references_from_payload(payload.get("references")),
    )


def _graph_association_from_payload(entry: dict) -> GraphAssociation:
    sources = tuple(
        GraphRecallSource(
            seed=str(source.get("seed") or ""),
            activation=_graph_float(source.get("activation")),
            hops=int(_graph_float(source.get("hops"))),
        )
        for source in entry.get("sources") or ()
        if isinstance(source, dict)
    )
    via = tuple(
        GraphRecallEdge(
            from_id=str(edge.get("from") or ""),
            to_id=str(edge.get("to") or ""),
            edge_type=str(edge.get("type") or ""),
            weight=_graph_float(edge.get("weight")),
            confidence=_graph_float(edge.get("confidence")),
            modality=str(edge.get("modality") or ""),
            source=str(edge.get("source") or ""),
        )
        for edge in entry.get("via") or ()
        if isinstance(edge, dict)
    )
    return GraphAssociation(
        node_id=str(entry.get("id") or ""),
        score=_graph_float(entry.get("score")),
        novelty=_graph_float(entry.get("novelty")),
        distance=int(_graph_float(entry.get("distance"))),
        source_count=int(_graph_float(entry.get("source_count"))),
        bridge=bool(entry.get("bridge")),
        labels=tuple(str(label) for label in entry.get("labels") or ()),
        references=_graph_references_from_payload(entry.get("references")),
        sources=sources,
        via=via,
    )


class CheetahFatalError(CheetahError):
    """Raised when the cheetah hot-path adapter becomes unusable."""


class CheetahClient(BinderCheetahClient):
    """The DB-SLM dialect on top of the generic Cheetah client.

    The transport, the command/response codec and the generic command surface
    come from the binder in ``cheetah-db/binders/python``
    (:class:`~cheetah_db.client.CheetahClient`). What stays here is what only
    DB-SLM needs: base64-wrapped fixed-size payloads, the reducer job flow with
    its legacy fallback, and the projections in
    :mod:`db_slm.cheetah_types`.
    """

    def __init__(
        self,
        host: str,
        port: int,
        *,
        database: str = "default",
        timeout: float = 1.0,
        idle_grace: float | None = None,
    ) -> None:
        super().__init__(
            host,
            port,
            database=database,
            timeout=timeout,
            idle_grace=idle_grace if (idle_grace and idle_grace > 0) else max(timeout * 30.0, READLINE_IDLE_GRACE_SECONDS),
        )
        async_raw = os.environ.get("CHEETAH_REDUCE_ASYNC", "1").strip().lower()
        self._async_reducers = async_raw not in {"0", "false", "no", "off"}
        poll_raw = os.environ.get("CHEETAH_REDUCE_POLL_INTERVAL_SECONDS", "").strip()
        try:
            poll_interval = float(poll_raw) if poll_raw else 5.0
        except ValueError:
            poll_interval = 5.0
        self._reduce_poll_interval = max(1.0, poll_interval)
        self._job_api: bool | None = None
        async_inherit_raw = os.environ.get("CHEETAH_PREDICT_INHERIT_ASYNC", "1").strip().lower()
        self._async_inherit = async_inherit_raw not in {"0", "false", "no", "off"}

    # ------------------------------------------------------------------ #
    # DB-SLM value layer
    #
    # Payloads are fixed-size binary records, so every value crosses the text
    # protocol base64-wrapped. `decode_reduced_payload` is the matching unwrap
    # on the reducer side.
    # ------------------------------------------------------------------ #
    def insert(self, payload: bytes) -> tuple[int | None, str | None]:
        encoded = self._encode_value(payload)
        response = self._command(f"INSERT:{len(encoded)} {encoded}")
        return self._parse_key_response(response), response

    def edit(self, key: int, payload: bytes) -> tuple[bool, str | None]:
        encoded = self._encode_value(payload)
        response = self._command(f"EDIT {key} {encoded}")
        return (response is not None and response.startswith("SUCCESS")), response

    def read(self, key: int) -> bytes | None:
        response = self._command(f"READ {key}")
        if not response or not response.startswith("SUCCESS"):
            return None
        parts = response.split(",")
        for part in parts:
            if part.startswith("value="):
                raw = part.split("=", 1)[1]
                return self._decode_value(raw)
        return None

    def delete(self, key: int) -> tuple[bool, str | None]:
        response = self._command(f"DELETE {key}")
        return (response is not None and response.startswith("SUCCESS")), response

    def pair_set(self, value: bytes, key: int) -> tuple[bool, str | None]:
        response = self._command(f"PAIR_SET x{value.hex()} {key}")
        return (response is not None and response.startswith("SUCCESS")), response

    def pair_set_hidden(self, value: bytes, key: int) -> tuple[bool, str | None]:
        response = self._command(f"PAIR_SET_HIDDEN x{value.hex()} {key}")
        return (response is not None and response.startswith("SUCCESS")), response

    def pair_get(self, value: bytes) -> int | None:
        response = self._command(f"PAIR_GET x{value.hex()}")
        return self._parse_key_response(response)

    def pair_del(self, value: bytes) -> tuple[bool, str | None]:
        response = self._command(f"PAIR_DEL x{value.hex()}")
        return (response is not None and response.startswith("SUCCESS")), response

    def pair_scan(
        self,
        prefix: bytes = b"",
        limit: int = 0,
        cursor: bytes | None = None,
        include_hidden: bool = False,
    ) -> tuple[list[tuple[bytes, int]], bytes | None] | None:
        arg = "*" if not prefix else f"x{prefix.hex()}"
        command = f"PAIR_SCAN {arg}"
        if limit > 0:
            command = f"{command} {limit}"
        if cursor:
            if limit <= 0:
                command = f"{command} 0"
            command = f"{command} x{cursor.hex()}"
        if include_hidden:
            command = f"{command} include_hidden=1"
        response = self._command(command)
        if not response or not response.startswith("SUCCESS"):
            return None
        return self._parse_pair_scan_response(response)

    def pair_reduce(
        self,
        mode: str,
        prefix: bytes = b"",
        limit: int = 0,
        cursor: bytes | None = None,
        include_hidden: bool = False,
    ) -> tuple[list[tuple[bytes, int, bytes | None]], bytes | None] | None:
        if self._async_reducers:
            result = self._pair_reduce_async(
                mode,
                prefix,
                limit=limit,
                cursor=cursor,
                include_hidden=include_hidden,
            )
            if result is not None:
                return result
        return self._pair_reduce_sync(
            mode,
            prefix,
            limit=limit,
            cursor=cursor,
            include_hidden=include_hidden,
        )

    def _pair_reduce_sync(
        self,
        mode: str,
        prefix: bytes,
        *,
        limit: int,
        cursor: bytes | None,
        include_hidden: bool,
    ) -> tuple[list[tuple[bytes, int, bytes | None]], bytes | None] | None:
        command = self._format_pair_reduce_command(
            "PAIR_REDUCE",
            mode,
            prefix,
            limit,
            cursor,
            include_hidden=include_hidden,
        )
        response = self._command(command)
        if not response or not response.startswith("SUCCESS"):
            return None
        return self._parse_pair_reduce_response(response)

    def _pair_reduce_async(
        self,
        mode: str,
        prefix: bytes,
        *,
        limit: int,
        cursor: bytes | None,
        include_hidden: bool,
    ) -> tuple[list[tuple[bytes, int, bytes | None]], bytes | None] | None:
        reduce_command = self._format_pair_reduce_command(
            "PAIR_REDUCE",
            mode,
            prefix,
            limit,
            cursor,
            include_hidden=include_hidden,
        )
        response: str | None = None
        canonical_job = False
        if self._job_api is not False:
            encoded = base64.b64encode(reduce_command.encode("utf-8")).decode("ascii")
            response = self._command(f"JOB submit command={encoded}")
            lowered = (response or "").lower()
            if response and response.startswith("SUCCESS"):
                self._job_api = True
                canonical_job = True
            elif lowered.startswith("error,unknown_command"):
                self._job_api = False
                response = None
            else:
                return None
        if response is None:
            legacy_command = self._format_pair_reduce_command(
                "PAIR_REDUCE_ASYNC",
                mode,
                prefix,
                limit,
                cursor,
                include_hidden=include_hidden,
            )
            response = self._command(legacy_command)
        if not response:
            return None
        lowered = response.lower()
        if lowered.startswith("error,unknown_command") or "async_reducer_unavailable" in lowered:
            self._async_reducers = False
            return None
        if not response.startswith("SUCCESS"):
            return None
        job_id = self._extract_response_field(response, "job")
        if not job_id:
            self._async_reducers = False
            return None
        target = self._describe_reduce_target(mode, prefix)
        logger.warning(
            "cheetah pair_reduce job %s queued (%s limit=%s cursor=%s)",
            job_id,
            target,
            limit or "default",
            "set" if cursor else "none",
        )
        return self._await_reduce_job(
            job_id,
            mode,
            target,
            canonical_job=canonical_job,
        )

    def _await_reduce_job(
        self,
        job_id: str,
        mode: str,
        target_label: str,
        *,
        canonical_job: bool = False,
    ) -> tuple[list[tuple[bytes, int, bytes | None]], bytes | None] | None:
        last_progress: float | None = None
        last_completed: int | None = None
        last_state: str | None = None
        last_log = 0.0
        log_interval = max(30.0, self._reduce_poll_interval * 6.0)
        while True:
            if canonical_job:
                response = self._command(f"JOB status id={job_id}")
            else:
                response = self._command(f"PAIR_REDUCE_FETCH {job_id}")
            if not response:
                return None
            if response.startswith("SUCCESS"):
                if canonical_job:
                    state = (
                        self._extract_response_field(response, "state") or "running"
                    ).lower()
                    if state == "completed":
                        fetched = self._command(f"JOB fetch id={job_id}")
                        if not fetched or not fetched.startswith("SUCCESS"):
                            logger.warning(
                                "cheetah reducer job %s (%s %s) fetch failed: %s",
                                job_id,
                                mode,
                                target_label,
                                fetched,
                            )
                            return None
                        logger.warning(
                            "cheetah reducer job %s (%s %s) completed",
                            job_id,
                            mode,
                            target_label,
                        )
                        return self._parse_pair_reduce_response(fetched)
                    if state == "failed":
                        fetched = self._command(f"JOB fetch id={job_id}")
                        logger.warning(
                            "cheetah reducer job %s (%s %s) failed: %s",
                            job_id,
                            mode,
                            target_label,
                            fetched or response,
                        )
                        return None
                    response = "PENDING," + response.split(",", 1)[1]
                else:
                    if last_state:
                        logger.warning(
                            "cheetah reducer job %s (%s %s) completed",
                            job_id,
                            mode,
                            target_label,
                        )
                    return self._parse_pair_reduce_response(response)
            if response.startswith("PENDING"):
                state = self._extract_response_field(response, "state") or "running"
                reducer = self._extract_response_field(response, "reducer") or mode
                progress = self._extract_float_field(response, "progress")
                completed = self._extract_int_field(response, "completed")
                total = self._extract_int_field(response, "total")
                now = time.monotonic()
                should_log = False
                if state != last_state:
                    should_log = True
                elif (
                    progress is not None
                    and last_progress is not None
                    and progress - last_progress >= 5.0
                ):
                    should_log = True
                elif completed is not None and completed != last_completed:
                    should_log = True
                elif now - last_log >= log_interval:
                    should_log = True
                if should_log:
                    percent_label = f"{progress:.1f}%" if progress is not None else "?"
                    if completed is not None and total:
                        extra = f"{completed}/{total}"
                    else:
                        extra = "in-progress"
                    logger.warning(
                        "cheetah reducer job %s (%s %s) state=%s progress=%s (%s)",
                        job_id,
                        reducer,
                        target_label,
                        state,
                        percent_label,
                        extra,
                    )
                    last_log = now
                last_state = state
                if progress is not None:
                    last_progress = progress
                if completed is not None:
                    last_completed = completed
                time.sleep(self._reduce_poll_interval)
                continue
            if response.startswith("ERROR"):
                logger.warning(
                    "cheetah reducer job %s (%s %s) failed: %s",
                    job_id,
                    mode,
                    target_label,
                    response,
                )
                return None
            return None

    def _format_pair_reduce_command(
        self,
        command_name: str,
        mode: str,
        prefix: bytes,
        limit: int,
        cursor: bytes | None,
        *,
        include_hidden: bool = False,
    ) -> str:
        arg = "*" if not prefix else f"x{prefix.hex()}"
        command = f"{command_name} {mode} {arg}"
        if limit != 0:
            command = f"{command} {limit}"
        if cursor:
            if limit == 0:
                command = f"{command} 0"
            command = f"{command} x{cursor.hex()}"
        if include_hidden:
            command = f"{command} include_hidden=1"
        return command

    @staticmethod
    def _describe_reduce_target(mode: str, prefix: bytes) -> str:
        if not prefix:
            return f"{mode} *"
        try:
            label = prefix.decode("utf-8")
        except UnicodeDecodeError:
            label = f"x{prefix.hex()}"
        return f"{mode} {label}"

    def pair_purge(
        self,
        prefix: bytes = b"",
        limit: int | None = None,
    ) -> tuple[int | None, str | None]:
        arg = "*" if not prefix else f"x{prefix.hex()}"
        command = f"PAIR_PURGE {arg}"
        if limit is not None and limit > 0:
            command = f"{command} {limit}"
        response = self._command(command)
        if not response or not response.startswith("SUCCESS"):
            return None, response
        removed: int | None = None
        for part in response.split(","):
            if part.startswith("purged="):
                try:
                    removed = int(part.split("=", 1)[1])
                except ValueError:
                    removed = None
                break
        return removed, response

    def predict_set(
        self,
        key: bytes,
        value: bytes,
        *,
        probability: float = 0.5,
        table: str | None = None,
        weights: Sequence[dict[str, object]] | None = None,
    ) -> tuple[bool, str | None]:
        prob = max(0.0, min(1.0, float(probability)))
        args = [
            f"key=x{key.hex()}",
            f"value=x{value.hex()}",
            f"prob={prob}",
        ]
        if table:
            args.append(f"table={table}")
        if weights:
            encoded = base64.b64encode(
                json.dumps(weights, separators=(",", ":")).encode("utf-8")
            ).decode("ascii")
            args.append(f"weights={encoded}")
        response = self._command(f"PREDICT_SET {' '.join(args)}")
        return (response is not None and response.startswith("SUCCESS")), response

    def predict_ctx(
        self,
        key: bytes,
        ctx_matrix: Sequence[Sequence[float]],
        *,
        table: str | None = None,
        mode: str | None = None,
        strength: float | None = None,
    ) -> tuple[bool, str | None]:
        if not ctx_matrix:
            return True, "SKIP"
        matrix_payload = base64.b64encode(
            json.dumps(ctx_matrix, separators=(",", ":")).encode("utf-8")
        ).decode("ascii")
        args = [f"key=x{key.hex()}", f"ctx={matrix_payload}"]
        if mode:
            args.append(f"mode={mode}")
        if strength is not None:
            args.append(f"strength={strength}")
        if table:
            args.append(f"table={table}")
        response = self._command(f"PREDICT_CTX {' '.join(args)}")
        return (response is not None and response.startswith("SUCCESS")), response

    def predict_train(
        self,
        key: bytes,
        target: bytes,
        *,
        context_matrix: Sequence[Sequence[float]] | dict[str, object] | None = None,
        learning_rate: float | None = None,
        table: str | None = None,
        negatives: Sequence[bytes | str] | None = None,
    ) -> tuple[bool, str | None]:
        args = [f"key=x{key.hex()}", f"target=x{target.hex()}"]
        matrix_payload = self._encode_matrix_payload(context_matrix)
        if matrix_payload:
            args.append(f"ctx={matrix_payload}")
        if learning_rate is not None:
            args.append(f"lr={learning_rate}")
        if table:
            args.append(f"table={table}")
        if negatives:
            formatted: list[str] = []
            for value in negatives:
                encoded_value = self._format_prediction_value(value)
                if encoded_value:
                    formatted.append(encoded_value)
            if formatted:
                args.append(f"negatives={','.join(formatted)}")
        response = self._command(f"PREDICT_TRAIN {' '.join(args)}")
        return (response is not None and response.startswith("SUCCESS")), response

    def predict_inherit(
        self,
        key: bytes,
        target: bytes,
        sources: Sequence[bytes | str],
        *,
        table: str | None = None,
        merge_mode: str | None = None,
    ) -> tuple[bool, str | None]:
        args = [f"key=x{key.hex()}", f"target=x{target.hex()}"]
        formatted_sources: list[str] = []
        for value in sources:
            encoded_value = self._format_prediction_value(value)
            if encoded_value:
                formatted_sources.append(encoded_value)
        if formatted_sources:
            args.append(f"sources={','.join(formatted_sources)}")
        if merge_mode:
            args.append(f"merge={merge_mode}")
        if table:
            args.append(f"table={table}")
        response = self._command(f"PREDICT_INHERIT {' '.join(args)}")
        return (response is not None and response.startswith("SUCCESS")), response

    def predict_inherit_batch(
        self,
        items: Sequence[dict[str, object]],
        *,
        table: str | None = None,
        merge_mode: str | None = None,
        async_mode: bool | None = None,
    ) -> tuple[bool, str | None]:
        payload = self._encode_inherit_batch_payload(items)
        if not payload:
            return False, "ERROR,empty_batch"
        args = [f"items={payload}"]
        if merge_mode:
            args.append(f"merge={merge_mode}")
        if table:
            args.append(f"table={table}")
        if async_mode is None:
            async_mode = self._async_inherit
        if async_mode:
            response = self._command(f"PREDICT_INHERIT_ASYNC {' '.join(args)}")
            if not response:
                return False, response
            lowered = response.lower()
            if lowered.startswith("error,unknown_command") or "async_inherit_unavailable" in lowered:
                self._async_inherit = False
                return self.predict_inherit_batch(
                    items,
                    table=table,
                    merge_mode=merge_mode,
                    async_mode=False,
                )
            return response.startswith("SUCCESS"), response
        response = self._command(f"PREDICT_INHERIT_BATCH {' '.join(args)}")
        if response and response.lower().startswith("error,unknown_command"):
            return False, response
        return (response is not None and response.startswith("SUCCESS")), response

    def predict_query(
        self,
        *,
        key: bytes | str | None = None,
        keys: Sequence[bytes | str] | None = None,
        context_matrix: Sequence[Sequence[float]] | dict[str, object] | None = None,
        windows: Sequence[Sequence[float]] | None = None,
        key_windows: Sequence[tuple[bytes | str, Sequence[Sequence[float]]]] | None = None,
        merge_mode: str | None = None,
        table: str | None = None,
    ) -> PredictionQueryResult | None:
        args: list[str] = []
        if key:
            formatted = self._format_prediction_value(key)
            if not formatted:
                return None
            args.append(f"key={formatted}")
        if keys:
            formatted_keys = []
            for candidate in keys:
                encoded_value = self._format_prediction_value(candidate)
                if encoded_value:
                    formatted_keys.append(encoded_value)
            if formatted_keys:
                args.append(f"keys={','.join(formatted_keys)}")
        ctx_payload = self._encode_matrix_payload(context_matrix)
        if ctx_payload:
            args.append(f"ctx={ctx_payload}")
        window_payload = self._encode_matrix_payload(windows)
        if window_payload:
            args.append(f"windows={window_payload}")
        key_window_payload = self._encode_key_windows_payload(key_windows)
        if key_window_payload:
            args.append(f"key_windows={key_window_payload}")
        if merge_mode:
            args.append(f"merge={merge_mode}")
        if table:
            trimmed = table.strip()
            if trimmed:
                args.append(f"table={trimmed}")
        if not args:
            return None
        response = self._command(f"PREDICT_QUERY {' '.join(args)}")
        if not response or not response.startswith("SUCCESS"):
            return None
        return self._parse_predict_query_response(response)

    # ------------------------------------------------------------------ #
    # Graph context memory (GRAPH_*)
    # ------------------------------------------------------------------ #
    def graph_node_set(
        self,
        node_id: str,
        *,
        labels: Sequence[str] | None = None,
        props: dict[str, object] | None = None,
        references: Sequence[dict[str, object]] | None = None,
        clear_references: bool = False,
    ) -> bool:
        """Upsert a node. Omitted fields keep their stored value server-side."""
        identifier = _graph_token(node_id)
        if not identifier:
            return False
        args = [f"id={identifier}"]
        label_tokens = [token for token in (_graph_token(label) for label in labels or ()) if token]
        if label_tokens:
            args.append(f"labels={','.join(label_tokens)}")
        if props:
            args.append(f"props={_graph_encode_json(props)}")
        if clear_references:
            args.append("references=-")
        elif references:
            args.append(f"references={_graph_encode_json(list(references))}")
        response = self._command(f"GRAPH_NODE_SET {' '.join(args)}")
        return bool(response and response.startswith("SUCCESS"))

    def graph_node_get(self, node_id: str) -> GraphNodeRecord | None:
        """Read one node record. A missing node answers ``ERROR,node_not_found``."""
        identifier = _graph_token(node_id)
        if not identifier:
            return None
        response = self._command(f"GRAPH_NODE_GET id={identifier}")
        if not response or not response.startswith("SUCCESS"):
            return None
        payload = self._decode_graph_payload(response)
        if not isinstance(payload, dict):
            return None
        return _graph_node_record_from_payload(payload, fallback_id=identifier)

    def graph_edge_set_batch(
        self,
        items: Sequence[dict[str, object]],
        *,
        continue_on_error: bool = True,
        default_type: str | None = None,
        default_props: dict[str, object] | None = None,
    ) -> GraphEdgeBatchResult | None:
        """Upsert many edges in one round-trip (``GRAPH_EDGE_SET_BATCH``)."""
        payload_items = [item for item in items if item]
        if not payload_items:
            return None
        args = [f"items={_graph_encode_json(payload_items)}"]
        type_token = _graph_token(default_type or "")
        if type_token:
            args.append(f"type={type_token}")
        if default_props:
            args.append(f"props={_graph_encode_json(default_props)}")
        if continue_on_error:
            args.append("continue_on_error=1")
        response = self._command(f"GRAPH_EDGE_SET_BATCH {' '.join(args)}")
        if not response or not response.startswith("SUCCESS"):
            return None
        fields = self._split_response_fields(response)
        return GraphEdgeBatchResult(
            requested=self._parse_int(fields.get("requested")),
            applied=self._parse_int(fields.get("applied")),
            created=self._parse_int(fields.get("created")),
            updated=self._parse_int(fields.get("updated")),
            failed=self._parse_int(fields.get("failed")),
        )

    def graph_recall(
        self,
        seeds: Sequence[str],
        *,
        precision: float | str | None = None,
        hops: int | None = None,
        min_sources: int | None = None,
        direction: str | None = None,
        edge_types: Sequence[str] | None = None,
        decay: float | None = None,
        expand: str | None = None,
        references: bool = False,
        reference_limit: int | None = None,
        include_seeds: bool = False,
        limit: int | None = None,
        branch_limit: int | None = None,
        budget: int | None = None,
    ) -> GraphRecallResult | None:
        """Spread activation from every seed at once (``GRAPH_RECALL``)."""
        seed_arg = _graph_encode_seeds(seeds)
        if not seed_arg:
            return None
        args = [f"seeds={seed_arg}"]
        if precision is not None:
            args.append(f"precision={_graph_format_precision(precision)}")
        if hops is not None:
            args.append(f"hops={max(1, min(int(hops), GRAPH_RECALL_MAX_HOPS))}")
        if min_sources is not None and int(min_sources) > 1:
            args.append(f"min_sources={int(min_sources)}")
        direction_token = (direction or "").strip().lower()
        if direction_token in {"out", "in", "both"}:
            args.append(f"direction={direction_token}")
        type_tokens = [token for token in (_graph_token(name) for name in edge_types or ()) if token]
        if type_tokens:
            args.append(f"type={','.join(type_tokens)}")
        if decay is not None:
            args.append(f"decay={float(decay):.4f}")
        expand_token = (expand or "").strip().lower()
        if expand_token:
            args.append(f"expand={expand_token}")
        if references:
            args.append("references=1")
            if reference_limit is not None:
                bounded = max(1, min(int(reference_limit), GRAPH_RECALL_MAX_REFERENCES))
                args.append(f"reference_limit={bounded}")
        if include_seeds:
            # Seed nodes are excluded from the answer by default. Their own
            # recorded sentences are the closest grounding a turn has, so ask for
            # them back whenever references are being hydrated.
            args.append("include_seeds=1")
        if limit is not None and int(limit) > 0:
            args.append(f"limit={int(limit)}")
        if branch_limit is not None and int(branch_limit) > 0:
            args.append(f"branch_limit={min(int(branch_limit), GRAPH_RECALL_MAX_BRANCH)}")
        if budget is not None and int(budget) > 0:
            args.append(f"budget={min(int(budget), GRAPH_RECALL_MAX_BUDGET)}")
        response = self._command(f"GRAPH_RECALL {' '.join(args)}")
        if not response or not response.startswith("SUCCESS"):
            return None
        return self._parse_graph_recall_response(response)

    def graph_similar(
        self,
        node_id: str,
        *,
        by: str | None = None,
        limit: int | None = None,
        precision: float | str | None = None,
    ) -> GraphSimilarResult | None:
        """Answer "what else behaves like this?" (``GRAPH_SIMILAR``)."""
        identifier = _graph_token(node_id)
        if not identifier:
            return None
        args = [f"id={identifier}"]
        by_token = (by or "").strip().lower()
        if by_token:
            args.append(f"by={by_token}")
        if limit is not None and int(limit) > 0:
            args.append(f"limit={int(limit)}")
        if precision is not None:
            args.append(f"precision={_graph_format_precision(precision)}")
        response = self._command(f"GRAPH_SIMILAR {' '.join(args)}")
        if not response or not response.startswith("SUCCESS"):
            return None
        fields = self._split_response_fields(response)
        payload = self._decode_graph_payload(response)
        matches: list[GraphSimilarMatch] = []
        if isinstance(payload, list):
            for entry in payload:
                if not isinstance(entry, dict):
                    continue
                matches.append(
                    GraphSimilarMatch(
                        node_id=str(entry.get("id") or ""),
                        score=_graph_float(entry.get("score")),
                        context=_graph_float(entry.get("context")),
                        lexical=_graph_float(entry.get("lexical")),
                        shared_count=int(_graph_float(entry.get("shared_count"))),
                        shared=tuple(str(item) for item in entry.get("shared") or ()),
                        labels=tuple(str(item) for item in entry.get("labels") or ()),
                    )
                )
        return GraphSimilarResult(
            node_id=fields.get("id", identifier),
            count=self._parse_int(fields.get("count")),
            truncated=self._parse_bool(fields.get("truncated")),
            matches=tuple(matches),
        )

    def graph_term_index(
        self,
        action: str = "stats",
        *,
        limit: int | None = None,
        cursor: str | None = None,
    ) -> GraphTermIndexStats | None:
        """Maintain the derived lexical index free-text seeds resolve through."""
        action_token = (action or "stats").strip().lower()
        if action_token not in {"stats", "status", "rebuild", "reindex", "drop", "clear"}:
            return None
        args = [f"action={action_token}"]
        if limit is not None and int(limit) > 0:
            args.append(f"limit={int(limit)}")
        cursor_token = (cursor or "").strip()
        if cursor_token:
            args.append(f"cursor={cursor_token}")
        response = self._command(f"GRAPH_TERM_INDEX {' '.join(args)}")
        if not response or not response.startswith("SUCCESS"):
            return None
        fields = self._split_response_fields(response)
        return GraphTermIndexStats(
            action=fields.get("action", action_token),
            enabled=self._parse_bool(fields.get("enabled")),
            entries=self._parse_int(fields.get("entries")),
            nodes=self._parse_int(fields.get("nodes")),
            terms=self._parse_int(fields.get("terms")),
            removed=self._parse_int(fields.get("removed")),
            next_cursor=fields.get("next_cursor", ""),
        )

    # ------------------------------------------------------------------ #
    # Commands delegated to the binder
    #
    # These have no DB-SLM projection: the binder already spells the command
    # and decodes the answer, so the only thing added here is the house style
    # of the rest of this client — a failure is a `None`/empty answer plus a
    # log line, not an exception through a hot path. The generic `execute`,
    # `execute_kv` and `send` from the binder stay available for one-off
    # commands that do not deserve a method.
    # ------------------------------------------------------------------ #
    def _binder_call(self, what: str, call: Callable[[], object], default: object = None) -> object:
        try:
            return call()
        except CheetahError as exc:
            logger.debug("cheetah %s failed: %s", what, exc)
            return default

    def pair_put_batch(
        self,
        entries: Sequence[tuple[bytes, bytes]],
        *,
        hidden: bool = False,
        want_keys: bool = False,
        continue_on_error: bool = False,
    ) -> list[int | None] | None:
        """Store and bind many payloads in one request (``PAIR_PUT_BATCH``).

        The DB-SLM transport is preserved: each payload is base64-wrapped
        exactly as :meth:`insert` wraps it, so a row written through the batch
        reads back through :meth:`read` and decodes through
        :meth:`decode_reduced_payload` like any other.

        Returns the assigned absolute keys when ``want_keys`` is set, an empty
        list otherwise, and ``None`` when the batch did not fully apply — the
        caller must fall back rather than assume the rows exist, because this
        is not a transaction.
        """
        if not entries:
            return []
        prepared = [(value, self._encode_value(payload)) for value, payload in entries]
        result = self._binder_call(
            f"PAIR_PUT_BATCH of {len(prepared)}",
            lambda: binder_kv.put_values_batch(
                self,
                prepared,
                hidden=hidden,
                want_keys=want_keys,
                continue_on_error=continue_on_error,
            ),
            default=None,
        )
        return result  # type: ignore[return-value]

    def pair_purge_prefix(self, prefix: bytes, *, limit: int = 0, payloads: bool = True) -> int:
        """``DEL pairs prefix=…`` — the micro-command form, with ``payloads=0``.

        ``payloads=False`` unlinks the names and leaves the values readable by
        absolute key, which :meth:`pair_purge` (the historical ``PAIR_PURGE``)
        cannot express.
        """
        return int(
            self._binder_call(
                "DEL pairs prefix",
                lambda: binder_kv.pair_purge(self, prefix, limit=limit, payloads=payloads),
                default=0,
            )
            or 0
        )

    def graph_node_delete(self, node_id: str, *, cascade: bool = False) -> bool:
        """Forget a node. Without ``cascade`` its incident edges are left dangling."""
        return bool(
            self._binder_call(
                f"DEL graph node={node_id}",
                lambda: binder_graph.delete_node(self, node_id, cascade=cascade),
                default=False,
            )
        )

    def graph_edge_delete(
        self,
        from_id: str,
        to_id: str,
        *,
        edge_type: str | None = None,
        directed: bool | None = None,
    ) -> bool:
        return bool(
            self._binder_call(
                f"DEL graph {from_id}->{to_id}",
                lambda: binder_graph.delete_edge(
                    self, from_id=from_id, to_id=to_id, edge_type=edge_type, directed=directed
                ),
                default=False,
            )
        )

    def graph_edge_get(
        self,
        from_id: str,
        to_id: str,
        *,
        edge_type: str | None = None,
        directed: bool | None = None,
    ) -> dict | None:
        record = self._binder_call(
            f"GRAPH_EDGE_GET {from_id}->{to_id}",
            lambda: binder_graph.get_edge(
                self, from_id=from_id, to_id=to_id, edge_type=edge_type, directed=directed
            ),
        )
        return record if isinstance(record, dict) else None

    def graph_degree(
        self,
        node_id: str,
        *,
        direction: str = "out",
        edge_type: str | None = None,
        weighted: bool = False,
    ) -> int:
        """How many edges a node carries — the cheapest graph question there is.

        No edge record is hydrated, which is what makes it usable as a hub test
        before deciding whether a seed is worth recalling on.
        """
        result = self._binder_call(
            f"GRAPH_DEGREE {node_id}",
            lambda: binder_graph.degree(
                self, node_id, direction=direction, edge_type=edge_type, weighted=weighted
            ),
            default={"degree": 0},
        )
        return int((result or {}).get("degree", 0))  # type: ignore[union-attr]

    def graph_neighbors(
        self,
        node_id: str,
        *,
        direction: str = "out",
        edge_type: str | None = None,
        limit: int | None = None,
        cursor: str | None = None,
    ) -> tuple[list[dict], str | None]:
        result = self._binder_call(
            f"GRAPH_NEIGHBORS {node_id}",
            lambda: binder_graph.neighbors(
                self,
                node_id,
                direction=direction,
                edge_type=edge_type,
                limit=limit,
                cursor=cursor,
            ),
            default=([], None),
        )
        edges, next_cursor = result  # type: ignore[misc]
        return [edge for edge in edges if isinstance(edge, dict)], next_cursor

    def graph_neighbor_types(
        self,
        node_id: str,
        *,
        direction: str = "out",
        limit: int | None = None,
        weighted: bool = False,
    ) -> list[dict]:
        result = self._binder_call(
            f"GRAPH_NEIGHBOR_TYPES {node_id}",
            lambda: binder_graph.neighbor_types(
                self, node_id, direction=direction, limit=limit, weighted=weighted
            ),
            default=([], None),
        )
        entries, _cursor = result  # type: ignore[misc]
        return [entry for entry in entries if isinstance(entry, dict)]

    def graph_query(self, clause: str) -> dict | None:
        """``GRAPH_QUERY`` — the clause dialect, passed through to the server."""
        result = self._binder_call(f"GRAPH_QUERY {clause}", lambda: binder_graph.query(self, clause))
        return result if isinstance(result, dict) else None

    def predict_backend(self, *, mode: str | None = None, table: str | None = None) -> str | None:
        """Read or switch a prediction table's merger. ``gpu`` is CPU fan-out."""
        result = self._binder_call(
            "PREDICT_BACKEND", lambda: binder_predict.backend(self, mode=mode, table=table)
        )
        return None if not isinstance(result, dict) else result.get("backend")

    def predict_bench(self, *, samples: int, window: int, table: str | None = None) -> dict | None:
        result = self._binder_call(
            "PREDICT_BENCH",
            lambda: binder_predict.bench(self, samples=samples, window=window, table=table),
        )
        return None if not isinstance(result, dict) else dict(result.get("fields") or {})

    def log_flush(self, limit: int = 0) -> list[str]:
        """Dump **and clear** the server's in-memory log ring. Keep one flusher."""
        entries = self._binder_call("LOG_FLUSH", lambda: binder_admin.log_flush(self, limit), default=[])
        return list(entries or ())  # type: ignore[arg-type]

    def file_checkpoint(
        self, *, idle: str | None = None, drop_cache: bool = False, close_handles: bool = False
    ) -> int:
        """Flush the managed-file layer now instead of at shutdown."""
        return int(
            self._binder_call(
                "FILE_CHECKPOINT",
                lambda: binder_admin.file_checkpoint(
                    self, idle=idle, drop_cache=drop_cache, close_handles=close_handles
                ),
                default=0,
            )
            or 0
        )

    def cluster_status(self) -> dict | None:
        result = self._binder_call("CLUSTER_STATUS", lambda: binder_admin.cluster_status(self))
        return None if not isinstance(result, dict) else dict(result.get("fields") or {})

    def fork_assign(self, prefix: bytes | None = None) -> dict | None:
        result = self._binder_call("FORK_ASSIGN", lambda: binder_admin.fork_assign(self, prefix))
        if not isinstance(result, dict):
            return None
        return {"fork_id": result.get("fork_id"), "nodes": result.get("nodes") or []}

    def supports_job_api(self) -> bool:
        """Whether this server knows the ``JOB`` micro-command.

        Cached after the first probe: it is a property of the server build, and
        the reducer path already discovers it lazily on its first submit.
        """
        if self._job_api is None:
            self._job_api = binder_jobs.supports_job_api(self)
        return bool(self._job_api)

    @classmethod
    def _decode_graph_payload(cls, response: str) -> object | None:
        """Decode the base64 JSON carried by ``payload=`` on a graph response."""
        encoded = cls._extract_response_field(response, "payload")
        if not encoded:
            return None
        try:
            raw = base64.b64decode(encoded.encode("ascii"))
        except (ValueError, binascii.Error):
            return None
        try:
            return json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            return None

    @classmethod
    def _parse_graph_recall_response(cls, response: str) -> GraphRecallResult:
        fields = cls._split_response_fields(response)
        payload = cls._decode_graph_payload(response)
        seed_resolutions: list[GraphRecallSeed] = []
        unresolved: list[str] = []
        associations: list[GraphAssociation] = []
        if isinstance(payload, dict):
            for entry in payload.get("seeds") or ():
                if not isinstance(entry, dict):
                    continue
                matches = tuple(
                    GraphRecallSeedMatch(
                        node_id=str(match.get("id") or ""),
                        score=_graph_float(match.get("score")),
                        match=str(match.get("match") or ""),
                    )
                    for match in entry.get("matches") or ()
                    if isinstance(match, dict)
                )
                seed_resolutions.append(
                    GraphRecallSeed(term=str(entry.get("term") or ""), matches=matches)
                )
            unresolved = [str(term) for term in payload.get("unresolved") or ()]
            for entry in payload.get("associations") or ():
                if not isinstance(entry, dict):
                    continue
                associations.append(_graph_association_from_payload(entry))
        return GraphRecallResult(
            seeds=cls._parse_int(fields.get("seeds")),
            resolved=cls._parse_int(fields.get("resolved")),
            visited=cls._parse_int(fields.get("visited")),
            expanded=cls._parse_int(fields.get("expanded")),
            hydrated=cls._parse_int(fields.get("hydrated")),
            reference_count=cls._parse_int(fields.get("references")),
            count=cls._parse_int(fields.get("count")),
            bridges=cls._parse_int(fields.get("bridges")),
            truncated=cls._parse_bool(fields.get("truncated")),
            precision=cls._parse_float(fields.get("precision")) or 0.0,
            seed_resolutions=tuple(seed_resolutions),
            unresolved=tuple(unresolved),
            associations=tuple(associations),
        )

    def pair_summary(
        self,
        prefix: bytes,
        *,
        depth: int = 1,
        branch_limit: int = 32,
        include_hidden: bool = False,
    ) -> NamespaceSummary | None:
        arg = "*" if not prefix else f"x{prefix.hex()}"
        command = f"PAIR_SUMMARY {arg}"
        if depth is not None:
            command = f"{command} {depth}"
        if branch_limit is not None:
            command = f"{command} {branch_limit}"
        if include_hidden:
            command = f"{command} include_hidden=1"
        response = self._command(command)
        if not response or not response.startswith("SUCCESS"):
            return None
        return self._parse_pair_summary_response(prefix, response)

    def system_stats(self) -> CheetahSystemStats | None:
        response = self._command("SYSTEM_STATS")
        if not response or not response.startswith("SUCCESS"):
            return None
        return self._parse_system_stats_response(response)

    def reset_database(self, name: str | None = None) -> tuple[bool, str | None]:
        target = (name or self.database or "default").strip() or "default"
        response = self._command(f"RESET_DB {target}")
        return (response is not None and response.startswith("SUCCESS")), response

    # ------------------------------------------------------------------ #
    # DB-SLM payload transport
    #
    # The socket, the reconnect and the response codec belong to the binder's
    # client; only the base64 wrapping of fixed-size binary payloads is ours.
    # ------------------------------------------------------------------ #
    def _encode_value(self, payload: bytes) -> str:
        return base64.b64encode(payload).decode("ascii")

    def _decode_value(self, encoded: str) -> bytes:
        return base64.b64decode(encoded.encode("ascii"))

    @staticmethod
    def decode_reduced_payload(payload: bytes | None) -> bytes | None:
        """Undo the transport encoding applied by ``insert`` after a reducer read."""
        if payload is None:
            return None
        try:
            return base64.b64decode(payload, validate=True)
        except (ValueError, binascii.Error):
            return None

    @staticmethod
    def _parse_key_response(response: str | None) -> int | None:
        if not response or not response.startswith("SUCCESS"):
            return None
        parts = response.split(",")
        for part in parts:
            if part.startswith("key="):
                try:
                    return int(part.split("=", 1)[1])
                except ValueError:
                    return None
        return None

    @staticmethod
    def _parse_pair_scan_response(response: str) -> tuple[list[tuple[bytes, int]], bytes | None]:
        if not response.startswith("SUCCESS"):
            return [], None
        header, payload = CheetahClient._split_response_sections(response)
        next_cursor = CheetahClient._extract_cursor(header)
        entries: list[tuple[bytes, int]] = []
        if payload:
            for item in payload.split(";"):
                if not item:
                    continue
                try:
                    value_hex, key_text = item.rsplit(":", 1)
                    value = bytes.fromhex(value_hex)
                    key = int(key_text)
                except (ValueError, TypeError):
                    continue
                entries.append((value, key))
        return entries, next_cursor

    @staticmethod
    def _parse_pair_reduce_response(
        response: str,
    ) -> tuple[list[tuple[bytes, int, bytes | None]], bytes | None]:
        if not response.startswith("SUCCESS"):
            return [], None
        header, payload = CheetahClient._split_response_sections(response)
        next_cursor = CheetahClient._extract_cursor(header)
        entries: list[tuple[bytes, int, bytes | None]] = []
        if payload:
            for item in payload.split(";"):
                if not item:
                    continue
                parts = item.split(":")
                if len(parts) < 2:
                    continue
                value_hex = parts[0]
                key_text = parts[1]
                blob: bytes | None = None
                if len(parts) > 2:
                    try:
                        blob = base64.b64decode(parts[2].encode("ascii"))
                    except (ValueError, binascii.Error):
                        blob = None
                try:
                    value = bytes.fromhex(value_hex)
                    key = int(key_text)
                except (ValueError, TypeError):
                    continue
                entries.append((value, key, blob))
        return entries, next_cursor

    @staticmethod
    def _split_response_sections(response: str) -> tuple[str, str]:
        marker = ",items="
        payload_start = response.find(marker)
        if payload_start == -1:
            return response, ""
        header = response[:payload_start]
        payload = response[payload_start + len(marker) :]
        return header, payload

    @staticmethod
    def _parse_predict_query_response(response: str) -> PredictionQueryResult | None:
        if not response.startswith("SUCCESS"):
            return None
        header, payload = CheetahClient._split_response_sections(response)
        meta = CheetahClient._split_response_fields(header)
        table = meta.get("table", "").strip() or "<unknown>"
        backend = meta.get("backend", "").strip() or "<unknown>"
        count = CheetahClient._parse_int(meta.get("count"), default=0)
        entries: list[PredictionValueResult] = []
        if payload:
            for item in payload.split(";"):
                if not item:
                    continue
                try:
                    value_hex, prob_text = item.split(":", 1)
                except ValueError:
                    continue
                try:
                    value = bytes.fromhex(value_hex)
                except (ValueError, TypeError):
                    continue
                try:
                    probability = float(prob_text)
                except ValueError:
                    continue
                entries.append(PredictionValueResult(value=value, probability=probability))
        if count <= 0:
            count = len(entries)
        return PredictionQueryResult(
            table=table,
            backend=backend,
            count=count,
            entries=tuple(entries),
        )

    @staticmethod
    def _extract_cursor(header: str) -> bytes | None:
        for part in header.split(","):
            if part.startswith("next_cursor="):
                token = part.split("=", 1)[1]
                if token.startswith("x"):
                    try:
                        return bytes.fromhex(token[1:])
                    except ValueError:
                        return None
        return None

    @staticmethod
    def _extract_response_field(response: str, key: str) -> str | None:
        prefix = f"{key}="
        for part in response.split(","):
            trimmed = part.strip()
            if trimmed.startswith(prefix):
                return trimmed.split("=", 1)[1]
        return None

    @staticmethod
    def _extract_int_field(response: str, key: str) -> int | None:
        value = CheetahClient._extract_response_field(response, key)
        if value is None:
            return None
        try:
            return int(value)
        except ValueError:
            return None

    @staticmethod
    def _extract_float_field(response: str, key: str) -> float | None:
        value = CheetahClient._extract_response_field(response, key)
        if value is None:
            return None
        try:
            return float(value)
        except ValueError:
            return None

    @staticmethod
    def _parse_pair_summary_response(prefix: bytes, response: str) -> NamespaceSummary | None:
        fields = CheetahClient._split_response_fields(response)
        if not fields:
            return None
        terminal_count = CheetahClient._parse_int(fields.get("count"), default=0)
        total_payload = CheetahClient._parse_int(fields.get("total_payload_bytes"), default=0)
        min_payload = CheetahClient._parse_int(fields.get("min_payload_bytes"), default=0)
        max_payload = CheetahClient._parse_int(fields.get("max_payload_bytes"), default=0)
        max_depth = CheetahClient._parse_int(fields.get("max_depth"), default=0)
        min_key = CheetahClient._parse_optional_int(fields.get("min_key"))
        max_key = CheetahClient._parse_optional_int(fields.get("max_key"))
        self_terminal = CheetahClient._parse_bool(fields.get("self_terminal"))
        branches: list[tuple[bytes, int]] = []
        raw_branches = fields.get("branches")
        if raw_branches:
            for chunk in raw_branches.split(";"):
                if not chunk:
                    continue
                try:
                    path_hex, count_text = chunk.split(":", 1)
                except ValueError:
                    continue
                try:
                    branch_bytes = bytes.fromhex(path_hex)
                except ValueError:
                    branch_bytes = path_hex.encode("utf-8", errors="ignore")
                branches.append(
                    (branch_bytes, CheetahClient._parse_int(count_text, default=0))
                )
        return NamespaceSummary(
            prefix=prefix,
            terminal_count=terminal_count,
            total_payload_bytes=total_payload,
            min_payload_bytes=min_payload,
            max_payload_bytes=max_payload,
            min_key=min_key,
            max_key=max_key,
            max_depth=max_depth,
            self_terminal=self_terminal,
            branches=tuple(branches),
        )

    @staticmethod
    def _parse_system_stats_response(response: str) -> CheetahSystemStats | None:
        fields = CheetahClient._split_response_fields(response)
        if not fields:
            return None
        raw_hints = fields.get("recommended_workers", "")
        recommended: list[tuple[int, int]] = []
        if raw_hints:
            for item in raw_hints.split(";"):
                if not item:
                    continue
                try:
                    pending_text, worker_text = item.split(":", 1)
                    recommended.append((int(pending_text), int(worker_text)))
                except ValueError:
                    continue
        return CheetahSystemStats(
            logical_cores=CheetahClient._parse_int(fields.get("logical_cores"), default=0),
            gomaxprocs=CheetahClient._parse_int(fields.get("gomaxprocs"), default=0),
            goroutines=CheetahClient._parse_int(fields.get("goroutines"), default=0),
            mem_alloc_bytes=CheetahClient._parse_int(fields.get("mem_alloc_bytes"), default=0),
            mem_sys_bytes=CheetahClient._parse_int(fields.get("mem_sys_bytes"), default=0),
            process_cpu_pct=CheetahClient._parse_float(fields.get("process_cpu_pct")),
            system_cpu_pct=CheetahClient._parse_float(fields.get("system_cpu_pct")),
            process_cpu_supported=CheetahClient._parse_bool(
                fields.get("process_cpu_supported")
            ),
            system_cpu_supported=CheetahClient._parse_bool(
                fields.get("system_cpu_supported")
            ),
            io_supported=CheetahClient._parse_bool(fields.get("io_supported")),
            io_read_bytes_per_sec=CheetahClient._parse_float(
                fields.get("io_read_bytes_per_sec")
            ),
            io_write_bytes_per_sec=CheetahClient._parse_float(
                fields.get("io_write_bytes_per_sec")
            ),
            timestamp=fields.get("timestamp"),
            recommended_workers=tuple(recommended),
            payload_cache_enabled=CheetahClient._parse_bool(
                fields.get("payload_cache_enabled")
            ),
            payload_cache_entries=CheetahClient._parse_int(
                fields.get("payload_cache_entries"), default=0
            ),
            payload_cache_max_entries=CheetahClient._parse_int(
                fields.get("payload_cache_max_entries"), default=0
            ),
            payload_cache_bytes=CheetahClient._parse_int(
                fields.get("payload_cache_bytes"), default=0
            ),
            payload_cache_max_bytes=CheetahClient._parse_int(
                fields.get("payload_cache_max_bytes"), default=0
            ),
            payload_cache_hits=CheetahClient._parse_int(fields.get("payload_cache_hits"), default=0),
            payload_cache_misses=CheetahClient._parse_int(
                fields.get("payload_cache_misses"), default=0
            ),
            payload_cache_evictions=CheetahClient._parse_int(
                fields.get("payload_cache_evictions"), default=0
            ),
            payload_cache_hit_pct=CheetahClient._parse_float(
                fields.get("payload_cache_hit_pct")
            ),
            payload_cache_advisory_bypass_bytes=CheetahClient._parse_optional_int(
                fields.get("payload_cache_advisory_bypass_bytes")
            ),
        )

    @staticmethod
    def _split_response_fields(response: str) -> dict[str, str]:
        fields: dict[str, str] = {}
        for part in response.split(","):
            if "=" not in part:
                continue
            key, value = part.split("=", 1)
            fields[key.strip()] = value.strip()
        return fields

    @staticmethod
    def _parse_int(raw: str | None, *, default: int = 0) -> int:
        if raw is None or raw == "":
            return default
        try:
            return int(raw, 10)
        except ValueError:
            return default

    @staticmethod
    def _parse_optional_int(raw: str | None) -> int | None:
        if raw is None or raw == "":
            return None
        try:
            return int(raw, 10)
        except ValueError:
            return None

    @staticmethod
    def _parse_float(raw: str | None) -> float | None:
        if raw is None or raw.strip().upper() == "NA" or raw == "":
            return None
        try:
            return float(raw)
        except ValueError:
            return None

    @staticmethod
    def _parse_bool(raw: str | None) -> bool:
        if raw is None:
            return False
        return raw.strip().lower() in {"1", "true", "yes"}

    @staticmethod
    def _normalize_prediction_value(value: bytes | str | bytearray) -> bytes:
        if isinstance(value, (bytes, bytearray)):
            return bytes(value)
        return str(value).encode("utf-8")

    @classmethod
    def _format_prediction_value(cls, value: bytes | str | bytearray) -> str | None:
        normalized = cls._normalize_prediction_value(value)
        if not normalized:
            return None
        return f"x{normalized.hex()}"

    @staticmethod
    def _encode_matrix_payload(matrix: Sequence[Sequence[float]] | dict[str, object] | None) -> str | None:
        if not matrix:
            return None
        if isinstance(matrix, dict):
            rows = matrix.get("rows") or matrix.get("matrix") or []
            weights = matrix.get("weights") or []
            serializable_rows: list[list[float]] = []
            for row in rows:
                if not row:
                    continue
                try:
                    serializable_rows.append([float(component) for component in row])
                except (TypeError, ValueError):
                    continue
            if not serializable_rows:
                return None
            payload: dict[str, object] = {"rows": serializable_rows}
            serializable_weights: list[float] = []
            for weight in weights:
                try:
                    serializable_weights.append(float(weight))
                except (TypeError, ValueError):
                    continue
            if serializable_weights:
                payload["weights"] = serializable_weights
            encoded = json.dumps(payload, separators=(",", ":")).encode("utf-8")
            return base64.b64encode(encoded).decode("ascii")
        serializable: list[list[float]] = []
        for row in matrix:
            if not row:
                continue
            try:
                serializable.append([float(component) for component in row])
            except (TypeError, ValueError):
                continue
        if not serializable:
            return None
        payload = json.dumps(serializable, separators=(",", ":")).encode("utf-8")
        return base64.b64encode(payload).decode("ascii")

    @classmethod
    def _encode_key_windows_payload(
        cls,
        specs: Sequence[tuple[bytes | str, Sequence[Sequence[float]]]] | None,
    ) -> str | None:
        if not specs:
            return None
        serialized: list[dict[str, object]] = []
        for key, windows in specs:
            formatted_key = cls._format_prediction_value(key)
            if not formatted_key:
                continue
            if not windows:
                continue
            normalized_windows: list[list[float]] = []
            for window in windows:
                if not window:
                    continue
                try:
                    normalized_windows.append([float(value) for value in window])
                except (TypeError, ValueError):
                    continue
            if not normalized_windows:
                continue
            serialized.append({"key": formatted_key, "windows": normalized_windows})
        if not serialized:
            return None
        payload = json.dumps(serialized, separators=(",", ":")).encode("utf-8")
        return base64.b64encode(payload).decode("ascii")

    @classmethod
    def _encode_inherit_batch_payload(
        cls,
        items: Sequence[dict[str, object]] | None,
    ) -> str | None:
        if not items:
            return None
        serialized: list[dict[str, object]] = []
        for item in items:
            if not item:
                continue
            key = item.get("key")
            target = item.get("target")
            sources = item.get("sources") or []
            merge = item.get("merge") or item.get("mode")
            formatted_key = cls._format_prediction_value(key) if key is not None else None
            formatted_target = (
                cls._format_prediction_value(target) if target is not None else None
            )
            if not formatted_key or not formatted_target:
                continue
            if isinstance(sources, (bytes, str, bytearray)):
                sources_iter = [sources]
            else:
                try:
                    sources_iter = list(sources)
                except TypeError:
                    sources_iter = []
            formatted_sources: list[str] = []
            for source in sources_iter:
                formatted = cls._format_prediction_value(source)
                if formatted:
                    formatted_sources.append(formatted)
            if not formatted_sources:
                continue
            spec: dict[str, object] = {
                "key": formatted_key,
                "target": formatted_target,
                "sources": formatted_sources,
            }
            if merge:
                spec["merge"] = merge
            serialized.append(spec)
        if not serialized:
            return None
        payload = json.dumps(serialized, separators=(",", ":")).encode("utf-8")
        return base64.b64encode(payload).decode("ascii")


@dataclass(frozen=True)
class TopKPayload:
    order: int
    ranked: list[tuple[int, int]]


@dataclass(frozen=True)
class ContextPayload:
    order_size: int
    token_ids: tuple[int, ...]


@dataclass(frozen=True)
class CountsPayload:
    order: int
    followers: tuple[tuple[int, int], ...]


@dataclass(frozen=True)
class ProbabilityPayload:
    order: int
    entries: tuple[tuple[int, int, int | None], ...]


@dataclass(frozen=True)
class ContinuationPayload:
    token_id: int
    num_contexts: int


class CheetahSerializer:
    """Binary codec for the cheetah hot-path payloads."""

    CONTEXT_VERSION = 1
    TOPK_VERSION = 1
    COUNTS_VERSION = 1
    PROBABILITY_VERSION = 1
    CONTINUATION_VERSION = 1
    MAX_TOPK = 32

    def encode_context(self, order_size: int, token_ids: Sequence[int]) -> bytes:
        if order_size > 255:
            raise CheetahError("order_size exceeds single-byte limit")
        if len(token_ids) > 254:
            raise CheetahError("token sequence too long for cheetah payload")
        buf = bytearray()
        buf.append(self.CONTEXT_VERSION)
        buf.append(order_size)
        buf.append(len(token_ids))
        for token_id in token_ids:
            buf.extend(struct.pack(">I", int(token_id)))
        return bytes(buf)

    def decode_context(self, payload: bytes) -> ContextPayload | None:
        if not payload or payload[0] != self.CONTEXT_VERSION or len(payload) < 3:
            return None
        order_size = payload[1]
        count = payload[2]
        expected = 3 + count * 4
        if len(payload) < expected:
            return None
        token_ids: list[int] = []
        offset = 3
        for _ in range(count):
            token_ids.append(struct.unpack(">I", payload[offset : offset + 4])[0])
            offset += 4
        return ContextPayload(order_size=order_size, token_ids=tuple(token_ids))

    def encode_counts(self, order: int, followers: Sequence[tuple[int, int]]) -> bytes:
        if order > 255:
            raise CheetahError("order exceeds single-byte limit for counts payload")
        follower_count = min(len(followers), 65535)
        buf = bytearray()
        buf.append(self.COUNTS_VERSION)
        buf.append(order & 0xFF)
        buf.extend(struct.pack(">H", follower_count))
        for token_id, count in followers[:follower_count]:
            buf.extend(struct.pack(">I", int(token_id)))
            buf.extend(struct.pack(">I", max(int(count), 0)))
        return bytes(buf)

    def decode_counts(self, payload: bytes) -> CountsPayload | None:
        if not payload or payload[0] != self.COUNTS_VERSION or len(payload) < 4:
            return None
        order = payload[1]
        follower_count = struct.unpack(">H", payload[2:4])[0]
        expected = 4 + follower_count * 8
        if len(payload) < expected:
            return None
        followers: list[tuple[int, int]] = []
        offset = 4
        for _ in range(follower_count):
            token_id = struct.unpack(">I", payload[offset : offset + 4])[0]
            offset += 4
            count = struct.unpack(">I", payload[offset : offset + 4])[0]
            offset += 4
            followers.append((token_id, count))
        return CountsPayload(order=order, followers=tuple(followers))

    def encode_topk(self, order: int, ranked: Sequence[tuple[int, int]]) -> bytes:
        if order > 255:
            raise CheetahError("order exceeds single-byte limit")
        buf = bytearray()
        buf.append(self.TOPK_VERSION)
        buf.append(order)
        clamped = list(ranked[: self.MAX_TOPK])
        actual_count = len(clamped)
        buf.append(actual_count)
        for idx in range(self.MAX_TOPK):
            if idx < actual_count:
                token_id, q = clamped[idx]
            else:
                token_id, q = 0, 0
            buf.extend(struct.pack(">I", int(token_id)))
            buf.append(int(q) & 0xFF)
        return bytes(buf)

    def decode_topk(self, payload: bytes) -> TopKPayload | None:
        if not payload or payload[0] != self.TOPK_VERSION or len(payload) < 3:
            return None
        order = payload[1]
        count = payload[2]
        expected = 3 + count * 5
        if len(payload) < expected:
            return None
        ranked: list[tuple[int, int]] = []
        offset = 3
        for _ in range(count):
            token_id = struct.unpack(">I", payload[offset : offset + 4])[0]
            offset += 4
            q = payload[offset]
            offset += 1
            ranked.append((token_id, q))
        return TopKPayload(order=order, ranked=ranked)

    def encode_probabilities(
        self,
        order: int,
        entries: Sequence[tuple[int, int, int | None]],
    ) -> bytes:
        if order > 255:
            raise CheetahError("order exceeds single-byte limit for probability payload")
        buf = bytearray()
        buf.append(self.PROBABILITY_VERSION)
        buf.append(order & 0xFF)
        count = min(len(entries), 65535)
        buf.extend(struct.pack(">H", count))
        for token_id, q_logprob, backoff in entries[:count]:
            buf.extend(struct.pack(">I", int(token_id)))
            buf.append(int(q_logprob) & 0xFF)
            if backoff is None:
                buf.extend(struct.pack(">H", 0xFFFF))
            else:
                buf.extend(struct.pack(">H", max(0, min(int(backoff), 0xFFFF))))
        return bytes(buf)

    def decode_probabilities(self, payload: bytes) -> ProbabilityPayload | None:
        if not payload or payload[0] != self.PROBABILITY_VERSION or len(payload) < 4:
            return None
        order = payload[1]
        count = struct.unpack(">H", payload[2:4])[0]
        expected = 4 + count * 7
        if len(payload) < expected:
            return None
        entries: list[tuple[int, int, int | None]] = []
        offset = 4
        for _ in range(count):
            token_id = struct.unpack(">I", payload[offset : offset + 4])[0]
            offset += 4
            q_logprob = payload[offset]
            offset += 1
            alpha_raw = struct.unpack(">H", payload[offset : offset + 2])[0]
            offset += 2
            backoff = None if alpha_raw == 0xFFFF else alpha_raw
            entries.append((token_id, q_logprob, backoff))
        return ProbabilityPayload(order=order, entries=tuple(entries))

    def encode_continuation(self, token_id: int, num_contexts: int) -> bytes:
        buf = bytearray()
        buf.append(self.CONTINUATION_VERSION)
        buf.extend(struct.pack(">I", int(token_id)))
        buf.extend(struct.pack(">I", max(int(num_contexts), 0)))
        return bytes(buf)

    def decode_continuation(self, payload: bytes) -> ContinuationPayload | None:
        if not payload or payload[0] != self.CONTINUATION_VERSION or len(payload) != 9:
            return None
        token_id = struct.unpack(">I", payload[1:5])[0]
        num_contexts = struct.unpack(">I", payload[5:9])[0]
        return ContinuationPayload(token_id=token_id, num_contexts=num_contexts)


class _ThreadLocalCheetahClientPool(ThreadLocalClientPool):
    """One cheetah-db client per thread.

    The behavior is generic and lives in the binder
    (:class:`~cheetah_db.client.ThreadLocalClientPool`); the name is kept here
    because it is the adapter's own vocabulary and appears in the handbook.
    """


class CheetahHotPathAdapter(HotPathAdapter):
    """Mirrors context metadata + Top-K slices into cheetah-db for low-latency reads."""

    def __init__(
        self,
        client: CheetahClient | None = None,
        *,
        client_factory: Callable[[], CheetahClient] | None = None,
        cache_size: int = 50000,
        serializer: CheetahSerializer | None = None,
        description: str | None = None,
    ) -> None:
        if client_factory is None and client is None:
            raise ValueError("Provide either client or client_factory.")
        if client_factory is None and client is not None:
            client_factory = lambda: client
        assert client_factory is not None
        self._client_pool = _ThreadLocalCheetahClientPool(
            client_factory,
            warm_client=client,
            description=description,
        )
        self._serializer = serializer or CheetahSerializer()
        self._vector_order = AbsoluteVectorOrder()
        self._key_cache: "OrderedDict[tuple[str, str], int]" = OrderedDict()
        self._cache_size = max(cache_size, 1024)
        self._enabled = True
        self._topk_total = 0
        self._topk_hits = 0
        self._description = self._client_pool.describe()
        self._reduce_page_limit = DEFAULT_REDUCE_PAGE_SIZE
        self._reduce_page_limit_deadline = 0.0
        self._async_workers = max(4, min(32, (os.cpu_count() or 4)))
        self._async_executor = ThreadPoolExecutor(
            max_workers=self._async_workers,
            thread_name_prefix="cheetah_hotpath",
        )
        self._async_lock = threading.Lock()
        self._async_futures: deque[Future] = deque()
        self._async_max_pending = self._async_workers * 4
        attempts = os.environ.get("CHEETAH_PAIR_REGISTER_ATTEMPTS", "").strip()
        try:
            parsed_attempts = int(attempts) if attempts else 4
        except ValueError:
            parsed_attempts = 4
        self._pair_register_attempts = max(1, parsed_attempts)
        backoff_raw = os.environ.get("CHEETAH_PAIR_REGISTER_BACKOFF_SECONDS", "").strip()
        try:
            self._pair_register_backoff = max(
                0.05, float(backoff_raw) if backoff_raw else 0.25
            )
        except ValueError:
            self._pair_register_backoff = 0.25
        batch_raw = os.environ.get("CHEETAH_PAIR_BATCH_SIZE", "").strip()
        try:
            parsed_batch = int(batch_raw) if batch_raw else 256
        except ValueError:
            parsed_batch = 256
        # The server refuses more than 10,000 items per request (pair_batch.go);
        # a few hundred already fill the connection.
        self._pair_batch_size = max(1, min(parsed_batch, PAIR_PUT_BATCH_MAX_ITEMS))
        reducer_retry_raw = os.environ.get("CHEETAH_REDUCER_RETRY_ATTEMPTS", "").strip()
        try:
            parsed_reducer_retry = int(reducer_retry_raw) if reducer_retry_raw else 3
        except ValueError:
            parsed_reducer_retry = 3
        self._reducer_retry_attempts = max(1, parsed_reducer_retry)
        reducer_delay_raw = os.environ.get("CHEETAH_REDUCER_RETRY_DELAY_SECONDS", "").strip()
        try:
            self._reducer_retry_delay = max(
                0.0, float(reducer_delay_raw) if reducer_delay_raw else 5.0
            )
        except ValueError:
            self._reducer_retry_delay = 5.0

    # ------------------------------------------------------------------ #
    # HotPathAdapter API
    # ------------------------------------------------------------------ #
    def publish_context(self, context_hash: str, order_size: int, token_ids: Sequence[int]) -> None:
        if not self._enabled:
            return
        namespace = "ctx"
        if self._lookup_key(namespace, context_hash=context_hash) is not None:
            return
        payload = self._serializer.encode_context(order_size, token_ids)
        try:
            key = self._insert(namespace, context_hash, payload)
            self._register_vector_alias(key, token_ids)
        except CheetahError as exc:
            self._disable(exc)

    def publish_topk(self, order: int, context_hash: str, ranked: Sequence[tuple[int, int]]) -> None:
        if not self._enabled or not ranked:
            return
        namespace = f"topk:{order}"
        payload = self._serializer.encode_topk(order, ranked)
        try:
            key = self._lookup_key(namespace, context_hash=context_hash)
            if key is None:
                self._submit_async(lambda: self._insert(namespace, context_hash, payload))
            else:
                self._submit_async(
                    lambda key=key, payload=payload: self._edit_or_reinsert(
                        namespace,
                        context_hash,
                        key,
                        payload,
                    )
                )
        except CheetahError as exc:
            self._disable(exc)

    def publish_counts(self, order: int, context_hash: str, followers: Sequence[tuple[int, int]]) -> None:
        if not self._enabled or not followers:
            return
        namespace = f"cnt:{order}"
        payload = self._serializer.encode_counts(order, followers)
        try:
            key = self._lookup_key(namespace, context_hash=context_hash)
            if key is None:
                self._submit_async(lambda: self._insert(namespace, context_hash, payload))
            else:
                self._submit_async(
                    lambda key=key, payload=payload: self._edit_or_reinsert(
                        namespace,
                        context_hash,
                        key,
                        payload,
                    )
                )
        except CheetahError as exc:
            self._disable(exc)

    def flush_pending(self) -> None:
        """Wait for asynchronous mirror writes without shutting down the adapter."""
        with self._async_lock:
            pending = list(self._async_futures)
            self._async_futures.clear()
        for future in pending:
            self._await_future(future)

    def publish_probabilities(
        self,
        order: int,
        context_hash: str,
        entries: Sequence[tuple[int, int, int | None]],
    ) -> None:
        if not self._enabled or not entries:
            return
        namespace = f"prob:{order}"
        payload = self._serializer.encode_probabilities(order, entries)
        try:
            key = self._lookup_key(namespace, context_hash=context_hash)
            if key is None:
                self._submit_async(lambda: self._insert(namespace, context_hash, payload))
            else:
                self._submit_async(
                    lambda key=key, payload=payload: self._edit_or_reinsert(
                        namespace,
                        context_hash,
                        key,
                        payload,
                    )
                )
        except CheetahError as exc:
            self._disable(exc)

    def publish_continuations(self, entries: Sequence[tuple[int, int]]) -> None:
        """Mirror per-token continuation counts.

        Unlike the other publish paths this one is called with a whole slice of
        the vocabulary at once, so the new rows are written through
        ``PAIR_PUT_BATCH`` — one request per page instead of two per token.
        Rows that already exist still take the edit path: the batch command
        only ever creates.
        """
        if not self._enabled or not entries:
            return
        namespace = "cont"
        fresh: list[tuple[str, bytes]] = []
        for token_id, num_contexts in entries:
            identifier = f"{int(token_id) & 0xFFFFFFFF:08x}"
            payload = self._serializer.encode_continuation(token_id, num_contexts)
            try:
                key = self._lookup_key(namespace, context_hash=identifier)
                if key is None:
                    fresh.append((identifier, payload))
                else:
                    self._submit_async(
                        lambda key=key, identifier=identifier, payload=payload: self._edit_or_reinsert(
                            namespace,
                            identifier,
                            key,
                            payload,
                        )
                    )
            except CheetahError as exc:
                self._disable(exc)
                return
        if not fresh:
            return
        try:
            self._publish_new_rows(namespace, fresh)
        except CheetahError as exc:
            self._disable(exc)

    def fetch_topk(self, order: int, context_hash: str, limit: int) -> list[tuple[int, int]] | None:
        if not self._enabled:
            return None
        self._topk_total += 1
        namespace = f"topk:{order}"
        key = self._lookup_key(namespace, context_hash=context_hash)
        if key is None:
            return None
        payload = self._client.read(key)
        if not payload:
            return None
        record = self._serializer.decode_topk(payload)
        if not record or record.order != order:
            return None
        self._topk_hits += 1
        return record.ranked[:limit]

    def fetch_context_tokens(self, context_hash: str) -> Sequence[int] | None:
        if not self._enabled:
            return None
        namespace = "ctx"
        key = self._lookup_key(namespace, context_hash=context_hash)
        if key is None:
            return None
        payload = self._client.read(key)
        if not payload:
            return None
        record = self._serializer.decode_context(payload)
        if not record:
            return None
        return list(record.token_ids)

    def write_metadata(self, key: str, value: str) -> None:
        if not self._enabled:
            return
        namespace = "meta"
        raw_value = key.encode("utf-8")
        payload = value.encode("utf-8")
        try:
            existing = self._lookup_key(namespace, raw_value=raw_value)
            if existing is None:
                new_key, response = self._client.insert(payload)
                if new_key is None:
                    logger.error(
                        "cheetah insert failed for metadata key=%s response=%s",
                        key,
                        response,
                    )
                    raise CheetahError("failed to insert metadata payload")
                self._register_pair(namespace, new_key, raw_value=raw_value)
            else:
                success, response = self._client.edit(existing, payload)
                if not success:
                    logger.error(
                        "cheetah edit failed for metadata key=%s entry=%s response=%s",
                        key,
                        existing,
                        response,
                    )
                    logger.info("Reinserting metadata key=%s after failed edit", key)
                    replacement_key, insert_response = self._client.insert(payload)
                    if replacement_key is None:
                        logger.error(
                            "cheetah insert retry failed for metadata key=%s response=%s",
                            key,
                            insert_response,
                        )
                        raise CheetahError("failed to edit metadata payload")
                    self._register_pair(namespace, replacement_key, raw_value=raw_value)
        except CheetahError as exc:
            self._disable(exc)

    def refresh_context_predictions(
        self,
        metadata_key: str,
        matrix: Sequence[Sequence[float]],
        payload: str,
    ) -> None:
        if not self._enabled or not matrix:
            return
        raw_key = f"meta:{metadata_key}".encode("utf-8")
        table_name = _CONTEXT_MATRIX_TABLE
        try:
            success, response = self._client.predict_set(
                raw_key,
                payload.encode("utf-8"),
                probability=1.0,
                table=table_name,
            )
            if not success:
                logger.warning(
                    "cheetah predict_set failed for metadata key=%s response=%s",
                    metadata_key,
                    response,
                )
                raise CheetahError("failed to upsert context prediction entry")
            success, response = self._client.predict_ctx(
                raw_key,
                matrix,
                table=table_name,
                mode="bias",
                strength=1.0,
            )
            if not success:
                logger.warning(
                    "cheetah predict_ctx failed for metadata key=%s response=%s",
                    metadata_key,
                    response,
                )
                raise CheetahError("failed to refresh context matrix weights")
        except CheetahError as exc:
            self._disable(exc)

    def read_metadata(self, key: str) -> str | None:
        if not self._enabled:
            return None
        namespace = "meta"
        raw_value = key.encode("utf-8")
        entry_key = self._lookup_key(namespace, raw_value=raw_value)
        if entry_key is None:
            return None
        payload = self._client.read(entry_key)
        if not payload:
            return None
        return payload.decode("utf-8", "replace")

    def scan_namespace(
        self,
        namespace: str,
        *,
        prefix: bytes = b"",
        limit: int = 0,
    ) -> Iterable[tuple[bytes, int]]:
        if not self._enabled:
            return []
        namespace_bytes = namespace.encode("utf-8") + b":"
        scoped_prefix = namespace_bytes + prefix
        trimmed: list[tuple[bytes, int]] = []
        cursor: bytes | None = None
        remaining = limit if limit > 0 else None
        while True:
            page_limit = remaining if remaining is not None else 0
            result = self._client.pair_scan(scoped_prefix, limit=page_limit, cursor=cursor)
            if result is None:
                self._disable(CheetahError("pair_scan failed"))
                return []
            entries, cursor = result
            if not entries:
                break
            for raw_value, key in entries:
                if not raw_value.startswith(namespace_bytes):
                    continue
                trimmed.append((raw_value[len(namespace_bytes) :], key))
                if remaining is not None:
                    remaining -= 1
                    if remaining <= 0:
                        return trimmed
            if not cursor:
                break
        return trimmed

    def _pair_reduce_with_retry(
        self,
        mode: str,
        namespace: bytes,
        *,
        limit: int,
        cursor: bytes | None,
    ) -> tuple[list[tuple[bytes, int, bytes | None]], bytes | None] | None:
        namespace_label = namespace.decode("utf-8", "replace")
        namespace_label = namespace_label.rstrip(":") or "<root>"
        attempts = self._reducer_retry_attempts
        delay = self._reducer_retry_delay
        client = self._client
        for attempt in range(1, attempts + 1):
            result = client.pair_reduce(
                mode,
                namespace,
                limit=limit,
                cursor=cursor,
            )
            if result is not None:
                return result
            wait = delay if attempt < attempts else 0.0
            logger.warning(
                "cheetah pair_reduce %s namespace=%s attempt %s/%s timed out; %s",
                mode,
                namespace_label,
                attempt,
                attempts,
                "retrying" if wait else "giving up",
            )
            client.close()
            if wait > 0:
                time.sleep(wait)
        return None

    def iter_counts(self, order: int) -> list[RawCountsProjection]:
        if not self._enabled:
            return []
        namespace = f"cnt:{order}"
        projections: list[RawCountsProjection] = []
        namespace_bytes = namespace.encode("utf-8") + b":"
        cursor: bytes | None = None
        page_limit = self._recommended_reduce_page_size()
        while True:
            result = self._pair_reduce_with_retry(
                "counts",
                namespace_bytes,
                limit=page_limit,
                cursor=cursor,
            )
            if result is None:
                self._disable(CheetahError("pair_reduce counts failed"))
                return []
            entries, cursor = result
            if not entries:
                break
            for raw_value, key, payload in entries:
                if not raw_value.startswith(namespace_bytes):
                    continue
                trimmed = raw_value[len(namespace_bytes) :]
                blob = payload
                if blob is not None:
                    blob = CheetahClient.decode_reduced_payload(blob)
                else:
                    blob = self._client.read(key)
                if not blob:
                    continue
                record = self._serializer.decode_counts(blob)
                if not record or record.order != order:
                    continue
                if trimmed == b"__root__":
                    context_hash = "__root__"
                else:
                    context_hash = trimmed.hex()
                totals = sum(count for _, count in record.followers)
                projections.append(
                    RawCountsProjection(
                        context_hash=context_hash,
                        order=order,
                        totals=totals,
                        followers=record.followers,
                    )
                )
            if not cursor:
                break
        return projections

    def iter_probabilities(self, order: int) -> list[RawProbabilityProjection]:
        if not self._enabled:
            return []
        namespace = f"prob:{order}"
        namespace_bytes = namespace.encode("utf-8") + b":"
        projections: list[RawProbabilityProjection] = []
        cursor: bytes | None = None
        page_limit = self._recommended_reduce_page_size()
        while True:
            result = self._pair_reduce_with_retry(
                "probabilities",
                namespace_bytes,
                limit=page_limit,
                cursor=cursor,
            )
            if result is None:
                self._disable(CheetahError("pair_reduce probabilities failed"))
                return []
            entries, cursor = result
            if not entries:
                break
            for raw_value, key, payload in entries:
                if not raw_value.startswith(namespace_bytes):
                    continue
                trimmed = raw_value[len(namespace_bytes) :]
                blob = payload
                if blob is not None:
                    blob = CheetahClient.decode_reduced_payload(blob)
                else:
                    blob = self._client.read(key)
                if not blob:
                    continue
                record = self._serializer.decode_probabilities(blob)
                if not record or record.order != order:
                    continue
                if trimmed == b"__root__":
                    context_hash = "__root__"
                else:
                    context_hash = trimmed.hex()
                projections.append(
                    RawProbabilityProjection(
                        context_hash=context_hash,
                        order=order,
                        followers=record.entries,
                    )
                )
            if not cursor:
                break
        return projections

    def iter_continuations(self) -> list[RawContinuationProjection]:
        if not self._enabled:
            return []
        namespace = "cont"
        namespace_bytes = namespace.encode("utf-8") + b":"
        projections: list[RawContinuationProjection] = []
        cursor: bytes | None = None
        page_limit = self._recommended_reduce_page_size()
        while True:
            result = self._pair_reduce_with_retry(
                "continuations",
                namespace_bytes,
                limit=page_limit,
                cursor=cursor,
            )
            if result is None:
                self._disable(CheetahError("pair_reduce continuations failed"))
                return []
            entries, cursor = result
            if not entries:
                break
            for raw_value, key, payload in entries:
                if not raw_value.startswith(namespace_bytes):
                    continue
                blob = payload
                if blob is not None:
                    blob = CheetahClient.decode_reduced_payload(blob)
                else:
                    blob = self._client.read(key)
                if not blob:
                    continue
                record = self._serializer.decode_continuation(blob)
                if not record:
                    continue
                projections.append(
                    RawContinuationProjection(
                        token_id=record.token_id,
                        num_contexts=record.num_contexts,
                    )
                )
            if not cursor:
                break
        return projections

    def context_relativism(
        self,
        context_tree,
        *,
        limit: int = 32,
        depth: int | None = None,
    ) -> list[RawContextProjection]:
        if not self._enabled:
            return []
        try:
            vector_prefix = self._vector_order.encode_tree(context_tree, depth_limit=depth)
        except (TypeError, ValueError) as exc:
            logger.debug("failed to encode context tree for relativism query: %s", exc)
            return []
        matches = self.scan_namespace("ctxv", prefix=vector_prefix, limit=limit)
        projections: list[RawContextProjection] = []
        for vector_bytes, key in matches:
            payload = self._client.read(key)
            if not payload:
                continue
            context_record = self._serializer.decode_context(payload)
            if not context_record:
                continue
            tokens = tuple(context_record.token_ids)
            context_hash = hash_tokens(tokens)
            ranked = self.fetch_topk(
                context_record.order_size,
                context_hash,
                self._serializer.MAX_TOPK,
            ) or []
            projections.append(
                RawContextProjection(
                    context_hash=context_hash,
                    order_size=context_record.order_size,
                    token_ids=tokens,
                    ranked=tuple(ranked),
                    cheetah_key=key,
                    vector_signature=bytes(vector_bytes),
                )
            )
        return projections

    def topk_hit_ratio(self) -> float:
        if self._topk_total <= 0:
            return 0.0
        return min(1.0, max(0.0, self._topk_hits / float(self._topk_total)))

    def describe(self) -> str:
        return self._description

    def namespace_summary(
        self,
        prefix: bytes,
        *,
        depth: int = 1,
        branch_limit: int = 32,
    ) -> NamespaceSummary | None:
        if not self._enabled:
            return None
        try:
            return self._client.pair_summary(prefix, depth=depth, branch_limit=branch_limit)
        except CheetahError as exc:
            self._disable(exc)
            return None

    def system_stats(self) -> CheetahSystemStats | None:
        if not self._enabled:
            return None
        try:
            return self._client.system_stats()
        except CheetahError as exc:
            self._disable(exc)
            return None

    # ------------------------------------------------------------------ #
    # Graph context memory
    # ------------------------------------------------------------------ #
    def graph_node_set(
        self,
        node_id: str,
        *,
        labels: Sequence[str] | None = None,
        props: dict[str, object] | None = None,
        references: Sequence[dict[str, object]] | None = None,
        clear_references: bool = False,
    ) -> bool:
        if not self._enabled:
            return False
        try:
            return self._client.graph_node_set(
                node_id,
                labels=labels,
                props=props,
                references=references,
                clear_references=clear_references,
            )
        except CheetahError as exc:
            self._disable(exc)
            return False

    def graph_node_get(self, node_id: str) -> GraphNodeRecord | None:
        if not self._enabled:
            return None
        try:
            return self._client.graph_node_get(node_id)
        except CheetahError as exc:
            self._disable(exc)
            return None

    def graph_edge_set_batch(
        self,
        items: Sequence[dict[str, object]],
        *,
        continue_on_error: bool = True,
        default_type: str | None = None,
        default_props: dict[str, object] | None = None,
    ) -> GraphEdgeBatchResult | None:
        if not self._enabled:
            return None
        try:
            return self._client.graph_edge_set_batch(
                items,
                continue_on_error=continue_on_error,
                default_type=default_type,
                default_props=default_props,
            )
        except CheetahError as exc:
            self._disable(exc)
            return None

    def graph_recall(
        self,
        seeds: Sequence[str],
        *,
        precision: float | str | None = None,
        hops: int | None = None,
        min_sources: int | None = None,
        direction: str | None = None,
        edge_types: Sequence[str] | None = None,
        decay: float | None = None,
        expand: str | None = None,
        references: bool = False,
        reference_limit: int | None = None,
        include_seeds: bool = False,
        limit: int | None = None,
        branch_limit: int | None = None,
        budget: int | None = None,
    ) -> GraphRecallResult | None:
        if not self._enabled:
            return None
        try:
            return self._client.graph_recall(
                seeds,
                precision=precision,
                hops=hops,
                min_sources=min_sources,
                direction=direction,
                edge_types=edge_types,
                decay=decay,
                expand=expand,
                references=references,
                reference_limit=reference_limit,
                include_seeds=include_seeds,
                limit=limit,
                branch_limit=branch_limit,
                budget=budget,
            )
        except CheetahError as exc:
            self._disable(exc)
            return None

    def graph_similar(
        self,
        node_id: str,
        *,
        by: str | None = None,
        limit: int | None = None,
        precision: float | str | None = None,
    ) -> GraphSimilarResult | None:
        if not self._enabled:
            return None
        try:
            return self._client.graph_similar(node_id, by=by, limit=limit, precision=precision)
        except CheetahError as exc:
            self._disable(exc)
            return None

    def graph_term_index(
        self,
        action: str = "stats",
        *,
        limit: int | None = None,
        cursor: str | None = None,
    ) -> GraphTermIndexStats | None:
        if not self._enabled:
            return None
        try:
            return self._client.graph_term_index(action, limit=limit, cursor=cursor)
        except CheetahError as exc:
            self._disable(exc)
            return None

    def predict_query(
        self,
        *,
        key: bytes | str | None = None,
        keys: Sequence[bytes | str] | None = None,
        context_matrix: Sequence[Sequence[float]] | dict[str, object] | None = None,
        windows: Sequence[Sequence[float]] | None = None,
        key_windows: Sequence[tuple[bytes | str, Sequence[Sequence[float]]]] | None = None,
        merge_mode: str | None = None,
        table: str | None = None,
    ) -> PredictionQueryResult | None:
        if not self._enabled:
            return None
        try:
            return self._client.predict_query(
                key=key,
                keys=keys,
                context_matrix=context_matrix,
                windows=windows,
                key_windows=key_windows,
                merge_mode=merge_mode,
                table=table,
            )
        except CheetahError as exc:
            self._disable(exc)
            return None

    def predict_set(
        self,
        *,
        key: bytes | str,
        value: bytes | str,
        probability: float = 0.5,
        table: str | None = None,
        weights: Sequence[dict[str, object]] | None = None,
    ) -> bool:
        if not self._enabled:
            return False
        key_bytes = self._ensure_bytes(key)
        value_bytes = self._ensure_bytes(value)
        try:
            success, response = self._client.predict_set(
                key_bytes,
                value_bytes,
                probability=probability,
                table=table,
                weights=weights,
            )
            if not success:
                logger.warning(
                    "cheetah predict_set failed for key=%s table=%s response=%s",
                    key,
                    table,
                    response,
                )
            return success
        except CheetahError as exc:
            self._disable(exc)
            return False

    def predict_train(
        self,
        *,
        key: bytes | str,
        target: bytes | str,
        context_matrix: Sequence[Sequence[float]] | dict[str, object] | None,
        learning_rate: float = 0.01,
        table: str | None = None,
        negatives: Sequence[bytes | str] | None = None,
    ) -> bool:
        if not self._enabled:
            return False
        key_bytes = self._ensure_bytes(key)
        target_bytes = self._ensure_bytes(target)
        try:
            success, response = self._client.predict_train(
                key_bytes,
                target_bytes,
                context_matrix=context_matrix,
                learning_rate=learning_rate,
                table=table,
                negatives=negatives,
            )
            if not success:
                logger.warning(
                    "cheetah predict_train failed for key=%s table=%s response=%s",
                    key,
                    table,
                    response,
                )
            return success
        except CheetahError as exc:
            self._disable(exc)
            return False

    def predict_inherit(
        self,
        *,
        key: bytes | str,
        target: bytes | str,
        sources: Sequence[bytes | str],
        table: str | None = None,
        merge_mode: str | None = None,
    ) -> bool:
        if not self._enabled:
            return False
        key_bytes = self._ensure_bytes(key)
        target_bytes = self._ensure_bytes(target)
        try:
            success, response = self._client.predict_inherit(
                key_bytes,
                target_bytes,
                sources,
                table=table,
                merge_mode=merge_mode,
            )
            if not success:
                logger.warning(
                    "cheetah predict_inherit failed for key=%s table=%s response=%s",
                    key,
                    table,
                    response,
                )
            return success
        except CheetahError as exc:
            self._disable(exc)
            return False

    def predict_inherit_batch(
        self,
        *,
        key: bytes | str,
        items: Sequence[dict[str, object]],
        table: str | None = None,
        merge_mode: str | None = None,
    ) -> bool:
        if not self._enabled:
            return False
        key_bytes = self._ensure_bytes(key)
        payload_items: list[dict[str, object]] = []
        for item in items or []:
            if not item:
                continue
            target = item.get("target")
            sources = item.get("sources") or []
            if target is None or not sources:
                continue
            payload = dict(item)
            payload["key"] = key_bytes
            payload_items.append(payload)
        if not payload_items:
            return False
        try:
            success, response = self._client.predict_inherit_batch(
                payload_items,
                table=table,
                merge_mode=merge_mode,
            )
            if not success:
                logger.warning(
                    "cheetah predict_inherit_batch failed for key=%s table=%s response=%s",
                    key,
                    table,
                    response,
                )
                return False
            job_id = self._client._extract_response_field(response or "", "job")
            if job_id:
                logger.warning(
                    "cheetah predict_inherit batch queued (job=%s table=%s count=%d)",
                    job_id,
                    table,
                    len(payload_items),
                )
            return True
        except CheetahError as exc:
            self._disable(exc)
            return False

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #
    def _insert(self, namespace: str, context_hash: str, payload: bytes) -> int:
        key, response = self._client.insert(payload)
        if key is None:
            logger.error(
                "cheetah insert failed for namespace=%s context_hash=%s response=%s",
                namespace,
                context_hash,
                response,
            )
            raise CheetahError("failed to insert cheetah payload")
        self._register_pair(namespace, key, context_hash=context_hash)
        return key

    @staticmethod
    def _ensure_bytes(value: bytes | str) -> bytes:
        if isinstance(value, bytes):
            return value
        if isinstance(value, str):
            return value.encode("utf-8")
        return bytes(value)

    def _register_pair(
        self,
        namespace: str,
        key: int,
        *,
        context_hash: str | None = None,
        raw_value: bytes | None = None,
    ) -> None:
        value = self._pair_value(namespace, context_hash=context_hash, raw_value=raw_value)
        identifier = self._normalize_identifier(context_hash=context_hash, raw_value=raw_value)
        attempts = getattr(self, "_pair_register_attempts", 1)
        backoff = getattr(self, "_pair_register_backoff", 0.25)
        client = self._client
        last_response: str | None = None
        for attempt in range(1, attempts + 1):
            success, response = client.pair_set(value, key)
            if success:
                if attempt > 1:
                    logger.info(
                        "cheetah pair_set recovered after %d attempt(s) for namespace=%s identifier=%s",
                        attempt,
                        namespace,
                        identifier or "<unknown>",
                    )
                self._set_cache(namespace, key, context_hash=context_hash, raw_value=raw_value)
                return
            last_response = response
            if self._confirm_pair_registration(value, key):
                logger.warning(
                    "cheetah pair_set missing acknowledgement for namespace=%s identifier=%s key=%s; "
                    "confirmed via PAIR_GET after %d attempt(s)",
                    namespace,
                    identifier or "<unknown>",
                    key,
                    attempt,
                )
                self._set_cache(namespace, key, context_hash=context_hash, raw_value=raw_value)
                return
            if attempt < attempts:
                delay = min(backoff * attempt, 2.0)
                logger.warning(
                    "cheetah pair_set attempt %d/%d failed for namespace=%s identifier=%s key=%s response=%s; "
                    "retrying in %.2fs",
                    attempt,
                    attempts,
                    namespace,
                    identifier or "<unknown>",
                    key,
                    response,
                    delay,
                )
                time.sleep(delay)
        logger.error(
            "cheetah pair_set failed after %d attempt(s) for namespace=%s identifier=%s key=%s response=%s",
            attempts,
            namespace,
            identifier or "<unknown>",
            key,
            last_response,
        )
        raise CheetahError("failed to register cheetah pair mapping")

    def _publish_new_rows(self, namespace: str, rows: Sequence[tuple[str, bytes]]) -> None:
        """Create several rows, batching the store+bind when the server can.

        A server without ``PAIR_PUT_BATCH`` (or a client stub that does not
        spell it) falls back to the per-row path, which is the same write in
        2N requests instead of one — slower, never different.
        """
        batch = getattr(self._client, "pair_put_batch", None)
        if not callable(batch) or len(rows) < 2:
            for identifier, payload in rows:
                self._submit_async(
                    lambda identifier=identifier, payload=payload: self._insert(
                        namespace, identifier, payload
                    )
                )
            return
        size = self._pair_batch_size
        for start in range(0, len(rows), size):
            page = list(rows[start : start + size])
            self._submit_async(lambda page=page: self._insert_batch(namespace, page))

    def _insert_batch(self, namespace: str, page: Sequence[tuple[str, bytes]]) -> None:
        """One ``PAIR_PUT_BATCH`` for a page, with a verified per-row fallback.

        The batch is not a transaction and reports its own accounting, so a
        page that did not fully apply is retried through ``_insert``, which
        confirms each binding with ``PAIR_GET``. Redoing an applied row is
        harmless (the name is rebound to a fresh value); leaving a hole in the
        namespace is not.
        """
        entries: list[tuple[bytes, bytes]] = []
        for identifier, payload in page:
            value = self._pair_value(namespace, context_hash=identifier)
            if value is None:
                raise CheetahError(f"cannot build a cheetah pair value for {namespace}:{identifier}")
            entries.append((value, payload))
        assigned = self._client.pair_put_batch(entries, want_keys=True)
        if assigned is None or len(assigned) != len(page):
            logger.warning(
                "cheetah pair_put_batch did not fully apply %d %s row(s); falling back to single writes",
                len(page),
                namespace,
            )
            for identifier, payload in page:
                self._insert(namespace, identifier, payload)
            return
        for (identifier, payload), key in zip(page, assigned):
            if key is None:
                self._insert(namespace, identifier, payload)
            else:
                self._set_cache(namespace, key, context_hash=identifier)

    def _register_vector_alias(self, key: int, token_ids: Sequence[int]) -> None:
        if not token_ids:
            return
        try:
            vector_bytes = self._vector_order.encode_tokens(token_ids)
        except (TypeError, ValueError) as exc:
            logger.debug("skipping vector alias for context: %s", exc)
            return
        self._register_pair("ctxv", key, raw_value=vector_bytes)

    def _lookup_key(
        self,
        namespace: str,
        *,
        context_hash: str | None = None,
        raw_value: bytes | None = None,
    ) -> int | None:
        cache_key = self._cache_key(namespace, context_hash=context_hash, raw_value=raw_value)
        if cache_key and cache_key in self._key_cache:
            key = self._key_cache[cache_key]
            self._key_cache.move_to_end(cache_key)
            return key
        value = self._pair_value(namespace, context_hash=context_hash, raw_value=raw_value)
        if value is None:
            return None
        key = self._client.pair_get(value)
        if key is not None:
            self._set_cache(namespace, key, context_hash=context_hash, raw_value=raw_value)
        return key

    def _set_cache(
        self,
        namespace: str,
        key: int,
        *,
        context_hash: str | None = None,
        raw_value: bytes | None = None,
    ) -> None:
        cache_key = self._cache_key(namespace, context_hash=context_hash, raw_value=raw_value)
        if cache_key is None:
            return
        self._key_cache[cache_key] = key
        self._key_cache.move_to_end(cache_key)
        if len(self._key_cache) > self._cache_size:
            self._key_cache.popitem(last=False)

    def _recommended_reduce_page_size(self) -> int:
        now = time.monotonic()
        if now < self._reduce_page_limit_deadline:
            return self._reduce_page_limit
        limit = DEFAULT_REDUCE_PAGE_SIZE
        stats = self.system_stats()
        if stats is not None:
            derived = stats.derive_reduce_page_limit()
            if derived is not None:
                limit = derived
        limit = max(PAIR_SCAN_MIN_LIMIT, min(PAIR_SCAN_MAX_LIMIT, limit))
        self._reduce_page_limit = limit
        self._reduce_page_limit_deadline = now + _REDUCE_LIMIT_CACHE_TTL_SECONDS
        return limit

    def _cache_key(
        self,
        namespace: str,
        *,
        context_hash: str | None = None,
        raw_value: bytes | None = None,
    ) -> tuple[str, str] | None:
        identifier = self._normalize_identifier(context_hash=context_hash, raw_value=raw_value)
        if identifier is None:
            return None
        return (namespace, identifier)

    def _pair_value(
        self,
        namespace: str,
        *,
        context_hash: str | None = None,
        raw_value: bytes | None = None,
    ) -> bytes | None:
        namespace_bytes = namespace.encode("utf-8") + b":"
        if raw_value is not None:
            return namespace_bytes + raw_value
        if context_hash is None:
            return None
        if context_hash == "__root__":
            context_bytes = b"__root__"
        else:
            try:
                context_bytes = bytes.fromhex(context_hash)
            except ValueError:
                context_bytes = context_hash.encode("utf-8")
        return namespace_bytes + context_bytes

    def _normalize_identifier(
        self,
        *,
        context_hash: str | None = None,
        raw_value: bytes | None = None,
    ) -> str | None:
        if raw_value is not None:
            return f"bytes:{raw_value.hex()}"
        if context_hash is not None:
            return f"hash:{context_hash}"
        return None

    def _confirm_pair_registration(self, value: bytes, key: int) -> bool:
        """Verify whether a failed PAIR_SET actually stuck on the backend."""
        try:
            existing = self._client.pair_get(value)
        except CheetahError:
            return False
        return existing == key

    def _disable(self, exc: Exception) -> NoReturn:
        if self._enabled:
            logger.warning("Disabling cheetah hot-path adapter: %s", exc)
            self._enabled = False
            self._shutdown_async()
            self._client_pool.close_all()
        reason = str(exc) or exc.__class__.__name__
        description = getattr(self, "_description", "cheetah")
        raise CheetahFatalError(f"{reason} (adapter={description})") from exc

    def _submit_async(self, action: Callable[[], None]) -> None:
        executor = getattr(self, "_async_executor", None)
        if executor is None:
            action()
            return
        future = executor.submit(action)
        future.add_done_callback(self._async_completion)
        with self._async_lock:
            self._async_futures.append(future)
            while len(self._async_futures) > self._async_max_pending:
                old = self._async_futures.popleft()
                self._await_future(old)

    def _async_completion(self, future: Future) -> None:
        exc = future.exception()
        if exc:
            logger.error("cheetah async publish failed: %s", exc)

    def _await_future(self, future: Future | None) -> None:
        if future is None:
            return
        try:
            future.result()
        except Exception as exc:  # pragma: no cover - log transport errors
            logger.error("cheetah async publish failed: %s", exc)

    def _shutdown_async(self) -> None:
        executor = getattr(self, "_async_executor", None)
        if executor is None:
            return
        self.flush_pending()
        executor.shutdown(wait=True)
        self._async_executor = None

    def _edit_or_reinsert(
        self,
        namespace: str,
        context_hash: str,
        key: int,
        payload: bytes,
    ) -> None:
        success, response = self._client.edit(key, payload)
        if success:
            return
        logger.warning(
            "cheetah edit failed for namespace=%s context_hash=%s key=%s response=%s; reinserting",
            namespace,
            context_hash,
            key,
            response,
        )
        self._insert(namespace, context_hash, payload)

    @property
    def _client(self) -> CheetahClient:
        return self._client_pool.acquire()


def build_cheetah_adapter(
    settings: DBSLMSettings,
    *,
    client: CheetahClient | None = None,
) -> HotPathAdapter:
    backend_active = settings.backend == "cheetah-db" or settings.cheetah_mirror
    if not backend_active:
        return NullHotPathAdapter()
    description = (
        f"cheetah-db://{settings.cheetah_host}:{settings.cheetah_port}/{settings.cheetah_database}"
    )
    idle_grace = settings.cheetah_idle_grace_seconds
    if idle_grace <= 0:
        idle_grace = None
    if client is not None:
        if idle_grace:
            client.set_idle_grace(idle_grace)
        if not client.connect():
            logger.warning(
                "cheetah hot-path backend unreachable (tried %s; last error: %s)",
                client.describe_targets(),
                client.describe_failures(),
            )
            raise SystemExit(1)
        return CheetahHotPathAdapter(client, description=description)

    def client_factory() -> CheetahClient:
        return CheetahClient(
            settings.cheetah_host,
            settings.cheetah_port,
            database=settings.cheetah_database,
            timeout=settings.cheetah_timeout_seconds,
            idle_grace=idle_grace,
        )

    warm_client = CheetahClient(
        settings.cheetah_host,
        settings.cheetah_port,
        database=settings.cheetah_database,
        timeout=settings.cheetah_timeout_seconds,
        idle_grace=idle_grace,
    )
    if not warm_client.connect():
        logger.warning(
            "cheetah hot-path backend unreachable (tried %s; last error: %s)",
            warm_client.describe_targets(),
            warm_client.describe_failures(),
        )
        raise SystemExit(1)
    return CheetahHotPathAdapter(
        warm_client,
        client_factory=client_factory,
        description=description,
    )


__all__ = [
    "CheetahClient",
    "CheetahError",
    "CheetahFatalError",
    "CheetahHotPathAdapter",
    "CheetahSerializer",
    "CountsPayload",
    "ContextPayload",
    "ContinuationPayload",
    "ProbabilityPayload",
    "TopKPayload",
    "build_cheetah_adapter",
]
