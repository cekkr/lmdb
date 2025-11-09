from __future__ import annotations

import argparse
import json
import random
import sys
import textwrap
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

from db_slm import DBSLMEngine
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


def read_corpora(paths: Iterable[Path], encoding: str, json_chunk_size: int) -> List[Tuple[str, str]]:
    corpora: list[tuple[str, str]] = []
    for path in paths:
        suffix = path.suffix.lower()
        if suffix in {".json", ".ndjson"}:
            corpora.extend(read_json_corpora(path, json_chunk_size))
            continue
        text = path.read_text(encoding=encoding)
        corpora.append((str(path), text))
    return corpora


def read_json_corpora(path: Path, chunk_size: int) -> List[Tuple[str, str]]:
    chunk_size = max(1, chunk_size)
    corpora: list[tuple[str, str]] = []
    buffer: list[str] = []
    chunk_index = 0
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
            buffer.append(segment)
            if len(buffer) >= chunk_size:
                chunk_index += 1
                corpora.append((f"{path}#chunk{chunk_index}", "\n\n".join(buffer)))
                buffer.clear()
    if buffer:
        chunk_index += 1
        corpora.append((f"{path}#chunk{chunk_index}", "\n\n".join(buffer)))
    print(
        f"[train] Prepared {chunk_index} chunk(s) from {path} using chunk size {chunk_size}."
    )
    return corpora


def load_eval_dataset(path: Path, max_records: int | None = None) -> List[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Evaluation dataset not found: {path}")
    records: list[dict[str, str]] = []
    reservoir = max_records if max_records is not None and max_records > 0 else None
    seen = 0
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
            seen += 1
            if reservoir is None or len(records) < reservoir:
                records.append(record)
            else:
                replacement_index = random.randint(0, seen - 1)
                if replacement_index < reservoir:
                    records[replacement_index] = record
    return records


def lexical_overlap_score(reference: str, candidate: str) -> float:
    ref_tokens = {token.strip(".,!?\"'").lower() for token in reference.split() if token.strip()}
    if not ref_tokens:
        return 0.0
    cand_tokens = {token.strip(".,!?\"'").lower() for token in candidate.split() if token.strip()}
    if not cand_tokens:
        return 0.0
    overlap = ref_tokens & cand_tokens
    return len(overlap) / len(ref_tokens)


def preview(text: str, width: int = 80) -> str:
    collapsed = " ".join(text.split())
    if not collapsed:
        return ""
    return textwrap.shorten(collapsed, width=width, placeholder="…")


class InferenceMonitor:
    def __init__(
        self,
        engine: DBSLMEngine,
        dataset: List[dict[str, str]],
        interval_tokens: int,
        samples_per_cycle: int,
    ) -> None:
        self.engine = engine
        self.dataset = dataset
        self.interval = interval_tokens
        self.samples = max(1, samples_per_cycle)
        self.next_threshold = interval_tokens

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
            score = lexical_overlap_score(record["response"], generated)
            print(
                f"[eval] #{idx}: emotion={record['emotion']} score={score:.2f} prompt='{preview(record['prompt'])}' response='{preview(generated)}'"
            )


def main() -> None:
    settings = load_settings()
    parser = build_parser(settings.sqlite_dsn())
    args = parser.parse_args()

    if not args.inputs and not args.stdin:
        parser.error("Provide at least one input path or enable --stdin")

    db_path_str, db_path = resolve_db_path(args.db, args.reset)
    engine = DBSLMEngine(db_path_str, ngram_order=args.ngram_order)

    corpora = read_corpora(
        collect_files(args.inputs, args.recursive), args.encoding, args.json_chunk_size
    )
    if args.stdin:
        stdin_payload = sys.stdin.read()
        if stdin_payload.strip():
            corpora.append(("<stdin>", stdin_payload))

    if not corpora:
        parser.error("No readable corpora found in the provided inputs")

    eval_dataset_path = args.eval_dataset or settings.dataset_path
    eval_records: list[dict[str, str]] = []
    monitor = InferenceMonitor(engine, eval_records, args.eval_interval, args.eval_samples)
    if args.eval_interval > 0:
        dataset_path = Path(eval_dataset_path).expanduser()
        try:
            eval_records = load_eval_dataset(dataset_path, args.eval_pool_size)
            monitor = InferenceMonitor(engine, eval_records, args.eval_interval, args.eval_samples)
            print(f"[eval] Loaded {len(eval_records)} held-out sample(s) from {dataset_path}.")
        except FileNotFoundError as exc:
            print(f"[eval] Warning: {exc}. Disabling evaluation probes.")
            monitor = InferenceMonitor(engine, [], 0, args.eval_samples)

    total_tokens = 0
    total_windows = 0
    print(
        f"[train] Starting ingest of {len(corpora)} corpus/corpora with order={engine.store.order} into {db_path_str}."
    )
    for label, corpus in corpora:
        print(f"[train] Processing {label} ({len(corpus)} bytes)...")
        token_count = engine.train_from_text(corpus)
        if token_count == 0:
            print(f"[train] Skipping {label} (corpus too small for order={engine.store.order})")
            continue
        window = max(0, token_count - engine.store.order + 1)
        total_tokens += token_count
        total_windows += window
        print(f"[train] Ingested {label}: {token_count} tokens -> {window} n-grams")
        monitor.maybe_run(total_tokens)

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
