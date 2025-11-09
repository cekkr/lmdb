from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

from db_slm import DBSLMEngine


def build_parser() -> argparse.ArgumentParser:
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
        default="var/db_slm.sqlite3",
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


def read_corpora(paths: Iterable[Path], encoding: str) -> List[Tuple[str, str]]:
    corpora: list[tuple[str, str]] = []
    for path in paths:
        text = path.read_text(encoding=encoding)
        corpora.append((str(path), text))
    return corpora


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if not args.inputs and not args.stdin:
        parser.error("Provide at least one input path or enable --stdin")

    db_path_str, db_path = resolve_db_path(args.db, args.reset)
    engine = DBSLMEngine(db_path_str, ngram_order=args.ngram_order)

    corpora = read_corpora(collect_files(args.inputs, args.recursive), args.encoding)
    if args.stdin:
        stdin_payload = sys.stdin.read()
        if stdin_payload.strip():
            corpora.append(("<stdin>", stdin_payload))

    if not corpora:
        parser.error("No readable corpora found in the provided inputs")

    total_tokens = 0
    total_windows = 0
    for label, corpus in corpora:
        token_count = engine.train_from_text(corpus)
        if token_count == 0:
            print(f"[train] Skipping {label} (corpus too small for order={engine.store.order})")
            continue
        window = max(0, token_count - engine.store.order + 1)
        total_tokens += token_count
        total_windows += window
        print(f"[train] Ingested {label}: {token_count} tokens -> {window} n-grams")

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
