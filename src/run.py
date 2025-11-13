from __future__ import annotations

import argparse
import sys
from pathlib import Path

from db_slm import DBSLMEngine
from db_slm.context_dimensions import format_context_dimensions, parse_context_dimensions_arg
from db_slm.inference_shared import issue_prompt
from db_slm.settings import load_settings

from log_helpers import log


def build_parser(default_db_path: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Interact with a trained DB-SLM instance for quick inference."
    )
    parser.add_argument(
        "--db",
        default=default_db_path,
        help="Path to the SQLite database produced by train.py (default: %(default)s).",
    )
    parser.add_argument(
        "--ngram-order",
        type=int,
        default=3,
        help="N-gram order used when the database was created (default: %(default)s).",
    )
    parser.add_argument(
        "--context-dimensions",
        help=(
            "Override the stored context-dimension penalties with ranges like '1-2,3-5' or "
            "length specs such as '4,8,4'. Use 'off' to disable; omit to reuse metadata from the database."
        ),
    )
    parser.add_argument(
        "--user",
        default="cli-user",
        help="User identifier recorded in tbl_l2_conversations (default: %(default)s).",
    )
    parser.add_argument(
        "--agent",
        default="db-slm",
        help="Agent name recorded for new conversations (default: %(default)s).",
    )
    parser.add_argument(
        "--conversation",
        help="Existing conversation ID to resume. Omit to start a new conversation.",
    )
    parser.add_argument(
        "--prompt",
        help="Single prompt to send in non-interactive mode. If omitted an interactive shell starts.",
    )
    parser.add_argument(
        "--max-turns",
        type=int,
        default=None,
        help="Optional limit for turns in interactive mode (default: unlimited).",
    )
    return parser


def resolve_db_path(raw: str) -> str:
    if raw == ":memory:":
        raise ValueError("run.py requires a persistent database path (not :memory:)")
    path = Path(raw).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    return str(path)


def conversation_exists(engine: DBSLMEngine, conversation_id: str) -> bool:
    rows = engine.db.query(
        "SELECT 1 FROM tbl_l2_conversations WHERE id = ? LIMIT 1", (conversation_id,)
    )
    return bool(rows)


def ensure_conversation(engine: DBSLMEngine, args: argparse.Namespace) -> str:
    if args.conversation:
        if not conversation_exists(engine, args.conversation):
            raise ValueError(f"Conversation '{args.conversation}' does not exist in this database")
        return args.conversation
    return engine.start_conversation(args.user, args.agent)


def respond_once(engine: DBSLMEngine, conversation_id: str, prompt: str) -> None:
    _, response = issue_prompt(engine, prompt, conversation_id)
    log(f"user> {prompt}")
    log(f"assistant> {response}")


def interactive_loop(engine: DBSLMEngine, conversation_id: str, max_turns: int | None) -> None:
    log("[run] Type ':exit' or press Ctrl+D to leave, ':history' to show the current context.")
    turns = 0
    while max_turns is None or turns < max_turns:
        try:
            user_input = input("you> ").strip()
        except EOFError:
            print()
            break
        except KeyboardInterrupt:
            log("[run] Interrupted. Exiting.")
            print()
            break
        if not user_input:
            continue
        if user_input in {":exit", ":quit"}:
            break
        if user_input == ":history":
            history = engine.memory.context_window(conversation_id)
            log("--- conversation context ---")
            log(history or "(empty)")
            log("----------------------------")
            continue
        conversation_id, response = issue_prompt(engine, user_input, conversation_id)
        log(f"assistant> {response}")
        turns += 1
    log(f"[run] Conversation {conversation_id} closed after {turns} turn(s).")


def main() -> None:
    settings = load_settings()
    parser = build_parser(settings.sqlite_dsn())
    args = parser.parse_args()
    try:
        context_dimensions = parse_context_dimensions_arg(args.context_dimensions, default=None)
    except ValueError as exc:
        parser.error(str(exc))

    engine: DBSLMEngine | None = None
    try:
        db_path = resolve_db_path(args.db)
        engine = DBSLMEngine(
            db_path,
            ngram_order=args.ngram_order,
            context_dimensions=context_dimensions,
            settings=settings,
        )
        conversation_id = ensure_conversation(engine, args)
        dims_label = format_context_dimensions(engine.context_dimensions)
        log(f"[run] Using conversation: {conversation_id} (context dims: {dims_label})")

        if args.prompt:
            respond_once(engine, conversation_id, args.prompt)
        else:
            interactive_loop(engine, conversation_id, args.max_turns)
    except ValueError as exc:
        parser.error(str(exc))
    finally:
        if engine is not None:
            engine.db.close()


if __name__ == "__main__":
    main()
