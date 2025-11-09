from __future__ import annotations

import argparse
import sys
from pathlib import Path

from db_slm import DBSLMEngine


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Interact with a trained DB-SLM instance for quick inference."
    )
    parser.add_argument(
        "--db",
        default="var/db_slm.sqlite3",
        help="Path to the SQLite database produced by train.py (default: %(default)s).",
    )
    parser.add_argument(
        "--ngram-order",
        type=int,
        default=3,
        help="N-gram order used when the database was created (default: %(default)s).",
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
    response = engine.respond(conversation_id, prompt)
    print(f"user> {prompt}")
    print(f"assistant> {response}")


def interactive_loop(engine: DBSLMEngine, conversation_id: str, max_turns: int | None) -> None:
    print("[run] Type ':exit' or press Ctrl+D to leave, ':history' to show the current context.")
    turns = 0
    while max_turns is None or turns < max_turns:
        try:
            user_input = input("you> ").strip()
        except EOFError:
            print()
            break
        except KeyboardInterrupt:
            print("\n[run] Interrupted. Exiting.")
            break
        if not user_input:
            continue
        if user_input in {":exit", ":quit"}:
            break
        if user_input == ":history":
            history = engine.memory.context_window(conversation_id)
            print("--- conversation context ---")
            print(history or "(empty)")
            print("----------------------------")
            continue
        response = engine.respond(conversation_id, user_input)
        print(f"assistant> {response}")
        turns += 1
    print(f"[run] Conversation {conversation_id} closed after {turns} turn(s).")


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    engine: DBSLMEngine | None = None
    try:
        db_path = resolve_db_path(args.db)
        engine = DBSLMEngine(db_path, ngram_order=args.ngram_order)
        conversation_id = ensure_conversation(engine, args)
        print(f"[run] Using conversation: {conversation_id}")

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
