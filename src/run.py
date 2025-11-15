from __future__ import annotations

import argparse
import multiprocessing
import sys
from pathlib import Path
from typing import Sequence

from db_slm import DBSLMEngine
from db_slm.context_dimensions import format_context_dimensions, parse_context_dimensions_arg
from db_slm.inference_shared import issue_prompt
from db_slm.settings import load_settings

from helpers.cheetah_cli import (
    collect_namespace_summary_lines,
    collect_system_stats_lines,
)
from log_helpers import log, log_verbose


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
    parser.add_argument(
        "--cheetah-summary",
        action="append",
        default=[],
        metavar="PREFIX",
        help=(
            "Namespace prefix to summarize via cheetah's PAIR_SUMMARY before the session starts "
            "(e.g., 'ctx:', 'prob:2'). Repeat to request multiple summaries."
        ),
    )
    parser.add_argument(
        "--cheetah-summary-depth",
        type=int,
        default=1,
        help="Relative depth for cheetah namespace summaries in run.py (default: %(default)s).",
    )
    parser.add_argument(
        "--cheetah-summary-branches",
        type=int,
        default=32,
        help="Maximum branch digests returned per summary (default: %(default)s).",
    )
    parser.add_argument(
        "--cheetah-system-stats",
        action="store_true",
        help="Log cheetah SYSTEM_STATS before handling prompts.",
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


class PromptWorker:
    """Runs decoder work inside a child process to exploit multiple cores."""

    def __init__(
        self,
        db_path: str,
        *,
        ngram_order: int,
        context_dimensions,
        settings,
        user: str,
        agent: str,
        conversation: str | None,
        cheetah_summary: Sequence[str],
        cheetah_summary_depth: int,
        cheetah_summary_branches: int,
        cheetah_system_stats: bool,
    ) -> None:
        self._ctx = multiprocessing.get_context("spawn")
        self._requests = self._ctx.Queue()
        self._responses = self._ctx.Queue()
        self._next_id = 0
        self._process = self._ctx.Process(
            target=_decoder_worker,
            args=(
                self._requests,
                self._responses,
                db_path,
                ngram_order,
                context_dimensions,
                settings,
                user,
                agent,
                conversation,
                cheetah_summary,
                cheetah_summary_depth,
                cheetah_summary_branches,
                cheetah_system_stats,
            ),
        )
        self._process.start()
        ready = self._responses.get()
        if ready.get("status") != "ready":
            raise RuntimeError(ready.get("error", "decoder worker failed to start"))
        for line in ready.get("cheetah_logs") or []:
            log(f"[run] {line}")
        self.conversation_id = ready.get("conversation_id", "")
        self.context_label = ready.get("context_dimensions")

    def request(self, prompt: str) -> str:
        self._next_id += 1
        message_id = self._next_id
        self._requests.put({"type": "prompt", "prompt": prompt, "id": message_id})
        while True:
            msg = self._responses.get()
            if msg.get("type") != "response" or msg.get("id") != message_id:
                continue
            if "error" in msg:
                raise RuntimeError(msg["error"])
            response = msg.get("response", "")
            self.conversation_id = msg.get("conversation_id", self.conversation_id)
            return response

    def history(self) -> str:
        self._next_id += 1
        message_id = self._next_id
        self._requests.put({"type": "history", "id": message_id})
        while True:
            msg = self._responses.get()
            if msg.get("type") != "history" or msg.get("id") != message_id:
                continue
            if "error" in msg:
                raise RuntimeError(msg["error"])
            return msg.get("history", "")

    def conversation_summary(self) -> dict[str, str]:
        self._next_id += 1
        message_id = self._next_id
        self._requests.put({"type": "conversation", "id": message_id})
        while True:
            msg = self._responses.get()
            if msg.get("type") != "conversation" or msg.get("id") != message_id:
                continue
            if "error" in msg:
                raise RuntimeError(msg["error"])
            summary = msg.get("summary") or {}
            conv_id = summary.get("conversation_id") or self.conversation_id
            if conv_id:
                self.conversation_id = conv_id
            if "context_dimensions" in summary:
                self.context_label = summary["context_dimensions"]
            return summary

    def close(self) -> None:
        try:
            self._requests.put({"type": "stop"})
        except Exception:
            pass
        self._process.join(timeout=5)
        if self._process.is_alive():
            self._process.kill()


def _decoder_worker(
    request_q: multiprocessing.Queue,
    response_q: multiprocessing.Queue,
    db_path: str,
    ngram_order: int,
    context_dimensions,
    settings,
    user: str,
    agent: str,
    conversation: str | None,
    cheetah_summary: Sequence[str],
    cheetah_summary_depth: int,
    cheetah_summary_branches: int,
    cheetah_system_stats: bool,
) -> None:
    engine: DBSLMEngine | None = None
    try:
        engine = DBSLMEngine(
            db_path,
            ngram_order=ngram_order,
            context_dimensions=context_dimensions,
            settings=settings,
        )
        conv_id = conversation
        if conv_id:
            if not conversation_exists(engine, conv_id):
                raise ValueError(f"Conversation '{conv_id}' does not exist in this database")
        else:
            conv_id = engine.start_conversation(user, agent)
        dims_label = format_context_dimensions(engine.context_dimensions)
        cheetah_lines: list[str] = []
        if cheetah_system_stats:
            cheetah_lines.extend(collect_system_stats_lines(engine.hot_path))
        if cheetah_summary:
            cheetah_lines.extend(
                collect_namespace_summary_lines(
                    engine.hot_path,
                    cheetah_summary,
                    depth=cheetah_summary_depth,
                    branch_limit=cheetah_summary_branches,
                )
            )
        ready_payload = {
            "status": "ready",
            "conversation_id": conv_id,
            "context_dimensions": dims_label,
        }
        if cheetah_lines:
            ready_payload["cheetah_logs"] = cheetah_lines
        response_q.put(ready_payload)
        while True:
            msg = request_q.get()
            if not isinstance(msg, dict):
                continue
            if msg.get("type") == "stop":
                break
            if msg.get("type") == "prompt":
                prompt = msg.get("prompt", "")
                msg_id = msg.get("id")
                try:
                    conv_id, response = issue_prompt(engine, prompt, conv_id)
                    response_q.put(
                        {
                            "type": "response",
                            "id": msg_id,
                            "prompt": prompt,
                            "conversation_id": conv_id,
                            "response": response,
                        }
                    )
                except Exception as exc:  # pragma: no cover - defensive barrier around child process
                    response_q.put({"type": "response", "id": msg_id, "error": str(exc)})
                continue
            if msg.get("type") == "history":
                msg_id = msg.get("id")
                try:
                    history = engine.memory.context_window(conv_id)
                    response_q.put({"type": "history", "id": msg_id, "history": history or "(empty)"})
                except Exception as exc:  # pragma: no cover - defensive barrier around child process
                    response_q.put({"type": "history", "id": msg_id, "error": str(exc)})
                continue
            if msg.get("type") == "conversation":
                msg_id = msg.get("id")
                try:
                    summary = {
                        "conversation_id": conv_id,
                        "context_dimensions": format_context_dimensions(engine.context_dimensions),
                        "history": engine.memory.context_window(conv_id) or "(empty)",
                    }
                    response_q.put({"type": "conversation", "id": msg_id, "summary": summary})
                except Exception as exc:  # pragma: no cover - defensive barrier around child process
                    response_q.put({"type": "conversation", "id": msg_id, "error": str(exc)})
                continue
    except Exception as exc:  # pragma: no cover - startup failure
        response_q.put({"status": "error", "error": str(exc)})
    finally:
        if engine is not None:
            engine.db.close()


def respond_once_worker(worker: PromptWorker, prompt: str) -> None:
    response = worker.request(prompt)
    log(f"user> {prompt}")
    log(f"assistant> {response}")


def interactive_loop(worker: PromptWorker, max_turns: int | None) -> None:
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
            log("--- conversation context ---")
            try:
                history = worker.history()
            except Exception as exc:
                history = f"(failed to fetch history: {exc})"
            log(history or "(empty)")
            log("----------------------------")
            continue
        if user_input in {":status", ":conversation"}:
            try:
                summary = worker.conversation_summary()
            except Exception as exc:
                log(f"[run] Unable to fetch conversation summary: {exc}")
                continue
            log(f"[run] Conversation summary -> id={summary.get('conversation_id', worker.conversation_id)}")
            if "context_dimensions" in summary:
                log(f"[run] Context dims: {summary['context_dimensions']}")
            if "history" in summary:
                preview = summary["history"].strip() or "(empty)"
                if len(preview) > 200:
                    preview = preview[:197] + "..."
                log(f"[run] History preview: {preview}")
            continue
        response = worker.request(user_input)
        log(f"assistant> {response}")
        turns += 1
    log(f"[run] Conversation {worker.conversation_id} closed after {turns} turn(s).")


def main() -> None:
    settings = load_settings()
    parser = build_parser(settings.sqlite_dsn())
    args = parser.parse_args()
    log_verbose(3, f"[run:v3] Parsed CLI arguments: {vars(args)}")
    try:
        context_dimensions = parse_context_dimensions_arg(args.context_dimensions, default=None)
    except ValueError as exc:
        parser.error(str(exc))
    log_verbose(
        3,
        f"[run:v3] Context dims argument '{args.context_dimensions}' resolved to "
        f"{format_context_dimensions(context_dimensions)}.",
    )
    worker: PromptWorker | None = None
    try:
        db_path = resolve_db_path(args.db)
        worker = PromptWorker(
            db_path,
            ngram_order=args.ngram_order,
            context_dimensions=context_dimensions,
            settings=settings,
            user=args.user,
            agent=args.agent,
            conversation=args.conversation,
            cheetah_summary=args.cheetah_summary or [],
            cheetah_summary_depth=args.cheetah_summary_depth,
            cheetah_summary_branches=args.cheetah_summary_branches,
            cheetah_system_stats=args.cheetah_system_stats,
        )
        dims_label = worker.context_label or format_context_dimensions(context_dimensions)
        log(f"[run] Using conversation: {worker.conversation_id} (context dims: {dims_label})")

        if args.prompt:
            respond_once_worker(worker, args.prompt)
        else:
            interactive_loop(worker, args.max_turns)
    except ValueError as exc:
        parser.error(str(exc))
    finally:
        if worker is not None:
            worker.close()


if __name__ == "__main__":
    main()
