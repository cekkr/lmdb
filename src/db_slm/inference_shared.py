from __future__ import annotations

from typing import Tuple

from .pipeline import DBSLMEngine


def issue_prompt(
    engine: DBSLMEngine,
    prompt: str,
    conversation_id: str | None = None,
    *,
    user_id: str = "trainer",
    agent_name: str = "db-slm",
    seed_history: bool = True,
) -> Tuple[str, str]:
    """
    Send a prompt through DBSLMEngine, starting a conversation when needed.

    Returns (conversation_id, response_text) so callers can keep reusing the
    same conversation across turns.
    """
    convo_id = conversation_id or engine.start_conversation(
        user_id, agent_name, seed_history=seed_history
    )
    response = engine.respond(convo_id, prompt)
    return convo_id, response
