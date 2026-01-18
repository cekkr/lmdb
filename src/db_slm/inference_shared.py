from __future__ import annotations

from typing import Tuple

from .decoder import DecoderConfig
from .scoring import ScoreObserver
from .pipeline import DBSLMEngine


def issue_prompt(
    engine: DBSLMEngine,
    prompt: str,
    conversation_id: str | None = None,
    *,
    user_id: str = "trainer",
    agent_name: str = "db-slm",
    seed_history: bool = True,
    min_response_words: int = 0,
    decoder_cfg: DecoderConfig | None = None,
    rng_seed: int | None = None,
    scaffold_response: bool = True,
    score_observer: ScoreObserver | None = None,
) -> Tuple[str, str]:
    """
    Send a prompt through DBSLMEngine, starting a conversation when needed.

    Returns (conversation_id, response_text) so callers can keep reusing the
    same conversation across turns. Provide score_observer to capture per-step
    scoring snapshots while debugging decode output.
    """
    convo_id = conversation_id or engine.start_conversation(
        user_id, agent_name, seed_history=seed_history
    )
    response = engine.respond(
        convo_id,
        prompt,
        decoder_cfg=decoder_cfg,
        min_response_words=min_response_words,
        rng_seed=rng_seed,
        scaffold_response=scaffold_response,
        score_observer=score_observer,
    )
    return convo_id, response
