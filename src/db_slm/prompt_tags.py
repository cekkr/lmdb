from __future__ import annotations


DEFAULT_RESPONSE_LABEL = "|RESPONSE|"


def ensure_response_prompt_tag(prompt: str, response_label: str | None = None) -> str:
    """
    Ensure prompts end with the response label so decoders predict assistant text
    instead of continuing the user frame.
    """
    base = (prompt or "").rstrip()
    label = (response_label or DEFAULT_RESPONSE_LABEL).strip() or DEFAULT_RESPONSE_LABEL
    sentinel = f"{label}:"
    sentinel_with_space = f"{sentinel} "
    if not base:
        return sentinel_with_space
    if base.endswith(sentinel_with_space):
        return base
    if base.endswith(sentinel):
        return f"{base} "
    return f"{base}\n{sentinel_with_space}"
