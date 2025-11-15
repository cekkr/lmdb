from __future__ import annotations

import os

_ALIAS_MAP = {
    "gpu": "cuda",
    "gpu0": "cuda:0",
    "gpu1": "cuda:1",
    "gpu2": "cuda:2",
    "gpu3": "cuda:3",
    "cuda0": "cuda:0",
    "cuda1": "cuda:1",
    "cuda2": "cuda:2",
    "cuda3": "cuda:3",
    "metal": "mps",
    "mps0": "mps",
}


def _normalize(value: str) -> str:
    normalized = value.strip().lower()
    return _ALIAS_MAP.get(normalized, normalized)


def requested_device() -> str | None:
    """Return the normalized DEVICE override when set (auto/default -> None)."""

    raw_value = os.environ.get("DEVICE")
    if not raw_value:
        return None
    normalized = _normalize(raw_value)
    if normalized in {"auto", "default", ""}:
        return None
    return normalized


def device_available(candidate: str) -> bool:
    """Best-effort availability probe for the requested torch device."""

    if not candidate:
        return False
    if candidate == "cpu":
        return True
    try:
        import torch  # type: ignore
    except Exception:
        return False
    if candidate.startswith("cuda"):
        if not torch.cuda.is_available():
            return False
        if ":" in candidate:
            try:
                index = int(candidate.split(":", 1)[1])
            except ValueError:
                return False
            return index < torch.cuda.device_count()
        return True
    if candidate.startswith("mps"):
        backend = getattr(torch.backends, "mps", None)
        return bool(backend and torch.backends.mps.is_available())
    return False


def auto_device(prefer_mps: bool = True) -> str:
    """Automatic accelerator detection for torch-backed workloads."""

    try:
        import torch  # type: ignore

        if torch.cuda.is_available():
            return "cuda"
        if prefer_mps:
            backend = getattr(torch.backends, "mps", None)
            if backend and torch.backends.mps.is_available():
                return "mps"
    except Exception:
        pass
    return "cpu"
