from __future__ import annotations

import os
import threading
from dataclasses import dataclass

__all__ = ["AdaptiveLoadController", "headroom_ratio", "suggest_worker_count"]


def headroom_ratio(default: float = 1.0) -> float:
    """
    Return the fraction (0.0–1.0) of CPU capacity that appears idle.

    Falls back to `default` whenever the platform does not expose load averages.
    """
    cores = os.cpu_count() or 1
    try:
        load1, _, _ = os.getloadavg()
    except (AttributeError, OSError):
        return default
    if cores <= 0:
        return default
    # Normalize the 1-minute load by the number of logical cores.
    normalized = load1 / cores
    headroom = max(0.0, 1.0 - normalized)
    return min(1.0, headroom)


def suggest_worker_count(base_workers: int, min_workers: int = 1, max_workers: int | None = None) -> int:
    """
    Scale the worker count based on available headroom so we opportunistically
    consume more CPU when the machine is idle.
    """
    headroom = headroom_ratio()
    target = base_workers
    if headroom < 0.25:
        target = max(1, int(round(base_workers * 0.5)))
    elif headroom > 0.6:
        target = int(round(base_workers * (1.0 + headroom)))
    if max_workers is not None:
        target = min(target, max_workers)
    return max(min_workers, target)


from dataclasses import dataclass, field


@dataclass
class AdaptiveLoadController:
    """
    Tiny helper that throttles expensive tasks when CPU saturation is high.

    Example:

        guard = AdaptiveLoadController(min_idle_ratio=0.3)
        if guard.allow_heavy_task():
            run_external_model()
    """

    min_idle_ratio: float = 0.3
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def allow_heavy_task(self) -> bool:
        """True when the observed idle capacity is above the configured floor."""
        with self._lock:
            return headroom_ratio() >= self.min_idle_ratio
