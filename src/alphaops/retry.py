"""Retry with exponential backoff and jitter, for transient upstream failures only."""

from __future__ import annotations

import logging
import random
import time
from collections.abc import Callable
from typing import TypeVar

T = TypeVar("T")
log = logging.getLogger(__name__)


def call_with_retry(
    fn: Callable[[], T],
    *,
    attempts: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 30.0,
    retry_on: tuple[type[BaseException], ...] = (Exception,),
    sleep: Callable[[float], None] = time.sleep,
    jitter: bool = True,
    description: str = "",
) -> T:
    """Call ``fn`` and retry on ``retry_on`` exceptions.

    Delay before attempt k (k >= 2) is ``min(max_delay, base_delay * 2**(k-2))``, multiplied
    by a random factor in [0.5, 1.0] when ``jitter`` is on (avoids thundering herds when many
    workers retry at once). Exceptions not in ``retry_on`` propagate immediately.
    """
    if attempts < 1:
        raise ValueError("attempts must be >= 1")
    for attempt in range(1, attempts + 1):
        try:
            return fn()
        except retry_on as exc:
            if attempt == attempts:
                raise
            delay = min(max_delay, base_delay * 2 ** (attempt - 1))
            if jitter:
                delay *= random.uniform(0.5, 1.0)
            log.warning(
                "transient failure, retrying",
                extra={
                    "what": description,
                    "attempt": attempt,
                    "max_attempts": attempts,
                    "delay_s": round(delay, 2),
                    "error": repr(exc),
                },
            )
            sleep(delay)
    raise AssertionError("unreachable")  # pragma: no cover
