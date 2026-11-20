"""Vietnamese equities (HOSE/HNX/UPCoM) via the open-source ``vnstock`` library.

``vnstock`` wraps public broker APIs (VCI, TCBS, ...). It is an optional dependency:
``pip install -e ".[vnstock]"``. Notes for operators:

* Prices are quoted in thousands of VND.
* Free/guest access is rate-limited, so requests are throttled (``min_interval_s``).
  A rate-limit or network error surfaces as ``TransientSourceError`` and is retried.
* The library's API has changed across major versions; both the 3.x ``Quote`` class and
  the older ``Vnstock().stock(...)`` entry point are supported.
* Check vnstock's licence terms before any commercial use.
"""

from __future__ import annotations

import logging
import time
from datetime import date

import pandas as pd

from alphaops.sources.base import (
    DataSource,
    PermanentSourceError,
    TransientSourceError,
    empty_bars,
    normalize_bars,
)

log = logging.getLogger(__name__)

# Messages that indicate the request will never succeed; everything else is retried.
_PERMANENT_HINTS = ("not found", "invalid symbol", "không tìm thấy", "no data")


class VnstockSource(DataSource):
    name = "vnstock"

    def __init__(self, provider: str = "VCI", min_interval_s: float = 3.1) -> None:
        try:
            import vnstock  # noqa: F401
        except ImportError as exc:  # pragma: no cover - depends on environment
            raise PermanentSourceError(
                "vnstock is not installed; run: pip install -e '.[vnstock]'"
            ) from exc
        self.provider = provider
        self.min_interval_s = min_interval_s
        self._last_call = 0.0

    def _throttle(self) -> None:
        wait = self.min_interval_s - (time.monotonic() - self._last_call)
        if wait > 0:
            time.sleep(wait)
        self._last_call = time.monotonic()

    def _history(self, symbol: str, start: date, end: date) -> pd.DataFrame:
        kwargs = {"start": start.isoformat(), "end": end.isoformat(), "interval": "1D"}
        try:
            from vnstock import Quote  # vnstock >= 3.x

            return Quote(symbol=symbol, source=self.provider).history(**kwargs)
        except ImportError:  # pragma: no cover - older vnstock
            from vnstock import Vnstock

            stock = Vnstock().stock(symbol=symbol, source=self.provider)
            return stock.quote.history(**kwargs)

    def fetch(self, symbol: str, start: date, end: date) -> pd.DataFrame:
        if start > end:
            return empty_bars()
        self._throttle()
        try:
            raw = self._history(symbol, start, end)
        except (ConnectionError, TimeoutError, OSError) as exc:
            raise TransientSourceError(f"{symbol}: network error: {exc}") from exc
        except SystemExit as exc:
            # Some vnstock versions call sys.exit() when the rate limit is hit.
            raise TransientSourceError(f"{symbol}: provider rate limit ({exc})") from exc
        except Exception as exc:
            msg = str(exc).lower()
            if any(hint in msg for hint in _PERMANENT_HINTS):
                raise PermanentSourceError(f"{symbol}: {exc}") from exc
            raise TransientSourceError(f"{symbol}: {type(exc).__name__}: {exc}") from exc
        bars = normalize_bars(raw, symbol)
        mask = (bars["date"] >= pd.Timestamp(start)) & (bars["date"] <= pd.Timestamp(end))
        return bars.loc[mask].reset_index(drop=True)
