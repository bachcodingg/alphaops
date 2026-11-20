"""HOSE equities via Yahoo Finance's public chart endpoint (tickers like ``FPT.VN``).

Only the standard library is used (``urllib``). Operational notes:

* Coverage: HOSE tickers use the ``.VN`` suffix. HNX/UPCoM are not covered; use the
  ``csv`` or ``vnstock`` source for those. Symbols are stored *without* the suffix.
* Units: prices are in VND (not thousands of VND as in vnstock), so do not mix sources
  in one database. The extreme-return check would flag the 1000x jump if you did.
* Adjustment: with ``adjusted=True`` (default) OHLC are scaled by adjclose/close and
  volume inversely, so splits and stock dividends do not show up as price jumps.
  Adjusted history is restated after every new corporate action. Incremental runs only
  re-fetch a short lookback window, so after a split the ex-date bar is quarantined as
  an extreme return. The runbook explains the fix (full backfill of that symbol).
* Quirks handled here (counts are logged per symbol):
    - a bar for *today* is dropped before the session is final (HOSE closes 14:45 ICT);
    - Yahoo emits placeholder bars on Vietnamese public holidays (Tet, 30/4, 2/9, ...):
      either all prices null, or O=H=L=C with zero volume. They are not sessions, and
      keeping them would create fake trading days with zero returns, so they are dropped.
      Partially-null bars are *not* dropped here; the data-quality layer quarantines them.
* HTTP 429/5xx and network errors -> ``TransientSourceError`` (retried by the caller);
  404 / "No data found" -> ``PermanentSourceError``.
"""

from __future__ import annotations

import json
import logging
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import UTC, date, datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd

from alphaops.config import SESSION_FINAL_HOUR, VN_TZ
from alphaops.sources.base import (
    DataSource,
    PermanentSourceError,
    TransientSourceError,
    empty_bars,
    normalize_bars,
)

log = logging.getLogger(__name__)

CHART_URL = "https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
USER_AGENT = "Mozilla/5.0 (compatible; alphaops/0.1)"


class YahooSource(DataSource):
    name = "yahoo"

    def __init__(
        self,
        suffix: str = ".VN",
        adjusted: bool = True,
        min_interval_s: float = 0.5,
        timeout_s: float = 20.0,
        now: datetime | None = None,
    ) -> None:
        self.suffix = suffix
        self.adjusted = adjusted
        self.min_interval_s = min_interval_s
        self.timeout_s = timeout_s
        self._now = now
        self._last_call = 0.0

    def ticker(self, symbol: str) -> str:
        return symbol if "." in symbol else f"{symbol}{self.suffix}"

    def _throttle(self) -> None:
        wait = self.min_interval_s - (time.monotonic() - self._last_call)
        if wait > 0:
            time.sleep(wait)
        self._last_call = time.monotonic()

    def _get_json(self, url: str) -> dict[str, Any]:
        req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                raise PermanentSourceError(f"HTTP 404 for {url}") from exc
            if exc.code == 429 or exc.code >= 500:
                raise TransientSourceError(f"HTTP {exc.code} for {url}") from exc
            raise PermanentSourceError(f"HTTP {exc.code} for {url}") from exc
        except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
            raise TransientSourceError(f"network error for {url}: {exc}") from exc
        except json.JSONDecodeError as exc:
            raise TransientSourceError(f"invalid JSON from {url}") from exc

    @staticmethod
    def parse_chart(payload: dict[str, Any], symbol: str, adjusted: bool = True) -> pd.DataFrame:
        chart = payload.get("chart") or {}
        if chart.get("error"):
            desc = chart["error"].get("description", "")
            raise PermanentSourceError(f"{symbol}: {desc or chart['error']}")
        results = chart.get("result") or []
        if not results or not results[0].get("timestamp"):
            return empty_bars()
        res = results[0]
        quote = res["indicators"]["quote"][0]
        offset = timedelta(seconds=res.get("meta", {}).get("gmtoffset", 25200))
        dates = [
            (datetime.fromtimestamp(ts, UTC) + offset).date().isoformat() for ts in res["timestamp"]
        ]
        df = pd.DataFrame(
            {
                "date": dates,
                **{k: quote.get(k) for k in ("open", "high", "low", "close", "volume")},
            }
        )
        bars = normalize_bars(df, symbol)
        adj = (res["indicators"].get("adjclose") or [{}])[0].get("adjclose")
        if adjusted and adj is not None:
            factor = pd.Series(adj, dtype="float64").to_numpy() / bars["close"].to_numpy()
            factor = np.where(np.isfinite(factor) & (factor > 0), factor, 1.0)
            for col in ("open", "high", "low", "close"):
                bars[col] = bars[col] * factor
            bars["volume"] = bars["volume"] / factor
        return bars

    def drop_placeholders(self, bars: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Remove bars that are not real sessions (see module docstring)."""
        if bars.empty:
            return bars
        now_local = (self._now or datetime.now(UTC)).astimezone(VN_TZ)
        today = pd.Timestamp(now_local.date())
        unfinished = (bars["date"] == today) & (now_local.hour < SESSION_FINAL_HOUR)
        ohlc = bars[["open", "high", "low", "close"]]
        all_null = ohlc.isna().all(axis=1)
        flat = ohlc.nunique(axis=1, dropna=False).eq(1)
        no_trade = flat & bars["volume"].fillna(0).eq(0)
        drop = unfinished | all_null | no_trade
        if drop.any():
            log.info(
                "dropped non-session bars",
                extra={
                    "symbol": symbol,
                    "unfinished": int(unfinished.sum()),
                    "all_null": int(all_null.sum()),
                    "flat_zero_volume": int((no_trade & ~all_null).sum()),
                    "last_dropped": bars.loc[drop, "date"].max().date(),
                },
            )
        return bars.loc[~drop].reset_index(drop=True)

    def fetch(self, symbol: str, start: date, end: date) -> pd.DataFrame:
        if start > end:
            return empty_bars()
        period1 = int(datetime.combine(start, datetime.min.time(), VN_TZ).timestamp())
        period2 = int(
            datetime.combine(end + timedelta(days=1), datetime.min.time(), VN_TZ).timestamp()
        )
        query = urllib.parse.urlencode(
            {
                "period1": period1,
                "period2": period2,
                "interval": "1d",
                "events": "split,div",
                "includeAdjustedClose": "true",
            }
        )
        url = CHART_URL.format(ticker=urllib.parse.quote(self.ticker(symbol))) + "?" + query
        self._throttle()
        bars = self.parse_chart(self._get_json(url), symbol, self.adjusted)
        bars = self.drop_placeholders(bars, symbol)
        mask = (bars["date"] >= pd.Timestamp(start)) & (bars["date"] <= pd.Timestamp(end))
        return bars.loc[mask].reset_index(drop=True)
