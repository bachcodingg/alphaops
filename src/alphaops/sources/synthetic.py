"""Deterministic synthetic market, for tests, CI and offline demos.

The generator is a small factor model (market + sector + idiosyncratic) with a weak
multi-day reversal in the idiosyncratic component, so classic reversal alphas have a
real (but modest) signal to find, even at delay 1. Daily returns are clipped to
Vietnamese exchange price limits (HOSE 7%, HNX 10%, UPCoM 15%).

Determinism: each symbol's full path from ``ORIGIN`` to ``HORIZON`` is generated from a
seed derived from (seed, symbol) and then sliced, so the bar for a given (symbol, date)
is identical no matter which window is requested. That property is what lets us test
incremental ingestion and idempotent re-runs.

Optional fault injection:
  * ``anomalies=True`` plants realistic bad data (missing rows, OHLC violations,
    fat-finger spikes, unadjusted splits, negative volume, null closes) for the
    data-quality layer to catch.
  * ``failure_rate`` raises random ``TransientSourceError`` (chaos testing of retries).
  * ``permanent_failures`` makes listed symbols always fail.
"""

from __future__ import annotations

import random
import re
import zlib
from datetime import date

import numpy as np
import pandas as pd

from alphaops.config import last_completed_session
from alphaops.sources.base import (
    PRICE_COLUMNS,
    DataSource,
    PermanentSourceError,
    TransientSourceError,
)

ORIGIN = date(2016, 1, 1)
HORIZON = date(2030, 12, 31)
SECTORS = ["Banks", "RealEstate", "Steel", "Retail", "Energy", "Securities", "Tech", "Food"]
EXCHANGE_LIMITS = {"HOSE": 0.07, "HNX": 0.10, "UPCOM": 0.15}
REVERSAL_DAYS = 5
REVERSAL_STRENGTH = 0.07
_SYMBOL_RE = re.compile(r"^SYN(\d{3})$")


def _seed(*parts: object) -> int:
    return zlib.crc32("|".join(map(str, parts)).encode())


class SyntheticSource(DataSource):
    name = "synthetic"

    def __init__(
        self,
        n_symbols: int = 50,
        seed: int = 7,
        *,
        anomalies: bool = False,
        anomaly_prob: float = 0.002,
        failure_rate: float = 0.0,
        permanent_failures: tuple[str, ...] = (),
        as_of: date | None = None,
    ) -> None:
        self.n_symbols = n_symbols
        self.seed = seed
        self.anomalies = anomalies
        self.anomaly_prob = anomaly_prob
        self.failure_rate = failure_rate
        self.permanent_failures = set(permanent_failures)
        self.as_of = as_of
        self._dates = pd.bdate_range(ORIGIN, HORIZON)
        self._cache: dict[str, pd.DataFrame] = {}
        self._market: np.ndarray | None = None
        self._sector_factors: dict[str, np.ndarray] = {}

    # ----------------------------------------------------------------- metadata
    def symbols(self) -> list[str]:
        return [f"SYN{i:03d}" for i in range(self.n_symbols)]

    def _exchange(self, i: int) -> str:
        r = i % 10
        return "HOSE" if r < 7 else ("HNX" if r < 9 else "UPCOM")

    def symbol_meta(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "symbol": self.symbols(),
                "exchange": [self._exchange(i) for i in range(self.n_symbols)],
                "sector": [SECTORS[i % len(SECTORS)] for i in range(self.n_symbols)],
            }
        )

    # --------------------------------------------------------------- generation
    def _market_factor(self) -> np.ndarray:
        if self._market is None:
            rng = np.random.default_rng(_seed(self.seed, "market"))
            self._market = rng.normal(0.0003, 0.011, len(self._dates))
        return self._market

    def _sector_factor(self, sector: str) -> np.ndarray:
        if sector not in self._sector_factors:
            rng = np.random.default_rng(_seed(self.seed, "sector", sector))
            self._sector_factors[sector] = rng.normal(0.0, 0.007, len(self._dates))
        return self._sector_factors[sector]

    def _generate(self, symbol: str, i: int) -> pd.DataFrame:
        n = len(self._dates)
        rng = np.random.default_rng(_seed(self.seed, symbol))
        exchange = self._exchange(i)
        limit = EXCHANGE_LIMITS[exchange] * 0.99

        beta = rng.uniform(0.6, 1.4)
        sigma = rng.uniform(0.012, 0.024)
        # Idiosyncratic short-term reversal: each shock is partly given back over the next
        # REVERSAL_DAYS sessions, so a delay-1 reversal alpha has a modest edge.
        z = rng.standard_normal(n) * sigma
        past = np.convolve(z, np.ones(REVERSAL_DAYS), mode="full")[:n]
        past = np.concatenate([[0.0], past[:-1]])  # sum of the previous REVERSAL_DAYS shocks
        idio = z - REVERSAL_STRENGTH * past / REVERSAL_DAYS
        ret = beta * self._market_factor() + self._sector_factor(SECTORS[i % len(SECTORS)]) + idio
        ret = np.clip(ret, -limit, limit)

        close = rng.uniform(10, 120) * np.cumprod(1.0 + ret)
        prev_close = np.concatenate([[close[0]], close[:-1]])
        open_ = prev_close * (1.0 + rng.normal(0, 0.3 * sigma, n))
        hi_base = np.maximum(open_, close)
        lo_base = np.minimum(open_, close)
        high = hi_base * (1.0 + np.abs(rng.normal(0, 0.4 * sigma, n)))
        low = lo_base * (1.0 - np.abs(rng.normal(0, 0.4 * sigma, n)))
        base_vol = rng.lognormal(13.5, 0.8)
        volume = base_vol * np.exp(rng.normal(0, 0.35, n)) * (1.0 + 15.0 * np.abs(ret))
        volume = np.round(volume / 100.0) * 100.0

        df = pd.DataFrame(
            {
                "symbol": symbol,
                "date": self._dates,
                "open": np.round(open_, 2),
                "high": np.round(high, 2),
                "low": np.round(low, 2),
                "close": np.round(close, 2),
                "volume": volume,
            }
        )
        if self.anomalies:
            df = self._inject_anomalies(df, symbol)
        return df[PRICE_COLUMNS]

    def _inject_anomalies(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        rng = np.random.default_rng(_seed(self.seed, symbol, "anomalies"))
        n = len(df)
        hits = np.flatnonzero(rng.random(n) < self.anomaly_prob)
        kinds = rng.choice(
            ["missing", "ohlc", "spike", "split", "neg_volume", "null_close"], size=len(hits)
        )
        df = df.copy()
        drop: list[int] = []
        for pos, kind in zip(hits, kinds, strict=True):
            if pos == 0:
                continue
            if kind == "missing":
                drop.append(pos)
            elif kind == "ohlc":
                df.loc[pos, "high"] = df.loc[pos, "close"] * 0.95
            elif kind == "spike":  # fat-finger: one bar 3x, then back to normal
                for col in ("open", "high", "low", "close"):
                    df.loc[pos, col] = df.loc[pos, col] * 3.0
            elif kind == "split":  # unadjusted 2:1 split from this bar onwards
                for col in ("open", "high", "low", "close"):
                    df.loc[pos:, col] = df.loc[pos:, col] / 2.0
                df.loc[pos:, "volume"] = df.loc[pos:, "volume"] * 2.0
            elif kind == "neg_volume":
                df.loc[pos, "volume"] = -df.loc[pos, "volume"]
            elif kind == "null_close":
                df.loc[pos, "close"] = np.nan
        return df.drop(index=drop).reset_index(drop=True)

    def _history(self, symbol: str) -> pd.DataFrame:
        if symbol not in self._cache:
            m = _SYMBOL_RE.match(symbol)
            if not m or int(m.group(1)) >= self.n_symbols:
                raise PermanentSourceError(f"unknown synthetic symbol {symbol!r}")
            self._cache[symbol] = self._generate(symbol, int(m.group(1)))
        return self._cache[symbol]

    # -------------------------------------------------------------------- fetch
    def fetch(self, symbol: str, start: date, end: date) -> pd.DataFrame:
        if symbol in self.permanent_failures:
            raise PermanentSourceError(f"{symbol}: configured permanent failure")
        if self.failure_rate and random.random() < self.failure_rate:
            raise TransientSourceError(f"{symbol}: injected transient failure")
        end = min(end, self.as_of or last_completed_session())
        hist = self._history(symbol)
        mask = (hist["date"] >= pd.Timestamp(start)) & (hist["date"] <= pd.Timestamp(end))
        return hist.loc[mask].reset_index(drop=True)
