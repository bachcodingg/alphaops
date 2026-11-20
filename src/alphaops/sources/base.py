"""Data source interface. Every source returns daily bars in one canonical schema."""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import date

import pandas as pd

PRICE_FIELDS = ["open", "high", "low", "close", "volume"]
PRICE_COLUMNS = ["symbol", "date", *PRICE_FIELDS]


class SourceError(Exception):
    """Base class for data source failures."""


class TransientSourceError(SourceError):
    """Temporary failure (timeout, rate limit, 5xx). Safe to retry."""


class PermanentSourceError(SourceError):
    """Will not succeed on retry (unknown symbol, bad request, missing file)."""


class DataSource(ABC):
    name: str = "base"

    @abstractmethod
    def fetch(self, symbol: str, start: date, end: date) -> pd.DataFrame:
        """Return daily bars for ``symbol`` with dates in [start, end], canonical schema."""

    def symbol_meta(self) -> pd.DataFrame | None:
        """Optional static metadata: columns symbol, exchange, sector."""
        return None


def empty_bars() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "symbol": pd.Series(dtype="object"),
            "date": pd.Series(dtype="datetime64[ns]"),
            **{c: pd.Series(dtype="float64") for c in PRICE_FIELDS},
        }
    )


def normalize_bars(raw: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """Coerce a source-specific frame into the canonical schema.

    Accepts either a ``date`` or ``time`` column, any column-name case, and returns
    columns ``PRICE_COLUMNS`` with ``date`` as midnight-normalized ``datetime64``.
    Values are *not* validated here; that is the data-quality layer's job.
    """
    if raw is None or len(raw) == 0:
        return empty_bars()
    df = raw.rename(columns={c: str(c).strip().lower() for c in raw.columns})
    if "date" not in df.columns and "time" in df.columns:
        df = df.rename(columns={"time": "date"})
    missing = [c for c in ["date", *PRICE_FIELDS] if c not in df.columns]
    if missing:
        raise PermanentSourceError(f"{symbol}: source frame missing columns {missing}")
    dates = pd.to_datetime(df["date"])
    if dates.dt.tz is not None:
        dates = dates.dt.tz_localize(None)
    out = pd.DataFrame({"symbol": symbol, "date": dates.dt.normalize().astype("datetime64[ns]")})
    for col in PRICE_FIELDS:
        out[col] = pd.to_numeric(df[col], errors="coerce").astype("float64")
    return out[PRICE_COLUMNS].reset_index(drop=True)
