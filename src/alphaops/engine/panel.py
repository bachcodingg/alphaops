"""Wide (date x symbol) matrices built from the long price table."""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from alphaops.engine.operators import ts_mean

BASE_FIELDS = ("open", "high", "low", "close", "volume")
FIELD_DOCS = {
    "open": "session open price",
    "high": "session high",
    "low": "session low",
    "close": "session close",
    "volume": "shares traded",
    "returns": "close-to-close return, close / ts_delay(close, 1) - 1",
    "vwap": "typical price (high + low + close) / 3, a VWAP proxy (no intraday data)",
    "dollar_volume": "close * volume (traded value)",
    "adv20": "20-day average traded value",
}
GROUP_DOCS = {
    "market": "every stock in one group",
    "sector": "sector from the universe file",
    "exchange": "listing exchange (HOSE / HNX / UPCOM)",
}


@dataclass
class Panel:
    fields: dict[str, pd.DataFrame]
    groups: dict[str, pd.Series] = field(default_factory=dict)

    @property
    def dates(self) -> pd.DatetimeIndex:
        return self.fields["close"].index

    @property
    def symbols(self) -> pd.Index:
        return self.fields["close"].columns

    @classmethod
    def from_prices(cls, prices: pd.DataFrame, meta: pd.DataFrame | None = None) -> Panel:
        if prices.empty:
            raise ValueError("no price data: run `alphaops ingest` first")
        wide = {
            f: prices.pivot(index="date", columns="symbol", values=f).sort_index()
            for f in BASE_FIELDS
        }
        close = wide["close"]
        wide["returns"] = close / close.shift(1) - 1.0
        wide["vwap"] = (wide["high"] + wide["low"] + close) / 3.0
        wide["dollar_volume"] = close * wide["volume"]
        wide["adv20"] = ts_mean(wide["dollar_volume"], 20)

        symbols = close.columns
        groups = {"market": pd.Series("ALL", index=symbols)}
        meta = meta if meta is not None else pd.DataFrame(columns=["symbol"])
        meta = meta.drop_duplicates("symbol").set_index("symbol")
        for g in ("sector", "exchange"):
            if g in meta.columns:
                groups[g] = meta[g].reindex(symbols).fillna("UNKNOWN")
            else:
                groups[g] = pd.Series("UNKNOWN", index=symbols)
        return cls(fields=wide, groups=groups)

    def with_fields(self, **overrides: pd.DataFrame) -> Panel:
        return Panel(fields={**self.fields, **overrides}, groups=dict(self.groups))
