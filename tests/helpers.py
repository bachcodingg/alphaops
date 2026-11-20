"""Test helpers (importable from tests as ``from helpers import ...``)."""

from __future__ import annotations

import numpy as np
import pandas as pd


def make_bars(
    symbol: str, closes: list[float], start: str = "2024-01-01", **overrides
) -> pd.DataFrame:
    """Well-formed bars for a list of closes; override any column with a list."""
    dates = pd.bdate_range(start, periods=len(closes))
    close = np.asarray(closes, dtype=float)
    df = pd.DataFrame(
        {
            "symbol": symbol,
            "date": dates,
            "open": close,
            "high": close * 1.01,
            "low": close * 0.99,
            "close": close,
            "volume": 1000.0,
        }
    )
    for col, values in overrides.items():
        df[col] = values
    return df
