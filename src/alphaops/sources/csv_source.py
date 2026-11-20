"""Load daily bars from a directory of ``<SYMBOL>.csv`` files.

Useful for data exported from any vendor or broker. Expected columns (any case):
``date`` (or ``time``), ``open``, ``high``, ``low``, ``close``, ``volume``.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd

from alphaops.sources.base import DataSource, PermanentSourceError, normalize_bars


class CsvSource(DataSource):
    name = "csv"

    def __init__(self, directory: str | Path) -> None:
        self.directory = Path(directory)
        if not self.directory.is_dir():
            raise PermanentSourceError(f"CSV directory not found: {self.directory}")

    def symbols(self) -> list[str]:
        return sorted(p.stem.upper() for p in self.directory.glob("*.csv"))

    def fetch(self, symbol: str, start: date, end: date) -> pd.DataFrame:
        path = self.directory / f"{symbol}.csv"
        if not path.exists():
            raise PermanentSourceError(f"{symbol}: no file {path}")
        bars = normalize_bars(pd.read_csv(path), symbol)
        mask = (bars["date"] >= pd.Timestamp(start)) & (bars["date"] <= pd.Timestamp(end))
        return bars.loc[mask].reset_index(drop=True)
