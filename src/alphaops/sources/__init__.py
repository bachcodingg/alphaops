"""Data source registry."""

from __future__ import annotations

from typing import Any

from alphaops.sources.base import (
    PRICE_COLUMNS,
    PRICE_FIELDS,
    DataSource,
    PermanentSourceError,
    SourceError,
    TransientSourceError,
    normalize_bars,
)

SOURCE_NAMES = ("synthetic", "yahoo", "csv", "vnstock")


def get_source(name: str, **kwargs: Any) -> DataSource:
    if name == "synthetic":
        from alphaops.sources.synthetic import SyntheticSource

        return SyntheticSource(**kwargs)
    if name == "yahoo":
        from alphaops.sources.yahoo import YahooSource

        return YahooSource(**kwargs)
    if name == "csv":
        from alphaops.sources.csv_source import CsvSource

        return CsvSource(**kwargs)
    if name == "vnstock":
        from alphaops.sources.vnstock_source import VnstockSource

        return VnstockSource(**kwargs)
    raise ValueError(f"unknown source {name!r}; choose from {SOURCE_NAMES}")


__all__ = [
    "PRICE_COLUMNS",
    "PRICE_FIELDS",
    "SOURCE_NAMES",
    "DataSource",
    "PermanentSourceError",
    "SourceError",
    "TransientSourceError",
    "get_source",
    "normalize_bars",
]
