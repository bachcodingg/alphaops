"""Universe files: CSV with columns ``symbol,exchange,sector`` (``#`` lines are comments)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_universe(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path, comment="#", skipinitialspace=True, dtype=str)
    df.columns = [c.strip().lower() for c in df.columns]
    if "symbol" not in df.columns:
        raise ValueError(f"{path}: universe file needs a 'symbol' column")
    df["symbol"] = df["symbol"].str.strip().str.upper()
    for col in ("exchange", "sector"):
        df[col] = df[col].str.strip() if col in df.columns else None
    dupes = df["symbol"][df["symbol"].duplicated()].tolist()
    if dupes:
        raise ValueError(f"{path}: duplicate symbols {dupes}")
    return df[["symbol", "exchange", "sector"]].reset_index(drop=True)
