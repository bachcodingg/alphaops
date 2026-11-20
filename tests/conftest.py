from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from alphaops.config import Settings
from alphaops.engine import Panel
from alphaops.sources.synthetic import SyntheticSource
from alphaops.storage import Store

AS_OF = date(2024, 12, 31)


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings.from_env(
        {
            "ALPHAOPS_DB": str(tmp_path / "test.db"),
            "ALPHAOPS_METRICS_DIR": str(tmp_path / "metrics"),
        }
    )


@pytest.fixture
def store(settings: Settings):
    with Store(settings.db_path) as s:
        yield s


@pytest.fixture
def synthetic() -> SyntheticSource:
    return SyntheticSource(n_symbols=12, as_of=AS_OF)


@pytest.fixture(scope="session")
def panel() -> Panel:
    src = SyntheticSource(n_symbols=30, as_of=AS_OF)
    prices = pd.concat(
        [src.fetch(s, date(2019, 1, 1), AS_OF) for s in src.symbols()], ignore_index=True
    )
    return Panel.from_prices(prices, src.symbol_meta())


@pytest.fixture
def small_frame() -> pd.DataFrame:
    """10 dates x 3 symbols with simple, hand-checkable values."""
    idx = pd.bdate_range("2024-01-01", periods=10)
    return pd.DataFrame(
        {
            "A": np.arange(1.0, 11.0),
            "B": np.arange(10.0, 0.0, -1.0),
            "C": [1.0, 3, 2, 5, 4, 7, 6, 9, 8, 10],
        },
        index=idx,
    )
