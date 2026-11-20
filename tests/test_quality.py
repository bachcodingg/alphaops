from datetime import date

import numpy as np
import pandas as pd

from alphaops.quality import (
    ERROR,
    run_dataset_checks,
    run_row_checks,
    split_quarantine,
    systemic_dates,
)
from helpers import make_bars


def failed(results, name):
    r = next(r for r in results if r.name == name)
    return sorted(r.failed["_row"].tolist()) if r.row_level else r.failed


def test_clean_data_passes():
    bars = make_bars("AAA", [10, 10.2, 10.1, 10.3, 10.25])
    results = run_row_checks(bars, as_of=date(2025, 1, 1))
    assert all(r.n_failed == 0 for r in results)


def test_intrinsic_errors_are_caught():
    bars = make_bars("AAA", [10, 10.1, 10.2, 10.3, 10.4, 10.5])
    bars.loc[1, "close"] = np.nan
    bars.loc[2, "low"] = -1
    bars.loc[3, "high"] = 5  # below open/close
    bars.loc[4, "volume"] = -100
    res = run_row_checks(bars, as_of=date(2025, 1, 1))
    assert failed(res, "missing_price") == [1]
    assert failed(res, "non_positive_price") == [2]
    assert 3 in failed(res, "ohlc_inconsistent")
    assert failed(res, "negative_volume") == [4]


def test_future_and_duplicate_rows():
    bars = make_bars("AAA", [10, 10.1, 10.2])
    bars = pd.concat([bars, bars.iloc[[1]]], ignore_index=True)
    res = run_row_checks(bars, as_of=date(2024, 1, 1))  # 2024-01-01 is the first bar
    assert failed(res, "future_date") == [1, 2, 3]
    assert failed(res, "duplicate_bar") == [3]


def test_fat_finger_spike_flags_only_the_spike():
    bars = make_bars("AAA", [10, 10.1, 30.3, 10.2, 10.3])
    res = run_row_checks(bars)
    assert failed(res, "extreme_return") == [2]


def test_unadjusted_split_flags_only_the_ex_date():
    bars = make_bars("AAA", [20, 20.2, 10.1, 10.2, 10.0])
    res = run_row_checks(bars)
    assert failed(res, "extreme_return") == [2]


def test_context_supplies_previous_close():
    history = make_bars("AAA", [10, 10.1], start="2024-01-01")
    today = make_bars("AAA", [25.0], start="2024-01-03")
    res = run_row_checks(today, history)
    assert failed(res, "extreme_return") == [0]


def test_price_limit_breach_uses_exchange():
    bars = make_bars("AAA", [10, 10.9])  # +9%
    res_hose = run_row_checks(bars, exchanges={"AAA": "HOSE"})
    res_hnx = run_row_checks(bars, exchanges={"AAA": "HNX"})
    assert failed(res_hose, "price_limit_breach") == [1]
    assert failed(res_hnx, "price_limit_breach") == []


def test_stale_and_zero_volume_are_warnings():
    bars = make_bars("AAA", [10, 10, 10, 10, 10, 10, 10.5])
    bars.loc[6, "volume"] = 0
    res = run_row_checks(bars)
    assert failed(res, "stale_price") == [5]
    assert failed(res, "zero_volume") == [6]
    clean, bad = split_quarantine(bars, res)
    assert bad.empty and len(clean) == len(bars)  # warnings never quarantine


def test_split_quarantine_collects_reasons():
    bars = make_bars("AAA", [10, 10.1, 10.2])
    bars.loc[1, "close"] = np.nan
    bars.loc[1, "volume"] = -5
    res = run_row_checks(bars)
    clean, bad = split_quarantine(bars, res)
    assert list(bad.index) == [1]
    assert bad.loc[1, "reasons"] == "missing_price,negative_volume"
    assert list(clean.index) == [0, 2]


def test_missing_symbols_and_bars():
    a = make_bars("AAA", [10, 10.1, 10.2, 10.3])
    b = make_bars("BBB", [5, 5.1, 5.2, 5.3]).drop(index=2)
    res = run_dataset_checks(pd.concat([a, b], ignore_index=True), ["AAA", "BBB", "CCC"])
    assert failed(res, "missing_symbol")["symbol"].tolist() == ["CCC"]
    gaps = failed(res, "missing_bar")
    assert gaps[["symbol"]].values.ravel().tolist() == ["BBB"]


def test_systemic_date_detects_vendor_incident():
    frames = [make_bars(f"S{i}", [10, 10.1, 10.2]) for i in range(10)]
    batch = pd.concat(frames, ignore_index=True)
    bad_rows = batch.index[(batch["date"] == batch["date"].iloc[1])][:6]
    batch.loc[bad_rows, "high"] = 1.0
    res = run_row_checks(batch)
    sys_res = systemic_dates(batch, res)
    assert sys_res.n_failed == 1
    assert "6/10" in sys_res.failed["detail"].iloc[0]
    assert all(r.severity != ERROR or r.n_failed in (0, 6) for r in res)
