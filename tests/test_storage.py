from datetime import date

import pandas as pd
import pytest

from alphaops.storage import Store
from helpers import make_bars


def test_upsert_is_idempotent(store):
    bars = make_bars("AAA", [10, 11, 12])
    assert store.upsert_prices(bars, source="t", run_id="r1") == (3, 0)
    assert store.upsert_prices(bars, source="t", run_id="r2") == (3, 0)
    assert store.count_prices() == 3


def test_restatement_is_counted(store):
    store.upsert_prices(make_bars("AAA", [10, 11, 12]), source="t", run_id="r1")
    written, restated = store.upsert_prices(
        make_bars("AAA", [10, 11.5, 12]), source="t", run_id="r2"
    )
    assert (written, restated) == (3, 1)
    assert store.load_prices()["close"].tolist() == [10, 11.5, 12]


def test_load_filters_and_tail(store):
    store.upsert_prices(make_bars("AAA", [1, 2, 3, 4, 5]), source="t", run_id="r")
    store.upsert_prices(make_bars("BBB", [9, 8, 7, 6, 5]), source="t", run_id="r")
    df = store.load_prices(start=date(2024, 1, 2), end=date(2024, 1, 4), symbols=["AAA"])
    assert df["close"].tolist() == [2, 3, 4]
    assert pd.api.types.is_datetime64_any_dtype(df["date"])
    tail = store.load_tail("BBB", date(2024, 1, 5), 2)
    assert tail["close"].tolist() == [7, 6]
    assert store.last_dates() == {"AAA": date(2024, 1, 5), "BBB": date(2024, 1, 5)}


def test_quarantine_dedup_and_release(store):
    bad = make_bars("AAA", [10.0]).assign(reasons="ohlc_inconsistent")
    assert store.quarantine_rows(bad, run_id="r1", source="t") == 1
    assert store.quarantine_rows(bad, run_id="r2", source="t") == 0  # same issue, not recounted
    qid = int(store.list_quarantine()["id"].iloc[0])
    store.resolve_quarantine(qid, "release")
    assert store.count_prices() == 1
    assert store.quarantine_backlog() == 0
    with pytest.raises(KeyError):
        store.resolve_quarantine(qid, "drop")


def test_run_lifecycle(store):
    run_id = store.start_run("ingest", {"x": 1})
    assert store.recent_runs()["status"].iloc[0] == "running"
    assert store.last_success("ingest") is None
    store.finish_run(run_id, "degraded", rows_written=5)
    assert store.last_success("ingest") is not None
    with pytest.raises(ValueError):
        store.finish_run(run_id, "weird")


def test_alpha_library_passed_only(store):
    pnl = pd.Series([0.01, -0.01], index=pd.bdate_range("2024-01-01", periods=2))
    store.save_alpha_result(
        run_id="r", name="a", expression="x", settings={}, metrics={"passed": True}, pnl=pnl
    )
    store.save_alpha_result(
        run_id="r", name="b", expression="y", settings={}, metrics={"passed": False}, pnl=pnl
    )
    assert set(store.latest_alpha_pnls()) == {"a", "b"}
    assert set(store.latest_alpha_pnls(passed_only=True)) == {"a"}
    pd.testing.assert_series_equal(
        store.latest_alpha_pnls()["a"], pnl, check_freq=False, check_names=False
    )


def test_integrity_and_reopen(tmp_path):
    path = tmp_path / "x.db"
    with Store(path) as s:
        s.upsert_prices(make_bars("AAA", [1.0]), source="t", run_id="r")
    with Store(path) as s:
        assert s.integrity_check() == "ok"
        assert s.count_prices() == 1
