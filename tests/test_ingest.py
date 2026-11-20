from datetime import date

import pytest

from alphaops.ingest import run_ingest
from alphaops.sources.base import DataSource, TransientSourceError
from alphaops.sources.synthetic import SyntheticSource

START, END = date(2024, 1, 1), date(2024, 6, 28)


def no_sleep(_: float) -> None:
    pass


def ingest(store, source, symbols=None, **kw):
    symbols = symbols or source.symbols()
    kw.setdefault("start", START)
    kw.setdefault("end", END)
    return run_ingest(store, source, symbols, sleep=no_sleep, **kw)


def test_clean_ingest_succeeds(store, synthetic):
    report = ingest(store, synthetic)
    assert report.status == "success"
    assert report.rows_written == store.count_prices() == report.rows_fetched > 1000
    assert set(store.symbol_meta()["symbol"]) == set(synthetic.symbols())
    assert store.recent_runs()["status"].iloc[0] == "success"


def test_rerun_is_idempotent_and_incremental(store, synthetic):
    first = ingest(store, synthetic)
    second = ingest(store, synthetic, lookback_days=7)
    assert store.count_prices() == first.rows_written
    assert second.rows_restated == 0
    # incremental: only the lookback window is re-fetched
    assert second.rows_fetched <= 6 * len(synthetic.symbols())


def test_incremental_extends_history(store, synthetic):
    ingest(store, synthetic, end=date(2024, 3, 29))
    n_before = store.count_prices()
    ingest(store, synthetic)
    full = SyntheticSource(n_symbols=12, as_of=END)
    expected = sum(len(full.fetch(s, START, END)) for s in full.symbols())
    assert store.count_prices() == expected > n_before


def test_anomalies_are_quarantined(store):
    src = SyntheticSource(n_symbols=20, anomalies=True, anomaly_prob=0.02, as_of=END)
    report = ingest(store, src)
    assert report.status == "degraded"
    assert report.rows_quarantined > 0
    assert report.rows_written + report.rows_rejected == report.rows_fetched
    q = store.list_quarantine()
    assert len(q) == report.rows_quarantined
    # nothing that failed an error check reached the prices table
    prices = store.load_prices()
    assert (prices["low"] <= prices[["open", "close"]].min(axis=1) + 1e-9).all()
    assert prices["close"].notna().all() and (prices["volume"] >= 0).all()


class FlakySource(DataSource):
    """Fails the first ``fail_times`` calls per symbol, then delegates."""

    name = "flaky"

    def __init__(self, inner, fail_times, always_fail=()):
        self.inner, self.fail_times, self.always_fail = inner, fail_times, set(always_fail)
        self.calls = {}

    def fetch(self, symbol, start, end):
        self.calls[symbol] = self.calls.get(symbol, 0) + 1
        if symbol in self.always_fail or self.calls[symbol] <= self.fail_times:
            raise TransientSourceError(f"{symbol}: boom")
        return self.inner.fetch(symbol, start, end)


def test_transient_errors_are_retried(store, synthetic):
    src = FlakySource(synthetic, fail_times=2)
    report = ingest(store, src, synthetic.symbols(), retry_attempts=3)
    assert report.status == "success"
    assert all(n == 3 for n in src.calls.values())


def test_failing_symbol_degrades_run(store, synthetic):
    syms = synthetic.symbols()
    src = FlakySource(synthetic, fail_times=0, always_fail=[syms[0]])
    report = ingest(store, src, syms, retry_attempts=2)
    assert report.status == "degraded"
    assert list(report.symbols_failed) == [syms[0]]
    assert src.calls[syms[0]] == 2


def test_upstream_outage_fails_run(store, synthetic):
    syms = synthetic.symbols()
    src = FlakySource(synthetic, fail_times=0, always_fail=syms[:6])
    report = ingest(store, src, syms, retry_attempts=1, max_symbol_failure_rate=0.2)
    assert report.status == "failed"
    assert "upstream likely down" in report.error
    assert store.recent_runs()["status"].iloc[0] == "failed"


def test_crash_is_recorded_not_raised(store, synthetic, monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("disk full")

    monkeypatch.setattr(store, "upsert_prices", boom)
    report = ingest(store, synthetic)
    assert report.status == "failed" and "disk full" in report.error
    runs = store.recent_runs()
    assert runs["status"].iloc[0] == "failed" and "disk full" in runs["error"].iloc[0]


@pytest.mark.parametrize("status_field", ["rows_fetched", "rows_written"])
def test_report_counters_exposed(store, synthetic, status_field):
    report = ingest(store, synthetic)
    assert report.counters()[status_field] == getattr(report, status_field)
