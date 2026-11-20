"""Batch evaluation, health checks, metrics, alerts and retry."""

import json
import urllib.request
from datetime import UTC, date, datetime, timedelta

import pandas as pd
import pytest

from alphaops import alerts
from alphaops.batch import load_alpha_file, max_correlation, run_batch
from alphaops.health import CRITICAL, OK, WARNING, overall_status, run_health
from alphaops.ingest import run_ingest
from alphaops.metrics import emit_job_metrics
from alphaops.retry import call_with_retry
from alphaops.sources.synthetic import SyntheticSource

END = date(2024, 12, 31)

ALPHAS = """
[defaults]
delay = 1
decay = 3

[checks]
min_sharpe = 0.3
min_fitness = 0.0

[[alpha]]
name = "rev"
expression = "rank(-ts_sum(returns, 5))"

[[alpha]]
name = "rev_copy"
expression = "rank(-ts_sum(returns, 5)) * 2"

[[alpha]]
name = "broken"
expression = "rank(nonexistent)"
"""


@pytest.fixture
def loaded_store(store):
    src = SyntheticSource(n_symbols=20, as_of=END)
    run_ingest(store, src, src.symbols(), start=date(2021, 1, 1), end=END, sleep=lambda _: None)
    return store


@pytest.fixture
def alpha_file(tmp_path):
    p = tmp_path / "alphas.toml"
    p.write_text(ALPHAS)
    return p


def test_load_alpha_file(alpha_file):
    specs, checks = load_alpha_file(alpha_file)
    assert [s.name for s in specs] == ["rev", "rev_copy", "broken"]
    assert specs[0].settings.decay == 3 and checks["min_sharpe"] == 0.3


@pytest.mark.parametrize(
    "bad, msg",
    [
        ('[[alpha]]\nname="a"\n', "needs 'name' and 'expression'"),
        ('[[alpha]]\nname="a"\nexpression="x"\n[[alpha]]\nname="a"\nexpression="y"\n', "duplicate"),
        ('[defaults]\nspeed=1\n[[alpha]]\nname="a"\nexpression="x"\n', "unknown"),
        ('[[alpha]]\nname="a"\nexpression="x"\nfoo=1\n', "unknown keys"),
        ("", "no \\[\\[alpha\\]\\]"),
    ],
)
def test_bad_alpha_files(tmp_path, bad, msg):
    p = tmp_path / "bad.toml"
    p.write_text(bad)
    with pytest.raises(ValueError, match=msg):
        load_alpha_file(p)


def test_batch_correlation_against_accepted(loaded_store, alpha_file):
    specs, checks = load_alpha_file(alpha_file)
    report = run_batch(loaded_store, specs, checks)
    by_name = {o.name: o for o in report.outcomes}
    assert report.status == "degraded"  # one alpha has an error
    assert by_name["rev"].passed
    assert by_name["rev_copy"].max_corr == pytest.approx(1.0)
    assert by_name["rev_copy"].checks["correlation"] is False
    assert not by_name["broken"].ok and "unknown field" in by_name["broken"].error
    assert report.counters() == {"alphas_evaluated": 2, "alphas_failed": 1, "alphas_passing": 1}
    # a rerun keeps the same verdicts (the library only holds passing alphas)
    again = {o.name: o for o in run_batch(loaded_store, specs, checks).outcomes}
    assert again["rev"].passed and not again["rev_copy"].passed


def test_max_correlation_requires_overlap():
    idx = pd.bdate_range("2024-01-01", periods=100)
    a = pd.Series(range(100), index=idx, dtype=float)
    corr, name = max_correlation(a, {"b": a * -1, "c": a.iloc[:10]}, 504, 60)
    assert (corr, name) == (pytest.approx(-1.0), "b")


def test_health_on_fresh_and_empty_db(store, settings, loaded_store):
    checks = {
        c.name: c
        for c in run_health(loaded_store, settings, now=datetime(2025, 1, 1, 12, tzinfo=UTC))
    }
    assert checks["db_integrity"].status == OK
    assert checks["data_freshness"].status == OK  # latest bar 2024-12-31


def test_health_flags_stale_pipeline(tmp_path, settings):
    from alphaops.storage import Store

    with Store(tmp_path / "empty.db") as empty:
        checks = run_health(empty, settings)
        assert overall_status(checks) == CRITICAL
        names = {c.name for c in checks if c.status == CRITICAL}
        assert {"ingest_freshness", "data_freshness"} <= names


def test_health_data_lag_levels(loaded_store, settings):
    def status_at(day):
        now = datetime.combine(day, datetime.min.time(), UTC) + timedelta(hours=12)
        return {c.name: c for c in run_health(loaded_store, settings, now=now)}["data_freshness"]

    assert status_at(date(2025, 1, 6)).status == WARNING  # 3 business days missing
    assert status_at(date(2025, 2, 3)).status == CRITICAL


def test_metrics_textfile_and_push_failure(settings, monkeypatch):
    from dataclasses import replace

    def broken_push(*a, **k):
        raise OSError("pushgateway down")

    monkeypatch.setattr("alphaops.metrics.push_to_gateway", broken_push)
    s = replace(settings, pushgateway_url="http://localhost:9")
    path = emit_job_metrics(
        s,
        "ingest",
        status="degraded",
        duration_s=1.5,
        last_success=None,
        counters={"rows_written": 10},
        dq_failures={("extreme_return", "error"): 2},
    )
    text = path.read_text()
    assert 'alphaops_job_status_code{pipeline="ingest"} 1.0' in text
    assert 'alphaops_rows_written{pipeline="ingest"} 10.0' in text
    assert 'check="extreme_return"' in text
    assert 'alphaops_job_last_success_timestamp_seconds{pipeline="ingest"} 0.0' in text


def test_alerts_delivery_and_failure_isolation(settings, monkeypatch):
    from dataclasses import replace

    sent = []

    class Resp:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            return b"ok"

    def fake_urlopen(req, timeout):
        sent.append((req.full_url, req.data))
        if "telegram" in req.full_url:
            raise OSError("network down")
        return Resp()

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    s = replace(
        settings,
        slack_webhook_url="https://hooks.example/x",
        telegram_bot_token="t",
        telegram_chat_id="1",
    )
    delivered = alerts.send_alert(s, "ingest FAILED", "details", "critical")
    assert delivered == 1  # Slack ok, Telegram failed but did not raise
    assert json.loads(sent[0][1])["text"].startswith("🔴 [CRITICAL] ingest FAILED")
    assert alerts.send_alert(settings, "t", "b") == 0  # no channels configured


def test_retry_backoff_and_non_retryable():
    delays, calls = [], []

    def flaky():
        calls.append(1)
        if len(calls) < 3:
            raise TimeoutError("slow")
        return "ok"

    assert (
        call_with_retry(
            flaky,
            attempts=3,
            base_delay=1,
            jitter=False,
            retry_on=(TimeoutError,),
            sleep=delays.append,
        )
        == "ok"
    )
    assert delays == [1, 2]

    def bad():
        raise KeyError("permanent")

    with pytest.raises(KeyError):
        call_with_retry(bad, retry_on=(TimeoutError,), sleep=delays.append)
    with pytest.raises(TimeoutError):
        call_with_retry(
            lambda: (_ for _ in ()).throw(TimeoutError()),
            attempts=2,
            retry_on=(TimeoutError,),
            sleep=lambda _: None,
        )
