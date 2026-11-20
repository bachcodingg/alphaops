import json
from datetime import UTC, date, datetime

import pandas as pd
import pytest

from alphaops.sources import PRICE_COLUMNS, get_source
from alphaops.sources.base import PermanentSourceError, normalize_bars
from alphaops.sources.csv_source import CsvSource
from alphaops.sources.synthetic import SyntheticSource
from alphaops.sources.yahoo import YahooSource


def test_synthetic_is_deterministic_across_windows():
    src = SyntheticSource(n_symbols=3, as_of=date(2024, 12, 31))
    long = src.fetch("SYN001", date(2024, 1, 1), date(2024, 12, 31))
    short = SyntheticSource(n_symbols=3, as_of=date(2024, 12, 31)).fetch(
        "SYN001", date(2024, 6, 3), date(2024, 6, 28)
    )
    merged = short.merge(long, on=["symbol", "date"], suffixes=("", "_long"))
    assert len(merged) == len(short) > 0
    assert (merged["close"] == merged["close_long"]).all()
    assert list(long.columns) == PRICE_COLUMNS


def test_synthetic_respects_as_of_and_limits():
    src = SyntheticSource(n_symbols=10, as_of=date(2024, 3, 1))
    bars = src.fetch("SYN000", date(2016, 1, 1), date(2030, 1, 1))
    assert bars["date"].max() <= pd.Timestamp("2024-03-01")
    ret = bars["close"].pct_change().abs().dropna()
    assert ret.max() < 0.071  # SYN000 is HOSE: +-7% limit (plus rounding)
    with pytest.raises(PermanentSourceError):
        src.fetch("NOPE", date(2024, 1, 1), date(2024, 2, 1))


def test_csv_source(tmp_path):
    pd.DataFrame(
        {
            "Time": ["2024-01-02", "2024-01-03"],
            "Open": [1, 2],
            "High": [1, 2],
            "Low": [1, 2],
            "Close": [1, 2],
            "Volume": [10, 20],
        }
    ).to_csv(tmp_path / "AAA.csv", index=False)
    src = CsvSource(tmp_path)
    assert src.symbols() == ["AAA"]
    bars = src.fetch("AAA", date(2024, 1, 3), date(2024, 1, 31))
    assert bars["close"].tolist() == [2.0]
    with pytest.raises(PermanentSourceError):
        src.fetch("BBB", date(2024, 1, 1), date(2024, 1, 31))


def test_normalize_rejects_missing_columns():
    with pytest.raises(PermanentSourceError, match="missing columns"):
        normalize_bars(pd.DataFrame({"date": ["2024-01-01"], "close": [1]}), "X")


def chart_payload(rows, adj=None):
    ts = [int(datetime.fromisoformat(d + "T02:00:00+00:00").timestamp()) for d, *_ in rows]
    quote = {
        k: [r[i + 1] for r in rows]
        for i, k in enumerate(["open", "high", "low", "close", "volume"])
    }
    ind = {"quote": [quote]}
    if adj is not None:
        ind["adjclose"] = [{"adjclose": adj}]
    return {
        "chart": {
            "result": [{"meta": {"gmtoffset": 25200}, "timestamp": ts, "indicators": ind}],
            "error": None,
        }
    }


def test_yahoo_parse_adjusts_prices():
    payload = chart_payload(
        [("2024-01-02", 100, 110, 90, 100, 1000), ("2024-01-03", 100, 105, 95, 100, 2000)],
        adj=[50.0, 100.0],
    )
    bars = YahooSource.parse_chart(payload, "FPT")
    assert bars["close"].tolist() == [50.0, 100.0]
    assert bars["volume"].tolist() == [2000.0, 2000.0]
    assert bars["date"].dt.strftime("%Y-%m-%d").tolist() == ["2024-01-02", "2024-01-03"]


def test_yahoo_error_payload_is_permanent():
    payload = {
        "chart": {"result": None, "error": {"code": "Not Found", "description": "No data found"}}
    }
    with pytest.raises(PermanentSourceError, match="No data found"):
        YahooSource.parse_chart(payload, "XXX")


def test_yahoo_drops_holiday_and_unfinished_bars():
    now = datetime(2024, 1, 5, 3, 0, tzinfo=UTC)  # 10:00 ICT: session not final
    src = YahooSource(now=now)
    bars = YahooSource.parse_chart(
        chart_payload(
            [
                ("2024-01-02", 10, 11, 9, 10, 100),
                ("2024-01-03", None, None, None, None, None),  # holiday placeholder (null)
                ("2024-01-04", 10, 10, 10, 10, 0),  # holiday placeholder (flat, no volume)
                ("2024-01-05", 10, 12, 9, 11, 50),  # today, unfinished
            ]
        ),
        "FPT",
        adjusted=False,
    )
    kept = src.drop_placeholders(bars, "FPT")
    assert kept["date"].dt.strftime("%Y-%m-%d").tolist() == ["2024-01-02"]


def test_yahoo_fetch_uses_http(monkeypatch):
    src = YahooSource(min_interval_s=0, now=datetime(2024, 1, 10, 12, tzinfo=UTC))
    seen = {}

    def fake_get(url):
        seen["url"] = url
        return chart_payload([("2024-01-02", 10, 11, 9, 10, 100)])

    monkeypatch.setattr(src, "_get_json", fake_get)
    bars = src.fetch("FPT", date(2024, 1, 1), date(2024, 1, 5))
    assert "FPT.VN" in seen["url"] and len(bars) == 1
    json.dumps(seen)  # url is a plain string


def test_registry():
    assert isinstance(get_source("synthetic", n_symbols=2), SyntheticSource)
    with pytest.raises(ValueError):
        get_source("bloomberg")


def test_universe_file(tmp_path):
    from alphaops.universe import load_universe

    p = tmp_path / "u.csv"
    p.write_text("# comment\nsymbol,exchange,sector\n fpt ,HOSE,Tech\nvcb,HOSE,Banks\n")
    uni = load_universe(p)
    assert uni["symbol"].tolist() == ["FPT", "VCB"]
    p.write_text("symbol\nFPT\nFPT\n")
    with pytest.raises(ValueError, match="duplicate"):
        load_universe(p)
    assert len(load_universe("config/universe_hose.csv")) >= 60
