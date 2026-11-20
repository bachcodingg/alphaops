import math

import numpy as np
import pandas as pd
import pytest

from alphaops.engine import Panel, SimSettings, evaluate, simulate
from alphaops.engine.simulator import build_weights, compute_metrics, universe_mask


def toy_panel(returns: pd.DataFrame) -> Panel:
    close = (1 + returns.fillna(0)).cumprod() * 100
    fields = {
        "close": close,
        "returns": returns,
        "adv20": pd.DataFrame(1.0, index=close.index, columns=close.columns),
    }
    return Panel(fields=fields, groups={"market": pd.Series("ALL", index=close.columns)})


def test_weights_are_neutral_and_gross_one(panel):
    alpha = evaluate("rank(-returns)", panel)
    w = build_weights(alpha, panel, SimSettings(truncation=0.2))
    active = w.abs().sum(axis=1) > 0
    assert np.allclose(w[active].abs().sum(axis=1), 1.0)
    assert np.allclose(w[active].sum(axis=1), 0.0, atol=1e-9)


def test_truncation_caps_weights(panel):
    alpha = evaluate("ts_rank(volume, 20) ^ 6", panel)
    w = build_weights(alpha, panel, SimSettings(truncation=0.05))
    assert w.abs().max().max() <= 0.05 + 1e-9


def test_sector_neutralization(panel):
    alpha = evaluate("rank(close)", panel)
    w = build_weights(alpha, panel, SimSettings(neutralization="sector"))
    by_sector = w.T.groupby(panel.groups["sector"]).sum().T
    assert np.allclose(by_sector, 0.0, atol=1e-9)


def test_universe_uses_yesterdays_liquidity(panel):
    mask = universe_mask(panel, 10)
    assert (mask.sum(axis=1) <= 10).all()
    adv_ranks = panel.fields["adv20"].shift(1).rank(axis=1, ascending=False, method="first")
    pd.testing.assert_frame_equal(mask, panel.fields["close"].notna() & (adv_ranks <= 10))


def test_hand_computed_pnl_and_delay():
    idx = pd.bdate_range("2024-01-01", periods=5)
    rets = pd.DataFrame(
        {"A": [0.0, 0.01, 0.02, -0.01, 0.03], "B": [0.0, -0.01, 0.0, 0.01, 0.0]}, index=idx
    )
    p = toy_panel(rets)
    alpha = pd.DataFrame({"A": 1.0, "B": -1.0}, index=idx)  # long A, short B every day
    r0 = simulate(alpha, p, SimSettings(delay=0, truncation=1.0))
    # weights +0.5/-0.5 from day 0; PnL(t) = 0.5*rA(t) - 0.5*rB(t) from day 1
    expected = 0.5 * rets["A"] - 0.5 * rets["B"]
    pd.testing.assert_series_equal(r0.pnl, expected.iloc[1:], check_names=False)
    r1 = simulate(alpha, p, SimSettings(delay=1, truncation=1.0))
    pd.testing.assert_series_equal(r1.pnl, expected.iloc[2:], check_names=False)


def test_constant_book_has_zero_turnover_after_entry():
    idx = pd.bdate_range("2024-01-01", periods=6)
    rets = pd.DataFrame({"A": 0.01, "B": -0.01}, index=idx)
    alpha = pd.DataFrame({"A": 1.0, "B": -1.0}, index=idx)
    r = simulate(alpha, toy_panel(rets), SimSettings(delay=0, truncation=1.0))
    assert r.turnover.iloc[0] == pytest.approx(1.0)  # initial entry
    assert r.turnover.iloc[1:].eq(0).all()


def test_costs_reduce_pnl_by_turnover(panel):
    alpha = evaluate("rank(-returns)", panel)
    free = simulate(alpha, panel, SimSettings())
    costly = simulate(alpha, panel, SimSettings(cost_bps=10))
    diff = (free.pnl - costly.pnl).sum()
    assert diff == pytest.approx(free.turnover.sum() * 10 / 1e4, rel=1e-6)


def test_synthetic_reversal_is_found_and_random_is_not(panel):
    """End-to-end sanity: the synthetic market has a planted 5-day reversal."""
    rev = simulate(evaluate("rank(-ts_sum(returns, 5))", panel), panel, SimSettings(decay=3))
    assert rev.metrics["sharpe"] > 0.5
    rng = np.random.default_rng(1)
    noise = pd.DataFrame(
        rng.standard_normal(panel.fields["close"].shape), index=panel.dates, columns=panel.symbols
    )
    rnd = simulate(noise, panel, SimSettings())
    assert abs(rnd.metrics["sharpe"]) < 1.0


def test_metrics_formulas():
    pnl = pd.Series([0.01, -0.005, 0.02, 0.0], index=pd.bdate_range("2024-01-01", periods=4))
    turnover = pd.Series([0.2, 0.2, 0.2, 0.2], index=pnl.index)
    pos = pd.DataFrame({"A": [0.5] * 4, "B": [-0.5] * 4}, index=pnl.index)
    m = compute_metrics(pnl, turnover, pos)
    sharpe = pnl.mean() / pnl.std() * math.sqrt(252)
    assert m["sharpe"] == pytest.approx(sharpe)
    assert m["returns"] == pytest.approx(pnl.mean() * 252)
    assert m["fitness"] == pytest.approx(sharpe * math.sqrt(abs(pnl.mean() * 252) / 0.2))
    assert m["max_drawdown"] == pytest.approx(0.005)
    assert m["margin_bps"] == pytest.approx(pnl.sum() / 0.8 * 1e4)
    assert m["long_count"] == 1 and m["short_count"] == 1


def test_os_split_and_yearly(panel):
    r = simulate(evaluate("rank(-returns)", panel), panel, SimSettings(os_start="2023-01-01"))
    assert {"is_sharpe", "os_sharpe"} <= set(r.metrics)
    assert list(r.yearly.index) == sorted(set(r.pnl.index.year))


@pytest.mark.parametrize(
    "kwargs",
    [
        {"delay": -1},
        {"decay": -2},
        {"neutralization": "country"},
        {"truncation": 0},
        {"universe": 1},
    ],
)
def test_invalid_settings(kwargs):
    with pytest.raises(ValueError):
        SimSettings(**kwargs)
