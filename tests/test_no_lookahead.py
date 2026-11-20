"""The single most important property of a backtester: results on date t must not depend
on data after t. We perturb every field after a cutoff and require that alpha values and
PnL up to the cutoff are bit-for-bit identical."""

import numpy as np
import pandas as pd
import pytest

from alphaops.engine import Panel, SimSettings, evaluate, simulate

EXPRESSIONS = [
    "rank(-ts_delta(close, 5))",
    "group_neutralize(-ts_sum(returns, 5), sector)",
    "ts_corr(rank(close), rank(volume), 10)",
    "ts_decay_linear(returns, 7) * ts_rank(volume, 20)",
    "zscore(ts_argmax(close, 15)) - ts_scale(vwap, 30)",
    "a = ts_zscore(dollar_volume, 20); trade_when(a > 1, -returns, a < 0)",
    "winsorize(ts_av_diff(close, 10) / ts_std(close, 10), std=3)",
]


def perturb_after(panel: Panel, cutoff: pd.Timestamp, seed: int = 0) -> Panel:
    rng = np.random.default_rng(seed)
    fields = {}
    for name, df in panel.fields.items():
        noisy = df.copy()
        future = noisy.index > cutoff
        noisy.loc[future] = noisy.loc[future] * rng.uniform(0.5, 1.5, noisy.loc[future].shape)
        fields[name] = noisy
    return Panel(fields=fields, groups=panel.groups)


@pytest.mark.parametrize("expr", EXPRESSIONS)
def test_alpha_values_do_not_look_ahead(panel, expr):
    cutoff = panel.dates[len(panel.dates) // 2]
    base = evaluate(expr, panel)
    pert = evaluate(expr, perturb_after(panel, cutoff))
    pd.testing.assert_frame_equal(base.loc[:cutoff], pert.loc[:cutoff])


@pytest.mark.parametrize("delay", [0, 1])
def test_pnl_does_not_look_ahead(panel, delay):
    cutoff = panel.dates[len(panel.dates) // 2]
    expr = "rank(-ts_sum(returns, 5))"
    s = SimSettings(delay=delay, decay=4, universe=20)
    perturbed = perturb_after(panel, cutoff)
    base = simulate(evaluate(expr, panel), panel, s)
    pert = simulate(evaluate(expr, perturbed), perturbed, s)
    pd.testing.assert_series_equal(base.pnl.loc[:cutoff], pert.pnl.loc[:cutoff])
    pd.testing.assert_frame_equal(base.positions.loc[:cutoff], pert.positions.loc[:cutoff])
