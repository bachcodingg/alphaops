"""Turn an alpha (a date x symbol signal) into a daily-rebalanced long/short book and PnL.

Timing convention (the most important thing to get right):

* The alpha value on date *t* is computed from data available at the close of *t*.
* ``delay = d`` means the position held from close *t* to close *t+1* is built from the
  alpha of date *t - d*. ``delay=1`` (the default and the realistic choice) trades a day
  after the signal is known. ``delay=0`` assumes you can trade at the same close the
  signal uses, which is optimistic.
* PnL on day *t+1* = sum over stocks of position(t) * return(t+1).

Pipeline per date: universe filter -> decay -> neutralize -> scale to gross 1 ->
truncate -> lag by delay -> PnL. All metrics are relative to gross exposure (1.0).
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd

from alphaops.engine.operators import group_neutralize, ts_decay_linear
from alphaops.engine.panel import Panel

TRADING_DAYS = 252
NEUTRALIZATIONS = ("none", "market", "sector", "exchange")


@dataclass(frozen=True)
class SimSettings:
    delay: int = 1
    decay: int = 0
    neutralization: str = "market"
    truncation: float = 0.1
    universe: int | None = None  # top-N stocks by 20-day traded value; None = all
    cost_bps: float = 0.0  # one-way transaction cost per unit traded
    start: str | None = None
    end: str | None = None
    os_start: str | None = None  # out-of-sample split date for IS/OS Sharpe

    def __post_init__(self) -> None:
        if self.delay not in (0, 1, 2, 3, 4, 5):
            raise ValueError("delay must be an integer in 0..5")
        if self.decay < 0:
            raise ValueError("decay must be >= 0")
        if self.neutralization not in NEUTRALIZATIONS:
            raise ValueError(f"neutralization must be one of {NEUTRALIZATIONS}")
        if not 0 < self.truncation <= 1:
            raise ValueError("truncation must be in (0, 1]")
        if self.universe is not None and self.universe < 2:
            raise ValueError("universe must be >= 2")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SimResult:
    settings: SimSettings
    weights: pd.DataFrame  # target weights by signal date (after universe/neutralize/scale)
    positions: pd.DataFrame  # weights actually held over (t, t+1], i.e. lagged by delay
    pnl: pd.Series  # daily PnL as a fraction of gross exposure, net of costs
    turnover: pd.Series
    metrics: dict[str, float]
    yearly: pd.DataFrame


def universe_mask(panel: Panel, top_n: int | None) -> pd.DataFrame:
    """Stocks eligible on each date. Liquidity rank uses *yesterday's* adv20 so the
    universe itself cannot leak today's information."""
    has_price = panel.fields["close"].notna()
    if top_n is None:
        return has_price
    adv = panel.fields["adv20"].shift(1)
    ranks = adv.rank(axis=1, ascending=False, method="first")
    return has_price & (ranks <= top_n)


def _truncate(w: pd.DataFrame, cap: float, iterations: int = 20) -> pd.DataFrame:
    for _ in range(iterations):
        if (w.abs().max(axis=1) <= cap + 1e-12).all():
            break
        w = w.clip(lower=-cap, upper=cap)
        w = w.div(w.abs().sum(axis=1).replace(0, np.nan), axis=0)
    return w


def build_weights(alpha: pd.DataFrame, panel: Panel, s: SimSettings) -> pd.DataFrame:
    mask = universe_mask(panel, s.universe)
    a = alpha.reindex(index=panel.dates, columns=panel.symbols).where(mask)
    if s.decay > 0:
        a = ts_decay_linear(a, s.decay).where(mask)
    if s.neutralization == "market":
        a = a.sub(a.mean(axis=1), axis=0)
    elif s.neutralization in ("sector", "exchange"):
        a = group_neutralize(a, panel.groups[s.neutralization])
    gross = a.abs().sum(axis=1).replace(0, np.nan)
    w = a.div(gross, axis=0)
    w = _truncate(w, s.truncation)
    return w.fillna(0.0)


def _sharpe(pnl: pd.Series) -> float:
    if len(pnl) < 2:
        return math.nan
    sd = pnl.std()
    return float(pnl.mean() / sd * math.sqrt(TRADING_DAYS)) if sd > 0 else math.nan


def _max_drawdown(pnl: pd.Series) -> float:
    if pnl.empty:
        return math.nan
    equity = pnl.cumsum()
    return float((equity.cummax().clip(lower=0) - equity).max())


def compute_metrics(
    pnl: pd.Series, turnover: pd.Series, positions: pd.DataFrame
) -> dict[str, float]:
    ann_ret = float(pnl.mean() * TRADING_DAYS) if len(pnl) else math.nan
    sharpe = _sharpe(pnl)
    avg_turnover = float(turnover.mean()) if len(turnover) else math.nan
    traded = float(turnover.sum())
    held = positions.loc[pnl.index] if len(pnl) else positions.iloc[0:0]
    fitness = (
        sharpe * math.sqrt(abs(ann_ret) / max(avg_turnover, 0.125))
        if not (math.isnan(sharpe) or math.isnan(avg_turnover))
        else math.nan
    )
    return {
        "sharpe": sharpe,
        "returns": ann_ret,
        "turnover": avg_turnover,
        "fitness": fitness,
        "max_drawdown": _max_drawdown(pnl),
        "margin_bps": float(pnl.sum() / traded * 1e4) if traded > 0 else math.nan,
        "hit_rate": float((pnl > 0).mean()) if len(pnl) else math.nan,
        "long_count": float((held > 0).sum(axis=1).mean()) if len(held) else math.nan,
        "short_count": float((held < 0).sum(axis=1).mean()) if len(held) else math.nan,
        "days": float(len(pnl)),
    }


def yearly_table(pnl: pd.Series, turnover: pd.Series) -> pd.DataFrame:
    rows = []
    for year, p in pnl.groupby(pnl.index.year):
        t = turnover.loc[p.index]
        rows.append(
            {
                "year": int(year),
                "sharpe": _sharpe(p),
                "returns": float(p.mean() * TRADING_DAYS),
                "turnover": float(t.mean()),
                "max_drawdown": _max_drawdown(p),
                "days": len(p),
            }
        )
    return pd.DataFrame(rows).set_index("year") if rows else pd.DataFrame()


def simulate(alpha: pd.DataFrame, panel: Panel, settings: SimSettings | None = None) -> SimResult:
    s = settings or SimSettings()
    weights = build_weights(alpha, panel, s)
    positions = weights.shift(s.delay).fillna(0.0)
    returns = panel.fields["returns"].fillna(0.0)

    pnl_gross = (positions.shift(1) * returns).sum(axis=1)
    turnover = (positions - positions.shift(1).fillna(0.0)).abs().sum(axis=1)
    # Cost is paid when trading into position(t), i.e. charged against the PnL of t+1.
    pnl = pnl_gross - turnover.shift(1).fillna(0.0) * s.cost_bps / 1e4
    turnover_on_pnl_day = turnover.shift(1).fillna(0.0)

    # Evaluate only once a book exists, and within the requested date range.
    active = positions.shift(1).abs().sum(axis=1) > 0
    if active.any():
        first = active.idxmax()
        pnl, turnover_on_pnl_day = pnl.loc[first:], turnover_on_pnl_day.loc[first:]
    else:
        pnl, turnover_on_pnl_day = pnl.iloc[0:0], turnover_on_pnl_day.iloc[0:0]
    if s.start:
        pnl, turnover_on_pnl_day = pnl.loc[s.start :], turnover_on_pnl_day.loc[s.start :]
    if s.end:
        pnl, turnover_on_pnl_day = pnl.loc[: s.end], turnover_on_pnl_day.loc[: s.end]

    # Turnover is attributed to the PnL day it affects; positions held into that day.
    held = positions.shift(1).fillna(0.0)
    metrics = compute_metrics(pnl, turnover_on_pnl_day, held)
    if s.os_start:
        metrics["is_sharpe"] = _sharpe(pnl.loc[: pd.Timestamp(s.os_start) - pd.Timedelta(days=1)])
        metrics["os_sharpe"] = _sharpe(pnl.loc[s.os_start :])
    return SimResult(
        settings=s,
        weights=weights,
        positions=positions,
        pnl=pnl,
        turnover=turnover_on_pnl_day,
        metrics=metrics,
        yearly=yearly_table(pnl, turnover_on_pnl_day),
    )
