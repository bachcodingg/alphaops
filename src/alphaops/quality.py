"""Data-quality checks for incoming daily bars.

Two severities:

* ``error``: the row is wrong and must not be used. It goes to the quarantine table.
* ``warn``: the row is suspicious but plausible (an illiquid stock with zero volume, a
  trading halt). It is stored and reported, and an alert fires if counts spike.

Row-level checks that need history (returns, stale runs) evaluate the new batch together
with the last few stored bars per symbol (the *context*), so the first bar of a daily
run is compared against yesterday's stored close rather than skipped.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from datetime import date

import pandas as pd

from alphaops.sources.base import PRICE_FIELDS

ERROR, WARN = "error", "warn"

DEFAULT_LIMITS = {"HOSE": 0.07, "HNX": 0.10, "UPCOM": 0.15}


@dataclass(frozen=True)
class DQConfig:
    max_abs_return: float = 0.35  # beyond any VN daily limit (max 15%, 20-30% on listing day)
    limit_tolerance: float = 0.005  # tick-size rounding lets moves slightly exceed the limit
    stale_days: int = 5
    exchange_limits: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_LIMITS))
    default_limit: float = 0.15
    min_date_coverage: float = 0.5  # a date counts as a trading day if >=50% of symbols have it


@dataclass
class CheckResult:
    name: str
    severity: str
    description: str
    failed: pd.DataFrame  # columns: symbol, date, detail (+ _row for row-level checks)
    row_level: bool = True

    @property
    def n_failed(self) -> int:
        return len(self.failed)

    def sample(self, n: int = 5) -> list[dict[str, object]]:
        cols = [c for c in ("symbol", "date", "detail") if c in self.failed.columns]
        out = self.failed[cols].head(n).copy()
        if "date" in out:
            out["date"] = out["date"].dt.strftime("%Y-%m-%d")
        return out.to_dict("records")


def _result(
    name: str,
    severity: str,
    desc: str,
    frame: pd.DataFrame,
    mask: pd.Series,
    detail: Callable[[pd.Index], pd.Series] | str,
) -> CheckResult:
    """Build a result from a boolean mask. ``detail`` is a function of the failing rows'
    index, so explanatory strings are only formatted for the (few) failures."""
    failed = frame.loc[mask & (frame["_row"] >= 0), ["_row", "symbol", "date"]].copy()
    if callable(detail):
        failed["detail"] = detail(failed.index) if len(failed) else pd.Series(dtype="object")
    else:
        failed["detail"] = detail
    return CheckResult(name, severity, desc, failed.reset_index(drop=True))


def _fmt(x: pd.Series) -> pd.Series:
    return x.map(lambda v: f"{v:.4g}")


def run_row_checks(
    batch: pd.DataFrame,
    context: pd.DataFrame | None = None,
    *,
    exchanges: dict[str, str] | None = None,
    as_of: date | None = None,
    config: DQConfig | None = None,
) -> list[CheckResult]:
    """Run all row-level checks. ``batch`` must have a unique index; failures reference it
    through the ``_row`` column so the caller can quarantine exactly those rows."""
    cfg = config or DQConfig()
    exchanges = exchanges or {}
    b = batch.copy()
    b["_row"] = batch.index.to_numpy()
    frames = [b]
    if context is not None and not context.empty:
        frames.insert(0, context.assign(_row=-1))
    f = pd.concat(frames, ignore_index=True).sort_values(["symbol", "date", "_row"])
    results: list[CheckResult] = []

    # --- intrinsic row checks (no history needed) ---------------------------------------
    nulls = f[["open", "high", "low", "close"]].isna()
    results.append(
        _result(
            "missing_price",
            ERROR,
            "open/high/low/close is null",
            f,
            nulls.any(axis=1),
            lambda ix: nulls.loc[ix].apply(lambda r: ",".join(r.index[r]) + " null", axis=1),
        )
    )
    prices = f[["open", "high", "low", "close"]]
    results.append(
        _result(
            "non_positive_price",
            ERROR,
            "a price is <= 0",
            f,
            (prices <= 0).any(axis=1),
            "price <= 0",
        )
    )
    eps = 1e-9
    ohlc_bad = (
        (f["low"] > f[["open", "close"]].min(axis=1) * (1 + eps))
        | (f["high"] < f[["open", "close"]].max(axis=1) * (1 - eps))
        | (f["low"] > f["high"])
    )
    results.append(
        _result(
            "ohlc_inconsistent",
            ERROR,
            "low/high do not bracket open/close",
            f,
            ohlc_bad,
            lambda ix: (
                "O="
                + _fmt(f.loc[ix, "open"])
                + " H="
                + _fmt(f.loc[ix, "high"])
                + " L="
                + _fmt(f.loc[ix, "low"])
                + " C="
                + _fmt(f.loc[ix, "close"])
            ),
        )
    )
    results.append(
        _result(
            "negative_volume",
            ERROR,
            "volume < 0",
            f,
            f["volume"] < 0,
            lambda ix: "volume=" + _fmt(f.loc[ix, "volume"]),
        )
    )
    if as_of is not None:
        results.append(
            _result(
                "future_date",
                ERROR,
                "bar dated after the run date",
                f,
                f["date"] > pd.Timestamp(as_of),
                "after " + as_of.isoformat(),
            )
        )
    dup = f.duplicated(["symbol", "date"], keep="first") & (f["_row"] >= 0)
    results.append(_result("duplicate_bar", ERROR, "duplicate (symbol, date)", f, dup, "duplicate"))

    # --- history-aware checks -----------------------------------------------------------
    intrinsic_bad = set()
    for r in results:
        intrinsic_bad.update(r.failed["_row"].tolist())
    usable = ~f["_row"].isin(intrinsic_bad) | (f["_row"] < 0)
    h = f[usable & ~f.duplicated(["symbol", "date"], keep="first")].copy()
    g = h.groupby("symbol", sort=False)["close"]
    prev1, prev2 = g.shift(1), g.shift(2)
    ret1 = h["close"] / prev1 - 1.0
    ret_skip = h["close"] / prev2 - 1.0
    extreme = ret1.abs() > cfg.max_abs_return
    # Day after a one-bar spike: the big move is a *reversion*; vs. two days ago it's normal.
    reversion = (
        extreme
        & extreme.groupby(h["symbol"]).shift(1, fill_value=False)
        & (ret_skip.abs() <= cfg.max_abs_return)
    )
    extreme &= ~reversion
    results.append(
        _result(
            "extreme_return",
            ERROR,
            f"|daily return| > {cfg.max_abs_return:.0%} (bad tick, unit change or "
            "unadjusted corporate action)",
            h,
            extreme,
            lambda ix: (
                "return="
                + _fmt(ret1.loc[ix])
                + " ratio="
                + (h.loc[ix, "close"] / prev1.loc[ix]).round(3).astype(str)
            ),
        )
    )

    limits = h["symbol"].map(exchanges).map(cfg.exchange_limits).fillna(cfg.default_limit)
    breach = (ret1.abs() > limits + cfg.limit_tolerance) & ~extreme & ~reversion
    results.append(
        _result(
            "price_limit_breach",
            WARN,
            "move exceeds the exchange daily limit",
            h,
            breach,
            lambda ix: "return=" + _fmt(ret1.loc[ix]) + " limit=" + _fmt(limits.loc[ix]),
        )
    )

    same = h["close"].eq(prev1)
    run_id = (~same).groupby(h["symbol"]).cumsum()
    run_len = same.groupby([h["symbol"], run_id]).cumsum()
    results.append(
        _result(
            "stale_price",
            WARN,
            f"close unchanged for >= {cfg.stale_days} sessions",
            h,
            run_len >= cfg.stale_days,
            lambda ix: "unchanged_for=" + run_len.loc[ix].astype(str),
        )
    )
    results.append(
        _result("zero_volume", WARN, "no shares traded", h, h["volume"] == 0, "volume=0")
    )
    return results


def systemic_dates(
    batch: pd.DataFrame,
    row_results: list[CheckResult],
    *,
    min_share: float = 0.2,
    min_symbols: int = 3,
) -> CheckResult:
    """Dates where many symbols fail error checks at once.

    One bad bar is a bad tick; the same failure across a large share of the market on one
    date is a vendor-side incident. Escalate it as one incident instead of reviewing
    dozens of quarantine rows one by one (see docs/runbook.md)."""
    errors = [r.failed for r in row_results if r.severity == ERROR and r.n_failed]
    empty = pd.DataFrame(columns=["symbol", "date", "detail"]).astype({"date": "datetime64[ns]"})
    failed = empty
    if errors and not batch.empty:
        bad = pd.concat(errors)[["symbol", "date"]].drop_duplicates()
        per_date = bad.groupby("date")["symbol"].nunique()
        universe = batch.groupby("date")["symbol"].nunique().reindex(per_date.index)
        share = per_date / universe
        hit = per_date[(per_date >= min_symbols) & (share >= min_share)]
        if not hit.empty:
            failed = pd.DataFrame(
                {
                    "symbol": "*",
                    "date": hit.index,
                    "detail": [
                        f"{n}/{int(universe[d])} symbols failed error checks"
                        for d, n in hit.items()
                    ],
                }
            )
    return CheckResult(
        "systemic_date",
        WARN,
        f"date where >= {min_share:.0%} of symbols failed error checks (vendor incident)",
        failed,
        row_level=False,
    )


def run_dataset_checks(
    batch: pd.DataFrame,
    requested: Iterable[str],
    *,
    config: DQConfig | None = None,
    row_results: list[CheckResult] | None = None,
) -> list[CheckResult]:
    """Checks on the batch as a whole. Reported only: no rows are quarantined."""
    cfg = config or DQConfig()
    requested = list(requested)
    results = []
    if row_results is not None:
        results.append(systemic_dates(batch, row_results))
    present = set(batch["symbol"].unique()) if not batch.empty else set()
    missing_syms = [s for s in requested if s not in present]
    results.append(
        CheckResult(
            "missing_symbol",
            WARN,
            "requested symbol returned no bars",
            pd.DataFrame({"symbol": missing_syms, "date": pd.NaT, "detail": "no bars"}).astype(
                {"date": "datetime64[ns]"}
            ),
            row_level=False,
        )
    )
    gaps = pd.DataFrame(columns=["symbol", "date", "detail"]).astype({"date": "datetime64[ns]"})
    if not batch.empty and len(present) > 1:
        counts = batch.groupby("date")["symbol"].nunique()
        trading_days = counts.index[counts >= cfg.min_date_coverage * len(present)]
        span = batch.groupby("symbol")["date"].agg(["min", "max"])
        have = set(zip(batch["symbol"], batch["date"], strict=True))
        rows = [
            (sym, d)
            for sym, (lo, hi) in span.iterrows()
            for d in trading_days
            if lo <= d <= hi and (sym, d) not in have
        ]
        if rows:
            gaps = pd.DataFrame(rows, columns=["symbol", "date"])
            gaps["detail"] = "missing on a trading day"
    results.append(
        CheckResult(
            "missing_bar",
            WARN,
            "symbol has no bar on a day most symbols traded",
            gaps,
            row_level=False,
        )
    )
    return results


def split_quarantine(
    batch: pd.DataFrame, results: list[CheckResult]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split ``batch`` into (clean, quarantined). Quarantined rows get a ``reasons`` column."""
    reasons: dict[object, list[str]] = {}
    for r in results:
        if r.severity == ERROR and r.row_level:
            for row in r.failed["_row"]:
                reasons.setdefault(row, []).append(r.name)
    bad_idx = list(reasons)
    bad = batch.loc[bad_idx].copy() if bad_idx else batch.iloc[0:0].copy()
    bad["reasons"] = [",".join(sorted(reasons[i])) for i in bad_idx]
    clean = batch.drop(index=bad_idx)
    return clean, bad


def summarize(results: list[CheckResult]) -> dict[str, int]:
    return {r.name: r.n_failed for r in results}


def assert_columns(batch: pd.DataFrame) -> None:
    missing = [c for c in ["symbol", "date", *PRICE_FIELDS] if c not in batch.columns]
    if missing:
        raise ValueError(f"batch missing columns: {missing}")


__all__ = [
    "ERROR",
    "WARN",
    "CheckResult",
    "DQConfig",
    "assert_columns",
    "run_dataset_checks",
    "run_row_checks",
    "split_quarantine",
    "summarize",
]
