"""Operator library for the alpha language.

Data model: every field is a ``DataFrame`` indexed by date (rows) and symbol (columns).
Time-series operators (``ts_*``) work down each column using only past and current
rows, so they cannot look ahead. Cross-sectional operators work across each row (one
date). Group operators work across each row within groups such as sector.

Rolling windows need at least ``ceil(0.75 * d)`` valid observations, so one missing bar
(e.g. a quarantined row) doesn't wipe out ``d`` days of signal.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

Frame = pd.DataFrame
Value = pd.DataFrame | float

MIN_PERIODS_FRAC = 0.75


def _mp(d: int) -> int:
    return max(1, math.ceil(MIN_PERIODS_FRAC * d))


def _clean(x: Value) -> Value:
    if isinstance(x, pd.DataFrame):
        return x.replace([np.inf, -np.inf], np.nan)
    if isinstance(x, float) and not math.isfinite(x):
        return math.nan
    return x


def _is_frame(x: object) -> bool:
    return isinstance(x, pd.DataFrame)


def _truthy(x: Value) -> Value:
    """1.0 where x is non-zero, 0.0 where zero, NaN where NaN."""
    if _is_frame(x):
        return (x != 0).astype("float64").where(x.notna())
    return math.nan if (isinstance(x, float) and math.isnan(x)) else float(x != 0)


# ------------------------------------------------------------------- time series
def ts_delay(x: Frame, d: int) -> Frame:
    return x.shift(d)


def ts_delta(x: Frame, d: int) -> Frame:
    return x - x.shift(d)


def ts_mean(x: Frame, d: int) -> Frame:
    return x.rolling(d, min_periods=_mp(d)).mean()


def ts_sum(x: Frame, d: int) -> Frame:
    # mean * d keeps the scale stable when a few observations are missing
    return x.rolling(d, min_periods=_mp(d)).mean() * d


def ts_std(x: Frame, d: int) -> Frame:
    return x.rolling(d, min_periods=max(2, _mp(d))).std()


def ts_min(x: Frame, d: int) -> Frame:
    return x.rolling(d, min_periods=_mp(d)).min()


def ts_max(x: Frame, d: int) -> Frame:
    return x.rolling(d, min_periods=_mp(d)).max()


def ts_rank(x: Frame, d: int) -> Frame:
    """Percentile rank (0..1] of today's value within the trailing ``d`` values."""
    return x.rolling(d, min_periods=_mp(d)).rank(pct=True)


def ts_zscore(x: Frame, d: int) -> Frame:
    return (x - ts_mean(x, d)) / ts_std(x, d)


def ts_av_diff(x: Frame, d: int) -> Frame:
    return x - ts_mean(x, d)


def ts_scale(x: Frame, d: int) -> Frame:
    lo, hi = ts_min(x, d), ts_max(x, d)
    return (x - lo) / (hi - lo)


def ts_corr(x: Frame, y: Frame, d: int) -> Frame:
    return x.rolling(d, min_periods=max(3, _mp(d))).corr(y)


def ts_cov(x: Frame, y: Frame, d: int) -> Frame:
    return x.rolling(d, min_periods=max(3, _mp(d))).cov(y)


def ts_decay_linear(x: Frame, d: int) -> Frame:
    """Linearly-weighted moving average, weights d, d-1, ..., 1 (today weighs most).
    Missing values are skipped and the remaining weights renormalized."""
    num = x.fillna(0.0) * float(d)
    den = x.notna().astype("float64") * float(d)
    for i in range(1, d):
        xi = x.shift(i)
        w = float(d - i)
        num = num + xi.fillna(0.0) * w
        den = den + xi.notna().astype("float64") * w
    return num / den.where(den > 0)


def _ts_arg(x: Frame, d: int, use_max: bool) -> Frame:
    arr = x.to_numpy(dtype="float64")
    out = np.full(arr.shape, np.nan)
    if len(arr) >= d:
        win = np.lib.stride_tricks.sliding_window_view(arr, d, axis=0)  # (T-d+1, N, d)
        nan_any = np.isnan(win).any(axis=-1)
        filled = np.where(np.isnan(win), -np.inf if use_max else np.inf, win)
        idx = filled.argmax(axis=-1) if use_max else filled.argmin(axis=-1)
        days_ago = (d - 1 - idx).astype("float64")
        days_ago[nan_any] = np.nan
        out[d - 1 :] = days_ago
    return pd.DataFrame(out, index=x.index, columns=x.columns)


def ts_argmax(x: Frame, d: int) -> Frame:
    """Days since the maximum of the trailing ``d`` values (0 = today)."""
    return _ts_arg(x, d, use_max=True)


def ts_argmin(x: Frame, d: int) -> Frame:
    """Days since the minimum of the trailing ``d`` values (0 = today)."""
    return _ts_arg(x, d, use_max=False)


def ts_backfill(x: Frame, d: int) -> Frame:
    """Carry the last valid value forward for up to ``d`` days."""
    return x.ffill(limit=d)


# ---------------------------------------------------------------- cross-section
def rank(x: Frame) -> Frame:
    """Cross-sectional percentile rank in (0, 1]."""
    return x.rank(axis=1, pct=True)


def zscore(x: Frame) -> Frame:
    return x.sub(x.mean(axis=1), axis=0).div(x.std(axis=1), axis=0)


def normalize(x: Frame) -> Frame:
    """Subtract the cross-sectional mean."""
    return x.sub(x.mean(axis=1), axis=0)


def scale(x: Frame, scale: float = 1.0) -> Frame:
    """Rescale so the sum of absolute values per date equals ``scale``."""
    return x.div(x.abs().sum(axis=1).replace(0, np.nan), axis=0) * scale


def winsorize(x: Frame, std: float = 4.0) -> Frame:
    mu, sd = x.mean(axis=1), x.std(axis=1)
    return x.clip(lower=mu - std * sd, upper=mu + std * sd, axis=0)


# ------------------------------------------------------------------------ groups
def _group_transform(x: Frame, groups: pd.Series, how: str) -> Frame:
    keys = groups.reindex(x.columns).to_numpy()
    grouped = x.T.groupby(keys, dropna=True)
    res = grouped.rank(pct=True) if how == "rank" else grouped.transform(how)
    return res.T.reindex(index=x.index, columns=x.columns)


def group_mean(x: Frame, groups: pd.Series) -> Frame:
    return _group_transform(x, groups, "mean")


def group_neutralize(x: Frame, groups: pd.Series) -> Frame:
    """Subtract the group mean, so each group is long/short balanced."""
    return x - group_mean(x, groups)


def group_rank(x: Frame, groups: pd.Series) -> Frame:
    return _group_transform(x, groups, "rank")


def group_zscore(x: Frame, groups: pd.Series) -> Frame:
    return (x - group_mean(x, groups)) / _group_transform(x, groups, "std")


# --------------------------------------------------------------------- elementwise
def abs_(x: Value) -> Value:
    return x.abs() if _is_frame(x) else abs(x)


def log(x: Value) -> Value:
    if _is_frame(x):
        return np.log(x.where(x > 0))
    return math.log(x) if x > 0 else math.nan


def sqrt(x: Value) -> Value:
    if _is_frame(x):
        return np.sqrt(x.where(x >= 0))
    return math.sqrt(x) if x >= 0 else math.nan


def sign(x: Value) -> Value:
    return np.sign(x) if _is_frame(x) else float(np.sign(x))


def signed_power(x: Value, e: float) -> Value:
    return sign(x) * abs_(x) ** e


def power(x: Value, e: Value) -> Value:
    return x**e


def max_(x: Value, y: Value) -> Value:
    if _is_frame(x) or _is_frame(y):
        a, b = (x, y) if _is_frame(x) else (y, x)
        return a.where(a >= b, b) if _is_frame(b) else a.clip(lower=b)
    return max(x, y)


def min_(x: Value, y: Value) -> Value:
    if _is_frame(x) or _is_frame(y):
        a, b = (x, y) if _is_frame(x) else (y, x)
        return a.where(a <= b, b) if _is_frame(b) else a.clip(upper=b)
    return min(x, y)


def inverse(x: Value) -> Value:
    return 1.0 / x


def reverse(x: Value) -> Value:
    return -x


def is_nan(x: Value) -> Value:
    if _is_frame(x):
        return x.isna().astype("float64")
    return float(math.isnan(x))


def if_else(cond: Value, a: Value, b: Value) -> Value:
    c = _truthy(cond)
    if not _is_frame(c):
        if isinstance(c, float) and math.isnan(c):
            return math.nan
        return a if c else b
    template = c
    a_f = a if _is_frame(a) else pd.DataFrame(a, index=template.index, columns=template.columns)
    b_f = b if _is_frame(b) else pd.DataFrame(b, index=template.index, columns=template.columns)
    return a_f.where(c == 1, b_f).where(c.notna())


def trade_when(trigger: Value, alpha: Frame, exit_cond: Value) -> Frame:
    """Take ``alpha`` when ``trigger`` is true, close (NaN) when ``exit_cond`` is true,
    otherwise hold the previous value."""
    if not _is_frame(alpha):
        raise TypeError("trade_when: alpha must be an expression over fields")

    def as_mask(v: Value) -> Frame:
        if _is_frame(v):
            return _truthy(v).fillna(0.0) == 1
        return pd.DataFrame(bool(v), index=alpha.index, columns=alpha.columns)

    enter, leave = as_mask(trigger), as_mask(exit_cond)
    marker = np.inf  # sentinel: "flat"; alpha itself never contains inf after cleaning
    state = alpha.where(enter)
    state = state.mask(leave & ~enter, marker)
    return state.ffill().replace(marker, np.nan)


# ------------------------------------------------------------------------- registry
@dataclass(frozen=True)
class OpSpec:
    name: str
    func: Callable[..., Value]
    args: tuple[str, ...]  # per positional arg: "frame" | "value" | "int" | "num" | "group"
    category: str
    doc: str
    optional: int = 0  # how many trailing args may be omitted
    kwargs: dict[str, str] = field(default_factory=dict)

    @property
    def signature(self) -> str:
        names = {"frame": "x", "value": "x", "int": "d", "num": "p", "group": "group"}
        parts = []
        seen: dict[str, int] = {}
        for kind in self.args:
            base = names[kind]
            seen[base] = seen.get(base, 0) + 1
            parts.append(base if seen[base] == 1 else f"{base}{seen[base]}")
        parts += [f"{k}=" for k in self.kwargs]
        return f"{self.name}({', '.join(parts)})"


def _spec(name, func, args, category, doc, **kw) -> OpSpec:
    return OpSpec(name, func, tuple(args), category, doc, **kw)


OPERATORS: dict[str, OpSpec] = {
    s.name: s
    for s in [
        _spec("ts_delay", ts_delay, ["frame", "int"], "time-series", "value d days ago"),
        _spec("ts_delta", ts_delta, ["frame", "int"], "time-series", "x - ts_delay(x, d)"),
        _spec("ts_mean", ts_mean, ["frame", "int"], "time-series", "rolling mean"),
        _spec("ts_sum", ts_sum, ["frame", "int"], "time-series", "rolling sum"),
        _spec("ts_std", ts_std, ["frame", "int"], "time-series", "rolling standard deviation"),
        _spec("ts_min", ts_min, ["frame", "int"], "time-series", "rolling minimum"),
        _spec("ts_max", ts_max, ["frame", "int"], "time-series", "rolling maximum"),
        _spec(
            "ts_rank",
            ts_rank,
            ["frame", "int"],
            "time-series",
            "percentile of today's value within the window",
        ),
        _spec("ts_zscore", ts_zscore, ["frame", "int"], "time-series", "(x - ts_mean) / ts_std"),
        _spec("ts_av_diff", ts_av_diff, ["frame", "int"], "time-series", "x - ts_mean(x, d)"),
        _spec(
            "ts_scale",
            ts_scale,
            ["frame", "int"],
            "time-series",
            "(x - ts_min) / (ts_max - ts_min)",
        ),
        _spec("ts_corr", ts_corr, ["frame", "frame", "int"], "time-series", "rolling correlation"),
        _spec("ts_cov", ts_cov, ["frame", "frame", "int"], "time-series", "rolling covariance"),
        _spec(
            "ts_decay_linear",
            ts_decay_linear,
            ["frame", "int"],
            "time-series",
            "linearly weighted moving average (today weighs most)",
        ),
        _spec(
            "ts_argmax", ts_argmax, ["frame", "int"], "time-series", "days since the window maximum"
        ),
        _spec(
            "ts_argmin", ts_argmin, ["frame", "int"], "time-series", "days since the window minimum"
        ),
        _spec(
            "ts_backfill",
            ts_backfill,
            ["frame", "int"],
            "time-series",
            "forward-fill missing values for up to d days",
        ),
        _spec("rank", rank, ["frame"], "cross-sectional", "percentile rank across stocks"),
        _spec("zscore", zscore, ["frame"], "cross-sectional", "standardize across stocks"),
        _spec(
            "normalize",
            normalize,
            ["frame"],
            "cross-sectional",
            "subtract the cross-sectional mean",
        ),
        _spec(
            "scale",
            scale,
            ["frame"],
            "cross-sectional",
            "rescale so sum(|x|) = scale (default 1)",
            kwargs={"scale": "num"},
        ),
        _spec(
            "winsorize",
            winsorize,
            ["frame"],
            "cross-sectional",
            "clip at mean ± std·sd (default 4)",
            kwargs={"std": "num"},
        ),
        _spec(
            "group_neutralize",
            group_neutralize,
            ["frame", "group"],
            "group",
            "subtract the group mean",
        ),
        _spec("group_rank", group_rank, ["frame", "group"], "group", "percentile rank in group"),
        _spec("group_zscore", group_zscore, ["frame", "group"], "group", "z-score in group"),
        _spec("group_mean", group_mean, ["frame", "group"], "group", "group mean"),
        _spec("abs", abs_, ["value"], "elementwise", "absolute value"),
        _spec("log", log, ["value"], "elementwise", "natural log (NaN for x <= 0)"),
        _spec("sqrt", sqrt, ["value"], "elementwise", "square root (NaN for x < 0)"),
        _spec("sign", sign, ["value"], "elementwise", "-1, 0 or 1"),
        _spec("signed_power", signed_power, ["value", "num"], "elementwise", "sign(x) * |x|^p"),
        _spec("power", power, ["value", "value"], "elementwise", "x^y"),
        _spec("max", max_, ["value", "value"], "elementwise", "elementwise maximum"),
        _spec("min", min_, ["value", "value"], "elementwise", "elementwise minimum"),
        _spec("inverse", inverse, ["value"], "elementwise", "1 / x"),
        _spec("reverse", reverse, ["value"], "elementwise", "-x"),
        _spec("is_nan", is_nan, ["value"], "elementwise", "1 where x is missing"),
        _spec(
            "if_else",
            if_else,
            ["value", "value", "value"],
            "logic",
            "a where cond is true, else b (same as cond ? a : b)",
        ),
        _spec(
            "trade_when",
            trade_when,
            ["value", "frame", "value"],
            "logic",
            "enter alpha on trigger, exit (NaN) on exit_cond, otherwise hold",
        ),
    ]
}


# Binary operators used by the evaluator --------------------------------------------
def _cmp(fn: Callable[[Value, Value], Value]) -> Callable[[Value, Value], Value]:
    def wrapped(a: Value, b: Value) -> Value:
        res = fn(a, b)
        if _is_frame(res):
            valid = pd.DataFrame(True, index=res.index, columns=res.columns)
            for v in (a, b):
                if _is_frame(v):
                    valid &= v.notna()
            return res.astype("float64").where(valid)
        return float(res)

    return wrapped


def _logical(fn: Callable[[Value, Value], Value]) -> Callable[[Value, Value], Value]:
    def wrapped(a: Value, b: Value) -> Value:
        return fn(_truthy(a), _truthy(b))

    return wrapped


BINARY_OPS: dict[str, Callable[[Value, Value], Value]] = {
    "+": lambda a, b: a + b,
    "-": lambda a, b: a - b,
    "*": lambda a, b: a * b,
    "/": lambda a, b: (a / b) if (_is_frame(a) or _is_frame(b)) else (a / b if b else math.nan),
    "^": lambda a, b: a**b,
    "<": _cmp(lambda a, b: a < b),
    "<=": _cmp(lambda a, b: a <= b),
    ">": _cmp(lambda a, b: a > b),
    ">=": _cmp(lambda a, b: a >= b),
    "==": _cmp(lambda a, b: a == b),
    "!=": _cmp(lambda a, b: a != b),
    "&&": _logical(lambda a, b: a * b),
    "||": _logical(lambda a, b: max_(a, b)),
}

clean = _clean
truthy = _truthy
