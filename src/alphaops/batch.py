"""Evaluate a library of alphas from a TOML file, apply submission-style checks, and
store results (metrics + daily PnL) so later runs can check correlation against them.

File format (see examples/alphas.toml)::

    [defaults]           # SimSettings applied to every alpha
    delay = 1
    decay = 3

    [checks]             # pass/fail thresholds
    min_sharpe = 1.0
    max_correlation = 0.7

    [[alpha]]
    name = "reversal_5d"
    expression = "rank(-ts_delta(close, 5))"
    decay = 5            # per-alpha override
"""

from __future__ import annotations

import logging
import math
import time
import tomllib
from dataclasses import dataclass, field, fields
from pathlib import Path

import pandas as pd

from alphaops.engine import ExpressionError, Panel, SimSettings, evaluate, simulate
from alphaops.storage import Store

log = logging.getLogger(__name__)

DEFAULT_CHECKS: dict[str, float] = {
    "min_sharpe": 1.0,
    "min_fitness": 0.5,
    "min_turnover": 0.01,
    "max_turnover": 0.7,
    "max_correlation": 0.7,
    "corr_lookback_days": 504,
    "min_corr_overlap": 60,
}
_SETTING_KEYS = {f.name for f in fields(SimSettings)}


@dataclass(frozen=True)
class AlphaSpec:
    name: str
    expression: str
    settings: SimSettings
    description: str = ""


@dataclass
class AlphaOutcome:
    name: str
    expression: str
    ok: bool
    error: str | None = None
    metrics: dict[str, float] = field(default_factory=dict)
    max_corr: float = math.nan
    max_corr_with: str | None = None
    checks: dict[str, bool] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return self.ok and bool(self.checks) and all(self.checks.values())


@dataclass
class BatchReport:
    run_id: str
    status: str = "running"
    duration_s: float = 0.0
    outcomes: list[AlphaOutcome] = field(default_factory=list)
    error: str | None = None

    def counters(self) -> dict[str, float]:
        return {
            "alphas_evaluated": sum(o.ok for o in self.outcomes),
            "alphas_failed": sum(not o.ok for o in self.outcomes),
            "alphas_passing": sum(o.passed for o in self.outcomes),
        }

    def table(self) -> pd.DataFrame:
        rows = []
        for o in self.outcomes:
            m = o.metrics
            rows.append(
                {
                    "name": o.name,
                    "sharpe": m.get("sharpe"),
                    "fitness": m.get("fitness"),
                    "returns": m.get("returns"),
                    "turnover": m.get("turnover"),
                    "drawdown": m.get("max_drawdown"),
                    "max_corr": o.max_corr,
                    "corr_with": o.max_corr_with or "",
                    "result": "PASS"
                    if o.passed
                    else (
                        "ERROR"
                        if not o.ok
                        else "fail: " + ",".join(k for k, v in o.checks.items() if not v)
                    ),
                }
            )
        return pd.DataFrame(rows)

    def summary(self) -> str:
        c = self.counters()
        text = (
            f"run {self.run_id}: {self.status.upper()} in {self.duration_s:.1f}s; "
            f"evaluated={c['alphas_evaluated']} failed={c['alphas_failed']} "
            f"passing={c['alphas_passing']}"
        )
        errors = [f"{o.name}: {o.error.splitlines()[0]}" for o in self.outcomes if o.error]
        if errors:
            text += "\nerrors: " + "; ".join(errors[:5])
        if self.error:
            text += f"\nerror: {self.error}"
        return text


def load_alpha_file(path: str | Path) -> tuple[list[AlphaSpec], dict[str, float]]:
    with open(path, "rb") as fh:
        data = tomllib.load(fh)
    defaults = data.get("defaults", {})
    bad = set(defaults) - _SETTING_KEYS
    if bad:
        raise ValueError(f"{path}: unknown [defaults] keys {sorted(bad)}")
    checks = {**DEFAULT_CHECKS, **data.get("checks", {})}
    bad = set(checks) - set(DEFAULT_CHECKS)
    if bad:
        raise ValueError(f"{path}: unknown [checks] keys {sorted(bad)}")
    specs, seen = [], set()
    for i, entry in enumerate(data.get("alpha", [])):
        name, expr = entry.get("name"), entry.get("expression")
        if not name or not expr:
            raise ValueError(f"{path}: alpha #{i + 1} needs 'name' and 'expression'")
        if name in seen:
            raise ValueError(f"{path}: duplicate alpha name {name!r}")
        seen.add(name)
        overrides = {
            k: v for k, v in entry.items() if k not in ("name", "expression", "description")
        }
        bad = set(overrides) - _SETTING_KEYS
        if bad:
            raise ValueError(f"{path}: alpha {name!r} has unknown keys {sorted(bad)}")
        specs.append(
            AlphaSpec(
                name, expr, SimSettings(**{**defaults, **overrides}), entry.get("description", "")
            )
        )
    if not specs:
        raise ValueError(f"{path}: no [[alpha]] entries")
    return specs, checks


def max_correlation(
    pnl: pd.Series, others: dict[str, pd.Series], lookback: int, min_overlap: int
) -> tuple[float, str | None]:
    best, best_name = math.nan, None
    recent = pnl.iloc[-lookback:]
    for name, other in others.items():
        joined = pd.concat([recent, other], axis=1, join="inner").dropna()
        if len(joined) < min_overlap:
            continue
        c = joined.iloc[:, 0].corr(joined.iloc[:, 1])
        if not math.isnan(c) and (math.isnan(best) or c > best):
            best, best_name = float(c), name
    return best, best_name


def apply_checks(
    metrics: dict[str, float], max_corr: float, checks: dict[str, float]
) -> dict[str, bool]:
    def ge(v: float, t: float) -> bool:
        return not math.isnan(v) and v >= t

    return {
        "sharpe": ge(metrics["sharpe"], checks["min_sharpe"]),
        "fitness": ge(metrics["fitness"], checks["min_fitness"]),
        "turnover": checks["min_turnover"] <= metrics["turnover"] <= checks["max_turnover"],
        "correlation": math.isnan(max_corr) or max_corr < checks["max_correlation"],
    }


def run_batch(
    store: Store,
    specs: list[AlphaSpec],
    checks: dict[str, float] | None = None,
    *,
    save: bool = True,
) -> BatchReport:
    checks = {**DEFAULT_CHECKS, **(checks or {})}
    run_id = store.start_run("evaluate", {"alphas": len(specs)})
    report = BatchReport(run_id=run_id)
    t0 = time.monotonic()
    try:
        panel = Panel.from_prices(store.load_prices(), store.symbol_meta())
        pnls: dict[str, pd.Series] = {}
        for spec in specs:
            outcome = AlphaOutcome(spec.name, spec.expression, ok=False)
            try:
                res = simulate(evaluate(spec.expression, panel), panel, spec.settings)
                outcome.ok, outcome.metrics = True, res.metrics
                pnls[spec.name] = res.pnl
            except (ExpressionError, ValueError, TypeError) as exc:
                outcome.error = str(exc)
                log.error("alpha failed", extra={"alpha": spec.name, "error": str(exc)})
            report.outcomes.append(outcome)

        # Like a self-correlation check: an alpha must add something new relative to the
        # alphas already accepted (previous passing runs, then earlier passes in this file).
        accepted = store.latest_alpha_pnls(passed_only=True)
        for o in report.outcomes:
            if not o.ok:
                continue
            others = {k: v for k, v in accepted.items() if k != o.name}
            o.max_corr, o.max_corr_with = max_correlation(
                pnls[o.name],
                others,
                int(checks["corr_lookback_days"]),
                int(checks["min_corr_overlap"]),
            )
            o.checks = apply_checks(o.metrics, o.max_corr, checks)
            if o.passed:
                accepted[o.name] = pnls[o.name]
            if save:
                spec = next(s for s in specs if s.name == o.name)
                store.save_alpha_result(
                    run_id=run_id,
                    name=o.name,
                    expression=o.expression,
                    settings=spec.settings.to_dict(),
                    metrics={**o.metrics, "max_corr": o.max_corr, "passed": o.passed},
                    pnl=pnls[o.name],
                )
        n_failed = sum(not o.ok for o in report.outcomes)
        report.status = (
            "failed" if n_failed == len(specs) else "degraded" if n_failed else "success"
        )
    except Exception as exc:
        report.status, report.error = "failed", f"{type(exc).__name__}: {exc}"
        log.exception("batch crashed", extra={"run_id": run_id})
    finally:
        report.duration_s = time.monotonic() - t0
        c = report.counters()
        store.finish_run(
            run_id,
            report.status,
            rows_written=int(c["alphas_evaluated"]),
            error=report.error,
            details=c,
        )
    return report
