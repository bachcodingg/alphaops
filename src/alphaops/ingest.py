"""Ingestion job: fetch -> validate -> quarantine bad rows -> upsert clean rows -> record.

Failure isolation: one symbol failing (after retries) does not stop the run; it is
recorded and the run is marked ``degraded``. If more than ``max_symbol_failure_rate``
of symbols fail, the upstream is probably down and the run is marked ``failed``.

Incremental mode re-fetches a short ``lookback_days`` window before the last stored
date. That makes the job self-healing after a missed day and lets us detect upstream
restatements of recent history.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Any

import pandas as pd

from alphaops.quality import (
    ERROR,
    CheckResult,
    DQConfig,
    run_dataset_checks,
    run_row_checks,
    split_quarantine,
)
from alphaops.retry import call_with_retry
from alphaops.sources.base import DataSource, PermanentSourceError, TransientSourceError
from alphaops.storage import Store

log = logging.getLogger(__name__)


@dataclass
class IngestReport:
    run_id: str
    status: str = "running"
    duration_s: float = 0.0
    rows_fetched: int = 0
    rows_written: int = 0
    rows_rejected: int = 0
    rows_quarantined: int = 0  # newly quarantined (already-known bad rows are not recounted)
    rows_restated: int = 0
    symbols_ok: list[str] = field(default_factory=list)
    symbols_failed: dict[str, str] = field(default_factory=dict)
    dq: list[CheckResult] = field(default_factory=list)
    error: str | None = None

    def counters(self) -> dict[str, float]:
        return {
            "rows_fetched": self.rows_fetched,
            "rows_written": self.rows_written,
            "rows_rejected": self.rows_rejected,
            "rows_quarantined": self.rows_quarantined,
            "rows_restated": self.rows_restated,
            "symbols_ok": len(self.symbols_ok),
            "symbols_failed": len(self.symbols_failed),
        }

    def dq_failures(self) -> dict[tuple[str, str], int]:
        return {(r.name, r.severity): r.n_failed for r in self.dq}

    def summary(self) -> str:
        lines = [
            f"run {self.run_id}: {self.status.upper()} in {self.duration_s:.1f}s",
            f"rows fetched={self.rows_fetched} written={self.rows_written} "
            f"rejected={self.rows_rejected} (new in quarantine={self.rows_quarantined}) "
            f"restated={self.rows_restated}",
            f"symbols ok={len(self.symbols_ok)} failed={len(self.symbols_failed)}",
        ]
        if self.symbols_failed:
            shown = list(self.symbols_failed.items())[:5]
            lines.append("failed: " + "; ".join(f"{s}: {e}" for s, e in shown))
        flagged = [f"{r.name}({r.severity})={r.n_failed}" for r in self.dq if r.n_failed]
        if flagged:
            lines.append("data quality: " + ", ".join(flagged))
        if self.error:
            lines.append(f"error: {self.error}")
        return "\n".join(lines)


def _fetch_one(
    source: DataSource,
    symbol: str,
    start: date,
    end: date,
    *,
    attempts: int,
    base_delay: float,
    sleep: Callable[[float], None],
) -> pd.DataFrame:
    return call_with_retry(
        lambda: source.fetch(symbol, start, end),
        attempts=attempts,
        base_delay=base_delay,
        retry_on=(TransientSourceError, ConnectionError, TimeoutError),
        sleep=sleep,
        description=f"fetch {symbol}",
    )


def run_ingest(
    store: Store,
    source: DataSource,
    symbols: Sequence[str],
    *,
    start: date,
    end: date,
    incremental: bool = True,
    lookback_days: int = 7,
    retry_attempts: int = 3,
    retry_base_delay: float = 1.0,
    max_symbol_failure_rate: float = 0.2,
    dq_config: DQConfig | None = None,
    as_of: date | None = None,
    sleep: Callable[[float], None] = time.sleep,
) -> IngestReport:
    params: dict[str, Any] = {
        "source": source.name,
        "symbols": len(symbols),
        "start": start,
        "end": end,
        "incremental": incremental,
        "lookback_days": lookback_days,
    }
    run_id = store.start_run("ingest", params)
    report = IngestReport(run_id=run_id)
    t0 = time.monotonic()
    log.info("ingest started", extra={"run_id": run_id, **params})
    try:
        meta = source.symbol_meta()
        if meta is not None:
            store.upsert_symbols(meta[meta["symbol"].isin(symbols)])
        last = store.last_dates() if incremental else {}

        frames = []
        for symbol in symbols:
            fetch_start = start
            if symbol in last:
                fetch_start = max(start, last[symbol] - timedelta(days=lookback_days))
            if fetch_start > end:
                report.symbols_ok.append(symbol)
                continue
            try:
                bars = _fetch_one(
                    source,
                    symbol,
                    fetch_start,
                    end,
                    attempts=retry_attempts,
                    base_delay=retry_base_delay,
                    sleep=sleep,
                )
            except (
                PermanentSourceError,
                TransientSourceError,
                ConnectionError,
                TimeoutError,
            ) as exc:
                report.symbols_failed[symbol] = str(exc)
                log.error(
                    "symbol fetch failed",
                    extra={"run_id": run_id, "symbol": symbol, "error": str(exc)},
                )
                continue
            report.symbols_ok.append(symbol)
            frames.append(bars)
            log.debug("fetched", extra={"symbol": symbol, "rows": len(bars)})

        batch = (
            pd.concat([f for f in frames if not f.empty], ignore_index=True)
            if any(not f.empty for f in frames)
            else pd.DataFrame()
        )
        report.rows_fetched = len(batch)

        if not batch.empty:
            exchanges = dict(store.symbol_meta()[["symbol", "exchange"]].dropna().values)
            context = pd.concat(
                [
                    store.load_tail(
                        sym, grp["date"].min().date(), (dq_config or DQConfig()).stale_days + 2
                    )
                    for sym, grp in batch.groupby("symbol")
                ],
                ignore_index=True,
            )
            row_results = run_row_checks(
                batch, context, exchanges=exchanges, as_of=as_of or end, config=dq_config
            )
            clean, bad = split_quarantine(batch, row_results)
            report.dq = row_results + run_dataset_checks(
                batch, symbols, config=dq_config, row_results=row_results
            )
            report.rows_rejected = len(bad)
            report.rows_quarantined = store.quarantine_rows(bad, run_id=run_id, source=source.name)
            report.rows_written, report.rows_restated = store.upsert_prices(
                clean, source=source.name, run_id=run_id
            )
        else:
            report.dq = run_dataset_checks(batch, symbols, config=dq_config)
        store.record_dq(run_id, report.dq)

        failure_rate = len(report.symbols_failed) / max(1, len(symbols))
        if failure_rate > max_symbol_failure_rate:
            report.status = "failed"
            report.error = (
                f"{len(report.symbols_failed)}/{len(symbols)} symbols failed "
                f"({failure_rate:.0%} > {max_symbol_failure_rate:.0%}); upstream likely down"
            )
        elif report.symbols_failed or report.rows_quarantined:
            report.status = "degraded"
        else:
            report.status = "success"
    except Exception as exc:
        report.status = "failed"
        report.error = f"{type(exc).__name__}: {exc}"
        log.exception("ingest crashed", extra={"run_id": run_id})
    finally:
        report.duration_s = time.monotonic() - t0
        store.finish_run(
            run_id,
            report.status,
            rows_written=report.rows_written,
            rows_quarantined=report.rows_quarantined,
            rows_restated=report.rows_restated,
            symbols_ok=len(report.symbols_ok),
            symbols_failed=len(report.symbols_failed),
            error=report.error,
            details={
                "rows_fetched": report.rows_fetched,
                "rows_rejected": report.rows_rejected,
                "failed_symbols": report.symbols_failed,
                "dq_errors": {
                    r.name: r.n_failed for r in report.dq if r.severity == ERROR and r.n_failed
                },
            },
        )
    log.info(
        "ingest finished",
        extra={
            "run_id": run_id,
            "status": report.status,
            "duration_s": round(report.duration_s, 2),
            **report.counters(),
        },
    )
    return report
