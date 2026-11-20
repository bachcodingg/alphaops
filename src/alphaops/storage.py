"""SQLite storage: prices, quarantine, run ledger, data-quality results, alpha results.

Design choices
--------------
* ``prices`` has primary key (symbol, date) and writes are UPSERTs, so any job can be
  re-run or backfilled safely (idempotency).
* Bad rows never reach ``prices``; they go to ``quarantine`` with the reasons, where an
  operator can *release* (accept) or *drop* them (see docs/runbook.md).
* Every job writes a row to ``runs`` (start, finish, status, counters, error) so the
  history of the pipeline is queryable: ``alphaops runs``.
* Upserts that change an existing close are counted as *restatements*: upstream silently
  rewriting history is something an operator wants to know about.
"""

from __future__ import annotations

import json
import sqlite3
import uuid
from collections.abc import Iterable
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from alphaops.sources.base import PRICE_COLUMNS, PRICE_FIELDS

SCHEMA = """
CREATE TABLE IF NOT EXISTS symbols (
    symbol   TEXT PRIMARY KEY,
    exchange TEXT,
    sector   TEXT
);
CREATE TABLE IF NOT EXISTS prices (
    symbol      TEXT NOT NULL,
    date        TEXT NOT NULL,
    open        REAL,
    high        REAL,
    low         REAL,
    close       REAL,
    volume      REAL,
    source      TEXT NOT NULL,
    run_id      TEXT NOT NULL,
    ingested_at TEXT NOT NULL,
    PRIMARY KEY (symbol, date)
);
CREATE INDEX IF NOT EXISTS idx_prices_date ON prices(date);
CREATE TABLE IF NOT EXISTS quarantine (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id         TEXT NOT NULL,
    symbol         TEXT NOT NULL,
    date           TEXT NOT NULL,
    open           REAL,
    high           REAL,
    low            REAL,
    close          REAL,
    volume         REAL,
    source         TEXT,
    reasons        TEXT NOT NULL,
    quarantined_at TEXT NOT NULL,
    resolved       INTEGER NOT NULL DEFAULT 0,
    resolution     TEXT,
    resolved_at    TEXT,
    UNIQUE (symbol, date, reasons)
);
CREATE TABLE IF NOT EXISTS runs (
    run_id           TEXT PRIMARY KEY,
    job              TEXT NOT NULL,
    started_at       TEXT NOT NULL,
    finished_at      TEXT,
    status           TEXT NOT NULL,
    params           TEXT,
    rows_written     INTEGER DEFAULT 0,
    rows_quarantined INTEGER DEFAULT 0,
    rows_restated    INTEGER DEFAULT 0,
    symbols_ok       INTEGER DEFAULT 0,
    symbols_failed   INTEGER DEFAULT 0,
    error            TEXT,
    details          TEXT
);
CREATE INDEX IF NOT EXISTS idx_runs_job ON runs(job, started_at);
CREATE TABLE IF NOT EXISTS dq_results (
    run_id     TEXT NOT NULL,
    check_name TEXT NOT NULL,
    severity   TEXT NOT NULL,
    n_failed   INTEGER NOT NULL,
    sample     TEXT,
    PRIMARY KEY (run_id, check_name)
);
CREATE TABLE IF NOT EXISTS alpha_results (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id       TEXT,
    name         TEXT NOT NULL,
    expression   TEXT NOT NULL,
    settings     TEXT NOT NULL,
    metrics      TEXT NOT NULL,
    pnl          TEXT NOT NULL,
    evaluated_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_alpha_name ON alpha_results(name, evaluated_at);
"""

RUN_STATUSES = ("running", "success", "degraded", "failed")


def utcnow() -> datetime:
    return datetime.now(UTC)


def _iso(ts: datetime) -> str:
    return ts.astimezone(UTC).isoformat(timespec="seconds")


def _records(df: pd.DataFrame, columns: list[str]) -> list[tuple[Any, ...]]:
    """DataFrame -> list of tuples with NaN converted to SQL NULL."""
    sub = df[columns].astype(object)
    sub = sub.where(sub.notna(), None)
    return [tuple(r) for r in sub.itertuples(index=False, name=None)]


class Store:
    def __init__(self, path: str | Path) -> None:
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.path, timeout=30)
        self.conn.row_factory = sqlite3.Row
        if self.path != ":memory:":
            self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.executescript(SCHEMA)

    def close(self) -> None:
        self.conn.close()

    def __enter__(self) -> Store:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # ------------------------------------------------------------------ symbols
    def upsert_symbols(self, meta: pd.DataFrame) -> None:
        if meta is None or meta.empty:
            return
        meta = meta.reindex(columns=["symbol", "exchange", "sector"])
        with self.conn:
            self.conn.executemany(
                "INSERT INTO symbols(symbol, exchange, sector) VALUES (?, ?, ?) "
                "ON CONFLICT(symbol) DO UPDATE SET "
                "exchange = COALESCE(excluded.exchange, exchange), "
                "sector = COALESCE(excluded.sector, sector)",
                _records(meta, ["symbol", "exchange", "sector"]),
            )

    def symbol_meta(self) -> pd.DataFrame:
        return pd.read_sql_query("SELECT symbol, exchange, sector FROM symbols", self.conn)

    # ------------------------------------------------------------------- prices
    def _existing_closes(self, rows: pd.DataFrame) -> pd.DataFrame:
        frames = []
        for symbol, grp in rows.groupby("symbol", sort=False):
            frames.append(
                pd.read_sql_query(
                    "SELECT symbol, date, close AS close_old FROM prices "
                    "WHERE symbol = ? AND date BETWEEN ? AND ?",
                    self.conn,
                    params=(symbol, grp["date"].min(), grp["date"].max()),
                )
            )
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    def upsert_prices(self, bars: pd.DataFrame, *, source: str, run_id: str) -> tuple[int, int]:
        """Insert or update bars. Returns (rows_written, rows_restated)."""
        if bars.empty:
            return 0, 0
        rows = bars[PRICE_COLUMNS].copy()
        rows["date"] = rows["date"].dt.strftime("%Y-%m-%d")
        existing = self._existing_closes(rows)
        restated = 0
        if not existing.empty:
            merged = rows.merge(existing, on=["symbol", "date"], how="inner")
            changed = ~np.isclose(
                merged["close"].to_numpy(float),
                merged["close_old"].to_numpy(float),
                rtol=1e-6,
                equal_nan=True,
            )
            restated = int(changed.sum())
        rows["source"] = source
        rows["run_id"] = run_id
        rows["ingested_at"] = _iso(utcnow())
        cols = [*PRICE_COLUMNS, "source", "run_id", "ingested_at"]
        updates = ", ".join(f"{c} = excluded.{c}" for c in cols[2:])
        with self.conn:
            self.conn.executemany(
                f"INSERT INTO prices({', '.join(cols)}) VALUES ({', '.join('?' * len(cols))}) "
                f"ON CONFLICT(symbol, date) DO UPDATE SET {updates}",
                _records(rows, cols),
            )
        return len(rows), restated

    def load_prices(
        self,
        start: date | None = None,
        end: date | None = None,
        symbols: Iterable[str] | None = None,
    ) -> pd.DataFrame:
        sql = f"SELECT {', '.join(PRICE_COLUMNS)} FROM prices WHERE 1=1"
        params: list[Any] = []
        if start:
            sql += " AND date >= ?"
            params.append(start.isoformat())
        if end:
            sql += " AND date <= ?"
            params.append(end.isoformat())
        if symbols is not None:
            symbols = list(symbols)
            sql += f" AND symbol IN ({', '.join('?' * len(symbols))})"
            params.extend(symbols)
        df = pd.read_sql_query(sql + " ORDER BY symbol, date", self.conn, params=params)
        df["date"] = pd.to_datetime(df["date"]).astype("datetime64[ns]")
        for col in PRICE_FIELDS:
            df[col] = df[col].astype("float64")
        return df

    def load_tail(self, symbol: str, before: date, n: int) -> pd.DataFrame:
        """The last ``n`` stored bars for ``symbol`` strictly before ``before``."""
        df = pd.read_sql_query(
            f"SELECT {', '.join(PRICE_COLUMNS)} FROM prices WHERE symbol = ? AND date < ? "
            "ORDER BY date DESC LIMIT ?",
            self.conn,
            params=(symbol, before.isoformat(), n),
        )
        df["date"] = pd.to_datetime(df["date"]).astype("datetime64[ns]")
        return df.iloc[::-1].reset_index(drop=True)

    def last_dates(self) -> dict[str, date]:
        rows = self.conn.execute("SELECT symbol, MAX(date) FROM prices GROUP BY symbol")
        return {sym: date.fromisoformat(d) for sym, d in rows}

    def latest_price_date(self) -> date | None:
        (d,) = self.conn.execute("SELECT MAX(date) FROM prices").fetchone()
        return date.fromisoformat(d) if d else None

    def count_prices(self) -> int:
        return self.conn.execute("SELECT COUNT(*) FROM prices").fetchone()[0]

    # --------------------------------------------------------------- quarantine
    def quarantine_rows(self, bad: pd.DataFrame, *, run_id: str, source: str) -> int:
        """Store rejected rows with their reasons. Returns the number of *new* rows."""
        if bad.empty:
            return 0
        rows = bad.copy()
        rows["date"] = rows["date"].dt.strftime("%Y-%m-%d")
        rows["run_id"] = run_id
        rows["source"] = source
        rows["quarantined_at"] = _iso(utcnow())
        cols = ["run_id", *PRICE_COLUMNS, "source", "reasons", "quarantined_at"]
        before = self.conn.total_changes
        with self.conn:
            self.conn.executemany(
                f"INSERT OR IGNORE INTO quarantine({', '.join(cols)}) "
                f"VALUES ({', '.join('?' * len(cols))})",
                _records(rows, cols),
            )
        return self.conn.total_changes - before

    def list_quarantine(self, include_resolved: bool = False) -> pd.DataFrame:
        sql = "SELECT * FROM quarantine"
        if not include_resolved:
            sql += " WHERE resolved = 0"
        return pd.read_sql_query(sql + " ORDER BY id", self.conn)

    def quarantine_backlog(self) -> int:
        return self.conn.execute("SELECT COUNT(*) FROM quarantine WHERE resolved = 0").fetchone()[0]

    def resolve_quarantine(self, qid: int, action: str) -> dict[str, Any]:
        """``action`` is 'release' (accept the row into prices) or 'drop'."""
        if action not in ("release", "drop"):
            raise ValueError("action must be 'release' or 'drop'")
        row = self.conn.execute(
            "SELECT * FROM quarantine WHERE id = ? AND resolved = 0", (qid,)
        ).fetchone()
        if row is None:
            raise KeyError(f"no unresolved quarantine row with id {qid}")
        if action == "release":
            bars = pd.DataFrame([{c: row[c] for c in PRICE_COLUMNS}])
            bars["date"] = pd.to_datetime(bars["date"])
            self.upsert_prices(bars, source=f"manual-release:{row['source']}", run_id=row["run_id"])
        with self.conn:
            self.conn.execute(
                "UPDATE quarantine SET resolved = 1, resolution = ?, resolved_at = ? WHERE id = ?",
                (action, _iso(utcnow()), qid),
            )
        return dict(row)

    # --------------------------------------------------------------------- runs
    def start_run(self, job: str, params: dict[str, Any] | None = None) -> str:
        now = utcnow()
        run_id = f"{job}-{now:%Y%m%dT%H%M%S}-{uuid.uuid4().hex[:6]}"
        with self.conn:
            self.conn.execute(
                "INSERT INTO runs(run_id, job, started_at, status, params) VALUES (?, ?, ?, ?, ?)",
                (run_id, job, _iso(now), "running", json.dumps(params or {}, default=str)),
            )
        return run_id

    def finish_run(
        self,
        run_id: str,
        status: str,
        *,
        rows_written: int = 0,
        rows_quarantined: int = 0,
        rows_restated: int = 0,
        symbols_ok: int = 0,
        symbols_failed: int = 0,
        error: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        if status not in RUN_STATUSES:
            raise ValueError(f"bad status {status!r}")
        with self.conn:
            self.conn.execute(
                "UPDATE runs SET finished_at = ?, status = ?, rows_written = ?, "
                "rows_quarantined = ?, rows_restated = ?, symbols_ok = ?, symbols_failed = ?, "
                "error = ?, details = ? WHERE run_id = ?",
                (
                    _iso(utcnow()),
                    status,
                    rows_written,
                    rows_quarantined,
                    rows_restated,
                    symbols_ok,
                    symbols_failed,
                    error,
                    json.dumps(details or {}, default=str),
                    run_id,
                ),
            )

    def last_success(self, job: str) -> datetime | None:
        row = self.conn.execute(
            "SELECT MAX(finished_at) FROM runs WHERE job = ? AND status IN ('success', 'degraded')",
            (job,),
        ).fetchone()
        return datetime.fromisoformat(row[0]) if row and row[0] else None

    def recent_runs(self, limit: int = 20, job: str | None = None) -> pd.DataFrame:
        sql = (
            "SELECT run_id, job, started_at, finished_at, status, rows_written, "
            "rows_quarantined, rows_restated, symbols_ok, symbols_failed, error FROM runs"
        )
        params: list[Any] = []
        if job:
            sql += " WHERE job = ?"
            params.append(job)
        sql += " ORDER BY started_at DESC LIMIT ?"
        params.append(limit)
        return pd.read_sql_query(sql, self.conn, params=params)

    def stale_running(self, older_than_hours: float) -> list[str]:
        """Runs stuck in 'running' (process killed before it could record an outcome)."""
        cutoff = _iso(utcnow() - pd.Timedelta(hours=older_than_hours))
        rows = self.conn.execute(
            "SELECT run_id FROM runs WHERE status = 'running' AND started_at < ?", (cutoff,)
        )
        return [r[0] for r in rows]

    # --------------------------------------------------------------- dq results
    def record_dq(self, run_id: str, results: Iterable[Any]) -> None:
        with self.conn:
            self.conn.executemany(
                "INSERT OR REPLACE INTO dq_results(run_id, check_name, severity, n_failed, sample) "
                "VALUES (?, ?, ?, ?, ?)",
                [
                    (run_id, r.name, r.severity, r.n_failed, json.dumps(r.sample(), default=str))
                    for r in results
                ],
            )

    def dq_results(self, run_id: str) -> pd.DataFrame:
        return pd.read_sql_query(
            "SELECT check_name, severity, n_failed, sample FROM dq_results WHERE run_id = ?",
            self.conn,
            params=(run_id,),
        )

    # ------------------------------------------------------------ alpha results
    def save_alpha_result(
        self,
        *,
        run_id: str | None,
        name: str,
        expression: str,
        settings: dict[str, Any],
        metrics: dict[str, Any],
        pnl: pd.Series,
    ) -> int:
        pnl_json = json.dumps({d.strftime("%Y-%m-%d"): round(float(v), 10) for d, v in pnl.items()})
        with self.conn:
            cur = self.conn.execute(
                "INSERT INTO alpha_results(run_id, name, expression, settings, metrics, pnl, "
                "evaluated_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    run_id,
                    name,
                    expression,
                    json.dumps(settings, default=str),
                    json.dumps(metrics, default=str),
                    pnl_json,
                    _iso(utcnow()),
                ),
            )
        return int(cur.lastrowid)

    def latest_alpha_pnls(self, passed_only: bool = False) -> dict[str, pd.Series]:
        """Most recent daily PnL series per alpha name. With ``passed_only``, only alphas
        whose latest evaluation passed all checks (the accepted 'library')."""
        sql = (
            "SELECT name, pnl FROM alpha_results a WHERE id = "
            "(SELECT MAX(id) FROM alpha_results b WHERE b.name = a.name)"
        )
        if passed_only:
            sql += " AND json_extract(metrics, '$.passed') = 1"
        rows = self.conn.execute(sql).fetchall()
        out = {}
        for name, pnl_json in rows:
            data = json.loads(pnl_json)
            s = pd.Series(data, dtype="float64")
            s.index = pd.to_datetime(s.index)
            out[name] = s
        return out

    def latest_alpha_results(self) -> pd.DataFrame:
        return pd.read_sql_query(
            "SELECT id, name, expression, metrics, evaluated_at FROM alpha_results a WHERE id = "
            "(SELECT MAX(id) FROM alpha_results b WHERE b.name = a.name) ORDER BY name",
            self.conn,
        )

    # ------------------------------------------------------------------- health
    def integrity_check(self) -> str:
        return self.conn.execute("PRAGMA quick_check").fetchone()[0]
