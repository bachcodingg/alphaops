"""Command-line interface.

Exit codes (stable, scripts depend on them):
  0  success
  1  job failed / error
  2  usage error (argparse)
  3  job degraded (completed with partial failures or quarantined rows)
``alphaops health`` instead uses the Nagios convention 0=OK, 1=WARNING, 2=CRITICAL.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from dataclasses import replace
from datetime import date
from pathlib import Path

import pandas as pd

from alphaops import __version__
from alphaops.config import Settings, today_vn
from alphaops.logging_setup import setup_logging
from alphaops.storage import Store

EXIT_OK, EXIT_FAILED, EXIT_DEGRADED = 0, 1, 3
STATUS_EXIT = {"success": EXIT_OK, "degraded": EXIT_DEGRADED, "failed": EXIT_FAILED}

log = logging.getLogger("alphaops")


def _date(s: str) -> date:
    try:
        return date.fromisoformat(s)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid date {s!r}, expected YYYY-MM-DD") from exc


def _fmt_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "(none)"
    with pd.option_context(
        "display.max_rows",
        200,
        "display.width",
        200,
        "display.max_colwidth",
        60,
        "display.float_format",
        "{:.3f}".format,
    ):
        return df.to_string(index=False)


# ------------------------------------------------------------------------- commands
def cmd_ingest(args: argparse.Namespace, settings: Settings) -> int:
    from alphaops.ingest import run_ingest
    from alphaops.jobs import observe
    from alphaops.quality import DQConfig
    from alphaops.sources import get_source
    from alphaops.universe import load_universe

    kwargs: dict[str, object] = {}
    if args.source == "synthetic":
        kwargs = {
            "n_symbols": args.n_symbols,
            "anomalies": args.anomalies,
            "failure_rate": args.chaos,
            "seed": args.seed,
        }
    elif args.source == "csv":
        if not args.csv_dir:
            print("--csv-dir is required for --source csv", file=sys.stderr)
            return EXIT_FAILED
        kwargs = {"directory": args.csv_dir}
    source = get_source(args.source, **kwargs)

    meta = None
    if args.symbols:
        symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    elif args.universe:
        meta = load_universe(args.universe)
        symbols = meta["symbol"].tolist()
    elif hasattr(source, "symbols"):
        symbols = source.symbols()
    else:
        print(f"--universe or --symbols is required for --source {args.source}", file=sys.stderr)
        return EXIT_FAILED

    end = args.end or today_vn()
    with Store(settings.db_path) as store:
        if meta is not None:
            store.upsert_symbols(meta)
        report = run_ingest(
            store,
            source,
            symbols,
            start=args.start,
            end=end,
            incremental=not args.full,
            lookback_days=args.lookback_days,
            retry_attempts=args.retries,
            max_symbol_failure_rate=settings.max_symbol_failure_rate,
            dq_config=DQConfig(max_abs_return=settings.max_abs_return),
        )
        observe(settings, store, "ingest", report, alerts=not args.no_alert)
    print(report.summary())
    return STATUS_EXIT[report.status]


def _settings_from_args(args: argparse.Namespace):
    from alphaops.engine import SimSettings

    return SimSettings(
        delay=args.delay,
        decay=args.decay,
        neutralization=args.neutralization,
        truncation=args.truncation,
        universe=args.universe_size,
        cost_bps=args.cost_bps,
        start=args.start,
        end=args.end,
        os_start=args.os_start,
    )


def cmd_simulate(args: argparse.Namespace, settings: Settings) -> int:
    from alphaops.engine import ExpressionError, Panel, evaluate, simulate

    sim_settings = _settings_from_args(args)
    with Store(settings.db_path) as store:
        panel = Panel.from_prices(store.load_prices(), store.symbol_meta())
    try:
        result = simulate(evaluate(args.expression, panel), panel, sim_settings)
    except ExpressionError as exc:
        print(f"expression error: {exc}", file=sys.stderr)
        return EXIT_FAILED
    if args.pnl_out:
        out = pd.DataFrame(
            {"pnl": result.pnl, "cum_pnl": result.pnl.cumsum(), "turnover": result.turnover}
        )
        out.to_csv(args.pnl_out, index_label="date")
    if args.json:
        print(
            json.dumps(
                {
                    "expression": args.expression,
                    "settings": sim_settings.to_dict(),
                    "metrics": result.metrics,
                },
                default=str,
                indent=2,
            )
        )
        return EXIT_OK
    m = result.metrics
    print(f"expression  : {args.expression}")
    print(f"settings    : {sim_settings.to_dict()}")
    print(
        f"period      : {result.pnl.index.min():%Y-%m-%d} .. {result.pnl.index.max():%Y-%m-%d}"
        f" ({int(m['days'])} days)"
        if len(result.pnl)
        else "period      : (no trading days)"
    )
    keys = [
        "sharpe",
        "fitness",
        "returns",
        "turnover",
        "max_drawdown",
        "margin_bps",
        "hit_rate",
        "long_count",
        "short_count",
        "is_sharpe",
        "os_sharpe",
    ]
    for k in keys:
        if k in m:
            v = m[k]
            missing = v is None or (isinstance(v, float) and math.isnan(v))
            print(f"{k:12s}: {'nan' if missing else f'{v:.4f}'}")
    if not result.yearly.empty:
        print("\nby year:")
        print(_fmt_table(result.yearly.reset_index()))
    return EXIT_OK


def cmd_batch(args: argparse.Namespace, settings: Settings) -> int:
    from alphaops.batch import load_alpha_file, run_batch
    from alphaops.jobs import observe

    specs, checks = load_alpha_file(args.file)
    if args.universe_size is not None:
        specs = [
            replace(s, settings=replace(s.settings, universe=args.universe_size)) for s in specs
        ]
    with Store(settings.db_path) as store:
        report = run_batch(store, specs, checks, save=not args.no_save)
        observe(settings, store, "evaluate", report, alerts=not args.no_alert)
    print(_fmt_table(report.table()))
    print()
    print(report.summary())
    return STATUS_EXIT[report.status]


def cmd_runs(args: argparse.Namespace, settings: Settings) -> int:
    with Store(settings.db_path) as store:
        df = store.recent_runs(args.limit, args.job)
    if not args.verbose:
        df = df.drop(columns=["error"])
    print(_fmt_table(df))
    return EXIT_OK


def cmd_dq(args: argparse.Namespace, settings: Settings) -> int:
    with Store(settings.db_path) as store:
        run_id = args.run_id
        if not run_id:
            runs = store.recent_runs(1, "ingest")
            if runs.empty:
                print("no ingest runs yet")
                return EXIT_OK
            run_id = runs["run_id"].iloc[0]
        df = store.dq_results(run_id)
    print(f"data quality for {run_id}")
    print(_fmt_table(df))
    return EXIT_OK


def cmd_quarantine(args: argparse.Namespace, settings: Settings) -> int:
    with Store(settings.db_path) as store:
        if args.action == "list":
            df = store.list_quarantine(include_resolved=args.all)
            cols = [
                "id",
                "symbol",
                "date",
                "open",
                "high",
                "low",
                "close",
                "volume",
                "reasons",
                "run_id",
            ] + (["resolution"] if args.all else [])
            print(_fmt_table(df[cols] if not df.empty else df))
            return EXIT_OK
        if args.id is None:
            print(f"quarantine {args.action} needs an id", file=sys.stderr)
            return EXIT_FAILED
        try:
            row = store.resolve_quarantine(args.id, args.action)
        except KeyError as exc:
            print(str(exc), file=sys.stderr)
            return EXIT_FAILED
    log.info(
        "quarantine resolved",
        extra={"id": args.id, "action": args.action, "symbol": row["symbol"], "date": row["date"]},
    )
    print(
        f"{args.action}d quarantine row {args.id} ({row['symbol']} {row['date']}: {row['reasons']})"
    )
    return EXIT_OK


def cmd_health(args: argparse.Namespace, settings: Settings) -> int:
    from alphaops.health import EXIT_CODES, overall_status, run_health

    with Store(settings.db_path) as store:
        checks = run_health(store, settings, max_data_lag_days=args.max_data_lag_days)
    status = overall_status(checks)
    if args.json:
        print(json.dumps({"status": status, "checks": [c.to_dict() for c in checks]}, indent=2))
    else:
        for c in checks:
            print(f"[{c.status.upper():8s}] {c.name:20s} {c.detail}")
        print(f"overall: {status.upper()}")
    return EXIT_CODES[status]


def cmd_alphas(args: argparse.Namespace, settings: Settings) -> int:
    with Store(settings.db_path) as store:
        df = store.latest_alpha_results()
    if df.empty:
        print("(no alpha results yet; run `alphaops batch examples/alphas.toml`)")
        return EXIT_OK
    metrics = pd.json_normalize(df["metrics"].map(json.loads))
    out = pd.concat(
        [
            df[["name", "evaluated_at"]],
            metrics[
                [c for c in ("sharpe", "fitness", "turnover", "max_corr", "passed") if c in metrics]
            ],
        ],
        axis=1,
    )
    print(_fmt_table(out))
    return EXIT_OK


def cmd_ops(args: argparse.Namespace, settings: Settings) -> int:
    from alphaops.engine import FIELD_DOCS, GROUP_DOCS, OPERATORS

    if args.markdown:
        print("| Operator | Category | Description |\n|---|---|---|")
        for spec in OPERATORS.values():
            print(f"| `{spec.signature}` | {spec.category} | {spec.doc} |")
        print("\n| Field | Description |\n|---|---|")
        for k, v in FIELD_DOCS.items():
            print(f"| `{k}` | {v} |")
        print("\n| Group | Description |\n|---|---|")
        for k, v in GROUP_DOCS.items():
            print(f"| `{k}` | {v} |")
        return EXIT_OK
    current = None
    for spec in OPERATORS.values():
        if spec.category != current:
            current = spec.category
            print(f"\n{current}:")
        print(f"  {spec.signature:34s} {spec.doc}")
    print("\nfields: " + ", ".join(FIELD_DOCS))
    print("groups: " + ", ".join(GROUP_DOCS))
    print("syntax: + - * / ^  < <= > >= == !=  && || !  cond ? a : b  name = expr; ...")
    return EXIT_OK


# --------------------------------------------------------------------------- parser
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="alphaops",
        description="Market data pipeline and alpha evaluation engine.",
    )
    p.add_argument("--version", action="version", version=f"alphaops {__version__}")
    p.add_argument("--db", type=Path, help="SQLite path (default $ALPHAOPS_DB or data/alphaops.db)")
    p.add_argument("--log-level", default="INFO")
    p.add_argument(
        "--log-format", choices=["json", "text"], help="default: text on a terminal, json otherwise"
    )
    sub = p.add_subparsers(dest="command", required=True)

    from alphaops.sources import SOURCE_NAMES

    ing = sub.add_parser("ingest", help="fetch, validate and store daily bars")
    ing.add_argument("--source", choices=SOURCE_NAMES, default="synthetic")
    ing.add_argument("--universe", type=Path, help="CSV with symbol,exchange,sector")
    ing.add_argument("--symbols", help="comma-separated symbols (overrides --universe)")
    ing.add_argument(
        "--start",
        type=_date,
        default=date(2018, 1, 1),
        help="earliest date to fetch (default 2018-01-01)",
    )
    ing.add_argument("--end", type=_date, help="last date to fetch (default: today, ICT)")
    ing.add_argument(
        "--full", action="store_true", help="re-fetch the whole [start, end] window (backfill)"
    )
    ing.add_argument(
        "--lookback-days",
        type=int,
        default=7,
        help="incremental re-fetch window before the last stored date",
    )
    ing.add_argument("--retries", type=int, default=3, help="attempts per symbol")
    ing.add_argument("--csv-dir", type=Path, help="directory for --source csv")
    ing.add_argument("--n-symbols", type=int, default=50, help="synthetic: number of symbols")
    ing.add_argument("--seed", type=int, default=7, help="synthetic: random seed")
    ing.add_argument("--anomalies", action="store_true", help="synthetic: inject bad data")
    ing.add_argument(
        "--chaos",
        type=float,
        default=0.0,
        help="synthetic: probability of a transient failure per request",
    )
    ing.add_argument("--no-alert", action="store_true", help="do not send alerts")
    ing.set_defaults(func=cmd_ingest)

    def sim_args(sp: argparse.ArgumentParser) -> None:
        sp.add_argument("--delay", type=int, default=1)
        sp.add_argument("--decay", type=int, default=0)
        sp.add_argument(
            "--neutralization", default="market", choices=["none", "market", "sector", "exchange"]
        )
        sp.add_argument("--truncation", type=float, default=0.1)
        sp.add_argument("--universe-size", type=int, help="top-N by 20-day traded value")
        sp.add_argument("--cost-bps", type=float, default=0.0)
        sp.add_argument("--start", help="first PnL date (YYYY-MM-DD)")
        sp.add_argument("--end", help="last PnL date (YYYY-MM-DD)")
        sp.add_argument("--os-start", help="out-of-sample start date for IS/OS Sharpe")

    sim = sub.add_parser("simulate", help="evaluate one alpha expression")
    sim.add_argument("expression")
    sim_args(sim)
    sim.add_argument("--pnl-out", type=Path, help="write daily PnL to CSV")
    sim.add_argument("--json", action="store_true")
    sim.set_defaults(func=cmd_simulate)

    bat = sub.add_parser("batch", help="evaluate an alpha library (TOML) with checks")
    bat.add_argument("file", type=Path)
    bat.add_argument("--universe-size", type=int, help="override universe for all alphas")
    bat.add_argument("--no-save", action="store_true")
    bat.add_argument("--no-alert", action="store_true")
    bat.set_defaults(func=cmd_batch)

    runs = sub.add_parser("runs", help="show recent job runs")
    runs.add_argument("--limit", type=int, default=15)
    runs.add_argument("--job", choices=["ingest", "evaluate"])
    runs.add_argument("-v", "--verbose", action="store_true", help="include error column")
    runs.set_defaults(func=cmd_runs)

    dq = sub.add_parser("dq", help="data-quality results of a run (default: latest ingest)")
    dq.add_argument("run_id", nargs="?")
    dq.set_defaults(func=cmd_dq)

    q = sub.add_parser("quarantine", help="review rejected rows")
    q.add_argument("action", choices=["list", "release", "drop"])
    q.add_argument("id", type=int, nargs="?")
    q.add_argument("--all", action="store_true", help="list: include resolved rows")
    q.set_defaults(func=cmd_quarantine)

    h = sub.add_parser("health", help="health check (exit 0 OK, 1 WARNING, 2 CRITICAL)")
    h.add_argument("--max-data-lag-days", type=int, default=2)
    h.add_argument("--json", action="store_true")
    h.set_defaults(func=cmd_health)

    al = sub.add_parser("alphas", help="latest result for each alpha in the library")
    al.set_defaults(func=cmd_alphas)

    ops = sub.add_parser("ops", help="list operators, fields and groups")
    ops.add_argument("--markdown", action="store_true")
    ops.set_defaults(func=cmd_ops)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    fmt = args.log_format or ("text" if sys.stderr.isatty() else "json")
    setup_logging(args.log_level, fmt)
    settings = Settings.from_env()
    if args.db:
        settings = replace(settings, db_path=args.db)
    try:
        return args.func(args, settings)
    except (ValueError, FileNotFoundError) as exc:
        log.error("command failed", extra={"command": args.command, "error": str(exc)})
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_FAILED


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
