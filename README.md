# alphaops

A production-style **market data pipeline and alpha evaluation engine** for Vietnamese
equities (HOSE).

It ingests daily bars on a schedule and validates every row with data-quality checks. Bad
data is quarantined instead of reaching research. Alpha ideas are written in a
Fast-Expression-style language and evaluated as daily-rebalanced long/short books. The
whole thing is run like a production service: idempotent jobs, retries, a run ledger,
Prometheus metrics, alerts, a health check, cron/systemd units and a runbook.

```text
$ alphaops ingest --source yahoo --universe config/universe_hose.csv --start 2020-01-01
run ingest-20261119T224727-e05603: DEGRADED in 33.3s
rows fetched=100877 written=100820 rejected=57 (new in quarantine=57) restated=0
symbols ok=62 failed=0
data quality: ohlc_inconsistent(error)=54, extreme_return(error)=3, price_limit_breach(warn)=307,
              stale_price(warn)=12, systemic_date(warn)=1, missing_bar(warn)=282
```

## What it found in real data

From the first full ingest of 62 liquid HOSE stocks via Yahoo Finance (2020-01 to 2026-11):

| Finding | How it was caught | What happens |
|---|---|---|
| **2021-11-02:** 50 of 62 symbols arrive with `close > high` | `ohlc_inconsistent` on every row, plus `systemic_date` because one date affects 81% of the universe | Rows quarantined; one *vendor incident* alert instead of 50 separate ones |
| **ACB, TPB, 2025-06-04/05:** `high` = 769 while the stock trades around 18,000, duplicated over two days | `ohlc_inconsistent` | Quarantined, so it never distorts a backtest |
| Placeholder bars on Tet, 30/4, 2/9 and other holidays: all-null, or flat with zero volume | Found while investigating 433 `zero_volume` warnings | Dropped in the source adapter; they would have created fake zero-return trading days |
| A flat, zero-volume bar for the latest session before it was final | Same rule | Dropped until the session is over |
| 307 moves beyond the ±7% HOSE limit (e.g. ACB +19.9% then −15.9% the next day) | `price_limit_breach` (warning) | Stored but flagged for review: on HOSE these are almost always data errors, not real moves |

A full re-run of the same window writes 0 new quarantine rows and reports 0 restatements,
which shows the pipeline is idempotent.

**The textbook alphas don't survive real data.** On the synthetic market (which has a
reversal effect planted on purpose) 5-day reversal scores Sharpe ≈ 1.4. On real HOSE large
caps it scores ≈ 0 before costs (−0.05). Its sector-neutral variant scores −0.47 before
costs and −2.4 with 15 bps costs. None of the six example alphas passes the checks. This
repository is about evaluating ideas honestly, not about producing a good-looking backtest.

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -e ".[dev]"

# Offline demo on the deterministic synthetic market, with injected bad data and flaky upstream:
alphaops ingest --source synthetic --anomalies --chaos 0.1     # exit 3 = degraded (expected)
alphaops quarantine list
alphaops batch examples/alphas.toml
alphaops simulate "group_neutralize(-ts_sum(returns, 5), sector)" --decay 3 --neutralization sector
alphaops health

# Real HOSE data:
alphaops --db data/hose.db ingest --source yahoo --universe config/universe_hose.csv --start 2020-01-01
alphaops --db data/hose.db batch examples/alphas.toml
```

Run the tests with `pytest` (137 tests, about 93% coverage) and lint with `ruff check src tests`.

## Commands

| Command | Purpose |
|---|---|
| `ingest` | fetch, validate and store bars (`--source synthetic\|yahoo\|csv\|vnstock`, `--full` to backfill) |
| `simulate EXPR` | evaluate one alpha: Sharpe, fitness, turnover, drawdown, by-year table, IS/OS split, costs |
| `batch FILE` | evaluate an alpha library (TOML) against submission-style checks, incl. correlation with accepted alphas |
| `runs`, `dq`, `alphas` | run ledger, data-quality results per run, latest result per alpha |
| `quarantine list\|release\|drop` | review rejected rows |
| `health` | freshness, integrity, backlog and stuck runs; exit 0/1/2 (Nagios convention) |
| `ops` | list operators, fields and groups |

Exit codes are stable for scripting: `0` success, `1` failed, `2` usage error, `3` degraded.

## Alpha language

```text
rank(-ts_delta(close, 5))
move = -ts_sum(returns, 5); surprise = volume / ts_mean(volume, 20); rank(move) * (surprise > 1.5 ? 1.5 : 1)
group_neutralize(ts_zscore(dollar_volume, 20), sector)
```

There are 39 operators (time-series, cross-sectional, group, element-wise, `trade_when`). Fields
are OHLCV, `returns`, `vwap`, `dollar_volume` and `adv20`; groups are `sector`, `exchange` and
`market`. Simulation settings: delay, decay, neutralization, truncation, liquidity universe
(top-N by lagged ADV) and transaction costs. Full reference:
[docs/expression-language.md](docs/expression-language.md).

## Operations

| Piece | Where |
|---|---|
| Daily job (flock, per-day logs, exit-code handling) | [`scripts/run_daily.sh`](scripts/run_daily.sh) |
| Scheduling | [`deploy/cron`](deploy/cron), [`deploy/systemd`](deploy/systemd) (hardened unit + timer after the HOSE close) |
| Metrics | `.prom` textfile per job and Pushgateway push; [Prometheus alert rules](deploy/prometheus/alert_rules.yml) |
| Dashboard | [`deploy/grafana`](deploy/grafana): `docker compose up -d`, then http://localhost:3000 |
| Alerts | Slack webhook and/or Telegram bot (`.env.example`); best-effort, never fail the job |
| Runbook | [docs/runbook.md](docs/runbook.md): one section per alert, with commands |
| Postmortems | [docs/postmortem-template.md](docs/postmortem-template.md) |
| CI | [`.github/workflows/ci.yml`](.github/workflows/ci.yml): ruff, shellcheck, tests on Python 3.11 to 3.13, end-to-end smoke run, Docker build |

Design decisions (quarantine, idempotency, lookback, circuit breaker, no look-ahead) and
known limitations (survivorship bias, no practical short selling in Vietnam, simple cost
model) are in [docs/architecture.md](docs/architecture.md).

## Layout

```text
src/alphaops/
  sources/        synthetic, yahoo, csv, vnstock adapters (one canonical schema)
  quality.py      row-level and dataset-level data-quality checks
  ingest.py       fetch -> validate -> quarantine -> upsert -> record
  storage.py      SQLite: prices, quarantine, runs, dq_results, alpha_results
  engine/         parser (recursive descent), operators, evaluator, simulator
  batch.py        alpha library evaluation and checks
  metrics.py, alerts.py, health.py, jobs.py, cli.py
tests/            parser, operators, no-look-ahead, simulator, quality, ingest, CLI, ...
deploy/           cron, systemd, Prometheus, Grafana
docs/             architecture, runbook, expression language, postmortem template
```

## Data sources and licences

Yahoo Finance and vnstock are unofficial or community sources, used here for personal
research. Check their terms before any other use. The vnstock adapter is optional and was
not exercised in this repository's test run. No market data is committed to the repository.
