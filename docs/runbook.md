# Runbook

This is what to do when an alert fires. Every Prometheus alert links to a section here.
Commands assume the project directory and the production `.env`.

**First 5 minutes, for any alert:**

1. `alphaops runs -v --limit 5`: which job, which status, what error?
2. `alphaops dq`: which data-quality checks failed in the latest ingest?
3. `alphaops health`: is the data fresh, and is anything stuck?
4. Logs: `logs/daily-YYYY-MM-DD.log` (JSON lines; `grep '"level": "ERROR"'`), or
   `journalctl -u alphaops-daily.service --since today` under systemd.
5. Post in the team channel: what's affected, what you're checking, when you'll update next.

Downstream impact to keep in mind: alpha evaluation reads whatever is in `prices`.
Quarantined rows are *absent*, not wrong, so a degraded ingest gives slightly less data.
It never gives bad data.

---

## AlphaopsJobFailed

`alphaops_job_status_code == 2`. The run did not complete, or too many symbols failed.

| Symptom in `alphaops runs -v` | Likely cause | Action |
|---|---|---|
| `N/M symbols failed (...) upstream likely down` | Vendor outage or rate limiting | Check one symbol by hand (see below). If the vendor is down, wait and re-run; the job is idempotent. |
| `OperationalError: database is locked` | Another process holds the SQLite write lock | `fuser data/alphaops.db` / `lsof data/alphaops.db`. Never run two ingests at once (cron uses `flock`). |
| `OperationalError: disk I/O error` / unusual slowness | DB on a network filesystem (NFS/SMB/`\\wsl$`) | SQLite WAL mode needs a local disk. Move the DB to local storage. |
| Python traceback | Bug, or a new vendor response format | Reproduce with `--log-level DEBUG`, fix with a test, deploy, re-run. |

Manual check of one symbol (Yahoo):

```bash
curl -s -A "Mozilla/5.0" "https://query1.finance.yahoo.com/v8/finance/chart/FPT.VN?range=5d&interval=1d" | head -c 400
```

Re-running is always safe: writes are upserts keyed on (symbol, date).

```bash
alphaops ingest --source yahoo --universe config/universe_hose.csv
```

## AlphaopsJobDegraded

The run completed but some symbols failed after retries, or new rows were quarantined.
This doesn't block anything downstream. Triage within the working day:

* **Symbols failed:** `alphaops runs -v` shows them. One or two symbols failing every day
  usually means a delisting or ticker change. Update `config/universe_hose.csv`.
* **Rows quarantined:** see [AlphaopsDataQualityErrors](#alphaopsdataqualityerrors).

## AlphaopsIngestStale

No successful ingest for more than 26 hours on a weekday. Either the job did not run, or
it runs and keeps failing.

1. Did it run? `systemctl list-timers alphaops-daily.timer` / `grep alphaops /var/log/syslog`
   (cron). Check that the log file for today exists.
2. Is a run stuck? `alphaops health` reports runs left in `running` (killed by OOM, reboot
   or timeout). They are harmless; the next run proceeds normally.
3. Is it a market holiday? Tet closes HOSE for about a week. Health reports *warning*
   (not critical) for lags of up to ~7 business days for this reason.

## AlphaopsMetricsMissing

Prometheus has never received metrics from ingest. Usually the job can't reach the
Pushgateway (`ALPHAOPS_PUSHGATEWAY_URL`), or it has never been scheduled. Pushing is
best-effort: the job logs `failed to push metrics; continuing` and still succeeds. The
metrics are also always written to `$ALPHAOPS_METRICS_DIR/*.prom`.

## AlphaopsDataQualityErrors

Rows failed an **error**-severity check and were quarantined (kept out of `prices`).

```bash
alphaops quarantine list          # what was rejected and why
alphaops dq                       # counts and samples per check
```

Decide per row:

| Reason | Typical cause | Decision |
|---|---|---|
| `ohlc_inconsistent` | Bad tick from the vendor (e.g. high = 769 for a stock trading at 18,000) | Drop. The bar is missing for that day; returns around it are NaN, which is safe. |
| `missing_price` | Vendor sent a partial bar | Drop. If the vendor later fixes it, the next incremental run (7-day lookback) picks it up. |
| `extreme_return` with ratio ≈ 0.5, 0.33, 0.83 | Corporate action (split, stock dividend) not yet in the adjusted history | Re-fetch the full adjusted history for that symbol: `alphaops ingest --source yahoo --symbols XYZ --full`. Check for restatements, then drop the quarantined row. |
| `extreme_return`, other ratio | Fat-finger tick or a units change | Compare with a second source (broker app or exchange website). Release only if it was real. |

```bash
alphaops quarantine drop 42       # reject permanently
alphaops quarantine release 42    # accept the row into prices (it was correct after all)
```

A dropped row is not re-quarantined when the same bad data arrives again (unique on
symbol, date and reasons), so the backlog does not refill every day.

## AlphaopsVendorIncident

`systemic_date`: on one date a large share of symbols failed error checks. It is one
upstream incident, not many independent bad ticks. Real example from Yahoo data: on
**2021-11-02**, 50 of 62 HOSE symbols came back with `close > high`.

1. Don't release rows one by one.
2. Confirm on a second source for two or three symbols.
3. Drop the rows for that date. Optionally backfill the date from another source
   (`--source csv` with a vendor export), then re-run `alphaops batch` so alpha results
   reflect the repaired data.
4. Write it up (see `docs/postmortem-template.md`) if it affected anything downstream.

## AlphaopsSymbolsFailing

See the first row of the [AlphaopsJobFailed](#alphaopsjobfailed) table. A permanent
`HTTP 404` / `No data found` means the ticker is wrong or delisted. Fix the universe file
rather than letting the alert fire every day.

---

## Routine operations

| Task | Command |
|---|---|
| Backfill a date range | `alphaops ingest --source yahoo --universe config/universe_hose.csv --start 2020-01-01 --full` |
| Add symbols | edit `config/universe_hose.csv`; the next run fetches their full history |
| Evaluate a new alpha | add it to `examples/alphas.toml`, then `alphaops batch examples/alphas.toml` |
| Inspect one alpha | `alphaops simulate "rank(-ts_delta(close, 5))" --decay 3 --cost-bps 15` |
| Back up the DB (online) | `sqlite3 data/alphaops.db ".backup data/backup-$(date +%F).db"` |
| Check DB integrity | `alphaops health` (runs `PRAGMA quick_check`) |
