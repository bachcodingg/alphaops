# Architecture

```mermaid
flowchart LR
    subgraph sources[Data sources]
        Y[Yahoo chart API<br/>HOSE .VN]
        V[vnstock<br/>optional]
        C[CSV exports]
        S[Synthetic market<br/>tests / CI / chaos]
    end

    subgraph ingest[Ingest job]
        F[fetch per symbol<br/>retry + backoff<br/>failure isolation]
        Q[data-quality checks<br/>row + dataset level]
    end

    subgraph db[(SQLite)]
        P[prices]
        QT[quarantine]
        R[runs]
        DQ[dq_results]
        A[alpha_results]
    end

    subgraph research[Research]
        E[expression engine<br/>parser → evaluator]
        SIM[simulator<br/>universe → decay → neutralize → truncate → PnL]
        B[batch + checks<br/>Sharpe, fitness, turnover, correlation]
    end

    subgraph ops[Operations]
        M[metrics: .prom file<br/>+ Pushgateway]
        AL[alerts: Slack / Telegram]
        H[health check<br/>exit 0/1/2]
        PR[Prometheus rules] --> G[Grafana]
    end

    sources --> F --> Q
    Q -- clean --> P
    Q -- error rows --> QT
    Q --> DQ
    F --> R
    P --> E --> SIM --> B --> A
    B --> R
    R --> M --> PR
    R --> AL
    db --> H
```

## Key decisions

**Quarantine, don't drop or fix.** Bad rows are kept, with reasons, in a separate table. Dropping
them silently loses evidence. Fixing them automatically (e.g. interpolating) puts made-up
numbers into research data. A quarantined bar is simply missing from `prices`, so returns
around it become NaN, which every operator handles. An operator releases or drops each
row explicitly.

**Idempotent writes.** `prices` is keyed on (symbol, date) and every write is an upsert, so
re-running any day, or backfilling a range, is always safe. That property is what makes
"just re-run it" a valid incident response. Quarantine rows are unique on
(symbol, date, reasons), so known bad data does not refill the backlog on every run.

**Incremental with lookback.** Each run re-fetches 7 calendar days before the last stored
bar. Missed days heal themselves, and upstream *restatements* (the vendor rewriting recent
history) are detected and counted (`alphaops_rows_restated`).

**Failure isolation with a circuit breaker.** One bad symbol makes the run `degraded`; more than
20% failing (`ALPHAOPS_MAX_SYMBOL_FAILURE_RATE`) makes it `failed`, because at that point it's an
upstream outage, not a symbol problem, and it needs a person.

**Only transient errors are retried.** Timeouts, HTTP 429 and 5xx are retried with exponential
backoff and jitter. A 404 or "no data found" fails immediately; retrying it only adds load.

**Observability is best-effort.** If the Pushgateway or Slack is down, the job logs a warning
and carries on. A monitoring outage must never cause a data outage.

**No look-ahead by construction, and verified by test.** Time-series operators only look
backwards, the liquidity universe uses *yesterday's* ADV, and `delay` shifts positions
after the signal. `test_no_lookahead.py` perturbs all data after a date and asserts
nothing before it changes.

**No `eval`.** Alpha expressions are parsed by a hand-written recursive-descent parser into
an AST, so a user-supplied expression can't execute code.

## Timing convention

```
close t-1        close t          close t+1
   |  signal(t) uses data ≤ t  |
   |                           |-- position held (t, t+1] = weights(t - delay)
   |                           |   PnL(t+1) = Σ position(t) · return(t+1)
```

`delay=1` (the default) means the signal from day *t* is traded at the close of *t+1*.

## Known limitations

* **Survivorship bias.** The universe is today's list of large caps, so backtests over past
  years exclude companies that were delisted or fell out of favour. A point-in-time
  universe would fix it.
* **No practical short selling in Vietnam.** Long/short results show the strength of a signal,
  not a strategy you could trade. A long-only variant would compare the long side to an index.
* **Costs.** A single `cost_bps` parameter; no market impact, no price-limit-lock fills
  (on a limit-up day you often can't buy).
* **Daily bars only.** `vwap` is the typical price (H+L+C)/3, a proxy.
* **SQLite** keeps operations simple for one writer. Many concurrent writers would need
  Postgres; the `Store` class is the only place that would change.
