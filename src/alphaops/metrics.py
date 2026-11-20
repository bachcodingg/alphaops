"""Prometheus metrics for batch jobs.

Batch jobs don't live long enough to be scraped, so each run:

1. writes a ``.prom`` file to ``ALPHAOPS_METRICS_DIR`` (for node_exporter's textfile
   collector, and easy to inspect by hand), and
2. pushes to a Prometheus Pushgateway when ``ALPHAOPS_PUSHGATEWAY_URL`` is set.

A monitoring outage must never fail the job, so push errors are logged, not raised.

``alphaops_job_last_success_timestamp_seconds`` comes from the run ledger (not "now"),
so a failed run still reports when the job last succeeded. Prometheus alerts on its age
(see deploy/prometheus/alert_rules.yml).
"""

from __future__ import annotations

import logging
import time
from datetime import datetime
from pathlib import Path

from prometheus_client import CollectorRegistry, Gauge, push_to_gateway, write_to_textfile

from alphaops.config import Settings

log = logging.getLogger(__name__)

STATUS_CODES = {"success": 0, "degraded": 1, "failed": 2}


def emit_job_metrics(
    settings: Settings,
    pipeline: str,
    *,
    status: str,
    duration_s: float,
    last_success: datetime | None,
    counters: dict[str, float] | None = None,
    dq_failures: dict[tuple[str, str], int] | None = None,
) -> Path:
    reg = CollectorRegistry()

    def gauge(name: str, doc: str, value: float) -> None:
        Gauge(name, doc, ["pipeline"], registry=reg).labels(pipeline).set(value)

    gauge(
        "alphaops_job_last_run_timestamp_seconds", "Unix time the last run finished.", time.time()
    )
    gauge(
        "alphaops_job_status_code",
        "Outcome of the last run: 0=success, 1=degraded, 2=failed.",
        STATUS_CODES[status],
    )
    gauge("alphaops_job_duration_seconds", "Wall-clock duration of the last run.", duration_s)
    gauge(
        "alphaops_job_last_success_timestamp_seconds",
        "Unix time of the last successful (or degraded) run; 0 if never.",
        last_success.timestamp() if last_success else 0.0,
    )
    for key, value in (counters or {}).items():
        gauge(f"alphaops_{key}", f"{key.replace('_', ' ')} in the last run.", float(value))
    if dq_failures:
        g = Gauge(
            "alphaops_dq_failed_rows",
            "Rows failing each data-quality check in the last run.",
            ["pipeline", "check", "severity"],
            registry=reg,
        )
        for (check, severity), n in dq_failures.items():
            g.labels(pipeline, check, severity).set(n)

    settings.metrics_dir.mkdir(parents=True, exist_ok=True)
    path = settings.metrics_dir / f"alphaops_{pipeline}.prom"
    write_to_textfile(str(path), reg)

    if settings.pushgateway_url:
        try:
            push_to_gateway(
                settings.pushgateway_url,
                job="alphaops",
                grouping_key={"pipeline": pipeline},
                registry=reg,
                timeout=10,
            )
        except Exception as exc:
            log.warning(
                "failed to push metrics; continuing",
                extra={"pushgateway": settings.pushgateway_url, "error": repr(exc)},
            )
    return path
