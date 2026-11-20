"""Job wrappers: run a pipeline step, then emit metrics and alerts for it.

The core functions (``run_ingest``, ``run_batch``) know nothing about Prometheus or
Slack; that keeps them easy to test and lets observability change independently.
"""

from __future__ import annotations

from alphaops.alerts import send_alert
from alphaops.batch import BatchReport
from alphaops.config import Settings
from alphaops.ingest import IngestReport
from alphaops.metrics import emit_job_metrics
from alphaops.storage import Store

JOB_NAMES = {"ingest": "ingest", "evaluate": "evaluate"}


def observe(
    settings: Settings,
    store: Store,
    pipeline: str,
    report: IngestReport | BatchReport,
    *,
    alerts: bool = True,
) -> None:
    emit_job_metrics(
        settings,
        pipeline,
        status=report.status,
        duration_s=report.duration_s,
        last_success=store.last_success(JOB_NAMES[pipeline]),
        counters=report.counters(),
        dq_failures=report.dq_failures() if isinstance(report, IngestReport) else None,
    )
    if not alerts:
        return
    if report.status == "failed":
        send_alert(settings, f"alphaops {pipeline} FAILED", report.summary(), "critical")
    elif report.status == "degraded":
        send_alert(settings, f"alphaops {pipeline} degraded", report.summary(), "warning")
