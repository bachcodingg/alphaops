"""Health checks for monitoring and cron "dead man's switch" use.

Exit codes follow the Nagios/Icinga convention so any monitoring system can run it:
0 = OK, 1 = WARNING, 2 = CRITICAL.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import UTC, datetime

import numpy as np

from alphaops.config import Settings, today_vn
from alphaops.storage import Store

OK, WARNING, CRITICAL = "ok", "warning", "critical"
EXIT_CODES = {OK: 0, WARNING: 1, CRITICAL: 2}


@dataclass(frozen=True)
class HealthCheck:
    name: str
    status: str
    detail: str

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


def run_health(
    store: Store,
    settings: Settings,
    *,
    now: datetime | None = None,
    max_data_lag_days: int = 2,
    stuck_after_hours: float = 6.0,
) -> list[HealthCheck]:
    now = now or datetime.now(UTC)
    checks: list[HealthCheck] = []

    integrity = store.integrity_check()
    checks.append(HealthCheck("db_integrity", OK if integrity == "ok" else CRITICAL, integrity))

    last_ok = store.last_success("ingest")
    if last_ok is None:
        checks.append(HealthCheck("ingest_freshness", CRITICAL, "ingest has never succeeded"))
    else:
        age_h = (now - last_ok).total_seconds() / 3600
        status = OK if age_h <= settings.stale_after_hours else CRITICAL
        checks.append(
            HealthCheck(
                "ingest_freshness",
                status,
                f"last successful ingest {age_h:.1f}h ago (limit {settings.stale_after_hours:g}h)",
            )
        )

    latest = store.latest_price_date()
    if latest is None:
        checks.append(HealthCheck("data_freshness", CRITICAL, "no price data"))
    else:
        today = today_vn(now)
        # business days strictly between the latest bar and today
        lag = int(np.busday_count(latest, today)) - 1 if latest < today else 0
        lag = max(lag, 0)
        # Tet closes the market for ~5 business days, so that range is only a warning.
        status = (
            OK
            if lag <= max_data_lag_days
            else (WARNING if lag <= max_data_lag_days + 5 else CRITICAL)
        )
        checks.append(
            HealthCheck(
                "data_freshness",
                status,
                f"latest bar {latest.isoformat()}, {lag} business day(s) missing since "
                f"(limit {max_data_lag_days}; public holidays such as Tet also count)",
            )
        )

    backlog = store.quarantine_backlog()
    checks.append(
        HealthCheck(
            "quarantine_backlog",
            OK if backlog == 0 else WARNING,
            f"{backlog} unresolved quarantined row(s)"
            + ("; review with `alphaops quarantine list`" if backlog else ""),
        )
    )

    stuck = store.stale_running(stuck_after_hours)
    checks.append(
        HealthCheck(
            "stuck_runs",
            OK if not stuck else WARNING,
            f"{len(stuck)} run(s) in 'running' for > {stuck_after_hours:g}h"
            + (f": {', '.join(stuck[:3])}" if stuck else ""),
        )
    )
    return checks


def overall_status(checks: list[HealthCheck]) -> str:
    if any(c.status == CRITICAL for c in checks):
        return CRITICAL
    if any(c.status == WARNING for c in checks):
        return WARNING
    return OK
