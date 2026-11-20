"""Runtime settings, read from environment variables (12-factor style).

Everything has a safe local default so `alphaops` works out of the box; production
deployments override via env vars or an `.env` file loaded by systemd/docker.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta, timezone
from pathlib import Path

# Vietnam has no daylight saving time, so a fixed offset is exact and avoids a
# dependency on the tz database (absent by default on Windows).
VN_TZ = timezone(timedelta(hours=7), name="ICT")


def today_vn(now: datetime | None = None) -> date:
    """Current calendar date in Vietnam (the market's local date)."""
    now = now or datetime.now(UTC)
    return now.astimezone(VN_TZ).date()


SESSION_FINAL_HOUR = 15  # HOSE closes 14:45 ICT; treat the day's bar as final from 15:00


def last_completed_session(now: datetime | None = None) -> date:
    """Most recent date whose session is over (weekends are handled by the calendar)."""
    now = (now or datetime.now(UTC)).astimezone(VN_TZ)
    return now.date() if now.hour >= SESSION_FINAL_HOUR else now.date() - timedelta(days=1)


def _opt(env: Mapping[str, str], key: str) -> str | None:
    value = env.get(key, "").strip()
    return value or None


@dataclass(frozen=True)
class Settings:
    db_path: Path
    metrics_dir: Path
    pushgateway_url: str | None
    slack_webhook_url: str | None
    telegram_bot_token: str | None
    telegram_chat_id: str | None
    max_symbol_failure_rate: float
    stale_after_hours: float
    max_abs_return: float

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> Settings:
        env = os.environ if env is None else env
        return cls(
            db_path=Path(env.get("ALPHAOPS_DB", "data/alphaops.db")),
            metrics_dir=Path(env.get("ALPHAOPS_METRICS_DIR", "data/metrics")),
            pushgateway_url=_opt(env, "ALPHAOPS_PUSHGATEWAY_URL"),
            slack_webhook_url=_opt(env, "ALPHAOPS_SLACK_WEBHOOK_URL"),
            telegram_bot_token=_opt(env, "ALPHAOPS_TELEGRAM_BOT_TOKEN"),
            telegram_chat_id=_opt(env, "ALPHAOPS_TELEGRAM_CHAT_ID"),
            max_symbol_failure_rate=float(env.get("ALPHAOPS_MAX_SYMBOL_FAILURE_RATE", "0.2")),
            stale_after_hours=float(env.get("ALPHAOPS_STALE_AFTER_HOURS", "26")),
            max_abs_return=float(env.get("ALPHAOPS_MAX_ABS_RETURN", "0.35")),
        )
